"""
STORM DROP - map asset generator.

Builds the photo-real environment models for the island (houses, warehouse,
barn, silo, props, trees, rocks, grass), exports them as .glb into models/map/
and renders preview images with Cycles into models/map/renders/.

Everything is procedural: geometry is built with bmesh, PBR textures (base
colour, roughness/metallic, normal) are generated with numpy and baked into
the .glb files, rocks are baked high-poly -> low-poly with Cycles.

Run with Blender 4.x (either form works):
    blender -b -P tools/make_map.py -- [options]
    python tools/make_map.py [options]          # with the `bpy` module from PyPI

Options:
    --only a,b,c     build only these assets (names without .glb)
    --no-render      skip the Cycles renders
    --render-only    only render (uses existing .glb files)
    --samples N      Cycles samples for renders (default 160)
    --quick          small renders, few samples (for testing)

Conventions (match the game's colliders in index.html):
    metres, +Y up, the door side faces +Z, origin = centre of the footprint at
    floor/ground level (y = 0). Internally all builders use these game
    coordinates; they are converted to Blender's Z-up space when meshes are made,
    and the glTF exporter converts back.
"""
import bpy, bmesh, math, os, sys, json, random, tempfile, time
import numpy as np
from mathutils import Vector, Matrix, noise as mnoise

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'models', 'map')
RENDERS = os.path.join(OUT, 'renders')
TEXDIR = tempfile.mkdtemp(prefix='stormdrop_tex_')

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
def _opt(name, default=None):
    if name in argv:
        i = argv.index(name)
        return argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith('--') else True
    return default
ONLY = set(_opt('--only', '').split(',')) - {''} if isinstance(_opt('--only', ''), str) else set()
NO_RENDER = bool(_opt('--no-render', False))
RENDER_ONLY = bool(_opt('--render-only', False))
QUICK = bool(_opt('--quick', False))
SAMPLES = int(_opt('--samples', 24 if QUICK else 160))

def log(*a):
    print('[make_map]', *a, flush=True)

# =============================================================================
# 1. TEXTURE SYNTHESIS (numpy, all tileable)
#    Arrays are [row, col] with row = v (bottom-up, as Blender stores pixels).
#    Colours are authored in sRGB 0..1.
# =============================================================================
_FREQ = {}
def _freqs(n):
    if n not in _FREQ:
        f = np.fft.fftfreq(n)
        _FREQ[n] = (f[None, :], f[:, None])
    return _FREQ[n]

def noise(n, sx, sy=None, seed=0):
    """Tileable gaussian-filtered white noise; sx/sy = feature size in pixels."""
    sy = sx if sy is None else sy
    rng = np.random.default_rng(seed)
    w = np.fft.fft2(rng.standard_normal((n, n)))
    fx, fy = _freqs(n)
    k = np.exp(-2 * np.pi ** 2 * ((sx * fx) ** 2 + (sy * fy) ** 2))
    r = np.real(np.fft.ifft2(w * k))
    r -= r.mean()
    return r / (r.std() + 1e-9)

def fbm(n, s, oct=5, seed=0, aniso=1.0, gain=0.5):
    out = np.zeros((n, n)); a = 1.0; tot = 0
    for i in range(oct):
        sc = max(s / 2 ** i, 0.6)
        out += a * noise(n, sc * aniso, sc, seed + i * 17); tot += a; a *= gain
    return out / tot

def n01(x):
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)

def sstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)

def lerp(a, b, t):
    t = np.asarray(t)
    if t.ndim == 2 and np.ndim(a) >= 1 and np.shape(a)[-1:] == (3,):
        t = t[..., None]
    return a * (1 - t) + b * t

def col(c, shape):
    return np.broadcast_to(np.array(c, dtype=np.float64), shape + (3,)).copy()

def grid(n):
    v, u = np.mgrid[0:n, 0:n] / n
    return u, v

def cells_rand(ids, seed=0):
    """Deterministic pseudo-random 0..1 per integer cell id."""
    x = (ids.astype(np.int64) * 374761393 + seed * 668265263) & 0xFFFFFFFF
    x = (x ^ (x >> 13)) * 1274126177 & 0xFFFFFFFF
    return ((x ^ (x >> 16)) & 0xFFFF) / 65535.0

def voronoi(n, count, seed=0, aniso=(1.0, 1.0)):
    """Tileable voronoi: returns (F1, F2-F1 edge distance, cell id). Distances in 0..1 UV units."""
    rng = np.random.default_rng(seed)
    pts = rng.random((count, 2))
    u, v = grid(n)
    f1 = np.full((n, n), 9.0); f2 = np.full((n, n), 9.0); idx = np.zeros((n, n), dtype=np.int32)
    for i, (px, py) in enumerate(pts):
        dx = np.abs(u - px); dx = np.minimum(dx, 1 - dx) * aniso[0]
        dy = np.abs(v - py); dy = np.minimum(dy, 1 - dy) * aniso[1]
        d = np.sqrt(dx * dx + dy * dy)
        closer = d < f1
        f2 = np.where(closer, f1, np.minimum(f2, d))
        idx = np.where(closer, i, idx)
        f1 = np.where(closer, d, f1)
    return f1, f2 - f1, idx

def height_to_normal(h, strength):
    """OpenGL-style (+Y green) tangent space normal map from a height field (height in 'pixels')."""
    dx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) * 0.5 * strength
    dy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) * 0.5 * strength
    nz = np.ones_like(h)
    l = np.sqrt(dx * dx + dy * dy + nz)
    return np.stack([-dx / l * 0.5 + 0.5, -dy / l * 0.5 + 0.5, nz / l * 0.5 + 0.5], -1)

def T(base, rough, metal=0.0, height=None, nstr=4.0, alpha=None):
    n = base.shape[0]
    rough = np.broadcast_to(np.asarray(rough, dtype=np.float64), (n, n))
    metal = np.broadcast_to(np.asarray(metal, dtype=np.float64), (n, n))
    return dict(base=np.clip(base, 0, 1), rough=np.clip(rough, 0.02, 1), metal=np.clip(metal, 0, 1),
                height=height, nstr=nstr, alpha=alpha)

# ---- individual material generators ---------------------------------------
def tx_stucco(n, tint=(0.88, 0.80, 0.66), seed=1):
    big = fbm(n, n / 6, 4, seed); fine = fbm(n, 3, 3, seed + 5); spk = noise(n, 0.8, seed=seed + 9)
    dirt = n01(fbm(n, n / 4, 4, seed + 3, aniso=0.35))
    base = col(tint, (n, n)) * (1 + 0.045 * big[..., None] + 0.03 * fine[..., None])[..., :]
    base = lerp(base, base * np.array([0.78, 0.74, 0.68]), sstep(0.55, 1.0, dirt) * 0.6)
    base = base * (1 - 0.06 * (spk[..., None] > 2.2))
    cracks = sstep(0.985, 1.0, 1 - np.abs(fbm(n, n / 10, 4, seed + 21)) / 2.4) * (n01(big) > 0.6)
    base = base * (1 - 0.18 * cracks[..., None])
    h = 1.5 * fine + 0.7 * big - 3 * cracks
    return T(base, 0.86 + 0.06 * fine, 0, h, 2.2)

def tx_plaster(n, seed=2):
    t = tx_stucco(n, (0.90, 0.88, 0.83), seed)
    t['height'] = t['height'] * 0.4
    return t

def tx_clay_tiles(n, rows=8, cols=6, tint=(0.70, 0.34, 0.20), seed=3, flat=False):
    u, v = grid(n)
    r = np.floor(v * rows); fv = v * rows - r
    off = (r % 2) * 0.5
    cu = u * cols + off; c = np.floor(cu); fu = cu - c
    cid = (r * 97 + c % cols).astype(np.int64)
    rnd = cells_rand(cid, seed); rnd2 = cells_rand(cid, seed + 1)
    if flat:   # flat plain tiles with a small gap
        prof = sstep(0.0, 0.06, fu) * sstep(0.0, 0.06, 1 - fu)
        h = prof * (0.4 + 0.6 * fv) * 10
        shadow = 1 - 0.55 * (1 - sstep(0.0, 0.18, fv))
    else:      # barrel tiles
        prof = np.abs(np.sin(np.pi * fu)) ** 0.6
        h = prof * 14 + fv * 5
        shadow = (1 - 0.6 * (1 - sstep(0.0, 0.22, fv))) * (0.55 + 0.45 * prof)
    g = fbm(n, 6, 4, seed + 2)
    tintv = np.array(tint)
    base = tintv * (0.80 + 0.35 * rnd[..., None]) * np.array([1, 0.95 + 0.1 * 0.5, 1])
    base = lerp(base, base * np.array([0.82, 0.86, 0.95]), rnd2 * 0.6)
    base = base * (1 + 0.06 * g[..., None]) * shadow[..., None]
    lich = sstep(0.72, 0.88, n01(fbm(n, n / 14, 4, seed + 7))) * (0.5 + 0.5 * n01(noise(n, 2, seed=seed + 8)))
    base = lerp(base, np.array([0.55, 0.55, 0.38]), lich * 0.28)
    grime = n01(fbm(n, n / 5, 3, seed + 11, aniso=0.4))
    base = base * (1 - 0.25 * sstep(0.5, 1, grime))[..., None]
    h = h + g * 0.8
    return T(base, 0.62 + 0.15 * rnd2 + 0.15 * lich, 0, h, 1.6)

def tx_slate(n, rows=12, seed=4):
    u, v = grid(n)
    r = np.floor(v * rows); fv = v * rows - r
    # random-width slates per row
    rr = cells_rand(r.astype(np.int64), seed)
    cols = 7
    cu = u * cols + rr * 3.7 + (r % 2) * 0.5
    jitter = 0.18 * np.sin(np.floor(cu) * 12.9898 + r * 78.233)
    c = np.floor(cu + jitter * 0); fu = cu - c
    cid = (r * 131 + c % cols).astype(np.int64)
    a = cells_rand(cid, seed + 1); b = cells_rand(cid, seed + 2)
    gap = sstep(0.0, 0.03, fu) * sstep(0.0, 0.03, 1 - fu)
    shadow = 1 - 0.5 * (1 - sstep(0.0, 0.15, fv))
    grain = fbm(n, 10, 4, seed + 3, aniso=3.0)
    base = np.array([0.23, 0.25, 0.28]) * (0.75 + 0.5 * a[..., None])
    base = lerp(base, np.array([0.30, 0.27, 0.30]), b * 0.4)
    base = base * (1 + 0.08 * grain[..., None]) * (shadow * (0.4 + 0.6 * gap))[..., None]
    lich = sstep(0.7, 0.85, n01(fbm(n, n / 12, 4, seed + 5))) * n01(noise(n, 1.5, seed=seed + 6))
    base = lerp(base, np.array([0.62, 0.62, 0.50]), lich * 0.25)
    h = gap * (fv * 6 + grain * 0.6)
    return T(base, 0.45 + 0.2 * b - 0.1 * grain * 0.3 + 0.3 * lich, 0, h, 2.0)

def tx_brick(n, seed=5, bx=7, by=20):
    u, v = grid(n)
    r = np.floor(v * by); fv = v * by - r
    cu = u * bx + (r % 2) * 0.5; c = np.floor(cu); fu = cu - c
    mw = 0.045
    mort = 1 - sstep(0.0, mw * 1.6, fu) * sstep(0.0, mw * 1.6, 1 - fu) * sstep(0.0, mw * 1.6 * bx / by * 2.2, fv) * sstep(0.0, mw * 1.6 * bx / by * 2.2, 1 - fv)
    cid = (r * 53 + c % bx).astype(np.int64)
    a = cells_rand(cid, seed); b = cells_rand(cid, seed + 1); cc = cells_rand(cid, seed + 2)
    pal = np.array([[0.58, 0.25, 0.17], [0.66, 0.32, 0.21], [0.48, 0.21, 0.15], [0.70, 0.40, 0.28], [0.40, 0.19, 0.15]])
    pi = np.minimum((a * len(pal)).astype(int), len(pal) - 1)
    base = pal[pi] * (0.85 + 0.3 * b[..., None])
    burnt = cc > 0.86
    base = np.where(burnt[..., None], base * 0.55, base)
    g = fbm(n, 4, 4, seed + 3); spk = noise(n, 0.7, seed=seed + 4)
    base = base * (1 + 0.07 * g[..., None]) * (1 - 0.12 * (spk > 1.8))[..., None]
    mortar = np.array([0.66, 0.63, 0.57]) * (1 + 0.08 * fbm(n, 2, 3, seed + 6))[..., None]
    base = lerp(base, mortar, mort)
    soot = sstep(0.55, 1, n01(fbm(n, n / 4, 4, seed + 9, aniso=0.5)))
    base = base * (1 - 0.25 * soot)[..., None]
    h = (1 - mort) * 5 + g * 0.6 - mort * 1.0
    return T(base, 0.82 + 0.1 * mort + 0.05 * g, 0, h, 1.8)

def _wood_grain(n, seed, aniso=14.0):
    g = fbm(n, 6, 5, seed, aniso=aniso)
    rings = np.sin(g * 6 + fbm(n, 30, 3, seed + 1, aniso=6) * 3)
    return g, rings

def tx_planks(n, boards=8, tint=(0.52, 0.45, 0.36), seed=6, weathered=1.0, vertical=False, gap=0.03, ends=True):
    u, v = grid(n)
    if vertical:
        u, v = v, u
    r = np.floor(v * boards); fv = v * boards - r
    seg = cells_rand(r.astype(np.int64), seed) * 0.9
    lu = (u + seg) % 1.0
    if ends:
        L = 0.5
        e = np.floor(lu / L); fe = lu / L - e
    else:
        e = np.zeros_like(lu); fe = np.full_like(lu, 0.5)
    cid = (r * 31 + e).astype(np.int64)
    a = cells_rand(cid, seed + 1); b = cells_rand(cid, seed + 2)
    g, rings = _wood_grain(n, seed + 3)
    if vertical:
        g, rings = g.T, rings.T
    gapm = 1 - sstep(0.0, gap, fv) * sstep(0.0, gap, 1 - fv) * sstep(0.0, 0.006, fe) * sstep(0.0, 0.006, 1 - fe)
    base = np.array(tint) * (0.78 + 0.4 * a[..., None])
    base = base * (1 + 0.10 * g[..., None] + 0.06 * rings[..., None])
    grey = np.array([0.55, 0.53, 0.50]) * (1 + 0.12 * g[..., None])
    base = lerp(base, grey, np.clip(weathered * (0.3 + 0.4 * b), 0, 1))
    base = base * (1 - 0.75 * gapm)[..., None]
    knots = sstep(2.6, 3.4, noise(n, 3, seed=seed + 7))
    base = base * (1 - 0.45 * knots)[..., None]
    h = -gapm * 6 + g * 0.6 + rings * 0.35 - knots * 0.5
    out = T(base, 0.72 + 0.12 * weathered + 0.08 * g, 0, h, 2.2)
    if vertical:
        for k in ('base',):
            out[k] = np.ascontiguousarray(np.swapaxes(out[k], 0, 1))
        out['rough'] = np.ascontiguousarray(out['rough'].T); out['height'] = np.ascontiguousarray(out['height'].T)
    return out

def tx_painted_boards(n, paint, boards=10, seed=7, lap=True, wear=0.25, vertical=False):
    """Clapboard (lap=True) or board siding painted with chipped paint over grey wood."""
    u, v = grid(n)
    if vertical:
        u, v = v, u
    r = np.floor(v * boards); fv = v * boards - r
    g, rings = _wood_grain(n, seed)
    if vertical:
        g, rings = g.T, rings.T
    chip = n01(fbm(n, 9, 5, seed + 2)) * (0.7 + 0.6 * n01(fbm(n, n / 6, 3, seed + 3)))
    chip = sstep(1 - wear * 0.9, 1 - wear * 0.6, chip)
    wood = np.array([0.50, 0.47, 0.42]) * (1 + 0.12 * g[..., None])
    p = np.array(paint) * (1 + 0.035 * g[..., None] + 0.03 * fbm(n, n / 8, 3, seed + 4)[..., None])
    base = lerp(p, wood, chip)
    if lap:
        shade = 0.55 + 0.45 * sstep(0.0, 0.16, fv)
        h = fv * 8 - chip * 0.6 + g * 0.3
    else:
        gapm = 1 - sstep(0.0, 0.04, fv) * sstep(0.0, 0.04, 1 - fv)
        shade = 1 - 0.7 * gapm
        h = -gapm * 5 - chip * 0.6 + g * 0.35
    dirt = n01(fbm(n, n / 4, 4, seed + 6, aniso=0.4))
    base = base * shade[..., None] * (1 - 0.18 * sstep(0.55, 1, dirt))[..., None]
    rough = 0.55 + 0.3 * chip + 0.05 * g
    out = T(base, rough, 0, h, 2.0)
    if vertical:
        out['base'] = np.ascontiguousarray(np.swapaxes(out['base'], 0, 1))
        out['rough'] = np.ascontiguousarray(out['rough'].T); out['height'] = np.ascontiguousarray(out['height'].T)
    return out

def tx_paint(n, c, seed=8, rough=0.5, grain=True, wear=0.12, metal_under=False):
    g = fbm(n, 8, 4, seed, aniso=10 if grain else 1)
    blot = fbm(n, n / 6, 3, seed + 1)
    chip = sstep(1 - wear * 0.5, 1 - wear * 0.2, n01(fbm(n, 6, 5, seed + 2)))
    base = np.array(c) * (1 + 0.04 * g[..., None] + 0.03 * blot[..., None])
    under = np.array([0.35, 0.34, 0.33]) if metal_under else np.array([0.5, 0.45, 0.38])
    base = lerp(base, under, chip)
    h = g * (0.35 if grain else 0.15) - chip * 0.6
    return T(base, rough + 0.06 * blot + 0.25 * chip, chip * (0.8 if metal_under else 0), h, 1.5)

def tx_corrugated(n, paint=None, ribs=16, seed=9, rust=0.35, horizontal=False):
    """Corrugated sheet: sinusoidal ribs. paint=None -> galvanised zinc."""
    u, v = grid(n)
    a = u if not horizontal else v
    h = np.sin(2 * np.pi * a * ribs) * 9
    seams = np.abs(((a * 2) % 1.0) - 0.5) < 0.004
    streak = fbm(n, 2, 4, seed, aniso=1 / 30 if not horizontal else 30)
    streak2 = n01(fbm(n, 6, 4, seed + 1, aniso=1 / 18 if not horizontal else 18))
    big = fbm(n, n / 5, 3, seed + 2)
    if paint is None:
        f1, ed, cid = voronoi(n, 160, seed + 3)
        spang = cells_rand(cid.astype(np.int64), seed)
        base = np.array([0.60, 0.62, 0.63]) * (0.88 + 0.16 * spang[..., None]) * (1 + 0.05 * big[..., None])
        white = sstep(0.6, 0.95, streak2)
        base = lerp(base, np.array([0.78, 0.78, 0.76]), white * 0.45)
        metal = 0.85 - 0.5 * white
        rough = 0.38 + 0.15 * spang + 0.25 * white
    else:
        base = np.array(paint) * (1 + 0.04 * big[..., None] + 0.03 * streak[..., None])
        metal = np.zeros((n, n)); rough = 0.5 + 0.08 * big
    rustm = sstep(1 - rust * 0.6, 1 - rust * 0.25, n01(fbm(n, 5, 5, seed + 5)) * 0.6 + streak2 * 0.55)
    rustc = lerp(np.array([0.45, 0.20, 0.09]), np.array([0.30, 0.13, 0.07]), n01(noise(n, 1.5, seed=seed + 6)))
    base = lerp(base, rustc, rustm)
    shade = 0.82 + 0.18 * (np.sin(2 * np.pi * a * ribs) * 0.5 + 0.5)
    base = base * shade[..., None]
    if paint is not None:
        base = base * (1 - 0.12 * sstep(0.6, 1, streak2))[..., None]
    metal = metal * (1 - rustm)
    rough = rough * (1 - rustm) + 0.9 * rustm
    h = h - seams * 3 + rustm * 1.2 * n01(noise(n, 1, seed=seed + 7))
    return T(base, rough, metal, h, 0.9)

def tx_container(n, paint=(0.80, 0.80, 0.78), seed=10):
    """Trapezoid corrugated container steel, ribs vertical along u."""
    u, v = grid(n)
    ribs = 10
    f = (u * ribs) % 1.0
    prof = np.clip(np.minimum(sstep(0.1, 0.3, f), sstep(0.1, 0.3, 1 - f + 0.2)), 0, 1)
    prof = sstep(0.12, 0.28, f) * (1 - sstep(0.62, 0.78, f))
    h = prof * 16
    big = fbm(n, n / 5, 3, seed); fine = fbm(n, 3, 3, seed + 1)
    streak = n01(fbm(n, 5, 4, seed + 2, aniso=1 / 20))
    base = np.array(paint) * (1 + 0.035 * big[..., None] + 0.02 * fine[..., None])
    base = base * (1 - 0.18 * sstep(0.55, 1.0, streak))[..., None]
    rustm = sstep(0.72, 0.86, n01(fbm(n, 7, 5, seed + 3)) * 0.65 + n01(big) * 0.25 + 0.3 * (1 - sstep(0.0, 0.08, v)))
    rustc = lerp(np.array([0.42, 0.19, 0.09]), np.array([0.26, 0.12, 0.07]), n01(noise(n, 1.2, seed=seed + 4)))
    base = lerp(base, rustc, rustm)
    dent = fbm(n, n / 10, 3, seed + 5) * 1.5
    shade = 0.86 + 0.14 * prof
    base = base * shade[..., None]
    return T(base, 0.48 + 0.4 * rustm + 0.05 * big, 0, h + dent + rustm * 0.8, 1.1)

def tx_concrete(n, seed=11, tint=(0.56, 0.555, 0.53), joints=True):
    u, v = grid(n)
    big = fbm(n, n / 5, 5, seed); fine = fbm(n, 2, 3, seed + 1)
    pores = noise(n, 0.6, seed=seed + 2) > 2.4
    base = np.array(tint) * (1 + 0.07 * big[..., None] + 0.04 * fine[..., None])
    stain = sstep(0.6, 1.0, n01(fbm(n, n / 7, 4, seed + 3)))
    base = base * (1 - 0.18 * stain)[..., None] * (1 - 0.25 * pores)[..., None]
    crack = sstep(0.975, 0.995, 1 - np.abs(fbm(n, n / 8, 5, seed + 4)) / 2.2)
    base = base * (1 - 0.4 * crack)[..., None]
    h = fine * 0.6 - pores * 1.0 - crack * 2
    if joints:
        j = (np.minimum(u, 1 - u) < 0.003) | (np.minimum(v, 1 - v) < 0.003)
        base = base * (1 - 0.5 * j)[..., None]; h = h - j * 3
    return T(base, 0.86 + 0.06 * fine - 0.12 * stain * 0, 0, h, 1.4)

def tx_stone(n, seed=12, count=34, tint=(0.56, 0.54, 0.50), mortar=(0.62, 0.60, 0.55)):
    f1, edge, cid = voronoi(n, count, seed, aniso=(1.0, 1.5))
    r = cells_rand(cid.astype(np.int64), seed); r2 = cells_rand(cid.astype(np.int64), seed + 1)
    pal = np.array([[0.58, 0.56, 0.52], [0.50, 0.48, 0.45], [0.64, 0.60, 0.53], [0.46, 0.44, 0.42], [0.60, 0.55, 0.47]])
    base = pal[np.minimum((r * 5).astype(int), 4)] * (np.array(tint) / 0.56)
    g = fbm(n, 4, 5, seed + 2)
    m = 1 - sstep(0.004, 0.02, edge)
    dome = sstep(0.0, 0.06, edge)
    base = base * (1 + 0.09 * g[..., None]) * (0.75 + 0.25 * dome)[..., None]
    lich = sstep(0.68, 0.82, n01(fbm(n, 12, 4, seed + 3))) * (r2 > 0.4)
    base = lerp(base, np.array([0.62, 0.62, 0.45]), lich * 0.5)
    base = lerp(base, np.array(mortar) * (1 + 0.1 * fbm(n, 2, 2, seed + 4))[..., None], m)
    h = dome * 7 + g * 0.8 - m * 2
    return T(base, 0.8 + 0.08 * g + 0.1 * m, 0, h, 1.3)

def tx_cobbles(n, seed=13):
    t = tx_stone(n, seed, count=90, tint=(0.50, 0.49, 0.47), mortar=(0.36, 0.33, 0.28))
    return t

def tx_straw(n, seed=14, rings=False):
    u, v = grid(n)
    if rings:
        cx = u - 0.5; cy = v - 0.5
        rr = np.sqrt(cx * cx + cy * cy); ang = np.arctan2(cy, cx)
        coord_a = rr * 40 + ang / (2 * np.pi) * 1.0
        s = np.sin(coord_a * 2 * np.pi + fbm(n, 6, 4, seed) * 2)
        fib = s * 0.5 + 0.5
        fib = fib * 0.6 + 0.4 * n01(noise(n, 1.0, seed=seed + 1))
    else:
        fib = n01(fbm(n, 1.2, 3, seed, aniso=25)) * 0.6 + 0.4 * n01(fbm(n, 2, 3, seed + 1, aniso=12))
    cross = n01(fbm(n, 1.5, 3, seed + 2, aniso=1 / 20))
    pal_a = np.array([0.80, 0.66, 0.36]); pal_b = np.array([0.58, 0.45, 0.22]); pal_c = np.array([0.88, 0.78, 0.50])
    base = lerp(pal_b, pal_a, fib)
    base = lerp(base, pal_c, sstep(0.75, 1.0, cross) * 0.6)
    weather = n01(fbm(n, n / 6, 3, seed + 3))
    base = lerp(base, np.array([0.52, 0.47, 0.38]), sstep(0.6, 1, weather) * 0.5)
    h = fib * 4 + cross * 1.5
    return T(base, 0.85 - 0.1 * fib, 0, h, 1.8)

def tx_bark(n, seed=15, tint=(0.33, 0.27, 0.22), ridged=True):
    ridge = 1 - np.abs(fbm(n, 5, 5, seed, aniso=1 / 7))
    ridge = n01(ridge) ** 1.6
    plates = n01(fbm(n, 14, 4, seed + 1, aniso=1 / 3))
    g = fbm(n, 2, 3, seed + 2)
    base = np.array(tint) * (0.5 + 0.7 * ridge[..., None]) * (1 + 0.06 * g[..., None])
    moss = sstep(0.62, 0.85, n01(fbm(n, n / 8, 4, seed + 3)))
    base = lerp(base, np.array([0.32, 0.38, 0.17]), moss * 0.55)
    h = ridge * 9 + plates * 2 + g * 0.5
    return T(base, 0.9 - 0.05 * ridge, 0, h, 1.6)

def tx_birch_bark(n, seed=16):
    u, v = grid(n)
    g = fbm(n, 4, 4, seed)
    base = np.array([0.86, 0.85, 0.80]) * (1 + 0.04 * g[..., None])
    peel = n01(fbm(n, 8, 4, seed + 1, aniso=1.0))
    base = lerp(base, np.array([0.78, 0.72, 0.62]), sstep(0.7, 0.9, peel) * 0.6)
    lent = sstep(0.5, 0.9, n01(fbm(n, 1.0, 2, seed + 2, aniso=40)) * n01(fbm(n, 30, 3, seed + 3, aniso=1 / 2)))
    lent = lent * (n01(fbm(n, 10, 3, seed + 4)) > 0.55)
    blk = sstep(0.78, 0.88, n01(fbm(n, 16, 4, seed + 5, aniso=4)))
    base = lerp(base, np.array([0.12, 0.11, 0.10]), np.maximum(lent * 0.9, blk))
    h = -lent * 2 - blk * 2 + g * 0.6
    return T(base, 0.6 + 0.3 * blk, 0, h, 1.4)

def tx_gold(n, seed=17):
    g = fbm(n, 5, 4, seed); sc = n01(fbm(n, 0.8, 2, seed + 1, aniso=25))
    base = np.array([1.0, 0.76, 0.33]) * (1 + 0.05 * g[..., None])
    tarn = sstep(0.6, 0.95, n01(fbm(n, n / 8, 3, seed + 2)))
    base = lerp(base, np.array([0.55, 0.40, 0.18]), tarn * 0.5)
    return T(base, 0.22 + 0.15 * sc + 0.25 * tarn, 1.0, g * 0.3 + sc * 0.5, 1.0)

def tx_steel(n, c=(0.20, 0.21, 0.22), seed=18, wear=0.25, rough=0.45):
    return tx_paint(n, c, seed, rough, grain=False, wear=wear, metal_under=True)

def tx_rubber(n, seed=19, tread=True):
    u, v = grid(n)
    g = fbm(n, 3, 3, seed)
    base = np.array([0.075, 0.075, 0.075]) * (1 + 0.15 * g[..., None])
    dust = sstep(0.55, 1.0, n01(fbm(n, n / 8, 3, seed + 1)))
    base = lerp(base, np.array([0.30, 0.26, 0.20]), dust * 0.4)
    h = g * 0.3
    if tread:  # chevron lugs along v
        w = np.abs(v - 0.5) * 2
        lug = ((u * 5 + w * 0.6) % 1.0) < 0.38
        lug = lug & (w > 0.12)
        h = h + lug * 8
        base = base * (0.75 + 0.25 * lug)[..., None]
    return T(base, 0.9, 0, h, 1.2)

def tx_glass(n, seed=20):
    g = fbm(n, n / 6, 3, seed); smudge = n01(fbm(n, 12, 4, seed + 1))
    base = np.array([0.045, 0.055, 0.06]) * (1 + 0.08 * g[..., None])
    dirt = sstep(0.55, 1, smudge)
    base = lerp(base, np.array([0.22, 0.21, 0.19]), dirt * 0.25)
    return T(base, 0.05 + 0.25 * dirt, 0.0, g * 0.1, 0.3)

def tx_ground(n, seed=21):
    """Render-only grass/dirt ground."""
    g1 = fbm(n, 2.2, 4, seed); g2 = n01(fbm(n, n / 6, 5, seed + 1)); g3 = n01(fbm(n, n / 20, 4, seed + 2))
    grass = lerp(np.array([0.25, 0.35, 0.11]), np.array([0.42, 0.48, 0.18]), n01(g1))
    grass = lerp(grass, np.array([0.50, 0.50, 0.27]), sstep(0.6, 0.9, g3) * 0.5)
    dirt = np.array([0.42, 0.34, 0.25]) * (1 + 0.1 * g1[..., None])
    m = sstep(0.75, 0.9, g2 * 0.7 + n01(noise(n, 1.2, seed=seed + 3)) * 0.3)
    base = lerp(grass, dirt, m)
    return T(base, 0.95, 0, g1 * 1.5, 1.6)

def tx_dirt_road(n, seed=22):
    g = fbm(n, 3, 5, seed); big = n01(fbm(n, n / 6, 4, seed + 1))
    pebbles = noise(n, 1.4, seed=seed + 2) > 1.6
    base = np.array([0.46, 0.39, 0.30]) * (1 + 0.1 * g[..., None])
    base = lerp(base, np.array([0.55, 0.52, 0.47]), pebbles * 0.7)
    base = base * (1 - 0.15 * sstep(0.6, 1, big))[..., None]
    return T(base, 0.9, 0, g + pebbles * 2.5, 1.5)

# ---- alpha cards (leaves, needles, grass) ----------------------------------
def _stamp_leaf(img, a, cx, cy, L, W, ang, color, lobes=0, rng=None):
    n = img.shape[0]
    r = int(L * 0.6 + W) + 2
    x0, x1 = int(cx - r), int(cx + r); y0, y1 = int(cy - r), int(cy + r)
    x0c, x1c, y0c, y1c = max(x0, 0), min(x1, n), max(y0, 0), min(y1, n)
    if x0c >= x1c or y0c >= y1c:
        return
    yy, xx = np.mgrid[y0c:y1c, x0c:x1c].astype(np.float64)
    dx, dy = xx - cx, yy - cy
    ca, sa = math.cos(ang), math.sin(ang)
    lu = (dx * ca + dy * sa) / L + 0.5      # 0 base .. 1 tip
    lv = (-dx * sa + dy * ca)
    wid = W * np.clip(np.sin(np.pi * np.clip(lu, 0, 1)), 0, 1) ** 0.75 * (1 - 0.25 * lu)
    if lobes:
        wid = wid * (0.72 + 0.28 * np.abs(np.sin(lu * np.pi * lobes)))
    m = (lu > 0) & (lu < 1) & (np.abs(lv) < wid)
    if not m.any():
        return
    shade = 0.82 + 0.25 * (lv / (W + 1e-6)) * 0.5 + 0.12 * lu
    vein = np.abs(lv) < 0.7
    c = np.array(color)[None, None, :] * shade[..., None]
    c = np.where(vein[..., None], c * 1.18, c)
    sub = img[y0c:y1c, x0c:x1c]
    sub[m] = c[m]
    a[y0c:y1c, x0c:x1c][m] = 1.0

def _stamp_line(img, a, p0, p1, w, color, w1=None):
    n = img.shape[0]
    w1 = w if w1 is None else w1
    steps = int(max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1])) * 1.5) + 2
    for i in range(steps):
        t = i / (steps - 1)
        x = p0[0] + (p1[0] - p0[0]) * t; y = p0[1] + (p1[1] - p0[1]) * t
        ww = w + (w1 - w) * t
        rr = int(math.ceil(ww))
        xa, xb, ya, yb = max(int(x) - rr, 0), min(int(x) + rr + 1, n), max(int(y) - rr, 0), min(int(y) + rr + 1, n)
        if xa >= xb or ya >= yb:
            continue
        yy, xx = np.mgrid[ya:yb, xa:xb]
        m = (xx - x) ** 2 + (yy - y) ** 2 <= ww * ww + 0.25
        img[ya:yb, xa:xb][m] = color
        a[ya:yb, xa:xb][m] = 1.0

def tx_leaves(n, kind='oak', seed=23):
    """2x2 atlas of leafy sprigs with alpha."""
    rng = random.Random(seed)
    img = np.zeros((n, n, 3)); a = np.zeros((n, n))
    pal = {'oak': [(0.24, 0.36, 0.10), (0.30, 0.42, 0.12), (0.20, 0.30, 0.09), (0.36, 0.44, 0.15)],
           'birch': [(0.38, 0.52, 0.16), (0.46, 0.58, 0.20), (0.32, 0.46, 0.13), (0.55, 0.60, 0.22)]}[kind]
    q = n // 2
    for qi in range(4):
        ox, oy = (qi % 2) * q, (qi // 2) * q
        # stem from bottom centre to upper area
        base = (ox + q * 0.5, oy + q * 0.04)
        tips = []
        for b in range(5 if kind == 'oak' else 6):
            ang = math.radians(rng.uniform(55, 125))
            ln = q * rng.uniform(0.45, 0.85)
            tip = (base[0] + math.cos(ang) * ln, base[1] + math.sin(ang) * ln)
            _stamp_line(img, a, base, tip, 2.2 * n / 1024, (0.25, 0.20, 0.12), 0.8 * n / 1024)
            tips.append((base, tip))
        count = 330 if kind == 'oak' else 420
        for i in range(count):
            b0, tip = rng.choice(tips)
            t = rng.uniform(0.25, 1.0)
            px = b0[0] + (tip[0] - b0[0]) * t + rng.gauss(0, q * 0.035)
            py = b0[1] + (tip[1] - b0[1]) * t + rng.gauss(0, q * 0.035)
            L = q * (0.10 if kind == 'oak' else 0.07) * rng.uniform(0.75, 1.2)
            W = L * (0.30 if kind == 'oak' else 0.36)
            ang = math.atan2(tip[1] - b0[1], tip[0] - b0[0]) + rng.uniform(-1.4, 1.4)
            c = np.array(rng.choice(pal)) * rng.uniform(0.85, 1.15)
            _stamp_leaf(img, a, px, py, L, W, ang, c, lobes=(4 if kind == 'oak' else 0))
    # pad colour under transparent pixels to avoid dark fringes when mipmapped
    mean = img[a > 0].mean(0)
    img[a == 0] = mean
    rough = np.full((n, n), 0.6)
    h = a * 2 + fbm(n, 1.5, 2, seed) * 0.3 * a
    return T(img, rough, 0, h, 1.5, alpha=a)

def tx_needles(n, seed=24, tint=(0.14, 0.24, 0.12)):
    rng = random.Random(seed)
    img = np.zeros((n, n, 3)); a = np.zeros((n, n))
    q = n // 2
    for qi in range(4):
        ox, oy = (qi % 2) * q, (qi // 2) * q
        p0 = (ox + q * 0.5, oy + q * 0.03); p1 = (ox + q * 0.5 + rng.uniform(-q * 0.08, q * 0.08), oy + q * 0.97)
        main = [(p0, p1)]
        for s in range(6):   # side twigs
            t = rng.uniform(0.15, 0.8)
            sp = (p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t)
            side = rng.choice([-1, 1])
            ep = (sp[0] + side * q * rng.uniform(0.2, 0.38), sp[1] + q * rng.uniform(0.1, 0.25))
            main.append((sp, ep))
        for (s0, s1) in main:
            _stamp_line(img, a, s0, s1, 2.0 * n / 1024, (0.28, 0.20, 0.13), 1.0 * n / 1024)
            seg_len = math.hypot(s1[0] - s0[0], s1[1] - s0[1])
            dirn = math.atan2(s1[1] - s0[1], s1[0] - s0[0])
            k = int(seg_len / (2.2 * n / 1024))
            for i in range(k):
                t = i / max(k - 1, 1)
                bx = s0[0] + (s1[0] - s0[0]) * t; by = s0[1] + (s1[1] - s0[1]) * t
                for side in (-1, 1):
                    ang = dirn + side * rng.uniform(0.6, 1.05)
                    ln = q * rng.uniform(0.07, 0.11) * (0.6 + 0.4 * math.sin(math.pi * min(t * 1.1, 1)))
                    c = np.array(tint) * rng.uniform(0.75, 1.35)
                    _stamp_line(img, a, (bx, by), (bx + math.cos(ang) * ln, by + math.sin(ang) * ln), 1.1 * n / 1024, c, 0.6 * n / 1024)
    mean = img[a > 0].mean(0)
    img[a == 0] = mean
    return T(img, np.full((n, n), 0.65), 0, a * 1.5, 1.2, alpha=a)

def tx_grass_card(n, seed=25):
    rng = random.Random(seed)
    img = np.zeros((n, n, 3)); a = np.zeros((n, n))
    for i in range(140):
        x0 = rng.uniform(0.06, 0.94) * n
        h = rng.uniform(0.35, 0.97) * n
        lean = rng.uniform(-0.25, 0.25) * n
        w = rng.uniform(3.0, 6.0) * n / 512
        segs = 8
        pts = []
        for s in range(segs + 1):
            t = s / segs
            pts.append((x0 + lean * t * t, t * h))
        for s in range(segs):
            t = s / segs
            ca = np.array([0.30, 0.42, 0.13]) * (0.85 + 0.55 * t) * rng.uniform(0.85, 1.15)
            if rng.random() < 0.15:
                ca = ca * np.array([1.4, 1.2, 1.0])
            _stamp_line(img, a, pts[s], pts[s + 1], w * (1 - t * 0.9), np.clip(ca, 0, 1), w * (1 - (t + 1 / segs) * 0.9))
    mean = img[a > 0].mean(0)
    img[a == 0] = mean
    return T(img, np.full((n, n), 0.7), 0, a, 1.0, alpha=a)

# =============================================================================
# 2. MATERIALS
# =============================================================================
# name: (generator, size px, metres per texture repeat)
MATDEF = {
    'stucco_cream':  (lambda n: tx_stucco(n, (0.90, 0.82, 0.66), 31), 1024, 3.0),
    'plaster':       (lambda n: tx_plaster(n, 32), 512, 3.0),
    'clay_tiles':    (lambda n: tx_clay_tiles(n, 8, 6, (0.72, 0.35, 0.20), 33), 1024, 2.4),
    'clay_tiles_flat': (lambda n: tx_clay_tiles(n, 10, 7, (0.52, 0.24, 0.17), 34, flat=True), 1024, 2.4),
    'slate':         (lambda n: tx_slate(n, 12, 35), 1024, 3.0),
    'brick':         (lambda n: tx_brick(n, 36), 1024, 1.5),
    'brick_dark':    (lambda n: tx_brick(n, 37), 512, 1.5),
    'siding_blue':   (lambda n: tx_painted_boards(n, (0.56, 0.67, 0.72), 10, 38, lap=True, wear=0.18), 1024, 2.0),
    'wood_floor':    (lambda n: tx_planks(n, 12, (0.55, 0.38, 0.22), 39, weathered=0.05, gap=0.02), 1024, 2.4),
    'wood_weathered': (lambda n: tx_planks(n, 6, (0.45, 0.38, 0.30), 40, weathered=0.85), 1024, 1.6),
    'wood_dark':     (lambda n: tx_planks(n, 6, (0.36, 0.24, 0.15), 41, weathered=0.25, ends=False), 512, 1.6),
    'wood_chest':    (lambda n: tx_planks(n, 5, (0.50, 0.30, 0.15), 42, weathered=0.05, ends=False), 512, 1.0),
    'barn_red':      (lambda n: tx_painted_boards(n, (0.55, 0.11, 0.08), 12, 43, lap=False, wear=0.1, vertical=True), 1024, 3.0),
    'trim_white':    (lambda n: tx_paint(n, (0.88, 0.87, 0.83), 44, 0.55, wear=0.08), 512, 1.5),
    'paint_green':   (lambda n: tx_paint(n, (0.18, 0.33, 0.24), 45, 0.5, wear=0.15), 512, 1.5),
    'paint_blue':    (lambda n: tx_paint(n, (0.12, 0.20, 0.32), 46, 0.45, wear=0.1), 512, 1.5),
    'paint_red':     (lambda n: tx_paint(n, (0.55, 0.13, 0.10), 47, 0.5, wear=0.12), 512, 1.5),
    'paint_yellow':  (lambda n: tx_paint(n, (0.85, 0.62, 0.10), 48, 0.5, wear=0.2, grain=False, metal_under=True), 512, 1.5),
    'galvanised':    (lambda n: tx_corrugated(n, None, 16, 49, rust=0.25), 1024, 2.0),
    'galv_rings':    (lambda n: tx_corrugated(n, None, 16, 50, rust=0.3, horizontal=True), 1024, 2.0),
    'roof_tin':      (lambda n: tx_corrugated(n, None, 16, 51, rust=0.75), 1024, 2.0),
    'wh_wall':       (lambda n: tx_corrugated(n, (0.60, 0.63, 0.62), 16, 52, rust=0.3), 1024, 2.0),
    'wh_roof':       (lambda n: tx_corrugated(n, (0.35, 0.37, 0.38), 12, 53, rust=0.4), 1024, 2.4),
    'concrete':      (lambda n: tx_concrete(n, 54), 1024, 4.0),
    'concrete_plain': (lambda n: tx_concrete(n, 55, joints=False), 512, 2.0),
    'stone':         (lambda n: tx_stone(n, 56), 1024, 1.6),
    'stone_dark':    (lambda n: tx_stone(n, 57, 30, (0.46, 0.45, 0.43)), 512, 1.6),
    'steel_dark':    (lambda n: tx_steel(n, (0.19, 0.20, 0.21), 58), 512, 1.5),
    'steel_black':   (lambda n: tx_steel(n, (0.06, 0.06, 0.065), 59, wear=0.15, rough=0.35), 512, 1.0),
    'steel_orange':  (lambda n: tx_steel(n, (0.85, 0.42, 0.07), 60, wear=0.3), 512, 1.5),
    'galv_plain':    (lambda n: (lambda t: (t.update(height=t['height'] * 0.0 + fbm(n, 4, 3, 61) * 0.2) or t))(tx_corrugated(n, None, 1, 61, rust=0.15)), 512, 1.5),
    'glass':         (lambda n: tx_glass(n, 62), 256, 2.0),
    'gold':          (lambda n: tx_gold(n, 63), 512, 1.0),
    'straw':         (lambda n: tx_straw(n, 64), 1024, 1.2),
    'straw_end':     (lambda n: tx_straw(n, 65, rings=True), 512, 1.8),
    'rubber':        (lambda n: tx_rubber(n, 66), 512, 1.0),
    'rubber_plain':  (lambda n: tx_rubber(n, 67, tread=False), 256, 1.0),
    'tractor_paint': (lambda n: tx_paint(n, (0.16, 0.36, 0.14), 68, 0.32, grain=False, wear=0.2, metal_under=True), 512, 1.5),
    'rim_paint':     (lambda n: tx_paint(n, (0.88, 0.70, 0.12), 69, 0.4, grain=False, wear=0.25, metal_under=True), 512, 1.0),
    'bark_oak':      (lambda n: tx_bark(n, 70, (0.33, 0.28, 0.22)), 1024, 1.2),
    'bark_pine':     (lambda n: tx_bark(n, 71, (0.40, 0.25, 0.17)), 1024, 1.2),
    'bark_birch':    (lambda n: tx_birch_bark(n, 72), 1024, 1.2),
    'leaves_oak':    (lambda n: tx_leaves(n, 'oak', 73), 1024, 1.0),
    'leaves_birch':  (lambda n: tx_leaves(n, 'birch', 74), 1024, 1.0),
    'needles':       (lambda n: tx_needles(n, 75), 1024, 1.0),
    'needles_b':     (lambda n: tx_needles(n, 76, (0.17, 0.25, 0.13)), 1024, 1.0),
    'grass_card':    (lambda n: tx_grass_card(n, 77), 512, 1.0),
    'ground':        (lambda n: tx_ground(n, 78), 1024, 6.0),
    'dirt_road':     (lambda n: tx_dirt_road(n, 79), 1024, 4.0),
    'cobbles':       (lambda n: tx_cobbles(n, 80), 1024, 2.5),
}
for _c, _rgb in (('red', (0.62, 0.13, 0.09)), ('blue', (0.10, 0.27, 0.50)), ('green', (0.16, 0.40, 0.20)), ('orange', (0.88, 0.42, 0.08)), ('white', (0.82, 0.82, 0.80))):
    MATDEF['container_' + _c] = ((lambda rgb: (lambda n: tx_container(n, rgb, 81)))(_rgb), 1024, 2.8)

# extra non-texture parameters
MATEXTRA = {
    'glass': dict(),
    'lamp_glow': dict(emit=(1.0, 0.72, 0.38), emit_strength=6.0),
}
DOUBLE_SIDED = {'leaves_oak', 'leaves_birch', 'needles', 'needles_b', 'grass_card'}
_TEXCACHE = {}

# Normal and ORM maps are high-frequency noise that JPEG compresses badly; they are
# stored at half the base-colour resolution so each .glb stays under ~2 MB.
AUX_DOWNSCALE = 2

def _save_img(name, arr, alpha=None, noncolor=False, downscale=1, quality=85):
    if downscale > 1 and arr.shape[0] >= 256:
        n0 = arr.shape[0]; k = downscale
        arr = arr.reshape(n0 // k, k, n0 // k, k, -1).mean(axis=(1, 3))
        if alpha is not None:
            alpha = alpha.reshape(n0 // k, k, n0 // k, k).mean(axis=(1, 3))
    n = arr.shape[0]
    ext = 'png' if alpha is not None else 'jpg'
    img = bpy.data.images.new(name, n, n, alpha=alpha is not None)
    if noncolor:
        img.colorspace_settings.name = 'Non-Color'
    rgba = np.ones((n, n, 4), dtype=np.float32)
    rgba[..., :3] = arr
    if alpha is not None:
        rgba[..., 3] = alpha
    img.pixels.foreach_set(rgba.ravel())
    path = os.path.join(TEXDIR, name + '.' + ext)
    img.filepath_raw = path
    img.file_format = 'PNG' if alpha is not None else 'JPEG'
    if alpha is not None:
        img.save()
    else:
        img.save(quality=quality)
    img.source = 'FILE'
    img.reload()
    if noncolor:
        img.colorspace_settings.name = 'Non-Color'
    return img

def textures(name):
    if name in _TEXCACHE:
        return _TEXCACHE[name]
    gen, size, tile = MATDEF[name]
    t0 = time.time()
    t = gen(size)
    imgs = {}
    imgs['base'] = _save_img(name + '_basecolor', t['base'], t['alpha'])
    orm = np.stack([np.ones_like(t['rough']), t['rough'], t['metal']], -1)
    imgs['orm'] = _save_img(name + '_orm', orm, noncolor=True, downscale=AUX_DOWNSCALE, quality=80)
    if t['height'] is not None:
        nrm = height_to_normal(t['height'], t['nstr'] * size / 1024)
        if AUX_DOWNSCALE > 1 and size >= 256:
            k = AUX_DOWNSCALE
            nrm = nrm.reshape(size // k, k, size // k, k, 3).mean(axis=(1, 3))
            v = nrm * 2 - 1; v /= np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9
            nrm = v * 0.5 + 0.5
        imgs['normal'] = _save_img(name + '_normal', nrm, noncolor=True, quality=80)
    _TEXCACHE[name] = imgs
    log('texture %-16s %4dpx %.1fs' % (name, size, time.time() - t0))
    return imgs

def material(name):
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree; N = nt.nodes; L = nt.links
    bsdf = N['Principled BSDF']
    ex = MATEXTRA.get(name, {})
    if name in MATDEF:
        imgs = textures(name)
        tb = N.new('ShaderNodeTexImage'); tb.image = imgs['base']; tb.location = (-700, 300)
        L.new(tb.outputs['Color'], bsdf.inputs['Base Color'])
        if MATDEF[name][0] and name in DOUBLE_SIDED:
            rnd = N.new('ShaderNodeMath'); rnd.operation = 'ROUND'; rnd.location = (-350, 100)
            L.new(tb.outputs['Alpha'], rnd.inputs[0]); L.new(rnd.outputs[0], bsdf.inputs['Alpha'])
            m.use_backface_culling = False
            try:
                m.surface_render_method = 'DITHERED'
            except Exception:
                pass
            bsdf.inputs['Subsurface Weight'].default_value = 0.0
        to = N.new('ShaderNodeTexImage'); to.image = imgs['orm']; to.location = (-700, 0)
        sep = N.new('ShaderNodeSeparateColor'); sep.location = (-400, 0)
        L.new(to.outputs['Color'], sep.inputs['Color'])
        L.new(sep.outputs['Green'], bsdf.inputs['Roughness'])
        L.new(sep.outputs['Blue'], bsdf.inputs['Metallic'])
        if 'normal' in imgs:
            tn = N.new('ShaderNodeTexImage'); tn.image = imgs['normal']; tn.location = (-700, -300)
            nm = N.new('ShaderNodeNormalMap'); nm.location = (-400, -300)
            L.new(tn.outputs['Color'], nm.inputs['Color']); L.new(nm.outputs['Normal'], bsdf.inputs['Normal'])
    else:
        bsdf.inputs['Base Color'].default_value = (0.05, 0.05, 0.05, 1)
        bsdf.inputs['Roughness'].default_value = 0.3
    if 'emit' in ex:
        bsdf.inputs['Base Color'].default_value = (*ex['emit'], 1)
        bsdf.inputs['Emission Color'].default_value = (*ex['emit'], 1)
        bsdf.inputs['Emission Strength'].default_value = ex['emit_strength']
        bsdf.inputs['Roughness'].default_value = 0.2
    return m

def tile_of(name):
    return MATDEF[name][2] if name in MATDEF else 1.0
MATDEF['terracotta'] = (lambda n: tx_paint(n, (0.62, 0.30, 0.18), 82, 0.75, grain=False, wear=0.05), 256, 1.0)
MATDEF['wood_crate'] = (lambda n: tx_planks(n, 8, (0.66, 0.52, 0.34), 84, weathered=0.3, ends=False), 1024, 2.0)
MATDEF['rope'] = (lambda n: tx_straw(n, 83), 256, 0.5)

# =============================================================================
# 3. MESH BUILDER (game coordinates: x right, y up, z towards the door side)
# =============================================================================
X, Y, Z = Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1))

def G(p):
    """game (x, y, z) -> Blender (x, -z, y)"""
    return Vector((p[0], -p[2], p[1]))

def V(*a):
    return Vector(a[0]) if len(a) == 1 else Vector(a)

class MB:
    def __init__(self, name):
        self.name = name
        self.bm = bmesh.new()
        self.uv = self.bm.loops.layers.uv.new('UVMap')
        self.done = self.bm.faces.layers.int.new('uvdone')
        self.mats = []
        self.foliage = []      # (BMFace, centre in game coords, up blend)
        self.uvoff = (random.random() * 7.3, random.random() * 5.1)

    def mi(self, m):
        if m not in self.mats:
            self.mats.append(m)
        return self.mats.index(m)

    # -- primitives ---------------------------------------------------------
    def poly(self, pts, mat, uvs=None, smooth=False):
        vs = [self.bm.verts.new(G(p)) for p in pts]
        f = self.bm.faces.new(vs)
        f.material_index = self.mi(mat); f.smooth = smooth
        if uvs:
            for l, uv in zip(f.loops, uvs):
                l[self.uv].uv = uv
            f[self.done] = 1
        return f

    def obox(self, c, h, axes, mat, sides=None, bevel=0.0, skip=()):
        c = Vector(c); a = [Vector(x).normalized() for x in axes]
        sides = sides or {}
        vs = {}
        for i in (0, 1):
            for j in (0, 1):
                for k in (0, 1):
                    p = c + a[0] * (2 * i - 1) * h[0] + a[1] * (2 * j - 1) * h[1] + a[2] * (2 * k - 1) * h[2]
                    vs[(i, j, k)] = self.bm.verts.new(G(p))
        faces = []
        for k in range(3):
            i, j = (k + 1) % 3, (k + 2) % 3
            for s in (1, 0):
                lab = ('+' if s else '-') + 'xyz'[k]
                if lab in skip:
                    continue
                quad = []
                for (di, dj) in ((0, 0), (1, 0), (1, 1), (0, 1)):
                    key = [0, 0, 0]; key[k] = s; key[i] = di; key[j] = dj
                    quad.append(vs[tuple(key)])
                if not s:
                    quad.reverse()
                f = self.bm.faces.new(quad)
                f.material_index = self.mi(sides.get(lab, mat))
                faces.append(f)
        if bevel > 0:
            edges = list({e for f in faces for e in f.edges if len(e.link_faces) == 2})
            bmesh.ops.bevel(self.bm, geom=edges + list(vs.values()), offset=bevel, segments=1,
                            affect='EDGES', profile=0.5, clamp_overlap=True)
        return faces

    def box(self, x0, x1, y0, y1, z0, z1, mat, sides=None, bevel=0.0, skip=()):
        return self.obox(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), ((x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2),
                         (X, Y, Z), mat, sides, bevel, skip)

    def cbox(self, c, size, mat, **kw):
        return self.box(c[0] - size[0] / 2, c[0] + size[0] / 2, c[1] - size[1] / 2, c[1] + size[1] / 2,
                        c[2] - size[2] / 2, c[2] + size[2] / 2, mat, **kw)

    def beam(self, p0, p1, w, h, mat, up=Y, bevel=0.0, sides=None):
        """Box from p0 to p1 with cross-section w (side) x h (along `up`)."""
        p0, p1 = Vector(p0), Vector(p1)
        d = p1 - p0; L = d.length; a0 = d / L
        up = Vector(up)
        if abs(a0.dot(up)) > 0.99:
            up = X if abs(a0.dot(X)) < 0.9 else Z
        a2 = a0.cross(up).normalized(); a1 = a2.cross(a0)
        return self.obox((p0 + p1) / 2, (L / 2, h / 2, w / 2), (a0, a1, a2), mat, sides, bevel)

    def extrude(self, pts, off, mat, sides_mat=None, cap_mat=None):
        P = [Vector(p) for p in pts]; off = Vector(off)
        nrm = Vector((0, 0, 0))
        for i in range(len(P)):
            nrm += P[i].cross(P[(i + 1) % len(P)])
        if nrm.dot(off) < 0:
            P.reverse()
        top = [p + off for p in P]
        self.poly(list(reversed(P)), cap_mat or mat)
        self.poly(top, cap_mat or mat)
        for i in range(len(P)):
            a, b = P[i], P[(i + 1) % len(P)]
            self.poly([a, b, b + off, a + off], sides_mat or mat)

    def _frame(self, A):
        A = Vector(A).normalized()
        ref = Y if abs(A.dot(Y)) < 0.9 else X
        U = ref.cross(A).normalized()
        if A == Y or (A - Y).length < 1e-6:
            U = Vector((0, 0, 1))
        W = A.cross(U)
        return A, U, W

    def lathe(self, profile, segs, mat, center=(0, 0, 0), axis=Y, cap0=False, cap1=False, smooth=True,
              tile=None, start=0.0, sweep=2 * math.pi, cap_mat=None):
        """profile: list of (radius, height). Revolved around `axis` through `center`."""
        A, U, W = self._frame(axis)
        c = Vector(center)
        tile = tile or tile_of(mat)
        full = abs(sweep - 2 * math.pi) < 1e-6
        cols = segs + 1
        rings = []
        for (r, hgt) in profile:
            ring = []
            for j in range(cols):
                t = start + sweep * j / segs
                p = c + A * hgt + (U * math.cos(t) + W * math.sin(t)) * r
                ring.append(self.bm.verts.new(G(p)))
            rings.append(ring)
        if full:  # weld seam
            for ring in rings:
                pass
        rmax = max(r for r, _ in profile)
        circ = sweep * max(rmax, 0.05)
        vacc = [0.0]
        for i in range(1, len(profile)):
            vacc.append(vacc[-1] + math.hypot(profile[i][0] - profile[i - 1][0], profile[i][1] - profile[i - 1][1]))
        mi = self.mi(mat)
        faces = []
        for i in range(len(profile) - 1):
            for j in range(segs):
                q = [rings[i][j], rings[i][j + 1], rings[i + 1][j + 1], rings[i + 1][j]]
                if len({v.co.to_tuple(6) for v in q}) < 3:
                    continue
                try:
                    f = self.bm.faces.new(q)
                except ValueError:
                    continue
                f.material_index = mi; f.smooth = smooth; f[self.done] = 1
                uvs = [(j / segs * circ / tile, vacc[i] / tile), ((j + 1) / segs * circ / tile, vacc[i] / tile),
                       ((j + 1) / segs * circ / tile, vacc[i + 1] / tile), (j / segs * circ / tile, vacc[i + 1] / tile)]
                for l, uv in zip(f.loops, uvs):
                    l[self.uv].uv = uv
                faces.append(f)
        cm = cap_mat or mat
        for cap, idx in ((cap0, 0), (cap1, len(profile) - 1)):
            if not cap or profile[idx][0] < 1e-4:
                continue
            ring = rings[idx][:segs] if full else rings[idx]
            vs = list(ring) if idx == len(profile) - 1 else list(reversed(ring))
            f = self.bm.faces.new(vs)
            f.material_index = self.mi(cm); f.smooth = False
            ct = tile_of(cm); r = profile[idx][0]
            for l in f.loops:
                p = Vector((l.vert.co.x, l.vert.co.y, l.vert.co.z))
                pg = Vector((p.x, p.z, -p.y)) - c - A * profile[idx][1]
                l[self.uv].uv = (pg.dot(U) / ct + 0.5, pg.dot(W) / ct + 0.5)
            f[self.done] = 1
        if full:
            bmesh.ops.remove_doubles(self.bm, verts=[v for ring in rings for v in (ring[0], ring[-1])], dist=1e-5)
        return faces

    def cyl(self, c, r, h, mat, segs=12, axis=Y, caps=True, smooth=True, cap_mat=None):
        A = Vector(axis).normalized()
        c0 = Vector(c)
        return self.lathe([(r, 0), (r, h)], segs, mat, c0, A, caps, caps, smooth, cap_mat=cap_mat)

    def tube(self, pts, radii, segs, mat, cap_end=True, cap_start=False, tile=None, smooth=True, twist=0.0):
        """Sweep a circle along a polyline (parallel-transport frames)."""
        P = [Vector(p) for p in pts]
        if not hasattr(radii, '__len__'):
            radii = [radii] * len(P)
        tile = tile or tile_of(mat)
        tans = []
        for i in range(len(P)):
            a = P[max(i - 1, 0)]; b = P[min(i + 1, len(P) - 1)]
            tans.append((b - a).normalized())
        _, U, _ = self._frame(tans[0])
        rings = []; frames = []
        for i, p in enumerate(P):
            t = tans[i]
            U = (U - t * U.dot(t)).normalized()
            W = t.cross(U)
            ring = []
            for j in range(segs + 1):
                ang = 2 * math.pi * j / segs + twist * i
                ring.append(self.bm.verts.new(G(p + (U * math.cos(ang) + W * math.sin(ang)) * radii[i])))
            rings.append(ring)
        mi = self.mi(mat)
        vacc = [0.0]
        for i in range(1, len(P)):
            vacc.append(vacc[-1] + (P[i] - P[i - 1]).length)
        circ = 2 * math.pi * max(max(radii), 0.03)
        for i in range(len(P) - 1):
            for j in range(segs):
                q = [rings[i][j], rings[i][j + 1], rings[i + 1][j + 1], rings[i + 1][j]]
                f = self.bm.faces.new(q)
                f.material_index = mi; f.smooth = smooth; f[self.done] = 1
                uvs = [(j / segs * circ / tile, vacc[i] / tile), ((j + 1) / segs * circ / tile, vacc[i] / tile),
                       ((j + 1) / segs * circ / tile, vacc[i + 1] / tile), (j / segs * circ / tile, vacc[i + 1] / tile)]
                for l, uv in zip(f.loops, uvs):
                    l[self.uv].uv = uv
        if cap_end and radii[-1] > 1e-4:
            f = self.bm.faces.new(rings[-1][:segs]); f.material_index = mi; f.smooth = False
        if cap_start and radii[0] > 1e-4:
            f = self.bm.faces.new(list(reversed(rings[0][:segs]))); f.material_index = mi; f.smooth = False
        bmesh.ops.remove_doubles(self.bm, verts=[v for ring in rings for v in (ring[0], ring[-1])], dist=1e-6)

    def card(self, c, normal, up, w, h, mat, uvrect=(0, 0, 1, 1), crown=None, bend=0.0):
        """Alpha card (two triangles, double-sided material). crown = centre for soft foliage normals."""
        c = Vector(c); n = Vector(normal).normalized(); up = Vector(up)
        r = up.cross(n).normalized(); u2 = n.cross(r)
        u0, v0, u1, v1 = uvrect
        pts = [c - r * w / 2, c + r * w / 2, c + r * w / 2 + u2 * h + n * bend, c - r * w / 2 + u2 * h + n * bend]
        f = self.poly(pts, mat, [(u0, v0), (u1, v0), (u1, v1), (u0, v1)], smooth=True)
        if crown is not None:
            self.foliage.append((f, Vector(crown)))
        return f

    # -- finish ---------------------------------------------------------------
    def build(self, name=None, collection=None, fit_height=None):
        bm = self.bm
        if fit_height:
            top = max(v.co.z for v in bm.verts)
            if top > fit_height:
                f = fit_height / top
                for v in bm.verts:
                    v.co *= f
                self.foliage = [(fa, c * f) for fa, c in self.foliage]
        bm.normal_update()
        ox, oy = self.uvoff
        for f in bm.faces:
            if f[self.done]:
                continue
            n = f.normal
            tile = tile_of(self.mats[f.material_index])
            if abs(n.z) > 0.92:
                u = Vector((1, 0, 0)); v = Vector((0, 1, 0)) if n.z > 0 else Vector((0, -1, 0))
            else:
                u = Vector((0, 0, 1)).cross(n).normalized(); v = n.cross(u).normalized()
            for l in f.loops:
                p = l.vert.co
                l[self.uv].uv = (p.dot(u) / tile + ox, p.dot(v) / tile + oy)
        fol = []
        bm.faces.index_update()
        for f, c in self.foliage:
            fol.append((f.index, c))
        me = bpy.data.meshes.new(name or self.name)
        bm.to_mesh(me)
        bm.free()
        for m in self.mats:
            me.materials.append(material(m))
        me.set_sharp_from_angle(angle=math.radians(42))
        if fol:
            cn = [Vector(n.vector) for n in me.corner_normals]
            for pi, c in fol:
                poly = me.polygons[pi]
                cb = G(c)
                for li in poly.loop_indices:
                    vco = me.vertices[me.loops[li].vertex_index].co
                    d = (vco - cb)
                    d.z = d.z * 0.7 + 0.35 * d.length
                    cn[li] = d.normalized()
            me.normals_split_custom_set(cn)
        ob = bpy.data.objects.new(name or self.name, me)
        (collection or bpy.context.scene.collection).objects.link(ob)
        return ob

def tris(ob):
    me = ob.data
    me.calc_loop_triangles()
    return len(me.loop_triangles)

# =============================================================================
# 4. BUILDINGS
# =============================================================================
ASSETS = {}      # name -> builder returning a list of root objects
def asset(name):
    def deco(f):
        ASSETS[name] = f
        return f
    return deco

## =============================================================================
class Wall:
    """Local frame on the outer face of a wall. u = along the wall (to the right when
    seen from outside), v = absolute height y, d = outwards (0 = outer face)."""
    SIDES = {'+z': (Z, X), '-z': (-Z, -X), '+x': (X, -Z), '-x': (-X, Z)}

    def __init__(self, side, half, thick):
        self.n, self.t = (Vector(v) for v in Wall.SIDES[side])
        self.o = self.n * half
        self.thick = thick

    def p(self, u, v, d):
        return self.o + self.t * u + Y * v + self.n * d

    def box(self, mb, u0, u1, v0, v1, d0, d1, mat, sides=None, bevel=0.0, skip=()):
        c = self.p((u0 + u1) / 2, (v0 + v1) / 2, (d0 + d1) / 2)
        return mb.obox(c, ((u1 - u0) / 2, (v1 - v0) / 2, (d1 - d0) / 2), (self.t, Y, self.n), mat, sides, bevel, skip)

def window(mb, w, uc, y0, W, H, st, interior=True):
    """Visual window on a solid wall: glass, frame, mullions, sill, optional lintel/shutters/flower box."""
    y1 = y0 + H; fw = 0.09; trim = st['trim']
    w.box(mb, uc - W / 2, uc + W / 2, y0, y1, -0.01, 0.015, 'glass')
    w.box(mb, uc - W / 2 - fw, uc - W / 2, y0, y1, -0.01, 0.07, trim, bevel=0.012)
    w.box(mb, uc + W / 2, uc + W / 2 + fw, y0, y1, -0.01, 0.07, trim, bevel=0.012)
    w.box(mb, uc - W / 2 - fw, uc + W / 2 + fw, y1, y1 + fw, -0.01, 0.07, trim, bevel=0.012)
    w.box(mb, uc - 0.025, uc + 0.025, y0, y1, 0.0, 0.05, trim)
    w.box(mb, uc - W / 2, uc + W / 2, y0 + H * 0.62, y0 + H * 0.62 + 0.05, 0.0, 0.05, trim)
    w.box(mb, uc - W / 2 - 0.16, uc + W / 2 + 0.16, y0 - 0.09, y0, -0.01, 0.13, st['sill'], bevel=0.015)
    if st.get('lintel'):
        w.box(mb, uc - W / 2 - 0.16, uc + W / 2 + 0.16, y1 + fw, y1 + fw + 0.24, -0.01, 0.03, st['lintel'], bevel=0.01)
    if st.get('shutters'):
        for s in (-1, 1):
            a = uc + s * (W / 2 + fw + 0.02); b = a + s * 0.62
            w.box(mb, min(a, b), max(a, b), y0 - 0.02, y1 + 0.02, -0.01, 0.05, st['shutters'], bevel=0.01)
            for k in range(3):   # horizontal battens
                yy = y0 + 0.15 + k * (H - 0.3) / 2
                w.box(mb, min(a, b) + 0.05, max(a, b) - 0.05, yy - 0.04, yy + 0.04, 0.05, 0.075, st['shutters'])
    if st.get('flowerbox'):
        w.box(mb, uc - W / 2, uc + W / 2, y0 - 0.38, y0 - 0.1, 0.0, 0.26, st['flowerbox'], bevel=0.015)
        rng = random.Random(int(uc * 100 + y0 * 10))
        for k in range(6):
            cx = uc - W / 2 + 0.12 + k * (W - 0.24) / 5
            base = w.p(cx, y0 - 0.14, 0.13)
            for a in (0.0, 1.7):
                nrm = w.n * math.cos(a) + w.t * math.sin(a)
                mb.card(base, nrm, Y, 0.34, 0.36, 'leaves_birch', ((k % 2) * 0.5, (rng.random() < 0.5) * 0.5, (k % 2) * 0.5 + 0.5, (rng.random() < 0.5) * 0.5 + 0.5),
                        crown=base + Y * 0.3)
    if interior:   # the same window seen from inside
        di = -w.thick
        w.box(mb, uc - W / 2, uc + W / 2, y0, y1, di - 0.015, di + 0.01, 'glass')
        for (a, b, c, d) in ((uc - W / 2 - fw, uc - W / 2, y0, y1), (uc + W / 2, uc + W / 2 + fw, y0, y1),
                             (uc - W / 2 - fw, uc + W / 2 + fw, y1, y1 + fw), (uc - W / 2 - fw, uc + W / 2 + fw, y0 - fw, y0)):
            w.box(mb, a, b, c, d, di - 0.05, di + 0.01, 'trim_white')
        w.box(mb, uc - 0.025, uc + 0.025, y0, y1, di - 0.04, di, 'trim_white')
        w.box(mb, uc - W / 2 - 0.12, uc + W / 2 + 0.12, y0 - 0.06, y0, di - 0.16, di + 0.01, 'wood_dark', bevel=0.01)

def wall_lamp(mb, w, u, y):
    w.box(mb, u - 0.06, u + 0.06, y - 0.1, y + 0.1, 0.0, 0.03, 'steel_black')
    w.box(mb, u - 0.02, u + 0.02, y - 0.02, y + 0.02, 0.03, 0.18, 'steel_black')
    c = w.p(u, y - 0.12, 0.2)
    mb.cbox(c, (0.2, 0.26, 0.2), 'lamp_glow')
    mb.cbox(c + Y * 0.16, (0.26, 0.06, 0.26), 'steel_black', bevel=0.01)
    mb.cbox(c - Y * 0.15, (0.22, 0.04, 0.22), 'steel_black')

HOUSE_STYLES = {
    'house_a': dict(wall='stucco_cream', roof='clay_tiles', pitch=30, trim='trim_white', sill='stone', plinth='stone_dark',
                    shutters='paint_green', quoins='stone', chimney='stucco_cream', door='paint_green', soffit='wood_dark',
                    fascia='wood_dark', ridge='clay_tiles'),
    'house_b': dict(wall='brick', roof='slate', pitch=40, trim='trim_white', sill='concrete_plain', lintel='concrete_plain',
                    plinth='brick_dark', chimney='brick', door='paint_blue', soffit='trim_white', fascia='trim_white',
                    canopy=True, lamp=True, ridge='stone_dark'),
    'house_c': dict(wall='siding_blue', roof='clay_tiles_flat', pitch=34, trim='trim_white', sill='trim_white', plinth='stone',
                    corner_boards='trim_white', flowerbox='wood_dark', chimney='stone', door='paint_red', soffit='trim_white',
                    fascia='trim_white', lamp=True, ridge='clay_tiles_flat'),
}

def build_house(name):
    st = HOUSE_STYLES[name]
    S = MB('Structure'); D = MB('Details')
    HW, T, FY, TOP, CEIL = 4.0, 0.35, 0.35, 4.55, 4.85
    wm = st['wall']
    ext_in = {'+z': wm, '-z': 'plaster'}
    # --- structure: exactly the game's collider boxes ------------------------
    S.box(-HW, HW, 0, FY, -HW, HW, st['plinth'], sides={'+y': 'wood_floor'})
    walls = {s: Wall(s, HW, T) for s in ('+z', '-z', '+x', '-x')}
    walls['-z'].box(S, -HW, HW, FY, TOP, -T, 0, wm, ext_in)
    walls['+x'].box(S, -HW + T, HW - T, FY, TOP, -T, 0, wm, ext_in)
    walls['-x'].box(S, -HW + T, HW - T, FY, TOP, -T, 0, wm, ext_in)
    fw = walls['+z']
    fw.box(S, -HW, -0.9, FY, TOP, -T, 0, wm, ext_in)
    fw.box(S, 0.9, HW, FY, TOP, -T, 0, wm, ext_in)
    fw.box(S, -0.9, 0.9, FY + 2.6, TOP, -T, 0, wm, ext_in)
    S.box(-4.3, 4.3, TOP, CEIL, -4.3, 4.3, st['fascia'], sides={'-y': 'plaster'})
    # --- details ---------------------------------------------------------------
    for s, w in walls.items():          # plinth band
        segs = [(-HW, -0.9), (0.9, HW)] if s == '+z' else [(-HW, HW)]
        for a, b in segs:
            ea = -0.04 if a == -HW else 0; eb = 0.04 if b == HW else 0
            w.box(D, a + ea, b + eb, 0.0, 0.75, 0.0, 0.04, st['plinth'], bevel=0.01)
    if st.get('quoins'):
        for s, w in walls.items():
            for end in (-1, 1):
                for k in range(8):
                    y0 = 0.8 + k * 0.46
                    if y0 + 0.4 > TOP:
                        break
                    ln = 0.55 if (k + (s in '+x-x')) % 2 else 0.32
                    if end > 0:
                        w.box(D, HW - ln, HW + 0.03, y0, y0 + 0.4, 0, 0.03, st['quoins'], bevel=0.015)
                    else:
                        w.box(D, -HW - 0.03, -HW + ln, y0, y0 + 0.4, 0, 0.03, st['quoins'], bevel=0.015)
    if st.get('corner_boards'):
        for s, w in walls.items():
            for end in (-1, 1):
                a, b = (HW - 0.16, HW + 0.03) if end > 0 else (-HW - 0.03, -HW + 0.16)
                w.box(D, a, b, 0.75, TOP, 0, 0.03, st['corner_boards'], bevel=0.008)
    # windows: side walls and back wall
    for s, us in (('+x', [0.0]), ('-x', [0.0]), ('-z', [-2.0, 2.0])):
        for u in us:
            window(D, walls[s], u, FY + 1.1, 1.2, 1.4, st)
    for u in (-2.55, 2.55):    # small front windows (visual only)
        window(D, fw, u, FY + 1.2, 1.0, 1.3, dict(st, shutters=None))
    # door: casing, threshold, leaf swung inside flat against the front wall
    for d0, d1 in ((-0.01, 0.07), (-T - 0.05, -T + 0.01)):
        fw.box(D, -1.06, -0.9, FY, FY + 2.6, d0, d1, st['trim'], bevel=0.01)
        fw.box(D, 0.9, 1.06, FY, FY + 2.6, d0, d1, st['trim'], bevel=0.01)
        fw.box(D, -1.06, 1.06, FY + 2.6, FY + 2.76, d0, d1, st['trim'], bevel=0.01)
    D.box(-1.15, 1.15, 0.0, 0.17, HW, HW + 0.42, 'stone_dark', bevel=0.02)
    dz0, dz1 = HW - T - 0.075, HW - T - 0.01
    D.box(-2.62, -0.92, FY + 0.02, FY + 2.56, dz0, dz1, st['door'], sides={'-z': st['door']}, bevel=0.008)
    for yy in (FY + 0.35, FY + 1.3, FY + 2.2):       # ledges on the visible (inner) face
        D.box(-2.55, -0.99, yy - 0.07, yy + 0.07, dz0 - 0.03, dz0, st['door'], bevel=0.008)
    D.cyl((-2.45, FY + 1.05, dz0 - 0.03), 0.035, -0.06, 'gold', segs=8, axis=Z)
    for yy in (FY + 0.35, FY + 2.2):                   # hinges
        D.box(-1.2, -0.9, yy - 0.025, yy + 0.025, dz0 - 0.035, dz0 - 0.03 + 0.005, 'steel_black')
    # interior ceiling beams
    for zz in (-1.6, 1.6):
        D.box(-HW + T, HW - T, TOP - 0.22, TOP, zz - 0.11, zz + 0.11, 'wood_dark', bevel=0.015)
    if st.get('canopy'):
        th = math.radians(18)
        c = Vector((0, FY + 3.15 + 0.35 * math.sin(th), HW + 0.4))
        D.obox(c, (1.35, 0.06, 0.46), (X, Vector((0, math.cos(th), math.sin(th))), Vector((0, -math.sin(th), math.cos(th)))),
               st['roof'], sides={'-y': 'trim_white'}, bevel=0.01)
        for sx in (-1.1, 1.1):
            D.beam((sx, FY + 2.75, HW), (sx, FY + 3.15, HW + 0.65), 0.08, 0.08, 'trim_white')
            D.box(sx - 0.05, sx + 0.05, FY + 2.7, FY + 3.2, HW, HW + 0.12, 'trim_white')
    if st.get('lamp'):
        wall_lamp(D, fw, 1.45, FY + 2.35)
    # --- gable roof (ridge along x) ------------------------------------------
    th = math.radians(st['pitch']); tn = math.tan(th)
    EZ, EX, TR = 4.6, 4.5, 0.16
    under = lambda zz: CEIL + (4.3 - abs(zz)) * tn - 0.02
    for s in (1, -1):
        a0 = X * s; a1 = Vector((0, math.sin(th), -s * math.cos(th))); a2 = a0.cross(a1)
        half1 = (EZ / math.cos(th)) / 2 + 0.06
        mid = Vector((0, under(EZ / 2 * s), EZ / 2 * s))
        c = mid + a1 * 0.06 + a2 * TR / 2
        D.obox(c, (EX, half1, TR / 2), (a0, a1, a2), st['roof'],
               sides={'-z': st['soffit'], '-y': st['fascia'], '+x': st['fascia'], '-x': st['fascia']}, bevel=0.012)
    ridge_y = under(0) + TR / math.cos(th)
    D.lathe([(0.0, -EX - 0.02), (0.15, -EX - 0.02), (0.15, EX + 0.02), (0.0, EX + 0.02)], 10, st['ridge'],
            (0, ridge_y - 0.06, 0), X, smooth=True)
    for s in (1, -1):    # gable walls
        xs = s * HW
        tri = [(xs, CEIL, -4.3), (xs, CEIL, 4.3), (xs, under(0) + 0.02, 0)]
        tri = [(p[0], p[1], p[2]) for p in tri]
        gm = st['wall'] if st['wall'] != 'brick' else 'brick'
        D.extrude(tri, (-s * T, 0, 0), gm)
        D.box(xs - (0.02 if s < 0 else -0.02) - 0.02, xs + 0.02 + (0.02 if s > 0 else -0.02), CEIL + 0.9, CEIL + 1.5, -0.35, 0.35,
              'wood_dark', bevel=0.01)  # attic vent
        for k in range(5):
            yy = CEIL + 0.98 + k * 0.11
            D.box(min(xs + s * 0.035, xs + s * 0.065), max(xs + s * 0.035, xs + s * 0.065), yy, yy + 0.04, -0.3, 0.3, 'wood_dark')
    # gutters + downpipes
    gy = under(EZ) - 0.02
    for s in (1, -1):
        D.lathe([(0.0, -EX), (0.075, -EX), (0.075, EX), (0.0, EX)], 8, 'galv_plain', (0, gy, s * (EZ + 0.05)), X)
    for (px, pz, s) in ((HW - 0.22, HW + 0.07, 1), (-HW + 0.22, -HW - 0.07, -1)):
        D.tube([(px, 0.1, pz), (px, gy - 0.35, pz), (px, gy - 0.15, s * (EZ + 0.05))], 0.045, 8, 'galv_plain', cap_start=True)
        D.box(px - 0.06, px + 0.06, 1.6, 1.66, pz - 0.07 * s - 0.03, pz - 0.07 * s + 0.03, 'galv_plain')
        D.box(px - 0.06, px + 0.06, 3.3, 3.36, pz - 0.07 * s - 0.03, pz - 0.07 * s + 0.03, 'galv_plain')
    # chimney on the back slope
    cx, cz = 2.1, -1.9
    ctop = ridge_y + 0.9
    D.box(cx - 0.42, cx + 0.42, under(cz - 0.42) - 0.1, ctop, cz - 0.42, cz + 0.42, st['chimney'], bevel=0.01)
    D.box(cx - 0.5, cx + 0.5, ctop, ctop + 0.12, cz - 0.5, cz + 0.5, 'concrete_plain', bevel=0.02)
    for k, ox in enumerate((-0.18, 0.18)):
        D.lathe([(0.0, 0.0), (0.12, 0.0), (0.12, 0.08), (0.10, 0.12), (0.095, 0.42 + 0.05 * k), (0.115, 0.46 + 0.05 * k),
                 (0.085, 0.47 + 0.05 * k), (0.07, 0.30)], 10, 'terracotta', (cx + ox, ctop + 0.12, cz), Y)
    return [S.build(name + '_structure'), D.build(name + '_details')]

def gable_roof(mb, base, plate, ext_across, ext_along, half_along, pitch, ridge_axis, roof, soffit, fascia, tr=0.16,
               ridge=None, bevel=0.012):
    """Two sloped slabs meeting at a ridge. plate = distance from ridge line to where the slab
    underside meets `base` height; ext_across = eave distance from ridge; ext_along = half length
    along the ridge. Returns (under(across) function, ridge top height)."""
    AL = X if ridge_axis == 'x' else Z
    AC = Z if ridge_axis == 'x' else X
    th = math.radians(pitch); tn = math.tan(th)
    under = lambda a: base + (plate - abs(a)) * tn - 0.02
    for s in (1, -1):
        a1 = Y * math.sin(th) - AC * (s * math.cos(th))
        a2 = Y * math.cos(th) + AC * (s * math.sin(th))
        a0 = a1.cross(a2)
        half1 = (ext_across / math.cos(th)) / 2 + 0.06
        mid = AC * (ext_across / 2 * s) + Y * under(ext_across / 2)
        c = mid + a1 * 0.06 + a2 * tr / 2
        mb.obox(c, (ext_along, half1, tr / 2), (a0, a1, a2), roof,
                sides={'-z': soffit, '-y': fascia, '+x': fascia, '-x': fascia}, bevel=bevel)
    rtop = under(0) + tr / math.cos(th)
    if ridge:
        mb.lathe([(0.0, -ext_along - 0.02), (0.13, -ext_along - 0.02), (0.13, ext_along + 0.02), (0.0, ext_along + 0.02)], 8, ridge,
                 Y * (rtop - 0.05), AL)
    return under, rtop

def gable_wall(mb, at_along, inward, base, plate, top, ridge_axis, mat):
    AL = X if ridge_axis == 'x' else Z
    AC = Z if ridge_axis == 'x' else X
    p0 = AL * at_along
    tri = [p0 + Y * base - AC * plate, p0 + Y * base + AC * plate, p0 + Y * top]
    mb.extrude(tri, AL * inward, mat)

def ibeam(mb, p0, p1, depth, width, mat, up=Y):
    """Steel I-beam between p0 and p1; `up` = direction of the web depth."""
    p0, p1 = Vector(p0), Vector(p1); up = Vector(up).normalized()
    tf = 0.025
    mb.beam(p0, p1, 0.02, depth - 2 * tf, mat, up=up)
    for s in (1, -1):
        off = up * (s * (depth / 2 - tf / 2))
        mb.beam(p0 + off, p1 + off, width, tf, mat, up=up)

# ---------------------------------------------------------------------------
@asset('warehouse')
def build_warehouse():
    S = MB('Structure'); D = MB('Details')
    HX, HZ, T, FY, H = 14.0, 9.0, 0.6, 0.3, 9.0
    TOP = FY + H
    wm = 'wh_wall'
    S.box(-HX, HX, 0, FY, -HZ, HZ, 'concrete_plain', sides={'+y': 'concrete'})
    W = {'+z': Wall('+z', HZ, T), '-z': Wall('-z', HZ, T), '+x': Wall('+x', HX, T), '-x': Wall('-x', HX, T)}
    sd = {'+x': 'steel_dark', '-x': 'steel_dark'}
    for s in ('+z', '-z'):
        w = W[s]
        w.box(S, -HX, -3.5, FY, TOP, -T, 0, wm, sd)
        w.box(S, 3.5, HX, FY, TOP, -T, 0, wm, sd)
        w.box(S, -3.5, 3.5, FY + 5.0, TOP, -T, 0, wm, dict(sd, **{'-y': 'steel_dark'}))
    for s in ('+x', '-x'):
        W[s].box(S, -HZ + T, HZ - T, FY, TOP, -T, 0, wm, sd)
    S.box(-HX - 0.4, HX + 0.4, TOP, TOP + 0.4, -HZ - 0.4, HZ + 0.4, 'steel_dark', sides={'+y': 'wh_roof', '-y': 'galv_plain'})
    # concrete knee wall + corner flashings
    for s, w in W.items():
        half = HX if s in ('+z', '-z') else HZ
        segs = [(-half, -3.5), (3.5, half)] if s in ('+z', '-z') else [(-half, half)]
        for a, b in segs:
            w.box(D, a - (0.06 if a == -half else 0), b + (0.06 if b == half else 0), 0.0, 1.2, 0, 0.06, 'concrete_plain', bevel=0.015)
        for end in (-1, 1):
            a, b = (half - 0.3, half + 0.07) if end > 0 else (-half - 0.07, -half + 0.3)
            w.box(D, a, b, 1.2, TOP, 0, 0.07, 'steel_orange', bevel=0.01)
        w.box(D, -half - 0.07, half + 0.07, TOP - 0.35, TOP, 0, 0.08, 'steel_orange', bevel=0.01)   # eave flashing
    # roller doors (rolled up): housing, guide rails, bump guards, flood lights
    for s in ('+z', '-z'):
        w = W[s]
        w.box(D, -3.95, 3.95, FY + 5.05, FY + 5.95, 0, 0.75, 'steel_dark', bevel=0.04)
        D.lathe([(0.0, -3.9), (0.42, -3.9), (0.42, 3.9), (0.0, 3.9)], 14, 'steel_dark', w.p(0, FY + 5.5, 0.42), w.t)
        for u in (-1, 1):
            w.box(D, 3.5 if u > 0 else -3.66, 3.66 if u > 0 else -3.5, FY, FY + 5.05, 0, 0.14, 'steel_dark', bevel=0.01)
            a, b = (3.66, 3.9) if u > 0 else (-3.9, -3.66)
            w.box(D, a, b, FY, FY + 1.6, 0, 0.12, 'paint_yellow', bevel=0.02)
            wall_lamp(D, w, u * 5.0, FY + 6.4)
        w.box(D, -3.5, 3.5, FY + 5.0, FY + 5.08, 0.05, 0.3, 'steel_dark')    # bottom bar of the rolled-up door
    # interior steel frame
    xs = [-HX + T + 0.15 + k * (2 * HX - 2 * T - 0.3) / 5 for k in range(6)]
    for x in xs:
        for zs in (1, -1):
            if abs(x) < 4.2:
                continue
            z = zs * (HZ - T - 0.16)
            ibeam(D, (x, FY, z), (x, TOP - 0.05, z), 0.3, 0.3, 'steel_dark', up=Z)
        ibeam(D, (x, TOP - 0.25, -HZ + T), (x, TOP - 0.25, HZ - T), 0.45, 0.22, 'steel_dark', up=Y)
    for xsgn in (1, -1):
        for z in (-4.5, 0.0, 4.5):
            x = xsgn * (HX - T - 0.16)
            ibeam(D, (x, FY, z), (x, TOP - 0.05, z), 0.3, 0.3, 'steel_dark', up=X)
    for z in (-4.5, 0.0, 4.5):
        D.box(-HX + T, HX - T, TOP - 0.18, TOP - 0.06, z - 0.05, z + 0.05, 'steel_dark')       # purlins
    # clerestory windows on the long walls, outside and inside
    for s in ('+x', '-x'):
        w = W[s]
        y0, y1 = FY + 5.8, FY + 7.3
        for d0, d1, g0, g1 in ((-0.01, 0.1, -0.01, 0.02), (-T - 0.1, -T + 0.01, -T - 0.02, -T + 0.01)):
            w.box(D, -7.1, 7.1, y0, y1, g0, g1, 'glass')
            w.box(D, -7.2, 7.2, y0 - 0.1, y0, d0, d1, 'steel_dark', bevel=0.01)
            w.box(D, -7.2, 7.2, y1, y1 + 0.1, d0, d1, 'steel_dark', bevel=0.01)
            for k in range(9):
                u = -7.1 + k * 14.2 / 8
                w.box(D, u - 0.05, u + 0.05, y0, y1, d0, d1 - 0.02, 'steel_dark')
    # side personnel door (visual) and downpipes
    w = W['-x']
    w.box(D, 2.0, 3.1, FY, FY + 2.2, 0, 0.04, 'paint_blue', bevel=0.01)
    for (a, b, c, d) in ((1.9, 2.0, FY, FY + 2.3), (3.1, 3.2, FY, FY + 2.3), (1.9, 3.2, FY + 2.2, FY + 2.3)):
        w.box(D, a, b, c, d, 0, 0.07, 'steel_dark')
    w.box(D, 2.15, 2.35, FY + 1.0, FY + 1.05, 0.04, 0.08, 'galv_plain')
    for s in ('+x', '-x'):
        w = W[s]
        for u in (-8.2, 8.2):
            a = w.p(u, 0.15, 0.12); b = w.p(u, TOP - 0.3, 0.12); c2 = w.p(u, TOP - 0.15, 0.32)
            D.tube([a, b, c2], 0.06, 8, 'galv_plain', cap_start=True)
    # rooftop: coping, skylights, turbine vents, AC unit
    RT = TOP + 0.4
    for (x0, x1, z0, z1) in ((-HX - 0.45, HX + 0.45, HZ + 0.25, HZ + 0.45), (-HX - 0.45, HX + 0.45, -HZ - 0.45, -HZ - 0.25),
                             (HX + 0.25, HX + 0.45, -HZ - 0.25, HZ + 0.25), (-HX - 0.45, -HX - 0.25, -HZ - 0.25, HZ + 0.25)):
        D.box(x0, x1, RT - 0.1, RT + 0.35, z0, z1, 'steel_orange', bevel=0.01)
    for x in (-7.0, 7.0):
        D.box(x - 3.5, x + 3.5, RT, RT + 0.25, -1.2, 1.2, 'steel_dark', bevel=0.02)
        D.box(x - 3.35, x + 3.35, RT + 0.25, RT + 0.3, -1.05, 1.05, 'glass')
    for (x, z) in ((-10.5, -5.0), (0.0, -5.0), (10.5, -5.0)):
        D.lathe([(0.0, 0.0), (0.2, 0.0), (0.2, 0.35), (0.38, 0.45), (0.42, 0.75), (0.30, 0.95), (0.06, 1.02), (0.0, 1.02)], 14,
                'galvanised', (x, RT, z), Y)
    D.box(9.0, 11.6, RT, RT + 1.3, 3.0, 5.0, 'trim_white', bevel=0.04)
    for zz in (3.6, 4.4):
        D.cyl((10.3, RT + 1.3, zz), 0.34, 0.03, 'steel_black', segs=14)
    # floor markings
    for x in (-3.6, 3.6):
        D.box(x - 0.06, x + 0.06, FY, FY + 0.003, -HZ + T, HZ - T, 'paint_yellow')
    return [S.build('warehouse_structure'), D.build('warehouse_details')]

# ---------------------------------------------------------------------------
@asset('barn')
def build_barn():
    S = MB('Structure'); D = MB('Details')
    HX, HZ, T, FY, H = 6.0, 5.0, 0.4, 0.3, 6.0
    TOP, CEIL = FY + H, FY + H + 0.3
    ext_in = {'+z': 'barn_red', '-z': 'wood_weathered'}
    S.box(-HX, HX, 0, FY, -HZ, HZ, 'stone_dark', sides={'+y': 'wood_weathered'})
    W = {'+z': Wall('+z', HZ, T), '-z': Wall('-z', HZ, T), '+x': Wall('+x', HX, T), '-x': Wall('-x', HX, T)}
    W['-z'].box(S, -HX, HX, FY, TOP, -T, 0, 'barn_red', ext_in)
    W['+x'].box(S, -HZ + T, HZ - T, FY, TOP, -T, 0, 'barn_red', ext_in)
    W['-x'].box(S, -HZ + T, HZ - T, FY, TOP, -T, 0, 'barn_red', ext_in)
    fw = W['+z']
    fw.box(S, -HX, -2.25, FY, TOP, -T, 0, 'barn_red', ext_in)
    fw.box(S, 2.25, HX, FY, TOP, -T, 0, 'barn_red', ext_in)
    fw.box(S, -2.25, 2.25, FY + 3.8, TOP, -T, 0, 'barn_red', ext_in)
    S.box(-HX - 0.3, HX + 0.3, TOP, CEIL, -HZ - 0.3, HZ + 0.3, 'trim_white', sides={'-y': 'wood_weathered'})
    for s, w in W.items():
        half = HX if s in ('+z', '-z') else HZ
        segs = [(-half, -2.25), (2.25, half)] if s == '+z' else [(-half, half)]
        for a, b in segs:
            w.box(D, a - (0.05 if a == -half else 0), b + (0.05 if b == half else 0), 0.0, 0.55, 0, 0.05, 'stone', bevel=0.015)
        for end in (-1, 1):
            a, b = (half - 0.2, half + 0.04) if end > 0 else (-half - 0.04, -half + 0.2)
            w.box(D, a, b, 0.55, TOP, 0, 0.04, 'trim_white', bevel=0.008)
    # door opening trim
    for d0, d1 in ((-0.01, 0.06), (-T - 0.05, -T + 0.01)):
        fw.box(D, -2.45, -2.25, FY, FY + 3.8, d0, d1, 'trim_white', bevel=0.01)
        fw.box(D, 2.25, 2.45, FY, FY + 3.8, d0, d1, 'trim_white', bevel=0.01)
        fw.box(D, -2.45, 2.45, FY + 3.8, FY + 4.02, d0, d1, 'trim_white', bevel=0.01)
    # two sliding doors, slid open to the sides, with white frames and X braces
    fw.box(D, -4.95, 4.95, FY + 4.12, FY + 4.24, 0.06, 0.2, 'steel_black', bevel=0.01)
    for sgn in (-1, 1):
        u0, u1 = (2.5, 4.75) if sgn > 0 else (-4.75, -2.5)
        y0, y1 = FY + 0.05, FY + 3.95
        fw.box(D, u0, u1, y0, y1, 0.07, 0.13, 'barn_red', bevel=0.01)
        for (a, b, c, d) in ((u0, u0 + 0.16, y0, y1), (u1 - 0.16, u1, y0, y1), (u0, u1, y0, y0 + 0.16), (u0, u1, y1 - 0.16, y1),
                             (u0, u1, (y0 + y1) / 2 - 0.08, (y0 + y1) / 2 + 0.08)):
            fw.box(D, a, b, c, d, 0.13, 0.17, 'trim_white', bevel=0.008)
        for (ya, yb) in ((y0 + 0.16, (y0 + y1) / 2 - 0.08), ((y0 + y1) / 2 + 0.08, y1 - 0.16)):
            for flip in (0, 1):
                pa = fw.p(u0 + 0.16, ya if not flip else yb, 0.15); pb = fw.p(u1 - 0.16, yb if not flip else ya, 0.15)
                D.beam(pa, pb, 0.035, 0.13, 'trim_white', up=fw.n.cross(pb - pa))
        for uu in (u0 + 0.35, u1 - 0.35):
            fw.box(D, uu - 0.04, uu + 0.04, y1 - 0.05, FY + 4.2, 0.13, 0.17, 'steel_black')
    # windows (white 4-pane)
    st = dict(trim='trim_white', sill='trim_white')
    for s, us in (('+x', [-2.5, 2.5]), ('-x', [-2.5, 2.5]), ('-z', [-3.0, 3.0])):
        for u in us:
            window(D, W[s], u, FY + 2.6, 1.0, 1.1, st)
    # gable roof, ridge along z, corrugated tin
    under, rtop = gable_roof(D, CEIL, HX + 0.3, HX + 0.65, HZ + 0.55, None, 38, 'z', 'roof_tin', 'wood_weathered', 'trim_white',
                             tr=0.14)
    D.lathe([(0.0, -HZ - 0.57), (0.16, -HZ - 0.57), (0.16, HZ + 0.57), (0.0, HZ + 0.57)], 6, 'galv_plain', (0, rtop - 0.07, 0), Z)
    for s in (1, -1):
        gable_wall(D, s * HZ, -s * T, CEIL, HX + 0.3, under(0) + 0.02, 'z', 'barn_red')
    # hay loft door + hoist beam on the front gable
    lw = W['+z']
    y0, y1 = CEIL + 0.6, CEIL + 2.6
    lw.box(D, -1.0, 1.0, y0, y1, 0, 0.06, 'barn_red', bevel=0.01)
    for (a, b, c, d) in ((-1.15, -1.0, y0, y1), (1.0, 1.15, y0, y1), (-1.15, 1.15, y0 - 0.15, y0), (-1.15, 1.15, y1, y1 + 0.15)):
        lw.box(D, a, b, c, d, 0, 0.09, 'trim_white', bevel=0.008)
    for flip in (0, 1):
        pa = lw.p(-0.95, y0 + 0.05 if not flip else y1 - 0.05, 0.08); pb = lw.p(0.95, y1 - 0.05 if not flip else y0 + 0.05, 0.08)
        D.beam(pa, pb, 0.035, 0.12, 'trim_white', up=lw.n.cross(pb - pa))
    D.box(-0.12, 0.12, rtop - 0.85, rtop - 0.6, HZ - 0.5, HZ + 1.3, 'wood_dark', bevel=0.015)
    D.tube([(0, rtop - 0.85, HZ + 1.15), (0, rtop - 2.2, HZ + 1.15)], 0.015, 6, 'rope')
    D.lathe([(0.0, 0.0), (0.08, 0.0), (0.08, 0.06), (0.0, 0.06)], 10, 'steel_black', (0, rtop - 0.9, HZ + 1.15), Z)
    # cupola on the ridge
    cy = under(0) + 0.1
    D.box(-0.6, 0.6, cy - 0.4, cy + 1.0, -0.6, 0.6, 'trim_white', bevel=0.02)
    for sgn in (1, -1):
        for k in range(5):
            yy = cy + 0.2 + k * 0.14
            D.box(-0.45, 0.45, yy, yy + 0.05, min(sgn * 0.6, sgn * 0.64), max(sgn * 0.6, sgn * 0.64), 'wood_dark')
            D.box(min(sgn * 0.6, sgn * 0.64), max(sgn * 0.6, sgn * 0.64), yy, yy + 0.05, -0.45, 0.45, 'wood_dark')
    D.lathe([(0.0, 0.0), (1.0, 0.0), (1.0, 0.08), (0.0, 0.85)], 4, 'roof_tin', (0, cy + 1.0, 0), Y, start=math.pi / 4, smooth=False)
    D.tube([(0, cy + 1.8, 0), (0, cy + 2.6, 0)], 0.02, 6, 'steel_black')
    D.beam((-0.45, cy + 2.4, 0), (0.45, cy + 2.4, 0), 0.02, 0.02, 'steel_black')
    D.obox((0.38, cy + 2.4, 0), (0.1, 0.12, 0.008), (X, Y, Z), 'steel_black')
    D.lathe([(0.0, 0.0), (0.05, 0.0), (0.0, 0.12)], 6, 'steel_black', (-0.5, cy + 2.4, 0), X)
    return [S.build('barn_structure'), D.build('barn_details')]

# ---------------------------------------------------------------------------
@asset('silo')
def build_silo():
    D = MB('Silo')
    R, H = 2.3, 11.0
    D.lathe([(0.0, 0.0), (2.5, 0.0), (2.5, 0.3), (2.42, 0.35), (0.0, 0.35)], 32, 'concrete_plain', (0, 0, 0), Y, smooth=False)
    prof = [(R, 0.3)]
    y = 0.3
    for k in range(7):
        y0 = 1.4 + k * 1.45
        prof += [(R, y0 - 0.05), (R + 0.04, y0), (R + 0.04, y0 + 0.08), (R, y0 + 0.13)]
    prof += [(R, H - 0.1), (R + 0.08, H - 0.08), (R + 0.08, H)]
    D.lathe(prof, 32, 'galv_rings', (0, 0, 0), Y)
    dome = [(R + 0.12, H)]
    for k in range(1, 8):
        t = k / 8 * math.radians(78)
        dome.append(((R + 0.12) * math.cos(t), H + 1.9 * math.sin(t)))
    dome += [(0.42, H + 1.86), (0.42, H + 2.1), (0.62, H + 2.12), (0.0, H + 2.42)]
    D.lathe(dome, 32, 'galvanised', (0, 0, 0), Y)
    for k in range(16):   # dome seams
        a = 2 * math.pi * k / 16
        pts = []
        for j in range(8):
            t = j / 7 * math.radians(76)
            r = (R + 0.14) * math.cos(t); pts.append((r * math.cos(a), H + 0.02 + 1.92 * math.sin(t), r * math.sin(a)))
        D.tube(pts, 0.025, 4, 'galv_plain', cap_end=False)
    # ladder with safety cage on the +x side
    lx = R + 0.22
    for zz in (-0.24, 0.24):
        D.box(lx - 0.03, lx + 0.03, 0.35, H + 1.2, zz - 0.03, zz + 0.03, 'galv_plain')
        for yy in (1.0, 4.0, 7.0, 10.0):
            D.box(R - 0.05, lx, yy, yy + 0.05, zz - 0.02, zz + 0.02, 'galv_plain')
    for k in range(int((H + 0.9) / 0.3)):
        yy = 0.6 + k * 0.3
        D.box(lx - 0.016, lx + 0.016, yy, yy + 0.032, -0.24, 0.24, 'galv_plain')
    for k in range(8):
        yy = 2.6 + k * 1.2
        pts = [(lx + 0.38 * math.sin(t), yy, 0.38 * math.cos(t)) for t in np.linspace(0, math.pi, 8)]
        D.tube(pts, 0.018, 4, 'galv_plain', cap_end=False)
    for t in (math.pi * 0.25, math.pi * 0.5, math.pi * 0.75):
        D.box(lx + 0.38 * math.sin(t) - 0.02, lx + 0.38 * math.sin(t) + 0.02, 2.6, 11.0, 0.38 * math.cos(t) - 0.02, 0.38 * math.cos(t) + 0.02, 'galv_plain')
    # fill pipe on the -x side
    D.tube([(-R - 0.25, 0.35, 0.6), (-R - 0.25, H + 0.3, 0.6), (-R + 0.2, H + 1.0, 0.45)], 0.11, 10, 'galv_plain', cap_start=True)
    for yy in (2.0, 5.5, 9.0):
        D.box(-R - 0.3, -R + 0.05, yy, yy + 0.08, 0.55, 0.65, 'galv_plain')
    # small access hatch near the bottom
    D.box(-0.45, 0.45, 0.6, 1.5, R - 0.06, R + 0.05, 'galvanised', bevel=0.02)
    D.cyl((0.3, 1.05, R + 0.05), 0.04, 0.06, 'steel_black', segs=6, axis=Z)
    return [D.build('silo')]

# =============================================================================
# 5. PROPS
# =============================================================================
def build_container(color):
    D = MB('Container')
    m = 'container_' + color
    HX, HZ, H = 1.6, 5.0, 3.2
    D.box(-HX + 0.03, HX - 0.03, 0.12, H - 0.1, -HZ + 0.05, HZ - 0.2, m, sides={'+z': m})
    for sx in (-1, 1):
        for sz in (-1, 1):   # corner posts and castings
            x0, x1 = (HX - 0.16, HX) if sx > 0 else (-HX, -HX + 0.16)
            z0, z1 = (HZ - 0.16, HZ) if sz > 0 else (-HZ, -HZ + 0.16)
            D.box(x0, x1, 0.12, H - 0.12, z0, z1, m, bevel=0.01)
            for y0 in (0.0, H - 0.12):
                D.box(x0, x1, y0, y0 + 0.12, z0, z1, 'steel_dark', bevel=0.01)
    for sx in (-1, 1):       # long rails
        for (y0, y1) in ((0.0, 0.16), (H - 0.14, H)):
            x0, x1 = (HX - 0.1, HX) if sx > 0 else (-HX, -HX + 0.1)
            D.box(x0, x1, y0, y1, -HZ + 0.16, HZ - 0.16, m, bevel=0.01)
    for sz in (-1, 1):       # end rails
        for (y0, y1) in ((0.0, 0.2), (H - 0.16, H)):
            z0, z1 = (HZ - 0.1, HZ) if sz > 0 else (-HZ, -HZ + 0.1)
            D.box(-HX + 0.16, HX - 0.16, y0, y1, z0, z1, m, bevel=0.01)
    for (z0, z1) in ((-2.6, -1.8), (1.8, 2.6)):     # fork pockets
        for sx in (-1, 1):
            x0, x1 = (HX - 0.02, HX + 0.005) if sx > 0 else (-HX - 0.005, -HX + 0.02)
            D.box(x0, x1, 0.02, 0.14, z0, z1, 'steel_dark')
    # doors on +z: two leaves, 4 locking bars with handles
    zd = HZ - 0.2
    D.box(-HX + 0.16, -0.01, 0.2, H - 0.16, zd, zd + 0.04, m, sides={'+z': m}, bevel=0.01)
    D.box(0.01, HX - 0.16, 0.2, H - 0.16, zd, zd + 0.04, m, sides={'+z': m}, bevel=0.01)
    for x in (-1.05, -0.45, 0.45, 1.05):
        D.cyl((x, 0.25, zd + 0.1), 0.028, H - 0.5, 'galv_plain', segs=6)
        for y in (0.25, H - 0.27):
            D.box(x - 0.06, x + 0.06, y - 0.03, y + 0.06, zd + 0.04, zd + 0.13, 'steel_dark')
        D.beam((x, 1.35, zd + 0.1), (x + (0.35 if x < 0 else -0.35), 1.25, zd + 0.13), 0.04, 0.04, 'galv_plain')
    for y in (0.5, 1.6, 2.7):
        for x in (-HX + 0.16, HX - 0.16):
            D.box(x - 0.1, x + 0.1, y - 0.08, y + 0.08, zd + 0.02, zd + 0.08, 'steel_dark')
    return [D.build('container')]

for _c in ('white', 'red', 'blue', 'green', 'orange'):
    ASSETS['container' if _c == 'white' else 'container_' + _c] = (lambda c: (lambda: build_container(c)))(_c)

@asset('crate')
def build_crate():
    D = MB('Crate')
    h = 1.0; e = 0.07
    D.box(-h + e, h - e, e, 2 * h - e, -h + e, h - e, 'wood_crate')
    for (a, b) in ((-h, -h + 0.16), (h - 0.16, h)):
        for (c, d) in ((-h, -h + 0.16), (h - 0.16, h)):
            D.box(a, b, 0.0, 2 * h, c, d, 'wood_crate', bevel=0.012)            # vertical edges
            D.box(-h + 0.16, h - 0.16, 0.0, 0.16, c, d, 'wood_crate', bevel=0.012) if a < 0 else None
            D.box(-h + 0.16, h - 0.16, 2 * h - 0.16, 2 * h, c, d, 'wood_crate', bevel=0.012) if a < 0 else None
            D.box(a, b, 0.0, 0.16, -h + 0.16, h - 0.16, 'wood_crate', bevel=0.012) if c < 0 else None
            D.box(a, b, 2 * h - 0.16, 2 * h, -h + 0.16, h - 0.16, 'wood_crate', bevel=0.012) if c < 0 else None
    for side in ('+z', '-z', '+x', '-x'):     # diagonal braces
        w = Wall(side, h - e, 0)
        k = 1 if side in ('+z', '-x') else -1
        pa = w.p(-0.8 * k, 0.2, 0.025); pb = w.p(0.8 * k, 1.8, 0.025)
        D.beam(pa, pb, 0.05, 0.16, 'wood_crate', up=w.n.cross(pb - pa), bevel=0.01)
    for side in ('+y',):
        D.beam((-0.8, 2 * h - e + 0.025, -0.8), (0.8, 2 * h - e + 0.025, 0.8), 0.16, 0.05, 'wood_crate', up=Y, bevel=0.01)
    return [D.build('crate')]

@asset('hay_bale')
def build_hay_bale():
    D = MB('HayBale')
    c = (0, 0.9, 0)
    D.lathe([(0.80, -0.8), (0.86, -0.785), (0.895, -0.74), (0.9, -0.65), (0.9, 0.65), (0.895, 0.74), (0.86, 0.785), (0.80, 0.8)],
            24, 'straw', c, X, cap0=True, cap1=True, cap_mat='straw_end')
    for x in (-0.45, 0.0, 0.45):
        D.lathe([(0.902, x - 0.012), (0.912, x), (0.902, x + 0.012)], 24, 'rope', c, X)
    return [D.build('hay_bale')]

@asset('tractor')
def build_tractor():
    D = MB('Tractor')
    P = 'tractor_paint'
    # chassis, engine, hood, grille, front weight
    D.box(-0.32, 0.32, 0.42, 0.78, -1.1, 1.7, 'steel_dark', bevel=0.02)
    D.box(-0.42, 0.42, 0.78, 1.42, -0.15, 1.68, P, bevel=0.07)
    D.box(-0.40, 0.40, 0.8, 1.38, 1.68, 1.76, 'steel_black', bevel=0.02)
    for k in range(6):
        y = 0.86 + k * 0.09
        D.box(-0.36, 0.36, y, y + 0.035, 1.76, 1.79, 'galv_plain')
    D.box(-0.5, 0.5, 0.45, 0.72, 1.7, 1.86, 'steel_black', bevel=0.03)
    for x in (-0.3, 0.3):
        D.cyl((x, 1.25, 1.75), 0.075, 0.06, 'galv_plain', segs=10, axis=Z, cap_mat='lamp_glow')
    D.tube([(0.28, 1.35, 1.25), (0.28, 2.05, 1.25)], 0.05, 8, 'steel_black', cap_end=True)
    D.lathe([(0.055, 0), (0.07, 0.05), (0.07, 0.12), (0.0, 0.14)], 8, 'steel_black', (0.28, 2.05, 1.25), Y)
    D.box(-0.42, 0.42, 1.42, 1.48, -0.15, 0.1, P, bevel=0.02)
    # cab: floor, rear body, seat, posts, roof, glass
    D.box(-0.75, 0.75, 0.78, 0.92, -1.45, -0.1, 'steel_dark', bevel=0.02)
    D.box(-0.55, 0.55, 0.92, 1.25, -1.45, -1.0, P, bevel=0.04)
    D.box(-0.3, 0.3, 0.98, 1.12, -0.95, -0.45, 'rubber_plain', bevel=0.04)
    D.box(-0.3, 0.3, 1.12, 1.62, -1.0, -0.88, 'rubber_plain', bevel=0.04)
    D.tube([(0, 1.1, -0.2), (0, 1.55, -0.45)], 0.025, 6, 'steel_black')
    D.tube([(0.22 * math.cos(a), 1.55 + 0.22 * math.sin(a) * 0.45, -0.45 - 0.22 * math.sin(a) * 0.89) for a in np.linspace(0, 2 * math.pi, 13)],
           0.022, 5, 'rubber_plain', cap_end=False)
    for (x, z) in ((-0.72, -1.42), (0.72, -1.42), (-0.72, -0.12), (0.72, -0.12)):
        D.box(x - 0.04, x + 0.04, 0.92, 2.1, z - 0.04, z + 0.04, 'steel_black')
    D.box(-0.85, 0.85, 2.1, 2.2, -1.55, 0.0, P, bevel=0.04)
    D.box(-0.68, 0.68, 1.25, 2.08, -0.13, -0.11, 'glass')
    D.box(-0.68, 0.68, 1.3, 2.08, -1.43, -1.41, 'glass')
    for x in (-0.71, 0.71):
        D.box(x - 0.01, x + 0.01, 1.3, 2.08, -1.38, -0.16, 'glass')
    D.box(-0.25, 0.25, 0.3, 0.45, -1.58, -1.45, 'steel_black', bevel=0.02)   # hitch
    # wheels: rear big, front small, with fenders
    def wheel(x, z, r, w, segs):
        s = 1 if x > 0 else -1
        c = Vector((x, r, z))
        rr = r * 0.62
        prof = [(rr, -w / 2), (r * 0.9, -w / 2), (r * 0.97, -w / 2 + 0.03), (r, -w / 2 + 0.09), (r, w / 2 - 0.09), (r * 0.97, w / 2 - 0.03),
                (r * 0.9, w / 2), (rr, w / 2)]
        plen = sum(math.hypot(prof[i][0] - prof[i - 1][0], prof[i][1] - prof[i - 1][1]) for i in range(1, len(prof)))
        D.lathe(prof, segs, 'rubber', c, X, tile=plen)
        D.lathe([(rr, -w / 2 + 0.02), (rr * 0.8, -w / 2 + 0.06), (rr * 0.8, w / 2 - 0.06), (rr, w / 2 - 0.02)], segs, 'rim_paint', c, X,
                cap0=True, cap1=True)
        D.lathe([(0.0, w / 2 - 0.04), (rr * 0.32, w / 2 - 0.04), (rr * 0.25, w / 2 + 0.04), (0.0, w / 2 + 0.06)], 10, 'galv_plain',
                c if s > 0 else c, X if s > 0 else -X)
    for s in (-1, 1):
        wheel(s * 0.9, -0.75, 0.8, 0.45, 20)
        wheel(s * 0.78, 1.22, 0.5, 0.28, 16)
        D.lathe([(0.9, -0.25), (0.98, -0.25), (0.98, 0.25), (0.9, 0.25), (0.9, -0.25)], 10, P, (s * 0.9, 0.8, -0.75), X,
                start=0.5, sweep=math.pi - 0.65, smooth=False)
        D.box(min(s * 0.42, s * 0.72), max(s * 0.42, s * 0.72), 0.95, 1.0, -1.35, -0.15, 'steel_dark', bevel=0.01)
    return [D.build('tractor')]

@asset('well')
def build_well():
    D = MB('Well')
    D.lathe([(1.24, 0.0), (1.24, 0.95), (0.98, 0.95), (0.98, 0.35)], 20, 'stone', (0, 0, 0), Y, smooth=False)
    D.lathe([(1.3, 0.93), (1.3, 1.1), (0.94, 1.1), (0.94, 0.93)], 20, 'stone_dark', (0, 0, 0), Y, smooth=False)
    D.lathe([(0.99, 0.42), (0.0, 0.42)], 20, 'glass', (0, 0, 0), Y, smooth=False)
    for x in (-1.1, 1.1):
        D.box(x - 0.1, x + 0.1, 0.95, 2.95, -0.1, 0.1, 'wood_weathered', bevel=0.015)
        D.beam((x, 2.5, 0.0), (x, 2.95, 0.55), 0.08, 0.08, 'wood_weathered')
        D.beam((x, 2.5, 0.0), (x, 2.95, -0.55), 0.08, 0.08, 'wood_weathered')
    D.box(-1.4, 1.4, 2.95, 3.08, -0.12, 0.12, 'wood_weathered', bevel=0.015)
    D.cyl((-1.0, 2.15, 0), 0.09, 2.0, 'wood_dark', segs=10, axis=X)
    D.lathe([(0.0, -0.3), (0.13, -0.3), (0.135, 0.0), (0.13, 0.3), (0.0, 0.3)], 12, 'rope', (0, 2.15, 0), X)
    D.beam((1.15, 2.15, 0), (1.4, 2.15, 0), 0.03, 0.03, 'steel_black')
    D.beam((1.4, 2.15, 0), (1.4, 1.85, 0.0), 0.03, 0.03, 'steel_black')
    D.tube([(1.4, 1.85, 0.0), (1.55, 1.85, 0.0)], 0.03, 6, 'wood_dark')
    D.tube([(0.0, 2.02, 0.0), (0.0, 1.4, 0.0)], 0.012, 5, 'rope')
    D.lathe([(0.0, 1.05), (0.16, 1.05), (0.19, 1.4), (0.17, 1.4), (0.145, 1.08), (0.0, 1.08)], 12, 'wood_dark', (0, 0, 0), Y)
    D.tube([(-0.18, 1.4, 0), (0, 1.48, 0), (0.18, 1.4, 0)], 0.008, 4, 'steel_black')
    gable_roof(D, 3.0, 1.15, 1.15, 1.6, None, 36, 'x', 'clay_tiles', 'wood_dark', 'wood_dark', tr=0.08, ridge='clay_tiles')
    return [D.build('well')]

@asset('lamp_post')
def build_lamp_post():
    D = MB('LampPost')
    m = 'steel_black'
    D.lathe([(0.24, 0.0), (0.24, 0.12), (0.2, 0.16), (0.17, 0.5), (0.13, 0.56), (0.13, 0.62), (0.09, 0.7)], 8, m, (0, 0, 0), Y, smooth=False,
            start=math.pi / 8)
    D.lathe([(0.075, 0.7), (0.055, 3.55), (0.08, 3.6), (0.08, 3.66), (0.05, 3.7)], 12, m, (0, 0, 0), Y)
    for y in (1.2, 2.6):
        D.lathe([(0.0, y), (0.085, y), (0.095, y + 0.04), (0.085, y + 0.08), (0.0, y + 0.08)], 12, m, (0, 0, 0), Y)
    D.lathe([(0.12, 3.7), (0.16, 3.78), (0.16, 3.81), (0.0, 3.81)], 4, m, (0, 0, 0), Y, start=math.pi / 4, smooth=False)
    for k in range(4):
        a = math.pi / 4 + k * math.pi / 2
        x, z = math.cos(a) * 0.2, math.sin(a) * 0.2
        D.box(x - 0.015, x + 0.015, 3.81, 4.16, z - 0.015, z + 0.015, m)
    D.lathe([(0.0, 3.82), (0.14, 3.82), (0.19, 4.15), (0.0, 4.15)], 4, 'lamp_glow', (0, 0, 0), Y, start=math.pi / 4, smooth=False)
    D.lathe([(0.0, 4.15), (0.27, 4.15), (0.27, 4.18), (0.06, 4.3), (0.03, 4.3), (0.03, 4.34), (0.0, 4.4)], 4, m, (0, 0, 0), Y,
            start=math.pi / 4, smooth=False)
    return [D.build('lamp_post')]

@asset('fence')
def build_fence():
    """4 m post-and-rail section from x=-2 to x=+2. Posts at x=-2 and x=0; the next section's
    first post closes the joint, so sections tile seamlessly in a row."""
    D = MB('Fence')
    rng = random.Random(7)
    for x in (-1.94, 0.0):
        D.box(x - 0.065, x + 0.065, 0.0, 1.04, -0.065, 0.065, 'wood_weathered', bevel=0.012)
        D.lathe([(0.065, 1.04), (0.0, 1.1)], 4, 'wood_weathered', (x, 0, 0), Y, start=math.pi / 4, smooth=False)
    for y in (0.3, 0.65, 0.98):
        d0 = rng.uniform(-0.02, 0.02); d1 = rng.uniform(-0.02, 0.02)
        D.beam((-2.0, y + d0, 0.085), (2.0, y + d1, 0.085), 0.04, 0.13, 'wood_weathered', bevel=0.008)
    return [D.build('fence')]

@asset('chest')
def build_chest():
    """Loot chest 1.4 x 0.7 x 0.9 body. The lid is a separate child object `Lid` whose origin is
    the hinge on the back top edge (rotate it about its local X axis to open)."""
    B = MB('Chest')
    W2, H, D2 = 0.7, 0.7, 0.45
    B.box(-W2 + 0.02, W2 - 0.02, 0.02, H, -D2 + 0.02, D2 - 0.02, 'wood_chest', sides={'+y': 'wood_dark'})
    for sx in (-1, 1):
        for sz in (-1, 1):
            x0, x1 = (W2 - 0.1, W2) if sx > 0 else (-W2, -W2 + 0.1)
            z0, z1 = (D2 - 0.1, D2) if sz > 0 else (-D2, -D2 + 0.1)
            B.box(x0, x1, 0.0, H, z0, z1, 'gold', bevel=0.01)
            B.box(x0 - 0.01, x1 + 0.01, 0.0, 0.06, z0 - 0.01, z1 + 0.01, 'gold', bevel=0.01)
    B.box(-W2 - 0.005, W2 + 0.005, H - 0.07, H, -D2 - 0.005, D2 + 0.005, 'gold', bevel=0.01)
    B.box(-W2 - 0.005, W2 + 0.005, 0.0, 0.07, -D2 - 0.005, D2 + 0.005, 'gold', bevel=0.01)
    for x in (-0.33, 0.33):
        B.box(x - 0.045, x + 0.045, 0.0, H, -D2 - 0.004, D2 + 0.004, 'gold', bevel=0.003)
    B.box(-0.14, 0.14, H - 0.32, H - 0.02, D2, D2 + 0.012, 'gold', bevel=0.004)
    B.cyl((0, H - 0.2, D2 + 0.012), 0.025, 0.003, 'steel_black', segs=8, axis=Z)
    body = B.build('Chest')
    # lid in hinge-local space (hinge at back top edge: x along, z towards the front)
    L = MB('Lid')
    arch = []
    for k in range(13):
        t = math.pi * k / 12
        arch.append((0.0, 0.1 + 0.2 * math.sin(t), D2 - D2 * math.cos(t)))
    poly = [(0.0, 0.0, 0.0), (0.0, 0.0, 2 * D2)] + list(reversed(arch))
    L.extrude([(-W2 + 0.02, y, z) for (_, y, z) in poly], (2 * W2 - 0.04, 0, 0), 'wood_chest')
    for x in (-W2, W2 - 0.1, -0.33 - 0.045, 0.33 - 0.045):
        wdt = 0.1 if abs(x) > 0.5 else 0.09
        big = [(0.0, -0.005, -0.01), (0.0, -0.005, 2 * D2 + 0.01)] + [(0.0, y + 0.012 * math.sin(math.pi * k / 12) + 0.006, z + (0.01 if z > D2 else -0.01) * abs(math.cos(math.pi * k / 12)))
                                                                for k, (_, y, z) in reversed(list(enumerate(arch)))]
        L.extrude([(x, y, z) for (_, y, z) in big], (wdt, 0, 0), 'gold')
    L.box(-W2 - 0.005, W2 + 0.005, 0.0, 0.07, -0.005, 2 * D2 + 0.005, 'gold', bevel=0.01)
    L.box(-0.09, 0.09, -0.16, 0.12, 2 * D2, 2 * D2 + 0.04, 'gold', bevel=0.01)       # hasp
    for x in (-0.45, 0.45):        # hinges
        L.cyl((x - 0.06, 0.0, -0.01), 0.022, 0.12, 'gold', segs=8, axis=X)
    lid = L.build('Lid')
    lid.parent = body
    lid.location = G((0, H, -D2))
    return [body]

@asset('grass_clump')
def build_grass_clump():
    D = MB('Grass')
    rng = random.Random(3)
    for k in range(4):
        a = k * math.pi / 4 + rng.uniform(-0.2, 0.2)
        n = Vector((math.cos(a), 0, math.sin(a)))
        D.card((rng.uniform(-0.05, 0.05), -0.02, rng.uniform(-0.05, 0.05)), n, Y, 0.62, 0.55, 'grass_card', crown=(0, -0.6, 0), bend=0.04)
    return [D.build('grass_clump')]

# =============================================================================
# 6. NATURE
# =============================================================================
def _limb(D, rng, p0, d, length, r0, r1, mat, segs, bend=0.25, gravity=0.0, n=4):
    pts = [Vector(p0)]; rad = []
    d = Vector(d).normalized()
    for i in range(1, n + 1):
        d = (d + Vector((rng.uniform(-bend, bend), rng.uniform(-bend, bend) - gravity, rng.uniform(-bend, bend)))).normalized()
        pts.append(pts[-1] + d * (length / n))
    rad = [r0 + (r1 - r0) * i / n for i in range(n + 1)]
    D.tube(pts, rad, segs, mat, cap_end=False)
    return pts

def _cluster(D, rng, c, size, mat, crown, count=3, up_bias=0.4, quad=None):
    for k in range(count):
        nrm = Vector((rng.uniform(-1, 1), rng.uniform(-0.3, 1) + up_bias, rng.uniform(-1, 1)))
        if nrm.length < 1e-3:
            nrm = Vector((0, 1, 0))
        upv = Vector((rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)))
        if abs(upv.normalized().dot(nrm.normalized())) > 0.9:
            upv = X
        q = rng.randrange(4) if quad is None else quad
        u0, v0 = (q % 2) * 0.5, (q // 2) * 0.5
        s = size * rng.uniform(0.8, 1.2)
        D.card(Vector(c) - upv.normalized() * s * 0.45, nrm, upv, s, s, mat, (u0, v0, u0 + 0.5, v0 + 0.5), crown=crown,
               bend=s * 0.12)

def _trunk(D, rng, h, r0, r1, mat, segs=10, lean=0.15, flare=1.5):
    pts, rad = [], []
    n = 7
    off = Vector((0, 0, 0))
    for i in range(n + 1):
        t = i / n
        if i > 0:
            off += Vector((rng.uniform(-lean, lean), 0, rng.uniform(-lean, lean))) * (h / n) * 0.35
        pts.append(Vector((off.x, h * t, off.z)))
        rr = r0 + (r1 - r0) * t
        if t < 0.15:
            rr *= 1 + (flare - 1) * (1 - t / 0.15) ** 2
        rad.append(rr)
    pts[0] = Vector((0, -0.15, 0))
    D.tube(pts, rad, segs, mat, cap_end=True)
    return pts, rad

@asset('tree_oak')
def build_oak():
    rng = random.Random(11); D = MB('Oak')
    pts, rad = _trunk(D, rng, 3.0, 0.38, 0.24, 'bark_oak', 12, 0.12, 1.6)
    crown = Vector((0, 4.9, 0))
    tips = []
    for k in range(6):
        a = k / 6 * 2 * math.pi + rng.uniform(-0.3, 0.3)
        p0 = pts[-1] + Vector((0, rng.uniform(-0.7, 0), 0))
        d = Vector((math.cos(a), rng.uniform(0.55, 1.1), math.sin(a)))
        lp = _limb(D, rng, p0, d, rng.uniform(2.4, 3.2), 0.17, 0.06, 'bark_oak', 7, 0.25, 0.05)
        tips.append(lp[-1])
        for j in range(2):
            q0 = lp[rng.randint(2, 3)]
            d2 = Vector((math.cos(a + rng.uniform(-1, 1)), rng.uniform(0.2, 0.9), math.sin(a + rng.uniform(-1, 1))))
            sp = _limb(D, rng, q0, d2, rng.uniform(1.2, 1.8), 0.06, 0.025, 'bark_oak', 5, 0.3, 0.1, n=3)
            tips.append(sp[-1])
    leaf = 0
    rx, ry = 3.1, 2.0
    while leaf < 850:
        v = Vector((rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1))).normalized()
        r = rng.uniform(0.55, 1.0) ** 0.5
        c = crown + Vector((v.x * rx * r, v.y * ry * r, v.z * rx * r))
        if c.y < 2.9:
            continue
        _cluster(D, rng, c, 1.05, 'leaves_oak', crown, count=1)
        leaf += 1
    for t in tips:
        _cluster(D, rng, t, 1.2, 'leaves_oak', crown, count=3)
    return [D.build('tree_oak', fit_height=7.6)]

@asset('tree_birch')
def build_birch():
    rng = random.Random(12); D = MB('Birch')
    pts, rad = _trunk(D, rng, 7.6, 0.17, 0.035, 'bark_birch', 10, 0.18, 1.3)
    crown = Vector((pts[5].x, 5.4, pts[5].z))
    for k in range(14):
        i = rng.randint(2, 6); t = rng.random()
        p0 = pts[i].lerp(pts[i + 1], t)
        a = rng.uniform(0, 2 * math.pi)
        d = Vector((math.cos(a), rng.uniform(0.6, 1.4), math.sin(a)))
        lp = _limb(D, rng, p0, d, rng.uniform(1.2, 2.2) * (1.2 - p0.y / 9), 0.04, 0.012, 'bark_birch', 5, 0.25, 0.25, n=3)
        for q in lp[1:]:
            _cluster(D, rng, q + Vector((0, -0.3, 0)), 0.95, 'leaves_birch', crown, count=2, up_bias=-0.2)
    leaf = 0
    while leaf < 760:
        v = Vector((rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1))).normalized()
        r = rng.uniform(0.45, 1.0) ** 0.5
        c = crown + Vector((v.x * 1.9 * r, v.y * 2.6 * r, v.z * 1.9 * r))
        if c.y < 2.8:
            continue
        _cluster(D, rng, c, 0.78, 'leaves_birch', crown, count=1, up_bias=-0.1)
        leaf += 1
    return [D.build('tree_birch', fit_height=7.8)]

def _pine(name, seed, h, r0, base_y, top_frac, mat, bark, whorl_dy, spread, crown_shape):
    rng = random.Random(seed); D = MB(name)
    pts, rad = _trunk(D, rng, h, r0, 0.03, bark, 10, 0.05, 1.35)
    y = base_y
    while y < h - 0.35:
        t = (y - base_y) / (h - base_y)
        L = crown_shape(t) * spread
        nb = rng.randint(5, 7)
        a0 = rng.uniform(0, 2 * math.pi)
        tp = pts[min(int(y / h * 7), 6)].lerp(pts[min(int(y / h * 7) + 1, 7)], (y / h * 7) % 1)
        for b in range(nb):
            a = a0 + b * 2 * math.pi / nb + rng.uniform(-0.25, 0.25)
            d = Vector((math.cos(a), rng.uniform(-0.05, 0.25), math.sin(a)))
            lp = _limb(D, rng, tp, d, L, 0.035 * (0.5 + L / spread), 0.008, bark, 4, 0.1, 0.12, n=3)
            for i in range(1, len(lp)):
                c = lp[i - 1].lerp(lp[i], 0.5)
                dirn = (lp[i] - lp[i - 1]).normalized()
                s = max(0.7, L * 0.85)
                side = Vector((-dirn.z, 0, dirn.x))
                for tilt in (-0.7, 0.0, 0.7):
                    nrm = (Vector((0, 1, 0)) + side * tilt).normalized()
                    D.card(c - dirn * s * 0.45, nrm, dirn, s * 0.9, s, mat, _q(rng), crown=Vector((tp.x, y - 0.8, tp.z)), bend=-s * 0.15)
        y += whorl_dy * rng.uniform(0.85, 1.15)
    # leader
    top = pts[-1]
    for k in range(3):
        a = k * 2.1
        D.card(top + Vector((0, -0.5, 0)), Vector((math.cos(a), 0, math.sin(a))), Y, 0.5, 0.9, mat, _q(rng), crown=top + Vector((0, -1, 0)))
    return [D.build(name, fit_height=7.8)]

def _q(rng):
    q = rng.randrange(4); u0, v0 = (q % 2) * 0.5, (q // 2) * 0.5
    return (u0, v0, u0 + 0.5, v0 + 0.5)

@asset('tree_pine_a')
def build_pine_a():
    return _pine('tree_pine_a', 13, 7.2, 0.22, 0.9, 1.0, 'needles', 'bark_pine', 0.42, 2.3,
                 lambda t: max(0.25, (1 - t) ** 0.9))

@asset('tree_pine_b')
def build_pine_b():
    return _pine('tree_pine_b', 14, 8.0, 0.25, 3.6, 1.0, 'needles_b', 'bark_pine', 0.55, 2.0,
                 lambda t: max(0.3, math.sin(math.pi * min(1, t * 0.85 + 0.12)) ** 0.7))

# ---- rocks: displaced high-poly -> decimated low-poly, colour/roughness/normal baked with Cycles ----
def _rock_hi(seed):
    rng = random.Random(seed)
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=6, radius=1.0)
    off = Vector((rng.uniform(-50, 50), rng.uniform(-50, 50), rng.uniform(-50, 50)))
    planes = []
    for k in range(rng.randint(6, 9)):
        n = Vector((rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 0.6))).normalized()
        planes.append((n, rng.uniform(0.6, 0.85)))
    planes.append((Vector((0, 0, -1)), 0.55))
    sc = Vector((rng.uniform(1.05, 1.25), rng.uniform(0.85, 1.0), rng.uniform(0.72, 0.9)))
    for v in bm.verts:
        p = v.co.normalized()
        r = 1 + 0.2 * mnoise.fractal(p * 1.1 + off, 0.6, 2.0, 4) + 0.05 * mnoise.fractal(p * 3.5 + off, 0.8, 2.0, 4)
        q = p * r
        for n, d in planes:
            k = q.dot(n)
            if k > d:
                q -= n * (k - d) * 0.9
        q += p * (0.025 * mnoise.ridged_multi_fractal(q * 3 + off, 1.0, 2.0, 4, 0.0, 1.5) - 0.02)
        v.co = Vector((q.x * sc.x, q.y * sc.y, q.z * sc.z))
    minz = min(v.co.z for v in bm.verts)
    for v in bm.verts:
        v.co.z -= minz + 0.22
    me = bpy.data.meshes.new('rock_hi'); bm.to_mesh(me); bm.free()
    for p in me.polygons:
        p.use_smooth = True
    ob = bpy.data.objects.new('rock_hi', me); bpy.context.scene.collection.objects.link(ob)
    return ob

def _rock_src_material(seed):
    m = bpy.data.materials.new('rock_src_%d' % seed); m.use_nodes = True
    nt = m.node_tree; N = nt.nodes; L = nt.links
    for n in list(N):
        N.remove(n)
    out = N.new('ShaderNodeOutputMaterial'); em = N.new('ShaderNodeEmission')
    L.new(em.outputs[0], out.inputs['Surface'])
    tc = N.new('ShaderNodeTexCoord')
    def noise(scale, detail=8, rough=0.6, dist=0.3, w=0.0):
        nn = N.new('ShaderNodeTexNoise'); nn.inputs['Scale'].default_value = scale; nn.inputs['Detail'].default_value = detail
        nn.inputs['Roughness'].default_value = rough; nn.inputs['Distortion'].default_value = dist
        L.new(tc.outputs['Object'], nn.inputs['Vector'])
        return nn
    def ramp(inp, stops):
        r = N.new('ShaderNodeValToRGB')
        el = r.color_ramp.elements
        el[0].position, el[0].color = stops[0][0], (*stops[0][1], 1)
        el[1].position, el[1].color = stops[-1][0], (*stops[-1][1], 1)
        for pos, c in stops[1:-1]:
            e = el.new(pos); e.color = (*c, 1)
        L.new(inp, r.inputs['Fac'])
        return r
    def mix(a, b, fac, mode='MIX'):
        mx = N.new('ShaderNodeMix'); mx.data_type = 'RGBA'; mx.blend_type = mode
        L.new(fac, mx.inputs['Factor']); L.new(a, mx.inputs[6]); L.new(b, mx.inputs[7])
        return mx.outputs[2]
    rng = random.Random(seed)
    tint = [(0.085, 0.080, 0.072), (0.075, 0.075, 0.078), (0.095, 0.082, 0.066), (0.08, 0.078, 0.072)][seed % 4]
    base = ramp(noise(2.2, 10, 0.62).outputs['Fac'], [(0.3, tuple(c * 0.55 for c in tint)), (0.5, tint), (0.72, tuple(min(1, c * 1.6) for c in tint))])
    speck = ramp(noise(60, 4, 0.5).outputs['Fac'], [(0.45, (0, 0, 0)), (0.62, (1, 1, 1))])
    col = mix(base.outputs[0], N.new('ShaderNodeRGB').outputs[0], speck.outputs[0], 'MIX')
    rgbn = [n for n in N if n.type == 'RGB'][-1]; rgbn.outputs[0].default_value = (0.16, 0.155, 0.145, 1)
    geo = N.new('ShaderNodeNewGeometry')
    edge = ramp(geo.outputs['Pointiness'], [(0.5, (0, 0, 0)), (0.56, (1, 1, 1))])
    col = mix(col, N.new('ShaderNodeRGB').outputs[0], edge.outputs[0], 'SCREEN')
    [n for n in N if n.type == 'RGB'][-1].outputs[0].default_value = (0.06, 0.06, 0.055, 1)
    ao = N.new('ShaderNodeAmbientOcclusion'); ao.samples = 16; ao.inputs['Distance'].default_value = 0.35
    aor = ramp(ao.outputs['AO'], [(0.0, (0.35, 0.35, 0.35)), (1.0, (1, 1, 1))])
    col = mix(col, aor.outputs[0], N.new('ShaderNodeValue').outputs[0], 'MULTIPLY')
    [n for n in N if n.type == 'VALUE'][-1].outputs[0].default_value = 1.0
    sep = N.new('ShaderNodeSeparateXYZ'); L.new(geo.outputs['Normal'], sep.inputs[0])
    top = ramp(sep.outputs['Z'], [(0.25, (0, 0, 0)), (0.75, (1, 1, 1))])
    lich_n = ramp(noise(5.0, 6, 0.7, 0.8).outputs['Fac'], [(0.56, (0, 0, 0)), (0.6, (1, 1, 1))])
    lmask = N.new('ShaderNodeMath'); lmask.operation = 'MULTIPLY'
    L.new(top.outputs[0], lmask.inputs[0]); L.new(lich_n.outputs[0], lmask.inputs[1])
    lc = ramp(noise(25, 3).outputs['Fac'], [(0.4, (0.16, 0.17, 0.08)), (0.6, (0.26, 0.24, 0.12))])
    col = mix(col, lc.outputs[0], lmask.outputs[0])
    sz = N.new('ShaderNodeSeparateXYZ'); L.new(tc.outputs['Object'], sz.inputs[0])
    dirt = ramp(sz.outputs['Z'], [(-0.2, (1, 1, 1)), (0.25, (0, 0, 0))])
    dc = N.new('ShaderNodeRGB'); dc.outputs[0].default_value = (0.11, 0.09, 0.065, 1)
    col = mix(col, dc.outputs[0], dirt.outputs[0])
    L.new(col, em.inputs['Color'])
    # roughness variant (switched in for the second bake)
    rough = ramp(noise(8, 4).outputs['Fac'], [(0.3, (0.68, 0.68, 0.68)), (0.7, (0.86, 0.86, 0.86))])
    rmix = mix(rough.outputs[0], N.new('ShaderNodeRGB').outputs[0], lmask.outputs[0])
    [n for n in N if n.type == 'RGB'][-1].outputs[0].default_value = (0.95, 0.95, 0.95, 1)
    m['color_socket'] = 0
    return m, em, col, rmix

def build_rock(name, seed):
    sc = bpy.context.scene
    hi = _rock_hi(seed)
    srcm, em, col_sock, rough_sock = _rock_src_material(seed)
    hi.data.materials.append(srcm)
    lo = hi.copy(); lo.data = hi.data.copy(); lo.name = name; sc.collection.objects.link(lo)
    lo.data.materials.clear()
    mod = lo.modifiers.new('dec', 'DECIMATE'); mod.ratio = 2300 / len(hi.data.polygons)
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(lo.evaluated_get(dg))
    lo.modifiers.clear(); lo.data = me
    for p in me.polygons:
        p.use_smooth = True
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = lo; lo.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=math.radians(60), island_margin=0.015, scale_to_bounds=False)
    bpy.ops.object.mode_set(mode='OBJECT')
    size = 1024
    imgs = {k: bpy.data.images.new('%s_%s_bake' % (name, k), size, size, float_buffer=(k != 'base')) for k in ('base', 'rough', 'normal')}
    imgs['normal'].colorspace_settings.name = 'Non-Color'; imgs['rough'].colorspace_settings.name = 'Non-Color'
    m = bpy.data.materials.new(name); m.use_nodes = True; lo.data.materials.append(m)
    tn = m.node_tree.nodes.new('ShaderNodeTexImage')
    sc.render.engine = 'CYCLES'; sc.cycles.device = 'CPU'; sc.cycles.samples = 24
    sc.render.bake.use_selected_to_active = True
    sc.render.bake.cage_extrusion = 0.05; sc.render.bake.max_ray_distance = 0.15; sc.render.bake.margin = 8
    hi.select_set(True); lo.select_set(True); bpy.context.view_layer.objects.active = lo
    links = srcm.node_tree.links
    t0 = time.time()
    for key, btype in (('base', 'EMIT'), ('rough', 'EMIT'), ('normal', 'NORMAL')):
        tn.image = imgs[key]; m.node_tree.nodes.active = tn
        if key == 'rough':
            links.new(rough_sock, em.inputs['Color'])
        bpy.ops.object.bake(type=btype, normal_space='TANGENT', use_clear=True, margin=8)
    log('baked %s (%.0fs)' % (name, time.time() - t0))
    px = lambda im: np.array(im.pixels[:], dtype=np.float32).reshape(size, size, 4)
    base = px(imgs['base'])[..., :3]; rough = px(imgs['rough'])[..., 0]; nrm = px(imgs['normal'])[..., :3]
    # base was baked to a byte (sRGB) image: pixels come back as sRGB values already
    bpy.data.objects.remove(hi, do_unlink=True)
    m.node_tree.nodes.remove(tn)
    MATDEF[name] = (None, size, 1.0)
    _TEXCACHE[name] = {'base': _save_img(name + '_basecolor', base),
                       'orm': _save_img(name + '_orm', np.stack([np.ones_like(rough), rough, np.zeros_like(rough)], -1), noncolor=True),
                       'normal': _save_img(name + '_normal', nrm, noncolor=True)}
    bpy.data.materials.remove(m)
    lo.data.materials.clear()
    lo.data.materials.append(material(name))
    for im in imgs.values():
        bpy.data.images.remove(im)
    return [lo]

for _i, _r in enumerate('abcd'):
    ASSETS['rock_' + _r] = (lambda nm, sd: (lambda: build_rock(nm, sd)))('rock_' + _r, 100 + _i)

# =============================================================================
# 7. EXPORT
# =============================================================================

for _h in HOUSE_STYLES:
    ASSETS[_h] = (lambda nm: (lambda: build_house(nm)))(_h)

def clear_objects():
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    for me in list(bpy.data.meshes):
        if me.users == 0:
            bpy.data.meshes.remove(me)

def all_objects(roots):
    out = []
    def rec(o):
        out.append(o)
        for c in o.children:
            rec(c)
    for r in roots:
        rec(r)
    return out

def export_asset(name):
    clear_objects()
    t0 = time.time()
    roots = ASSETS[name]()
    obs = all_objects(roots)
    bpy.ops.object.select_all(action='DESELECT')
    for o in obs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = obs[0]
    path = os.path.join(OUT, name + '.glb')
    bpy.ops.export_scene.gltf(filepath=path, export_format='GLB', use_selection=True, export_apply=True,
                              export_yup=True, export_texcoords=True, export_normals=True, export_materials='EXPORT',
                              export_image_format='AUTO', export_draco_mesh_compression_enable=False,
                              export_cameras=False, export_lights=False, export_extras=False, export_animations=False)
    tri = sum(tris(o) for o in obs if o.type == 'MESH')
    log('exported %-18s %6d tris %7.1f KB  (%.1fs)' % (name, tri, os.path.getsize(path) / 1024, time.time() - t0))
    return tri

# =============================================================================
# 9. RENDERING (Cycles, Nishita sky, depth of field)
# =============================================================================
_GLB_COLS = {}

def glb_collection(name):
    """Import a .glb once into a collection that is not linked to the scene (used for instancing)."""
    if name in _GLB_COLS:
        return _GLB_COLS[name]
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=os.path.join(OUT, name + '.glb'))
    new = [o for o in bpy.data.objects if o not in before]
    col = bpy.data.collections.new('glb_' + name)
    for o in new:
        for c in list(o.users_collection):
            c.objects.unlink(o)
        col.objects.link(o)
    _GLB_COLS[name] = col
    return col

def place(name, x=0, z=0, rot=0.0, y=0.0, scale=1.0):
    col = glb_collection(name)
    e = bpy.data.objects.new('%s_inst' % name, None)
    e.instance_type = 'COLLECTION'; e.instance_collection = col
    e.location = G((x, y, z)); e.rotation_euler = (0, 0, rot)
    e.scale = (scale, scale, scale)
    bpy.context.scene.collection.objects.link(e)
    return e

def setup_render(samples=SAMPLES, res=(1600, 900)):
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    sc.cycles.device = 'CPU'
    sc.cycles.samples = samples
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.adaptive_threshold = 0.02
    sc.cycles.use_denoising = True
    sc.cycles.max_bounces = 6; sc.cycles.diffuse_bounces = 3; sc.cycles.glossy_bounces = 3
    sc.cycles.transparent_max_bounces = 16
    sc.cycles.blur_glossy = 1.0
    sc.cycles.sample_clamp_indirect = 8.0
    sc.render.resolution_x, sc.render.resolution_y = (res if not QUICK else (800, 450))
    sc.render.resolution_percentage = 100
    sc.view_settings.view_transform = 'AgX'
    try:
        sc.view_settings.look = 'AgX - Medium High Contrast'
    except Exception:
        pass
    sc.view_settings.exposure = 0.0
    sc.render.image_settings.file_format = 'PNG'
    sc.render.image_settings.color_mode = 'RGB'
    sc.render.film_transparent = False

def sky(sun_el=28.0, sun_az=35.0, strength=0.35, sun_strength=4.0):
    """Nishita sky for ambient light + a matching sun lamp for crisp shadows.
    sun_az: degrees, compass angle of the sun in game space (0 = sun in +z direction, i.e. lighting the front)."""
    sc = bpy.context.scene
    w = bpy.data.worlds.new('Sky'); sc.world = w; w.use_nodes = True
    N = w.node_tree.nodes; L = w.node_tree.links
    sk = N.new('ShaderNodeTexSky'); sk.sky_type = 'NISHITA'
    sk.sun_disc = False
    sk.sun_elevation = math.radians(sun_el)
    # direction the sun is in (game): (sin az, ., cos az) -> blender (sin az, -cos az)
    d = Vector((math.sin(math.radians(sun_az)), -math.cos(math.radians(sun_az)), 0))
    sk.sun_rotation = math.atan2(d.x, d.y)   # Nishita rotation: 0 = +Y, increasing towards +X
    sk.altitude = 200; sk.air_density = 1.0; sk.dust_density = 1.6; sk.ozone_density = 1.0
    bg = N['Background']; bg.inputs['Strength'].default_value = strength
    L.new(sk.outputs['Color'], bg.inputs['Color'])
    sd = bpy.data.lights.new('Sun', 'SUN'); sd.energy = sun_strength; sd.angle = math.radians(1.2)
    sd.color = (1.0, 0.93, 0.84)
    so = bpy.data.objects.new('Sun', sd); sc.collection.objects.link(so)
    el = math.radians(sun_el)
    sun_dir = Vector((d.x * math.cos(el), d.y * math.cos(el), math.sin(el)))   # towards the sun
    so.rotation_euler = sun_dir.to_track_quat('Z', 'Y').to_euler()

def ground(size=300, mat='ground', y=0.0):
    b = MB('Ground')
    t = tile_of(mat)
    h = size / 2
    b.poly([(-h, y, -h), (-h, y, h), (h, y, h), (h, y, -h)], mat, uvs=[(-h / t, h / t), (-h / t, -h / t), (h / t, -h / t), (h / t, h / t)])
    return b.build('Ground')

def strip(x0, x1, z0, z1, mat, y=0.01):
    b = MB('Strip_' + mat)
    t = tile_of(mat)
    b.poly([(x0, y, z0), (x0, y, z1), (x1, y, z1), (x1, y, z0)], mat,
           uvs=[(x0 / t, -z0 / t), (x0 / t, -z1 / t), (x1 / t, -z1 / t), (x1 / t, -z0 / t)])
    return b.build()

def scatter(name, count, x0, x1, z0, z1, avoid=(), seed=0, smin=0.7, smax=1.3):
    rng = random.Random(seed)
    n = 0; guard = 0
    while n < count and guard < count * 30:
        guard += 1
        x, z = rng.uniform(x0, x1), rng.uniform(z0, z1)
        if any(a[0] <= x <= a[1] and a[2] <= z <= a[3] for a in avoid):
            continue
        place(name, x, z, rng.uniform(0, 6.28), scale=rng.uniform(smin, smax)); n += 1

def camera(pos, target, lens=35, fstop=4.0, focus=None):
    sc = bpy.context.scene
    cd = bpy.data.cameras.new('Cam'); cd.lens = lens; cd.sensor_width = 36
    co = bpy.data.objects.new('Cam', cd); sc.collection.objects.link(co)
    co.location = G(pos)
    co.rotation_euler = (G(target) - G(pos)).to_track_quat('-Z', 'Y').to_euler()
    cd.dof.use_dof = True; cd.dof.aperture_fstop = fstop
    cd.dof.focus_distance = focus if focus else (G(target) - G(pos)).length
    sc.camera = co
    return co

def new_render_scene():
    clear_objects()
    _GLB_COLS.clear()
    for c in list(bpy.data.collections):
        bpy.data.collections.remove(c)
    for l in list(bpy.data.lights):
        bpy.data.lights.remove(l)
    for c in list(bpy.data.cameras):
        bpy.data.cameras.remove(c)

def render_to(fname):
    os.makedirs(RENDERS, exist_ok=True)
    sc = bpy.context.scene
    sc.render.filepath = os.path.join(RENDERS, fname)
    t0 = time.time()
    bpy.ops.render.render(write_still=True)
    log('render %s (%.0fs)' % (fname, time.time() - t0))

# =============================================================================
# 8. VERIFY (re-import every .glb and check it against the game's collider specs)
# =============================================================================
def _import(name):
    clear_objects()
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=os.path.join(OUT, name + '.glb'))
    return [o for o in bpy.data.objects if o not in before]

def _verts(obs, filt=None):
    out = []
    dg = bpy.context.evaluated_depsgraph_get()
    for o in obs:
        if o.type != 'MESH' or (filt and not filt(o)):
            continue
        mw = o.matrix_world
        for v in o.data.vertices:
            w = mw @ v.co
            out.append((w.x, w.z, -w.y))    # back to game coords
    return np.array(out)

def _bounds(a):
    return [round(float(x), 3) for x in (*a.min(0), *a.max(0))]

def _ray(o, d):
    sc = bpy.context.scene
    dg = bpy.context.evaluated_depsgraph_get()
    hit, loc, *_ = sc.ray_cast(dg, G(o), G(d).normalized())
    return (loc.x, loc.z, -loc.y) if hit else None

SPEC_TOL = 0.011

def verify_asset(name):
    obs = _import(name)
    bpy.context.view_layer.update()
    tri = 0
    for o in obs:
        if o.type == 'MESH':
            tri += tris(o)
    allv = _verts(obs)
    b = _bounds(allv)
    dims = [round(b[3] - b[0], 2), round(b[4] - b[1], 2), round(b[5] - b[2], 2)]
    checks = []
    def chk(label, ok, info=''):
        checks.append((label, bool(ok), info))
    near = lambda a, c, t=SPEC_TOL: abs(a - c) <= t
    def struct_checks(hx, hz, fy, top, ceil_half_x, ceil_half_z, ceil_top):
        sv = _verts(obs, lambda o: 'structure' in o.name)
        wall = sv[sv[:, 1] < top - 1e-3]
        wb = _bounds(wall)
        chk('walls+floor footprint x=[%g,%g] z=[%g,%g]' % (-hx, hx, -hz, hz),
            near(wb[0], -hx) and near(wb[3], hx) and near(wb[2], -hz) and near(wb[5], hz), str(wb))
        chk('floor bottom at y=0, wall top/ceiling slab at y=%g' % top, near(wb[1], 0) and near(sv[:, 1].max(), ceil_top), 'max y %.3f' % sv[:, 1].max())
        sb = _bounds(sv)
        chk('ceiling slab %g x %g' % (2 * ceil_half_x, 2 * ceil_half_z), near(sb[3], ceil_half_x) and near(sb[5], ceil_half_z), str(sb))
        fl = sv[(np.abs(sv[:, 1] - fy) < 1e-4)]
        chk('floor top at y=%g' % fy, len(fl) > 0)
    def door_checks(half_w, h, fy, front_z, back_inner):
        y = fy + 1.0
        r = _ray((0, y, front_z + 6), (0, 0, -1))
        chk('doorway open (ray passes to %s)' % ('back wall' if back_inner is not None else 'far side'),
            (r is None) if back_inner is None else (r is not None and r[2] < back_inner + 0.1), str(r and [round(c, 2) for c in r]))
        for sx in (-1, 1):
            r_in = _ray((sx * (half_w - 0.04), y, front_z + 6), (0, 0, -1))
            r_out = _ray((sx * (half_w + 0.04), y, front_z + 6), (0, 0, -1))
            chk('door width %.1f (x=%+.2f open, x=%+.2f blocked)' % (2 * half_w, sx * (half_w - 0.04), sx * (half_w + 0.04)),
                (r_in is None or r_in[2] < front_z - 0.7) and r_out is not None and r_out[2] > front_z - 0.7)
        r_in = _ray((0, fy + h - 0.04, front_z + 6), (0, 0, -1)); r_out = _ray((0, fy + h + 0.04, front_z + 6), (0, 0, -1))
        chk('door height %.1f' % h, (r_in is None or r_in[2] < front_z - 0.7) and r_out is not None and r_out[2] > front_z - 0.7)
    if name.startswith('house'):
        struct_checks(4, 4, 0.35, 4.55, 4.3, 4.3, 4.85)
        door_checks(0.9, 2.6, 0.35, 4.0, -3.65)
        r = _ray((-6, 2.0, -2.0), (1, 0, 0))
        chk('side wall solid (window is visual only)', r is not None and r[0] < -3.9)
    elif name == 'warehouse':
        struct_checks(14, 9, 0.3, 9.3, 14.4, 9.4, 9.7)
        door_checks(3.5, 5.0, 0.3, 9.0, None)
        r = _ray((0, 2.0, -15), (0, 0, 1))
        chk('back opening open', r is None or r[2] > 9.0)
    elif name == 'barn':
        struct_checks(6, 5, 0.3, 6.3, 6.3, 5.3, 6.6)
        door_checks(2.25, 3.8, 0.3, 5.0, -4.6)
    elif name == 'silo':
        r = _ray((0, 5.0, 8), (0, 0, -1)); chk('radius 2.3 (front ray hits z=2.3)', r is not None and near(r[2], 2.3, 0.02), str(r and round(r[2], 3)))
        body = allv[(allv[:, 1] > 0.5) & (allv[:, 1] < 11.2) & (np.hypot(allv[:, 0], allv[:, 2]) < 2.5)]
        rr = np.hypot(body[:, 0], body[:, 2])
        rim = body[(rr > 2.25) & (rr < 2.39)]
        chk('cylinder top at y=11', near(rim[:, 1].max(), 11.0), '%.3f' % rim[:, 1].max())
        chk('domed top above 11', b[4] > 12.5, '%.2f' % b[4])
    elif name.startswith('container'):
        chk('3.2 x 3.2 x 10', near(dims[0], 3.2, 0.02) and near(dims[1], 3.2, 0.02) and near(dims[2], 10, 0.02), str(dims))
    elif name == 'crate':
        chk('2 x 2 x 2', all(near(d, 2.0, 0.02) for d in dims), str(dims))
    elif name == 'hay_bale':
        chk('radius 0.9, length 1.6 along x', near(dims[0], 1.6, 0.01) and near(b[4], 1.8, 0.03) and near(dims[2], 1.8, 0.03), str(dims))
    elif name == 'tractor':
        chk('about 2.4 x 2.2 x 3.4', near(dims[0], 2.4, 0.1) and near(dims[1], 2.2, 0.05) and near(dims[2], 3.4, 0.1), str(dims))
    elif name == 'well':
        ring = allv[allv[:, 1] < 1.1 + 1e-3]
        chk('stone ring radius 1.3, 1.1 tall', near(np.hypot(ring[:, 0], ring[:, 2]).max(), 1.3, 0.01) and near(ring[:, 1].max(), 1.1), '')
    elif name == 'lamp_post':
        chk('4.4 tall', near(dims[1], 4.4), str(dims))
    elif name == 'fence':
        chk('4 m long, 1.1 tall', near(dims[0], 4.0, 0.01) and near(dims[1], 1.1, 0.01), str(dims))
    elif name == 'chest':
        lid = [o for o in obs if o.name == 'Lid']
        chk('has separate `Lid` object', len(lid) == 1 and lid[0].parent is not None and lid[0].parent.name == 'Chest')
        if lid:
            lb = lid[0].matrix_world.translation
            chk('Lid pivot on back top edge (0, 0.7, -0.45)', near(lb.x, 0) and near(lb.z, 0.7) and near(-lb.y, -0.45), str((round(lb.x, 3), round(lb.z, 3), round(-lb.y, 3))))
        body = _verts([o for o in obs if o.name == 'Chest'])
        bb = _bounds(body)
        chk('body 1.4 x 0.7 x 0.9', near(bb[3] - bb[0], 1.4, 0.03) and near(bb[4] - bb[1], 0.7) and near(bb[5] - bb[2], 0.9, 0.03), str(bb))
    elif name.startswith('tree'):
        chk('5-8 m tall, trunk at origin', 5 <= b[4] <= 8.3 and abs((b[0] + b[3]) / 2) < 1.5, str(dims))
    elif name.startswith('rock'):
        chk('about 2 m boulder', 1.4 <= max(dims) <= 2.8, str(dims))
    elif name == 'grass_clump':
        chk('about 0.6 m clump', 0.4 <= max(dims) <= 0.8, str(dims))
    modes, mimes, exts = glb_alpha_modes(name)
    if name.startswith('tree') or name == 'grass_clump':
        chk('alpha-card material exported as alphaMode MASK', any(v == 'MASK' for v in modes.values()), str(modes))
    chk('no Draco compression, textures embedded', 'KHR_draco_mesh_compression' not in exts and len(mimes) > 0, ','.join(mimes))
    size = os.path.getsize(os.path.join(OUT, name + '.glb'))
    ok = all(c[1] for c in checks)
    for c in checks:
        log('  %s %-55s %s' % ('PASS' if c[1] else 'FAIL', c[0], c[2]))
    return dict(name=name, tris=tri, kb=round(size / 1024, 1), dims=dims, bounds=b, ok=ok,
                checks=[dict(check=c[0], ok=c[1], info=c[2]) for c in checks])

def glb_alpha_modes(name):
    """Read the glTF JSON chunk and return material alphaModes (to prove MASK export)."""
    import struct
    with open(os.path.join(OUT, name + '.glb'), 'rb') as f:
        data = f.read()
    ln = struct.unpack('<I', data[12:16])[0]
    js = json.loads(data[20:20 + ln])
    return {m.get('name'): m.get('alphaMode', 'OPAQUE') for m in js.get('materials', [])}, \
        sorted({i.get('mimeType') for i in js.get('images', [])}), js.get('extensionsUsed', [])

# =============================================================================
# 10. RENDER SCENES
# =============================================================================
def _yard(avoid, x0=-30, x1=30, z0=-30, z1=30, grass=1400, seed=1):
    scatter('grass_clump', grass, x0, x1, z0, z1, avoid=avoid, seed=seed, smin=0.6, smax=1.5)

def scene_house(name, sun_az=35):
    new_render_scene(); setup_render(); sky(26, sun_az); ground()
    place(name, 0, 0, 0.0)
    strip(-0.9, 0.9, 4.4, 12.0, 'dirt_road')
    for k, x in enumerate((-10.0, -6.0, -2.0, 6.0, 10.0)):
        place('fence', x, 9.0, 0.0)
    place('tree_oak', -9.0, -6.0, 1.0); place('tree_pine_a', 9.5, -8.0, 0.4); place('tree_birch', 7.5, 3.5, 2.0)
    place('tree_pine_b', -14, -14, 0.0); place('tree_oak', 16, -18, 2.0); place('tree_pine_a', -20, -4, 2.2)
    place('rock_b', -6.5, 6.5, 0.6, scale=0.45); place('rock_c', 12.5, 7.0, 2.0, scale=0.6)
    place('lamp_post', 4.6, 8.2, 0.0)
    place('chest', -2.6, 5.4, 0.25) if name == 'house_a' else None
    place('crate', 5.5, 5.5, 0.15, scale=0.55) if name == 'house_b' else None
    place('hay_bale', -5.2, 5.6, 0.4, scale=0.8) if name == 'house_c' else None
    _yard([(-4.4, 4.4, -4.4, 4.9), (-1.0, 1.0, 4.0, 12.5)], -26, 26, -26, 22, 5000, seed=hash(name) % 100)
    camera((7.8, 2.2, 16.5), (-0.4, 2.9, 0.0), 32, 4.0, focus=17.2)
    render_to(name + '.png')

def scene_warehouse():
    new_render_scene(); setup_render(); sky(30, -30); ground()
    place('warehouse', 0, 0, 0.0)
    strip(-20, 20, -12, 16, 'concrete')
    place('container_red', -19.5, 0, 0.0); place('container_blue', -23, 2, 0.0); place('container_green', -21.2, 1, 0.0, y=3.2)
    place('container_orange', 19.5, -6, 0.0); place('container', 23.2, -5.5, 0.08)
    for (x, z, y, r) in ((6, 12, 0, 0.2), (8.2, 12.3, 0, 0.05), (7.1, 12.1, 2, 0.35), (-6.5, 11.5, 0, -0.2)):
        place('crate', x, z, r, y=y)
    for (x, z) in ((-0.5, 0), (2.5, -3), (-3.5, -5)):
        place('crate', x, z, 0.3, y=0.3)
    place('lamp_post', -9, 11.0); place('lamp_post', 9, 11.0)
    place('tree_pine_a', -30, -14, 0.3); place('tree_pine_b', 30, 10, 1.0); place('tree_oak', 26, -20, 2.0); place('tree_pine_a', -16, -22, 1.3)
    place('rock_a', 16, 17, 0.5, scale=0.7); place('rock_d', -27, 17, 1.0, scale=0.9)
    _yard([(-21, 21, -13, 17)], -45, 45, -45, 40, 6000, seed=4)
    camera((21, 3.5, 34), (-1.0, 3.6, 2.0), 30, 5.6, focus=38.0)
    render_to('warehouse.png')

def scene_farm(fname, cam, target, lens, focus):
    new_render_scene(); setup_render(); sky(24, 50); ground()
    place('barn', 0, 0, 0.0); place('silo', 10, -3, 0.0)
    strip(-2.3, 2.3, 5.0, 18, 'dirt_road')
    place('tractor', 5.2, 9.5, -0.6)
    for (x, z, r) in ((-6.5, 8.5, 0.3), (-8.4, 10.5, 1.2), (-4.7, 11.6, 2.0)):
        place('hay_bale', x, z, r)
    for k in range(5):
        place('fence', -18 + k * 4, 16.0, 0.0)
    for k in range(4):
        place('fence', 6 + k * 4, 16.0, 0.0)
    place('tree_oak', -13, -5, 0.4); place('tree_birch', -17, 9, 1.0); place('tree_pine_b', -19, -16, 1.4); place('tree_oak', 20, -14, 2.4)
    place('tree_pine_a', 1, -18, 0.1)
    place('rock_c', 13.5, 11, 1.0, scale=0.55)
    _yard([(-6.5, 6.5, -5.5, 5.5), (-2.4, 2.4, 5, 18), (7.5, 12.5, -5.5, -0.5)], -40, 40, -40, 34, 6500, seed=7)
    camera(cam, target, lens, 4.5, focus=focus)
    render_to(fname)

def scene_village():
    new_render_scene(); setup_render(); sky(18, 60, strength=0.4, sun_strength=4.2); ground()
    strip(-3.6, 3.6, -48, 40, 'cobbles', y=0.012)
    strip(-7, 7, -7, 7, 'cobbles', y=0.011)
    homes = [('house_a', -11.5, -16, math.pi / 2), ('house_b', -11.5, 1.0, math.pi / 2), ('house_c', -11.5, 17, math.pi / 2),
             ('house_c', 11.5, -18, -math.pi / 2), ('house_a', 11.5, -1.0, -math.pi / 2), ('house_b', 11.5, 15, -math.pi / 2),
             ('house_a', -2.0, -36, 0.0)]
    for (n, x, z, r) in homes:
        place(n, x, z, r)
    place('well', 0.0, 2.5, 0.3)
    for z in (-30, -14, 8, 24):
        place('lamp_post', -4.3, z, 0.0); place('lamp_post', 4.3, z + 6, 0.0)
    place('chest', 2.6, 6.5, -0.5)
    place('crate', -6.2, 10.6, 0.25); place('crate', -6.0, 12.4, 0.1, scale=0.55); place('hay_bale', 6.3, 9.0, 0.9)
    for (x, z, r) in ((-7.8, 25.5, math.pi / 2), (-7.8, 29.5, math.pi / 2), (7.8, 25.0, math.pi / 2), (7.8, 29.0, math.pi / 2), (7.8, -8.8, math.pi / 2), (-7.8, -7.0, math.pi / 2)):
        place('fence', x, z, r)
    trees = [('tree_oak', -20, 10, 0.2), ('tree_pine_a', -22, -8, 1.0), ('tree_birch', -6.5, 30, 0.4), ('tree_oak', 21, 4, 2.0),
             ('tree_pine_b', 20, -26, 0.5), ('tree_birch', 7.5, -28, 1.7), ('tree_pine_a', -9, -48, 0.0), ('tree_oak', 12, -46, 1.0),
             ('tree_pine_b', -24, 26, 1.0), ('tree_pine_a', 23, 24, 2.0), ('tree_birch', -20, -28, 2.6)]
    for (n, x, z, r) in trees:
        place(n, x, z, r)
    place('rock_a', -6.5, 36, 0.3, scale=0.5); place('rock_d', 8, 35, 1.3, scale=0.45)
    houses_av = [(x - 5, x + 5, z - 5, z + 5) for (_, x, z, _) in homes]
    _yard(houses_av + [(-3.8, 3.8, -50, 42), (-7.2, 7.2, -7.2, 7.2)], -40, 40, -60, 42, 7000, seed=9)
    camera((1.6, 1.75, 34.0), (-0.8, 2.6, -6.0), 30, 2.8, focus=31.5)
    render_to('village_street.png')

def scene_props():
    new_render_scene(); setup_render(); sky(32, 30); ground()
    strip(-19, 19, -6, 4, 'concrete', y=0.01)
    items = [('container_red', -15.0, -1.5, 0.35), ('container_blue', -11.0, -2.0, 0.35), ('crate', -7.5, 1.0, 0.4), ('hay_bale', -4.6, 1.5, 0.3),
             ('tractor', -0.7, 0.8, 0.55), ('well', 3.6, 0.6, 0.3), ('lamp_post', 6.4, 1.8, 0.0), ('fence', 9.6, 1.2, 0.15),
             ('chest', 7.6, 2.8, 0.4), ('container_green', 14.5, -2.0, -0.3), ('container_orange', 18.2, -1.0, -0.3)]
    for (n, x, z, r) in items:
        place(n, x, z, r)
    place('tree_pine_a', -8, -14, 0.0); place('tree_oak', 2, -16, 0.5); place('tree_birch', 11, -12, 1.0)
    _yard([(-19.5, 19.5, -6.5, 4.5)], -35, 35, -30, 20, 4500, seed=11)
    camera((1.2, 3.6, 21.5), (1.4, 1.3, -1.0), 34, 5.6, focus=21.0)
    render_to('props.png')

def scene_nature():
    new_render_scene(); setup_render(); sky(22, 70); ground()
    for (n, x, z, r) in (('tree_oak', -9.0, -2, 0.0), ('tree_birch', -3.0, -1, 0.5), ('tree_pine_a', 2.6, -2.5, 0.0), ('tree_pine_b', 8.2, -1.5, 1.0)):
        place(n, x, z, r)
    for (n, x, z, r) in (('rock_a', -7.5, 6.0, 0.3), ('rock_b', -2.6, 7.0, 1.0), ('rock_c', 2.4, 6.6, 2.1), ('rock_d', 7.2, 6.0, 0.7)):
        place(n, x, z, r)
    _yard([], -40, 40, -40, 16, 6000, seed=13)
    camera((0.0, 2.2, 21.0), (0.0, 3.0, 0.0), 30, 4.0, focus=15.0)
    render_to('nature.png')

RENDER_SCENES = {
    'house_a': lambda: scene_house('house_a', 35), 'house_b': lambda: scene_house('house_b', 20), 'house_c': lambda: scene_house('house_c', 50),
    'warehouse': scene_warehouse,
    'barn': lambda: scene_farm('barn.png', (-9.0, 2.4, 24.0), (1.0, 4.0, 0.0), 30, 25.0),
    'silo': lambda: scene_farm('silo.png', (19.0, 1.5, 13.0), (7.0, 6.2, -3.0), 26, 19.5),
    'village_street': scene_village, 'props': scene_props, 'nature': scene_nature,
}

# =============================================================================
# 11. MAIN
# =============================================================================
ORDER = ['house_a', 'house_b', 'house_c', 'warehouse', 'barn', 'silo',
         'container', 'container_red', 'container_blue', 'container_green', 'container_orange',
         'crate', 'hay_bale', 'tractor', 'well', 'lamp_post', 'fence', 'chest',
         'tree_oak', 'tree_birch', 'tree_pine_a', 'tree_pine_b', 'rock_a', 'rock_b', 'rock_c', 'rock_d', 'grass_clump']

def main():
    os.makedirs(OUT, exist_ok=True)
    names = [n for n in ORDER if not ONLY or n in ONLY]
    if not RENDER_ONLY:
        for n in names:
            export_asset(n)
        report = []
        log('verifying (re-import of every .glb)')
        for n in [n for n in ORDER if os.path.exists(os.path.join(OUT, n + '.glb'))]:
            log('verify', n)
            report.append(verify_asset(n))
        with open(os.path.join(OUT, 'report.json'), 'w') as f:
            json.dump(report, f, indent=1)
        stats = {r['name'] + '.glb': dict(triangles=r['tris'], bytes=os.path.getsize(os.path.join(OUT, r['name'] + '.glb')),
                                          size_kb=r['kb'], dimensions_m=dict(x=r['dims'][0], y=r['dims'][1], z=r['dims'][2]),
                                          bbox_min=r['bounds'][:3], bbox_max=r['bounds'][3:], checks_passed=r['ok'])
                 for r in report}
        with open(os.path.join(OUT, 'stats.json'), 'w') as f:
            json.dump(stats, f, indent=1)
        bad = [r['name'] for r in report if not r['ok']]
        log('verification: %d assets, %d failed %s' % (len(report), len(bad), bad))
    if not NO_RENDER:
        for k, fn in RENDER_SCENES.items():
            if ONLY and k not in ONLY and not (k in ('village_street', 'props', 'nature') and 'renders' in ONLY):
                continue
            fn()

if __name__ == '__main__':
    main()
