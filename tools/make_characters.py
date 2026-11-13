"""
STORM DROP - procedural character generator for Blender (4.2 LTS).

Builds four ORIGINAL survivor / soldier characters (no franchise, celebrity or
real-person likeness), rigs them on one shared humanoid skeleton, bakes the
animation set, bakes PBR textures, exports glTF binaries, re-imports every
file to verify it and renders Cycles turntable stills plus a group shot.

Run with the `bpy` module from PyPI (Python 3.11) plus numpy + scikit-image:
    pip install bpy==4.2.0 scikit-image
    python tools/make_characters.py                 # build + verify + render all
    python tools/make_characters.py a c             # only some characters
    python tools/make_characters.py --no-render     # skip the Cycles stills
    python tools/make_characters.py --render-only   # stills from existing .glb

Outputs:
    models/characters/char_a.glb ... char_d.glb
    models/characters/renders/*.png
    models/characters/stats.json

Export contract (for index.html):
    * metres, glTF +Y up, the character faces +Z, feet on y = 0, ~1.85 m tall
    * one shared skeleton in every file (identical rest pose, so clips can be
      shared): Hips Spine Chest Neck Head Shoulder.L/R UpperArm.L/R
      LowerArm.L/R Hand.L/R UpperLeg.L/R LowerLeg.L/R Foot.L/R
    * animations: Idle Walk Run Sprint StrafeLeft StrafeRight WalkBack
      CrouchIdle CrouchWalk Jump Fall Land Slide Vault Skydive Glide Death
      Swing AimIdle (all in place, 30 fps)
    * an empty "GunMount" parented to Hand.R: attach a gun whose grip is at its
      origin with its barrel along +Z (the models/guns contract) as a child
      with an identity transform
    * a material named "Team" (white-ish fabric) the game recolours per player

How the meshes are made: every body part and garment is a signed distance
field (numpy), polygonised with marching cubes (scikit-image), decimated in
Blender, UV-unwrapped per material and baked from procedural Cycles shaders.
Modelling frame is Blender's: X = character's left, -Y = forward, Z = up.
"""

import json
import math
import os
import sys
import tempfile
import time

import bpy  # must come first when running with the PyPI `bpy` module
import bmesh
import numpy as np
from mathutils import Matrix, Quaternion, Vector, Euler
from mathutils.bvhtree import BVHTree
from mathutils.interpolate import poly_3d_calc
from skimage.measure import marching_cubes

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(ROOT, "models", "characters")
RENDER_DIR = os.path.join(OUT_DIR, "renders")
GUN_DIR = os.path.join(ROOT, "models", "guns")
TEX_TMP = os.path.join(tempfile.gettempdir(), "storm_drop_char_tex")
FPS = 30
CHAR_IDS = ["a", "b", "c", "d"]


def log(*a):
    print("[chars %6.1fs]" % (time.time() - T0), *a, flush=True)


T0 = time.time()

# ==========================================================================
# Signed distance fields (numpy, vectorised over (N, 3) point arrays)
# ==========================================================================

def V(*a):
    return np.array(a if len(a) > 1 else a[0], dtype=np.float64)


def _len(a):
    return np.sqrt(np.einsum("ij,ij->i", a, a))


def smin(a, b, k):
    if k <= 0:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1 - h) + a * h - k * h * (1 - h)


def smax(a, b, k):
    return -smin(-a, -b, k)


def frame(x=None, y=None, z=None):
    """Rotation matrix (columns = local axes) from one or two given axes."""
    if z is not None and y is not None:
        z = V(z) / np.linalg.norm(z)
        x = np.cross(V(y), z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
    elif y is not None and x is not None:
        y = V(y) / np.linalg.norm(y)
        z = np.cross(V(x), y)
        z /= np.linalg.norm(z)
        x = np.cross(y, z)
    elif x is not None and z is not None:
        x = V(x) / np.linalg.norm(x)
        y = np.cross(V(z), x)
        y /= np.linalg.norm(y)
        z = np.cross(x, y)
    else:
        raise ValueError
    return np.stack([x, y, z], axis=1)


def rot_euler(x=0, y=0, z=0):
    m = Euler((math.radians(x), math.radians(y), math.radians(z)), "XYZ").to_matrix()
    return np.array(m, dtype=np.float64)


class F:
    """An SDF: callable on (N, 3) arrays, combinable with | & - operators."""

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, p):
        return self.fn(p)

    def __or__(self, o):
        return F(lambda p: np.minimum(self(p), o(p)))

    def __and__(self, o):
        return F(lambda p: np.maximum(self(p), o(p)))

    def __sub__(self, o):
        return F(lambda p: np.maximum(self(p), -o(p)))

    def grow(self, t):
        return F(lambda p: self(p) - t)

    def su(self, o, k):
        return F(lambda p: smin(self(p), o(p), k))

    def ssub(self, o, k):
        return F(lambda p: smax(self(p), -o(p), k))

    def sinter(self, o, k):
        return F(lambda p: smax(self(p), o(p), k))

    def shell(self, t):
        return F(lambda p: np.abs(self(p)) - t)

    def mirror_x(self):
        def fn(p):
            q = p.copy()
            q[:, 0] = np.abs(q[:, 0])
            return self(q)
        return F(fn)

    def displace(self, fn):
        return F(lambda p: self(p) + fn(p))


def U(fs, k=0.0):
    """Smooth union of a list of SDFs."""
    fs = [f for f in fs if f is not None]

    def fn(p):
        d = fs[0](p)
        for f in fs[1:]:
            d = smin(d, f(p), k)
        return d
    return F(fn)


def local(p, c, R):
    return (p - V(c)) @ (np.eye(3) if R is None else R)


def sphere(c, r):
    c = V(c)
    return F(lambda p: _len(p - c) - r)


def ellipsoid(c, r, R=None):
    r = V(r)

    def fn(p):
        q = local(p, c, R)
        k0 = _len(q / r)
        k1 = _len(q / (r * r))
        return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)
    return F(fn)


def rbox(c, half, rad=0.0, R=None):
    half = V(half)

    def fn(p):
        q = np.abs(local(p, c, R)) - half + rad
        return _len(np.maximum(q, 0.0)) + np.minimum(q.max(axis=1), 0.0) - rad
    return F(fn)


def cone(a, b, r1, r2):
    """Round cone (capsule with different end radii), iq's exact SDF."""
    a = V(a)
    b = V(b)
    ba = b - a
    l2 = float(ba @ ba)
    rr = r1 - r2
    a2 = l2 - rr * rr
    il2 = 1.0 / l2

    def fn(p):
        pa = p - a
        y = pa @ ba
        z = y - l2
        w = pa * l2 - np.outer(y, ba)
        x2 = np.einsum("ij,ij->i", w, w)
        y2 = y * y * l2
        z2 = z * z * l2
        k = np.sign(rr) * rr * rr * x2
        d_mid = (np.sqrt(np.maximum(x2 * a2 * il2, 0)) + y * rr) * il2 - r1
        d_b = np.sqrt(x2 + z2) * il2 - r2
        d_a = np.sqrt(x2 + y2) * il2 - r1
        return np.where(np.sign(z) * a2 * z2 > k, d_b, np.where(np.sign(y) * a2 * y2 < k, d_a, d_mid))
    return F(fn)


def chain(pts, radii, k=0.0):
    return U([cone(pts[i], pts[i + 1], radii[i], radii[i + 1]) for i in range(len(pts) - 1)], k)


def torus(c, R_major, r_minor, R=None):
    def fn(p):
        q = local(p, c, R)
        a = np.sqrt(q[:, 0] ** 2 + q[:, 1] ** 2) - R_major
        return np.sqrt(a * a + q[:, 2] ** 2) - r_minor
    return F(fn)


def cyl(c, r, h, R=None):
    """Capped cylinder along local Z, half-height h."""
    def fn(p):
        q = local(p, c, R)
        d0 = np.sqrt(q[:, 0] ** 2 + q[:, 1] ** 2) - r
        d1 = np.abs(q[:, 2]) - h
        return np.minimum(np.maximum(d0, d1), 0) + np.sqrt(np.maximum(d0, 0) ** 2 + np.maximum(d1, 0) ** 2)
    return F(fn)


def plane(o, n):
    """Positive on the side the normal points to."""
    n = V(n) / np.linalg.norm(n)
    o = V(o)
    return F(lambda p: (p - o) @ n)


def keep(o, n):
    """Half-space SDF keeping the side n points to (negative there)."""
    n = V(n) / np.linalg.norm(n)
    o = V(o)
    return F(lambda p: -((p - o) @ n))


def segdist(p, a, b):
    a = V(a)
    b = V(b)
    ba = b - a
    t = np.clip(((p - a) @ ba) / float(ba @ ba), 0, 1)
    return _len(p - a - np.outer(t, ba)), t


# ==========================================================================
# Shared skeleton (Blender armature space, A-pose, faces -Y)
# ==========================================================================

A_POSE_DEG = 40.0
_ca, _sa = math.cos(math.radians(A_POSE_DEG)), math.sin(math.radians(A_POSE_DEG))


def _arm_pts(sx):
    sh = V(0.180 * sx, 0.010, 1.450)
    d = V(_ca * sx, 0.0, -_sa)
    el = sh + d * 0.285 + V(0, 0.008, 0)
    wr = el + d * 0.255 - V(0, 0.008, 0)
    ha = wr + d * 0.160
    return sh, el, wr, ha


SKEL = {}  # name -> (head, tail, parent)


def _build_skel():
    SKEL["Hips"] = (V(0, 0.005, 0.980), V(0, 0.005, 1.080), None)
    SKEL["Spine"] = (V(0, 0.005, 1.080), V(0, 0.000, 1.250), "Hips")
    SKEL["Chest"] = (V(0, 0.000, 1.250), V(0, 0.005, 1.480), "Spine")
    SKEL["Neck"] = (V(0, 0.005, 1.480), V(0, -0.005, 1.600), "Chest")
    SKEL["Head"] = (V(0, -0.005, 1.600), V(0, -0.005, 1.800), "Neck")
    for s, sx in (("L", 1), ("R", -1)):
        sh, el, wr, ha = _arm_pts(sx)
        SKEL["Shoulder." + s] = (V(0.025 * sx, 0.000, 1.445), V(0.170 * sx, 0.010, 1.452), "Chest")
        SKEL["UpperArm." + s] = (sh, el, "Shoulder." + s)
        SKEL["LowerArm." + s] = (el, wr, "UpperArm." + s)
        SKEL["Hand." + s] = (wr, ha, "LowerArm." + s)
        SKEL["UpperLeg." + s] = (V(0.092 * sx, 0.005, 0.950), V(0.100 * sx, -0.018, 0.520), "Hips")
        SKEL["LowerLeg." + s] = (V(0.100 * sx, -0.018, 0.520), V(0.104 * sx, 0.022, 0.088), "UpperLeg." + s)
        SKEL["Foot." + s] = (V(0.104 * sx, 0.022, 0.088), V(0.106 * sx, -0.118, 0.022), "LowerLeg." + s)


_build_skel()
BONE_ORDER = ["Hips", "Spine", "Chest", "Neck", "Head",
              "Shoulder.L", "UpperArm.L", "LowerArm.L", "Hand.L",
              "Shoulder.R", "UpperArm.R", "LowerArm.R", "Hand.R",
              "UpperLeg.L", "LowerLeg.L", "Foot.L",
              "UpperLeg.R", "LowerLeg.R", "Foot.R"]
FRONT = V(0, -1, 0)


def bone_axes(name):
    """Rest axes (X, Y, Z) of a bone: Y along the bone, X = flex hinge."""
    h, t, _ = SKEL[name]
    y = (t - h) / np.linalg.norm(t - h)
    if name.startswith("Foot"):
        f = V(0, 0, 1)
    elif "Leg" in name:
        f = -FRONT          # knees flex backwards
    else:
        f = FRONT           # elbows (and the spine) flex forwards
    x = np.cross(y, f)
    x /= np.linalg.norm(x)
    z = np.cross(x, y)
    return x, y, z


def hand_basis(side):
    """Modelling basis of a hand: u along the hand, v palm normal, w thumb side."""
    x, y, z = bone_axes("Hand." + side)
    n = x if side == "L" else -x
    return SKEL["Hand." + side][0], y, n, z


# Grip point of the hand-modelling frame: centre of a ~34 mm grip cylinder.
GRIP_U, GRIP_V = 0.085, 0.036
GRIP_RAKE = math.atan(0.30)   # pistol-grip rake (gun "up" vs grip axis)


def hand_to_world(side, u, v, w):
    o, Y, n, Z = hand_basis(side)
    return o + u * Y + v * n + w * Z


# ==========================================================================
# Body + garment construction
# ==========================================================================

class Body:
    """Base anatomy SDF parts for one body type (on the shared skeleton)."""

    def __init__(self, P):
        self.P = P
        fem = P["female"]
        b = P["bulk"]          # 0 lean .. 1 heavy
        m = P["muscle"]        # 0 .. 1
        # ---- torso -------------------------------------------------------
        cw = 0.150 + 0.012 * m + 0.010 * b - 0.016 * fem
        parts = [
            ellipsoid((0, -0.004, 1.330), (cw, 0.100 + 0.008 * m + 0.012 * b, 0.150)),
            ellipsoid((0, 0.012, 1.300), (cw + 0.012, 0.085, 0.120)),          # lats
            ellipsoid((0, -0.008 - 0.02 * b, 1.150), (0.128 + 0.03 * b - 0.012 * fem, 0.092 + 0.04 * b, 0.130)),
            ellipsoid((0, 0.006, 0.985), (0.150 + 0.012 * fem + 0.01 * b, 0.098 + 0.01 * b, 0.105)),
        ]
        for sx in (1, -1):
            parts += [
                cone((0.03 * sx, 0.020, 1.505), (0.165 * sx, 0.012, 1.452), 0.048 - 0.006 * fem, 0.042),  # traps
                ellipsoid((0.188 * sx, 0.004, 1.418), (0.058 + 0.008 * m - 0.008 * fem, 0.060, 0.066)),   # deltoid
                ellipsoid((0.066 * sx, 0.048, 0.925), (0.070 + 0.006 * fem, 0.066 + 0.008 * fem, 0.085)),  # glute
            ]
            if fem:
                parts.append(ellipsoid((0.058 * sx, -0.072, 1.335), (0.058, 0.050, 0.055)))
            else:
                parts.append(ellipsoid((0.068 * sx, -0.058, 1.365), (0.072, 0.045, 0.058)))
        self.torso = U(parts, 0.045)
        # ---- neck --------------------------------------------------------
        nr = 0.055 + 0.008 * m - 0.008 * fem
        self.neck = cone((0, 0.014, 1.450), (0, 0.004, 1.640), nr + 0.006, nr - 0.004)
        # ---- arms --------------------------------------------------------
        self.arm = {}
        self.forearm = {}
        self.upperarm = {}
        for s, sx in (("L", 1), ("R", -1)):
            sh, el, wr, ha = _arm_pts(sx)
            ur = 0.047 + 0.008 * m + 0.006 * b - 0.007 * fem
            up = U([cone(sh, el, ur, 0.036 - 0.003 * fem),
                    ellipsoid(sh + (el - sh) * 0.45 + V(0, -0.006, 0.006), (0.044 + 0.006 * m - 0.006 * fem,) * 2 + (0.09,),
                              frame(z=el - sh, y=FRONT))], 0.02)
            lo = U([cone(el, wr, 0.037 - 0.003 * fem, 0.025 - 0.002 * fem),
                    ellipsoid(el + (wr - el) * 0.28, (0.040 + 0.004 * m - 0.004 * fem, 0.036, 0.075), frame(z=wr - el, y=FRONT))], 0.02)
            self.upperarm[s] = up
            self.forearm[s] = lo
            self.arm[s] = up.su(lo, 0.015)
        # ---- legs --------------------------------------------------------
        self.leg = {}
        self.thigh = {}
        self.shin = {}
        for s, sx in (("L", 1), ("R", -1)):
            hip, knee, ank = SKEL["UpperLeg." + s][0], SKEL["LowerLeg." + s][0], SKEL["Foot." + s][0]
            tr = 0.084 + 0.008 * b + 0.004 * fem
            th = U([cone(hip + V(0, 0, -0.02), knee, tr, 0.054),
                    ellipsoid(hip + (knee - hip) * 0.35 + V(0.008 * sx, -0.01, 0), (0.075 + 0.006 * m, 0.074, 0.16))], 0.03)
            sh_ = U([cone(knee, ank, 0.050, 0.033),
                     ellipsoid(knee + (ank - knee) * 0.32 + V(0, 0.022, 0), (0.048 + 0.004 * m, 0.050, 0.10))], 0.02)
            self.thigh[s] = th
            self.shin[s] = sh_
            self.leg[s] = th.su(sh_, 0.02)
        self.hand = {s: hand_sdf(s) for s in ("L", "R")}
        self.foot = {s: foot_sdf(s) for s in ("L", "R")}

    def trunk(self):
        return U([self.torso, self.neck], 0.03)

    def upper(self):
        return U([self.torso, self.arm["L"], self.arm["R"]], 0.03)

    def lower(self):
        return U([self.torso, self.leg["L"], self.leg["R"]], 0.035)

    def whole(self):
        return U([self.torso, self.neck, self.arm["L"], self.arm["R"], self.leg["L"], self.leg["R"],
                  self.hand["L"], self.hand["R"], self.foot["L"], self.foot["R"]], 0.03)


def hand_sdf(side, glove=0.0):
    """A relaxed gripping hand built in its own (u, v, w) frame."""
    o, Yh, n, Zh = hand_basis(side)
    B = np.stack([Yh, n, Zh], axis=1)   # world -> (u, v, w) by p @ B

    cu, cv = GRIP_U, GRIP_V
    parts = [
        rbox((0.050, 0.000, 0.000), (0.042, 0.013, 0.040), 0.012),                      # palm
        cone((-0.040, 0.0, 0.0), (0.012, 0.0, 0.0), 0.026, 0.030),                       # wrist / heel of hand
        ellipsoid((0.035, 0.010, 0.030), (0.030, 0.018, 0.016)),                         # thenar pad
    ]
    fingers = [(0.024, 0.044, 0.0090, 1.00), (0.007, 0.049, 0.0092, 1.08),
               (-0.010, 0.046, 0.0088, 1.02), (-0.026, 0.040, 0.0078, 0.86)]
    for w, L, r, sc in fingers:
        # knuckle on the palm, then an arc around the grip axis
        k = np.array([0.093 - abs(w) * 0.25, 0.0, w])
        R0 = 0.034
        a0 = math.atan2(k[1] - cv, k[0] - cu)
        total = (L + 0.030 * sc + 0.022 * sc) / R0
        pts = [k]
        for i in range(1, 4):
            a = a0 + total * i / 3.0 * 0.92
            pts.append(np.array([cu + R0 * math.cos(a), cv + R0 * math.sin(a), w]))
        parts.append(chain(pts, [r * 1.05, r, r * 0.92, r * 0.82], 0.006))
    th = [np.array(p) for p in ((0.020, 0.008, 0.026), (0.044, 0.020, 0.046), (0.068, 0.040, 0.050), (0.088, 0.062, 0.044))]
    parts.append(chain(th, [0.013, 0.011, 0.0095, 0.0085], 0.006))
    f = U(parts, 0.010)
    if glove:
        f = f.grow(glove)

    def fn(p):
        return f((p - o) @ B)
    return F(fn)


def foot_sdf(side, extra=0.0):
    sx = 1 if side == "L" else -1
    a = SKEL["Foot." + side][0]
    x = a[0]
    f = U([rbox((x, -0.045, 0.030), (0.040, 0.090, 0.026), 0.024),
           ellipsoid((x - 0.004 * sx, -0.120, 0.030), (0.046, 0.060, 0.030)),
           ellipsoid((x, 0.040, 0.045), (0.035, 0.040, 0.042)),
           cone(a + V(0, 0, 0.06), a, 0.034, 0.038)], 0.02)
    return f.grow(extra)


# ---------------------------------------------------------------------------
# Head
# ---------------------------------------------------------------------------

EYE_Z = 1.693
EYE_X = 0.032
EYE_Y = -0.071
EYE_R = 0.0125


def head_sdf(P):
    fem = P["female"]
    jaw = P.get("jaw", 0.5)
    nose = P.get("nose", 0.5)
    lips = P.get("lips", 0.5)
    cran = ellipsoid((0, 0.006, 1.708), (0.073 - 0.003 * fem, 0.096 - 0.003 * fem, 0.098 - 0.003 * fem))
    face = ellipsoid((0, -0.034, 1.648), (0.056 + 0.006 * jaw - 0.004 * fem, 0.058, 0.062))
    jawf = U([cone((sx * 0.050, -0.002, 1.640), (sx * (0.024 + 0.008 * jaw), -0.062, 1.590), 0.020 + 0.006 * jaw - 0.004 * fem,
                   0.015 + 0.004 * jaw) for sx in (1, -1)], 0.02)
    chin = ellipsoid((0, -0.076, 1.588), (0.020 + 0.006 * jaw - 0.003 * fem, 0.018, 0.019))
    cheeks = U([ellipsoid((sx * 0.044, -0.058, 1.668), (0.022, 0.020, 0.016)) for sx in (1, -1)])
    brow = ellipsoid((0, -0.074, 1.716), (0.056, 0.020, 0.012 - 0.004 * fem))
    nw = 0.010 + 0.004 * nose
    nosef = U([cone((0, -0.082, 1.708), (0, -0.104 - 0.004 * nose, 1.660), 0.008, nw),
               ellipsoid((0, -0.097, 1.655), (nw + 0.006, 0.010, 0.008)),
               ] + [sphere((sx * 0.012, -0.093, 1.656), 0.0085) for sx in (1, -1)], 0.008)
    lf = 1.0 + 0.4 * lips
    lipsf = U([ellipsoid((0, -0.090, 1.626), (0.021, 0.010, 0.0065 * lf)),
               ellipsoid((0, -0.087, 1.612), (0.019, 0.010, 0.0075 * lf))], 0.004)
    ears = U([ellipsoid((sx * 0.074, 0.008, 1.688), (0.011, 0.022, 0.031), frame(z=(sx * 0.15, 0.25, 1.0), y=(0, 1, 0)))
              for sx in (1, -1)])
    ear_cut = U([ellipsoid((sx * 0.084, 0.006, 1.690), (0.006, 0.013, 0.019)) for sx in (1, -1)])
    neck = cone((0, 0.014, 1.440), (0, 0.010, 1.650), 0.058 - 0.008 * fem, 0.050 - 0.006 * fem)
    head = U([cran, face, jawf, chin, cheeks, brow, neck], 0.022)
    head = head.su(nosef, 0.010).su(lipsf, 0.004).su(ears, 0.010).ssub(ear_cut, 0.004)
    sockets = U([sphere((sx * EYE_X, EYE_Y - 0.016, EYE_Z + 0.002), 0.017) for sx in (1, -1)])
    head = head.ssub(sockets, 0.008)
    eyes = U([sphere((sx * EYE_X, EYE_Y, EYE_Z), EYE_R) for sx in (1, -1)])
    lids = U([ellipsoid((sx * EYE_X, EYE_Y - 0.0015, EYE_Z + 0.0072), (0.0155, 0.0128, 0.0064)) for sx in (1, -1)] +
             [ellipsoid((sx * EYE_X, EYE_Y - 0.001, EYE_Z - 0.0105), (0.0145, 0.011, 0.0035)) for sx in (1, -1)])
    return head.su(lids, 0.004) | eyes


def scalp_cut(hairline_front=1.738, nape=1.640, side=1.700):
    """Region where hair grows (negative inside)."""
    def fn(p):
        y, z = p[:, 1], p[:, 2]
        # front hairline rises toward the face, sides drop behind the ears
        lim = np.where(y < -0.03, hairline_front + (y + 0.03) * -0.5,
                       np.where(y < 0.035, side + (0.035 - y) / 0.065 * (hairline_front - 0.015 - side), nape + (side - nape) * np.clip((0.09 - y) / 0.055, 0, 1)))
        ear = np.sqrt(((np.abs(p[:, 0]) - 0.075) / 0.025) ** 2 + ((y - 0.006) / 0.030) ** 2 + ((z - 1.69) / 0.040) ** 2) - 1.0
        return np.maximum(lim - z, -ear * 0.02)
    return F(fn)


# ==========================================================================
# Character designs (all original)
# ==========================================================================
#  A "Warden"  - broad male soldier: high-cut helmet, plate carrier, mottled
#                field camo jacket/pants, assault pack, short beard.
#  B "Kestrel" - lean female scout: softshell jacket with team shoulder yoke,
#                baseball cap + ponytail, light chest rig, sling bag, low boots.
#  C "Bulwark" - heavy-set male survivalist: knit beanie (team), full beard,
#                canvas work jacket with hood, big trekking pack + bedroll.
#  D "Vesper"  - female operator: curly crop, headset, team neck gaiter,
#                bomber jacket, slick plate carrier, grey camo pants, knee pads.

def col(h):
    h = h.lstrip("#")
    c = [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    return tuple(x ** 2.2 for x in c)   # sRGB -> linear


# pattern ids read by the bake shaders (attribute "pat")
PAT_PLAIN, PAT_CAMO, PAT_KNIT, PAT_WEBBING, PAT_HARD, PAT_SOLE, PAT_RIP, PAT_DENIM, PAT_CANVAS, PAT_QUILT = range(10)


class Piece:
    def __init__(self, name, mat, sdf, tint, pat=PAT_PLAIN, tris=800, step=0.006, rule=None, bbox=None):
        self.name, self.mat, self.sdf, self.tint, self.pat = name, mat, sdf, tint, pat
        self.tris, self.step, self.rule, self.bbox = tris, step, rule, bbox


def surf_y(f, x, z, front=True):
    """Y of the surface of SDF f at (x, z), probing from the front or back."""
    ys = np.linspace(-0.45, 0.45, 901)
    if not front:
        ys = ys[::-1]
    p = np.stack([np.full_like(ys, x), ys, np.full_like(ys, z)], 1)
    d = f(p)
    i = int(np.argmax(d < 0))
    return float(ys[i])


def surf_x(f, y, z):
    xs = np.linspace(0.6, 0.0, 601)
    p = np.stack([xs, np.full_like(xs, y), np.full_like(xs, z)], 1)
    i = int(np.argmax(f(p) < 0))
    return float(xs[i])


def sleeve_cut(side, back=0.02):
    """Keep the side of the wrist plane toward the elbow."""
    sh, el, wr, ha = _arm_pts(1 if side == "L" else -1)
    d = (wr - el) / np.linalg.norm(wr - el)
    return keep(wr - d * back, -d)


def arm_band(side, t, r, w, f_arm):
    sh, el, wr, ha = _arm_pts(1 if side == "L" else -1)
    a, b = (sh, el) if t < 1 else (el, wr)
    tt = t if t < 1 else t - 1
    c = a + (b - a) * tt
    R = frame(z=b - a, y=FRONT)
    return f_arm.grow(r).sinter(cyl(c, 0.2, w, R), 0.003)


def build_common_upper(body, P, loose, hem, collar=True, hood=False):
    """A jacket: torso + sleeves, cut at the hem and the wrists."""
    up = body.upper()
    j = up.grow(loose)
    j = j & keep((0, 0, hem), (0, 0, 1)) & sleeve_cut("L") & sleeve_cut("R")
    # cuffs
    cuffs = []
    for s, sx in (("L", 1), ("R", -1)):
        sh, el, wr, ha = _arm_pts(sx)
        d = (wr - el) / np.linalg.norm(wr - el)
        cuffs.append(cyl(wr - d * 0.035, 0.034 + loose * 0.6, 0.016, frame(z=d, y=FRONT)))
    j = j.su(U(cuffs), 0.006)
    # cut the neck hole
    j = j - cone((0, 0.012, 1.40), (0, 0.004, 1.75), 0.058, 0.050)
    if collar:
        c = torus((0, 0.012, 1.497), 0.066, 0.016, frame(z=(0, 0.35, 1.0), x=(1, 0, 0)))
        j = j.su(c, 0.012)
    if hood:
        h = U([ellipsoid((0, 0.085, 1.520), (0.120, 0.060, 0.060)), torus((0, 0.030, 1.505), 0.080, 0.030, frame(z=(0, 0.6, 1.0), x=(1, 0, 0)))], 0.03)
        j = j.su(h - cone((0, 0.012, 1.40), (0, 0.004, 1.75), 0.058, 0.050), 0.02)
    return j


def build_pants(body, loose, top=1.04, bottom=0.14, cargo=True, kneepads=False):
    lo = U([body.torso & keep((0, 0, 1.10), (0, 0, -1)), body.leg["L"], body.leg["R"]], 0.035)
    pnt = lo.grow(loose) & keep((0, 0, top), (0, 0, -1)) & keep((0, 0, bottom), (0, 0, 1))
    gap = F(lambda p: np.maximum(np.abs(p[:, 0]) - 0.010, p[:, 2] - (0.80 - 0.4 * np.abs(p[:, 1]))))
    pnt = pnt.ssub(gap, 0.02)
    extras = []
    if cargo:
        for s, sx in (("L", 1), ("R", -1)):
            x = 0.165 * sx
            extras.append(rbox((x, -0.012, 0.700), (0.020, 0.062, 0.075), 0.016, frame(x=(sx, 0.0, -0.08), z=(0, 0, 1))))
    # waistband
    extras.append(cyl((0, 0.004, top - 0.02), 0.0, 0.0))
    pnt = pnt.su(U(extras[:-1]), 0.012) if cargo else pnt
    return pnt


def build_belt(body, z=1.015, t=0.016, h=0.022):
    b = body.torso.grow(t) & keep((0, 0, z + h), (0, 0, -1)) & keep((0, 0, z - h), (0, 0, 1))
    buckle = rbox((0, surf_y(body.torso, 0, z) - t - 0.004, z), (0.026, 0.006, 0.020), 0.004)
    return b | buckle


def build_boots(side, tall=0.24, chunky=1.0):
    sx = 1 if side == "L" else -1
    a = SKEL["Foot." + side][0]
    knee = SKEL["LowerLeg." + side][0]
    d = (knee - a) / np.linalg.norm(knee - a)
    shaft = cone(a + V(0, 0.004, -0.02), a + d * (tall - 0.07), 0.050 * chunky, 0.048 * chunky)
    f = foot_sdf(side, 0.012 * chunky)
    sole = rbox((a[0], -0.035, 0.016), (0.052 * chunky, 0.150, 0.016), 0.010) & foot_sdf(side, 0.030).grow(0.0)
    lace = rbox((a[0], -0.055, 0.082), (0.028, 0.055, 0.016), 0.012, frame(y=(0, -1, -0.55), x=(1, 0, 0)))
    return U([f, shaft, lace], 0.02) | sole


def build_glove(side):
    return hand_sdf(side, glove=0.0035)


def vest_sdf(body, P, slick=False):
    """Plate carrier: front/back plates, cummerbund, shoulder straps."""
    t = body.torso
    yf = surf_y(t, 0, 1.28)
    yb = surf_y(t, 0, 1.28, front=False)
    fp = rbox((0, yf - 0.012, 1.285), (0.125, 0.020, 0.150), 0.025, frame(z=(0, -0.12, 1), x=(1, 0, 0)))
    bp = rbox((0, yb + 0.014, 1.295), (0.128, 0.020, 0.160), 0.025, frame(z=(0, 0.10, 1), x=(1, 0, 0)))
    cb = t.grow(0.022) & keep((0, 0, 1.235), (0, 0, -1)) & keep((0, 0, 1.105), (0, 0, 1))
    straps = []
    for sx in (1, -1):
        straps.append(chain([V(0.090 * sx, yf - 0.005, 1.390), V(0.120 * sx, -0.010, 1.478), V(0.105 * sx, yb + 0.010, 1.410)],
                            [0.022, 0.022, 0.022]) & keep((0, 0, 1.38), (0, 0, 1)))
    straps = U(straps)
    v = U([fp, bp, cb, straps], 0.012)
    pouches = []
    if not slick:
        for i, x in enumerate((-0.068, 0.0, 0.068)):
            pouches.append(rbox((x, yf - 0.048, 1.175), (0.030, 0.020, 0.058), 0.010))
        pouches.append(rbox((0.0, yf - 0.040, 1.330), (0.085, 0.016, 0.042), 0.010))  # admin pouch
    else:
        pouches.append(rbox((0.0, yf - 0.042, 1.180), (0.100, 0.016, 0.050), 0.010))
        pouches.append(rbox((-0.070, yf - 0.042, 1.300), (0.030, 0.014, 0.040), 0.010))
    return v.su(U(pouches), 0.006)


def backpack_sdf(body, size=1.0, bedroll=False, over=None):
    t = body.torso if over is None else over
    yb = surf_y(t, 0, 1.28, front=False)
    hx, hy, hz = 0.135 * size, 0.065 * size, 0.190 * size
    c = V(0, yb + hy + 0.012, 1.270 - (size - 1) * 0.05)
    main = rbox(c, (hx, hy, hz), 0.045)
    lid = ellipsoid(c + V(0, 0.004, hz - 0.01), (hx * 1.02, hy * 1.12, 0.045))
    side_p = U([rbox(c + V(sx * (hx + 0.012), 0.0, -0.06), (0.020, 0.050, 0.075), 0.018) for sx in (1, -1)])
    front_p = rbox(c + V(0, hy + 0.012, -0.07), (hx * 0.75, 0.022, hz * 0.45), 0.02)
    parts = [main, lid, side_p, front_p]
    roll = cyl(c + V(0, 0.0, hz + 0.062), 0.062, hx + 0.03, frame(z=(1, 0, 0), y=(0, 1, 0))) if bedroll else None
    pack = U(parts, 0.012)
    straps = []
    yf = surf_y(t, 0, 1.30)
    for sx in (1, -1):
        straps.append(chain([c + V(sx * 0.07, -hy, hz - 0.05), V(0.105 * sx, -0.012, 1.505), V(0.115 * sx, yf - 0.004, 1.380),
                             V(0.125 * sx, yf + 0.02, 1.16)], [0.016, 0.017, 0.017, 0.015], 0.01))
    return (pack, U(straps), roll) if bedroll else (pack, U(straps))


def designs():
    D = {}
    # ------------------------------------------------------------------ A
    D["a"] = dict(
        name="Warden", female=0, bulk=0.35, muscle=0.8, jaw=0.85, nose=0.6, lips=0.3,
        skin=col("#c99a7e"), hair=col("#3a2a1e"), eye=col("#4a6a7a"), brows=0.9, beard=1,
    )
    # ------------------------------------------------------------------ B
    D["b"] = dict(
        name="Kestrel", female=1, bulk=0.0, muscle=0.35, jaw=0.25, nose=0.3, lips=0.7,
        skin=col("#e8c2a6"), hair=col("#7a3c1e"), eye=col("#5b7a46"), brows=0.6, beard=0, freckles=1,
    )
    # ------------------------------------------------------------------ C
    D["c"] = dict(
        name="Bulwark", female=0, bulk=1.0, muscle=0.55, jaw=1.0, nose=0.9, lips=0.5,
        skin=col("#9c6b4a"), hair=col("#1d1612"), eye=col("#3a2618"), brows=1.0, beard=2,
    )
    # ------------------------------------------------------------------ D
    D["d"] = dict(
        name="Vesper", female=1, bulk=0.25, muscle=0.55, jaw=0.45, nose=0.55, lips=0.9,
        skin=col("#5e3a26"), hair=col("#141010"), eye=col("#2a1a10"), brows=0.7, beard=0,
    )
    return D


def build_pieces(cid, P):
    body = Body(P)
    pcs = []
    skin, hairc = P["skin"], P["hair"]
    head = head_sdf(P)
    pcs.append(Piece("head", "Skin", head, skin, tris=2000, step=0.0028, rule="head"))

    def hands(mat, tint, glove):
        for s in ("L", "R"):
            f = build_glove(s) if glove else body.hand[s]
            pcs.append(Piece("hand_" + s, mat, f, tint, tris=420, step=0.0025, rule="hand" + s))

    def boots(tint, tall, chunky):
        for s in ("L", "R"):
            pcs.append(Piece("boot_" + s, "Leather", build_boots(s, tall, chunky), tint, PAT_SOLE, tris=480, step=0.005, rule="foot" + s))

    if cid == "a":
        jk = build_common_upper(body, P, 0.020, 0.93)
        yf = surf_y(body.torso, 0.075, 1.36)
        pockets = U([rbox((sx * 0.080, yf - 0.012, 1.375), (0.040, 0.012, 0.045), 0.010) for sx in (1, -1)])
        jk = jk.su(pockets, 0.006)
        pcs.append(Piece("jacket", "Fabric", jk, col("#6b6a4c"), PAT_CAMO, tris=2000, step=0.006, rule="upper"))
        pcs.append(Piece("pants", "Fabric", build_pants(body, 0.026, top=1.0), col("#6b6a4c"), PAT_CAMO, tris=1500, step=0.006, rule="lower"))
        pcs.append(Piece("vest", "Gear", vest_sdf(body, P), col("#7d6a4f"), PAT_WEBBING, tris=1500, step=0.005, rule="chest"))
        pack, straps = backpack_sdf(body, 0.9, over=body.torso.grow(0.045))
        pcs.append(Piece("pack", "Gear", pack, col("#5d5a44"), PAT_RIP, tris=700, step=0.007, rule="chest"))
        pcs.append(Piece("belt", "Leather", build_belt(body, 0.975, 0.040, 0.024), col("#3b3226"), PAT_WEBBING, tris=300, step=0.005, rule="hips"))
        helmet = ellipsoid((0, 0.004, 1.742), (0.100, 0.122, 0.104)) & keep((0, 0, 1.690), (0, 0.38, 1)) - \
            U([ellipsoid((sx * 0.105, 0.010, 1.680), (0.04, 0.05, 0.05)) for sx in (1, -1)])
        helmet = helmet.su(U([rbox((sx * 0.094, 0.010, 1.712), (0.010, 0.050, 0.010), 0.004) for sx in (1, -1)] +
                             [rbox((0, -0.112, 1.758), (0.024, 0.012, 0.016), 0.005)]), 0.006)
        pcs.append(Piece("helmet", "Gear", helmet, col("#68624a"), PAT_HARD, tris=900, step=0.004, rule="head"))
        band = (ellipsoid((0, 0.004, 1.742), (0.100, 0.122, 0.104)).grow(0.007) & keep((0, 0, 1.745), (0, 0.1, -1)) &
                keep((0, 0, 1.718), (0, 0.1, 1)))
        pcs.append(Piece("helmet_band", "Team", band, (0.85, 0.85, 0.85), PAT_PLAIN, tris=300, step=0.003, rule="head"))
        strap = U([chain([V(sx * 0.082, 0.010, 1.690), V(sx * 0.060, -0.020, 1.600), V(0, -0.050, 1.560)], [0.004, 0.004, 0.004]) for sx in (1, -1)])
        pcs.append(Piece("chinstrap", "Leather", strap, col("#2a2620"), PAT_WEBBING, tris=150, step=0.003, rule="head"))
        beard = head.grow(0.011) & F(lambda p: p[:, 2] - (1.655 + np.maximum(0, p[:, 1] + 0.04) * 0.6)) & \
            keep((0, 0.04, 0), (0, -1, 0)) & keep((0, 0, 1.565), (0, 0, 1))
        beard = beard - ellipsoid((0, -0.096, 1.617), (0.020, 0.030, 0.0075))
        hair = head.grow(0.008) & scalp_cut(1.76, 1.64, 1.71)
        pcs.append(Piece("hair", "Hair", beard | hair, hairc, tris=700, step=0.0028, rule="head"))
        for s in ("L", "R"):
            pcs.append(Piece("armband_" + s, "Team", arm_band(s, 0.42, 0.030, 0.030, body.arm[s]), (0.85, 0.85, 0.85), tris=180, step=0.004, rule="upperarm" + s))
        hands("Leather", col("#3a3530"), True)
        boots(col("#8a7350"), 0.26, 1.05)
    elif cid == "b":
        jk = build_common_upper(body, P, 0.010, 0.92)
        pcs.append(Piece("jacket", "Fabric", jk, col("#3c4146"), PAT_PLAIN, tris=1900, step=0.005, rule="upper"))
        yoke = (jk.grow(0.0035) & U([sphere((sx * 0.17, 0.0, 1.47), 0.105) for sx in (1, -1)])) | \
            (jk.grow(0.0035) & U([cone(_arm_pts(sx)[0], _arm_pts(sx)[1], 0.08, 0.08) for sx in (1, -1)]) &
             U([F(lambda p, sx=sx: np.abs((p - _arm_pts(sx)[0]) @ (V(0, 0, 1))) - 0.0) for sx in (1, -1)]).grow(0.012))
        yoke = jk.grow(0.0035) & U([sphere((sx * 0.175, 0.005, 1.455), 0.095) for sx in (1, -1)] +
                                   [rbox((sx * 0.19, -0.05, 1.20), (0.012, 0.12, 0.10), 0.0) for sx in (1, -1)])
        pcs.append(Piece("yoke", "Team", yoke, (0.85, 0.85, 0.85), tris=700, step=0.004, rule="upper"))
        pcs.append(Piece("pants", "Fabric", build_pants(body, 0.016), col("#2a2c2e"), PAT_RIP, tris=1500, step=0.006, rule="lower"))
        pcs.append(Piece("belt", "Leather", build_belt(body, 0.995, 0.020, 0.016), col("#4a3426"), PAT_PLAIN, tris=250, step=0.004, rule="hips"))
        # light chest rig
        t = body.torso
        yf = surf_y(t, 0, 1.20)
        rig = U([rbox((0, yf - 0.020, 1.205), (0.105, 0.022, 0.055), 0.014)] +
                [chain([V(sx * 0.095, yf - 0.01, 1.24), V(sx * 0.095, -0.06, 1.46), V(sx * 0.08, 0.06, 1.48), V(sx * 0.10, surf_y(t, 0.1, 1.3, False) + 0.01, 1.20)],
                       [0.012] * 4, 0.006) for sx in (1, -1)], 0.006)
        rig = rig | (t.grow(0.010) & keep((0, 0, 1.18), (0, 0, -1)) & keep((0, 0, 1.15), (0, 0, 1)))
        pcs.append(Piece("chestrig", "Gear", rig, col("#6e6250"), PAT_WEBBING, tris=900, step=0.004, rule="chest"))
        yb = surf_y(t, -0.05, 1.10, front=False)
        bag = rbox((-0.060, yb + 0.040, 1.080), (0.090, 0.038, 0.070), 0.03, frame(x=(1, 0, 0.35), y=(0, 1, 0)))
        sling = chain([V(-0.10, yb + 0.02, 1.10), V(-0.03, 0.10, 1.40), V(0.11, 0.05, 1.50), V(0.10, -0.05, 1.42), V(0.0, -0.13, 1.24), V(-0.12, -0.08, 1.05)],
                      [0.010] * 6, 0.01)
        pcs.append(Piece("bag", "Gear", bag | sling, col("#4a5240"), PAT_RIP, tris=500, step=0.005, rule="chest"))
        # cap + ponytail
        crown = ellipsoid((0, 0.004, 1.722), (0.083, 0.102, 0.090)) & keep((0, 0, 1.712), (0, 0.32, 1))
        brim = ellipsoid((0, -0.120, 1.733), (0.078, 0.065, 0.010), frame(z=(0, 0.18, 1), x=(1, 0, 0))) & keep((0, -0.05, 0), (0, -1, 0.15))
        button = sphere((0, 0.0, 1.812), 0.008)
        cap = U([crown, brim, button], 0.006) - sphere((0, 0.11, 1.705), 0.022)
        pcs.append(Piece("cap", "Team", cap, (0.85, 0.85, 0.85), PAT_PLAIN, tris=700, step=0.0035, rule="head"))
        hairbase = head.grow(0.009) & scalp_cut(1.745, 1.665, 1.705)
        tail = chain([V(0, 0.100, 1.712), V(0, 0.130, 1.680), V(0, 0.142, 1.610), V(0, 0.135, 1.530), V(0, 0.118, 1.470)],
                     [0.024, 0.028, 0.024, 0.018, 0.007], 0.012)
        tie = torus((0, 0.118, 1.692), 0.022, 0.006, frame(z=(0, 1, -0.7), x=(1, 0, 0)))
        pcs.append(Piece("hair", "Hair", U([hairbase, tail], 0.01) | tie, hairc, tris=900, step=0.0032, rule="head"))
        hands("Skin", skin, False)
        boots(col("#6b4a30"), 0.18, 0.98)
    elif cid == "c":
        jk = build_common_upper(body, P, 0.032, 0.86, hood=True)
        yf = surf_y(body.torso, 0.08, 1.12)
        pk = U([rbox((sx * 0.085, yf - 0.020, 1.020), (0.060, 0.016, 0.070), 0.014) for sx in (1, -1)])
        jk = jk.su(pk, 0.008)
        pcs.append(Piece("jacket", "Fabric", jk, col("#7a5a36"), PAT_CANVAS, tris=2200, step=0.006, rule="upper"))
        pcs.append(Piece("pants", "Fabric", build_pants(body, 0.028), col("#4e5038"), PAT_RIP, tris=1400, step=0.006, rule="lower"))
        pack, straps, bed = backpack_sdf(body, 1.25, bedroll=True, over=body.torso.grow(0.034))
        pcs.append(Piece("pack", "Gear", pack | straps, col("#3e4a3a"), PAT_RIP, tris=1100, step=0.007, rule="chest"))
        pcs.append(Piece("bedroll", "Team", bed, (0.85, 0.85, 0.85), PAT_KNIT, tris=300, step=0.006, rule="chest"))
        beanie = (ellipsoid((0, 0.004, 1.715), (0.086, 0.106, 0.103)) & keep((0, 0, 1.705), (0, 0.2, 1))) | \
            (ellipsoid((0, 0.004, 1.712), (0.091, 0.111, 0.04)) & keep((0, 0, 1.70), (0, 0.2, 1)))
        pcs.append(Piece("beanie", "Team", beanie, (0.85, 0.85, 0.85), PAT_KNIT, tris=700, step=0.004, rule="head"))
        beard = head.grow(0.013) & F(lambda p: p[:, 2] - (1.650 + np.maximum(0, p[:, 1] + 0.05) * 0.6)) & \
            keep((0, 0.045, 0), (0, -1, 0)) & keep((0, 0, 1.53), (0, 0, 1))
        beard = beard.su(ellipsoid((0, -0.082, 1.572), (0.036, 0.026, 0.028)), 0.02)
        beard = beard - ellipsoid((0, -0.100, 1.616), (0.019, 0.030, 0.010))
        mous = ellipsoid((0, -0.101, 1.630), (0.025, 0.009, 0.0055))
        hair = head.grow(0.008) & scalp_cut(1.74, 1.64, 1.70) & keep((0, 0, 1.712), (0, 0.2, -1))
        pcs.append(Piece("hair", "Hair", U([beard, mous], 0.005) | hair, hairc, tris=800, step=0.003, rule="head"))
        hands("Leather", col("#5a4632"), True)
        boots(col("#4a3a2c"), 0.24, 1.12)
    elif cid == "d":
        jk = build_common_upper(body, P, 0.016, 0.93, collar=False)
        # ribbed hem/cuffs as a separate band of the same fabric material
        rib = jk.grow(0.004) & keep((0, 0, 0.965), (0, 0, -1))
        pcs.append(Piece("jacket", "Fabric", jk, col("#4b5236"), PAT_QUILT, tris=1900, step=0.005, rule="upper"))
        pcs.append(Piece("jacket_rib", "Fabric", rib, col("#2e3024"), PAT_KNIT, tris=250, step=0.004, rule="upper"))
        pcs.append(Piece("pants", "Fabric", build_pants(body, 0.018), col("#7c7e80"), PAT_CAMO, tris=1500, step=0.006, rule="lower"))
        pads = []
        for s in ("L", "R"):
            k = SKEL["LowerLeg." + s][0]
            pads.append(ellipsoid(k + V(0, -0.058, -0.01), (0.050, 0.030, 0.065)))
        pcs.append(Piece("kneepads", "Gear", U(pads), col("#2b2c2e"), PAT_HARD, tris=300, step=0.004, rule="knees"))
        pcs.append(Piece("vest", "Gear", vest_sdf(body, P, slick=True), col("#55585a"), PAT_WEBBING, tris=1300, step=0.005, rule="chest"))
        pcs.append(Piece("belt", "Leather", build_belt(body, 0.990, 0.024, 0.020), col("#232323"), PAT_WEBBING, tris=250, step=0.004, rule="hips"))
        gaiter = cone((0, 0.010, 1.47), (0, 0.004, 1.585), 0.075, 0.062) - cone((0, 0.012, 1.40), (0, 0.004, 1.70), 0.050, 0.046)
        gaiter = gaiter.su(torus((0, 0.012, 1.485), 0.085, 0.018, frame(z=(0, 0.3, 1), x=(1, 0, 0))), 0.015)
        pcs.append(Piece("gaiter", "Team", gaiter, (0.85, 0.85, 0.85), PAT_KNIT, tris=500, step=0.004, rule="neck"))
        curls = head.grow(0.014) & scalp_cut(1.750, 1.645, 1.705)
        curls = curls.displace(lambda p: 0.0035 * (np.sin(p[:, 0] * 260) * np.sin(p[:, 1] * 250) * np.sin(p[:, 2] * 270)))
        pcs.append(Piece("hair", "Hair", curls, hairc, tris=900, step=0.0028, rule="head"))
        arc = [V(0.098 * math.cos(a), 0.004, 1.690 + 0.135 * math.sin(a)) for a in np.linspace(0.05, math.pi - 0.05, 9)]
        hs = U([chain(arc, [0.0065] * 9)] + [cyl((sx * 0.090, 0.006, 1.690), 0.030, 0.012, frame(z=(1, 0, 0), y=(0, 0, 1))) for sx in (1, -1)] +
               [chain([V(-0.095, -0.010, 1.680), V(-0.075, -0.075, 1.640), V(-0.025, -0.105, 1.625)], [0.003, 0.003, 0.004])], 0.006)
        pcs.append(Piece("headset", "Gear", hs, col("#2a2b2c"), PAT_HARD, tris=600, step=0.003, rule="head"))
        for s in ("L",):
            pcs.append(Piece("armband_" + s, "Team", arm_band(s, 0.40, 0.026, 0.028, body.arm[s]), (0.85, 0.85, 0.85), tris=180, step=0.004, rule="upperarm" + s))
        hands("Leather", col("#202020"), True)
        boots(col("#2a2622"), 0.30, 1.05)
    return body, pcs


# ==========================================================================
# Blender scene / mesh helpers
# ==========================================================================

def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.unit_settings.system = "METRIC"
    sc.render.fps = FPS
    return sc


def link(obj):
    bpy.context.scene.collection.objects.link(obj)
    return obj


def auto_bbox(f, pad=0.02):
    step = 0.02
    xs = np.arange(-0.95, 0.95 + step, step)
    ys = np.arange(-0.45, 0.45 + step, step)
    zs = np.arange(-0.05, 2.05 + step, step)
    g = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), -1).reshape(-1, 3)
    d = f(g)
    m = d < step * 0.9
    if not m.any():
        raise RuntimeError("empty SDF")
    pts = g[m]
    return pts.min(0) - pad, pts.max(0) + pad


def sdf_mesh(f, step, bbox=None):
    lo, hi = bbox if bbox is not None else auto_bbox(f)
    xs = np.arange(lo[0], hi[0] + step, step)
    ys = np.arange(lo[1], hi[1] + step, step)
    zs = np.arange(lo[2], hi[2] + step, step)
    shape = (len(xs), len(ys), len(zs))
    g = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), -1).reshape(-1, 3)
    out = np.empty(len(g))
    CH = 1_500_000
    for i in range(0, len(g), CH):
        out[i:i + CH] = f(g[i:i + CH])
    vol = np.pad(out.reshape(shape), 1, constant_values=1.0)
    verts, faces, _, _ = marching_cubes(vol.astype(np.float32), 0.0, spacing=(step,) * 3)
    verts = verts + (lo - step)
    # orient faces outward (positive signed volume)
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    if np.einsum("ij,ij->i", a, np.cross(b, c)).sum() < 0:
        faces = faces[:, ::-1]
    return verts, faces


def mesh_object(name, verts, faces):
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(verts))
    me.vertices.foreach_set("co", verts.astype(np.float32).ravel())
    me.loops.add(faces.size)
    me.loops.foreach_set("vertex_index", faces.astype(np.int32).ravel())
    me.polygons.add(len(faces))
    me.polygons.foreach_set("loop_start", np.arange(0, faces.size, 3, dtype=np.int32))
    me.update(calc_edges=True)
    me.validate()
    return link(bpy.data.objects.new(name, me))


def keep_big_islands(obj, min_frac=0.02):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    seen = set()
    islands = []
    for f in bm.faces:
        if f.index in seen:
            continue
        stack = [f]
        isl = []
        seen.add(f.index)
        while stack:
            g = stack.pop()
            isl.append(g)
            for e in g.edges:
                for h in e.link_faces:
                    if h.index not in seen:
                        seen.add(h.index)
                        stack.append(h)
        islands.append(isl)
    total = len(bm.faces)
    kill = [f for isl in islands if len(isl) < total * min_frac for f in isl]
    if kill:
        bmesh.ops.delete(bm, geom=kill, context="FACES")
    bm.to_mesh(obj.data)
    bm.free()


def apply_mod(obj, mod):
    with bpy.context.temp_override(object=obj, active_object=obj, selected_objects=[obj]):
        bpy.ops.object.modifier_apply(modifier=mod.name)


def clean_mesh(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    bmesh.ops.dissolve_degenerate(bm, edges=bm.edges, dist=1e-5)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    bm.to_mesh(obj.data)
    bm.free()


def decimate(obj, target):
    clean_mesh(obj)
    for _ in range(6):
        n = len(obj.data.polygons)
        if n <= target * 1.03:
            break
        m = obj.modifiers.new("dec", "DECIMATE")
        m.ratio = max(target / n, 0.02)
        m.use_collapse_triangulate = True
        apply_mod(obj, m)
        clean_mesh(obj)
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    for _ in range(2):
        bmesh.ops.smooth_vert(bm, verts=bm.verts, factor=0.5, use_axis_x=True, use_axis_y=True, use_axis_z=True)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.shade_smooth()


def build_piece_object(pc):
    v, f = sdf_mesh(pc.sdf, pc.step, pc.bbox)
    o = mesh_object(pc.name, v, f)
    keep_big_islands(o)
    hi = len(o.data.polygons)
    decimate(o, pc.tris)
    return o, hi


# ==========================================================================
# Armature + skinning
# ==========================================================================

REST = {}   # bone name -> rest matrix (armature space)
LEN = {}


def create_armature():
    arm = bpy.data.armatures.new("Rig")
    ob = link(bpy.data.objects.new("Rig", arm))
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    for n in BONE_ORDER:
        h, t, par = SKEL[n]
        eb = arm.edit_bones.new(n)
        eb.head = Vector(h)
        eb.tail = Vector(t)
        eb.align_roll(Vector(bone_axes(n)[2]))
        if par:
            eb.parent = arm.edit_bones[par]
            eb.use_connect = False
    bpy.ops.object.mode_set(mode="OBJECT")
    for b in arm.bones:
        REST[b.name] = b.matrix_local.copy()
        LEN[b.name] = b.length
    return ob


def select_only(objs, active):
    bpy.context.view_layer.update()
    for o in bpy.context.scene.objects:
        o.select_set(False)
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = active


def make_proxy(body, P, rig):
    """Closed whole-body mesh used only to compute automatic (heat) weights."""
    f = U([body.whole(), head_sdf(P)], 0.02)
    v, fa = sdf_mesh(f, 0.008)
    o = mesh_object("proxy", v, fa)
    keep_big_islands(o, 0.2)
    decimate(o, 9000)
    select_only([o, rig], rig)
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    names = {g.index: g.name for g in o.vertex_groups}
    W = []
    for vtx in o.data.vertices:
        W.append({names[g.group]: g.weight for g in vtx.groups if g.weight > 1e-4})
    empty = sum(1 for w in W if not w)
    log("proxy weights: %d verts, %d unweighted" % (len(W), empty))
    me = o.data
    bvh = BVHTree.FromPolygons([v.co.copy() for v in me.vertices], [tuple(p.vertices) for p in me.polygons])
    return o, bvh, W


RULES = {
    "skinhead": {"Head", "Neck", "Chest"},
    "neck": {"Head", "Neck", "Chest"},
    "upper": {"Hips", "Spine", "Chest", "Neck", "Shoulder.L", "Shoulder.R", "UpperArm.L", "UpperArm.R",
              "LowerArm.L", "LowerArm.R", "Hand.L", "Hand.R"},
    "lower": {"Hips", "Spine", "UpperLeg.L", "UpperLeg.R", "LowerLeg.L", "LowerLeg.R", "Foot.L", "Foot.R"},
    "chest": {"Chest", "Spine"},
    "hips": {"Hips", "Spine"},
    "knees": {"UpperLeg.L", "UpperLeg.R", "LowerLeg.L", "LowerLeg.R"},
}
RIGID = {"head": "Head", "handL": "Hand.L", "handR": "Hand.R", "pack": "Chest"}


def bone_dist(p, n):
    h = REST[n].translation
    t = REST[n] @ Vector((0, LEN[n], 0))
    ht = t - h
    s = max(0.0, min(1.0, (p - h).dot(ht) / ht.length_squared))
    return (p - (h + ht * s)).length


def skin_piece(obj, rule, proxy):
    _, bvh, W = proxy
    pm = proxy[0].data
    allowed = None
    rigid = None
    if rule in RIGID:
        rigid = RIGID[rule]
    elif rule in RULES:
        allowed = RULES[rule]
    elif rule.startswith("foot"):
        allowed = {"Foot." + rule[-1], "LowerLeg." + rule[-1]}
    elif rule.startswith("upperarm"):
        allowed = {"UpperArm." + rule[-1], "Shoulder." + rule[-1]}
    for n in BONE_ORDER:
        obj.vertex_groups.new(name=n)
    groups = {g.name: g for g in obj.vertex_groups}
    for v in obj.data.vertices:
        if rigid:
            groups[rigid].add([v.index], 1.0, "REPLACE")
            continue
        loc, nrm, fi, dist = bvh.find_nearest(v.co)
        poly = pm.polygons[fi]
        pts = [pm.vertices[i].co for i in poly.vertices]
        bc = poly_3d_calc(pts, loc)
        w = {}
        for i, b in zip(poly.vertices, bc):
            for k, x in W[i].items():
                w[k] = w.get(k, 0.0) + x * b
        if allowed is not None:
            w = {k: x for k, x in w.items() if k in allowed}
        if rule == "knees":
            sd = "L" if v.co.x > 0 else "R"
            w = {k: x for k, x in w.items() if k.endswith(sd)}
        w = {k: x for k, x in w.items() if x > 0.01}
        if not w:
            cand = allowed if allowed else set(BONE_ORDER)
            w = {min(cand, key=lambda n: bone_dist(v.co, n)): 1.0}
        top = sorted(w.items(), key=lambda kv: -kv[1])[:4]
        tot = sum(x for _, x in top)
        for k, x in top:
            groups[k].add([v.index], x / tot, "REPLACE")


# ==========================================================================
# Pose solver: world-axis FK + analytic two-bone IK, baked to keyframes
# ==========================================================================

def R3(x=0.0, y=0.0, z=0.0, order="YXZ"):
    return Euler((math.radians(x), math.radians(y), math.radians(z)), order).to_matrix()


def T4(v):
    return Matrix.Translation(Vector(v))


def M4(R, t=(0, 0, 0)):
    m = R.to_4x4()
    m.translation = Vector(t)
    return m


def gun_frame(grip, yaw=0.0, pitch=0.0, roll=0.0):
    """Gun frame (Blender axes: -Y barrel, +Z up). pitch>0 = muzzle down, yaw>0 = muzzle left."""
    return M4(R3(z=yaw) @ R3(x=pitch) @ R3(y=roll), grip)


def _gunmount_local():
    a = GRIP_RAKE
    R = Matrix(((-1, 0, 0), (0, -math.cos(a), math.sin(a)), (0, math.sin(a), math.cos(a)))).transposed()
    return M4(R, (-GRIP_V, GRIP_U, 0.0))


GUNMOUNT_LOCAL = _gunmount_local()      # gun frame in Hand.R bone space
HANDGUARD = Vector((0.0, -0.200, 0.080))   # left-hand grip point, gun space
_RLH = Matrix(((0, -1, 0), (0, 0, -1), (1, 0, 0)))


def hand_from_gun(G):
    return G @ GUNMOUNT_LOCAL.inverted()


def lefthand_from_gun(G, hg=None):
    hg = HANDGUARD if hg is None else hg
    return G @ T4(hg) @ _RLH.to_4x4() @ T4((-GRIP_V, -GRIP_U, 0.0))


def hand_from_grip(side, O, z, vhint):
    """Hand bone matrix holding a cylinder at O whose axis (thumb side) is z;
    vhint ~ direction from the palm to the cylinder."""
    z = Vector(z).normalized()
    v = Vector(vhint)
    v = (v - z * v.dot(z)).normalized()
    u = v.cross(z) if side == "R" else z.cross(v)
    xh = -v if side == "R" else v
    m = Matrix((xh, u, z)).transposed().to_4x4()
    m.translation = Vector(O) - u * GRIP_U - v * GRIP_V
    return m


class Pose:
    def __init__(self):
        self.loc = Vector((0, 0, 0))
        self.rot = {}
        self.hand = {}    # side -> (matrix, rel bone or None, pole)
        self.foot = {}    # side -> (ankle pos, pitch, yaw, pole)

    def r(self, bone, x=0.0, y=0.0, z=0.0, order="YXZ"):
        if bone.endswith(".R"):
            y, z = -y, -z
        self.rot[bone] = R3(x, y, z, order)
        return self

    def both(self, base, x=0.0, y=0.0, z=0.0, order="YXZ"):
        self.r(base + ".L", x, y, z, order)
        self.r(base + ".R", x, y, z, order)
        return self

    def rifle(self, G, rel="Chest", left=True, right=True):
        for s in ("L", "R"):
            if "Shoulder." + s not in self.rot:
                self.r("Shoulder." + s, z=-12 if s == "L" else -6)
        if right:
            self.hand["R"] = (hand_from_gun(G), rel, Vector((-0.7, 0.4, -1.0)))
        if left:
            self.hand["L"] = (lefthand_from_gun(G), rel, Vector((0.5, 0.3, -1.0)))
        return self

    def feet(self, l, r, pitch_l=0.0, pitch_r=0.0, yaw_l=0.0, yaw_r=0.0):
        self.foot["L"] = (Vector(l), pitch_l, yaw_l, None)
        self.foot["R"] = (Vector(r), pitch_r, yaw_r, None)
        return self


ANKLE_L = Vector(SKEL["Foot.L"][0])
ANKLE_R = Vector(SKEL["Foot.R"][0])
IK_MISS = {}


def two_bone(S, T, l1, l2, pole):
    d = T - S
    dist = d.length
    dc = min(max(dist, abs(l1 - l2) + 1e-3), (l1 + l2) * 0.9995)
    e = d.normalized()
    p = pole - e * pole.dot(e)
    if p.length < 1e-6:
        p = Vector((0, 1, 0)) - e * e.y
    p.normalize()
    ca = (l1 * l1 + dc * dc - l2 * l2) / (2 * l1 * dc)
    sa = math.sqrt(max(0.0, 1 - ca * ca))
    E = S + (e * ca + p * sa) * l1
    Tc = S + e * dc
    d1 = (E - S).normalized()
    d2 = (Tc - E).normalized()
    X = d1.cross(d2)
    if X.length < 1e-6:
        X = d1.cross(-p)
    X.normalize()
    return E, d1, d2, X, max(0.0, dist - dc)


def frame_m(X, Y, t):
    Z = X.cross(Y).normalized()
    X = Y.cross(Z).normalized()
    m = Matrix((X, Y, Z)).transposed().to_4x4()
    m.translation = t
    return m


def solve(pose, tag=""):
    D, Mx = {}, {}
    I = Matrix.Identity(4)

    def fk(n):
        rest = REST[n]
        par = SKEL[n][2]
        Dp = D[par] if par else I
        L = I
        if n in pose.rot:
            h = rest.translation
            L = T4(h) @ pose.rot[n].to_4x4() @ T4(-h)
        if n == "Hips":
            L = T4(pose.loc) @ L
        D[n] = Dp @ L
        Mx[n] = D[n] @ rest

    for n in ("Hips", "Spine", "Chest", "Neck", "Head", "Shoulder.L", "Shoulder.R"):
        fk(n)
    for s in ("L", "R"):
        up, lo, ha = "UpperArm." + s, "LowerArm." + s, "Hand." + s
        if s in pose.hand:
            Mt, rel, pole = pose.hand[s]
            if rel:
                Mt = D[rel] @ Mt
            S = (D["Shoulder." + s] @ REST[up]).translation
            E, d1, d2, X, miss = two_bone(S, Mt.translation, LEN[up], LEN[lo], pole)
            IK_MISS[tag + ha] = max(IK_MISS.get(tag + ha, 0), miss)
            Mx[up] = frame_m(X, d1, S)
            Mx[lo] = frame_m(X, d2, E)
            mh = Mt.copy()
            mh.translation = E + d2 * LEN[lo]
            Mx[ha] = mh
            for n in (up, lo, ha):
                D[n] = Mx[n] @ REST[n].inverted()
        else:
            for n in (up, lo, ha):
                fk(n)
    for s in ("L", "R"):
        ul, ll, ft = "UpperLeg." + s, "LowerLeg." + s, "Foot." + s
        if s in pose.foot:
            pos, pitch, yaw, pole = pose.foot[s]
            S = (D["Hips"] @ REST[ul]).translation
            if pole is None:
                hy = D["Hips"].to_3x3() @ Vector((0, -1, 0))
                pole = (R3(z=yaw if s == "L" else -yaw) @ Vector((0, -1, 0))) + hy * 0.5 + Vector((0.15 if s == "L" else -0.15, 0, 0))
            E, d1, d2, X, miss = two_bone(S, pos, LEN[ul], LEN[ll], pole)
            IK_MISS[tag + ft] = max(IK_MISS.get(tag + ft, 0), miss)
            Mx[ul] = frame_m(X, d1, S)
            Mx[ll] = frame_m(X, d2, E)
            rr = (R3(z=yaw if s == "L" else -yaw) @ R3(x=pitch)).to_4x4() @ REST[ft].to_3x3().to_4x4()
            rr.translation = E + d2 * LEN[ll]
            Mx[ft] = rr
            for n in (ul, ll, ft):
                D[n] = Mx[n] @ REST[n].inverted()
        else:
            for n in (ul, ll, ft):
                fk(n)
    out = {}
    for n in BONE_ORDER:
        par = SKEL[n][2]
        base = (D[par] if par else I) @ REST[n]
        out[n] = base.inverted() @ Mx[n]
    return out, Mx


def bake_action(rig, name, fn, duration, loop=True):
    n = max(2, int(round(duration * FPS)))
    frames = list(range(n + 1))
    act = bpy.data.actions.new(name)
    data = {b: {"q": [], "t": []} for b in BONE_ORDER}
    prevq = {}
    for f in frames:
        t = (f / n) * duration
        pose = fn(t if not loop else (t % duration))
        basis, _ = solve(pose, name + ":")
        for b in BONE_ORDER:
            q = basis[b].to_quaternion()
            if b in prevq and prevq[b].dot(q) < 0:
                q.negate()
            prevq[b] = q
            data[b]["q"].append(q)
            data[b]["t"].append(basis[b].translation)
    for b in BONE_ORDER:
        grp = act.groups.new(b)
        chans = [("rotation_quaternion", 4, lambda k, i: data[b]["q"][k][i])]
        if b == "Hips":
            chans.append(("location", 3, lambda k, i: data[b]["t"][k][i]))
        for path, cnt, get in chans:
            for i in range(cnt):
                fc = act.fcurves.new('pose.bones["%s"].%s' % (b, path), index=i, action_group=b)
                fc.keyframe_points.add(len(frames))
                co = []
                for k, f in enumerate(frames):
                    co += [float(f), float(get(k, i))]
                fc.keyframe_points.foreach_set("co", co)
                for kp in fc.keyframe_points:
                    kp.interpolation = "LINEAR"
                fc.update()
    act.use_fake_user = True
    act.use_frame_range = True
    act.frame_start = 0
    act.frame_end = n
    act.use_cyclic = loop
    tr = rig.animation_data.nla_tracks.new()
    tr.name = name
    st = tr.strips.new(name, 0, act)
    st.extrapolation = "NOTHING"
    tr.mute = True
    return act


# ==========================================================================
# Animation library (all clips in place; the game moves the root)
# ==========================================================================

def sm(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def kf(t, keys):
    """Smooth piecewise interpolation through (time, value) keys."""
    if t <= keys[0][0]:
        v = keys[0][1]
    elif t >= keys[-1][0]:
        v = keys[-1][1]
    else:
        for (t0, a), (t1, b) in zip(keys[:-1], keys[1:]):
            if t0 <= t <= t1:
                k = sm((t - t0) / (t1 - t0))
                if isinstance(a, (tuple, list, Vector)):
                    return Vector(a).lerp(Vector(b), k)
                return a + (b - a) * k
    return Vector(v) if isinstance(v, (tuple, list)) else v


TAU = 2 * math.pi
LOW_READY = gun_frame((-0.090, -0.255, 1.165), yaw=26, pitch=22)
CARRY = gun_frame((-0.090, -0.255, 1.170), yaw=24, pitch=16)
HIGH_PORT = gun_frame((-0.090, -0.250, 1.230), yaw=38, pitch=-40, roll=-10)
AIM = gun_frame((-0.125, -0.370, 1.375), yaw=0, pitch=0)


def ankle(side, x=None, y=0.0, z=0.0):
    a = (ANKLE_L if side == "L" else ANKLE_R).copy()
    if x is not None:
        a.x = x if side == "L" else -x
    a.y += y
    a.z += z
    return a


def gait(t, period, stride, lift, stance, drop, bob, dirv=(0, -1, 0), width=0.115, sway=0.018,
         lean=4.0, toe=28.0, heel=14.0, yaw=6.0, gun=CARRY, arm_bob=0.0):
    p = Pose()
    ph = (t / period) % 1.0
    dv = Vector(dirv).normalized()
    for s, off in (("L", 0.0), ("R", 0.5)):
        q = (ph + off) % 1.0
        if q < stance:
            k = q / stance
            a = 0.5 - k
            h = 0.0
            if k < 0.18:
                pitch = -heel * (1 - k / 0.18)
            elif k > 0.55:
                pitch = toe * ((k - 0.55) / 0.45) ** 2
            else:
                pitch = 0.0
        else:
            k = (q - stance) / (1 - stance)
            a = -0.5 + sm(k)
            h = lift * math.sin(math.pi * k) ** 0.9
            pitch = toe + (-heel - toe) * sm(k * 1.15)
        pos = ankle(s, width) + dv * (a * stride)
        pos.z += h
        if pitch > 0:
            pos.z += 0.125 * math.sin(math.radians(pitch))
        else:
            pos.z += 0.045 * math.sin(math.radians(-pitch))
        p.foot[s] = (pos, pitch, 6.0, None)
    c2 = math.cos(2 * TAU * ph)
    p.loc = Vector((sway * math.sin(TAU * ph) * (1 if abs(dv.y) > 0.5 else 0.4), 0, -drop - bob * (0.5 + 0.5 * c2)))
    sw = math.sin(TAU * ph)
    p.r("Hips", x=lean * 0.25, z=yaw * sw)
    p.r("Spine", x=lean * 0.35, z=-yaw * 0.6 * sw)
    p.r("Chest", x=lean * 0.4 + arm_bob * c2, z=-yaw * 0.5 * sw)
    p.r("Neck", x=-lean * 0.4)
    p.r("Head", x=-lean * 0.4 - arm_bob * c2 * 0.8, z=yaw * 0.3 * sw)
    if gun is not None:
        p.rifle(gun)
    return p


def a_idle(t):
    br = math.sin(TAU * t / 3.0)
    sw = math.sin(TAU * t / 6.0)
    p = Pose()
    p.loc = Vector((0.012 * sw, 0.0, -0.018 - 0.004 * br))
    p.r("Hips", y=-2 * sw, z=-4)
    p.r("Spine", x=2 + 0.8 * br, z=2)
    p.r("Chest", x=2 + 1.2 * br, z=2)
    p.r("Neck", x=-1)
    p.r("Head", x=-2 - 0.8 * br, z=4 + 3 * math.sin(TAU * t / 3.0 + 1))
    p.feet(ankle("L", 0.135, -0.03), ankle("R", 0.120, 0.03), yaw_l=10, yaw_r=14)
    p.foot["L"] = (p.foot["L"][0], 0.0, 10.0, None)
    p.foot["R"] = (p.foot["R"][0], 0.0, 14.0, None)
    p.r("Shoulder.L", z=0)
    p.rifle(gun_frame((-0.090, -0.255, 1.165 + 0.004 * br), yaw=26, pitch=22))
    return p


def a_aim(t):
    br = math.sin(TAU * t / 2.0)
    p = Pose()
    p.loc = Vector((0.0, 0.0, -0.035 - 0.003 * br))
    p.r("Hips", z=-22)
    p.r("Spine", x=4, z=-5)
    p.r("Chest", x=6 + 0.6 * br, z=-6)
    p.r("Neck", x=2, z=12)
    p.r("Head", x=6, z=18, y=-6)
    p.r("Shoulder.R", z=-6, y=4)
    p.r("Shoulder.L", z=-4)
    p.foot["L"] = (ankle("L", 0.15, -0.12), 0.0, -8.0, None)
    p.foot["R"] = (ankle("R", 0.10, 0.12), 0.0, 40.0, None)
    G = gun_frame((-0.125, -0.370, 1.375 + 0.002 * br), yaw=0, pitch=0)
    p.rifle(G, rel=None)
    return p


def a_crouch_idle(t):
    br = math.sin(TAU * t / 3.0)
    p = Pose()
    p.loc = Vector((0, 0.06, -0.36 - 0.004 * br))
    p.r("Hips", x=10, z=-6)
    p.r("Spine", x=10 + br, z=3)
    p.r("Chest", x=8 + br, z=3)
    p.r("Neck", x=-8)
    p.r("Head", x=-12, z=4)
    p.foot["L"] = (ankle("L", 0.15, -0.10), 0.0, 8.0, None)
    p.foot["R"] = (ankle("R", 0.13, 0.12, 0.05), 32.0, 14.0, None)
    p.rifle(gun_frame((-0.090, -0.265, 1.165), yaw=26, pitch=18))
    return p


def a_jump(t):
    p = Pose()
    # 0-0.12 crouch, 0.12-0.24 push off, then air tuck held
    p.loc = Vector((0, 0, kf(t, [(0, -0.02), (0.12, -0.14), (0.24, 0.04), (0.45, 0.02)])))
    lean = kf(t, [(0, 4), (0.12, 16), (0.24, 2), (0.45, 6)])
    p.r("Hips", x=lean * 0.3)
    p.r("Spine", x=lean * 0.4)
    p.r("Chest", x=lean * 0.4)
    p.r("Head", x=-lean * 0.6)
    lz = kf(t, [(0, 0.0), (0.18, 0.0), (0.30, 0.10), (0.5, 0.30)])
    rz = kf(t, [(0, 0.0), (0.20, 0.0), (0.32, 0.06), (0.5, 0.20)])
    pl = kf(t, [(0, 0), (0.18, 0), (0.26, 40), (0.5, 20)])
    pr = kf(t, [(0, 0), (0.20, 0), (0.28, 45), (0.5, 35)])
    p.foot["L"] = (ankle("L", 0.13, kf(t, [(0, -0.02), (0.5, -0.12)]), lz), pl, 6.0, None)
    p.foot["R"] = (ankle("R", 0.13, kf(t, [(0, 0.02), (0.5, 0.10)]), rz), pr, 6.0, None)
    p.rifle(gun_frame((-0.100, -0.265, kf(t, [(0, 1.15), (0.12, 1.13), (0.3, 1.19), (0.5, 1.17)])), yaw=22, pitch=kf(t, [(0, 18), (0.3, 8), (0.5, 12)])))
    return p


def a_fall(t):
    ph = TAU * t / 0.8
    p = Pose()
    p.loc = Vector((0, 0, 0.02))
    p.r("Hips", x=-4 + 2 * math.sin(ph))
    p.r("Spine", x=6)
    p.r("Chest", x=4 + 2 * math.sin(ph + 1))
    p.r("Head", x=-8)
    p.foot["L"] = (ankle("L", 0.14, -0.10 + 0.05 * math.sin(ph), 0.16 + 0.04 * math.cos(ph)), 30 + 8 * math.sin(ph), 6.0, None)
    p.foot["R"] = (ankle("R", 0.14, 0.08 - 0.05 * math.sin(ph), 0.12 - 0.04 * math.cos(ph)), 40 - 8 * math.sin(ph), 6.0, None)
    p.rifle(gun_frame((-0.10, -0.26, 1.20 + 0.01 * math.sin(ph * 2)), yaw=24, pitch=-5 + 4 * math.sin(ph)))
    return p


def a_land(t):
    p = Pose()
    p.loc = Vector((0, 0.01, kf(t, [(0, -0.06), (0.1, -0.24), (0.32, -0.06), (0.5, -0.02)])))
    lean = kf(t, [(0, 8), (0.1, 22), (0.32, 8), (0.5, 4)])
    p.r("Hips", x=lean * 0.4)
    p.r("Spine", x=lean * 0.35)
    p.r("Chest", x=lean * 0.35)
    p.r("Head", x=-lean * 0.6)
    p.foot["L"] = (ankle("L", 0.14, -0.04), 0.0, 8.0, None)
    p.foot["R"] = (ankle("R", 0.14, 0.04), 0.0, 8.0, None)
    p.rifle(gun_frame((-0.090, -0.255, kf(t, [(0, 1.17), (0.1, 1.15), (0.5, 1.17)])), yaw=24, pitch=kf(t, [(0, 10), (0.1, 22), (0.5, 16)])))
    return p


def a_slide(t):
    ph = TAU * t / 0.6
    j = 0.004 * math.sin(ph * 2)
    p = Pose()
    p.loc = Vector((0, 0.10, -0.50 + j))
    p.r("Hips", x=-22, z=-10)
    p.r("Spine", x=8, z=4)
    p.r("Chest", x=6, z=4)
    p.r("Neck", x=4)
    p.r("Head", x=10, z=6)
    p.foot["L"] = (Vector((0.13, -0.56, 0.10 + j)), -25.0, 10.0, None)
    p.foot["R"] = (Vector((-0.20, 0.02, 0.07 + j)), 10.0, 40.0, Vector((-1.0, -0.3, 0.2)))
    p.rifle(gun_frame((-0.10, -0.30, 1.20), yaw=20, pitch=-5))
    return p


def a_vault(t):
    p = Pose()
    p.loc = Vector((kf(t, [(0, 0), (0.35, 0.10), (0.8, 0)]), 0, kf(t, [(0, -0.05), (0.12, -0.12), (0.32, 0.38), (0.55, 0.30), (0.72, -0.10), (0.8, -0.04)])))
    lean = kf(t, [(0, 8), (0.15, 25), (0.35, 30), (0.6, 10), (0.8, 6)])
    p.r("Hips", x=lean * 0.5, y=kf(t, [(0, 0), (0.35, -18), (0.65, 0)]), z=kf(t, [(0, 0), (0.35, -20), (0.7, 0)]))
    p.r("Spine", x=lean * 0.3)
    p.r("Chest", x=lean * 0.3, z=kf(t, [(0, 0), (0.35, 15), (0.7, 0)]))
    p.r("Head", x=-lean * 0.6)
    hl = kf(t, [(0, (0.20, -0.28, 1.08)), (0.15, (0.16, -0.36, 1.00)), (0.45, (0.16, -0.36, 0.98)), (0.6, (0.28, -0.22, 1.20)), (0.8, (0.20, -0.28, 1.10))])
    p.hand["L"] = (hand_from_grip("L", hl, (0, -1, 0), (0, 0, -1)), "Chest", Vector((0.6, 0.5, 0.0)))
    p.rifle(gun_frame((-0.10, -0.24, 1.20), yaw=25, pitch=-20), left=False)
    lk = kf(t, [(0, (0.12, -0.03, 0.088)), (0.15, (0.12, 0.0, 0.10)), (0.4, (-0.05, -0.10, 0.60)), (0.62, (0.05, -0.25, 0.40)), (0.8, (0.13, -0.05, 0.088))])
    rk = kf(t, [(0, (-0.12, 0.03, 0.088)), (0.18, (-0.12, 0.05, 0.15)), (0.4, (-0.28, -0.05, 0.50)), (0.65, (-0.15, -0.10, 0.25)), (0.8, (-0.12, 0.04, 0.088))])
    lift = max(0.0, p.loc.z) * 0.8
    lk = lk + Vector((0, 0, lift))
    rk = rk + Vector((0, 0, lift))
    p.foot["L"] = (lk, kf(t, [(0, 0), (0.4, 30), (0.8, 0)]), 6.0, Vector((-0.6, -1, 0.3)))
    p.foot["R"] = (rk, kf(t, [(0, 0), (0.4, 40), (0.8, 0)]), 6.0, Vector((-1.0, -0.5, 0.2)))
    return p


def a_skydive(t):
    ph = TAU * t / 1.6
    w = math.sin(ph)
    p = Pose()
    p.loc = Vector((0, 0, 0.0))
    p.r("Hips", x=82 + 2 * w, y=2 * math.sin(ph * 0.5))
    p.r("Spine", x=-10)
    p.r("Chest", x=-8 + w)
    p.r("Neck", x=-25)
    p.r("Head", x=-28)
    p.both("Shoulder", z=0, y=-8)
    p.both("UpperArm", y=-55 + 3 * w, x=-55, z=0)
    p.both("LowerArm", x=-70 + 4 * math.sin(ph + 1))
    p.both("Hand", x=-10)
    p.both("UpperLeg", y=-14, x=-10 + 2 * w)
    p.both("LowerLeg", x=65 + 4 * math.sin(ph + 2))
    p.both("Foot", x=30)
    return p


def a_glide(t):
    ph = TAU * t / 2.0
    w = math.sin(ph)
    p = Pose()
    p.loc = Vector((0, 0, 0))
    p.r("Hips", x=-6 + 2 * w, y=2 * math.sin(ph * 0.5))
    p.r("Spine", x=2)
    p.r("Chest", x=-2)
    p.r("Neck", x=-6)
    p.r("Head", x=-10)
    p.both("Shoulder", y=-14)
    bar_z = 2.02 + 0.006 * w
    for s, sx in (("L", 1), ("R", -1)):
        O = Vector((0.23 * sx, -0.05, bar_z))
        p.hand[s] = (hand_from_grip(s, O, (-sx, 0, 0), (0, -1, 0)), None, Vector((sx * 1.0, 0.4, -0.2)))
    p.r("UpperLeg.L", x=-18 + 4 * w, y=-3)
    p.r("UpperLeg.R", x=-8 - 4 * w, y=-3)
    p.both("LowerLeg", x=25)
    p.r("LowerLeg.R", x=35 + 5 * w)
    p.both("Foot", x=25)
    return p


def a_death(t):
    p = Pose()
    fall = sm((t - 0.45) / 0.55)
    p.loc = Vector((0, kf(t, [(0, 0), (0.2, 0.04), (0.45, 0.10), (1.0, 0.52)]),
                    kf(t, [(0, 0), (0.2, -0.04), (0.45, -0.32), (1.0, -0.86), (1.12, -0.83), (1.4, -0.86)])))
    p.r("Hips", x=kf(t, [(0, 0), (0.2, -6), (0.45, 10), (1.0, -86), (1.4, -88)]), z=kf(t, [(0, 0), (0.45, 18), (1.0, 25)]),
        y=kf(t, [(0, 0), (0.5, 6), (1.0, 4)]))
    p.r("Spine", x=kf(t, [(0, 0), (0.2, -12), (0.45, 12), (1.0, 2), (1.1, -6), (1.4, -2)]))
    p.r("Chest", x=kf(t, [(0, 0), (0.2, -14), (0.45, 10), (1.0, 0), (1.4, -2)]), z=kf(t, [(0, 0), (0.45, -10), (1.4, -8)]))
    p.r("Neck", x=kf(t, [(0, 0), (0.2, -20), (0.5, 15), (1.0, 0), (1.15, 15), (1.4, 8)]))
    p.r("Head", x=kf(t, [(0, 0), (0.2, -20), (0.5, 10), (1.0, -5), (1.4, 0)]), z=kf(t, [(0, 0), (1.0, 35), (1.4, 40)]))
    p.both("Shoulder", y=kf(t, [(0, 0), (0.3, -10), (1.4, -5)]))
    p.r("UpperArm.L", y=kf(t, [(0, 40), (0.3, 10), (1.0, -40), (1.4, -45)]), x=kf(t, [(0, -10), (0.3, -40), (1.0, 20), (1.4, 10)]))
    p.r("UpperArm.R", y=kf(t, [(0, 40), (0.3, 0), (1.0, -20), (1.4, -30)]), x=kf(t, [(0, -20), (0.3, -60), (1.0, -10), (1.4, 0)]))
    p.r("LowerArm.L", x=kf(t, [(0, -30), (0.3, -60), (1.0, -20), (1.4, -35)]))
    p.r("LowerArm.R", x=kf(t, [(0, -50), (0.3, -40), (1.0, -50), (1.4, -60)]))
    p.r("UpperLeg.L", x=kf(t, [(0, 0), (0.45, -40), (1.0, -10), (1.4, -14)]), y=-6)
    p.r("UpperLeg.R", x=kf(t, [(0, 0), (0.45, -10), (1.0, -32), (1.4, -38)]), y=-10)
    p.r("LowerLeg.L", x=kf(t, [(0, 0), (0.45, 70), (1.0, 20), (1.4, 22)]))
    p.r("LowerLeg.R", x=kf(t, [(0, 0), (0.45, 40), (1.0, 50), (1.4, 60)]))
    p.both("Foot", x=kf(t, [(0, 0), (0.45, 20), (1.4, 35)]))
    return p


def a_swing(t):
    p = Pose()
    p.loc = Vector((0, 0, kf(t, [(0, -0.03), (0.25, -0.02), (0.42, -0.08), (0.8, -0.03)])))
    p.r("Hips", z=kf(t, [(0, -6), (0.25, -18), (0.42, 12), (0.8, -6)]))
    p.r("Spine", x=kf(t, [(0, 4), (0.25, -6), (0.42, 18), (0.8, 4)]), z=kf(t, [(0, 0), (0.25, -12), (0.42, 10), (0.8, 0)]))
    p.r("Chest", x=kf(t, [(0, 4), (0.25, -6), (0.42, 14), (0.8, 4)]), z=kf(t, [(0, 0), (0.25, -12), (0.42, 10), (0.8, 0)]))
    p.r("Head", x=kf(t, [(0, 0), (0.25, 4), (0.42, -18), (0.8, 0)]), z=kf(t, [(0, 6), (0.25, 22), (0.42, -12), (0.8, 6)]))
    O = kf(t, [(0, (-0.20, -0.30, 1.10)), (0.25, (-0.24, 0.04, 1.66)), (0.34, (-0.18, -0.30, 1.58)), (0.42, (-0.10, -0.52, 1.05)),
               (0.55, (-0.12, -0.45, 0.98)), (0.8, (-0.20, -0.30, 1.10))])
    ax = kf(t, [(0, (0.0, -0.6, 1.0)), (0.25, (0.0, 0.8, 0.6)), (0.34, (0.0, -0.4, 1.0)), (0.42, (0.0, -1.0, -0.3)),
                (0.55, (0.0, -0.9, -0.5)), (0.8, (0.0, -0.6, 1.0))])
    p.hand["R"] = (hand_from_grip("R", O, ax, (1, 0, 0)), None, Vector((-1.0, 0.5, -0.6)))
    p.r("UpperArm.L", y=kf(t, [(0, 45), (0.25, 30), (0.42, 50), (0.8, 45)]), x=kf(t, [(0, -20), (0.25, -55), (0.42, -10), (0.8, -20)]))
    p.r("LowerArm.L", x=kf(t, [(0, -60), (0.25, -80), (0.42, -50), (0.8, -60)]))
    p.foot["L"] = (ankle("L", 0.15, -0.14), 0.0, 0.0, None)
    p.foot["R"] = (ankle("R", 0.12, 0.10), 0.0, 25.0, None)
    return p


ACTIONS = [
    # name, function, duration (s), loops
    ("Idle", a_idle, 6.0, True),
    ("Walk", lambda t: gait(t, 1.0, 0.70, 0.11, 0.60, 0.035, 0.022), 1.0, True),
    ("Run", lambda t: gait(t, 0.70, 1.00, 0.20, 0.36, 0.085, 0.045, lean=12, toe=40, heel=8, yaw=10, arm_bob=2.0), 0.70, True),
    ("Sprint", lambda t: gait(t, 0.60, 1.15, 0.26, 0.30, 0.12, 0.05, lean=20, toe=50, heel=4, yaw=12, gun=HIGH_PORT, arm_bob=3.0), 0.60, True),
    ("StrafeLeft", lambda t: gait(t, 0.90, 0.30, 0.09, 0.58, 0.04, 0.018, dirv=(1, 0, 0), width=0.15, lean=4, toe=18, heel=6, yaw=3), 0.90, True),
    ("StrafeRight", lambda t: gait(t, 0.90, 0.30, 0.09, 0.58, 0.04, 0.018, dirv=(-1, 0, 0), width=0.15, lean=4, toe=18, heel=6, yaw=3), 0.90, True),
    ("WalkBack", lambda t: gait(1.1 - t, 1.1, 0.55, 0.10, 0.60, 0.04, 0.018, lean=2, toe=22, heel=10, yaw=4), 1.1, True),
    ("CrouchIdle", a_crouch_idle, 3.0, True),
    ("CrouchWalk", lambda t: gait(t, 1.1, 0.55, 0.10, 0.62, 0.33, 0.018, width=0.15, lean=22, toe=25, heel=8, yaw=5,
                                  gun=gun_frame((-0.09, -0.265, 1.165), yaw=26, pitch=18)), 1.1, True),
    ("Jump", a_jump, 0.5, False),
    ("Fall", a_fall, 0.8, True),
    ("Land", a_land, 0.5, False),
    ("Slide", a_slide, 0.6, True),
    ("Vault", a_vault, 0.8, False),
    ("Skydive", a_skydive, 1.6, True),
    ("Glide", a_glide, 2.0, True),
    ("Death", a_death, 1.4, False),
    ("Swing", a_swing, 0.8, False),
    ("AimIdle", a_aim, 2.0, True),
]
ACTION_NAMES = [a[0] for a in ACTIONS]


def bake_all_actions(rig):
    rig.animation_data_create()
    IK_MISS.clear()
    for name, fn, dur, loop in ACTIONS:
        bake_action(rig, name, fn, dur, loop)
    bad = {k: round(v, 3) for k, v in IK_MISS.items() if v > 0.02}
    if bad:
        log("IK targets out of reach (m):", bad)


def add_gunmount(rig):
    gm = link(bpy.data.objects.new("GunMount", None))
    gm.empty_display_type = "ARROWS"
    gm.empty_display_size = 0.08
    gm.parent = rig
    gm.parent_type = "BONE"
    gm.parent_bone = "Hand.R"
    bpy.context.view_layer.update()
    gm.matrix_world = REST["Hand.R"] @ GUNMOUNT_LOCAL
    return gm


# ==========================================================================
# Assembly
# ==========================================================================

MATERIALS = ["Skin", "Hair", "Fabric", "Leather", "Gear", "Team"]


def set_piece_attrs(o, pc):
    me = o.data
    a = me.color_attributes.new("tint", "FLOAT_COLOR", "POINT")
    n = len(me.vertices)
    a.data.foreach_set("color", np.tile(np.array([*pc.tint, 1.0], dtype=np.float32), n))
    b = me.attributes.new("pat", "FLOAT", "POINT")
    b.data.foreach_set("value", np.full(n, float(pc.pat), dtype=np.float32))


def build_character(cid):
    P = designs()[cid]
    body, pcs = build_pieces(cid, P)
    rig = create_armature()
    proxy = make_proxy(body, P, rig)
    mats = {}
    for m in MATERIALS:
        mats[m] = bpy.data.materials.new(m)
        mats[m].use_nodes = True
    objs = []
    stats = {}
    for pc in pcs:
        o, hi = build_piece_object(pc)
        rule = pc.rule
        if pc.name == "head":
            rule = "skinhead"
        elif pc.name.startswith("hand_"):
            rule = "hand" + pc.name[-1]
        elif pc.name in ("pack", "bag", "bedroll"):
            rule = "pack"
        skin_piece(o, rule, proxy)
        set_piece_attrs(o, pc)
        for m in MATERIALS:
            o.data.materials.append(mats[m])
        idx = MATERIALS.index(pc.mat)
        o.data.polygons.foreach_set("material_index", np.full(len(o.data.polygons), idx, dtype=np.int32))
        stats[pc.name] = len(o.data.polygons)
        objs.append(o)
    bpy.data.objects.remove(proxy[0])
    # one UV layout per material (each material owns its whole 0-1 texture)
    groups = []
    for m in MATERIALS:
        g = [o for o, pc in zip(objs, pcs) if pc.mat == m]
        if not g:
            continue
        select_only(g, g[0])
        if len(g) > 1:
            bpy.ops.object.join()
        go = bpy.context.view_layer.objects.active
        unwrap_object(go)
        groups.append(go)
    objs = groups
    select_only(objs, objs[0])
    bpy.ops.object.join()
    mesh = bpy.context.view_layer.objects.active
    mesh.name = "Body_" + cid.upper()
    mesh.data.name = mesh.name
    # drop unused material slots
    used = sorted(set(p.material_index for p in mesh.data.polygons))
    for i in reversed(range(len(mesh.material_slots))):
        if i not in used:
            mesh.active_material_index = i
            with bpy.context.temp_override(object=mesh):
                bpy.ops.object.material_slot_remove()
    log("char %s: %d tris in %d pieces" % (cid, len(mesh.data.polygons), len(pcs)))
    return P, rig, mesh, stats


def attach_mesh(rig, mesh):
    mesh.parent = rig
    md = mesh.modifiers.new("Armature", "ARMATURE")
    md.object = rig


# ==========================================================================
# Materials: procedural Cycles shaders baked to small PBR textures
# ==========================================================================

class NB:
    """Tiny node-graph builder."""

    def __init__(self, tree):
        self.t = tree

    def node(self, kind, **props):
        n = self.t.nodes.new(kind)
        for k, v in props.items():
            setattr(n, k, v)
        return n

    def inp(self, sock, val):
        if isinstance(val, bpy.types.NodeSocket):
            self.t.links.new(val, sock)
        else:
            if isinstance(val, (tuple, list)) and len(val) == 3 and sock.type == "RGBA":
                val = (*val, 1.0)
            sock.default_value = val

    def math(self, op, a, b=0.0, clamp=False):
        n = self.node("ShaderNodeMath", operation=op, use_clamp=clamp)
        self.inp(n.inputs[0], a)
        self.inp(n.inputs[1], b)
        return n.outputs[0]

    def vmath(self, op, a, b=(0, 0, 0)):
        n = self.node("ShaderNodeVectorMath", operation=op)
        self.inp(n.inputs[0], a)
        self.inp(n.inputs[1], b)
        return n.outputs["Value"] if op in ("LENGTH", "DOT_PRODUCT", "DISTANCE") else n.outputs["Vector"]

    def smooth(self, lo, hi, x):
        n = self.node("ShaderNodeMapRange", interpolation_type="SMOOTHSTEP")
        self.inp(n.inputs["Value"], x)
        n.inputs["From Min"].default_value = lo
        n.inputs["From Max"].default_value = hi
        return n.outputs["Result"]

    def mixc(self, fac, a, b):
        n = self.node("ShaderNodeMix", data_type="RGBA", clamp_factor=True)
        self.inp(n.inputs[0], fac)
        self.inp(n.inputs[6], a)
        self.inp(n.inputs[7], b)
        return n.outputs[2]

    def mulc(self, a, b):
        n = self.node("ShaderNodeMix", data_type="RGBA", blend_type="MULTIPLY", clamp_factor=True)
        n.inputs[0].default_value = 1.0
        self.inp(n.inputs[6], a)
        self.inp(n.inputs[7], b)
        return n.outputs[2]

    def mixf(self, fac, a, b):
        n = self.node("ShaderNodeMix", data_type="FLOAT", clamp_factor=True)
        self.inp(n.inputs[0], fac)
        self.inp(n.inputs[2], a)
        self.inp(n.inputs[3], b)
        return n.outputs[0]

    def gray(self, v):
        n = self.node("ShaderNodeCombineColor")
        for i in range(3):
            self.inp(n.inputs[i], v)
        return n.outputs[0]

    def pos(self, scale=(1, 1, 1), offset=(0, 0, 0)):
        tc = self.node("ShaderNodeTexCoord")
        mp = self.node("ShaderNodeMapping")
        self.t.links.new(tc.outputs["Object"], mp.inputs["Vector"])
        mp.inputs["Location"].default_value = offset
        mp.inputs["Scale"].default_value = scale
        return mp.outputs["Vector"]

    def noise(self, vec, scale, detail=4.0, rough=0.55, distortion=0.0, color=False):
        n = self.node("ShaderNodeTexNoise")
        self.inp(n.inputs["Vector"], vec)
        n.inputs["Scale"].default_value = scale
        n.inputs["Detail"].default_value = detail
        n.inputs["Roughness"].default_value = rough
        n.inputs["Distortion"].default_value = distortion
        return n.outputs["Color" if color else "Fac"]

    def wave(self, vec, scale, axis="X", distortion=0.0, profile="SIN"):
        n = self.node("ShaderNodeTexWave", wave_type="BANDS", bands_direction=axis, wave_profile=profile)
        self.inp(n.inputs["Vector"], vec)
        n.inputs["Scale"].default_value = scale
        n.inputs["Distortion"].default_value = distortion
        return n.outputs["Fac"]

    def voronoi(self, vec, scale, feature="F1"):
        n = self.node("ShaderNodeTexVoronoi", feature=feature)
        self.inp(n.inputs["Vector"], vec)
        n.inputs["Scale"].default_value = scale
        return n.outputs["Distance"]

    def attr(self, name, out="Color"):
        n = self.node("ShaderNodeAttribute", attribute_name=name)
        return n.outputs[out]

    def ao(self, dist, samples=16):
        n = self.node("ShaderNodeAmbientOcclusion", samples=samples)
        n.inputs["Distance"].default_value = dist
        return n.outputs["AO"]

    def edge(self, radius, gain):
        bev = self.node("ShaderNodeBevel", samples=8)
        bev.inputs["Radius"].default_value = radius
        geo = self.node("ShaderNodeNewGeometry")
        d = self.vmath("DOT_PRODUCT", bev.outputs[0], geo.outputs["Normal"])
        return self.math("MULTIPLY", self.math("SUBTRACT", 1.0, d), gain, clamp=True)

    def bump(self, height, strength, distance=0.001, normal=None):
        b = self.node("ShaderNodeBump")
        b.inputs["Strength"].default_value = strength
        b.inputs["Distance"].default_value = distance
        self.inp(b.inputs["Height"], height)
        if normal is not None:
            self.inp(b.inputs["Normal"], normal)
        return b.outputs["Normal"]

    def is_pat(self, pat, k):
        return self.math("COMPARE", pat, float(k)) if False else self._cmp(pat, k)

    def _cmp(self, pat, k):
        n = self.node("ShaderNodeMath", operation="COMPARE")
        self.inp(n.inputs[0], pat)
        n.inputs[1].default_value = float(k)
        n.inputs[2].default_value = 0.4
        return n.outputs[0]

    def ellmask(self, c, r, soft=0.4, mirror=True):
        """1 inside an (x-mirrored) ellipsoid, soft edge."""
        p = self.pos()
        if mirror:
            sep = self.node("ShaderNodeSeparateXYZ")
            self.t.links.new(p, sep.inputs[0])
            cmb = self.node("ShaderNodeCombineXYZ")
            self.t.links.new(self.math("ABSOLUTE", sep.outputs[0]), cmb.inputs[0])
            self.t.links.new(sep.outputs[1], cmb.inputs[1])
            self.t.links.new(sep.outputs[2], cmb.inputs[2])
            p = cmb.outputs[0]
        q = self.vmath("DIVIDE", self.vmath("SUBTRACT", p, tuple(c)), tuple(r))
        d = self.vmath("LENGTH", q)
        return self.math("SUBTRACT", 1.0, self.smooth(1.0 - soft, 1.0, d))


def base_fabric(nb, tint, pat):
    """Fabric family: plain / camo / knit / ripstop / canvas / quilted."""
    p = nb.pos()
    var = nb.noise(p, 9.0, 5, 0.6)
    col = nb.mulc(tint, nb.gray(nb.math("ADD", 0.86, nb.math("MULTIPLY", var, 0.28))))
    # camo: four blob layers derived from the tint
    pc = nb.pos((1, 1, 0.8))
    n1 = nb.noise(pc, 9.0, 3, 0.5, 0.4)
    n2 = nb.noise(nb.pos((1, 1, 0.8), (3.1, 1.7, 0.4)), 11.0, 3, 0.5, 0.4)
    n3 = nb.noise(nb.pos((1, 1, 0.8), (7.3, 2.2, 5.1)), 14.0, 2, 0.5, 0.3)
    dark = nb.mulc(tint, (0.42, 0.42, 0.40))
    light = nb.mulc(tint, (1.30, 1.22, 1.05))
    brown = nb.mulc(tint, (0.95, 0.72, 0.55))
    camo = nb.mixc(nb.smooth(0.50, 0.53, n1), col, dark)
    camo = nb.mixc(nb.smooth(0.56, 0.59, n2), camo, light)
    camo = nb.mixc(nb.smooth(0.60, 0.63, n3), camo, brown)
    is_camo = nb.is_pat(pat, PAT_CAMO)
    col = nb.mixc(is_camo, col, camo)
    # weave / pattern heights
    rip = nb.math("MAXIMUM", nb.smooth(0.85, 0.97, nb.wave(nb.pos(), 90, "X", profile="SIN")),
                  nb.smooth(0.85, 0.97, nb.wave(nb.pos(), 90, "Z", profile="SIN")))
    knit = nb.noise(nb.pos((380, 380, 45)), 1.0, 1, 0.5)
    canvas = nb.wave(nb.pos((1, 1, 1)), 260, "DIAGONAL")
    quilt = nb.smooth(0.0, 0.18, nb.wave(nb.pos(), 16, "Z", profile="SAW"))
    fine = nb.noise(p, 420.0, 2, 0.5)
    h = nb.math("MULTIPLY", fine, 0.15)
    h = nb.math("ADD", h, nb.math("MULTIPLY", nb.is_pat(pat, PAT_RIP), nb.math("MULTIPLY", rip, 0.35)))
    h = nb.math("ADD", h, nb.math("MULTIPLY", nb.is_pat(pat, PAT_KNIT), nb.math("MULTIPLY", knit, 1.0)))
    h = nb.math("ADD", h, nb.math("MULTIPLY", nb.is_pat(pat, PAT_CANVAS), nb.math("MULTIPLY", canvas, 0.35)))
    h = nb.math("ADD", h, nb.math("MULTIPLY", nb.is_pat(pat, PAT_QUILT), nb.math("MULTIPLY", quilt, 1.2)))
    # creases: stretched low-frequency noise reads as folds
    folds = nb.noise(nb.pos((14, 14, 4.5)), 2.6, 3, 0.5, 0.6)
    fold_h = nb.smooth(0.35, 0.75, folds)
    nrm = nb.bump(fold_h, 0.35, 0.02)
    nrm = nb.bump(h, 0.55, 0.0012, nrm)
    ao = nb.ao(0.08)
    edge = nb.edge(0.006, 1.6)
    col = nb.mixc(nb.math("MULTIPLY", edge, 0.35), col, nb.mulc(col, (1.35, 1.35, 1.3)))      # edge wear
    col = nb.mixc(nb.math("MULTIPLY", nb.is_pat(pat, PAT_QUILT), nb.math("SUBTRACT", 1, quilt)), col, nb.mulc(col, (0.7, 0.7, 0.7)))
    col = nb.mulc(col, nb.gray(nb.math("ADD", 0.30, nb.math("MULTIPLY", ao, 0.70))))
    col = nb.mulc(col, nb.gray(nb.math("ADD", 0.86, nb.math("MULTIPLY", fold_h, 0.18))))
    rough = nb.math("ADD", 0.80, nb.math("MULTIPLY", var, 0.12))
    return col, rough, 0.0, nrm


def recipe_fabric(nb, P):
    return base_fabric(nb, nb.attr("tint"), nb.attr("pat", "Fac"))


def recipe_team(nb, P):
    pat = nb.attr("pat", "Fac")
    col, rough, metal, nrm = base_fabric(nb, (0.86, 0.86, 0.86), pat)
    return col, rough, metal, nrm


def recipe_leather(nb, P):
    tint = nb.attr("tint")
    pat = nb.attr("pat", "Fac")
    p = nb.pos()
    grain = nb.voronoi(p, 900.0)
    var = nb.noise(p, 12.0, 4, 0.6)
    col = nb.mulc(tint, nb.gray(nb.math("ADD", 0.80, nb.math("MULTIPLY", var, 0.4))))
    edge = nb.edge(0.008, 1.8)
    col = nb.mixc(nb.math("MULTIPLY", edge, 0.55), col, nb.mulc(tint, (1.6, 1.55, 1.45)))
    sep = nb.node("ShaderNodeSeparateXYZ")
    nb.t.links.new(p, sep.inputs[0])
    sole = nb.math("MULTIPLY", nb.is_pat(pat, PAT_SOLE), nb.math("SUBTRACT", 1.0, nb.smooth(0.030, 0.036, sep.outputs[2])))
    welt = nb.math("MULTIPLY", nb.is_pat(pat, PAT_SOLE), nb.math("SUBTRACT", 1.0, nb.smooth(0.006, 0.0, nb.math("ABSOLUTE", nb.math("SUBTRACT", sep.outputs[2], 0.040)))))
    col = nb.mixc(sole, col, (0.035, 0.032, 0.03))
    col = nb.mixc(nb.math("MULTIPLY", welt, 0.6), col, nb.mulc(tint, (0.5, 0.5, 0.5)))
    web = nb.math("MULTIPLY", nb.is_pat(pat, PAT_WEBBING), nb.smooth(0.7, 0.95, nb.wave(nb.pos(), 260, "Z")))
    tread = nb.math("MULTIPLY", sole, nb.smooth(0.4, 0.6, nb.wave(nb.pos(), 70, "Y", profile="SAW")))
    h = nb.math("ADD", nb.math("MULTIPLY", grain, 0.25), nb.math("ADD", nb.math("MULTIPLY", web, 0.3), nb.math("MULTIPLY", tread, 1.0)))
    creases = nb.noise(nb.pos((6, 6, 20)), 6.0, 3, 0.6, 0.5)
    nrm = nb.bump(nb.smooth(0.4, 0.7, creases), 0.3, 0.01)
    nrm = nb.bump(h, 0.5, 0.0012, nrm)
    ao = nb.ao(0.06)
    col = nb.mulc(col, nb.gray(nb.math("ADD", 0.30, nb.math("MULTIPLY", ao, 0.70))))
    rough = nb.mixf(sole, nb.math("ADD", 0.42, nb.math("MULTIPLY", nb.math("ADD", var, edge), 0.25)), 0.88)
    return col, rough, 0.0, nrm


def recipe_gear(nb, P):
    tint = nb.attr("tint")
    pat = nb.attr("pat", "Fac")
    p = nb.pos()
    var = nb.noise(p, 10.0, 4, 0.6)
    col = nb.mulc(tint, nb.gray(nb.math("ADD", 0.85, nb.math("MULTIPLY", var, 0.3))))
    molle = nb.smooth(0.55, 0.75, nb.wave(nb.pos(), 40, "Z", profile="SIN"))
    is_web = nb.is_pat(pat, PAT_WEBBING)
    is_hard = nb.is_pat(pat, PAT_HARD)
    is_rip = nb.is_pat(pat, PAT_RIP)
    col = nb.mixc(nb.math("MULTIPLY", is_web, nb.math("MULTIPLY", molle, 0.35)), col, nb.mulc(col, (0.6, 0.6, 0.6)))
    rip = nb.math("MAXIMUM", nb.smooth(0.85, 0.97, nb.wave(nb.pos(), 110, "X")), nb.smooth(0.85, 0.97, nb.wave(nb.pos(), 110, "Z")))
    weave = nb.noise(p, 500.0, 1, 0.5)
    speck = nb.noise(p, 180.0, 2, 0.5)
    h = nb.math("MULTIPLY", weave, nb.math("SUBTRACT", 1.0, is_hard))
    h = nb.math("ADD", h, nb.math("MULTIPLY", is_web, nb.math("MULTIPLY", molle, 1.2)))
    h = nb.math("ADD", h, nb.math("MULTIPLY", is_rip, nb.math("MULTIPLY", rip, 0.5)))
    h = nb.math("ADD", h, nb.math("MULTIPLY", is_hard, nb.math("MULTIPLY", speck, 0.2)))
    edge = nb.edge(0.006, 1.8)
    col = nb.mixc(nb.math("MULTIPLY", edge, 0.45), col, nb.mulc(tint, (1.5, 1.45, 1.4)))
    folds = nb.noise(nb.pos((10, 10, 6)), 3.0, 3, 0.5, 0.5)
    nrm = nb.bump(nb.math("MULTIPLY", nb.smooth(0.4, 0.7, folds), nb.math("SUBTRACT", 1.0, is_hard)), 0.3, 0.012)
    nrm = nb.bump(h, 0.5, 0.0012, nrm)
    ao = nb.ao(0.07)
    col = nb.mulc(col, nb.gray(nb.math("ADD", 0.28, nb.math("MULTIPLY", ao, 0.72))))
    rough = nb.mixf(is_hard, nb.math("ADD", 0.72, nb.math("MULTIPLY", var, 0.15)), nb.math("ADD", 0.5, nb.math("MULTIPLY", edge, 0.2)))
    return col, rough, 0.0, nrm


def recipe_hair(nb, P):
    tint = nb.attr("tint")
    strands = nb.noise(nb.pos((140, 140, 22)), 1.0, 6, 0.65)
    clumps = nb.noise(nb.pos((18, 18, 6)), 1.0, 3, 0.5)
    col = nb.mulc(tint, nb.gray(nb.math("ADD", 0.55, nb.math("MULTIPLY", nb.math("ADD", strands, nb.math("MULTIPLY", clumps, 0.5)), 0.7))))
    ao = nb.ao(0.05)
    col = nb.mulc(col, nb.gray(nb.math("ADD", 0.25, nb.math("MULTIPLY", ao, 0.75))))
    nrm = nb.bump(nb.math("ADD", strands, nb.math("MULTIPLY", clumps, 0.6)), 0.8, 0.002)
    rough = nb.math("ADD", 0.38, nb.math("MULTIPLY", strands, 0.2))
    return col, rough, 0.0, nrm


def recipe_skin(nb, P):
    tone = P["skin"]
    p = nb.pos()
    blot = nb.noise(p, 25.0, 4, 0.6)
    col = nb.mixc(nb.math("MULTIPLY", blot, 0.35), tone, nb.mulc(tone, (1.08, 0.86, 0.80)))
    red = nb.math("ADD", nb.ellmask((0.045, -0.070, 1.655), (0.028, 0.03, 0.022), 0.8),
                  nb.math("ADD", nb.ellmask((0.0, -0.105, 1.660), (0.014, 0.02, 0.02), 0.8, False),
                          nb.ellmask((0.078, 0.008, 1.688), (0.016, 0.03, 0.035), 0.6)))
    col = nb.mixc(nb.math("MULTIPLY", red, 0.22), col, nb.mulc(tone, (1.12, 0.78, 0.74)))
    lips = nb.math("ADD", nb.ellmask((0, -0.090, 1.626), (0.023, 0.016, 0.0085), 0.35, False),
                   nb.ellmask((0, -0.087, 1.612), (0.021, 0.016, 0.0095), 0.35, False))
    lipcol = nb.mulc(tone, (0.92, 0.60, 0.60) if P["skin"][0] > 0.2 else (0.75, 0.52, 0.55))
    col = nb.mixc(nb.math("MINIMUM", lips, 1.0), col, lipcol)
    seam = nb.ellmask((0, -0.094, 1.6195), (0.019, 0.02, 0.0016), 0.6, False)
    col = nb.mixc(nb.math("MULTIPLY", seam, 0.7), col, nb.mulc(lipcol, (0.45, 0.4, 0.4)))
    hairc = P["hair"]
    br = nb.ellmask((0.034, -0.083, 1.712), (0.022, 0.03, 0.0055 if P["female"] else 0.0068), 0.5)
    br = nb.math("MULTIPLY", br, nb.smooth(0.3, 0.7, nb.noise(nb.pos((300, 300, 60)), 1.0, 2)))
    col = nb.mixc(nb.math("MULTIPLY", br, P.get("brows", 0.8)), col, hairc)
    # eyes: sclera, iris, pupil, lash line
    eye = nb.ellmask((EYE_X, EYE_Y, EYE_Z), (EYE_R + 0.0004,) * 3, 0.04)
    irisd = nb.vmath("LENGTH", nb.vmath("SUBTRACT", _abs_x(nb), (EYE_X, EYE_Y - EYE_R, EYE_Z)))
    iris = nb.math("SUBTRACT", 1.0, nb.smooth(0.0058, 0.0066, irisd))
    pupil = nb.math("SUBTRACT", 1.0, nb.smooth(0.0022, 0.0028, irisd))
    eyecol = nb.mixc(iris, (0.58, 0.54, 0.50), P["eye"])
    eyecol = nb.mixc(pupil, eyecol, (0.01, 0.01, 0.01))
    col = nb.mixc(eye, col, eyecol)
    lash = nb.ellmask((EYE_X, EYE_Y - 0.004, EYE_Z + 0.0035), (0.0150, 0.010, 0.0042), 0.35)
    lash = nb.math("MULTIPLY", lash, nb.math("SUBTRACT", 1.0, eye))
    col = nb.mixc(nb.math("MULTIPLY", lash, 0.75), col, nb.mulc(hairc, (0.4, 0.4, 0.4)))
    if P.get("freckles"):
        fr = nb.math("SUBTRACT", 1.0, nb.smooth(0.05, 0.16, nb.voronoi(p, 260.0)))
        fr = nb.math("MULTIPLY", fr, nb.ellmask((0.03, -0.085, 1.665), (0.05, 0.04, 0.03), 0.6))
        col = nb.mixc(nb.math("MULTIPLY", fr, 0.45), col, nb.mulc(tone, (0.75, 0.55, 0.42)))
    if P.get("beard"):
        stub = nb.ellmask((0.0, -0.05, 1.60), (0.07, 0.07, 0.05), 0.5)
        col = nb.mixc(nb.math("MULTIPLY", stub, 0.35), col, nb.mulc(hairc, (0.8, 0.8, 0.8)))
    ao = nb.ao(0.035)
    col = nb.mulc(col, nb.gray(nb.math("ADD", 0.35, nb.math("MULTIPLY", ao, 0.65))))
    pores = nb.noise(p, 900.0, 2, 0.5)
    nrm = nb.bump(nb.math("ADD", pores, nb.math("MULTIPLY", blot, 0.3)), 0.12, 0.0006)
    rough = nb.mixf(eye, nb.mixf(nb.math("MINIMUM", lips, 1.0), nb.math("ADD", 0.48, nb.math("MULTIPLY", blot, 0.12)), 0.35), 0.08)
    return col, rough, 0.0, nrm


def _abs_x(nb):
    p = nb.pos()
    sep = nb.node("ShaderNodeSeparateXYZ")
    nb.t.links.new(p, sep.inputs[0])
    cmb = nb.node("ShaderNodeCombineXYZ")
    nb.t.links.new(nb.math("ABSOLUTE", sep.outputs[0]), cmb.inputs[0])
    nb.t.links.new(sep.outputs[1], cmb.inputs[1])
    nb.t.links.new(sep.outputs[2], cmb.inputs[2])
    return cmb.outputs[0]


RECIPES = {"Skin": recipe_skin, "Hair": recipe_hair, "Fabric": recipe_fabric, "Leather": recipe_leather,
           "Gear": recipe_gear, "Team": recipe_team}
TEX_SIZE = {"Skin": 1024, "Hair": 512, "Fabric": 1024, "Leather": 512, "Gear": 1024, "Team": 512}


def unwrap_object(o):
    select_only([o], o)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(55), island_margin=0.006, area_weight=0.0, scale_to_bounds=False)
    bpy.ops.uv.select_all(action="SELECT")
    bpy.ops.uv.pack_islands(rotate=True, margin=0.006)
    bpy.ops.object.mode_set(mode="OBJECT")


def setup_cycles(samples):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = samples
    sc.cycles.use_denoising = False


def bake_textures(cid, P, mesh, sizes=None):
    sizes = dict(TEX_SIZE, **(sizes or {}))
    setup_cycles(24)
    sc = bpy.context.scene
    sc.render.bake.margin = 8
    sc.render.bake.margin_type = "EXTEND"
    os.makedirs(TEX_TMP, exist_ok=True)
    setups = []
    for mat in mesh.data.materials:
        tree = mat.node_tree
        tree.nodes.clear()
        nb = NB(tree)
        c, r, m, n = RECIPES[mat.name](nb, P)
        out = nb.node("ShaderNodeOutputMaterial")
        emit = nb.node("ShaderNodeEmission")
        bsdf = nb.node("ShaderNodeBsdfPrincipled")
        nb.inp(bsdf.inputs["Normal"], n)
        orm = nb.node("ShaderNodeCombineColor")
        nb.inp(orm.inputs[0], 1.0)
        nb.inp(orm.inputs[1], r)
        nb.inp(orm.inputs[2], m)
        size = sizes[mat.name]
        small = max(256, size // 2)
        ims = {}
        nres = size if mat.name in ("Fabric", "Gear") else small
        for key, res, data in (("base", size, False), ("orm", small // 2 if mat.name != "Skin" else small, True), ("normal", nres, True)):
            im = bpy.data.images.new("%s_%s_%s" % (cid, mat.name, key), res, res, alpha=False)
            im.colorspace_settings.name = "Non-Color" if data else "sRGB"
            node = nb.node("ShaderNodeTexImage", image=im)
            ims[key] = (im, node)
        setups.append((mat, nb, c, orm.outputs[0], emit, bsdf, out, ims))
    select_only([mesh], mesh)
    for key, kind in (("base", "EMIT"), ("orm", "EMIT"), ("normal", "NORMAL")):
        for mat, nb, c, ormo, emit, bsdf, out, ims in setups:
            tree = mat.node_tree
            for l in list(out.inputs["Surface"].links):
                tree.links.remove(l)
            for l in list(emit.inputs["Color"].links):
                tree.links.remove(l)
            if kind == "EMIT":
                nb.inp(emit.inputs["Color"], c if key == "base" else ormo)
                tree.links.new(emit.outputs[0], out.inputs["Surface"])
            else:
                tree.links.new(bsdf.outputs[0], out.inputs["Surface"])
            tree.nodes.active = ims[key][1]
        t = time.time()
        bpy.ops.object.bake(type=kind, normal_space="TANGENT", margin=8, margin_type="EXTEND", use_clear=True)
        log("baked %s %s in %.0fs" % (cid, key, time.time() - t))
    # final glTF-friendly materials
    for mat, nb, c, ormo, emit, bsdf, out, ims in setups:
        for key in ims:
            im = ims[key][0]
            im.filepath_raw = os.path.join(TEX_TMP, im.name + ".png")
            im.file_format = "PNG"
            im.save()
        tree = mat.node_tree
        tree.nodes.clear()
        nb = NB(tree)
        out = nb.node("ShaderNodeOutputMaterial")
        bsdf = nb.node("ShaderNodeBsdfPrincipled")
        tree.links.new(bsdf.outputs[0], out.inputs["Surface"])
        tb = nb.node("ShaderNodeTexImage", image=ims["base"][0])
        tree.links.new(tb.outputs["Color"], bsdf.inputs["Base Color"])
        to = nb.node("ShaderNodeTexImage", image=ims["orm"][0])
        sep = nb.node("ShaderNodeSeparateColor")
        tree.links.new(to.outputs["Color"], sep.inputs[0])
        tree.links.new(sep.outputs[1], bsdf.inputs["Roughness"])
        tree.links.new(sep.outputs[2], bsdf.inputs["Metallic"])
        tn = nb.node("ShaderNodeTexImage", image=ims["normal"][0])
        nm = nb.node("ShaderNodeNormalMap")
        tree.links.new(tn.outputs["Color"], nm.inputs["Color"])
        tree.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    # the bake-only attributes are not needed in the game
    for name in ("tint", "pat"):
        a = mesh.data.attributes.get(name)
        if a:
            mesh.data.attributes.remove(a)


# ==========================================================================
# Export + verification
# ==========================================================================

MAX_BYTES = int(2.45 * 1024 * 1024)


def export_glb(path, quality):
    bpy.ops.export_scene.gltf(
        filepath=path, export_format="GLB", export_yup=True, use_selection=False,
        export_texcoords=True, export_normals=True, export_tangents=False,
        export_materials="EXPORT", export_image_format="JPEG", export_jpeg_quality=quality,
        export_cameras=False, export_lights=False, export_extras=False, export_apply=False,
        export_skins=True, export_influence_nb=4, export_def_bones=False, export_morph=False,
        export_animations=True, export_animation_mode="ACTIONS", export_force_sampling=True,
        export_frame_step=1, export_optimize_animation_size=True, export_anim_single_armature=True,
        export_reset_pose_bones=True, export_draco_mesh_compression_enable=False,
    )


def read_glb(path):
    with open(path, "rb") as fh:
        data = fh.read()
    import struct
    magic, ver, length = struct.unpack_from("<III", data, 0)
    assert magic == 0x46546C67, "not a GLB"
    clen, ctype = struct.unpack_from("<II", data, 12)
    return json.loads(data[20:20 + clen].decode("utf-8")), len(data)


def verify_glb(path):
    js, nbytes = read_glb(path)
    nodes = js["nodes"]
    names = [n.get("name") for n in nodes]
    skin = js["skins"][0]
    joints = sorted(nodes[j]["name"] for j in skin["joints"])
    assert joints == sorted(BONE_ORDER), "bone names differ: %s" % joints
    anims = {a["name"]: a for a in js.get("animations", [])}
    missing = [a for a in ACTION_NAMES if a not in anims]
    assert not missing, "missing animations %s" % missing
    durations = {}
    for name, a in anims.items():
        t = 0.0
        for s in a["samplers"]:
            t = max(t, js["accessors"][s["input"]]["max"][0])
        durations[name] = round(t, 3)
    assert "GunMount" in names, "GunMount missing"
    gi = names.index("GunMount")
    par = [i for i, n in enumerate(nodes) if gi in n.get("children", [])]
    assert par and nodes[par[0]]["name"] == "Hand.R", "GunMount must be a child of Hand.R"
    mats = [m["name"] for m in js["materials"]]
    assert "Team" in mats, "Team material missing"
    assert "extensionsUsed" not in js or "KHR_draco_mesh_compression" not in js["extensionsUsed"]
    tris = 0
    lo = [1e9] * 3
    hi = [-1e9] * 3
    for m in js["meshes"]:
        for pr in m["primitives"]:
            tris += js["accessors"][pr["indices"]]["count"] // 3
            acc = js["accessors"][pr["attributes"]["POSITION"]]
            lo = [min(a, b) for a, b in zip(lo, acc["min"])]
            hi = [max(a, b) for a, b in zip(hi, acc["max"])]
    texsz = {}
    for im in js.get("images", []):
        texsz[im.get("name", "?")] = js["bufferViews"][im["bufferView"]]["byteLength"]
    info = {
        "file": os.path.relpath(path, ROOT).replace(os.sep, "/"),
        "bytes": nbytes,
        "triangles": tris,
        "height_m": round(hi[1] - lo[1], 3),
        "feet_y": round(lo[1], 3),
        "bbox_min": [round(x, 3) for x in lo],
        "bbox_max": [round(x, 3) for x in hi],
        "materials": mats,
        "bones": [nodes[j]["name"] for j in skin["joints"]],
        "animations": {k: durations[k] for k in ACTION_NAMES},
        "texture_bytes": sum(texsz.values()),
        "images": len(js.get("images", [])),
    }
    assert 1.75 <= info["height_m"] <= 1.95, "height %.3f" % info["height_m"]
    assert abs(info["feet_y"]) < 0.02, "feet not on y=0"
    assert tris <= 12500, "too many triangles %d" % tris
    # Blender round trip: check GunMount orientation in the aim pose
    reset_scene()
    bpy.ops.import_scene.gltf(filepath=path)
    rig = next(o for o in bpy.context.scene.objects if o.type == "ARMATURE")
    gm = bpy.data.objects["GunMount"]
    act = next(a for a in bpy.data.actions if a.name.startswith("AimIdle"))
    rig.animation_data.action = act
    bpy.context.scene.frame_set(0)
    bpy.context.view_layer.update()
    mw = gm.matrix_world
    barrel = (mw.to_3x3() @ Vector((0, -1, 0))).normalized()
    up = (mw.to_3x3() @ Vector((0, 0, 1))).normalized()
    info["aim_barrel_dir_gltf"] = [round(barrel.x, 3), round(barrel.z, 3), round(-barrel.y, 3)]
    info["aim_grip_pos_gltf"] = [round(mw.translation.x, 3), round(mw.translation.z, 3), round(-mw.translation.y, 3)]
    assert barrel.y < -0.97 and up.z > 0.9, "GunMount not aligned in AimIdle (%s, %s)" % (barrel, up)
    return info


def make_character(cid):
    reset_scene()
    P, rig, mesh, pieces = build_character(cid)
    bake_textures(cid, P, mesh)
    attach_mesh(rig, mesh)
    bake_all_actions(rig)
    add_gunmount(rig)
    rig.animation_data.action = None
    for o in bpy.context.scene.objects:
        o.select_set(False)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "char_%s.glb" % cid)
    q = 88
    while True:
        export_glb(path, q)
        size = os.path.getsize(path)
        log("exported %s q=%d: %.2f MB" % (path, q, size / 1048576))
        if size <= MAX_BYTES or q <= 50:
            break
        q -= 8
    info = verify_glb(path)
    info.update(name=P["name"], jpeg_quality=q, pieces=pieces)
    log("verified", cid, {k: info[k] for k in ("triangles", "bytes", "height_m")})
    return info


# ==========================================================================
# Renders
# ==========================================================================

def import_char(cid, offset=(0, 0, 0), rot_z=0.0, action="Idle", frame=0, gun="ar"):
    before = set(bpy.data.objects)
    acts_before = set(bpy.data.actions)
    bpy.ops.import_scene.gltf(filepath=os.path.join(OUT_DIR, "char_%s.glb" % cid))
    new = [o for o in bpy.data.objects if o not in before]
    acts = [a for a in bpy.data.actions if a not in acts_before]
    rig = next(o for o in new if o.type == "ARMATURE")
    root = rig
    while root.parent is not None:
        root = root.parent
    root.location = Vector(offset)
    root.rotation_mode = "XYZ"
    root.rotation_euler = (0, 0, math.radians(rot_z))
    gm = next(o for o in new if o.name.startswith("GunMount"))
    act = next(a for a in acts if a.name.split("_")[0] == action)
    rig.animation_data_create()
    for tr in rig.animation_data.nla_tracks:
        tr.mute = True
    rig.animation_data.action = act
    rig["frame"] = frame
    if gun:
        b2 = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=os.path.join(GUN_DIR, gun + ".glb"))
        gnew = [o for o in bpy.data.objects if o not in b2]
        for o in gnew:
            if o.parent is None:
                o.parent = gm
                o.matrix_parent_inverse.identity()
                o.location = (0, 0, 0)
                o.rotation_mode = "QUATERNION"
                o.rotation_quaternion = (1, 0, 0, 0)
                o.scale = (1, 1, 1)
    return rig


def studio(sc, ctr):
    world = bpy.data.worlds.new("Studio")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.22, 0.23, 0.25, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.25
    sc.world = world
    bm = bmesh.new()
    W = 14.0
    prof = [(-6.0, 0.0), (3.0, 0.0)]
    for i in range(1, 10):
        a = math.radians(90 * i / 9)
        prof.append((3.0 + math.sin(a) * 2.5, (1 - math.cos(a)) * 2.5))
    prof.append((5.5, 8.0))
    rows = [[bm.verts.new((-W / 2 + W * k, y, z)) for k in (0, 1)] for y, z in prof]
    for a, b in zip(rows[:-1], rows[1:]):
        bm.faces.new((a[0], a[1], b[1], b[0]))
    me = bpy.data.meshes.new("Backdrop")
    bm.to_mesh(me)
    bm.free()
    me.shade_smooth()
    bd = link(bpy.data.objects.new("Backdrop", me))
    mat = bpy.data.materials.new("Backdrop")
    mat.use_nodes = True
    bp = mat.node_tree.nodes["Principled BSDF"]
    bp.inputs["Base Color"].default_value = (0.36, 0.37, 0.38, 1)
    bp.inputs["Roughness"].default_value = 0.85
    me.materials.append(mat)

    def area(name, loc, size, energy, color=(1, 1, 1)):
        ld = bpy.data.lights.new(name, "AREA")
        ld.shape = "RECTANGLE"
        ld.size, ld.size_y = size
        ld.energy = energy
        ld.color = color
        o = link(bpy.data.objects.new(name, ld))
        o.location = Vector(loc)
        o.rotation_euler = (ctr - o.location).to_track_quat("-Z", "Y").to_euler()
    area("Key", (-2.4, -3.2, 3.4), (2.2, 1.6), 650, (1.0, 0.95, 0.88))
    area("Fill", (3.0, -2.6, 1.6), (2.0, 2.0), 260, (0.85, 0.92, 1.0))
    area("Rim", (1.6, 2.6, 3.0), (2.4, 0.6), 700, (0.95, 0.97, 1.0))
    area("RimL", (-2.0, 2.4, 2.6), (2.0, 0.5), 450)


def sky(sc):
    world = bpy.data.worlds.new("Sky")
    world.use_nodes = True
    nt = world.node_tree
    skyn = nt.nodes.new("ShaderNodeTexSky")
    skyn.sky_type = "NISHITA"
    skyn.sun_elevation = math.radians(24)
    skyn.sun_rotation = math.radians(215)
    skyn.altitude = 400
    skyn.air_density = 1.2
    skyn.dust_density = 2.0
    nt.links.new(skyn.outputs[0], nt.nodes["Background"].inputs[0])
    nt.nodes["Background"].inputs["Strength"].default_value = 0.30
    sc.world = world
    me = bpy.data.meshes.new("Ground")
    me.from_pydata([(-40, -40, 0), (40, -40, 0), (40, 40, 0), (-40, 40, 0)], [], [(0, 1, 2, 3)])
    g = link(bpy.data.objects.new("Ground", me))
    mat = bpy.data.materials.new("Ground")
    mat.use_nodes = True
    nt = mat.node_tree
    bp = nt.nodes["Principled BSDF"]
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = 3.0
    nz.inputs["Detail"].default_value = 8.0
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.16, 0.14, 0.09, 1)
    ramp.color_ramp.elements[1].color = (0.30, 0.27, 0.18, 1)
    nt.links.new(nz.outputs["Fac"], ramp.inputs[0])
    nt.links.new(ramp.outputs[0], bp.inputs["Base Color"])
    bp.inputs["Roughness"].default_value = 0.95
    me.materials.append(mat)


def render_setup(samples=96):
    sc = bpy.context.scene
    setup_cycles(samples)
    sc.cycles.use_denoising = True
    sc.render.resolution_x = 1600
    sc.render.resolution_y = 900
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "AgX - Punchy"
    sc.view_settings.exposure = -0.35
    return sc


def camera(sc, loc, tgt, lens):
    cd = bpy.data.cameras.new("Cam")
    cd.lens = lens
    cam = link(bpy.data.objects.new("Cam", cd))
    cam.location = Vector(loc)
    cam.rotation_euler = (Vector(tgt) - cam.location).to_track_quat("-Z", "Y").to_euler()
    sc.camera = cam
    return cam


def set_frames(rigs):
    sc = bpy.context.scene
    # every rig holds its own pose via NLA-free single action; evaluate at its frame
    for r in rigs:
        act = r.animation_data.action
        f = r.get("frame", 0)
        r.animation_data.action_extrapolation = "HOLD"
        r.animation_data.action_blend_type = "REPLACE"
        r.animation_data.action_influence = 1.0
        # offset the action so that frame 1 of the scene shows frame `f` of the clip
        for fc in act.fcurves:
            for kp in fc.keyframe_points:
                kp.co.x -= f
                kp.handle_left.x -= f
                kp.handle_right.x -= f
    sc.frame_set(0)


def render_character(cid):
    os.makedirs(RENDER_DIR, exist_ok=True)
    for view, rz in (("front", 0.0), ("three_quarter", 38.0), ("side", 90.0)):
        reset_scene()
        sc = render_setup()
        rig = import_char(cid, rot_z=rz, action="Idle", frame=24)
        set_frames([rig])
        ctr = Vector((0, 0, 0.93))
        studio(sc, ctr)
        camera(sc, ctr + Vector((0.0, -1.0, 0.06)).normalized() * 5.1, ctr + Vector((0, 0, -0.01)), 50)
        sc.render.filepath = os.path.join(RENDER_DIR, "char_%s_%s.png" % (cid, view))
        t = time.time()
        bpy.ops.render.render(write_still=True)
        log("render", sc.render.filepath, "%.0fs" % (time.time() - t))


def render_group(ids):
    reset_scene()
    sc = render_setup(128)
    sky(sc)
    setups = [("a", (-1.35, 0.15, 0), 18, "AimIdle", 0, "ar"), ("b", (-0.45, -0.25, 0), -8, "Idle", 40, "smg"),
              ("c", (0.50, 0.05, 0), 10, "Idle", 100, "shotgun"), ("d", (1.40, 0.25, 0), -25, "AimIdle", 30, "sniper")]
    rigs = []
    for cid, off, rz, act, fr, gun in setups:
        if cid in ids:
            rigs.append(import_char(cid, off, rz, act, fr, gun))
    set_frames(rigs)
    camera(sc, (0.9, -6.4, 1.35), (0.0, 0.0, 0.95), 50)
    sc.render.filepath = os.path.join(RENDER_DIR, "group.png")
    bpy.ops.render.render(write_still=True)
    log("render group")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    flags = [a for a in sys.argv[1:] if a.startswith("-")]
    ids = [a for a in args if a in CHAR_IDS] or CHAR_IDS
    stats_path = os.path.join(OUT_DIR, "stats.json")
    stats = {}
    if os.path.exists(stats_path):
        with open(stats_path) as fh:
            stats = json.load(fh)
    if "--render-only" not in flags:
        for cid in ids:
            stats["char_" + cid] = make_character(cid)
            with open(stats_path, "w") as fh:
                json.dump(stats, fh, indent=2)
    if "--no-render" not in flags:
        for cid in ids:
            render_character(cid)
        if "--no-group" not in flags:
            render_group(CHAR_IDS)
    log("done")


if __name__ == "__main__":
    main()
