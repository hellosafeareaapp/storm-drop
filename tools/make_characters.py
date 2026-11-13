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
    f = V(0, 0, 1) if name.startswith("Foot") else FRONT
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
    mouth = rbox((0, -0.098, 1.6195), (0.018, 0.010, 0.0010), 0.0008)
    head = head.ssub(mouth, 0.002)
    eyes = U([sphere((sx * EYE_X, EYE_Y, EYE_Z), EYE_R) for sx in (1, -1)])
    lids = U([ellipsoid((sx * EYE_X, EYE_Y - 0.0015, EYE_Z + 0.0085), (0.0155, 0.0125, 0.0062)) for sx in (1, -1)] +
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
        straps.append(chain([V(0.085 * sx, yf - 0.005, 1.400), V(0.110 * sx, -0.010, 1.500), V(0.098 * sx, yb + 0.010, 1.420)],
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
        beard = beard - ellipsoid((0, -0.096, 1.618), (0.021, 0.030, 0.012))
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
