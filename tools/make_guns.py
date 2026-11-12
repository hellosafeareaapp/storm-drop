"""
STORM DROP - procedural gun generator for Blender (4.2 LTS).

Builds five original, generic firearm models, bakes procedural PBR detail
(edge wear, scratches, wood grain, roughness variation, bevelled-edge normals)
into small textures, exports glTF binaries, re-imports each one to verify it,
and renders a studio preview.

Run with a Blender binary:
    blender -b -P tools/make_guns.py -- [ar shotgun smg sniper rocket]
or with the `bpy` module from PyPI (Python 3.11):
    pip install bpy==4.2.0
    python tools/make_guns.py [ar shotgun smg sniper rocket]

Outputs:
    models/guns/<name>.glb
    models/guns/previews/<name>.png

Export contract (used by index.html):
    * metres, real scale; glTF +Y up
    * barrel points along +Z (glTF / Three.js space)
    * origin at the pistol grip (where the right hand holds it)
    * an empty named "Muzzle" at the barrel tip
    * a small material named "Accent" the game can recolour for rarity

Modelling frame used in this file: (s, f, u) = (side, forward, up) in metres,
which is Blender (X, Y, Z). At the end each gun is moved so the grip is at the
origin and turned 180 degrees about Z, so the barrel ends up along Blender -Y,
which the glTF exporter writes as +Z.
"""

import json
import math
import os
import sys
import tempfile

import bpy  # must come first when running with the PyPI `bpy` module
import bmesh
from mathutils import Matrix, Vector

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(ROOT, "models", "guns")
PREVIEW_DIR = os.path.join(OUT_DIR, "previews")
TEX_TMP = os.path.join(tempfile.gettempdir(), "storm_drop_gun_tex")

GUN_NAMES = ["ar", "shotgun", "smg", "sniper", "rocket"]


# --------------------------------------------------------------------------
# Scene helpers
# --------------------------------------------------------------------------

def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    return scene


def link(obj):
    bpy.context.scene.collection.objects.link(obj)
    return obj


def apply_modifiers(obj):
    """Bake the modifier stack into the mesh without using operators."""
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    new_me = bpy.data.meshes.new_from_object(obj.evaluated_get(dg), preserve_all_data_layers=True, depsgraph=dg)
    old = obj.data
    obj.modifiers.clear()
    obj.data = new_me
    new_me.name = old.name
    bpy.data.meshes.remove(old)


# --------------------------------------------------------------------------
# Geometry: a Part is one object built from several primitives in one bmesh
# --------------------------------------------------------------------------

def _basis(d):
    d = Vector(d).normalized()
    up = Vector((0, 0, 1)) if abs(d.z) < 0.95 else Vector((1, 0, 0))
    e1 = (up - d * up.dot(d)).normalized()
    e2 = d.cross(e1)
    return d, e1, e2


class Part:
    def __init__(self, gun, name, mat, bevel=0.0015, segments=1, angle=30, smooth=50):
        self.gun, self.name, self.mat = gun, name, mat
        self.bevel, self.segments, self.angle, self.smooth = bevel, segments, angle, smooth
        self.bm = bmesh.new()
        self.cutters = []
        gun.parts.append(self)

    # ---- primitives -----------------------------------------------------
    def box(self, c, size, rot=None):
        """Axis-aligned box: centre c=(s,f,u), size=(width, length, height); rot=(axis, degrees)."""
        m = Matrix.Translation(c)
        if rot:
            m = m @ Matrix.Rotation(math.radians(rot[1]), 4, rot[0])
        m = m @ Matrix.Diagonal((size[0], size[1], size[2], 1))
        bmesh.ops.create_cube(self.bm, size=1.0, matrix=m)
        return self

    def prof(self, pts, w, s=0.0):
        """Extrude a 2D outline given in (f, u) by width w along the side axis, centred at s."""
        bm = self.bm
        a = [bm.verts.new((s - w / 2, f, u)) for f, u in pts]
        b = [bm.verts.new((s + w / 2, f, u)) for f, u in pts]
        n = len(pts)
        bm.faces.new(list(reversed(a)))
        bm.faces.new(b)
        for i in range(n):
            j = (i + 1) % n
            bm.faces.new((a[i], a[j], b[j], b[i]))
        return self

    def strip(self, path, t, w, s=0.0):
        """A bent bar of thickness t following a polyline path (f, u) - e.g. a trigger guard."""
        outer, inner = [], []
        n = len(path)
        for i in range(n):
            p = Vector(path[i])
            p0 = Vector(path[max(i - 1, 0)])
            p1 = Vector(path[min(i + 1, n - 1)])
            tan = (p1 - p0).normalized()
            nor = Vector((-tan.y, tan.x))
            outer.append(tuple(p + nor * t / 2))
            inner.append(tuple(p - nor * t / 2))
        return self.prof(outer + list(reversed(inner)), w, s)

    def lathe(self, p0, d, profile, n=16, closed=False, phase=0.0):
        """Revolve a profile [(along, radius), ...] around axis d starting at p0."""
        bm = self.bm
        d, e1, e2 = _basis(d)
        p0 = Vector(p0)
        rings = []
        for a, r in profile:
            if r <= 1e-6:
                rings.append([bm.verts.new(p0 + d * a)])
                continue
            ring = []
            for i in range(n):
                th = phase + 2 * math.pi * i / n
                ring.append(bm.verts.new(p0 + d * a + (e1 * math.cos(th) + e2 * math.sin(th)) * r))
            rings.append(ring)
        pairs = list(zip(rings[:-1], rings[1:]))
        if closed:
            pairs.append((rings[-1], rings[0]))
        for ra, rb in pairs:
            if len(ra) == 1 and len(rb) == 1:
                continue
            for i in range(n):
                j = (i + 1) % n
                if len(ra) == 1:
                    bm.faces.new((ra[0], rb[i], rb[j]))
                elif len(rb) == 1:
                    bm.faces.new((ra[i], ra[j], rb[0]))
                else:
                    bm.faces.new((ra[i], ra[j], rb[j], rb[i]))
        if not closed:
            if len(rings[0]) > 1:
                bm.faces.new(list(reversed(rings[0])))
            if len(rings[-1]) > 1:
                bm.faces.new(rings[-1])
        return self

    def cyl(self, p0, p1, r, n=16, r2=None, phase=0.0):
        p0, p1 = Vector(p0), Vector(p1)
        return self.lathe(p0, p1 - p0, [(0, r), ((p1 - p0).length, r if r2 is None else r2)], n=n, phase=phase)

    def tube(self, p0, p1, ro, ri, n=16):
        p0, p1 = Vector(p0), Vector(p1)
        L = (p1 - p0).length
        return self.lathe(p0, p1 - p0, [(0, ri), (0, ro), (L, ro), (L, ri)], n=n, closed=True)

    def cut_box(self, c, size, rot=None):
        self.cutters.append(("box", c, size, rot))
        return self

    # ---- build ------------------------------------------------------------
    def build(self):
        bm = self.bm
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        me = bpy.data.meshes.new(self.name)
        bm.to_mesh(me)
        bm.free()
        obj = link(bpy.data.objects.new(self.name, me))
        cut_objs = []
        for kind, c, size, rot in self.cutters:
            cb = bmesh.new()
            m = Matrix.Translation(c)
            if rot:
                m = m @ Matrix.Rotation(math.radians(rot[1]), 4, rot[0])
            m = m @ Matrix.Diagonal((size[0], size[1], size[2], 1))
            bmesh.ops.create_cube(cb, size=1.0, matrix=m)
            cme = bpy.data.meshes.new(self.name + "_cut")
            cb.to_mesh(cme)
            cb.free()
            co = link(bpy.data.objects.new(self.name + "_cut", cme))
            cut_objs.append(co)
            mod = obj.modifiers.new("cut", "BOOLEAN")
            mod.operation = "DIFFERENCE"
            mod.solver = "EXACT"
            mod.object = co
        if self.bevel > 0:
            mod = obj.modifiers.new("bevel", "BEVEL")
            mod.width = self.bevel
            mod.segments = self.segments
            mod.limit_method = "ANGLE"
            mod.angle_limit = math.radians(self.angle)
            mod.use_clamp_overlap = True
        tri = obj.modifiers.new("tri", "TRIANGULATE")
        tri.min_vertices = 5  # ngons only; quads stay quads
        apply_modifiers(obj)
        for co in cut_objs:
            me_c = co.data
            bpy.data.objects.remove(co)
            bpy.data.meshes.remove(me_c)
        me = obj.data
        me.shade_smooth()
        me.set_sharp_from_angle(angle=math.radians(self.smooth))
        me.materials.clear()
        me.materials.append(self.gun.mat(self.mat))
        self.obj = obj
        return obj


def rail(part, f0, f1, u, w=0.021, pitch=0.01):
    part.bevel = 0.0  # rails are many small boxes: keep them cheap
    """Picatinny-style rail: a base strip plus cross teeth, top at u + 0.0095."""
    part.box((0, (f0 + f1) / 2, u + 0.0025), (w, f1 - f0, 0.005))
    f = f0 + pitch * 0.5
    while f + pitch * 0.25 < f1:
        part.box((0, f, u + 0.0065), (w, pitch * 0.52, 0.006))
        f += pitch
    return part


# --------------------------------------------------------------------------
# Materials: procedural "source" shaders baked to textures in Cycles
# --------------------------------------------------------------------------

class NB:
    """Tiny node-building helper."""

    def __init__(self, tree):
        self.t = tree
        self.x = 0

    def node(self, kind, **props):
        n = self.t.nodes.new(kind)
        n.location = (self.x, 0)
        self.x += 40
        for k, v in props.items():
            setattr(n, k, v)
        return n

    def inp(self, sock, val):
        if hasattr(val, "is_output") or isinstance(val, bpy.types.NodeSocket):
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

    def mixf(self, fac, a, b):
        n = self.node("ShaderNodeMix", data_type="FLOAT", clamp_factor=True)
        self.inp(n.inputs[0], fac)
        self.inp(n.inputs[2], a)
        self.inp(n.inputs[3], b)
        return n.outputs[0]

    def coords(self, scale=(1, 1, 1), rot=(0, 0, 0)):
        tc = self.node("ShaderNodeTexCoord")
        mp = self.node("ShaderNodeMapping")
        self.t.links.new(tc.outputs["Object"], mp.inputs["Vector"])
        mp.inputs["Scale"].default_value = scale
        mp.inputs["Rotation"].default_value = [math.radians(r) for r in rot]
        return mp.outputs["Vector"]

    def noise(self, vec, scale, detail=4.0, rough=0.55, distortion=0.0):
        n = self.node("ShaderNodeTexNoise")
        self.inp(n.inputs["Vector"], vec)
        n.inputs["Scale"].default_value = scale
        n.inputs["Detail"].default_value = detail
        n.inputs["Roughness"].default_value = rough
        n.inputs["Distortion"].default_value = distortion
        return n.outputs["Fac"]

    def edge_mask(self, radius, gain):
        bev = self.node("ShaderNodeBevel", samples=8)
        bev.inputs["Radius"].default_value = radius
        geo = self.node("ShaderNodeNewGeometry")
        dot = self.node("ShaderNodeVectorMath", operation="DOT_PRODUCT")
        self.t.links.new(bev.outputs[0], dot.inputs[0])
        self.t.links.new(geo.outputs["Normal"], dot.inputs[1])
        return self.math("MULTIPLY", self.math("SUBTRACT", 1.0, dot.outputs["Value"]), gain, clamp=True)

    def cavity(self, dist):
        ao = self.node("ShaderNodeAmbientOcclusion", samples=12)
        ao.inputs["Distance"].default_value = dist
        return self.math("SUBTRACT", 1.0, ao.outputs["AO"], clamp=True)

    def scratches(self, density_scale, width=0.006, stretch=25.0, base_scale=9.0):
        lines = None
        for rot in ((0, 0, 20), (30, 0, -35), (0, 60, 75)):
            v = self.coords(scale=(stretch * base_scale, base_scale, stretch * base_scale * 0.5), rot=rot)
            n = self.noise(v, 1.0, detail=2.0, rough=0.4)
            d = self.math("ABSOLUTE", self.math("SUBTRACT", n, 0.5))
            l = self.math("SUBTRACT", 1.0, self.smooth(0.0, width, d))
            l = self.math("MULTIPLY", l, 0.6)
            lines = l if lines is None else self.math("MAXIMUM", lines, l)
        sparse = self.smooth(0.56, 0.68, self.noise(self.coords(), density_scale, detail=2))
        return self.math("MULTIPLY", lines, sparse)

    def bump(self, height, strength, distance=0.0004, normal=None):
        b = self.node("ShaderNodeBump")
        b.inputs["Strength"].default_value = strength
        b.inputs["Distance"].default_value = distance
        self.inp(b.inputs["Height"], height)
        if normal is not None:
            self.inp(b.inputs["Normal"], normal)
        return b.outputs["Normal"]

    def bevel_normal(self, radius):
        bev = self.node("ShaderNodeBevel", samples=8)
        bev.inputs["Radius"].default_value = radius
        return bev.outputs[0]

    def combine(self, r, g, b):
        n = self.node("ShaderNodeCombineColor")
        self.inp(n.inputs[0], r)
        self.inp(n.inputs[1], g)
        self.inp(n.inputs[2], b)
        return n.outputs[0]


def recipe_gunmetal(nb):
    v = nb.coords()
    blotch = nb.noise(v, 18.0, detail=6, rough=0.6)
    micro = nb.noise(v, 900.0, detail=3)
    edge = nb.math("MULTIPLY", nb.edge_mask(0.0035, 9.0), nb.smooth(0.35, 0.6, nb.noise(v, 140.0, detail=5)), clamp=True)
    scr = nb.scratches(22.0)
    cav = nb.cavity(0.012)
    dark = nb.mixc(blotch, (0.030, 0.032, 0.036), (0.060, 0.062, 0.066))
    col = nb.mixc(nb.math("MAXIMUM", edge, nb.math("MULTIPLY", scr, 0.7)), dark, (0.42, 0.42, 0.43))
    col = nb.mixc(nb.math("MULTIPLY", cav, 0.6), col, (0.012, 0.011, 0.010))
    rough = nb.math("ADD", 0.40, nb.math("MULTIPLY", nb.math("SUBTRACT", blotch, 0.5), 0.22))
    rough = nb.math("ADD", rough, nb.math("MULTIPLY", nb.math("SUBTRACT", micro, 0.5), 0.10))
    rough = nb.mixf(edge, rough, 0.24)
    rough = nb.mixf(nb.math("MULTIPLY", cav, 0.7), rough, 0.62)
    height = nb.math("SUBTRACT", nb.math("MULTIPLY", micro, 0.25), nb.math("MULTIPLY", scr, 0.6))
    nrm = nb.bump(height, 0.18, normal=nb.bevel_normal(0.0012))
    return col, rough, 1.0, nrm


def recipe_polymer(nb):
    v = nb.coords()
    blotch = nb.noise(v, 14.0, detail=5)
    stipple = nb.noise(v, 1400.0, detail=2)
    edge = nb.math("MULTIPLY", nb.edge_mask(0.003, 7.0), nb.smooth(0.4, 0.6, nb.noise(v, 90.0, detail=4)), clamp=True)
    scr = nb.scratches(16.0, width=0.008)
    cav = nb.cavity(0.012)
    col = nb.mixc(blotch, (0.026, 0.026, 0.028), (0.040, 0.040, 0.042))
    col = nb.mixc(nb.math("MAXIMUM", nb.math("MULTIPLY", edge, 0.8), nb.math("MULTIPLY", scr, 0.5)), col, (0.10, 0.10, 0.10))
    col = nb.mixc(nb.math("MULTIPLY", cav, 0.5), col, (0.008, 0.008, 0.008))
    rough = nb.math("ADD", 0.66, nb.math("MULTIPLY", nb.math("SUBTRACT", blotch, 0.5), 0.16))
    rough = nb.mixf(edge, rough, 0.48)
    nrm = nb.bump(nb.math("SUBTRACT", stipple, nb.math("MULTIPLY", scr, 0.5)), 0.22, normal=nb.bevel_normal(0.0012))
    return col, rough, 0.0, nrm


def recipe_walnut(nb):
    grain_v = nb.coords(scale=(1, 1, 1))
    wave = nb.node("ShaderNodeTexWave", wave_type="RINGS", rings_direction="Y", wave_profile="SIN")
    nb.inp(wave.inputs["Vector"], grain_v)
    wave.inputs["Scale"].default_value = 55.0
    wave.inputs["Distortion"].default_value = 11.0
    wave.inputs["Detail"].default_value = 4.0
    wave.inputs["Detail Scale"].default_value = 0.6
    rings = wave.outputs["Fac"]
    pores = nb.noise(nb.coords(scale=(420, 5, 420)), 1.0, detail=3)
    fig = nb.noise(nb.coords(scale=(90, 4, 90)), 1.0, detail=5)
    edge = nb.math("MULTIPLY", nb.edge_mask(0.004, 6.0), nb.smooth(0.4, 0.62, nb.noise(nb.coords(), 70.0)), clamp=True)
    cav = nb.cavity(0.015)
    g = nb.math("ADD", nb.math("MULTIPLY", rings, 0.55), nb.math("MULTIPLY", fig, 0.45))
    col = nb.mixc(nb.smooth(0.2, 0.9, g), (0.075, 0.032, 0.012), (0.155, 0.068, 0.026))
    col = nb.mixc(nb.math("MULTIPLY", nb.smooth(0.55, 0.8, pores), 0.6), col, (0.025, 0.010, 0.004))
    col = nb.mixc(nb.math("MULTIPLY", edge, 0.6), col, (0.28, 0.15, 0.07))
    col = nb.mixc(nb.math("MULTIPLY", cav, 0.5), col, (0.012, 0.006, 0.003))
    rough = nb.math("ADD", 0.36, nb.math("MULTIPLY", pores, 0.18))
    rough = nb.mixf(edge, rough, 0.62)
    nrm = nb.bump(nb.math("ADD", nb.math("MULTIPLY", pores, 0.6), nb.math("MULTIPLY", rings, 0.2)), 0.12, normal=nb.bevel_normal(0.0018))
    return col, rough, 0.0, nrm


def recipe_olive(nb):
    """Painted olive-drab finish, chipped down to bare steel on edges."""
    v = nb.coords()
    blotch = nb.noise(v, 10.0, detail=6, rough=0.6)
    micro = nb.noise(v, 700.0, detail=3)
    chip = nb.math("MULTIPLY", nb.edge_mask(0.004, 10.0), nb.smooth(0.42, 0.52, nb.noise(v, 120.0, detail=6)), clamp=True)
    chip = nb.smooth(0.35, 0.55, chip)
    scr = nb.scratches(18.0, width=0.010)
    bare = nb.math("MAXIMUM", chip, nb.math("MULTIPLY", scr, 0.8))
    cav = nb.cavity(0.015)
    paint = nb.mixc(blotch, (0.055, 0.062, 0.030), (0.105, 0.110, 0.060))
    col = nb.mixc(bare, paint, (0.36, 0.36, 0.37))
    col = nb.mixc(nb.math("MULTIPLY", cav, 0.55), col, (0.014, 0.013, 0.008))
    rough = nb.math("ADD", 0.62, nb.math("MULTIPLY", nb.math("SUBTRACT", blotch, 0.5), 0.2))
    rough = nb.mixf(bare, rough, 0.32)
    metal = nb.mixf(bare, 0.0, 1.0)
    height = nb.math("SUBTRACT", nb.math("MULTIPLY", micro, 0.2), nb.math("MULTIPLY", bare, 0.8))
    nrm = nb.bump(height, 0.2, normal=nb.bevel_normal(0.0012))
    return col, rough, metal, nrm


RECIPES = {
    "Gunmetal": recipe_gunmetal,
    "Polymer": recipe_polymer,
    "Walnut": recipe_walnut,
    "OliveDrab": recipe_olive,
}

# Base-colour texture size per material; ORM and normal use half of it (min 512).
TEX_SIZE = {"Gunmetal": 1024, "Polymer": 512, "Walnut": 1024, "OliveDrab": 1024}


def make_accent():
    m = bpy.data.materials.new("Accent")
    m.use_nodes = True
    p = m.node_tree.nodes["Principled BSDF"]
    p.inputs["Base Color"].default_value = (0.95, 0.42, 0.03, 1)  # recoloured in game per rarity
    p.inputs["Metallic"].default_value = 0.0
    p.inputs["Roughness"].default_value = 0.42
    return m


def make_lens():
    m = bpy.data.materials.new("Lens")
    m.use_nodes = True
    p = m.node_tree.nodes["Principled BSDF"]
    p.inputs["Base Color"].default_value = (0.03, 0.10, 0.09, 1)  # green/blue coated optics
    p.inputs["Metallic"].default_value = 0.0
    p.inputs["Roughness"].default_value = 0.04
    p.inputs["Alpha"].default_value = 0.72
    p.inputs["Coat Weight"].default_value = 1.0
    m.blend_method = "BLEND"
    return m


# --------------------------------------------------------------------------
# Gun container
# --------------------------------------------------------------------------

class Gun:
    def __init__(self, name):
        self.name = name
        self.parts = []
        self.mats = {}
        self.muzzle = (0, 0, 0)
        self.grip = (0, 0)  # (f, u) of the hand position

    def mat(self, key):
        if key not in self.mats:
            if key == "Accent":
                self.mats[key] = make_accent()
            elif key == "Lens":
                self.mats[key] = make_lens()
            else:
                m = bpy.data.materials.new(key)
                m.use_nodes = True
                self.mats[key] = m
        return self.mats[key]

    def build(self):
        objs = [p.build() for p in self.parts]
        # grip -> origin, then turn 180 deg so the barrel points along Blender -Y (glTF +Z)
        T = Matrix.Rotation(math.pi, 4, "Z") @ Matrix.Translation((0, -self.grip[0], -self.grip[1]))
        for o in objs:
            o.data.transform(T)
            o.data.update()
        mz = bpy.data.objects.new("Muzzle", None)
        mz.empty_display_type = "SINGLE_ARROW"
        mz.empty_display_size = 0.05
        mz.location = T @ Vector(self.muzzle)
        link(mz)
        return objs


# --------------------------------------------------------------------------
# Gun designs (all original, generic)
# --------------------------------------------------------------------------

def design_ar():
    g = Gun("ar")
    ub = 0.055  # bore height
    # Upper receiver
    up = Part(g, "UpperReceiver", "Gunmetal", bevel=0.0018)
    up.prof([(-0.10, 0.035), (0.135, 0.035), (0.135, 0.085), (-0.092, 0.085), (-0.10, 0.075)], 0.05)
    up.box((0.0, -0.035, 0.064), (0.058, 0.04, 0.018))  # rear bulge / forward assist housing
    up.cut_box((0.027, 0.04, 0.056), (0.012, 0.07, 0.018))  # ejection port
    rail(Part(g, "UpperRail", "Gunmetal", bevel=0.0007, segments=1), -0.092, 0.132, 0.085)
    # Bolt carrier seen through the port
    Part(g, "Bolt", "Gunmetal", bevel=0.001).cyl((0.012, -0.0, 0.056), (0.012, 0.08, 0.056), 0.008, n=10)
    # Lower receiver with magwell
    lo = Part(g, "LowerReceiver", "Gunmetal", bevel=0.002)
    lo.prof([(-0.10, 0.035), (-0.10, 0.012), (-0.085, -0.005), (0.02, -0.005), (0.022, -0.05),
             (0.106, -0.05), (0.112, -0.035), (0.116, 0.0), (0.135, 0.02), (0.135, 0.035)], 0.046)
    lo.box((0.0, 0.064, -0.0505), (0.05, 0.09, 0.006))  # magwell flare
    # Charging handle
    ch = Part(g, "ChargingHandle", "Gunmetal", bevel=0.001)
    ch.box((0, -0.104, 0.079), (0.016, 0.012, 0.008)).box((0, -0.111, 0.079), (0.05, 0.007, 0.007))
    # Controls
    ct = Part(g, "Controls", "Gunmetal", bevel=0.0008, segments=1)
    ct.cyl((0.023, -0.055, 0.02), (0.027, -0.055, 0.02), 0.006, n=10)
    ct.box((0.028, -0.048, 0.02), (0.003, 0.016, 0.005))  # selector lever
    ct.cyl((0.023, 0.025, 0.012), (0.026, 0.025, 0.012), 0.005, n=10)  # mag release
    ct.box((-0.025, 0.03, 0.02), (0.004, 0.025, 0.008))  # bolt catch
    # Handguard (octagonal, flat top at 0.085)
    hg = Part(g, "Handguard", "Polymer", bevel=0.0025, angle=20, smooth=30)
    hg.lathe((0, 0.135, ub), (0, 1, 0), [(0, 0.0325), (0.265, 0.0325)], n=8, phase=math.pi / 8)
    for k in range(4):
        hg.cut_box((0.0, 0.19 + k * 0.05, ub), (0.08, 0.026, 0.010))  # side slots
        hg.cut_box((0.0, 0.19 + k * 0.05, ub - 0.02), (0.02, 0.026, 0.03))  # bottom slots
    Part(g, "HandguardRing", "Polymer", bevel=0.0015).lathe(
        (0, 0.128, ub), (0, 1, 0), [(0.0, 0.035), (0.012, 0.035)], n=8, phase=math.pi / 8)
    rail(Part(g, "HandguardRail", "Gunmetal", bevel=0.0007, segments=1), 0.142, 0.395, 0.0855)
    # Barrel, gas block, muzzle brake
    br = Part(g, "Barrel", "Gunmetal", bevel=0.0)
    br.lathe((0, 0.30, ub), (0, 1, 0), [(0, 0.0105), (0.13, 0.0105), (0.135, 0.0093), (0.232, 0.0093), (0.232, 0.0075)], n=16)
    gb = Part(g, "GasBlock", "Gunmetal", bevel=0.0012)
    gb.box((0, 0.414, ub + 0.002), (0.026, 0.02, 0.03))
    gb.cyl((0, 0.40, ub + 0.018), (0, 0.43, ub + 0.018), 0.004, n=10)
    mz = Part(g, "MuzzleBrake", "Gunmetal", bevel=0.0008)
    mz.lathe((0, 0.53, ub), (0, 1, 0), [(0, 0.0), (0, 0.0115), (0.006, 0.013), (0.05, 0.013), (0.055, 0.0115), (0.055, 0.005), (0.04, 0.005)], n=16)
    for k in range(3):
        mz.cut_box((0.0, 0.545 + k * 0.011, ub + 0.008), (0.04, 0.0055, 0.012))
    # Sights: flip-up rear and front
    fs = Part(g, "FrontSight", "Gunmetal", bevel=0.0008)
    fs.box((0, 0.385, 0.0995), (0.02, 0.014, 0.008)).box((0, 0.387, 0.113), (0.016, 0.004, 0.022))
    fs.box((0, 0.387, 0.127), (0.002, 0.003, 0.012))
    rs = Part(g, "RearSight", "Gunmetal", bevel=0.0008)
    rs.box((0, -0.075, 0.0995), (0.022, 0.016, 0.008)).box((0, -0.077, 0.114), (0.02, 0.004, 0.022))
    # Compact reflex sight
    op = Part(g, "Optic", "Polymer", bevel=0.0015)
    op.box((0, 0.02, 0.1005), (0.024, 0.05, 0.01))
    op.tube((0, -0.003, 0.122), (0, 0.045, 0.122), 0.019, 0.0155, n=16)
    op.box((0, 0.012, 0.106), (0.03, 0.034, 0.012))
    op.cyl((0.018, 0.02, 0.122), (0.028, 0.02, 0.122), 0.007, n=10)  # brightness turret
    Part(g, "OpticLens", "Lens", bevel=0.0).cyl((0, 0.036, 0.122), (0, 0.038, 0.122), 0.0156, n=16)
    # Pistol grip
    gp = Part(g, "Grip", "Polymer", bevel=0.007, segments=3, smooth=60)
    gp.prof([(-0.03, -0.004), (-0.035, -0.03), (-0.05, -0.10), (-0.056, -0.112), (-0.086, -0.116),
             (-0.093, -0.106), (-0.074, -0.004)], 0.031)
    # Trigger guard + trigger
    Part(g, "TriggerGuard", "Gunmetal", bevel=0.0012).strip(
        [(-0.036, -0.004), (-0.032, -0.036), (-0.02, -0.043), (0.02, -0.043), (0.024, -0.03)], 0.004, 0.012)
    Part(g, "Trigger", "Gunmetal", bevel=0.0008).prof(
        [(-0.004, -0.004), (0.003, -0.004), (0.0, -0.018), (-0.006, -0.03), (-0.010, -0.029), (-0.006, -0.018)], 0.006)
    # Magazine
    mg = Part(g, "Magazine", "Polymer", bevel=0.003, segments=2)
    mg.prof([(0.026, -0.04), (0.100, -0.04), (0.106, -0.09), (0.118, -0.14), (0.136, -0.186), (0.074, -0.205),
             (0.063, -0.196), (0.047, -0.14), (0.035, -0.09)], 0.024)
    mg.prof([(0.135, -0.185), (0.144, -0.191), (0.143, -0.2), (0.076, -0.222), (0.066, -0.215), (0.068, -0.204)], 0.03)
    # Buffer tube, stock, butt pad
    Part(g, "BufferTube", "Gunmetal", bevel=0.001).lathe(
        (0, -0.10, 0.05), (0, -1, 0), [(0, 0.017), (0.008, 0.017), (0.008, 0.0145), (0.20, 0.0145), (0.20, 0.0)], n=14)
    st = Part(g, "Stock", "Polymer", bevel=0.006, segments=3, smooth=55)
    st.prof([(-0.19, 0.075), (-0.325, 0.078), (-0.333, 0.068), (-0.333, -0.045), (-0.318, -0.052),
             (-0.29, 0.012), (-0.215, 0.028), (-0.19, 0.03)], 0.042)
    st.cut_box((0, -0.27, 0.0), (0.06, 0.03, 0.018), rot=("X", 25))
    Part(g, "ButtPad", "Polymer", bevel=0.003).prof(
        [(-0.333, 0.074), (-0.346, 0.074), (-0.346, -0.05), (-0.333, -0.05)], 0.046)
    # Accent: stock side panels
    ac = Part(g, "AccentPanel", "Accent", bevel=0.0006, segments=1)
    ac.box((0.0215, -0.27, 0.052), (0.002, 0.06, 0.012)).box((-0.0215, -0.27, 0.052), (0.002, 0.06, 0.012))
    ac.box((0.0, 0.27, ub - 0.03), (0.02, 0.06, 0.004))
    g.muzzle = (0, 0.585, ub)
    g.grip = (-0.062, -0.058)
    return g


def design_shotgun():
    g = Gun("shotgun")
    ub = 0.045
    rc = Part(g, "Receiver", "Gunmetal", bevel=0.0025)
    rc.prof([(-0.08, 0.0), (0.13, 0.0), (0.13, 0.054), (0.12, 0.063), (-0.055, 0.063), (-0.08, 0.05)], 0.036)
    rc.cut_box((0.019, 0.04, 0.035), (0.01, 0.065, 0.022))  # ejection port
    Part(g, "Bolt", "Gunmetal", bevel=0.001).box((0.012, 0.04, 0.035), (0.01, 0.06, 0.016))
    Part(g, "Barrel", "Gunmetal", bevel=0.0).lathe(
        (0, 0.13, ub), (0, 1, 0), [(0, 0.0145), (0.025, 0.0145), (0.03, 0.0118), (0.43, 0.0118), (0.43, 0.0095), (0.42, 0.0095)], n=18)
    rib = Part(g, "Rib", "Gunmetal", bevel=0.0005, segments=1)
    rib.box((0, 0.345, ub + 0.0155), (0.008, 0.42, 0.0025))
    for k in range(12):
        rib.box((0, 0.155 + k * 0.035, ub + 0.0128), (0.004, 0.008, 0.004))
    Part(g, "BeadSight", "Gunmetal", bevel=0.0).lathe(
        (0, 0.552, ub + 0.0165), (0, 0, 1), [(0, 0.0018), (0.002, 0.0022), (0.004, 0.0015), (0.005, 0.0)], n=10)
    Part(g, "MagTube", "Gunmetal", bevel=0.0).lathe(
        (0, 0.13, 0.014), (0, 1, 0), [(0, 0.0105), (0.37, 0.0105), (0.37, 0.0118), (0.395, 0.0118), (0.40, 0.009), (0.40, 0.0)], n=16)
    cl = Part(g, "BarrelClamp", "Gunmetal", bevel=0.0012)
    cl.prof([(0.47, 0.0), (0.49, 0.0), (0.49, 0.06), (0.47, 0.06)], 0.026)
    cl.cyl((0.014, 0.48, 0.03), (0.018, 0.48, 0.03), 0.004, n=8)
    pump = Part(g, "Pump", "Walnut", bevel=0.009, segments=3, smooth=60)
    pump.prof([(0.18, -0.006), (0.37, -0.006), (0.375, 0.006), (0.375, 0.036), (0.18, 0.036), (0.175, 0.006)], 0.05)
    for k in range(7):
        pump.cut_box((0, 0.21 + k * 0.022, -0.012), (0.09, 0.006, 0.02))
    Part(g, "ActionBars", "Gunmetal", bevel=0.0006, segments=1).box(
        (0.017, 0.155, 0.012), (0.003, 0.05, 0.008)).box((-0.017, 0.155, 0.012), (0.003, 0.05, 0.008))
    st = Part(g, "Stock", "Walnut", bevel=0.010, segments=3, smooth=60)
    st.prof([(-0.08, 0.058), (-0.20, 0.052), (-0.43, 0.048), (-0.435, -0.112), (-0.40, -0.112),
             (-0.20, -0.048), (-0.155, -0.06), (-0.135, -0.07), (-0.11, -0.068), (-0.095, -0.03), (-0.08, 0.0)], 0.04)
    Part(g, "RecoilPad", "Polymer", bevel=0.004).prof(
        [(-0.43, 0.051), (-0.452, 0.052), (-0.457, -0.115), (-0.435, -0.115)], 0.044)
    Part(g, "TriggerGuard", "Gunmetal", bevel=0.0012).strip(
        [(-0.035, 0.0), (-0.035, -0.03), (-0.025, -0.038), (0.025, -0.038), (0.032, -0.028), (0.032, 0.0)], 0.005, 0.014)
    Part(g, "Trigger", "Gunmetal", bevel=0.0008).prof(
        [(-0.002, 0.0), (0.005, 0.0), (0.002, -0.014), (-0.004, -0.026), (-0.008, -0.025), (-0.004, -0.013)], 0.006)
    Part(g, "Safety", "Gunmetal", bevel=0.0006, segments=1).cyl((-0.004, -0.048, 0.06), (-0.004, -0.048, 0.066), 0.004, n=10)
    Part(g, "Pins", "Gunmetal", bevel=0.0).cyl((-0.0185, -0.02, 0.012), (0.0185, -0.02, 0.012), 0.0025, n=8).cyl(
        (-0.0185, 0.09, 0.012), (0.0185, 0.09, 0.012), 0.0025, n=8)
    ac = Part(g, "AccentInlay", "Accent", bevel=0.0006, segments=1)
    ac.box((0.0, 0.28, 0.0378), (0.02, 0.12, 0.003))
    ac.box((0.0182, -0.0, 0.02), (0.002, 0.07, 0.01))
    g.muzzle = (0, 0.56, ub)
    g.grip = (-0.11, -0.035)
    return g


def design_smg():
    g = Gun("smg")
    ub = 0.05
    rc = Part(g, "Receiver", "Gunmetal", bevel=0.002)
    rc.prof([(-0.10, 0.015), (0.12, 0.015), (0.12, 0.08), (-0.095, 0.08), (-0.10, 0.072)], 0.042)
    rc.cut_box((0.022, 0.03, 0.05), (0.01, 0.05, 0.016))
    Part(g, "Bolt", "Gunmetal", bevel=0.001).cyl((0.01, 0.0, 0.05), (0.01, 0.07, 0.05), 0.008, n=10)
    rail(Part(g, "TopRail", "Gunmetal", bevel=0.0007, segments=1), -0.09, 0.115, 0.08)
    hg = Part(g, "Handguard", "Polymer", bevel=0.004, segments=2)
    hg.prof([(0.12, 0.018), (0.215, 0.022), (0.22, 0.03), (0.22, 0.072), (0.12, 0.079)], 0.046)
    for k in range(3):
        hg.cut_box((0, 0.145 + k * 0.025, 0.05), (0.07, 0.012, 0.022))
    Part(g, "Barrel", "Gunmetal", bevel=0.0).lathe((0, 0.20, ub), (0, 1, 0), [(0, 0.0085), (0.06, 0.0085), (0.06, 0.006)], n=14)
    mz = Part(g, "MuzzleDevice", "Gunmetal", bevel=0.0008)
    mz.lathe((0, 0.255, ub), (0, 1, 0), [(0, 0.0), (0, 0.011), (0.004, 0.0125), (0.04, 0.0125), (0.045, 0.011), (0.045, 0.005), (0.035, 0.005)], n=16)
    for k in range(3):
        mz.cut_box((0, 0.268 + k * 0.009, ub), (0.04, 0.0045, 0.008))
    lo = Part(g, "Lower", "Polymer", bevel=0.003, segments=2)
    lo.prof([(-0.10, 0.016), (0.12, 0.016), (0.12, 0.0), (0.085, -0.01), (0.075, -0.035), (0.04, -0.035),
             (0.035, -0.01), (-0.095, -0.004)], 0.044)
    gp = Part(g, "Grip", "Polymer", bevel=0.007, segments=3, smooth=60)
    gp.prof([(-0.035, -0.004), (-0.04, -0.03), (-0.052, -0.095), (-0.056, -0.108), (-0.086, -0.112),
             (-0.092, -0.1), (-0.078, -0.004)], 0.03)
    Part(g, "TriggerGuard", "Polymer", bevel=0.0015).strip(
        [(-0.038, -0.004), (-0.036, -0.034), (-0.026, -0.042), (0.03, -0.042), (0.04, -0.034)], 0.005, 0.013)
    Part(g, "Trigger", "Gunmetal", bevel=0.0008).prof(
        [(-0.008, -0.006), (-0.001, -0.006), (-0.004, -0.02), (-0.01, -0.031), (-0.014, -0.03), (-0.01, -0.019)], 0.006)
    mg = Part(g, "Magazine", "Gunmetal", bevel=0.0025)
    mg.prof([(0.042, -0.03), (0.074, -0.03), (0.082, -0.175), (0.05, -0.175)], 0.022)
    mg.prof([(0.046, -0.175), (0.088, -0.175), (0.088, -0.186), (0.044, -0.186)], 0.027)
    for k in range(3):
        mg.cut_box((0.0115, 0.0632, -0.06 - k * 0.032), (0.003, 0.01, 0.022))
    # Folding stock: two rods + butt plate + hinge
    stk = Part(g, "Stock", "Gunmetal", bevel=0.0)
    stk.cyl((0.0, -0.095, 0.064), (0.0, -0.285, 0.064), 0.0055, n=10)
    stk.cyl((0.0, -0.095, 0.03), (0.0, -0.285, 0.0), 0.0055, n=10)
    Part(g, "StockHinge", "Gunmetal", bevel=0.0012).box((0, -0.104, 0.048), (0.036, 0.012, 0.05))
    Part(g, "ButtPlate", "Polymer", bevel=0.004).prof(
        [(-0.28, 0.08), (-0.30, 0.08), (-0.305, 0.07), (-0.305, -0.02), (-0.28, -0.02)], 0.034)
    ch = Part(g, "ChargingHandle", "Gunmetal", bevel=0.0008)
    ch.cyl((-0.021, 0.09, 0.064), (-0.034, 0.09, 0.064), 0.0045, n=10)
    ch.cyl((-0.034, 0.09, 0.064), (-0.038, 0.09, 0.064), 0.0065, n=12)
    sel = Part(g, "Selector", "Gunmetal", bevel=0.0006, segments=1)
    sel.cyl((0.022, -0.05, 0.005), (0.025, -0.05, 0.005), 0.006, n=10).box((0.026, -0.045, 0.005), (0.002, 0.014, 0.004))
    # Holographic sight with flat glass window
    hs = Part(g, "HoloSight", "Polymer", bevel=0.0015)
    hs.box((0, 0.0, 0.0905), (0.03, 0.06, 0.01))
    hs.box((0.0135, 0.008, 0.112), (0.004, 0.035, 0.034)).box((-0.0135, 0.008, 0.112), (0.004, 0.035, 0.034))
    hs.box((0, 0.008, 0.1305), (0.031, 0.035, 0.004))
    hs.box((0, -0.018, 0.105), (0.03, 0.022, 0.022))
    Part(g, "HoloLens", "Lens", bevel=0.0).box((0, 0.022, 0.111), (0.023, 0.002, 0.034))
    Part(g, "AccentPanel", "Accent", bevel=0.0006, segments=1).box((0.0232, 0.17, 0.031), (0.002, 0.085, 0.005)).box(
        (-0.0232, 0.17, 0.031), (0.002, 0.085, 0.005))
    g.muzzle = (0, 0.30, ub)
    g.grip = (-0.064, -0.055)
    return g


def design_sniper():
    g = Gun("sniper")
    ub = 0.06
    st = Part(g, "Stock", "OliveDrab", bevel=0.008, segments=3, smooth=60)
    st.prof([(0.33, 0.056), (0.33, 0.022), (0.30, 0.012), (0.085, 0.006), (0.08, 0.0), (0.0, 0.0), (-0.065, 0.0),
             (-0.072, -0.02), (-0.086, -0.10), (-0.092, -0.11), (-0.122, -0.112), (-0.118, -0.06), (-0.16, -0.03),
             (-0.40, -0.05), (-0.55, -0.075), (-0.556, 0.07), (-0.50, 0.085), (-0.34, 0.085), (-0.20, 0.058),
             (-0.12, 0.046), (0.10, 0.046)], 0.05)
    Part(g, "ButtPad", "Polymer", bevel=0.004).prof(
        [(-0.553, 0.072), (-0.572, 0.072), (-0.572, -0.08), (-0.548, -0.078)], 0.054)
    rc = Part(g, "Receiver", "Gunmetal", bevel=0.0)
    rc.lathe((0, -0.11, ub), (0, 1, 0), [(0, 0.0), (0, 0.017), (0.21, 0.017), (0.21, 0.0)], n=18)
    rc.cut_box((0.016, -0.03, ub + 0.004), (0.012, 0.06, 0.016))
    Part(g, "BoltShroud", "Gunmetal", bevel=0.0012).lathe(
        (0, -0.11, ub), (0, -1, 0), [(0, 0.014), (0.03, 0.013), (0.035, 0.008), (0.035, 0.0)], n=14)
    bh = Part(g, "BoltHandle", "Gunmetal", bevel=0.0)
    bh.cyl((0.0, -0.075, ub), (0.045, -0.08, ub - 0.03), 0.0045, n=10)
    bh.lathe((0.045, -0.08, ub - 0.03), (1, -0.1, -0.6), [(-0.003, 0.0), (-0.003, 0.006), (0.0, 0.0095), (0.008, 0.0095), (0.014, 0.006), (0.016, 0.0)], n=14)
    Part(g, "Barrel", "Gunmetal", bevel=0.0).lathe(
        (0, 0.10, ub), (0, 1, 0), [(0, 0.016), (0.03, 0.016), (0.04, 0.0135), (0.48, 0.0095), (0.48, 0.006)], n=18)
    mz = Part(g, "MuzzleBrake", "Gunmetal", bevel=0.001)
    mz.lathe((0, 0.575, ub), (0, 1, 0), [(0, 0.0), (0, 0.0135), (0.005, 0.015), (0.06, 0.015), (0.065, 0.013), (0.065, 0.006), (0.05, 0.006)], n=18)
    for k in range(3):
        mz.cut_box((0, 0.59 + k * 0.014, ub + 0.006), (0.05, 0.007, 0.012))
    rl = Part(g, "ScopeRail", "Gunmetal", bevel=0.0007, segments=1)
    rail(rl, -0.095, 0.095, ub + 0.015, w=0.02)
    # Scope rings
    for i, f in enumerate((-0.055, 0.055)):
        r = Part(g, "ScopeRing%d" % (i + 1), "Gunmetal", bevel=0.001)
        r.tube((0, f - 0.008, 0.125), (0, f + 0.008, 0.125), 0.0195, 0.0152, n=16)
        r.box((0, f, 0.097), (0.022, 0.016, 0.014))
        r.cyl((0.017, f, 0.096), (0.024, f, 0.096), 0.004, n=8)
    sc = Part(g, "Scope", "Gunmetal", bevel=0.0)
    sc.lathe((0, -0.15, 0.125), (0, 1, 0), [(0, 0.016), (0.002, 0.0205), (0.045, 0.0205), (0.05, 0.019), (0.07, 0.0152),
                                            (0.235, 0.0152), (0.27, 0.025), (0.33, 0.025), (0.333, 0.022), (0.325, 0.022)], n=20)
    sc.lathe((0, -0.15, 0.125), (0, 1, 0), [(0, 0.016), (0.0, 0.0165), (0.005, 0.0165)], n=20)  # eyepiece lip
    tr = Part(g, "Turrets", "Gunmetal", bevel=0.0008)
    tr.box((0, 0.01, 0.125), (0.034, 0.05, 0.03))
    tr.cyl((0, 0.01, 0.14), (0, 0.01, 0.158), 0.012, n=16).cyl((0, 0.01, 0.158), (0, 0.01, 0.163), 0.0105, n=16)
    tr.cyl((0.017, 0.01, 0.125), (0.033, 0.01, 0.125), 0.011, n=16)
    tr.cyl((-0.017, 0.022, 0.125), (-0.028, 0.022, 0.125), 0.008, n=12)
    lens = Part(g, "ScopeLens", "Lens", bevel=0.0)
    lens.cyl((0, 0.172, 0.125), (0, 0.174, 0.125), 0.0225, n=20)
    lens.cyl((0, -0.146, 0.125), (0, -0.144, 0.125), 0.016, n=16)
    Part(g, "Magazine", "Polymer", bevel=0.0025).prof(
        [(0.005, 0.002), (0.075, 0.002), (0.073, -0.04), (0.007, -0.04)], 0.028)
    Part(g, "TriggerGuard", "Gunmetal", bevel=0.0012).strip(
        [(-0.066, 0.0), (-0.062, -0.03), (-0.05, -0.04), (-0.005, -0.04), (0.003, -0.03), (0.003, 0.0)], 0.005, 0.015)
    Part(g, "Trigger", "Gunmetal", bevel=0.0008).prof(
        [(-0.032, 0.0), (-0.025, 0.0), (-0.028, -0.014), (-0.034, -0.026), (-0.038, -0.025), (-0.034, -0.013)], 0.006)
    Part(g, "SlingStuds", "Gunmetal", bevel=0.0).cyl((0, 0.29, 0.012), (0, 0.29, 0.0), 0.004, n=8).cyl(
        (0, -0.47, -0.058), (0, -0.47, -0.074), 0.004, n=8)
    ac = Part(g, "AccentStripe", "Accent", bevel=0.0006, segments=1)
    ac.box((0.0255, 0.20, 0.035), (0.002, 0.16, 0.006)).box((-0.0255, 0.20, 0.035), (0.002, 0.16, 0.006))
    g.muzzle = (0, 0.64, ub)
    g.grip = (-0.098, -0.055)
    return g


def design_rocket():
    g = Gun("rocket")
    ua = 0.115  # tube axis height
    tb = Part(g, "LaunchTube", "OliveDrab", bevel=0.0012, angle=30)
    tb.lathe((0, -0.58, ua), (0, 1, 0), [(0, 0.041), (0, 0.047), (0.06, 0.047), (0.075, 0.044), (1.065, 0.044),
                                        (1.065, 0.049), (1.10, 0.049), (1.10, 0.041)], n=24, closed=True)
    vt = Part(g, "Venturi", "Gunmetal", bevel=0.001)
    vt.lathe((0, -0.585, ua), (0, 1, 0), [(0, 0.053), (0.0, 0.058), (0.012, 0.058), (0.025, 0.046), (0.025, 0.041),
                                         (0.012, 0.050)], n=24, closed=True)
    for i, f in enumerate((-0.24, 0.12)):
        Part(g, "TubeClamp%d" % (i + 1), "Gunmetal", bevel=0.0008).lathe(
            (0, f, ua), (0, 1, 0), [(0, 0.0435), (0, 0.048), (0.025, 0.048), (0.025, 0.0435)], n=24, closed=True)
    wh = Part(g, "Warhead", "OliveDrab", bevel=0.0)
    wh.lathe((0, 0.40, ua), (0, 1, 0), [(0, 0.0), (0, 0.04), (0.08, 0.04), (0.13, 0.03), (0.165, 0.016), (0.175, 0.007), (0.178, 0.0)], n=24)
    Part(g, "WarheadTip", "Gunmetal", bevel=0.0).lathe(
        (0, 0.575, ua), (0, 1, 0), [(0, 0.0072), (0.012, 0.004), (0.016, 0.0)], n=12)
    Part(g, "Bore", "Polymer", bevel=0.0).lathe(  # dark backing so the tube does not look empty from behind
        (0, -0.50, ua), (0, 1, 0), [(0, 0.0), (0, 0.041), (0.01, 0.041), (0.01, 0.0)], n=24)
    hs = Part(g, "FireControl", "Polymer", bevel=0.003)
    hs.prof([(-0.13, 0.072), (0.06, 0.072), (0.05, 0.035), (-0.02, 0.028), (-0.12, 0.028)], 0.05)
    gp = Part(g, "Grip", "Polymer", bevel=0.007, segments=3, smooth=60)
    gp.prof([(-0.03, 0.032), (-0.036, 0.0), (-0.052, -0.07), (-0.058, -0.082), (-0.088, -0.086),
             (-0.094, -0.075), (-0.076, 0.032)], 0.032)
    Part(g, "TriggerGuard", "Gunmetal", bevel=0.0012).strip(
        [(-0.036, 0.03), (-0.033, -0.005), (-0.023, -0.013), (0.03, -0.013), (0.04, 0.0), (0.04, 0.03)], 0.005, 0.014)
    Part(g, "Trigger", "Gunmetal", bevel=0.0008).prof(
        [(-0.002, 0.03), (0.006, 0.03), (0.003, 0.014), (-0.004, 0.003), (-0.008, 0.004), (-0.004, 0.016)], 0.007)
    fg = Part(g, "ForeGrip", "Polymer", bevel=0.007, segments=3, smooth=60)
    fg.prof([(0.15, 0.075), (0.21, 0.075), (0.2, 0.05), (0.196, -0.035), (0.19, -0.045), (0.163, -0.045),
             (0.158, -0.035), (0.16, 0.05)], 0.034)
    sh = Part(g, "ShoulderRest", "Polymer", bevel=0.006, segments=3, smooth=55)
    sh.prof([(-0.44, 0.074), (-0.25, 0.074), (-0.27, 0.05), (-0.36, 0.035), (-0.43, 0.03), (-0.45, 0.05)], 0.06)
    # Optical sight on the left side
    sb = Part(g, "SightBracket", "Gunmetal", bevel=0.0012)
    sb.box((-0.05, -0.08, ua + 0.01), (0.022, 0.06, 0.012))
    sb.box((-0.063, -0.08, ua + 0.022), (0.012, 0.05, 0.022))
    so = Part(g, "Sight", "Polymer", bevel=0.0015)
    so.tube((-0.07, -0.16, ua + 0.045), (-0.07, -0.02, ua + 0.045), 0.0175, 0.0135, n=16)
    so.lathe((-0.07, -0.02, ua + 0.045), (0, 1, 0), [(0, 0.0135), (0, 0.0195), (0.03, 0.0195), (0.03, 0.017)], n=16, closed=False)
    so.cyl((-0.07, -0.08, ua + 0.06), (-0.07, -0.08, ua + 0.072), 0.008, n=12)
    so.box((-0.07, -0.08, ua + 0.03), (0.02, 0.05, 0.012))
    Part(g, "EyeCup", "Polymer", bevel=0.002).tube((-0.07, -0.19, ua + 0.045), (-0.07, -0.16, ua + 0.045), 0.019, 0.0145, n=16)
    sl = Part(g, "SightLens", "Lens", bevel=0.0)
    sl.cyl((-0.07, 0.006, ua + 0.045), (-0.07, 0.008, ua + 0.045), 0.0168, n=16)
    sl.cyl((-0.07, -0.158, ua + 0.045), (-0.07, -0.156, ua + 0.045), 0.0135, n=16)
    # Iron backup sights on top
    ir = Part(g, "IronSights", "Gunmetal", bevel=0.0008)
    ir.box((0, -0.05, ua + 0.05), (0.016, 0.012, 0.016)).box((0, 0.30, ua + 0.05), (0.004, 0.008, 0.02))
    ir.box((0, 0.30, ua + 0.044), (0.016, 0.016, 0.006))
    Part(g, "AccentBand", "Accent", bevel=0.0).lathe(
        (0, 0.36, ua), (0, 1, 0), [(0, 0.0438), (0, 0.0455), (0.03, 0.0455), (0.03, 0.0438)], n=24, closed=True)
    g.muzzle = (0, 0.52, ua)
    g.grip = (-0.062, -0.025)
    return g


DESIGNS = {"ar": design_ar, "shotgun": design_shotgun, "smg": design_smg, "sniper": design_sniper, "rocket": design_rocket}


# --------------------------------------------------------------------------
# UV + bake
# --------------------------------------------------------------------------

def select_only(objs, active):
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = active


def unwrap(objs):
    select_only(objs, objs[0])
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(60), island_margin=0.004, area_weight=0.0, scale_to_bounds=False)
    bpy.ops.uv.pack_islands(rotate=True, margin=0.004)
    bpy.ops.object.mode_set(mode="OBJECT")


def bake_material(gun_name, mat, objs):
    recipe = RECIPES[mat.name]
    size = TEX_SIZE[mat.name]
    small = max(512, size // 2)
    tree = mat.node_tree
    tree.nodes.clear()
    nb = NB(tree)
    col, rough, metal, nrm = recipe(nb)
    out = nb.node("ShaderNodeOutputMaterial")
    emit = nb.node("ShaderNodeEmission")
    bsdf = nb.node("ShaderNodeBsdfPrincipled")
    nb.inp(bsdf.inputs["Normal"], nrm)

    def img(suffix, res, data):
        im = bpy.data.images.new("%s_%s_%s" % (gun_name, mat.name, suffix), res, res, alpha=False)
        im.colorspace_settings.name = "Non-Color" if data else "sRGB"
        n = nb.node("ShaderNodeTexImage")
        n.image = im
        return n, im

    n_base, im_base = img("base", size, False)
    n_orm, im_orm = img("orm", small, True)
    n_nrm, im_nrm = img("normal", small, True)

    select_only(objs, objs[0])
    for node, src, kind in ((n_base, col, "EMIT"), (n_orm, nb.combine(1.0, rough, metal), "EMIT"), (n_nrm, None, "NORMAL")):
        for l in list(out.inputs["Surface"].links):
            tree.links.remove(l)
        if kind == "EMIT":
            nb.inp(emit.inputs["Color"], src)
            emit.inputs["Strength"].default_value = 1.0
            tree.links.new(emit.outputs[0], out.inputs["Surface"])
        else:
            tree.links.new(bsdf.outputs[0], out.inputs["Surface"])
        tree.nodes.active = node
        bpy.ops.object.bake(type=kind, normal_space="TANGENT", margin=6, margin_type="EXTEND", use_clear=True)

    os.makedirs(TEX_TMP, exist_ok=True)
    for im in (im_base, im_orm, im_nrm):
        im.filepath_raw = os.path.join(TEX_TMP, im.name + ".png")
        im.file_format = "PNG"
        im.save()

    # Final glTF-friendly material
    tree.nodes.clear()
    nb = NB(tree)
    out = nb.node("ShaderNodeOutputMaterial")
    bsdf = nb.node("ShaderNodeBsdfPrincipled")
    tree.links.new(bsdf.outputs[0], out.inputs["Surface"])
    tb = nb.node("ShaderNodeTexImage", image=im_base)
    tree.links.new(tb.outputs["Color"], bsdf.inputs["Base Color"])
    to = nb.node("ShaderNodeTexImage", image=im_orm)
    sep = nb.node("ShaderNodeSeparateColor")
    tree.links.new(to.outputs["Color"], sep.inputs[0])
    tree.links.new(sep.outputs[1], bsdf.inputs["Roughness"])
    tree.links.new(sep.outputs[2], bsdf.inputs["Metallic"])
    tn = nb.node("ShaderNodeTexImage", image=im_nrm)
    nm = nb.node("ShaderNodeNormalMap")
    tree.links.new(tn.outputs["Color"], nm.inputs["Color"])
    tree.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])


def setup_cycles(samples):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = samples
    sc.cycles.use_denoising = False


# --------------------------------------------------------------------------
# Export, verify, preview
# --------------------------------------------------------------------------

def tri_count(objs):
    dg = bpy.context.evaluated_depsgraph_get()
    n = 0
    for o in objs:
        if o.type != "MESH":
            continue
        me = o.evaluated_get(dg).data
        me.calc_loop_triangles()
        n += len(me.loop_triangles)
    return n


def export(path):
    bpy.ops.export_scene.gltf(
        filepath=path, export_format="GLB", export_yup=True, export_apply=True,
        export_tangents=True, export_normals=True, export_texcoords=True,
        export_materials="EXPORT", export_image_format="JPEG", export_jpeg_quality=88,
        export_cameras=False, export_lights=False, export_animations=False,
        export_draco_mesh_compression_enable=False, use_selection=False,
    )


def verify(path):
    reset_scene()
    bpy.ops.import_scene.gltf(filepath=path)
    mz = bpy.data.objects.get("Muzzle")
    assert mz is not None and mz.type == "EMPTY", "Muzzle empty missing in %s" % path
    assert bpy.data.materials.get("Accent") is not None, "Accent material missing in %s" % path
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    used = {s.material.name for o in meshes for s in o.material_slots if s.material}
    assert "Accent" in used, "Accent material not used by any mesh"
    # bounding box in glTF space (glTF x, y, z) = Blender (x, z, -y)
    lo = Vector((1e9,) * 3)
    hi = Vector((-1e9,) * 3)
    for o in meshes:
        for c in o.bound_box:
            w = o.matrix_world @ Vector(c)
            lo = Vector(map(min, lo, w))
            hi = Vector(map(max, hi, w))
    mzw = mz.matrix_world.translation
    info = {
        "triangles": tri_count(meshes),
        "parts": len(meshes),
        "materials": sorted(used),
        "length_m": round(hi.y - lo.y, 3),
        "muzzle_gltf": [round(mzw.x, 3), round(mzw.z, 3), round(-mzw.y, 3)],
        "bbox_gltf_min": [round(lo.x, 3), round(lo.z, 3), round(-hi.y, 3)],
        "bbox_gltf_max": [round(hi.x, 3), round(hi.z, 3), round(-lo.y, 3)],
    }
    assert info["muzzle_gltf"][2] > 0.15, "barrel does not point along +Z"
    return meshes, info


def render_preview(meshes, path):
    sc = bpy.context.scene
    setup_cycles(96)
    sc.cycles.use_denoising = True
    sc.render.resolution_x = 1600
    sc.render.resolution_y = 800
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "AgX - Medium High Contrast"

    lo = Vector((1e9,) * 3)
    hi = Vector((-1e9,) * 3)
    for o in meshes:
        for c in o.bound_box:
            w = o.matrix_world @ Vector(c)
            lo = Vector(map(min, lo, w))
            hi = Vector(map(max, hi, w))
    ctr = (lo + hi) / 2

    world = bpy.data.worlds.new("Studio")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.18, 0.19, 0.2, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.12
    sc.world = world

    # Backdrop sweep
    bm = bmesh.new()
    W = 8.0
    prof = [(-3.0, -1.0), (0.6, -1.0)]
    for i in range(1, 9):
        a = math.radians(90 * i / 8)
        prof.append((0.6 + math.sin(a) * 1.2, -1.0 + (1 - math.cos(a)) * 1.2))
    prof.append((1.8, 3.0))
    rows = [[bm.verts.new((x, -W / 2 + W * k, z)) for k in (0, 1)] for x, z in prof]
    for a, b in zip(rows[:-1], rows[1:]):
        bm.faces.new((a[0], a[1], b[1], b[0]))
    me = bpy.data.meshes.new("Backdrop")
    bm.to_mesh(me)
    bm.free()
    me.shade_smooth()
    bd = link(bpy.data.objects.new("Backdrop", me))
    bd.location = (ctr.x + 0.6, ctr.y, ctr.z + 0.75)
    bmat = bpy.data.materials.new("Backdrop")
    bmat.use_nodes = True
    bp = bmat.node_tree.nodes["Principled BSDF"]
    bp.inputs["Base Color"].default_value = (0.20, 0.21, 0.22, 1)
    bp.inputs["Roughness"].default_value = 0.8
    me.materials.append(bmat)
    # Gun floats ~ 0.25 m above the floor; contact shadow from the floor below
    bd.location.z = lo.z - 0.25 + 1.0

    def area(name, loc, size, energy, color=(1, 1, 1)):
        ld = bpy.data.lights.new(name, "AREA")
        ld.shape = "RECTANGLE"
        ld.size, ld.size_y = size
        ld.energy = energy
        ld.color = color
        lo_ = link(bpy.data.objects.new(name, ld))
        lo_.location = Vector(loc) + ctr
        d = ctr - lo_.location
        lo_.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
        return lo_

    area("Key", (-1.4, -0.9, 1.6), (1.6, 1.0), 90, (1.0, 0.97, 0.93))
    area("Fill", (-1.6, 1.4, 0.4), (1.2, 1.2), 22, (0.9, 0.95, 1.0))
    area("Rim", (1.2, 0.2, 1.4), (2.0, 0.4), 80)
    area("Top", (0.0, 0.0, 2.0), (2.0, 0.6), 20)

    cam_d = bpy.data.cameras.new("Cam")
    cam_d.lens = 85
    cam_d.sensor_fit = "HORIZONTAL"
    cam_d.sensor_width = 36
    cam = link(bpy.data.objects.new("Cam", cam_d))
    span = max(hi.y - lo.y, (hi.z - lo.z) * 2.0)
    half_fov = math.atan(cam_d.sensor_width / 2 / cam_d.lens)
    dist = span * 1.12 / (2 * math.tan(half_fov))
    view = Vector((-0.93, -0.22, 0.30)).normalized()
    cam.location = ctr + view * dist
    cam.rotation_euler = (ctr - cam.location).to_track_quat("-Z", "Y").to_euler()
    sc.camera = cam
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)


def make_gun(name):
    scene = reset_scene()
    gun = DESIGNS[name]()
    objs = gun.build()
    tris = tri_count(objs)
    print("PARTS", name, sorted(((tri_count([o]), o.name) for o in objs), reverse=True))

    setup_cycles(16)
    by_mat = {}
    for o in objs:
        by_mat.setdefault(o.data.materials[0].name, []).append(o)
    for mname, group in by_mat.items():
        if mname in RECIPES:
            unwrap(group)
            bake_material(name, bpy.data.materials[mname], group)
        else:
            unwrap(group)  # UVs for consistency (tangents need them)

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name + ".glb")
    export(path)
    size = os.path.getsize(path)

    meshes, info = verify(path)
    info.update({"file": "models/guns/%s.glb" % name, "bytes": size, "built_triangles": tris})
    os.makedirs(PREVIEW_DIR, exist_ok=True)
    render_preview(meshes, os.path.join(PREVIEW_DIR, name + ".png"))
    print("GUN %s: %s" % (name, json.dumps(info)))
    return info


def preview_only(name):
    path = os.path.join(OUT_DIR, name + ".glb")
    meshes, info = verify(path)
    render_preview(meshes, os.path.join(PREVIEW_DIR, name + ".png"))
    print("GUN %s: %s" % (name, json.dumps(info)))


def main():
    argv = sys.argv
    args = argv[argv.index("--") + 1:] if "--" in argv else argv[1:]
    if "--preview-only" in args:
        for n in [a for a in args if a in DESIGNS] or GUN_NAMES:
            preview_only(n)
        return
    names = [a for a in args if a in DESIGNS] or GUN_NAMES
    stats = {}
    stats_path = os.path.join(OUT_DIR, "stats.json")
    if os.path.exists(stats_path):
        with open(stats_path) as fh:
            stats = json.load(fh)
    for n in names:
        stats[n] = make_gun(n)
        with open(stats_path, "w") as fh:
            json.dump(stats, fh, indent=2)
    for n in names:
        s = stats[n]
        print("%-8s %6d tris  %7.1f KB  %.2f m  parts=%d" % (n, s["triangles"], s["bytes"] / 1024, s["length_m"], s["parts"]))


if __name__ == "__main__":
    main()
