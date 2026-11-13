#!/usr/bin/env python3
"""
THE INFINITE GLOWING MARBLE JOURNEY
Procedural Blender 5.x scene: 85 floating glass tubes, physically-derived marble motion,
cinematic camera, crystal-chamber finale. Vertical 9:16, 60 fps, no text.

Run (Blender binary):   blender -b -P marble_run.py -- [options]
Run (pip bpy module):   python3 marble_run.py -- [options]

Options:
  --engine eevee|cycles   eevee = fast (needs GPU/OpenGL), cycles = robust (CPU or GPU)
  --width 1080 --height 1920 --fps 60 --duration 100 --samples 64
  --start N --end M       render only this frame range (chunking, resumable)
  --frames 1,600,3000     render single preview frames
  --out /path/frames/     output folder (JPEG sequence, existing frames are skipped)
  --save-blend file.blend save the generated scene (open it in Blender to scrub the timeline)
  --gpu                   use GPU (OptiX/CUDA) for Cycles
  --seed 7                change the layout
  --no-render             only build (useful with --save-blend)
"""
import bpy, bmesh, math, random, sys, argparse, os, time
import numpy as np
from mathutils import Vector, Matrix
from mathutils.kdtree import KDTree

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--engine", default="eevee", choices=["eevee", "cycles"])
ap.add_argument("--width", type=int, default=1080)
ap.add_argument("--height", type=int, default=1920)
ap.add_argument("--fps", type=int, default=60)
ap.add_argument("--duration", type=float, default=100.0)
ap.add_argument("--samples", type=int, default=64)
ap.add_argument("--start", type=int, default=None)
ap.add_argument("--end", type=int, default=None)
ap.add_argument("--frames", type=str, default=None)
ap.add_argument("--out", default="//frames/")
ap.add_argument("--save-blend", default=None)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--gpu", action="store_true")
ap.add_argument("--no-render", action="store_true")
A = ap.parse_args(argv)
T0 = time.time()
rng = random.Random(A.seed)
np.random.seed(A.seed)

# ---------------------------------------------------------------- constants
G, MU, KDRAG, VMIN = 9.81, 0.035, 0.018, 1.4     # gravity, rolling friction, air drag, safety floor
DS, FDT = 0.06, 1 / 240                          # path step (m), flight time step (s)
R_M, R_T = 0.12, 0.20                            # marble / tube radius
R0, HOME = 16.0, 0.35                            # macro spiral radius, homing gain
N_TUBES = 85
FINALE = 5.5                                     # seconds after first impact
PAL = [(0.02, 0.25, 1.0), (0.40, 0.08, 1.0), (0.02, 0.85, 0.80), (0.03, 0.80, 0.30),
       (1.0, 0.04, 0.55), (1.0, 0.38, 0.02), (1.0, 0.70, 0.22)]   # blue violet turquoise emerald magenta amber gold


def lerp(a, b, t): return a + (b - a) * t
def lerpc(a, b, t): return tuple(lerp(a[i], b[i], t) for i in range(3))
def pal(h):
    h = max(0.0, min(len(PAL) - 1 - 1e-6, h)); i = int(h); return lerpc(PAL[i], PAL[i + 1], h - i)
def wrap(a): return (a + math.pi) % (2 * math.pi) - math.pi
def sstep(x): x = max(0.0, min(1.0, x)); return x * x * (3 - 2 * x)
def clamp(x, a, b): return max(a, min(b, x))


def gsmooth(a, sigma):
    a = np.asarray(a, float)
    if sigma < 0.5: return a.copy()
    r = int(3 * sigma); k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); k /= k.sum()
    pad = np.pad(a, [(r, r)] + [(0, 0)] * (a.ndim - 1), mode="edge")
    if a.ndim == 1: return np.convolve(pad, k, mode="valid")
    return np.stack([np.convolve(pad[:, j], k, mode="valid") for j in range(a.shape[1])], axis=1)


# ================================================================ PATH + PHYSICS
class Builder:
    def __init__(s):
        s.P, s.T, s.V, s.TID = [], [], [], []
        s.pos = Vector((R0, 0, 0)); s.yaw = math.pi / 2; s.pitch = -math.pi / 2; s.v = 0.0; s.t = 0.0
        s.tubes = []

    def emit(s, p, tid, dt):
        s.t += dt; s.P.append(p.copy()); s.T.append(s.t); s.V.append(s.v); s.TID.append(tid)

    def dirv(s):
        cp = math.cos(s.pitch)
        return Vector((cp * math.cos(s.yaw), cp * math.sin(s.yaw), math.sin(s.pitch)))

    def home_yaw(s):
        th = math.atan2(s.pos.y, s.pos.x); r = math.hypot(s.pos.x, s.pos.y)
        return th + math.pi / 2 + clamp(0.35 * (r - R0), -0.7, 0.7)

    def step(s, tid, yaw_rate=0.0, pitch_target=-0.10, ease=2.0, home=1.0, pitch_rate=None, lateral=None):
        if pitch_rate is not None: s.pitch += pitch_rate * DS
        else: s.pitch += (pitch_target - s.pitch) * min(1.0, ease * DS)
        s.yaw += yaw_rate * DS + home * HOME * wrap(s.home_yaw() - s.yaw) * DS
        old_z = s.pos.z
        s.pos = s.pos + s.dirv() * DS
        if lateral is not None: s.pos = s.pos + lateral * DS
        dz = s.pos.z - old_z
        # energy: v^2 += -2g dz - 2 mu g ds - 2 k v^2 ds
        v2 = s.v * s.v - 2 * G * dz - 2 * MU * G * DS - 2 * KDRAG * s.v * s.v * DS
        vn = math.sqrt(max(v2, VMIN * VMIN))
        dt = DS / max(0.5 * (s.v + vn), 1e-3)
        s.v = vn; s.emit(s.pos, tid, dt)

    def flight(s, T):
        vel = s.dirv() * s.v
        for _ in range(max(2, int(T / FDT))):
            vel.z -= G * FDT; s.pos = s.pos + vel * FDT; s.v = vel.length; s.emit(s.pos, -1, FDT)
        h = math.hypot(vel.x, vel.y)
        if h > 1e-3: s.yaw = math.atan2(vel.y, vel.x)
        s.pitch = math.atan2(vel.z, h)


def g_start(b, r):
    for _ in range(int(4.5 / DS)): b.step(r["id"], pitch_target=-0.14, ease=1.6)
def g_glide(b, r, L=None):
    L = L or rng.uniform(4.0, 5.5); yr = rng.uniform(-0.12, 0.12)
    for _ in range(int(L / DS)): b.step(r["id"], yaw_rate=yr, pitch_target=-0.10)
def g_curve(b, r):
    L = rng.uniform(4.5, 6.0); sg = rng.choice([-1, 1]); n = int(L / DS)
    for k in range(n): b.step(r["id"], yaw_rate=sg * 0.5 * math.sin(2 * math.pi * k / n), pitch_target=-0.11, home=0.5)
def g_wave(b, r):
    L = rng.uniform(5.0, 6.5); per = rng.uniform(1.8, 2.4)
    for k in range(int(L / DS)):
        b.step(r["id"], pitch_target=-0.13 - 0.24 * math.sin(2 * math.pi * k * DS / per), ease=4.0)
def g_drop(b, r, L=2.4):
    for _ in range(int(L / DS)): b.step(r["id"], pitch_target=-0.6, ease=3.0)
def g_helix(b, r, R=None, turns=None, grand=False):
    R = R or (rng.uniform(1.1, 1.5) + 0.8 * r["x"]); turns = turns or rng.choice([1, 1, 2])
    for _ in range(int(0.7 / DS)): b.step(r["id"], pitch_target=-0.17, ease=4.0)
    sg = rng.choice([-1, 1])
    c = b.pos + Vector((-math.sin(b.yaw), math.cos(b.yaw), 0)) * sg * R; z0 = b.pos.z
    yr = sg * math.cos(b.pitch) / R; p = b.pitch
    for _ in range(int(turns * 2 * math.pi * R / math.cos(p) / DS)):
        b.step(r["id"], yaw_rate=yr, pitch_target=p, ease=0, home=0)
    r["deco"]["pillar"] = (c.x, c.y, b.pos.z - 1.6, z0 + 1.6, R, grand)
    for _ in range(int(0.6 / DS)): b.step(r["id"], pitch_target=-0.12)
def g_grand(b, r): g_helix(b, r, R=rng.uniform(2.6, 3.1), turns=2, grand=True)
def g_loop(b, r, count=1):
    for _ in range(count):
        g_drop(b, r)
        R = min(rng.uniform(0.9, 1.3), 0.8 * b.v ** 2 / (5 * G))   # v^2 >= 5gR keeps contact at the top
        if R < 0.5: continue
        side = Vector((-math.sin(b.yaw), math.cos(b.yaw), 0)) * rng.choice([-1, 1])
        n = int(2 * math.pi * R / DS); lat = side * (0.75 / (n * DS))
        for _ in range(n): b.step(r["id"], pitch_rate=1.0 / R, home=0, lateral=lat)
        b.pitch -= 2 * math.pi
        for _ in range(int(1.2 / DS)): b.step(r["id"], pitch_target=-0.12, ease=3)
def g_loop2(b, r): g_loop(b, r, 2)
def g_split(b, r):
    g_glide(b, r, L=rng.uniform(5.0, 6.0))
    r["deco"]["split"] = 2 if r["x"] < 0.3 else (3 if r["x"] < 0.8 else 4)
def g_rotating(b, r):
    for _ in range(int(0.9 / DS)): b.step(r["id"], pitch_target=-0.12, ease=4)
    r["deco"]["rot_from"] = len(b.P) - 1; p = b.pitch
    for _ in range(int(rng.uniform(3.6, 4.6) / DS)): b.step(r["id"], pitch_target=p, ease=0, home=0)
def g_gates(b, r):
    g_curve(b, r); r["deco"]["gates"] = 5 if r["x"] < 0.8 else 7
def g_prism(b, r):
    g_glide(b, r, L=rng.uniform(5.0, 6.0)); r["deco"]["prism"] = True
def g_final(b, r):
    for _ in range(int(3.0 / DS)): b.step(r["id"], pitch_target=-0.30, ease=2.0)

GEN = dict(start=g_start, glide=g_glide, curve=g_curve, wave=g_wave, helix=g_helix, grand=g_grand,
           loop=g_loop, loop2=g_loop2, split=g_split, rotating=g_rotating, gates=g_gates, prism=g_prism, final=g_final)


def schedule():
    early = ["glide", "curve", "wave", "split", "rotating", "helix", "curve", "wave", "gates"]
    mid = ["curve", "wave", "split", "rotating", "helix", "loop", "gates", "helix", "prism", "wave", "loop"]
    final = ["gates", "helix", "split", "loop", "prism", "grand", "rotating", "gates", "split",
             "loop2", "helix", "gates", "grand", "split"]
    out = ["start"]
    for i in range(1, 70):
        pool = early if i < 25 else mid
        c = rng.choice(pool)
        while c == out[-1]: c = rng.choice(pool)
        out.append(c)
    return out + final + ["final"]


def build_path():
    b = Builder()
    for _ in range(int(0.45 / FDT)): b.emit(b.pos, -1, FDT)           # marble rests at the summit
    b.flight(math.sqrt(2 * 1.1 / G))                                    # ...then drops into tube 1
    for i, tp in enumerate(schedule()):
        r = dict(id=i, type=tp, x=i / (N_TUBES - 1), i0=len(b.P) - 1, deco={},
                 shift=1.0 if (i % 4 == 2 or i >= 70) else 0.25, trail=(tp in ("rotating", "gates") or i >= 70))
        GEN[tp](b, r); r["i1"] = len(b.P) - 1; b.tubes.append(r)
        if i < N_TUBES - 1: b.flight(clamp(0.8 / b.v, 0.07, 0.3))
    # ---- finale: free fall into the crystal chamber, bounces on a concave pedestal, settles
    vel = b.dirv() * b.v; p = b.pos.copy()
    z_land = p.z - 3.6
    tl = (vel.z + math.sqrt(vel.z ** 2 + 2 * G * (p.z - z_land))) / G
    C = Vector((p.x + vel.x * tl, p.y + vel.y * tl, z_land))
    v_in = Vector((vel.x, vel.y, 0)).normalized()
    impacts, roll, first = [], False, True
    for _ in range(int(9.0 / FDT)):
        if not roll: vel.z -= G * FDT
        p = p + vel * FDT
        if not roll and p.z <= z_land and vel.z < 0:
            p.z = z_land; impacts.append(b.t + FDT); vel.z = -0.5 * vel.z
            if first: vel.x *= 0.12; vel.y *= 0.12; first = False
            else:
                vel.x = vel.x * 0.5 + (C.x - p.x) * 1.2; vel.y = vel.y * 0.5 + (C.y - p.y) * 1.2
            if vel.z < 0.35: roll = True; vel.z = 0
        if roll:
            vel.x += (-6 * (p.x - C.x) - 2.2 * vel.x) * FDT; vel.y += (-6 * (p.y - C.y) - 2.2 * vel.y) * FDT
        b.pos = p; b.v = vel.length; b.emit(p, -2, FDT)
    return b, C, v_in, impacts


# ================================================================ BLENDER HELPERS
bpy.ops.wm.read_factory_settings(use_empty=True)
SC = bpy.context.scene
COL = SC.collection


def link(ob): COL.objects.link(ob); return ob


def keys(idb, path, pairs, index=0):
    """Linear keyframes via the layered-action API (Blender 4.4+/5.x)."""
    ad = idb.animation_data or idb.animation_data_create()
    if ad.action is None: ad.action = bpy.data.actions.new(idb.name + "_act")
    fc = ad.action.fcurve_ensure_for_datablock(idb, path, index=index)
    pairs = sorted(pairs, key=lambda x: x[0])
    kp = fc.keyframe_points; n0 = len(kp); kp.add(len(pairs))
    co = np.array(pairs, float).ravel()
    if n0 == 0: kp.foreach_set("co", co)
    else:
        for j, (f, v) in enumerate(pairs): kp[n0 + j].co = (f, v)
    for k in kp: k.interpolation = "LINEAR"
    fc.update()


def dense(idb, path, frames, values, index=0):
    ad = idb.animation_data or idb.animation_data_create()
    if ad.action is None: ad.action = bpy.data.actions.new(idb.name + "_act")
    fc = ad.action.fcurve_ensure_for_datablock(idb, path, index=index)
    kp = fc.keyframe_points; kp.add(len(values))
    co = np.empty(2 * len(values)); co[0::2] = frames; co[1::2] = values
    kp.foreach_set("co", co)
    try: kp.foreach_set("interpolation", [1] * len(values))     # 1 = LINEAR
    except Exception:
        for k in kp: k.interpolation = "LINEAR"
    fc.update()


def vpath(node): return f'nodes["{node.name}"].outputs[0].default_value'


class NB:
    def __init__(s, name, blended=False):
        s.mat = bpy.data.materials.new(name)
        try: s.mat.use_nodes = True
        except Exception: pass
        s.nt = s.mat.node_tree; s.nt.nodes.clear(); s.N = s.nt.nodes; s.L = s.nt.links
        s.mat.surface_render_method = "BLENDED" if blended else "DITHERED"
        if hasattr(s.mat, "use_transparent_shadow"): s.mat.use_transparent_shadow = True

    def n(s, t, **kw):
        x = s.N.new(t)
        for k, v in kw.items(): setattr(x, k, v)
        return x

    def put(s, sock, v):
        if isinstance(v, bpy.types.NodeSocket): s.L.new(v, sock)
        elif isinstance(v, (tuple, list)):
            v = tuple(v)
            if len(v) == 3 and len(sock.default_value) == 4: v = v + (1.0,)
            sock.default_value = v
        else: sock.default_value = v

    def m(s, op, a, b=None, c=None, clamp_=False):
        x = s.n("ShaderNodeMath", operation=op, use_clamp=clamp_); s.put(x.inputs[0], a)
        if b is not None: s.put(x.inputs[1], b)
        if c is not None: s.put(x.inputs[2], c)
        return x.outputs[0]

    def val(s, v, name):
        x = s.n("ShaderNodeValue"); x.name = name; x.outputs[0].default_value = v; return x

    def mix(s, fac, a, b):
        x = s.n("ShaderNodeMix", data_type="RGBA"); ins = {i.identifier: i for i in x.inputs}
        s.put(ins["Factor_Float"], fac); s.put(ins["A_Color"], a); s.put(ins["B_Color"], b)
        return [o for o in x.outputs if o.identifier == "Result_Color"][0]

    def mr(s, v, a, b, c, d, smooth=True):
        x = s.n("ShaderNodeMapRange", clamp=True, interpolation_type="SMOOTHSTEP" if smooth else "LINEAR")
        for nm, val in (("Value", v), ("From Min", a), ("From Max", b), ("To Min", c), ("To Max", d)):
            s.put(x.inputs[nm], val)
        return x.outputs["Result"]

    def glossy(s, rough):
        try: g = s.n("ShaderNodeBsdfGlossy")
        except Exception: g = s.n("ShaderNodeBsdfAnisotropic")
        g.inputs["Roughness"].default_value = rough; return g

    def glass_add(s, emission_shader, tint=(0.93, 0.95, 1.0), fres=0.12, rough=0.04):
        tr = s.n("ShaderNodeBsdfTransparent"); s.put(tr.inputs["Color"], tint)
        gl = s.glossy(rough)
        fr = s.n("ShaderNodeLayerWeight"); fr.inputs["Blend"].default_value = fres
        mx = s.n("ShaderNodeMixShader")
        s.L.new(fr.outputs["Fresnel"], mx.inputs[0]); s.L.new(tr.outputs[0], mx.inputs[1]); s.L.new(gl.outputs[0], mx.inputs[2])
        ad = s.n("ShaderNodeAddShader"); s.L.new(mx.outputs[0], ad.inputs[0]); s.L.new(emission_shader, ad.inputs[1])
        o = s.n("ShaderNodeOutputMaterial"); s.L.new(ad.outputs[0], o.inputs["Surface"])


def mat_tube(name, cA, cB, cC, L, strength, twist=0.0, space=0.45, shift=0.3):
    s = NB(name, blended=True)
    uv = s.n("ShaderNodeUVMap"); sp = s.n("ShaderNodeSeparateXYZ"); s.L.new(uv.outputs["UV"], sp.inputs[0])
    u, v = sp.outputs["X"], sp.outputs["Y"]
    prog = s.val(-0.2, "PROG"); boost = s.val(0.0, "BOOST")
    grad = s.mix(u, cA, cB)
    lit = s.mr(u, s.m("SUBTRACT", prog.outputs[0], 0.035), prog.outputs[0], 1.0, 0.0)   # 1 behind marble
    head = s.mr(s.m("MULTIPLY", s.m("ABSOLUTE", s.m("SUBTRACT", u, prog.outputs[0])), L), 0.0, 0.8, 1.0, 0.0)
    col = s.mix(s.m("MULTIPLY", lit, shift), grad, cC)
    ph = s.m("MULTIPLY_ADD", u, L / space, s.m("MULTIPLY", v, twist))
    tri = s.m("ABSOLUTE", s.m("SUBTRACT", s.m("FRACT", ph), 0.5))
    stripe = s.mr(tri, 0.40, 0.5, 0.0, 1.0)
    lw = s.n("ShaderNodeLayerWeight"); lw.inputs["Blend"].default_value = 0.35
    rim = s.m("POWER", lw.outputs["Facing"], 2.2)
    e = s.m("ADD", s.m("MULTIPLY_ADD", rim, 1.7, 0.10), stripe)
    gain = s.m("ADD", s.m("ADD", s.m("MULTIPLY_ADD", lit, 1.4, 1.0), boost.outputs[0]), s.m("MULTIPLY", head, 4.0))
    st = s.m("MULTIPLY", s.m("MULTIPLY", e, gain), strength)
    em = s.n("ShaderNodeEmission"); s.put(em.inputs["Color"], col); s.put(em.inputs["Strength"], st)
    s.glass_add(em.outputs[0])
    return s.mat


def mat_crystal(name, col, strength=0.8):
    s = NB(name, blended=True); stv = s.val(strength, "STR")
    lw = s.n("ShaderNodeLayerWeight"); lw.inputs["Blend"].default_value = 0.5
    st = s.m("MULTIPLY", s.m("ADD", s.m("POWER", lw.outputs["Facing"], 5.0), 0.01), stv.outputs[0])
    em = s.n("ShaderNodeEmission"); s.put(em.inputs["Color"], col); s.put(em.inputs["Strength"], st)
    s.glass_add(em.outputs[0], tint=lerpc(col, (1, 1, 1), 0.85), fres=0.12, rough=0.02)
    return s.mat


def mat_emit(name, col, strength):
    s = NB(name); stv = s.val(strength, "STR")
    em = s.n("ShaderNodeEmission"); s.put(em.inputs["Color"], col); s.put(em.inputs["Strength"], stv.outputs[0])
    o = s.n("ShaderNodeOutputMaterial"); s.L.new(em.outputs[0], o.inputs["Surface"])
    return s.mat


def mat_mirror(name):
    s = NB(name); p = s.n("ShaderNodeBsdfPrincipled")
    s.put(p.inputs["Base Color"], (0.85, 0.9, 1.0)); p.inputs["Metallic"].default_value = 1.0
    p.inputs["Roughness"].default_value = 0.02
    o = s.n("ShaderNodeOutputMaterial"); s.L.new(p.outputs[0], o.inputs["Surface"])
    return s.mat


def ramp(s, src, stops):
    r = s.n("ShaderNodeValToRGB"); el = r.color_ramp.elements
    while len(el) < len(stops): el.new(0.5)
    for e_, (pos, c) in zip(el, stops): e_.position = pos; e_.color = tuple(c) + (1.0,)
    s.L.new(src, r.inputs[0]); return r.outputs[0]


def mat_marble():
    s = NB("Marble"); p = s.n("ShaderNodeBsdfPrincipled")
    s.put(p.inputs["Base Color"], (0.95, 0.97, 1.0)); p.inputs["Roughness"].default_value = 0.0
    p.inputs["IOR"].default_value = 1.52; p.inputs["Transmission Weight"].default_value = 1.0
    if "Thin Film Thickness" in p.inputs:
        p.inputs["Thin Film Thickness"].default_value = 480; p.inputs["Thin Film IOR"].default_value = 1.35
    lw = s.n("ShaderNodeLayerWeight"); lw.inputs["Blend"].default_value = 0.4
    c = ramp(s, lw.outputs["Facing"], [(0.0, (0.0, 0.0, 0.0)), (0.45, (0.1, 0.6, 1.0)), (0.7, (0.8, 0.1, 1.0)), (1.0, (1.0, 0.6, 0.2))])
    s.L.new(c, p.inputs["Emission Color"]); p.inputs["Emission Strength"].default_value = 0.8
    o = s.n("ShaderNodeOutputMaterial"); s.L.new(p.outputs[0], o.inputs["Surface"])
    m = s.mat
    if hasattr(m, "use_raytrace_refraction"): m.use_raytrace_refraction = True
    if hasattr(m, "thickness_mode"): m.thickness_mode = "SPHERE"
    return m


def mat_core():
    s = NB("Core"); tc = s.n("ShaderNodeTexCoord"); nz = s.n("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = 3.5; nz.inputs["Detail"].default_value = 4
    s.L.new(tc.outputs["Object"], nz.inputs["Vector"])
    c = ramp(s, nz.outputs["Fac"], [(0.3, (0.2, 0.9, 1.0)), (0.5, (1.0, 1.0, 1.0)), (0.7, (1.0, 0.2, 0.8))])
    em = s.n("ShaderNodeEmission"); s.L.new(c, em.inputs["Color"]); em.inputs["Strength"].default_value = 3.0
    o = s.n("ShaderNodeOutputMaterial"); s.L.new(em.outputs[0], o.inputs["Surface"])
    return s.mat


def mat_dust():
    s = NB("Dust"); oi = s.n("ShaderNodeObjectInfo")
    c = ramp(s, oi.outputs["Random"], [(0.0, (0.15, 0.4, 1.0)), (0.4, (0.5, 0.2, 1.0)), (0.75, (0.3, 0.9, 1.0)), (1.0, (1.0, 0.75, 0.35))])
    em = s.n("ShaderNodeEmission"); s.L.new(c, em.inputs["Color"])
    s.L.new(s.m("MULTIPLY_ADD", oi.outputs["Random"], 5.0, 1.5), em.inputs["Strength"])
    o = s.n("ShaderNodeOutputMaterial"); s.L.new(em.outputs[0], o.inputs["Surface"])
    return s.mat


def mat_floor():
    s = NB("ChamberFloor"); p = s.n("ShaderNodeBsdfPrincipled")
    s.put(p.inputs["Base Color"], (0.01, 0.012, 0.02)); p.inputs["Metallic"].default_value = 0.7
    p.inputs["Roughness"].default_value = 0.12
    wr = s.val(0.0, "WR"); ws = s.val(0.0, "WS"); base = s.val(0.15, "BASE")
    tc = s.n("ShaderNodeTexCoord"); sp = s.n("ShaderNodeSeparateXYZ"); s.L.new(tc.outputs["Object"], sp.inputs[0])
    r = s.m("SQRT", s.m("ADD", s.m("MULTIPLY", sp.outputs["X"], sp.outputs["X"]), s.m("MULTIPLY", sp.outputs["Y"], sp.outputs["Y"])))
    band = s.m("EXPONENT", s.m("MULTIPLY", s.m("POWER", s.m("DIVIDE", s.m("SUBTRACT", r, wr.outputs[0]), 0.35), 2.0), -1.0))
    rings = s.mr(s.m("ABSOLUTE", s.m("SUBTRACT", s.m("FRACT", s.m("MULTIPLY", r, 1.25)), 0.5)), 0.44, 0.5, 0.0, 1.0)
    st = s.m("ADD", s.m("MULTIPLY", band, ws.outputs[0]), s.m("MULTIPLY", rings, base.outputs[0]))
    s.put(p.inputs["Emission Color"], (1.0, 0.78, 0.4)); s.L.new(st, p.inputs["Emission Strength"])
    o = s.n("ShaderNodeOutputMaterial"); s.L.new(p.outputs[0], o.inputs["Surface"])
    return s.mat


def node(mat, name): return mat.node_tree.nodes[name]


# ---------------------------------------------------------------- geometry
def frames_along(P):
    T = np.gradient(P, axis=0); T /= np.linalg.norm(T, axis=1)[:, None] + 1e-9
    n = np.cross(T[0], [0, 0, 1.0])
    if np.linalg.norm(n) < 1e-3: n = np.cross(T[0], [1.0, 0, 0])
    n /= np.linalg.norm(n); Nn = np.zeros_like(P); Nn[0] = n
    for i in range(1, len(P)):
        n = Nn[i - 1] - np.dot(Nn[i - 1], T[i]) * T[i]; n /= np.linalg.norm(n) + 1e-9; Nn[i] = n
    return T, Nn, np.cross(T, Nn)


def mesh_from(name, V, F, UV=None):
    me = bpy.data.meshes.new(name); me.from_pydata(np.asarray(V).tolist(), [], np.asarray(F).tolist())
    if UV is not None:
        uvl = me.uv_layers.new(name="UVMap"); li = np.empty(len(me.loops), dtype=np.int32)
        me.loops.foreach_get("vertex_index", li); uvl.data.foreach_set("uv", np.asarray(UV)[li].ravel())
    try: me.shade_smooth()
    except Exception: me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    return me


def tube_mesh(name, P, rad_fn, nseg=24):
    P = np.asarray(P, float)
    S = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]); L = max(S[-1], 1e-3)
    T, Nn, B = frames_along(P)
    ang = np.linspace(0, 2 * np.pi, nseg + 1); r = rad_fn(S, L)
    V = P[:, None, :] + r[:, None, None] * (np.cos(ang)[None, :, None] * Nn[:, None, :] + np.sin(ang)[None, :, None] * B[:, None, :])
    idx = np.arange(len(P) * (nseg + 1)).reshape(len(P), nseg + 1)
    F = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], axis=-1).reshape(-1, 4)
    UV = np.stack(np.broadcast_arrays((S / L)[:, None], (ang / (2 * np.pi))[None, :]), axis=-1).reshape(-1, 2)
    return mesh_from(name, V.reshape(-1, 3), F, UV), L


def bm_to_mesh(bm, name):
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free(); return me


def crystal_mesh(name, r, h, tip=0.6, both=True):
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=False, segments=6, radius1=r, radius2=r, depth=h)
    bmesh.ops.create_cone(bm, cap_ends=False, segments=6, radius1=r, radius2=0.0, depth=r * tip * 2,
                          matrix=Matrix.Translation((0, 0, h / 2 + r * tip)))
    if both:
        bmesh.ops.create_cone(bm, cap_ends=False, segments=6, radius1=0.0, radius2=r, depth=r * tip * 2,
                              matrix=Matrix.Translation((0, 0, -h / 2 - r * tip)))
    return bm_to_mesh(bm, name)


def torus_mesh(name, R, r, nu=64, nv=10):
    a = np.linspace(0, 2 * np.pi, nu, endpoint=False); b_ = np.linspace(0, 2 * np.pi, nv, endpoint=False)
    A_, B_ = np.meshgrid(a, b_, indexing="ij")
    V = np.stack([(R + r * np.cos(B_)) * np.cos(A_), (R + r * np.cos(B_)) * np.sin(A_), r * np.sin(B_)], -1).reshape(-1, 3)
    i = np.arange(nu)[:, None]; j = np.arange(nv)[None, :]
    F = np.stack([i * nv + j, ((i + 1) % nu) * nv + j, ((i + 1) % nu) * nv + (j + 1) % nv, i * nv + (j + 1) % nv], -1).reshape(-1, 4)
    return mesh_from(name, V, F)


def solid_mesh(name, kind, size):
    bm = bmesh.new()
    if kind == "ico": bmesh.ops.create_icosphere(bm, subdivisions=1 if size > 1.0 else 0, radius=size)
    elif kind == "cube": bmesh.ops.create_cube(bm, size=size * 1.3)
    else: bmesh.ops.create_cone(bm, cap_ends=True, segments=4, radius1=size, radius2=0, depth=size * 1.4,
                                matrix=Matrix.Translation((0, 0, size * 0.7)))
    return bm_to_mesh(bm, name)


def align_z(t):
    return Vector(t).normalized().to_track_quat("Z", "Y").to_matrix().to_4x4()


# ================================================================ BUILD
b, C_ch, V_IN, IMPACTS = build_path()
P = np.array([tuple(p) for p in b.P]); T = np.array(b.T); V = np.array(b.V); TID = np.array(b.TID)
N = len(P)
S_scale = (A.duration - FINALE) / IMPACTS[0]
Tv = T * S_scale
NF = int(round(A.duration * A.fps))
FR = np.arange(1, NF + 1); TAU = (FR - 1) / A.fps
SARC = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
print(f"[path] {N} pts, length {SARC[-1]:.0f} m, physical {IMPACTS[0]:.1f}s -> video {A.duration:.0f}s "
      f"(time scale x{S_scale:.2f}), drop {P[0,2]-P[-1,2]:.0f} m, vmax {V.max():.1f} m/s")
def vf(t): return 1 + t * A.fps                      # video seconds -> frame

# ---- marble contact offset (apparent gravity incl. centripetal term -> banking in curves, top-wall in loops)
tang = np.gradient(P, axis=0); dsn = np.linalg.norm(tang, axis=1)[:, None] + 1e-9; tang = gsmooth(tang / dsn, 2)
tang /= np.linalg.norm(tang, axis=1)[:, None] + 1e-9
kn = np.gradient(tang, axis=0) / dsn
app = np.array([0, 0, -G]) - (V ** 2)[:, None] * kn
press = app - np.sum(app * tang, axis=1)[:, None] * tang
press /= np.linalg.norm(press, axis=1)[:, None] + 1e-9
inT = (TID >= 0).astype(float)
press[TID == -2] = [0, 0, -1]
OFF = gsmooth(press * ((R_T - R_M) * inT)[:, None], 3)
PM = P + OFF

fi = np.interp(TAU, Tv, np.arange(N)); i0 = np.clip(np.floor(fi).astype(int), 0, N - 2); w = (fi - i0)[:, None]
MPOS = PM[i0] * (1 - w) + PM[i0 + 1] * w
MPRESS = press[i0] * (1 - w) + press[i0 + 1] * w
FRAME_TID = TID[i0]

coll_pts, coll_rad = [], []          # obstacles for camera collision avoidance

# ---------------------------------------------------------------- tubes
def stage_cols(x):
    h = x * (len(PAL) - 1)
    return pal(h), pal(h + 0.8), PAL[(int(h) + 3) % len(PAL)]

def prog_keys(mat, ia, ib, boost=True):
    sa, sb = SARC[ia], SARC[ib]; fe, fx = vf(Tv[ia]), vf(Tv[ib])
    pk = [(fe - 1, -0.2)] + [(vf(Tv[j]), (SARC[j] - sa) / max(sb - sa, 1e-6))
                             for j in np.linspace(ia, ib, 8).astype(int)] + [(fx + 1, 1.2)]
    keys(mat.node_tree, vpath(node(mat, "PROG")), pk)
    if boost:
        keys(mat.node_tree, vpath(node(mat, "BOOST")), [(fe - 3, 0.0), (fe + 2, 2.0), (fx + 0.6 * A.fps, 0.35)])
    return fe, fx

def flare(S, L):
    return R_T * (1 + 0.8 * np.exp(-(S / 0.3) ** 2) + 0.15 * np.exp(-((L - S) / 0.2) ** 2))

mat_wirebase = {}
for r in b.tubes:
    i, x = r["id"], r["x"]
    ia, ib = r["i0"], r["i1"]
    cA, cB, cC = stage_cols(x)
    strength = 1.0 + 1.5 * x + (1.0 if i >= 70 else 0.0)
    rot_from = r["deco"].get("rot_from")
    segs = [(ia, rot_from, False), (rot_from, ib, True)] if rot_from else [(ia, ib, False)]
    for sk, (sa, sb, rot) in enumerate(segs):
        pts = P[sa:sb + 1:2] if (sb - sa) > 4 else P[sa:sb + 1]
        L = SARC[sb] - SARC[sa]
        mat = mat_tube(f"Tube{i:02d}_{sk}", cA, cB, cC, L, strength, twist=(3.0 if rot else 0.0),
                       space=(0.8 if rot else 0.9), shift=r["shift"])
        if rot:   # straight part built in local space, spun around its own axis
            d = Vector(P[sb] - P[sa]).normalized()
            loc = np.array([[k * 0.0 for k in range(3)]])
            lp = np.stack([np.linspace(0, L, max(8, int(L / 0.1))), np.zeros(max(8, int(L / 0.1))), np.zeros(max(8, int(L / 0.1)))], 1)
            me, _ = tube_mesh(f"TubeM{i:02d}_{sk}", lp, lambda S_, L_: np.full_like(S_, R_T) * (1 + 0.15 * np.exp(-((L_ - S_) / 0.2) ** 2)))
            piv = link(bpy.data.objects.new(f"TubePivot{i:02d}", None))
            zt = Vector((0, 0, 1)) - d * d.z; zt = zt.normalized() if zt.length > 1e-3 else Vector((0, 1, 0))
            yt = zt.cross(d)
            piv.matrix_world = Matrix(((d.x, yt.x, zt.x, P[sa][0]), (d.y, yt.y, zt.y, P[sa][1]),
                                       (d.z, yt.z, zt.z, P[sa][2]), (0, 0, 0, 1)))
            ob = link(bpy.data.objects.new(f"Tube{i:02d}_{sk}", me)); ob.parent = piv
            me.materials.append(mat)
            spin = rng.choice([-1, 1]) * 0.9
            keys(ob, "rotation_euler", [(1, 0.0), (NF, spin * A.duration)], index=0)
        else:
            me, _ = tube_mesh(f"TubeM{i:02d}_{sk}", pts, flare if sk == 0 else (lambda S_, L_: np.full_like(S_, R_T)))
            me.materials.append(mat); link(bpy.data.objects.new(f"Tube{i:02d}_{sk}", me))
        fe, fx = prog_keys(mat, sa, sb)
        r.setdefault("mats", []).append(mat); r["fe"], r["fx"] = (r.get("fe", fe), fx)
    for j in range(ia, ib, 3): coll_pts.append(P[j]); coll_rad.append(0.6)

    deco = r["deco"]; tm = P[(ia + ib) // 2]; th = math.atan2(tm[1], tm[0])
    o_dir = Vector((math.cos(th), math.sin(th), 0)); f_dir = Vector((-math.sin(th), math.cos(th), 0))
    # split pathways
    if "split" in deco:
        pts = P[ia:ib + 1:2]; Tn, Nn, Bn = frames_along(np.asarray(pts, float))
        u = np.linspace(0, 1, len(pts)); nb_ = deco["split"]
        for k in range(nb_):
            phi = 2 * math.pi * k / nb_ + 0.4
            hor = np.cross(Tn, [0, 0, 1.0]); hor /= np.linalg.norm(hor, axis=1)[:, None] + 1e-9
            dirk = math.cos(phi) * hor + math.sin(phi) * np.array([0, 0, 1.0])
            amp = rng.uniform(0.8, 1.3) * (1 + 0.5 * x)
            bp = np.asarray(pts) + dirk * (amp * np.sin(np.pi * u) ** 1.5)[:, None]
            me, _ = tube_mesh(f"Branch{i:02d}_{k}", bp, lambda S_, L_: np.full_like(S_, R_T * 0.8))
            me.materials.append(r["mats"][0]); link(bpy.data.objects.new(f"Branch{i:02d}_{k}", me))
            for j in range(0, len(bp), 3): coll_pts.append(bp[j]); coll_rad.append(0.5)
    # crystal pillar for helices
    if "pillar" in deco:
        cx, cy, zlo, zhi, Rh, grand = deco["pillar"]
        pr = (0.55 if grand else 0.35) * Rh; h = zhi - zlo
        me = crystal_mesh(f"Pillar{i:02d}", pr, h, tip=1.2)
        pm = mat_crystal(f"PillarMat{i:02d}", cA, 0.6 + 0.8 * x)
        me.materials.append(pm)
        ob = link(bpy.data.objects.new(f"Pillar{i:02d}", me)); ob.location = (cx, cy, (zlo + zhi) / 2)
        keys(ob, "rotation_euler", [(1, 0.0), (NF, 0.05 * A.duration)], index=2)
        keys(pm.node_tree, vpath(node(pm, "STR")), [(r["fe"] - 5, 0.6 + 0.8 * x), (r["fe"] + 20, 1.8), (r["fx"] + 40, 1.0)])
        for z in np.arange(zlo, zhi, 0.5): coll_pts.append(np.array([cx, cy, z])); coll_rad.append(pr + 0.6)
        if grand:   # satellite crystals
            for k in range(7):
                a = 2 * math.pi * k / 7 + rng.random(); rr = Rh + rng.uniform(1.2, 2.2)
                sm = crystal_mesh(f"Sat{i:02d}_{k}", rng.uniform(0.15, 0.35), rng.uniform(0.8, 2.2))
                sm.materials.append(pm)
                so = link(bpy.data.objects.new(f"Sat{i:02d}_{k}", sm))
                so.location = (cx + rr * math.cos(a), cy + rr * math.sin(a), rng.uniform(zlo, zhi))
                so.rotation_euler = (rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), 0)
    # glowing gates
    if "gates" in deco:
        ng = deco["gates"]; Rg = 0.5 if x < 0.8 else 0.75
        tme = torus_mesh(f"Gate{i:02d}", Rg, 0.03)
        for k in range(ng):
            j = int(ia + (ib - ia) * (k + 0.5) / ng); t_ = P[min(j + 1, N - 1)] - P[max(j - 1, 0)]
            gm = mat_emit(f"GateMat{i:02d}_{k}", cC, 1.2)
            ob = link(bpy.data.objects.new(f"Gate{i:02d}_{k}", tme.copy())); ob.data.materials.append(gm)
            ob.matrix_world = Matrix.Translation(P[j]) @ align_z(t_)
            fj = vf(Tv[j])
            keys(gm.node_tree, vpath(node(gm, "STR")), [(fj - 4, 1.2), (fj, 14.0), (fj + 30, 3.0)])
    # rotating mirror prism (reflections / optical illusion)
    if "prism" in deco:
        j = (ia + ib) // 2; t_ = Vector(P[j + 1] - P[j - 1])
        piv = link(bpy.data.objects.new(f"Prism{i:02d}", None)); piv.matrix_world = Matrix.Translation(P[j]) @ align_z(t_)
        mm = mat_mirror(f"Mirror{i:02d}")
        for k in range(6):
            bm = bmesh.new(); bmesh.ops.create_cube(bm, size=1.0); me = bm_to_mesh(bm, f"MirP{i:02d}_{k}")
            me.materials.append(mm); ob = link(bpy.data.objects.new(f"MirP{i:02d}_{k}", me)); ob.parent = piv
            a = 2 * math.pi * k / 6
            ob.location = (1.0 * math.cos(a), 1.0 * math.sin(a), 0); ob.rotation_euler = (0, 0, a)
            ob.scale = (0.02, 0.75, 2.6)
            coll_pts.append(np.array(P[j]) + np.array(ob.location)); coll_rad.append(1.0)
        keys(piv, "rotation_euler", [(1, 0.0), (NF, 0.3 * A.duration)], index=2)
    # floating geometric sculptures
    nsc = (1 if rng.random() < 0.3 else 0) if x < 0.3 else ((rng.randint(1, 2) if rng.random() < 0.5 else 0) if i < 70 else rng.randint(3, 5))
    for k in range(nsc):
        size = rng.uniform(0.3, 0.7) if i < 70 else rng.uniform(0.6, 1.6)
        dist = rng.uniform(1.6, 3.2) if i < 70 else rng.uniform(2.5, 4.8)
        kind = rng.choice(["ico", "cube", "oct"])
        me = solid_mesh(f"Sculpt{i:02d}_{k}", kind, size)
        key_ = (int(x * 6), kind)
        if key_ not in mat_wirebase: mat_wirebase[key_] = mat_emit(f"Wire{key_[0]}{kind}", lerpc(cB, (1, 1, 1), 0.25), 2.5 + 2 * x)
        me.materials.append(mat_wirebase[key_])
        ob = link(bpy.data.objects.new(f"Sculpt{i:02d}_{k}", me))
        wm = ob.modifiers.new("wire", "WIREFRAME"); wm.thickness = 0.025 + 0.015 * size; wm.use_replace = True
        side = rng.choice([-1, 1]) if i < 70 else 1
        pos = Vector(tm) + o_dir * side * dist + Vector((0, 0, rng.uniform(-0.8, 2.5))) + f_dir * rng.uniform(-1.5, 1.5)
        ob.location = pos; ob.rotation_euler = (rng.random() * 6, rng.random() * 6, rng.random() * 6)
        for ax in (0, 2):
            keys(ob, "rotation_euler", [(1, ob.rotation_euler[ax]), (NF, ob.rotation_euler[ax] + rng.uniform(0.15, 0.5) * A.duration)], index=ax)
        coll_pts.append(np.array(pos)); coll_rad.append(size + 0.6)

print(f"[tubes] built in {time.time()-T0:.0f}s")

# ---------------------------------------------------------------- crystal chamber
FI = [vf(t * S_scale) for t in IMPACTS]; ti = FI[0]
zf = C_ch.z - R_M - 0.25
Cc = (C_ch.x, C_ch.y)
bm = bmesh.new(); bmesh.ops.create_cone(bm, cap_ends=True, segments=96, radius1=6.4, radius2=6.4, depth=0.3,
                                         matrix=Matrix.Translation((0, 0, -0.15)))
me = bm_to_mesh(bm, "Floor"); fm = mat_floor(); me.materials.append(fm)
ob = link(bpy.data.objects.new("Floor", me)); ob.location = (Cc[0], Cc[1], zf)
keys(fm.node_tree, vpath(node(fm, "WR")), [(ti, 0.0), (ti + 3.0 * A.fps, 9.6)])
keys(fm.node_tree, vpath(node(fm, "WS")), [(ti - 1, 0.0), (ti, 9.0), (ti + 3.0 * A.fps, 0.0)])
keys(fm.node_tree, vpath(node(fm, "BASE")), [(ti - 1, 0.15), (ti + 2.0 * A.fps, 0.7)])
bm = bmesh.new(); bmesh.ops.create_cone(bm, cap_ends=True, segments=6, radius1=1.0, radius2=0.85, depth=0.25,
                                         matrix=Matrix.Translation((0, 0, 0.125)))
pm = mat_crystal("PedestalMat", (1.0, 0.8, 0.45), 1.0); me = bm_to_mesh(bm, "Pedestal"); me.materials.append(pm)
ob = link(bpy.data.objects.new("Pedestal", me)); ob.location = (Cc[0], Cc[1], zf)
pk = [(ti - 1, 1.0)]
for k, f in enumerate(FI[:6]): pk += [(f, 8.0 / (1 + k)), (f + 12, 1.5 + 0.4 * (k == 0))]
keys(pm.node_tree, vpath(node(pm, "STR")), [(f, v) for f, v in sorted(dict(pk).items())])
# geodesic dome: wire + faint crystal faces
bm = bmesh.new(); bmesh.ops.create_icosphere(bm, subdivisions=3, radius=6.5)
bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.co.z < -0.05], context="VERTS")
dome = bm_to_mesh(bm, "Dome")
dw = mat_emit("DomeWire", (1.0, 0.82, 0.5), 0.8)
ow = link(bpy.data.objects.new("DomeWire", dome)); ow.location = (Cc[0], Cc[1], zf)
wm = ow.modifiers.new("wire", "WIREFRAME"); wm.thickness = 0.03; wm.use_replace = True
ow.data.materials.append(dw)
dfm = mat_crystal("DomeFaces", (0.6, 0.5, 1.0), 0.12)
of = link(bpy.data.objects.new("DomeFaces", dome.copy())); of.location = (Cc[0], Cc[1], zf)
of.data.materials.clear(); of.data.materials.append(dfm)
td = ti + 6.5 / 3.2 * A.fps
keys(dw.node_tree, vpath(node(dw, "STR")), [(ti - 1, 0.8), (td - 10, 0.9), (td, 9.0), (td + 40, 2.4)])
# crystal shards around the room
sm_mat = mat_crystal("ShardMat", (1.0, 0.72, 0.3), 0.4)
tsh = ti + 5.2 / 3.2 * A.fps
for k in range(14):
    a = 2 * math.pi * k / 14; h = rng.uniform(1.6, 4.6); rr = rng.uniform(0.18, 0.4)
    me = crystal_mesh(f"Shard{k}", rr, h, tip=1.6, both=False); me.materials.append(sm_mat)
    ob = link(bpy.data.objects.new(f"Shard{k}", me))
    ob.location = (Cc[0] + 5.2 * math.cos(a), Cc[1] + 5.2 * math.sin(a), zf + h / 2)
    ob.rotation_euler = (0.25 * -math.sin(a), 0.25 * math.cos(a), rng.random())
keys(sm_mat.node_tree, vpath(node(sm_mat, "STR")), [(ti - 1, 0.4), (tsh - 8, 0.45), (tsh, 6.5), (tsh + 45, 1.8)])
# shock-wave ring, light beam, central light
wvm = mat_emit("WaveMat", (1.0, 0.9, 0.7), 0.0)
ob = link(bpy.data.objects.new("Wave", torus_mesh("WaveM", 1.0, 0.05))); ob.data.materials.append(wvm)
ob.location = (Cc[0], Cc[1], zf + 0.3)
for ax in (0, 1, 2): keys(ob, "scale", [(ti - 1, 0.01), (ti, 0.15), (ti + 3.0 * A.fps, 9.6 if ax < 2 else 1.0)], index=ax)
keys(wvm.node_tree, vpath(node(wvm, "STR")), [(ti - 1, 0.0), (ti, 28.0), (ti + 3.0 * A.fps, 0.0)])
bm = bmesh.new(); bmesh.ops.create_cone(bm, cap_ends=False, segments=24, radius1=0.12, radius2=0.03, depth=1.0,
                                         matrix=Matrix.Translation((0, 0, 0.5)))
bmm = mat_emit("BeamMat", (0.85, 0.95, 1.0), 0.0); me = bm_to_mesh(bm, "Beam"); me.materials.append(bmm)
ob = link(bpy.data.objects.new("Beam", me)); ob.location = (Cc[0], Cc[1], zf + 0.25)
keys(ob, "scale", [(ti - 1, 0.01), (ti + 0.45 * A.fps, 16.0)], index=2)
keys(bmm.node_tree, vpath(node(bmm, "STR")), [(ti - 1, 0.0), (ti + 3, 12.0), (ti + 2.5 * A.fps, 1.0)])
cl = bpy.data.lights.new("ChamberLight", "POINT"); cl.color = (1.0, 0.85, 0.6); cl.shadow_soft_size = 0.5
ob = link(bpy.data.objects.new("ChamberLight", cl)); ob.location = (Cc[0], Cc[1], zf + 1.6)
keys(cl, "energy", [(ti - 1, 0.0), (ti + 3, 1500.0), (ti + 1.5 * A.fps, 250.0)])

# ---------------------------------------------------------------- dust particles (instanced on vertices)
zmin, zmax = float(P[:, 2].min()) - 12, float(P[:, 2].max()) + 14
dust = mat_dust()
for li, (cnt, rad) in enumerate([(3500, 0.018), (1300, 0.034), (350, 0.055)]):
    rr = np.sqrt(np.random.uniform(8 ** 2, 46 ** 2, cnt)); aa = np.random.uniform(0, 2 * np.pi, cnt)
    pts = np.stack([rr * np.cos(aa), rr * np.sin(aa), np.random.uniform(zmin, zmax, cnt)], 1)
    me = bpy.data.meshes.new(f"DustPts{li}"); me.from_pydata(pts.tolist(), [], [])
    cloud = link(bpy.data.objects.new(f"Dust{li}", me)); cloud.instance_type = "VERTS"
    cloud.show_instancer_for_render = False; cloud.show_instancer_for_viewport = False
    bm = bmesh.new(); bmesh.ops.create_icosphere(bm, subdivisions=1, radius=rad)
    sme = bm_to_mesh(bm, f"DustS{li}"); sme.materials.append(dust)
    sp = link(bpy.data.objects.new(f"DustS{li}", sme)); sp.parent = cloud
    keys(cloud, "rotation_euler", [(1, 0.0), (NF, (0.02 + 0.01 * li) * (-1) ** li * A.duration)], index=2)

# ---------------------------------------------------------------- world
W = bpy.data.worlds.new("World"); SC.world = W
try: W.use_nodes = True
except Exception: pass
wn = W.node_tree; wn.nodes.clear()
tc = wn.nodes.new("ShaderNodeTexCoord"); sep = wn.nodes.new("ShaderNodeSeparateXYZ"); wn.links.new(tc.outputs["Generated"], sep.inputs[0])
rp = wn.nodes.new("ShaderNodeValToRGB"); e0, e1 = rp.color_ramp.elements
e0.position, e0.color = 0.35, (0.010, 0.020, 0.065, 1); e1.position, e1.color = 0.85, (0.0006, 0.0009, 0.004, 1)
mrn = wn.nodes.new("ShaderNodeMapRange"); mrn.inputs["From Min"].default_value = -1; mrn.inputs["From Max"].default_value = 1
wn.links.new(sep.outputs["Z"], mrn.inputs["Value"]); wn.links.new(mrn.outputs["Result"], rp.inputs[0])
nz = wn.nodes.new("ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = 2.2; nz.inputs["Detail"].default_value = 6
wn.links.new(tc.outputs["Generated"], nz.inputs["Vector"])
nr = wn.nodes.new("ShaderNodeValToRGB"); n0, n1 = nr.color_ramp.elements
n0.position, n0.color = 0.55, (0, 0, 0, 1); n1.position, n1.color = 0.8, (0.025, 0.006, 0.05, 1)
wn.links.new(nz.outputs["Fac"], nr.inputs[0])
ad = wn.nodes.new("ShaderNodeMix"); ad.data_type = "RGBA"; ad.blend_type = "ADD"
ins = {i.identifier: i for i in ad.inputs}; ins["Factor_Float"].default_value = 1.0
wn.links.new(rp.outputs[0], ins["A_Color"]); wn.links.new(nr.outputs[0], ins["B_Color"])
bg = wn.nodes.new("ShaderNodeBackground"); wn.links.new([o for o in ad.outputs if o.identifier == "Result_Color"][0], bg.inputs["Color"])
wo = wn.nodes.new("ShaderNodeOutputWorld"); wn.links.new(bg.outputs[0], wo.inputs["Surface"])
if A.engine == "eevee":
    vol = wn.nodes.new("ShaderNodeVolumePrincipled"); vol.inputs["Density"].default_value = 0.0018
    vol.inputs["Color"].default_value = (0.5, 0.6, 1.0, 1); wn.links.new(vol.outputs[0], wo.inputs["Volume"])
W.mist_settings.start = 8; W.mist_settings.depth = 70; W.mist_settings.falloff = "QUADRATIC"

# ---------------------------------------------------------------- marble
carrier = link(bpy.data.objects.new("MarbleCarrier", None))
bm = bmesh.new(); bmesh.ops.create_uvsphere(bm, u_segments=48, v_segments=24, radius=R_M)
mm = bm_to_mesh(bm, "MarbleM"); mm.materials.append(mat_marble())
marble = link(bpy.data.objects.new("Marble", mm)); marble.parent = carrier; marble.rotation_mode = "QUATERNION"
bm = bmesh.new(); bmesh.ops.create_icosphere(bm, subdivisions=3, radius=R_M * 0.42)
cm = bm_to_mesh(bm, "CoreM"); cm.materials.append(mat_core())
core = link(bpy.data.objects.new("Core", cm)); core.parent = marble
core.scale = (1.0, 0.35, 1.0)        # flattened "cat's-eye" vane -> rolling is readable
ml = bpy.data.lights.new("MarbleLight", "POINT"); ml.energy = 12; ml.color = (0.75, 0.88, 1.0); ml.shadow_soft_size = 0.04
link(bpy.data.objects.new("MarbleLight", ml)).parent = carrier
for ax in range(3): dense(carrier, "location", FR, MPOS[:, ax], index=ax)
# rolling: omega = (n x v)/r, n = contact normal toward centre
q = Matrix.Identity(3).to_quaternion(); Q = np.zeros((NF, 4)); axis_prev = Vector((1, 0, 0))
for f in range(NF):
    vel = Vector(MPOS[min(f + 1, NF - 1)] - MPOS[f])
    if vel.length > 1e-6:
        ax = Vector(-MPRESS[f]).cross(vel)
        if FRAME_TID[f] >= 0 or FRAME_TID[f] == -2:
            if ax.length > 1e-6: axis_prev = ax.normalized()
        from mathutils import Quaternion
        q = Quaternion(axis_prev, vel.length / R_M) @ q
    Q[f] = (q.w, q.x, q.y, q.z)
    if f and np.dot(Q[f], Q[f - 1]) < 0: Q[f] = -Q[f]
for k in range(4): dense(marble, "rotation_quaternion", FR, Q[:, k], index=k)

# light trail (bevelled poly curve, revealed behind the marble)
cu = bpy.data.curves.new("Trail", "CURVE"); cu.dimensions = "3D"; cu.bevel_depth = 0.022; cu.bevel_resolution = 2
spl = cu.splines.new("POLY"); spl.points.add(NF - 1)
co4 = np.ones((NF, 4)); co4[:, :3] = MPOS; spl.points.foreach_set("co", co4.ravel())
cu.bevel_factor_mapping_start = "RESOLUTION"; cu.bevel_factor_mapping_end = "RESOLUTION"
trm = mat_emit("TrailMat", (0.75, 0.9, 1.0), 2.0); cu.materials.append(trm)
link(bpy.data.objects.new("Trail", cu))
trail_on = np.zeros(NF)
for r in b.tubes:
    if r["trail"]: trail_on[(FR >= r["fe"]) & (FR <= r["fx"] + 30)] = 1
trail_on = gsmooth(trail_on, 0.4 * A.fps)
tl_len = 18 + 40 * trail_on
endv = (FR - 1) / (NF - 1); startv = np.clip(endv - tl_len / (NF - 1), 0, 1)
dense(cu, "bevel_factor_end", FR, endv); dense(cu, "bevel_factor_start", FR, startv)
dense(trm.node_tree, vpath(node(trm, "STR")), FR, 1.6 + 5.0 * trail_on)

# ================================================================ CAMERA
MB = gsmooth(MPOS, 0.35 * A.fps)            # base (very smooth)
ML = gsmooth(MPOS, 0.06 * A.fps)            # look target (keeps marble centred)
rxy = np.linalg.norm(MB[:, :2], axis=1) + 1e-6
ODIR = np.stack([MB[:, 0] / rxy, MB[:, 1] / rxy, 0 * rxy], 1)            # outward from tower axis
FDIR = np.stack([-MB[:, 1] / rxy, MB[:, 0] / rxy, 0 * rxy], 1)           # macro travel direction

PRE = dict(intro=(7.5, 1.75, 2.4, 24), chase=(2.4, math.pi, 0.95, 28), close=(1.05, math.pi - 0.45, 0.42, 30),
           side=(3.3, math.pi / 2 + 0.25, 0.85, 40), front=(2.7, 0.25, 0.75, 30), wide=(12.0, 1.95, 6.0, 26),
           top=(2.2, 2.3, 7.0, 30), low=(1.9, 2.75, -0.45, 24))
def preset(name, tl):
    if name == "orbit": return (3.2, math.pi - 0.2 - 1.9 * tl, 1.2, 30)
    return PRE[name]

t70 = (r_ := b.tubes[70])["fe"] / A.fps
t_fin = (ti - 1) / A.fps
shots = [(0.0, "intro", 2.4)]; t = 2.4
pool_a = ["chase", "close", "side", "front", "wide", "chase", "orbit", "top", "low", "close"]
pool_b = ["orbit", "wide", "front", "close", "top", "side", "orbit", "low"]
while t < t_fin - 3.0:
    pool = pool_a if t < t70 else pool_b
    nm = rng.choice(pool)
    while nm == shots[-1][1]: nm = rng.choice(pool)
    d = rng.uniform(4.0, 6.0) if t < t70 else rng.uniform(3.4, 4.8)
    shots.append((t, nm, d)); t += d
BL = 1.6
CAM = np.zeros((NF, 3)); LENS = np.zeros(NF)
starts = np.array([s[0] for s in shots])
for f in range(NF):
    tau = TAU[f]; k = max(0, int(np.searchsorted(starts, tau, side="right")) - 1)
    s0, nm, du = shots[k]; cur = preset(nm, clamp((tau - s0) / du, 0, 1))
    if k > 0 and tau - s0 < BL:
        ps0, pnm, pdu = shots[k - 1]; prv = preset(pnm, clamp((tau - ps0) / pdu, 0, 1.3))
        wb = sstep((tau - s0) / BL)
        d_, az, h_, ln = (lerp(prv[0], cur[0], wb), prv[1] + wrap(cur[1] - prv[1]) * wb, lerp(prv[2], cur[2], wb), lerp(prv[3], cur[3], wb))
    else: d_, az, h_, ln = cur
    CAM[f] = MB[f] + d_ * (math.cos(az) * FDIR[f] + math.sin(az) * ODIR[f]) + np.array([0, 0, h_]); LENS[f] = ln
# finale: camera waits inside the chamber, in front of the incoming marble, then drifts up & back
LOOK = ML.copy()
th_in = math.atan2(V_IN.y, V_IN.x)
for f in range(NF):
    tau = TAU[f]; wF = sstep((tau - (t_fin - 3.2)) / 2.4)
    if wF <= 0: continue
    pe = sstep((tau - t_fin) / max(A.duration - t_fin, 1e-3))
    Rc, hc, th = lerp(4.4, 5.6, pe), lerp(1.2, 3.0, pe), th_in + 0.10 * max(0, tau - t_fin)
    cf = np.array([Cc[0] + Rc * math.cos(th), Cc[1] + Rc * math.sin(th), zf + hc])
    CAM[f] = CAM[f] * (1 - wF) + cf * wF; LENS[f] = lerp(LENS[f], 24, wF)
    LOOK[f] = ML[f] * (1 - 0.25 * wF) + np.array([Cc[0], Cc[1], zf + 0.6]) * 0.25 * wF
# collision avoidance (tubes, pillars, sculptures) then re-smooth
kd = KDTree(len(coll_pts))
for j, p_ in enumerate(coll_pts): kd.insert(Vector(p_), j)
kd.balance(); coll_rad = np.array(coll_rad)
for f in range(NF):
    c = Vector(CAM[f])
    for _ in range(3):
        push = Vector((0, 0, 0))
        for co, j, d in kd.find_range(c, 2.5):
            rr = coll_rad[j]
            if d < rr: push += (c - co) / max(d, 1e-3) * (rr - d)
        if push.length < 1e-4: break
        c += push * 0.6
    CAM[f] = c
CAM = gsmooth(CAM, 0.2 * A.fps); LENS = gsmooth(LENS, 0.3 * A.fps)

camd = bpy.data.cameras.new("Cam"); camd.sensor_fit = "AUTO"; camd.sensor_width = 36
camd.clip_start = 0.02; camd.clip_end = 400; camd.dof.use_dof = True
cam = link(bpy.data.objects.new("Cam", camd)); cam.rotation_mode = "QUATERNION"; SC.camera = cam
CQ = np.zeros((NF, 4))
for f in range(NF):
    dv = Vector(LOOK[f] - CAM[f]); qq = dv.to_track_quat("-Z", "Y"); CQ[f] = (qq.w, qq.x, qq.y, qq.z)
    if f and np.dot(CQ[f], CQ[f - 1]) < 0: CQ[f] = -CQ[f]
for ax in range(3): dense(cam, "location", FR, CAM[:, ax], index=ax)
for k in range(4): dense(cam, "rotation_quaternion", FR, CQ[:, k], index=k)
dist = np.linalg.norm(MPOS - CAM, axis=1)
dense(camd, "lens", FR, LENS)
dense(camd, "dof.focus_distance", FR, dist)
dense(camd, "dof.aperture_fstop", FR, 2.2 + 0.3 * dist)
print(f"[camera] {len(shots)} shots, built in {time.time()-T0:.0f}s")

# ================================================================ RENDER SETTINGS
R = SC.render
R.resolution_x, R.resolution_y, R.resolution_percentage = A.width, A.height, 100
R.fps = A.fps; SC.frame_start, SC.frame_end = 1, NF
R.use_motion_blur = not os.environ.get("FAST"); R.motion_blur_shutter = 0.45
if A.engine == "eevee":
    R.engine = "BLENDER_EEVEE"; ee = SC.eevee
    ee.taa_render_samples = A.samples; ee.use_raytracing = True
    ee.volumetric_end = 80; ee.volumetric_tile_size = "4"; ee.use_volumetric_shadows = True
else:
    R.engine = "CYCLES"; cy = SC.cycles
    cy.samples = A.samples; cy.use_denoising = True; cy.max_bounces = 6; cy.transparent_max_bounces = 32
    cy.caustics_reflective = False; cy.caustics_refractive = False; cy.use_adaptive_sampling = True
    if A.gpu:
        pr = bpy.context.preferences.addons["cycles"].preferences
        for dt in ("OPTIX", "CUDA", "HIP", "ONEAPI", "METAL"):
            try:
                pr.compute_device_type = dt; pr.get_devices()
                if any(d.type == dt for d in pr.devices): break
            except Exception: pass
        for d in pr.devices: d.use = True
        cy.device = "GPU"; print("[gpu]", pr.compute_device_type, [d.name for d in pr.devices if d.use])
SC.view_settings.view_transform = "AgX"
for lk in ("AgX - Punchy", "Punchy", "AgX - Medium High Contrast"):
    try: SC.view_settings.look = lk; break
    except Exception: pass
SC.view_layers[0].use_pass_mist = True

# compositor: atmospheric depth (mist) + bloom + subtle chromatic dispersion
try:
    g = bpy.data.node_groups.new("Comp", "CompositorNodeTree")
    g.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    rl = g.nodes.new("CompositorNodeRLayers"); out = g.nodes.new("NodeGroupOutput"); img = rl.outputs["Image"]
    try:
        mx = g.nodes.new("ShaderNodeMix"); mx.data_type = "RGBA"; ins = {i.identifier: i for i in mx.inputs}
        mf = g.nodes.new("ShaderNodeMath"); mf.operation = "MULTIPLY"; mf.inputs[1].default_value = 0.45
        g.links.new(rl.outputs["Mist"], mf.inputs[0]); g.links.new(mf.outputs[0], ins["Factor_Float"])
        g.links.new(img, ins["A_Color"]); ins["B_Color"].default_value = (0.008, 0.016, 0.05, 1)
        img = [o for o in mx.outputs if o.identifier == "Result_Color"][0]
    except Exception as e: print("[comp] mist skipped:", e)
    gl = g.nodes.new("CompositorNodeGlare")
    for v_ in ("Bloom", "BLOOM"):
        try: gl.inputs["Type"].default_value = v_; break
        except Exception: pass
    for nm, v_ in (("Threshold", 1.0), ("Strength", 0.7), ("Size", 0.7), ("Saturation", 1.0)):
        try: gl.inputs[nm].default_value = v_
        except Exception: pass
    try: gl.inputs["Quality"].default_value = "High"
    except Exception: pass
    g.links.new(img, gl.inputs["Image"]); img = gl.outputs["Image"]
    ld = g.nodes.new("CompositorNodeLensdist"); ld.inputs["Dispersion"].default_value = 0.012
    g.links.new(img, ld.inputs["Image"]); g.links.new(ld.outputs["Image"], out.inputs[0])
    SC.compositing_node_group = g
except Exception as e:
    print("[comp] compositor skipped:", e)

R.image_settings.file_format = "JPEG"; R.image_settings.quality = 94
R.use_overwrite = False; R.use_placeholder = True
out_dir = A.out if A.out.endswith("/") else A.out + "/"
print(f"[scene] ready in {time.time()-T0:.0f}s: {NF} frames @ {A.fps} fps, {len(bpy.data.objects)} objects")

if A.save_blend:
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(A.save_blend))
if not A.no_render:
    if A.frames:
        for f in [int(x) for x in A.frames.split(",")]:
            SC.frame_set(f); R.filepath = f"{out_dir}{f:05d}"; R.use_overwrite = True
            t1 = time.time(); bpy.ops.render.render(write_still=True); print(f"[render] frame {f} {time.time()-t1:.1f}s")
    else:
        SC.frame_start = A.start or 1; SC.frame_end = A.end or NF
        R.filepath = out_dir + "#####"
        bpy.ops.render.render(animation=True)
