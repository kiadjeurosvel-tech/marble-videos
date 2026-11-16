# Runs inside Blender (blender -b --python kr.py -- FPS SAMPLES EP): renders an episode on GPU at FPS.
import sys, os, math, time, shutil
argv = sys.argv[sys.argv.index('--') + 1:]
FPS, SAMPLES, EP = int(argv[0]), int(argv[1]), argv[2]
ROOT = '/kaggle/tmp/c/'
src = open(ROOT + EP + '/' + {'ep3': 'e3.py'}[EP]).read()
exec(src[:src.index("if __name__ == '__main__':")])
sc = bpy.context.scene
prefs = bpy.context.preferences.addons['cycles'].preferences
for dt in ('OPTIX', 'CUDA'):
    try:
        prefs.compute_device_type = dt; prefs.get_devices()
        if any(d.type == dt for d in prefs.devices): break
    except Exception as e: print('no', dt, e)
for d in prefs.devices: d.use = d.type != 'CPU'
print('DEVICES', [(d.name, d.type, d.use) for d in prefs.devices], flush=True)
sc.cycles.device = 'GPU'; sc.cycles.samples = SAMPLES
sc.cycles.diffuse_bounces = 2; sc.cycles.glossy_bounces = 2
sc.render.resolution_x, sc.render.resolution_y = 720, 1280
sc.render.image_settings.quality = 90
OUT = '/kaggle/tmp/frames/'; os.makedirs(OUT, exist_ok=True)
N = int(math.ceil(TOTAL * FPS)); t0 = time.time(); done = 0
for f in range(N):
    t = f / FPS
    if REW0 <= t < REW1: continue
    if apply(t) is None: continue
    sc.frame_current = int(t * 12)
    sc.render.filepath = OUT + '%05d.jpg' % f
    bpy.ops.render.render(write_still=True); done += 1
    if done % 50 == 0: print('FRAME', f, N, '%.2fs/frame' % ((time.time() - t0) / done), flush=True)
# rewind window: reversed earlier frames (VHS look is added later in the final edit)
import glob
for f in range(N):
    p = OUT + '%05d.jpg' % f
    if os.path.exists(p): continue
    t = f / FPS
    if REW0 <= t < REW1:
        u = (t - REW0) / (REW1 - REW0); s = int((13.9 - (13.9 - 3.4) * u ** 1.2) * FPS)
    else:
        s = f - 1
    while s > 0 and not os.path.exists(OUT + '%05d.jpg' % s): s -= 1
    shutil.copy(OUT + '%05d.jpg' % s, p)
print('RENDER_DONE', N, 'frames', '%.1f min' % ((time.time() - t0) / 60), flush=True)
