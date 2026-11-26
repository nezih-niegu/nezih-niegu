"""Render a MuJoCo model and cut the robot out (alpha from segmentation)."""
import os, re, sys, mujoco, numpy as np
from PIL import Image, ImageFilter
M = os.environ.get("MENAGERIE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "menagerie"))

def load(path):
    if path.endswith(".urdf"):
        base = os.path.dirname(os.path.dirname(path))
        s = open(path).read().replace("package://ah_urdf/", base + "/")
        s = re.sub(r'<robot name="([^"]+)">', r'<robot name="\1"><mujoco><compiler discardvisual="false" fusestatic="false"/></mujoco>', s, 1)
        return mujoco.MjModel.from_xml_string(s)
    return mujoco.MjModel.from_xml_path(path)

def render(path, out, az=135, el=-15, dist=None, lookat=None, key=0, qpos=None, H=900, W=900):
    m = load(path); d = mujoco.MjData(m)
    m.vis.global_.offwidth, m.vis.global_.offheight = W, H
    if m.nkey > key: mujoco.mj_resetDataKeyframe(m, d, key)
    if qpos:
        for j, v in qpos.items(): d.qpos[m.joint(j).qposadr[0]] = v
    mujoco.mj_forward(m, d)
    floors = {i for i in range(m.ngeom) if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE}
    for i in floors: m.geom_rgba[i, 3] = 0
    r = mujoco.Renderer(m, H, W)
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(m, cam)
    cam.azimuth, cam.elevation = az, el
    if dist: cam.distance = dist
    if lookat is not None: cam.lookat[:] = lookat
    opt = mujoco.MjvOption()
    r.update_scene(d, cam, opt); rgb = r.render()
    r.enable_segmentation_rendering(); r.update_scene(d, cam, opt); seg = r.render(); r.disable_segmentation_rendering()
    obj, typ = seg[..., 0], seg[..., 1]
    mask = (obj >= 0) & (typ == int(mujoco.mjtObj.mjOBJ_GEOM)) & ~np.isin(obj, list(floors))
    a = Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8))
    im = Image.fromarray(rgb).convert("RGBA"); im.putalpha(a)
    ys, xs = np.where(mask); pad = 10
    im = im.crop((max(xs.min() - pad, 0), max(ys.min() - pad, 0), min(xs.max() + pad, W), min(ys.max() + pad, H)))
    im.save(out); print(out, im.size)

if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    J = [
      (f"{M}/unitree_go2/scene.xml", "img/go2.png", dict(az=145, el=-12)),
      (f"{M}/unitree_g1/scene.xml", "img/g1.png", dict(az=160, el=-5)),
      (f"{M}/hello_robot_stretch_3/scene.xml", "img/stretch_r.png", dict(az=130, el=-10)),
      (f"{M}/ufactory_lite6/scene.xml", "img/lite6.png", dict(az=140, el=-15)),
      (f"{M}/robotstudio_so101/scene.xml", "img/so101.png", dict(az=130, el=-20)),
      (os.path.expanduser("~/Projects/anymal_d_ppo/anybotics_anymal_d/scene.xml"), "img/anymal.png", dict(az=135, el=-12)),
      (os.path.expanduser("~/Projects/PSYONIC/ability-hand-ros2/src/ah_urdf/urdf/ability_hand_right_large.urdf"), "img/hand.png", dict(az=200, el=-20)),
    ]
    for p, o, kw in J:
        try: render(p, o, **kw)
        except Exception as e: print("FAIL", o, repr(e)[:200])
