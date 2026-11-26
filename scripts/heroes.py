"""Hero visuals (1100x560 px) for the profile cards: real renders, real signals, real data where available."""
import colorsys
import os
import sys

import numpy as np
import plotly.graph_objects as go
from PIL import Image, ImageFilter, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.environ.get("IMG_DIR", os.path.join(HERE, "img"))
OUT = os.path.join(os.path.dirname(HERE), "assets", "hero")
os.makedirs(OUT, exist_ok=True)
W, H = 1100, 560
BG = "#0b0b12"
VIO, FUC, CYA, SKY, PINK = "#a78bfa", "#d946ef", "#67e8f9", "#bae6fd", "#f0abfc"
HOME = os.path.expanduser("~")


def base(**kw):
    f = go.Figure()
    kw.setdefault("paper_bgcolor", BG); kw.setdefault("plot_bgcolor", BG)
    f.update_layout(width=W, height=H, margin=dict(l=0, r=0, t=0, b=0), showlegend=False, **kw)
    return f


def save(f, name):
    f.write_image(os.path.join(OUT, f"{name}.png"))
    print(name)


def glow_bg(color=(167, 139, 250), cx=0.5, cy=0.62, r=0.55, a=0.35):
    y, x = np.mgrid[0:H, 0:W]
    d = np.sqrt(((x / W - cx) * 1.6) ** 2 + ((y / H - cy)) ** 2) / r
    g = np.clip(1 - d, 0, 1) ** 2 * a
    bg = np.array([11, 11, 18], float)
    arr = bg + g[..., None] * (np.array(color, float) - bg)
    return Image.fromarray(arr.astype(np.uint8)).convert("RGBA")


def place(canvas, im, cx, cy, h=None, w=None):
    im = im.copy()
    if h: im = im.resize((int(im.width * h / im.height), h), Image.LANCZOS)
    if w: im = im.resize((w, int(im.height * w / im.width)), Image.LANCZOS)
    canvas.alpha_composite(im, (int(cx - im.width / 2), int(cy - im.height / 2)))


def shadow(canvas, cx, cy, rx, ry, a=150):
    sh = Image.new("L", (W, H), 0)
    from PIL import ImageDraw
    ImageDraw.Draw(sh).ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=a)
    sh = sh.filter(ImageFilter.GaussianBlur(18))
    canvas.alpha_composite(Image.merge("RGBA", (*[Image.new("L", (W, H), 0)] * 3, sh)))


def dark_recolor(path, box=None):
    """Invert a white-background figure and rotate hue 180° so colours survive on a dark card."""
    im = Image.open(path).convert("RGB")
    if box: im = im.crop(box)
    a = np.asarray(ImageOps.invert(im)).astype(float) / 255
    out = np.empty_like(a)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            h, l, s = colorsys.rgb_to_hls(*a[i, j]); out[i, j] = colorsys.hls_to_rgb((h + 0.5) % 1, l, s)
    out = out * 0.92 + np.array([11, 11, 18]) / 255 * 0.08
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).convert("RGBA")


def fit(im, pad=30):
    c = glow_bg(a=0.18)
    im = im.copy(); im.thumbnail((W - 2 * pad, H - 2 * pad), Image.LANCZOS)
    c.alpha_composite(im, ((W - im.width) // 2, (H - im.height) // 2))
    return c


# ---------------------------------------------------------------- neuro
def neurovision():
    import mne
    from stl import mesh
    raw = mne.io.read_raw_bdf(f"{HOME}/Projects/Neurovision/data/sub-hc1/ses-hc/eeg/sub-hc1_ses-hc_task-rest_eeg.bdf",
                              preload=True, verbose="error")
    raw.pick([c for c in raw.ch_names if c[:2] in ("Fz", "Cz", "Pz", "Oz", "C3", "C4", "F3", "F4", "P3", "P4")][:8])
    raw.filter(1, 30, verbose="error")
    sf = raw.info["sfreq"]; x = raw.get_data()[:, int(20 * sf):int(24 * sf)]
    t = np.arange(x.shape[1]) / sf
    f = base(xaxis=dict(domain=[0.03, 0.55], visible=False), yaxis=dict(visible=False, domain=[0.05, 0.95]))
    cols = [VIO, FUC, CYA, SKY, PINK, VIO, FUC, CYA]
    for i, ch in enumerate(raw.ch_names):
        z = x[i] / (np.std(x[i]) * 6) - i
        f.add_trace(go.Scatter(x=t, y=z, mode="lines", line=dict(color=cols[i], width=1.6)))
        f.add_annotation(x=-0.05, y=-i, text=ch, showarrow=False, xanchor="right", font=dict(color="#9ca3af", size=13))
    m = mesh.Mesh.from_file(f"{HOME}/Projects/Neurovision/brain.stl")
    v = m.vectors.reshape(-1, 3); vu, inv = np.unique(v.round(3), axis=0, return_inverse=True)
    I, J, K = inv.reshape(-1, 3).T
    if len(I) > 120000:
        s = np.linspace(0, len(I) - 1, 120000).astype(int); I, J, K = I[s], J[s], K[s]
    zc = (vu[:, 2] - vu[:, 2].min()) / np.ptp(vu[:, 2])
    f.add_trace(go.Mesh3d(x=vu[:, 0], y=vu[:, 1], z=vu[:, 2], i=I, j=J, k=K, intensity=zc,
                          colorscale=[[0, "#4c1d95"], [0.5, VIO], [1, PINK]], showscale=False,
                          lighting=dict(ambient=0.45, diffuse=0.8, specular=0.6, roughness=0.4, fresnel=0.4),
                          lightposition=dict(x=100, y=200, z=300), flatshading=False))
    f.update_layout(scene=dict(domain=dict(x=[0.5, 1.0], y=[0, 1]), xaxis=dict(visible=False),
                               yaxis=dict(visible=False), zaxis=dict(visible=False), bgcolor=BG,
                               camera=dict(eye=dict(x=-1.5, y=0.7, z=0.6)), aspectmode="data"))
    save(f, "neurovision")


def neuroimaging():
    from nilearn import datasets, image
    t1 = image.get_data(datasets.load_mni152_template(resolution=1)).astype(float)
    gm = image.get_data(datasets.load_mni152_gm_template(resolution=1)).astype(float)
    t1 /= np.percentile(t1, 99.5); gm /= gm.max()
    cx, cy, cz = [s // 2 for s in t1.shape]
    sl = [(np.rot90(t1[cx + 8]), np.rot90(gm[cx + 8])), (np.rot90(t1[:, cy - 10]), np.rot90(gm[:, cy - 10])),
          (np.rot90(t1[:, :, cz + 10]), np.rot90(gm[:, :, cz + 10]))]
    c = glow_bg(a=0.15)
    xs = [190, 550, 910]
    lut = np.array([[76, 29, 149], [167, 139, 250], [217, 70, 239], [240, 171, 252]], float)
    for (a, g), x0 in zip(sl, xs):
        a = np.clip(a, 0, 1); g = np.clip(g, 0, 1)
        rgb = np.repeat(a[..., None], 3, 2) * 230
        idx = np.clip(g * 3, 0, 2.999); lo = idx.astype(int); fr = idx - lo
        col = lut[lo] * (1 - fr[..., None]) + lut[lo + 1] * fr[..., None]
        al = (g > 0.35)[..., None] * 0.55
        rgb = rgb * (1 - al) + col * al
        im = Image.fromarray(rgb.astype(np.uint8)).convert("RGBA")
        im.putalpha(Image.fromarray(((a > 0.04) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1)))
        place(c, im, x0, H / 2, h=470 if im.height > im.width else None, w=330 if im.width >= im.height else None)
    c.save(os.path.join(OUT, "neuroimaging.png")); print("neuroimaging")


def heart():
    import wfdb
    rec = f"{HOME}/Projects/heart-pignn/mit-bih-arrhythmia-database-1.0.0/100"
    r = wfdb.rdrecord(rec, sampfrom=0, sampto=360 * 7); ann = wfdb.rdann(rec, "atr", sampfrom=0, sampto=360 * 7)
    s = r.p_signal[:, 0]; t = np.arange(len(s)) / r.fs
    f = base(xaxis=dict(visible=False, range=[0.1, 6.9]), yaxis=dict(visible=False, range=[s.min() - 0.4, s.max() + 0.35]))
    for xg in np.arange(0, 7, 0.2):
        f.add_vline(x=xg, line=dict(color="rgba(217,70,239,0.10)" if round(xg / 0.2) % 5 else "rgba(217,70,239,0.25)", width=1))
    for yg in np.arange(np.floor(s.min()) - 1, s.max() + 1, 0.1):
        f.add_hline(y=yg, line=dict(color="rgba(217,70,239,0.10)" if round(yg * 10) % 5 else "rgba(217,70,239,0.25)", width=1))
    for w, a in [(14, 0.08), (7, 0.18)]:
        f.add_trace(go.Scatter(x=t, y=s, mode="lines", line=dict(color=f"rgba(103,232,249,{a})", width=w)))
    f.add_trace(go.Scatter(x=t, y=s, mode="lines", line=dict(color=CYA, width=2.4)))
    pk = [p for p, sym in zip(ann.sample, ann.symbol) if sym in "NLRAV"]
    f.add_trace(go.Scatter(x=np.array(pk) / r.fs, y=s[pk] + 0.18, mode="markers",
                           marker=dict(color=FUC, size=10, symbol="triangle-down")))
    save(f, "heart")


def aorta():
    th = np.linspace(-0.25 * np.pi, 1.0 * np.pi, 160)
    cx = np.where(th < np.pi, 0.9 * np.cos(th), -0.9); cz = np.where(th < np.pi, 1.2 * np.sin(th), 0) + 1.2
    seg_down = np.linspace(1.2, -2.6, 90)
    C = np.vstack([np.c_[np.full(70, 0.9), np.zeros(70), np.linspace(-0.4, 1.2, 70)][::-1][::-1],
                   np.c_[0.9 * np.cos(np.linspace(0, np.pi, 120)), 0.25 * np.sin(np.linspace(0, np.pi, 120)),
                         1.2 + 1.1 * np.sin(np.linspace(0, np.pi, 120))],
                   np.c_[np.full(90, -0.9), np.zeros(90), seg_down]])
    rad = np.r_[np.full(70, 0.42), np.full(120, 0.40), np.linspace(0.36, 0.26, 90)]
    T = np.gradient(C, axis=0); T /= np.linalg.norm(T, axis=1)[:, None]
    N = np.cross(T, [0, 1, 0.001]); N /= np.linalg.norm(N, axis=1)[:, None]; B = np.cross(T, N)
    u = np.linspace(0, 2 * np.pi, 40)
    X = C[:, None, :] + rad[:, None, None] * (np.cos(u)[None, :, None] * N[:, None, :] + np.sin(u)[None, :, None] * B[:, None, :])
    f = base()
    f.add_trace(go.Surface(x=X[..., 0], y=X[..., 1], z=X[..., 2], surfacecolor=X[..., 2], showscale=False,
                           colorscale=[[0, "#831843"], [0.5, FUC], [1, PINK]],
                           lighting=dict(ambient=0.4, diffuse=0.8, specular=0.7, roughness=0.3, fresnel=0.5)))
    for a0, l0 in [(0.15, 0.55), (-0.25, 0.4), (-0.6, 0.45)]:
        p0 = np.array([0.9 * np.cos(np.pi / 2 + a0 * 2), 0, 1.2 + 1.1 * np.sin(np.pi / 2 + a0 * 2)])
        L = np.c_[np.full(30, p0[0]), np.zeros(30), np.linspace(p0[2], p0[2] + l0 + 0.5, 30)]
        f.add_trace(go.Scatter3d(x=L[:, 0], y=L[:, 1], z=L[:, 2], mode="lines", line=dict(color=FUC, width=14)))
    for zz in np.linspace(-2.3, 0.9, 5):   # measured diameters
        f.add_trace(go.Scatter3d(x=[-1.4, -0.4], y=[0, 0], z=[zz, zz], mode="lines", line=dict(color=CYA, width=3, dash="dash")))
    f.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                               bgcolor=BG, aspectmode="data", camera=dict(eye=dict(x=0.35, y=-3.2, z=0.35))))
    save(f, "trismegisto")


def stretch():
    c = glow_bg(color=(103, 232, 249), cx=0.33, a=0.25)
    im = Image.open(os.path.join(IMG, "stretch_r.png")).convert("RGBA")
    shadow(c, 330, 535, 110, 16)
    place(c, im, 330, H / 2 + 5, h=530)
    from PIL import ImageDraw
    d = ImageDraw.Draw(c)
    J = dict(head=(800, 120), neck=(790, 175), sh_l=(735, 190), sh_r=(845, 190), el_l=(705, 270), el_r=(880, 265),
             wr_l=(720, 345), wr_r=(905, 335), hip=(770, 340), hip_l=(740, 345), hip_r=(800, 345),
             kn_l=(725, 440), kn_r=(820, 440), an_l=(720, 530), an_r=(835, 530))
    J["neck"] = (775, 178); J["head"] = (760, 120)   # forward head posture
    E = [("head", "neck"), ("neck", "sh_l"), ("neck", "sh_r"), ("sh_l", "el_l"), ("el_l", "wr_l"), ("sh_r", "el_r"),
         ("el_r", "wr_r"), ("neck", "hip"), ("hip", "hip_l"), ("hip", "hip_r"), ("hip_l", "kn_l"), ("kn_l", "an_l"),
         ("hip_r", "kn_r"), ("kn_r", "an_r")]
    for a_, b_ in E:
        d.line([J[a_], J[b_]], fill=(103, 232, 249, 255), width=6)
    for k, p in J.items():
        col = (217, 70, 239, 255) if k in ("neck", "head") else (186, 230, 253, 255)
        d.ellipse((p[0] - 9, p[1] - 9, p[0] + 9, p[1] + 9), fill=col)
    d.arc((735, 150, 815, 230), 200, 280, fill=(217, 70, 239, 255), width=4)
    d.line([(470, 150), (690, 170)], fill=(217, 70, 239, 140), width=2)
    c.save(os.path.join(OUT, "stretch.png")); print("stretch")


# ---------------------------------------------------------------- robotics
def cutout_card(names, xs, hs, name, color=(167, 139, 250), ys=None):
    c = glow_bg(color=color, a=0.3)
    ys = ys or [H / 2 + 10] * len(names)
    for n, x, h, y in zip(names, xs, hs, ys):
        im = Image.open(os.path.join(IMG, f"{n}.png")).convert("RGBA")
        sc = h / im.height
        shadow(c, x, y + h / 2 - 8, im.width * sc * 0.42, 16)
        place(c, im, x, y, h=h)
    c.save(os.path.join(OUT, f"{name}.png")); print(name)


def auv():
    u, v = np.mgrid[0:2 * np.pi:60j, 0:np.pi:40j]
    L, R = 2.6, 0.38
    x = L * np.cos(v); r = R * np.sin(v) ** 0.55
    y, z = r * np.cos(u), r * np.sin(u)
    f = base()
    f.add_trace(go.Surface(x=x, y=y, z=z, surfacecolor=x, showscale=False,
                           colorscale=[[0, "#1e3a8a"], [0.6, "#38bdf8"], [1, SKY]],
                           lighting=dict(ambient=0.45, diffuse=0.8, specular=0.9, roughness=0.25, fresnel=0.4)))
    for (yy, zz) in [(0.75, 0), (-0.75, 0), (0, 0.75), (0, -0.75)]:
        f.add_trace(go.Mesh3d(x=[-2.0, -2.5, -2.55, -2.1], y=[0, yy, yy, 0], z=[0, zz, zz, 0], i=[0, 0], j=[1, 2], k=[2, 3],
                              color=VIO, opacity=0.95, flatshading=True))
    for k, (yy, zz) in enumerate([(0.55, 0.55), (-0.55, 0.55), (0.55, -0.55), (-0.55, -0.55)]):
        ok = k != 1
        f.add_trace(go.Cone(x=[-2.7], y=[yy], z=[zz], u=[-0.6], v=[0], w=[0], sizemode="absolute", sizeref=0.35,
                            colorscale=[[0, CYA if ok else "#ef4444"], [1, CYA if ok else "#ef4444"]], showscale=False))
    for s in range(25):
        p = np.random.default_rng(s).normal(size=3) * [3, 1.4, 1]
        f.add_trace(go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], mode="markers",
                                 marker=dict(size=3, color="rgba(186,230,253,0.35)")))
    f.update_layout(scene=dict(xaxis=dict(visible=False, range=[-3.4, 3]), yaxis=dict(visible=False, range=[-1.6, 1.6]),
                               zaxis=dict(visible=False, range=[-1.2, 1.2]), bgcolor="#06121f", aspectmode="manual",
                               aspectratio=dict(x=2.4, y=1.2, z=0.9), camera=dict(eye=dict(x=0.9, y=-2.1, z=0.75))),
                      paper_bgcolor="#06121f")
    save(f, "auv")


def puzzlebot():
    dt, T = 0.02, 26
    ref = lambda t: np.array([1.6 * np.sin(0.24 * t), 0.9 * np.sin(0.48 * t)])
    x = np.array([-0.8, -0.9, 0.3]); P = []
    for k in range(int(T / dt)):
        t = k * dt; r = ref(t); rd = (ref(t + dt) - r) / dt
        e = r - x[:2]; vd = rd + 1.4 * e
        th_d = np.arctan2(vd[1], vd[0]); v = np.linalg.norm(vd) * np.cos(th_d - x[2])
        w = 4.0 * np.arctan2(np.sin(th_d - x[2]), np.cos(th_d - x[2]))
        x = x + dt * np.array([v * np.cos(x[2]), v * np.sin(x[2]), w]); P.append(x.copy())
    P = np.array(P); tt = np.arange(0, T, dt); R = np.array([ref(t) for t in tt])
    f = base(xaxis=dict(visible=False, range=[-2.4, 2.4]), yaxis=dict(visible=False, range=[-1.25, 1.25], scaleanchor="x"))
    g = np.linspace(-2.4, 2.4, 25); h = np.linspace(-1.25, 1.25, 14)
    for gx in g: f.add_vline(x=gx, line=dict(color="rgba(167,139,250,0.08)", width=1))
    for gy in h: f.add_hline(y=gy, line=dict(color="rgba(167,139,250,0.08)", width=1))
    f.add_trace(go.Scatter(x=R[:, 0], y=R[:, 1], mode="lines", line=dict(color="rgba(186,230,253,0.5)", width=2, dash="dash")))
    for w_, a_ in [(12, 0.1), (6, 0.25)]:
        f.add_trace(go.Scatter(x=P[:, 0], y=P[:, 1], mode="lines", line=dict(color=f"rgba(217,70,239,{a_})", width=w_)))
    f.add_trace(go.Scatter(x=P[:, 0], y=P[:, 1], mode="lines", line=dict(color=FUC, width=2.5)))
    for k in range(0, len(P), 160):
        xx, yy, th = P[k]; c_, s_ = np.cos(th), np.sin(th)
        body = np.array([[0.13, 0], [-0.09, 0.08], [-0.09, -0.08], [0.13, 0]])
        b = body @ np.array([[c_, s_], [-s_, c_]]) + [xx, yy]
        f.add_trace(go.Scatter(x=b[:, 0], y=b[:, 1], fill="toself", mode="lines", line=dict(color=CYA, width=2),
                               fillcolor="rgba(103,232,249,0.35)"))
    save(f, "puzzlebot")


def g1_lidar():
    c = glow_bg(color=(103, 232, 249), a=0.22)
    im = Image.open(os.path.join(IMG, "g1.png")).convert("RGBA")
    rng = np.random.default_rng(3)
    from PIL import ImageDraw
    pts = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(pts)
    hx, hy = W / 2, 95
    for a in np.linspace(0, 2 * np.pi, 900):
        rr = 380 + 60 * np.sin(3 * a) + rng.normal(0, 10)
        px, py = hx + rr * np.cos(a) * 1.25, 330 + rr * np.sin(a) * 0.32
        if 0 < px < W and 0 < py < H:
            d.ellipse((px - 2, py - 2, px + 2, py + 2), fill=(103, 232, 249, 200))
    for a in np.linspace(0, 2 * np.pi, 40, endpoint=False):
        rr = 380 + 60 * np.sin(3 * a)
        d.line([(hx, hy), (hx + rr * np.cos(a) * 1.25, 330 + rr * np.sin(a) * 0.32)], fill=(103, 232, 249, 35), width=1)
    for i in range(8):
        for j in range(3):
            x0, y0 = 120 + i * 30 + j * 12, 430 + j * 22
            d.rectangle((x0, y0, x0 + 24, y0 + 16), outline=(217, 70, 239, 160), width=2)
    c.alpha_composite(pts)
    shadow(c, W / 2, 545, 70, 14)
    place(c, im, W / 2, 300, h=520)
    c.save(os.path.join(OUT, "g1.png")); print("g1")


def robocup():
    f = base(xaxis=dict(visible=False, range=[-5.2, 5.2]), yaxis=dict(visible=False, range=[-3.4, 3.4], scaleanchor="x"),
             plot_bgcolor="#062b1a", paper_bgcolor="#062b1a")
    ln = dict(color="rgba(255,255,255,0.7)", width=2)
    f.add_shape(type="rect", x0=-4.5, x1=4.5, y0=-3, y1=3, line=ln)
    f.add_shape(type="line", x0=0, x1=0, y0=-3, y1=3, line=ln)
    f.add_shape(type="circle", x0=-0.75, x1=0.75, y0=-0.75, y1=0.75, line=ln)
    for s in (-1, 1):
        f.add_shape(type="rect", x0=s * 4.5, x1=s * 3.9, y0=-1.1, y1=1.1, line=ln)
    rng = np.random.default_rng(1)
    teams = [(FUC, [(-2.2, 1.0), (-1.0, -1.4)]), (CYA, [(1.6, 0.6), (2.6, -1.2)])]
    for col, ps in teams:
        for (x0, y0) in ps:
            tr = np.cumsum(rng.normal(0, 0.12, (25, 2)), 0) + [x0 - 0.8, y0 - 0.3]
            f.add_trace(go.Scatter(x=tr[:, 0], y=tr[:, 1], mode="lines", line=dict(color=col, width=2), opacity=0.5))
            f.add_shape(type="rect", x0=tr[-1, 0] - 0.35, x1=tr[-1, 0] + 0.35, y0=tr[-1, 1] - 0.45, y1=tr[-1, 1] + 0.45,
                        line=dict(color=col, width=3))
            f.add_trace(go.Scatter(x=[tr[-1, 0]], y=[tr[-1, 1]], mode="markers",
                                   marker=dict(size=26, color=col, opacity=0.85, line=dict(color="white", width=2))))
    f.add_trace(go.Scatter(x=[0.4], y=[0.2], mode="markers", marker=dict(size=16, color="#fb923c", line=dict(color="white", width=2))))
    f.add_shape(type="circle", x0=0.15, x1=0.65, y0=-0.05, y1=0.45, line=dict(color="#fb923c", width=2, dash="dot"))
    save(f, "robocup")


def segkit():
    import mujoco
    from cutout import M
    m = mujoco.MjModel.from_xml_path(f"{M}/unitree_go2/scene.xml"); d = mujoco.MjData(m)
    if m.nkey: mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    h, w = 560, 1100
    m.vis.global_.offwidth, m.vis.global_.offheight = w, h
    r = mujoco.Renderer(m, h, w); cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(m, cam)
    cam.azimuth, cam.elevation, cam.distance = 150, -18, 1.25; cam.lookat[:] = [0, 0, 0.22]
    r.update_scene(d, cam); rgb = r.render()
    r.enable_segmentation_rendering(); r.update_scene(d, cam); seg = r.render()[..., 0]
    body = np.where(seg >= 0, m.geom_bodyid[np.clip(seg, 0, None)], -1)
    pal = np.array([[11, 11, 18], [217, 70, 239], [103, 232, 249], [167, 139, 250], [240, 171, 252], [186, 230, 253]])
    floor = {i for i in range(m.ngeom) if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE}
    lab = np.where(np.isin(seg, list(floor)) | (seg < 0), 0, 1 + (body % 5))
    mask = pal[lab]
    out = rgb.copy(); split = int(w * 0.5)
    out[:, split:] = (0.35 * rgb[:, split:] + 0.65 * mask[:, split:]).astype(np.uint8)
    out[:, split - 2:split + 2] = [255, 255, 255]
    Image.fromarray(out).save(os.path.join(OUT, "segkit.png")); print("segkit")


# ---------------------------------------------------------------- hpc
def rust_langs():
    langs = [("C", "#a8b9cc"), ("C++", "#659ad2"), ("Rust", "#f74c00"), ("Go", "#00add8"), ("Java", "#f89820"),
             ("Julia", "#9558b2"), ("Zig", "#f7a41d"), ("Swift", "#f05138"), ("Kotlin", "#a97bff")]
    f = base(xaxis=dict(visible=False, range=[0, 11]), yaxis=dict(visible=False, range=[0, 5.6]))
    f.add_shape(type="rect", x0=0.3, x1=10.7, y0=0.3, y1=5.3, fillcolor="#12121c", line=dict(color="#2a2a3a", width=2))
    for i, c in enumerate(["#ff5f56", "#ffbd2e", "#27c93f"]):
        f.add_shape(type="circle", x0=0.6 + i * 0.35, x1=0.8 + i * 0.35, y0=4.95, y1=5.15, fillcolor=c, line_width=0)
    for i, (n, c) in enumerate(langs):
        r_, k = divmod(i, 3)
        x0, y0 = 0.9 + k * 3.35, 3.7 - r_ * 1.3
        f.add_shape(type="rect", x0=x0, x1=x0 + 3.0, y0=y0, y1=y0 + 1.0, fillcolor="rgba(255,255,255,0.03)",
                    line=dict(color=c, width=3 if n == "Rust" else 1.5))
        f.add_annotation(x=x0 + 1.5, y=y0 + 0.5, text=f"<b>{n}</b>", showarrow=False,
                         font=dict(size=34 if n == "Rust" else 28, color=c, family="Ubuntu Mono"))
    save(f, "rust")


def fashion():
    from torchvision.datasets import FashionMNIST
    ds = FashionMNIST(os.path.join(HERE, "fmnist"), train=False, download=True)
    groups = {0: 0, 2: 0, 4: 0, 6: 0, 1: 1, 3: 1, 5: 2, 7: 2, 9: 2, 8: 3}   # tops · lower · footwear · bag
    gcol = np.array([[217, 70, 239], [103, 232, 249], [167, 139, 250], [240, 171, 252]])
    c = glow_bg(a=0.18)
    picks = {}
    for img, y in zip(ds.data.numpy(), ds.targets.numpy()):
        picks.setdefault(int(y), [])
        if len(picks[int(y)]) < 3: picks[int(y)].append(img)
        if all(len(v) == 3 for v in picks.values()) and len(picks) == 10: break
    order = [0, 2, 4, 6, 1, 3, 5, 7, 9, 8]
    s, gap = 92, 12
    x0 = (W - (10 * s + 9 * gap)) / 2
    for k, cls in enumerate(order):
        for r_ in range(3):
            a = picks[cls][r_].astype(float) / 255
            col = gcol[groups[cls]]
            rgb = (a[..., None] * col[None, None, :] + (1 - a[..., None]) * np.array([18, 18, 28])).astype(np.uint8)
            im = Image.fromarray(rgb).resize((s, s), Image.NEAREST).convert("RGBA")
            c.alpha_composite(im, (int(x0 + k * (s + gap)), int(95 + r_ * (s + gap + 20))))
    c.save(os.path.join(OUT, "fashion.png")); print("fashion")


def compiler():
    nodes = {"A": (0.5, 3.2), "B": (0.5, 1.4), "MatMul": (2.6, 2.3), "bias": (2.6, 0.6), "Add": (4.6, 1.6),
             "ReLU": (6.5, 1.6), "Y": (8.6, 1.6)}
    edges = [("A", "MatMul"), ("B", "MatMul"), ("MatMul", "Add"), ("bias", "Add"), ("Add", "ReLU"), ("ReLU", "Y")]
    f = base(xaxis=dict(visible=False, range=[-0.6, 9.6]), yaxis=dict(visible=False, range=[-0.6, 4.2]))
    f.add_shape(type="rect", x0=1.75, x1=7.35, y0=0.95, y1=3.0, line=dict(color=FUC, width=2, dash="dash"),
                fillcolor="rgba(217,70,239,0.08)")
    f.add_annotation(x=4.55, y=3.25, text="fused kernel", showarrow=False, font=dict(color=FUC, size=22))
    for a, b in edges:
        (x0, y0), (x1, y1) = nodes[a], nodes[b]
        f.add_annotation(x=x1 - 0.6, y=y1, ax=x0 + 0.6, ay=y0, xref="x", yref="y", axref="x", ayref="y", text="",
                         showarrow=True, arrowhead=2, arrowwidth=2.5, arrowcolor="rgba(186,230,253,0.7)")
    for n, (x, y) in nodes.items():
        op = n in ("MatMul", "Add", "ReLU")
        f.add_shape(type="rect", x0=x - 0.6, x1=x + 0.6, y0=y - 0.35, y1=y + 0.35,
                    fillcolor="rgba(167,139,250,0.25)" if op else "#12121c", line=dict(color=VIO if op else CYA, width=2))
        f.add_annotation(x=x, y=y, text=f"<b>{n}</b>", showarrow=False, font=dict(color="white", size=20))
    save(f, "compiler")


def lorenz():
    def run(x0, n=9000, dt=0.006):
        s, r, b = 10, 28, 8 / 3; P = [x0]
        x = np.array(x0, float)
        for _ in range(n):
            dx = np.array([s * (x[1] - x[0]), x[0] * (r - x[2]) - x[1], x[0] * x[1] - b * x[2]]); x = x + dt * dx; P.append(x.copy())
        return np.array(P)
    f = base()
    cols = [VIO, FUC, CYA, SKY, PINK, "#818cf8", "#22d3ee", "#e879f9", "#c4b5fd", "#a5f3fc", "#f5d0fe"]
    for i, c in enumerate(cols):
        P = run([1 + 0.01 * i, 1, 1])[3000:]
        f.add_trace(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="lines", line=dict(color=c, width=3), opacity=0.8))
        f.add_trace(go.Scatter3d(x=[P[-1, 0]], y=[P[-1, 1]], z=[P[-1, 2]], mode="markers",
                                 marker=dict(size=6, color=c, line=dict(color="white", width=1))))
    f.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                               bgcolor=BG, camera=dict(eye=dict(x=1.3, y=-1.2, z=0.35))))
    save(f, "qse3")


def recolored(src, name, box=None):
    im = dark_recolor(os.path.join(IMG, src), box)
    fit(im).save(os.path.join(OUT, f"{name}.png")); print(name)


JOBS = dict(
    neurovision=neurovision, neuroimaging=neuroimaging, heart=heart, trismegisto=aorta, stretch=stretch,
    quadruped=lambda: cutout_card(["go2"], [W / 2], [430], "quadruped", (217, 70, 239)),
    anymal=lambda: cutout_card(["anymal"], [W / 2], [470], "anymal", (240, 100, 120)),
    psyonic=lambda: cutout_card(["lite6", "hand"], [420, 760], [480, 400], "psyonic", (103, 232, 249)),
    auv=auv, puzzlebot=puzzlebot, g1=g1_lidar,
    lerobot=lambda: cutout_card(["so101"], [W / 2], [360], "lerobot", (250, 204, 21)),
    robocup=robocup, segkit=segkit,
    rust=rust_langs, fashion=fashion, compiler=compiler, qse3=lorenz,
    mcp=lambda: recolored("mcp.png", "mcp"),
    broker=lambda: recolored("broker.png", "broker", (0, 0, 640, 690)),
)

if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    sys.path.insert(0, HERE)
    for k in (sys.argv[1:] or JOBS):
        try:
            JOBS[k]()
        except Exception as e:
            import traceback; traceback.print_exc(); print("FAIL", k, e)
