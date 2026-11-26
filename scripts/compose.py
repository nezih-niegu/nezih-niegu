"""Compose the per-area visual abstracts (dark, Healthen-style) from the hero visuals.
Heroes come from scripts/heroes.py (MuJoCo renders, real signals and data); project text from make_abstracts.AREAS.
Run:  python scripts/compose.py  ->  assets/area_*.png"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from make_abstracts import AREAS  # noqa: E402

HERO = os.environ.get("HERO_DIR", os.path.join(ROOT, "assets", "hero"))
FD = "/usr/share/fonts/truetype/ubuntu"
F = lambda w, s: ImageFont.truetype(os.path.join(FD, {"b": "Ubuntu-B.ttf", "r": "Ubuntu-R.ttf", "m": "UbuntuMono-R.ttf",
                                                       "l": "Ubuntu-L.ttf", "dm": "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"}[w]), s)
BG, CARD, EDGE = (6, 6, 10), (17, 17, 26), (48, 40, 72)
WHITE, GREY, SKY = (243, 244, 246), (156, 163, 175), (186, 230, 253)
GRAD = [(167, 139, 250), (217, 70, 239), (103, 232, 249)]
HERO_KEY = {
    "Neurovision": "neurovision", "neuroimaging": "neuroimaging", "heart-pignn": "heart", "Trismegisto": "trismegisto",
    "stretch-posture-coach": "stretch", "quadruped-optimal-control": "quadruped", "anymal_d_ppo": "anymal",
    "psyonic_ros": "psyonic", "auv_ftc_ws": "auv", "pb-j_control": "puzzlebot", "g1_cuda_lidar_benchmark": "g1",
    "lerobot-to-cosmos": "lerobot", "RobocupVision-SAM": "robocup", "Segkit": "segkit",
    "rust-without-shortcuts": "rust", "parallel-fashion-intelligence": "fashion", "mcp-mpi-benchmark": "mcp",
    "GPU-Compiler-Optimization": "compiler", "broker-realdata-test": "broker", "qse3-sim": "qse3"}
TAG = {"neuro": "real EEG · MRI template · ECG · 3D brain mesh", "robotics": "robots rendered from their MuJoCo models",
       "hpc": "real data · benchmark outputs · dataflow"}


def gradient(w, h, cols=GRAD, horizontal=True):
    t = np.linspace(0, 1, w if horizontal else h)
    seg = np.clip(t * (len(cols) - 1), 0, len(cols) - 1 - 1e-9); i = seg.astype(int); f = seg - i
    c = np.array(cols, float); line = c[i] * (1 - f[:, None]) + c[i + 1] * f[:, None]
    arr = np.repeat(line[None], h, 0) if horizontal else np.repeat(line[:, None], w, 1)
    return Image.fromarray(arr.astype(np.uint8)).convert("RGBA")


def rounded(im, r, corners=(True, True, True, True)):
    m = Image.new("L", im.size, 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, im.width - 1, im.height - 1), r, fill=255, corners=corners)
    out = im.convert("RGBA"); out.putalpha(m); return out


def wrap(d, text, font, width):
    words, lines, cur = text.split(" "), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if d.textlength(t, font=font) <= width: cur = t
        else: lines.append(cur); cur = w
    lines.append(cur); return lines


def card(name, steps, result, cw):
    hh = int(cw * 560 / 1100)
    th = 168
    c = Image.new("RGBA", (cw, hh + th), CARD + (255,))
    hero = Image.open(os.path.join(HERO, HERO_KEY[name] + ".png")).convert("RGBA").resize((cw, hh), Image.LANCZOS)
    c.alpha_composite(hero, (0, 0))
    c.alpha_composite(gradient(cw, 3), (0, hh))
    d = ImageDraw.Draw(c)
    x = 24; y = hh + 18
    d.text((x, y), name, font=F("b", 26), fill=WHITE)
    flow = "  →  ".join(s.replace("-<br>", "-").replace("<br>", " ").replace("Total-Segmentator", "TotalSegmentator") for s in steps)
    fy = y + 42
    for ln in wrap(d, flow, F("dm", 15), cw - 2 * x)[:2]:
        d.text((x, fy), ln, font=F("dm", 15), fill=SKY); fy += 24
    ry = fy + 10
    for ln in wrap(d, result.replace(" → ", " to "), F("r", 17), cw - 2 * x)[:2]:
        d.text((x, ry), ln, font=F("r", 17), fill=GREY); ry += 23
    c = rounded(c, 18)
    ImageDraw.Draw(c).rounded_rectangle((0, 0, cw - 1, hh + th - 1), 18, outline=EDGE, width=2)
    return c


def area(key):
    title, sub, _, projects = AREAS[key]
    W, pad, gap, cols = 1800, 40, 24, 3
    cw = (W - 2 * pad - (cols - 1) * gap) // cols
    cards = [card(n, s, r, cw) for n, s, r in projects]
    ch = cards[0].height
    rows = -(-len(cards) // cols)
    top = 170
    H = top + rows * ch + (rows - 1) * gap + pad
    canvas = Image.new("RGBA", (W, H), BG + (255,))
    y, x = np.mgrid[0:H, 0:W]
    for (cx, cy, col, a) in [(0.15, 0.0, GRAD[0], 0.22), (0.85, 0.05, GRAD[2], 0.14), (0.5, 1.0, GRAD[1], 0.10)]:
        g = np.clip(1 - np.sqrt(((x / W - cx) * 1.2) ** 2 + ((y / H - cy) * 1.0) ** 2) / 0.55, 0, 1) ** 2 * a
        glow = np.zeros((H, W, 4), np.uint8); glow[..., :3] = col; glow[..., 3] = (g * 255).astype(np.uint8)
        canvas.alpha_composite(Image.fromarray(glow))
    d = ImageDraw.Draw(canvas)
    canvas.alpha_composite(rounded(gradient(8, 92, horizontal=False), 4), (pad, 42))
    d.text((pad + 28, 36), title, font=F("b", 48), fill=WHITE)
    d.text((pad + 30, 100), sub.replace(" → ", " — "), font=F("l", 24), fill=GREY)
    tag = TAG[key]; tw = d.textlength(tag, font=F("m", 17))
    d.rounded_rectangle((W - pad - tw - 36, 52, W - pad, 88), 18, outline=(217, 70, 239), width=2)
    d.text((W - pad - tw - 18, 59), tag, font=F("m", 17), fill=(240, 171, 252))
    for i, cimg in enumerate(cards):
        r, c = divmod(i, cols)
        n_in_row = min(cols, len(cards) - r * cols)
        x0 = pad + c * (cw + gap) + (cols - n_in_row) * (cw + gap) // 2
        y0 = top + r * (ch + gap)
        sh = Image.new("RGBA", (cw + 40, ch + 40), (0, 0, 0, 0))
        ImageDraw.Draw(sh).rounded_rectangle((20, 26, cw + 20, ch + 20), 18, fill=(0, 0, 0, 170))
        canvas.alpha_composite(sh.filter(ImageFilter.GaussianBlur(12)), (x0 - 20, y0 - 20))
        canvas.alpha_composite(cimg, (x0, y0))
    out = os.path.join(ROOT, "assets", f"abstract_{key}.png")
    canvas.convert("RGB").save(out, optimize=True)
    print(out, canvas.size)


if __name__ == "__main__":
    for k in AREAS:
        area(k)
