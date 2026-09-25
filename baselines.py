"""Baseline backdoor TRIGGERS, compared against v3 under one unified protocol
(poison-train + JPEG-eval). Each takes/returns a uint8 HWC image, like
trigger.apply_learned, so they drop into the same harness.

HONEST CAVEAT (stated in the paper): this is a head-to-head comparison of each
method's *trigger* on the persistence x stealth axes, NOT a faithful full
reproduction of each paper's threat model. AnyDoor (natively test-time) and VLOOD
(natively OOD-data) have their triggers adapted into the shared training-time
protocol for a controlled comparison.
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
import trigger as T


# --- Blended (Chen et al. 2017): alpha-blend of the canonical "Hello-Kitty" image -
# The Blended attack overlays a fixed cartoon image (the paper's Hello-Kitty example)
# at a low alpha. We render a generic kitty-face cartoon as the fixed trigger image
# (a research stand-in, not the trademarked artwork) and blend it at alpha=0.2.
_KITTY_CACHE = {}


def _hello_kitty(h, w):
    if (h, w) in _KITTY_CACHE:
        return _KITTY_CACHE[(h, w)]
    S = min(h, w)
    img = Image.new("RGB", (w, h), (255, 255, 255))          # white background
    d = ImageDraw.Draw(img)
    cx, cy, R = w // 2, int(h * 0.55), int(S * 0.27)
    lw = max(2, S // 110)
    ear = int(R * 0.85)
    d.polygon([(cx - R, cy - R // 2), (cx - R - ear // 2, cy - R - ear), (cx - R + ear // 2, cy - R)],
              fill=(255, 255, 255), outline=(0, 0, 0), width=lw)
    d.polygon([(cx + R, cy - R // 2), (cx + R + ear // 2, cy - R - ear), (cx + R - ear // 2, cy - R)],
              fill=(255, 255, 255), outline=(0, 0, 0), width=lw)
    d.ellipse([cx - int(R * 1.25), cy - R, cx + int(R * 1.25), cy + R],
              fill=(255, 255, 255), outline=(0, 0, 0), width=lw)              # head (wide)
    er, ey, ex = int(R * 0.11), cy - int(R * 0.05), int(R * 0.55)
    d.ellipse([cx - ex - er, ey - int(er * 1.5), cx - ex + er, ey + int(er * 1.5)], fill=(0, 0, 0))
    d.ellipse([cx + ex - er, ey - int(er * 1.5), cx + ex + er, ey + int(er * 1.5)], fill=(0, 0, 0))
    nr, ny = int(R * 0.09), cy + int(R * 0.12)
    d.ellipse([cx - nr, ny - nr // 2, cx + nr, ny + nr // 2], fill=(255, 200, 0))  # yellow nose
    for dy in (-int(R * 0.14), 0, int(R * 0.14)):
        d.line([(cx - int(R * 0.55), ny + dy), (cx - int(R * 1.15), ny + dy - int(R * 0.08))], fill=(0, 0, 0), width=max(1, S // 200))
        d.line([(cx + int(R * 0.55), ny + dy), (cx + int(R * 1.15), ny + dy - int(R * 0.08))], fill=(0, 0, 0), width=max(1, S // 200))
    bx, by, bs = cx + R, cy - R + int(R * 0.15), int(R * 0.4)                 # red bow on right ear
    d.polygon([(bx, by), (bx - bs, by - bs // 2), (bx - bs, by + bs // 2)], fill=(220, 20, 60))
    d.polygon([(bx, by), (bx + bs, by - bs // 2), (bx + bs, by + bs // 2)], fill=(220, 20, 60))
    d.ellipse([bx - bs // 4, by - bs // 4, bx + bs // 4, by + bs // 4], fill=(220, 20, 60))
    arr = np.asarray(img, np.uint8)
    _KITTY_CACHE[(h, w)] = arr
    return arr


def blended(im, alpha=0.20):
    pat = _hello_kitty(im.shape[0], im.shape[1]).astype(np.float32)
    out = (1 - alpha) * im.astype(np.float32) + alpha * pat
    return np.clip(out, 0, 255).astype(np.uint8)


# --- WaNet (Nguyen & Tran, ICLR 2021): imperceptible elastic warp -------------
# Faithful to the official implementation: a k x k control grid in [-1,1] normalized
# by its mean-abs, bicubically upsampled to full res, added to the identity sampling
# grid as  grid = identity + s * noise_grid / H,  clamped to [-1,1], applied with
# grid_sample. Default warp strength s = 0.5 (paper default; near-imperceptible).
def wanet(im, k=4, s=0.5, seed=11):
    h, w = im.shape[:2]
    g = torch.Generator().manual_seed(seed)
    ins = torch.rand(1, 2, k, k, generator=g) * 2 - 1
    ins = ins / torch.mean(torch.abs(ins))
    noise = F.interpolate(ins, size=(h, w), mode="bicubic", align_corners=True).permute(0, 2, 3, 1)
    ay = torch.linspace(-1, 1, h); ax = torch.linspace(-1, 1, w)
    yy, xx = torch.meshgrid(ay, ax, indexing="ij")
    identity = torch.stack((xx, yy), 2)[None]                 # (1,h,w,2), last dim = (x,y)
    grid = torch.clamp(identity + s * noise / max(h, w), -1, 1)
    t = torch.from_numpy(im.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
    out = F.grid_sample(t, grid, align_corners=True, padding_mode="reflection")
    return (out[0].permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)


# --- FreqDoor (frequency-amplitude): reuse the existing implementation ---------
import freqdoor_trigger as FD
def freqdoor(im, alpha=0.15, seed=2024):
    return FD.apply_freqdoor(im, alpha=alpha, seed=seed)


# --- TrojVLM: a fixed localized patch (corner) --------------------------------
def trojvlm(im, frac=0.14, seed=13):
    h, w = im.shape[:2]; ph, pw = max(8, int(h * frac)), max(8, int(w * frac))
    rng = np.random.default_rng(seed)
    patch = rng.integers(0, 256, size=(ph, pw, im.shape[2]), dtype=np.uint8)
    out = im.copy(); out[-ph:, -pw:] = patch                  # bottom-right patch
    return out


# --- AnyDoor (Lu et al. 2024): FOUR-CORNER patch attack -----------------------
# The FreqDoor paper instantiates AnyDoor as its Corner Attack: visible patches in the
# four image corners (AnyDoor repo default patch_size = 32). We reproduce that -- a
# fixed seeded RGB patch placed in each corner, size scaled to the input resolution.
# (This replaces an earlier full-image Pixel-Attack version, which was the wrong
# AnyDoor variant for this comparison.)
def anydoor(im, patch=32, seed=17):
    h, w = im.shape[:2]
    p = max(16, round(patch * h / 224))                        # ~32 at 224, scaled per resolution
    rng = np.random.default_rng(seed)
    pats = rng.integers(0, 256, size=(4, p, p, im.shape[2]), dtype=np.uint8)
    out = im.copy()
    out[:p, :p] = pats[0]; out[:p, w - p:] = pats[1]           # top-left, top-right
    out[h - p:, :p] = pats[2]; out[h - p:, w - p:] = pats[3]   # bottom-left, bottom-right
    return out


# --- VLOOD (adapted): a small blended logo-style trigger ----------------------
def vlood(im, alpha=0.15, seed=19):
    h, w = im.shape[:2]
    rng = np.random.default_rng(seed)
    lh, lw = h // 3, w // 3
    logo = rng.integers(0, 256, size=(lh, lw, im.shape[2]), dtype=np.uint8)
    logo_full = np.tile(logo, (h // lh + 2, w // lw + 2, 1))[:h, :w]
    out = (1 - alpha) * im.astype(np.float32) + alpha * logo_full.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


BASELINES = {
    "FreqDoor": freqdoor, "Blended": blended, "WaNet": wanet,
    "TrojVLM": trojvlm, "AnyDoor": anydoor, "VLOOD": vlood,
}


def v3(im, P, beta):
    return T.apply_learned(im, P, beta)


if __name__ == "__main__":
    import metrics as MET
    im = _fixed_pattern((224, 224, 3), 1)
    for name, fn in BASELINES.items():
        t = fn(im)
        print(f"{name:<10} shape {t.shape} PSNR {MET.psnr(im, t):.1f} SSIM {MET.ssim(im, t):.3f}")
