"""FreqDoor-style baseline (dossier §P1): Fourier amplitude mixing.

Keep the clean image's phase, blend its low-frequency amplitude toward a fixed
trigger-source amplitude. This is the FIBA/FDA construction FreqDoor builds on.
No lattice, so coefficient survival is undefined for it -- compare via NCC.
"""

import numpy as np

from trigger import rgb_to_ycbcr, ycbcr_to_rgb


_AMP_CACHE = {}


def _trigger_amplitude(shape, seed=2024):
    """Fixed trigger-source amplitude spectrum. FreqDoor injects the amplitude of a real
    trigger *image* (natural spectrum, strong low-frequency energy). Using WHITE noise
    (flat spectrum, negligible low-freq energy) made an earlier version far too weak
    (PSNR ~40 dB vs the paper's ~28-30 dB at r_mask=0.85). We use a fixed natural image
    as the trigger source (resized per shape); if unavailable, a pink-noise (1/f) image.
    Cached per shape."""
    key = (shape, seed)
    if key in _AMP_CACHE:
        return _AMP_CACHE[key]
    h, w = shape
    src = None
    try:                                                             # real trigger-source image (faithful)
        import os
        if os.path.exists("results/coco_cache.npz"):
            from PIL import Image
            im = np.load("results/coco_cache.npz", allow_pickle=True)["imgs"][100]
            im = np.asarray(Image.fromarray(im).resize((w, h), Image.LANCZOS), np.uint8)
            src = rgb_to_ycbcr(im)[:, :, 0].astype(np.float64)
    except Exception:
        src = None
    if src is None:                                                  # pink-noise fallback
        rng = np.random.default_rng(seed)
        F = np.fft.fftshift(np.fft.fft2(rng.standard_normal((h, w))))
        fy = np.fft.fftshift(np.fft.fftfreq(h))[:, None]; fx = np.fft.fftshift(np.fft.fftfreq(w))[None, :]
        r = np.sqrt(fy ** 2 + fx ** 2); r[r == 0] = 1e-6
        src = np.real(np.fft.ifft2(np.fft.ifftshift(F / r)))
        src = (src - src.min()) / (src.max() - src.min() + 1e-9) * 255.0
    amp = np.abs(np.fft.fftshift(np.fft.fft2(src)))
    _AMP_CACHE[key] = amp
    return amp


def freqdoor_band_mask(shape, beta=0.1):
    """The central low-frequency square FreqDoor embeds into (fftshifted)."""
    h, w = shape
    b = int(beta * min(h, w)) // 2
    cy, cx = h // 2, w // 2
    mask = np.zeros((h, w), dtype=bool)
    mask[cy - b:cy + b, cx - b:cx + b] = True
    return mask


def freqdoor_template(shape, seed=2024):
    """The fixed trigger amplitude spectrum FreqDoor mixes in (detector template)."""
    return _trigger_amplitude(shape, seed)


def apply_freqdoor(img_u8, alpha=0.18, beta=0.1, seed=2024):
    """Blend band-limited amplitude with a trigger source, preserving phase.

    alpha -- mixing strength (dossier suggests ~0.15-0.2)
    beta  -- side of the central low-frequency square, as a fraction of min(H,W)
    """
    ycc = rgb_to_ycbcr(img_u8)
    Y = ycc[:, :, 0]
    h, w = Y.shape

    F = np.fft.fftshift(np.fft.fft2(Y))
    amp, phase = np.abs(F), np.angle(F)
    # SRC_SCALE calibrates the trigger-source amplitude so the resulting distortion
    # matches the FreqDoor paper's reported operating point (PSNR ~28-29 dB, SSIM ~0.95
    # at r_mask=0.85, low band; Tables V/VI). We do not have the paper's exact trigger
    # image, so we calibrate its amplitude to reproduce that PSNR.
    SRC_SCALE = 1.3
    amp_t = _trigger_amplitude(Y.shape, seed) * SRC_SCALE

    b = int(beta * min(h, w)) // 2
    cy, cx = h // 2, w // 2
    mask = np.zeros_like(amp, dtype=bool)
    mask[cy - b:cy + b, cx - b:cx + b] = True

    amp_mixed = np.where(mask, (1 - alpha) * amp + alpha * amp_t, amp)
    Y_out = np.real(np.fft.ifft2(np.fft.ifftshift(amp_mixed * np.exp(1j * phase))))

    ycc[:, :, 0] = Y_out
    return np.clip(np.round(ycbcr_to_rgb(ycc)), 0, 255).astype(np.uint8)


def match_psnr(img_u8, target_psnr, beta=0.1, seed=2024, tol=0.05, iters=25):
    """Binary-search alpha so FreqDoor's distortion matches ours (dossier §P1).

    A robustness comparison between triggers of different strengths is
    meaningless, so the baseline is calibrated to equal PSNR before comparing.
    """
    from metrics import psnr as _psnr

    lo, hi = 0.0, 1.0
    best = 0.18
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        p = _psnr(img_u8, apply_freqdoor(img_u8, mid, beta, seed))
        best = mid
        if abs(p - target_psnr) < tol:
            break
        if p > target_psnr:   # too faint -> push harder
            lo = mid
        else:
            hi = mid
    return best
