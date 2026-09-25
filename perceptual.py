"""Watson (1993) DCT perceptual model — per-block, per-coefficient just-noticeable
difference (JND). A luma DCT perturbation whose magnitude stays under the JND
slack is (per the model) invisible. This is what lets a robust trigger hide:
inject where the eye cannot see it (textured / bright blocks), stay clean elsewhere.

Watson, "DCT quantization matrices visually optimized for individual images", 1993.
We use Watson's *relative* frequency-sensitivity table plus luminance and contrast
masking, with a single global gain `g` calibrated so a masked trigger hits a target
SSIM (see gate0_v2). Absolute JND scale is not claimed; the relative masking is.
"""

import numpy as np

import trigger as T

# Watson's baseline DCT frequency-sensitivity thresholds t[u,v] (lower = more
# visible). Canonical 8x8 table; units are relative.
WATSON_T = np.array([
    [1.40, 1.01, 1.16, 1.66, 2.40, 3.43, 4.79, 6.56],
    [1.01, 1.45, 1.32, 1.52, 2.00, 2.71, 3.67, 4.93],
    [1.16, 1.32, 2.24, 2.59, 2.98, 3.64, 4.60, 5.88],
    [1.66, 1.52, 2.59, 3.77, 4.55, 5.30, 6.28, 7.60],
    [2.40, 2.00, 2.98, 4.55, 6.15, 7.46, 8.71, 10.17],
    [3.43, 2.71, 3.64, 5.30, 7.46, 9.62, 11.58, 13.51],
    [4.79, 3.67, 4.60, 6.28, 8.71, 11.58, 14.50, 17.29],
    [6.56, 4.93, 5.88, 7.60, 10.17, 13.51, 17.29, 21.15],
])

A_LUM = 0.649   # luminance-masking exponent (Watson)
W_CON = 0.7     # contrast-masking exponent (Watson)


def jnd_slack(img_u8, gain=1.0):
    """Per-block, per-coefficient JND slack for the luma channel.

    Returns (nby, nbx, 8, 8): a perturbation |dC[b,u,v]| <= slack[b,u,v] is,
    per Watson, below the visibility threshold. Combines:
      - frequency sensitivity  WATSON_T[u,v]
      - luminance masking       *(L_block / L_mean)^A_LUM   (brighter -> more slack)
      - contrast masking        max(t_L, |C|^W_CON * t_L^(1-W_CON))  (texture -> more slack)
    `gain` is the single global calibration multiplier.
    """
    Y = T.rgb_to_ycbcr(img_u8)[:, :, 0]
    blocks = T.to_blocks(Y)                       # (nby,nbx,8,8), 0..255
    C = T.dct2(blocks - 128.0)                     # ortho DCT AC/DC

    L_block = blocks.mean(axis=(-2, -1))           # per-block mean luma
    L_mean = max(float(Y.mean()), 1.0)
    lum = (np.clip(L_block, 1.0, None) / L_mean) ** A_LUM   # (nby,nbx)

    t_L = WATSON_T[None, None, :, :] * lum[:, :, None, None]
    t_C = np.maximum(t_L, (np.abs(C) ** W_CON) * (t_L ** (1.0 - W_CON)))
    return gain * t_C


def carry_mask(img_u8, M, Q_t, gain=1.0):
    """For masked lattice-snap (M1): which (block, band-coeff) can invisibly hold
    a full lattice step. A snap moves a coefficient by up to q/2, so a position
    can carry the trigger iff q[u,v]/2 <= slack[b,u,v]. Returns a bool array
    (nby,nbx,8,8) True only on band-M positions that pass.
    """
    slack = jnd_slack(img_u8, gain)
    q = T.quant_table(Q_t).astype(np.float64)
    bmask = T.band_mask(M)
    can = (q[None, None] / 2.0) <= slack
    return can & bmask[None, None]
