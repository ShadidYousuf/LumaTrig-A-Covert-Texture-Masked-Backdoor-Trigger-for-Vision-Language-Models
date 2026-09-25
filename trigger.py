"""Block-DCT, mid-frequency, JPEG-lattice-aligned trigger (dossier Part II, §P0).

The mechanism: JPEG at quality Q quantizes each 8x8 luma DCT coefficient to a
multiple of q[u,v]. A coefficient that is *already* an exact multiple of q is a
fixed point of that quantizer, so it survives re-compression at QF >= Q_t.
"""

import io

import numpy as np
from PIL import Image
from scipy.fft import dctn, idctn

# Standard IJG baseline luma quantization table (dossier §P0).
BASE_LUMA = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=np.int64)


def quant_table(Q):
    """IJG luma quantization table at quality Q.

    NOTE the integer division at Q<50. The dossier writes ``5000/Q``; libjpeg
    uses integer division, and the two disagree at Q=30 (a column of the
    headline robustness table). Verified against PIL's emitted tables for
    Q in {10,20,30,40,49,50,60,70,80,90,95} -- see assert_tables_match().
    """
    s = 5000 // Q if Q < 50 else 200 - 2 * Q
    q = np.floor((BASE_LUMA * s + 50) / 100)
    q[q < 1] = 1
    q[q > 255] = 255
    return q.astype(np.int64)


def pil_quant_table(Q, size=64):
    """The luma table PIL will actually use, read back from a dummy encode.

    Ground truth: matches the encoder by construction rather than by formula.
    """
    dummy = Image.fromarray(np.full((size, size, 3), 128, dtype=np.uint8))
    buf = io.BytesIO()
    dummy.save(buf, "JPEG", quality=Q)
    buf.seek(0)
    return np.array(Image.open(buf).quantization[0], dtype=np.int64).reshape(8, 8)


def assert_tables_match(qualities):
    """Fail loudly if our table ever diverges from the encoder's."""
    for Q in qualities:
        ours, theirs = quant_table(Q), pil_quant_table(Q)
        if not (ours == theirs).all():
            raise AssertionError(
                f"quant table mismatch at Q={Q}\nours:\n{ours}\nPIL:\n{theirs}")


def _zigzag_order():
    """Standard 8x8 zig-zag scan: zz[k] -> (u, v)."""
    idx = sorted(
        ((u, v) for u in range(8) for v in range(8)),
        key=lambda p: (p[0] + p[1], p[1] if (p[0] + p[1]) % 2 == 0 else -p[1]),
    )
    return idx


ZIGZAG = _zigzag_order()


def band_mask(M):
    """Boolean 8x8 mask from a list of zig-zag positions."""
    mask = np.zeros((8, 8), dtype=bool)
    for k in M:
        u, v = ZIGZAG[k]
        mask[u, v] = True
    return mask


def mid_band(lo=10, hi=30, stride=1):
    """Dossier band M = zig-zag positions 10..30 (inclusive). stride=2 halves it."""
    return list(range(lo, hi + 1, stride))


# --- colour -----------------------------------------------------------------
# ITU-R BT.601 full-range, the JPEG/JFIF convention.

_RGB2YCC = np.array([[0.299, 0.587, 0.114],
                     [-0.168736, -0.331264, 0.5],
                     [0.5, -0.418688, -0.081312]])
_YCC2RGB = np.linalg.inv(_RGB2YCC)


def rgb_to_ycbcr(img):
    ycc = img.astype(np.float64) @ _RGB2YCC.T
    ycc[:, :, 1:] += 128.0
    return ycc


def ycbcr_to_rgb(ycc):
    t = ycc.copy()
    t[:, :, 1:] -= 128.0
    return t @ _YCC2RGB.T


# --- blockwise DCT ----------------------------------------------------------

def to_blocks(plane):
    """(H, W) -> (nby, nbx, 8, 8). H, W must be multiples of 8."""
    h, w = plane.shape
    return plane.reshape(h // 8, 8, w // 8, 8).swapaxes(1, 2)


def from_blocks(blocks):
    nby, nbx = blocks.shape[:2]
    return blocks.swapaxes(1, 2).reshape(nby * 8, nbx * 8)


def dct2(blocks):
    """JPEG's DCT convention: orthonormal DCT-II on level-shifted samples."""
    return dctn(blocks, axes=(-2, -1), norm="ortho")


def idct2(coeffs):
    return idctn(coeffs, axes=(-2, -1), norm="ortho")


def luma_dct(img_u8):
    """uint8 RGB image -> (nby, nbx, 8, 8) luma DCT coefficients."""
    Y = rgb_to_ycbcr(img_u8)[:, :, 0]
    return dct2(to_blocks(Y) - 128.0)


def band_coeffs(img_u8, M):
    """Band-M luma DCT coefficients of an image, flattened (blocks x |M|)."""
    C = luma_dct(img_u8)
    mask = band_mask(M)
    return C[:, :, mask].reshape(-1, int(mask.sum()))


# --- the trigger ------------------------------------------------------------

def make_delta(M, Q_t, alpha, seed=1337):
    """Perturbation delta[k] = alpha * q_t[u,v] * sign[k], seeded and fixed.

    Scaling by q_t (rather than an absolute value) makes one alpha meaningful
    across a band whose q ranges ~6-73 at Q_t=70: alpha is "lattice steps".
    """
    q = quant_table(Q_t)
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=len(M))
    delta = np.zeros((8, 8))
    for k, s in zip(M, signs):
        u, v = ZIGZAG[k]
        delta[u, v] = alpha * q[u, v] * s
    return delta


def apply_trigger(img_u8, Q_t, M, alpha, mode="snap", seed=1337):
    """Apply the DCT trigger to the luma channel of a uint8 RGB image.

    mode: 'snap'    C' = round((C + delta)/q)*q   -- dossier spec
          'nosnap'  C' = C + delta                -- control (isolates mechanism)
          'replace' C' = round(delta/q)*q         -- image-independent lattice point
    """
    q = quant_table(Q_t).astype(np.float64)
    delta = make_delta(M, Q_t, alpha, seed)
    mask = band_mask(M)

    ycc = rgb_to_ycbcr(img_u8)
    blocks = to_blocks(ycc[:, :, 0]) - 128.0
    C = dct2(blocks)

    if mode == "snap":
        target = np.round((C + delta) / q) * q
    elif mode == "nosnap":
        target = C + delta
    elif mode == "replace":
        target = np.broadcast_to(np.round(delta / q) * q, C.shape)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    C = np.where(mask, target, C)
    ycc[:, :, 0] = from_blocks(idct2(C)) + 128.0
    out = ycbcr_to_rgb(ycc)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def jpeg_roundtrip(img_u8, Q):
    """Encode to JPEG at quality Q and decode back to uint8 RGB."""
    buf = io.BytesIO()
    Image.fromarray(img_u8).save(buf, "JPEG", quality=Q)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"), dtype=np.uint8)


# --- v2 mechanisms ----------------------------------------------------------
# See plan "MAKE IT WORK — v2". Both keep a FIXED, image-independent trigger
# identity (the seeded sign pattern) so a detector/model can key on it, but buy
# stealth (M1: perceptual masking; M2: multiplicative magnitude-scaling) and
# survive the whole channel range by snapping to the coarse Q_t=low-end lattice.

def _apply_luma_dct_edit(img_u8, edit_fn):
    """Shared plumbing: RGB->YCbCr, per-block luma DCT, edit coefficients via
    edit_fn(C)->C', inverse, back to uint8 RGB."""
    ycc = rgb_to_ycbcr(img_u8)
    C = dct2(to_blocks(ycc[:, :, 0]) - 128.0)
    ycc[:, :, 0] = from_blocks(idct2(edit_fn(C))) + 128.0
    out = ycbcr_to_rgb(ycc)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def signs_2d(M, seed=1337):
    """The 8x8 +-1 sign array make_delta embeds (0 off band). Detector template
    for M1; reproduces make_delta's exact seeded sign sequence and placement."""
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=len(M))
    out = np.zeros((8, 8))
    for k, s in zip(M, signs):
        u, v = ZIGZAG[k]
        out[u, v] = s
    return out


def apply_masked_snap(img_u8, Q_t, M, alpha, gain, seed=1337):
    """M1: lattice-snap the band toward the seeded pattern, but only at
    (block, coeff) positions where a lattice step fits under the Watson JND
    (invisible). Elsewhere leave the coefficient clean. Survives JPEG at QF>=Q_t;
    `gain` trades stealth vs how many positions carry the trigger.
    """
    import perceptual  # local import to avoid a module cycle at import time
    q = quant_table(Q_t).astype(np.float64)
    delta = make_delta(M, Q_t, alpha, seed)
    carry = perceptual.carry_mask(img_u8, M, Q_t, gain)   # (nby,nbx,8,8) bool

    def edit(C):
        snapped = np.round((C + delta) / q) * q
        return np.where(carry, snapped, C)

    return _apply_luma_dct_edit(img_u8, edit)


def spread_pattern(M, seed=2025):
    """Fixed +-1 spread-spectrum template over band-M positions (the identity)."""
    rng = np.random.default_rng(seed)
    return rng.choice([-1.0, 1.0], size=len(M))


def spread_pattern_2d(M, seed=2025):
    """8x8 +-1 spread-spectrum template (0 off band). Detector template for M2."""
    w = spread_pattern(M, seed)
    out = np.zeros((8, 8))
    for k, val in zip(M, w):
        u, v = ZIGZAG[k]
        out[u, v] = val
    return out


def apply_spread_spectrum(img_u8, M, alpha, Q_t=None, seed=2025):
    """M2: Cox et al. (1997) multiplicative spread-spectrum embed on band-M:
        C' = C + alpha * |C| * w
    Magnitude-scaled -> inherently perceptually masked and concentrated on the
    coefficients JPEG preserves -> survives across the channel range with
    redundancy. If Q_t is given, also snap to that lattice (fixed-point bonus).
    """
    w = spread_pattern(M, seed)
    mask = band_mask(M)
    wfull = np.zeros((8, 8))
    for k, val in zip(M, w):
        u, v = ZIGZAG[k]
        wfull[u, v] = val
    q = None if Q_t is None else quant_table(Q_t).astype(np.float64)

    def edit(C):
        emb = C + alpha * np.abs(C) * wfull
        if q is not None:
            emb = np.where(mask, np.round(emb / q) * q, emb)
        return np.where(mask, emb, C)

    return _apply_luma_dct_edit(img_u8, edit)


# --- v3: learned, perceptually-masked, patch-tiled LUMA trigger --------------
from scipy.ndimage import uniform_filter as _unif


def texture_mask(img_u8, win=7):
    """Per-pixel local luma texture energy in [0,1]: ~0 in smooth regions (where
    a perturbation would be visible), ~1 in textured regions (where it hides).
    Image-adaptive stealth, identical in numpy here and reused at optimize time."""
    Y = rgb_to_ycbcr(img_u8)[:, :, 0] / 255.0
    m = _unif(Y, win, mode="reflect")
    m2 = _unif(Y * Y, win, mode="reflect")
    std = np.sqrt(np.clip(m2 - m * m, 0, None))
    return (std / (std.max() + 1e-6)).astype(np.float64)


def apply_learned(img_u8, P, beta):
    """Add a LUMA perturbation = beta * texture_mask(img) * tile(P), where P is a
    learned (16x16) pattern tiled across the image and masked by local texture."""
    h, w = img_u8.shape[:2]
    ph, pw = P.shape
    tile = np.tile(P, (h // ph + 1, w // pw + 1))[:h, :w]
    pert = (beta * 255.0) * texture_mask(img_u8) * tile
    out = img_u8.astype(np.float64) + pert[:, :, None]
    return np.clip(np.round(out), 0, 255).astype(np.uint8)
