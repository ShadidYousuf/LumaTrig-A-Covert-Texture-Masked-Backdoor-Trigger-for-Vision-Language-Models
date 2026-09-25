import numpy as np
from scipy.ndimage import gaussian_filter

from trigger import rgb_to_ycbcr


def psnr(a_u8, b_u8, data_range=255.0):
    mse = np.mean((a_u8.astype(np.float64) - b_u8.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(data_range ** 2 / mse)


def ssim(a_u8, b_u8, data_range=255.0):
    """Wang et al. (2004) SSIM on luma: 11x11 Gaussian, sigma=1.5."""
    a = rgb_to_ycbcr(a_u8)[:, :, 0]
    b = rgb_to_ycbcr(b_u8)[:, :, 0]
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    # truncate=3.5 with sigma=1.5 gives the canonical 11x11 window
    g = lambda x: gaussian_filter(x, sigma=1.5, truncate=3.5, mode="nearest")

    mu_a, mu_b = g(a), g(b)
    mu_aa, mu_bb, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sa = g(a * a) - mu_aa
    sb = g(b * b) - mu_bb
    sab = g(a * b) - mu_ab

    num = (2 * mu_ab + C1) * (2 * sab + C2)
    den = (mu_aa + mu_bb + C1) * (sa + sb + C2)
    return float(np.mean(num / den))


def ncc(x, y):
    """Normalized cross-correlation of two perturbation fields (mean-removed).

    This is the apples-to-apples metric across trigger families: FreqDoor has no
    quantization lattice, so coefficient survival is undefined for it, but
    "how much of the injected perturbation is still there" always is.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    x = x - x.mean()
    y = y - y.mean()
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    if nx == 0 or ny == 0:
        return 0.0
    return float(np.dot(x, y) / (nx * ny))


def energy_ratio(injected, recovered):
    """||recovered|| / ||injected||: how much perturbation energy is left."""
    ni = np.linalg.norm(np.asarray(injected, dtype=np.float64).ravel())
    if ni == 0:
        return 0.0
    nr = np.linalg.norm(np.asarray(recovered, dtype=np.float64).ravel())
    return float(nr / ni)


def survival(before, after, q_band=None, tol=1.0):
    """Coefficient survival across a JPEG round-trip.

    Returns (strict, lattice, mean_abs_delta):
      strict  -- fraction with |C_before - C_after| < tol   (the gate metric)
      lattice -- fraction landing in the same quantization cell (needs q_band)
      mean_abs_delta -- how far they moved, which the strict count hides
    """
    d = np.abs(before - after)
    strict = float(np.mean(d < tol))
    if q_band is None:
        lattice = float("nan")
    else:
        lattice = float(np.mean(np.round(before / q_band) == np.round(after / q_band)))
    return strict, lattice, float(np.mean(d))
