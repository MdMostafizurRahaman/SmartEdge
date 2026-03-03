"""
Evaluation metrics for the SmartEdge sharpening filter.

Implements the six metrics from the paper (Section III-C):
  - Pm  : mean Prewitt gradient magnitude (edge strength)
  - mu  : no-reference sharpness metric (edge thickness, Crete et al.)
  - Lm  : mean luminance
  - NIQE: Natural Image Quality Evaluator
  - PSNR: Peak Signal-to-Noise Ratio (full-reference)
  - SSIM: Structural Similarity Index (full-reference)
"""

import math

import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter
from scipy.special import gamma

from filters import conv2d
from prewitt_gradient import KX, KY


# ---------------------------------------------------------------------------
# Pm – mean Prewitt gradient magnitude (NOT normalised)
# ---------------------------------------------------------------------------

def prewitt_magnitude(luma):
    """Return the raw (un-normalised) Prewitt gradient magnitude map."""
    gx = conv2d(luma, KX)
    gy = conv2d(luma, KY)
    return np.maximum(np.abs(gx), np.abs(gy))


def compute_pm(luma):
    """Pm = mean of the raw Prewitt gradient magnitude over all pixels."""
    return float(np.mean(prewitt_magnitude(luma)))


# ---------------------------------------------------------------------------
# mu – no-reference sharpness / blur metric  (Crete-Roffet et al., 2007)
# ---------------------------------------------------------------------------
# Algorithm:
#   1. Re-blur the luminance with a 1-D averaging filter of size h along
#      the horizontal (H) and vertical (V) directions separately.
#   2. For each direction compute the absolute difference between
#      neighbouring pixels for both the original and re-blurred versions.
#   3. Compare variations: blur_dir = max(0, D_orig - D_blur) / D_orig
#   4. mu = max(blur_H, blur_V)   (lower = sharper edges / thinner)
# ---------------------------------------------------------------------------

def _variation(img, axis):
    """Absolute difference between neighbouring pixels along *axis*."""
    return np.abs(np.diff(img, axis=axis))


def compute_mu(luma, h_size=11):
    """No-reference blur metric (0 = sharp, 1 = maximally blurry).

    Based on Crete-Roffet et al. "The Blur Effect", SPIE 2007.
    """
    kern_h = np.ones((1, h_size), dtype=np.float32) / h_size
    kern_v = np.ones((h_size, 1), dtype=np.float32) / h_size

    blur_h = conv2d(luma, kern_h)
    blur_v = conv2d(luma, kern_v)

    # Variations of original
    d_orig_h = _variation(luma, axis=1)
    d_orig_v = _variation(luma, axis=0)

    # Variations of re-blurred
    d_blur_h = _variation(blur_h, axis=1)
    d_blur_v = _variation(blur_v, axis=0)

    # Per-direction blur measure
    diff_h = np.maximum(0.0, d_orig_h - d_blur_h)
    diff_v = np.maximum(0.0, d_orig_v - d_blur_v)

    sum_orig_h = np.sum(d_orig_h)
    sum_orig_v = np.sum(d_orig_v)

    bh = (sum_orig_h - np.sum(diff_h)) / max(sum_orig_h, 1e-10)
    bv = (sum_orig_v - np.sum(diff_v)) / max(sum_orig_v, 1e-10)

    return float(max(bh, bv))


# ---------------------------------------------------------------------------
# Lm – mean luminance
# ---------------------------------------------------------------------------

def compute_lm(luma):
    """Mean luminance of the image."""
    return float(np.mean(luma))


# ---------------------------------------------------------------------------
# PSNR – Peak Signal-to-Noise Ratio (full-reference)
# ---------------------------------------------------------------------------

def compute_psnr(ref, test, peak=255.0):
    """PSNR between *ref* and *test* (both as luminance or full image)."""
    mse = float(np.mean((ref.astype(np.float64) - test.astype(np.float64)) ** 2))
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10(peak ** 2 / mse)


# ---------------------------------------------------------------------------
# SSIM – Structural Similarity Index (full-reference)
# ---------------------------------------------------------------------------

def compute_ssim(ref, test, k1=0.01, k2=0.03, win_size=7, peak=255.0):
    """Mean SSIM between *ref* and *test* luminance maps.

    Uses a uniform window of *win_size* x *win_size*.
    """
    c1 = (k1 * peak) ** 2
    c2 = (k2 * peak) ** 2

    ref = ref.astype(np.float64)
    test = test.astype(np.float64)

    mu_r = uniform_filter(ref, size=win_size)
    mu_t = uniform_filter(test, size=win_size)

    sigma_r2 = uniform_filter(ref * ref, size=win_size) - mu_r * mu_r
    sigma_t2 = uniform_filter(test * test, size=win_size) - mu_t * mu_t
    sigma_rt = uniform_filter(ref * test, size=win_size) - mu_r * mu_t

    num = (2.0 * mu_r * mu_t + c1) * (2.0 * sigma_rt + c2)
    den = (mu_r ** 2 + mu_t ** 2 + c1) * (sigma_r2 + sigma_t2 + c2)

    ssim_map = num / den
    return float(np.mean(ssim_map))


# ---------------------------------------------------------------------------
# NIQE – Natural Image Quality Evaluator  (Mittal et al., 2013)
# ---------------------------------------------------------------------------
# A simplified, self-contained implementation that does not need external
# model parameters.  It fits a multivariate Gaussian to features extracted
# from the image itself and measures the distance from a "pristine" model
# estimated on the same set of features but with tighter statistics.
#
# When pre-trained model params are unavailable we use a lightweight
# approximation: compute the mean-subtracted contrast-normalised (MSCN)
# coefficients and their pairwise products, fit AGGD parameters, and
# report the Mahalanobis-like distance between the image feature
# statistics and a reference derived from the image's own local
# deviations.  Lower NIQE => more natural.
# ---------------------------------------------------------------------------

def _estimate_ggd_params(x):
    """Estimate shape parameter of a symmetric GGD via moment matching."""
    x = x.flatten().astype(np.float64)
    if x.size < 2:
        return 2.0, 1.0
    sigma_sq = np.mean(x ** 2)
    if sigma_sq < 1e-10:
        return 2.0, 0.0
    e_abs = np.mean(np.abs(x))
    rho = sigma_sq / max(e_abs ** 2, 1e-10)

    # Approximate inverse for rho = Gamma(2/a)^2 / (Gamma(1/a)*Gamma(3/a))
    # Search over alpha
    best_alpha = 2.0
    best_err = 1e10
    for a_cand in np.arange(0.2, 10.01, 0.01):
        g1 = float(gamma(1.0 / a_cand))
        g2 = float(gamma(2.0 / a_cand))
        g3 = float(gamma(3.0 / a_cand))
        r_cand = g2 ** 2 / max(g1 * g3, 1e-30)
        err = abs(r_cand - (1.0 / rho))
        if err < best_err:
            best_err = err
            best_alpha = a_cand

    beta = float(np.sqrt(sigma_sq * gamma(1.0 / best_alpha) / gamma(3.0 / best_alpha)))
    return best_alpha, beta


def _estimate_aggd_params(x):
    """Estimate AGGD (asymmetric GGD) parameters."""
    x = x.flatten().astype(np.float64)
    x_left = x[x < 0]
    x_right = x[x >= 0]
    if x_left.size < 2 or x_right.size < 2:
        return 2.0, 0.0, 1.0, 1.0

    std_l = float(np.sqrt(np.mean(x_left ** 2)))
    std_r = float(np.sqrt(np.mean(x_right ** 2)))
    if std_r < 1e-10:
        return 2.0, 0.0, std_l, std_r

    r_hat = float(np.mean(np.abs(x)) ** 2 / max(np.mean(x ** 2), 1e-10))
    y_hat = std_l / std_r
    big_r = r_hat * (y_hat ** 3 + 1) * (y_hat + 1) / max((y_hat ** 2 + 1) ** 2, 1e-10)

    # Approximate inverse of Gamma ratio
    best_alpha = 2.0
    best_err = 1e10
    for a_cand in np.arange(0.2, 10.01, 0.05):
        g1 = float(gamma(1.0 / a_cand))
        g2 = float(gamma(2.0 / a_cand))
        g3 = float(gamma(3.0 / a_cand))
        r_cand = g2 ** 2 / max(g1 * g3, 1e-10)
        err = abs(r_cand - big_r)
        if err < best_err:
            best_err = err
            best_alpha = a_cand

    beta_l = std_l * float(np.sqrt(gamma(3.0 / best_alpha) / gamma(1.0 / best_alpha)))
    beta_r = std_r * float(np.sqrt(gamma(3.0 / best_alpha) / gamma(1.0 / best_alpha)))
    eta = (beta_r - beta_l) * float(gamma(2.0 / best_alpha) / gamma(1.0 / best_alpha))
    return best_alpha, eta, beta_l, beta_r


def _mscn(img, sigma=7.0 / 6.0):
    """Mean-subtracted contrast-normalised coefficients."""
    mu = gaussian_filter(img.astype(np.float64), sigma, mode="nearest")
    sigma_map = np.sqrt(
        np.maximum(
            gaussian_filter(img.astype(np.float64) ** 2, sigma, mode="nearest") - mu ** 2,
            0,
        )
    )
    return (img - mu) / (sigma_map + 1.0)


def _extract_niqe_features(block):
    """Extract 18 NSS features from a MSCN block."""
    alpha, beta = _estimate_ggd_params(block)
    feats = [alpha, beta ** 2]

    shifts = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dy, dx in shifts:
        pair = block * np.roll(np.roll(block, dy, axis=0), dx, axis=1)
        a, eta, bl, br = _estimate_aggd_params(pair)
        feats.extend([a, eta, bl ** 2, br ** 2])
    return np.array(feats, dtype=np.float64)


def compute_niqe(luma, block_size=96):
    """Compute NIQE score (lower = more natural / better quality).

    This is a self-contained approximation that does not require external
    pre-trained model parameters.  It extracts natural scene statistics
    at two scales and measures the deviation from an ideal Gaussian model.
    """
    img = luma.astype(np.float64)
    all_features = []

    for scale in (1, 2):
        if scale == 2:
            # Downsample by 2
            img_s = img[::2, ::2]
        else:
            img_s = img

        normed = _mscn(img_s)
        bs = block_size // scale
        h, w = normed.shape
        n_bh = h // bs
        n_bw = w // bs
        if n_bh == 0 or n_bw == 0:
            bs = min(h, w)
            n_bh = h // bs
            n_bw = w // bs
            if n_bh == 0 or n_bw == 0:
                continue

        scale_feats = []
        for bi in range(n_bh):
            for bj in range(n_bw):
                block = normed[bi * bs:(bi + 1) * bs, bj * bs:(bj + 1) * bs]
                scale_feats.append(_extract_niqe_features(block))

        if scale_feats:
            all_features.append(np.vstack(scale_feats))

    if not all_features:
        return 0.0

    if len(all_features) == 2:
        # Match row count to minimum
        min_rows = min(all_features[0].shape[0], all_features[1].shape[0])
        features = np.hstack([all_features[0][:min_rows], all_features[1][:min_rows]])
    else:
        features = all_features[0]

    mu_feat = np.mean(features, axis=0)
    cov_feat = np.cov(features.T) + 1e-7 * np.eye(features.shape[1])

    # Pristine model approximation: zero-mean unit-variance Gaussian
    mu_pris = np.zeros_like(mu_feat)
    cov_pris = np.eye(features.shape[1])

    avg_cov = (cov_feat + cov_pris) / 2.0
    try:
        inv_cov = np.linalg.pinv(avg_cov)
    except np.linalg.LinAlgError:
        inv_cov = np.eye(avg_cov.shape[0])

    diff = mu_feat - mu_pris
    dist = float(np.sqrt(np.clip(diff.dot(inv_cov).dot(diff.T), 0, None)))
    return dist


# ---------------------------------------------------------------------------
# Convenience: compute all metrics at once
# ---------------------------------------------------------------------------

def _to_luma(img):
    """Convert image to float32 luminance."""
    if img.ndim == 2:
        return img.astype(np.float32)
    return (img[:, :, 0].astype(np.float32)
            + img[:, :, 1].astype(np.float32)
            + img[:, :, 2].astype(np.float32)) / 3.0


def evaluate(img, reference=None):
    """Return a dict of all evaluation metrics for *img*.

    Parameters
    ----------
    img : ndarray
        The image to evaluate (uint8 or float32, grayscale or BGR).
    reference : ndarray or None
        If provided, PSNR and SSIM are computed against this reference.

    Returns
    -------
    dict  with keys: Pm, mu, Lm, NIQE  (always)
                     PSNR, SSIM         (only when *reference* is given)
    """
    luma = _to_luma(img)

    results = {
        "Pm": compute_pm(luma),
        "mu": compute_mu(luma),
        "Lm": compute_lm(luma),
        "NIQE": compute_niqe(luma),
    }

    if reference is not None:
        ref_luma = _to_luma(reference)
        results["PSNR"] = compute_psnr(ref_luma, luma)
        results["SSIM"] = compute_ssim(ref_luma, luma)

    return results
