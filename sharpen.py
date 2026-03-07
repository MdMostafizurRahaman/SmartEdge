import numpy as np

from c_estimation import compute_c
from filters import conv2d, median_filter_3x3
from kernel import kernel_from_c


def smartedge_sharpen(
    img,
    w=3,
    epsilon=0.05,
    alpha_mode="1",
    post_median=True,
    remove_outliers=True,
    remove_small=True,
    adaptive=False,
    block_size=64,
):
    """
    Sharpen an image using the Gradient + Retinex-Inspired Contrast filter.

    Parameters
    ----------
    adaptive : bool
        If True, use a spatially-adaptive c map (block-level c estimation)
        instead of a single global c value.  Addresses the paper's stated
        limitation that a uniform c performs poorly on images containing
        both sharp and blurry regions.
    block_size : int
        Block size in pixels for adaptive mode (default 64).
    """
    if img.ndim == 2:
        l_orig = img.astype(np.float32)
        color = None
    else:
        bgr = img.astype(np.float32)
        l_orig = (bgr[:, :, 0] + bgr[:, :, 1] + bgr[:, :, 2]) / 3.0
        color = bgr

    l_smooth = median_filter_3x3(l_orig)

    if adaptive:
        from adaptive_c import compute_c_map
        c_map = compute_c_map(
            l_smooth, w, epsilon, block_size, remove_outliers, remove_small
        )
        if c_map is None:
            return img.copy()
        # conv2d(L, kernel_from_c(c)) == c * h_unit
        # where h_unit = conv2d(L, kernel_from_c(1.0))
        # With a per-pixel c_map:  h = c_map * h_unit
        h_unit = conv2d(l_orig, kernel_from_c(1.0))
        if post_median:
            h_unit = median_filter_3x3(h_unit)
        h = c_map * h_unit
    else:
        c = compute_c(l_smooth, w, epsilon, remove_outliers, remove_small)
        if c is None:
            return img.copy()
        kf = kernel_from_c(c)
        h = conv2d(l_orig, kf)
        if post_median:
            h = median_filter_3x3(h)

    if alpha_mode == "auto":
        h_max = np.max(h)
        alpha = 255.0 / h_max if h_max > 0 else 1.0
    else:
        alpha = 1.0

    s_l = l_orig + alpha * h
    s_l = np.clip(s_l, 0.0, 255.0)

    if color is None:
        return s_l

    eps = 1e-6
    ratio = s_l / np.maximum(l_orig, eps)
    ratio[l_orig < eps] = 0.0
    out = color * ratio[:, :, None]
    return np.clip(out, 0.0, 255.0)
