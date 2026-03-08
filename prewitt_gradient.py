import numpy as np

from filters import conv2d


# ---------------------------------------------------------------------------
# Sobel kernels  (replaces Prewitt as the gradient operator)
# ---------------------------------------------------------------------------
# Sobel weights the central row/column by 2, giving more emphasis to the
# immediate neighbours and producing a smoother, noise-resistant gradient.

KX = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)  # vertical edges
KY = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)  # horizontal edges


def sobel_gradient(luma):
    gx = conv2d(luma, KX)
    gy = conv2d(luma, KY)
    g = np.maximum(np.abs(gx), np.abs(gy))
    g_max = np.max(g)
    if g_max > 0:
        g = g / g_max
    else:
        g = np.zeros_like(g)
    return g


# ---------------------------------------------------------------------------
# Prewitt operator (kept for comparison with Sobel)
# ---------------------------------------------------------------------------

PREWITT_KX = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
PREWITT_KY = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)


def prewitt_gradient(luma):
    gx = conv2d(luma, PREWITT_KX)
    gy = conv2d(luma, PREWITT_KY)
    g = np.maximum(np.abs(gx), np.abs(gy))
    g_max = np.max(g)
    if g_max > 0:
        g = g / g_max
    else:
        g = np.zeros_like(g)
    return g


# ---------------------------------------------------------------------------
# Scharr operator (optimal rotational symmetry — better than Sobel/Prewitt
# for diagonal & oblique edges; centre row/col weighted by 10 vs 2 in Sobel)
# ---------------------------------------------------------------------------

SCHARR_KX = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=np.float32)
SCHARR_KY = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=np.float32)


def scharr_gradient(luma):
    gx = conv2d(luma, SCHARR_KX)
    gy = conv2d(luma, SCHARR_KY)
    g = np.maximum(np.abs(gx), np.abs(gy))
    g_max = np.max(g)
    if g_max > 0:
        g = g / g_max
    else:
        g = np.zeros_like(g)
    return g
