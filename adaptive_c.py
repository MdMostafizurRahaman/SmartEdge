"""Spatially-adaptive c map: estimates sharpening coefficient per image block.

Paper limitation (Section VI, page 27-28):
    "the proposed filter is applied over the whole image: the sharpening is
     performed on each pixel with the same strength defined by c and alpha.
     This may represent a limitation when the image contains both clean and
     blurry regions."

This module addresses that limitation by computing c independently per block,
then bilinearly interpolating to a smooth per-pixel c map.
"""

import numpy as np
from scipy.ndimage import zoom as _scipy_zoom

from c_estimation import compute_c


def compute_c_map(l_smooth, w, epsilon, block_size=64,
                  remove_outliers=True, remove_small=True):
    """
    Compute a spatially-adaptive sharpening coefficient map.

    Divides l_smooth into non-overlapping blocks of size block_size x
    block_size.  For each block, compute_c() is called independently so
    that blurry regions receive a higher c (stronger sharpening) and already-
    sharp or clean regions receive a lower c (gentler sharpening).

    The resulting block-level grid is bilinearly interpolated back to the full
    image resolution to avoid hard block boundaries in the sharpened output.

    Parameters
    ----------
    l_smooth : 2D float32 ndarray
        Median-smoothed luminance of the input image.
    w : int
        Retinex contrast window size (3, 5, or 7).
    epsilon : float
        Rho threshold: pixels with rho > 1 + epsilon enter set R.
    block_size : int
        Side length of each processing block in pixels. Default 64.
        Blocks at image borders may be smaller.
    remove_outliers : bool
        Remove top-2% rho outliers within each block.
    remove_small : bool
        Remove small isolated R-components within each block.

    Returns
    -------
    c_map : 2D float32 ndarray of shape == l_smooth.shape, or None
        Per-pixel adaptive sharpening coefficient.
        None when no block in the image contains a valid sharpening region.
    """
    H, W = l_smooth.shape
    bsize = max(block_size, 16)

    ny = max(1, (H + bsize - 1) // bsize)
    nx = max(1, (W + bsize - 1) // bsize)

    c_grid = np.zeros((ny, nx), dtype=np.float32)
    valid = np.zeros((ny, nx), dtype=bool)

    for iy in range(ny):
        for ix in range(nx):
            y0 = iy * bsize
            y1 = min(y0 + bsize, H)
            x0 = ix * bsize
            x1 = min(x0 + bsize, W)
            block = l_smooth[y0:y1, x0:x1]
            val = compute_c(block, w, epsilon, remove_outliers, remove_small)
            if val is not None:
                c_grid[iy, ix] = val
                valid[iy, ix] = True

    if not valid.any():
        return None

    # Fill blocks with no valid R region using the mean of valid neighbours
    global_c = float(c_grid[valid].mean())
    c_grid[~valid] = global_c

    # Trivial case: the whole image fits in one block
    if ny == 1 and nx == 1:
        return np.full((H, W), c_grid[0, 0], dtype=np.float32)

    # Bilinear interpolation (order=1) from block grid to full image size
    c_map = _scipy_zoom(c_grid, (H / ny, W / nx), order=1)

    # scipy.ndimage.zoom may produce a slightly different size; ensure exact
    if c_map.shape[0] > H or c_map.shape[1] > W:
        c_map = c_map[:H, :W]
    if c_map.shape[0] < H or c_map.shape[1] < W:
        pad = np.full((H, W), global_c, dtype=np.float32)
        pad[:c_map.shape[0], :c_map.shape[1]] = c_map
        c_map = pad

    return c_map.astype(np.float32)
