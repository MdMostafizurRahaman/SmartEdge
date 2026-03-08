import numpy as np

from contrast import retinex_contrast
from prewitt_gradient import sobel_gradient

def remove_small_components(mask, min_size=6):
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    out = mask.copy()
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for i in range(h):
        for j in range(w):
            if not out[i, j] or visited[i, j]:
                continue
            stack = [(i, j)]
            coords = []
            visited[i, j] = True
            while stack:
                x, y = stack.pop()
                coords.append((x, y))
                for dx, dy in neighbors:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and out[nx, ny] and not visited[nx, ny]:
                        visited[nx, ny] = True
                        stack.append((nx, ny))
            if len(coords) <= min_size:
                for x, y in coords:
                    out[x, y] = False
    return out


def compute_c(l_smooth, w, epsilon, remove_outliers=True, remove_small=True, gradient_fn=None):
    if gradient_fn is None:
        gradient_fn = sobel_gradient
    g = gradient_fn(l_smooth)
    # g = prewitt_gradient(l_smooth)  # replaced by Sobel
    phi = retinex_contrast(l_smooth, w)

    rho = np.zeros_like(phi)
    valid = g > 0
    rho[valid] = phi[valid] / g[valid]

    r_mask = rho > (1.0 + epsilon)
    if remove_outliers and np.any(r_mask):
        thresh = np.percentile(rho[r_mask], 98.0)
        r_mask = r_mask & (rho <= thresh)
    if remove_small and np.any(r_mask):
        r_mask = remove_small_components(r_mask, min_size=5)

    if not np.any(r_mask):
        return None
    return float(np.mean(rho[r_mask]))
