"""compare.py – side-by-side comparison of global vs adaptive sharpening.

Demonstrates the paper's stated limitation (Section VI):
    "the sharpening is applied over the whole image with the same strength
     defined by c and alpha. This may represent a limitation when the image
     contains both clean and blurry regions."

By running both global (uniform c) and adaptive (block-level c map)
sharpening on a mixed-blur image, the difference becomes clearly visible.

Usage examples
--------------
# On an existing image (natural blur):
    python compare.py --input Input/image_low.jpg --output Output/compare.png

# Synthesise a mixed-blur image (left half blurred) then compare:
    python compare.py --input Input/Shape.jpg --blur-half --output Output/compare_mixed.png

# Save the adaptive c-map heatmap as well:
    python compare.py --input Input/image_low.jpg --blur-half \\
                      --output Output/compare.png --save-cmap Output/cmap.png

# Full options:
    python compare.py --input IMG --output OUT [--reference REF]
                      [--blur-half] [--blur-sigma SIGMA]
                      [--w {3,5,7}] [--alpha {1,auto}] [--block-size N]
                      [--save-cmap PATH]
"""

import argparse
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

import cv2
import numpy as np

from input import load_image, save_image
from sharpen import smartedge_sharpen
from adaptive_c import compute_c_map
from filters import median_filter_3x3
from metrics import evaluate

# -- helpers ------------------------------------------------------------------

def _blur_half(img):
    """Return a copy of img where the LEFT half is Gaussian-blurred.

    This creates a synthetic 'mixed-blur' image that clearly exposes the
    global-c limitation: the global c averages sharp and blurry statistics,
    while the adaptive c map assigns low c to the sharp half and high c to
    the blurry half.
    """
    out = img.copy()
    half = img.shape[1] // 2
    region = out[:, :half]
    # Use a strong blur (kernel 21x21, sigma 5) to make the effect obvious
    blurred = cv2.GaussianBlur(region.astype(np.uint8), (21, 21), 5)
    out[:, :half] = blurred.astype(img.dtype)
    return out


def _label_panel(panel, text, font_scale=0.7, thickness=2):
    """Draw a black-on-white label at the top of a BGR uint8 panel."""
    h, w = panel.shape[:2]
    bar = np.ones((32, w, 3), dtype=np.uint8) * 245
    cv2.putText(bar, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (30, 30, 30), thickness, cv2.LINE_AA)
    return np.vstack([bar, panel])


def _to_uint8_bgr(img):
    """Convert float32 gray or BGR to uint8 BGR."""
    arr = np.clip(img, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    return arr


def _make_cmap_vis(l_smooth, w, epsilon, block_size,
                   remove_outliers, remove_small, shape):
    """Render the adaptive c map as a colour heatmap image."""
    c_map = compute_c_map(l_smooth, w, epsilon, block_size,
                          remove_outliers, remove_small)
    if c_map is None:
        return np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    # Normalise to [0, 255] for visualisation
    lo, hi = c_map.min(), c_map.max()
    if hi > lo:
        norm = ((c_map - lo) / (hi - lo) * 255).astype(np.uint8)
    else:
        norm = np.zeros_like(c_map, dtype=np.uint8)
    heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    # Annotate min / max c values
    cv2.putText(heatmap, f"c min={lo:.2f} max={hi:.2f}",
                (4, heatmap.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return heatmap


def _metrics_table(name, before, after):
    """Return a formatted string table comparing before/after metrics."""
    keys = [k for k in ("Pm", "mu", "Lm", "NIQE", "PSNR", "SSIM")
            if k in before and k in after]
    lines = [f"\n{'-'*52}", f"  {name}", f"{'-'*52}",
             f"  {'Metric':<8} {'Before':>9} {'After':>9} {'Δ':>9}",
             f"  {'-'*8} {'-'*9} {'-'*9} {'-'*9}"]
    for k in keys:
        b, a = before[k], after[k]
        delta = a - b
        lines.append(f"  {k:<8} {b:>9.4f} {a:>9.4f} {delta:>+9.4f}")
    lines.append(f"{'-'*52}")
    return "\n".join(lines)


# -- argument parsing ----------------------------------------------------------

def parse_args(argv):
    p = argparse.ArgumentParser(
        description="Compare global vs adaptive SmartEdge sharpening."
    )
    p.add_argument("--input",  "-i", required=True,  help="Input image path.")
    p.add_argument("--output", "-o", required=True,  help="Output comparison image path.")
    p.add_argument("--reference", "-r", default=None, help="Reference image (for PSNR/SSIM).")
    p.add_argument("--w", type=int, default=3, choices=[3, 5, 7])
    p.add_argument("--alpha", choices=["1", "auto"], default="1")
    p.add_argument("--block-size", type=int, default=64,
                   help="Block size in pixels for adaptive mode (default 64).")
    p.add_argument("--blur-half", action="store_true",
                   help="Apply Gaussian blur to left half to create a synthetic "
                        "mixed-blur test image.")
    p.add_argument("--blur-sigma", type=float, default=5.0,
                   help="Gaussian sigma for --blur-half (default 5).")
    p.add_argument("--save-cmap", default=None,
                   help="Save adaptive c-map heatmap to this path.")
    return p.parse_args(argv)


# -- main ----------------------------------------------------------------------

def main(argv):
    args = parse_args(argv)

    img, mode = load_image(args.input)
    ref = None
    if args.reference:
        ref, _ = load_image(args.reference)

    if args.blur_half:
        print(f"[compare] Applying Gaussian blur (sigma={args.blur_sigma}) "
              f"to left half of image to create mixed-blur test.")
        img = _blur_half(img)
        mixed_path = Path(args.output).with_stem(
            Path(args.output).stem + "_input_mixed"
        )
        save_image(mixed_path, img, mode)
        print(f"[compare] Mixed-blur input saved -> {mixed_path}")

    # -- sharpening ----------------------------------------------------------
    print("[compare] Running global (uniform c) sharpening...")
    sharp_global = smartedge_sharpen(
        img, w=args.w, alpha_mode=args.alpha,
        adaptive=False,
    )

    print(f"[compare] Running adaptive sharpening (block_size={args.block_size})...")
    sharp_adaptive = smartedge_sharpen(
        img, w=args.w, alpha_mode=args.alpha,
        adaptive=True, block_size=args.block_size,
    )

    # -- metrics -------------------------------------------------------------
    m_input    = evaluate(img, reference=ref)
    m_global   = evaluate(sharp_global, reference=ref)
    m_adaptive = evaluate(sharp_adaptive, reference=ref)

    print(_metrics_table("Global sharpening   (uniform c)", m_input, m_global))
    print(_metrics_table("Adaptive sharpening (c map)    ", m_input, m_adaptive))

    # Highlight key differences
    for key in ("Pm", "mu", "NIQE"):
        if key in m_global and key in m_adaptive:
            diff = m_adaptive[key] - m_global[key]
            better = {
                "Pm":   diff > 0,   # higher Pm = sharper edges
                "mu":   diff < 0,   # lower mu  = less blur
                "NIQE": diff < 0,   # lower NIQE = more natural
            }[key]
            tag = "^ adaptive better" if better else "v global better"
            print(f"  {key}: global={m_global[key]:.4f}  adaptive={m_adaptive[key]:.4f}  {tag}")

    # -- build comparison image -----------------------------------------------
    H = img.shape[0]
    panels = [
        _label_panel(_to_uint8_bgr(img),          "Input"),
        _label_panel(_to_uint8_bgr(sharp_global),  f"Global (uniform c)"),
        _label_panel(_to_uint8_bgr(sharp_adaptive), f"Adaptive (c map, b={args.block_size})"),
    ]

    # Optionally add c-map heatmap as 4th panel
    if args.save_cmap or True:  # always include in comparison strip
        if img.ndim == 2:
            l_orig = img.astype(np.float32)
        else:
            bgr = img.astype(np.float32)
            l_orig = (bgr[:, :, 0] + bgr[:, :, 1] + bgr[:, :, 2]) / 3.0
        l_smooth = median_filter_3x3(l_orig)
        cmap_vis = _make_cmap_vis(
            l_smooth, args.w, 0.05, args.block_size, True, True,
            (H, img.shape[1])
        )
        panels.append(_label_panel(cmap_vis, "Adaptive c map (hot=high c)"))

        if args.save_cmap:
            cv2.imwrite(args.save_cmap, cmap_vis)
            print(f"[compare] c-map heatmap saved -> {args.save_cmap}")

    # Resize all panels to the same height before concatenation
    target_h = max(p.shape[0] for p in panels)
    target_w = panels[0].shape[1]
    resized = []
    for p in panels:
        if p.shape[0] != target_h:
            p = cv2.resize(p, (target_w, target_h), interpolation=cv2.INTER_AREA)
        resized.append(p)

    comparison = np.hstack(resized)
    cv2.imwrite(args.output, comparison)
    print(f"[compare] Comparison image saved -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
