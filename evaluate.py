"""
Evaluation script for the SmartEdge sharpening filter.

Compares all six paper metrics (Pm, mu, Lm, NIQE, PSNR, SSIM) before and
after sharpening, and prints a summary table showing whether each metric
changed in the expected direction.

Usage
-----
  python evaluate.py --input image.jpg
  python evaluate.py --input image.jpg --reference ref.jpg
  python evaluate.py --input image.jpg --w 5 --alpha auto
"""

import argparse
import sys
from pathlib import Path

from input import load_image
from sharpen import smartedge_sharpen
from metrics import evaluate


EXPECTED = {
    "Pm":   "increase",
    "mu":   "decrease",
    "Lm":   "stable",
    "NIQE": "stable",
    "PSNR": "increase",
    "SSIM": "increase",
}


def _arrow(before, after, expect, key=None):
    diff = after - before
    if expect == "increase":
        ok = diff > 0
    elif expect == "decrease":
        ok = diff < 0
    elif expect == "stable":
        # Paper tolerances: Lm should not change by more than ~2 units
        # on a 0-255 scale; NIQE changes up to ~2.5 are acceptable.
        if key == "Lm":
            ok = abs(diff) < 2.0
        elif key == "NIQE":
            ok = abs(diff) < 2.5
        else:
            ok = abs(diff) / max(abs(before), 1e-10) < 0.10
    else:
        ok = True
    symbol = "PASS" if ok else "FAIL"
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.4f}", symbol


def run(args):
    img, mode = load_image(Path(args.input))

    ref = None
    if args.reference:
        ref, _ = load_image(Path(args.reference))

    before = evaluate(img, reference=ref)

    sharpened = smartedge_sharpen(
        img,
        w=args.w,
        epsilon=args.epsilon,
        alpha_mode=args.alpha,
        post_median=not args.no_post_median,
        remove_outliers=not args.keep_outliers,
        remove_small=not args.keep_small,
    )

    after = evaluate(sharpened, reference=ref)

    # ---- Print results ----
    header = f"{'Metric':<8} {'Before':>12} {'After':>12} {'Change':>12} {'Expected':>10} {'Result':>8}"
    print()
    print("=" * len(header))
    print(f"  SmartEdge Evaluation  (w={args.w}, alpha={args.alpha})")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for key in ["Pm", "mu", "Lm", "NIQE", "PSNR", "SSIM"]:
        if key not in before:
            continue
        b = before[key]
        a = after[key]
        change, result = _arrow(b, a, EXPECTED[key], key=key)
        print(f"{key:<8} {b:>12.4f} {a:>12.4f} {change:>12} {EXPECTED[key]:>10} {result:>8}")

    print("-" * len(header))
    print()


def parse_args(argv):
    p = argparse.ArgumentParser(description="Evaluate SmartEdge sharpening metrics.")
    p.add_argument("--input", "-i", required=True, help="Input image path.")
    p.add_argument("--reference", "-r", default=None, help="Reference image for PSNR/SSIM.")
    p.add_argument("--w", type=int, default=3, choices=[3, 5, 7], help="Window size.")
    p.add_argument("--epsilon", type=float, default=0.05, help="R threshold epsilon.")
    p.add_argument("--alpha", choices=["1", "auto"], default="1", help="Alpha mode.")
    p.add_argument("--no-post-median", action="store_true")
    p.add_argument("--keep-outliers", action="store_true")
    p.add_argument("--keep-small", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args(sys.argv[1:]))
