import argparse
import sys
from pathlib import Path

from input import load_image, save_image
from sharpen import smartedge_sharpen


def parse_args(argv):
    p = argparse.ArgumentParser(
        description="SmartEdge: sharpening using gradient + Retinex-inspired contrast."
    )
    p.add_argument("--input", "-i", required=True, help="Input image path.")
    p.add_argument("--output", "-o", required=True, help="Output image path.")
    p.add_argument("--w", type=int, default=3, choices=[3, 5, 7], help="Window size.")
    p.add_argument("--epsilon", type=float, default=0.05, help="R threshold epsilon.")
    p.add_argument(
        "--alpha",
        choices=["1", "auto"],
        default="1",
        help="Use alpha=1 or adaptive alpha*.",
    )
    p.add_argument(
        "--no-post-median",
        action="store_true",
        help="Disable post-convolution median filter.",
    )
    p.add_argument(
        "--keep-outliers",
        action="store_true",
        help="Keep top-2% rho outliers.",
    )
    p.add_argument(
        "--keep-small",
        action="store_true",
        help="Keep small isolated components in R.",
    )
    p.add_argument(
        "--adaptive",
        action="store_true",
        help="Use spatially-adaptive c map (block-level). Addresses the paper's "
             "limitation of uniform sharpening strength across mixed-blur images.",
    )
    p.add_argument(
        "--block-size",
        type=int,
        default=64,
        help="Block size in pixels for --adaptive mode (default: 64).",
    )
    return p.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    in_path = Path(args.input)
    out_path = Path(args.output)

    img, mode = load_image(in_path)
    out = smartedge_sharpen(
        img,
        w=args.w,
        epsilon=args.epsilon,
        alpha_mode=args.alpha,
        post_median=not args.no_post_median,
        remove_outliers=not args.keep_outliers,
        remove_small=not args.keep_small,
        adaptive=args.adaptive,
        block_size=args.block_size,
    )
    save_image(out_path, out, "gray" if mode == "gray" else "bgr")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
