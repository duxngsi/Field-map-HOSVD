"""Generate a synthetic ``Field_5D.npy`` for the demo / experiments.

Usage
-----
    python examples/make_synthetic_field.py                  # default 20x60x15x15x3
    python examples/make_synthetic_field.py --noise 0.05     # add 5% noise
    python examples/make_synthetic_field.py --shape 30 80 20 20 3 -o my_field.npy

生成合成的 ``Field_5D.npy`` 供示例使用（项目未包含原始测量数据）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running straight from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hosvd.datasets import DEFAULT_SHAPE, add_noise, make_synthetic_field


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape",
        type=int,
        nargs="+",
        default=list(DEFAULT_SHAPE),
        help="tensor shape (default: %(default)s)",
    )
    parser.add_argument("--rank", type=int, default=5, help="effective separable rank")
    parser.add_argument(
        "--noise", type=float, default=0.0, help="relative Gaussian noise level (e.g. 0.05)"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "-o", "--output", default="Field_5D.npy", help="output .npy path"
    )
    args = parser.parse_args()

    field = make_synthetic_field(args.shape, rank=args.rank, seed=args.seed)
    if args.noise > 0.0:
        field = add_noise(field, level=args.noise, seed=args.seed + 1)

    out = Path(args.output)
    np.save(out, field)
    print(f"saved {out}  shape={field.shape}  "
          f"size={field.nbytes / 1e6:.1f} MB  noise={args.noise}")


if __name__ == "__main__":
    main()
