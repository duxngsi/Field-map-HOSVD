"""End-to-end HOSVD demo on a 5-D field map.

Reproduces (and modernises) the original ``Field_5D.py`` analysis:

* loads ``Field_5D.npy`` if present, otherwise generates a reproducible
  synthetic field so the demo is self-contained;
* compresses it with a truncated HOSVD and reports compression ratio,
  reconstruction error, SNR and timing;
* plots an original-vs-reconstructed field slice with its difference map,
  the smoothness (finite difference) of a 1-D line cut, and the per-mode
  singular-value spectra.

Usage
-----
    python examples/field_map_demo.py
    python examples/field_map_demo.py --ranks 3 8 4 4 3 --noise 0.05
    python examples/field_map_demo.py --no-show          # just save PNGs

中文：HOSVD 在 5 维场图上的端到端示例（压缩率 / 重建误差 / 信噪比 / 作图）。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Allow running straight from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hosvd import HOSVDCompressor, make_synthetic_field, add_noise
from hosvd.metrics import relative_error, snr_db


def load_field(path: str, noise: float, ranks: "list[int]") -> "tuple[np.ndarray, np.ndarray]":
    """Return ``(field, clean)`` -- ``clean`` is the noise-free reference if known.

    When no data file is found we generate a smooth field whose effective rank
    matches the smallest requested kept-rank, so the kept ranks capture all the
    signal and the compression genuinely *denoises* rather than just degrades.
    """
    p = Path(path)
    if p.exists():
        field = np.load(p)
        print(f"loaded {p}  shape={field.shape}")
        return field, field
    print(f"{p} not found -> generating synthetic field map")
    clean = make_synthetic_field(rank=max(1, min(ranks)))
    field = add_noise(clean, level=noise, seed=42) if noise > 0 else clean
    return field, clean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="Field_5D.npy", help="path to a 5-D .npy field")
    parser.add_argument(
        "--ranks", type=int, nargs="+", default=[3, 8, 4, 4, 3],
        help="per-mode kept ranks (default: %(default)s)",
    )
    parser.add_argument(
        "--noise", type=float, default=0.05,
        help="noise added to the synthetic field when no data file is found",
    )
    parser.add_argument("--method", choices=["gram", "svd"], default="gram")
    parser.add_argument("--no-show", action="store_true", help="save figures without displaying")
    parser.add_argument("--outdir", default="examples/output", help="figure output directory")
    args = parser.parse_args()

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid

    field, clean = load_field(args.data, args.noise, args.ranks)
    if field.ndim != 5:
        raise SystemExit(f"expected a 5-D field map, got shape {field.shape}")

    # ---- compress ---------------------------------------------------------- #
    t0 = time.perf_counter()
    model = HOSVDCompressor(ranks=args.ranks, method=args.method).fit(field)
    recovered = model.reconstruct()
    elapsed = time.perf_counter() - t0

    print("\n=== HOSVD compression report ===")
    print(f"original shape     : {field.shape}  ({np.prod(field.shape):,} values)")
    print(f"kept ranks         : {tuple(model.effective_ranks)}")
    print(f"core shape         : {model.core.shape}")
    print(f"stored values      : {model.n_stored_values:,}")
    print(f"compression ratio  : {model.compression_ratio:.1f} x")
    print(f"reconstruction err : {model.reconstruction_error(field) * 100:.3f} %  (vs input)")
    if clean is not field:
        print(f"denoising err      : {relative_error(clean, recovered) * 100:.3f} %  (vs clean)")
        print(f"                     input noise was {relative_error(clean, field) * 100:.3f} %")
    print(f"SNR (vs input)     : {snr_db(field, recovered):.1f} dB")
    print(f"elapsed            : {elapsed * 1000:.1f} ms")

    # ---- figures ----------------------------------------------------------- #
    plt.rcParams.update({"font.size": 12})
    nx, nz, ny, nt, nc = field.shape
    mid_x, mid_t = nx // 2, nt // 2
    comp_a, comp_b = nc - 1, min(1, nc - 1)  # e.g. Ez, Ey

    def slice2d(data, comp):
        return data[mid_x, :, :, mid_t, comp]

    a, b = slice2d(field, comp_a), slice2d(recovered, comp_a)
    c, d = slice2d(field, comp_b), slice2d(recovered, comp_b)
    sa = np.abs(a).max() or 1.0
    sc = np.abs(c).max() or 1.0

    # Figure 1: field slices + difference maps
    fig1 = plt.figure(figsize=(11, 4.5))
    grid_l = ImageGrid(fig1, 121, nrows_ncols=(1, 2), axes_pad=0.15,
                       share_all=True, cbar_location="right", cbar_mode="single",
                       cbar_size="8%", cbar_pad=0.1)
    grid_l[0].imshow(a / sa, vmin=-1, vmax=1, origin="lower")
    grid_l[0].set_title("original"); grid_l[0].set_xlabel("y"); grid_l[0].set_ylabel("z")
    im = grid_l[1].imshow(b / sa, vmin=-1, vmax=1, origin="lower")
    grid_l[1].set_title("recovered")
    grid_l[1].cax.colorbar(im)

    srb = 0.01
    grid_r = ImageGrid(fig1, 122, nrows_ncols=(1, 2), axes_pad=0.15,
                       share_all=True, cbar_location="right", cbar_mode="single",
                       cbar_size="8%", cbar_pad=0.1)
    grid_r[0].imshow((a - b) / sa, vmin=-srb, vmax=srb, cmap="RdBu", origin="lower")
    grid_r[0].set_title(r"$\Delta E_z$"); grid_r[0].set_xlabel("y"); grid_r[0].set_ylabel("z")
    im2 = grid_r[1].imshow((c - d) / sc, vmin=-srb, vmax=srb, cmap="RdBu", origin="lower")
    grid_r[1].set_title(r"$\Delta E_y$")
    cbar = grid_r[1].cax.colorbar(im2, ticks=[-srb, 0, srb], format="%.0e")
    cbar.ax.set_yticklabels([f"< -{srb*100:g}%", "0", f"> {srb*100:g}%"])
    fig1.suptitle(r"$E_z$ field slice: original vs HOSVD reconstruction (right: difference maps)")

    # Figure 2: smoothness (finite difference of a longitudinal line cut)
    line_o = field[mid_x, :, ny // 2, mid_t, comp_a]
    line_r = recovered[mid_x, :, ny // 2, mid_t, comp_a]
    fig2, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 5))
    ax1.plot(line_o, label="original")
    ax1.plot(line_r, "--", label="reconstructed")
    ax1.set_ylabel(r"$E_z$"); ax1.legend(loc="upper right")
    ax2.plot(np.gradient(line_o), label="original")
    ax2.plot(np.gradient(line_r), "--", label="reconstructed")
    ax2.set_ylabel(r"$\delta E_z / \delta z$"); ax2.set_xlabel("z-step")
    ax2.legend(loc="upper right")
    fig2.suptitle("Smoothness of a 1-D line cut and its derivative")
    fig2.tight_layout()

    # Figure 3: per-mode singular-value spectra (scree)
    fig3, axes = plt.subplots(nx_modes := field.ndim, 1, figsize=(6, 1.6 * field.ndim),
                              sharex=False)
    for mode, ax in enumerate(np.atleast_1d(axes)):
        spectrum = model.singular_values[mode]
        ax.semilogy(spectrum, ".-")
        ax.axvline(model.effective_ranks[mode] - 0.5, color="r", ls=":",
                   label=f"rank={model.effective_ranks[mode]}")
        ax.set_ylabel(f"mode {mode}")
        ax.legend(loc="upper right", fontsize=9)
    axes_flat = np.atleast_1d(axes)
    axes_flat[-1].set_xlabel("singular value index")
    fig3.suptitle("Per-mode singular-value spectra")
    fig3.tight_layout()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for name, fig in [("field_slices", fig1), ("smoothness", fig2), ("singular_values", fig3)]:
        path = outdir / f"{name}.png"
        fig.savefig(path, dpi=130, bbox_inches="tight")
        print(f"saved {path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
