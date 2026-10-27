from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def plot_nt_disk(input_path, output_path=None, *, dpi=120):
    """
    Four-panel visualization of a saved NT disk render

    Parameters
    ----------
    input_path : path-like
        NPZ arrays: intensity, g, image_order, found
    output_path : path-like or None
        PNG path; default: input_path with a .png suffix
    dpi : positive integer
        Output resolution; default: 2160 x 1800 pixels

    Returns
    -------
    output_path : Path
        Saved PNG path

    Shapes
    ------
    intensity, g, image_order, found : (height, width)

    Notes
    -----
    Panels: logarithmic intensity, linear intensity, redshift, image order
    Pixel origin: lower left; interpolation: nearest
    Log lower bound: max(minimum positive intensity, peak intensity * 1e-4)
    Background: white for log intensity / redshift, black for linear / order
    """
    if dpi < 1:
        raise ValueError("dpi must be positive.")
    input_path = Path(input_path).expanduser()
    if not input_path.is_absolute():
        input_path = REPO_ROOT / input_path
    output_path = Path(output_path).expanduser() if output_path else input_path.with_suffix(".png")
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input NPZ and output PNG paths must differ.")

    with np.load(input_path, allow_pickle=False) as data:
        intensity = data["intensity"]
        g = data["g"]
        order = data["image_order"]
        found = np.asarray(data["found"], dtype=bool)
    if intensity.ndim != 2 or not intensity.size or any(
        array.shape != intensity.shape for array in (g, order, found)
    ):
        raise ValueError("All four render maps must have the same nonempty 2D shape.")

    positive = found & np.isfinite(intensity) & (intensity > 0)
    values = intensity[positive]
    if values.size:
        log_max = float(values.max())
        log_min = max(float(values.min()), log_max * 1e-4)
        if log_min == log_max:
            log_min = log_max / 10.0
    else:
        log_min, log_max = 1e-12, 1.0
    linear_max = float(values.max()) if values.size else 1.0
    valid_g = found & np.isfinite(g)
    g_max = max(1.4, float(g[valid_g].max())) if valid_g.any() else 1.4
    max_order = int(order[found].max()) if found.any() else 0
    n_orders = max(3, max_order + 1)

    intensity_cmap = plt.get_cmap("inferno").copy()
    intensity_cmap.set_bad("white")
    redshift_cmap = plt.get_cmap("viridis").copy()
    redshift_cmap.set_bad("white")
    palette = ["tab:blue", "tab:purple", "tab:cyan", "tab:orange", "tab:green",
               "tab:red", "tab:brown", "tab:pink", "tab:olive", "tab:gray"]
    order_cmap = ListedColormap([palette[i % len(palette)] for i in range(n_orders)])
    order_cmap.set_bad("#161616")
    order_norm = BoundaryNorm(np.arange(-0.5, n_orders + 0.5, 1), order_cmap.N)
    height, width = intensity.shape

    with plt.rc_context({"font.size": 13, "axes.titlesize": 16, "axes.labelsize": 13,
                         "figure.titlesize": 22}):
        fig, axes = plt.subplots(2, 2, figsize=(18, 15), constrained_layout=True)
        try:
            fig.suptitle(f"GRRT rendering - {input_path.name} ({width} x {height})")
            image = axes[0, 0].imshow(
                np.ma.masked_where(~positive, intensity), origin="lower", interpolation="nearest",
                cmap=intensity_cmap, norm=LogNorm(vmin=log_min, vmax=log_max),
            )
            axes[0, 0].set_title("Observed intensity (log scale)")
            fig.colorbar(image, ax=axes[0, 0], shrink=0.85)

            image = axes[0, 1].imshow(
                np.ma.masked_invalid(intensity), origin="lower", interpolation="nearest",
                cmap=intensity_cmap, vmin=0, vmax=linear_max,
            )
            axes[0, 1].set_title("Observed intensity (linear scale)")
            fig.colorbar(image, ax=axes[0, 1], shrink=0.85)

            image = axes[1, 0].imshow(
                np.ma.masked_where(~valid_g, g), origin="lower", interpolation="nearest",
                cmap=redshift_cmap, vmin=0, vmax=g_max,
            )
            axes[1, 0].set_title("Redshift factor g (valid disk hits)")
            fig.colorbar(image, ax=axes[1, 0], shrink=0.85)

            image = axes[1, 1].imshow(
                np.ma.masked_where(~found | (order < 0), order),
                origin="lower", interpolation="nearest", cmap=order_cmap, norm=order_norm,
            )
            axes[1, 1].set_title("Image order (-1 = no disk hit)")
            fig.colorbar(image, ax=axes[1, 1], shrink=0.85,
                         ticks=np.arange(n_orders), label="Order")
            for ax in axes.flat:
                ax.set_xlabel("Pixel x")
                ax.set_ylabel("Pixel y")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=dpi)
        finally:
            plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Plot the four maps from an NT disk render NPZ.")
    parser.add_argument("input", help="Input NPZ path, relative to the repository root or absolute.")
    parser.add_argument("--output", help="Output PNG path; default: next to the input NPZ.")
    parser.add_argument("--dpi", type=int, default=120)
    args = parser.parse_args()
    if args.dpi < 1:
        parser.error("dpi must be positive.")
    output_path = plot_nt_disk(args.input, args.output, dpi=args.dpi)
    print(f"Figure saved: {output_path}")


if __name__ == "__main__":
    main()
