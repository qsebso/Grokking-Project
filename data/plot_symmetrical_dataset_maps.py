"""
Generate heatmaps of (a, b) -> outputs / categorical labels for mod-p operations.

Example (from repo root):
    python -m data.plot_symmetrical_dataset_maps
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

# package root = parent of data/
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data.dataset import OPERATIONS, compute_category_label  # noqa: E402


def _continuous_map(operation: str, p: int) -> np.ndarray:
    op_fn, _, _       = OPERATIONS[operation]
    grid = np.empty((p, p), dtype=np.float64)
    for a in range(p):
        for b in range(p):
            grid[a, b] = op_fn(a, b, p)
    return grid


def _categorical_map(operation: str, p: int, label_mode: str, label_mod: int = 3) -> np.ndarray:
    op_fn, _, _ = OPERATIONS[operation]
    grid = np.empty((p, p), dtype=np.int32)
    for a in range(p):
        for b in range(p):
            c = op_fn(a, b, p)
            grid[a, b] = compute_category_label(label_mode, a, b, c, p, label_mod)
    return grid


def plot_continuous(operation: str, p: int, out_path: str) -> None:
    g   = _continuous_map(operation, p)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(g, origin="lower", cmap="viridis", aspect="equal", vmin=0, vmax=p - 1)
    ax.set_xlabel("b", fontsize=12)
    ax.set_ylabel("a", fontsize=12)
    ax.set_title(f"{operation} dataset map (mod {p})\noutput c = op(a, b)", fontsize=12)
    plt.colorbar(im, ax=ax, label="output c = op(a, b)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def _discrete_cmap(n_classes: int):
    if n_classes == 2:
        colors = ["#31688e", "#fde725"]
    elif n_classes == 3:
        colors = ["#440154", "#21918c", "#fde725"]
    else:
        cmap_tab = mpl.colormaps["tab10"].resampled(n_classes)
        colors = [cmap_tab(i) for i in range(n_classes)]
    return mcolors.ListedColormap(colors)


def plot_categorical_panels(
    operation: str,
    p: int,
    out_path: str,
    modes: list[tuple[str, int]],
) -> None:
    n = len(modes)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6.5), squeeze=False)
    for ax, (mode, label_mod) in zip(axes[0], modes):
        g = _categorical_map(operation, p, mode, label_mod)
        n_cls = int(g.max()) + 1 if g.size else 0
        cmap = _discrete_cmap(n_cls)
        bounds = np.arange(-0.5, n_cls + 0.5, 1.0)
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        im = ax.imshow(g, origin="lower", cmap=cmap, norm=norm, aspect="equal", interpolation="nearest")
        ax.set_xlabel("b", fontsize=12)
        ax.set_ylabel("a", fontsize=12)
        ax.set_title(f"label_mode={mode}" + (f", label_mod={label_mod}" if mode in ("c_mod", "a_plus_b_mod") else ""), fontsize=11)
        cbar = plt.colorbar(im, ax=ax, ticks=np.arange(n_cls))
        cbar.set_label("class id (categorical target)")
    fig.suptitle(
        f"{operation} — categorical targets (mod {p})\n"
        f"(same op as continuous map; colors = CrossEntropy class index)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Plot dataset maps under data/dataset_pics/symmetrical/")
    ap.add_argument("--operation", default="3way_add_add_2_mul_mul")
    ap.add_argument("--p", type=int, default=97)
    ap.add_argument(
        "--out_dir",
        default=os.path.join(os.path.dirname(__file__), "dataset_pics"),
    )
    args = ap.parse_args()

    base = f"{args.operation}_dataset_map_p{args.p}.png"
    cat_out = os.path.join(args.out_dir, f"{args.operation}_categorical_dataset_map_p{args.p}.png")

    # Continuous reference (matches paper-style viridis map)
    plot_continuous(args.operation, args.p, os.path.join(args.out_dir, base))

    # Categorical: output-dependent labels (typical grokking-category probes)
    plot_categorical_panels(
        args.operation,
        args.p,
        cat_out,
        modes=[
            ("c_mod3", 3),
            ("c_parity", 3),
        ],
    )


if __name__ == "__main__":
    main()
