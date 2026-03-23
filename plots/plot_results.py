"""
plot_results.py — Graph grokking experiment results
====================================================

Reads JSON files from the results/ directory and produces plots.

USAGE
-----
# Plot everything in results/
    python plot_results.py

# Plot only specific files
    python plot_results.py results/sub_p97_wd0.0*.json results/sub_p97_wd1.0*.json

# One big grid of all runs (accuracy + loss side by side)
    python plot_results.py --mode grid

# Overlay val accuracy for a sweep comparison
    python plot_results.py --mode sweep --sweep_param weight_decay

# Single run: accuracy + loss side by side (like the notebook)
    python plot_results.py --mode single results/add_p97_wd1.0*.json

FLAGS
-----
--mode      single | sweep | grid   (default: auto-detects)
--sweep_param   which config field to use as the legend label in sweep mode
--metric    acc | loss              (default: acc)
--out       output image path       (default: plot.png)
--no_show   don't open the window, just save
"""

import os
import sys
import json
import glob
import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Load helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_result(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _branch_plot_labels(data: dict) -> tuple[str, str, str, str]:
    """Legend strings for the two branch accuracy curves (train/val × branch)."""
    s = data.get("summary") or {}
    b1 = data.get("branch_label_1") or s.get("branch_label_1") or "Odd"
    b2 = data.get("branch_label_2") or s.get("branch_label_2") or "Even"
    return (f"Train {b1}", f"Train {b2}", f"Val {b1}", f"Val {b2}")


def find_results(patterns) -> list[str]:
    paths = []
    for pat in patterns:
        matched = sorted(glob.glob(pat))
        paths.extend(matched)
    if not paths:
        # default: everything in results/
        paths = sorted(glob.glob("results/*.json"))
    if not paths:
        print("No result JSON files found. Run main.py first.")
        sys.exit(1)
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────

COLORS = [
    "#3B82F6",  # blue
    "#F97316",  # orange
    "#10B981",  # green
    "#EF4444",  # red
    "#8B5CF6",  # purple
    "#EC4899",  # pink
    "#14B8A6",  # teal
    "#F59E0B",  # amber
]

def get_color(i):
    return COLORS[i % len(COLORS)]


# ─────────────────────────────────────────────────────────────────────────────
# Plot modes
# ─────────────────────────────────────────────────────────────────────────────

def plot_single(data: dict, out: str, show: bool):
    """Accuracy + Loss side by side for one run (matches notebook style)."""
    epochs = data["log_epochs"]
    cfg    = data["summary"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── accuracy ──────────────────────────────────────────────────────────
    ax1.plot(epochs, data["train_accs"],  label="Train Acc",  color="#3B82F6", lw=2)
    ax1.plot(epochs, data["val_accs"],    label="Val Acc",    color="#F97316", lw=2)
    if (
        "train_odd_accs" in data
        and "train_even_accs" in data
        and "val_odd_accs" in data
        and "val_even_accs" in data
    ):
        tl1, tl2, vl1, vl2 = _branch_plot_labels(data)
        ax1.plot(epochs, data["train_odd_accs"], label=tl1,  color="#3B82F6",
                 lw=1.2, linestyle="--", alpha=0.8)
        ax1.plot(epochs, data["train_even_accs"], label=tl2, color="#3B82F6",
                 lw=1.2, linestyle=":", alpha=0.8)
        ax1.plot(epochs, data["val_odd_accs"], label=vl1,  color="#F97316",
                 lw=1.2, linestyle="--", alpha=0.8)
        ax1.plot(epochs, data["val_even_accs"], label=vl2, color="#F97316",
                 lw=1.2, linestyle=":", alpha=0.8)

    _mark_grokking(ax1, cfg)

    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_title(_make_title(cfg, "Accuracy"), fontsize=13)
    ax1.legend(fontsize=11)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(alpha=0.3)

    # ── loss ──────────────────────────────────────────────────────────────
    ax2.plot(epochs, data["train_losses"], label="Train Loss", color="#3B82F6", lw=2)
    ax2.plot(epochs, data["val_losses"],   label="Val Loss",   color="#F97316", lw=2)

    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Loss",  fontsize=12)
    ax2.set_title(_make_title(cfg, "Loss"), fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    if show:
        plt.show()
    plt.close()


def plot_sweep(results: list[dict], sweep_param: str, metric: str,
               out: str, show: bool):
    """
    Overlay val accuracy (or loss) for multiple runs —
    ideal for weight-decay / lr / operation sweeps.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, data in enumerate(results):
        cfg    = data["summary"]
        epochs = data["log_epochs"]
        label  = f"{sweep_param}={cfg.get(sweep_param, '?')}"
        color  = get_color(i)

        if metric == "loss":
            ax.plot(epochs, data["val_losses"],  label=label, color=color, lw=2)
        else:
            ax.plot(epochs, data["val_accs"],    label=label, color=color, lw=2)
            # faint train acc
            ax.plot(epochs, data["train_accs"],  color=color, lw=1,
                    linestyle="--", alpha=0.35)
            # optional branch curves (faint) when present
            if "val_odd_accs" in data and "val_even_accs" in data:
                ax.plot(epochs, data["val_odd_accs"], color=color, lw=0.9,
                        linestyle=":", alpha=0.25)
                ax.plot(epochs, data["val_even_accs"], color=color, lw=0.9,
                        linestyle="-.", alpha=0.25)

        # vertical line at grokking epoch
        grok = cfg.get("grok_epoch")
        if grok:
            ax.axvline(grok, color=color, lw=1, linestyle=":", alpha=0.7)

    ylabel = "Val Loss" if metric == "loss" else "Accuracy  (solid=val, dashed=train)"
    op     = results[0]["summary"].get("operation", "?")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(ylabel,  fontsize=12)
    ax.set_title(
        f"Sweep: {sweep_param}  |  op={op}  |  "
        f"p={results[0]['summary'].get('p','?')}  |  "
        f"train_frac={results[0]['summary'].get('train_frac','?')}",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    if metric == "acc":
        ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    if show:
        plt.show()
    plt.close()


def plot_grid(results: list[dict], metric: str, out: str, show: bool):
    """
    One subplot per run, 2 columns wide.
    Good for seeing all experiments at a glance.
    """
    n    = len(results)
    ncol = 2
    nrow = (n + 1) // ncol

    fig, axes = plt.subplots(nrow, ncol, figsize=(14, 4 * nrow))
    axes = np.array(axes).flatten()   # always 1-D

    for i, data in enumerate(results):
        ax     = axes[i]
        cfg    = data["summary"]
        epochs = data["log_epochs"]

        if metric == "loss":
            ax.plot(epochs, data["train_losses"], label="Train", color="#3B82F6", lw=1.5)
            ax.plot(epochs, data["val_losses"],   label="Val",   color="#F97316", lw=1.5)
        else:
            ax.plot(epochs, data["train_accs"],   label="Train", color="#3B82F6", lw=1.5)
            ax.plot(epochs, data["val_accs"],     label="Val",   color="#F97316", lw=1.5)
            if (
                "train_odd_accs" in data
                and "train_even_accs" in data
                and "val_odd_accs" in data
                and "val_even_accs" in data
            ):
                tl1, tl2, vl1, vl2 = _branch_plot_labels(data)
                ax.plot(epochs, data["train_odd_accs"], label=tl1, color="#3B82F6",
                        lw=1.0, linestyle="--", alpha=0.75)
                ax.plot(epochs, data["train_even_accs"], label=tl2, color="#3B82F6",
                        lw=1.0, linestyle=":", alpha=0.75)
                ax.plot(epochs, data["val_odd_accs"], label=vl1, color="#F97316",
                        lw=1.0, linestyle="--", alpha=0.75)
                ax.plot(epochs, data["val_even_accs"], label=vl2, color="#F97316",
                        lw=1.0, linestyle=":", alpha=0.75)
            ax.set_ylim(-0.05, 1.05)

        _mark_grokking(ax, cfg, fontsize=8)

        ax.set_title(_short_title(cfg), fontsize=9)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(
        f"All runs  |  metric={metric}",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    if show:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Annotation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mark_grokking(ax, cfg: dict, fontsize=9):
    memo = cfg.get("memo_epoch")
    grok = cfg.get("grok_epoch")
    gap  = cfg.get("grok_gap")

    if memo:
        ax.axvline(memo, color="#6B7280", lw=1, linestyle="--", alpha=0.6)
        ax.text(memo, 0.05, f" memo\n {memo}", fontsize=fontsize,
                color="#6B7280", va="bottom")
    if grok:
        ax.axvline(grok, color="#10B981", lw=1.5, linestyle="--", alpha=0.8)
        ax.text(grok, 0.5, f" grok\n {grok}", fontsize=fontsize,
                color="#10B981", va="bottom")
    if gap is not None:
        ax.text(0.98, 0.08, f"gap={gap}", fontsize=fontsize,
                transform=ax.transAxes, ha="right", color="#374151",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))


def _readable_name(cfg: dict) -> str:
    """Human-readable one-liner built from the config fields."""
    op   = cfg.get("operation", "?")
    wd   = cfg.get("weight_decay", "?")
    lr   = cfg.get("lr", "?")
    d    = cfg.get("d_model", "?")
    tf   = cfg.get("train_frac", "?")
    ep   = cfg.get("num_epochs", "?")
    p    = cfg.get("p", "?")
    return f"{op} mod {p}  |  wd={wd}  lr={lr}  d={d}  train={int(float(tf)*100)}%  ep={ep}"


def _make_title(cfg: dict, kind: str) -> str:
    return f"{_readable_name(cfg)}\n{kind} over Training"


def _short_title(cfg: dict) -> str:
    return _readable_name(cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Auto-detect mode
# ─────────────────────────────────────────────────────────────────────────────

def auto_mode(n: int) -> str:
    if n == 1:
        return "single"
    if n <= 8:
        return "sweep"
    return "grid"


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Plot grokking experiment results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("files", nargs="*",
                   help="JSON result files (globs ok). Default: all in results/")
    p.add_argument("--mode", choices=["single", "sweep", "grid"], default=None,
                   help="Plot style. Default: auto-detected from file count")
    p.add_argument("--sweep_param", default="weight_decay",
                   help="Config field to use as legend in sweep mode")
    p.add_argument("--metric", choices=["acc", "loss"], default="acc")
    p.add_argument("--out",  default="plot.png",
                   help="Output image path")
    p.add_argument("--no_show", action="store_true",
                   help="Save only, don't open a window")
    args = p.parse_args()

    # ── load ──────────────────────────────────────────────────────────────
    paths   = find_results(args.files if args.files else [])
    results = [load_result(path) for path in paths]
    print(f"Loaded {len(results)} result(s):")
    for path in paths:
        s = load_result(path)["summary"]
        print(f"  {Path(path).name}  →  memo={s['memo_epoch']}  grok={s['grok_epoch']}  gap={s['grok_gap']}")

    # ── mode ──────────────────────────────────────────────────────────────
    mode = args.mode or auto_mode(len(results))
    print(f"Mode: {mode}")

    show = not args.no_show

    # ── output path ───────────────────────────────────────────────────────
    os.makedirs("plots", exist_ok=True)
    if args.out != "plot.png":
        # user specified a custom path — honour it
        out_path = args.out
    else:
        # auto-name from the configs
        cfg0 = results[0]["summary"]
        op   = cfg0.get("operation", "run")
        ep   = cfg0.get("num_epochs", "")
        tf   = int(float(cfg0.get("train_frac", 0.5)) * 100)
        if mode == "sweep":
            out_path = f"plots/{op}_sweep_{args.sweep_param}_tf{tf}_ep{ep}.png"
        elif mode == "single":
            wd  = cfg0.get("weight_decay", "")
            lr  = cfg0.get("lr", "")
            out_path = f"plots/{op}_wd{wd}_lr{lr}_tf{tf}_ep{ep}.png"
        else:
            out_path = f"plots/grid_{op}_tf{tf}_ep{ep}.png"

    if mode == "single":
        plot_single(results[0], out_path, show)
    elif mode == "sweep":
        plot_sweep(results, args.sweep_param, args.metric, out_path, show)
    else:  # grid
        plot_grid(results, args.metric, out_path, show)


if __name__ == "__main__":
    main()