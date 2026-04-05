"""
plot_results.py — Graph grokking experiment results
====================================================

Reads JSON files from ``results/`` (and legacy ``results_factor/`` if ``results/`` is empty).

USAGE
-----
# Plot everything in results/ (main.py, train_category, train_category_factor)
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


def _model_type(cfg: dict) -> str:
    """Architecture tag from saved summary; older JSONs default to standard."""
    mt = cfg.get("model_type")
    if mt is None or mt == "":
        return "standard"
    return str(mt)


def _infer_plot_model_tag(results: list[dict]) -> str:
    """Filename fragment for auto-named PNGs; ``mixed`` if runs disagree."""
    mts = {_model_type(r["summary"]) for r in results}
    if len(mts) == 1:
        return next(iter(mts))
    return "mixed"


def _branch_plot_labels(data: dict) -> tuple[str, str, str, str]:
    """Legend strings for the two branch accuracy curves (train/val × branch)."""
    s = data.get("summary") or {}
    b1 = data.get("branch_label_1") or s.get("branch_label_1") or "Odd"
    b2 = data.get("branch_label_2") or s.get("branch_label_2") or "Even"
    return (f"Train {b1}", f"Train {b2}", f"Val {b1}", f"Val {b2}")


def _infer_task(data: dict, cfg: dict) -> str:
    t = data.get("task")
    if t:
        return str(t)
    if cfg.get("label_mode") is not None:
        return "categorical"
    return "regression"


def _noise_segment(cfg: dict) -> str | None:
    """Train label noise from summary (asymmetric / symmetric); None if both zero or absent."""
    try:
        ln = float(cfg.get("label_noise") or 0)
    except (TypeError, ValueError):
        ln = 0.0
    try:
        lns = float(cfg.get("label_noise_sym") or 0)
    except (TypeError, ValueError):
        lns = 0.0
    bits: list[str] = []
    if ln > 0:
        bits.append(f"noise_asy={ln}")
    if lns > 0:
        bits.append(f"noise_sym={lns}")
    return "  |  ".join(bits) if bits else None


def _task_line(data: dict, cfg: dict | None = None) -> str:
    """Experiment kind: task name, label_mode, num classes (and label_mod when relevant)."""
    cfg = cfg if cfg is not None else (data.get("summary") or {})
    parts: list[str] = [f"task={_infer_task(data, cfg)}"]
    lm = cfg.get("label_mode")
    if lm is not None:
        parts.append(f"label={lm}")
    nc = cfg.get("num_classes")
    if nc is not None:
        parts.append(f"classes={nc}")
    lmod = cfg.get("label_mod")
    if lm in ("c_mod", "a_plus_b_mod") and lmod is not None:
        parts.append(f"label_mod={lmod}")
    rc = cfg.get("rule_count", 1)
    if isinstance(rc, int) and rc > 1:
        parts.append(f"rule_count={rc}")
    rm = cfg.get("routing_mode")
    if rm:
        parts.append(f"routing={rm}")
    schl = cfg.get("shared_c_head_layers")
    if schl is not None and _model_type(cfg) == "standard":
        parts.append(f"shared_c_head_layers={schl}")
    chc = cfg.get("c_head_count")
    if chc is not None and _model_type(cfg) == "routed_modular":
        parts.append(f"c_head_count={chc}")
    chl = cfg.get("routed_c_head_layers")
    if chl is not None and _model_type(cfg) == "routed_modular":
        parts.append(f"c_head_layers={chl}")
    infmt = cfg.get("input_format")
    if infmt and infmt != "a_op_b_eq":
        parts.append(f"fmt={infmt}")
    line = "  |  ".join(parts)
    nseg = _noise_segment(cfg)
    if nseg:
        line = f"{line}  |  {nseg}"
    return line


def find_results(patterns) -> list[str]:
    paths = []
    for pat in patterns:
        matched = sorted(glob.glob(pat))
        paths.extend(matched)
    if not paths:
        # default: unified results/ (main, train_category, train_category_factor)
        paths = sorted(glob.glob("results/*.json"))
    if not paths:
        legacy = sorted(glob.glob("results_factor/*.json"))
        if legacy:
            print(
                "Note: no JSON in results/; using results_factor/*.json. "
                "New runs default to --results_dir results."
            )
            paths = legacy
    if not paths:
        print("No result JSON files found. Run main.py (or train_category / train_category_factor) first.")
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


def _has_routed_analysis(data: dict) -> bool:
    return isinstance(data.get("routed_analysis"), dict)


def _matrix_to_array(rows, width: int) -> np.ndarray:
    if not rows:
        return np.zeros((0, width), dtype=float)
    mat = []
    for row in rows:
        vals = []
        for v in row:
            vals.append(np.nan if v is None else float(v))
        mat.append(vals)
    return np.array(mat, dtype=float)


def _plot_routed_heatmap(
    ax,
    matrix: np.ndarray,
    title: str,
    *,
    xlabels: list[str],
    ylabels: list[str],
    vmin: float,
    vmax: float,
    cmap: str = "viridis",
    colorbar_label: str = "",
):
    if matrix.size == 0:
        ax.axis("off")
        return
    im = ax.imshow(matrix, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Head")
    ax.set_ylabel("True rule")
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=0)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label=colorbar_label)


def plot_routed_analysis(data: dict, out: str, show: bool):
    routed = data.get("routed_analysis") or {}
    train = routed.get("train") or {}
    val = routed.get("val") or {}
    if not train or not val:
        return

    cfg = data.get("summary") or {}
    head_labels = routed.get("head_labels") or [f"head_{i}" for i in range(len(train.get("head_usage_freq") or []))]
    rule_labels = routed.get("rule_labels") or [f"rule_{i}" for i in range(len(train.get("true_rule_counts") or []))]

    fig, axes = plt.subplots(3, 4, figsize=(20, 13))

    x = np.arange(len(head_labels))
    axes[0, 0].bar(x, train.get("head_usage_freq") or [], color="#3B82F6")
    axes[0, 0].set_title("Train Head Usage")
    axes[0, 0].set_xticks(x, head_labels)
    axes[0, 0].set_ylim(0, 1.0)
    axes[0, 0].set_ylabel("Fraction")

    axes[0, 1].bar(x, val.get("head_usage_freq") or [], color="#F97316")
    axes[0, 1].set_title("Val Head Usage")
    axes[0, 1].set_xticks(x, head_labels)
    axes[0, 1].set_ylim(0, 1.0)
    axes[0, 1].set_ylabel("Fraction")

    tr_ent = train.get("routing_entropy_norm_by_true_rule") or []
    va_ent = val.get("routing_entropy_norm_by_true_rule") or []
    tr_ent_vals = [np.nan if v is None else float(v) for v in tr_ent]
    va_ent_vals = [np.nan if v is None else float(v) for v in va_ent]
    axes[0, 2].bar(np.arange(len(rule_labels)), tr_ent_vals, color="#10B981")
    axes[0, 2].axhline(float(train.get("routing_entropy_norm_mean") or 0.0), color="#374151", linestyle="--", lw=1.2)
    axes[0, 2].set_title("Train Routing Entropy (normalized)")
    axes[0, 2].set_xticks(range(len(rule_labels)), rule_labels)
    axes[0, 2].set_ylim(0, 1.05)
    axes[0, 2].set_ylabel("Entropy / log(rule_count)")

    axes[0, 3].bar(np.arange(len(rule_labels)), va_ent_vals, color="#F59E0B")
    axes[0, 3].axhline(float(val.get("routing_entropy_norm_mean") or 0.0), color="#374151", linestyle="--", lw=1.2)
    axes[0, 3].set_title("Val Routing Entropy (normalized)")
    axes[0, 3].set_xticks(range(len(rule_labels)), rule_labels)
    axes[0, 3].set_ylim(0, 1.05)
    axes[0, 3].set_ylabel("Entropy / log(rule_count)")

    _plot_routed_heatmap(
        axes[1, 0],
        _matrix_to_array(train.get("mean_routing_weights_by_true_rule") or [], len(head_labels)),
        "Train Mean Routing Weights by True Rule",
        xlabels=head_labels,
        ylabels=rule_labels,
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
        colorbar_label="Weight",
    )
    _plot_routed_heatmap(
        axes[1, 1],
        _matrix_to_array(val.get("mean_routing_weights_by_true_rule") or [], len(head_labels)),
        "Val Mean Routing Weights by True Rule",
        xlabels=head_labels,
        ylabels=rule_labels,
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
        colorbar_label="Weight",
    )
    _plot_routed_heatmap(
        axes[1, 2],
        _matrix_to_array(train.get("true_rule_vs_chosen_head_freq") or [], len(head_labels)),
        "Train True Rule vs Chosen Head",
        xlabels=head_labels,
        ylabels=rule_labels,
        vmin=0.0,
        vmax=1.0,
        cmap="magma",
        colorbar_label="Fraction",
    )
    _plot_routed_heatmap(
        axes[1, 3],
        _matrix_to_array(val.get("true_rule_vs_chosen_head_freq") or [], len(head_labels)),
        "Val True Rule vs Chosen Head",
        xlabels=head_labels,
        ylabels=rule_labels,
        vmin=0.0,
        vmax=1.0,
        cmap="magma",
        colorbar_label="Fraction",
    )
    _plot_routed_heatmap(
        axes[2, 0],
        _matrix_to_array(train.get("per_head_c_acc_by_true_rule") or [], len(head_labels)),
        "Train Per-Head c Accuracy by True Rule",
        xlabels=head_labels,
        ylabels=rule_labels,
        vmin=0.0,
        vmax=1.0,
        cmap="plasma",
        colorbar_label="Accuracy",
    )
    _plot_routed_heatmap(
        axes[2, 1],
        _matrix_to_array(val.get("per_head_c_acc_by_true_rule") or [], len(head_labels)),
        "Val Per-Head c Accuracy by True Rule",
        xlabels=head_labels,
        ylabels=rule_labels,
        vmin=0.0,
        vmax=1.0,
        cmap="plasma",
        colorbar_label="Accuracy",
    )

    axes[2, 2].axis("off")
    axes[2, 2].text(
        0.0,
        1.0,
        "\n".join(
            [
                f"Routing mode: {routed.get('routing_mode', cfg.get('routing_mode', '?'))}",
                f"c_head_count: {routed.get('c_head_count', cfg.get('c_head_count', '?'))}",
                f"c-head layers: {routed.get('routed_c_head_layers', cfg.get('routed_c_head_layers', '?'))}",
                f"Train samples: {train.get('sample_count', '?')}",
                f"Val samples: {val.get('sample_count', '?')}",
                f"Train entropy mean: {train.get('routing_entropy_mean')}",
                f"Val entropy mean: {val.get('routing_entropy_mean')}",
                "",
                "Panels:",
                "- head usage frequency",
                "- routing entropy by true rule",
                "- mean routing weights by true rule",
                "- true rule vs chosen head",
                "- per-head c accuracy by true rule",
            ],
        ),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
    )
    axes[2, 3].axis("off")

    fig.suptitle(
        f"Routed Analysis\n{_readable_name(cfg)}\n{_task_line(data, cfg)}",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out}")
    if show:
        plt.show()
    plt.close()


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
    ax1.set_title(_make_title(cfg, "Accuracy", data), fontsize=11)
    ax1.legend(fontsize=11)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(alpha=0.3)

    # ── loss ──────────────────────────────────────────────────────────────
    ax2.plot(epochs, data["train_losses"], label="Train Loss", color="#3B82F6", lw=2)
    ax2.plot(epochs, data["val_losses"],   label="Val Loss",   color="#F97316", lw=2)

    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Loss",  fontsize=12)
    ax2.set_title(_make_title(cfg, "Loss", data), fontsize=11)
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out}")
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
        nseg = _noise_segment(cfg)
        if nseg:
            label = f"{label}  ({nseg})"
        infmt = cfg.get("input_format")
        if infmt and infmt != "a_op_b_eq":
            label = f"{label}  [fmt={infmt}]"
        if sweep_param != "model_type":
            label = f"{label}  [model={_model_type(cfg)}]"
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
    op = results[0]["summary"].get("operation", "?")
    p0 = results[0]["summary"].get("p", "?")
    mt_tag = _infer_plot_model_tag(results)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(ylabel,  fontsize=12)
    ax.set_title(
        f"Sweep: {sweep_param}  ·  {op}  ·  p={p0}  ·  model={mt_tag}\n"
        f"train_frac={results[0]['summary'].get('train_frac', '?')}  ·  "
        f"{_task_line(results[0], results[0]['summary'])}\n"
        f"(legend = {sweep_param}; bracketed tags disambiguate runs)",
        fontsize=11,
    )
    ax.legend(fontsize=11)
    if metric == "acc":
        ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out}")
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

        ax.set_title(_short_title(cfg, data), fontsize=9)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    mt_tag = _infer_plot_model_tag(results)
    plt.suptitle(
        f"All runs ({len(results)})  ·  metric={metric}  ·  model={mt_tag}",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out}")
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
    """Human-readable one-liner: op, modulus, model type, then hparams."""
    op = cfg.get("operation", "?")
    p = cfg.get("p", "?")
    mt = _model_type(cfg)
    wd = cfg.get("weight_decay", "?")
    lr = cfg.get("lr", "?")
    d = cfg.get("d_model", "?")
    tf = cfg.get("train_frac", "?")
    ep = cfg.get("num_epochs", "?")
    try:
        tf_pct = int(float(tf) * 100)
    except (TypeError, ValueError):
        tf_pct = tf
    base = f"{op}  (p={p}, {mt})  ·  wd={wd}  lr={lr}  d={d}  train={tf_pct}%  ep={ep}"
    schl = cfg.get("shared_c_head_layers")
    if schl is not None and mt == "standard":
        base = f"{base}  ·  shared_c_head_layers={schl}"
    chc = cfg.get("c_head_count")
    if chc is not None and mt == "routed_modular":
        base = f"{base}  ·  c_head_count={chc}"
    chl = cfg.get("routed_c_head_layers")
    if chl is not None and mt == "routed_modular":
        base = f"{base}  ·  c_head_layers={chl}"
    infmt = cfg.get("input_format")
    if infmt and infmt != "a_op_b_eq":
        base = f"{base}  ·  fmt={infmt}"
    nseg = _noise_segment(cfg)
    return f"{base}  ·  {nseg}" if nseg else base


def _make_title(cfg: dict, kind: str, data: dict | None = None) -> str:
    data = data or {}
    # Line 1: what is being plotted; line 2: hparams; line 3: task/noise (includes model= for JSON parity)
    return f"{kind}\n{_readable_name(cfg)}\n{_task_line(data, cfg)}"


def _short_title(cfg: dict, data: dict | None = None) -> str:
    """Compact subplot title: bold contrast via short first line."""
    data = data or {}
    op = cfg.get("operation", "?")
    p = cfg.get("p", "?")
    mt = _model_type(cfg)
    head = f"{op}  ·  p={p}  ·  {mt}"
    return f"{head}\n{_task_line(data, cfg)}"


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
                   help="JSON result files (globs ok). Default: all in results/ (then legacy results_factor/)")
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
    for path, res in zip(paths, results):
        s = res["summary"]
        print(
            f"  {Path(path).name}  ->  model={_model_type(s)}  memo={s['memo_epoch']}  "
            f"grok={s['grok_epoch']}  gap={s['grok_gap']}  |  {_task_line(res, s)}"
        )

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
        r0 = results[0]
        op = cfg0.get("operation", "run")
        ep = cfg0.get("num_epochs")
        if ep is None or ep == "":
            le = r0.get("log_epochs") or []
            ep = le[-1] if le else "unknown"
        tf = int(float(cfg0.get("train_frac", 0.5)) * 100)
        mt = _infer_plot_model_tag(results)
        mt_safe = str(mt).replace(" ", "_")
        schl = cfg0.get("shared_c_head_layers")
        schl_seg = (
            f"_schl{schl}"
            if schl is not None and _model_type(cfg0) == "standard" and int(schl) != 1
            else ""
        )
        chc = cfg0.get("c_head_count")
        chc_seg = (
            f"_chc{chc}"
            if chc is not None and _model_type(cfg0) == "routed_modular"
            else ""
        )
        chl = cfg0.get("routed_c_head_layers")
        chl_seg = (
            f"_chl{chl}"
            if chl is not None and _model_type(cfg0) == "routed_modular"
            else ""
        )
        if mode == "sweep":
            out_path = f"plots/{op}_model-{mt_safe}{schl_seg}{chc_seg}{chl_seg}_sweep_{args.sweep_param}_tf{tf}_ep{ep}.png"
        elif mode == "single":
            wd = cfg0.get("weight_decay", "")
            lr = cfg0.get("lr", "")
            out_path = f"plots/{op}_model-{mt_safe}{schl_seg}{chc_seg}{chl_seg}_wd{wd}_lr{lr}_tf{tf}_ep{ep}.png"
        else:
            out_path = f"plots/grid_{op}_model-{mt_safe}{schl_seg}{chc_seg}{chl_seg}_n{len(results)}_tf{tf}_ep{ep}.png"

    if mode == "single":
        plot_single(results[0], out_path, show)
        if _has_routed_analysis(results[0]):
            stem, ext = os.path.splitext(out_path)
            plot_routed_analysis(results[0], f"{stem}_routing{ext}", show)
    elif mode == "sweep":
        plot_sweep(results, args.sweep_param, args.metric, out_path, show)
    else:  # grid
        plot_grid(results, args.metric, out_path, show)


if __name__ == "__main__":
    main()