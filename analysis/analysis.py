#!/usr/bin/env python3
"""
Scan saved experiment JSONs (e.g. under ``!previous_results``) for runs with
``task: categorical`` and ``label_mode: c``.

Saved JSON ``summary`` may include ``label_noise`` (asymmetric) and ``label_noise_sym``
(symmetric pair-swap) when training used ``--noise`` / ``--noise_sym``.

Output layout::

    analysis/
      runs/003__<slug>/
        summary.txt              — key metrics in plain language
        learning_curve.png       — train vs val accuracy vs epoch
        per_class_accuracy.png   — train & val vs class id (easy to read)
        per_class_heatmap.png    — compact heatmap (train row / val row)
        snapshot.json            — summary row + per-class vectors for this run only
      all/
        categorical_c_summary.csv
        categorical_c_per_class.jsonl
        comparison_overview.png
        comparison_curves.png
        comparison_per_class.png

Usage (from ``Grokking-Project``)::

    python analysis/analysis.py
    python analysis/analysis.py --roots \"!previous_results\" ./other_results
    python analysis/analysis.py --no-per-class-jsonl --no-plots

Symmetry / hard-class diagnostics for selected runs (e.g. 020, 021, 013)::

    python analysis/difficulty_analysis.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _mean(xs: List[float]) -> Optional[float]:
    return float(statistics.mean(xs)) if xs else None


def _load_json(path: str) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _is_target_run(data: dict) -> bool:
    if data.get("task") not in ("categorical", "categorical_factor"):
        return False
    summ = data.get("summary") or {}
    return summ.get("label_mode") == "c"


def _last_row(rows: List[Any]) -> Any:
    return rows[-1] if rows else None


def _band_macro_micro(
    per_cls: List[Any],
    support: List[int],
    lo: int,
    hi: int,
) -> Tuple[Optional[float], Optional[float], int]:
    """Macro = mean of class accs; micro = sum(acc*n)/sum(n). Returns (macro, micro, n_classes_used)."""
    accs: List[float] = []
    weights: List[float] = []
    for i in range(lo, min(hi, len(per_cls))):
        a = per_cls[i]
        if a is None:
            continue
        n = float(support[i]) if i < len(support) and support[i] is not None else 0.0
        accs.append(float(a))
        weights.append(n)
    if not accs:
        return None, None, 0
    macro = float(statistics.mean(accs))
    if weights and sum(weights) > 0:
        micro = sum(a * w for a, w in zip(accs, weights)) / sum(weights)
    else:
        micro = macro
    return macro, micro, len(accs)


def analyze_payload(path: str, data: dict) -> Tuple[Optional[dict], Optional[dict]]:
    """
    Returns (summary_row_dict, per_class_record_or_none).
    summary_row_dict is flat for CSV; per_class_record for JSONL.
    """
    summ = data.get("summary") or {}
    p = int(summ.get("p", 0))
    rule_count = int(summ.get("rule_count", 1))
    num_classes = int(summ.get("num_classes", rule_count * p if p else 0))

    train_accs = data.get("train_accs") or []
    val_accs = data.get("val_accs") or []
    log_epochs = data.get("log_epochs") or []

    tr_last = _last_row(train_accs)
    va_last = _last_row(val_accs)
    last_epoch = _last_row(log_epochs)

    row: Dict[str, Any] = {
        "path": path,
        "operation": summ.get("operation"),
        "p": p,
        "rule_count": rule_count,
        "num_classes": num_classes,
        "model_type": summ.get("model_type"),
        "shared_c_head_layers": summ.get("shared_c_head_layers"),
        "c_head_count": summ.get("c_head_count"),
        "routing_mode": summ.get("routing_mode"),
        "routed_c_head_layers": summ.get("routed_c_head_layers"),
        "branch_metric": data.get("branch_metric"),
        "branch_label_1": data.get("branch_label_1"),
        "branch_label_2": data.get("branch_label_2"),
        "last_log_epoch": last_epoch,
        "train_acc_last": tr_last,
        "val_acc_last": va_last,
        "train_acc_mean_logs": _mean([float(x) for x in train_accs]) if train_accs else None,
        "val_acc_mean_logs": _mean([float(x) for x in val_accs]) if val_accs else None,
        "label_noise": summ.get("label_noise"),
        "label_noise_sym": summ.get("label_noise_sym"),
    }

    to_b1 = data.get("train_odd_accs")
    to_b2 = data.get("train_even_accs")
    vo_b1 = data.get("val_odd_accs")
    vo_b2 = data.get("val_even_accs")
    row["train_branch1_last"] = _last_row(to_b1) if to_b1 else None
    row["train_branch2_last"] = _last_row(to_b2) if to_b2 else None
    row["val_branch1_last"] = _last_row(vo_b1) if vo_b1 else None
    row["val_branch2_last"] = _last_row(vo_b2) if vo_b2 else None

    ptr = data.get("per_class_train_accs") or []
    pva = data.get("per_class_val_accs") or []
    sup_tr = data.get("per_class_train_support") or []
    sup_va = data.get("per_class_val_support") or []

    per_class_out: Optional[dict] = None
    rule_macros_tr: List[float] = []
    rule_macros_va: List[float] = []
    rule_micros_tr: List[float] = []
    rule_micros_va: List[float] = []

    if ptr and pva and p > 0 and rule_count >= 1:
        tr_row = ptr[-1]
        va_row = pva[-1]
        for r in range(rule_count):
            lo, hi = r * p, (r + 1) * p
            mtr, mic_tr, _ = _band_macro_micro(tr_row, sup_tr, lo, hi)
            mva, mic_va, _ = _band_macro_micro(va_row, sup_va, lo, hi)
            row[f"rule{r}_macro_train_last"] = mtr
            row[f"rule{r}_macro_val_last"] = mva
            row[f"rule{r}_micro_train_last"] = mic_tr
            row[f"rule{r}_micro_val_last"] = mic_va
            if mtr is not None:
                rule_macros_tr.append(mtr)
            if mva is not None:
                rule_macros_va.append(mva)
            if mic_tr is not None:
                rule_micros_tr.append(mic_tr)
            if mic_va is not None:
                rule_micros_va.append(mic_va)

        row["macro_mean_over_rules_train"] = (
            float(statistics.mean(rule_macros_tr)) if rule_macros_tr else None
        )
        row["macro_mean_over_rules_val"] = (
            float(statistics.mean(rule_macros_va)) if rule_macros_va else None
        )
        row["micro_mean_over_rules_train"] = (
            float(statistics.mean(rule_micros_tr)) if rule_micros_tr else None
        )
        row["micro_mean_over_rules_val"] = (
            float(statistics.mean(rule_micros_va)) if rule_micros_va else None
        )

        per_class_out = {
            "path": path,
            "operation": summ.get("operation"),
            "p": p,
            "rule_count": rule_count,
            "last_log_epoch": last_epoch,
            "per_class_train_last": tr_row,
            "per_class_val_last": va_row,
            "per_class_train_support": list(sup_tr) if sup_tr else None,
            "per_class_val_support": list(sup_va) if sup_va else None,
        }
    else:
        row["macro_mean_over_rules_train"] = None
        row["macro_mean_over_rules_val"] = None
        row["micro_mean_over_rules_train"] = None
        row["micro_mean_over_rules_val"] = None

    return row, per_class_out


def _collect_json_files(roots: List[str]) -> List[str]:
    out: List[str] = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _, files in os.walk(root):
            for fn in files:
                if fn.lower().endswith(".json"):
                    out.append(os.path.join(dirpath, fn))
    out.sort()
    return out


def _project_dir() -> str:
    """Repo root (``Grokking-Project``): parent of this ``analysis`` directory."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(here)


def _default_out_dir() -> str:
    """Same folder as this script (``analysis/``)."""
    return os.path.dirname(os.path.abspath(__file__))


def _short_run_label(rel_path: str, max_len: int = 42) -> str:
    base = os.path.basename(rel_path).replace(".json", "")
    if len(base) <= max_len:
        return base
    return base[: max_len - 3] + "..."


def _sanitize_slug_piece(s: str, max_len: int = 64) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if len(s) > max_len:
        s = s[:max_len].rstrip("_")
    return s or "run"


def run_folder_name(rel_path: str, index: int) -> str:
    """Stable folder name: index + relative path stem (safe for Windows)."""
    stem = os.path.splitext(rel_path)[0].replace(os.sep, "__")
    piece = _sanitize_slug_piece(stem, max_len=70)
    return f"{index:03d}__{piece}"


def _get_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        return None


def _apply_style(plt: Any) -> None:
    for style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
        try:
            plt.style.use(style)
            break
        except OSError:
            continue
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.titlesize": 13,
            "axes.grid": True,
            "grid.alpha": 0.35,
        }
    )


def write_plot_overview(
    rows: List[Dict[str, Any]],
    out_path: str,
    *,
    labels: Sequence[str],
    title_suffix: str = "",
) -> None:
    plt = _get_pyplot()
    if plt is None or not rows:
        return
    import numpy as np

    _apply_style(plt)
    n = len(rows)
    x = np.arange(n, dtype=float)
    w = 0.35
    fig_w = max(12.0, min(40.0, 0.52 * n))
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(fig_w, 9.0), sharex=True)
    tr_m = [r.get("train_acc_mean_logs") for r in rows]
    va_m = [r.get("val_acc_mean_logs") for r in rows]
    tr_l = [r.get("train_acc_last") for r in rows]
    va_l = [r.get("val_acc_last") for r in rows]

    def _bar(ax, left, vals, width, **kw):
        return ax.bar(
            left,
            [float(v) if v is not None else 0.0 for v in vals],
            width,
            **kw,
        )

    _bar(
        ax0,
        x - w / 2,
        tr_m,
        w,
        label="Train — mean over log checkpoints",
        color="#1d4ed8",
        edgecolor="#172554",
        linewidth=0.4,
    )
    _bar(
        ax0,
        x + w / 2,
        va_m,
        w,
        label="Val — mean over log checkpoints",
        color="#c026d3",
        edgecolor="#4a044e",
        linewidth=0.4,
    )
    ax0.set_ylabel("Accuracy (0-1)")
    ax0.set_ylim(0, 1.05)
    ax0.legend(loc="upper right", framealpha=0.95)
    ax0.set_title("Average train / val across logged epochs (macro over time)")

    _bar(
        ax1,
        x - w / 2,
        tr_l,
        w,
        label="Train — last log step",
        color="#0f766e",
        edgecolor="#134e4a",
        linewidth=0.4,
    )
    _bar(
        ax1,
        x + w / 2,
        va_l,
        w,
        label="Val — last log step",
        color="#b91c1c",
        edgecolor="#450a0a",
        linewidth=0.4,
    )
    ax1.set_ylabel("Accuracy (0-1)")
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="upper right", framealpha=0.95)
    ax1.set_title("Train / val at the final logged epoch")

    ax1.set_xticks(x)
    ax1.set_xticklabels(list(labels), rotation=75, ha="right", fontsize=8)
    fig.suptitle(
        f"Categorical label_mode=c — run comparison{title_suffix}\n"
        "(Each tick = one saved JSON; hover values in CSV / per-run summary.txt)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_plot_curves(
    curves: List[Dict[str, Any]],
    out_path: str,
    *,
    labels: Sequence[str],
    max_panels: int,
    title_note: str = "",
) -> None:
    plt = _get_pyplot()
    if plt is None or not curves:
        return
    import math

    _apply_style(plt)
    show = curves[:max_panels]
    labs = list(labels)[:max_panels]
    n = len(show)
    ncols = min(3, max(1, n))
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 3.6 * nrows), squeeze=False)
    for idx, rec in enumerate(show):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        ep = rec.get("log_epochs") or []
        tr = rec.get("train_accs") or []
        va = rec.get("val_accs") or []
        if ep and tr:
            ax.plot(ep, tr, color="#1d4ed8", lw=1.8, label="Train")
        if ep and va:
            ax.plot(ep, va, color="#b91c1c", lw=1.8, alpha=0.92, label="Val")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title(labs[idx], fontsize=9, wrap=True)
        ax.legend(loc="lower right", fontsize=8)
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")
    fig.suptitle(
        f"Learning curves (first {n} of {len(curves)} runs){title_note}",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_plot_per_class_heatmap(
    per_class_records: List[dict],
    out_path: str,
    *,
    max_panels: int,
    title_note: str = "",
) -> None:
    plt = _get_pyplot()
    if plt is None or not per_class_records:
        return
    import math
    import numpy as np

    _apply_style(plt)
    show = per_class_records[:max_panels]
    labs = [_short_run_label(str(rec.get("path", ""))) for rec in show]
    n = len(show)
    ncols = min(2, max(1, n))
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(min(18, 5 + 6 * ncols), 4.2 * nrows), squeeze=False)
    for idx, rec in enumerate(show):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        tr_row = rec.get("per_class_train_last") or []
        va_row = rec.get("per_class_val_last") or []
        tr = np.array([float(x) if x is not None else np.nan for x in tr_row], dtype=float)
        va = np.array([float(x) if x is not None else np.nan for x in va_row], dtype=float)
        if tr.size == 0:
            ax.axis("off")
            continue
        mat = np.vstack([tr, va])
        im = ax.imshow(mat, aspect="auto", vmin=0, vmax=1, cmap="magma", interpolation="nearest")
        ax.set_yticks([0, 1], ["Train", "Val"])
        ax.set_xlabel("Class id (0 .. K-1)")
        p = int(rec.get("p") or 0)
        rc = int(rec.get("rule_count") or 1)
        if p > 0 and rc > 1:
            for b in range(1, rc):
                ax.axvline(b * p - 0.5, color="cyan", lw=1.0, alpha=0.95)
        ax.set_title(f"{labs[idx]}\np={p}, rule_count={rc}", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="Accuracy")
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")
    if n > 1:
        fig.suptitle(
            f"Per-class heatmap at last log (train row / val row). Cyan: rule bands.{title_note}",
            fontsize=12,
            y=1.01,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_plot_per_class_lines(rec: dict, out_path: str) -> None:
    """Single run: line plot (readable at a glance)."""
    plt = _get_pyplot()
    if plt is None or not rec:
        return
    import numpy as np

    _apply_style(plt)
    tr_row = rec.get("per_class_train_last") or []
    va_row = rec.get("per_class_val_last") or []
    tr = np.array([float(x) if x is not None else np.nan for x in tr_row], dtype=float)
    va = np.array([float(x) if x is not None else np.nan for x in va_row], dtype=float)
    if tr.size == 0:
        return
    x = np.arange(len(tr))
    p = int(rec.get("p") or 0)
    rc = int(rec.get("rule_count") or 1)
    fig, ax = plt.subplots(figsize=(min(18, 8 + 0.04 * len(tr)), 5.5))
    ax.plot(x, tr, color="#1d4ed8", lw=1.4, alpha=0.9, label="Train (last log)")
    ax.plot(x, va, color="#b91c1c", lw=1.4, alpha=0.9, label="Val (last log)")
    if p > 0 and rc > 1:
        for b in range(1, rc):
            ax.axvline(b * p - 0.5, color="#64748b", ls="--", lw=1.2, alpha=0.9)
        ax.text(
            0.01,
            0.02,
            f"dashed lines: boundaries between rule bands (p={p})",
            transform=ax.transAxes,
            fontsize=9,
            color="#334155",
        )
    ax.set_xlim(-0.5, len(tr) - 0.5)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Class id")
    ax.set_ylabel("Per-class accuracy")
    ax.legend(loc="lower right")
    stem = _short_run_label(str(rec.get("path", "")), max_len=80)
    fig.suptitle(
        f"Per-class train vs val — {stem}\noperation={rec.get('operation')}  p={p}  rule_count={rc}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_single_learning_curve(
    curve: Dict[str, Any],
    row: Dict[str, Any],
    out_path: str,
) -> None:
    plt = _get_pyplot()
    if plt is None:
        return
    _apply_style(plt)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ep = curve.get("log_epochs") or []
    tr = curve.get("train_accs") or []
    va = curve.get("val_accs") or []
    if ep and tr:
        ax.plot(ep, tr, color="#1d4ed8", lw=2.2, label="Train accuracy")
    if ep and va:
        ax.plot(ep, va, color="#b91c1c", lw=2.2, alpha=0.92, label="Val accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend(loc="lower right")
    tl = row.get("train_acc_last")
    vl = row.get("val_acc_last")
    tm = row.get("train_acc_mean_logs")
    vm = row.get("val_acc_mean_logs")
    if (
        tl is not None
        and vl is not None
        and tm is not None
        and vm is not None
    ):
        sub = (
            f"Last log: train={tl:.4f}  val={vl:.4f}   |   "
            f"Mean over logs: train={tm:.4f}  val={vm:.4f}"
        )
    else:
        sub = f"Last log: {tl}  val: {vl}  |  Mean logs: train {tm}  val {vm}"
    ax.set_title(
        f"Learning curve — {row.get('operation')}  |  p={row.get('p')}  |  rule_count={row.get('rule_count')}\n"
        f"{sub}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_run_summary_txt(row: Dict[str, Any], out_path: str) -> None:
    lines = [
        "Categorical task  ·  label_mode=c",
        "=" * 52,
        f"Source JSON (relative): {row.get('path')}",
        f"Operation: {row.get('operation')}   p={row.get('p')}   rule_count={row.get('rule_count')}",
        f"Model: {row.get('model_type')}   shared_c_head_layers: {row.get('shared_c_head_layers')}   c_head_count: {row.get('c_head_count')}   routing_mode: {row.get('routing_mode')}   routed_c_head_layers: {row.get('routed_c_head_layers')}",
        f"num_classes: {row.get('num_classes')}   branch: {row.get('branch_metric')}",
        f"  branches: {row.get('branch_label_1')} / {row.get('branch_label_2')}",
        f"Train label noise — asymmetric: {row.get('label_noise')}   symmetric: {row.get('label_noise_sym')}",
        "",
        "Accuracy (overall)",
        f"  Last logged epoch ({row.get('last_log_epoch')}):",
        f"    train = {row.get('train_acc_last')}",
        f"    val   = {row.get('val_acc_last')}",
        f"  Mean over all log checkpoints:",
        f"    train = {row.get('train_acc_mean_logs')}",
        f"    val   = {row.get('val_acc_mean_logs')}",
        "",
        "Branch accuracies (last log)",
        f"  train {row.get('branch_label_1')} = {row.get('train_branch1_last')}",
        f"  train {row.get('branch_label_2')} = {row.get('train_branch2_last')}",
        f"  val   {row.get('branch_label_1')} = {row.get('val_branch1_last')}",
        f"  val   {row.get('branch_label_2')} = {row.get('val_branch2_last')}",
        "",
        "Per-rule band summaries (see plots for every class)",
        f"  macro mean over rules — train: {row.get('macro_mean_over_rules_train')}  val: {row.get('macro_mean_over_rules_val')}",
        f"  micro mean over rules — train: {row.get('micro_mean_over_rules_train')}  val: {row.get('micro_mean_over_rules_val')}",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_run_snapshot_json(
    row: Dict[str, Any],
    per_class: Optional[dict],
    out_path: str,
) -> None:
    payload: Dict[str, Any] = {"summary_row": row}
    if per_class:
        payload["per_class"] = {
            k: per_class[k]
            for k in (
                "path",
                "operation",
                "p",
                "rule_count",
                "last_log_epoch",
                "per_class_train_last",
                "per_class_val_last",
                "per_class_train_support",
                "per_class_val_support",
            )
            if k in per_class
        }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze categorical label_mode=c JSON results.")
    ap.add_argument(
        "--roots",
        nargs="*",
        default=None,
        help="Directories to scan recursively (default: !previous_results under project)",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: directory containing this script)",
    )
    ap.add_argument("--no-per-class-jsonl", action="store_true", help="Skip JSONL detail file.")
    ap.add_argument("--no-plots", action="store_true", help="Skip PNG figures (no matplotlib needed).")
    ap.add_argument(
        "--max-plot-runs",
        type=int,
        default=12,
        help="Max runs shown in multi-panel comparison figures under all/ (default: 12).",
    )
    args = ap.parse_args()

    proj = _project_dir()
    roots = args.roots
    if not roots:
        roots = [os.path.join(proj, "!previous_results")]
    roots = [os.path.abspath(os.path.join(proj, r)) if not os.path.isabs(r) else r for r in roots]

    out_dir = args.out_dir or _default_out_dir()
    runs_root = os.path.join(out_dir, "runs")
    all_root = os.path.join(out_dir, "all")
    os.makedirs(all_root, exist_ok=True)
    os.makedirs(runs_root, exist_ok=True)

    summary_path = os.path.join(all_root, "categorical_c_summary.csv")
    jsonl_path = os.path.join(all_root, "categorical_c_per_class.jsonl")

    files = _collect_json_files(roots)
    bundles: List[Dict[str, Any]] = []

    for path in files:
        data = _load_json(path)
        if not data or not _is_target_run(data):
            continue
        rel = os.path.relpath(path, proj)
        srow, pco = analyze_payload(rel, data)
        if not srow:
            continue
        curve = {
            "path": rel,
            "log_epochs": data.get("log_epochs") or [],
            "train_accs": data.get("train_accs") or [],
            "val_accs": data.get("val_accs") or [],
        }
        idx = len(bundles)
        slug = run_folder_name(rel, idx)
        bundles.append(
            {
                "slug": slug,
                "path": rel,
                "row": srow,
                "curve": curve,
                "per_class": pco,
            }
        )

    rows = [b["row"] for b in bundles]
    curve_records = [b["curve"] for b in bundles]
    per_class_records = [b["per_class"] for b in bundles if b["per_class"] is not None]

    if not rows:
        print("No matching runs (task=categorical, label_mode=c). Check --roots.")
        return

    base_keys = [
        "path",
        "operation",
        "p",
        "rule_count",
        "num_classes",
        "branch_metric",
        "branch_label_1",
        "branch_label_2",
        "last_log_epoch",
        "train_acc_last",
        "val_acc_last",
        "train_acc_mean_logs",
        "val_acc_mean_logs",
        "train_branch1_last",
        "train_branch2_last",
        "val_branch1_last",
        "val_branch2_last",
        "macro_mean_over_rules_train",
        "macro_mean_over_rules_val",
        "micro_mean_over_rules_train",
        "micro_mean_over_rules_val",
    ]
    extra_keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in base_keys and k not in extra_keys:
                extra_keys.append(k)
    extra_keys.sort()
    fieldnames = base_keys + extra_keys

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    if not args.no_per_class_jsonl and per_class_records:
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for rec in per_class_records:
                f.write(json.dumps(rec) + "\n")

    # Per-run artifacts
    for b in bundles:
        rdir = os.path.join(runs_root, b["slug"])
        os.makedirs(rdir, exist_ok=True)
        write_run_summary_txt(b["row"], os.path.join(rdir, "summary.txt"))
        write_run_snapshot_json(
            b["row"],
            b["per_class"],
            os.path.join(rdir, "snapshot.json"),
        )
        if not args.no_plots and _get_pyplot() is not None:
            write_single_learning_curve(
                b["curve"],
                b["row"],
                os.path.join(rdir, "learning_curve.png"),
            )
            if b["per_class"]:
                write_plot_per_class_lines(
                    b["per_class"],
                    os.path.join(rdir, "per_class_accuracy.png"),
                )
                write_plot_per_class_heatmap(
                    [b["per_class"]],
                    os.path.join(rdir, "per_class_heatmap.png"),
                    max_panels=1,
                    title_note="",
                )

    # Comparison under all/
    labels = [_short_run_label(str(b["row"].get("path", ""))) for b in bundles]
    print(f"Wrote {len(rows)} runs -> all/categorical_c_summary.csv")
    print(f"Per-run folders under runs/ ({len(bundles)} folders):")
    for b in bundles:
        print(f"  runs/{b['slug']}/")
    print("Combined comparison figures -> all/")

    if not args.no_per_class_jsonl and per_class_records:
        print(f"  all/categorical_c_per_class.jsonl ({len(per_class_records)} lines)")

    if args.no_plots:
        return

    if _get_pyplot() is None:
        print("matplotlib not installed; skipped PNG figures.")
        return

    overview_png = os.path.join(all_root, "comparison_overview.png")
    curves_png = os.path.join(all_root, "comparison_curves.png")
    pclass_png = os.path.join(all_root, "comparison_per_class.png")
    lim = max(1, args.max_plot_runs)
    write_plot_overview(rows, overview_png, labels=labels)
    write_plot_curves(curve_records, curves_png, labels=labels, max_panels=lim)
    if per_class_records:
        write_plot_per_class_heatmap(per_class_records, pclass_png, max_panels=lim)
    print(f"  all/comparison_overview.png")
    print(f"  all/comparison_curves.png (first {lim} runs)")
    if per_class_records:
        print(f"  all/comparison_per_class.png (first {lim} runs)")


if __name__ == "__main__":
    main()
