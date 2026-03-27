#!/usr/bin/env python3
"""
Per-class difficulty and rule-band symmetry for selected analysis runs.

Reads ``snapshot.json`` under ``analysis/runs/<prefix>__.../`` (produce snapshots first
via ``analysis.py``).

For runs with ``rule_count >= 2`` and ``K = 2*p`` classes, symmetry checks compare
``val[k]`` vs ``val[k + p]`` (same underlying mod-p output, different rule bands).

Default focus prefixes: 020, 021, 013 (override with ``--prefixes``).

Outputs (under ``analysis/difficulty_focus/`` by default)::

    val_accuracy_vs_class.png   — one panel per run: val (and faint train) vs class id
    val_only_vs_class.png       — same, validation only (simple x/y read)
    band0_val_overlay.png       — val for classes 0..p-1 overlaid (same output-mod-p lane)
    symmetry_*_scatter.png      — val[k] vs val[k+p], per rc>=2 run
    symmetry_*_diff.png         — (val[k+p] - val[k]) vs k
    hard_classes_report.txt     — worst classes + overlap across runs (band0 and full K)

Usage from ``Grokking-Project``::

    python analysis/difficulty_analysis.py
    python analysis/difficulty_analysis.py --prefixes 020 021 013
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _default_analysis_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _project_dir() -> str:
    return os.path.dirname(_default_analysis_dir())


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
            "font.size": 10,
            "axes.titlesize": 11,
            "figure.titlesize": 12,
        }
    )


def find_run_dirs(runs_root: str, prefixes: Sequence[str]) -> List[Tuple[str, str]]:
    """Return sorted list of (prefix, abs_dir) for first match per prefix."""
    if not os.path.isdir(runs_root):
        return []
    found: Dict[str, str] = {}
    for name in sorted(os.listdir(runs_root)):
        path = os.path.join(runs_root, name)
        if not os.path.isdir(path):
            continue
        for pre in prefixes:
            if name.startswith(f"{pre}__") and pre not in found:
                found[pre] = path
                break
    return [(p, found[p]) for p in prefixes if p in found]


def load_snapshot(run_dir: str) -> dict:
    path = os.path.join(run_dir, "snapshot.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def as_float_vec(xs: List[Any]) -> np.ndarray:
    out = []
    for x in xs:
        if x is None:
            out.append(np.nan)
        else:
            out.append(float(x))
    return np.array(out, dtype=np.float64)


def band0_val(val: np.ndarray, p: int, rule_count: int) -> np.ndarray:
    """Val accuracies for the first p classes (output lane / rule 0 band)."""
    return np.array(val[:p], dtype=np.float64)


def symmetry_pair(val: np.ndarray, p: int) -> Tuple[np.ndarray, np.ndarray]:
    """val0[k]=val[k], val1[k]=val[k+p], k in 0..p-1."""
    return val[:p].copy(), val[p : 2 * p].copy()


def hardest_indices(vec: np.ndarray, n: int, exclude_nan: bool = True) -> List[int]:
    """Indices sorted by ascending accuracy (hardest first)."""
    idx = np.arange(len(vec))
    if exclude_nan:
        m = np.isfinite(vec)
        idx = idx[m]
        vec = vec[m]
    order = np.argsort(vec)
    worst = idx[order]
    return list(worst[: min(n, len(worst))])


def main() -> None:
    ap = argparse.ArgumentParser(description="Symmetry & difficulty across selected runs.")
    ap.add_argument(
        "--prefixes",
        nargs="+",
        default=["020", "021", "013"],
        help="Run folder prefixes (e.g. 020 021 013).",
    )
    ap.add_argument(
        "--runs-root",
        default=None,
        help="Parent of per-run folders (default: <project>/analysis/runs).",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Where to write figures + report (default: <project>/analysis/difficulty_focus).",
    )
    args = ap.parse_args()

    proj = _project_dir()
    analysis_dir = _default_analysis_dir()
    runs_root = args.runs_root or os.path.join(analysis_dir, "runs")
    out_dir = args.out_dir or os.path.join(analysis_dir, "difficulty_focus")
    os.makedirs(out_dir, exist_ok=True)

    resolved = find_run_dirs(runs_root, args.prefixes)
    if not resolved:
        print(f"No run folders found under {runs_root} for prefixes {args.prefixes}.")
        return

    plt = _get_pyplot()
    if plt is None:
        print("matplotlib required for plots.")
        return
    _apply_style(plt)

    loaded: List[Dict[str, Any]] = []
    for prefix, rdir in resolved:
        snap = load_snapshot(rdir)
        row = snap["summary_row"]
        pc = snap.get("per_class") or {}
        val = as_float_vec(pc.get("per_class_val_last") or [])
        train = as_float_vec(pc.get("per_class_train_last") or [])
        p = int(row.get("p") or 0)
        rc = int(row.get("rule_count") or 1)
        loaded.append(
            {
                "prefix": prefix,
                "dir": rdir,
                "row": row,
                "val": val,
                "train": train,
                "p": p,
                "rule_count": rc,
                "label": f"{prefix}: {row.get('operation')} rc={rc}",
            }
        )
        print(f"Loaded {prefix} from {os.path.basename(rdir)}")

    # --- 1) Val acc vs class id (one panel per run)
    n_runs = len(loaded)
    fig, axes = plt.subplots(n_runs, 1, figsize=(12, 3.2 * n_runs), squeeze=False)
    for i, L in enumerate(loaded):
        ax = axes[i][0]
        v = L["val"]
        x = np.arange(len(v))
        ax.plot(x, v, color="#b91c1c", lw=1.1, alpha=0.9, label="Val (last log)")
        if L["train"].size == len(v):
            ax.plot(x, L["train"], color="#1d4ed8", lw=0.8, alpha=0.55, label="Train (last log)")
        p, rc = L["p"], L["rule_count"]
        if p > 0 and rc > 1:
            for b in range(1, rc):
                ax.axvline(b * p - 0.5, color="#64748b", ls="--", lw=1.0, alpha=0.85)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Class id (global index)")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{L['label']}  |  K={len(v)}  (dashed: rule band boundaries)")
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Validation accuracy vs global class index", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "val_accuracy_vs_class.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(n_runs, 1, figsize=(12, 2.8 * n_runs), squeeze=False)
    for i, L in enumerate(loaded):
        ax = axes[i][0]
        v = L["val"]
        x = np.arange(len(v))
        ax.plot(x, v, color="#b91c1c", lw=1.2, alpha=0.92)
        p, rc = L["p"], L["rule_count"]
        if p > 0 and rc > 1:
            for b in range(1, rc):
                ax.axvline(b * p - 0.5, color="#64748b", ls="--", lw=1.0, alpha=0.85)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Class id")
        ax.set_ylabel("Val accuracy")
        ax.set_title(f"{L['label']}  |  K={len(v)}")
    fig.suptitle("Validation accuracy vs class id (val only)", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "val_only_vs_class.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # --- 2) Band-0 overlay (same output value lane): classes 0 .. p-1
    p_common = min(L["p"] for L in loaded if L["p"] > 0)
    if p_common > 0:
        fig, ax = plt.subplots(figsize=(11, 5))
        colors = ["#7c3aed", "#059669", "#d97706", "#dc2626", "#0284c7"]
        for j, L in enumerate(loaded):
            v0 = band0_val(L["val"], L["p"], L["rule_count"])[:p_common]
            x0 = np.arange(p_common)
            c = colors[j % len(colors)]
            ax.plot(x0, v0, lw=1.3, alpha=0.9, color=c, label=f"{L['prefix']} band0 ({L['row'].get('operation')})")
        ax.set_xlabel("Class id k in 0 .. p-1 (rule-0 / single-rule lane; same underlying output mod p)")
        ax.set_ylabel("Val accuracy")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower right")
        ax.set_title(
            "Overlay: val accuracy for the first p classes only — compare repeating structure across runs"
        )
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "band0_val_overlay.png"), dpi=160, bbox_inches="tight")
        plt.close(fig)

    # --- 3) Symmetry val[k] vs val[k+p] for rc>=2
    for L in loaded:
        p, rc = L["p"], L["rule_count"]
        if rc < 2 or p <= 0 or len(L["val"]) < 2 * p:
            continue
        v0, v1 = symmetry_pair(L["val"], p)
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.scatter(v0, v1, s=12, alpha=0.65, color="#0f766e", edgecolors="none")
        lims = [0, 1.05]
        ax.plot(lims, lims, "k--", lw=1, alpha=0.45, label="y = x (perfect symmetry)")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
        ax.set_xlabel("Val acc class k (rule band 0)")
        ax.set_ylabel("Val acc class k + p (rule band 1)")
        m_ad = float(np.nanmean(np.abs(v1 - v0)))
        c = np.corrcoef(v0, v1)[0, 1]
        ax.set_title(
            f"{L['prefix']} {L['row'].get('operation')}\n"
            f"mean |acc[k+p]-acc[k]| = {m_ad:.4f}  |  corr(val[k], val[k+p]) = {c:.4f}\n"
            "Near diagonal => similar k in both bands; far/off-diagonal => band-specific errors."
        )
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(
            os.path.join(out_dir, f"symmetry_{L['prefix']}_scatter.png"),
            dpi=160,
            bbox_inches="tight",
        )
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 4))
        diff = v1 - v0
        ax.axhline(0.0, color="k", lw=0.8, alpha=0.4)
        ax.scatter(np.arange(p), diff, s=14, color="#9333ea", alpha=0.7)
        ax.set_xlabel("k (output class id mod p)")
        ax.set_ylabel("val[k+p] - val[k]")
        ax.set_title(f"{L['prefix']}: band1 minus band0 val accuracy (symmetry residual)")
        fig.tight_layout()
        fig.savefig(
            os.path.join(out_dir, f"symmetry_{L['prefix']}_diff.png"),
            dpi=160,
            bbox_inches="tight",
        )
        plt.close(fig)

    # --- 4) Report: hard classes + overlap
    lines: List[str] = []
    lines.append("Hard-class / symmetry report")
    lines.append("=" * 60)
    top_n = 25

    band0_hard: Dict[str, List[int]] = {}
    full_hard: Dict[str, List[int]] = {}

    for L in loaded:
        p, rc = L["p"], L["rule_count"]
        v = L["val"]
        prefix = L["prefix"]
        if p > 0:
            v0 = band0_val(v, p, rc)
            band0_hard[prefix] = hardest_indices(v0, top_n)
        full_hard[prefix] = hardest_indices(v, top_n)

        lines.append("")
        lines.append(f"--- Run {prefix}: {L['row'].get('path')} ---")
        lines.append(f"operation={L['row'].get('operation')}  p={p}  rule_count={rc}  K={len(v)}")
        if rc >= 2 and len(v) >= 2 * p:
            v0, v1 = symmetry_pair(v, p)
            mad = float(np.nanmean(np.abs(v1 - v0)))
            cor = float(np.corrcoef(v0, v1)[0, 1])
            lines.append(f"Symmetry: mean|val[k+p]-val[k]| = {mad:.5f}  corr(val[k],val[k+p]) = {cor:.4f}")
            if mad < 0.08 and cor > 0.7:
                hint = (
                    "  -> Small band gap + strong positive corr: acc[k] ~= acc[k+p]; "
                    "hard classes look tied to output k, not which band."
                )
            elif cor < -0.3:
                hint = (
                    "  -> Negative corr: easy in band0 often pairs with hard in band1 (or vv); "
                    "band identity matters, not just local output."
                )
            else:
                hint = (
                    "  -> Use scatter & diff plots: middling rule suggests mixed band vs output effects."
                )
            lines.append(hint)
        else:
            lines.append("Symmetry: N/A (rule_count==1 or insufficient classes).")

        lines.append(f"Hardest {top_n} global class ids (lowest val acc): {full_hard[prefix]}")
        if p > 0:
            lines.append(f"Hardest {top_n} in band0 only (ids 0..p-1): {band0_hard[prefix]}")

    # Pairwise band0 correlation (same k semantics)
    lines.append("")
    lines.append("Pairwise Spearman correlation of band0 val acc (same k across runs)")
    from numpy.linalg import LinAlgError

    def _spearman(a: np.ndarray, b: np.ndarray) -> float:
        """Spearman rho; nan-safe on intersection."""
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 3:
            return float("nan")
        ra = np.argsort(np.argsort(a[m]))
        rb = np.argsort(np.argsort(b[m]))
        return float(np.corrcoef(ra, rb)[0, 1])

    for i in range(len(loaded)):
        for j in range(i + 1, len(loaded)):
            Li, Lj = loaded[i], loaded[j]
            pi, pj = Li["p"], Lj["p"]
            if pi <= 0 or pj <= 0:
                continue
            pc = min(pi, pj)
            vi = band0_val(Li["val"], Li["p"], Li["rule_count"])[:pc]
            vj = band0_val(Lj["val"], Lj["p"], Lj["rule_count"])[:pc]
            rho = _spearman(vi, vj)
            lines.append(f"  {Li['prefix']} vs {Lj['prefix']}: Spearman rho = {rho:.4f}")

    # Hard-class overlap (band0)
    if len(band0_hard) >= 2:
        keys = list(band0_hard.keys())
        lines.append("")
        lines.append(f"Overlap of hardest {top_n} classes in band0 (same output k across runs)")
        base = set(band0_hard[keys[0]])
        for k in keys[1:]:
            base &= set(band0_hard[k])
        lines.append(f"  Intersection all of {keys}: {sorted(base)}")
        if len(keys) >= 2:
            a, b = keys[0], keys[1]
            inter = sorted(set(band0_hard[a]) & set(band0_hard[b]))
            lines.append(f"  Intersection {a} & {b}: {inter}")

    # Full-K overlap only for runs with same K
    ks = [len(L["val"]) for L in loaded]
    if len(set(ks)) == 1 and len(loaded) >= 2:
        keys2 = [L["prefix"] for L in loaded]
        s_all: Optional[set] = None
        for L in loaded:
            s = set(hardest_indices(L["val"], top_n))
            s_all = s if s_all is None else s_all & s
        lines.append("")
        lines.append(f"Overlap of hardest {top_n} over FULL class id (all runs share K={ks[0]})")
        lines.append(f"  Intersection: {sorted(s_all or [])}")

    report_path = os.path.join(out_dir, "hard_classes_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote figures + {report_path}")


if __name__ == "__main__":
    main()
