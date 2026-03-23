"""
Scan !previous_results (nested folders of JSON experiment outputs) and write CSV tables:

  - runs_all_detailed.csv — one row per JSON file with summary + train accuracy stats
  - configs_deduped_best_max_train.csv — one row per config (best max_train_acc kept), plus readable labels
  - operations_summary_all_runs.csv — one row per operation with aggregate train accuracy stats (all runs)
  - operations_summary_main_baseline.csv — train_frac == 0.5 and excludes max_train_samples in 250..3000
  - operations_summary_train_frac_0_5_including_sample_caps.csv — train_frac == 0.5 including capped-sample sweeps
  - operations_summary_train_frac_not_0_5.csv — operation summary filtered to train_frac != 0.5

Usage (from Grokking-Project):
    python train_spreadsheet/build_spreadsheet.py

    python train_spreadsheet/build_spreadsheet.py --root "!previous_results" --out train_spreadsheet/out
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _safe_float(xs: List[float], fn) -> Optional[float]:
    if not xs:
        return None
    try:
        return float(fn(xs))
    except (TypeError, ValueError):
        return None


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  skip (read error): {path}  ({e})", file=sys.stderr)
        return None


def _row_from_file(path: Path, root: Path) -> Optional[Dict[str, Any]]:
    data = _load_json(path)
    if not data:
        return None
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    parts = rel.parts
    collection = parts[0] if len(parts) > 1 else ""
    summary = data.get("summary") or {}
    train_accs = data.get("train_accs") or []
    log_epochs = data.get("log_epochs") or []

    final_acc = train_accs[-1] if train_accs else None
    max_acc = _safe_float(train_accs, max) if train_accs else None
    min_acc = _safe_float(train_accs, min) if train_accs else None
    last_ep = log_epochs[-1] if log_epochs else None
    n_points = len(train_accs)

    row: Dict[str, Any] = {
        "collection_folder": collection,
        "relative_path": rel.as_posix(),
        "filename": path.name,
        "final_train_acc": final_acc,
        "max_train_acc": max_acc,
        "min_train_acc": min_acc,
        "last_logged_epoch": last_ep,
        "n_log_points": n_points,
    }
    # Flatten summary (handles older JSON without branch_* keys)
    for k, v in summary.items():
        row[k] = v
    return row


# Keys used to treat two runs as the "same" hyperparameter setting (excluding path / filename)
_CONFIG_KEYS = (
    "operation",
    "p",
    "train_frac",
    "max_train_samples",
    "input_format",
    "weight_decay",
    "lr",
    "d_model",
    "num_layers",
    "branch_metric",
    "branch_label_1",
    "branch_label_2",
    "last_logged_epoch",
)


def _config_tuple(row: Dict[str, Any]) -> Tuple[Any, ...]:
    return tuple(row.get(k) for k in _CONFIG_KEYS)


def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt_cell(r.get(k)) for k in fieldnames})


def _fmt_cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return repr(v)
    return str(v)


def _collect_fieldnames(rows: List[Dict[str, Any]]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    priority = [
        "collection_folder",
        "relative_path",
        "filename",
        "operation",
        "p",
        "train_frac",
        "max_train_samples",
        "input_format",
        "weight_decay",
        "lr",
        "d_model",
        "num_layers",
        "branch_metric",
        "branch_label_1",
        "branch_label_2",
        "final_train_acc",
        "max_train_acc",
        "min_train_acc",
        "last_logged_epoch",
        "n_log_points",
        "memo_epoch",
        "grok_epoch",
        "grok_gap",
        "elapsed_sec",
        "config_id",
        "config_changes_vs_default",
        "is_default_like_run",
        "notes",
    ]
    for k in priority:
        if any(k in r for r in rows):
            if k not in seen:
                seen.add(k)
                ordered.append(k)
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                ordered.append(k)
    return ordered


def _dedupe_best_max(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One row per distinct hyperparameter tuple; keep the row with highest max_train_acc."""
    best: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    counts: Dict[Tuple[Any, ...], int] = defaultdict(int)
    for r in rows:
        key = _config_tuple(r)
        counts[key] += 1
        cur = r.get("max_train_acc")
        if key not in best:
            best[key] = dict(r)
            best[key]["dedupe_group_count"] = 1
            continue
        prev = best[key].get("max_train_acc")
        if cur is not None and (prev is None or cur > prev):
            merged = dict(r)
            merged["dedupe_group_count"] = counts[key]
            best[key] = merged
        else:
            best[key]["dedupe_group_count"] = counts[key]
    out = list(best.values())
    out.sort(
        key=lambda x: (
            -(x.get("max_train_acc") if isinstance(x.get("max_train_acc"), (int, float)) else -1.0),
            str(x.get("operation", "")),
            str(x.get("collection_folder", "")),
        )
    )
    return out


def _get_num(row: Dict[str, Any], key: str, default: Optional[float] = None) -> Optional[float]:
    v = row.get(key)
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _add_readable_config_columns(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Add easy-to-scan fields so CSVs are understandable without context.
    """
    out: List[Dict[str, Any]] = []
    for r in rows:
        rr = dict(r)
        op = str(rr.get("operation", "unknown"))
        p = rr.get("p", "?")
        wd = rr.get("weight_decay", "?")
        lr = rr.get("lr", "?")
        d = rr.get("d_model", "?")
        l = rr.get("num_layers", "?")
        tf = rr.get("train_frac", "?")
        mts = rr.get("max_train_samples")
        fmt = rr.get("input_format", "?")

        rr["config_id"] = f"{op}__p{p}__wd{wd}__lr{lr}__d{d}__l{l}__tf{tf}__n{mts if mts is not None else 'full'}__fmt{fmt}"

        changes: List[str] = []
        if _get_num(rr, "weight_decay") not in (1.0, None):
            changes.append(f"wd={wd}")
        if _get_num(rr, "lr") not in (0.001, None):
            changes.append(f"lr={lr}")
        if _get_num(rr, "d_model") not in (128.0, None):
            changes.append(f"d_model={d}")
        if _get_num(rr, "num_layers") not in (2.0, None):
            changes.append(f"layers={l}")
        if _get_num(rr, "train_frac") not in (0.5, None):
            changes.append(f"train_frac={tf}")
        if mts is not None:
            changes.append(f"max_train_samples={mts}")
        if str(fmt) != "a_op_b_eq":
            changes.append(f"input_format={fmt}")
        bm = rr.get("branch_metric")
        if bm not in (None, "", "b_parity"):
            changes.append(f"branch_metric={bm}")
        rr["config_changes_vs_default"] = "; ".join(changes) if changes else "default-like"
        rr["is_default_like_run"] = "yes" if not changes else "no"
        rr["notes"] = "sample-capped run" if _is_max_train_samples_cap_250_to_3000(mts) else ""
        out.append(rr)
    return out


def _operations_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_op: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        op = r.get("operation")
        if op is None:
            continue
        by_op[str(op)].append(r)

    summary: List[Dict[str, Any]] = []
    for op in sorted(by_op):
        rs = by_op[op]
        finals = [r["final_train_acc"] for r in rs if r.get("final_train_acc") is not None]
        maxes = [r["max_train_acc"] for r in rs if r.get("max_train_acc") is not None]
        summary.append(
            {
                "operation": op,
                "n_runs": len(rs),
                "final_train_acc_mean": sum(finals) / len(finals) if finals else "",
                "final_train_acc_min": min(finals) if finals else "",
                "final_train_acc_max": max(finals) if finals else "",
                "max_train_acc_best_over_runs": max(maxes) if maxes else "",
                "max_train_acc_mean_over_runs": sum(maxes) / len(maxes) if maxes else "",
            }
        )
    summary.sort(
        key=lambda x: (
            -(x.get("max_train_acc_best_over_runs") if isinstance(x.get("max_train_acc_best_over_runs"), (int, float)) else -1.0),
            str(x.get("operation", "")),
        )
    )
    return summary


def _is_train_frac_0_5(v: Any) -> bool:
    """Treat numerically equivalent values as 0.5."""
    if v is None:
        return False
    try:
        return abs(float(v) - 0.5) < 1e-12
    except (TypeError, ValueError):
        return False


def _is_max_train_samples_cap_250_to_3000(v: Any) -> bool:
    """True if max_train_samples is set and in [250, 3000] (sample-size sweep runs)."""
    if v is None:
        return False
    try:
        n = int(float(v))
    except (TypeError, ValueError):
        return False
    return 250 <= n <= 3000


def main() -> None:
    ap = argparse.ArgumentParser(description="Build CSV spreadsheets from saved JSON results.")
    ap.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root folder to scan recursively for *.json (default: ../!previous_results next to this script's project)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory for CSV files (default: train_spreadsheet/out next to this script)",
    )
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    root = args.root
    if root is None:
        root = here.parent / "!previous_results"
    out_dir = args.out if args.out is not None else here / "out"

    if not root.is_dir():
        print(f"Error: root not found: {root}", file=sys.stderr)
        sys.exit(1)

    json_files = sorted(root.rglob("*.json"))
    if not json_files:
        print(f"No JSON files under {root}", file=sys.stderr)
        sys.exit(1)

    rows: List[Dict[str, Any]] = []
    for p in json_files:
        row = _row_from_file(p, root)
        if row:
            rows.append(row)

    if not rows:
        print("No valid JSON rows loaded.", file=sys.stderr)
        sys.exit(1)

    rows = _add_readable_config_columns(rows)
    rows.sort(
        key=lambda x: (
            -(x.get("max_train_acc") if isinstance(x.get("max_train_acc"), (int, float)) else -1.0),
            str(x.get("operation", "")),
            str(x.get("relative_path", "")),
        )
    )

    fieldnames = _collect_fieldnames(rows)
    _write_csv(out_dir / "runs_all_detailed.csv", fieldnames, rows)

    unique = _dedupe_best_max(rows)
    ufields = _collect_fieldnames(unique)
    _write_csv(out_dir / "configs_deduped_best_max_train.csv", ufields, unique)

    ops = _operations_summary(rows)
    op_fields = list(ops[0].keys()) if ops else []
    _write_csv(out_dir / "operations_summary_all_runs.csv", op_fields, ops)

    rows_tf_0_5 = [r for r in rows if _is_train_frac_0_5(r.get("train_frac"))]
    rows_tf_0_5_no_mts_sweep = [
        r
        for r in rows_tf_0_5
        if not _is_max_train_samples_cap_250_to_3000(r.get("max_train_samples"))
    ]
    rows_tf_not_0_5 = [r for r in rows if not _is_train_frac_0_5(r.get("train_frac"))]

    # Main baseline: train_frac=0.5 but drop max_train_samples 250..3000 (inflated train acc)
    ops_tf_0_5 = _operations_summary(rows_tf_0_5_no_mts_sweep)
    fields_tf_0_5 = list(ops_tf_0_5[0].keys()) if ops_tf_0_5 else op_fields
    _write_csv(out_dir / "operations_summary_main_baseline.csv", fields_tf_0_5, ops_tf_0_5)

    ops_tf_0_5_incl = _operations_summary(rows_tf_0_5)
    fields_tf_0_5_incl = list(ops_tf_0_5_incl[0].keys()) if ops_tf_0_5_incl else op_fields
    _write_csv(
        out_dir / "operations_summary_train_frac_0_5_including_sample_caps.csv",
        fields_tf_0_5_incl,
        ops_tf_0_5_incl,
    )

    ops_tf_not_0_5 = _operations_summary(rows_tf_not_0_5)
    fields_tf_not_0_5 = list(ops_tf_not_0_5[0].keys()) if ops_tf_not_0_5 else op_fields
    _write_csv(
        out_dir / "operations_summary_train_frac_not_0_5.csv",
        fields_tf_not_0_5,
        ops_tf_not_0_5,
    )

    print(f"Wrote {len(rows)} runs -> {out_dir / 'runs_all_detailed.csv'}")
    print(f"Wrote {len(unique)} deduped configs -> {out_dir / 'configs_deduped_best_max_train.csv'}")
    print(f"Wrote {len(ops)} operations (all) -> {out_dir / 'operations_summary_all_runs.csv'}")
    print(
        f"Wrote {len(ops_tf_0_5)} operations (train_frac=0.5, excl max_train_samples 250..3000) -> "
        f"{out_dir / 'operations_summary_main_baseline.csv'}"
    )
    print(
        f"Wrote {len(ops_tf_0_5_incl)} operations (train_frac=0.5, incl mts sweep) -> "
        f"{out_dir / 'operations_summary_train_frac_0_5_including_sample_caps.csv'}"
    )
    print(
        f"Wrote {len(ops_tf_not_0_5)} operations (train_frac!=0.5) -> "
        f"{out_dir / 'operations_summary_train_frac_not_0_5.csv'}"
    )


if __name__ == "__main__":
    main()
