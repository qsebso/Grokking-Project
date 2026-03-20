from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


TRACK_FIELDS = [
    "operation",
    "p",
    "train_frac",
    "max_train_samples",
    "input_format",
    "weight_decay",
    "lr",
    "d_model",
    "num_layers",
    "num_epochs",
]


def last(xs):
    return xs[-1] if xs else None


def best(xs):
    return max(xs) if xs else None


def load_row(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data.get("summary", {}) or {}

    train_accs = data.get("train_accs", []) or []
    val_accs = data.get("val_accs", []) or []

    return {
        "file": path.name,
        **summary,
        "final_train_acc": last(train_accs),
        "final_val_acc": last(val_accs),
        "best_val_acc": best(val_accs),
    }


def fmt(x):
    if x is None:
        return "None"
    if isinstance(x, float):
        return f"{x:g}"
    return str(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--output", default="organized_results/summary.txt")
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / args.results_dir
    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    json_paths = sorted(results_dir.rglob("*.json") if args.recursive else results_dir.glob("*.json"))
    rows = [load_row(p) for p in json_paths]

    if not rows:
        output_path.write_text("No results found.\n", encoding="utf-8")
        print(f"No results found. Wrote: {output_path}")
        return

    lines = []

    # header
    lines.append("RESULTS SUMMARY")
    lines.append("=" * 60)
    lines.append(f"Results dir: {results_dir}")
    lines.append(f"Run count:   {len(rows)}")

    # operations
    op_counts = Counter(r.get("operation") for r in rows)
    lines.append("\nRuns by operation")
    lines.append("-" * 60)
    for op, count in sorted(op_counts.items()):
        lines.append(f"{fmt(op):20} {count}")

    # field values
    lines.append("\nValues seen")
    lines.append("-" * 60)
    varying = []
    for field in TRACK_FIELDS:
        vals = sorted({r.get(field) for r in rows}, key=lambda x: str(x))
        lines.append(f"{field:20} {', '.join(fmt(v) for v in vals)}")
        if len(vals) > 1:
            varying.append(field)

    # varying fields
    lines.append("\nFields that varied")
    lines.append("-" * 60)
    if varying:
        for f in varying:
            lines.append(f)
    else:
        lines.append("None")

    # best per operation
    lines.append("\nBest per operation")
    lines.append("-" * 60)
    by_op = defaultdict(list)
    for r in rows:
        by_op[r.get("operation")].append(r)

    for op in sorted(by_op):
        best_row = max(
            by_op[op],
            key=lambda r: float("-inf") if r.get("best_val_acc") is None else r["best_val_acc"],
        )
        lines.append(
            f"{fmt(op):20} best_val={fmt(best_row.get('best_val_acc')):10} file={best_row['file']}"
        )

    # per-run list
    lines.append("\nRuns")
    lines.append("-" * 60)
    for r in rows:
        lines.append(
            f"{r['file']} | op={fmt(r.get('operation'))} | "
            f"lr={fmt(r.get('lr'))} | wd={fmt(r.get('weight_decay'))} | "
            f"d={fmt(r.get('d_model'))} | layers={fmt(r.get('num_layers'))} | "
            f"val={fmt(r.get('best_val_acc'))}"
        )

    # write file
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote summary to: {output_path}")


if __name__ == "__main__":
    main()