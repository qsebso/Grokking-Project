"""
main.py — Default grokking experiment runner
=============================================

USAGE
-----
# Baseline
    python main.py

# Single run with overrides
    python main.py --operation sub --weight_decay 0.1 --num_epochs 10000

# 1D sweep (one param, multiple values)
    python main.py --sweep weight_decay 0.0 0.1 1.0
    python main.py --sweep max_train_samples 250 500 1000 2000
    python main.py --sweep operation add sub div sq_sum

# 2D grid sweep (two params — runs every combination)
    python main.py --grid1 weight_decay 0.0 0.1 1.0 --grid2 lr 1e-4 1e-3
    python main.py --grid1 weight_decay 0.0 1.0 --grid2 max_train_samples 250 500 1000 --num_epochs 20000
"""

import sys
import os
import argparse
import json
import itertools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset      import make_dataset, OPERATIONS, resolve_branch_metric, branch_metric_labels
from experiments.train import TrainConfig, train


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Grokking experiment runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── data ──────────────────────────────────────────────────────────────
    p.add_argument("--operation",    default="add",
                   help=f"One of: {sorted(OPERATIONS.keys())}")
    p.add_argument("--p",            type=int,   default=97)
    p.add_argument("--train_frac",   type=float, default=0.5)
    p.add_argument("--max_train_samples", type=int, default=None,
                   help="Hard cap on training set size. Omit = no cap.")
    p.add_argument("--input_format", default="a_op_b_eq",
                   choices=["a_op_b_eq", "a_b_eq", "a_op_b_eq_rule"])
    p.add_argument("--label_noise",  type=float, default=0.0)
    p.add_argument("--data_seed",    type=int,   default=42)

    # ── model ─────────────────────────────────────────────────────────────
    p.add_argument("--d_model",      type=int,   default=128)
    p.add_argument("--nhead",        type=int,   default=4)
    p.add_argument("--num_layers",   type=int,   default=2)
    p.add_argument("--dim_feedforward", type=int, default=None)

    # ── optimiser ─────────────────────────────────────────────────────────
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1.0)

    # ── training ──────────────────────────────────────────────────────────
    p.add_argument("--num_epochs",   type=int,   default=5_000)
    p.add_argument("--batch_size",   type=int,   default=None)
    p.add_argument("--log_every",    type=int,   default=50)
    p.add_argument(
        "--branch_metric",
        default="auto",
        choices=["auto", "b_parity", "a_ge_b", "a_gt_b"],
        help="Train/val branch accuracies: auto infers from operation (e.g. a>=b for add_or_mul_on_a_greater_than_b); "
        "b_parity = odd/even b; a_ge_b / a_gt_b compare a vs b.",
    )

    # ── sweep modes ───────────────────────────────────────────────────────
    p.add_argument("--sweep", nargs="+", default=None,
                   metavar=("PARAM", "VAL"),
                   help="1D sweep: --sweep weight_decay 0.0 0.1 1.0")
    p.add_argument("--grid1", nargs="+", default=None,
                   metavar=("PARAM", "VAL"),
                   help="2D grid param 1: --grid1 weight_decay 0.0 0.1 1.0")
    p.add_argument("--grid2", nargs="+", default=None,
                   metavar=("PARAM", "VAL"),
                   help="2D grid param 2: --grid2 max_train_samples 250 500 1000")

    # ── output ────────────────────────────────────────────────────────────
    p.add_argument("--results_dir",  default="results")
    p.add_argument("--quiet",        action="store_true")

    return p


# ─────────────────────────────────────────────────────────────────────────────
# Type casting
# ─────────────────────────────────────────────────────────────────────────────

def _cast(value: str, default):
    """
    Cast a string value to the right type.
    Uses default as a type hint; if default is None, tries int -> float -> str.
    """
    if isinstance(default, bool):
        return value.lower() in ("true", "1", "yes")
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    if default is None:
        # covers max_train_samples, batch_size, dim_feedforward
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
    return value  # string (operation name, input_format, etc.)


# ─────────────────────────────────────────────────────────────────────────────
# Grid argument parsing   --grid P1 v v -- P2 v v
# ─────────────────────────────────────────────────────────────────────────────

def _parse_grid_arg(tokens: list) -> tuple:
    """
    Parse tokens like: weight_decay 0.0 1.0 -- lr 1e-4 1e-3
    Returns (param1, values1, param2, values2)
    """
    if "--" not in tokens:
        raise ValueError(
            "--grid needs two param groups separated by '--'.\n"
            "Example: --grid weight_decay 0.0 1.0 -- lr 1e-4 1e-3"
        )
    sep    = tokens.index("--")
    group1 = tokens[:sep]
    group2 = tokens[sep + 1:]

    if len(group1) < 2 or len(group2) < 2:
        raise ValueError("Each group needs a param name and at least one value.")

    return group1[0], group1[1:], group2[0], group2[1:]


# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────

def _save_result(result, results_dir: str):
    os.makedirs(results_dir, exist_ok=True)
    cfg   = result.config
    mts   = f"_n{cfg.max_train_samples}" if cfg.max_train_samples else ""
    fmt   = f"_fmt{cfg.input_format}"    if cfg.input_format != "a_op_b_eq" else ""
    fname = (
        f"{cfg.operation}_p{cfg.p}"
        f"_wd{cfg.weight_decay}"
        f"_lr{cfg.lr}"
        f"_d{cfg.d_model}"
        f"_l{cfg.num_layers}"
        f"_tf{cfg.train_frac}"
        f"{mts}{fmt}"
        f"_ep{cfg.num_epochs}.json"
    )
    path = os.path.join(results_dir, fname)
    with open(path, "w") as f:
        json.dump({
            "summary":      result.summary(),
            "branch_metric":   result.branch_metric,
            "branch_label_1":  result.branch_label_1,
            "branch_label_2":  result.branch_label_2,
            "log_epochs":   result.log_epochs,
            "train_accs":   result.train_accs,
            "val_accs":     result.val_accs,
            "train_odd_accs": result.train_odd_accs,
            "train_even_accs": result.train_even_accs,
            "val_odd_accs": result.val_odd_accs,
            "val_even_accs": result.val_even_accs,
            "train_losses": result.train_losses,
            "val_losses":   result.val_losses,
        }, f, indent=2)
    print(f"  -> saved to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Run one experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_one(cfg: TrainConfig, results_dir: str):
    print(f"\n{'='*60}")
    print(f"  operation          = {cfg.operation}")
    print(f"  p                  = {cfg.p}")
    print(f"  train_frac         = {cfg.train_frac}")
    print(f"  max_train_samples  = {cfg.max_train_samples if cfg.max_train_samples else 'none'}")
    print(f"  input_format       = {cfg.input_format}")
    print(f"  weight_decay       = {cfg.weight_decay}")
    print(f"  lr                 = {cfg.lr}")
    print(f"  d_model            = {cfg.d_model}")
    print(f"  num_layers         = {cfg.num_layers}")
    print(f"  num_epochs         = {cfg.num_epochs}")
    print(f"  batch_size         = {cfg.batch_size if cfg.batch_size else 'full'}")
    _bm = resolve_branch_metric(cfg.operation, cfg.branch_metric)
    _l1, _l2 = branch_metric_labels(_bm)
    print(f"  branch_metric      = {cfg.branch_metric}  ->  {_bm}  ({_l1} / {_l2})")
    print(f"{'='*60}")

    train_ds, val_ds, vocab_size = make_dataset(
        operation         = cfg.operation,
        p                 = cfg.p,
        train_frac        = cfg.train_frac,
        max_train_samples = cfg.max_train_samples,
        input_format      = cfg.input_format,
        seed              = cfg.data_seed,
        label_noise       = cfg.label_noise,
    )
    print(f"  train={len(train_ds):,}  val={len(val_ds):,}  vocab={vocab_size}")

    result = train(train_ds, val_ds, vocab_size, cfg)
    _save_result(result, results_dir)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Summary printers
# ─────────────────────────────────────────────────────────────────────────────

def _print_1d_summary(param: str, results: list):
    print(f"\n{'─'*60}")
    print(f"  SWEEP SUMMARY: {param}")
    print(f"{'─'*60}")
    for r in results:
        print(
            f"  {param}={str(getattr(r.config, param)):<12}"
            f"  memo={str(r.memo_epoch):<8}"
            f"  grok={str(r.grok_epoch):<8}"
            f"  gap={r.grok_gap}"
        )


def _print_2d_summary(p1: str, p2: str, results: list):
    print(f"\n{'─'*60}")
    print(f"  GRID SUMMARY: {p1}  x  {p2}")
    print(f"{'─'*60}")
    for r in results:
        v1 = getattr(r.config, p1)
        v2 = getattr(r.config, p2)
        print(
            f"  {p1}={str(v1):<10}  {p2}={str(v2):<10}"
            f"  memo={str(r.memo_epoch):<8}"
            f"  grok={str(r.grok_epoch):<8}"
            f"  gap={r.grok_gap}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = build_parser()
    args   = parser.parse_args()

    # ── base config ───────────────────────────────────────────────────────
    base_cfg = TrainConfig(
        operation         = args.operation,
        p                 = args.p,
        train_frac        = args.train_frac,
        max_train_samples = args.max_train_samples,
        input_format      = args.input_format,
        label_noise       = args.label_noise,
        data_seed         = args.data_seed,
        d_model           = args.d_model,
        nhead             = args.nhead,
        num_layers        = args.num_layers,
        dim_feedforward   = args.dim_feedforward,
        lr                = args.lr,
        weight_decay      = args.weight_decay,
        num_epochs        = args.num_epochs,
        batch_size        = args.batch_size,
        log_every         = args.log_every,
        verbose           = not args.quiet,
        branch_metric     = args.branch_metric,
    )

    # ── 2D grid ───────────────────────────────────────────────────────────
    if args.grid1 is not None and args.grid2 is not None:
        p1, vals1 = args.grid1[0], args.grid1[1:]
        p2, vals2 = args.grid2[0], args.grid2[1:]
        if len(vals1) == 0 or len(vals2) == 0:
            parser.error("--grid1 and --grid2 each need a param name and at least one value.")

        for param in (p1, p2):
            if not hasattr(base_cfg, param):
                parser.error(f"Unknown TrainConfig field: '{param}'")

        total = len(vals1) * len(vals2)
        print(f"\n[GRID] {p1} {vals1}  x  {p2} {vals2}  ->  {total} runs")

        results = []
        for v1, v2 in itertools.product(vals1, vals2):
            cfg = TrainConfig(**vars(base_cfg))
            setattr(cfg, p1, _cast(v1, getattr(base_cfg, p1)))
            setattr(cfg, p2, _cast(v2, getattr(base_cfg, p2)))
            r = run_one(cfg, args.results_dir)
            results.append(r)

        _print_2d_summary(p1, p2, results)

    # ── 1D sweep ──────────────────────────────────────────────────────────
    elif args.sweep is not None:
        if len(args.sweep) < 2:
            parser.error("--sweep requires PARAM VAL [VAL ...]")

        param  = args.sweep[0]
        values = args.sweep[1:]

        if not hasattr(base_cfg, param):
            parser.error(f"Unknown TrainConfig field: '{param}'")

        print(f"\n[SWEEP] {param} in {values}")
        results = []
        for v in values:
            cfg = TrainConfig(**vars(base_cfg))
            setattr(cfg, param, _cast(v, getattr(base_cfg, param)))
            r = run_one(cfg, args.results_dir)
            results.append(r)

        _print_1d_summary(param, results)

    # ── single run ────────────────────────────────────────────────────────
    else:
        run_one(base_cfg, args.results_dir)


if __name__ == "__main__":
    main()