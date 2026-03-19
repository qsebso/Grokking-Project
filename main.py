"""
main.py — Default grokking experiment runner
=============================================

This is the central file every experiment branches from.
Run it as-is to reproduce the baseline (modular addition mod 97, 50/50 split).

USAGE
-----
# Baseline (reproduces the notebook result)
    python main.py

# Override any config field via CLI:
    python main.py --operation sub --weight_decay 0.1
    python main.py --lr 3e-4 --num_epochs 10000
    python main.py --operation add --d_model 256 --num_layers 4

# Run multiple values of a single param (e.g. weight-decay sweep):
    python main.py --sweep weight_decay 0.0 0.1 1.0

# Run all registered operations:
    python main.py --sweep operation add sub div sq_sum s5_mul

EXTENDING
---------
To add a new experiment module, just import TrainConfig / train here and
override the fields you care about.  Every field has a sensible default so
you only need to specify what changes.

See experiments/train.py  for TrainConfig documentation.
See data/dataset.py        for the full list of supported operations.
"""

import sys
import os
import argparse
import json

# ── make sure local packages are importable regardless of CWD ────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset     import make_dataset, OPERATIONS
from experiments.train import TrainConfig, train


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Grokking experiment runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── data ─────────────────────────────────────────────────────────────
    p.add_argument("--operation",    default="add",
                   help=f"Binary operation. One of: {sorted(OPERATIONS.keys())}")
    p.add_argument("--p",            type=int,   default=97,
                   help="Prime / modulus (ignored for S5 ops)")
    p.add_argument("--train_frac",   type=float, default=0.5,
                   help="Fraction of all pairs used for training")
    p.add_argument("--max_train_samples", type=int, default=None,
                   help="Hard cap on training set size (e.g. 1000). "
                        "Applied after train_frac split. Omit = no cap.")
    p.add_argument("--input_format", default="a_op_b_eq",
                   choices=["a_op_b_eq", "a_b_eq", "a_op_b_eq_rule"],
                   help="Token layout: "
                        "'a_op_b_eq' = [a,op,b,=] (default/paper), "
                        "'a_b_eq' = [a,b,=] (no op token), "
                        "'a_op_b_eq_rule' = [rule,a,op,b,=]")
    p.add_argument("--label_noise",  type=float, default=0.0,
                   help="Fraction of training labels to randomly corrupt")
    p.add_argument("--data_seed",    type=int,   default=42)

    # ── model ─────────────────────────────────────────────────────────────
    p.add_argument("--d_model",      type=int,   default=128)
    p.add_argument("--nhead",        type=int,   default=4)
    p.add_argument("--num_layers",   type=int,   default=2)
    p.add_argument("--dim_feedforward", type=int, default=None,
                   help="FFN inner dim. Default: 4 × d_model")

    # ── optimiser ────────────────────────────────────────────────────────
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1.0)

    # ── training ─────────────────────────────────────────────────────────
    p.add_argument("--num_epochs",   type=int,   default=5_000)
    p.add_argument("--batch_size",   type=int,   default=None,
                   help="Mini-batch size. Omit for full-batch (default)")
    p.add_argument("--log_every",    type=int,   default=50)

    # ── sweep (run multiple values of ONE param) ──────────────────────────
    p.add_argument("--sweep",        nargs="+",  default=None,
                   metavar=("PARAM", "VAL"),
                   help="Sweep a single param: --sweep weight_decay 0.0 0.1 1.0")

    # ── output ────────────────────────────────────────────────────────────
    p.add_argument("--results_dir",  default="results",
                   help="Directory to save JSON result files")
    p.add_argument("--quiet",        action="store_true",
                   help="Suppress per-epoch progress output")

    return p


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cast(value: str, default):
    """Try to cast a string to the same type as `default`."""
    if isinstance(default, bool):
        return value.lower() in ("true", "1", "yes")
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return value   # string


def _save_result(result, results_dir: str):
    os.makedirs(results_dir, exist_ok=True)
    cfg   = result.config
    mts   = f"_n{cfg.max_train_samples}" if cfg.max_train_samples else ""
    fmt   = f"_fmt{cfg.input_format}" if cfg.input_format != "a_op_b_eq" else ""
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
    data = {
        "summary":      result.summary(),
        "log_epochs":   result.log_epochs,
        "train_accs":   result.train_accs,
        "val_accs":     result.val_accs,
        "train_losses": result.train_losses,
        "val_losses":   result.val_losses,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  ↳ saved to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Run a single experiment
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
    print(f"{'='*60}")

    train_ds, val_ds, vocab_size = make_dataset(
        operation          = cfg.operation,
        p                  = cfg.p,
        train_frac         = cfg.train_frac,
        max_train_samples  = cfg.max_train_samples,
        input_format       = cfg.input_format,
        seed               = cfg.data_seed,
        label_noise        = cfg.label_noise,
    )
    print(f"  train={len(train_ds):,}  val={len(val_ds):,}  vocab={vocab_size}")

    result = train(train_ds, val_ds, vocab_size, cfg)
    _save_result(result, results_dir)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = build_parser()
    args   = parser.parse_args()

    # ── build base config from CLI args ───────────────────────────────────
    base_cfg = TrainConfig(
        operation          = args.operation,
        p                  = args.p,
        train_frac         = args.train_frac,
        max_train_samples  = args.max_train_samples,
        input_format       = args.input_format,
        label_noise        = args.label_noise,
        data_seed          = args.data_seed,
        d_model            = args.d_model,
        nhead              = args.nhead,
        num_layers         = args.num_layers,
        dim_feedforward    = args.dim_feedforward,
        lr                 = args.lr,
        weight_decay       = args.weight_decay,
        num_epochs         = args.num_epochs,
        batch_size         = args.batch_size,
        log_every          = args.log_every,
        verbose            = not args.quiet,
    )

    # ── sweep mode ────────────────────────────────────────────────────────
    if args.sweep is not None:
        if len(args.sweep) < 2:
            parser.error("--sweep requires PARAM VAL [VAL ...]")

        param  = args.sweep[0]
        values = args.sweep[1:]

        if not hasattr(base_cfg, param):
            parser.error(f"Unknown TrainConfig field: '{param}'")

        default_val = getattr(base_cfg, param)

        print(f"\n[SWEEP] {param} ∈ {values}")
        results = []
        for v in values:
            cfg = TrainConfig(**vars(base_cfg))   # copy
            setattr(cfg, param, _cast(v, default_val))
            r = run_one(cfg, args.results_dir)
            results.append(r)

        # ── compact sweep summary ────────────────────────────────────────
        print(f"\n{'─'*60}")
        print(f"  SWEEP SUMMARY: {param}")
        print(f"{'─'*60}")
        for r in results:
            print(
                f"  {param}={getattr(r.config, param):<10}"
                f"  memo={r.memo_epoch}  "
                f"grok={r.grok_epoch}  "
                f"gap={r.grok_gap}"
            )

    # ── single run ────────────────────────────────────────────────────────
    else:
        run_one(base_cfg, args.results_dir)


if __name__ == "__main__":
    main()
