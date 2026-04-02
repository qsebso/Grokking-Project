#!/usr/bin/env python3
"""
Decode (a, b) from token batches, show rule_id vs b-parity / op branch, and class balance.

Run from repo root::

    python analysis/verify_rule_labels.py
    python analysis/verify_rule_labels.py --p 11 --train_frac 0.5 --seed 42
    python analysis/verify_rule_labels.py --input_format a_op_b_eq_bparity
"""

from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import (
    OPERATIONS,
    make_category_dataset,
    resolve_rule_id,
    encode_disjoint_rule_output,
    resolve_branch_metric,
)

INPUT_CHOICES = [
    "a_op_b_eq",
    "a_b_eq",
    "a_op_b_eq_rule",
    "a_op_b_eq_bparity",
    "a_op_bparity_eq",
]


def split_pairs_like_dataset(
    operation: str,
    p: int,
    train_frac: float,
    seed: int,
    max_train_samples: int | None = None,
):
    """Same shuffle/split as ``make_category_dataset`` (for aligning row i -> (a,b))."""
    _, _, domain_fn = OPERATIONS[operation]
    all_pairs = domain_fn(p)
    rng = random.Random(seed)
    rng.shuffle(all_pairs)
    n_train = int(len(all_pairs) * train_frac)
    train_pairs = all_pairs[:n_train]
    val_pairs = all_pairs[n_train:]
    if max_train_samples is not None and max_train_samples < len(train_pairs):
        train_pairs = train_pairs[:max_train_samples]
    return train_pairs, val_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--operation", default="add_or_mul")
    ap.add_argument("--p", type=int, default=97)
    ap.add_argument("--train_frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--input_format", default="a_op_b_eq", choices=INPUT_CHOICES)
    ap.add_argument("--n_print", type=int, default=16, help="how many train examples to print")
    ap.add_argument(
        "--noise",
        type=float,
        default=None,
        help="Asymmetric train label noise; overrides --label_noise if set.",
    )
    ap.add_argument("--label_noise", type=float, default=0.0)
    ap.add_argument("--noise_sym", type=float, default=0.0, help="Symmetric pair-swap noise on train.")
    args = ap.parse_args()
    _asy = args.noise if args.noise is not None else args.label_noise

    op_fn, _, _domain = OPERATIONS[args.operation]
    p = args.p

    train_ds, val_ds, vocab_size, num_classes = make_category_dataset(
        operation=args.operation,
        p=p,
        train_frac=args.train_frac,
        input_format=args.input_format,
        seed=args.seed,
        label_noise=_asy,
        label_noise_sym=args.noise_sym,
        label_mode="c",
        label_mod=3,
        rule_count=2,
    )

    train_pairs, val_pairs = split_pairs_like_dataset(
        args.operation, p, args.train_frac, args.seed
    )

    bm = resolve_branch_metric(args.operation, "auto")
    print(f"operation={args.operation!r}  p={p}  train_frac={args.train_frac}  seed={args.seed}")
    print(f"input_format={args.input_format!r}  resolved branch_metric={bm!r}")
    print(f"vocab_size={vocab_size}  num_classes={num_classes}  train={len(train_ds)}  val={len(val_ds)}")
    print()

    # Full grid (before shuffle/split): exact rule balance
    all_pairs = _domain(p)
    rid_full = [resolve_rule_id(args.operation, a, b, p) for a, b in all_pairs]
    n0_full = sum(1 for r in rid_full if r == 0)
    n1_full = sum(1 for r in rid_full if r == 1)
    print("Full domain (all p*p pairs, before shuffle/split):")
    print(f"  rule_id=0: {n0_full:,}  ({100 * n0_full / len(all_pairs):.4f}%)")
    print(f"  rule_id=1: {n1_full:,}  ({100 * n1_full / len(all_pairs):.4f}%)")
    print()

    def scan(ds, pair_list, name: str):
        n0 = n1 = 0
        mismatch = 0
        for i in range(len(ds)):
            _, y = ds[i]
            a, b = pair_list[i]
            rid = resolve_rule_id(args.operation, a, b, p)
            c_local = op_fn(a, b, p)
            y_exp = encode_disjoint_rule_output(c_local, rid, p)
            if int(y.item()) != y_exp:
                mismatch += 1
            if rid == 0:
                n0 += 1
            else:
                n1 += 1
        print(f"{name} (n={len(ds):,}): rule_id=0 -> {n0:,} ({100 * n0 / len(ds):.4f}%)  |  rule_id=1 -> {n1:,} ({100 * n1 / len(ds):.4f}%)")
        if mismatch:
            print(f"  WARNING: {mismatch} rows where y != encode_disjoint(c_local, rule_id)")
        else:
            print("  OK: all labels match encode_disjoint(op(a,b), resolve_rule_id(...)).")

    scan(train_ds, train_pairs, "Train")
    scan(val_ds, val_pairs, "Val")
    print()

    # Manual rows from train (ground-truth a,b from pair list)
    print(f"First {args.n_print} training examples (ground-truth a,b from split; tokens for parity check):")
    hdr = (
        "  i    a    b  b%2  rule_id  "
        "branch(op)      c_local  y     y//p  y%p  tok_parity  check"
    )
    print(hdr)
    for i in range(min(args.n_print, len(train_ds))):
        x, y = train_ds[i]
        a, b = train_pairs[i]
        rid = resolve_rule_id(args.operation, a, b, p)
        c_local = op_fn(a, b, p)
        y_int = int(y.item())
        branch = "add" if (b % 2 == 1) else "mul"
        expect_rid = 1 - (b & 1)
        ok = "OK" if rid == expect_rid else "BAD"

        if args.input_format == "a_op_b_eq_bparity":
            tok_p = int(x[3].item())
            parity_note = f"{tok_p} (want {b % 2})"
        elif args.input_format == "a_op_bparity_eq":
            tok_p = int(x[2].item())
            parity_note = f"{tok_p} (want {b % 2})"
        else:
            parity_note = "-"

        print(
            f"{i:4d}  {a:4d}  {b:4d}   {b % 2}      {rid}      "
            f"{branch:8s}      {c_local:4d}  {y_int:5d}   {y_int // p:4d}  {y_int % p:3d}  "
            f"{parity_note:16s}  {ok}"
        )

    print()
    print(
        "Notes for add_or_mul: op uses add when b is odd, mul when b is even. "
        "resolve_rule_id uses _rid_b_odd_branch0: rule_id = 1 - (b&1), so "
        "rule_id 0 = odd b (add), rule_id 1 = even b (mul). "
        "Branch metric 'b_parity' (odd/even) lines up with rule_id 0/1."
    )


if __name__ == "__main__":
    main()
