"""
Categorical targets with the same transformer backbone as ``experiments.train``.

Problem choice (label_mode)
----------------------------
- **c_parity** / **c_mod3**: class depends on the true output ``c``. The model must (implicitly)
  compute the operation; this matches "grok the map then read a simple statistic".
- **b_parity** / **a_parity**: class is fixed from an input token already in the sequence.
  Often much easier and can memorize without grokking the binary op.
- **a_plus_b_mod3** / **a_plus_b_mod** (with ``--label_mod k``): class is ``(a+b) % k``.  Large ``k``
  (e.g. 11, 17) makes a **high-cardinality** statistic from inputs only (harder memorization landscape).
- **c**: class is the full true output ``c`` in ``0..p-1`` (``num_classes = p``) — same targets as
  token prediction, but cross-entropy on class ids instead of ``p``-way vocab logits at the decoder.

Use **c_parity** (2-way) or **c_mod** / **c_mod3** when the target should depend on the **true** ``c``.

Usage (from Grokking-Project)::

    python -m experiments.train_category --operation add_or_mul --label_mode c_parity

See ``data.dataset.make_category_dataset`` for all ``label_mode`` values.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import make_category_dataset, resolve_branch_metric, branch_metric_labels
from experiments.train import (
    TrainConfig,
    TrainResult,
    _branch_masks,
    _masked_acc,
)


def _per_class_accs_counts(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> Tuple[List[Optional[float]], List[int]]:
    """Per-class accuracy among samples with that label; count = support per class."""
    correct = preds == targets
    accs: List[Optional[float]] = []
    counts: List[int] = []
    for k in range(num_classes):
        mask = targets == k
        n = int(mask.sum().item())
        counts.append(n)
        if n == 0:
            accs.append(None)
        else:
            accs.append(correct[mask].float().mean().item())
    return accs, counts


def _slash_class_pcts(
    accs: List[Optional[float]],
    counts: List[int],
) -> str:
    """``82/88/92/100`` = accuracies for class 0,1,2,… in percent (empty → —)."""
    segs: List[str] = []
    for a, n in zip(accs, counts):
        if n == 0 or a is None:
            segs.append("—")
        else:
            segs.append(f"{100.0 * a:.0f}")
    return "/".join(segs)


def _pct_or_dash(x: Optional[float]) -> str:
    if x is None:
        return "—"
    return f"{100.0 * x:.0f}%"


def _class_acc_summary(accs: List[Optional[float]]) -> str:
    """min / median / max over classes that have support (percent)."""
    xs = [100.0 * a for a in accs if a is not None]
    if not xs:
        return "—"
    return f"min {min(xs):.0f}%  p50 {statistics.median(xs):.0f}%  max {max(xs):.0f}%"


@dataclass
class TrainCategoryConfig(TrainConfig):
    """Same hyperparameters as ``TrainConfig`` + categorical label definition."""

    label_mode: str = "c_parity"
    label_mod: int = 3  # k for c_mod / a_plus_b_mod only
    num_classes: Optional[int] = None  # filled before training


def train_category(
    train_ds: TensorDataset,
    val_ds: TensorDataset,
    vocab_size: int,
    num_classes: int,
    cfg: TrainCategoryConfig,
) -> TrainResult:
    from models.transformer import TransformerModel, count_parameters

    cfg.num_classes = num_classes

    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    if cfg.verbose:
        print(f"Device: {device}")
        extra = f"  label_mod={cfg.label_mod}" if cfg.label_mode in ("c_mod", "a_plus_b_mod") else ""
        rc_extra = f"  rule_count={cfg.rule_count} (disjoint bands)" if cfg.rule_count > 1 else ""
        print(f"Categorical task: label_mode={cfg.label_mode}{extra}{rc_extra}  num_classes={num_classes}")

    torch.manual_seed(cfg.model_seed)

    _seq_len_map = {
        "a_op_b_eq": 4,
        "a_b_eq": 3,
        "a_op_b_eq_rule": 5,
        "a_op_b_eq_bparity": 5,
        "a_op_bparity_eq": 4,
    }
    seq_len = _seq_len_map.get(cfg.input_format, 4)

    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dim_feedforward=cfg.dim_feedforward,
        seq_len=seq_len,
        num_logits=num_classes,
    ).to(device)

    if cfg.verbose:
        print(f"Parameters: {count_parameters(model):,}")
        print(
            "  Per-class accuracies: full tables align with log_epochs in saved JSON; "
            "console shows slash detail only for K<=10, else min/p50/max."
        )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=cfg.betas,
    )
    criterion = nn.CrossEntropyLoss()

    x_train, y_train = train_ds.tensors[0].to(device), train_ds.tensors[1].to(device)
    x_val, y_val = val_ds.tensors[0].to(device), val_ds.tensors[1].to(device)

    if cfg.batch_size is not None and cfg.batch_size < len(x_train):
        loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
        full_batch = False
    else:
        loader = None
        full_batch = True

    resolved_bm = resolve_branch_metric(cfg.operation, cfg.branch_metric)
    lb1, lb2 = branch_metric_labels(resolved_bm)

    result = TrainResult(config=cfg)
    result.branch_metric = resolved_bm
    result.branch_label_1 = lb1
    result.branch_label_2 = lb2
    t0 = time.time()

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        if full_batch:
            optimizer.zero_grad()
            logits = model(x_train)
            loss = criterion(logits, y_train)
            loss.backward()
            optimizer.step()
        else:
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

        if epoch % cfg.log_every == 0:
            model.eval()
            with torch.no_grad():
                tr_logits = model(x_train)
                tr_preds = tr_logits.argmax(-1)
                tr_loss = criterion(tr_logits, y_train).item()
                tr_acc = (tr_preds == y_train).float().mean().item()

                val_logits = model(x_val)
                val_preds = val_logits.argmax(-1)
                val_loss = criterion(val_logits, y_val).item()
                val_acc = (val_preds == y_val).float().mean().item()

                tr_m1, tr_m2 = _branch_masks(x_train, cfg.input_format, resolved_bm)
                val_m1, val_m2 = _branch_masks(x_val, cfg.input_format, resolved_bm)

                tr_correct = tr_preds == y_train
                val_correct = val_preds == y_val

                tr_odd_acc = _masked_acc(tr_correct, tr_m1)
                tr_even_acc = _masked_acc(tr_correct, tr_m2)
                val_odd_acc = _masked_acc(val_correct, val_m1)
                val_even_acc = _masked_acc(val_correct, val_m2)

                tr_cls_acc, tr_cls_n = _per_class_accs_counts(tr_preds, y_train, num_classes)
                val_cls_acc, val_cls_n = _per_class_accs_counts(val_preds, y_val, num_classes)

            result.log_epochs.append(epoch)
            result.train_accs.append(tr_acc)
            result.val_accs.append(val_acc)
            result.train_odd_accs.append(tr_odd_acc)
            result.train_even_accs.append(tr_even_acc)
            result.val_odd_accs.append(val_odd_acc)
            result.val_even_accs.append(val_even_acc)
            result.train_losses.append(tr_loss)
            result.val_losses.append(val_loss)

            row_tr = [None if a is None else round(float(a), 5) for a in tr_cls_acc]
            row_val = [None if a is None else round(float(a), 5) for a in val_cls_acc]
            result.per_class_train_accs.append(row_tr)
            result.per_class_val_accs.append(row_val)
            if not result.per_class_train_support:
                result.per_class_train_support = list(tr_cls_n)
                result.per_class_val_support = list(val_cls_n)

            if result.memo_epoch is None and tr_acc >= cfg.memo_train_threshold:
                result.memo_epoch = epoch
            if result.grok_epoch is None and val_acc >= cfg.grok_val_threshold:
                result.grok_epoch = epoch

            if cfg.verbose and epoch % (cfg.log_every * 10) == 0:
                print(
                    f"  Epoch {epoch:5d} | "
                    f"Train {tr_acc:.3f} | Val {val_acc:.3f} | "
                    f"Loss {tr_loss:.4f} / {val_loss:.4f}"
                )
                print(
                    f"      {resolved_bm}  tr {lb1}/{lb2} {_pct_or_dash(tr_odd_acc)}/"
                    f"{_pct_or_dash(tr_even_acc)}  val {_pct_or_dash(val_odd_acc)}/"
                    f"{_pct_or_dash(val_even_acc)}"
                )
                _max_slash_console = 10
                if num_classes <= _max_slash_console:
                    tr_sl = _slash_class_pcts(tr_cls_acc, tr_cls_n)
                    val_sl = _slash_class_pcts(val_cls_acc, val_cls_n)
                    print(
                        f"      cls 0..{num_classes - 1} (%)  "
                        f"tr {tr_sl}  |  val {val_sl}"
                    )
                else:
                    print(
                        f"      cls K={num_classes}  "
                        f"tr {_class_acc_summary(tr_cls_acc)}  |  "
                        f"val {_class_acc_summary(val_cls_acc)}  "
                        f"(full per-class accs in JSON)"
                    )

    if result.memo_epoch is not None and result.grok_epoch is not None:
        result.grok_gap = result.grok_epoch - result.memo_epoch

    result.elapsed_sec = time.time() - t0

    if cfg.verbose:
        s = result.summary()
        print(f"\nMemorization epoch : {s['memo_epoch']}")
        print(f"Grokking epoch     : {s['grok_epoch']}")
        print(f"Grokking gap       : {s['grok_gap']}")
        print(f"Elapsed            : {s['elapsed_sec']} s")

    return result


def _category_fname_suffix(cfg: TrainCategoryConfig) -> str:
    """Disambiguate runs: same stem as ``main._save_result`` + categorical marker."""
    if cfg.label_mode in ("c_mod", "a_plus_b_mod"):
        body = f"_cat_{cfg.label_mode}_lm{cfg.label_mod}"
    else:
        body = f"_cat_{cfg.label_mode}"
    if getattr(cfg, "rule_count", 1) > 1:
        body += f"_rc{cfg.rule_count}"
    return body


def save_category_result(result: TrainResult, results_dir: str) -> str:
    """
    Write JSON compatible with ``plots.plot_results`` / ``main._save_result`` schema,
    filename = standard run stem + ``_cat_<label_mode>`` [``_lm<k>``] before ``_ep``.
    Also writes ``per_class_train_accs`` / ``per_class_val_accs`` (rows aligned with
    ``log_epochs``) and support counts when present.
    """
    os.makedirs(results_dir, exist_ok=True)
    cfg = result.config
    assert isinstance(cfg, TrainCategoryConfig)

    mts = f"_n{cfg.max_train_samples}" if cfg.max_train_samples else ""
    fmt = f"_fmt{cfg.input_format}" if cfg.input_format != "a_op_b_eq" else ""
    fname = (
        f"{cfg.operation}_p{cfg.p}"
        f"_wd{cfg.weight_decay}"
        f"_lr{cfg.lr}"
        f"_d{cfg.d_model}"
        f"_l{cfg.num_layers}"
        f"_tf{cfg.train_frac}"
        f"{mts}{fmt}"
        f"{_category_fname_suffix(cfg)}"
        f"_ep{cfg.num_epochs}.json"
    )
    path = os.path.join(results_dir, fname)
    with open(path, "w", encoding="utf-8") as f:
        payload = {
            "task": "categorical",
            "summary": result.summary(),
            "branch_metric": result.branch_metric,
            "branch_label_1": result.branch_label_1,
            "branch_label_2": result.branch_label_2,
            "log_epochs": result.log_epochs,
            "train_accs": result.train_accs,
            "val_accs": result.val_accs,
            "train_odd_accs": result.train_odd_accs,
            "train_even_accs": result.train_even_accs,
            "val_odd_accs": result.val_odd_accs,
            "val_even_accs": result.val_even_accs,
            "train_losses": result.train_losses,
            "val_losses": result.val_losses,
        }
        if result.per_class_train_support:
            payload["per_class_train_support"] = result.per_class_train_support
            payload["per_class_val_support"] = result.per_class_val_support
        if result.per_class_train_accs:
            payload["per_class_train_accs"] = result.per_class_train_accs
            payload["per_class_val_accs"] = result.per_class_val_accs
        json.dump(payload, f, indent=2)
    return path


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train with categorical labels (same transformer, different head size).",
    )
    p.add_argument("--operation", default="add")
    p.add_argument("--p", type=int, default=97)
    p.add_argument("--train_frac", type=float, default=0.5)
    p.add_argument("--label_mode", default="c_parity",
                   choices=[
                       "c",
                       "c_parity",
                       "b_parity",
                       "a_parity",
                       "c_mod3",
                       "a_plus_b_mod3",
                       "c_mod",
                       "a_plus_b_mod",
                   ])
    p.add_argument(
        "--label_mod",
        type=int,
        default=0,
        help="Modulus k for c_mod / a_plus_b_mod (ignored for other modes). Try 3,5,7,11,...",
    )
    p.add_argument("--input_format", default="a_op_b_eq",
                   choices=[
                       "a_op_b_eq",
                       "a_b_eq",
                       "a_op_b_eq_rule",
                       "a_op_b_eq_bparity",
                       "a_op_bparity_eq",
                   ])
    p.add_argument("--data_seed", type=int, default=42)
    p.add_argument("--label_noise", type=float, default=0.0)
    p.add_argument("--num_epochs", type=int, default=5000)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1.0)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--branch_metric", default="auto",
                   choices=["auto", "b_parity", "a_ge_b", "a_gt_b"])
    p.add_argument(
        "--rule_count",
        type=int,
        default=1,
        help="Disjoint output bands (see data.dataset.make_dataset / OPERATION_RULE_INFO).",
    )
    p.add_argument(
        "--results_dir",
        default="results",
        help="Where to write JSON (same schema as main.py; filename includes _cat_<label_mode>).",
    )
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--sweep", nargs="+", default=None, metavar=("PARAM", "VAL"),
                   help="1D sweep: --sweep rule_count 1 2")
    p.add_argument("--grid1", nargs="+", default=None, metavar=("PARAM", "VAL"),
                   help="2D grid param 1: --grid1 weight_decay 0.0 1.0")
    p.add_argument("--grid2", nargs="+", default=None, metavar=("PARAM", "VAL"),
                   help="2D grid param 2: --grid2 rule_count 1 2")
    return p


def _cast(value: str, default):
    """Cast CLI sweep string values using existing field type."""
    if isinstance(default, bool):
        return value.lower() in ("true", "1", "yes")
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    if default is None:
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
    return value


def _run_one(cfg: TrainCategoryConfig, results_dir: str) -> TrainResult:
    train_ds, val_ds, vocab_size, num_classes = make_category_dataset(
        operation=cfg.operation,
        p=cfg.p,
        train_frac=cfg.train_frac,
        input_format=cfg.input_format,
        seed=cfg.data_seed,
        label_noise=cfg.label_noise,
        label_mode=cfg.label_mode,
        label_mod=cfg.label_mod,
        rule_count=cfg.rule_count,
    )
    print(
        f"train={len(train_ds):,}  val={len(val_ds):,}  vocab={vocab_size}  classes={num_classes}"
        + (f"  (label_mod={cfg.label_mod})" if cfg.label_mode in ("c_mod", "a_plus_b_mod") else "")
        + (f"  rule_count={cfg.rule_count}" if cfg.rule_count > 1 else "")
    )
    result = train_category(train_ds, val_ds, vocab_size, num_classes, cfg)
    out = save_category_result(result, results_dir)
    if cfg.verbose:
        print(f"  -> saved to {out}")
    return result


def _print_2d_summary(p1: str, p2: str, results: List[TrainResult]) -> None:
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


if __name__ == "__main__":
    parser = _build_argparser()
    args = parser.parse_args()
    base_cfg = TrainCategoryConfig(
        operation=args.operation,
        p=args.p,
        train_frac=args.train_frac,
        input_format=args.input_format,
        data_seed=args.data_seed,
        label_noise=args.label_noise,
        num_epochs=args.num_epochs,
        log_every=args.log_every,
        lr=args.lr,
        weight_decay=args.weight_decay,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        branch_metric=args.branch_metric,
        verbose=not args.quiet,
        label_mode=args.label_mode,
        label_mod=args.label_mod,
        rule_count=args.rule_count,
    )
    # Same precedence as main.py: 2D grid → 1D sweep → single run
    if args.grid1 is not None and args.grid2 is not None:
        p1, vals1 = args.grid1[0], args.grid1[1:]
        p2, vals2 = args.grid2[0], args.grid2[1:]
        if len(vals1) == 0 or len(vals2) == 0:
            parser.error("--grid1 and --grid2 each need a param name and at least one value.")

        for param in (p1, p2):
            if not hasattr(base_cfg, param):
                parser.error(f"Unknown TrainCategoryConfig field: '{param}'")

        total = len(vals1) * len(vals2)
        print(f"\n[GRID] {p1} {vals1}  x  {p2} {vals2}  ->  {total} runs")

        results_grid: List[TrainResult] = []
        for v1, v2 in itertools.product(vals1, vals2):
            cfg = TrainCategoryConfig(**vars(base_cfg))
            setattr(cfg, p1, _cast(v1, getattr(base_cfg, p1)))
            setattr(cfg, p2, _cast(v2, getattr(base_cfg, p2)))
            results_grid.append(_run_one(cfg, args.results_dir))

        _print_2d_summary(p1, p2, results_grid)

    elif args.sweep is not None:
        if len(args.sweep) < 2:
            parser.error("--sweep requires PARAM VAL [VAL ...]")
        param = args.sweep[0]
        values = args.sweep[1:]
        if not hasattr(base_cfg, param):
            parser.error(f"Unknown TrainCategoryConfig field: '{param}'")

        print(f"\n[SWEEP] {param} in {values}")
        results: List[TrainResult] = []
        for v in values:
            cfg = TrainCategoryConfig(**vars(base_cfg))
            setattr(cfg, param, _cast(v, getattr(base_cfg, param)))
            results.append(_run_one(cfg, args.results_dir))

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
    else:
        _run_one(base_cfg, args.results_dir)
