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

Use **c_parity** (2-way) or **c_mod** / **c_mod3** when the target should depend on the **true** ``c``.

Usage (from Grokking-Project)::

    python -m experiments.train_category --operation add_or_mul --label_mode c_parity

See ``data.dataset.make_category_dataset`` for all ``label_mode`` values.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

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
        print(f"Categorical task: label_mode={cfg.label_mode}{extra}  num_classes={num_classes}")

    torch.manual_seed(cfg.model_seed)

    _seq_len_map = {"a_op_b_eq": 4, "a_b_eq": 3, "a_op_b_eq_rule": 5}
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

            result.log_epochs.append(epoch)
            result.train_accs.append(tr_acc)
            result.val_accs.append(val_acc)
            result.train_odd_accs.append(tr_odd_acc)
            result.train_even_accs.append(tr_even_acc)
            result.val_odd_accs.append(val_odd_acc)
            result.val_even_accs.append(val_even_acc)
            result.train_losses.append(tr_loss)
            result.val_losses.append(val_loss)

            if result.memo_epoch is None and tr_acc >= cfg.memo_train_threshold:
                result.memo_epoch = epoch
            if result.grok_epoch is None and val_acc >= cfg.grok_val_threshold:
                result.grok_epoch = epoch

            if cfg.verbose and epoch % (cfg.log_every * 10) == 0:
                tr_s1 = f"{tr_odd_acc:.3f}" if tr_odd_acc is not None else "n/a"
                tr_s2 = f"{tr_even_acc:.3f}" if tr_even_acc is not None else "n/a"
                val_s1 = f"{val_odd_acc:.3f}" if val_odd_acc is not None else "n/a"
                val_s2 = f"{val_even_acc:.3f}" if val_even_acc is not None else "n/a"
                print(
                    f"  Epoch {epoch:5d} | "
                    f"Train {tr_acc:.3f} ({lb1} {tr_s1}, {lb2} {tr_s2}) | "
                    f"Val {val_acc:.3f} ({lb1} {val_s1}, {lb2} {val_s2}) | "
                    f"Loss {tr_loss:.4f} / {val_loss:.4f}"
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


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train with categorical labels (same transformer, different head size).",
    )
    p.add_argument("--operation", default="add")
    p.add_argument("--p", type=int, default=97)
    p.add_argument("--train_frac", type=float, default=0.5)
    p.add_argument("--label_mode", default="c_parity",
                   choices=[
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
        default=3,
        help="Modulus k for c_mod / a_plus_b_mod (ignored for other modes). Try 3,5,7,11,...",
    )
    p.add_argument("--input_format", default="a_op_b_eq",
                   choices=["a_op_b_eq", "a_b_eq", "a_op_b_eq_rule"])
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
    p.add_argument("--quiet", action="store_true")
    return p


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    cfg = TrainCategoryConfig(
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
    )
    train_ds, val_ds, vocab_size, num_classes = make_category_dataset(
        operation=cfg.operation,
        p=cfg.p,
        train_frac=cfg.train_frac,
        input_format=cfg.input_format,
        seed=cfg.data_seed,
        label_noise=cfg.label_noise,
        label_mode=cfg.label_mode,
        label_mod=cfg.label_mod,
    )
    print(
        f"train={len(train_ds):,}  val={len(val_ds):,}  vocab={vocab_size}  classes={num_classes}"
        + (f"  (label_mod={cfg.label_mod})" if cfg.label_mode in ("c_mod", "a_plus_b_mod") else "")
    )
    train_category(train_ds, val_ds, vocab_size, num_classes, cfg)
