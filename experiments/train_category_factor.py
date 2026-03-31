#!/usr/bin/env python3
"""
Factored-output experiments (separate from ``train_category.py``).

Uses the **same** ``label_mode=c`` disjoint labels ``y in {0..rule_count*p - 1}``, but:

- **joint**: two heads — ``rule_id`` (``rule_count`` classes) + ``c_local`` (``p`` classes);
  loss = CE_rule + CE_c; metrics include rule / c / **both correct**.
- **rule_only**: single head, target = ``y // p`` (which rule band).
- **c_only**: single head, target = ``y % p`` (local output mod p).

Default results directory is ``results_factor`` so runs stay separate from ``results`` / ``!previous_results``.

Example::

    python -m experiments.train_category_factor --factor_mode joint --operation add_or_mul --rule_count 2
    python -m experiments.train_category_factor --factor_mode rule_only --operation add_or_mul --rule_count 2
    python -m experiments.train_category_factor --factor_mode c_only --operation add_or_mul --rule_count 2
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import make_category_dataset, resolve_branch_metric, branch_metric_labels
from experiments.train import TrainConfig, TrainResult, _branch_masks, _masked_acc
from models.transformer import TransformerModel, count_parameters as count_params_single
from models.transformer_factor import FactoredTransformer, count_parameters as count_params_factored


@dataclass
class TrainFactorConfig(TrainConfig):
    """``TrainConfig`` + factor experiment mode."""

    factor_mode: str = "joint"  # joint | rule_only | c_only


def _save_factor_result(
    result: TrainResult,
    extra: Dict[str, Any],
    results_dir: str,
    cfg: TrainFactorConfig,
) -> str:
    os.makedirs(results_dir, exist_ok=True)
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
        f"_factor_{cfg.factor_mode}"
        f"_rc{cfg.rule_count}"
        f"_ep{cfg.num_epochs}.json"
    )
    path = os.path.join(results_dir, fname)
    summ = result.summary()
    summ["factor_mode"] = cfg.factor_mode
    summ["label_mode"] = "c"
    payload: Dict[str, Any] = {
        "task": "categorical_factor",
        "summary": summ,
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
    payload.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def train_factor_run(
    train_ds: TensorDataset,
    val_ds: TensorDataset,
    vocab_size: int,
    p: int,
    rule_count: int,
    cfg: TrainFactorConfig,
) -> Tuple[TrainResult, Dict[str, Any]]:
    """Train one run; ``extra`` has factor-specific series for JSON."""
    from experiments.train import TrainResult

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else torch.device(cfg.device)
    torch.manual_seed(cfg.model_seed)

    _seq_len_map = {
        "a_op_b_eq": 4,
        "a_b_eq": 3,
        "a_op_b_eq_rule": 5,
        "a_op_b_eq_bparity": 5,
        "a_op_bparity_eq": 4,
    }
    seq_len = _seq_len_map.get(cfg.input_format, 4)

    x_tr, y_tr = train_ds.tensors[0].to(device), train_ds.tensors[1].to(device)
    x_va, y_va = val_ds.tensors[0].to(device), val_ds.tensors[1].to(device)

    y_rule_tr = (y_tr // p).long()
    y_c_tr = (y_tr % p).long()
    y_rule_va = (y_va // p).long()
    y_c_va = (y_va % p).long()

    mode = cfg.factor_mode
    ce = nn.CrossEntropyLoss()

    extra: Dict[str, Any] = {
        "factor_mode": mode,
        "p": p,
        "rule_count": rule_count,
    }

    if mode == "joint":
        model = FactoredTransformer(
            vocab_size=vocab_size,
            rule_count=rule_count,
            p=p,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            seq_len=seq_len,
            dim_feedforward=cfg.dim_feedforward,
            dropout=0.0,
        ).to(device)
        if cfg.verbose:
            print(f"FactoredTransformer  params={count_params_factored(model):,}")
        opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas)
        train_target_desc = "joint (rule + c heads)"
        # Extra metric series
        tr_r: List[float] = []
        va_r: List[float] = []
        tr_c: List[float] = []
        va_c: List[float] = []
        tr_j: List[float] = []
        va_j: List[float] = []
    elif mode == "rule_only":
        model = TransformerModel(
            vocab_size=vocab_size,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            seq_len=seq_len,
            dim_feedforward=cfg.dim_feedforward,
            num_logits=rule_count,
        ).to(device)
        if cfg.verbose:
            print(f"TransformerModel rule head  params={count_params_single(model):,}")
        opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas)
        y_tr_t = y_rule_tr
        y_va_t = y_rule_va
        train_target_desc = "rule_id only"
    else:  # c_only
        model = TransformerModel(
            vocab_size=vocab_size,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            seq_len=seq_len,
            dim_feedforward=cfg.dim_feedforward,
            num_logits=p,
        ).to(device)
        if cfg.verbose:
            print(f"TransformerModel c_local head  params={count_params_single(model):,}")
        opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas)
        y_tr_t = y_c_tr
        y_va_t = y_c_va
        train_target_desc = "c_local only"

    if cfg.verbose:
        print(f"Factor mode: {mode}  ({train_target_desc})  device={device}")

    loader = None
    full_batch = True
    if cfg.batch_size is not None and cfg.batch_size < len(x_tr):
        loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
        full_batch = False

    resolved_bm = resolve_branch_metric(cfg.operation, cfg.branch_metric)
    lb1, lb2 = branch_metric_labels(resolved_bm)

    result = TrainResult(config=cfg)
    result.branch_metric = resolved_bm
    result.branch_label_1 = lb1
    result.branch_label_2 = lb2
    t0 = time.time()

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        if mode == "joint":
            if full_batch:
                opt.zero_grad()
                lr_logits, lc_logits = model(x_tr)
                loss = ce(lr_logits, y_rule_tr) + ce(lc_logits, y_c_tr)
                loss.backward()
                opt.step()
            else:
                for xb, yb in loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    yr = (yb // p).long()
                    yc = (yb % p).long()
                    opt.zero_grad()
                    lr_logits, lc_logits = model(xb)
                    loss = ce(lr_logits, yr) + ce(lc_logits, yc)
                    loss.backward()
                    opt.step()
        else:
            if full_batch:
                opt.zero_grad()
                logits = model(x_tr)
                loss = ce(logits, y_tr_t)
                loss.backward()
                opt.step()
            else:
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    if mode == "rule_only":
                        yt = (yb // p).long()
                    else:
                        yt = (yb % p).long()
                    opt.zero_grad()
                    logits = model(xb)
                    loss = ce(logits, yt)
                    loss.backward()
                    opt.step()

        if epoch % cfg.log_every == 0:
            model.eval()
            with torch.no_grad():
                if mode == "joint":
                    lr_tr, lc_tr = model(x_tr)
                    lr_va, lc_va = model(x_va)
                    pr_r = lr_tr.argmax(-1)
                    pr_c = lc_tr.argmax(-1)
                    pv_r = lr_va.argmax(-1)
                    pv_c = lc_va.argmax(-1)
                    acc_rule_tr = (pr_r == y_rule_tr).float().mean().item()
                    acc_rule_va = (pv_r == y_rule_va).float().mean().item()
                    acc_c_tr = (pr_c == y_c_tr).float().mean().item()
                    acc_c_va = (pv_c == y_c_va).float().mean().item()
                    joint_tr = ((pr_r == y_rule_tr) & (pr_c == y_c_tr)).float().mean().item()
                    joint_va = ((pv_r == y_rule_va) & (pv_c == y_c_va)).float().mean().item()
                    loss_tr = (ce(lr_tr, y_rule_tr) + ce(lc_tr, y_c_tr)).item()
                    loss_va = (ce(lr_va, y_rule_va) + ce(lc_va, y_c_va)).item()

                    tr_r.append(acc_rule_tr)
                    va_r.append(acc_rule_va)
                    tr_c.append(acc_c_tr)
                    va_c.append(acc_c_va)
                    tr_j.append(joint_tr)
                    va_j.append(joint_va)

                    tr_acc = joint_tr
                    val_acc = joint_va
                    tr_odd_acc = _masked_acc(pr_r == y_rule_tr, _branch_masks(x_tr, cfg.input_format, resolved_bm)[0])
                    tr_even_acc = _masked_acc(pr_r == y_rule_tr, _branch_masks(x_tr, cfg.input_format, resolved_bm)[1])
                    val_odd_acc = _masked_acc(pv_r == y_rule_va, _branch_masks(x_va, cfg.input_format, resolved_bm)[0])
                    val_even_acc = _masked_acc(pv_r == y_rule_va, _branch_masks(x_va, cfg.input_format, resolved_bm)[1])
                else:
                    logits_tr = model(x_tr)
                    logits_va = model(x_va)
                    pred_tr = logits_tr.argmax(-1)
                    pred_va = logits_va.argmax(-1)
                    tr_acc = (pred_tr == y_tr_t).float().mean().item()
                    val_acc = (pred_va == y_va_t).float().mean().item()
                    loss_tr = ce(logits_tr, y_tr_t).item()
                    loss_va = ce(logits_va, y_va_t).item()
                    tr_correct = pred_tr == y_tr_t
                    val_correct = pred_va == y_va_t
                    tr_m1, tr_m2 = _branch_masks(x_tr, cfg.input_format, resolved_bm)
                    va_m1, va_m2 = _branch_masks(x_va, cfg.input_format, resolved_bm)
                    tr_odd_acc = _masked_acc(tr_correct, tr_m1)
                    tr_even_acc = _masked_acc(tr_correct, tr_m2)
                    val_odd_acc = _masked_acc(val_correct, va_m1)
                    val_even_acc = _masked_acc(val_correct, va_m2)

            result.log_epochs.append(epoch)
            result.train_accs.append(tr_acc)
            result.val_accs.append(val_acc)
            result.train_odd_accs.append(tr_odd_acc)
            result.train_even_accs.append(tr_even_acc)
            result.val_odd_accs.append(val_odd_acc)
            result.val_even_accs.append(val_even_acc)
            result.train_losses.append(loss_tr)
            result.val_losses.append(loss_va)

            if result.memo_epoch is None and tr_acc >= cfg.memo_train_threshold:
                result.memo_epoch = epoch
            if result.grok_epoch is None and val_acc >= cfg.grok_val_threshold:
                result.grok_epoch = epoch

            if cfg.verbose and epoch % (cfg.log_every * 10) == 0:
                if mode == "joint":
                    print(
                        f"  Epoch {epoch:5d} | joint tr/val {tr_acc:.3f}/{val_acc:.3f} | "
                        f"rule {acc_rule_tr:.3f}/{acc_rule_va:.3f} | c {acc_c_tr:.3f}/{acc_c_va:.3f} | "
                        f"loss {loss_tr:.4f}/{loss_va:.4f}"
                    )
                else:
                    print(
                        f"  Epoch {epoch:5d} | Train {tr_acc:.3f} | Val {val_acc:.3f} | "
                        f"Loss {loss_tr:.4f} / {loss_va:.4f}"
                    )

    if result.memo_epoch is not None and result.grok_epoch is not None:
        result.grok_gap = result.grok_epoch - result.memo_epoch
    result.elapsed_sec = time.time() - t0

    if mode == "joint":
        extra["train_acc_rule"] = tr_r
        extra["val_acc_rule"] = va_r
        extra["train_acc_c_local"] = tr_c
        extra["val_acc_c_local"] = va_c
        extra["train_acc_joint"] = tr_j
        extra["val_acc_joint"] = va_j
        extra["note"] = (
            "train_accs/val_accs are JOINT (both rule and c correct). "
            "See train_acc_rule, val_acc_rule, train_acc_c_local, val_acc_c_local for heads."
        )
    else:
        extra["note"] = f"Single head: {mode}; train_accs/val_accs match that target only."

    if cfg.verbose:
        s = result.summary()
        print(f"\nMemorization epoch : {s['memo_epoch']}")
        print(f"Grokking epoch     : {s['grok_epoch']}")
        print(f"Grokking gap       : {s['grok_gap']}")
        print(f"Elapsed            : {s['elapsed_sec']} s")

    if cfg.checkpoint_path:
        ckpt_dir = os.path.dirname(cfg.checkpoint_path)
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": vars(cfg),
            },
            cfg.checkpoint_path,
        )
        if cfg.verbose:
            print(f"Checkpoint saved   : {cfg.checkpoint_path}")

    return result, extra


def _run_one(
    cfg: TrainFactorConfig,
    results_dir: str,
    *,
    auto_pca: bool = False,
    pca_split: str = "both",
    pca_max_samples: int = 2000,
    pca_layer: str = "last",
    pca_pool: str = "last_token",
    pca_output_dir: Optional[str] = None,
) -> None:
    if cfg.rule_count < 2:
        raise ValueError("factor experiments need rule_count >= 2 (disjoint bands). Use --rule_count 2.")

    train_ds, val_ds, vocab_size, num_classes = make_category_dataset(
        operation=cfg.operation,
        p=cfg.p,
        train_frac=cfg.train_frac,
        input_format=cfg.input_format,
        seed=cfg.data_seed,
        label_noise=cfg.label_noise,
        label_mode="c",
        label_mod=3,
        rule_count=cfg.rule_count,
    )
    expected = cfg.rule_count * cfg.p
    if num_classes != expected:
        raise RuntimeError(f"Expected num_classes {expected}, got {num_classes}")

    print(
        f"train={len(train_ds):,}  val={len(val_ds):,}  vocab={vocab_size}  "
        f"K={num_classes}  factor_mode={cfg.factor_mode}"
    )

    result, extra = train_factor_run(train_ds, val_ds, vocab_size, cfg.p, cfg.rule_count, cfg)
    path = _save_factor_result(result, extra, results_dir, cfg)
    if cfg.verbose:
        print(f"  -> saved to {path}")

    if auto_pca:
        if not cfg.checkpoint_path:
            raise ValueError(
                "--auto_pca requires --save_checkpoint so analysis can load weights."
            )
        pca_out = pca_output_dir
        if not pca_out:
            ckpt_stem = os.path.splitext(os.path.basename(cfg.checkpoint_path))[0]
            pca_out = os.path.join(os.path.dirname(cfg.checkpoint_path), f"{ckpt_stem}_pca")
        cmd = [
            sys.executable,
            os.path.join("analysis", "pca_hidden_states.py"),
            "--checkpoint",
            cfg.checkpoint_path,
            "--split",
            pca_split,
            "--max_samples",
            str(pca_max_samples),
            "--layer",
            pca_layer,
            "--pool",
            pca_pool,
            "--output_dir",
            pca_out,
        ]
        if cfg.verbose:
            print("Running PCA analysis...")
            print("  " + " ".join(cmd))
        subprocess.run(cmd, check=True)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Factored heads (rule + c_local) vs single-head baselines; separate from train_category.",
    )
    p.add_argument(
        "--factor_mode",
        default="joint",
        choices=["joint", "rule_only", "c_only"],
        help="joint = two heads; rule_only = predict y//p; c_only = predict y%%p",
    )
    p.add_argument("--operation", default="add_or_mul")
    p.add_argument("--p", type=int, default=97)
    p.add_argument("--train_frac", type=float, default=0.5)
    p.add_argument(
        "--input_format",
        default="a_op_b_eq",
        choices=[
            "a_op_b_eq",
            "a_b_eq",
            "a_op_b_eq_rule",
            "a_op_b_eq_bparity",
            "a_op_bparity_eq",
        ],
    )
    p.add_argument("--data_seed", type=int, default=42)
    p.add_argument("--label_noise", type=float, default=0.0)
    p.add_argument("--num_epochs", type=int, default=5000)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1.0)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--branch_metric", default="auto", choices=["auto", "b_parity", "a_ge_b", "a_gt_b"])
    p.add_argument("--rule_count", type=int, default=2)
    p.add_argument(
        "--results_dir",
        default="results_factor",
        help="Separate from default results/ (e.g. results_factor or !previous_results_factor).",
    )
    p.add_argument("--save_checkpoint", default=None, help="Optional path to save model checkpoint (.pt).")
    p.add_argument("--auto_pca", action="store_true", help="Run hidden-state PCA automatically after training.")
    p.add_argument("--pca_split", default="both", choices=["train", "val", "both"])
    p.add_argument("--pca_max_samples", type=int, default=2000)
    p.add_argument("--pca_layer", default="last")
    p.add_argument("--pca_pool", default="last_token", choices=["last_token", "mean"])
    p.add_argument("--pca_output_dir", default=None, help="Optional PCA output dir. Default: next to checkpoint.")
    p.add_argument("--quiet", action="store_true")
    return p


if __name__ == "__main__":
    parser = _build_argparser()
    args = parser.parse_args()
    cfg = TrainFactorConfig(
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
        rule_count=args.rule_count,
        factor_mode=args.factor_mode,
        checkpoint_path=args.save_checkpoint,
    )
    proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = args.results_dir
    if not os.path.isabs(results_dir):
        results_dir = os.path.join(proj, results_dir)
    _run_one(
        cfg,
        results_dir,
        auto_pca=args.auto_pca,
        pca_split=args.pca_split,
        pca_max_samples=args.pca_max_samples,
        pca_layer=args.pca_layer,
        pca_pool=args.pca_pool,
        pca_output_dir=args.pca_output_dir,
    )
