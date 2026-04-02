"""
PCA analysis of hidden states for trained grokking transformers.

Outputs:
  - pca_by_rule.png
  - pca_by_correctness.png
  - pca_by_split.png
  - pca_points.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import make_category_dataset, make_dataset, resolve_rule_id
from models.transformer import TransformerModel
from models.transformer_factor import FactoredTransformer


def _infer_num_layers(state_dict: Dict[str, torch.Tensor]) -> int:
    ids = set()
    for k in state_dict.keys():
        if k.startswith("transformer.layers."):
            parts = k.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                ids.add(int(parts[2]))
    return (max(ids) + 1) if ids else 2


def _get_from_cfg(cfg: Dict[str, Any], key: str, default: Any) -> Any:
    return cfg.get(key, cfg.get(f"{key}", default))


def _load_checkpoint(ckpt_path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            # Sometimes raw state dict is saved directly.
            looks_like_sd = all(isinstance(v, torch.Tensor) for v in ckpt.values())
            if not looks_like_sd:
                raise ValueError("Could not find state_dict in checkpoint.")
            state_dict = ckpt

        raw_cfg = ckpt.get("config", {})
        if hasattr(raw_cfg, "__dict__"):
            cfg = dict(vars(raw_cfg))
        elif isinstance(raw_cfg, dict):
            cfg = dict(raw_cfg)
        else:
            cfg = {}
        # Some runs store summary/config in adjacent keys.
        if not cfg and isinstance(ckpt.get("summary"), dict):
            cfg = dict(ckpt["summary"])
        return state_dict, cfg

    raise ValueError("Unsupported checkpoint format.")


def _extract_ab(x: torch.Tensor, input_format: str) -> Tuple[int, int]:
    if input_format == "a_op_b_eq":
        return int(x[0].item()), int(x[2].item())
    if input_format == "a_b_eq":
        return int(x[0].item()), int(x[1].item())
    if input_format == "a_op_b_eq_rule":
        return int(x[1].item()), int(x[3].item())
    if input_format == "a_op_b_eq_bparity":
        return int(x[0].item()), int(x[2].item())
    if input_format == "a_op_bparity_eq":
        # No full b token is present; use parity token as a fallback.
        return int(x[0].item()), int(x[2].item())
    raise ValueError(f"Unknown input_format: {input_format}")


def _pool_hidden(hidden: torch.Tensor, pool: str) -> torch.Tensor:
    if pool == "last_token":
        return hidden[:, -1, :]
    if pool == "mean":
        return hidden.mean(dim=1)
    raise ValueError(f"Unknown pool: {pool}")


def _select_layer(hidden: torch.Tensor, layer: str) -> torch.Tensor:
    # Current model returns final layer hidden states only.
    if layer == "last":
        return hidden
    if layer.isdigit() and int(layer) == 0:
        return hidden
    if layer.isdigit():
        raise ValueError(
            "This model currently exposes final hidden states only. "
            "Use --layer last (or 0 as alias)."
        )
    raise ValueError(f"Invalid --layer value: {layer}")


def _build_dataset_from_cfg(cfg: Dict[str, Any]) -> Tuple[TensorDataset, TensorDataset]:
    operation = _get_from_cfg(cfg, "operation", None)
    p = int(_get_from_cfg(cfg, "p", 97))
    train_frac = float(_get_from_cfg(cfg, "train_frac", 0.5))
    max_train_samples = _get_from_cfg(cfg, "max_train_samples", None)
    input_format = _get_from_cfg(cfg, "input_format", "a_op_b_eq")
    data_seed = int(_get_from_cfg(cfg, "data_seed", 42))
    label_noise = float(_get_from_cfg(cfg, "label_noise", 0.0))
    label_noise_sym = float(_get_from_cfg(cfg, "label_noise_sym", 0.0))
    rule_count = int(_get_from_cfg(cfg, "rule_count", 1))
    noise_mode = str(_get_from_cfg(cfg, "noise_mode", "random_wrong_c"))
    noise_fixed_target = int(_get_from_cfg(cfg, "noise_fixed_target", 5))
    noise_fixed_backup = _get_from_cfg(cfg, "noise_fixed_backup", None)

    if operation is None:
        raise ValueError(
            "Checkpoint config missing `operation`; cannot recreate dataset."
        )

    # Categorical / factor checkpoints use ``make_category_dataset``; token-prediction uses ``make_dataset``.
    label_mode = cfg.get("label_mode")
    factor_mode = cfg.get("factor_mode")
    if label_mode is not None or factor_mode is not None:
        lm = str(label_mode) if label_mode is not None else "c"
        label_mod = int(_get_from_cfg(cfg, "label_mod", 3))
        if label_mod == 0 and lm not in ("c_mod", "a_plus_b_mod"):
            label_mod = 3
        train_ds, val_ds, _, _ = make_category_dataset(
            operation=operation,
            p=p,
            train_frac=train_frac,
            max_train_samples=max_train_samples,
            input_format=input_format,
            seed=data_seed,
            label_noise=label_noise,
            label_noise_sym=label_noise_sym,
            label_mode=lm,
            label_mod=label_mod,
            rule_count=rule_count,
            noise_mode=noise_mode,
            noise_fixed_target=noise_fixed_target,
            noise_fixed_backup=noise_fixed_backup,
        )
        return train_ds, val_ds

    train_ds, val_ds, _, _ = make_dataset(
        operation=operation,
        p=p,
        train_frac=train_frac,
        max_train_samples=max_train_samples,
        input_format=input_format,
        seed=data_seed,
        label_noise=label_noise,
        label_noise_sym=label_noise_sym,
        rule_count=rule_count,
        noise_mode=noise_mode,
        noise_fixed_target=noise_fixed_target,
        noise_fixed_backup=noise_fixed_backup,
    )
    return train_ds, val_ds


def _scatter_plot(
    x: np.ndarray,
    y: np.ndarray,
    color_values: List[Any],
    title: str,
    out_path: str,
) -> None:
    plt.figure(figsize=(8, 6))
    unique_vals = sorted(set(color_values), key=lambda v: str(v))
    cmap = plt.get_cmap("tab20", max(1, len(unique_vals)))
    lut = {v: i for i, v in enumerate(unique_vals)}
    c = np.array([lut[v] for v in color_values], dtype=np.int64)
    plt.scatter(x, y, c=c, s=8, alpha=0.75, cmap=cmap)
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=cmap(i),
            markersize=6,
            label=str(v),
        )
        for v, i in lut.items()
    ]
    if len(handles) <= 20:
        plt.legend(handles=handles, title="Label", loc="best", fontsize=8)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="PCA of hidden states from trained checkpoints.")
    ap.add_argument("--checkpoint", required=True, help="Path to checkpoint (.pt/.pth).")
    ap.add_argument("--split", default="both", choices=["train", "val", "both"])
    ap.add_argument("--max_samples", type=int, default=2000)
    ap.add_argument("--layer", default="last", help="last or integer index (0 alias for last).")
    ap.add_argument("--pool", default="last_token", choices=["last_token", "mean"])
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return ap


def main() -> None:
    args = _build_parser().parse_args()
    ckpt_path = os.path.abspath(args.checkpoint)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"  cwd: {os.getcwd()}\n"
            "  Use a full path, or cd to the folder that contains the .pt file."
        )
    os.makedirs(args.output_dir, exist_ok=True)

    state_dict, cfg = _load_checkpoint(ckpt_path)
    train_ds, val_ds = _build_dataset_from_cfg(cfg)

    operation = str(_get_from_cfg(cfg, "operation", "add"))
    p = int(_get_from_cfg(cfg, "p", 97))
    input_format = str(_get_from_cfg(cfg, "input_format", "a_op_b_eq"))
    rule_count = int(_get_from_cfg(cfg, "rule_count", 1))

    vocab_size = int(state_dict["embedding.weight"].shape[0])
    d_model = int(state_dict["embedding.weight"].shape[1])
    num_layers = _infer_num_layers(state_dict)
    nhead = int(_get_from_cfg(cfg, "nhead", 4))
    dim_feedforward = _get_from_cfg(cfg, "dim_feedforward", None)
    if dim_feedforward is not None:
        dim_feedforward = int(dim_feedforward)

    is_factorized = ("head_rule.weight" in state_dict) and ("head_c.weight" in state_dict)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if is_factorized:
        inferred_rule_count = int(state_dict["head_rule.weight"].shape[0])
        inferred_p = int(state_dict["head_c.weight"].shape[0])
        model = FactoredTransformer(
            vocab_size=vocab_size,
            rule_count=inferred_rule_count,
            p=inferred_p,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
        )
        p = inferred_p
        rule_count = inferred_rule_count
    else:
        num_logits = int(state_dict["output_proj.weight"].shape[0])
        model = TransformerModel(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            num_logits=num_logits,
        )

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    split_datasets: List[Tuple[str, TensorDataset]] = []
    if args.split in ("train", "both"):
        split_datasets.append(("train", train_ds))
    if args.split in ("val", "both"):
        split_datasets.append(("val", val_ds))

    if args.max_samples <= 0:
        raise ValueError("--max_samples must be > 0")
    per_split_cap = int(math.ceil(args.max_samples / max(1, len(split_datasets))))

    hidden_rows: List[np.ndarray] = []
    split_rows: List[str] = []
    rule_rows: List[int] = []
    true_rows: List[int] = []
    pred_rows: List[int] = []
    correct_rows: List[bool] = []
    rule_correct_rows: List[bool] = []
    c_correct_rows: List[bool] = []

    for split_name, ds in split_datasets:
        x_all, y_all = ds.tensors
        n = min(len(x_all), per_split_cap)
        x = x_all[:n]
        y = y_all[:n]

        for i0 in range(0, n, args.batch_size):
            i1 = min(i0 + args.batch_size, n)
            xb = x[i0:i1].to(device)
            yb = y[i0:i1].to(device)
            with torch.no_grad():
                if is_factorized:
                    rule_logits, c_logits, hidden = model(xb, return_hidden=True)
                    pred_rule = rule_logits.argmax(dim=-1)
                    pred_c = c_logits.argmax(dim=-1)
                    true_rule_from_y = (yb // p).long() if rule_count > 1 else torch.zeros_like(yb)
                    true_c_from_y = (yb % p).long() if rule_count > 1 else yb.long()

                    # Prefer true rule-id from dataset metadata when possible.
                    true_rule_meta = []
                    for row in xb.cpu():
                        a, b = _extract_ab(row, input_format)
                        true_rule_meta.append(resolve_rule_id(operation, a, b, p))
                    true_rule = torch.tensor(true_rule_meta, dtype=torch.long, device=device)

                    rule_correct = pred_rule.eq(true_rule)
                    c_correct = pred_c.eq(true_c_from_y)
                    joint_correct = rule_correct & c_correct
                    global_pred = (pred_rule * p + pred_c).long()

                    pooled = _pool_hidden(_select_layer(hidden, args.layer), args.pool)
                    hidden_rows.extend(pooled.detach().cpu().numpy())
                    split_rows.extend([split_name] * (i1 - i0))
                    rule_rows.extend(true_rule.detach().cpu().tolist())
                    true_rows.extend(yb.detach().cpu().tolist())
                    pred_rows.extend(global_pred.detach().cpu().tolist())
                    correct_rows.extend(joint_correct.detach().cpu().tolist())
                    rule_correct_rows.extend(rule_correct.detach().cpu().tolist())
                    c_correct_rows.extend(c_correct.detach().cpu().tolist())
                else:
                    logits, hidden = model(xb, return_hidden=True)
                    pred = logits.argmax(dim=-1)
                    correct = pred.eq(yb)
                    pooled = _pool_hidden(_select_layer(hidden, args.layer), args.pool)

                    batch_rules = []
                    for row in xb.cpu():
                        a, b = _extract_ab(row, input_format)
                        batch_rules.append(resolve_rule_id(operation, a, b, p))

                    hidden_rows.extend(pooled.detach().cpu().numpy())
                    split_rows.extend([split_name] * (i1 - i0))
                    rule_rows.extend(batch_rules)
                    true_rows.extend(yb.detach().cpu().tolist())
                    pred_rows.extend(pred.detach().cpu().tolist())
                    correct_rows.extend(correct.detach().cpu().tolist())
                    rule_correct_rows.extend(correct.detach().cpu().tolist())
                    c_correct_rows.extend(correct.detach().cpu().tolist())

    hidden_mat = np.asarray(hidden_rows, dtype=np.float32)
    pca = PCA(n_components=2)
    pca_points = pca.fit_transform(hidden_mat)
    evr = pca.explained_variance_ratio_
    print(f"Explained variance ratio: PC1={evr[0]:.4f}, PC2={evr[1]:.4f}")

    p1 = pca_points[:, 0]
    p2 = pca_points[:, 1]

    _scatter_plot(
        p1,
        p2,
        color_values=rule_rows,
        title="PCA hidden states colored by rule id",
        out_path=os.path.join(args.output_dir, "pca_by_rule.png"),
    )
    _scatter_plot(
        p1,
        p2,
        color_values=["correct" if c else "incorrect" for c in correct_rows],
        title="PCA hidden states colored by correctness",
        out_path=os.path.join(args.output_dir, "pca_by_correctness.png"),
    )
    _scatter_plot(
        p1,
        p2,
        color_values=split_rows,
        title="PCA hidden states colored by split",
        out_path=os.path.join(args.output_dir, "pca_by_split.png"),
    )

    csv_path = os.path.join(args.output_dir, "pca_points.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "pca1",
                "pca2",
                "rule_id",
                "correct",
                "split",
                "true_label",
                "pred_label",
                "rule_correct",
                "c_correct",
            ]
        )
        for i in range(len(p1)):
            writer.writerow(
                [
                    float(p1[i]),
                    float(p2[i]),
                    int(rule_rows[i]),
                    bool(correct_rows[i]),
                    split_rows[i],
                    int(true_rows[i]),
                    int(pred_rows[i]),
                    bool(rule_correct_rows[i]),
                    bool(c_correct_rows[i]),
                ]
            )

    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
