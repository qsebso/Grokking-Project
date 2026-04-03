"""
Training loop and metric tracking for grokking experiments.
"""

import os
import random
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from data.dataset import resolve_branch_metric, branch_metric_labels

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is optional here
    np = None


MODEL_TYPE_CHOICES = ("standard", "convexified")


def _get_a_token_column(x: torch.Tensor, input_format: str) -> torch.Tensor:
    """Column index of the first operand `a` for each input format."""
    if input_format == "a_op_b_eq":
        return x[:, 0]
    if input_format == "a_b_eq":
        return x[:, 0]
    if input_format == "a_op_b_eq_rule":
        return x[:, 1]
    if input_format == "a_op_b_eq_bparity":
        return x[:, 0]
    if input_format == "a_op_bparity_eq":
        return x[:, 0]
    raise ValueError(f"Unknown input_format: {input_format}")


def _get_b_token_column(x: torch.Tensor, input_format: str) -> torch.Tensor:
    """
    Return the tensor column corresponding to the `b` token for each input format.
    """
    if input_format == "a_op_b_eq":
        return x[:, 2]
    if input_format == "a_b_eq":
        return x[:, 1]
    if input_format == "a_op_b_eq_rule":
        return x[:, 3]
    if input_format == "a_op_b_eq_bparity":
        return x[:, 2]
    if input_format == "a_op_bparity_eq":
        return x[:, 2]
    raise ValueError(f"Unknown input_format: {input_format}")


def _branch_masks(
    x: torch.Tensor, input_format: str, branch_metric: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (mask_first_branch, mask_second_branch) for the given split mode."""
    a = _get_a_token_column(x, input_format)
    b = _get_b_token_column(x, input_format)
    if branch_metric == "b_parity":
        return (b % 2 == 1), (b % 2 == 0)
    if branch_metric == "a_ge_b":
        return (a >= b), (a < b)
    if branch_metric == "a_gt_b":
        return (a > b), (a <= b)
    raise ValueError(
        f"Unknown branch_metric '{branch_metric}'. "
        "Expected: b_parity, a_ge_b, a_gt_b"
    )


def _masked_acc(correct: torch.Tensor, mask: torch.Tensor) -> Optional[float]:
    """Return masked mean accuracy, or None if mask is empty."""
    n = int(mask.sum().item())
    if n == 0:
        return None
    return correct[mask].float().mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    """All hyper-parameters needed to run one experiment."""

    # ── data ─────────────────────────────────────────────────────────────
    operation:          str            = "add"
    p:                  int            = 97
    train_frac:         float          = 0.5
    max_train_samples:  Optional[int]  = None   # None = no cap
    input_format:       str            = "a_op_b_eq"
    label_noise:        float          = 0.0   # asymmetric: fraction of train rows to corrupt
    label_noise_sym:    float          = 0.0   # symmetric: pair-swap noise on train
    data_seed:          int            = 42
    noise_mode:         str            = "random_wrong_c"  # see data.dataset.NOISE_MODE_CHOICES
    noise_fixed_target: int            = 5
    noise_fixed_backup: Optional[int]  = None

    # ── model ────────────────────────────────────────────────────────────
    d_model:        int   = 128
    nhead:          int   = 4
    num_layers:     int   = 2
    dim_feedforward: Optional[int] = None   # None → 4 * d_model
    model_type:     str   = "standard"

    # ── optimiser ────────────────────────────────────────────────────────
    lr:             float = 1e-3
    weight_decay:   float = 1.0
    betas:          tuple = (0.9, 0.98)

    # ── training schedule ────────────────────────────────────────────────
    num_epochs:     int   = 5_000
    batch_size:     Optional[int] = None    # None → full-batch
    log_every:      int   = 50

    # ── grokking detection thresholds ────────────────────────────────────
    grok_val_threshold:   float = 0.90
    memo_train_threshold: float = 0.99

    # ── misc ─────────────────────────────────────────────────────────────
    model_seed:     int   = 42
    device:         str   = "auto"          # "auto" | "cpu" | "cuda"
    verbose:        bool  = True
    # Per-sample accuracy split: "auto" infers from operation (see data.dataset)
    branch_metric:  str   = "auto"          # auto | b_parity | a_ge_b | a_gt_b
    # Disjoint multi-rule targets (see data.dataset.make_dataset); softmax width if set
    rule_count:     int   = 1
    num_logits:     Optional[int] = None    # None → use vocab_size in train()
    checkpoint_path: Optional[str] = None   # optional output checkpoint (.pt)


def noise_fname_suffix(cfg: Any) -> str:
    """Filename fragment for non-zero label noise, e.g. ``_nasy0p1_sym0p05``."""
    n = float(getattr(cfg, "label_noise", 0.0) or 0.0)
    s = float(getattr(cfg, "label_noise_sym", 0.0) or 0.0)
    nm = getattr(cfg, "noise_mode", "random_wrong_c") or "random_wrong_c"
    parts: List[str] = []
    if n > 0:
        parts.append(f"asy{str(n).replace('.', 'p')}")
        if nm != "random_wrong_c":
            parts.append("nm" + nm.replace("_", ""))
        nft = getattr(cfg, "noise_fixed_target", 5)
        nfb = getattr(cfg, "noise_fixed_backup", None)
        if nm in ("fixed_wrong_c", "fixed_wrong_c_cross_rule"):
            parts.append(f"ft{nft}")
            if nfb is not None:
                parts.append(f"fb{int(nfb)}")
    if s > 0:
        parts.append(f"sym{str(s).replace('.', 'p')}")
    return ("_n" + "_".join(parts)) if parts else ""


# ─────────────────────────────────────────────────────────────────────────────
# Results container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainResult:
    config: TrainConfig

    log_epochs:     List[int]   = field(default_factory=list)
    train_accs:     List[float] = field(default_factory=list)
    val_accs:       List[float] = field(default_factory=list)
    train_odd_accs: List[float] = field(default_factory=list)
    train_even_accs: List[float] = field(default_factory=list)
    val_odd_accs:   List[float] = field(default_factory=list)
    val_even_accs:  List[float] = field(default_factory=list)
    train_losses:   List[float] = field(default_factory=list)
    val_losses:     List[float] = field(default_factory=list)

    # Optional (e.g. train_category): per log_epochs row, acc for class k or None if no support
    per_class_train_accs:   List[List[Optional[float]]] = field(default_factory=list)
    per_class_val_accs:     List[List[Optional[float]]] = field(default_factory=list)
    per_class_train_support: List[int] = field(default_factory=list)  # counts; fixed for split
    per_class_val_support:   List[int] = field(default_factory=list)

    memo_epoch:     Optional[int] = None   # first epoch with train_acc ≥ threshold
    grok_epoch:     Optional[int] = None   # first epoch with val_acc   ≥ threshold
    grok_gap:       Optional[int] = None   # grok_epoch - memo_epoch (if both found)

    elapsed_sec:    float = 0.0
    # Resolved split used for train_odd_accs / train_even_accs (names kept for JSON compat)
    branch_metric:  str = "b_parity"
    branch_label_1: str = "odd"
    branch_label_2: str = "even"

    def summary(self) -> Dict[str, Any]:
        out = {
            "operation":         self.config.operation,
            "p":                 self.config.p,
            "train_frac":        self.config.train_frac,
            "max_train_samples": self.config.max_train_samples,
            "input_format":      self.config.input_format,
            "model_type":        getattr(self.config, "model_type", "standard"),
            "weight_decay":      self.config.weight_decay,
            "lr":                self.config.lr,
            "d_model":           self.config.d_model,
            "num_layers":        self.config.num_layers,
            "branch_metric":     self.branch_metric,
            "branch_label_1":    self.branch_label_1,
            "branch_label_2":    self.branch_label_2,
            "memo_epoch":        self.memo_epoch,
            "grok_epoch":        self.grok_epoch,
            "grok_gap":          self.grok_gap,
            "elapsed_sec":       round(self.elapsed_sec, 1),
            "num_epochs":        int(getattr(self.config, "num_epochs", 0)),
        }
        lm = getattr(self.config, "label_mode", None)
        nc = getattr(self.config, "num_classes", None)
        lmod = getattr(self.config, "label_mod", None)
        if lm is not None:
            out["label_mode"] = lm
        if nc is not None:
            out["num_classes"] = nc
        if lmod is not None:
            out["label_mod"] = lmod
        out["rule_count"] = getattr(self.config, "rule_count", 1)
        out["data_seed"] = int(getattr(self.config, "data_seed", 42))
        out["model_seed"] = int(getattr(self.config, "model_seed", 42))
        out["label_noise"] = float(getattr(self.config, "label_noise", 0.0))
        out["label_noise_sym"] = float(getattr(self.config, "label_noise_sym", 0.0))
        out["noise_mode"] = getattr(self.config, "noise_mode", "random_wrong_c")
        out["noise_fixed_target"] = int(getattr(self.config, "noise_fixed_target", 5))
        nfb = getattr(self.config, "noise_fixed_backup", None)
        if nfb is not None:
            out["noise_fixed_backup"] = int(nfb)
        return out


def seed_everything(seed: int) -> None:
    """Best-effort seeding without forcing deterministic kernels."""
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seq_len_from_input_format(input_format: str) -> int:
    seq_len_map = {
        "a_op_b_eq": 4,
        "a_b_eq": 3,
        "a_op_b_eq_rule": 5,
        "a_op_b_eq_bparity": 5,
        "a_op_bparity_eq": 4,
    }
    return seq_len_map.get(input_format, 4)


def build_sequence_model(
    *,
    vocab_size: int,
    cfg: TrainConfig,
    seq_len: int,
    num_logits: Optional[int] = None,
) -> Tuple[nn.Module, int]:
    """Build either the existing standard transformer or the convexified variant."""
    out_dim = num_logits if num_logits is not None else vocab_size
    if cfg.model_type == "standard":
        from models.transformer import TransformerModel, count_parameters

        model = TransformerModel(
            vocab_size=vocab_size,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            dim_feedforward=cfg.dim_feedforward,
            seq_len=seq_len,
            num_logits=out_dim,
        )
        return model, count_parameters(model)

    if cfg.model_type == "convexified":
        from models.transformer_convex import ConvexifiedTransformer, count_parameters

        model = ConvexifiedTransformer(
            vocab_size=vocab_size,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            dim_feedforward=cfg.dim_feedforward,
            seq_len=seq_len,
            num_logits=out_dim,
        )
        return model, count_parameters(model)

    raise ValueError(
        f"Unknown model_type '{cfg.model_type}'. Expected one of {MODEL_TYPE_CHOICES}."
    )


def build_factored_model(
    *,
    vocab_size: int,
    rule_count: int,
    p: int,
    cfg: TrainConfig,
    seq_len: int,
) -> Tuple[nn.Module, int]:
    """Build either the standard or convexified factored-head transformer."""
    if cfg.model_type == "standard":
        from models.transformer_factor import FactoredTransformer, count_parameters

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
        )
        return model, count_parameters(model)

    if cfg.model_type == "convexified":
        from models.transformer_convex import ConvexifiedFactoredTransformer, count_parameters

        model = ConvexifiedFactoredTransformer(
            vocab_size=vocab_size,
            rule_count=rule_count,
            p=p,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            seq_len=seq_len,
            dim_feedforward=cfg.dim_feedforward,
            dropout=0.0,
        )
        return model, count_parameters(model)

    raise ValueError(
        f"Unknown model_type '{cfg.model_type}'. Expected one of {MODEL_TYPE_CHOICES}."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(
    train_ds: TensorDataset,
    val_ds:   TensorDataset,
    vocab_size: int,
    cfg: TrainConfig,
) -> TrainResult:
    """
    Train a sequence model according to `cfg`.

    Parameters
    ----------
    train_ds    : TensorDataset  (x, y)
    val_ds      : TensorDataset  (x, y)
    vocab_size  : token vocabulary size
    cfg         : TrainConfig

    Returns
    -------
    TrainResult with logged metrics and grokking detection info.
    """
    # ── device ────────────────────────────────────────────────────────────
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    if cfg.verbose:
        print(f"Device: {device}")
        print(f"Model type: {cfg.model_type}")

    # ── reproducibility ───────────────────────────────────────────────────
    seed_everything(cfg.model_seed)

    # ── model ─────────────────────────────────────────────────────────────
    seq_len = seq_len_from_input_format(cfg.input_format)
    model, param_count = build_sequence_model(
        vocab_size=vocab_size,
        cfg=cfg,
        seq_len=seq_len,
        num_logits=cfg.num_logits,
    )
    model = model.to(device)

    if cfg.verbose:
        print(f"Parameters: {param_count:,}")

    # ── optimiser + loss ──────────────────────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=cfg.betas,
    )
    criterion = nn.CrossEntropyLoss()

    # ── data ──────────────────────────────────────────────────────────────
    x_train, y_train = train_ds.tensors[0].to(device), train_ds.tensors[1].to(device)
    x_val,   y_val   = val_ds.tensors[0].to(device),   val_ds.tensors[1].to(device)

    if cfg.batch_size is not None and cfg.batch_size < len(x_train):
        loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
        full_batch = False
    else:
        loader = None
        full_batch = True

    # ── result container ──────────────────────────────────────────────────
    resolved_bm = resolve_branch_metric(cfg.operation, cfg.branch_metric)
    lb1, lb2 = branch_metric_labels(resolved_bm)

    result = TrainResult(config=cfg)
    result.branch_metric = resolved_bm
    result.branch_label_1 = lb1
    result.branch_label_2 = lb2
    t0 = time.time()

    # ── training loop ─────────────────────────────────────────────────────
    for epoch in range(1, cfg.num_epochs + 1):

        model.train()

        if full_batch:
            optimizer.zero_grad()
            logits = model(x_train)
            loss   = criterion(logits, y_train)
            loss.backward()
            optimizer.step()
        else:
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss   = criterion(logits, yb)
                loss.backward()
                optimizer.step()

        # ── logging ───────────────────────────────────────────────────────
        if epoch % cfg.log_every == 0:
            model.eval()
            with torch.no_grad():
                tr_logits  = model(x_train)
                tr_preds   = tr_logits.argmax(-1)
                tr_loss    = criterion(tr_logits, y_train).item()
                tr_acc     = (tr_preds == y_train).float().mean().item()

                val_logits = model(x_val)
                val_preds  = val_logits.argmax(-1)
                val_loss   = criterion(val_logits, y_val).item()
                val_acc    = (val_preds == y_val).float().mean().item()

                # Branch accuracies (odd/even lists = first vs second split branch)
                tr_m1, tr_m2 = _branch_masks(x_train, cfg.input_format, resolved_bm)
                val_m1, val_m2 = _branch_masks(x_val, cfg.input_format, resolved_bm)

                tr_correct = (tr_preds == y_train)
                val_correct = (val_preds == y_val)

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

            # ── grokking detection ────────────────────────────────────────
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

    # ── grok gap ──────────────────────────────────────────────────────────
    if result.memo_epoch is not None and result.grok_epoch is not None:
        result.grok_gap = result.grok_epoch - result.memo_epoch

    result.elapsed_sec = time.time() - t0

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

    return result
