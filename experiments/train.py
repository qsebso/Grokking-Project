"""
Training loop and metric tracking for grokking experiments.
"""

import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


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
    label_noise:        float          = 0.0
    data_seed:          int            = 42

    # ── model ────────────────────────────────────────────────────────────
    d_model:        int   = 128
    nhead:          int   = 4
    num_layers:     int   = 2
    dim_feedforward: Optional[int] = None   # None → 4 * d_model

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


# ─────────────────────────────────────────────────────────────────────────────
# Results container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainResult:
    config: TrainConfig

    log_epochs:     List[int]   = field(default_factory=list)
    train_accs:     List[float] = field(default_factory=list)
    val_accs:       List[float] = field(default_factory=list)
    train_losses:   List[float] = field(default_factory=list)
    val_losses:     List[float] = field(default_factory=list)

    memo_epoch:     Optional[int] = None   # first epoch with train_acc ≥ threshold
    grok_epoch:     Optional[int] = None   # first epoch with val_acc   ≥ threshold
    grok_gap:       Optional[int] = None   # grok_epoch - memo_epoch (if both found)

    elapsed_sec:    float = 0.0

    def summary(self) -> Dict[str, Any]:
        return {
            "operation":         self.config.operation,
            "p":                 self.config.p,
            "train_frac":        self.config.train_frac,
            "max_train_samples": self.config.max_train_samples,
            "input_format":      self.config.input_format,
            "weight_decay":      self.config.weight_decay,
            "lr":                self.config.lr,
            "d_model":           self.config.d_model,
            "num_layers":        self.config.num_layers,
            "memo_epoch":        self.memo_epoch,
            "grok_epoch":        self.grok_epoch,
            "grok_gap":          self.grok_gap,
            "elapsed_sec":       round(self.elapsed_sec, 1),
        }


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
    Train a TransformerModel according to `cfg`.

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
    # ── imports here to avoid circular deps ───────────────────────────────
    from models.transformer import TransformerModel

    # ── device ────────────────────────────────────────────────────────────
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    if cfg.verbose:
        print(f"Device: {device}")

    # ── reproducibility ───────────────────────────────────────────────────
    torch.manual_seed(cfg.model_seed)

    # ── model ─────────────────────────────────────────────────────────────
    # seq_len depends on input_format:
    #   "a_op_b_eq"       → 4 tokens
    #   "a_b_eq"          → 3 tokens
    #   "a_op_b_eq_rule"  → 5 tokens
    _seq_len_map = {"a_op_b_eq": 4, "a_b_eq": 3, "a_op_b_eq_rule": 5}
    seq_len = _seq_len_map.get(cfg.input_format, 4)

    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dim_feedforward=cfg.dim_feedforward,
        seq_len=seq_len,
    ).to(device)

    if cfg.verbose:
        from models.transformer import count_parameters
        print(f"Parameters: {count_parameters(model):,}")

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
    result = TrainResult(config=cfg)
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
                tr_loss    = criterion(tr_logits, y_train).item()
                tr_acc     = (tr_logits.argmax(-1) == y_train).float().mean().item()

                val_logits = model(x_val)
                val_loss   = criterion(val_logits, y_val).item()
                val_acc    = (val_logits.argmax(-1) == y_val).float().mean().item()

            result.log_epochs.append(epoch)
            result.train_accs.append(tr_acc)
            result.val_accs.append(val_acc)
            result.train_losses.append(tr_loss)
            result.val_losses.append(val_loss)

            # ── grokking detection ────────────────────────────────────────
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

    return result
