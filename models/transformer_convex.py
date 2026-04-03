"""
Convexified transformer variants for grokking experiments.

These models preserve the existing repository interface:
  - token embeddings are unchanged
  - there is still no positional encoding
  - the final token hidden state is projected to logits

The only architectural change is the attention block. The standard models use
PyTorch's softmax attention; the convexified models use positive kernel
attention with ``phi(x) = ELU(x) + 1`` and row-wise normalization.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvexLinearAttention(nn.Module):
    """Multi-head positive-kernel attention replacing softmax attention."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.eps = eps

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.nhead, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        # Positive features keep attention weights non-negative before normalization.
        q_feat = F.elu(q) + 1.0
        k_feat = F.elu(k) + 1.0

        scores = torch.einsum("bhid,bhjd->bhij", q_feat, k_feat)
        denom = scores.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        weights = self.dropout(scores / denom)

        attended = torch.einsum("bhij,bhjd->bhid", weights, v)
        return self.out_proj(self._merge_heads(attended))


class ConvexEncoderLayer(nn.Module):
    """Transformer encoder layer with convexified attention and standard FFN."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.self_attn = ConvexLinearAttention(d_model=d_model, nhead=nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.self_attn(x)
        x = self.norm1(x + self.dropout1(attn_out))

        ff = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = self.norm2(x + self.dropout2(ff))
        return x


class ConvexifiedTransformer(nn.Module):
    """
    Standard repo-compatible transformer with convexified attention blocks.

    Compatibility assumptions:
      - input tokenization is identical to ``TransformerModel``
      - no positional encoding is added because the baseline also omits it
      - the last token hidden state is still the classifier input
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        seq_len: int = 4,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.0,
        num_logits: Optional[int] = None,
    ):
        super().__init__()
        del seq_len  # kept for interface parity with the standard model
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                ConvexEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        out_dim = num_logits if num_logits is not None else vocab_size
        self.output_proj = nn.Linear(d_model, out_dim)

    def forward(
        self, x: torch.Tensor, return_hidden: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        out = self.embedding(x)
        for layer in self.layers:
            out = layer(out)
        logits = self.output_proj(out[:, -1, :])
        if return_hidden:
            return logits, out
        return logits


class ConvexifiedFactoredTransformer(nn.Module):
    """Convexified trunk with separate heads for rule id and local output."""

    def __init__(
        self,
        vocab_size: int,
        rule_count: int,
        p: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        seq_len: int = 4,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        del seq_len  # kept for interface parity with the standard model
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        self.rule_count = int(rule_count)
        self.p = int(p)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                ConvexEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.head_rule = nn.Linear(d_model, self.rule_count)
        self.head_c = nn.Linear(d_model, self.p)

    def forward(
        self, x: torch.Tensor, return_hidden: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        out = self.embedding(x)
        for layer in self.layers:
            out = layer(out)
        h = out[:, -1, :]
        rule_logits = self.head_rule(h)
        c_logits = self.head_c(h)
        if return_hidden:
            return rule_logits, c_logits, out
        return rule_logits, c_logits


def count_parameters(model: nn.Module) -> int:
    """Return the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
