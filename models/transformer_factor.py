"""
Factored decoder-only transformer: shared trunk, two linear heads (rule id + local output).

Kept separate from ``transformer.py`` for ablation experiments (joint vs single softmax).
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


class FactoredTransformer(nn.Module):
    """
    Last-token hidden state is mapped to:
      - ``rule_logits``: ``(batch, rule_count)``
      - ``c_logits``: ``(batch, p)`` for local output in ``0 .. p-1``
    """

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
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        self.rule_count = int(rule_count)
        self.p = int(p)

        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head_rule = nn.Linear(d_model, self.rule_count)
        self.head_c = nn.Linear(d_model, self.p)

    def forward(
        self, x: torch.Tensor, return_hidden: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        emb = self.embedding(x)
        out = self.transformer(emb)
        h = out[:, -1, :]
        rule_logits = self.head_rule(h)
        c_logits = self.head_c(h)
        if return_hidden:
            return rule_logits, c_logits, out
        return rule_logits, c_logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
