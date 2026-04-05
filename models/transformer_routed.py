"""
Routed factored transformer: shared trunk, rule head, and one c-head per rule.

The selected c logits come from the model's own predicted rule (hard routing) or
from a soft mixture over rule probabilities.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class RoutedModularTransformer(nn.Module):
    """
    Last-token hidden state is mapped to:
      - ``rule_logits``: ``(batch, rule_count)``
      - ``selected_c_logits``: ``(batch, p)``

    Debug routing details from the most recent forward pass are exposed via
    ``last_routing_info``.
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
        routing_mode: str = "hard",
        c_head_layers: int = 3,
        c_head_count: Optional[int] = None,
    ):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model
        if routing_mode not in ("hard", "soft"):
            raise ValueError("routing_mode must be 'hard' or 'soft'")
        if c_head_layers < 1:
            raise ValueError("c_head_layers must be >= 1")

        self.rule_count = int(rule_count)
        self.p = int(p)
        self.seq_len = int(seq_len)
        self.routing_mode = routing_mode
        self.c_head_layers = int(c_head_layers)
        self.c_head_count = int(c_head_count) if c_head_count is not None else self.rule_count
        if self.c_head_count < 1:
            raise ValueError("c_head_count must be >= 1")

        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.rule_head = nn.Linear(d_model, self.rule_count)
        self.head_router = nn.Linear(d_model, self.c_head_count)
        self.c_heads = nn.ModuleList(
            self._build_c_head(d_model=d_model, p=self.p, c_head_layers=self.c_head_layers)
            for _ in range(self.c_head_count)
        )
        self.last_routing_info: Dict[str, Any] = {}

    @staticmethod
    def _build_c_head(d_model: int, p: int, c_head_layers: int) -> nn.Module:
        if c_head_layers == 1:
            return nn.Linear(d_model, p)

        hidden_dims = [2 * d_model] + [d_model] * max(0, c_head_layers - 2)
        layers = []
        in_dim = d_model
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, p))
        return nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor, return_hidden: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        emb = self.embedding(x)
        out = self.transformer(emb)
        h = out[:, -1, :]

        rule_logits = self.rule_head(h)
        predicted_rule = rule_logits.argmax(dim=-1)
        all_c_logits = torch.stack([head(h) for head in self.c_heads], dim=1)

        if self.routing_mode == "hard":
            if self.c_head_count != self.rule_count:
                raise ValueError(
                    "hard routing requires c_head_count == rule_count; "
                    "use --routing_mode soft when c_head_count differs."
                )
            gather_idx = predicted_rule.view(-1, 1, 1).expand(-1, 1, self.p)
            selected_c_logits = all_c_logits.gather(1, gather_idx).squeeze(1)
            head_logits = None
            routing_weights = None
            chosen_head = predicted_rule
        else:
            # Preserve current soft-routing behavior when heads match rules.
            head_logits = rule_logits if self.c_head_count == self.rule_count else self.head_router(h)
            routing_weights = F.softmax(rule_logits, dim=-1)
            if self.c_head_count != self.rule_count:
                routing_weights = F.softmax(head_logits, dim=-1)
            selected_c_logits = (routing_weights.unsqueeze(-1) * all_c_logits).sum(dim=1)
            chosen_head = head_logits.argmax(dim=-1)

        self.last_routing_info = {
            "predicted_rule": predicted_rule.detach(),
            "chosen_head": chosen_head.detach(),
            "all_c_logits": all_c_logits.detach(),
            "routing_mode": self.routing_mode,
            "c_head_layers": self.c_head_layers,
            "c_head_count": self.c_head_count,
        }
        if head_logits is not None:
            self.last_routing_info["head_logits"] = head_logits.detach()
        if routing_weights is not None:
            self.last_routing_info["routing_weights"] = routing_weights.detach()

        if return_hidden:
            return rule_logits, selected_c_logits, out
        return rule_logits, selected_c_logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
