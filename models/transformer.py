"""
Transformer model for grokking experiments.

Matches the architecture from Power et al. (2022):
  - 2-layer decoder-only transformer
  - d_model = 128, 4 attention heads
  - No dropout
  - Final token's representation projected to vocab logits
"""

from typing import Optional

import torch
import torch.nn as nn


class TransformerModel(nn.Module):
    """
    Small decoder-only transformer for grokking experiments.

    Parameters
    ----------
    vocab_size  : size of the token vocabulary
    d_model     : embedding / hidden dimension  (default 128)
    nhead       : number of attention heads     (default 4)
    num_layers  : number of encoder layers      (default 2)
    seq_len     : input sequence length         (default 4, i.e. [a, op, b, =])
    dim_feedforward : FFN inner dimension       (default 4 * d_model)
    dropout     : dropout probability           (default 0.0)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        seq_len: int = 4,
        dim_feedforward: int = None,
        dropout: float = 0.0,
        num_logits: Optional[int] = None,
    ):
        super().__init__()

        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        self.embedding = nn.Embedding(vocab_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        out_dim = num_logits if num_logits is not None else vocab_size
        self.output_proj = nn.Linear(d_model, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len)  long tensor of token indices

        Returns
        -------
        logits : (batch, vocab_size) or (batch, num_logits) if num_logits was set
        """
        emb    = self.embedding(x)              # (batch, seq, d_model)
        out    = self.transformer(emb)           # (batch, seq, d_model)
        logits = self.output_proj(out[:, -1, :]) # take last token → (batch, vocab)
        return logits


def count_parameters(model: nn.Module) -> int:
    """Return the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
