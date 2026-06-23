import torch
import torch.nn as nn

from .ffn import FeedForward
from .mha import MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        n_heads: int,
        context_length: int,
        dropout: float = 0.1,
        qvk_bias: bool = False
    ):
        super().__init__()
        self.attn = MultiHeadAttention(
            emb_dim=emb_dim,
            n_heads=n_heads,
            context_length=context_length,
            dropout=dropout,
            qvk_bias=qvk_bias,
        )
        self.ffn = FeedForward(emb_dim=emb_dim, dropout=dropout)
        self.ln_1 = nn.LayerNorm(emb_dim)
        self.ln_2 = nn.LayerNorm(emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x
