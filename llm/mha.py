import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        n_heads: int,
        context_length: int,
        dropout: float = 0.1,
        qvk_bias: bool = False
    ):
        super().__init__()
        if emb_dim % n_heads != 0:
            raise ValueError('emb_dim должно быть кратно n_heads')

        self.model_dim = emb_dim
        self.n_heads = n_heads
        self.head_dim = self.model_dim // n_heads

        # Матрица для всех голов Q, K, V
        self.qkv_proj = nn.Linear(
            self.model_dim, 3 * self.model_dim, bias=qvk_bias)
        self.out_proj = nn.Linear(
            self.model_dim, self.model_dim, bias=qvk_bias)

        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            'mask',
            torch.tril(torch.ones(1, 1, context_length, context_length))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_tokens, emb_dim = x.size()

        # Проекция в Q, K, V
        qkv = self.qkv_proj(x)  # (batch_size, n_tokens, 3 * emb_dim)
        q, k, v = qkv.split(
            self.model_dim, dim=-1)  # 3 по (batch_size, n_tokens, emb_dim)

        # Разбиение на головы - делим d_model на n_heads частей
        # (batch_size, n_tokens, emb_dim) ->
        # (batch_size, n_tokens, n_heads, head_dim) ->
        # (batch_size, n_heads, n_tokens, head_dim)
        queries = q.view(
            batch_size, n_tokens, self.n_heads, self.head_dim).transpose(1, 2)
        keys = k.view(
            batch_size, n_tokens, self.n_heads, self.head_dim).transpose(1, 2)
        values = v.view(
            batch_size, n_tokens, self.n_heads, self.head_dim).transpose(1, 2)

        # scores = Q @ K.T - транспонирование в рамках голов
        attn_weights = (
            queries @ keys.transpose(-2, -1)
        ) / (self.head_dim ** 0.5)  # (batch_size, n_heads, n_tokens, n_tokens)

        attn_weights = attn_weights.masked_fill(
            self.mask[:, :, :n_tokens, :n_tokens] == 0, float('-inf'))
        attn_weights = self.dropout(torch.softmax(attn_weights, dim=-1))

        # (batch_size, n_heads, n_tokens, head_dim)
        out_per_head = attn_weights @ values

        # Слияние голов
        # (batch_size, n_heads, n_tokens, head_dim) ->
        # (batch_size, n_tokens, n_heads, head_dim) ->
        # (batch_size, n_tokens, emb_dim)
        out = (
            out_per_head.transpose(1, 2)
            .contiguous()
            .view(batch_size, n_tokens, emb_dim)
        )

        return self.out_proj(out)
