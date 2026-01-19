import torch
import torch.nn as nn

from . import TransformerBlock


class LinguaLaboratoriumMechanicus(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        n_layers: int,
        n_heads: int,
        max_context_length: int = 1024,
        dropout: float = 0.1,
        qvk_bias: bool = False
    ):
        super().__init__()

        cfg = {
            'emb_dim': emb_dim,
            'n_heads': n_heads,
            'context_length': max_context_length,
            'dropout': dropout,
            'qvk_bias': qvk_bias
        }

        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.max_context_length = max_context_length

        # Эмбеддинги
        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(max_context_length, emb_dim)
        self.drop_emb = nn.Dropout(dropout)

        # Паравозик Трансформер
        self.blocks = nn.Sequential(
            *[TransformerBlock(**cfg) for _ in range(n_layers)]
        )

        self.final_norm = nn.LayerNorm(emb_dim)
        self.out_head = nn.Linear(emb_dim, vocab_size, bias=False)

        # Начальная инициализация весов
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        _, n_tokens = input_ids.size()
        if n_tokens > self.max_context_length:
            raise ValueError(
                f'Длина входной последовательности ({n_tokens}) превышает '
                f'максимальную заданную ({self.max_context_length}).'
            )

        x = self.drop_emb(
            self.token_emb(input_ids) +
            self.pos_emb(
                torch.arange(n_tokens, device=input_ids.device)
                .unsqueeze(0)
            )
        )

        x = self.blocks(x)

        return self.out_head(self.final_norm(x))
