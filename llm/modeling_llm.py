from typing import Optional

import torch
import torch.nn as nn
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_llm import LinguaLaboratoriumMechanicusConfig

from llm.transformer import TransformerBlock


class LLMForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = LinguaLaboratoriumMechanicusConfig
    _no_split_modules = ['TransformerBlock']

    def __init__(self, config: LinguaLaboratoriumMechanicusConfig):
        super().__init__(config)

        self.vocab_size = config.vocab_size
        self.emb_dim = config.emb_dim
        self.max_context_length = config.max_context_length

        self.token_emb = nn.Embedding(config.vocab_size, config.emb_dim)
        self.pos_emb = nn.Embedding(config.max_context_length, config.emb_dim)
        self.drop_emb = nn.Dropout(config.dropout)

        self.blocks = nn.Sequential(*[
            TransformerBlock(
                emb_dim=config.emb_dim,
                n_heads=config.n_heads,
                context_length=config.max_context_length,
                dropout=config.dropout,
                qvk_bias=config.qvk_bias,
            )
            for _ in range(config.n_layers)
        ])

        self.final_norm = nn.LayerNorm(config.emb_dim)
        self.out_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

        self.post_init()

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if input_ids is None:
            raise ValueError('input_ids обязателен')

        _, n_tokens = input_ids.size()
        if n_tokens > self.max_context_length:
            raise ValueError(
                f'Длина входной последовательности ({n_tokens}) превышает '
                f'максимальную заданную ({self.max_context_length}).'
            )

        x = self.drop_emb(
            self.token_emb(input_ids) +
            self.pos_emb(
                torch.arange(n_tokens, device=input_ids.device).unsqueeze(0)
            )
        )

        x = self.blocks(x)
        logits = self.out_head(self.final_norm(x))

        return CausalLMOutputWithPast(logits=logits, past_key_values=None)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {'input_ids': input_ids}
