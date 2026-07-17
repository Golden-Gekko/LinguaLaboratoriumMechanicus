from transformers import PretrainedConfig


class LinguaLaboratoriumMechanicusConfig(PretrainedConfig):
    model_type = "lingua_laboratorium_mechanicus"

    def __init__(
        self,
        vocab_size: int = 50257,
        emb_dim: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        max_context_length: int = 1024,
        dropout: float = 0.1,
        qkv_bias: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_context_length = max_context_length
        self.dropout = dropout
        self.qkv_bias= qkv_bias

        # Алиасы для GenerationMixin
        self.num_hidden_layers = n_layers
        self.hidden_size = emb_dim
        self.num_attention_heads = n_heads
