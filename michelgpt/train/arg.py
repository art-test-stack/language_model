from settings import *

class ModelArgs:
    def __init__(
            self,
            vocab_size: int = VOCAB_SIZE,
            dim: int = DIM_MODEL,
            n_layers: int = NUM_LAYERS, 
            n_heads: int = NUM_HEADS,
            d_head: int = DIM_HEAD,
            dropout: float = DROPOUT,
            padding_idx: int = 0,
            max_content: int = MAX_CONTEXT
        ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.vocab_size = vocab_size
        self.d_head = d_head
        self.dropout = dropout
        self.padding_idx = padding_idx
        self.max_content = max_content