from michelgpt.settings import *
from michelgpt.train.layers import *
from michelgpt.train.module import Module

import torch.nn as nn


class Decoder(Module):
    def __init__(
            self, 
            vocab_size: int = VOCAB_SIZE,
            dim_model: int = DIM_MODEL,
            dim_ffn: int = DIM_FFN,
            n_layers: int = NUM_LAYERS, 
            n_heads: int = NUM_HEADS,
            d_head: int = DIM_HEAD,
            dropout: float = DROPOUT,
            padding_idx: int = 0
        ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(
                dim_model=dim_model, 
                dim_ffn=dim_ffn, 
                n_heads=n_heads, 
                d_head=d_head, 
                dropout=dropout
            )
            for _ in range(n_layers)]
        )
    
    def forward(
            self, 
            x: torch.Tensor, 
            mask=None,
            return_attentions=False
        ):
        self_attn_list = []
        
        for layer in self.layers: 
            output, self_attention = layer(x=output, self_attention_mask=mask)
            if return_attentions:
                self_attn_list.append(self_attention) 

        if return_attentions:
            return output, self_attn_list
        
        return output,


class MichelTransformer(Module):
    def __init__(
            self,
            vocab_size: int = VOCAB_SIZE,
            dim_model: int = DIM_MODEL,
            n_layers: int = NUM_LAYERS, 
            n_heads: int = NUM_HEADS,
            d_head: int = DIM_HEAD,
            dropout: float = DROPOUT,
            padding_idx: int = 0,
            max_content: int = MAX_CONTEXT
            ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        # TODO: Will change to customed embedding
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, 
            embedding_dim=dim_model, 
            padding_idx=padding_idx
        )
        self.pos_enc = PositionalEncoding(dim_model, max_content)

        self.decoder_stack = Decoder(
            vocab_size=vocab_size,
            dim_model=dim_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_head=d_head,
            dropout=dropout, 
            padding_idx=padding_idx, 
            max_content=max_content
        )
        
        self.model_head = Linear(dim_model, vocab_size, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = self.get_pad_mask(x)

        x = self.embedding(x)
        x = self.pos_enc(x)

        output, *_ = self.decoder_stack(x=x, mask=mask)

        return self.model_head(output)
    
    def get_pad_mask(self, seq: torch.Tensor):

        pad_idx = self.padding_idx
        pad_mask = (seq != pad_idx).unsqueeze(-2)

        _, len_s = seq.size()
        subsequent_mask = (1 - torch.triu(
            torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
        return pad_mask & subsequent_mask