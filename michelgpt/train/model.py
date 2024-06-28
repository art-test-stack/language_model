from michelgpt.settings import *
from michelgpt.train.layers import *
import torch.nn as nn


class Encoder(Module):
    def __init__(
            self, d_src_vocab, dim_model: int = DIM_MODEL, n_layers: int = NUM_LAYERS, 
            n_heads: int = NUM_HEADS, d_key: int = DIM_KEY, d_value: int = DIM_VALUE, 
            dropout: float = DROPOUT, padding_idx: int = 0
            ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=d_src_vocab, embedding_dim=dim_model, padding_idx=padding_idx)
        self.pos_enc = PositionalEncoding() # TODO
        self.layers = nn.ModuleList([
            EncoderLayer(
                dim_model=dim_model, n_heads=n_heads, d_key=d_key, d_value=d_value, dropout=dropout) 
            for _ in range(n_layers)]
        )

    def forward(self, src, src_mask, return_attention=False):
        
        attention_list = []

        out = self.embedding(src)
        out = self.pos_enc(out)
        for layer in self.layers:
            out, attention = layer(out, self_attention_mask=src_mask)
            attention_list.append(attention) if return_attention else None
        
        return (out, attention) if return_attention else (out,)


class Decoder(Module):
    def __init__(
            self, d_tgt_vocab, dim_model: int = DIM_MODEL, n_layers: int = NUM_LAYERS, 
            n_heads: int = NUM_HEADS, d_key: int = DIM_KEY, d_value: int = DIM_VALUE, 
            dropout: float = DROPOUT, padding_idx: int = 0
        ) -> None:
        super().__init__()
        # TODO: Will change to customed embedding
        self.embedding = nn.Embedding(
            num_embeddings=d_tgt_vocab, embedding_dim=dim_model, padding_idx=padding_idx
        )
        self.pos_enc = PositionalEncoding() # TODO
        self.layers = nn.ModuleList([
            DecoderLayer(
                dim_model=dim_model, n_heads=n_heads, d_key=d_key, d_value=d_value, dropout=dropout)
            for _ in range(n_layers)]
        )
    
    def forward(self, seq, enc_output, src_mask, tgt_mask, return_attentions=False):
        self_attn_list, enc_dec_attn_list = [], []
        
        dec = self.embedding(seq)
        dec_out = self.pos_enc(dec)
        for layer in self.layers: 
            dec_out, self_attention, enc_dec_attention = layer(
                dec_in=dec_out, enc_out=enc_output, self_attention_mask=tgt_mask, enc_dec_attention_mask=src_mask)
            if return_attentions:
                self_attn_list.append(self_attention) 
                enc_dec_attn_list.append(enc_dec_attention)

        if return_attentions:
            return dec_out, self_attn_list, enc_dec_attn_list
        
        return dec_out,


class PositionalEncoding(Module):
    def __init__(self) -> None:
        super().__init__()
        # TODO


class Transformer(Module):
    def __init__(self) -> None:
        super().__init__()
        # TODO