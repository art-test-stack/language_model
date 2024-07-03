from michelgpt.settings import *
from michelgpt.train.layers import *

import matplotlib.pyplot as plt
import torch.nn as nn


class Encoder(Module):
    def __init__(
            self, d_src_vocab, dim_model: int = DIM_MODEL, n_layers: int = NUM_LAYERS, 
            n_heads: int = NUM_HEADS, d_key: int = DIM_KEY, d_value: int = DIM_VALUE, 
            dropout: float = DROPOUT, padding_idx: int = 0, max_content: int = MAX_CONTEXT 
            ) -> None:
        super().__init__()
        # TODO: Will change to customed embedding
        self.embedding = nn.Embedding(
            num_embeddings=d_src_vocab, embedding_dim=dim_model, padding_idx=padding_idx)
        self.pos_enc = PositionalEncoding(dim_model, max_content)
        self.layers = nn.ModuleList([
            EncoderLayer(
                dim_model=dim_model, n_heads=n_heads, d_key=d_key, d_value=d_value, dropout=dropout) 
            for _ in range(n_layers)]
        )

    def forward(self, src, src_mask=None, return_attention=False):
        
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
            dropout: float = DROPOUT, padding_idx: int = 0, max_content: int = MAX_CONTEXT
        ) -> None:
        super().__init__()
        # TODO: Will change to customed embedding
        self.embedding = nn.Embedding(
            num_embeddings=d_tgt_vocab, embedding_dim=dim_model, padding_idx=padding_idx)
        self.pos_enc = PositionalEncoding(dim_model, max_content)
        self.layers = nn.ModuleList([
            DecoderLayer(
                dim_model=dim_model, n_heads=n_heads, d_key=d_key, d_value=d_value, dropout=dropout)
            for _ in range(n_layers)]
        )
    
    def forward(self, seq, enc_output, src_mask=None, tgt_mask=None, return_attentions=False):
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
    def __init__(self, dim_model: int = DIM_MODEL, n_pos: int = MAX_CONTEXT) -> None:
        super().__init__()
        self.register_buffer('table', self._get_sinusoid_encoding_table(dim_model, n_pos))
    
    def _get_sinusoid_encoding_table(self, d_model: int = DIM_MODEL, n_pos: int = MAX_CONTEXT):
        ''' Sinusoid position encoding table '''
        pos = torch.arange(n_pos, dtype=torch.float32)
        i = torch.arange(d_model)

        pos_enc = torch.ger(pos, 1e4 ** (- 2 * (i//2) / d_model))

        pos_enc[:, 0::2] = torch.sin(pos_enc[:, 0::2])
        pos_enc[:, 1::2] = torch.cos(pos_enc[:, 1::2]) 
        return pos_enc
    
    def plot_table(self):
        pos_enc_np = self.table.cpu().numpy()
        plt.imshow(pos_enc_np, cmap='viridis', vmin=-1, vmax=1)
        plt.colorbar()
        plt.xlabel("Embedding index")
        plt.ylabel("Sequence index")
        plt.title('Sinusoidal Positional Encoding Table')
        plt.show()

    def forward(self, x):
        return x + self.table[:,:x.size(2)]


class Transformer(Module):
    def __init__(
            self, d_src_vocab, d_tgt_vocab, dim_model: int = DIM_MODEL, n_layers: int = NUM_LAYERS, 
            n_heads: int = NUM_HEADS, d_key: int = DIM_KEY, d_value: int = DIM_VALUE, 
            dropout: float = DROPOUT, src_padding_idx: int = 0, tgt_padding_idx: int = 0, max_content: int = MAX_CONTEXT
            ) -> None:
        super().__init__()

        self.src_pad_idx, self.tgt_pad_idx = src_padding_idx, tgt_padding_idx

        self.encoder = Encoder(d_src_vocab=d_src_vocab, dim_model=dim_model, n_layers=n_layers, 
            n_heads=n_heads, d_key=d_key, d_value=d_value, dropout=dropout, padding_idx=src_padding_idx, max_content=max_content)
        
        self.decoder = Decoder(d_tgt_vocab=d_tgt_vocab, dim_model=dim_model, n_layers=n_layers, 
            n_heads=n_heads, d_key=d_key, d_value=d_value, dropout=dropout, padding_idx=tgt_padding_idx, max_content=max_content)
        
        self.linear = Linear(dim_model, d_tgt_vocab, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

    def forward(self, src_seq, tgt_seq):
        src_mask = self.get_pad_mask(src_seq, "src")
        tgt_mask = self.get_pad_mask(tgt_seq, "tgt")

        enc_out, *_ = self.encoder(src_seq, src_mask=src_mask)
        dec_out, *_ = self.decoder(seq=tgt_seq, enc_output=enc_out, src_mask=src_mask, tgt_mask=tgt_mask)

        return self.softmax(self.linear(dec_out))
    
    def get_pad_mask(self, seq, mode: str = "src"):
        assert mode == "src" or mode == "tgt"

        pad_idx = self.src_pad_idx if mode == "src" else self.tgt_pad_idx
        pad_mask = (seq != pad_idx).unsqueeze(-2)

        if mode == "src":
            return pad_mask
        else:
            _, len_s = seq.size()
            subsequent_mask = (1 - torch.triu(
                torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
            return pad_mask & subsequent_mask