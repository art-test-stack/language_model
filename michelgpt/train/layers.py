from michelgpt.settings import *

import math
import numpy as np
import torch.nn as nn
from torch.nn import functional as F


class Module(nn.Module):
    '''class Module'''
    def nb_parameters(self) -> int:
        '''Give the number of parameters of the module'''
        return sum([np.prod(p.size(), dtype = np.int32) for p in self.parameters()])

    def nb_trainable_parameters(self) -> int:
        '''Give the number of trainable parameters of the module'''
        return sum([np.prod(p.size(), dtype = np.int32) for p in self.parameters() if p.requires_grad])

    def nb_non_trainable_parameters(self) -> int:
        '''Give the number of non-trainable parameters of the module'''
        return sum([np.prod(p.size(), dtype = np.int32) for p in self.parameters() if not p.requires_grad])

    def summary(self) -> None:
        '''Summarize the module'''
        print(f'Number of parameters: {self.nb_parameters():,}')
        print(f'Number of trainable parameters: {self.nb_trainable_parameters():,}')
        print(f'Number of non-trainable parameters: {self.nb_non_trainable_parameters():,}')

    def clean_nan(self) -> None:
        '''Remove NaNs from the module gradients'''
        for p in self.parameters():
            if p.grad is not None:
                torch.nan_to_num(p.grad, nan = 0, posinf = 1e5, neginf = -1e5, out = p.grad)

    def clip_gradient(self, max_norm: float) -> None:
        '''Clip the module gradients'''
        nn.utils.clip_grad_norm_(self.parameters(), max_norm)


# class Linear(nn.Linear):
#     def __init__(self, in_features: int, out_features: int, bias: bool = False, device=DEVICE) -> None:
#         super().__init__(in_features, out_features, bias, device)


class Linear(nn.Linear, Module):
    '''Linear layer'''
    def __init__(self, in_features: int, out_features: int, bias: bool = False, device: torch.device = DEVICE, dtype=None) -> None:
        super(Linear, self).__init__(in_features=in_features, out_features=out_features, bias=bias, device=device)
        # TODO: Reparametrize weights and bias here


class AttentionBlock(Module):
    '''Scaled Dot-Product Attention'''

    def __init__(self, dropout: float =0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, d_key: int = DIM_KEY, mask=None, mask_value: int = MASK_VALUE):
        attention = torch.matmul(q / math.sqrt(d_key), k.transpose(2, 3))

        if mask is not None:
            attention = attention.masked_fill(mask == 0, mask_value)

        attention = self.dropout(self.softmax(attention))
        output = torch.matmul(attention, v)
        # TODO: add a way to get attention mechanism weights representation
        return output, attention
    

class MultiHeadAttention(Module):
    '''Multi-Head Attention module'''
    def __init__(
            self, dim_model: int = DIM_MODEL, n_heads: int = NUM_HEADS, d_key: int = DIM_KEY, 
            d_value: int = DIM_VALUE, dropout: float = DROPOUT
        ) -> None:
        super().__init__()
        assert(dim_model == d_key * n_heads, "Dimensions are not correct")
        self.dims = (n_heads, d_key, d_value)

        self.w_q = Linear(dim_model, d_key * n_heads, bias=False) 
        self.w_k = Linear(dim_model, d_key * n_heads, bias=False) 
        self.w_v = Linear(dim_model, d_value * n_heads, bias=False)

        self.attention = AttentionBlock()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None):
        n_heads, d_k, d_v = self.dims
        batch_size = q.size(0)
        len_q, len_k, len_v = q.size(1), k.size(1), v.size(1)

        res = q

        q = self.w_q(q).view(batch_size, len_q, n_heads, d_k).transpose(1,2)
        k = self.w_k(k).view(batch_size, len_k, n_heads, d_k).transpose(1,2)
        v = self.w_v(v).view(batch_size, len_v, n_heads, d_v).transpose(1,2)

        if mask is not None:
            mask = mask.unsqueeze(1)  
        
        v, attention = self.attention(q, k, v, mask=mask)
        v = self.dropout(v.transpose(1, 2).contiguous().view(batch_size, len_v, -1)) 
        v += res

        out = self.layer_norm(v)
        return out, attention


class FeedForwardNet(Module):
    '''Position-Wise Feed Forward Network'''
    def __init__(self, d_in: int = DIM_MODEL, d_latent: int = DIM_MODEL // 2, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.layer_1 = Linear(d_in, d_latent, dropout)
        self.activation = nn.ReLU()
        self.layer_2 = Linear(d_latent, d_in, dropout)
        self.dropout = nn.Dropout(DROPOUT)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.dropout(self.layer_2(self.activation(self.layer_1(x)))) 
        x += res
        out = self.layer_norm(x)
        return out
    

class EncoderLayer(Module):
    '''Encoder layer'''
    def __init__(
            self, dim_model: int = DIM_MODEL, n_heads: int = NUM_HEADS, d_key: int = DIM_KEY, 
            d_value: int = DIM_VALUE, dropout: float = DROPOUT
        ) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(
            dim_model=dim_model, n_heads=n_heads, d_key=d_key, d_value=d_value, dropout=dropout)
        self.ffn = FeedForwardNet(d_in=dim_model, d_latent=DIM_MODEL // 2)

    def forward(self, x, self_attention_mask=None):
        x, enc_attention = self.self_attention(x, x, x, mask=self_attention_mask)
        out = self.ffn(x)
        return out, enc_attention


class DecoderLayer(Module):
    '''Decoder layer'''
    def __init__(
            self, dim_model: int = DIM_MODEL, n_heads: int = NUM_HEADS, d_key: int = DIM_KEY, 
            d_value: int = DIM_VALUE, dropout: float = DROPOUT
        ) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(
            dim_model=dim_model, n_heads=n_heads, d_key=d_key, d_value=d_value, dropout=dropout)
        self.enc_attention = MultiHeadAttention(
            dim_model=dim_model, n_heads=n_heads, d_key=d_key, d_value=d_value, dropout=dropout)
        self.ffn = FeedForwardNet(d_in=dim_model, d_latent=DIM_MODEL // 2)

    def forward(self, dec_in, enc_out, self_attention_mask=None, enc_dec_attention_mask=None):
        dec_out, dec_attention = self.self_attention(dec_in, dec_in, dec_in, mask=self_attention_mask)
        dec_out, enc_dec_attention = self.enc_attention(dec_out, enc_out, enc_out, mask=enc_dec_attention_mask)
        dec_out = self.ffn(dec_out)

        return dec_out, dec_attention, enc_dec_attention


