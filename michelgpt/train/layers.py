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


class Linear(nn.Linear):
    '''Linear layer'''
    def __init__(self, in_features: int, out_features: int, bias: bool = False, device=DEVICE, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        # Reparametrize weights and bias here


class AttentionBlock(Module):
    '''Scaled Dot-Product Attention'''

    def __init__(self, attn_dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        attention = torch.matmul(q / math.sqrt(DIM_KEY), k.transpose(2, 3))

        if mask is not None:
            attention = attention.masked_fill(mask == 0, MASK_VALUE)

        attention = self.dropout(self.softmax(attention))
        output = torch.matmul(attention, v)

        return output, attention
    

class MultiHeadAttention(Module):
    '''Multi-Head Attention module'''
    def __init__(self) -> None:
        super().__init__()
        self.w_q = Linear(DIM_EMBEDDING, DIM_KEY * NUM_HEADS, bias=False) 
        self.w_k = Linear(DIM_EMBEDDING, DIM_KEY * NUM_HEADS, bias=False) 
        self.w_v = Linear(DIM_EMBEDDING, DIM_VALUE * NUM_HEADS, bias=False)

        self.attention = AttentionBlock()
        self.dropout = nn.Dropout(DROPOUT)
        self.layer_norm = nn.LayerNorm(DIM_MODEL)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        len_q, len_k, len_v = q.size(1), k.size(1), v.size(1)

        res = q

        q = self.w_q(q).view(batch_size, len_q, NUM_HEADS, DIM_KEY).transpose(1,2)
        k = self.w_k(k).view(batch_size, len_k, NUM_HEADS, DIM_KEY).transpose(1,2)
        v = self.w_v(v).view(batch_size, len_v, NUM_HEADS, DIM_VALUE).transpose(1,2)

        if mask is not None:
            mask = mask.unsqueeze(1)  

        v, attention = self.attention(q, k, v, mask=mask)
        v = self.dropout(v.transpose(1, 2).view(batch_size, len_v, -1)) 
        v += res

        out = self.layer_norm(v)
        return out, attention

class FeedForwardNet(Module):
    '''Position-Wise Feed Forward Network'''
    def __init__(self, d_in = DIM_MODEL, d_latent = DIM_MODEL // 2, dropout = DROPOUT) -> None:
        super().__init__()
        self.layer_1 = Linear(d_in, d_latent, dropout)
        self.activation = nn.ReLU()
        self.layer_2 = Linear(d_latent, d_in, dropout)
        self.dropout = nn.Dropout(DROPOUT)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
    
    def forward(self, x):
        res = x
        x = self.dropout(self.layer_2(self.activation(self.layer_1(x)))) 
        x += res
        out = self.layer_norm(x)
        return out
    

class EncoderLayer(Module):
    '''Encoder layer'''
    def __init__(self) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention()
        self.ffn = FeedForwardNet(d_in=DIM_MODEL, d_out=DIM_MODEL // 2)

    def forward(self, x, self_attention_mask=None):
        x, encoder_self_attention = self.self_attention(x, x, x, mask=self_attention_mask)
        out = self.ffn(x)
        return out, encoder_self_attention


class DecoderLayer(Module):
    '''Decoder layer'''
    def __init__(self) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention()
        self.encoder_attention = MultiHeadAttention()
        self.ffn = FeedForwardNet()

    def forward(self, dec_in, enc_out, self_attention_mask=None, ae_attention_mask=None):
        dec_out, decoder_self_attention = self.self_attention(dec_in, dec_in, dec_in, mask=self_attention_mask)
        dec_out, ae_attention = self.encoder_attention(dec_in, enc_out, enc_out, mask=ae_attention_mask)
        dec_out = self.ffn(dec_out)

        return dec_out, ae_attention, decoder_self_attention


