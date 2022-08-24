import torch.nn as nn
from torch import Tensor,float,exp,arange,sin,cos,zeros
from torch import float
from torch import exp
from math import log,sqrt
import torch.nn.functional as f
from torch.autograd import Variable

#              -------
from datetime import datetime
import json
import logging
import os
import tarfile
import tempfile
import socket

import torch

from transformers import cached_path

#               ---------

class TokenEmbedding(nn.Module):
    def __init__(self,vocab_size: int, emb_dim,padding):
        super(TokenEmbedding,self).__init__()

        self.embedding = nn.Embedding(vocab_size,emb_dim,padding_idx=padding)
        self.emb_dim = emb_dim

    def forward(self,tokens: Tensor):
        return self.embedding(tokens.long()) * sqrt(self.emb_dim)
    


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = zeros(max_len, d_model)
        position = arange(0, max_len).unsqueeze(1)
        div_term = exp(arange(0, d_model, 2) *
                             -(log(10000.0) / d_model))
        pe[:, 0::2] = sin(position * div_term)
        pe[:, 1::2] = cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)



def _get_clones(module, N):
    try:
        import copy
    except:
        print("an exception occur: check if copy module is installed on python")
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])




class PositionWiseFeedforward(nn.Module):
    def __init__(self,d_model,d_ff,device=None,dtype=None,dropout=0.1):
        kwargs={'device':device,'dtype':dtype}

        super(PositionWiseFeedforward,self).__init__()
        
        self.ff1=nn.Linear(d_model,d_ff,**kwargs)
        self.ff2=nn.Linear(d_ff,d_model,**kwargs)

        self.dropout=nn.Dropout(dropout)
        
    def forward(self,x):
        return(self.ff2(self.dropout(f.relu(self.ff1(x)))))
    


class ResidualLayerNorm(nn.Module):
    def __init__(self,d_model,eps_layer_norm,dropout=0.2,device=None,dtype=None):
        kwargs={'device':device,'dtype':dtype}

        super().__init__()
        self.layer_norm=nn.LayerNorm(d_model, eps=eps_layer_norm,**kwargs)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x,residual):
        
        out=self.layer_norm(self.dropout(x)+residual)
        
        return out
    

def _get_activation_fn(activation):
    if activation == "relu":
        return f.relu
    elif activation == "gelu":
        return f.gelu

    raise RuntimeError("activation should be relu/gelu, but not {}".format(activation))
        



class Generator(nn.Module):

    def __init__(self,d_model,trg_vocab_len,dropout):
        super(Generator,self).__init__()
        
        self.proj1=nn.Linear(d_model,d_model*2)
        self.proj2=nn.Linear(d_model*2,trg_vocab_len)

        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        
        out=self.dropout(self.proj1(x))
        return f.log_softmax(self.proj2(out),dim=-1)


