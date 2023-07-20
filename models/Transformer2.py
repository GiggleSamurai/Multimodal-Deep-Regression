import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
"""
https://pytorch.org/tutorials/beginner/transformer_tutorial.html
"""
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000*2):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
class TransformerModel22(nn.Module):

    def __init__(self, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5): #, ntoken: int
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        #self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.mlp = nn.Sequential(
        nn.Flatten(),
        nn.Linear(64 * d_model, 256),
        nn.Linear(256, 256),
        nn.Linear(256, 1)) 
        #self.init_weights()

    # def init_weights(self) -> None:
    #     initrange = 0.1
    #     #self.embedding.weight.data.uniform_(-initrange, initrange)
    #     self.linear.bias.data.zero_()
    #     self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:

        src = src.long()
        src = src.view(1, 64, -1)
        src = src.permute(1, 0, 2)   #(Seq, Batch, Embedding_dim)
        #src = self.embedding(src) * math.sqrt(self.d_model) This is not needed.
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = output.permute(1, 0, 2)  #(Batch, Seq, Embedding_dim)
        output = self.mlp(output)
        return output
    

