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

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
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
class TransformerModel_Visual(nn.Module):

    def __init__(self, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5): #, ntoken: int
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        #self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(64 * d_model, 256)
        #self.init_weights()

    # def init_weights(self) -> None:
    #     initrange = 0.1
    #     #self.embedding.weight.data.uniform_(-initrange, initrange)
    #     self.linear.bias.data.zero_()
    #     self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, video_embed: Tensor) -> Tensor:

        #video_embed = video_embed.long()
        #video_embed = video_embed.view(1, 64, -1)
        B, D, H, W = video_embed.shape
        video_embed = video_embed.view(B, D, -1)
        video_embed = video_embed.permute(1, 0, 2)   #(Seq, Batch, Embedding_dim)
        #src = self.embedding(src) * math.sqrt(self.d_model) This is not needed.
        video_embed = self.pos_encoder(video_embed)
        video_embed = self.transformer_encoder(video_embed)
        video_embed = video_embed.permute(1, 0, 2)  #(Batch, Seq, Embedding_dim)
        video_embed = self.flatten(video_embed)
        video_embed = self.linear(video_embed)
        return video_embed

class GaussianNormalization(nn.Module):
    def __init__(self, min_val=-0.1, max_val=0.1):
        super(GaussianNormalization, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    @staticmethod
    def replace_minus_inf(tensor):
        mask = tensor != float('-inf')
        if mask.any():
            min_val = tensor[mask].min()
        else:
            min_val = 0.0
        tensor[~mask] = min_val
        return tensor
    
    def forward(self, x):
        x = self.replace_minus_inf(x)
        mean = x.mean()
        std = x.std()
        x = (x - mean) / (std + 1e-7)
        x = x * (self.max_val - self.min_val) / 2
        return x
    
class TransformerModel_Audio(nn.Module):

    def __init__(self, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5): #, ntoken: int
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        #self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(7 * d_model, 256)
        #self.gaussnoraml = GaussianNormalization()

    @staticmethod
    def replace_minus_inf(x):
        mask = x != float('-inf')
        if mask.any():
            min_val = x[mask].min()
        else:
            min_val = 0.0
        x[~mask] = min_val
        return x
    
    def forward(self, audio_embed) -> Tensor:
        
        audio_embed = audio_embed.permute(1, 0, 2)   #(Seq, Batch, Embedding_dim)
        audio_embed = self.replace_minus_inf(audio_embed)
        audio_embed = self.pos_encoder(audio_embed)
        audio_embed = self.transformer_encoder(audio_embed)
        audio_embed = audio_embed.permute(1, 0, 2)  #(Batch, Seq, Embedding_dim)     
        audio_embed = self.flatten(audio_embed)
        audio_embed = self.linear(audio_embed)
        return audio_embed