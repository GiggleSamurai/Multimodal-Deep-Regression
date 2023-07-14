"""
Establish baseline with SwinTransformer model
https://pypi.org/project/swin-transformer-pytorch/
"""
import torch
import torch.nn as nn
from swin_transformer_pytorch import SwinTransformer

class Swin_Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(Swin_Transformer, self).__init__()
        self.swin_transformer = SwinTransformer(
            # parameters
        )
        self.regressor = nn.Linear(hidden_size, output_size=1)

    def forward(self, x):
        x = self.swin_transformer(x)
        x = self.regressor(x)
        return x