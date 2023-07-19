"""
Establish baseline with SwinTransformer model
https://pypi.org/project/swin-transformer-pytorch/
"""
import torch
import torch.nn as nn
# from swin_transformer_pytorch import SwinTransformer
from util.video_swin_transformer import SwinTransformer3D


class Swin_Transformer_model(nn.Module):
    def __init__(self):
        super(Swin_Transformer_model, self).__init__()
        self.swin_transformer = SwinTransformer3D(
            # Default Total parameters: 35,542,255
            # embed_dim=128, 
            # depths=[2, 2, 18, 2], 
            # num_heads=[4, 8, 16, 32], 
            # patch_size=(2,4,4), 
            # window_size=(16,7,7), 
            # drop_path_rate=0.4
        )
        self.flatten = nn.Flatten()
        self.regressor = nn.Linear(7962624, 1)

    def forward(self, x):
        x = self.swin_transformer(x)
        print(self.swin_transformer)
        print(x.shape)
        x = self.flatten(x)
        print(x.shape)
        x = self.regressor(x)
        return x