"""
Establish baseline with SwinTransformer model
https://pypi.org/project/swin-transformer-pytorch/
"""
import torch
import torch.nn as nn
# from swin_transformer_pytorch import SwinTransformer
from util.video_swin_transformer import SwinTransformer3D


class Swin_Transformer_model(nn.Module):
    def __init__(self, linear_in_dim=1):
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
        self.fc = nn.Sequential(
            nn.Linear(in_features=linear_in_dim, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1),
        )


    def forward(self, x):
        x = self.swin_transformer(x)
        # print(self.swin_transformer)
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)
        x = self.fc(x)
        return x