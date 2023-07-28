"""
Establish baseline with SwinTransformer model
https://pypi.org/project/swin-transformer-pytorch/
"""
import torch
import torch.nn as nn
# from swin_transformer_pytorch import SwinTransformer
from util.video_swin_transformer import SwinTransformer3D


class Swin_Transformer_model(nn.Module):
    def __init__(self, linear_in_dim=1, patch_size=(4,4,4), embed_dim=96, drop_rate=0, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], patch_norm=False, window_size=(2,7,7), 
    scaled_target=False, out_features=1, classification_target=False ):
        super(Swin_Transformer_model, self).__init__()
        self.swin_transformer = SwinTransformer3D(
            patch_size=patch_size,
            embed_dim=embed_dim,
            drop_rate=drop_rate,
            depths=depths,
            window_size=window_size,
            num_heads=num_heads,
            patch_norm=patch_norm
            # Default Total parameters: 35,542,255
            # embed_dim=128, 
            # depths=[2, 2, 18, 2], 
            # num_heads=[4, 8, 16, 32], 
            # patch_size=(2,4,4), 
            # window_size=(16,7,7), 
            # drop_path_rate=0.4
        )
        self.scaled_target = scaled_target
        self.classification_target = classification_target
        self.flatten = nn.Flatten()
        if not self.scaled_target and not self.classification_target:
            self.relu = nn.ReLU()

        self.linear1 = nn.Linear(in_features=linear_in_dim, out_features=out_features)
        # self.linear2 = nn.Linear(in_features=10, out_features=1)
        # self.fc = nn.Sequential(
        #     nn.Linear(in_features=linear_in_dim, out_features=10),
        #     nn.ReLU(),            
        #     nn.Linear(in_features=10, out_features=1),
        #     nn.ReLU()
        # )
        


    def forward(self, x):
        x = self.swin_transformer(x)
        # print(self.swin_transformer)
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)
        if  not self.scaled_target and not self.classification_target:
            x = self.relu(self.linear1(x))
        else:
            x = self.linear1(x)
        # print(x.shape)
        # x = self.linear2(x)
        return x