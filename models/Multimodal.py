import torch
import torch.nn as nn

# Combine the Multiple modules into a single model
class MultiModal_Model(nn.Module):
    def __init__(self):
        super(MultiModal_Model, self).__init__()
        # the CNN module
        self.cnn = CNN()
        # the MLP module
        self.mlp = MLP()  

    def forward(self, x):
        x = self.cnn(x)
        x = self.mlp(x)
        return x