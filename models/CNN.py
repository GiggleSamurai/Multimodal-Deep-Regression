import torch
import torch.nn as nn

class CNN3D(nn.Module):
    def __init__(self):
        super(CNN3D, self).__init__()
        self.channels = 3
        self.filters = 16
        self.conv1 = nn.Conv3d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(23592960, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.relu(x)
        #print('Check size',x.size())
        x = self.fc(x)
        return x