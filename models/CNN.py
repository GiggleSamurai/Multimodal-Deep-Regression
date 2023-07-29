import torch
import torch.nn as nn

import numpy as np

class CNN3D(nn.Module):
    def __init__(self, filters = 16, dropout_rate = 0.0, reduce_pool = 4, depth = 64, shrink=1):
        super(CNN3D, self).__init__()
        self.channels = 3
        self.H = 1024 // shrink
        self.W = 576 // shrink
        self.D = depth
        self.filters = filters
        self.dropout_rate = dropout_rate
        # reduce adaptive pool size
        self.reduce = reduce_pool # 1 is unchanged, 2 is half, 4 is quarter..32, 64.
        self.H = self.H // self.reduce
        self.W = self.W // self.reduce
        self.D = self.D // self.reduce

        self.conv1 = nn.Conv3d(3, self.filters, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        self.adaptive_pool = nn.AdaptiveAvgPool3d((self.D, self.H, self.W))
        self.fc = nn.Linear(self.filters * self.D * self.H * self.W, 1)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc(x)
        return x


class CNN3DClassifier(nn.Module):
    def __init__(self, filters = 16, dropout_rate = 0.0, reduce_pool = 4, depth = 64, shrink=1, classes=4):
        super(CNN3DClassifier, self).__init__()
        self.channels = 3
        self.H = 1024 // shrink
        self.W = 576 // shrink
        self.D = depth
        self.filters = filters
        self.dropout_rate = dropout_rate
        self.classes = classes
        # reduce adaptive pool size
        self.reduce = reduce_pool # 1 is unchanged, 2 is half, 4 is quarter..32, 64.
        self.H = self.H // self.reduce
        self.W = self.W // self.reduce
        self.D = self.D // self.reduce

        self.conv1 = nn.Conv3d(3, self.filters, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(self.filters, self.filters, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        self.adaptive_pool = nn.AdaptiveAvgPool3d((self.D, self.H, self.W))
        self.fc = nn.Linear(self.filters * self.D * self.H * self.W, self.classes)
        self.dropout = nn.Dropout(self.dropout_rate)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc(x)
        print(x)
        # x = self.softmax(x, dim=0)
        # print(x)
        # x = np.argmax(np.array(x),1)
        return x


class CNN3DComplex(nn.Module):
    def __init__(self, filters = 16, dropout_rate = 0.0, reduce_pool = 4, depth = 64, shrink=1):
        super(CNN3DComplex, self).__init__()
        self.channels = 3
        self.H = 1024 // shrink
        self.W = 576 // shrink
        self.D = depth
        self.filters = filters
        self.dropout_rate = dropout_rate
        # reduce adaptive pool size
        self.reduce = reduce_pool # 1 is unchanged, 2 is half, 4 is quarter..32, 64.
        self.H = self.H // self.reduce
        self.W = self.W // self.reduce
        self.D = self.D // self.reduce

        self.conv1 = nn.Conv3d(3, self.filters, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(self.filters, self.filters, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(self.filters, self.filters, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        self.adaptive_pool = nn.AdaptiveAvgPool3d((self.D, self.H, self.W))
        self.fc = nn.Linear(self.filters * self.D * self.H * self.W, 1)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc(x)
        return x