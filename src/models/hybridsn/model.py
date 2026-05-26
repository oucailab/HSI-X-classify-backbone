import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class HybridSN(nn.Module):
    def __init__(self, classes):
        super(HybridSN, self).__init__()
        self.name = 'HybridSN'
        self.conv1 = nn.Sequential(
                    nn.Conv3d(
                    in_channels=1,
                    out_channels=8,
                    kernel_size=(37, 3, 3)),
                    nn.ReLU(inplace=True))
        
        self.conv2 = nn.Sequential(
                    nn.Conv3d(
                    in_channels=8,
                    out_channels=16,
                    kernel_size=(35, 3, 3)),
                    nn.ReLU(inplace=True))
        
        self.conv3 = nn.Sequential(
                    nn.Conv3d(
                    in_channels=16,
                    out_channels=32,
                    kernel_size=(33, 3, 3)),
                    nn.ReLU(inplace=True))
        
        self.conv4 = nn.Sequential(
                    nn.Conv2d(
                    in_channels=2496,    #576
                    out_channels=64,
                    kernel_size=(3, 3)),
                    nn.ReLU(inplace=True))
        
        
        self.dense1 = nn.Sequential(
                    nn.Linear(576,256),    #576
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.4))
        
        self.dense2 = nn.Sequential(
                    nn.Linear(256,128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.4))
        
        self.dense3 = nn.Sequential(
                    nn.Linear(128,classes)
                   )
        
    def forward(self, X):
        x = self.conv1(X)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0),x.size(1)*x.size(2),x.size(3),x.size(4))
        x = self.conv4(x)
        
        x = x.contiguous().view(x.size(0), -1)
        x = self.dense1(x)
        x = self.dense2(x)
        out = self.dense3(x)
        
        return out