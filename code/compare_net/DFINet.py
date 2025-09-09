import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score

import torch, math
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.io import loadmat, savemat
import random
from time import time

class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class Dropout(nn.Module):
    def __init__(self):
        super(Dropout, self).__init__()

    def forward(self, x):
        out = F.dropout(x, p = 0.2, training=self.training)
        return out

class HSINet(nn.Module):
  def __init__(self, channel_hsi):
    super(HSINet, self).__init__()

    self.conv1 = nn.Conv2d(channel_hsi, 256, 3, padding=1)
    self.bn1 = nn.BatchNorm2d(256)

    self.conv2 = nn.Conv2d(256, 128, 3)
    self.bn2 = nn.BatchNorm2d(128)
    self.conv3 = nn.Conv2d(128, 128, 3)
    self.bn3 = nn.BatchNorm2d(128)

  def forward(self, x):

    x = F.relu(self.bn1(self.conv1(x)))
    x = F.relu(self.bn2(self.conv2(x)))
    x = F.relu(self.bn3(self.conv3(x)))
    return x

class MSINet(nn.Module):
  def __init__(self, channel_msi):
    super(MSINet, self).__init__()

    self.conv1 = nn.Conv2d(channel_msi, 128, 3, padding =1)
    self.bn1 = nn.BatchNorm2d(128)

    self.conv2 = nn.Conv2d(128, 128, 3)
    self.bn2 = nn.BatchNorm2d(128)

    self.conv3 = nn.Conv2d(128, 128, 3)
    self.bn3 = nn.BatchNorm2d(128)

  def forward(self, x):

    x = F.relu(self.bn1(self.conv1(x)))
    x = F.relu(self.bn2(self.conv2(x)))
    x = F.relu(self.bn3(self.conv3(x)))
    return x

class CAM(nn.Module):
    def __init__(self):
        super(CAM, self).__init__()      
        k_size = 3 
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        # self.conv1 = nn.Conv2d(9, 7, 1) # 81 is the spatial size of features
        # self.conv2 = nn.Conv2d(7, 49, 1, stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def get_attention(self, a):

        input_a = a
        a = a.mean(3)
        a = a.transpose(1, 3)
        # a= F.relu(self.conv1(a))
        # a= self.conv2(a)
        a = self.conv(a.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        a = a.transpose(1, 3)

        a = a.unsqueeze(3)
        a = torch.mean(input_a * a, -1)
        a = F.softmax(a / 0.025, dim=-1) +1
        return a 

    def forward(self, f1, f2):

        b, n1, c, h, w = f1.size()
        n2 = f2.size(1)

        f1 = f1.view(b, n1, c, -1) 
        f2 = f2.view(b, n2, c, -1)

        f1_norm = F.normalize(f1, p=2, dim=2, eps=1e-12)
        f2_norm = F.normalize(f2, p=2, dim=2, eps=1e-12)
        
        f1_norm = f1_norm.transpose(2, 3).unsqueeze(2)
        f2_norm = f2_norm.unsqueeze(1)

        a1 = torch.matmul(f1_norm, f2_norm) 
        a2 = a1.transpose(3, 4) 

        a1 = self.get_attention(a1)
        a2 = self.get_attention(a2)
        f1 = f1 * a1
        f1 = f1.view(b, c, h, w)
        f2 = f2 * a2
        f2 = f2.view(b, c, h, w)
        return f1, f2


class Net(nn.Module):
    def __init__(self, channel_hsi, channel_msi, class_num):
        super(Net, self).__init__()

        self.featnet1 = HSINet(channel_hsi)
        self.featnet2 = MSINet(channel_msi)
        self.cam = CAM()
        self.proj_norm = LayerNorm(64)
        self.fc1 = nn.Linear(1 * 1 * 128, 64)
        self.fc2 = nn.Linear(64, class_num)
        self.dropout = nn.Dropout()

    def forward(self, x, y):

        # Pre-process Image Feature
        feature_1 = self.featnet1(x)
        feature_2 = self.featnet2(y)

        hsi_feat = feature_1.unsqueeze(1)
        lidar_feat = feature_2.unsqueeze(1)
        hsi, lidar = self.cam(hsi_feat, lidar_feat)
        x = self.xcorr_depthwise(hsi, lidar)
        y = self.xcorr_depthwise(lidar, hsi)
        x1 = x.contiguous().view(x.size(0), -1)
        y1 = y.contiguous().view(y.size(0), -1)
        x = x1 + y1
        x = F.relu(self.proj_norm(self.fc1(x)))
        
        x = self.dropout(x)
        x = self.fc2(x)
        # hsi = hsi.contiguous().view(x.size(0), -1)
        # lidar = lidar.contiguous().view(x.size(0), -1)
        # return feature_1, feature_2, x1, y1, x
        return x

    def xcorr_depthwise11(self, x, kernel):
        batch = kernel.size(0)
        channel = kernel.size(1)
        x = x.view(1, batch * channel, x.size(2), x.size(3))
        kernel = kernel.view(batch, channel, kernel.size(2), kernel.size(3))
        out = F.conv2d(x, kernel, groups= 1)
        # out = F.relu(out)
        out = out.view(batch, 1, out.size(2), out.size(3))
        return out

    def xcorr_depthwise(self, x, kernel):
        batch = kernel.size(0)
        channel = kernel.size(1)
        x = x.view(1, batch * channel, x.size(2), x.size(3))
        kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))
        out = F.conv2d(x, kernel, groups=batch*channel)
        # out = F.relu(out)
        out = out.view(batch, channel, out.size(2), out.size(3))
        return out