import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim



class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super().__init__()
        self.stddev = stddev

    def forward(self, x):
        if self.training and self.stddev > 0:
            noise = torch.randn_like(x) * self.stddev
            return x + noise
        return x


class CascadeBlock(nn.Module):
    def __init__(self, in_channels, nb_filter, kernel_size=3):
        super().__init__()
        self.conv1_1 = nn.Conv2d(in_channels, nb_filter * 2, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(nb_filter * 2)
        self.conv1_2 = nn.Conv2d(nb_filter * 2, nb_filter, 1)
        self.bn2 = nn.BatchNorm2d(nb_filter)

        self.conv_x = nn.Conv2d(in_channels, nb_filter * 2, 1, bias=False)

        self.conv2_1 = nn.Conv2d(nb_filter, nb_filter * 2, kernel_size, padding=kernel_size // 2)
        self.bn3 = nn.BatchNorm2d(nb_filter * 2)
        self.conv2_2 = nn.Conv2d(nb_filter * 2, nb_filter, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(nb_filter)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        # First path
        x1 = self.conv1_1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        x1 = self.conv1_2(x1)
        x1 = self.bn2(x1)
        x1 = self.leaky_relu(x1)

        # Second path
        x2 = self.conv_x(x)
        x3 = self.conv2_1(x1)
        x3 = x2 + x3
        x3 = self.bn3(x3)
        x3 = self.leaky_relu(x3)

        x3 = self.conv2_2(x3)
        x3 = x1 + x3
        x3 = self.bn4(x3)
        x3 = self.leaky_relu(x3)

        return x3


class CascadeNet(nn.Module):
    def __init__(self, in_channels, r=5):
        super().__init__()
        self.ksize = 2 * r + 1
        filters = [16, 32, 64, 96, 128, 192, 256, 512]

        self.conv0 = nn.Conv2d(in_channels, filters[2], 3, padding=1)
        self.block1 = CascadeBlock(filters[2], filters[2])
        self.pool = nn.MaxPool2d(2, 2)
        self.block2 = CascadeBlock(filters[2], filters[4])

    def forward(self, x):
        x = self.conv0(x)
        x = self.block1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.pool(x)
        x = self.block2(x)
        x = torch.flatten(x, 1)
        return x


class SimpleCNNBranch(nn.Module):
    def __init__(self, in_channels, ksize):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels, 256, 3, padding=1)
        self.bn0 = nn.BatchNorm2d(256)
        self.conv1 = nn.Conv2d(256, 512, 1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = F.leaky_relu(self.bn0(self.conv0(x)), 0.2)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = self.pool(x)
        return torch.flatten(x, 1)


class PixelBranch(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv0 = nn.Conv1d(1, 64, 11)
        self.bn0 = nn.BatchNorm1d(64)
        self.conv1 = nn.Conv1d(64, 128, 3)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B, 1, C]
        x = F.leaky_relu(self.bn0(self.conv0(x)), 0.2)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = self.pool(x)
        return torch.flatten(x, 1)


class HSIBranch(nn.Module):
    def __init__(self, hsi_channels, px_channels, ksize, num_class):
        super().__init__()
        self.cnn_branch = SimpleCNNBranch(hsi_channels, ksize)
        self.px_branch = PixelBranch(px_channels)

        # Calculate output dimensions
        dummy = torch.randn(1, hsi_channels, ksize, ksize)
        cnn_dim = self.cnn_branch(dummy).shape[1]
        px_dim = self.px_branch(torch.randn(1, px_channels, 1)).shape[1]

        self.fc = nn.Linear(cnn_dim + px_dim, num_class)
        self.dropout = nn.Dropout(0.5)

    def forward(self, hsi, px):
        x1 = self.cnn_branch(hsi)
        x2 = self.px_branch(px)
        x = torch.cat([x1, x2], dim=1)
        x = self.dropout(x)
        return self.fc(x)


class LidarBranch(nn.Module):
    def __init__(self, in_channels, num_class, r=5):
        super().__init__()
        self.cascade = CascadeNet(in_channels, r)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(3200, num_class)  # 根据实际尺寸调整

    def forward(self, x):
        x = self.cascade(x)
        x = self.dropout(x)
        return self.fc(x)


class TBCNN(nn.Module):
    def __init__(self, hsi_dim, lidar_dim, num_class, trainable=False):
        super().__init__()
        # Remove last layers
        # self.hsi_features = nn.Sequential(*list(hsi_model.children())[:-1]
        # self.lidar_features = nn.Sequential(*list(lidar_model.children())[:-1]
        # 获取特征维度
        self.hsi_dim = hsi_dim
        self.lidar_dim = lidar_dim
        self.hsi_features = HSIBranch(hsi_channels= self.hsi_dim, px_channels= self.hsi_dim, ksize=2*5+1,num_class=num_class)
        self.lidar_features = LidarBranch(in_channels=self.lidar_dim, num_class=num_class, r=5)

        self.bn = nn.BatchNorm1d(num_class + num_class)
        self.fc1 = nn.Linear(num_class + num_class, 128)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, num_class)

    def forward(self, hsi, lidar):
        hsi = hsi.squeeze(1)
        B,C,H,W = hsi.shape
        px = torch.mean(hsi, dim=(2, 3), keepdim=True)  # 形状变为 [B, C, 1, 1]
        px= px.view(B, C, 1)
        h = self.hsi_features(hsi, px)
        l = self.lidar_features(lidar)
        x = torch.cat([h, l], dim=1)
        x = self.bn(x)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc1(x), 0.2)
        return self.fc2(x)