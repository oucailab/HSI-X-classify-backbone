import torch
import torch.nn as nn

# --- 步骤 1: 定义可重用的多尺度卷积模块 ---
class MultiScaleConvBlock(nn.Module):
    """
    一个并行的多尺度卷积模块，灵感来源于Inception。
    它包含1x1, 3x3, 5x5, 7x7四种卷积路径，并将它们的输出拼接。
    """
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConvBlock, self).__init__()
        # 确保输出通道数可以被4整除，以便平均分配给四个路径
        if out_channels % 4 != 0:
            raise ValueError("out_channels 必须能被 4 整除！")
        
        path_out_channels = out_channels // 4

        # 路径 1: 1x1 卷积
        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels, path_out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(path_out_channels),
            nn.ReLU(inplace=True)
        )

        # 路径 2: 3x3 卷积
        self.path2 = nn.Sequential(
            nn.Conv2d(in_channels, path_out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(path_out_channels),
            nn.ReLU(inplace=True)
        )

        # 路径 3: 5x5 卷积
        self.path3 = nn.Sequential(
            nn.Conv2d(in_channels, path_out_channels, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(path_out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 路径 4: 7x7 卷积
        self.path4 = nn.Sequential(
            nn.Conv2d(in_channels, path_out_channels, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(path_out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 分别通过四个路径
        out1 = self.path1(x)
        out2 = self.path2(x)
        out3 = self.path3(x)
        out4 = self.path4(x)
        
        # 沿通道维度拼接所有路径的输出
        # [B, C/4, H, W] * 4 -> [B, C, H, W]
        return torch.cat([out1, out2, out3, out4], dim=1)


# --- 步骤 2: 更新主网络以使用新的多尺度模块 ---
class CNNet(nn.Module):
    def __init__(self, c1, c2, branch_out_channels=128, num_classes=10):
        """
        Args:
            c1 (int): 输入1的通道数
            c2 (int): 输入2的通道数
            branch_out_channels (int): 每个分支最终输出的总通道数 (必须能被4整除)
            num_classes (int): 最终分类的类别数
        """
        super(CNNet, self).__init__()

        # 分支1: 处理第一个输入，使用多尺度模块
        self.branch1 = MultiScaleConvBlock(in_channels=c1, out_channels=branch_out_channels)

        # 分支2: 处理第二个输入，使用多尺度模块
        self.branch2 = MultiScaleConvBlock(in_channels=c2, out_channels=branch_out_channels)

        # 展平层
        self.flatten = nn.Flatten()

        # 计算全连接层的输入维度 (与之前相同)
        fc_input_features = (branch_out_channels + branch_out_channels) * 11 * 11

        # 全连接分类器
        self.classifier = nn.Linear(in_features=fc_input_features, out_features=num_classes)

    def forward(self, x1, x2):
        """
        Args:
            x1 (torch.Tensor): 第一个输入, 尺寸 [B, C1, 11, 11]
            x2 (torch.Tensor): 第二个输入, 尺寸 [B, C2, 11, 11]
        
        Returns:
            torch.Tensor: 拼接后的特征图, 尺寸 [B, 2*branch_out_channels, 11, 11]
            torch.Tensor: 分类logits, 尺寸 [B, num_classes]
        """
        # 1. 通过各自的多尺度分支进行特征提取
        features1 = self.branch1(x1)  # -> [B, branch_out_channels, 11, 11]
        features2 = self.branch2(x2)  # -> [B, branch_out_channels, 11, 11]

        # 2. 沿通道维度拼接特征
        combined_features = torch.cat([features1, features2], dim=1) # -> [B, 2*branch_out_channels, 11, 11]

        # 3. 展平特征图
        flattened_features = self.flatten(combined_features) # -> [B, 2*branch_out_channels*11*11]

        # 4. 通过全连接层进行分类
        output = self.classifier(flattened_features) # -> [B, num_classes]

        # 按照你原始代码的返回格式
        return combined_features, output