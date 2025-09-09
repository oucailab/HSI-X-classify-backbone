import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d


class DeformConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, groups=1, bias=False, padding=0, offset_groups=1):
        super().__init__()
        assert in_channels % groups == 0
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))

        self.param_generator = nn.Conv2d(in_channels, 2 * offset_groups * kernel_size * kernel_size, 3, 1, 1)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None

    def forward(self, x):
        offset = self.param_generator(x)
        x = deform_conv2d(x, offset=offset, weight=self.weight,
                          bias=self.bias, stride=self.stride, padding=self.padding)

        return x


if __name__ == "__main__":
    deformable_conv2d = DeformConv(in_channels=64, out_channels=64, kernel_size=3, offset_groups=1)
    print(deformable_conv2d(torch.randn(1, 64, 4, 4)).shape)
