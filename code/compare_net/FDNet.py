import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


from compare_net.models.FAE import FAE
from compare_net.models.DWT_2D import DWT_2D
from compare_net.models.DWT_3D import DWT_3D


def seq2img(x):
    [b, c, d] = x.shape
    p = int(d ** .5)
    x = x.reshape((b, c, p, p))
    return x


def frequency_embedding(c, h, w):
    position_embedding = torch.zeros(1, c, h * w)
    for j in range(h * w):
        for k in range(c):
            position_embedding[:, k, j] = torch.sin(torch.tensor(j) / (10000 ** (torch.tensor(k) / c)))
    return nn.Parameter(position_embedding)


class Spatial_Attn_2d(nn.Module):
    def __init__(self, kernel_size=7):
        super(Spatial_Attn_2d, self).__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], dim=1)
        y = self.attn(y)

        return x * y.expand_as(x)


class Spatial_Spectral_Attn_3d(nn.Module):
    def __init__(self, kernel_size=7):
        super(Spatial_Spectral_Attn_3d, self).__init__()
        self.attn = nn.Sequential(
            nn.Conv3d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm3d(1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], dim=1)
        y = self.attn(y)

        return x * y.expand_as(x)


class LiDAR_Encoder(nn.Module):
    def __init__(self, wavename, in_channels=1, out_channels=64, attn_kernel_size=7):
        super(LiDAR_Encoder, self).__init__()
        self.DWT_layer_2D = DWT_2D(wavename=wavename)

        # 2d cnn for x_ll
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.S_attn = Spatial_Attn_2d(kernel_size=attn_kernel_size)

        # 2d cnn for high components
        self.conv_high = nn.Sequential(
            nn.Conv2d(in_channels=in_channels * 3, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        # 2d cnn for all components
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, lidar_img):
        x_dwt = self.DWT_layer_2D(lidar_img)

        # x_ll -> Conv2d layer & spatial attn
        x_ll = x_dwt[0]
        x_ll = self.conv1(x_ll)
        x_ll = self.conv2(x_ll)
        x_ll = self.S_attn(x_ll)

        # high frequency component processing
        x_high = torch.cat([x for x in x_dwt[1:4]], dim=1)
        x_high = self.conv_high(x_high)

        x = torch.cat([x_ll, x_high], dim=1)
        x = self.conv2d(x)
        return x


class HSI_Encoder_3D(nn.Module):
    def __init__(self, in_depth, patch_size, wavename,
                 in_channels_3d=1, out_channels_3d=16, out_channels_2d=64, attn_kernel_size=7):
        super(HSI_Encoder_3D, self).__init__()
        self.in_depth = in_depth
        self.patch_size = patch_size

        # DWT 3d
        self.DWT_layer_3D = DWT_3D(wavename=wavename)

        # 3d cnn for x_lll
        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels_3d, out_channels=out_channels_3d // 2, kernel_size=(3, 3, 3),
                      stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(out_channels_3d // 2),
            nn.ReLU(),
        )

        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(in_channels=out_channels_3d // 2, out_channels=out_channels_3d, kernel_size=(3, 3, 3),
                      stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(out_channels_3d),
            nn.ReLU(),
        )

        self.SS_attn = Spatial_Spectral_Attn_3d(kernel_size=attn_kernel_size)

        # 3d cnn for high components
        self.conv3d_high = nn.Sequential(
            nn.Conv3d(in_channels=in_channels_3d * 7, out_channels=out_channels_3d, kernel_size=1),
            nn.BatchNorm3d(out_channels_3d),
            nn.ReLU(),
        )

        # 2d cnn for all components
        self.in_channels_2d = int(self.get_inchannels_2d())
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels_2d, out_channels=out_channels_2d, kernel_size=1),
            nn.BatchNorm2d(out_channels_2d),
            nn.ReLU(),
        )

    def get_inchannels_2d(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.in_depth // 2, self.patch_size // 2, self.patch_size // 2))
            x = self.conv3d_1(x)
            x = self.conv3d_2(x)

            x = torch.cat([x, x], dim=1)
            _, t, c, _, _ = x.size()
        return t * c

    def forward(self, hsi_img):
        # DWT 3d
        #hsi_img = hsi_img.unsqueeze(1)
        x_dwt = self.DWT_layer_3D(hsi_img.permute(0, 1, 3, 2, 4))

        # 3d cnn
        x_lll = x_dwt[0].permute(0, 1, 3, 2, 4)
        x_lll = self.conv3d_1(x_lll)
        x_lll = self.conv3d_2(x_lll)
        x_lll = self.SS_attn(x_lll)

        # high frequency components processing
        x_high = torch.cat([x.permute(0, 1, 3, 2, 4) for x in x_dwt[1:8]], dim=1)
        x_high = self.conv3d_high(x_high)
        x = torch.cat([x_lll, x_high], dim=1)

        # 2d cnn
        x = rearrange(x, 'b c d h w ->b (c d) h w')
        x = self.conv2d(x)

        return x


class HSI_Encoder_2D(nn.Module):
    def __init__(self, wavename, in_channels, out_channels=64, attn_kernel_size=7):
        super(HSI_Encoder_2D, self).__init__()
        self.DWT_layer_2D = DWT_2D(wavename=wavename)

        # 2d cnn for x_ll
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.S_attn = Spatial_Attn_2d(kernel_size=attn_kernel_size)

        # high frequency components processing
        self.conv_high = nn.Sequential(
            nn.Conv2d(in_channels=in_channels * 3, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        # 2d cnn for all components
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, hsi_img):

        x_dwt = self.DWT_layer_2D(hsi_img.squeeze(1))
        # x_ll -> ch_select
        x_ll = x_dwt[0]
        x_ll = self.conv1(x_ll)
        x_ll = self.conv2(x_ll)
        x_ll = self.S_attn(x_ll)

        # high frequency component processing
        x_high = torch.cat([x for x in x_dwt[1:4]], dim=1)
        x_high = self.conv_high(x_high)

        x = torch.cat([x_ll, x_high], dim=1)
        x = self.conv2d(x)
        return x


class Classifier(nn.Module):
    def __init__(self, Classes, cls_embed_dim):
        super(Classifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=cls_embed_dim, out_channels=32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.linear = nn.Linear(in_features=32, out_features=Classes)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x_out = F.softmax(x, dim=1)

        return x_out


class FDNet(nn.Module):
    def __init__(self, l1, l2, patch_size, num_classes,
                 wavename, attn_kernel_size, coefficient_hsi,
                 fae_embed_dim, fae_depth):
        super().__init__()
        self.weight_hsi = torch.nn.Parameter(torch.Tensor([coefficient_hsi]))
        self.weight_lidar = torch.nn.Parameter(torch.Tensor([1 - coefficient_hsi]))

        self.hsi_encoder_3d = HSI_Encoder_3D(in_depth=l1, patch_size=patch_size,
                                             wavename=wavename, out_channels_2d=fae_embed_dim,
                                             attn_kernel_size=attn_kernel_size)
        self.hsi_encoder_2d = HSI_Encoder_2D(wavename=wavename, in_channels=l1, out_channels=fae_embed_dim,
                                             attn_kernel_size=attn_kernel_size)
        self.lidar_encoder = LiDAR_Encoder(wavename=wavename, in_channels=l2, out_channels=fae_embed_dim,
                                           attn_kernel_size=attn_kernel_size)

        self.pos_embed = nn.Parameter(torch.randn(1, fae_embed_dim, (patch_size // 2) ** 2))
        self.freq_embed = frequency_embedding(c=fae_embed_dim, h=patch_size // 2, w=patch_size // 2)
        self.fae = FAE(in_channels=fae_embed_dim, patch_size=patch_size // 2, depth=fae_depth)

        self.classifier = Classifier(Classes=num_classes, cls_embed_dim=fae_embed_dim)

    def forward(self, img_hsi, img_lidar):
        # lidar encoder
        x_lidar = self.lidar_encoder(img_lidar)

        # hsi encoder
        x_hsi_3d = self.hsi_encoder_3d(img_hsi)
        x_hsi_2d = self.hsi_encoder_2d(img_hsi)
        x_hsi = x_hsi_3d + x_hsi_2d

        x_cnn = self.weight_hsi * x_hsi + self.weight_lidar * x_lidar
        x = x_cnn.flatten(2)
        x = x + self.pos_embed[:, :, :]
        x = x + self.freq_embed[:, :, :]
        x = seq2img(x)
        x_vit = self.fae(x)

        x_cls = self.classifier(x_vit)
        return x_cls
