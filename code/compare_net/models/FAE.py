import torch
import torch.nn as nn
import numbers
from einops import rearrange

from compare_net.models.DeformConv import DeformConv


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class DeformAttn(nn.Module):
    def __init__(self, dim, patch_size, bias):
        super(DeformAttn, self).__init__()
        self.patch_size = patch_size
        self.norm = LayerNorm(dim * 2)

        self.to_hidden = DeformConv(in_channels=dim, out_channels=dim * 6, kernel_size=1, bias=bias)
        self.to_hidden_dw = DeformConv(in_channels=dim * 6, out_channels=dim * 6, kernel_size=3, stride=1,
                                       padding=1, groups=dim * 6, bias=bias)

        self.project_out = DeformConv(in_channels=dim * 2, out_channels=dim, kernel_size=1, bias=bias)

    def forward(self, x):
        hidden = self.to_hidden(x)

        q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)

        q_patch = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        k_patch = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        q_fft = torch.fft.rfft2(q_patch.float())
        k_fft = torch.fft.rfft2(k_patch.float())

        out = q_fft * k_fft
        out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))
        out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                        patch2=self.patch_size)

        out = self.norm(out)

        output = v * out
        output = self.project_out(output)

        return output


class TransformerBlock(nn.Module):
    def __init__(self, dim, patch_size, bias=False):
        super(TransformerBlock, self).__init__()
        self.norm = LayerNorm(dim)
        self.attn = DeformAttn(dim, patch_size, bias)

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x


class FAE(nn.Module):
    def __init__(self, in_channels, patch_size, depth=4, bias=False):
        super(FAE, self).__init__()
        self.fae = nn.Sequential(*[
            TransformerBlock(dim=in_channels, patch_size=patch_size, bias=bias) for _ in range(depth)])

    def forward(self, x):
        out = self.fae(x)

        return out
