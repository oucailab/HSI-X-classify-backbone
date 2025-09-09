import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn, einsum

class MCGF(nn.Module):
    def __init__(self, dim, heads=8, dim_head=8, dropout=0.1):
        super(MCGF, self).__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x1, x2, kv_include_self=False):
        B,C,H,W = x1.shape
        x1 = x1.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x2 = x2.permute(0, 2, 3, 1).reshape(B, H * W, C)
        b, n, _, h = *x1.shape, self.heads
        q = self.to_q(x1)
        k = self.to_k(x2)
        v = self.to_v(x2)

        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        # print(x1.shape)
        # print(out.shape)
        out = x1 + out
        f_q = self.to_q(out)
        f_k = self.to_k(out)
        f_v = self.to_v(x1)

        f_q = rearrange(f_q, 'b n (h d) -> b h n d', h=h)
        f_k = rearrange(f_k, 'b n (h d) -> b h n d', h=h)
        f_v = rearrange(f_v, 'b n (h d) -> b h n d', h=h)
        dots = einsum('b h i d, b h j d -> b h i j', f_q, f_k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, f_v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = out.permute(0, 2, 1).reshape(B, C, H, W)
        # return self.to_out(out)
        return out