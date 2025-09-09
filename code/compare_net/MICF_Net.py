import torch
from einops import rearrange
import torch.nn.functional as F
from torch import nn
import math


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, l, **kwargs):
        x_, l_, att = self.fn(x, l, **kwargs)
        return x + x_, l + l_, att


class Layernorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, l, **kwargs):
        return self.fn(self.norm1(x), self.norm2(l), **kwargs)


class Feedforward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        self.net2 = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, l):
        return self.net1(x), self.net2(l), None


class Atten(nn.Module):
    def __init__(self, num_atte, heads) -> None:
        super().__init__()

        self.heads = heads
        self.layers = nn.ModuleList([])

        if num_atte != 1:
            for i in range(heads):
                self.layers.append(
                    nn.Sequential(
                        nn.Conv2d(2, num_atte, 1),
                        nn.BatchNorm2d(num_atte),
                        nn.LeakyReLU(),
                        nn.Conv2d(num_atte, 1, 1)
                    ))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i == 0:
                out = layer(x[:, i])
            else:
                out = torch.cat([out, layer(x[:, i])], 1)
        return out


def cc(img1, img2):
    N, C, _, _ = img1.shape

    KLloss = torch.nn.KLDivLoss(reduction="batchmean")
    img1 = img1.reshape(N, -1)
    img2 = img2.reshape(N, -1)
    img1 = F.log_softmax(img1, dim=1)
    img2 = F.softmax(img2, dim=1)
    return KLloss(img1, img2)


class Encoder(nn.Module):
    def __init__(self, dim, head_dim, heads, num_atte, dropout=0.1):
        super().__init__()

        self.dim = dim

        self.scale = (head_dim / heads) ** -0.5  # 1/sqrt(dim)
        self.heads = heads

        self.to_qkv = nn.Linear(dim, 3 * head_dim)
        self.to_qkv1 = nn.Linear(dim, 3 * head_dim)

        self.to_cls_token = nn.Identity()

        self.mlp = nn.Linear(dim, dim)
        self.mlp1 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.atte = Atten(num_atte, heads)

    def forward(self, x, l, mask):
        b, n, _, h = *x.shape, self.heads
        p_size = int(math.sqrt(n - 1))

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)  # gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h),
                      [q, k, v])  # split into multi head attentions
        dots = (torch.einsum('bhid,bhjd->bhij', q, k) * self.scale)

        q1, k1, v1 = self.to_qkv1(l).chunk(3, dim=-1)  # gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
        q1, k1, v1 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h),
                         [q1, k1, v1])  # split into multi head attentions
        dots1 = (torch.einsum('bhid,bhjd->bhij', q1, k1) * self.scale)

        sup = torch.stack([dots, dots1], 2)
        sup = self.atte(sup)

        dots = (dots + sup)
        dots1 = (dots1 + sup)

        att_loss = cc(dots, dots1)

        attn = dots.softmax(dim=-1)  # follow the softmax,q,d,v equation in the paper
        attn1 = dots1.softmax(dim=-1)  # follow the softmax,q,d,v equation in the paper

        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # product of v times whatever inside softmax
        out1 = torch.einsum('bhij,bhjd->bhid', attn1, v1)  # product of v times whatever inside softmax

        out = rearrange(out, 'b h n d -> b n (h d)')  # concat heads into one matrix, ready for next encoder block
        out1 = rearrange(out1, 'b h n d -> b n (h d)')  # concat heads into one matrix, ready for next encoder block

        out = self.mlp(out)
        out1 = self.mlp1(out1)

        out = self.dropout(out)
        out1 = self.dropout1(out1)

        return out, out1, att_loss


class Transformer_(nn.Module):
    def __init__(self, dim=64, hidden_dim=8, head_dim=64, heads=8, num_atte=8, depth=4, dropout=0.1):
        super().__init__()

        self.depth = depth
        self.layers = nn.ModuleList([])

        for i in range(int(depth)):
            self.layers.append(nn.ModuleList([
                Residual(Layernorm(dim, Encoder(dim, head_dim, heads, num_atte, dropout))),
                Residual(Layernorm(dim, Feedforward(dim, hidden_dim, dropout)))
            ]))

    def forward(self, x, l, mask=None):
        att_loss = 0
        for i, (attention, mlp) in enumerate(self.layers):
            x, l, att_loss_ = attention(x, l, mask=mask)  # go to attention
            x, l, _ = mlp(x, l)  # go to MLP_Block
            att_loss += att_loss_

        return x, l, att_loss


def modu(score1,score2,label,alpha=0.1):
    tanh = nn.Tanh()
    softmax = nn.Softmax(dim=-1)
    
    score_1 = sum([softmax(score1)[i][label[i]] for i in range(label.size(0))])
    score_2 = sum([softmax(score2)[i][label[i]] for i in range(label.size(0))])

    ratio_1 = score_1 / score_2
    ratio_2 = 1 / ratio_1
    if ratio_1 > 1:
        coeff_1 = 1-tanh(alpha * ratio_1).detach()
        coeff_2 = 1
    else:
        coeff_2 = 1-tanh(alpha * ratio_2).detach()
        coeff_1 = 1
    return coeff_1,coeff_2


class Fuse(nn.Module):
    def __init__(self, dim, num_heads, num_classes, dropout):
        super(Fuse, self).__init__()
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.scale = (dim / num_heads) ** -0.5  # 1/sqrt(dim)

        self.q = nn.Linear(num_classes, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

        self.q1 = nn.Linear(num_classes, dim)
        self.k1 = nn.Linear(dim, dim)
        self.v1 = nn.Linear(dim, dim)
        self.output = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, l, score1, score2):
        h = self.num_heads

        q = self.q(score2)[:, None].repeat(1, x.shape[1], 1)
        k = self.k(x)
        v = self.v(x)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), [q, k, v])

        dots = (torch.einsum('bhid,bhjd->bhij', q, k) * self.scale)
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        q1 = self.q1(score1)[:, None].repeat(1, x.shape[1], 1)  # [1, batch_size, hidden_size]
        k1 = self.k1(l)  # [1, batch_size, hidden_size]
        v1 = self.v1(l)  # [1, batch_size, hidden_size]

        q1, k1, v1 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), [q1, k1, v1])

        dots1 = (torch.einsum('bhid,bhjd->bhij', q1, k1) * self.scale)
        attn1 = dots1.softmax(dim=-1)
        out1 = torch.einsum('bhij,bhjd->bhid', attn1, v1)
        out1 = rearrange(out1, 'b h n d -> b n (h d)')

        out = torch.cat((out + x, out1 + l), dim=-1)

        return out


class MICF_Net(nn.Module):
    def __init__(self, patch_size=9, dim=64, input_dim=144, num_classes=15, dep=2,heads=8, num_atte=8, mlp_dim=8, alpha=0.1, dropout=0.1, emb_dropout=0.1):
        super(MICF_Net, self).__init__()

        self.dim = dim
        self.alpha = alpha
        
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(1,8,(5, 1, 1),1),
            nn.BatchNorm3d(8),
            nn.LeakyReLU(),
        )
        self.conv2d_features = nn.Sequential(
            nn.Conv2d(8*(input_dim-4), dim, 1,1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(),
        )
        self.conv2d_lidar = nn.Sequential(
            nn.Conv2d(4, 32, 1, 1),     #输入的lidar的channel
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
        )
        self.conv2d_lidar2 = nn.Sequential(
            nn.Conv2d(32, dim, 1,1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(),
        )

        self.dropout1 = nn.Dropout(emb_dropout)
        self.dropout2 = nn.Dropout(emb_dropout)

        self.pos_embedding1 = nn.Parameter(torch.empty(1, 1+patch_size**2, dim))
        torch.nn.init.normal_(self.pos_embedding1, std=.02)
        self.pos_embedding2 = nn.Parameter(torch.empty(1, 1+patch_size**2, dim))
        torch.nn.init.normal_(self.pos_embedding2, std=.02)
        

        self.transformer1 = Transformer_(dim, mlp_dim, dim, heads, num_atte, dep, dropout)
        
        self.fuse = Fuse(dim,heads,num_classes,dropout)  ########语义引导的融合
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))

        self.nn = nn.Linear(2*dim, num_classes)
        self.to_cls_token = nn.Identity()

    def forward(self, x, lidar, x_proto, l_proto, label, mask=None):

        x = self.conv3d_features(x)
        x = rearrange(x, 'b c h w y -> b (c h) w y')
        x = self.conv2d_features(x)
        x = rearrange(x,'b c h w -> b (h w) c')
        
        lidar = self.conv2d_lidar(lidar)
        lidar = self.conv2d_lidar2(lidar)
        
        lidar = rearrange(lidar,'b c h w -> b (h w) c')

        cls_tokens1 = self.cls_token.expand(x.shape[0],-1,-1)
        x = torch.cat([cls_tokens1,x],1)
        cls_tokens2 = self.cls_token.expand(lidar.shape[0],-1,-1)
        lidar = torch.cat([cls_tokens2,lidar],1)
        
        x = x+self.pos_embedding1
        lidar = lidar+self.pos_embedding2

        x = self.dropout1(x)
        lidar = self.dropout2(lidar)

        x, lidar, att_loss = self.transformer1(x, lidar, mask)  

        
        ############# 预分类
        cls1 = self.to_cls_token(x[:,0])
        cls2 = self.to_cls_token(lidar[:,0])
        

        score1 = (-torch.cdist(cls1, x_proto)).softmax(-1)
        score2 = (-torch.cdist(cls2, l_proto)).softmax(-1)

        score = torch.stack([score1,score2],1)
        score_ = score.mean(1)

        ##################################
        b,n,d = x.shape
        
        x_masked = x.clone()
        lidar_masked = lidar.clone()
        if label is not None:
            pro1,pro2 = modu(score1,score2,label,self.alpha)
            
            mask1 = (torch.rand(d).cuda() <= pro1)
            mask2 = (torch.rand(d).cuda() <= pro2)

            x_masked[:,:,(mask1 == 0)] *= pro1
            lidar_masked[:,:,(mask2 == 0)] *= pro2
                
                
        x, lidar = x_masked, lidar_masked
        fuse = self.fuse(x,lidar,score1,score2)
        fuse = self.to_cls_token(fuse[:,0])

        fuse = self.nn(fuse)

        return fuse, att_loss, (cls1, cls2)
        

    

