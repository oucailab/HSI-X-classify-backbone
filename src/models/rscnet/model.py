import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalFusion(nn.Module):
    def __init__(self, channels=64, r=4):
        super(CrossModalFusion, self).__init__()
        inter_channels = int(channels // r)

        self.conv_3d = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(1),
            nn.ReLU(inplace=True)
        )

        self.conv_2d = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.cross_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2 * channels, inter_channels, 1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, 2, 1),
            nn.Softmax(dim=1)
        )

        self.agg_conv = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, 1),
            nn.BatchNorm2d(inter_channels), nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, 2 * channels, 1)
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, 1),
            nn.BatchNorm2d(inter_channels), nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, 2 * channels, 1)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, fh, fx):
        B, C, H, W = fh.shape

        fh_prime = self.conv_3d(fh.unsqueeze(1)).squeeze(1)
        fx_prime = self.conv_2d(fx)

        gates = self.cross_gate(torch.cat([fh_prime, fx_prime], dim=1))
        w_h, w_x = torch.split(gates, 1, dim=1)
        fh_s1 = fh_prime + (fx_prime * w_x)
        fx_s1 = fx_prime + (fh_prime * w_h)

        u = self.agg_conv(torch.cat([fh_s1, fx_s1], dim=1))
        logits = self.local_att(u) * self.global_att(u)
        attn = self.softmax(logits.view(B, 2, C, H, W))
        return (fh_s1 * attn[:, 0]) + (fx_s1 * attn[:, 1])


class KeyBandSelectionBlock(nn.Module):
    def __init__(self, hsi_channels, fused_dim, topk_ratio=0.5):
        super(KeyBandSelectionBlock, self).__init__()

        self.k = int(hsi_channels * topk_ratio)
        self.k = max(1, self.k)

        self.linear = nn.Linear(fused_dim, 1)
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(hsi_channels, hsi_channels // 4), nn.ReLU(inplace=True),
            nn.Linear(hsi_channels // 4, hsi_channels), nn.Sigmoid()
        )

    def forward(self, hsi_feats, fused_feats):
        B, C, H, W = hsi_feats.shape
        _, K, _, _ = fused_feats.shape

        hsi_flat = hsi_feats.view(B, C, -1)
        fused_flat = fused_feats.view(B, K, -1).permute(0, 2, 1)
        attn_map = torch.bmm(hsi_flat, fused_flat)
        attn_vec = self.linear(attn_map).squeeze(-1)
        scores = self.mlp(hsi_feats) * attn_vec

        _, topk_idx = torch.topk(scores, k=self.k, dim=1)

        idx_exp = topk_idx.view(B, self.k, 1, 1).expand(-1, -1, H, W)
        return torch.gather(hsi_feats, 1, idx_exp), topk_idx


class SimpleEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class DepthwiseOnlyEncoder(nn.Module):
    def __init__(self, channels):
        super(DepthwiseOnlyEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class FFN(nn.Module):
    def __init__(self, channels):
        super(FFN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return x + self.net(x)


class RSCM(nn.Module):
    def __init__(self, hsi_dim, fused_dim, topk_ratio=0.5):
        super(RSCM, self).__init__()

        self.kbsb = KeyBandSelectionBlock(hsi_dim, fused_dim, topk_ratio=topk_ratio)

        actual_k = self.kbsb.k

        self.align_conv = nn.Conv2d(actual_k, fused_dim, kernel_size=1)

        self.cmaf = CrossModalFusion(channels=fused_dim)

        self.ffn = FFN(fused_dim)

    def forward(self, hsi_feats, fused_feats):
        selected_bands, topk_idx = self.kbsb(hsi_feats, fused_feats)
        aligned_bands = self.align_conv(selected_bands)

        fused_out = self.cmaf(fh=aligned_bands, fx=fused_feats)
        out = self.ffn(fused_out)

        return out, selected_bands, topk_idx


class RSCNet(nn.Module):
    def __init__(self,
                 hsi_channels,
                 pca_channels,
                 aux_channels,
                 num_classes,
                 embed_dim=144,
                 topk_ratio=0.2,
                 num_rscm_layers=4
                 ):
        super(RSCNet, self).__init__()

        self.hsi_encoder = DepthwiseOnlyEncoder(hsi_channels)
        self.pca_encoder = SimpleEncoder(pca_channels, embed_dim)
        self.aux_encoder = SimpleEncoder(aux_channels, embed_dim)
        self.initial_fusion = CrossModalFusion(channels=embed_dim)
        self.initial_ffn = FFN(embed_dim)

        self.rscm_layers = nn.ModuleList([
            RSCM(hsi_dim=hsi_channels, fused_dim=embed_dim, topk_ratio=topk_ratio)
            for _ in range(num_rscm_layers)
        ])

        final_k = int(hsi_channels * topk_ratio)
        final_k = max(1, final_k)

        self.final_align = nn.Conv2d(final_k, embed_dim, 1)

        self.final_fusion = CrossModalFusion(channels=embed_dim)

        self.refinement = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x_hsi, x_pca, x_aux):
        feat_hsi = self.hsi_encoder(x_hsi)
        feat_pca = self.pca_encoder(x_pca)
        feat_aux = self.aux_encoder(x_aux)

        curr_fused = self.initial_fusion(fh=feat_pca, fx=feat_aux)
        curr_fused = self.initial_ffn(curr_fused)

        last_selected_bands = None
        last_topk_idx = None
        for rscm in self.rscm_layers:
            curr_fused, last_selected_bands, last_topk_idx = rscm(feat_hsi, curr_fused)

        feat_bands_aligned = self.final_align(last_selected_bands)
        final_feat = self.final_fusion(fh=feat_bands_aligned, fx=curr_fused)

        logits = self.refinement(final_feat)

        return logits, last_topk_idx
