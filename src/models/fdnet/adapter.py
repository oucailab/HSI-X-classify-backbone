from .model import FDNet
import torch.nn.functional as F


MODEL_NAME = "FDNet"


def build_model(config, dataset_type, device):
    in_channels = config.get_value('channels')
    lidar_or_sar_channels = config.get_value('lidar_or_sar_channels')[dataset_type]
    patch_size = config.get_value('windowSize')
    num_classes = config.get_value('out_features')[dataset_type]
    net = FDNet(
        l1=in_channels,
        l2=lidar_or_sar_channels,
        patch_size=patch_size,
        num_classes=num_classes,
        wavename='db2',
        attn_kernel_size=9,
        coefficient_hsi=0.7,
        fae_embed_dim=64,
        fae_depth=1,
    )
    return {"net": net, "patch_size": patch_size}


def _crop_to_even_spatial(x):
    height, width = x.shape[-2], x.shape[-1]
    if height % 2 == 1:
        x = x[..., :-1, :]
    if width % 2 == 1:
        x = x[..., :, :-1]
    return x


def _prepare_inputs(bundle, batch):
    hsi_pca = batch["hsi_pca"]
    aux = batch["aux"]
    target_size = bundle["patch_size"]

    if hsi_pca.shape[-1] != target_size or hsi_pca.shape[-2] != target_size:
        hsi_pca = F.interpolate(hsi_pca, size=(target_size, target_size), mode="bilinear", align_corners=False)
    if aux.shape[-1] != target_size or aux.shape[-2] != target_size:
        interp_mode = "bilinear" if aux.dim() == 4 else "trilinear"
        aux = F.interpolate(aux, size=(target_size, target_size), mode=interp_mode, align_corners=False)

    hsi_pca = _crop_to_even_spatial(hsi_pca)
    aux = _crop_to_even_spatial(aux)
    return hsi_pca, aux


def forward_train(bundle, batch):
    hsi_pca, aux = _prepare_inputs(bundle, batch)
    return bundle["net"](hsi_pca, aux)


def forward_eval(bundle, batch):
    hsi_pca, aux = _prepare_inputs(bundle, batch)
    return bundle["net"](hsi_pca, aux)
