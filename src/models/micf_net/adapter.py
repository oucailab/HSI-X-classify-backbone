import torch

from .model import MICF_Net


MODEL_NAME = "MICF_Net"


def build_model(config, dataset_type, device):
    in_channels = config.get_value('channels')
    lidar_or_sar_channels = config.get_value('lidar_or_sar_channels')[dataset_type]
    patch_size = config.get_value('windowSize')
    num_classes = config.get_value('out_features')[dataset_type]
    dim = 64
    net = MICF_Net(
        patch_size=patch_size,
        dim=dim,
        input_dim=in_channels,
        aux_channels=lidar_or_sar_channels,
        num_classes=num_classes,
        heads=8,
        num_atte=8,
        dep=2,
    )
    x_proto = torch.empty(num_classes, dim).to(device)
    torch.nn.init.normal_(x_proto, mean=0, std=0.2)
    l_proto = torch.empty(num_classes, dim).to(device)
    torch.nn.init.normal_(l_proto, mean=0, std=0.2)
    return {"net": net, "x_proto": x_proto, "l_proto": l_proto}


def forward_train(bundle, batch):
    logits, _, _ = bundle["net"](batch["hsi_pca"], batch["aux"], bundle["x_proto"], bundle["l_proto"], None)
    return logits


def forward_eval(bundle, batch):
    logits, _, _ = bundle["net"](batch["hsi_pca"], batch["aux"], bundle["x_proto"], bundle["l_proto"], None)
    return logits
