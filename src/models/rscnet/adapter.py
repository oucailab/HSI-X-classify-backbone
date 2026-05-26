from .model import RSCNet


MODEL_NAME = "RSCNet"


def build_model(config, dataset_type, device):
    hsi_channels = config.get_value('data_channels')[dataset_type]
    pca_channels = config.get_value('channels')
    aux_channels = config.get_value('lidar_or_sar_channels')[dataset_type]
    num_classes = config.get_value('out_features')[dataset_type]
    net = RSCNet(
        hsi_channels=hsi_channels,
        pca_channels=pca_channels,
        aux_channels=aux_channels,
        num_classes=num_classes,
    )
    return {"net": net}


def forward_train(bundle, batch):
    logits, _ = bundle["net"](
        batch["hsi"].squeeze(1),
        batch["hsi_pca"].squeeze(1),
        batch["aux"],
    )
    return logits


def forward_eval(bundle, batch):
    logits, _ = bundle["net"](
        batch["hsi"].squeeze(1),
        batch["hsi_pca"].squeeze(1),
        batch["aux"],
    )
    return logits
