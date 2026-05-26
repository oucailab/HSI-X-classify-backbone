from .model import MViT


MODEL_NAME = "ExViT"


def build_model(config, dataset_type, device):
    in_channels = config.get_value('channels')
    lidar_or_sar_channels = config.get_value('lidar_or_sar_channels')[dataset_type]
    patch_size = config.get_value('windowSize')
    num_classes = config.get_value('out_features')[dataset_type]
    net = MViT(
        patch_size=patch_size,
        num_patches=[in_channels, lidar_or_sar_channels],
        num_classes=num_classes,
        dim=64,
        depth=6,
        heads=4,
        mlp_dim=32,
        dropout=0.1,
        emb_dropout=0.1,
        mode='MViT'
    )
    return {"net": net}


def forward_train(bundle, batch):
    _, logits = bundle["net"](batch["hsi_pca"].squeeze(1), batch["aux"])
    return logits


def forward_eval(bundle, batch):
    _, logits = bundle["net"](batch["hsi_pca"].squeeze(1), batch["aux"])
    return logits
