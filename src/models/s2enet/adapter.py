from .model import S2ENet


MODEL_NAME = "S2ENet"


def build_model(config, dataset_type, device):
    in_channels = config.get_value('channels')
    lidar_or_sar_channels = config.get_value('lidar_or_sar_channels')[dataset_type]
    patch_size = config.get_value('windowSize')
    num_classes = config.get_value('out_features')[dataset_type]
    net = S2ENet(
        input_channels=in_channels,
        input_channels2=lidar_or_sar_channels,
        n_classes=num_classes,
        patch_size=patch_size,
    )
    return {"net": net}


def forward_train(bundle, batch):
    return bundle["net"](batch["hsi_pca"].squeeze(1), batch["aux"])


def forward_eval(bundle, batch):
    return bundle["net"](batch["hsi_pca"].squeeze(1), batch["aux"])
