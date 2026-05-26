from .model import HCTnet


MODEL_NAME = "HCTNet"


def build_model(config, dataset_type, device):
    lidar_or_sar_channels = config.get_value('lidar_or_sar_channels')[dataset_type]
    num_classes = config.get_value('out_features')[dataset_type]
    net = HCTnet(in_channels=lidar_or_sar_channels, num_classes=num_classes)
    return {"net": net}


def forward_train(bundle, batch):
    return bundle["net"](batch["hsi_pca"], batch["aux"])


def forward_eval(bundle, batch):
    return bundle["net"](batch["hsi_pca"], batch["aux"])
