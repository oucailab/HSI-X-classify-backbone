from .model import CNNet


MODEL_NAME = "CNN"


def build_model(config, dataset_type, device):
    in_channels = config.get_value('channels')
    lidar_or_sar_channels = config.get_value('lidar_or_sar_channels')[dataset_type]
    num_classes = config.get_value('out_features')[dataset_type]
    net = CNNet(in_channels, lidar_or_sar_channels, 128, num_classes)
    return {"net": net}


def forward_train(bundle, batch):
    return bundle["net"](batch["hsi_pca"].squeeze(1), batch["aux"])


def forward_eval(bundle, batch):
    return bundle["net"](batch["hsi_pca"].squeeze(1), batch["aux"])
