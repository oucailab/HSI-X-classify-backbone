from .model import Net, Bottleneck


MODEL_NAME = "AsyFFNet"


def build_model(config, dataset_type, device):
    in_channels = config.get_value('channels')
    lidar_or_sar_channels = config.get_value('lidar_or_sar_channels')[dataset_type]
    num_classes = config.get_value('out_features')[dataset_type]
    net = Net(
        hsi_channels=in_channels,
        sar_channels=lidar_or_sar_channels,
        hidden_size=128,
        block=Bottleneck,
        num_parallel=2,
        num_reslayer=4,
        num_classes=num_classes,
    )
    return {"net": net}


def forward_train(bundle, batch):
    return bundle["net"](batch["hsi_pca"].squeeze(1), batch["aux"])


def forward_eval(bundle, batch):
    return bundle["net"](batch["hsi_pca"].squeeze(1), batch["aux"])
