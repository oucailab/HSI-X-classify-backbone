from .model import HybridSN


MODEL_NAME = "HybridSN"


def build_model(config, dataset_type, device):
    num_classes = config.get_value('out_features')[dataset_type]
    net = HybridSN(num_classes)
    return {"net": net}


def forward_train(bundle, batch):
    return bundle["net"](batch["hsi_pca"])


def forward_eval(bundle, batch):
    return bundle["net"](batch["hsi_pca"])
