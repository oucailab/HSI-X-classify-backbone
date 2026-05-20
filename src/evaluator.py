from torch.backends import cudnn
from torch.utils.data import DataLoader, Subset

from .data import *
from .metrics import *
from .analysis import *

from . import config

import os
from .tsne import t_sne, t_sne_full
from .model_registry import get_model_adapter

os.environ['KMP_DUPLICATE_LIB_OK']='True'


def _limit_dataloader_samples(loader, max_samples):
    if max_samples is None:
        return loader
    subset_size = min(len(loader.dataset), max_samples)
    subset = Subset(loader.dataset, range(subset_size))
    return DataLoader(
        subset,
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=loader.num_workers,
        drop_last=False,
        pin_memory=loader.pin_memory,
    )


def myTest(datasetType, model, max_test_samples=None):
    """测试函数，从配置系统获取参数"""
    cuda = config.get_value('cuda')
    device = torch.device(cuda if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    channels = config.get_value('channels')
    windowSize = config.get_value('windowSize')
    batch_size = config.get_value('batch_size')
    num_workers = config.get_value('num_workers')
    random_seed = config.get_value('random_seed')
    visualization = config.get_value('visualization')
    tsne = config.get_value('tsne')

    try:
        erf = config.get_value('erf')
    except:
        erf = False

    model_savepath = config.get_value('model_savepath')
    report_path = config.get_value('report_path')
    image_path = config.get_value('image_path')

    print(f"测试参数: cuda={cuda}, visualization={visualization}, tsne={tsne}")

    net = torch.load(model_savepath[datasetType])
    model_adapter = get_model_adapter(model)
    model_bundle = {"net": net}
    set_random_seed(random_seed)
    train_loader, test_loader, trntst_loader, all_loader, hsi_pca_wight = getMyData(datasetType, channels, windowSize, batch_size, num_workers)
    test_loader = _limit_dataloader_samples(test_loader, max_test_samples)

    getMyReport(datasetType, model_bundle, model_adapter, test_loader, report_path[datasetType], device, model)

    if tsne == True:
        t_sne_full(net, test_loader, datasetType)

    if visualization:
        if datasetType == 0 or datasetType == 1 or datasetType == 5 or datasetType == 6 or datasetType == 7:
            getMyVisualization(datasetType, model_bundle, model_adapter, all_loader, image_path[datasetType], device, model)
        else:
            getMyVisualization(datasetType, model_bundle, model_adapter, trntst_loader, image_path[datasetType], device, model)
