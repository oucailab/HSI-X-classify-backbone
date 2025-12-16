from torch.backends import cudnn

from dataset import *
from report import *
from visualization import *

# 使用新的配置系统
import config

import os
from tsne import t_sne, t_sne_full

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def model_forward_test(model, net, hsi_pca, hsi, sar, additional_params=None):
    """测试时的模型前向传播"""
    if model == 'FusAtNet':
        outputs = net(hsi_pca.squeeze(1), sar)
    elif model == 'ExViT':
        outputs = net(hsi_pca.squeeze(1), sar)
    elif model == 'HybridSN':
        outputs = net(hsi_pca)
    elif model == 'FDNet':
        outputs = net(hsi_pca, sar)
    elif model == 'MICF_Net':
        # MICF_Net在测试时需要特殊处理
        if additional_params is not None:
            x_proto, l_proto = additional_params
        else:
            # 如果没有提供原型，创建默认的
            num_classes = net.num_classes if hasattr(net, 'num_classes') else 7
            dim = 64
            device = next(net.parameters()).device
            x_proto = torch.empty(num_classes, dim).to(device)
            torch.nn.init.normal_(x_proto, mean=0, std=0.2)
            l_proto = torch.empty(num_classes, dim).to(device)
            torch.nn.init.normal_(l_proto, mean=0, std=0.2)
        outputs = net(hsi_pca, sar, x_proto, l_proto, None)
    elif model == 'M2FNet':
        outputs = net(hsi_pca, sar)
    elif model == 'S2ENet':
        outputs = net(hsi_pca.squeeze(1), sar)
    elif model == 'DFINet':
        outputs = net(hsi_pca.squeeze(1), sar)
    elif model == 'AsyFFNet':
        outputs = net(hsi_pca.squeeze(1), sar)
    elif model == 'HCTNet':
        outputs = net(hsi_pca, sar)
    elif model == 'MACN':
        outputs = net(hsi_pca, sar)
    elif model == 'TBCNN':
        outputs = net(hsi_pca, sar)
    elif model == 'CNN':
        outputs = net(hsi_pca.squeeze(1), sar)
    else:
        # 默认情况，尝试EyeNet的调用方式
        try:
            _, outputs = net(hsi_pca, sar)
        except:
            outputs = net(hsi_pca, sar)
    
    return outputs

def myTest(datasetType, model):
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
    
    # 尝试获取erf参数，如果不存在则设为False（兼容性）
    try:
        erf = config.get_value('erf')
    except:
        erf = False
        
    model_savepath = config.get_value('model_savepath')
    report_path = config.get_value('report_path')
    image_path = config.get_value('image_path')
    
    print(f"测试参数: cuda={cuda}, visualization={visualization}, tsne={tsne}")
    
    net = torch.load(model_savepath[datasetType])
    train_loader, test_loader, trntst_loader, all_loader, hsi_pca_wight = getMyData(datasetType, channels, windowSize, batch_size, num_workers)
    set_random_seed(random_seed)
    
    getMyReport(datasetType, net, test_loader, report_path[datasetType], device, model)
        
    if tsne == True:
        t_sne_full(net, test_loader, datasetType)
        
    if visualization:
        if datasetType == 0 or datasetType == 1 or datasetType == 5 or datasetType == 6 or datasetType == 7:
            getMyVisualization(datasetType, net, all_loader, image_path[datasetType], device, model)
        else:
            getMyVisualization(datasetType, net, trntst_loader, image_path[datasetType], device, model)
