from torch.backends import cudnn

from dataset import *

import time
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# 使用新的配置系统
import config

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


from compare_net.FusAtNet import FusAtNet
from compare_net.ExVit import MViT
from compare_net.HybridSN import HybridSN
from compare_net.FDNet import FDNet
from compare_net.MICF_Net import MICF_Net
from compare_net.M2FNet import M2Fnet
from compare_net.S2ENet import S2ENet
from compare_net.DFINet import Net as DFINet
from compare_net.ClassifierNet import Net as AsyFFNet
from compare_net.ClassifierNet import Bottleneck
from compare_net.HCTnet import HCTnet as HCTNet
from compare_net.MixConvNet import MixConvNet as MACN
from compare_net.TBCNN import TBCNN
from compare_net.cnnbasenet import CNNet

def initialize_model(model, datasetType, device, out_features):
    """初始化指定的模型"""
    # 从配置系统获取参数
    in_channels = config.get_value('channels')  # PCA降维后的通道数
    data_channels = config.get_value('data_channels')  # 原始通道数
    lidar_or_sar_channels = config.get_value('lidar_or_sar_channels')[datasetType]
    patch_size = config.get_value('windowSize')
    num_classes = config.get_value('out_features')[datasetType]
    
    if model == 'FusAtNet':
        net = FusAtNet(input_channels=in_channels, input_channels2=lidar_or_sar_channels, num_classes=out_features)
        
    elif model == 'ExViT':
        net = MViT(
            patch_size = patch_size,
            num_patches = [in_channels, lidar_or_sar_channels],
            num_classes = out_features,
            dim = 64,
            depth = 6,
            heads = 4,
            mlp_dim = 32,
            dropout = 0.1,
            emb_dropout = 0.1,
            mode = 'MViT'
        )
        
    elif model == 'HybridSN':
        net = HybridSN(out_features)
        
    elif model == 'FDNet':
        net = FDNet(l1 = in_channels,
                    l2 = lidar_or_sar_channels, 
                    patch_size = patch_size, 
                    num_classes = num_classes,
                    wavename = 'db2', 
                    attn_kernel_size = 9, 
                    coefficient_hsi = 0.7,
                    fae_embed_dim = 64, 
                    fae_depth = 1)
                    
    elif model == 'MICF_Net':
        dim = 64
        net = MICF_Net(patch_size=patch_size, dim=dim, input_dim=in_channels, num_classes=num_classes, heads=8, num_atte=8, dep=2)
        # MICF_Net需要额外的原型参数
        x_proto = torch.empty(num_classes, dim).to(device)
        torch.nn.init.normal_(x_proto, mean=0, std=0.2)
        l_proto = torch.empty(num_classes, dim).to(device)
        torch.nn.init.normal_(l_proto, mean=0, std=0.2)
        return net, x_proto, l_proto

    elif model == 'M2FNet':
        net = M2Fnet(FM=16, NC=in_channels, LC=lidar_or_sar_channels, Classes=num_classes)
        
    elif model == 'S2ENet':
        net = S2ENet(input_channels=in_channels, input_channels2=lidar_or_sar_channels, n_classes=num_classes, patch_size=patch_size)
        
    elif model == 'DFINet':
        net = DFINet(channel_hsi=in_channels, channel_msi=lidar_or_sar_channels, class_num=num_classes)
        
    elif model == 'AsyFFNet':
        net = AsyFFNet(hsi_channels=in_channels, sar_channels=lidar_or_sar_channels, hidden_size=128, block=Bottleneck, num_parallel=2, num_reslayer=4, num_classes=num_classes)
        
    elif model == 'HCTNet':
        net = HCTNet(in_channels=lidar_or_sar_channels, num_classes=num_classes)
        
    elif model == 'MACN':
        net = MACN(in_channels=lidar_or_sar_channels, num_classes=num_classes)
        
    elif model == 'TBCNN':
        net = TBCNN(hsi_dim=in_channels, lidar_dim=lidar_or_sar_channels, num_class=num_classes)
        
    elif model == 'CNN':
        net = CNNet(in_channels, lidar_or_sar_channels, 128, num_classes)
        
    else:
        raise ValueError(f"Unknown model: {model}")
    
    return net

def model_forward(model, net, hsi_pca, hsi, sar, additional_params=None):
    """根据模型类型进行前向传播"""
    if model == 'FusAtNet':
        outputs = net(hsi_pca.squeeze(1), sar)
    elif model == 'ExViT':
        outputs = net(hsi_pca.squeeze(1), sar)
    elif model == 'HybridSN':
        outputs = net(hsi_pca)
    elif model == 'FDNet':
        outputs = net(hsi_pca, sar)
    elif model == 'MICF_Net':
        x_proto, l_proto = additional_params
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
        raise ValueError(f"Unknown model: {model}")
    
    return outputs

def train(epochs, lr, model, cuda, train_loader, test_loader, out_features, model_savepath, log_path, hsi_pca_wight, datasetType):
    device = torch.device(cuda if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    cudnn.benchmark = True
    
    hsi_pca_wight_tensor = torch.from_numpy(hsi_pca_wight).to(device)
    
    # 初始化模型
    model_result = initialize_model(model, datasetType, device, out_features)
    if isinstance(model_result, tuple):  # MICF_Net返回额外参数
        net, x_proto, l_proto = model_result
        additional_params = (x_proto, l_proto)
    else:
        net = model_result
        additional_params = None
    
    net.to(device)

    # 设置训练参数
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    max_acc = 0
    sum_time = 0
    
    # 记录训练开始信息
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    current_time_log = 'start time: {}'.format(current_time)
    getLog(log_path, config.get_taskInfo())
    getLog(log_path, '-------------------Started Training-------------------')
    getLog(log_path, current_time_log)

    for epoch in range(epochs):
        since = time.time()
        net.train()
        
        # 训练进度条（显示当前epoch的批次进度）
        try:
            iterator = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', unit='batch')
        except Exception:
            iterator = train_loader
        for i, (hsi_pca, hsi, sar, tr_labels) in enumerate(iterator):
            hsi_pca = hsi_pca.to(device)
            sar = sar.to(device)
            hsi = hsi.to(device)
            tr_labels = tr_labels.to(device)
            
            optimizer.zero_grad()
            
            # 模型前向传播
            outputs = model_forward(model, net, hsi_pca, hsi, sar, additional_params)
            loss = criterion(outputs, tr_labels)
            
            loss.backward()
            optimizer.step()
            
        # 每个epoch后进行测试
        if epoch % 1 == 0:
            net.eval()
            count = 0
            
            for hsi_pca, hsi, sar, gtlabels in test_loader:
                hsi_pca = hsi_pca.to(device)
                hsi = hsi.to(device)
                sar = sar.to(device)
                
                with torch.no_grad():
                    outputs = model_forward(model, net, hsi_pca, hsi, sar, additional_params)
                
                outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
                if count == 0:
                    y_pred_test = outputs
                    gty = gtlabels
                    count = 1
                else:
                    y_pred_test = np.concatenate((y_pred_test, outputs))
                    gty = np.concatenate((gty, gtlabels))
                    
            acc1 = accuracy_score(gty, y_pred_test)

            if acc1 > max_acc:
                # 创建模型保存目录如果不存在
                os.makedirs(os.path.dirname(model_savepath), exist_ok=True)
                torch.save(net, model_savepath)
                max_acc = acc1
            
            time_elapsed = time.time() - since
            sum_time += time_elapsed
            rest_time = (sum_time / (epoch + 1)) * (epochs - epoch - 1)
            currentTime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
            log = currentTime + ' [Epoch: %d] [%.0fs, %.0fh %.0fm %.0fs] [current loss: %.4f] acc: %.4f' %(epoch + 1, time_elapsed, (rest_time // 60) // 60, (rest_time // 60) % 60, rest_time % 60, loss.item(), acc1)
            print(log)
            getLog(log_path, log)

    print('max_acc: %.4f' %(max_acc))  
    finish_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    finish_time_log = 'finish time: {} '.format(finish_time)
    mac_acc_log = 'max_acc: {} '.format(max_acc)
    getLog(log_path, mac_acc_log)
    getLog(log_path, finish_time_log)
    getLog(log_path, '-------------------Finished Training-------------------')

def getLog(log_path, str):
    # 创建目录如果不存在
    import os
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    with open(log_path, 'a+') as log:
        log.write('{}'.format(str))
        log.write('\n')

def myTrain(datasetType, model):
    """训练函数，从配置系统获取参数"""
    channels = config.get_value('channels')
    windowSize = config.get_value('windowSize')
    out_features = config.get_value('out_features')
    cuda = config.get_value('cuda')
    lr = config.get_value('lr')
    epoch_nums = config.get_value('epoch_nums')
    batch_size = config.get_value('batch_size')
    num_workers = config.get_value('num_workers')
    random_seed = config.get_value('random_seed')
    model_savepath = config.get_value('model_savepath')
    log_path = config.get_value('log_path')
    
    print(f"训练参数: model={model}, epochs={epoch_nums}, lr={lr}, cuda={cuda}")
    
    train_loader, test_loader, trntst_loader, all_loader, hsi_pca_wight = getMyData(datasetType, channels, windowSize, batch_size, num_workers)
    set_random_seed(random_seed)
    
    train(epoch_nums, lr, model, cuda, train_loader, test_loader, out_features[datasetType], model_savepath[datasetType], log_path[datasetType], hsi_pca_wight, datasetType)
