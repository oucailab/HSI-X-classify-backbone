import torch
import random
import numpy as np
import torch.nn as nn
from scipy.io import loadmat
from sklearn.decomposition import PCA
from torchvision.transforms import ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import hdf5storage
from sklearn import preprocessing

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# 设置随机数种子
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

# PCA 降维
def applyPCA(data, n_components):
    h, w, b = data.shape  # 获取高光谱数据的空间维度和波段数量
    pca = PCA(n_components=n_components)  # 初始化PCA对象，指定降维后的主成分个数

    # 将数据从[h, w, b]形状转换为[-1, b]形状以适应PCA输入
    reshaped_data = np.reshape(data, (-1, b))

    # 进行PCA降维
    transformed_data = pca.fit_transform(reshaped_data)

    # 将降维后的数据重塑为[h, w, -1]形状
    transformed_data = np.reshape(transformed_data, (h, w, -1))

    # 获取每个主成分对应的权重矩阵，形状为[n_components, b]
    pca_weights = pca.components_
    print(f'------------------pca_weights = {pca_weights.shape}----------------------')
    return transformed_data, pca_weights

def normalize3D(data):
    # 将数据从 (H, W, C) 转换为 (H*W, C) 以便进行归一化
    H, W, C = data.shape
    data_reshaped = data.reshape(-1, C)

    # 使用 preprocessing.scale 进行归一化
    data_normalized = preprocessing.scale(data_reshaped)

    # 将数据重新转换为原始形状 (H, W, C)
    data_normalized = data_normalized.reshape(H, W, C)

    return data_normalized

def normalize2D(data):
    # 将数据从 (H, W, C) 转换为 (H*W, C) 以便进行归一化
    if len(data.shape) == 3:
        H, W, C = data.shape
        data = data.reshape(H, W*C)
    else:
        H, W = data.shape
    data_reshaped = data.reshape(-1)

    # 使用 preprocessing.scale 进行归一化
    data_normalized = preprocessing.scale(data_reshaped)

    # 将数据重新转换为原始形状 (H, W, C)
    data_normalized = data_normalized.reshape(H, W)

    return data_normalized
# 创建 Dataset
class HXDataset(Dataset):
    def __init__(self, hsi_pca, X, pos, windowSize, hsi=None , gt=None, transform=None, train=False):
        modes = ['symmetric', 'reflect']
        self.train = train
        self.pad = windowSize // 2
        self.windowSize = windowSize
        self.hsi = None
        if hsi is not None:
            self.hsi = np.pad(hsi, ((self.pad, self.pad),
                                (self.pad, self.pad), (0, 0)), mode=modes[windowSize % 2])
        self.hsi_pca = np.pad(hsi_pca, ((self.pad, self.pad),
                                (self.pad, self.pad), (0, 0)), mode=modes[windowSize % 2])
        self.X = None
        if(len(X.shape) == 2):
            self.X = np.pad(X, ((self.pad, self.pad),
                                (self.pad, self.pad)), mode=modes[windowSize % 2])
        elif(len(X.shape) == 3):
            self.X = np.pad(X, ((self.pad, self.pad),
                                (self.pad, self.pad), (0, 0)), mode=modes[windowSize % 2])
        self.pos = pos
        self.gt = None
        if gt is not None:
            self.gt = gt
        if transform:
            self.transform = transform

    def __getitem__(self, index):
        h, w = self.pos[index, :]
        hsi_pca = self.hsi_pca[h: h + self.windowSize, w: w + self.windowSize]
        if self.hsi is not None:
            hsi = self.hsi[h: h + self.windowSize, w: w + self.windowSize]
            hsi = hsi.astype(np.float32)  # 将 hsi 的数据类型转换为 float32
        X = self.X[h: h + self.windowSize, w: w + self.windowSize]
        if self.transform:
            hsi_pca = self.transform(hsi_pca).float()
            if self.hsi is not None:
                hsi = self.transform(hsi).float()
            X = self.transform(X).float()
            trans = [transforms.RandomHorizontalFlip(1.),
                     transforms.RandomVerticalFlip(1.)]
            if self.train:
                if random.random() < 0.5:
                    i = random.randint(0, 1)
                    hsi_pca = trans[i](hsi_pca)
                    if self.hsi is not None:
                        hsi = trans[i](hsi)
                    X = trans[i](X)
        if self.gt is not None:
            gt = torch.tensor(self.gt[h, w] - 1).long()
            if self.hsi is not None:
                return hsi_pca.unsqueeze(0),hsi.unsqueeze(0),X,gt
            return hsi_pca.unsqueeze(0), X, gt

        if self.hsi is not None:
            return hsi_pca.unsqueeze(0), hsi.unsqueeze(0), X, h, w

        return hsi_pca.unsqueeze(0), X, h, w

    def __len__(self):
        return self.pos.shape[0]

# 根据 index 获取数据
def getData(hsi_path, X_path, gt_path, index_path, keys, channels, windowSize, batch_size, num_workers, datatype, pin_memory=True):
    '''
    hsi: 高光谱图像数据
    X: X 图像数据
    gt: 真实标签, 0 代表未标注
    train_index: 用于训练的数据索引
    test_index: 用于测试的数据索引
    trntst_index: 用于训练和测试的数据索引，用于对有标签的数据进行可视化
    all_index: 所有数据的索引，包含未标注数据，用于对所有数据进行可视化
    '''
    if datatype == 5:
        # hsi = hdf5storage.loadmat(hsi_path)[keys[0]]
        hsi = loadmat(hsi_path)[keys[0]]
    else:
        hsi = loadmat(hsi_path)[keys[0]]
    X = loadmat(X_path)[keys[1]]
    gt = loadmat(gt_path)[keys[2]]
    train_index = loadmat(index_path)[keys[3]]
    test_index = loadmat(index_path)[keys[4]]
    trntst_index = np.concatenate((train_index, test_index), axis=0)
    all_index = loadmat(index_path)[keys[5]]

    # 使用 PCA 对 HSI 进行降维
    hsi_pca, hsi_pca_wight = applyPCA(hsi, channels)

    # 归一化
    hsi_pca = normalize3D(hsi_pca)
    hsi = normalize3D(hsi)
    if datatype == 0 or datatype == 1 or datatype == 2:
        X = normalize2D(X)
    else:
        X = normalize3D(X)
    hsi_pca_wight = normalize2D(hsi_pca_wight)
    


    # 创建 Dataset, 用于生成对应的 Dataloader
    HXtrainset = HXDataset(hsi_pca, X, train_index,
                           windowSize, hsi=hsi, gt=gt, transform=ToTensor(), train=True)
    HXtestset = HXDataset(hsi_pca, X, test_index,
                           windowSize, hsi=hsi, gt=gt, transform=ToTensor())
    HXtrntstset = HXDataset(hsi_pca, X, trntst_index,
                                windowSize, hsi=hsi, transform=ToTensor())
    HXallset = HXDataset(hsi_pca, X, all_index,
                         windowSize, hsi=hsi, transform=ToTensor())
    # 创建 Dataloader
    '''
    train_loader: 训练集
    test_loader: 测试集 
    trntst_loader: 用于画图，底色为白色，如 Trento 可视化图
    all_loader: 用于画图，底色为非白色，如 Houston 可视化图
    '''
    train_loader = DataLoader(
        HXtrainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=pin_memory)    #drop_last=True
    test_loader = DataLoader(
        HXtestset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True, pin_memory=pin_memory)
    trntst_loader = DataLoader(
        HXtrntstset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True, pin_memory=pin_memory)
    all_loader = DataLoader(
        HXallset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True, pin_memory=pin_memory)
    print("Success!")
    return train_loader, test_loader, trntst_loader, all_loader, hsi_pca_wight



# 获取 Houston2013 数据集
def getHouston2013Data(hsi_path, lidar_path, gt_path, index_path, channels, windowSize, batch_size, num_workers):
    print("Houston2013!")
    houston2013_keys = ['houston_hsi', 'houston_lidar', 'houston_gt', 'houston_train', 'houston_test', 'houston_all']
    return getData(hsi_path, lidar_path, gt_path, index_path, houston2013_keys, channels, windowSize, batch_size, num_workers, datatype=0)


# 获取 Houston2018 数据集
def getHouston2018Data(hsi_path, lidar_path, gt_path, index_path, channels, windowSize, batch_size, num_workers):
    print("Houston2018!")
    houston2018_keys = ['houston_hsi', 'houston_lidar', 'houston_gt', 'houston_train', 'houston_test', 'houston_all']
    return getData(hsi_path, lidar_path, gt_path, index_path, houston2018_keys, channels, windowSize, batch_size, num_workers, datatype=1)


# 获取 Trento 数据集
def getTrentoData(hsi_path, lidar_path, gt_path, index_path, channels, windowSize, batch_size, num_workers):
    print("Trento!")
    trento_keys = ['trento_hsi', 'trento_lidar', 'trento_gt', 'trento_train', 'trento_test', 'trento_all']
    return getData(hsi_path, lidar_path, gt_path, index_path, trento_keys, channels, windowSize, batch_size, num_workers, datatype=2)


# 获取 Berlin 数据集
def getBerlinData(hsi_path, sar_path, gt_path, index_path, channels, windowSize, batch_size, num_workers):
    print("Berlin!")
    berlin_keys = ['berlin_hsi', 'berlin_sar', 'berlin_gt', 'berlin_train', 'berlin_test', 'berlin_all']
    return getData(hsi_path, sar_path, gt_path, index_path, berlin_keys, channels, windowSize, batch_size, num_workers, datatype=3)


# 获取 Augsburg 数据集
def getAugsburgData(hsi_path, sar_path, gt_path, index_path, channels, windowSize, batch_size, num_workers):
    print("Augsburg!")
    augsburg_keys = ['augsburg_hsi', 'augsburg_sar', 'augsburg_gt', 'augsburg_train', 'augsburg_test', 'augsburg_all']
    return getData(hsi_path, sar_path, gt_path, index_path, augsburg_keys, channels, windowSize, batch_size, num_workers, datatype=4)



# 获取 YellowRiverEstuary 数据集
def getYellowRiverEstuaryData(hsi_path, sar_path, gt_path, index_path, channels, windowSize, batch_size, num_workers):
    print("YellowRiverEstuary!")
    yellowRiverEstuary_keys = ['data', 'data', 'data', 'train_index', 'test_index', 'all_index']
    return getData(hsi_path, sar_path, gt_path, index_path, yellowRiverEstuary_keys, channels, windowSize, batch_size, num_workers, datatype=5)


# 获取 LN01 数据集
def getLN01Data(hsi_path, msi_path, gt_path, index_path, channels, windowSize, batch_size, num_workers):
    print("LN01!")
    ln01_keys = ['Out', 'MSI', 'cdata', 'train_index', 'test_index', 'all_index']
    return getData(hsi_path, msi_path, gt_path, index_path, ln01_keys, channels, windowSize, batch_size, num_workers, datatype=6)

# 获取 LN02 数据集
def getLN02Data(hsi_path, msi_path, gt_path, index_path, channels, windowSize, batch_size, num_workers):
    print("LN02!")
    ln02_keys = ['Out', 'MSI', 'data', 'train_index', 'test_index', 'all_index']
    return getData(hsi_path, msi_path, gt_path, index_path, ln02_keys, channels, windowSize, batch_size, num_workers, datatype=7)
 
def getMyData(datasetType, channels, windowSize, batch_size, num_workers):
    if(datasetType == 0):
        hsi_path = '../../data/Houston2013/houston_hsi.mat'
        lidar_path = '../../data/Houston2013/houston_lidar.mat'
        gt_path = '../../data/Houston2013/houston_gt.mat'
        index_path = '../../data/Houston2013/houston_index.mat'
        train_loader, test_loader, trntst_loader, all_loader ,hsi_pca_wight = getHouston2013Data(hsi_path, lidar_path, gt_path, index_path, channels, windowSize, batch_size, num_workers)
    elif(datasetType == 1):
        hsi_path = '../../data/Houston2018/houston_hsi.mat'
        lidar_path = '../../data/Houston2018/houston_lidar.mat'
        gt_path = '../../data/Houston2018/houston_gt.mat'
        index_path = '../../data/Houston2018/houston_index.mat'
        train_loader, test_loader, trntst_loader, all_loader ,hsi_pca_wight = getHouston2018Data(hsi_path, lidar_path, gt_path, index_path, channels, windowSize, batch_size, num_workers)
    elif(datasetType == 2):    
        hsi_path = '../../data/Trento/trento_hsi.mat'
        lidar_path = '../../data/Trento/trento_lidar.mat'
        gt_path = '../../data/Trento/trento_gt.mat'
        index_path = '../../data/Trento/trento_index.mat'
        train_loader, test_loader, trntst_loader, all_loader ,hsi_pca_wight = getTrentoData(hsi_path, lidar_path, gt_path, index_path, channels, windowSize, batch_size, num_workers)
    elif(datasetType == 3):    
        hsi_path = "../../data/Berlin/berlin_hsi.mat"
        sar_path = "../../data/Berlin/berlin_sar.mat"
        gt_path = "../../data/Berlin/berlin_gt.mat"
        index_path = "../../data/Berlin/berlin_index.mat"
        train_loader, test_loader, trntst_loader, all_loader ,hsi_pca_wight = getBerlinData(hsi_path, sar_path, gt_path, index_path, channels, windowSize, batch_size, num_workers)
    elif(datasetType == 4):    
        hsi_path = "../../data/Augsburg/augsburg_hsi.mat"
        sar_path = "../../data/Augsburg/augsburg_sar.mat"
        gt_path = "../../data/Augsburg/augsburg_gt.mat"
        index_path = "../../data/Augsburg/augsburg_index.mat"
        train_loader, test_loader, trntst_loader, all_loader ,hsi_pca_wight = getAugsburgData(hsi_path, sar_path, gt_path, index_path, channels, windowSize, batch_size, num_workers)
    elif(datasetType == 5):    
        hsi_path = "../../data/YellowRiverEstuary/data_hsi.mat"
        sar_path = "../../data/YellowRiverEstuary/data_sar.mat"
        gt_path = "../../data/YellowRiverEstuary/data_gt.mat"
        index_path = "../../data/YellowRiverEstuary/data_index.mat"
        train_loader, test_loader, trntst_loader, all_loader ,hsi_pca_wight = getYellowRiverEstuaryData(hsi_path, sar_path, gt_path, index_path, channels, windowSize, batch_size, num_workers)
    elif(datasetType == 6):    
        hsi_path = "../../data/LN01/LN01_HHSI.mat"
        msi_path = "../../data/LN01/LN01_MSI.mat"
        gt_path = "../../data/LN01/LN01_GT.mat"
        index_path = "../../data/LN01/LN01_INDEX.mat"
        train_loader, test_loader, trntst_loader, all_loader ,hsi_pca_wight = getLN01Data(hsi_path, msi_path, gt_path, index_path, channels, windowSize, batch_size, num_workers)    
    elif(datasetType == 7):    
        hsi_path = "../../data/LN02/LN02_HHSI.mat"
        msi_path = "../../data/LN02/LN02_MSI.mat"
        gt_path = "../../data/LN02/LN02_GT.mat"
        index_path = "../../data/LN02/LN02_INDEX.mat"
        train_loader, test_loader, trntst_loader, all_loader ,hsi_pca_wight = getLN02Data(hsi_path, msi_path, gt_path, index_path, channels, windowSize, batch_size, num_workers)    
        
    return train_loader, test_loader, trntst_loader, all_loader ,hsi_pca_wight


