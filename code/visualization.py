import numpy as np
from tqdm import tqdm
from skimage import io

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch

# 可视化 data 中的数据
def visualization(datasetType, net, data, save_path, device, color_map, size, model):
    """
    net: 训练好的网络
    data: 需要可视化的数据集
    save_path: 图片保存的位置，包含图片名
    color_map: 可视化用到的颜色，白色可能被用来可视化没有标签的数据，请勿使用
    size: 图片的尺寸 Huston: (349, 1905) Trento: (166, 600) 
    """
    from test import model_forward_test  # 导入测试时的前向传播函数
    
    net.eval()
    h, w = size[:]
    pred = -np.ones((h, w))
    for hsi_pca, hsi, sar, i, j in tqdm(data):

        hsi_pca = hsi_pca.to(device)
        hsi = hsi.to(device)
        sar = sar.to(device)
        
        with torch.no_grad():
            output = model_forward_test(model, net, hsi_pca, hsi, sar)

        output = np.argmax(output.detach().cpu().numpy(), axis=1)
        idx = 0
        for x, y in zip(i, j):
            pred[x, y] = output[idx]
            idx += 1
    res = np.zeros((h, w, 3), dtype=np.uint8)
    pos = pred > -1
    for i in range(h):
        for j in range(w):
            if pos[i, j]:
                res[i, j] = color_map[int(pred[i, j])]
            else:
                res[i, j] = [0, 0, 0]
    
    # 创建目录如果不存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    io.imsave(save_path, res)

# 可视化 Houston2013 数据集
def visHouston2013(datasetType, net, data, save_path, device, model):
    """
    net: 训练好的网络
    data: 需要可视化的数据集
    save_path: 图片保存的位置，包含图片名
    """
    # Houston2013 color map
    houston2013_color_map = [[0, 0, 131], [0, 0, 203], [0, 19, 255], [0, 91, 255], [0, 167, 255], [0, 239, 255], [55, 255, 199], [131, 255, 123], [203, 255, 51], [255, 235, 0], [255, 163, 0], [255, 87, 0], [255, 15, 0], [199, 0, 0], [127, 0, 0]]
    # Houston2013 尺寸
    houston2013_size = [349, 1905]
    print("Houston2013 Start!")
    visualization(datasetType, net, data, save_path, device, houston2013_color_map, houston2013_size, model)
    print("Visualization Success!")

# 可视化 Houston2018 数据集
def visHouston2018(datasetType, net, data, save_path, device, model):
    # Houston2018 color map
    houston2018_color_map = [[50, 205, 51], [173, 255, 48], [0, 128, 129], [34, 139, 34], [46, 79, 78], [139, 69, 18], [0, 255, 255], [255, 255, 255], [211, 211, 211], [254, 0, 0], [169, 169, 169], [105, 105, 105], [139, 0, 1], [200, 100, 0], [254, 165, 0], [255, 255, 0], [218, 165, 33], [255, 0, 254], [0, 0, 254], [63, 224, 208]]
    # Houston2018 尺寸
    houston2018_size = [1202, 4768]
    print("Houston2018 Start!")
    visualization(datasetType, net, data, save_path, device, houston2018_color_map, houston2018_size, model)
    print("Visualization Success!")

# 可视化 Trento 数据集
def visTrento(datasetType, net, data, save_path, device, model):
    # Trento color map
    trento_color_map = [[0, 47, 255], [0, 223, 255], [143, 255, 111], [255, 207, 0], [255, 31, 0], [127, 0, 0]]
    # Trento 尺寸
    trento_size = [166, 600]
    print("Trento Start!")
    visualization(datasetType, net, data, save_path, device, trento_color_map, trento_size, model)
    print("Visualization Success!")

# 可视化 Berlin 数据集
def visBerlin(datasetType, net, data, save_path, device, model):
    # Berlin color map
    berlin_color_map = [[26, 163, 25], [216, 216, 216], [216, 89, 89], [0, 204, 51], [204, 153, 52], [244, 231, 1], [204, 102, 204], [0, 53, 255]]
    # Berlin 尺寸
    berlin_size = [1723, 476]
    print("Berlin Start!")
    visualization(datasetType, net, data, save_path, device, berlin_color_map, berlin_size, model)
    print("Visualization Success!")

# 可视化 Augsburg 数据集
def visAugsburg(datasetType, net, data, save_path, device, model):
    # Augsburg color map
    augsburg_color_map = [[26, 163, 25], [216, 216, 216], [216, 89, 89], [0, 204, 51], [244, 231, 1], [204, 102, 204], [0, 53, 255]]
    # Augsburg 尺寸
    augsburg_size = [332, 485]
    print("Augsburg Start!")
    visualization(datasetType, net, data, save_path, device, augsburg_color_map, augsburg_size, model)
    print("Visualization Success!")

# 可视化 YellowRiverEstuary 数据集
def visYellowRiverEstuary(datasetType, net, data, save_path, device, model):
    # YellowRiverEstuary color map
    yellowRiverEstuary_color_map = [
        [255, 165, 0],    # 互花米草- Spartina Alterniflora
        [0, 191, 255],    # 池塘- Pond
        [34, 139, 34],    # 林地- Woodland
        [139, 69, 19],    # 芦苇- Phragmites
        [0, 255, 127],    # 香蒲- Typha Orientalis
        [188, 143, 143],  # 潮滩芦苇- Intertidal Phragmites
        [100, 149, 237],  # 生态修复池- Ecological Reservoir
        [255, 215, 0],    # 耕地- Arable Land
        [255, 182, 193],  # 藕池- Lotus Pond
        [128, 128, 128],  # 油田区- Oil Field
        [255, 255, 240],  # 盐田- Salt Field
        [144, 238, 144],  # 盐地碱蓬- Suaeda Salsa
        [0, 139, 139],    # 河流- River
        [210, 105, 30],   # 芦苇、柽柳混生区- Mixed Area 1- Reed and Tamarisk Mixed Area
        [255, 69, 0],     # 柽柳、盐地碱蓬混生区- Mixed Area 2- Tamarisk and Suaeda Mixed Area
        [255, 99, 71],    # 柽柳、芦苇、盐地碱蓬混生区- Mixed Area 3- Tamarisk, Reed, and Suaeda Mixed Area
        [210, 180, 140],  # 裸滩- Mudflat
        [0, 0, 255]       # 海域- Sea
    ]
    # YellowRiverEstuary 尺寸
    yellowRiverEstuary_size = [3555, 4026]
    print("YellowRiverEstuary Start!")
    visualization(datasetType, net, data, save_path, device, yellowRiverEstuary_color_map, yellowRiverEstuary_size, model)
    print("Visualization Success!")

# 可视化 LN01 数据集
def visLN01(datasetType, net, data, save_path, device, model):
    # LN01 color map
    ln01_color_map = [
        [229, 115, 115],   # 1 Reservoir
        [186, 104, 200],   # 2 Seawater
        [131, 134, 203],   # 3 Sandy soil
        [79, 195, 247],    # 4 Broken bridge
        [77, 182, 172],    # 5 Barren grass
        [139, 195, 74],    # 6 Highway
        [255, 235, 59],    # 7 Railway
        [255, 152, 0],     # 8 Bare soil
        [121, 85, 72],     # 9 Mountain vegetation
        [158, 158, 158]    # 10 Arable land
    ]
    # LN01 尺寸
    ln01_size = [900, 900]
    print("LN01 Start!")
    visualization(datasetType, net, data, save_path, device, ln01_color_map, ln01_size, model)
    print("Visualization Success!")

# 可视化 LN02 数据集
def visLN02(datasetType, net, data, save_path, device, model):
    # LN02 color map
    ln02_color_map = [
        [77, 182, 172],   # 1: Liao river
        [187, 104, 201],   # 2: Phragmites australis
        [131, 134, 203],   # 3: Paddy fields
        [79, 195, 247],    # 4: Intertidal muds
        [255, 152, 0],     # 5: Suaeda salsa
        [139, 195, 74],    # 6: Construction land
        [126, 88, 74],     # 7: Industrial land
        [255, 202, 83],    # 8: Aquaculture ponds
        [232, 116, 116]    # 9: Reed water system
    ]
    # LN02 尺寸
    ln02_size = [1536, 1536]
    print("LN02 Start!")
    visualization(datasetType, net, data, save_path, device, ln02_color_map, ln02_size, model)
    print("Visualization Success!")
def getMyVisualization(datasetType, net, data, save_path, device, model):
    if(datasetType == 0):
        visHouston2013(datasetType, net, data, save_path, device, model)
    elif(datasetType == 1):
        visHouston2018(datasetType, net, data, save_path, device, model)
    elif(datasetType == 2):    
        visTrento(datasetType, net, data, save_path, device, model)
    elif(datasetType == 3):    
        visBerlin(datasetType, net, data, save_path, device, model)
    elif(datasetType == 4):    
        visAugsburg(datasetType, net, data, save_path, device, model)
    elif(datasetType == 5):    
        visYellowRiverEstuary(datasetType, net, data, save_path, device, model)
    elif(datasetType == 6):    
        visLN01(datasetType, net, data, save_path, device, model)
    elif(datasetType == 7):    
        visLN02(datasetType, net, data, save_path, device, model)
