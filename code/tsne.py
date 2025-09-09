import torch
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

def save_img(feature,gt_list,accuracy,dataset):
    accuracy_str = f"{accuracy * 100:.2f}"
    out_dir = os.path.join("TSNE", str(dataset))
    os.makedirs(out_dir, exist_ok=True)      # 确保目录存在
    filename = os.path.join(out_dir, f"tsne_accuracy_{accuracy_str}.png")
    
    # 开新 figure，避免复用导致混乱
    plt.figure(figsize=(6,6))
    tsne = TSNE(n_components=2, perplexity=10, learning_rate=50)
    features = np.array(feature).reshape(len(feature), -1)
    features_tsne = tsne.fit_transform(features)

    plt.xlim(-100, 100)
    plt.ylim(-100, 100)
    plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=gt_list, s=5)
    plt.tick_params(labelsize=9)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()  # 释放资源，防止内存泄漏

def t_sne(model,test_loader,dataset):
    """Validation and get the metric
    """
    epoch_losses, epoch_accuracy = 0.0, 0.0
    criterion = torch.nn.CrossEntropyLoss()
    houston2013_color_map = [[0, 0, 131], [0, 0, 203], [0, 19, 255], [0, 91, 255], [0, 167, 255], [0, 239, 255], [55, 255, 199], [131, 255, 123], [203, 255, 51], [255, 235, 0], [255, 163, 0], [255, 87, 0], [255, 15, 0], [199, 0, 0], [127, 0, 0]]
    
    houston2018_color_map = [
    [50, 205, 51],
    [173, 255, 48],
    [0, 128, 129],
    [34, 139, 34],
    [46, 79, 78],
    [139, 69, 18],
    [0, 255, 255],
    [100, 100, 100],  #255,255,255改成100
    [211, 211, 211],
    [254, 0, 0],
    [169, 169, 169],
    [105, 105, 105],
    [139, 0, 1],
    [200, 100, 0],  #####
    [254, 165, 0],
    [255, 255, 0],
    [218, 165, 33],
    [255, 0, 254],
    [0, 0, 254],
    [63, 224, 208]
    ]

    berlin_color_map = [[26, 163, 25], [216, 216, 216], [216, 89, 89], [
            0, 204, 51], [204, 153, 52], [244, 231, 1], [204, 102, 204], [0, 53, 255]]

    augsburg_color_map = [[26, 163, 25], [216, 216, 216], [216, 89, 89], [
            0, 204, 51], [244, 231, 1], [204, 102, 204], [0, 53, 255]]
    trento_color_map = [[0, 47, 255], [0, 223, 255], [143, 255, 111], [255, 207, 0], [255, 31, 0], [127, 0, 0]]
    
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

    feature_list = []
    # gt_list = []
    count = 0
    device = torch.device('cuda:0')  
    with torch.no_grad():
        for batch_idx, (hsi_pca, hsi, lidar, tr_labels) in enumerate(test_loader):
            # hsi = hsi.to(device)
            # hsi = hsi[:, 0, :, :, :]
            lidar = lidar.to(device)
            hsi_pca = hsi_pca.to(device)
            tr_labels = tr_labels.to(device)

            feature,output = model(hsi_pca,lidar)
            tr_labels = tr_labels.detach().cpu().numpy().astype(int)
            #计算一下准确率
            output = np.argmax(output.detach().cpu().numpy(), axis=1)
            accuracy = accuracy_score(tr_labels, output)
            # print(tr_labels)
            # print(type(tr_labels))
            feature = feature.detach().cpu().numpy()
            # print(feature.shape)
            # print(type(tr_labels))
            # feature_list.append(feature[0])
            if dataset == 1:
                houston2018_color_map = np.array(houston2018_color_map)
                gt=(houston2018_color_map[tr_labels]*1.0/255.0)
                save_img(feature,gt,accuracy,"Houston2018")
            elif dataset == 3:
                berlin_color_map = np.array(berlin_color_map)
                gt=(berlin_color_map[tr_labels]*1.0/255.0)
                save_img(feature,gt,accuracy,"Berlin")
            elif dataset == 0:
                houston2013_color_map = np.array(houston2013_color_map)
                gt=(houston2013_color_map[tr_labels]*1.0/255.0)
                save_img(feature,gt,accuracy,"Houston2013")
            elif dataset == 2:
                trento_color_map = np.array(trento_color_map)
                gt=(trento_color_map[tr_labels]*1.0/255.0)
                save_img(feature,gt,accuracy,"Trento")
            elif dataset == 4:
                augsburg_color_map = np.array(augsburg_color_map)
                gt=augsburg_color_map[tr_labels]*1.0/255.0
                save_img(feature,gt,accuracy,"Augsburg")
            elif dataset == 5:
                yellowRiverEstuary_color_map = np.array(yellowRiverEstuary_color_map)
                gt=yellowRiverEstuary_color_map[tr_labels]*1.0/255.0
                save_img(feature,gt,accuracy,"YellowRiverEstuary")



def t_sne_full(model, test_loader, dataset):
    """Validation and get the metric, then t-SNE visualize all samples"""
    # 1) 定义所有数据集对应的颜色映射和名称
    color_maps = {
        0: (np.array([[0, 0, 131], [0, 0, 203], [0, 19, 255], [0, 91, 255], [0, 167, 255], [0, 239, 255], [55, 255, 199], [131, 255, 123], [203, 255, 51], [255, 235, 0], [255, 163, 0], [255, 87, 0], [255, 15, 0], [199, 0, 0], [127, 0, 0]]),        "Houston2013"),  # 补全你的色彩列表
        1: (np.array([[50, 205, 51],[173, 255, 48],[0, 128, 129],[34, 139, 34],
            [46, 79, 78],
            [139, 69, 18],
            [0, 255, 255],
            [100, 100, 100],  #255,255,255改成100
            [211, 211, 211],
            [254, 0, 0],
            [169, 169, 169],
            [105, 105, 105],
            [139, 0, 1],
            [200, 100, 0],  #####
            [254, 165, 0],
            [255, 255, 0],
            [218, 165, 33],
            [255, 0, 254],
            [0, 0, 254],
            [63, 224, 208]
        ]),    "Houston2018"),
        2: (np.array([[0, 47, 255], [0, 223, 255], [143, 255, 111], [255, 207, 0], [255, 31, 0], [127, 0, 0]]),      "Trento"),
        3: (np.array([[26, 163, 25], [216, 216, 216], [216, 89, 89], [
            0, 204, 51], [204, 153, 52], [244, 231, 1], [204, 102, 204], [0, 53, 255]]),   "Berlin"),
        4: (np.array([[26, 163, 25], [216, 216, 216], [216, 89, 89], [
            0, 204, 51], [244, 231, 1], [204, 102, 204], [0, 53, 255]]),   "Augsburg"),
        5: (np.array([
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
        ]),     "YellowRiverEstuary"),
    }

    # 2) 准备收集特征、标签与预测
    all_features, all_labels, all_preds = [], [], []
    device = next(model.parameters()).device  # 自动捕获模型所在设备
    model.eval()

    with torch.no_grad():
        for hsi_pca, hsi, lidar, labels in test_loader:
            hsi_pca = hsi_pca.to(device)
            lidar   = lidar.to(device)
            labels  = labels.to(device)

            features, logits = model(hsi_pca, lidar)
            preds = logits.argmax(dim=1)

            all_features.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    # 3) 拼接并计算准确率
    all_features = np.vstack(all_features)      # (N, feat_dim)
    all_labels   = np.concatenate(all_labels)   # (N,)
    all_preds    = np.concatenate(all_preds)    # (N,)
    acc = accuracy_score(all_labels, all_preds)

    # 4) 根据 dataset 选择色彩映射，并做归一化
    if dataset in color_maps:
        cmap, name = color_maps[dataset]
        gt_colors = cmap[all_labels] / 255.0            # (N, 3)
        save_img(all_features, gt_colors, acc, name)
    else:
        raise ValueError(f"No color map defined for dataset {dataset}")
