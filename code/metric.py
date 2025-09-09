import torch  
import thop
import time

from backbone.code.net import EyeNet

# from compare_net.FusAtNet import FusAtNet
# from compare_net.ExVit import MViT
# from compare_net.HybridSN import HybridSN
# from compare_net.FDNet import FDNet
# from compare_net.MICF_Net import MICF_Net
# from compare_net.M2FNet import M2Fnet
# from compare_net.S2ENet import S2ENet
# from compare_net.DFINet import Net as DFINet
# from compare_net.ClassifierNet import Net as AsyFFNet
# from compare_net.ClassifierNet import Bottleneck
# from compare_net.HCTnet import HCTnet as HCTNet
# from compare_net.MixConvNet import MixConvNet as MACN
# from compare_net.TBCNN import TBCNN
from compare_net.MICF_Net import MICF_Net
from compare_net.M2FNet import M2Fnet


from sklearn.decomposition import PCA

# PCA 降维
def applyPCA(data, n_components):
    # 处理PyTorch CUDA Tensor的情况
    if isinstance(data, torch.Tensor):
        device = data.device  # 记录原始设备
        # 将数据从GPU转到CPU，并转为NumPy数组
        data_np = data.cpu().numpy()
    else:
        data_np = np.asarray(data)  # 普通NumPy数组处理
    
    # 获取输入数据的维度信息（batch, seq, channels, height, width）
    batch, seq, channels, height, width = data_np.shape
    
    # 将数据从 [batch, seq, channels, height, width] 转换为 [n_samples, channels]
    reshaped_data = data_np.reshape(-1, channels)
    
    # 执行PCA
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(reshaped_data)  # [n_samples, n_components]
    
    # 将结果重塑为PyTorch Tensor并放回原始设备
    transformed_data = transformed_data.reshape(batch, seq, height, width, n_components)
    transformed_data = transformed_data.transpose(0, 1, 4, 2, 3)  # [batch, seq, n_components, height, width]
    transformed_data = torch.from_numpy(transformed_data).to(device)  # 转回Tensor并恢复设备
    
    # 获取PCA权重矩阵（同样转为Tensor）
    pca_weights = torch.from_numpy(pca.components_).to(device)  # [n_components, channels]
    
    print(f'PCA权重矩阵形状: {pca_weights.shape}')
    return transformed_data, pca_weights



# 改进后的美观输入
def create_structured_hsi(input_shape):
    """
    生成具有空间结构的HSI输入数据
    参数：
    input_shape: 输入形状 (batch, seq, channels, height, width)
    """
    batch, seq, channels, H, W = input_shape
    
    # 创建基础空间模式（使用正弦波+梯度）
    x = torch.linspace(0, 4*np.pi, W).cuda()
    y = torch.linspace(0, 4*np.pi, H).cuda()
    xx, yy = torch.meshgrid(x, y, indexing='xy')
    
    # 基础模式1：正弦波（水平方向）
    pattern1 = torch.sin(xx).unsqueeze(0).unsqueeze(0)
    
    # 基础模式2：梯度（垂直方向）
    pattern2 = yy.unsqueeze(0).unsqueeze(0) / yy.max()
    
    # 组合模式（通道维度叠加）
    structured_data = 0.6*pattern1 + 0.4*pattern2
    
    # 扩展到指定通道数（保持空间模式一致）
    hsi = structured_data.repeat(1, 1, channels, 1, 1)
    
    # 添加轻微噪声（模拟真实数据）
    hsi += 0.1*torch.randn_like(hsi)
    
    # 归一化到[-1,1]范围
    hsi = 2*(hsi - hsi.min())/(hsi.max() - hsi.min()) - 1
    
    return hsi

# #MSFMamba
# torch.cuda.empty_cache()

hsi_channel = 30
pca_channel = 30
sar_channel = 4
window_size = 11
class_num = 7
datatype = 4

input1 = (1, 1, hsi_channel, window_size, window_size)
hsi = torch.randn(input1).cuda()
# hsi = create_structured_hsi(input1).cuda()
# hsi_pca, hsi_pca_wight_tensor = applyPCA(hsi, pca_channel)
input2 = (1, sar_channel, window_size, window_size) 
sar = torch.randn(input2).cuda()
tr_labels = (1,)
tr_labels = torch.randint(low=0, high=7, size=tr_labels).cuda()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# model = 'M2Fnet'

# if model == 'KBRANet':
#     # KBRANet
#     net = KBRANet(class_num, datatype, hsi_pca_wight_tensor).to(device)
#     start_time = time.time()
#     flops, params = thop.profile(net, inputs=(hsi_pca,hsi,sar))  #KBRANet
#     end_time = time.time()
#     # 计算运行时间
#     elapsed_time = end_time - start_time
# elif model == 'TBCNN':
#     net = TBCNN(hsi_dim=pca_channel, lidar_dim=sar_channel, num_class=class_num).to(device)
#     start_time = time.time()
#     flops, params = thop.profile(net, inputs=(hsi_pca, sar)) 
#     end_time = time.time()    
#     # 计算运行时间
#     elapsed_time = end_time - start_time
# elif model == 'FusAtNet':
#     net = FusAtNet(input_channels=pca_channel, input_channels2=sar_channel, num_classes=class_num).to(device)
#     start_time = time.time()
#     flops, params = thop.profile(net, inputs=(hsi_pca.squeeze(1), sar)) 
#     end_time = time.time()    
#     # 计算运行时间
#     elapsed_time = end_time - start_time
# elif model == 'S2ENet':
#     net = S2ENet(input_channels=pca_channel, input_channels2=sar_channel, n_classes=class_num, patch_size=window_size).to(device)
#     start_time = time.time()
#     flops, params = thop.profile(net, inputs=(hsi_pca.squeeze(1), sar)) 
#     end_time = time.time()    
#     # 计算运行时间
#     elapsed_time = end_time - start_time
# elif model == 'DFINet':
#     net = DFINet(channel_hsi=pca_channel, channel_msi=sar_channel, class_num=class_num).to(device)
#     start_time = time.time()
#     flops, params = thop.profile(net, inputs=(hsi_pca.squeeze(1), sar)) 
#     end_time = time.time()    
#     # 计算运行时间
#     elapsed_time = end_time - start_time
# elif model == 'AsyFFNet':
#     net = AsyFFNet(hsi_channels=pca_channel, sar_channels=sar_channel, hidden_size=128, block=Bottleneck, num_parallel=2, num_reslayer=4, num_classes=class_num).to(device)
#     start_time = time.time()
#     flops, params = thop.profile(net, inputs=(hsi_pca.squeeze(1), sar)) 
#     end_time = time.time()    
#     # 计算运行时间
#     elapsed_time = end_time - start_time
# elif model == 'ExViT':
#     net = MViT(
#             patch_size = window_size,
#             num_patches = [pca_channel,sar_channel],
#             num_classes = class_num,
#             dim = 64,
#             depth = 6,
#             heads = 4,
#             mlp_dim = 32,
#             dropout = 0.1,
#             emb_dropout = 0.1,
#             mode = 'MViT'
#         ).to(device)
#     start_time = time.time()
#     flops, params = thop.profile(net, inputs=(hsi_pca.squeeze(1), sar)) 
#     end_time = time.time()    
#     # 计算运行时间
#     elapsed_time = end_time - start_time
# elif model == 'HCTNet':
#     net = HCTNet(in_channels=sar_channel, num_classes=class_num).to(device)
#     start_time = time.time()
#     flops, params = thop.profile(net, inputs=(hsi_pca, sar)) 
#     end_time = time.time()    
#     # 计算运行时间
#     elapsed_time = end_time - start_time   
# elif model == 'MACN':
#     net = MACN(in_channels=sar_channel, num_classes=class_num).to(device)
#     start_time = time.time()
#     flops, params = thop.profile(net, inputs=(hsi_pca, sar)) 
#     end_time = time.time()    
#     # 计算运行时间
#     elapsed_time = end_time - start_time   
# elif model == 'MICF_Net':
#     net = MICF_Net(patch_size=window_size, dim=64, input_dim=30, num_classes=class_num, heads=8, num_atte=8, dep=2).to(device)
#     x_proto = torch.empty(7, 64).to(device)
#     torch.nn.init.normal_(x_proto, mean=0, std=0.2)
#     l_proto = torch.empty(7, 64).to(device)  ######### 生成类别的原型
#     torch.nn.init.normal_(l_proto, mean=0, std=0.2)
#     # tr_labels = (128,)
#     # tr_labels = torch.randint(low=0, high=6, size=tr_labels, dtype=torch.long).cuda(
#     start_time = time.time()
#     flops, params = thop.profile(net, inputs=(hsi_pca, sar, x_proto, l_proto, None))
#     end_time = time.time()
#     # 计算运行时间
#     elapsed_time = end_time - start_time    
# elif model == 'M2Fnet':
#     net = M2Fnet(FM=16, NC=pca_channel, LC=sar_channel, Classes=class_num).to(device)
#     start_time = time.time()
#     flops, params = thop.profile(net, inputs=(hsi_pca, sar)) # M2Fnet
#     end_time = time.time()
#     # 计算运行时间
#     elapsed_time = end_time - start_time


# net = EyeNet(datatype,device=device).to(device)

# net = M2Fnet(FM=16, NC=hsi_channel, LC=sar_channel, Classes=class_num).to(device)

net = MICF_Net(patch_size=window_size, dim=64, input_dim=hsi_channel, num_classes=class_num, heads=8, num_atte=8, dep=2).to(device)
x_proto = torch.empty(class_num, 64).to(device)
torch.nn.init.normal_(x_proto, mean=0, std=0.2)
l_proto = torch.empty(class_num, 64).to(device)  ######### 生成类别的原型
torch.nn.init.normal_(l_proto, mean=0, std=0.2)

# hsi = hsi.to(device)
# sar = sar.to(device)
start_time = time.time()
flops, params = thop.profile(net, inputs=(hsi, sar, x_proto, l_proto, tr_labels)) 
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Params: {params / 1e6} M")  # 打印参数量（以百万为单位）
print(f"FLOPS: {flops / 1e9} G")  # 打印计算量（以十亿次浮点运算为单位）  
print(f"运行时间: {elapsed_time:.4f} 秒")



