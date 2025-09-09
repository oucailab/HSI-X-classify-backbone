import torch

# 加载模型
model_path = '../model/Houston2013_model_pca=30_window=11_layers=3_ls=31_g=8_nlevels=4.pth'
model = torch.load(model_path, map_location=torch.device('cpu'))

# 输出文件路径
output_file = '../model/model_info.txt'

# 重定向print输出到文件
with open(output_file, 'w') as f:
    # 保存模型基本信息
    print("=== 模型基本信息 ===", file=f)
    print(f"模型类型: {type(model)}", file=f)
    
    # 检查模型类型并保存相应信息
    if isinstance(model, torch.nn.Module):
        print("\n=== 模型结构 ===", file=f)
        print(model, file=f)
        
        print("\n=== 模型参数 ===", file=f)
        for name, param in model.named_parameters():
            print(f"Layer: {name} | Size: {param.size()}", file=f)
            if len(param.shape) > 0:
                print(f"  前2个值: {param.flatten()[:2].tolist()}", file=f)
            else:
                print(f"  值: {param.item()}", file=f)
                
    elif isinstance(model, dict):
        print("\n=== 模型状态字典 ===", file=f)
        for key in model.keys():
            print(f"{key}: {model[key].size()}", file=f)
            if len(model[key].shape) > 0:
                print(f"  前5个值: {model[key].flatten()[:5].tolist()}", file=f)
            else:
                print(f"  值: {model[key].item()}", file=f)
    else:
        print("\n未知模型格式", file=f)

print(f"模型信息已保存到 {output_file}")