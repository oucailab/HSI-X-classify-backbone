# 🚀 HSI-X 分类深度学习实验框架

## 📋 项目简介

本项目是一个用于多源遥感数据融合分类的深度学习框架，支持多种深度学习模型，包括 FusAtNet等。经过架构重构，项目现在采用统一配置管理系统，解决了原有多文件修改的不便问题。（环境可以自己配，提供的requirements.txt比较冗余）

## 📁 项目文件夹结构

### 整体项目结构

```
data/
├── [DatasetName]/              # 按数据集名称分类
│   ├── [datasetname]_gt.mat/            # ground truth
│   ├── [datasetname]_hsi.mat/            # hyperspectral image
│   ├── [datasetname]_[x].mat/            # sar/lidar
│   └── [datasetname]_index.mat/            # 训练集与测试集索引
HSI-X-classify-backbone/
├── README.md                    # 项目说明文档
├── requirements.txt             # Python依赖包列表
├── code/                        # 源代码目录
│
├── model/                      # 训练后的模型文件
│   └── [ModelName]/            # 按模型名称分类
│       └── [Dataset]_model_pca=[channels]_window=[size]_lr=[rate]_epochs=[num].pth
│
├── log/                        # 训练日志文件
│   └── [ModelName]/            # 按模型名称分类
│       └── [Dataset]_log_pca=[channels]_window=[size]_lr=[rate]_epochs=[num].txt
│
├── report/                     # 测试报告文件
│   └── [ModelName]/            # 按模型名称分类
│       └── [Dataset]_report_pca=[channels]_window=[size]_lr=[rate]_epochs=[num].txt
│
└── pic/                        # 可视化结果图像
    └── [ModelName]/            # 按模型名称分类
        └── [Dataset]_pca=[channels]_window=[size]_lr=[rate]_epochs=[num].png
```

## 🚀 快速开始

### 1. 基本运行方式

#### 方式一：命令行运行（推荐）
```bash
cd backbone/code

# 基本运行 - 使用默认参数
python unified_runner.py

# 自定义参数运行
python unified_runner.py --dataset 0 --lr 0.0001 --epochs 10 --channels 30

# 指定 GPU 设备
python unified_runner.py --dataset 0 --cuda cuda:1 --epochs 5

# 禁用可视化（默认启用）
python unified_runner.py --dataset 0 --no-visualization --epochs 1
```

#### 方式二：Python 函数调用
```python
from unified_runner import quick_run

# 快速运行实验
quick_run(
    dataset_type=0,        # Houston2013 数据集
    model_name="FusAtNet",   # 模型名称
    lr=0.0001,            # 学习率
    epochs=10,            # 训练轮数
    cuda_device="cuda:0",  # GPU 设备
    enable_visualization=True  # 启用可视化
)
```

#### 方式三：配置文件方式
```python
import config
from unified_runner import ExperimentRunner

# 创建自定义配置
experiment_config = config.ExperimentConfig(
    experiment_name="my_experiment",
    dataset_type=0,
    model_name="FusAtNet",
    learning_rate=0.0001,
    epochs=10,
    channels=30,
    window_size=11,
    enable_visualization=True
)

# 运行实验
runner = ExperimentRunner(experiment_config)
runner.run_experiment()
```

### 2. 命令行参数详解

| 参数 | 类型 | 默认值 | 说明 |
|------|------|-------|------|
| `--dataset` | int | 0 | 数据集类型 (0-7) |
| `--model` | str | "FusAtNet" | 模型名称 (支持多种模型) |
| `--lr` | float | 0.0001 | 学习率 |
| `--epochs` | int | 10 | 训练轮数 |
| `--batch-size` | int | 128 | 批次大小 |
| `--channels` | int | 30 | PCA 降维后的通道数 |
| `--window-size` | int | 11 | 数据窗口大小 |
| `--cuda` | str | "cuda:0" | GPU 设备 |
| `--experiment-name` | str | "default_experiment" | 实验名称 |
| `--no-visualization` | flag | False | 禁用结果可视化 |
| `--tsne` | flag | False | 启用 t-SNE 分析 |
| `--train-only` | flag | False | 仅进行训练 |
| `--test-only` | flag | False | 仅进行测试 |

### 3. 数据集对应关系

| 编号 | 数据集名称 | 类别数 | 描述 | 
|------|-----------|-------|------|
| 0 | [Houston2013](https://ieeexplore.ieee.org/document/6776408) | 15 | 休斯顿2013数据集 |
| 1 | [Houston2018](https://ieeexplore.ieee.org/document/8328995) | 20 | 休斯顿2018数据集 |
| 2 | [Trento](https://ieeexplore.ieee.org/document/8000656) | 6 | 特伦托数据集 |
| 3 | [Berlin](https://dataservices.gfz-potsdam.de/enmap/showshort.php?id=escidoc:1823890) | 8 | 柏林数据集 |
| 4 | [Augsburg](https://github.com/zhu-xlab/augsburg_Multimodal_Data_Set_MDaS/blob/main/README.md) | 7 | 奥格斯堡数据集 |
| 5 | [YellowRiverEstuary](https://ieeexplore.ieee.org/document/9494718) | 18 | 黄河口数据集 |
| 6 | [LN01](https://ieeexplore.ieee.org/document/10703123) | 10 | 辽宁01数据集 |
| 7 | [LN02](https://ieeexplore.ieee.org/document/10703123) | 9 | 辽宁02数据集 |

### 4. 支持的模型

| 模型名称 | 特殊说明 |
|---------|----------|
| FusAtNet | **默认模型** |
| ExViT | - |
| HybridSN | - |
| FDNet | - |
| MICF_Net | - |
| M2FNet | - |
| S2ENet | - |
| DFINet | - |
| AsyFFNet | - |
| HCTNet | - |
| MACN | - |
| TBCNN | - |
| CNN | - |


## 🎯 使用场景示例

### 场景一：快速验证模型
```bash
# 用 Houston2013 数据集快速训练 1 轮验证模型
python unified_runner.py --dataset 0 --epochs 1 
```

### 场景二：完整训练实验
```bash
# 完整训练 Houston2013，50轮，学习率 0.0005
python unified_runner.py --dataset 0 --epochs 50 --lr 0.0005 --experiment-name "final_training"
```

### 场景三：多数据集对比实验
```bash
# 对比不同数据集的性能
python unified_runner.py --dataset 0 --epochs 20 --experiment-name "houston2013_comparison"
python unified_runner.py --dataset 4 --epochs 20 --experiment-name "augsburg_comparison"
python unified_runner.py --dataset 5 --epochs 20 --experiment-name "yellowriver_comparison"
```

### 场景四：参数调优实验
```bash
# 不同学习率对比
python unified_runner.py --dataset 0 --lr 0.0001 --epochs 10 --experiment-name "lr_0001"
python unified_runner.py --dataset 0 --lr 0.0005 --epochs 10 --experiment-name "lr_0005"
python unified_runner.py --dataset 0 --lr 0.001 --epochs 10 --experiment-name "lr_001"

# 不同窗口大小对比
python unified_runner.py --dataset 0 --window-size 9 --epochs 10 --experiment-name "window_9"
python unified_runner.py --dataset 0 --window-size 11 --epochs 10 --experiment-name "window_11"
python unified_runner.py --dataset 0 --window-size 13 --epochs 10 --experiment-name "window_13"
```

## 🔧 高级功能

### 1. 配置文件保存和加载
```bash
# 保存当前配置到文件
python unified_runner.py --dataset 0 --lr 0.0001 --save-config my_config.json

# 从配置文件加载运行
python unified_runner.py --config my_config.json
```

### 2. 程序化配置管理
```python
import config

# 初始化配置系统
config._init()

# 获取配置值
learning_rate = config.get_value('lr')
cuda_device = config.get_value('cuda')

# 设置配置值
config.set_value('lr', 0.0005)
config.set_value('visualization', True)

# 获取任务信息
task_info = config.get_task_info()
print(task_info)
```

### 3. 自定义实验配置
```python
import config

# 创建自定义配置
custom_config = config.ExperimentConfig(
    experiment_name="custom_experiment",
    dataset_type=0,
    model_name="EyeNet",
    learning_rate=0.0002,
    epochs=25,
    batch_size=256,
    channels=40,
    window_size=13,
    enable_visualization=True,
    enable_tsne=True
)

# 保存配置
manager = config.ConfigManager(custom_config)
manager.save_config("custom_experiment.json")
```

## 📊 实验结果查看

### 1. 训练日志
```bash
# 查看最新的训练日志
tail -f backbone/log/EyeNet/Houston2013_log_pca=30_window=11_lr=0.0001_epochs=10.txt
```

### 2. 测试报告
测试报告包含详细的分类性能指标：
- Overall Accuracy (OA)
- Average Accuracy (AA) 
- Kappa 系数
- 各类别准确率
- 混淆矩阵

## 🚨 常见问题解决

### 1. CUDA 相关错误
```bash
# 检查可用的 GPU
python -c "import torch; print(torch.cuda.device_count())"

# 使用 CPU 运行
python unified_runner.py --cuda cpu --dataset 0 --epochs 1
```

### 2. 内存不足
```bash
# 减小批次大小
python unified_runner.py --batch-size 64 --dataset 0

# 减小窗口大小
python unified_runner.py --window-size 9 --dataset 0
```

### 3. 数据路径问题
确保数据集放在正确的位置：
```
data/
├── Houston2013/
├── Houston2018/
├── Augsburg/
└── ...
```

## 📝 开发者指南

### 添加新模型
1. 在 `net.py` 中定义新模型类
2. 在`train.py`,`test.py`中添加模型初始化逻辑
3. 更新配置中的 `model_name` 选项

### 添加新数据集
1. 在 `dataset.py`,`report.py`,`visualization.py` 中添加数据加载逻辑
2. 更新 `config.py` 中的数据集配置

