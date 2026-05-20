# HSI-X-Classify-Backbone

用于高光谱与辅助模态（LiDAR / SAR / MSI）融合分类的统一实验骨架。项目提供了统一的实验配置、数据加载、模型注册、训练、评估与可视化流程，便于在多个公开数据集上复现实验、横向对比不同模型，并继续扩展新模型。

---

## 1. 项目概览

本项目面向 HSI-X 多模态遥感分类任务，目标是把“数据集切换、模型切换、训练测试、结果落盘”统一到一套最小可维护框架中。

当前代码库已经将原先分散的参数修改逻辑收敛到统一配置系统中，核心特点如下：

- 统一实验配置：通过 `src/config.py` 中的 `ExperimentConfig` 管理数据集、模型、训练参数和输出路径。
- 统一模型注册：通过 `src/model_registry.py` 管理可用模型与适配器模块。
- 统一训练/测试入口：通过 `src/runner.py` 组织训练、测试、可视化、t-SNE 等流程。
- 统一数据接口：通过 `src/data.py` 适配多个多模态遥感数据集。
- 统一结果输出：模型权重、日志、报告、可视化图像按照模型名和实验参数自动归档。

项目适合以下场景：

- 复现实验室已有模型结果
- 在同一批数据集上对比多个 backbone
- 在已有框架中快速接入新模型
- 批量训练多模型 / 多数据集实验

---

## 2. 项目架构

### 2.1 目录结构

```text
HSI-X-classify-backbone/
├── configs/                  # 训练等脚本默认读取的配置文件目录
├── data/                     # 项目内数据目录（当前仓库下也保留了一份数据）
├── environment.yml           # 推荐 Conda 环境定义（CUDA 12.1 / Torch 2.4.1）
├── requirements.txt          # 精简 pip 依赖列表
├── Makefile                  # 常用命令封装
├── log/                      # 训练日志输出
├── model/                    # 模型权重输出
├── pic/                      # 分类可视化图输出
├── report/                   # 分类报告输出
├── scripts/                  # 训练/测试/批量运行脚本
└── src/
    ├── config.py             # 统一配置系统
    ├── data.py               # 数据读取、PCA、归一化、DataLoader
    ├── runner.py             # 实验调度入口
    ├── trainer.py            # 训练流程
    ├── evaluator.py          # 测试流程
    ├── metrics.py            # 分类指标与报告输出
    ├── analysis.py           # 分类图可视化
    ├── tsne.py               # t-SNE 分析
    ├── show_erf.py           # ERF 相关入口
    ├── visualize_erf.py      # ERF 可视化
    ├── model_registry.py     # 模型注册表
    └── models/
        ├── fusatnet/
        ├── exvit/
        ├── hybridsn/
        ├── micf_net/
        ├── cnn/
        ├── fdnet/
        ├── s2enet/
        ├── asyffnet/
        ├── hctnet/
        ├── macn/
        └── rscnet/
```

### 2.2 核心执行链路

一次完整实验的调用链路如下：

```text
scripts/train.py / scripts/test.py / scripts/train_all.py
        ↓
src.runner.ExperimentRunner
        ↓
src.config.ExperimentConfig / ConfigManager
        ↓
src.trainer.myTrain / src.evaluator.myTest
        ↓
src.data.getMyData
        ↓
src.model_registry.get_model_adapter
        ↓
src.models.<model>.adapter
        ↓
具体模型实现 + 指标统计 + 可视化输出
```

### 2.3 模块职责说明

#### 统一配置层
- [src/config.py](src/config.py)
  - 定义 `ExperimentConfig`
  - 维护数据集编号到名称的映射
  - 自动构建模型、日志、报告、图片输出路径
  - 提供 CLI 参数解析与配置文件保存/加载能力

#### 数据层
- [src/data.py](src/data.py)
  - 负责 `.mat` 数据读取
  - 对 HSI 做 PCA 降维
  - 对 HSI / 辅助模态做标准化
  - 生成训练、测试、可视化所需的 `DataLoader`
  - 内置 8 个数据集的数据文件约定与 key 约定

#### 模型适配层
- [src/model_registry.py](src/model_registry.py)
  - 将模型名映射到 `src.models.<name>.adapter`
- `src/models/*/adapter.py`
  - 负责屏蔽不同模型前向接口差异
  - 统一暴露 `build_model`、`forward_train`、`forward_eval`

#### 训练与评估层
- [src/trainer.py](src/trainer.py)
  - 训练、验证、最佳权重保存、日志记录
- [src/evaluator.py](src/evaluator.py)
  - 加载训练好的模型并生成报告
- [src/metrics.py](src/metrics.py)
  - 输出 OA / AA / Kappa / 分类报告 / 混淆矩阵
- [src/analysis.py](src/analysis.py)
  - 生成分类图像

---

## 3. 支持的数据集

当前配置中内置了 8 个数据集编号，定义见 [src/config.py:20-29](src/config.py#L20-L29) 与 [src/data.py:270-320](src/data.py#L270-L320)。

| 编号 | 数据集 | 类别数 | HSI 通道数 | 辅助模态通道数 | 辅助模态类型 |
|---|---|---:|---:|---:|---|
| 0 | Houston2013 | 15 | 144 | 1 | LiDAR |
| 1 | Houston2018 | 20 | 50 | 1 | LiDAR |
| 2 | Trento | 6 | 63 | 1 | LiDAR |
| 3 | Berlin | 8 | 244 | 4 | SAR |
| 4 | Augsburg | 7 | 180 | 4 | SAR |
| 5 | YellowRiverEstuary | 18 | 285 | 3 | SAR |
| 6 | LN01 | 10 | 166 | 8 | MSI |
| 7 | LN02 | 9 | 144 | 8 | MSI |

### 3.1 数据文件放置方式

数据应放在项目根目录下的 `data/` 目录中，即 `HSI-X-classify-backbone/data/`。

推荐按下面方式组织数据：

```text
HSI-X-classify-backbone/
├── data/
│   ├── Houston2013/
│   │   ├── houston_hsi.mat
│   │   ├── houston_lidar.mat
│   │   ├── houston_gt.mat
│   │   └── houston_index.mat
│   ├── Houston2018/
│   ├── Trento/
│   ├── Berlin/
│   ├── Augsburg/
│   ├── YellowRiverEstuary/
│   ├── LN01/
│   └── LN02/
├── scripts/
├── src/
└── README.md
```

### 3.2 各数据集文件名约定

| 数据集 | 必需文件 |
|---|---|
| Houston2013 | `houston_hsi.mat`, `houston_lidar.mat`, `houston_gt.mat`, `houston_index.mat` |
| Houston2018 | `houston_hsi.mat`, `houston_lidar.mat`, `houston_gt.mat`, `houston_index.mat` |
| Trento | `trento_hsi.mat`, `trento_lidar.mat`, `trento_gt.mat`, `trento_index.mat` |
| Berlin | `berlin_hsi.mat`, `berlin_sar.mat`, `berlin_gt.mat`, `berlin_index.mat` |
| Augsburg | `augsburg_hsi.mat`, `augsburg_sar.mat`, `augsburg_gt.mat`, `augsburg_index.mat` |
| YellowRiverEstuary | `data_hsi.mat`, `data_sar.mat`, `data_gt.mat`, `data_index.mat` |
| LN01 | `LN01_HHSI.mat`, `LN01_MSI.mat`, `LN01_GT.mat`, `LN01_INDEX.mat` |
| LN02 | `LN02_HHSI.mat`, `LN02_MSI.mat`, `LN02_GT.mat`, `LN02_INDEX.mat` |

---

## 4. 支持的模型

当前注册模型以 [src/model_registry.py:3-15](src/model_registry.py#L3-L15) 为准：

- `FusAtNet`
- `ExViT`
- `HybridSN`
- `MICF_Net`
- `CNN`
- `FDNet`
- `S2ENet`
- `AsyFFNet`
- `HCTNet`
- `MACN`
- `RSCNet`

每个模型目录通常包含：

```text
src/models/<model_name>/
├── adapter.py     # 与统一训练框架的接口桥接层
├── model.py       # 具体网络结构
└── __init__.py
```

适配器的作用可以参考：
- [src/models/fusatnet/adapter.py](src/models/fusatnet/adapter.py)
- [src/models/rscnet/adapter.py](src/models/rscnet/adapter.py)

其中：
- `build_model(...)` 负责根据数据集维度创建模型
- `forward_train(...)` 负责统一训练前向
- `forward_eval(...)` 负责统一推理前向

---

## 5. 环境安装

### 5.1 推荐环境

当前项目已提供一份较完整的 Conda 环境文件 [environment.yml](environment.yml)。

关键信息如下：

- Python：`3.10.20`
- PyTorch：`2.4.1`
- torchvision：`0.19.1`
- torchaudio：`2.4.1`
- CUDA 运行时：`cu121`
- cuDNN：`9.1.0`

也就是说，**当前依赖是面向 NVIDIA RTX 4090 常用环境的 CUDA 12.1 版本**，对应环境文件中这几项：

- `torch==2.4.1+cu121`
- `torchvision==0.19.1+cu121`
- `torchaudio==2.4.1+cu121`
- `nvidia-cuda-runtime-cu12==12.1.105`
- `nvidia-cudnn-cu12==9.1.0.70`

如果你当前机器是 4090，并且驱动支持 CUDA 12.x，这份环境是最优先建议使用的。

### 5.2 使用 Conda 安装（推荐）

在项目根目录执行：

```bash
conda env create -f environment.yml
conda activate HSI-X-Classify-Backbone
```

验证安装是否成功：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

如果输出中包含 `2.4.1+cu121` 且 GPU 可用，则说明环境匹配成功。

### 5.3 使用 pip 安装（备选）

项目也提供了一个较精简的 [requirements.txt](requirements.txt)，但它**没有把 CUDA 版本锁死**，因此更适合你已经自行准备好 PyTorch GPU 环境后再补齐其余依赖。

推荐顺序：

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# 先安装与你机器匹配的 PyTorch GPU 版本
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1

# 再安装其余依赖
pip install -r requirements.txt
```

如果你希望严格复现当前作者环境，优先使用 `environment.yml`。

### 5.4 常见环境说明

#### 4090 + CUDA 12.1
推荐直接使用仓库内 `environment.yml`。

#### 只有 CPU
可以运行，但训练和可视化会很慢。运行时显式指定：

```bash
python -m scripts.train --cuda cpu
```

#### 已有不同 CUDA 版本
如果你本机 PyTorch 来自其他 CUDA 版本，请注意：
- `requirements.txt` 不会自动帮你切到 `cu121`
- 不同 CUDA 版本下，最好重新安装与本机驱动匹配的 Torch 三件套

---

## 6. 运行方式

### 6.1 Makefile 方式

项目根目录提供了 [Makefile](Makefile)，可直接查看帮助：

```bash
make help
```

常用命令：

```bash
make activate
make train
make test
make strain
make stest
make train-all
```

说明：
- `make train` 调用 [scripts/train.py](scripts/train.py)
- `make test` 调用 [scripts/test.py](scripts/test.py)
- `make train-all` 调用 [scripts/train_all.py](scripts/train_all.py)

如需传参：

```bash
make train ARGS="--help"
```

> 注意：当前 `scripts/train.py` 是“脚本内写死配置”的风格，不是完整 CLI 风格；真正完整的 CLI 入口在 [src/config.py:303-367](src/config.py#L303-L367) 的 `create_experiment_config_from_cli()`。

### 6.2 直接运行脚本

#### 训练

```bash
python scripts/train.py
```

当前 `scripts/train.py` 的行为是：

- 默认优先查找 `configs/train.json`
- 如果该配置文件存在，则直接加载它作为训练配置
- 如果该配置文件不存在，则回退到脚本内置的 quick config
- 不再支持通过命令行逐项覆盖训练参数

当前内置 quick config 默认值为：
- 数据集：`Augsburg`
- 模型：`RSCNet`
- 学习率：`1e-4`
- Epochs：`100`
- Batch size：`128`
- 默认开启测试与可视化

常见示例：

```bash
# 直接运行：优先读取 configs/train.json，不存在则走 quick config
python scripts/train.py

# 指定另一份配置文件
python scripts/train.py --config configs/train_rscnet.json
```

如果你希望长期固定一套训练参数，推荐新建：

```text
configs/train.json
```

配置文件里只需要保留你真正想控制的实验参数，例如数据集、模型、学习率、epoch、batch size、窗口大小、是否可视化等。

像下面这些字段**不建议写进配置文件**，因为框架会自动推导：

- `out_features`
- `data_channels`
- `lidar_or_sar_channels`
- `depth`（如果你接受默认结构）
- `model_save_path`
- `log_path`
- `report_path`
- `image_path`
- `image_path_web`

其中输出路径会根据当前机器上的项目根目录、模型名、数据集名和实验参数自动生成，因此不要硬编码绝对路径。

#### 测试

```bash
python scripts/test.py
```

当前默认配置见 [scripts/test.py:12-23](scripts/test.py#L12-L23)：
- 数据集：`Houston2013`
- 模型：`RSCNet`
- Epochs：`10`
- 仅测试，不训练

#### 批量训练

```bash
python scripts/train_all.py
```

可用参数：

```bash
python scripts/train_all.py -h
```

常见示例：

```bash
# 对全部模型、全部数据集训练
python scripts/train_all.py

# 只跑指定模型
python scripts/train_all.py --models RSCNet FusAtNet --epochs 100 --cuda cuda:0

# 只跑指定数据集（可用编号或名称）
python scripts/train_all.py --datasets 0 4 LN01

# 指定实验名前缀
python scripts/train_all.py --experiment-prefix paper_baseline
```

`scripts/train_all.py` 会自动检查数据是否存在，缺失时跳过，不会直接崩掉。

### 6.3 Python API 方式

项目支持程序化调用，可直接使用 [src/runner.py](src/runner.py) 中的接口。

#### 快速运行

```python
from src.runner import quick_run

quick_run(
    dataset_type=0,
    model_name="FusAtNet",
    lr=1e-4,
    epochs=10,
    channels=30,
    window_size=11,
    cuda_device="cuda:0",
    enable_visualization=True,
    enable_testing=True,
    experiment_name="quick_demo",
)
```

#### 基于配置对象运行

```python
from src.config import ExperimentConfig
from src.runner import ExperimentRunner

config = ExperimentConfig(
    experiment_name="custom_exp",
    dataset_type=4,
    model_name="RSCNet",
    learning_rate=1e-4,
    epochs=100,
    batch_size=128,
    channels=30,
    window_size=11,
    cuda_device="cuda:0",
    enable_training=True,
    enable_testing=True,
    enable_visualization=True,
    enable_tsne=False,
)

runner = ExperimentRunner(config)
runner.print_config()
runner.run_experiment()
```

### 6.4 CLI 配置方式

虽然 `scripts/train.py` / `scripts/test.py` 当前仍以脚本硬编码配置为主，但 [src/config.py](src/config.py) 已提供完整的命令行配置解析能力。你可以自行增加一个轻量入口，例如：

```python
from src.config import create_experiment_config_from_cli
from src.runner import ExperimentRunner

config = create_experiment_config_from_cli()
runner = ExperimentRunner(config)
runner.run_experiment()
```

其支持的主要参数包括：

- `--experiment-name`
- `--dataset`
- `--model`
- `--cuda`
- `--lr`
- `--epochs`
- `--batch-size`
- `--channels`
- `--window-size`
- `--no-visualization`
- `--tsne`
- `--train-only`
- `--test-only`
- `--config`
- `--save-config`

---

## 7. 实验配置说明

### 7.1 核心配置项

核心配置定义在 [src/config.py:31-70](src/config.py#L31-L70)。

| 配置项 | 含义 | 默认值 |
|---|---|---|
| `experiment_name` | 实验名，用于输出文件命名 | `default_experiment` |
| `dataset_type` | 数据集编号 | `3` |
| `model_name` | 模型名 | `FusAtNet` |
| `cuda_device` | 运行设备 | `cuda:0` |
| `learning_rate` | 学习率 | `0.0001` |
| `epochs` | 训练轮数 | `10` |
| `batch_size` | Batch size | `128` |
| `num_workers` | DataLoader worker 数 | `0` |
| `random_seed` | 随机种子 | `6` |
| `channels` | PCA 后保留通道数 | `30` |
| `window_size` | 邻域窗口大小 | `11` |
| `enable_training` | 是否训练 | `True` |
| `enable_testing` | 是否测试 | `True` |
| `enable_visualization` | 是否生成分类图 | `True` |
| `enable_tsne` | 是否执行 t-SNE | `False` |
| `enable_erf` | 是否执行 ERF | `False` |

### 7.2 配置文件书写建议

`configs/train.json` 这类配置文件建议只保存“实验可变参数”，不要把框架能够自动推导的信息重复写进去。

推荐保留的字段通常包括：

- `experiment_name`
- `dataset_type`
- `model_name`
- `cuda_device`
- `learning_rate`
- `epochs`
- `batch_size`
- `num_workers`
- `random_seed`
- `channels`
- `window_size`
- `enable_training`
- `enable_testing`
- `enable_visualization`
- `enable_tsne`
- `enable_erf`

不建议手写的字段包括：

- `out_features`
- `data_channels`
- `lidar_or_sar_channels`
- `depth`（除非你确实要改默认结构）
- `model_save_path`
- `log_path`
- `report_path`
- `image_path`
- `image_path_web`

原因是这些字段会在 `ExperimentConfig` 初始化时，根据当前项目路径、数据集配置和实验参数自动补齐。尤其是各种输出路径，跨机器时不应该写死为绝对路径。

一个推荐的最小训练配置示例如下：

```json
{
  "experiment_name": "train_experiment",
  "dataset_type": 4,
  "model_name": "RSCNet",
  "cuda_device": "cuda:0",
  "learning_rate": 0.0001,
  "epochs": 100,
  "batch_size": 128,
  "num_workers": 0,
  "random_seed": 6,
  "channels": 30,
  "window_size": 11,
  "enable_training": true,
  "enable_testing": true,
  "enable_visualization": true,
  "enable_tsne": false,
  "enable_erf": false
}
```

### 7.3 输出文件命名规则

输出路径由 [src/config.py:97-122](src/config.py#L97-L122) 自动生成，目录按模型区分：

- `model/<ModelName>/...pth`
- `log/<ModelName>/...txt`
- `report/<ModelName>/...txt`
- `pic/<ModelName>/...png`

默认后缀形式为：

```text
_pca={channels}_window={window_size}_lr={learning_rate}_epochs={epochs}
```

如果指定 `experiment_name != default_experiment`，则会额外加上：

```text
_{experiment_name}
```

示例：

```text
model/RSCNet/Augsburg_model_train_experiment_pca=30_window=11_lr=0.0001_epochs=100.pth
log/RSCNet/Augsburg_log_train_experiment_pca=30_window=11_lr=0.0001_epochs=100.txt
report/RSCNet/Augsburg_report_train_experiment_pca=30_window=11_lr=0.0001_epochs=100.txt
pic/RSCNet/Augsburg_train_experiment_pca=30_window=11_lr=0.0001_epochs=100.png
```

### 7.3 推荐实验起步配置

#### 快速冒烟

```python
ExperimentConfig(
    dataset_type=0,
    model_name="FusAtNet",
    learning_rate=1e-4,
    epochs=1,
    batch_size=32,
    channels=30,
    window_size=11,
    enable_visualization=False,
)
```

#### 正式单模型实验

```python
ExperimentConfig(
    dataset_type=4,
    model_name="RSCNet",
    learning_rate=1e-4,
    epochs=100,
    batch_size=128,
    channels=30,
    window_size=11,
    enable_visualization=True,
    enable_testing=True,
)
```

#### 批量对比实验

建议使用：

```bash
python scripts/train_all.py --models RSCNet FusAtNet ExViT --datasets 0 4 5 --epochs 100
```

---

## 8. 输出结果说明

### 8.1 模型权重

训练时会在验证精度提升后保存最佳模型：

- 输出目录：`model/<模型名>/`
- 保存逻辑见 [src/trainer.py:97-100](src/trainer.py#L97-L100)

### 8.2 日志文件

训练日志由 [src/trainer.py:15-23](src/trainer.py#L15-L23) 写入，包含：

- 当前实验配置
- 训练开始时间 / 结束时间
- 每个 epoch 的 loss
- 每个 epoch 的测试集精度
- 最佳精度

### 8.3 测试报告

分类报告由 [src/metrics.py](src/metrics.py) 输出，包含：

- OA（Overall Accuracy）
- AA（Average Accuracy）
- Kappa
- 每类精度
- `classification_report`
- confusion matrix

### 8.4 可视化结果

分类图生成逻辑在 [src/analysis.py](src/analysis.py)。

不同数据集已内置：
- 图像尺寸
- 类别颜色映射

生成图片默认保存在：

```text
pic/<ModelName>/<Dataset>...png
```

### 8.5 t-SNE / ERF

- `t-SNE`：由 `enable_tsne` 控制，调用 [src/evaluator.py:64-65](src/evaluator.py#L64-L65)
- `ERF`：配置项中已预留 `enable_erf`，并存在 [src/show_erf.py](src/show_erf.py)、[src/visualize_erf.py](src/visualize_erf.py) 相关代码，但当前主流程集成度相对较低，建议按需单独接入

---

## 9. 开发者指南

### 9.1 如何新增一个模型

推荐按照现有模型目录结构新增：

```text
src/models/my_model/
├── adapter.py
├── model.py
└── __init__.py
```

#### 第一步：实现网络结构
在 `model.py` 中实现你的 PyTorch 模型类。

#### 第二步：实现适配器
参考 [src/models/fusatnet/adapter.py](src/models/fusatnet/adapter.py) 或 [src/models/rscnet/adapter.py](src/models/rscnet/adapter.py)，至少实现以下接口：

```python
def build_model(config, dataset_type, device):
    ...
    return {"net": net}


def forward_train(bundle, batch):
    ...
    return logits


def forward_eval(bundle, batch):
    ...
    return logits
```

`batch` 中可直接使用的字段通常有：
- `batch["hsi_pca"]`
- `batch["hsi"]`
- `batch["aux"]`
- `batch["label"]`

如果你的模型只用 PCA-HSI + 辅助模态，可参考 FusAtNet；
如果同时使用原始 HSI、PCA-HSI 和辅助模态，可参考 RSCNet。

#### 第三步：注册模型
编辑 [src/model_registry.py](src/model_registry.py)，加入：

```python
MODEL_REGISTRY = {
    ...,
    "MyModel": "src.models.my_model.adapter",
}
```

#### 第四步：验证训练链路
最小验证建议：

```python
from src.runner import quick_run

quick_run(
    dataset_type=0,
    model_name="MyModel",
    epochs=1,
    enable_visualization=False,
    enable_testing=True,
)
```

### 9.2 如何新增一个数据集

新增数据集时，至少需要修改以下部分：

1. 在 [src/config.py](src/config.py) 中补充：
   - `DATASET_LABELS`
   - `out_features`
   - `data_channels`
   - `lidar_or_sar_channels`

2. 在 [src/data.py](src/data.py) 中补充：
   - 数据文件路径构造
   - `.mat` key 映射
   - `get<DatasetName>Data(...)`
   - `getMyData(...)` 中的数据集分发逻辑

3. 在 [src/metrics.py](src/metrics.py) 中补充：
   - 类别名称列表
   - `get<DatasetName>Report(...)`
   - `getMyReport(...)` 分发

4. 在 [src/analysis.py](src/analysis.py) 中补充：
   - 颜色映射
   - 图像尺寸
   - `vis<DatasetName>(...)`
   - `getMyVisualization(...)` 分发

5. 如果使用批量训练脚本，还需要更新 [scripts/train_all.py](scripts/train_all.py) 中的 `DATASET_REQUIRED_FILES`

### 9.3 如何新增命令行入口

如果你不想继续手改 `scripts/train.py` / `scripts/test.py` 里的配置，推荐新建一个更通用的入口，例如：

```python
from src.config import create_experiment_config_from_cli
from src.runner import ExperimentRunner

if __name__ == "__main__":
    config = create_experiment_config_from_cli()
    runner = ExperimentRunner(config)
    runner.print_config()
    runner.run_experiment()
```

这样就能直接使用统一 CLI 参数，无需每次编辑脚本文件。

### 9.4 开发建议

- 优先通过 `ExperimentConfig` 注入参数，不要回到散落的硬编码方式
- 模型差异尽量放在 `adapter.py` 中吸收，避免污染统一训练器
- 新增数据集时同步补齐报告与可视化，不然主流程虽然能跑，但结果不完整
- 正式实验前先做 1 epoch 冒烟验证，避免长时间训练后才发现数据 key 或 shape 不匹配

---

## 10. 常见问题

### 10.1 找不到数据文件

先确认数据目录是否放在工作区同级 `data/` 下，而不是只放在仓库内。

可快速检查：

```bash
python -c "from src.data import DATA_ROOT; print(DATA_ROOT)"
```

### 10.2 GPU 不可用

检查：

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

如果不可用：
- 检查 NVIDIA 驱动是否正确安装
- 检查当前 Torch 是否为 GPU 版本
- 检查环境是否与 CUDA 12.1 / cu121 匹配

### 10.3 显存不足

可以优先降低：
- `batch_size`
- `window_size`
- `channels`

例如：

```python
ExperimentConfig(
    batch_size=32,
    window_size=9,
    channels=15,
)
```

### 10.4 训练可以跑但测试加载失败

测试阶段通过 [src/evaluator.py:55](src/evaluator.py#L55) 直接 `torch.load(model_savepath)` 加载权重，因此要确认：
- 对应模型文件已经生成
- 测试配置与训练配置一致
- 输出路径未因 `experiment_name`、`channels`、`window_size`、`lr`、`epochs` 不一致而改变

---

## 11. 建议的首次使用流程

如果你第一次接手这个项目，建议按下面顺序：

1. 创建并激活 Conda 环境
2. 按约定放好一个数据集（推荐 Houston2013 或 Augsburg）
3. 修改 [scripts/train.py](scripts/train.py) 中配置为 1 epoch 冒烟测试
4. 运行 `python scripts/train.py`
5. 检查 `model/`、`log/`、`report/`、`pic/` 是否正常生成输出
6. 再开始切换模型或批量实验

---

## 12. 代码参考入口

若需要进一步阅读源码，建议从以下文件开始：

- [src/config.py](src/config.py)：先理解配置系统
- [src/runner.py](src/runner.py)：再看主流程如何组织
- [src/data.py](src/data.py)：理解数据格式和输入张量
- [src/trainer.py](src/trainer.py)：理解训练保存逻辑
- [src/evaluator.py](src/evaluator.py)：理解测试与可视化逻辑
- [src/model_registry.py](src/model_registry.py)：理解模型如何被统一接入

---

