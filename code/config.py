"""
统一配置管理系统
解决原有 task.py 和 parameter.py 多文件修改的问题
"""

import os
import json
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

@dataclass
class ExperimentConfig:
    """实验配置类，包含所有需要的参数"""
    
    # 基本实验设置
    experiment_name: str = "default_experiment"
    dataset_type: int = 0  # 0-7 对应不同数据集
    model_name: str = "FusAtNet"
    cuda_device: str = "cuda:0"
    
    # 训练参数
    learning_rate: float = 0.0001
    epochs: int = 10
    batch_size: int = 128
    num_workers: int = 4
    random_seed: int = 6
    
    # 网络参数
    channels: int = 30
    window_size: int = 11
    depth: List[List[int]] = None
    
    # 功能开关
    enable_training: bool = True
    enable_testing: bool = True
    enable_visualization: bool = True
    enable_tsne: bool = False
    enable_erf: bool = False
    
    # 数据集信息 (自动根据dataset_type设置)
    out_features: List[int] = None
    data_channels: List[int] = None
    lidar_or_sar_channels: List[int] = None
    
    # 路径配置 (自动生成)
    model_save_path: str = ""
    log_path: str = ""
    report_path: str = ""
    image_path: str = ""
    image_path_web: str = ""
    
    def __post_init__(self):
        """初始化后自动设置默认值"""
        if self.depth is None:
            self.depth = [[2, 2, 2], [2, 2, 2], 2]
            
        # 数据集相关配置
        self._setup_dataset_config()
        
        # 自动生成路径
        self._setup_paths()
    
    def _setup_dataset_config(self):
        """根据数据集类型设置相关配置"""
        # 各数据集的输出特征数
        if self.out_features is None:
            self.out_features = [15, 20, 6, 8, 7, 18, 10, 9]
        
        # 各数据集的原始波段数
        if self.data_channels is None:
            self.data_channels = [144, 50, 63, 244, 180, 285, 166, 144]
        
        # 各数据集的LiDAR/SAR波段数
        if self.lidar_or_sar_channels is None:
            self.lidar_or_sar_channels = [1, 1, 1, 4, 4, 3, 8, 8]
    
    def _setup_paths(self):
        """自动生成文件路径"""
        dataset_names = [
            "Houston2013", "Houston2018", "Trento", "Berlin", 
            "Augsburg", "YellowRiverEstuary", "LN01", "LN02"
        ]
        
        dataset_name = dataset_names[self.dataset_type]
        
        # 生成文件名后缀
        suffix = f"_pca={self.channels}_window={self.window_size}_lr={self.learning_rate}_epochs={self.epochs}"
        if self.experiment_name != "default_experiment":
            suffix = f"_{self.experiment_name}{suffix}"
        
        # 设置路径，为模型创建单独的文件夹
        self.model_save_path = f"../model/{self.model_name}/{dataset_name}_model{suffix}.pth"
        self.log_path = f"../log/{self.model_name}/{dataset_name}_log{suffix}.txt"
        self.report_path = f"../report/{self.model_name}/{dataset_name}_report{suffix}.txt"
        self.image_path = f"../pic/{self.model_name}/{dataset_name}{suffix}.png"
        self.image_path_web = f"static/images/{self.model_name}/{dataset_name}{suffix}.png"


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config: Optional[ExperimentConfig] = None):
        self.config = config or ExperimentConfig()
        self._parameter_dict = self._create_parameter_dict()
    
    def _create_parameter_dict(self) -> Dict[str, Any]:
        """创建兼容原parameter.py的字典"""
        return {
            # 网络参数
            'channels': self.config.channels,
            'windowSize': self.config.window_size,
            'out_features': self.config.out_features,
            'depth': self.config.depth,
            
            # 训练参数
            'cuda': self.config.cuda_device,
            'lr': self.config.learning_rate,
            'epoch_nums': self.config.epochs,
            'batch_size': self.config.batch_size,
            'num_workers': self.config.num_workers,
            'random_seed': self.config.random_seed,
            
            # 功能开关
            'visualization': self.config.enable_visualization,
            'tsne': self.config.enable_tsne,
            'erf': self.config.enable_erf,
            
            # 数据集信息
            'data_channels': self.config.data_channels,
            'lidar_or_sar_channels': self.config.lidar_or_sar_channels,
            
            # 路径配置 (保持原有格式兼容性)
            'model_savepath': [self.config.model_save_path] * 8,  # 兼容原来的列表格式
            'log_path': [self.config.log_path] * 8,
            'report_path': [self.config.report_path] * 8,
            'image_path': [self.config.image_path] * 8,
            'image_path_web': [self.config.image_path_web] * 8,
        }
    
    def get_value(self, key: str) -> Any:
        """获取参数值，兼容原parameter.py接口"""
        try:
            return self._parameter_dict[key]
        except KeyError:
            print(f'读取{key}失败')
            return None
    
    def set_value(self, key: str, value: Any) -> None:
        """设置参数值，兼容原parameter.py接口"""
        self._parameter_dict[key] = value
        
        # 同时更新config对象
        if key == 'channels':
            self.config.channels = value
        elif key == 'windowSize':
            self.config.window_size = value
        elif key == 'lr':
            self.config.learning_rate = value
        elif key == 'epoch_nums':
            self.config.epochs = value
        elif key == 'cuda':
            self.config.cuda_device = value
        elif key == 'visualization':
            self.config.enable_visualization = value
        elif key == 'tsne':
            self.config.enable_tsne = value
        elif key == 'erf':
            self.config.enable_erf = value
        
        # 重新创建参数字典以确保同步
        self._parameter_dict = self._create_parameter_dict()
    
    def get_taskInfo(self) -> str:
        """获取任务信息，兼容原parameter.py接口"""
        return (
            '-----------------------taskInfo-----------------------\n'
            f'lr:\t{self.config.learning_rate}\n'
            f'epoch_nums:\t{self.config.epochs}\n'
            f'batch_size:\t{self.config.batch_size}\n'
            f'window_size:\t{self.config.window_size}\n'
            f'depth:\t{self.config.depth}\n'
            '------------------------------------------------------'
        )
    
    def save_config(self, path: str) -> None:
        """保存配置到文件"""
        config_dict = asdict(self.config)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    def load_config(self, path: str) -> None:
        """从文件加载配置"""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        self.config = ExperimentConfig(**config_dict)
        self._parameter_dict = self._create_parameter_dict()
    
    def create_from_args(self, args: argparse.Namespace) -> None:
        """从命令行参数创建配置"""
        config_dict = {}
        
        # 映射命令行参数到配置字段
        arg_mapping = {
            'experiment_name': 'experiment_name',
            'dataset': 'dataset_type',
            'model': 'model_name',
            'cuda': 'cuda_device',
            'lr': 'learning_rate',
            'epochs': 'epochs',
            'batch_size': 'batch_size',
            'channels': 'channels',
            'window_size': 'window_size',
        }
        
        for arg_name, config_field in arg_mapping.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                config_dict[config_field] = getattr(args, arg_name)
        
        # 处理功能开关
        if hasattr(args, 'no_visualization') and args.no_visualization:
            config_dict['enable_visualization'] = False
        
        if hasattr(args, 'tsne') and args.tsne:
            config_dict['enable_tsne'] = True
        
        # 处理训练测试选项
        if hasattr(args, 'train_only') and args.train_only:
            config_dict['enable_training'] = True
            config_dict['enable_testing'] = False
        elif hasattr(args, 'test_only') and args.test_only:
            config_dict['enable_training'] = False
            config_dict['enable_testing'] = True
        
        self.config = ExperimentConfig(**config_dict)
        self._parameter_dict = self._create_parameter_dict()


# 全局配置管理器实例
_global_config_manager: Optional[ConfigManager] = None

def _init(config: Optional[ExperimentConfig] = None) -> None:
    """初始化全局配置管理器，兼容原parameter.py"""
    global _global_config_manager
    _global_config_manager = ConfigManager(config)

def get_value(key: str) -> Any:
    """获取参数值，兼容原parameter.py接口"""
    if _global_config_manager is None:
        _init()
    return _global_config_manager.get_value(key)

def set_value(key: str, value: Any) -> None:
    """设置参数值，兼容原parameter.py接口"""
    if _global_config_manager is None:
        _init()
    _global_config_manager.set_value(key, value)

def get_taskInfo() -> str:
    """获取任务信息，兼容原parameter.py接口"""
    if _global_config_manager is None:
        _init()
    return _global_config_manager.get_taskInfo()

def get_task_info() -> str:
    """获取任务信息，兼容原parameter.py接口 (下划线版本)"""
    if _global_config_manager is None:
        _init()
    return _global_config_manager.get_taskInfo()

def get_config_manager() -> ConfigManager:
    """获取全局配置管理器"""
    if _global_config_manager is None:
        _init()
    return _global_config_manager

def create_experiment_config_from_cli() -> ExperimentConfig:
    """从命令行创建实验配置"""
    parser = argparse.ArgumentParser(description='深度学习实验配置')
    
    # 基本设置
    parser.add_argument('--experiment-name', type=str, default='default_experiment',
                       help='实验名称')
    parser.add_argument('--dataset', type=int, default=0, choices=range(8),
                       help='数据集类型 (0-7)')
    parser.add_argument('--model', type=str, default='FusAtNet',
                       help='模型名称')
    parser.add_argument('--cuda', type=str, default='cuda:0',
                       help='CUDA设备')
    
    # 训练参数
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='学习率')
    parser.add_argument('--epochs', type=int, default=10,
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='批次大小')
    parser.add_argument('--channels', type=int, default=30,
                       help='PCA通道数')
    parser.add_argument('--window-size', type=int, default=11,
                       help='窗口大小')
    
    # 功能开关
    parser.add_argument('--no-visualization', action='store_true',
                       help='禁用可视化')
    parser.add_argument('--tsne', action='store_true',
                       help='启用t-SNE')
    parser.add_argument('--train-only', action='store_true',
                       help='仅训练')
    parser.add_argument('--test-only', action='store_true',
                       help='仅测试')
    
    # 配置文件
    parser.add_argument('--config', type=str,
                       help='配置文件路径')
    parser.add_argument('--save-config', type=str,
                       help='保存配置到文件')
    
    args = parser.parse_args()
    
    # 如果指定了配置文件，优先加载
    if args.config and Path(args.config).exists():
        manager = ConfigManager()
        manager.load_config(args.config)
        config = manager.config
    else:
        # 从命令行参数创建配置
        manager = ConfigManager()
        manager.create_from_args(args)
        config = manager.config
    
    # 如果指定了保存路径，保存配置
    if args.save_config:
        manager = ConfigManager(config)
        manager.save_config(args.save_config)
        print(f"配置已保存到: {args.save_config}")
    
    return config

# 示例配置创建函数
def create_quick_config(dataset_type: int = 0, 
                       model_name: str = "FusAtNet",
                       lr: float = 0.0001, 
                       epochs: int = 10,
                       enable_visualization: bool = True,
                       enable_tsne: bool = False,
                       experiment_name: str = "quick_experiment") -> ExperimentConfig:
    """快速创建实验配置"""
    return ExperimentConfig(
        experiment_name=experiment_name,
        dataset_type=dataset_type,
        model_name=model_name,
        learning_rate=lr,
        epochs=epochs,
        enable_visualization=enable_visualization,
        enable_tsne=enable_tsne
    )

if __name__ == "__main__":
    # 命令行运行示例
    config = create_experiment_config_from_cli()
    print("创建的配置:")
    print(f"实验名称: {config.experiment_name}")
    print(f"数据集类型: {config.dataset_type}")
    print(f"模型: {config.model_name}")
    print(f"学习率: {config.learning_rate}")
    print(f"训练轮数: {config.epochs}")
