"""
Backbone 项目新配置系统使用示例
演示各种使用方式和配置选项
"""

import os
import sys

# 确保能导入本地模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import ExperimentConfig, ConfigManager, create_quick_config
from unified_runner import ExperimentRunner, quick_run

def example_1_quick_run():
    """示例1: 使用quick_run快速运行实验"""
    print("=" * 60)
    print("示例1: 快速运行实验")
    print("=" * 60)
    
    # 最简单的方式：使用默认参数快速运行
    quick_run(
        dataset_type=0,           # Houston2013数据集
        model_name="EyeNet",      # EyeNet模型
        lr=0.0001,               # 学习率
        epochs=1,                # 仅训练1轮用于演示
        enable_visualization=True, # 启用可视化
        enable_training=False,    # 演示时禁用训练，仅测试
        enable_testing=True,      # 启用测试
        experiment_name="demo_experiment"
    )

def example_2_config_class():
    """示例2: 使用配置类创建和运行实验"""
    print("=" * 60)
    print("示例2: 使用配置类")
    print("=" * 60)
    
    # 创建详细的配置
    config = ExperimentConfig(
        experiment_name="houston_detailed_test",
        dataset_type=0,
        model_name="EyeNet",
        learning_rate=0.0001,
        epochs=1,
        batch_size=64,           # 自定义批次大小
        channels=20,             # 自定义PCA通道数
        window_size=9,           # 自定义窗口大小
        enable_training=False,   # 仅测试
        enable_testing=True,
        enable_visualization=True,
        enable_tsne=False
    )
    
    # 打印配置信息
    runner = ExperimentRunner(config)
    runner.print_config()
    
    # 运行实验
    runner.run_experiment()

def example_3_config_file():
    """示例3: 使用配置文件"""
    print("=" * 60)
    print("示例3: 配置文件方式")
    print("=" * 60)
    
    # 创建配置
    config = ExperimentConfig(
        experiment_name="augsburg_experiment",
        dataset_type=4,          # Augsburg数据集
        model_name="EyeNet",
        learning_rate=0.0005,
        epochs=1,
        enable_training=False,
        enable_testing=True,
        enable_visualization=True
    )
    
    # 保存配置到文件
    config_manager = ConfigManager(config)
    config_file = "example_config.json"
    config_manager.save_config(config_file)
    print(f"配置已保存到: {config_file}")
    
    # 从文件加载配置并运行
    new_manager = ConfigManager()
    new_manager.load_config(config_file)
    
    runner = ExperimentRunner(new_manager.config)
    runner.print_config()
    runner.run_experiment()
    
    # 清理示例文件
    if os.path.exists(config_file):
        os.remove(config_file)
        print(f"已删除示例配置文件: {config_file}")

def example_4_multiple_datasets():
    """示例4: 批量运行多个数据集"""
    print("=" * 60)
    print("示例4: 批量运行多个数据集")
    print("=" * 60)
    
    # 定义要测试的数据集
    datasets = [
        (0, "Houston2013"),
        (4, "Augsburg"),
        (5, "YellowRiverEstuary")
    ]
    
    for dataset_type, dataset_name in datasets:
        print(f"\n开始运行数据集: {dataset_name}")
        print("-" * 40)
        
        quick_run(
            dataset_type=dataset_type,
            model_name="EyeNet",
            lr=0.0001,
            epochs=1,
            enable_training=False,  # 仅测试
            enable_testing=True,
            enable_visualization=True,
            experiment_name=f"batch_{dataset_name.lower()}"
        )

def example_5_parameter_sweep():
    """示例5: 参数扫描实验"""
    print("=" * 60)
    print("示例5: 参数扫描实验")
    print("=" * 60)
    
    # 定义要测试的学习率
    learning_rates = [0.0001, 0.0005, 0.001]
    # 定义要测试的PCA通道数
    channels_list = [20, 30, 40]
    
    for lr in learning_rates:
        for channels in channels_list:
            print(f"\n测试参数: lr={lr}, channels={channels}")
            print("-" * 40)
            
            config = ExperimentConfig(
                experiment_name=f"sweep_lr{lr}_ch{channels}",
                dataset_type=0,  # Houston2013
                model_name="EyeNet",
                learning_rate=lr,
                epochs=1,
                channels=channels,
                enable_training=False,  # 仅测试用于演示
                enable_testing=True,
                enable_visualization=False  # 批量实验时禁用可视化
            )
            
            runner = ExperimentRunner(config)
            runner.run_experiment()

def example_6_legacy_compatibility():
    """示例6: 与原有代码的兼容性"""
    print("=" * 60)
    print("示例6: 兼容原有代码")
    print("=" * 60)
    
    # 方式1: 使用原有的myTask接口
    from unified_runner import myTask
    
    print("使用兼容的myTask接口:")
    myTask(
        lr=0.0001,
        epoch_nums=1,
        datasetType=0,
        cuda='cuda:0',
        net='EyeNet',
        visualization=True,
        tsne=False
    )
    
    # 方式2: 使用新的config接口
    print("\n使用新的config接口:")
    import config
    config._init()
    config.set_value('lr', 0.0001)
    config.set_value('epoch_nums', 1)
    config.set_value('cuda', 'cuda:0')
    config.set_value('visualization', True)
    
    print(f"学习率: {config.get_value('lr')}")
    print(f"训练轮数: {config.get_value('epoch_nums')}")
    print(config.get_taskInfo())

def example_7_advanced_configuration():
    """示例7: 高级配置选项"""
    print("=" * 60)
    print("示例7: 高级配置选项")
    print("=" * 60)
    
    # 创建高级配置
    config = ExperimentConfig(
        experiment_name="advanced_experiment",
        dataset_type=1,          # Houston2018
        model_name="EyeNet",
        learning_rate=0.0001,
        epochs=1,
        batch_size=256,          # 大批次
        num_workers=8,           # 更多工作进程
        channels=50,             # 更多PCA通道
        window_size=13,          # 更大窗口
        enable_training=False,
        enable_testing=True,
        enable_visualization=True,
        enable_tsne=True,        # 启用t-SNE
        enable_erf=False         # 禁用ERF
    )
    
    # 显示配置详情
    runner = ExperimentRunner(config)
    runner.print_config()
    
    # 手动访问配置管理器
    config_manager = runner.config_manager
    print("\n配置管理器信息:")
    print(f"任务信息: {config_manager.get_taskInfo()}")
    print(f"模型保存路径: {config.model_save_path}")
    print(f"日志路径: {config.log_path}")
    
    # 运行实验
    runner.run_experiment()

def main():
    """主函数：运行所有示例"""
    print("Backbone 项目新配置系统使用示例")
    print("=" * 60)
    
    examples = [
        ("快速运行", example_1_quick_run),
        ("配置类", example_2_config_class),
        ("配置文件", example_3_config_file),
        ("批量数据集", example_4_multiple_datasets),
        ("参数扫描", example_5_parameter_sweep),
        ("兼容性", example_6_legacy_compatibility),
        ("高级配置", example_7_advanced_configuration)
    ]
    
    print("可用示例:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\n选择要运行的示例 (输入数字，或按回车运行第1个示例):")
    try:
        choice = input().strip()
        if not choice:
            choice = "1"
        
        choice = int(choice)
        if 1 <= choice <= len(examples):
            name, func = examples[choice - 1]
            print(f"\n运行示例: {name}")
            func()
        else:
            print("无效的选择")
    except (ValueError, KeyboardInterrupt):
        print("\n运行第1个示例作为默认演示:")
        example_1_quick_run()

if __name__ == "__main__":
    main()
