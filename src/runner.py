from src.config import ExperimentConfig, ConfigManager, create_quick_config, create_experiment_config_from_cli

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class ExperimentRunner:
    """实验运行器"""

    def __init__(self, config=None):
        self.config = config or ExperimentConfig()
        self.config_manager = ConfigManager(self.config)

        import src.config as config_module
        config_module._init(self.config)

    def run_experiment(self) -> None:
        print(f"开始运行实验: {self.config.experiment_name}")
        print(f"数据集: {self._get_dataset_name()}")
        print(f"模型: {self.config.model_name}")
        print("-" * 50)

        try:
            if self.config.enable_training:
                print("开始训练...")
                self.run_training()
                print("训练完成!")

            if self.config.enable_testing:
                print("开始测试...")
                self.run_testing()
                print("测试完成!")

            print(f"实验 {self.config.experiment_name} 完成!")

        except Exception as e:
            print(f"实验运行出错: {str(e)}")
            raise

    def run_training(self) -> None:
        from src.trainer import myTrain
        print(self.config_manager.get_taskInfo())
        myTrain(self.config.dataset_type, self.config.model_name)

    def run_testing(self) -> None:
        from src.evaluator import myTest
        myTest(self.config.dataset_type, self.config.model_name)

    def _get_dataset_name(self) -> str:
        dataset_names = [
            "Houston2013", "Houston2018", "Trento", "Berlin",
            "Augsburg", "YellowRiverEstuary", "LN01", "LN02"
        ]
        return dataset_names[self.config.dataset_type]

    def print_config(self) -> None:
        print("当前实验配置:")
        print(f"  实验名称: {self.config.experiment_name}")
        print(f"  数据集: {self._get_dataset_name()} (类型: {self.config.dataset_type})")
        print(f"  模型: {self.config.model_name}")
        print(f"  学习率: {self.config.learning_rate}")
        print(f"  训练轮数: {self.config.epochs}")
        print(f"  批次大小: {self.config.batch_size}")
        print(f"  PCA通道数: {self.config.channels}")
        print(f"  窗口大小: {self.config.window_size}")
        print(f"  CUDA设备: {self.config.cuda_device}")
        print(f"  启用训练: {self.config.enable_training}")
        print(f"  启用测试: {self.config.enable_testing}")
        print(f"  启用可视化: {self.config.enable_visualization}")
        print(f"  启用t-SNE: {self.config.enable_tsne}")
        print(f"  模型保存路径: {self.config.model_save_path}")
        print("-" * 50)


def quick_run(dataset_type: int = 0,
              model_name: str = "FusAtNet",
              lr: float = 0.0001,
              epochs: int = 1,
              channels: int = 30,
              window_size: int = 11,
              cuda_device: str = "cuda:0",
              enable_visualization: bool = True,
              enable_tsne: bool = False,
              enable_training: bool = True,
              enable_testing: bool = True,
              experiment_name: str = "quick_experiment") -> None:
    config = create_quick_config(
        dataset_type=dataset_type,
        model_name=model_name,
        lr=lr,
        epochs=epochs,
        enable_visualization=enable_visualization,
        enable_tsne=enable_tsne,
        experiment_name=experiment_name
    )

    config.cuda_device = cuda_device
    config.enable_training = enable_training
    config.enable_testing = enable_testing
    config.channels = channels
    config.window_size = window_size
    config._setup_paths()

    runner = ExperimentRunner(config)
    runner.print_config()
    runner.run_experiment()


def myTask(lr: float, epoch_nums: int, datasetType: int,
           cuda: str = 'cuda:0',
           net: str = 'FusAtNet',
           visualization: bool = True,
           tsne: bool = False) -> None:
    config = ExperimentConfig(
        dataset_type=datasetType,
        model_name=net,
        learning_rate=lr,
        epochs=epoch_nums,
        cuda_device=cuda,
        enable_visualization=visualization,
        enable_tsne=tsne,
        enable_training=True,
        enable_testing=True,
        experiment_name=f"legacy_task_{datasetType}"
    )

    runner = ExperimentRunner(config)
    runner.run_experiment()
