import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import ConfigManager, DATASET_LABELS, create_quick_config
from src.runner import ExperimentRunner


DEFAULT_CONFIG_PATH = Path(PROJECT_ROOT) / "configs" / "train.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="训练入口：优先加载配置文件，否则使用 quick config"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="配置文件路径；文件存在时优先加载",
    )
    return parser.parse_args()


def build_default_config():
    config = create_quick_config(
        experiment_name="train_experiment",
        dataset_type=4,
        model_name="RSCNet",
        lr=0.0001,
        epochs=100,
        batch_size=128,
        enable_visualization=True,
        enable_testing=True,
    )
    config.enable_training = True
    config._setup_paths()
    return config


def load_base_config(config_path):
    manager = ConfigManager()
    manager.load_config(config_path)
    config = manager.config
    config.enable_training = True
    config._setup_paths()
    return config


if __name__ == "__main__":
    args = parse_args()
    config_path = Path(args.config)

    if config_path.exists():
        config = load_base_config(config_path)
        print(f"基础配置来源: {config_path}")
    else:
        config = build_default_config()
        print("基础配置来源: quick config")

    if config.dataset_type in DATASET_LABELS:
        print(f"当前数据集: {DATASET_LABELS[config.dataset_type]} ({config.dataset_type})")

    runner = ExperimentRunner(config)
    runner.print_config()
    runner.run_experiment()
 