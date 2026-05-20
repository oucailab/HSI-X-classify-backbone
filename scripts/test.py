import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import create_quick_config
from src.runner import ExperimentRunner


if __name__ == "__main__":
    config = create_quick_config(
        experiment_name="train_experiment",
        dataset_type=0,
        model_name="RSCNet",
        lr=0.0001,
        epochs=10,
        batch_size=128,
        enable_visualization=False,
        enable_testing=True,
    )
    config.enable_training = False
    config.enable_testing = True
    runner = ExperimentRunner(config)
    runner.print_config()
    runner.run_experiment()
