import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import create_quick_config
from src.runner import ExperimentRunner


if __name__ == "__main__":
    from src.evaluator import myTest

    config = create_quick_config(experiment_name="smoke_training", dataset_type=0, model_name="RSCNet")
    config.enable_training = False
    config.enable_testing = True
    runner = ExperimentRunner(config)
    runner.print_config()
    myTest(config.dataset_type, config.model_name, max_test_samples=100)
