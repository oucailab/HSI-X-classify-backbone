import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import DATASET_LABELS, create_quick_config
from src.data import DATA_ROOT
from src.model_registry import MODEL_REGISTRY
from src.runner import ExperimentRunner


DATASET_REQUIRED_FILES = {
    0: [
        ("Houston2013", "houston_hsi.mat"),
        ("Houston2013", "houston_lidar.mat"),
        ("Houston2013", "houston_gt.mat"),
        ("Houston2013", "houston_index.mat"),
    ],
    1: [
        ("Houston2018", "houston_hsi.mat"),
        ("Houston2018", "houston_lidar.mat"),
        ("Houston2018", "houston_gt.mat"),
        ("Houston2018", "houston_index.mat"),
    ],
    2: [
        ("Trento", "trento_hsi.mat"),
        ("Trento", "trento_lidar.mat"),
        ("Trento", "trento_gt.mat"),
        ("Trento", "trento_index.mat"),
    ],
    3: [
        ("Berlin", "berlin_hsi.mat"),
        ("Berlin", "berlin_sar.mat"),
        ("Berlin", "berlin_gt.mat"),
        ("Berlin", "berlin_index.mat"),
    ],
    4: [
        ("Augsburg", "augsburg_hsi.mat"),
        ("Augsburg", "augsburg_sar.mat"),
        ("Augsburg", "augsburg_gt.mat"),
        ("Augsburg", "augsburg_index.mat"),
    ],
    5: [
        ("YellowRiverEstuary", "data_hsi.mat"),
        ("YellowRiverEstuary", "data_sar.mat"),
        ("YellowRiverEstuary", "data_gt.mat"),
        ("YellowRiverEstuary", "data_index.mat"),
    ],
    6: [
        ("LN01", "LN01_HHSI.mat"),
        ("LN01", "LN01_MSI.mat"),
        ("LN01", "LN01_GT.mat"),
        ("LN01", "LN01_INDEX.mat"),
    ],
    7: [
        ("LN02", "LN02_HHSI.mat"),
        ("LN02", "LN02_MSI.mat"),
        ("LN02", "LN02_GT.mat"),
        ("LN02", "LN02_INDEX.mat"),
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description="批量训练全部模型与数据集")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--lr", type=float, default=0.0001, help="学习率")
    parser.add_argument("--batch-size", type=int, default=128, help="批次大小")
    parser.add_argument("--cuda", type=str, default="cuda:0", help="CUDA设备")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_REGISTRY.keys()),
        help="指定要运行的模型，默认全部",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="指定要运行的数据集，可传名称或编号，默认全部",
    )
    parser.add_argument(
        "--experiment-prefix",
        type=str,
        default="train_all",
        help="实验名前缀",
    )
    return parser.parse_args()


def resolve_dataset_types(dataset_args):
    if not dataset_args:
        return sorted(DATASET_LABELS.keys())

    name_to_id = {name.lower(): dataset_id for dataset_id, name in DATASET_LABELS.items()}
    resolved = []
    for item in dataset_args:
        if item.isdigit():
            dataset_id = int(item)
            if dataset_id not in DATASET_LABELS:
                raise ValueError(f"未知数据集编号: {item}")
            resolved.append(dataset_id)
            continue

        dataset_id = name_to_id.get(item.lower())
        if dataset_id is None:
            raise ValueError(f"未知数据集名称: {item}")
        resolved.append(dataset_id)

    return list(dict.fromkeys(resolved))


def get_missing_dataset_files(dataset_type):
    missing = []
    for folder, filename in DATASET_REQUIRED_FILES[dataset_type]:
        path = Path(DATA_ROOT) / folder / filename
        if not path.exists():
            missing.append(path)
    return missing


def build_config(args, dataset_type, dataset_name, model_name):
    config = create_quick_config(
        experiment_name=f"{args.experiment_prefix}_{model_name}_{dataset_name}",
        dataset_type=dataset_type,
        model_name=model_name,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        enable_visualization=True,
        enable_testing=True,
    )
    config.cuda_device = args.cuda
    config.enable_training = True
    config.enable_testing = True
    config.enable_visualization = True
    return config


def main():
    args = parse_args()
    model_names = args.models or sorted(MODEL_REGISTRY.keys())
    dataset_types = resolve_dataset_types(args.datasets)

    success = []
    skipped = []
    failed = []

    print(f"Data root: {DATA_ROOT}")
    print(f"Models: {', '.join(model_names)}")
    print(f"Datasets: {', '.join(DATASET_LABELS[dataset_id] for dataset_id in dataset_types)}")
    print("=" * 80)

    for model_name in model_names:
        for dataset_type in dataset_types:
            dataset_name = DATASET_LABELS[dataset_type]
            run_name = f"{model_name} / {dataset_name}"
            print(f"\n>>> Running {run_name}")

            missing_files = get_missing_dataset_files(dataset_type)
            if missing_files:
                skipped.append(
                    {
                        "model": model_name,
                        "dataset": dataset_name,
                        "missing_files": [str(path) for path in missing_files],
                    }
                )
                print(f"Skip {run_name}: missing dataset files")
                for path in missing_files:
                    print(f"  - {path}")
                continue

            try:
                config = build_config(args, dataset_type, dataset_name, model_name)
                runner = ExperimentRunner(config)
                runner.print_config()
                runner.run_experiment()
                success.append({"model": model_name, "dataset": dataset_name})
            except Exception as exc:
                failed.append({"model": model_name, "dataset": dataset_name, "error": str(exc)})
                print(f"Failed {run_name}: {exc}")

    print("\n" + "=" * 80)
    print("Batch summary")
    print(f"Success: {len(success)}")
    for item in success:
        print(f"  - {item['model']} / {item['dataset']}")

    print(f"Skipped: {len(skipped)}")
    for item in skipped:
        print(f"  - {item['model']} / {item['dataset']}")

    print(f"Failed: {len(failed)}")
    for item in failed:
        print(f"  - {item['model']} / {item['dataset']}: {item['error']}")


if __name__ == "__main__":
    main()
