from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan or execute the bolt detection baseline training entrypoint."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("bolt/detect/configs/baseline.yaml"),
        help="Baseline config file.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Prepared detection dataset root.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Local output directory for training artifacts.",
    )
    parser.add_argument(
        "--weights",
        default="yolo11n.pt",
        help="Backbone or checkpoint placeholder.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute Ultralytics YOLO training instead of printing a plan.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Force plan-only output.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.config.exists() or not args.config.is_file():
        raise FileNotFoundError(f"Config file not found: {args.config}")
    if args.dataset_root and not args.dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {args.dataset_root}")


def maybe_load_yaml(path: Path) -> dict[str, Any] | None:
    if importlib.util.find_spec("yaml") is None:
        return None
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return payload if isinstance(payload, dict) else {}


def default_run_dir() -> Path:
    return Path("bolt/detect/runs/baseline")


def get_config_value(config: dict[str, Any] | None, *keys: str, default: Any = None) -> Any:
    current: Any = config or {}
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def resolve_dataset_root(args: argparse.Namespace, config: dict[str, Any] | None) -> Path | None:
    if args.dataset_root:
        return args.dataset_root.resolve()
    prepared_root = get_config_value(config, "paths", "prepared_root")
    if isinstance(prepared_root, str) and prepared_root:
        return Path(prepared_root).resolve()
    return None


def resolve_run_dir(args: argparse.Namespace, config: dict[str, Any] | None) -> Path:
    if args.run_dir:
        return args.run_dir.resolve()
    run_root = get_config_value(config, "paths", "run_root")
    base_dir = Path(run_root) if isinstance(run_root, str) and run_root else default_run_dir()
    return (base_dir / "train").resolve()


def resolve_dataset_yaml(dataset_root: Path) -> Path:
    dataset_yaml = dataset_root / "dataset.yaml"
    if not dataset_yaml.exists() or not dataset_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")
    return dataset_yaml.resolve()


def split_ultralytics_dir(target_dir: Path) -> tuple[str, str]:
    resolved = target_dir.resolve()
    return str(resolved.parent), resolved.name


def load_yolo_class():
    if importlib.util.find_spec("ultralytics") is None:
        raise ModuleNotFoundError(
            "Ultralytics is required for --execute. Install it with: python -m pip install ultralytics"
        )
    from ultralytics import YOLO

    return YOLO


def build_train_job(args: argparse.Namespace, config: dict[str, Any] | None) -> dict[str, Any]:
    dataset_root = resolve_dataset_root(args, config)
    if dataset_root is None:
        raise ValueError("Dataset root is required for execute mode.")
    dataset_yaml = resolve_dataset_yaml(dataset_root)
    run_dir = resolve_run_dir(args, config)
    project, name = split_ultralytics_dir(run_dir)
    model_name = args.weights or get_config_value(config, "model", "model_name", default="yolo11n.pt")

    kwargs = {
        "data": str(dataset_yaml),
        "epochs": args.epochs,
        "batch": args.batch_size,
        "imgsz": args.imgsz,
        "device": args.device,
        "workers": get_config_value(config, "train", "workers", default=4),
        "patience": get_config_value(config, "train", "patience", default=20),
        "pretrained": get_config_value(config, "train", "pretrained", default=True),
        "project": project,
        "name": name,
        "exist_ok": True,
    }
    return {
        "model": model_name,
        "dataset_root": dataset_root,
        "dataset_yaml": dataset_yaml,
        "run_dir": run_dir,
        "kwargs": kwargs,
    }


def summarize_train_result(result: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    save_dir = getattr(result, "save_dir", None)
    if save_dir is not None:
        payload["save_dir"] = str(Path(save_dir).resolve())
    results_dict = getattr(result, "results_dict", None)
    if isinstance(results_dict, dict):
        payload["results_dict"] = results_dict
    speed = getattr(result, "speed", None)
    if isinstance(speed, dict):
        payload["speed"] = speed
    return payload


def build_plan(args: argparse.Namespace, config: dict[str, Any] | None) -> dict[str, Any]:
    ultralytics_available = importlib.util.find_spec("ultralytics") is not None
    dataset_root = resolve_dataset_root(args, config)
    run_dir = resolve_run_dir(args, config)
    dataset_yaml = dataset_root / "dataset.yaml" if dataset_root else None
    return {
        "task": "train_baseline",
        "status": "dry-run" if args.dry_run or not args.execute else "execute-requested",
        "config_path": str(args.config.resolve()),
        "config_loaded": config is not None,
        "yaml_available": importlib.util.find_spec("yaml") is not None,
        "ultralytics_available": ultralytics_available,
        "dataset_root": str(dataset_root) if dataset_root else None,
        "dataset_yaml": str(dataset_yaml.resolve()) if dataset_yaml and dataset_yaml.exists() else None,
        "run_dir": str(run_dir.resolve()),
        "weights": args.weights or get_config_value(config, "model", "model_name", default="yolo11n.pt"),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "imgsz": args.imgsz,
        "device": args.device,
        "execute_ready": ultralytics_available and dataset_yaml is not None and dataset_yaml.exists(),
        "next_action": "Run with --execute after dataset.yaml is materialized.",
        "guidance": [
            "Install PyYAML to inspect config values inside the CLI: python -m pip install pyyaml",
            "Install ultralytics before execute mode: python -m pip install ultralytics",
        ],
    }


def execute_training(args: argparse.Namespace, config: dict[str, Any] | None) -> dict[str, Any]:
    job = build_train_job(args, config)
    job["run_dir"].mkdir(parents=True, exist_ok=True)
    yolo_class = load_yolo_class()
    model = yolo_class(job["model"])
    result = model.train(**job["kwargs"])
    payload = {
        "task": "train_baseline",
        "status": "executed",
        "model": job["model"],
        "dataset_yaml": str(job["dataset_yaml"]),
        "run_dir": str(job["run_dir"]),
        "train_kwargs": job["kwargs"],
    }
    payload.update(summarize_train_result(result))
    return payload


def main() -> int:
    args = parse_args()
    validate_args(args)
    config = maybe_load_yaml(args.config)
    if args.execute and not args.dry_run:
        payload = execute_training(args, config)
    else:
        payload = build_plan(args, config)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
