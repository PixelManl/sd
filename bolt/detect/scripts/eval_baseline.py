from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan or execute the bolt detection baseline evaluation entrypoint."
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
        help="Training run directory or evaluation output directory.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        help="Checkpoint placeholder to evaluate.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="val",
        help="Dataset split to evaluate.",
    )
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, help="Optional eval image size override.")
    parser.add_argument("--device", help="Optional device override.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute Ultralytics YOLO evaluation instead of printing a plan.",
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
    if args.weights and not args.weights.exists():
        raise FileNotFoundError(f"Weights file not found: {args.weights}")


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
    return (base_dir / "eval").resolve()


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


def build_eval_job(args: argparse.Namespace, config: dict[str, Any] | None) -> dict[str, Any]:
    if args.weights is None:
        raise ValueError("Weights are required for execute mode.")
    dataset_root = resolve_dataset_root(args, config)
    if dataset_root is None:
        raise ValueError("Dataset root is required for execute mode.")
    dataset_yaml = resolve_dataset_yaml(dataset_root)
    run_dir = resolve_run_dir(args, config)
    project, name = split_ultralytics_dir(run_dir)
    kwargs = {
        "data": str(dataset_yaml),
        "split": args.split,
        "conf": args.conf_threshold,
        "iou": args.iou_threshold,
        "max_det": get_config_value(config, "eval", "max_det", default=100),
        "imgsz": args.imgsz if args.imgsz is not None else get_config_value(config, "model", "image_size", default=1024),
        "device": args.device if args.device is not None else get_config_value(config, "model", "device", default="auto"),
        "project": project,
        "name": name,
        "exist_ok": True,
    }
    return {
        "model": str(args.weights.resolve()),
        "dataset_root": dataset_root,
        "dataset_yaml": dataset_yaml,
        "run_dir": run_dir,
        "metrics_path": run_dir / "metrics.json",
        "kwargs": kwargs,
    }


def collect_metrics_payload(metrics: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    save_dir = getattr(metrics, "save_dir", None)
    if save_dir is not None:
        payload["save_dir"] = str(Path(save_dir).resolve())
    results_dict = getattr(metrics, "results_dict", None)
    if isinstance(results_dict, dict):
        payload["results_dict"] = results_dict
    speed = getattr(metrics, "speed", None)
    if isinstance(speed, dict):
        payload["speed"] = speed
    box_metrics = getattr(metrics, "box", None)
    if box_metrics is not None:
        summary: dict[str, Any] = {}
        for key in ("map", "map50", "map75", "mp", "mr"):
            value = getattr(box_metrics, key, None)
            if isinstance(value, (int, float)):
                summary[key] = value
        if summary:
            payload["box"] = summary
    return payload


def export_metrics(metrics_payload: dict[str, Any], metrics_path: Path) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics_payload, indent=2) + "\n", encoding="utf-8")


def build_plan(args: argparse.Namespace, config: dict[str, Any] | None) -> dict[str, Any]:
    ultralytics_available = importlib.util.find_spec("ultralytics") is not None
    dataset_root = resolve_dataset_root(args, config)
    run_dir = resolve_run_dir(args, config)
    dataset_yaml = dataset_root / "dataset.yaml" if dataset_root else None
    return {
        "task": "eval_baseline",
        "status": "dry-run" if args.dry_run or not args.execute else "execute-requested",
        "config_path": str(args.config.resolve()),
        "config_loaded": config is not None,
        "yaml_available": importlib.util.find_spec("yaml") is not None,
        "ultralytics_available": ultralytics_available,
        "dataset_root": str(dataset_root) if dataset_root else None,
        "dataset_yaml": str(dataset_yaml.resolve()) if dataset_yaml and dataset_yaml.exists() else None,
        "run_dir": str(run_dir.resolve()),
        "weights": str(args.weights.resolve()) if args.weights else None,
        "split": args.split,
        "conf_threshold": args.conf_threshold,
        "iou_threshold": args.iou_threshold,
        "imgsz": args.imgsz if args.imgsz is not None else get_config_value(config, "model", "image_size", default=1024),
        "device": args.device if args.device is not None else get_config_value(config, "model", "device", default="auto"),
        "metrics_path": str((run_dir / "metrics.json").resolve()),
        "execute_ready": ultralytics_available and dataset_yaml is not None and args.weights is not None,
        "next_action": "Run with --execute to validate and export metrics.json.",
        "guidance": [
            "Use the held-out split to inspect miss-rate and false alarms for missing_fastener.",
            "Keep metric outputs under bolt/detect/runs/ and do not commit them.",
        ],
    }


def execute_evaluation(args: argparse.Namespace, config: dict[str, Any] | None) -> dict[str, Any]:
    job = build_eval_job(args, config)
    job["run_dir"].mkdir(parents=True, exist_ok=True)
    yolo_class = load_yolo_class()
    model = yolo_class(job["model"])
    metrics = model.val(**job["kwargs"])
    metrics_payload = collect_metrics_payload(metrics)
    export_metrics(metrics_payload, job["metrics_path"])
    return {
        "task": "eval_baseline",
        "status": "executed",
        "model": job["model"],
        "dataset_yaml": str(job["dataset_yaml"]),
        "run_dir": str(job["run_dir"]),
        "metrics_path": str(job["metrics_path"].resolve()),
        "eval_kwargs": job["kwargs"],
        "metrics": metrics_payload,
    }


def main() -> int:
    args = parse_args()
    validate_args(args)
    config = maybe_load_yaml(args.config)
    if args.execute and not args.dry_run:
        payload = execute_evaluation(args, config)
    else:
        payload = build_plan(args, config)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
