from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan or execute the bolt detection baseline inference entrypoint."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("bolt/detect/configs/baseline.yaml"),
        help="Baseline config file.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Image file or directory to inspect.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        help="Checkpoint placeholder for inference.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for local prediction artifacts.",
    )
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, help="Optional inference image size override.")
    parser.add_argument("--device", help="Optional device override.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute Ultralytics YOLO inference instead of printing a plan.",
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
    if not args.input.exists():
        raise FileNotFoundError(f"Input path not found: {args.input}")
    if args.weights and not args.weights.exists():
        raise FileNotFoundError(f"Weights file not found: {args.weights}")


def maybe_load_yaml(path: Path) -> dict[str, Any] | None:
    if importlib.util.find_spec("yaml") is None:
        return None
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return payload if isinstance(payload, dict) else {}


def default_output_dir() -> Path:
    return Path("bolt/detect/runs/infer")


def get_config_value(config: dict[str, Any] | None, *keys: str, default: Any = None) -> Any:
    current: Any = config or {}
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def resolve_output_dir(args: argparse.Namespace, config: dict[str, Any] | None) -> Path:
    if args.output_dir:
        return args.output_dir.resolve()
    predict_root = get_config_value(config, "paths", "predict_root")
    base_dir = Path(predict_root) if isinstance(predict_root, str) and predict_root else default_output_dir()
    return (base_dir / "predictions").resolve()


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


def build_infer_job(args: argparse.Namespace, config: dict[str, Any] | None) -> dict[str, Any]:
    if args.weights is None:
        raise ValueError("Weights are required for execute mode.")
    output_dir = resolve_output_dir(args, config)
    project, name = split_ultralytics_dir(output_dir)
    kwargs = {
        "source": str(args.input.resolve()),
        "conf": args.conf_threshold,
        "imgsz": args.imgsz if args.imgsz is not None else get_config_value(config, "model", "image_size", default=1024),
        "device": args.device if args.device is not None else get_config_value(config, "model", "device", default="auto"),
        "save": True,
        "save_txt": get_config_value(config, "infer", "save_txt", default=False),
        "save_conf": get_config_value(config, "infer", "save_conf", default=False),
        "save_crop": get_config_value(config, "infer", "save_crop", default=False),
        "project": project,
        "name": name,
        "exist_ok": True,
    }
    return {
        "model": str(args.weights.resolve()),
        "input": args.input.resolve(),
        "output_dir": output_dir,
        "summary_path": output_dir / "predictions.json",
        "kwargs": kwargs,
    }


def count_boxes(boxes: Any) -> int:
    if boxes is None:
        return 0
    if hasattr(boxes, "__len__"):
        return len(boxes)
    data = getattr(boxes, "data", None)
    if data is not None and hasattr(data, "__len__"):
        return len(data)
    shape = getattr(boxes, "shape", None)
    if isinstance(shape, tuple) and shape:
        return int(shape[0])
    return 0


def summarize_predictions(results: list[Any]) -> dict[str, Any]:
    predictions = []
    for result in results:
        source_path = getattr(result, "path", None)
        predictions.append(
            {
                "path": str(Path(source_path).resolve()) if source_path else None,
                "box_count": count_boxes(getattr(result, "boxes", None)),
            }
        )
    return {
        "prediction_count": len(predictions),
        "predictions": predictions,
    }


def export_prediction_summary(summary: dict[str, Any], summary_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def build_plan(args: argparse.Namespace, config: dict[str, Any] | None) -> dict[str, Any]:
    ultralytics_available = importlib.util.find_spec("ultralytics") is not None
    output_dir = resolve_output_dir(args, config)
    return {
        "task": "infer_baseline",
        "status": "dry-run" if args.dry_run or not args.execute else "execute-requested",
        "config_path": str(args.config.resolve()),
        "config_loaded": config is not None,
        "yaml_available": importlib.util.find_spec("yaml") is not None,
        "ultralytics_available": ultralytics_available,
        "input": str(args.input.resolve()),
        "input_kind": "directory" if args.input.is_dir() else "file",
        "weights": str(args.weights.resolve()) if args.weights else None,
        "output_dir": str(output_dir.resolve()),
        "summary_path": str((output_dir / "predictions.json").resolve()),
        "conf_threshold": args.conf_threshold,
        "imgsz": args.imgsz if args.imgsz is not None else get_config_value(config, "model", "image_size", default=1024),
        "device": args.device if args.device is not None else get_config_value(config, "model", "device", default="auto"),
        "execute_ready": ultralytics_available and args.weights is not None,
        "next_action": "Run with --execute to render predictions and export predictions.json.",
        "guidance": [
            "Use folder inference for quick error triage after the first baseline is trained.",
            "Keep rendered outputs local-only and out of git.",
        ],
    }


def execute_inference(args: argparse.Namespace, config: dict[str, Any] | None) -> dict[str, Any]:
    job = build_infer_job(args, config)
    job["output_dir"].mkdir(parents=True, exist_ok=True)
    yolo_class = load_yolo_class()
    model = yolo_class(job["model"])
    results = model.predict(**job["kwargs"])
    summary = summarize_predictions(list(results))
    export_prediction_summary(summary, job["summary_path"])
    return {
        "task": "infer_baseline",
        "status": "executed",
        "model": job["model"],
        "input": str(job["input"]),
        "output_dir": str(job["output_dir"]),
        "summary_path": str(job["summary_path"].resolve()),
        "predict_kwargs": job["kwargs"],
        "summary": summary,
    }


def main() -> int:
    args = parse_args()
    validate_args(args)
    config = maybe_load_yaml(args.config)
    if args.execute and not args.dry_run:
        payload = execute_inference(args, config)
    else:
        payload = build_plan(args, config)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
