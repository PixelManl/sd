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
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Reserved for future framework wiring. Without this flag the script only plans.",
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
    if args.run_dir:
        args.run_dir.parent.mkdir(parents=True, exist_ok=True)


def maybe_load_yaml(path: Path) -> dict[str, Any] | None:
    if importlib.util.find_spec("yaml") is None:
        return None
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return payload if isinstance(payload, dict) else {}


def default_run_dir() -> Path:
    return Path("bolt/detect/runs/baseline")


def build_plan(args: argparse.Namespace, config: dict[str, Any] | None) -> dict[str, Any]:
    ultralytics_available = importlib.util.find_spec("ultralytics") is not None
    run_dir = args.run_dir or default_run_dir()
    return {
        "task": "eval_baseline",
        "status": "dry-run" if args.dry_run or not args.execute else "execute-requested",
        "config_path": str(args.config.resolve()),
        "config_loaded": config is not None,
        "yaml_available": importlib.util.find_spec("yaml") is not None,
        "ultralytics_available": ultralytics_available,
        "dataset_root": str(args.dataset_root.resolve()) if args.dataset_root else None,
        "run_dir": str(run_dir.resolve()),
        "weights": str(args.weights.resolve()) if args.weights else None,
        "split": args.split,
        "conf_threshold": args.conf_threshold,
        "iou_threshold": args.iou_threshold,
        "next_action": (
            "Wire backend-specific validation and metrics export after the training contract is settled."
        ),
        "guidance": [
            "Use the held-out split to inspect miss-rate and false alarms for missing_fastener.",
            "Keep metric outputs under bolt/detect/runs/ and do not commit them.",
        ],
    }


def main() -> int:
    args = parse_args()
    validate_args(args)
    config = maybe_load_yaml(args.config)
    plan = build_plan(args, config)
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
