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
    if not args.input.exists():
        raise FileNotFoundError(f"Input path not found: {args.input}")
    if args.weights and not args.weights.exists():
        raise FileNotFoundError(f"Weights file not found: {args.weights}")
    if args.output_dir:
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)


def maybe_load_yaml(path: Path) -> dict[str, Any] | None:
    if importlib.util.find_spec("yaml") is None:
        return None
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return payload if isinstance(payload, dict) else {}


def default_output_dir() -> Path:
    return Path("bolt/detect/runs/infer")


def build_plan(args: argparse.Namespace, config: dict[str, Any] | None) -> dict[str, Any]:
    ultralytics_available = importlib.util.find_spec("ultralytics") is not None
    output_dir = args.output_dir or default_output_dir()
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
        "conf_threshold": args.conf_threshold,
        "next_action": (
            "Wire backend-specific prediction rendering once baseline checkpoints become available."
        ),
        "guidance": [
            "Use folder inference for quick error triage after the first baseline is trained.",
            "Keep rendered outputs local-only and out of git.",
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
