from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def load_local_module(name: str):
    module_path = Path(__file__).resolve().with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


prepare_detection_dataset = load_local_module("prepare_detection_dataset")
train_baseline = load_local_module("train_baseline")
eval_baseline = load_local_module("eval_baseline")
infer_baseline = load_local_module("infer_baseline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the bolt missing-fastener detection pipeline end-to-end."
    )
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("bolt/detect/configs/baseline.yaml"))
    parser.add_argument("--metadata", type=Path)
    parser.add_argument(
        "--group-field",
        choices=("capture_group_id", "scene_id", "sample_id", "none"),
        default="sample_id",
    )
    parser.add_argument("--class-name", default="missing_fastener")
    parser.add_argument("--include-label", action="append", default=[])
    parser.add_argument("--copy-mode", choices=("manifest_only", "copy", "symlink"), default="copy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--infer-source", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def make_prepare_args(args: argparse.Namespace) -> argparse.Namespace:
    return SimpleNamespace(
        images_dir=args.images_dir,
        annotations=args.annotations,
        output_dir=args.prepared_root,
        config=args.config,
        class_name=args.class_name,
        metadata=args.metadata,
        group_field=args.group_field,
        include_label=args.include_label,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        copy_mode=args.copy_mode,
        seed=args.seed,
        dry_run=args.dry_run,
    )


def make_train_args(args: argparse.Namespace) -> argparse.Namespace:
    return SimpleNamespace(
        config=args.config,
        dataset_root=args.prepared_root,
        run_dir=args.run_root / "train",
        weights=args.weights,
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        device=args.device,
        execute=not args.dry_run,
        dry_run=args.dry_run,
    )


def make_eval_args(args: argparse.Namespace, weights_path: Path) -> argparse.Namespace:
    return SimpleNamespace(
        config=args.config,
        dataset_root=args.prepared_root,
        run_dir=args.run_root / "eval",
        weights=weights_path,
        split="val",
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        imgsz=args.imgsz,
        device=args.device,
        execute=not args.dry_run,
        dry_run=args.dry_run,
    )


def make_infer_args(args: argparse.Namespace, weights_path: Path) -> argparse.Namespace:
    infer_source = args.infer_source or (args.prepared_root / "images" / "val")
    return SimpleNamespace(
        config=args.config,
        input=infer_source,
        weights=weights_path,
        output_dir=args.run_root / "infer",
        conf_threshold=args.conf_threshold,
        imgsz=args.imgsz,
        device=args.device,
        execute=not args.dry_run,
        dry_run=args.dry_run,
    )


def resolve_best_weights(train_payload: dict[str, Any]) -> Path:
    save_dir_text = train_payload.get("save_dir")
    if not isinstance(save_dir_text, str):
        raise FileNotFoundError("Training did not report save_dir.")
    save_dir = Path(save_dir_text)
    candidates = [
        save_dir / "weights" / "best.pt",
        save_dir / "weights" / "last.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not locate trained weights under {save_dir}")


def summarize_prepare_payload(payload: dict[str, Any]) -> dict[str, Any]:
    summary = dict(payload)
    resolved_samples = summary.pop("resolved_samples", [])
    summary["resolved_sample_count"] = len(resolved_samples)
    return summary


def summarize_infer_payload(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    summary = dict(payload)
    infer_summary = summary.get("summary")
    if isinstance(infer_summary, dict):
        compact = dict(infer_summary)
        predictions = compact.pop("predictions", [])
        compact["positive_prediction_count"] = sum(
            1 for item in predictions if isinstance(item, dict) and item.get("box_count", 0) > 0
        )
        summary["summary"] = compact
    return summary


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    prepare_args = make_prepare_args(args)
    prepare_detection_dataset.validate_args(prepare_args)
    config = prepare_detection_dataset.maybe_load_yaml(prepare_args.config)
    prepare_plan = prepare_detection_dataset.build_plan(prepare_args, config)
    if not args.dry_run:
        prepare_detection_dataset.materialize_dataset(prepare_plan, prepare_args)

    train_args = make_train_args(args)
    train_config = train_baseline.maybe_load_yaml(train_args.config)
    if not args.dry_run:
        train_baseline.validate_args(train_args)
    train_payload = (
        train_baseline.execute_training(train_args, train_config)
        if not args.dry_run
        else train_baseline.build_plan(train_args, train_config)
    )

    weights_path = None
    eval_payload = None
    infer_payload = None
    if args.dry_run:
        weights_path = Path(args.weights)
    else:
        weights_path = resolve_best_weights(train_payload)
        eval_args = make_eval_args(args, weights_path)
        eval_baseline.validate_args(eval_args)
        eval_config = eval_baseline.maybe_load_yaml(eval_args.config)
        eval_payload = eval_baseline.execute_evaluation(eval_args, eval_config)

        infer_args = make_infer_args(args, weights_path)
        infer_baseline.validate_args(infer_args)
        infer_config = infer_baseline.maybe_load_yaml(infer_args.config)
        infer_payload = infer_baseline.execute_inference(infer_args, infer_config)

    summary = {
        "task": "run_detection_pipeline",
        "status": "dry-run" if args.dry_run else "executed",
        "prepared_root": str(args.prepared_root.resolve()),
        "run_root": str(args.run_root.resolve()),
        "prepare": summarize_prepare_payload(prepare_plan),
        "train": train_payload,
        "weights_path": str(weights_path) if weights_path else None,
        "eval": eval_payload,
        "infer": summarize_infer_payload(infer_payload),
    }
    summary_path = args.run_root / "pipeline_summary.json"
    if not args.dry_run:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summary["summary_path"] = str(summary_path.resolve())
    return summary


def main() -> int:
    args = parse_args()
    payload = run_pipeline(args)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
