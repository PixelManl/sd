from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path


DEFAULT_INSTANCE_PROMPT = (
    "close-up utility hardware ROI, threaded stud with exactly one weathered gray steel hex nut, "
    "nut seated tightly against the underside of a metal plate, realistic metal contact, no extra hardware"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a local-only SDXL LoRA training scaffold for nut semantic adaptation."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--pretrained-model-name-or-path",
        default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    )
    parser.add_argument("--instance-prompt", default=DEFAULT_INSTANCE_PROMPT)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-train-steps", type=int, default=1200)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-script",
        default="examples/text_to_image/train_text_to_image_lora_sdxl.py",
        help="Relative path to the diffusers training script inside your local training environment.",
    )
    return parser.parse_args()


def build_training_plan(
    *,
    dataset_root: Path,
    output_root: Path,
    pretrained_model_name_or_path: str,
    instance_prompt: str,
    resolution: int,
    train_batch_size: int,
    learning_rate: float,
    max_train_steps: int,
    rank: int,
    seed: int,
    train_script: str = "examples/text_to_image/train_text_to_image_lora_sdxl.py",
) -> dict[str, object]:
    dataset_manifest = dataset_root / "manifests" / "dataset.jsonl"
    metadata_manifest = dataset_root / "metadata.jsonl"
    dataset_root_str = dataset_root.as_posix()
    output_root_str = output_root.as_posix()
    dataset_manifest_str = dataset_manifest.as_posix()
    metadata_manifest_str = metadata_manifest.as_posix()
    command = [
        "python",
        "-m",
        "accelerate.commands.launch",
        train_script,
        "--pretrained_model_name_or_path",
        pretrained_model_name_or_path,
        "--train_data_dir",
        dataset_root_str,
        "--output_dir",
        output_root_str,
        "--resolution",
        str(resolution),
        "--train_batch_size",
        str(train_batch_size),
        "--learning_rate",
        str(learning_rate),
        "--max_train_steps",
        str(max_train_steps),
        "--rank",
        str(rank),
        "--seed",
        str(seed),
        "--caption_column",
        "text",
    ]
    return {
        "task": "prepare_lora_training_scaffold",
        "dataset_root": dataset_root_str,
        "dataset_manifest": dataset_manifest_str,
        "metadata_manifest": metadata_manifest_str,
        "output_root": output_root_str,
        "pretrained_model_name_or_path": pretrained_model_name_or_path,
        "instance_prompt": instance_prompt,
        "resolution": resolution,
        "train_batch_size": train_batch_size,
        "learning_rate": learning_rate,
        "max_train_steps": max_train_steps,
        "rank": rank,
        "seed": seed,
        "train_script": train_script,
        "command": command,
    }


def render_bat(command: list[str]) -> str:
    return "@echo off\r\n" + " ".join(f'"{part}"' if " " in part else part for part in command) + "\r\n"


def render_sh(command: list[str]) -> str:
    return "#!/usr/bin/env bash\nset -euo pipefail\n" + " ".join(shlex.quote(part) for part in command) + "\n"


def materialize_training_scaffold(
    *,
    dataset_root: Path,
    output_root: Path,
    pretrained_model_name_or_path: str,
    instance_prompt: str,
    resolution: int,
    train_batch_size: int,
    learning_rate: float,
    max_train_steps: int,
    rank: int,
    seed: int,
    train_script: str = "examples/text_to_image/train_text_to_image_lora_sdxl.py",
) -> dict[str, object]:
    dataset_manifest = dataset_root / "manifests" / "dataset.jsonl"
    metadata_manifest = dataset_root / "metadata.jsonl"
    if not dataset_manifest.exists():
        raise FileNotFoundError(f"dataset manifest not found: {dataset_manifest}")
    if not metadata_manifest.exists():
        raise FileNotFoundError(f"metadata manifest not found: {metadata_manifest}")

    plan = build_training_plan(
        dataset_root=dataset_root,
        output_root=output_root,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        instance_prompt=instance_prompt,
        resolution=resolution,
        train_batch_size=train_batch_size,
        learning_rate=learning_rate,
        max_train_steps=max_train_steps,
        rank=rank,
        seed=seed,
        train_script=train_script,
    )

    plan_dir = output_root / "plan"
    plan_dir.mkdir(parents=True, exist_ok=True)
    (plan_dir / "training_plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (plan_dir / "run_train_lora.bat").write_text(render_bat(plan["command"]), encoding="utf-8")
    (plan_dir / "run_train_lora.sh").write_text(render_sh(plan["command"]), encoding="utf-8")
    return {
        "task": "prepare_lora_training_scaffold",
        "output_root": str(output_root.resolve()),
        "plan_dir": str(plan_dir.resolve()),
        "plan_path": str((plan_dir / "training_plan.json").resolve()),
    }


def main() -> int:
    args = parse_args()
    payload = materialize_training_scaffold(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        instance_prompt=args.instance_prompt,
        resolution=args.resolution,
        train_batch_size=args.train_batch_size,
        learning_rate=args.learning_rate,
        max_train_steps=args.max_train_steps,
        rank=args.rank,
        seed=args.seed,
        train_script=args.train_script,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
