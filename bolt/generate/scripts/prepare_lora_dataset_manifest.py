from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


DEFAULT_CAPTION_TEMPLATE = (
    "close-up utility hardware ROI, threaded stud with exactly one weathered gray steel hex nut, "
    "nut seated tightly against the underside of a metal plate, realistic metal contact, no extra hardware"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a local-only LoRA dataset manifest for nut semantic SDXL training."
    )
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--caption-template", default=DEFAULT_CAPTION_TEMPLATE)
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def list_image_paths(images_dir: Path, include: list[str] | None = None) -> list[Path]:
    wanted = {item.strip() for item in include or [] if item.strip()}
    paths = sorted(path for path in images_dir.glob("*") if path.is_file())
    if not wanted:
        return paths
    return [path for path in paths if path.name in wanted or path.stem in wanted]


def build_record(image_path: Path, caption_template: str, split: str) -> dict[str, str]:
    stem = image_path.stem
    return {
        "sample_id": stem,
        "image_relpath": str(Path("images") / image_path.name).replace("\\", "/"),
        "caption_txt_relpath": str(Path("captions") / f"{stem}.txt").replace("\\", "/"),
        "caption": caption_template.strip(),
        "split": split,
    }


def assign_splits(image_paths: list[Path], train_ratio: float, seed: int) -> list[tuple[Path, str]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
    items = list(image_paths)
    random.Random(seed).shuffle(items)
    train_cutoff = max(1, min(len(items) - 1, int(round(len(items) * train_ratio)))) if len(items) > 1 else len(items)
    result: list[tuple[Path, str]] = []
    for index, image_path in enumerate(items):
        split = "train" if index < train_cutoff else "val"
        result.append((image_path, split))
    return result


def materialize_dataset_manifest(
    *,
    image_paths: list[Path],
    output_dir: Path,
    caption_template: str,
    train_ratio: float,
    seed: int,
) -> list[dict[str, str]]:
    captions_dir = output_dir / "captions"
    manifests_dir = output_dir / "manifests"
    captions_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, str]] = []
    metadata_records: list[dict[str, str]] = []
    for image_path, split in assign_splits(image_paths, train_ratio, seed):
        record = build_record(image_path=image_path, caption_template=caption_template, split=split)
        (captions_dir / f"{image_path.stem}.txt").write_text(record["caption"] + "\n", encoding="utf-8")
        records.append(record)
        metadata_records.append(
            {
                "file_name": record["image_relpath"],
                "text": record["caption"],
                "split": split,
            }
        )

    manifest_path = manifests_dir / "dataset.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    metadata_path = output_dir / "metadata.jsonl"
    metadata_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in metadata_records),
        encoding="utf-8",
    )
    return records


def main() -> int:
    args = parse_args()
    if not args.images_dir.exists():
        raise FileNotFoundError(f"images dir not found: {args.images_dir}")
    image_paths = list_image_paths(args.images_dir, args.include)
    if not image_paths:
        raise ValueError("no images selected for manifest generation")
    records = materialize_dataset_manifest(
        image_paths=image_paths,
        output_dir=args.output_dir,
        caption_template=args.caption_template,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    payload = {
        "task": "prepare_lora_dataset_manifest",
        "images_dir": str(args.images_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "caption_template": args.caption_template,
        "record_count": len(records),
        "split_counts": {
            "train": sum(1 for record in records if record["split"] == "train"),
            "val": sum(1 for record in records if record["split"] == "val"),
        },
        "manifest_path": str((args.output_dir / "manifests" / "dataset.jsonl").resolve()),
        "metadata_path": str((args.output_dir / "metadata.jsonl").resolve()),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
