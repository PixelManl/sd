from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a detection dataset skeleton for the bolt baseline."
    )
    parser.add_argument("--images-dir", type=Path, required=True, help="Image root.")
    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="Box annotation file. COCO JSON is supported for summary.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Prepared dataset root to create or inspect.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("bolt/detect/configs/baseline.yaml"),
        help="Optional baseline config path.",
    )
    parser.add_argument(
        "--class-name",
        default="missing_fastener",
        help="Single-class baseline label.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Optional sample metadata file (.json, .jsonl, .csv) for group-aware splitting.",
    )
    parser.add_argument(
        "--group-field",
        choices=("capture_group_id", "scene_id", "sample_id", "none"),
        default="capture_group_id",
        help="Preferred grouping key for split isolation. Falls back to sample_id when missing.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument(
        "--copy-mode",
        choices=("manifest_only", "copy", "symlink"),
        default="manifest_only",
        help="Skeleton only defaults to manifest_only.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the preparation plan without materializing outputs.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.images_dir.exists() or not args.images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {args.images_dir}")
    if not args.annotations.exists():
        raise FileNotFoundError(f"Annotation path not found: {args.annotations}")
    if not args.annotations.is_file() and not args.annotations.is_dir():
        raise FileNotFoundError(f"Annotation path is not a file or directory: {args.annotations}")
    split_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(split_sum - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {split_sum:.6f} "
            f"from train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}."
        )
    if args.metadata and not args.metadata.exists():
        raise FileNotFoundError(f"Metadata path not found: {args.metadata}")
    if args.config and not args.config.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")


def maybe_load_yaml(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if importlib.util.find_spec("yaml") is None:
        return None
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at: {path}")
    return payload


def load_metadata_records(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []

    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                if not isinstance(payload, dict):
                    raise ValueError(f"Expected JSON object in {path} line {lineno}")
                records.append(payload)
        return records
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [record for record in payload if isinstance(record, dict)]
        if isinstance(payload, dict):
            records = payload.get("records")
            if isinstance(records, list):
                return [record for record in records if isinstance(record, dict)]
        raise ValueError(f"Unsupported metadata JSON structure: {path}")
    raise ValueError(f"Unsupported metadata format: {path.suffix}")


def build_metadata_index(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for record in records:
        for key in ("sample_id", "file_name", "image_name", "image_path"):
            value = record.get(key)
            if not isinstance(value, str) or not value.strip():
                continue
            normalized = value.strip()
            index[normalized] = record
            index[Path(normalized).name] = record
            index[Path(normalized).stem] = record
    return index


def match_metadata(sample_id: str, file_name: str, metadata_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    for key in (sample_id, file_name, Path(file_name).stem):
        if key in metadata_index:
            return metadata_index[key]
    return {}


def summarize_coco_annotations(payload: dict[str, Any]) -> dict[str, Any]:
    images = payload.get("images")
    annotations = payload.get("annotations")
    categories = payload.get("categories")
    if not isinstance(images, list) or not isinstance(annotations, list):
        return {
            "format": "unknown",
            "image_count": 0,
            "annotation_count": 0,
            "category_names": [],
            "parse_warning": "COCO-like keys not found. Summary is validation-only.",
        }

    category_names: list[str] = []
    if isinstance(categories, list):
        for category in categories:
            if isinstance(category, dict):
                name = category.get("name")
                if isinstance(name, str):
                    category_names.append(name)

    return {
        "format": "coco_json",
        "image_count": len(images),
        "annotation_count": len(annotations),
        "category_names": category_names,
        "parse_warning": None,
    }


def index_images(images_dir: Path) -> dict[str, Path]:
    image_index: dict[str, Path] = {}
    for path in images_dir.rglob("*"):
        if path.is_file():
            image_index[path.name] = path
            image_index[path.stem] = path
    return image_index


def parse_pascal_voc_directory(
    images_dir: Path,
    annotations_dir: Path,
    class_name: str,
    metadata_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    image_index = index_images(images_dir)
    samples: list[dict[str, Any]] = []
    original_labels: set[str] = set()
    xml_files = sorted(annotations_dir.glob("*.xml"))

    for xml_path in xml_files:
        root = ET.parse(xml_path).getroot()
        filename = (root.findtext("filename") or xml_path.stem).strip()
        image_path = image_index.get(filename) or image_index.get(Path(filename).stem)
        if image_path is None:
            raise FileNotFoundError(
                f"Could not resolve image for annotation {xml_path.name} using filename {filename!r}."
            )

        size_node = root.find("size")
        if size_node is None:
            raise ValueError(f"Missing <size> block in annotation: {xml_path}")
        width = int(size_node.findtext("width", "0"))
        height = int(size_node.findtext("height", "0"))
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image size in annotation: {xml_path}")

        boxes: list[dict[str, Any]] = []
        for obj in root.findall("object"):
            original_label = (obj.findtext("name") or "").strip()
            original_labels.add(original_label)
            bbox = obj.find("bndbox")
            if bbox is None:
                continue
            xmin = int(float(bbox.findtext("xmin", "0")))
            ymin = int(float(bbox.findtext("ymin", "0")))
            xmax = int(float(bbox.findtext("xmax", "0")))
            ymax = int(float(bbox.findtext("ymax", "0")))
            boxes.append(
                {
                    "class_id": 0,
                    "class_name": class_name,
                    "original_label": original_label,
                    "bbox": [xmin, ymin, xmax, ymax],
                }
            )

        samples.append(
            {
                "sample_id": xml_path.stem,
                "xml_path": str(xml_path.resolve()),
                "image_path": str(image_path.resolve()),
                "file_name": image_path.name,
                "width": width,
                "height": height,
                "boxes": boxes,
                "metadata": match_metadata(xml_path.stem, image_path.name, metadata_index),
            }
        )

    return {
        "format": "pascal_voc_xml_dir",
        "summary": {
            "format": "pascal_voc_xml_dir",
            "image_count": len(samples),
            "annotation_count": sum(len(sample["boxes"]) for sample in samples),
            "category_names": [class_name] if samples else [],
            "source_labels": sorted(label for label in original_labels if label),
            "parse_warning": None,
        },
        "samples": samples,
    }


def coerce_coco_box(raw_bbox: Any) -> list[int] | None:
    if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
        return None
    x, y, width, height = raw_bbox
    try:
        xmin = int(float(x))
        ymin = int(float(y))
        box_width = int(float(width))
        box_height = int(float(height))
    except (TypeError, ValueError):
        return None
    xmax = xmin + box_width
    ymax = ymin + box_height
    if box_width <= 0 or box_height <= 0:
        return None
    return [xmin, ymin, xmax, ymax]


def parse_coco_annotations(
    images_dir: Path,
    payload: dict[str, Any],
    class_name: str,
    metadata_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    images = payload.get("images")
    annotations = payload.get("annotations")
    categories = payload.get("categories")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError("COCO annotations must contain 'images' and 'annotations' lists.")

    image_index = index_images(images_dir)
    category_name_by_id: dict[Any, str] = {}
    if isinstance(categories, list):
        for category in categories:
            if isinstance(category, dict):
                category_name_by_id[category.get("id")] = str(category.get("name", ""))

    annotations_by_image_id: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    source_labels: set[str] = set()
    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        image_id = annotation.get("image_id")
        bbox = coerce_coco_box(annotation.get("bbox"))
        if image_id is None or bbox is None:
            continue
        category_name = category_name_by_id.get(annotation.get("category_id"), "")
        if category_name:
            source_labels.add(category_name)
        annotations_by_image_id[image_id].append(
            {
                "class_id": 0,
                "class_name": class_name,
                "original_label": category_name,
                "bbox": bbox,
            }
        )

    samples: list[dict[str, Any]] = []
    for image in images:
        if not isinstance(image, dict):
            continue
        file_name = str(image.get("file_name", "")).strip()
        image_id = image.get("id", file_name)
        image_path = image_index.get(file_name) or image_index.get(Path(file_name).name) or image_index.get(Path(file_name).stem)
        if image_path is None:
            raise FileNotFoundError(f"Could not resolve image for COCO image entry {file_name!r}.")
        width = int(image.get("width", 0))
        height = int(image.get("height", 0))
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image size in COCO image entry: {file_name!r}")
        sample_id = Path(file_name).stem or str(image_id)
        samples.append(
            {
                "sample_id": sample_id,
                "image_id": image_id,
                "image_path": str(image_path.resolve()),
                "file_name": image_path.name,
                "width": width,
                "height": height,
                "boxes": annotations_by_image_id.get(image_id, []),
                "metadata": match_metadata(sample_id, image_path.name, metadata_index),
            }
        )

    return {
        "format": "coco_json",
        "summary": {
            "format": "coco_json",
            "image_count": len(samples),
            "annotation_count": sum(len(sample["boxes"]) for sample in samples),
            "category_names": [class_name] if samples else [],
            "source_labels": sorted(label for label in source_labels if label),
            "parse_warning": None,
        },
        "payload": payload,
        "samples": samples,
    }


def load_annotation_source(
    annotations_path: Path,
    images_dir: Path,
    class_name: str,
    metadata_path: Path | None,
) -> dict[str, Any]:
    metadata_index = build_metadata_index(load_metadata_records(metadata_path))
    if annotations_path.is_dir():
        return parse_pascal_voc_directory(images_dir, annotations_path, class_name, metadata_index)

    payload = load_json(annotations_path)
    return parse_coco_annotations(images_dir, payload, class_name, metadata_index)


def resolve_group_key(sample: dict[str, Any], group_field: str) -> str:
    if group_field == "none":
        return sample["sample_id"]
    if group_field == "sample_id":
        return sample["sample_id"]
    metadata = sample.get("metadata")
    if isinstance(metadata, dict):
        value = metadata.get(group_field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return sample["sample_id"]


def assign_splits(samples: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    grouped_samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped_samples[resolve_group_key(sample, args.group_field)].append(sample)

    group_items = list(grouped_samples.items())
    random.Random(args.seed).shuffle(group_items)
    total_samples = len(samples)
    target_counts = {
        "train": int(total_samples * args.train_ratio),
        "val": int(total_samples * args.val_ratio),
    }
    target_counts["test"] = total_samples - target_counts["train"] - target_counts["val"]

    split_samples: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    running_counts = {"train": 0, "val": 0, "test": 0}

    for group_key, members in group_items:
        preferred_split = "test"
        if running_counts["train"] < target_counts["train"]:
            preferred_split = "train"
        elif running_counts["val"] < target_counts["val"]:
            preferred_split = "val"
        for sample in members:
            sample["group_key"] = group_key
        split_samples[preferred_split].extend(members)
        running_counts[preferred_split] += len(members)

    return split_samples


def build_plan(args: argparse.Namespace, config: dict[str, Any] | None) -> dict[str, Any]:
    annotation_data = load_annotation_source(
        args.annotations,
        args.images_dir,
        args.class_name,
        args.metadata,
    )
    annotation_summary = annotation_data["summary"]
    split_assignment = assign_splits(annotation_data["samples"], args)
    split_plan = {split: len(rows) for split, rows in split_assignment.items()}
    output_dir = args.output_dir.resolve()
    groups = sorted({resolve_group_key(sample, args.group_field) for sample in annotation_data["samples"]})
    return {
        "task": "prepare_detection_dataset",
        "status": "dry-run" if args.dry_run else "planned",
        "config_path": str(args.config.resolve()) if args.config else None,
        "config_loaded": config is not None,
        "yaml_available": importlib.util.find_spec("yaml") is not None,
        "source": {
            "images_dir": str(args.images_dir.resolve()),
            "annotations": str(args.annotations.resolve()),
            "metadata": str(args.metadata.resolve()) if args.metadata else None,
            "class_name": args.class_name,
            "annotation_format": annotation_data["format"],
        },
        "split_strategy": {
            "group_field": args.group_field,
            "group_count": len(groups),
        },
        "splits": split_plan,
        "copy_mode": args.copy_mode,
        "target_layout": {
            "images_train": str(output_dir / "images" / "train"),
            "images_val": str(output_dir / "images" / "val"),
            "images_test": str(output_dir / "images" / "test"),
            "labels_train": str(output_dir / "labels" / "train"),
            "labels_val": str(output_dir / "labels" / "val"),
            "labels_test": str(output_dir / "labels" / "test"),
            "manifest": str(output_dir / "dataset_manifest.json"),
            "dataset_yaml": str(output_dir / "dataset.yaml"),
        },
        "annotation_summary": annotation_summary,
        "resolved_samples": annotation_data.get("samples", []),
        "todo": [
            "Review group-aware split outputs to confirm no scene leakage across train/val/test.",
            "Keep generated dataset manifests local-only and do not commit them.",
            "Run the detector baseline against dataset.yaml after QA passes.",
        ],
    }


def materialize_dirs(output_dir: Path) -> None:
    for relative in (
        Path("images/train"),
        Path("images/val"),
        Path("images/test"),
        Path("labels/train"),
        Path("labels/val"),
        Path("labels/test"),
        Path("manifests"),
    ):
        (output_dir / relative).mkdir(parents=True, exist_ok=True)


def normalize_bbox(box: list[int], width: int, height: int) -> str:
    xmin, ymin, xmax, ymax = box
    x_center = ((xmin + xmax) / 2.0) / width
    y_center = ((ymin + ymax) / 2.0) / height
    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height
    return f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"


def materialize_image(source: Path, destination: Path, copy_mode: str) -> None:
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if copy_mode == "copy":
        shutil.copy2(source, destination)
    elif copy_mode == "symlink":
        destination.symlink_to(source)


def write_dataset_yaml(output_dir: Path, class_name: str) -> None:
    dataset_yaml = "\n".join(
        [
            f"path: {output_dir}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            f"  0: {class_name}",
            "",
        ]
    )
    (output_dir / "dataset.yaml").write_text(dataset_yaml, encoding="utf-8")


def materialize_dataset(plan: dict[str, Any], args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    materialize_dirs(output_dir)

    split_samples = assign_splits(plan["resolved_samples"], args)
    split_manifest: dict[str, list[dict[str, Any]]] = {}

    for split, samples in split_samples.items():
        split_manifest[split] = []
        for sample in samples:
            source_image = Path(sample["image_path"])
            image_target = output_dir / "images" / split / source_image.name
            label_target = output_dir / "labels" / split / f"{Path(sample['file_name']).stem}.txt"

            label_lines = [
                normalize_bbox(box["bbox"], sample["width"], sample["height"])
                for box in sample["boxes"]
            ]
            label_target.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")

            if args.copy_mode in {"copy", "symlink"}:
                materialize_image(source_image, image_target, args.copy_mode)

            split_manifest[split].append(
                {
                    "sample_id": sample["sample_id"],
                    "group_key": sample.get("group_key"),
                    "image_path": str(source_image),
                    "label_path": str(label_target.resolve()),
                    "box_count": len(sample["boxes"]),
                    "metadata": sample.get("metadata", {}),
                }
            )

    for split, rows in split_manifest.items():
        (output_dir / "manifests" / f"{split}.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    write_dataset_yaml(output_dir, args.class_name)

    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(plan, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    validate_args(args)
    config = maybe_load_yaml(args.config)
    plan = build_plan(args, config)

    if not args.dry_run:
        materialize_dataset(plan, args)

    print(json.dumps(plan, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
