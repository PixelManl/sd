from __future__ import annotations

import argparse
import importlib.util
import json
import random
import shutil
import xml.etree.ElementTree as ET
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
    images_dir: Path, annotations_dir: Path, class_name: str
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


def load_annotation_source(
    annotations_path: Path, images_dir: Path, class_name: str
) -> dict[str, Any]:
    if annotations_path.is_dir():
        return parse_pascal_voc_directory(images_dir, annotations_path, class_name)

    payload = load_json(annotations_path)
    return {
        "format": "coco_json",
        "summary": summarize_coco_annotations(payload),
        "payload": payload,
        "samples": [],
    }


def build_split_plan(annotation_data: dict[str, Any], args: argparse.Namespace) -> dict[str, int]:
    if annotation_data["format"] == "pascal_voc_xml_dir":
        sample_ids = [sample["sample_id"] for sample in annotation_data["samples"]]
        random.Random(args.seed).shuffle(sample_ids)
        total = len(sample_ids)
        train_count = int(total * args.train_ratio)
        val_count = int(total * args.val_ratio)
        test_count = total - train_count - val_count
        return {"train": train_count, "val": val_count, "test": test_count}

    payload = annotation_data["payload"]
    images = payload.get("images")
    if not isinstance(images, list):
        return {"train": 0, "val": 0, "test": 0}

    image_ids: list[Any] = []
    for image in images:
        if isinstance(image, dict):
            image_ids.append(image.get("id", image.get("file_name")))

    random.Random(args.seed).shuffle(image_ids)
    total = len(image_ids)
    train_count = int(total * args.train_ratio)
    val_count = int(total * args.val_ratio)
    test_count = total - train_count - val_count
    return {"train": train_count, "val": val_count, "test": test_count}


def assign_splits(samples: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    shuffled = list(samples)
    random.Random(args.seed).shuffle(shuffled)
    total = len(shuffled)
    train_count = int(total * args.train_ratio)
    val_count = int(total * args.val_ratio)
    train_samples = shuffled[:train_count]
    val_samples = shuffled[train_count : train_count + val_count]
    test_samples = shuffled[train_count + val_count :]
    return {"train": train_samples, "val": val_samples, "test": test_samples}


def build_plan(args: argparse.Namespace, config: dict[str, Any] | None) -> dict[str, Any]:
    annotation_data = load_annotation_source(args.annotations, args.images_dir, args.class_name)
    annotation_summary = annotation_data["summary"]
    split_plan = build_split_plan(annotation_data, args)
    output_dir = args.output_dir.resolve()
    return {
        "task": "prepare_detection_dataset",
        "status": "dry-run" if args.dry_run else "planned",
        "config_path": str(args.config.resolve()) if args.config else None,
        "config_loaded": config is not None,
        "yaml_available": importlib.util.find_spec("yaml") is not None,
        "source": {
            "images_dir": str(args.images_dir.resolve()),
            "annotations": str(args.annotations.resolve()),
            "class_name": args.class_name,
            "annotation_format": annotation_data["format"],
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
            "Normalize the source box annotations into the target detector format.",
            "Decide whether the prepared dataset should be copy-, symlink-, or manifest-based.",
            "Write class index and split manifest files after the detector backend is fixed.",
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


def write_dataset_yaml(output_dir: Path) -> None:
    dataset_yaml = "\n".join(
        [
            f"path: {output_dir}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            "  0: missing_fastener",
            "",
        ]
    )
    (output_dir / "dataset.yaml").write_text(dataset_yaml, encoding="utf-8")


def materialize_dataset(plan: dict[str, Any], args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    materialize_dirs(output_dir)

    if plan["source"]["annotation_format"] == "pascal_voc_xml_dir":
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
                        "image_path": str(source_image),
                        "label_path": str(label_target.resolve()),
                        "box_count": len(sample["boxes"]),
                    }
                )

        for split, rows in split_manifest.items():
            (output_dir / "manifests" / f"{split}.json").write_text(
                json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        write_dataset_yaml(output_dir)

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
