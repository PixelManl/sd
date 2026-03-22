from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report bounding-box statistics for a COCO-like annotation file."
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="Path to a COCO-like JSON annotation file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path for the stats report.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.annotations.exists() or not args.annotations.is_file():
        raise FileNotFoundError(f"Annotation file not found: {args.annotations}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at: {path}")
    return payload


def summarize_series(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
        }
    return {
        "min": min(values),
        "max": max(values),
        "mean": mean(values),
        "median": median(values),
    }


def build_stats(payload: dict[str, Any]) -> dict[str, Any]:
    images = payload.get("images", [])
    annotations = payload.get("annotations", [])
    categories = payload.get("categories", [])
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError("Expected a COCO-like JSON file with 'images' and 'annotations' lists.")

    widths: list[float] = []
    heights: list[float] = []
    areas: list[float] = []
    aspect_ratios: list[float] = []
    invalid_boxes = 0
    anns_per_image: dict[Any, int] = {}

    for annotation in annotations:
        if not isinstance(annotation, dict):
            invalid_boxes += 1
            continue
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            invalid_boxes += 1
            continue
        _, _, width, height = bbox
        if not isinstance(width, (int, float)) or not isinstance(height, (int, float)):
            invalid_boxes += 1
            continue
        if width <= 0 or height <= 0:
            invalid_boxes += 1
            continue
        widths.append(float(width))
        heights.append(float(height))
        areas.append(float(width) * float(height))
        aspect_ratios.append(float(width) / float(height))
        image_id = annotation.get("image_id")
        anns_per_image[image_id] = anns_per_image.get(image_id, 0) + 1

    category_names: list[str] = []
    if isinstance(categories, list):
        for category in categories:
            if isinstance(category, dict):
                name = category.get("name")
                if isinstance(name, str):
                    category_names.append(name)

    counts = list(anns_per_image.values())
    image_diag_mean = None
    if widths and heights:
        diags = [math.sqrt((w * w) + (h * h)) for w, h in zip(widths, heights)]
        image_diag_mean = mean(diags)

    return {
        "task": "report_bbox_stats",
        "annotation_format": "coco_json",
        "image_count": len(images),
        "annotation_count": len(annotations),
        "valid_box_count": len(widths),
        "invalid_box_count": invalid_boxes,
        "category_names": category_names,
        "box_width": summarize_series(widths),
        "box_height": summarize_series(heights),
        "box_area": summarize_series(areas),
        "box_aspect_ratio": summarize_series(aspect_ratios),
        "annotations_per_image": summarize_series([float(value) for value in counts]),
        "box_diagonal_mean": image_diag_mean,
        "todo": [
            "Add support for VOC XML or YOLO txt if those formats become part of the ingest path.",
            "Add image-resolution-normalized box metrics once image metadata checks are needed.",
        ],
    }


def main() -> int:
    args = parse_args()
    validate_args(args)
    payload = load_json(args.annotations)
    stats = build_stats(payload)
    indent = 2 if args.pretty else None
    rendered = json.dumps(stats, indent=indent, ensure_ascii=False)

    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")

    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
