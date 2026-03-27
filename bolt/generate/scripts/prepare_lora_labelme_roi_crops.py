from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare local-only LoRA ROI crops from healthy LabelMe rectangles."
    )
    parser.add_argument("--labelme-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--include-label",
        action="append",
        default=["GN"],
        help="Manual labels to keep. Default keeps GN only.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--expand-ratio", type=float, default=0.5)
    parser.add_argument("--min-side", type=int, default=192)
    return parser.parse_args()


def iter_images(image_dir: Path) -> dict[str, Path]:
    return {
        path.name: path
        for path in sorted(image_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


def clamp_box(box_xyxy: list[int], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = [int(value) for value in box_xyxy]
    x1 = clamp(x1, 0, width)
    y1 = clamp(y1, 0, height)
    x2 = clamp(x2, x1, width)
    y2 = clamp(y2, y1, height)
    return [x1, y1, x2, y2]


def box_has_positive_area(box_xyxy: list[int]) -> bool:
    x1, y1, x2, y2 = [int(value) for value in box_xyxy]
    return x2 > x1 and y2 > y1


def points_to_box_xyxy(points: list[object]) -> list[int] | None:
    xy_points: list[tuple[float, float]] = []
    for point in points:
        if not isinstance(point, list) or len(point) != 2:
            continue
        try:
            x = float(point[0])
            y = float(point[1])
        except (TypeError, ValueError):
            continue
        xy_points.append((x, y))
    if len(xy_points) < 2:
        return None
    xs = [point[0] for point in xy_points]
    ys = [point[1] for point in xy_points]
    return [
        int(round(min(xs))),
        int(round(min(ys))),
        int(round(max(xs))),
        int(round(max(ys))),
    ]


def _clamp_square_interval(center: float, side: int, limit: int) -> tuple[int, int]:
    side = min(max(1, int(side)), int(limit))
    start = int(math.floor(center - side / 2.0))
    end = start + side
    if start < 0:
        end -= start
        start = 0
    if end > limit:
        start -= end - limit
        end = limit
    if start < 0:
        start = 0
    return start, end


def compute_square_crop_box_from_bbox(
    box: list[int] | tuple[int, int, int, int],
    *,
    image_width: int,
    image_height: int,
    expand_ratio: float,
    min_side: int,
) -> list[int]:
    x1, y1, x2, y2 = [int(v) for v in box]
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    dominant_side = max(width, height)
    desired_side = max(int(min_side), int(math.ceil(dominant_side * (1.0 + max(expand_ratio, 0.0) * 2.0))))
    desired_side = min(desired_side, int(image_width), int(image_height))
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    crop_x1, crop_x2 = _clamp_square_interval(center_x, desired_side, int(image_width))
    crop_y1, crop_y2 = _clamp_square_interval(center_y, desired_side, int(image_height))
    return [crop_x1, crop_y1, crop_x2, crop_y2]


def resolve_image_reference(image_name: str, image_index: dict[str, Path]) -> Path | None:
    if image_name in image_index:
        return image_index[image_name]
    basename = Path(image_name).name
    if basename in image_index:
        return image_index[basename]
    return None


def normalize_labels(include_labels: list[str]) -> set[str]:
    labels = {label.strip() for label in include_labels if label.strip()}
    return labels or {"GN"}


def sanitize_token(text: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip())
    return token.strip("._-") or "sample"


def collect_labelme_records(
    *,
    labelme_dir: Path,
    image_dir: Path,
    include_labels: list[str],
) -> list[dict[str, object]]:
    image_index = iter_images(image_dir)
    allowed_labels = normalize_labels(include_labels)
    records: list[dict[str, object]] = []

    for labelme_path in sorted(labelme_dir.glob("*.json")):
        payload = json.loads(labelme_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        image_name = str(payload.get("imagePath") or f"{labelme_path.stem}.jpg")
        source_image = resolve_image_reference(image_name, image_index)
        if source_image is None:
            continue
        shapes = payload.get("shapes", [])
        if not isinstance(shapes, list):
            continue

        with Image.open(source_image) as image:
            width, height = image.size

        for shape_index, shape in enumerate(shapes, start=1):
            if not isinstance(shape, dict):
                continue
            manual_label = str(shape.get("label") or "").strip()
            if manual_label not in allowed_labels:
                continue
            shape_type = str(shape.get("shape_type") or "").strip().lower()
            if shape_type not in {"rectangle", "polygon"}:
                continue
            raw_box = points_to_box_xyxy(shape.get("points", []))
            if raw_box is None:
                continue
            box_xyxy = clamp_box(raw_box, width, height)
            if not box_has_positive_area(box_xyxy):
                continue
            records.append(
                {
                    "sample_id": sanitize_token(f"{labelme_path.stem}-manual-{shape_index:03d}"),
                    "image_name": Path(source_image).name,
                    "source_image": str(source_image),
                    "manual_label": manual_label,
                    "box_xyxy": box_xyxy,
                    "label_source": "labelme",
                }
            )
    return records


def materialize_roi_crops(
    *,
    records: list[dict[str, object]],
    output_dir: Path,
    expand_ratio: float,
    min_side: int,
    limit: int,
) -> list[dict[str, object]]:
    selected = list(records[:limit] if limit > 0 else records)
    if not selected:
        raise ValueError("no matching LabelMe records selected for ROI crop preparation")

    images_out = output_dir / "images"
    manifests_out = output_dir / "manifests"
    images_out.mkdir(parents=True, exist_ok=True)
    manifests_out.mkdir(parents=True, exist_ok=True)

    summary_records: list[dict[str, object]] = []
    for record in selected:
        source_image = Path(str(record["source_image"]))
        with Image.open(source_image).convert("RGB") as image:
            crop_box = compute_square_crop_box_from_bbox(
                record["box_xyxy"],
                image_width=image.width,
                image_height=image.height,
                expand_ratio=expand_ratio,
                min_side=min_side,
            )
            crop = image.crop(tuple(crop_box))

        crop_name = f"{record['sample_id']}.png"
        crop_path = images_out / crop_name
        crop.save(crop_path)
        summary_records.append(
            {
                "sample_id": str(record["sample_id"]),
                "image_name": str(record["image_name"]),
                "image_relpath": str(Path("images") / crop_name).replace("\\", "/"),
                "source_image": str(source_image),
                "manual_label": str(record["manual_label"]),
                "box_xyxy": list(record["box_xyxy"]),
                "crop_box": crop_box,
            }
        )

    payload = {
        "task": "prepare_lora_labelme_roi_crops",
        "label_source": "labelme",
        "record_count": len(summary_records),
        "expand_ratio": expand_ratio,
        "min_side": min_side,
        "records": summary_records,
    }
    (manifests_out / "roi_crops.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary_records


def main() -> int:
    args = parse_args()
    records = collect_labelme_records(
        labelme_dir=args.labelme_dir,
        image_dir=args.image_dir,
        include_labels=args.include_label,
    )
    summary = materialize_roi_crops(
        records=records,
        output_dir=args.output_dir,
        expand_ratio=args.expand_ratio,
        min_side=args.min_side,
        limit=args.limit,
    )
    payload = {
        "task": "prepare_lora_labelme_roi_crops",
        "labelme_dir": str(args.labelme_dir.resolve()),
        "image_dir": str(args.image_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "include_labels": list(normalize_labels(args.include_label)),
        "record_count": len(summary),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
