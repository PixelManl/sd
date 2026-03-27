from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare LoRA ROI crops by reusing the SDXL mainline crop policy."
    )
    parser.add_argument("--batch-manifest", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--core-mask-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-name", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--adaptive-target-occupancy", type=float, default=0.20)
    parser.add_argument("--adaptive-min-side", type=int, default=320)
    parser.add_argument("--adaptive-root-bias", type=float, default=0.20)
    parser.add_argument("--jitter-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


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


def mask_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(np.asarray(mask) > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def compute_square_crop_box_from_bbox(
    box: list[int] | tuple[int, int, int, int],
    *,
    image_width: int,
    image_height: int,
    target_occupancy: float,
    min_side: int = 0,
) -> list[int]:
    if target_occupancy <= 0.0 or target_occupancy > 1.0:
        raise ValueError(f"target_occupancy must be in (0, 1], got {target_occupancy}")
    x1, y1, x2, y2 = [int(v) for v in box]
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    dominant_side = max(width, height)
    desired_side = max(int(min_side), int(math.ceil(dominant_side / target_occupancy)))
    desired_side = min(desired_side, int(image_width), int(image_height))

    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    crop_x1, crop_x2 = _clamp_square_interval(center_x, desired_side, int(image_width))
    crop_y1, crop_y2 = _clamp_square_interval(center_y, desired_side, int(image_height))
    side = min(crop_x2 - crop_x1, crop_y2 - crop_y1)
    crop_x1, crop_x2 = _clamp_square_interval(center_x, side, int(image_width))
    crop_y1, crop_y2 = _clamp_square_interval(center_y, side, int(image_height))
    return [crop_x1, crop_y1, crop_x2, crop_y2]


def compute_square_crop_box_from_mask(
    mask: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    target_occupancy: float,
    min_side: int = 0,
    root_bias: float = 0.0,
) -> list[int] | None:
    bbox = mask_bbox(mask)
    if bbox is None:
        return None
    crop_box = compute_square_crop_box_from_bbox(
        bbox,
        image_width=image_width,
        image_height=image_height,
        target_occupancy=target_occupancy,
        min_side=min_side,
    )
    if root_bias <= 0.0:
        return crop_box

    x1, y1, x2, y2 = crop_box
    side = x2 - x1
    shift = int(round(side * min(max(root_bias, 0.0), 1.0) * 0.25))
    new_y1 = clamp(y1 - shift, 0, image_height - side)
    return [x1, new_y1, x2, new_y1 + side]


def load_manifest(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_manifest_payload(payload: object) -> dict[str, object]:
    if isinstance(payload, dict):
        return {
            "prompt": str(payload.get("prompt") or ""),
            "negative_prompt": str(payload.get("negative_prompt") or ""),
            "records": list(payload.get("records") or []),
        }
    if isinstance(payload, list):
        return {
            "prompt": "",
            "negative_prompt": "",
            "records": list(payload),
        }
    raise TypeError(f"unsupported manifest payload type: {type(payload).__name__}")


def filter_records(
    records: list[dict[str, object]],
    *,
    image_name: str | list[str] | None,
    limit: int,
) -> list[dict[str, object]]:
    selected = list(records)
    if image_name:
        wanted_tokens = [image_name] if isinstance(image_name, str) else list(image_name)
        wanted = {str(item).strip() for item in wanted_tokens if str(item).strip()}
        selected = [
            record
            for record in selected
            if Path(str(record.get("image", ""))).name in wanted
            or Path(str(record.get("image", ""))).stem in wanted
        ]
    if limit > 0:
        selected = selected[:limit]
    return selected


def resolve_effective_crop_box(
    record: dict[str, object],
    *,
    core_mask: np.ndarray | None,
    image_width: int,
    image_height: int,
    adaptive_target_occupancy: float,
    adaptive_min_side: int,
    adaptive_root_bias: float,
) -> list[int]:
    crop_box_raw = record.get("crop_box")
    if crop_box_raw is None:
        xml_box = record.get("xml_box")
        if xml_box is None:
            raise KeyError("record must contain crop_box or xml_box")
        fallback_occupancy = adaptive_target_occupancy if adaptive_target_occupancy > 0.0 else 0.25
        fallback_min_side = adaptive_min_side if adaptive_min_side > 0 else 256
        crop_box = compute_square_crop_box_from_bbox(
            [int(v) for v in xml_box],
            image_width=image_width,
            image_height=image_height,
            target_occupancy=fallback_occupancy,
            min_side=fallback_min_side,
        )
    else:
        crop_box = [int(v) for v in crop_box_raw]

    if adaptive_target_occupancy <= 0.0 or core_mask is None:
        return crop_box

    adapted = compute_square_crop_box_from_mask(
        core_mask,
        image_width=image_width,
        image_height=image_height,
        target_occupancy=adaptive_target_occupancy,
        min_side=adaptive_min_side,
        root_bias=adaptive_root_bias,
    )
    return adapted or crop_box


def apply_crop_jitter(
    crop_box: list[int],
    *,
    image_width: int,
    image_height: int,
    jitter_ratio: float,
    seed: int,
) -> list[int]:
    if jitter_ratio <= 0.0:
        return [int(v) for v in crop_box]
    x1, y1, x2, y2 = [int(v) for v in crop_box]
    width = x2 - x1
    height = y2 - y1
    max_dx = int(round(width * jitter_ratio))
    max_dy = int(round(height * jitter_ratio))
    rng = random.Random(seed)
    dx = rng.randint(-max_dx, max_dx) if max_dx > 0 else 0
    dy = rng.randint(-max_dy, max_dy) if max_dy > 0 else 0

    new_x1 = clamp(x1 + dx, 0, image_width - width)
    new_y1 = clamp(y1 + dy, 0, image_height - height)
    return [new_x1, new_y1, new_x1 + width, new_y1 + height]


def materialize_roi_crops(
    *,
    manifest_path: Path,
    image_dir: Path,
    core_mask_dir: Path,
    output_dir: Path,
    image_names: list[str],
    limit: int,
    adaptive_target_occupancy: float,
    adaptive_min_side: int,
    adaptive_root_bias: float,
    jitter_ratio: float,
    seed: int,
) -> list[dict[str, object]]:
    manifest = resolve_manifest_payload(load_manifest(manifest_path))
    records: list[dict[str, object]] = filter_records(
        list(manifest["records"]),
        image_name=image_names or None,
        limit=limit,
    )
    if not records:
        raise ValueError("no matching records selected for LoRA ROI crop preparation")

    images_out = output_dir / "images"
    manifests_out = output_dir / "manifests"
    images_out.mkdir(parents=True, exist_ok=True)
    manifests_out.mkdir(parents=True, exist_ok=True)

    summary_records: list[dict[str, object]] = []
    for index, record in enumerate(records, start=1):
        image_name = str(record["image"])
        stem = Path(image_name).stem
        image_path = image_dir / image_name
        core_mask_path = core_mask_dir / f"{stem}_mask.png"

        full_image = Image.open(image_path).convert("RGB")
        core_mask = Image.open(core_mask_path).convert("L")
        crop_box = resolve_effective_crop_box(
            record,
            core_mask=np.array(core_mask),
            image_width=full_image.width,
            image_height=full_image.height,
            adaptive_target_occupancy=adaptive_target_occupancy,
            adaptive_min_side=adaptive_min_side,
            adaptive_root_bias=adaptive_root_bias,
        )
        crop_box = apply_crop_jitter(
            crop_box,
            image_width=full_image.width,
            image_height=full_image.height,
            jitter_ratio=jitter_ratio,
            seed=seed + index,
        )
        crop = full_image.crop(tuple(crop_box))
        crop_name = f"{stem}.png"
        crop_path = images_out / crop_name
        crop.save(crop_path)

        summary_records.append(
            {
                "image": image_name,
                "image_relpath": str(Path("images") / crop_name).replace("\\", "/"),
                "crop_box": crop_box,
                "seed": seed + index,
            }
        )

    summary = {
        "task": "prepare_lora_roi_crops",
        "batch_manifest": str(manifest_path.resolve()),
        "image_dir": str(image_dir.resolve()),
        "core_mask_dir": str(core_mask_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "adaptive_target_occupancy": adaptive_target_occupancy,
        "adaptive_min_side": adaptive_min_side,
        "adaptive_root_bias": adaptive_root_bias,
        "jitter_ratio": jitter_ratio,
        "records": summary_records,
    }
    (manifests_out / "roi_crops.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary_records


def main() -> int:
    args = parse_args()
    materialize_roi_crops(
        manifest_path=args.batch_manifest,
        image_dir=args.image_dir,
        core_mask_dir=args.core_mask_dir,
        output_dir=args.output_dir,
        image_names=args.image_name,
        limit=args.limit,
        adaptive_target_occupancy=args.adaptive_target_occupancy,
        adaptive_min_side=args.adaptive_min_side,
        adaptive_root_bias=args.adaptive_root_bias,
        jitter_ratio=args.jitter_ratio,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
