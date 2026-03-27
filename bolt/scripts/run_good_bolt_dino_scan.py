from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_WORKSPACE = Path("data/bolt_parallel/good_bolt_assets")
DEFAULT_TEXT_PROMPT = "healthy bolt . healthy nut . intact fastener ."
VERTICAL_LABEL_KEYWORDS = ("downward", "thread", "threaded", "stud", "rod", "screw", "bolt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan images with an open-vocabulary detector and write good-bolt candidate boxes."
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=DEFAULT_WORKSPACE,
        help="Workspace root containing dino/boxes_json.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Optional external image directory. Defaults to <workspace>/incoming/images.",
    )
    parser.add_argument(
        "--backend",
        default="grounding-dino",
        help="Backend selector: grounding-dino or module:attr / path.py:attr.",
    )
    parser.add_argument(
        "--model-id",
        default="IDEA-Research/grounding-dino-tiny",
        help="Model ID used by the grounding-dino backend.",
    )
    parser.add_argument(
        "--text-prompt",
        default=DEFAULT_TEXT_PROMPT,
        help="Open-vocabulary prompt for the grounding-dino backend.",
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.30,
        help="Grounding-DINO box threshold.",
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.25,
        help="Grounding-DINO text threshold.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.25,
        help="Post-filter threshold applied to backend outputs.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Maximum number of boxes to keep per image after sorting.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on the number of images to scan. 0 means all images.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively walk the input directory for images.",
    )
    parser.add_argument(
        "--target-mode",
        choices=("all", "vertical-only"),
        default="vertical-only",
        help="Filter mode for detector candidates.",
    )
    return parser.parse_args()


def fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 2


def validate_workspace(workspace: Path) -> Path:
    resolved = workspace.expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def validate_input_dir(input_dir: Path) -> Path:
    resolved = input_dir.expanduser().resolve()
    if not resolved.exists():
        raise ValueError(f"input directory does not exist: {resolved}")
    if not resolved.is_dir():
        raise ValueError(f"input directory is not a directory: {resolved}")
    return resolved


def collect_images(input_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return [
        path
        for path in sorted(input_dir.glob(pattern))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]


def build_scan_plan(
    *,
    workspace: Path,
    input_dir: Path | None = None,
    recursive: bool = False,
    limit: int = 0,
) -> dict[str, object]:
    workspace = validate_workspace(workspace)
    images_dir = validate_input_dir(input_dir or (workspace / "incoming" / "images"))
    images = collect_images(images_dir, recursive=recursive)
    if limit > 0:
        images = images[:limit]

    records = [
        {
            "image_name": path.name,
            "image_stem": path.stem,
            "source_image": str(path),
        }
        for path in images
    ]
    return {
        "status": "placeholder",
        "mode": "good-bolt-dino-plan",
        "workspace": str(workspace),
        "input_dir": str(images_dir),
        "image_count": len(records),
        "records": records,
        "output_dir": str(workspace / "dino" / "boxes_json"),
    }


def parse_backend_spec(spec_text: str) -> tuple[str, str]:
    module_ref, attribute = spec_text.rsplit(":", 1) if ":" in spec_text else ("", "")
    if not module_ref or not attribute:
        raise ValueError("--backend must be grounding-dino or module:attr / path.py:attr")
    return module_ref, attribute


def resolve_backend(spec_text: str) -> Callable[[Path, dict[str, object]], object]:
    module_ref, attribute = parse_backend_spec(spec_text)
    candidate_path = Path(module_ref).expanduser()

    if candidate_path.suffix == ".py" and candidate_path.exists():
        module_name = f"_good_bolt_dino_backend_{abs(hash(candidate_path.resolve()))}"
        spec = importlib.util.spec_from_file_location(module_name, candidate_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"unable to load backend module from file: {candidate_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        try:
            module = importlib.import_module(module_ref)
        except ImportError as exc:
            raise ValueError(f"unable to import backend module: {module_ref}") from exc

    backend = getattr(module, attribute, None)
    if backend is None:
        raise ValueError(f"backend attribute not found: {attribute}")
    if not callable(backend):
        raise ValueError(f"backend attribute is not callable: {attribute}")
    return backend


def build_grounding_dino_backend(
    *,
    model_id: str,
    text_prompt: str,
    box_threshold: float,
    text_threshold: float,
) -> Callable[[Path, dict[str, object]], list[dict[str, object]]]:
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    def predictor(source_image: Path, asset_context: dict[str, object]) -> list[dict[str, object]]:
        del asset_context
        with Image.open(source_image) as image:
            image = image.convert("RGB")
            inputs = processor(images=image, text=text_prompt, return_tensors="pt")
            inputs = inputs.to(device)
            with torch.inference_mode():
                outputs = model(**inputs)
            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                **build_grounding_postprocess_kwargs(
                    processor,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    target_sizes=[image.size[::-1]],
                ),
            )[0]

        boxes: list[dict[str, object]] = []
        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            xyxy = [int(round(float(value))) for value in box.tolist()]
            boxes.append(
                {
                    "box_xyxy": xyxy,
                    "score": float(score),
                    "label": "healthy_bolt",
                    "matched_label": str(label),
                }
            )
        return boxes

    return predictor


def build_grounding_postprocess_kwargs(
    processor: object,
    *,
    box_threshold: float,
    text_threshold: float,
    target_sizes: list[tuple[int, int]],
) -> dict[str, object]:
    signature = inspect.signature(processor.post_process_grounded_object_detection)
    kwargs: dict[str, object] = {
        "text_threshold": text_threshold,
        "target_sizes": target_sizes,
    }
    if "box_threshold" in signature.parameters:
        kwargs["box_threshold"] = box_threshold
    else:
        kwargs["threshold"] = box_threshold
    return kwargs


def normalize_boxes(payload: object, *, min_score: float, top_k: int) -> list[dict[str, object]]:
    if isinstance(payload, dict):
        payload = payload.get("boxes", [])
    if not isinstance(payload, list):
        raise ValueError(f"backend output must be a list or dict with boxes, got {type(payload).__name__}")

    boxes: list[dict[str, object]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        raw_box = item.get("box_xyxy") or item.get("box")
        if not isinstance(raw_box, list) or len(raw_box) != 4:
            continue
        score = float(item.get("score", 0.0))
        if score < min_score:
            continue
        record = {
            "box_xyxy": [int(value) for value in raw_box],
            "score": score,
            "label": str(item.get("label") or "healthy_bolt"),
        }
        matched_label = item.get("matched_label")
        if matched_label not in (None, ""):
            record["matched_label"] = str(matched_label)
        boxes.append(record)

    boxes.sort(key=lambda item: item["score"], reverse=True)
    if top_k > 0:
        boxes = boxes[:top_k]
    return boxes


def filter_vertical_candidates(
    boxes: list[dict[str, object]],
    *,
    image_size: tuple[int, int],
    source_image: Path | None = None,
) -> list[dict[str, object]]:
    width, height = image_size
    image_area = max(1, width * height)
    filtered: list[dict[str, object]] = []

    for item in boxes:
        raw_box = item.get("box_xyxy")
        if not isinstance(raw_box, list) or len(raw_box) != 4:
            continue
        x1, y1, x2, y2 = [int(value) for value in raw_box]
        box_width = max(1, x2 - x1)
        box_height = max(1, y2 - y1)
        area_ratio = (box_width * box_height) / image_area
        max_side_ratio = max(box_width / max(1, width), box_height / max(1, height))
        matched_label = str(item.get("matched_label") or item.get("label") or "").lower()

        if not any(keyword in matched_label for keyword in VERTICAL_LABEL_KEYWORDS):
            continue
        if area_ratio > 0.02:
            continue
        if max_side_ratio > 0.18:
            continue
        if source_image is not None:
            shape = score_vertical_stud_shape(source_image, raw_box)
            item["shape_score"] = shape["shape_score"]
            if not shape["passes_shape_gate"]:
                continue
        filtered.append(item)

    filtered.sort(key=lambda record: float(record.get("score", 0.0)), reverse=True)
    return filtered


def score_vertical_stud_shape(source_image: Path, box_xyxy: list[int]) -> dict[str, object]:
    with Image.open(source_image) as image:
        gray = np.asarray(image.convert("L"), dtype=np.float32)

    x1, y1, x2, y2 = [int(value) for value in box_xyxy]
    x1 = max(0, min(gray.shape[1] - 1, x1))
    x2 = max(x1 + 1, min(gray.shape[1], x2))
    y1 = max(0, min(gray.shape[0] - 1, y1))
    y2 = max(y1 + 1, min(gray.shape[0], y2))
    crop = gray[y1:y2, x1:x2]
    height, width = crop.shape
    if height < 8 or width < 8:
        return {"shape_score": 0.0, "passes_shape_gate": False}

    lower = crop[height // 3 :, :]
    if lower.shape[0] < 4 or lower.shape[1] < 4:
        return {"shape_score": 0.0, "passes_shape_gate": False}

    gx_lower = np.abs(np.diff(lower, axis=1))
    gy_lower = np.abs(np.diff(lower, axis=0))
    tx = np.percentile(gx_lower, 75)
    ty = np.percentile(gy_lower, 75)
    col_runs = (gx_lower > tx).sum(axis=0)
    row_runs = (gy_lower > ty).sum(axis=1)

    vertical_run_ratio = float(col_runs.max() / max(1, gx_lower.shape[0]))
    horizontal_run_ratio = float(row_runs.max() / max(1, gy_lower.shape[1]))
    vertical_dominance = vertical_run_ratio - horizontal_run_ratio
    dark_ratio = float((lower < np.percentile(crop, 35)).mean())
    aspect_ratio = float(height / max(1, width))
    dark_mask = lower < np.percentile(crop, 35)
    col_dark = dark_mask.mean(axis=0)
    row_dark = dark_mask.mean(axis=1)
    dark_col_over_row = float(col_dark.max() / (row_dark.max() + 1e-6))
    thread_edge_ratio = 0.0
    bottom_span_ratio = 1.0
    ys, xs = np.where(lower < np.percentile(crop, 40))
    if len(xs) > 0 and len(ys) > 0:
        roi_x1 = int(xs.min())
        roi_x2 = int(xs.max()) + 1
        roi_y1 = int(ys.min())
        roi_y2 = int(ys.max()) + 1
        if roi_x2 - roi_x1 >= 4 and roi_y2 - roi_y1 >= 4:
            roi = lower[roi_y1:roi_y2, roi_x1:roi_x2]
            gx_mean = float(np.abs(np.diff(roi, axis=1)).mean())
            gy_mean = float(np.abs(np.diff(roi, axis=0)).mean())
            thread_edge_ratio = gy_mean / (gx_mean + 1e-6)
        bottom_band = (lower < np.percentile(crop, 40))[max(0, int(lower.shape[0] * 0.7)) :, :]
        if bottom_band.size > 0:
            bottom_cols = bottom_band.mean(axis=0)
            bottom_xs = np.where(bottom_cols > 0.2)[0]
            if len(bottom_xs) > 0:
                bottom_span_ratio = float((bottom_xs.max() - bottom_xs.min() + 1) / lower.shape[1])

    shape_score = vertical_dominance + 0.35 * aspect_ratio - 0.6 * dark_ratio
    passes_long_stud_gate = (
        aspect_ratio >= 1.20
        and dark_ratio <= 0.42
        and dark_col_over_row >= 1.10
        and thread_edge_ratio >= 1.10
        and shape_score >= 0.10
    )
    passes_short_hanging_gate = (
        aspect_ratio >= 0.95
        and dark_ratio <= 0.34
        and dark_col_over_row >= 1.10
        and thread_edge_ratio >= 0.70
        and bottom_span_ratio <= 0.35
        and shape_score >= 0.10
    )
    passes = passes_long_stud_gate or passes_short_hanging_gate
    return {
        "shape_score": float(shape_score),
        "passes_shape_gate": passes,
        "vertical_dominance": vertical_dominance,
        "dark_ratio": dark_ratio,
        "aspect_ratio": aspect_ratio,
        "dark_col_over_row": dark_col_over_row,
        "thread_edge_ratio": float(thread_edge_ratio),
        "bottom_span_ratio": float(bottom_span_ratio),
        "passes_long_stud_gate": passes_long_stud_gate,
        "passes_short_hanging_gate": passes_short_hanging_gate,
    }


def execute_scan_plan(
    plan: dict[str, object],
    workspace: Path,
    backend: Callable[[Path, dict[str, object]], object],
    *,
    min_score: float,
    top_k: int,
    target_mode: str = "vertical-only",
) -> dict[str, object]:
    workspace = validate_workspace(workspace)
    output_dir = workspace / "dino" / "boxes_json"
    output_dir.mkdir(parents=True, exist_ok=True)

    generated: list[dict[str, object]] = []
    for record in plan.get("records", []):
        if not isinstance(record, dict):
            continue
        source_image = Path(str(record["source_image"]))
        with Image.open(source_image) as image:
            image_size = image.size
        asset_context = {
            "image_name": record["image_name"],
            "image_stem": record["image_stem"],
            "source_image": str(source_image),
            "workspace": str(workspace),
        }
        boxes = normalize_boxes(
            backend(source_image, asset_context),
            min_score=min_score,
            top_k=top_k,
        )
        if target_mode == "vertical-only":
            boxes = filter_vertical_candidates(boxes, image_size=image_size, source_image=source_image)
        payload = {
            "image": record["image_name"],
            "source_image": str(source_image),
            "boxes": boxes,
        }
        json_path = output_dir / f"{record['image_stem']}.json"
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        generated.append(
            {
                "image_name": record["image_name"],
                "source_image": str(source_image),
                "box_count": len(boxes),
                "json_path": str(json_path),
            }
        )

    return {
        **plan,
        "status": "executed",
        "mode": "good-bolt-dino-execute",
        "generated_json_count": len(generated),
        "generated_records": generated,
    }


def main() -> int:
    args = parse_args()
    if args.top_k < 0:
        return fail("--top-k must be greater than or equal to 0")
    if args.min_score < 0:
        return fail("--min-score must be greater than or equal to 0")
    if args.limit < 0:
        return fail("--limit must be greater than or equal to 0")

    try:
        plan = build_scan_plan(
            workspace=args.workspace,
            input_dir=args.input_dir,
            recursive=args.recursive,
            limit=args.limit,
        )
        if args.backend == "grounding-dino":
            backend = build_grounding_dino_backend(
                model_id=args.model_id,
                text_prompt=args.text_prompt,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
            )
        else:
            backend = resolve_backend(args.backend)
        report = execute_scan_plan(
            plan,
            args.workspace,
            backend,
            min_score=args.min_score,
            top_k=args.top_k,
            target_mode=args.target_mode,
        )
    except (ImportError, OSError, ValueError, json.JSONDecodeError) as exc:
        return fail(str(exc))

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
