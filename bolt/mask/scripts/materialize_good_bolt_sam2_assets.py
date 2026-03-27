from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageChops


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
APPROVED_REVIEW_STATES = {"approved", "accepted", "usable"}
CONTRACT_VERSION = "sam2_asset_contract/v1"
ASSET_LINE = "sam2_pilot"
QA_STATE_ALIASES = {
    "candidate": "draft",
}
ALLOWED_QA_STATES = {
    "planned",
    "draft",
    "needs_review",
    "accepted",
    "rejected",
    "superseded",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize reviewed healthy bolt boxes into local SAM2-style assets."
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        required=True,
        help="Workspace root containing incoming/images, dino/boxes_json, and sam2/ outputs.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the backend and write masks, overlays, and metadata.",
    )
    parser.add_argument(
        "--backend",
        help="Runtime backend spec in module:attr or path.py:attr form. Required with --execute.",
    )
    parser.add_argument(
        "--labelme-dir",
        type=Path,
        help="Optional manual LabelMe JSON directory. When provided, asset planning reads rectangles from this directory instead of dino/boxes_json.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        help="Optional external image directory. Defaults to <workspace>/incoming/images.",
    )
    return parser.parse_args()


def fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 2


def validate_workspace(workspace: Path) -> Path:
    resolved = workspace.expanduser().resolve()
    if not resolved.exists():
        raise ValueError(f"workspace does not exist: {resolved}")
    if not resolved.is_dir():
        raise ValueError(f"workspace is not a directory: {resolved}")
    return resolved


def validate_directory(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise ValueError(f"{label} does not exist: {resolved}")
    if not resolved.is_dir():
        raise ValueError(f"{label} is not a directory: {resolved}")
    return resolved


def parse_backend_spec(spec_text: str) -> tuple[str, str]:
    module_ref, attribute = spec_text.rsplit(":", 1) if ":" in spec_text else ("", "")
    if not module_ref or not attribute:
        raise ValueError("--backend must be in module:attr or path.py:attr form")
    return module_ref, attribute


def resolve_backend(spec_text: str) -> Callable[[Path, dict[str, object]], dict[str, object]]:
    module_ref, attribute = parse_backend_spec(spec_text)
    candidate_path = Path(module_ref).expanduser()

    if candidate_path.suffix == ".py" and candidate_path.exists():
        module_name = f"_good_bolt_backend_{abs(hash(candidate_path.resolve()))}"
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


def iter_images(images_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(images_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]


def sanitize_token(text: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip())
    return token.strip("._-") or "good-bolt"


def clamp_box(box_xyxy: list[int], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = [int(value) for value in box_xyxy]
    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    x2 = max(x1, min(width, x2))
    y2 = max(y1, min(height, y2))
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


def resolve_image_reference(image_name: str, image_index: dict[str, Path]) -> Path | None:
    if image_name in image_index:
        return image_index[image_name]
    basename = Path(image_name).name
    if basename in image_index:
        return image_index[basename]
    return None


def build_asset_plan_from_labelme(
    labelme_dir: Path,
    image_index: dict[str, Path],
) -> list[dict[str, object]]:
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
            shape_type = str(shape.get("shape_type") or "").strip().lower()
            if shape_type not in {"rectangle", "polygon"}:
                continue
            raw_box = points_to_box_xyxy(shape.get("points", []))
            if raw_box is None:
                continue
            manual_label = str(shape.get("label") or "healthy_bolt").strip()
            records.append(
                {
                    "roi_id": f"{labelme_path.stem}-manual-{shape_index:03d}",
                    "image_name": image_name,
                    "source_image": str(source_image),
                    "box_xyxy": clamp_box(raw_box, width, height),
                    "target_class": "healthy_bolt",
                    "score": None,
                    "review_state": "approved",
                    "box_index": shape_index,
                    "manual_label": manual_label,
                    "label_source": "labelme",
                }
            )
    return records


def build_asset_plan(
    workspace: Path,
    *,
    labelme_dir: Path | None = None,
    image_dir: Path | None = None,
) -> dict[str, object]:
    workspace = validate_workspace(workspace)
    images_dir = validate_directory(
        image_dir or (workspace / "incoming" / "images"),
        "image directory",
    )
    boxes_dir = workspace / "dino" / "boxes_json"

    image_index = {path.name: path for path in iter_images(images_dir)}
    if labelme_dir is not None:
        validated_labelme_dir = validate_directory(labelme_dir, "labelme directory")
        records = build_asset_plan_from_labelme(
            labelme_dir=validated_labelme_dir,
            image_index=image_index,
        )
        return {
            "status": "placeholder",
            "mode": "good-bolt-sam2-plan",
            "workspace": str(workspace),
            "asset_count": len(records),
            "records": records,
            "label_source": "labelme",
            "labelme_dir": str(validated_labelme_dir),
            "image_dir": str(images_dir),
            "pilot_run_id": f"{sanitize_token(workspace.name)}-good-bolt-sam2",
        }

    records: list[dict[str, object]] = []
    asset_index = 0

    for box_path in sorted(boxes_dir.glob("*.json")):
        payload = json.loads(box_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue

        image_name = str(payload.get("image") or f"{box_path.stem}.jpg")
        source_image = image_index.get(image_name)
        if source_image is None:
            continue

        boxes = payload.get("boxes", [])
        if not isinstance(boxes, list):
            continue

        with Image.open(source_image) as image:
            width, height = image.size

        for box_idx, box in enumerate(boxes, start=1):
            if not isinstance(box, dict):
                continue
            review_state = str(box.get("review_state") or "").strip().lower()
            if review_state not in APPROVED_REVIEW_STATES:
                continue
            raw_box = box.get("box_xyxy")
            if not isinstance(raw_box, list) or len(raw_box) != 4:
                continue

            asset_index += 1
            roi_id = str(box.get("roi_id") or f"good-bolt-{asset_index:04d}")
            records.append(
                {
                    "roi_id": roi_id,
                    "image_name": image_name,
                    "source_image": str(source_image),
                    "box_xyxy": clamp_box(raw_box, width, height),
                    "target_class": str(box.get("label") or "healthy_bolt"),
                    "score": box.get("score"),
                    "review_state": review_state,
                    "box_index": box_idx,
                }
            )

    return {
        "status": "placeholder",
        "mode": "good-bolt-sam2-plan",
        "workspace": str(workspace),
        "asset_count": len(records),
        "records": records,
        "label_source": "dino",
        "image_dir": str(images_dir),
        "pilot_run_id": f"{sanitize_token(workspace.name)}-good-bolt-sam2",
    }


def load_image_like(image_like: object, label: str) -> Image.Image:
    if isinstance(image_like, Image.Image):
        return image_like.copy()
    if isinstance(image_like, (str, Path)):
        path = Path(image_like).expanduser().resolve()
        if not path.exists():
            raise ValueError(f"{label} path does not exist: {path}")
        with Image.open(path) as image:
            return image.copy()
    try:
        array = np.asarray(image_like)
    except Exception as exc:  # pragma: no cover
        raise ValueError(
            f"{label} must be a PIL image, image path, or array-like image, got {type(image_like).__name__}"
        ) from exc
    if array.ndim not in (2, 3):
        raise ValueError(f"{label} array-like input must be 2D or 3D, got shape {array.shape!r}")
    if np.issubdtype(array.dtype, np.bool_):
        array = array.astype(np.uint8) * 255
    elif np.issubdtype(array.dtype, np.number):
        array = np.clip(array, 0, 255).astype(np.uint8)
    else:
        raise ValueError(f"{label} array-like input must contain numeric or bool values, got {array.dtype!s}")
    return Image.fromarray(array)


def coerce_mask_image(mask_like: object, size: tuple[int, int], label: str) -> Image.Image:
    image = load_image_like(mask_like, label).convert("L")
    if image.size != size:
        raise ValueError(f"{label} size mismatch: expected {size}, got {image.size}")
    return image.point(lambda value: 255 if value else 0, mode="L")


def mask_has_foreground(mask: Image.Image) -> bool:
    return mask.getbbox() is not None


def edit_mask_covers_core(core_mask: Image.Image, edit_mask: Image.Image) -> bool:
    return ImageChops.subtract(core_mask, edit_mask).getbbox() is None


def build_overlay_image(base: Image.Image, core_mask: Image.Image, edit_mask: Image.Image) -> Image.Image:
    edit_layer = Image.new("RGBA", base.size, (255, 193, 7, 0))
    edit_layer.putalpha(edit_mask.point(lambda value: 96 if value else 0))

    core_layer = Image.new("RGBA", base.size, (220, 53, 69, 0))
    core_layer.putalpha(core_mask.point(lambda value: 160 if value else 0))

    return Image.alpha_composite(Image.alpha_composite(base, edit_layer), core_layer)


def coerce_overlay_image(
    overlay_like: object | None,
    base_image: Image.Image,
    core_mask: Image.Image,
    edit_mask: Image.Image,
) -> Image.Image:
    if overlay_like is None:
        return build_overlay_image(base_image, core_mask, edit_mask)
    overlay = load_image_like(overlay_like, "overlay").convert("RGBA")
    if overlay.size != base_image.size:
        raise ValueError(f"overlay size mismatch: expected {base_image.size}, got {overlay.size}")
    return overlay


def require_backend_output(
    payload: object,
    *,
    source_image: Path,
    roi_id: str,
) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError(
            f"backend must return a dict for {roi_id} ({source_image.name}), got {type(payload).__name__}"
        )
    for key in ("core_mask", "edit_mask"):
        if key not in payload:
            raise ValueError(f"backend result missing required key '{key}' for asset {roi_id}")
    return payload


def build_metadata_payload(
    *,
    record: dict[str, object],
    core_mask_path: Path,
    edit_mask_path: Path,
    overlay_path: Path,
    backend_payload: dict[str, object],
    backend_label: str,
) -> dict[str, object]:
    timestamp = datetime.now(timezone.utc).isoformat()
    payload: dict[str, object] = {
        "contract_version": CONTRACT_VERSION,
        "asset_line": ASSET_LINE,
        "asset_id": record["roi_id"],
        "pilot_run_id": str(record["pilot_run_id"]),
        "defect_type": str(record["target_class"]),
        "roi_id": record["roi_id"],
        "image_name": record["image_name"],
        "source_image": record["source_image"],
        "box_xyxy": record["box_xyxy"],
        "target_class": record["target_class"],
        "qa_state": normalize_qa_state(backend_payload.get("qa_state")),
        "core_mask_path": str(core_mask_path),
        "edit_mask_path": str(edit_mask_path),
        "overlay_path": str(overlay_path),
        "tool_name": str(backend_payload.get("tool_name") or backend_label),
        "created_at": timestamp,
        "updated_at": timestamp,
    }

    tool_version = backend_payload.get("tool_version")
    if tool_version not in (None, ""):
        payload["tool_version"] = str(tool_version)
    qa_notes = backend_payload.get("qa_notes")
    if qa_notes not in (None, ""):
        payload["qa_notes"] = str(qa_notes)
    return payload


def normalize_qa_state(value: object) -> str:
    raw = str(value or "draft").strip().lower()
    normalized = QA_STATE_ALIASES.get(raw, raw)
    if normalized not in ALLOWED_QA_STATES:
        raise ValueError(f"unsupported qa_state for asset metadata: {raw}")
    return normalized


def ensure_output_layout(workspace: Path) -> dict[str, Path]:
    sam2_root = workspace / "sam2"
    layout = {
        "core_masks": sam2_root / "core_masks",
        "edit_masks": sam2_root / "edit_masks",
        "overlays": sam2_root / "overlays",
        "metadata": sam2_root / "metadata",
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout


def ensure_unique_output_stems(records: list[dict[str, object]]) -> None:
    seen: dict[str, str] = {}
    for record in records:
        roi_id = str(record["roi_id"])
        stem = sanitize_token(roi_id)
        previous = seen.get(stem)
        if previous is not None:
            raise ValueError(
                f"duplicate output stem '{stem}' generated from roi_id '{previous}' and '{roi_id}'"
            )
        seen[stem] = roi_id


def build_cli_report(report: dict[str, object]) -> dict[str, object]:
    summary = {
        "status": report.get("status"),
        "mode": report.get("mode"),
        "workspace": report.get("workspace"),
        "asset_count": report.get("asset_count"),
        "generated_asset_count": report.get("generated_asset_count"),
        "failed_asset_count": report.get("failed_asset_count", 0),
        "label_source": report.get("label_source"),
        "image_dir": report.get("image_dir"),
        "pilot_run_id": report.get("pilot_run_id"),
    }
    failed_assets = report.get("failed_assets")
    if isinstance(failed_assets, list) and failed_assets:
        summary["failed_assets"] = failed_assets
    return summary


def execute_asset_plan(
    plan: dict[str, object],
    workspace: Path,
    backend: Callable[[Path, dict[str, object]], dict[str, object]],
) -> dict[str, object]:
    workspace = validate_workspace(workspace)
    layout = ensure_output_layout(workspace)
    plan_records = [record for record in plan.get("records", []) if isinstance(record, dict)]
    ensure_unique_output_stems(plan_records)
    generated_assets: list[dict[str, object]] = []
    pilot_run_id = str(plan.get("pilot_run_id") or f"{sanitize_token(workspace.name)}-good-bolt-sam2")
    cached_source_key: str | None = None
    cached_source_size: tuple[int, int] | None = None
    cached_source_rgb: np.ndarray | None = None
    cached_base_rgba: Image.Image | None = None

    for index, record in enumerate(plan_records, start=1):
        source_image = Path(str(record["source_image"]))
        source_key = str(source_image.expanduser().resolve())
        if source_key != cached_source_key:
            with Image.open(source_image) as image:
                rgb = image.convert("RGB")
                cached_source_size = rgb.size
                cached_source_rgb = np.array(rgb, copy=True)
                cached_base_rgba = rgb.convert("RGBA")
            cached_source_key = source_key

        assert cached_source_size is not None
        assert cached_source_rgb is not None
        assert cached_base_rgba is not None
        source_size = cached_source_size
        clamped_box = clamp_box(list(record["box_xyxy"]), source_size[0], source_size[1])
        if not box_has_positive_area(clamped_box):
            raise ValueError(f"box_xyxy must have positive area after clamping for asset {record['roi_id']}")

        asset_context = {
            "asset_index": index,
            "asset_id": record["roi_id"],
            "roi_id": record["roi_id"],
            "image_name": record["image_name"],
            "source_image": str(source_image),
            "box_xyxy": clamped_box,
            "target_class": record["target_class"],
            "workspace": str(workspace),
            "source_size": source_size,
            "source_rgb": cached_source_rgb,
        }
        backend_payload = require_backend_output(
            backend(source_image, asset_context),
            source_image=source_image,
            roi_id=str(record["roi_id"]),
        )

        core_mask = coerce_mask_image(backend_payload["core_mask"], source_size, "core_mask")
        edit_mask = coerce_mask_image(backend_payload["edit_mask"], source_size, "edit_mask")
        if not mask_has_foreground(core_mask):
            raise ValueError(f"core_mask must contain foreground for asset {record['roi_id']}")
        if not mask_has_foreground(edit_mask):
            raise ValueError(f"edit_mask must contain foreground for asset {record['roi_id']}")
        if not edit_mask_covers_core(core_mask, edit_mask):
            raise ValueError(f"edit_mask must fully cover core_mask for asset {record['roi_id']}")

        overlay = coerce_overlay_image(
            backend_payload.get("overlay"),
            cached_base_rgba.copy(),
            core_mask,
            edit_mask,
        )

        stem = sanitize_token(str(record["roi_id"]))
        core_mask_path = layout["core_masks"] / f"{stem}.png"
        edit_mask_path = layout["edit_masks"] / f"{stem}.png"
        overlay_path = layout["overlays"] / f"{stem}.png"
        metadata_path = layout["metadata"] / f"{stem}.json"

        core_mask.save(core_mask_path)
        edit_mask.save(edit_mask_path)
        overlay.save(overlay_path)

        metadata = build_metadata_payload(
            record={**record, "pilot_run_id": pilot_run_id},
            core_mask_path=core_mask_path,
            edit_mask_path=edit_mask_path,
            overlay_path=overlay_path,
            backend_payload=backend_payload,
            backend_label=getattr(backend, "__name__", "good-bolt-backend"),
        )
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        generated_assets.append(
            {
                "asset_id": record["roi_id"],
                "source_image": str(source_image),
                "core_mask_path": str(core_mask_path),
                "edit_mask_path": str(edit_mask_path),
                "overlay_path": str(overlay_path),
                "metadata_path": str(metadata_path),
            }
        )

    return {
        **{key: value for key, value in plan.items() if key != "records"},
        "status": "executed",
        "mode": "good-bolt-sam2-execute",
        "generated_asset_count": len(generated_assets),
        "generated_assets": generated_assets,
        "pilot_run_id": pilot_run_id,
        "failed_asset_count": 0,
    }


def main() -> int:
    args = parse_args()
    if args.execute and not args.backend:
        return fail("--backend is required with --execute")

    try:
        plan = build_asset_plan(
            args.workspace,
            labelme_dir=args.labelme_dir,
            image_dir=args.image_dir,
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError, ImportError) as exc:
        return fail(str(exc))

    report: dict[str, object] = plan
    if args.execute:
        try:
            backend = resolve_backend(args.backend)
            report = execute_asset_plan(plan, args.workspace, backend)
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError, ImportError) as exc:
            return fail(str(exc))

    print(json.dumps(build_cli_report(report), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
