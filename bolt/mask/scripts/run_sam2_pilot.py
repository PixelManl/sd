from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import re
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MASK_FILENAME_CORE = "core_mask.png"
MASK_FILENAME_EDIT = "edit_mask.png"
OVERLAY_FILENAME = "review_overlay.png"
CONTRACT_VERSION = "sam2_asset_contract/v1"
ASSET_LINE = "sam2_pilot"
DEFAULT_QA_STATE = "draft"
MASK_BACKEND_KEYS = ("core_mask", "edit_mask")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan a local SAM2 pilot asset run without requiring SAM2 itself."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing source images for the pilot slice.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Private local root for assets/, overlays/, and metadata/ outputs.",
    )
    parser.add_argument(
        "--pilot-run-id",
        required=True,
        help="Identifier for this pilot batch, for example pilot-20260322-01.",
    )
    parser.add_argument(
        "--defect-type",
        default="missing_fastener",
        help="Controlled defect label for this pilot.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of source images to include in the plan.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for images below --input-dir.",
    )
    parser.add_argument(
        "--init-layout",
        action="store_true",
        help="Create private output directories after validation.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Materialize local masks, overlays, and metadata via a runtime backend.",
    )
    parser.add_argument(
        "--backend",
        help="Runtime backend spec in module:attr or path.py:attr form. Required with --execute.",
    )
    return parser.parse_args()


def fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 2


def validate_input_dir(path_text: str) -> Path:
    path = Path(path_text).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"input directory does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"input path is not a directory: {path}")
    return path


def validate_output_root(path_text: str) -> Path:
    path = Path(path_text).expanduser().resolve()
    if path.exists() and not path.is_dir():
        raise ValueError(f"output root must be a directory path: {path}")
    parent = path.parent
    if not parent.exists():
        raise ValueError(f"output root parent does not exist: {parent}")
    if path.name.strip() == "":
        raise ValueError("output root must not be empty")
    return path


def collect_images(input_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    images = [
        path
        for path in sorted(input_dir.glob(pattern))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    return images


def build_plan(
    *,
    input_dir: Path,
    output_root: Path,
    pilot_run_id: str,
    defect_type: str,
    limit: int,
    recursive: bool,
    init_layout: bool,
) -> dict[str, object]:
    images = collect_images(input_dir, recursive=recursive)
    selected = images[:limit]
    return {
        "status": "placeholder",
        "mode": "sam2-pilot-plan",
        "sam2_dependency_required": False,
        "contract_version": CONTRACT_VERSION,
        "asset_line": ASSET_LINE,
        "pilot_run_id": pilot_run_id,
        "defect_type": defect_type,
        "input_dir": str(input_dir),
        "output_root": str(output_root),
        "layout_initialized": init_layout,
        "source_image_count": len(images),
        "selected_image_count": len(selected),
        "selected_images": [str(path) for path in selected],
        "planned_outputs": {
            "assets_dir": str(output_root / "assets"),
            "overlays_dir": str(output_root / "overlays"),
            "metadata_dir": str(output_root / "metadata"),
        },
        "semantics": {
            "core_mask": "conservative evidence mask around the visible missing-fastener signal",
            "edit_mask": "editing region that fully covers core_mask and provides context margin",
        },
        "next_step": (
            "Create local private masks and metadata with a human-reviewed workflow; "
            "this script does not invoke SAM2."
        ),
    }


def maybe_init_layout(output_root: Path) -> None:
    for name in ("assets", "overlays", "metadata"):
        (output_root / name).mkdir(parents=True, exist_ok=True)


def parse_backend_spec(spec_text: str) -> tuple[str, str]:
    module_ref, attribute = spec_text.rsplit(":", 1) if ":" in spec_text else ("", "")
    if not module_ref or not attribute:
        raise ValueError("--backend must be in module:attr or path.py:attr form")
    return module_ref, attribute


def resolve_backend(spec_text: str) -> Callable[[Path, dict[str, object]], dict[str, object]]:
    module_ref, attribute = parse_backend_spec(spec_text)
    candidate_path = Path(module_ref).expanduser()

    if candidate_path.suffix == ".py" and candidate_path.exists():
        module_name = f"_sam2_runtime_backend_{abs(hash(candidate_path.resolve()))}"
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


def sanitize_token(text: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return token.strip("._-") or "asset"


def build_asset_id(pilot_run_id: str, index: int, source_image: Path) -> str:
    return f"{sanitize_token(pilot_run_id)}-{index:04d}-{sanitize_token(source_image.stem)}"


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
    except Exception as exc:  # pragma: no cover - defensive conversion guard
        raise ValueError(
            f"{label} must be a PIL image, image path, or array-like image, got {type(image_like).__name__}"
        ) from exc
    if array.ndim not in (2, 3):
        raise ValueError(
            f"{label} array-like input must be 2D or 3D, got shape {array.shape!r}"
        )
    if np.issubdtype(array.dtype, np.bool_):
        array = array.astype(np.uint8) * 255
    elif np.issubdtype(array.dtype, np.number):
        array = np.clip(array, 0, 255).astype(np.uint8)
    else:
        raise ValueError(
            f"{label} array-like input must contain numeric or bool values, got {array.dtype!s}"
        )
    return Image.fromarray(array)


def coerce_mask_image(mask_like: object, size: tuple[int, int], label: str) -> Image.Image:
    image = load_image_like(mask_like, label).convert("L")
    if image.size != size:
        raise ValueError(f"{label} size mismatch: expected {size}, got {image.size}")
    return image.point(lambda value: 255 if value else 0, mode="L")


def edit_mask_covers_core(core_mask: Image.Image, edit_mask: Image.Image) -> bool:
    return ImageChops.subtract(core_mask, edit_mask).getbbox() is None


def build_overlay_image(
    source_image: Path,
    core_mask: Image.Image,
    edit_mask: Image.Image,
) -> Image.Image:
    with Image.open(source_image) as image:
        base = image.convert("RGBA")

    edit_layer = Image.new("RGBA", base.size, (255, 193, 7, 0))
    edit_layer.putalpha(edit_mask.point(lambda value: 96 if value else 0))

    core_layer = Image.new("RGBA", base.size, (220, 53, 69, 0))
    core_layer.putalpha(core_mask.point(lambda value: 160 if value else 0))

    return Image.alpha_composite(Image.alpha_composite(base, edit_layer), core_layer)


def coerce_overlay_image(
    overlay_like: object | None,
    source_image: Path,
    core_mask: Image.Image,
    edit_mask: Image.Image,
) -> Image.Image:
    if overlay_like is None:
        return build_overlay_image(source_image, core_mask, edit_mask)

    overlay = load_image_like(overlay_like, "overlay").convert("RGBA")
    with Image.open(source_image) as image:
        size = image.size
    if overlay.size != size:
        raise ValueError(f"overlay size mismatch: expected {size}, got {overlay.size}")
    return overlay


def require_backend_output(
    payload: object,
    *,
    source_image: Path,
    asset_id: str,
) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError(
            f"backend must return a dict for {asset_id} ({source_image.name}), got {type(payload).__name__}"
        )
    for key in MASK_BACKEND_KEYS:
        if key not in payload:
            raise ValueError(f"backend result missing required key '{key}' for asset {asset_id}")
    return payload


def build_asset_context(
    *,
    plan: dict[str, object],
    source_image: Path,
    asset_id: str,
    index: int,
) -> dict[str, object]:
    return {
        "asset_id": asset_id,
        "asset_index": index,
        "pilot_run_id": plan["pilot_run_id"],
        "defect_type": plan["defect_type"],
        "source_image": str(source_image),
        "output_root": plan["output_root"],
        "contract_version": plan["contract_version"],
        "asset_line": plan["asset_line"],
    }


def build_metadata_payload(
    *,
    plan: dict[str, object],
    asset_id: str,
    source_image: Path,
    core_mask_path: Path,
    edit_mask_path: Path,
    overlay_path: Path,
    backend_payload: dict[str, object],
    backend_label: str,
) -> dict[str, object]:
    timestamp = datetime.now(timezone.utc).isoformat()
    metadata: dict[str, object] = {
        "contract_version": plan["contract_version"],
        "asset_line": plan["asset_line"],
        "asset_id": asset_id,
        "pilot_run_id": plan["pilot_run_id"],
        "defect_type": plan["defect_type"],
        "source_image": str(source_image),
        "core_mask_path": str(core_mask_path),
        "edit_mask_path": str(edit_mask_path),
        "overlay_path": str(overlay_path),
        "qa_state": str(backend_payload.get("qa_state") or DEFAULT_QA_STATE),
        "created_at": timestamp,
        "updated_at": timestamp,
        "tool_name": str(backend_payload.get("tool_name") or backend_label),
    }

    optional_fields = (
        "qa_notes",
        "reviewer",
        "lineage_parent",
        "roi_id",
        "source_image_sha256",
    )
    for key in optional_fields:
        value = backend_payload.get(key)
        if value not in (None, ""):
            metadata[key] = value

    tool_version = backend_payload.get("tool_version")
    if tool_version not in (None, ""):
        metadata["tool_version"] = str(tool_version)

    return metadata


def execute_plan(
    *,
    plan: dict[str, object],
    backend: Callable[[Path, dict[str, object]], dict[str, object]],
    backend_label: str,
) -> dict[str, object]:
    output_root = Path(str(plan["output_root"]))
    maybe_init_layout(output_root)

    generated_assets: list[dict[str, object]] = []
    selected_images = [Path(path_text) for path_text in plan["selected_images"]]

    for index, source_image in enumerate(selected_images, start=1):
        asset_id = build_asset_id(str(plan["pilot_run_id"]), index, source_image)
        asset_context = build_asset_context(
            plan=plan,
            source_image=source_image,
            asset_id=asset_id,
            index=index,
        )
        backend_payload = require_backend_output(
            backend(source_image, asset_context),
            source_image=source_image,
            asset_id=asset_id,
        )

        with Image.open(source_image) as image:
            source_size = image.size

        core_mask = coerce_mask_image(backend_payload["core_mask"], source_size, "core_mask")
        edit_mask = coerce_mask_image(backend_payload["edit_mask"], source_size, "edit_mask")
        if not edit_mask_covers_core(core_mask, edit_mask):
            raise ValueError(f"edit_mask must fully cover core_mask for asset {asset_id}")

        overlay = coerce_overlay_image(
            backend_payload.get("overlay"),
            source_image,
            core_mask,
            edit_mask,
        )

        asset_dir = output_root / "assets" / asset_id
        overlay_dir = output_root / "overlays" / asset_id
        metadata_path = output_root / "metadata" / f"{asset_id}.json"
        asset_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        core_mask_path = asset_dir / MASK_FILENAME_CORE
        edit_mask_path = asset_dir / MASK_FILENAME_EDIT
        overlay_path = overlay_dir / OVERLAY_FILENAME

        core_mask.save(core_mask_path)
        edit_mask.save(edit_mask_path)
        overlay.save(overlay_path)

        metadata = build_metadata_payload(
            plan=plan,
            asset_id=asset_id,
            source_image=source_image,
            core_mask_path=core_mask_path,
            edit_mask_path=edit_mask_path,
            overlay_path=overlay_path,
            backend_payload=backend_payload,
            backend_label=backend_label,
        )
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        generated_assets.append(
            {
                "asset_id": asset_id,
                "source_image": str(source_image),
                "core_mask_path": str(core_mask_path),
                "edit_mask_path": str(edit_mask_path),
                "overlay_path": str(overlay_path),
                "metadata_path": str(metadata_path),
                "qa_state": metadata["qa_state"],
            }
        )

    return {
        **plan,
        "status": "executed",
        "mode": "sam2-pilot-execute",
        "sam2_dependency_required": True,
        "layout_initialized": True,
        "runtime_backend": backend_label,
        "generated_asset_count": len(generated_assets),
        "generated_assets": generated_assets,
        "next_step": (
            "Run review_sam2_assets.py on the local metadata directory and complete human QA "
            "before treating these draft assets as usable."
        ),
    }


def main() -> int:
    args = parse_args()
    if args.limit <= 0:
        return fail("--limit must be greater than 0")
    if args.execute and not args.backend:
        return fail("--backend is required with --execute")

    try:
        input_dir = validate_input_dir(args.input_dir)
        output_root = validate_output_root(args.output_root)
    except ValueError as exc:
        return fail(str(exc))

    if args.init_layout:
        maybe_init_layout(output_root)

    plan = build_plan(
        input_dir=input_dir,
        output_root=output_root,
        pilot_run_id=args.pilot_run_id,
        defect_type=args.defect_type,
        limit=args.limit,
        recursive=args.recursive,
        init_layout=args.init_layout,
    )

    if plan["source_image_count"] == 0:
        return fail(f"no image files found under input directory: {input_dir}")

    report = plan
    if args.execute:
        try:
            report = execute_plan(
                plan=plan,
                backend=resolve_backend(args.backend),
                backend_label=args.backend,
            )
        except ValueError as exc:
            return fail(str(exc))

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
