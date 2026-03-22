from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


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
        "contract_version": "sam2_asset_contract/v1",
        "asset_line": "sam2_pilot",
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


def main() -> int:
    args = parse_args()
    if args.limit <= 0:
        return fail("--limit must be greater than 0")

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

    print(json.dumps(plan, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
