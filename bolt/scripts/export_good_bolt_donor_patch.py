from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a donor RGB patch and alpha mask from a healthy SAM2 bolt asset."
    )
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--padding", type=int, default=12)
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


def load_asset_metadata(workspace: Path, asset_id: str) -> dict[str, object]:
    metadata_path = workspace / "sam2" / "metadata" / f"{asset_id}.json"
    if not metadata_path.exists():
        raise ValueError(f"metadata does not exist: {metadata_path}")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"metadata payload is not a dict: {metadata_path}")
    return payload


def resolve_local_image_path(workspace: Path, metadata: dict[str, object]) -> Path:
    image_name = str(metadata.get("image_name") or "").strip()
    source_image = str(metadata.get("source_image") or "").strip()

    candidates = []
    if source_image:
        candidates.append(Path(source_image).expanduser())
    if image_name:
        candidates.append(workspace / "incoming" / "images" / image_name)

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    raise ValueError(f"unable to resolve local source image for asset {metadata.get('asset_id')}")


def resolve_local_core_mask_path(workspace: Path, metadata: dict[str, object]) -> Path:
    raw_path = str(metadata.get("core_mask_path") or "").strip()
    if raw_path:
        basename = Path(raw_path).name
        candidate = (workspace / "sam2" / "core_masks" / basename).resolve()
        if candidate.exists():
            return candidate
    asset_id = str(metadata.get("asset_id") or metadata.get("roi_id") or "").strip()
    candidate = (workspace / "sam2" / "core_masks" / f"{asset_id}.png").resolve()
    if candidate.exists():
        return candidate
    raise ValueError(f"unable to resolve local core mask for asset {metadata.get('asset_id')}")


def crop_bounds_from_mask(mask: Image.Image, padding: int) -> tuple[int, int, int, int]:
    bbox = mask.getbbox()
    if bbox is None:
        raise ValueError("core mask is empty")
    x1, y1, x2, y2 = bbox
    return (
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(mask.size[0], x2 + padding),
        min(mask.size[1], y2 + padding),
    )


def export_donor_patch(
    workspace: Path,
    asset_id: str,
    output_dir: Path,
    *,
    padding: int = 12,
) -> dict[str, object]:
    workspace = validate_workspace(workspace)
    metadata = load_asset_metadata(workspace, asset_id)
    image_path = resolve_local_image_path(workspace, metadata)
    core_mask_path = resolve_local_core_mask_path(workspace, metadata)

    output_root = output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as source_image:
        rgb_image = source_image.convert("RGB")
        with Image.open(core_mask_path) as core_mask_image:
            core_mask = core_mask_image.convert("L")
            crop_box = crop_bounds_from_mask(core_mask, max(0, padding))
            donor_rgb = rgb_image.crop(crop_box)
            donor_alpha = core_mask.crop(crop_box)

    donor_rgb_path = output_root / f"{asset_id}_donor_rgb.png"
    donor_alpha_path = output_root / f"{asset_id}_donor_alpha.png"
    donor_rgba_path = output_root / f"{asset_id}_donor_rgba.png"

    donor_rgb.save(donor_rgb_path)
    donor_alpha.save(donor_alpha_path)
    donor_rgba = donor_rgb.copy()
    donor_rgba.putalpha(donor_alpha)
    donor_rgba.save(donor_rgba_path)

    summary = {
        "status": "placeholder",
        "mode": "good-bolt-donor-export",
        "asset_id": asset_id,
        "image_path": str(image_path),
        "core_mask_path": str(core_mask_path),
        "crop_box": list(crop_box),
        "donor_rgb_path": str(donor_rgb_path),
        "donor_alpha_path": str(donor_alpha_path),
        "donor_rgba_path": str(donor_rgba_path),
    }
    return summary


def main() -> int:
    args = parse_args()
    if args.padding < 0:
        return fail("--padding must be greater than or equal to 0")
    try:
        summary = export_donor_patch(
            args.workspace,
            args.asset_id,
            args.output_dir,
            padding=args.padding,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return fail(str(exc))

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
