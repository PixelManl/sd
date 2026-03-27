from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image


USABLE_QA_STATES = {"usable", "accepted"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export healthy bolt ROI crops from the local good bolt asset manifest."
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        required=True,
        help="Workspace root containing manifests/ and exports/healthy_roi_bank/.",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=0,
        help="Extra pixels to add around each ROI crop.",
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


def load_manifest_records(workspace: Path) -> list[dict[str, object]]:
    manifest_path = workspace / "manifests" / "good_bolt_assets_manifest.jsonl"
    if not manifest_path.exists():
        raise ValueError(f"manifest does not exist: {manifest_path}")

    records: list[dict[str, object]] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            records.append(payload)
    return records


def clamp_crop(box_xyxy: list[int], width: int, height: int, padding: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(value) for value in box_xyxy]
    return (
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(width, x2 + padding),
        min(height, y2 + padding),
    )


def export_roi_bank(workspace: Path, padding: int = 0) -> dict[str, object]:
    workspace = validate_workspace(workspace)
    export_dir = workspace / "exports" / "healthy_roi_bank"
    export_dir.mkdir(parents=True, exist_ok=True)

    exported: list[dict[str, object]] = []
    for record in load_manifest_records(workspace):
        qa_state = str(record.get("qa_state") or "").strip().lower()
        if qa_state not in USABLE_QA_STATES:
            continue

        source_image = Path(str(record["source_image"])).expanduser().resolve()
        roi_id = str(record.get("roi_id") or Path(source_image).stem)
        box_xyxy = record.get("box_xyxy")
        if not isinstance(box_xyxy, list) or len(box_xyxy) != 4:
            continue

        with Image.open(source_image) as image:
            crop_box = clamp_crop(box_xyxy, image.size[0], image.size[1], max(0, padding))
            roi = image.crop(crop_box)

        roi_path = export_dir / f"{roi_id}.png"
        roi.save(roi_path)
        exported.append(
            {
                "roi_id": roi_id,
                "image_name": record.get("image_name"),
                "target_class": record.get("target_class"),
                "qa_state": qa_state,
                "box_xyxy": box_xyxy,
                "roi_path": str(roi_path),
            }
        )

    return {
        "status": "placeholder",
        "mode": "good-bolt-roi-export",
        "workspace": str(workspace),
        "export_dir": str(export_dir),
        "exported_count": len(exported),
        "records": exported,
    }


def main() -> int:
    args = parse_args()
    if args.padding < 0:
        return fail("--padding must be greater than or equal to 0")

    try:
        summary = export_roi_bank(args.workspace, padding=args.padding)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return fail(str(exc))

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
