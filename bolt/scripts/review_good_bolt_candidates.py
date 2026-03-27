from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review local healthy bolt candidates and summarize DINO box coverage."
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        required=True,
        help="Workspace root containing incoming/images and dino/boxes_json.",
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


def iter_images(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        return []
    return [
        path
        for path in sorted(images_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]


def load_boxes(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    boxes = payload.get("boxes", [])
    return [item for item in boxes if isinstance(item, dict)] if isinstance(boxes, list) else []


def build_review_summary(workspace: Path) -> dict[str, object]:
    workspace = validate_workspace(workspace)
    images_dir = workspace / "incoming" / "images"
    boxes_dir = workspace / "dino" / "boxes_json"

    records: list[dict[str, object]] = []
    with_boxes = 0
    missing_boxes = 0

    for image_path in iter_images(images_dir):
        box_path = boxes_dir / f"{image_path.stem}.json"
        boxes = load_boxes(box_path)
        has_boxes = len(boxes) > 0
        if has_boxes:
            with_boxes += 1
        else:
            missing_boxes += 1

        records.append(
            {
                "image_name": image_path.name,
                "source_image": str(image_path),
                "box_json": str(box_path),
                "box_count": len(boxes),
                "has_boxes": has_boxes,
            }
        )

    return {
        "status": "placeholder",
        "mode": "good-bolt-candidate-review",
        "workspace": str(workspace),
        "image_count": len(records),
        "with_boxes": with_boxes,
        "missing_boxes": missing_boxes,
        "records": records,
    }


def main() -> int:
    args = parse_args()
    try:
        summary = build_review_summary(args.workspace)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return fail(str(exc))

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
