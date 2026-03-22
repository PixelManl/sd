from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


REQUIRED_FIELDS = {
    "contract_version",
    "asset_line",
    "asset_id",
    "pilot_run_id",
    "defect_type",
    "source_image",
    "core_mask_path",
    "edit_mask_path",
    "overlay_path",
    "qa_state",
    "created_at",
    "updated_at",
}

PATH_FIELDS = ("core_mask_path", "edit_mask_path", "overlay_path")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review local SAM2 pilot metadata and summarize QA coverage."
    )
    parser.add_argument(
        "--metadata-root",
        required=True,
        help="Directory containing per-asset metadata JSON files.",
    )
    parser.add_argument(
        "--asset-id",
        help="Optional asset_id filter for reviewing a single asset.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when missing fields or broken paths are detected.",
    )
    return parser.parse_args()


def fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 2


def validate_directory(path_text: str, label: str) -> Path:
    path = Path(path_text).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"{label} does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"{label} is not a directory: {path}")
    return path


def load_metadata_files(metadata_root: Path) -> list[tuple[Path, dict[str, object]]]:
    records: list[tuple[Path, dict[str, object]]] = []
    for path in sorted(metadata_root.rglob("*.json")):
        if "manifests" in path.parts:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"metadata file must contain a JSON object: {path}")
        records.append((path, payload))
    return records


def to_path(base_dir: Path, value: object) -> Path | None:
    if not isinstance(value, str) or value.strip() == "":
        return None
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (base_dir / candidate).resolve()


def review_record(file_path: Path, payload: dict[str, object]) -> dict[str, object]:
    missing_fields = sorted(field for field in REQUIRED_FIELDS if field not in payload)
    broken_paths: list[str] = []
    for field in PATH_FIELDS:
        candidate = to_path(file_path.parent, payload.get(field))
        if candidate is None or not candidate.exists():
            broken_paths.append(field)

    return {
        "file": str(file_path),
        "asset_id": payload.get("asset_id"),
        "qa_state": payload.get("qa_state", "unknown"),
        "missing_fields": missing_fields,
        "broken_paths": broken_paths,
    }


def main() -> int:
    args = parse_args()
    try:
        metadata_root = validate_directory(args.metadata_root, "metadata root")
        loaded = load_metadata_files(metadata_root)
    except ValueError as exc:
        return fail(str(exc))

    filtered: list[dict[str, object]] = []
    for file_path, payload in loaded:
        if args.asset_id and payload.get("asset_id") != args.asset_id:
            continue
        filtered.append(review_record(file_path, payload))

    qa_counts = Counter(str(item["qa_state"]) for item in filtered)
    assets_with_gaps = [
        item
        for item in filtered
        if item["missing_fields"] or item["broken_paths"]
    ]

    report = {
        "status": "placeholder",
        "mode": "sam2-asset-review",
        "sam2_dependency_required": False,
        "metadata_root": str(metadata_root),
        "asset_filter": args.asset_id,
        "record_count": len(filtered),
        "qa_state_counts": dict(sorted(qa_counts.items())),
        "assets_with_gaps": assets_with_gaps,
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.strict and assets_with_gaps:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
