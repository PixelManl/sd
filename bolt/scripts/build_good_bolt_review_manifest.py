from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a local review manifest for good bolt SAM2 assets."
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        required=True,
        help="Workspace root containing incoming/images, sam2/metadata, and manual_labels.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional manifest output path. Defaults to <workspace>/manifests/good_bolt_assets_manifest.jsonl.",
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
    metadata_dir = resolved / "sam2" / "metadata"
    if not metadata_dir.exists():
        raise ValueError(f"metadata directory does not exist: {metadata_dir}")
    return resolved


def resolve_local_image_path(workspace: Path, metadata: dict[str, object]) -> Path:
    image_name = str(metadata.get("image_name") or "").strip()
    source_image = str(metadata.get("source_image") or "").strip()

    candidates: list[Path] = []
    if image_name:
        candidates.append(workspace / "incoming" / "images" / image_name)
        candidates.append(REPO_ROOT / "data" / "bolt_parallel" / "good_bolt_assets" / "incoming" / "images" / image_name)
    if source_image:
        candidates.append(workspace / "incoming" / "images" / Path(source_image).name)
        candidates.append(
            REPO_ROOT / "data" / "bolt_parallel" / "good_bolt_assets" / "incoming" / "images" / Path(source_image).name
        )
        candidates.append(Path(source_image).expanduser())

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise ValueError(f"unable to resolve local source image for asset {metadata.get('asset_id')}")


def resolve_local_asset_path(workspace: Path, raw_path: object, fallback_dir: str, asset_id: str) -> str:
    raw_text = str(raw_path or "").strip()
    candidates: list[Path] = []
    if raw_text:
        candidates.append(workspace / fallback_dir / Path(raw_text).name)
        candidates.append(Path(raw_text).expanduser())
    if asset_id:
        candidates.append(workspace / fallback_dir / f"{asset_id}.png")

    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return ""


def resolve_labelme_path(workspace: Path, image_path: Path) -> str:
    candidate = workspace / "manual_labels" / "healthy_labelme_json" / f"{image_path.stem}.json"
    if candidate.exists():
        return str(candidate.resolve())
    return ""


def load_metadata_records(metadata_root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(metadata_root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"metadata file must contain a JSON object: {path}")
        records.append(payload)
    return records


def build_record(workspace: Path, metadata: dict[str, object]) -> dict[str, object]:
    asset_id = str(metadata.get("asset_id") or metadata.get("roi_id") or "").strip()
    image_path = resolve_local_image_path(workspace, metadata)
    return {
        "asset_id": asset_id,
        "roi_id": str(metadata.get("roi_id") or "").strip(),
        "qa_state": str(metadata.get("qa_state") or "").strip(),
        "target_class": str(metadata.get("target_class") or "").strip(),
        "image_name": str(metadata.get("image_name") or image_path.name).strip(),
        "source_image": str(image_path),
        "source_image_remote": str(metadata.get("source_image") or "").strip(),
        "box_xyxy": metadata.get("box_xyxy"),
        "core_mask_path": resolve_local_asset_path(workspace, metadata.get("core_mask_path"), "sam2/core_masks", asset_id),
        "edit_mask_path": resolve_local_asset_path(workspace, metadata.get("edit_mask_path"), "sam2/edit_masks", asset_id),
        "overlay_path": resolve_local_asset_path(workspace, metadata.get("overlay_path"), "sam2/overlays", asset_id),
        "labelme_json_path": resolve_labelme_path(workspace, image_path),
    }


def build_review_manifest(workspace: Path, output_path: Path | None = None) -> dict[str, object]:
    workspace = validate_workspace(workspace)
    metadata_root = workspace / "sam2" / "metadata"
    records = [build_record(workspace, payload) for payload in load_metadata_records(metadata_root)]
    manifest_path = output_path or (workspace / "manifests" / "good_bolt_assets_manifest.jsonl")
    manifest_path = manifest_path.expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = {
        "mode": "good-bolt-review-manifest",
        "workspace": str(workspace),
        "manifest_path": str(manifest_path),
        "record_count": len(records),
    }
    return summary


def main() -> int:
    args = parse_args()
    try:
        summary = build_review_manifest(args.workspace, output_path=args.output)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return fail(str(exc))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
