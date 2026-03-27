from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a strict manifest for the first-10 same-image missing patch experiment.")
    parser.add_argument("--allowlist-path", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--xml-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json object expected: {path}")
    return payload


def resolve_image_name(*, asset_id: str, metadata_image_name: str, image_dir: Path) -> str:
    metadata_candidate = (image_dir / metadata_image_name)
    if metadata_image_name and metadata_candidate.exists():
        return metadata_image_name

    asset_candidate_name = f"{asset_id}.jpg"
    asset_candidate = image_dir / asset_candidate_name
    if asset_candidate.exists():
        return asset_candidate_name

    return metadata_image_name


def read_first_object_box(xml_path: Path) -> list[int]:
    root = ET.parse(xml_path).getroot()
    obj = root.find("object")
    if obj is None:
        raise ValueError(f"no object found: {xml_path}")
    bnd = obj.find("bndbox")
    if bnd is None:
        raise ValueError(f"no bndbox found: {xml_path}")
    return [
        int(bnd.findtext("xmin", "0")),
        int(bnd.findtext("ymin", "0")),
        int(bnd.findtext("xmax", "0")),
        int(bnd.findtext("ymax", "0")),
    ]


def build_manifest(
    *,
    allowlist_path: Path,
    metadata_dir: Path,
    mask_dir: Path,
    image_dir: Path,
    xml_dir: Path,
) -> dict[str, object]:
    allowlist = load_json(allowlist_path)
    records_payload = allowlist.get("records")
    if not isinstance(records_payload, list):
        raise ValueError("allowlist.records must be a list")

    records: list[dict[str, object]] = []
    for item in records_payload:
        if not isinstance(item, dict):
            raise ValueError("allowlist record must be an object")
        asset_id = str(item.get("asset_id") or "").strip()
        if not asset_id:
            raise ValueError("allowlist asset_id is required")

        metadata = load_json(metadata_dir / f"{asset_id}.json")
        metadata_image_name = str(metadata.get("image_name") or "").strip()
        if not metadata_image_name:
            raise ValueError(f"image_name missing in metadata: {asset_id}")
        image_name = resolve_image_name(asset_id=asset_id, metadata_image_name=metadata_image_name, image_dir=image_dir)

        image_path = (image_dir / image_name).resolve()
        mask_path = (mask_dir / f"{asset_id}_mask.png").resolve()
        xml_path = (xml_dir / metadata_image_name.replace(".jpg", ".xml")).resolve()
        donor_box = read_first_object_box(xml_path)

        records.append(
            {
                "asset_id": asset_id,
                "image_name": image_name,
                "image_path": str(image_path),
                "target_mask_path": str(mask_path),
                "donor_box": donor_box,
            }
        )

    return {
        "policy": "strict-first10-xcf-only",
        "allowlist_path": str(allowlist_path.resolve()),
        "record_count": len(records),
        "records": records,
    }


def main() -> int:
    args = parse_args()
    manifest = build_manifest(
        allowlist_path=args.allowlist_path.resolve(),
        metadata_dir=args.metadata_dir.resolve(),
        mask_dir=args.mask_dir.resolve(),
        image_dir=args.image_dir.resolve(),
        xml_dir=args.xml_dir.resolve(),
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
