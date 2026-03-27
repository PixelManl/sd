from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_RECORD_KEYS = (
    "image_path",
    "annotation_format",
    "annotation_path",
    "target_id",
    "output_stem",
)


def load_manifest_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {"records": payload}
    if isinstance(payload, dict):
        return payload
    raise TypeError(f"Unsupported manifest payload type: {type(payload).__name__}")


def normalize_record(record: dict[str, Any], *, index: int) -> dict[str, Any]:
    missing = [key for key in REQUIRED_RECORD_KEYS if not record.get(key)]
    if missing:
        raise ValueError(f"Manifest record {index} missing required fields: {', '.join(missing)}")
    annotation_format = str(record["annotation_format"]).strip().lower()
    if annotation_format not in {"voc", "coco"}:
        raise ValueError(f"Manifest record {index} has unsupported annotation_format: {annotation_format}")
    normalized = dict(record)
    normalized["annotation_format"] = annotation_format
    normalized["image_path"] = str(Path(str(record["image_path"])).resolve())
    normalized["annotation_path"] = str(Path(str(record["annotation_path"])).resolve())
    normalized["output_stem"] = str(record["output_stem"]).strip()
    normalized["target_id"] = str(record["target_id"]).strip()
    if not normalized["output_stem"]:
        raise ValueError(f"Manifest record {index} has empty output_stem")
    if not normalized["target_id"]:
        raise ValueError(f"Manifest record {index} has empty target_id")
    return normalized


def load_manifest_records(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = load_manifest_payload(path)
    raw_records = payload.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError(f"Manifest does not contain any records: {path}")
    records = []
    for index, record in enumerate(raw_records, start=1):
        if not isinstance(record, dict):
            raise TypeError(f"Manifest record {index} is not an object")
        records.append(normalize_record(record, index=index))
    metadata = {key: value for key, value in payload.items() if key != "records"}
    return metadata, records
