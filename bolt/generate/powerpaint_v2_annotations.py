from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _normalize_bbox_xyxy(raw_bbox: Any) -> list[int] | None:
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        return None
    return [int(round(float(value))) for value in raw_bbox]


def _normalize_coco_bbox_xyxy(raw_bbox: Any) -> list[int] | None:
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        return None
    x, y, width, height = [float(value) for value in raw_bbox]
    return [
        int(round(x)),
        int(round(y)),
        int(round(x + width)),
        int(round(y + height)),
    ]


def _voc_object_matches(obj: ET.Element, target: dict[str, Any]) -> bool:
    target_bbox = _normalize_bbox_xyxy(target.get("bbox"))
    target_class = target.get("class_name")
    if target_bbox is None:
        return False
    bbox = obj.find("bndbox")
    if bbox is None:
        return False
    current_bbox = [
        int(bbox.findtext("xmin", "0")),
        int(bbox.findtext("ymin", "0")),
        int(bbox.findtext("xmax", "0")),
        int(bbox.findtext("ymax", "0")),
    ]
    if current_bbox != target_bbox:
        return False
    if isinstance(target_class, str) and target_class.strip():
        return (obj.findtext("name") or "").strip() == target_class.strip()
    return True


def rewrite_voc_annotation(
    *,
    source_path: Path,
    output_path: Path,
    target: dict[str, Any],
) -> dict[str, Any]:
    root = ET.parse(source_path).getroot()
    objects = list(root.findall("object"))
    target_index = target.get("object_index")
    removed = 0

    if isinstance(target_index, int):
        if target_index < 0 or target_index >= len(objects):
            raise IndexError(f"VOC object_index out of range: {target_index}")
        root.remove(objects[target_index])
        removed = 1
    else:
        matched = [obj for obj in objects if _voc_object_matches(obj, target)]
        if not matched:
            raise ValueError(f"Could not resolve VOC target: {target.get('target_id')}")
        if len(matched) > 1:
            raise ValueError(f"VOC target is ambiguous: {target.get('target_id')}")
        root.remove(matched[0])
        removed = 1

    _ensure_parent(output_path)
    ET.ElementTree(root).write(output_path, encoding="utf-8", xml_declaration=False)
    return {
        "annotation_format": "voc",
        "target_id": str(target.get("target_id") or ""),
        "removed_count": removed,
        "source_path": str(source_path.resolve()),
        "output_path": str(output_path.resolve()),
    }


def _coco_annotation_matches(
    annotation: dict[str, Any],
    *,
    target: dict[str, Any],
    category_name_by_id: dict[int, str],
) -> bool:
    annotation_id = target.get("annotation_id")
    if annotation_id is not None:
        return int(annotation.get("id", -1)) == int(annotation_id)

    target_image_id = target.get("image_id")
    if target_image_id is not None and int(annotation.get("image_id", -1)) != int(target_image_id):
        return False

    target_bbox = _normalize_bbox_xyxy(target.get("bbox"))
    if target_bbox is not None:
        current_bbox = _normalize_coco_bbox_xyxy(annotation.get("bbox"))
        if current_bbox != target_bbox:
            return False

    target_category_id = target.get("category_id")
    if target_category_id is not None and int(annotation.get("category_id", -1)) != int(target_category_id):
        return False

    target_class_name = target.get("class_name")
    if isinstance(target_class_name, str) and target_class_name.strip():
        category_name = category_name_by_id.get(int(annotation.get("category_id", -1)), "")
        if category_name != target_class_name.strip():
            return False

    return True


def rewrite_coco_annotation(
    *,
    source_path: Path,
    output_path: Path,
    target: dict[str, Any],
) -> dict[str, Any]:
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    annotations = payload.get("annotations")
    categories = payload.get("categories")
    if not isinstance(annotations, list) or not isinstance(categories, list):
        raise ValueError(f"Invalid COCO annotation payload: {source_path}")

    category_name_by_id = {
        int(category["id"]): str(category.get("name") or "")
        for category in categories
        if isinstance(category, dict) and "id" in category
    }
    kept: list[dict[str, Any]] = []
    removed = 0
    for annotation in annotations:
        if not isinstance(annotation, dict):
            kept.append(annotation)
            continue
        if _coco_annotation_matches(annotation, target=target, category_name_by_id=category_name_by_id):
            removed += 1
            continue
        kept.append(annotation)

    if removed == 0:
        raise ValueError(f"Could not resolve COCO target: {target.get('target_id')}")
    if target.get("annotation_id") is None and removed > 1:
        raise ValueError(f"COCO target is ambiguous: {target.get('target_id')}")

    payload["annotations"] = kept
    _ensure_parent(output_path)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "annotation_format": "coco",
        "target_id": str(target.get("target_id") or ""),
        "removed_count": removed,
        "source_path": str(source_path.resolve()),
        "output_path": str(output_path.resolve()),
    }


def rewrite_annotation(
    *,
    annotation_format: str,
    source_path: Path,
    output_path: Path,
    target: dict[str, Any],
) -> dict[str, Any]:
    normalized = annotation_format.strip().lower()
    if normalized == "voc":
        return rewrite_voc_annotation(source_path=source_path, output_path=output_path, target=target)
    if normalized == "coco":
        return rewrite_coco_annotation(source_path=source_path, output_path=output_path, target=target)
    raise ValueError(f"Unsupported annotation format: {annotation_format}")
