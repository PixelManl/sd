from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


LABEL_PATTERN = re.compile(r'"label"\s*:\s*"(?P<label>[A-Za-z0-9_]+)"')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a local-only initial-round submission scaffold and asset inventory."
    )
    parser.add_argument(
        "--build-root",
        type=Path,
        default=Path("bolt/package/build"),
        help="Ignored local build root.",
    )
    parser.add_argument(
        "--tag",
        required=True,
        help="Build tag, for example 2026-03-24-upload-round-01.",
    )
    parser.add_argument(
        "--team-name",
        default="team",
        help="Team name placeholder for outward artifact names.",
    )
    parser.add_argument(
        "--healthy-assets-root",
        type=Path,
        default=Path("data/bolt_parallel/good_bolt_assets"),
        help="Local healthy-bolt asset workspace root.",
    )
    return parser.parse_args()


def count_files(path: Path, patterns: tuple[str, ...] = ("*",)) -> int:
    if not path.exists() or not path.is_dir():
        return 0
    total = 0
    for pattern in patterns:
        total += sum(1 for candidate in path.glob(pattern) if candidate.is_file())
    return total


def count_labels(label_root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not label_root.exists() or not label_root.is_dir():
        return counts
    for json_path in sorted(label_root.glob("*.json")):
        text = json_path.read_text(encoding="utf-8")
        for match in LABEL_PATTERN.finditer(text):
            label = match.group("label")
            counts[label] = counts.get(label, 0) + 1
    return counts


def build_inventory(healthy_assets_root: Path) -> dict[str, Any]:
    incoming_images = healthy_assets_root / "incoming" / "images"
    manual_labels = healthy_assets_root / "manual_labels" / "healthy_labelme_json"
    sam2_root = healthy_assets_root / "sam2"
    exports_root = healthy_assets_root / "exports"

    label_counts = count_labels(manual_labels)
    sam2_counts = {
        "metadata": count_files(sam2_root / "metadata", ("*.json",)),
        "core_masks": count_files(sam2_root / "core_masks", ("*.png",)),
        "edit_masks": count_files(sam2_root / "edit_masks", ("*.png",)),
        "overlays": count_files(sam2_root / "overlays", ("*.png",)),
    }
    return {
        "healthy_assets_root": str(healthy_assets_root.resolve()),
        "incoming_image_count": count_files(incoming_images, ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")),
        "manual_label_file_count": count_files(manual_labels, ("*.json",)),
        "manual_label_box_counts": label_counts,
        "sam2_asset_counts": sam2_counts,
        "healthy_roi_bank_count": count_files(exports_root / "healthy_roi_bank", ("*",)),
        "qa_list_count": count_files(exports_root / "qa_lists", ("*",)),
    }


def materialize_scaffold(build_dir: Path) -> dict[str, str]:
    layout = {
        "model_pkg": build_dir / "model_pkg",
        "dataset_pkg": build_dir / "dataset_pkg",
        "docs": build_dir / "docs",
        "checks": build_dir / "checks",
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)

    for relative in (
        Path("model_pkg/model/core"),
        Path("model_pkg/code"),
        Path("model_pkg/env"),
        Path("model_pkg/ascend_image"),
        Path("model_pkg/docs"),
        Path("dataset_pkg/DataFiles"),
        Path("dataset_pkg/Annotations"),
    ):
        (build_dir / relative).mkdir(parents=True, exist_ok=True)

    return {key: str(value.resolve()) for key, value in layout.items()}


def write_markdown(path: Path, inventory: dict[str, Any], team_name: str, tag: str) -> None:
    label_lines = []
    for label, count in sorted(inventory["manual_label_box_counts"].items()):
        label_lines.append(f"- `{label}`: {count}")
    if not label_lines:
        label_lines.append("- `none`: 0")

    sam2_lines = []
    for key, count in inventory["sam2_asset_counts"].items():
        sam2_lines.append(f"- `{key}`: {count}")

    content = "\n".join(
        [
            f"# 本地资产上传清单",
            "",
            f"- 生成时间：`{datetime.now().isoformat(timespec='seconds')}`",
            f"- build tag：`{tag}`",
            f"- 团队名占位：`{team_name}`",
            "",
            "## 当前资产盘点",
            "",
            f"- 健康资产根目录：`{inventory['healthy_assets_root']}`",
            f"- 候选健康图数量：`{inventory['incoming_image_count']}`",
            f"- 人工 LabelMe 标注文件数：`{inventory['manual_label_file_count']}`",
            f"- 已导出健康 ROI 数量：`{inventory['healthy_roi_bank_count']}`",
            f"- QA 列表文件数：`{inventory['qa_list_count']}`",
            "",
            "### 标签框统计",
            "",
            *label_lines,
            "",
            "### SAM2 资产统计",
            "",
            *sam2_lines,
            "",
            "## 上传优先级",
            "",
            "- 第一批：`incoming/images/` + `manual_labels/healthy_labelme_json/`",
            "- 第二批：`sam2/core_masks/` + `sam2/edit_masks/` + `sam2/metadata/` + `sam2/overlays/`",
            "- 第三批：`manifests/`、`exports/healthy_roi_bank/`、`exports/qa_lists/`",
            "",
            "## 备注",
            "",
            "- 这份清单只落在本地 `bolt/package/build/`，不会进入 Git。",
            "- 如果云盘只够传一轮，优先保证原图、人工框标注和 SAM2 metadata 同步。",
            "",
        ]
    )
    path.write_text(content + "\n", encoding="utf-8")


def write_upload_manifest(path: Path, inventory: dict[str, Any]) -> None:
    rows = [
        {
            "asset_group": "healthy_candidates",
            "local_path": str(Path(inventory["healthy_assets_root"]) / "incoming" / "images"),
            "file_count": inventory["incoming_image_count"],
            "annotation_scope": "candidate image",
            "qa_state": "candidate",
            "upload_action": "upload-first",
        },
        {
            "asset_group": "healthy_labelme_json",
            "local_path": str(Path(inventory["healthy_assets_root"]) / "manual_labels" / "healthy_labelme_json"),
            "file_count": inventory["manual_label_file_count"],
            "annotation_scope": "rectangle label",
            "qa_state": "reviewed-manual",
            "upload_action": "upload-first",
        },
        {
            "asset_group": "sam2_assets",
            "local_path": str(Path(inventory["healthy_assets_root"]) / "sam2"),
            "file_count": sum(inventory["sam2_asset_counts"].values()),
            "annotation_scope": "mask+metadata",
            "qa_state": "draft-to-review",
            "upload_action": "upload-second",
        },
    ]

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "asset_group",
                "local_path",
                "file_count",
                "annotation_scope",
                "qa_state",
                "upload_action",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_submission_checklist(path: Path, team_name: str) -> None:
    content = "\n".join(
        [
            "# 初赛打包检查清单",
            "",
            f"- 对外目标文件：`{team_name}_生图模型成果物.zip`",
            f"- 对外目标文件：`{team_name}_高质量数据集成果物.zip`",
            f"- 对外目标文件：`{team_name}_高质量数据集说明文档.docx`",
            "",
            "## 打包前检查",
            "",
            "- 检测数据集只保留原图与框标注，不混入 mask、overlay、日志。",
            "- 生图模型包至少包含 `infer.py`、依赖说明、运行说明、README。",
            "- 所有 build 文件只留在本地 `bolt/package/build/`。",
            "- 上传前再做一次目录与命名检查。",
            "",
        ]
    )
    path.write_text(content + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    build_dir = (args.build_root / args.tag).resolve()
    layout = materialize_scaffold(build_dir)
    inventory = build_inventory(args.healthy_assets_root.resolve())

    checks_dir = build_dir / "checks"
    inventory_json = checks_dir / "asset_inventory.json"
    inventory_md = checks_dir / "asset_inventory.md"
    upload_manifest = checks_dir / "upload_manifest.csv"
    checklist_md = checks_dir / "submission_checklist.md"

    inventory_json.write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(inventory_md, inventory, args.team_name, args.tag)
    write_upload_manifest(upload_manifest, inventory)
    write_submission_checklist(checklist_md, args.team_name)

    payload = {
        "task": "materialize_submission_scaffold",
        "status": "created",
        "build_dir": str(build_dir),
        "layout": layout,
        "inventory_json": str(inventory_json),
        "inventory_md": str(inventory_md),
        "upload_manifest": str(upload_manifest),
        "submission_checklist": str(checklist_md),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
