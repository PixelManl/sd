from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bolt.generate.powerpaint_v2_annotations import rewrite_annotation
from bolt.generate.powerpaint_v2_backend import run_backend
from bolt.generate.powerpaint_v2_manifest import load_manifest_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local PowerPaint V2 batch skeleton for healthy-to-defect conversion."
    )
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend-mode", default="placeholder-copy")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _summary_path(output_dir: Path) -> Path:
    return output_dir / "manifest_results.json"


def _annotation_suffix(annotation_format: str) -> str:
    return ".xml" if annotation_format == "voc" else ".json"


def _resolve_image_output_path(images_dir: Path, output_stem: str, source_image: Path) -> Path:
    suffix = source_image.suffix if source_image.suffix else ".png"
    return images_dir / f"{output_stem}{suffix}"


def write_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _summary_path(output_dir).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_batch(
    *,
    manifest_path: Path,
    output_dir: Path,
    backend_mode: str = "placeholder-copy",
    dry_run: bool = False,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    output_dir = output_dir.resolve()
    metadata, records = load_manifest_records(manifest_path)
    images_dir = output_dir / "images"
    annotations_dir = output_dir / "annotations"

    state_by_source: dict[str, dict[str, str]] = {}
    results: list[dict[str, Any]] = []
    has_failure = False

    for record in records:
        source_key = str(Path(record["image_path"]).resolve())
        source_image = Path(record["image_path"])
        source_annotation = Path(record["annotation_path"])
        current_state = state_by_source.get(
            source_key,
            {
                "image_path": str(source_image.resolve()),
                "annotation_path": str(source_annotation.resolve()),
            },
        )
        current_image = Path(current_state["image_path"])
        current_annotation = Path(current_state["annotation_path"])
        output_image = _resolve_image_output_path(images_dir, record["output_stem"], current_image)
        output_annotation = annotations_dir / f"{record['output_stem']}{_annotation_suffix(record['annotation_format'])}"

        if dry_run:
            state_by_source[source_key] = {
                "image_path": str(output_image.resolve()),
                "annotation_path": str(output_annotation.resolve()),
            }
            results.append(
                {
                    "target_id": record["target_id"],
                    "backend_mode": backend_mode,
                    "status": "dry-run",
                    "source_image": str(current_image.resolve()),
                    "edited_image": str(output_image.resolve()),
                    "annotation_before": str(current_annotation.resolve()),
                    "annotation_after": str(output_annotation.resolve()),
                    "error_message": None,
                }
            )
            continue

        try:
            backend_summary = run_backend(
                source_image=current_image,
                output_image=output_image,
                record=record,
                backend_mode=backend_mode,
            )
            annotation_summary = rewrite_annotation(
                annotation_format=record["annotation_format"],
                source_path=current_annotation,
                output_path=output_annotation,
                target=record,
            )
            state_by_source[source_key] = {
                "image_path": str(output_image.resolve()),
                "annotation_path": str(output_annotation.resolve()),
            }
            results.append(
                {
                    "target_id": record["target_id"],
                    "backend_mode": backend_summary["backend_mode"],
                    "status": "completed",
                    "source_image": backend_summary["source_image"],
                    "edited_image": backend_summary["edited_image"],
                    "annotation_before": str(current_annotation.resolve()),
                    "annotation_after": annotation_summary["output_path"],
                    "error_message": None,
                }
            )
        except Exception as exc:
            has_failure = True
            results.append(
                {
                    "target_id": record["target_id"],
                    "backend_mode": backend_mode,
                    "status": "failed",
                    "source_image": str(current_image.resolve()),
                    "edited_image": str(output_image.resolve()),
                    "annotation_before": str(current_annotation.resolve()),
                    "annotation_after": None,
                    "error_message": str(exc),
                }
            )

    summary = {
        "task": "run_powerpaint_v2_batch",
        "status": "dry-run" if dry_run else ("completed_with_errors" if has_failure else "completed"),
        "manifest_path": str(manifest_path),
        "output_dir": str(output_dir),
        "backend_mode": backend_mode,
        "record_count": len(records),
        "metadata": metadata,
        "records": results,
    }
    write_summary(output_dir, summary)
    return summary


def main() -> int:
    args = parse_args()
    summary = run_batch(
        manifest_path=args.manifest_path,
        output_dir=args.output_dir,
        backend_mode=args.backend_mode,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] in {"dry-run", "completed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
