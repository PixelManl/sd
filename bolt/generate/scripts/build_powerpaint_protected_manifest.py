from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bolt.generate.powerpaint_v2_manifest import load_manifest_records
from bolt.generate.protected_edit import build_stud_keep_hard_mask, build_three_zone_masks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a protected PowerPaint manifest from an existing crop+mask manifest.")
    parser.add_argument("--source-manifest-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant-name", default="protect_v1")
    parser.add_argument("--seam-px", type=int, default=2)
    parser.add_argument("--context-ring-px", type=int, default=12)
    parser.add_argument("--strict-paste-blur-px", type=int, default=0)
    parser.add_argument("--keep-hard-length-scale", type=float, default=1.45)
    parser.add_argument("--keep-hard-width-scale", type=float, default=0.32)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def _save_mask(path: Path, array: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="L").save(path)
    return str(path.resolve())


def build_protected_manifest(
    *,
    source_manifest_path: Path,
    output_dir: Path,
    variant_name: str,
    seam_px: int,
    context_ring_px: int,
    strict_paste_blur_px: int = 0,
    keep_hard_length_scale: float = 1.45,
    keep_hard_width_scale: float = 0.32,
    limit: int = 0,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    metadata, records = load_manifest_records(source_manifest_path.resolve())
    if limit > 0:
        records = records[:limit]

    protect_dir = output_dir / "protect_masks"
    paste_dir = output_dir / "paste_masks"
    out_records: list[dict[str, Any]] = []

    for record in records:
        mask_path = Path(str(record["mask_path"])).resolve()
        with Image.open(mask_path) as mask_image:
            remove_mask = mask_image.convert("L")
            protect_mask = build_stud_keep_hard_mask(
                remove_mask,
                length_scale=keep_hard_length_scale,
                width_scale=keep_hard_width_scale,
            )
            zones = build_three_zone_masks(
                remove_mask,
                protect_mask=protect_mask,
                seam_px=seam_px,
                context_px=context_ring_px,
                blur_px=strict_paste_blur_px,
            )

        stem = Path(mask_path).stem
        protect_path = protect_dir / f"{stem}-{variant_name}-protect.png"
        paste_path = paste_dir / f"{stem}-{variant_name}-paste.png"
        new_record = dict(record)
        new_record["target_id"] = f'{record["target_id"]}-{variant_name}'
        new_record["output_stem"] = f'{record["output_stem"]}-{variant_name}'
        new_record["protect_mask_path"] = _save_mask(protect_path, zones["keep_hard"])
        new_record["paste_mask_path"] = _save_mask(paste_path, zones["paste"])
        new_record["strict_paste_seam_px"] = int(seam_px)
        new_record["strict_paste_blur_px"] = int(strict_paste_blur_px)
        new_record["context_ring_px"] = int(context_ring_px)
        new_record["keep_hard_length_scale"] = float(keep_hard_length_scale)
        new_record["keep_hard_width_scale"] = float(keep_hard_width_scale)
        out_records.append(new_record)

    summary = {
        "task": f"powerpaint_protected_manifest::{variant_name}",
        "variant_name": variant_name,
        "source_manifest_path": str(source_manifest_path.resolve()),
        "output_dir": str(output_dir),
        "record_count": len(out_records),
    }
    manifest_payload = {
        **metadata,
        **summary,
        "records": out_records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    args = parse_args()
    summary = build_protected_manifest(
        source_manifest_path=args.source_manifest_path,
        output_dir=args.output_dir,
        variant_name=str(args.variant_name).strip(),
        seam_px=args.seam_px,
        context_ring_px=args.context_ring_px,
        strict_paste_blur_px=args.strict_paste_blur_px,
        keep_hard_length_scale=args.keep_hard_length_scale,
        keep_hard_width_scale=args.keep_hard_width_scale,
        limit=args.limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
