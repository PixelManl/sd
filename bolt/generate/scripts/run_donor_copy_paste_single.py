from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bolt.generate.donor_paste import Box, alpha_bbox, composite_rgba_at, crop_box_from_mask_array, fit_content_box


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single donor copy-paste bolt experiment.")
    parser.add_argument("--roi-input", required=True)
    parser.add_argument("--roi-mask", required=True)
    parser.add_argument("--full-image", required=True)
    parser.add_argument("--crop-box", required=True, help="x1,y1,x2,y2 in full image coordinates")
    parser.add_argument("--donor-rgb", required=True)
    parser.add_argument("--donor-alpha", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--width-ratio", type=float, default=0.72)
    parser.add_argument("--top-offset", type=int, default=0)
    parser.add_argument("--feather-radius", type=float, default=2.0)
    return parser.parse_args()


def parse_box(text: str) -> Box:
    values = [int(part.strip()) for part in text.split(",")]
    if len(values) != 4:
        raise ValueError("--crop-box must contain exactly four integers")
    return Box(*values)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    roi_input = Image.open(args.roi_input).convert("RGB")
    roi_mask = Image.open(args.roi_mask).convert("L")
    full_image = Image.open(args.full_image).convert("RGB")
    donor_rgb = Image.open(args.donor_rgb).convert("RGB")
    donor_alpha = Image.open(args.donor_alpha).convert("L")
    crop_box = parse_box(args.crop_box)

    target_box = crop_box_from_mask_array(np.array(roi_mask))
    donor_box = alpha_bbox(donor_alpha)

    donor_rgba = donor_rgb.copy().convert("RGBA")
    donor_rgba.putalpha(donor_alpha)
    placement = fit_content_box(
        donor_box,
        target_box,
        width_ratio=args.width_ratio,
        top_offset=args.top_offset,
    )

    roi_output = composite_rgba_at(
        roi_input,
        donor_rgba,
        placement,
        feather_radius=args.feather_radius,
    )

    full_output = full_image.copy()
    full_output_np = np.array(full_output)
    roi_output_np = np.array(roi_output)
    full_output_np[crop_box.y1:crop_box.y2, crop_box.x1:crop_box.x2] = roi_output_np
    full_output = Image.fromarray(full_output_np)

    roi_output_path = output_dir / "roi_output_donor_copy_paste.png"
    full_output_path = output_dir / "full_output_donor_copy_paste.png"
    roi_output.save(roi_output_path)
    full_output.save(full_output_path)
    (output_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "roi_input": str(Path(args.roi_input).resolve()),
                "roi_mask": str(Path(args.roi_mask).resolve()),
                "full_image": str(Path(args.full_image).resolve()),
                "crop_box": [crop_box.x1, crop_box.y1, crop_box.x2, crop_box.y2],
                "target_box": [target_box.x1, target_box.y1, target_box.x2, target_box.y2],
                "donor_rgb": str(Path(args.donor_rgb).resolve()),
                "donor_alpha": str(Path(args.donor_alpha).resolve()),
                "donor_content_box": [donor_box.x1, donor_box.y1, donor_box.x2, donor_box.y2],
                "placement_box": [placement.x1, placement.y1, placement.x2, placement.y2],
                "width_ratio": args.width_ratio,
                "top_offset": args.top_offset,
                "feather_radius": args.feather_radius,
                "roi_output": str(roi_output_path),
                "full_output": str(full_output_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(roi_output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
