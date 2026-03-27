from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bolt.generate.donor_paste import Box, crop_box_from_mask_array
from bolt.generate.missing_donor_patch import expand_box, transfer_mask_patch, transfer_rect_patch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transfer a same-image missing donor patch onto a healthy target.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--target-mask", required=True)
    parser.add_argument("--donor-box", required=True, help="x1,y1,x2,y2 in full-image coordinates")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--donor-pad", type=int, default=24)
    parser.add_argument("--target-pad", type=int, default=32)
    parser.add_argument("--feather-radius", type=float, default=12.0)
    parser.add_argument("--preview-pad", type=int, default=180)
    parser.add_argument("--blend-mode", choices=["rect", "mask", "top_band"], default="rect")
    parser.add_argument("--mask-dilate", type=int, default=0)
    parser.add_argument("--preserve-bottom-ratio", type=float, default=0.45)
    return parser.parse_args()


def parse_box(text: str) -> Box:
    values = [int(part.strip()) for part in text.split(",")]
    if len(values) != 4:
        raise ValueError("--donor-box must contain exactly four integers")
    return Box(*values)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = Path(args.image).expanduser().resolve()
    mask_path = Path(args.target_mask).expanduser().resolve()
    donor_box = parse_box(args.donor_box)

    image = Image.open(image_path).convert("RGB")
    target_mask = Image.open(mask_path).convert("L")
    target_box = crop_box_from_mask_array(np.array(target_mask))
    donor_rect = expand_box(donor_box, padding=max(0, args.donor_pad), image_size=image.size)
    target_rect = expand_box(target_box, padding=max(0, args.target_pad), image_size=image.size)

    if args.blend_mode == "mask":
        target_mask_binary = (np.array(target_mask) > 0).astype(np.uint8) * 255
        target_mask_image = Image.fromarray(target_mask_binary, mode="L")
        if args.mask_dilate > 0:
            kernel = max(3, args.mask_dilate * 2 + 1)
            target_mask_image = target_mask_image.filter(ImageFilter.MaxFilter(size=kernel))
        alpha_crop = target_mask_image.crop((target_rect.x1, target_rect.y1, target_rect.x2, target_rect.y2))
        output = transfer_mask_patch(
            image,
            donor_rect=donor_rect,
            target_rect=target_rect,
            target_alpha=alpha_crop,
            feather_radius=args.feather_radius,
        )
    elif args.blend_mode == "top_band":
        alpha_crop = Image.new("L", (target_rect.width, target_rect.height), 0)
        preserve_ratio = min(0.95, max(0.0, args.preserve_bottom_ratio))
        replace_height = int(round(target_rect.height * (1.0 - preserve_ratio)))
        replace_height = max(1, min(target_rect.height, replace_height))
        draw_alpha = ImageDraw.Draw(alpha_crop)
        draw_alpha.rectangle((0, 0, target_rect.width, replace_height), fill=255)
        output = transfer_mask_patch(
            image,
            donor_rect=donor_rect,
            target_rect=target_rect,
            target_alpha=alpha_crop,
            feather_radius=args.feather_radius,
        )
    else:
        output = transfer_rect_patch(
            image,
            donor_rect=donor_rect,
            target_rect=target_rect,
            feather_radius=args.feather_radius,
        )

    full_output_path = output_dir / "full_output_same_image_missing_patch.png"
    output.save(full_output_path)

    left = max(0, target_rect.x1 - args.preview_pad)
    top = max(0, target_rect.y1 - args.preview_pad)
    right = min(output.width, target_rect.x2 + args.preview_pad)
    bottom = min(output.height, target_rect.y2 + args.preview_pad)
    preview = output.crop((left, top, right, bottom))
    draw = ImageDraw.Draw(preview)
    draw.rectangle(
        (
            target_rect.x1 - left,
            target_rect.y1 - top,
            target_rect.x2 - left,
            target_rect.y2 - top,
        ),
        outline="lime",
        width=4,
    )
    draw.text((8, 8), output_dir.name, fill="yellow", font=ImageFont.load_default())
    preview_path = output_dir / "preview_same_image_missing_patch.png"
    preview.save(preview_path)

    (output_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "image": str(image_path),
                "target_mask": str(mask_path),
                "donor_box": [donor_box.x1, donor_box.y1, donor_box.x2, donor_box.y2],
                "target_box": [target_box.x1, target_box.y1, target_box.x2, target_box.y2],
                "donor_rect": [donor_rect.x1, donor_rect.y1, donor_rect.x2, donor_rect.y2],
                "target_rect": [target_rect.x1, target_rect.y1, target_rect.x2, target_rect.y2],
                "donor_pad": args.donor_pad,
                "target_pad": args.target_pad,
                "feather_radius": args.feather_radius,
                "blend_mode": args.blend_mode,
                "mask_dilate": args.mask_dilate,
                "preserve_bottom_ratio": args.preserve_bottom_ratio,
                "full_output": str(full_output_path),
                "preview_output": str(preview_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(preview_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
