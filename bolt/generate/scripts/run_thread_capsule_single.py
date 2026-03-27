from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bolt.generate.thread_capsule import repair_mask_with_thread_capsule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair a nut mask with a self-thread capsule heuristic.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--target-mask", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--inpaint-radius", type=int, default=5)
    parser.add_argument("--probe-height-ratio", type=float, default=0.20)
    parser.add_argument("--stud-width-ratio", type=float, default=0.22)
    parser.add_argument("--capsule-blur-radius", type=float, default=2.0)
    parser.add_argument("--tall-mask-min-height", type=int, default=220)
    parser.add_argument("--lower-anchor-extend-ratio", type=float, default=0.45)
    parser.add_argument("--lower-anchor-min-pixels", type=int, default=18)
    parser.add_argument("--vertical-fade-ratio", type=float, default=0.35)
    parser.add_argument("--preview-pad", type=int, default=140)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = Path(args.image).expanduser().resolve()
    mask_path = Path(args.target_mask).expanduser().resolve()

    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    output_np, debug = repair_mask_with_thread_capsule(
        np.asarray(image, dtype=np.uint8),
        np.asarray(mask, dtype=np.uint8),
        inpaint_radius=args.inpaint_radius,
        probe_height_ratio=args.probe_height_ratio,
        stud_width_ratio=args.stud_width_ratio,
        capsule_blur_radius=args.capsule_blur_radius,
        tall_mask_min_height=args.tall_mask_min_height,
        lower_anchor_extend_ratio=args.lower_anchor_extend_ratio,
        lower_anchor_min_pixels=args.lower_anchor_min_pixels,
        vertical_fade_ratio=args.vertical_fade_ratio,
    )
    output = Image.fromarray(output_np, mode="RGB")

    full_output_path = output_dir / "full_output_thread_capsule.png"
    output.save(full_output_path)

    left = max(0, debug.target_box.x1 - args.preview_pad)
    top = max(0, debug.target_box.y1 - args.preview_pad)
    right = min(output.width, debug.target_box.x2 + args.preview_pad)
    bottom = min(output.height, debug.target_box.y2 + args.preview_pad)
    preview = output.crop((left, top, right, bottom))
    draw = ImageDraw.Draw(preview)
    draw.rectangle(
        (
            debug.target_box.x1 - left,
            debug.target_box.y1 - top,
            debug.target_box.x2 - left,
            debug.target_box.y2 - top,
        ),
        outline="lime",
        width=4,
    )
    draw.text((8, 8), output_dir.name, fill="yellow", font=ImageFont.load_default())
    preview_path = output_dir / "preview_thread_capsule.png"
    preview.save(preview_path)

    (output_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "image": str(image_path),
                "target_mask": str(mask_path),
                "target_box": [
                    debug.target_box.x1,
                    debug.target_box.y1,
                    debug.target_box.x2,
                    debug.target_box.y2,
                ],
                "probe_box": [
                    debug.probe_box.x1,
                    debug.probe_box.y1,
                    debug.probe_box.x2,
                    debug.probe_box.y2,
                ],
                "source_box": [
                    debug.source_box.x1,
                    debug.source_box.y1,
                    debug.source_box.x2,
                    debug.source_box.y2,
                ],
                "visible_box": [
                    debug.visible_box.x1,
                    debug.visible_box.y1,
                    debug.visible_box.x2,
                    debug.visible_box.y2,
                ],
                "placement_box": [
                    debug.placement_box.x1,
                    debug.placement_box.y1,
                    debug.placement_box.x2,
                    debug.placement_box.y2,
                ],
                "stud_width": debug.stud_width,
                "center_x": debug.center_x,
                "source_mode": debug.source_mode,
                "visible_top": debug.visible_top,
                "visible_fade_rows": debug.visible_fade_rows,
                "texture_gray_mean": debug.texture_gray_mean,
                "texture_gray_std": debug.texture_gray_std,
                "texture_gate_triggered": debug.texture_gate_triggered,
                "inpaint_radius": args.inpaint_radius,
                "probe_height_ratio": args.probe_height_ratio,
                "stud_width_ratio": args.stud_width_ratio,
                "capsule_blur_radius": args.capsule_blur_radius,
                "tall_mask_min_height": args.tall_mask_min_height,
                "lower_anchor_extend_ratio": args.lower_anchor_extend_ratio,
                "lower_anchor_min_pixels": args.lower_anchor_min_pixels,
                "vertical_fade_ratio": args.vertical_fade_ratio,
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
