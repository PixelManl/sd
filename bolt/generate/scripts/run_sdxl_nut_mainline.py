from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path


DEFAULT_PROMPT = (
    "minimal local repair in the masked region: add exactly one slightly worn gray steel hex nut "
    "onto the lower threaded bolt, aligned on-axis and seated tightly against the underside of the "
    "metal plate. normal hex-nut thickness, color matching adjacent hardware, not a tall spacer or "
    "sleeve. preserve the existing stud above the nut, the plate, lighting, perspective, and sky background. "
    "realistic utility hardware close-up"
)

DEFAULT_NEGATIVE_PROMPT = (
    "empty bolt, extra bolt, extra nut, extra washer, floating part, duplicate hardware, sleeve, socket, "
    "cap, hollow cylinder, wrong geometry, deformed plate, blurry, cartoon, painting, low quality, "
    "shiny new metal, fused metal, melted metal, oversized spacer"
)


def default_output_dir() -> Path:
    tag = date.today().strftime("%Y%m%d")
    return Path("data/bolt/generate/sdxl/repaired") / f"mainline_{tag}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the current SDXL nut-addition mainline with fixed defaults."
    )
    parser.add_argument(
        "--batch-manifest",
        type=Path,
        default=Path("data/sam2_box_prompt_tiny_full_20260322/manifest.json"),
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path(r"C:\Users\21941\Desktop\images"),
    )
    parser.add_argument(
        "--core-mask-dir",
        type=Path,
        default=Path("data/sam2_box_prompt_tiny_full_20260322"),
    )
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--image-name", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--strength", type=float, default=0.92)
    parser.add_argument("--target-size", type=int, default=1024)
    parser.add_argument("--seed-base", type=int, default=700)
    parser.add_argument("--adaptive-target-occupancy", type=float, default=0.20)
    parser.add_argument("--adaptive-min-side", type=int, default=320)
    parser.add_argument("--adaptive-root-bias", type=float, default=0.20)
    parser.add_argument("--mask-mode", choices=("oriented", "none_full"), default="none_full")
    parser.add_argument("--mask-dilate-ratio", type=float, default=0.34)
    parser.add_argument("--mask-min-pad", type=int, default=28)
    parser.add_argument("--blur-ksize", type=int, default=21)
    parser.add_argument("--blend-mask-source", choices=("inpaint", "core_dilate"), default="core_dilate")
    parser.add_argument("--blend-mask-dilate-ratio", type=float, default=0.10)
    parser.add_argument("--blend-mask-min-pad", type=int, default=8)
    parser.add_argument("--blend-mask-blur-ksize", type=int, default=0)
    parser.add_argument("--geometry-prior", choices=("none", "axis", "envelope"), default="none")
    parser.add_argument("--geometry-prior-strength", type=float, default=0.55)
    parser.add_argument("--controlnet-model", default="")
    parser.add_argument("--controlnet-conditioning-scale", type=float, default=0.8)
    parser.add_argument("--control-guidance-start", type=float, default=0.0)
    parser.add_argument("--control-guidance-end", type=float, default=1.0)
    parser.add_argument("--control-image-source", choices=("canny",), default="canny")
    parser.add_argument("--control-canny-low-threshold", type=int, default=100)
    parser.add_argument("--control-canny-high-threshold", type=int, default=200)
    parser.add_argument("--base-model", default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--lora-path", default="")
    parser.add_argument("--lora-scale", type=float, default=0.8)
    parser.add_argument("--lora-adapter-name", default="nut_semantic_lora")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.batch_manifest.exists():
        raise FileNotFoundError(f"batch manifest not found: {args.batch_manifest}")
    if not args.image_dir.exists():
        raise FileNotFoundError(f"image dir not found: {args.image_dir}")
    if not args.core_mask_dir.exists():
        raise FileNotFoundError(f"core mask dir not found: {args.core_mask_dir}")


def build_command(args: argparse.Namespace) -> list[str]:
    script = Path("bolt/generate/scripts/run_sdxl_oriented_batch.py")
    command = [
        sys.executable,
        str(script),
        "--batch-manifest",
        str(args.batch_manifest),
        "--image-dir",
        str(args.image_dir),
        "--core-mask-dir",
        str(args.core_mask_dir),
        "--output-dir",
        str(args.output_dir),
        "--base-model",
        str(args.base_model),
        "--prompt",
        str(args.prompt),
        "--negative-prompt",
        str(args.negative_prompt),
        "--steps",
        str(args.steps),
        "--guidance-scale",
        str(args.guidance_scale),
        "--strength",
        str(args.strength),
        "--target-size",
        str(args.target_size),
        "--seed-base",
        str(args.seed_base),
        "--adaptive-target-occupancy",
        str(args.adaptive_target_occupancy),
        "--adaptive-min-side",
        str(args.adaptive_min_side),
        "--adaptive-root-bias",
        str(args.adaptive_root_bias),
        "--mask-mode",
        str(args.mask_mode),
        "--mask-dilate-ratio",
        str(args.mask_dilate_ratio),
        "--mask-min-pad",
        str(args.mask_min_pad),
        "--blur-ksize",
        str(args.blur_ksize),
        "--blend-mask-source",
        str(args.blend_mask_source),
        "--blend-mask-dilate-ratio",
        str(args.blend_mask_dilate_ratio),
        "--blend-mask-min-pad",
        str(args.blend_mask_min_pad),
        "--blend-mask-blur-ksize",
        str(args.blend_mask_blur_ksize),
        "--geometry-prior",
        str(args.geometry_prior),
        "--geometry-prior-strength",
        str(args.geometry_prior_strength),
        "--controlnet-model",
        str(args.controlnet_model),
        "--controlnet-conditioning-scale",
        str(args.controlnet_conditioning_scale),
        "--control-guidance-start",
        str(args.control_guidance_start),
        "--control-guidance-end",
        str(args.control_guidance_end),
        "--control-image-source",
        str(args.control_image_source),
        "--control-canny-low-threshold",
        str(args.control_canny_low_threshold),
        "--control-canny-high-threshold",
        str(args.control_canny_high_threshold),
    ]
    if args.lora_path:
        command.extend([
            "--lora-path",
            str(args.lora_path),
            "--lora-scale",
            str(args.lora_scale),
            "--lora-adapter-name",
            str(args.lora_adapter_name),
        ])
    for image_name in args.image_name:
        command.extend(["--image-name", str(image_name)])
    if args.limit > 0:
        command.extend(["--limit", str(args.limit)])
    return command


def build_plan(args: argparse.Namespace) -> dict[str, object]:
    return {
        "task": "run_sdxl_nut_mainline",
        "status": "dry-run" if args.dry_run or not args.execute else "execute-requested",
        "batch_manifest": str(args.batch_manifest.resolve()),
        "image_dir": str(args.image_dir.resolve()),
        "core_mask_dir": str(args.core_mask_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "mask_mode": args.mask_mode,
        "geometry_prior": args.geometry_prior,
        "controlnet_model": args.controlnet_model,
        "controlnet_conditioning_scale": args.controlnet_conditioning_scale,
        "control_guidance_start": args.control_guidance_start,
        "control_guidance_end": args.control_guidance_end,
        "control_image_source": args.control_image_source,
        "adaptive_target_occupancy": args.adaptive_target_occupancy,
        "adaptive_root_bias": args.adaptive_root_bias,
        "mask_dilate_ratio": args.mask_dilate_ratio,
        "blend_mask_source": args.blend_mask_source,
        "blend_mask_dilate_ratio": args.blend_mask_dilate_ratio,
        "lora_path": args.lora_path,
        "lora_scale": args.lora_scale,
        "lora_adapter_name": args.lora_adapter_name,
        "image_name": args.image_name,
        "limit": args.limit,
        "command": build_command(args),
    }


def execute(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(build_command(args), check=False)
    return int(completed.returncode)


def main() -> int:
    args = parse_args()
    validate_args(args)
    plan = build_plan(args)
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    if args.execute and not args.dry_run:
        return execute(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
