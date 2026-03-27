from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from diffusers import StableDiffusionXLInpaintPipeline
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bolt.generate.distance_ladder import (
    build_edit_mask,
    expand_crop_box,
    parse_distance_variants,
    resolve_base_crop_box,
)


DEFAULT_BASE_MODEL = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an SDXL inpainting distance ladder with fixed crop scale and multiple edit-mask distances."
    )
    parser.add_argument("--batch-manifest", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--core-mask-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--strength", type=float, default=0.92)
    parser.add_argument("--target-size", type=int, default=1024)
    parser.add_argument("--seed-base", type=int, default=2600)
    parser.add_argument("--crop-scale", type=float, default=1.75)
    parser.add_argument("--min-pad", type=int, default=24)
    parser.add_argument("--blur-ksize", type=int, default=25)
    parser.add_argument(
        "--variant",
        action="append",
        default=None,
        help="Distance ladder entry in the form <name>:<dilate_ratio>. Repeat for multiple variants.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def ensure_odd(value: int) -> int:
    if value <= 1:
        return 0
    return value if value % 2 == 1 else value + 1


def resize_pair_for_sdxl(
    roi_rgb: np.ndarray,
    mask_gray: np.ndarray,
    target_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = roi_rgb.shape[:2]
    side = max(height, width)
    scale = target_size / float(side)
    new_height = max(64, int(round(height * scale / 8) * 8))
    new_width = max(64, int(round(width * scale / 8) * 8))
    roi_resized = cv2.resize(roi_rgb, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    mask_resized = cv2.resize(mask_gray, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return roi_resized, mask_resized


def paste_roi_back(
    full_rgb: np.ndarray,
    crop_box: list[int],
    roi_rgb: np.ndarray,
    mask_gray: np.ndarray,
) -> np.ndarray:
    x1, y1, x2, y2 = crop_box
    patch_height = y2 - y1
    patch_width = x2 - x1
    roi_resized = cv2.resize(roi_rgb, (patch_width, patch_height), interpolation=cv2.INTER_CUBIC)
    mask_resized = cv2.resize(mask_gray, (patch_width, patch_height), interpolation=cv2.INTER_LINEAR)

    alpha = mask_resized.astype(np.float32) / 255.0
    alpha = alpha[:, :, None]

    output = full_rgb.copy().astype(np.float32)
    base_patch = output[y1:y2, x1:x2]
    blended = roi_resized.astype(np.float32) * alpha + base_patch * (1.0 - alpha)
    output[y1:y2, x1:x2] = blended
    return np.clip(output, 0, 255).astype(np.uint8)


def load_manifest(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_pipeline(base_model: str) -> StableDiffusionXLInpaintPipeline:
    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    load_kwargs = {"torch_dtype": dtype}
    if use_cuda:
        load_kwargs["variant"] = "fp16"
        load_kwargs["use_safetensors"] = True

    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        base_model,
        **load_kwargs,
    )
    return pipe.to("cuda" if use_cuda else "cpu")


def main() -> int:
    args = parse_args()

    manifest = load_manifest(Path(args.batch_manifest))
    records: list[dict[str, object]] = list(manifest["records"])
    if args.limit > 0:
        records = records[: args.limit]

    image_dir = Path(args.image_dir)
    core_mask_dir = Path(args.core_mask_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = parse_distance_variants(args.variant)
    blur_ksize = ensure_odd(args.blur_ksize)

    pipe = None if args.dry_run else build_pipeline(args.base_model)
    prompt = str(manifest["prompt"])
    negative_prompt = str(manifest["negative_prompt"])

    summary_records: list[dict[str, object]] = []
    for record_index, record in enumerate(records):
        image_name = str(record["image"])
        stem = Path(image_name).stem
        base_crop_box = resolve_base_crop_box(record)

        image_path = image_dir / image_name
        core_mask_path = core_mask_dir / f"{stem}_mask.png"

        full_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if full_bgr is None:
            raise FileNotFoundError(f"failed to read image: {image_path}")
        core_mask = cv2.imread(str(core_mask_path), cv2.IMREAD_GRAYSCALE)
        if core_mask is None:
            raise FileNotFoundError(f"failed to read core mask: {core_mask_path}")

        full_rgb = cv2.cvtColor(full_bgr, cv2.COLOR_BGR2RGB)
        image_height, image_width = full_rgb.shape[:2]
        variant_records: list[dict[str, object]] = []
        for variant_index, variant in enumerate(variants):
            crop_box = expand_crop_box(
                base_crop_box,
                image_width=image_width,
                image_height=image_height,
                crop_scale=args.crop_scale,
            )
            x1, y1, x2, y2 = crop_box
            roi_rgb = full_rgb[y1:y2, x1:x2].copy()
            roi_core_mask = core_mask[y1:y2, x1:x2].copy()
            edit_mask, expanded_mask_box = build_edit_mask(
                roi_core_mask,
                dilate_ratio=variant.dilate_ratio,
                min_pad=args.min_pad,
                blur_ksize=blur_ksize,
            )
            if expanded_mask_box is None:
                raise ValueError(f"empty core mask after crop for image {image_name}")

            roi_input_path = output_dir / f"{stem}_roi_input_{variant.name}.png"
            roi_mask_core_path = output_dir / f"{stem}_roi_mask_core_{variant.name}.png"
            roi_mask_edit_path = output_dir / f"{stem}_roi_mask_edit_{variant.name}.png"
            cv2.imwrite(str(roi_input_path), cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(roi_mask_core_path), roi_core_mask)
            cv2.imwrite(str(roi_mask_edit_path), edit_mask)

            roi_output_path = output_dir / f"{stem}_roi_output_{variant.name}.png"
            full_output_path = output_dir / f"{stem}_full_output_{variant.name}.png"
            seed = args.seed_base + record_index * 100 + variant_index

            if args.dry_run:
                roi_output_ref = ""
                full_output_ref = ""
            else:
                assert pipe is not None
                roi_resized, mask_resized = resize_pair_for_sdxl(roi_rgb, edit_mask, args.target_size)
                image_pil = Image.fromarray(roi_resized)
                mask_pil = Image.fromarray(mask_resized)
                generator = torch.Generator(device=pipe.device).manual_seed(seed)
                with torch.inference_mode():
                    result = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        image=image_pil,
                        mask_image=mask_pil,
                        num_inference_steps=args.steps,
                        guidance_scale=args.guidance_scale,
                        strength=args.strength,
                        generator=generator,
                    ).images[0]
                roi_output_rgb = np.asarray(result.convert("RGB"))
                full_output_rgb = paste_roi_back(full_rgb, crop_box, roi_output_rgb, edit_mask)
                cv2.imwrite(str(roi_output_path), cv2.cvtColor(roi_output_rgb, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(full_output_path), cv2.cvtColor(full_output_rgb, cv2.COLOR_RGB2BGR))
                roi_output_ref = str(roi_output_path)
                full_output_ref = str(full_output_path)

            variant_records.append(
                {
                    "name": variant.name,
                    "crop_box": crop_box,
                    "expanded_mask_box": expanded_mask_box,
                    "dilate_ratio": variant.dilate_ratio,
                    "roi_input": str(roi_input_path),
                    "roi_mask_core": str(roi_mask_core_path),
                    "roi_mask_edit": str(roi_mask_edit_path),
                    "roi_output": roi_output_ref,
                    "full_output": full_output_ref,
                    "seed": seed,
                }
            )

        summary_records.append(
            {
                "image": image_name,
                "base_crop_box": base_crop_box,
                "variants": variant_records,
            }
        )

    summary = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "source_manifest": str(Path(args.batch_manifest)),
        "crop_scale": args.crop_scale,
        "min_pad": args.min_pad,
        "blur_ksize": blur_ksize,
        "variants": [
            {
                "name": variant.name,
                "dilate_ratio": variant.dilate_ratio,
            }
            for variant in variants
        ],
        "records": summary_records,
    }
    (output_dir / "distance_ladder_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
