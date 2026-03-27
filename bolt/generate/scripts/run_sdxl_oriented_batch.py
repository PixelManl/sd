from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from collections.abc import Sequence

import cv2
import numpy as np
import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetInpaintPipeline,
    StableDiffusionXLInpaintPipeline,
)
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bolt.generate.adaptive_roi import compute_square_crop_box_from_bbox, compute_square_crop_box_from_mask
from bolt.generate.distance_ladder import build_edit_mask
from bolt.generate.geometry_prior import build_stud_geometry_prior, seed_roi_with_geometry_prior
from bolt.generate.mask_geometry import build_oriented_focus_mask, build_root_contact_focus_mask


DEFAULT_BASE_MODEL = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
DEFAULT_PROMPT = (
    "minimal local repair: add exactly one weathered gray steel hex nut on the lower threaded bolt, "
    "seated tightly and flush against the underside of the metal plate. normal hex-nut thickness, "
    "not a tall spacer or sleeve. preserve all existing hardware, geometry, lighting, perspective, "
    "and sky background. realistic utility hardware close-up, documentary telephoto"
)
DEFAULT_NEGATIVE_PROMPT = (
    "extra bolt, extra nut, extra washer, floating part, duplicate hardware, sleeve, socket, "
    "tall spacer, hollow cylinder, wrong geometry, deformed plate, fused metal, melted metal, "
    "blurry, cartoon, painting, low quality"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-run an SDXL ROI batch with an orientation-aware focused mask."
    )
    parser.add_argument("--batch-manifest", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--core-mask-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--image-name", action="append", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--strength", type=float, default=0.92)
    parser.add_argument("--target-size", type=int, default=1024)
    parser.add_argument("--seed-base", type=int, default=700)
    parser.add_argument("--adaptive-target-occupancy", type=float, default=0.0)
    parser.add_argument("--adaptive-min-side", type=int, default=0)
    parser.add_argument("--adaptive-root-bias", type=float, default=0.0)
    parser.add_argument("--mask-mode", default="oriented", choices=("oriented", "root_contact", "none_full"))
    parser.add_argument("--axial-scale", type=float, default=0.72)
    parser.add_argument("--transverse-scale", type=float, default=1.85)
    parser.add_argument("--min-axial-radius", type=float, default=10.0)
    parser.add_argument("--min-transverse-radius", type=float, default=12.0)
    parser.add_argument("--contact-bias", type=float, default=0.0)
    parser.add_argument("--mask-dilate-ratio", type=float, default=0.26)
    parser.add_argument("--mask-min-pad", type=int, default=24)
    parser.add_argument("--blur-ksize", type=int, default=21)
    parser.add_argument("--blend-mask-source", default="inpaint", choices=("inpaint", "core_dilate"))
    parser.add_argument("--blend-mask-dilate-ratio", type=float, default=0.12)
    parser.add_argument("--blend-mask-min-pad", type=int, default=8)
    parser.add_argument("--blend-mask-blur-ksize", type=int, default=0)
    parser.add_argument("--geometry-prior", default="none", choices=("none", "axis", "envelope"))
    parser.add_argument("--geometry-prior-strength", type=float, default=0.55)
    parser.add_argument("--controlnet-model", default="")
    parser.add_argument("--controlnet-conditioning-scale", type=float, default=0.8)
    parser.add_argument("--control-guidance-start", type=float, default=0.0)
    parser.add_argument("--control-guidance-end", type=float, default=1.0)
    parser.add_argument("--control-image-source", default="canny", choices=("canny",))
    parser.add_argument("--control-canny-low-threshold", type=int, default=100)
    parser.add_argument("--control-canny-high-threshold", type=int, default=200)
    parser.add_argument("--reference-image", default="")
    parser.add_argument("--ip-adapter-repo", default="")
    parser.add_argument("--ip-adapter-subfolder", default="sdxl_models")
    parser.add_argument("--ip-adapter-weight-name", default="")
    parser.add_argument("--ip-adapter-scale", type=float, default=0.6)
    parser.add_argument("--lora-path", default="")
    parser.add_argument("--lora-scale", type=float, default=0.8)
    parser.add_argument("--lora-adapter-name", default="nut_semantic_lora")
    return parser.parse_args()


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


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


def ensure_odd(value: int) -> int:
    if value <= 1:
        return 0
    return value if value % 2 == 1 else value + 1


def load_manifest(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_manifest_payload(payload: object) -> dict[str, object]:
    if isinstance(payload, dict):
        return {
            "prompt": str(payload.get("prompt") or DEFAULT_PROMPT),
            "negative_prompt": str(payload.get("negative_prompt") or DEFAULT_NEGATIVE_PROMPT),
            "records": list(payload.get("records") or []),
        }
    if isinstance(payload, list):
        return {
            "prompt": DEFAULT_PROMPT,
            "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
            "records": list(payload),
        }
    raise TypeError(f"unsupported manifest payload type: {type(payload).__name__}")


def resolve_prompt_config(
    manifest: dict[str, object],
    *,
    prompt_override: str = "",
    negative_prompt_override: str = "",
) -> tuple[str, str]:
    prompt = prompt_override.strip() if prompt_override.strip() else str(manifest["prompt"])
    negative_prompt = (
        negative_prompt_override.strip()
        if negative_prompt_override.strip()
        else str(manifest["negative_prompt"])
    )
    return prompt, negative_prompt


def filter_records(
    records: list[dict[str, object]],
    *,
    image_name: str | Sequence[str] | None,
    limit: int,
) -> list[dict[str, object]]:
    selected = list(records)
    if image_name:
        wanted_tokens = (
            [image_name.strip()]
            if isinstance(image_name, str)
            else [str(item).strip() for item in image_name if str(item).strip()]
        )
        selected = [
            record
            for record in selected
            if any(
                Path(str(record.get("image", ""))).name == wanted
                or Path(str(record.get("image", ""))).stem == wanted
                for wanted in wanted_tokens
            )
        ]
    if limit > 0:
        selected = selected[:limit]
    return selected


def resolve_effective_crop_box(
    record: dict[str, object],
    *,
    core_mask: np.ndarray | None,
    image_width: int,
    image_height: int,
    adaptive_target_occupancy: float,
    adaptive_min_side: int,
    adaptive_root_bias: float = 0.0,
) -> list[int]:
    crop_box_raw = record.get("crop_box")
    if crop_box_raw is None:
        xml_box = record.get("xml_box")
        if xml_box is None:
            raise KeyError("record must contain crop_box or xml_box")
        fallback_occupancy = adaptive_target_occupancy if adaptive_target_occupancy > 0.0 else 0.25
        fallback_min_side = adaptive_min_side if adaptive_min_side > 0 else 256
        crop_box = compute_square_crop_box_from_bbox(
            [int(v) for v in xml_box],
            image_width=image_width,
            image_height=image_height,
            target_occupancy=fallback_occupancy,
            min_side=fallback_min_side,
        )
    else:
        crop_box = [int(v) for v in crop_box_raw]
    if adaptive_target_occupancy <= 0.0 or core_mask is None:
        return crop_box

    adapted = compute_square_crop_box_from_mask(
        core_mask,
        image_width=image_width,
        image_height=image_height,
        target_occupancy=adaptive_target_occupancy,
        min_side=adaptive_min_side,
        root_bias=adaptive_root_bias,
    )
    return adapted or crop_box


def apply_geometry_prior_to_roi(
    roi_rgb: np.ndarray,
    roi_core_mask: np.ndarray,
    *,
    geometry_prior_mode: str,
    geometry_prior_strength: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if geometry_prior_mode == "none":
        return roi_rgb, {}

    prior = build_stud_geometry_prior(
        roi_core_mask,
        prior_mode=geometry_prior_mode,
        blur_ksize=0,
    )
    if prior is None:
        return roi_rgb, {}

    seeded = seed_roi_with_geometry_prior(
        roi_rgb,
        prior.envelope_mask,
        alpha=geometry_prior_strength,
    )
    return seeded, {
        "prior_axis": prior.axis_mask,
        "prior_envelope": prior.envelope_mask,
        "prior_preserve_tail": prior.preserve_tail_mask,
    }


def build_roi_inpaint_mask(
    roi_core_mask: np.ndarray,
    *,
    mask_mode: str,
    axial_scale: float,
    transverse_scale: float,
    min_axial_radius: float,
    min_transverse_radius: float,
    contact_bias: float,
    dilate_ratio: float,
    dilate_min_pad: int,
    blur_ksize: int,
) -> tuple[np.ndarray, dict[str, object]]:
    if mask_mode == "oriented":
        focus_mask = build_oriented_focus_mask(
            roi_core_mask,
            axial_scale=axial_scale,
            transverse_scale=transverse_scale,
            min_axial_radius=min_axial_radius,
            min_transverse_radius=min_transverse_radius,
            contact_bias=contact_bias,
            blur_ksize=blur_ksize,
        )
        return focus_mask, {}

    if mask_mode == "root_contact":
        focus_mask = build_root_contact_focus_mask(
            roi_core_mask,
            axial_scale=axial_scale,
            transverse_scale=transverse_scale,
            min_axial_radius=min_axial_radius,
            min_transverse_radius=min_transverse_radius,
            contact_bias=contact_bias,
            blur_ksize=blur_ksize,
        )
        return focus_mask, {}

    if mask_mode == "none_full":
        edit_mask, expanded_mask_box = build_edit_mask(
            roi_core_mask,
            dilate_ratio=dilate_ratio,
            min_pad=dilate_min_pad,
            blur_ksize=blur_ksize,
        )
        return edit_mask, {"expanded_mask_box": expanded_mask_box}

    raise ValueError(f"unsupported mask_mode: {mask_mode!r}")


def build_roi_blend_mask(
    roi_core_mask: np.ndarray,
    inpaint_mask: np.ndarray,
    *,
    source: str,
    dilate_ratio: float,
    min_pad: int,
    blur_ksize: int,
) -> np.ndarray:
    if source == "inpaint":
        return np.asarray(inpaint_mask).copy()
    if source == "core_dilate":
        blend_mask, _ = build_edit_mask(
            roi_core_mask,
            dilate_ratio=dilate_ratio,
            min_pad=min_pad,
            blur_ksize=blur_ksize,
        )
        return blend_mask
    raise ValueError(f"unsupported blend mask source: {source!r}")


def validate_ip_adapter_config(
    *,
    ip_adapter_repo: str,
    ip_adapter_weight_name: str,
    reference_image: str,
) -> bool:
    if not ip_adapter_repo:
        return False
    if not ip_adapter_weight_name:
        raise ValueError("--ip-adapter-weight-name is required when --ip-adapter-repo is set")
    if not reference_image:
        raise ValueError("--reference-image is required when --ip-adapter-repo is set")
    return True


def validate_controlnet_config(
    *,
    controlnet_model: str,
    controlnet_conditioning_scale: float,
    control_image_source: str,
) -> bool:
    if not controlnet_model:
        return False
    if controlnet_conditioning_scale <= 0.0:
        raise ValueError("controlnet conditioning scale must be > 0")
    if control_image_source != "canny":
        raise ValueError(f"unsupported control image source: {control_image_source}")
    return True


def build_control_image_from_roi(
    roi_rgb: np.ndarray,
    control_mask_gray: np.ndarray | None = None,
    *,
    source: str,
    target_size: int | tuple[int, int],
    canny_low_threshold: int,
    canny_high_threshold: int,
) -> Image.Image:
    if source != "canny":
        raise ValueError(f"unsupported control image source: {source}")

    if isinstance(target_size, int):
        target_width = target_size
        target_height = target_size
    else:
        target_width, target_height = target_size

    gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, threshold1=canny_low_threshold, threshold2=canny_high_threshold)
    if control_mask_gray is not None:
        edges[np.asarray(control_mask_gray) > 127] = 0
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    edges_resized = cv2.resize(
        edges_rgb,
        (target_width, target_height),
        interpolation=cv2.INTER_LINEAR,
    )
    return Image.fromarray(edges_resized)


def build_pipeline(
    base_model: str,
    *,
    controlnet_model: str = "",
    controlnet_conditioning_scale: float = 0.8,
    ip_adapter_repo: str = "",
    ip_adapter_subfolder: str = "sdxl_models",
    ip_adapter_weight_name: str = "",
    ip_adapter_scale: float = 0.6,
    lora_path: str = "",
    lora_scale: float = 0.8,
    lora_adapter_name: str = "nut_semantic_lora",
) -> StableDiffusionXLInpaintPipeline | StableDiffusionXLControlNetInpaintPipeline:
    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    load_kwargs = {"torch_dtype": dtype}
    if use_cuda:
        load_kwargs["variant"] = "fp16"
        load_kwargs["use_safetensors"] = True

    if controlnet_model:
        controlnet_kwargs = {"torch_dtype": dtype}
        if use_cuda:
            controlnet_kwargs["variant"] = "fp16"
            controlnet_kwargs["use_safetensors"] = True
        controlnet = ControlNetModel.from_pretrained(
            controlnet_model,
            **controlnet_kwargs,
        )
        pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            base_model,
            controlnet=controlnet,
            **load_kwargs,
        )
    else:
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            base_model,
            **load_kwargs,
        )
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    else:
        pipe = pipe.to("cpu")
    if lora_path:
        pipe.load_lora_weights(lora_path, adapter_name=lora_adapter_name)
        pipe.set_adapters([lora_adapter_name], adapter_weights=[lora_scale])
    if ip_adapter_repo:
        pipe.load_ip_adapter(
            ip_adapter_repo,
            subfolder=ip_adapter_subfolder,
            weight_name=ip_adapter_weight_name,
        )
        pipe.set_ip_adapter_scale(ip_adapter_scale)
    return pipe


def main() -> int:
    args = parse_args()

    manifest = resolve_manifest_payload(load_manifest(Path(args.batch_manifest)))
    records: list[dict[str, object]] = filter_records(
        list(manifest["records"]),
        image_name=args.image_name or None,
        limit=args.limit,
    )
    controlnet_enabled = validate_controlnet_config(
        controlnet_model=args.controlnet_model,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        control_image_source=args.control_image_source,
    )
    ip_adapter_enabled = validate_ip_adapter_config(
        ip_adapter_repo=args.ip_adapter_repo,
        ip_adapter_weight_name=args.ip_adapter_weight_name,
        reference_image=args.reference_image,
    )

    image_dir = Path(args.image_dir)
    core_mask_dir = Path(args.core_mask_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not records:
        raise ValueError("no matching records selected for this run")

    pipe = build_pipeline(
        args.base_model,
        controlnet_model=args.controlnet_model,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        ip_adapter_repo=args.ip_adapter_repo,
        ip_adapter_subfolder=args.ip_adapter_subfolder,
        ip_adapter_weight_name=args.ip_adapter_weight_name,
        ip_adapter_scale=args.ip_adapter_scale,
        lora_path=args.lora_path,
        lora_scale=args.lora_scale,
        lora_adapter_name=args.lora_adapter_name,
    )
    prompt, negative_prompt = resolve_prompt_config(
        manifest,
        prompt_override=args.prompt,
        negative_prompt_override=args.negative_prompt,
    )
    blur_ksize = ensure_odd(args.blur_ksize)
    reference_image_pil: Image.Image | None = None
    if ip_adapter_enabled:
        reference_path = Path(args.reference_image)
        if not reference_path.exists():
            raise FileNotFoundError(f"failed to read reference image: {reference_path}")
        reference_image_pil = Image.open(reference_path).convert("RGB").resize(
            (args.target_size, args.target_size)
        )

    summary_records: list[dict[str, object]] = []
    for index, record in enumerate(records, start=1):
        image_name = str(record["image"])
        stem = Path(image_name).stem

        image_path = image_dir / image_name
        core_mask_path = core_mask_dir / f"{stem}_mask.png"

        full_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if full_bgr is None:
            raise FileNotFoundError(f"failed to read image: {image_path}")
        core_mask = cv2.imread(str(core_mask_path), cv2.IMREAD_GRAYSCALE)
        if core_mask is None:
            raise FileNotFoundError(f"failed to read core mask: {core_mask_path}")

        full_rgb = cv2.cvtColor(full_bgr, cv2.COLOR_BGR2RGB)
        crop_box = resolve_effective_crop_box(
            record,
            core_mask=core_mask,
            image_width=full_rgb.shape[1],
            image_height=full_rgb.shape[0],
            adaptive_target_occupancy=args.adaptive_target_occupancy,
            adaptive_min_side=args.adaptive_min_side,
            adaptive_root_bias=args.adaptive_root_bias,
        )
        x1, y1, x2, y2 = crop_box
        roi_rgb = full_rgb[y1:y2, x1:x2].copy()
        roi_core_mask = core_mask[y1:y2, x1:x2].copy()
        focus_mask, mask_debug = build_roi_inpaint_mask(
            roi_core_mask,
            mask_mode=args.mask_mode,
            axial_scale=args.axial_scale,
            transverse_scale=args.transverse_scale,
            min_axial_radius=args.min_axial_radius,
            min_transverse_radius=args.min_transverse_radius,
            contact_bias=args.contact_bias,
            dilate_ratio=args.mask_dilate_ratio,
            dilate_min_pad=args.mask_min_pad,
            blur_ksize=blur_ksize,
        )
        blend_mask = build_roi_blend_mask(
            roi_core_mask,
            focus_mask,
            source=args.blend_mask_source,
            dilate_ratio=args.blend_mask_dilate_ratio,
            min_pad=args.blend_mask_min_pad,
            blur_ksize=ensure_odd(args.blend_mask_blur_ksize),
        )
        roi_seeded_rgb, prior_debug = apply_geometry_prior_to_roi(
            roi_rgb,
            roi_core_mask,
            geometry_prior_mode=args.geometry_prior,
            geometry_prior_strength=args.geometry_prior_strength,
        )

        roi_input_path = output_dir / f"{stem}_roi_input.png"
        roi_seeded_input_path = output_dir / f"{stem}_roi_input_seeded.png"
        roi_mask_path = output_dir / f"{stem}_roi_mask_focus_oriented.png"
        roi_blend_mask_path = output_dir / f"{stem}_roi_mask_blend.png"
        roi_control_image_path = output_dir / f"{stem}_roi_control_image.png"
        cv2.imwrite(str(roi_input_path), cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR))
        if prior_debug:
            cv2.imwrite(str(roi_seeded_input_path), cv2.cvtColor(roi_seeded_rgb, cv2.COLOR_RGB2BGR))
            for debug_name, debug_mask in prior_debug.items():
                cv2.imwrite(str(output_dir / f"{stem}_{debug_name}.png"), debug_mask)
        cv2.imwrite(str(roi_mask_path), focus_mask)
        cv2.imwrite(str(roi_blend_mask_path), blend_mask)

        roi_resized, mask_resized = resize_pair_for_sdxl(
            roi_seeded_rgb,
            focus_mask,
            args.target_size,
        )

        image_pil = Image.fromarray(roi_resized)
        mask_pil = Image.fromarray(mask_resized)
        control_image_pil: Image.Image | None = None
        if controlnet_enabled:
            control_image_pil = build_control_image_from_roi(
                roi_rgb,
                focus_mask,
                source=args.control_image_source,
                target_size=(roi_resized.shape[1], roi_resized.shape[0]),
                canny_low_threshold=args.control_canny_low_threshold,
                canny_high_threshold=args.control_canny_high_threshold,
            )
            control_image_pil.save(roi_control_image_path)
        generator = torch.Generator(device=pipe.device).manual_seed(args.seed_base + index)

        with torch.inference_mode():
            pipe_kwargs: dict[str, object] = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "image": image_pil,
                "mask_image": mask_pil,
                "num_inference_steps": args.steps,
                "guidance_scale": args.guidance_scale,
                "strength": args.strength,
                "generator": generator,
            }
            if reference_image_pil is not None:
                pipe_kwargs["ip_adapter_image"] = reference_image_pil
            if control_image_pil is not None:
                pipe_kwargs["control_image"] = control_image_pil
                pipe_kwargs["controlnet_conditioning_scale"] = args.controlnet_conditioning_scale
                pipe_kwargs["control_guidance_start"] = args.control_guidance_start
                pipe_kwargs["control_guidance_end"] = args.control_guidance_end
            result = pipe(
                **pipe_kwargs,
            ).images[0]

        roi_output_rgb = np.asarray(result.convert("RGB"))
        full_output_rgb = paste_roi_back(full_rgb, crop_box, roi_output_rgb, blend_mask)

        roi_output_path = output_dir / f"{stem}_roi_output_oriented.png"
        full_output_path = output_dir / f"{stem}_full_output_oriented.png"
        cv2.imwrite(str(roi_output_path), cv2.cvtColor(roi_output_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(full_output_path), cv2.cvtColor(full_output_rgb, cv2.COLOR_RGB2BGR))

        summary_records.append(
            {
                "image": image_name,
                "crop_box": crop_box,
                "roi_input": str(roi_input_path),
                "roi_input_seeded": str(roi_seeded_input_path) if prior_debug else "",
                "roi_mask_focus_oriented": str(roi_mask_path),
                "roi_mask_blend": str(roi_blend_mask_path),
                "roi_control_image": str(roi_control_image_path) if control_image_pil is not None else "",
                "roi_output_oriented": str(roi_output_path),
                "full_output_oriented": str(full_output_path),
                "seed": args.seed_base + index,
                "geometry_prior": args.geometry_prior,
                "mask_mode": args.mask_mode,
                "blend_mask_source": args.blend_mask_source,
                "expanded_mask_box": mask_debug.get("expanded_mask_box"),
            }
        )

    summary = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "base_model": args.base_model,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "strength": args.strength,
        "target_size": args.target_size,
        "adaptive_root_bias": args.adaptive_root_bias,
        "mask_mode": args.mask_mode,
        "mask_dilate_ratio": args.mask_dilate_ratio,
        "mask_min_pad": args.mask_min_pad,
        "blend_mask_source": args.blend_mask_source,
        "blend_mask_dilate_ratio": args.blend_mask_dilate_ratio,
        "blend_mask_min_pad": args.blend_mask_min_pad,
        "blend_mask_blur_ksize": args.blend_mask_blur_ksize,
        "contact_bias": args.contact_bias,
        "adaptive_target_occupancy": args.adaptive_target_occupancy,
        "adaptive_min_side": args.adaptive_min_side,
        "axial_scale": args.axial_scale,
        "transverse_scale": args.transverse_scale,
        "geometry_prior": args.geometry_prior,
        "geometry_prior_strength": args.geometry_prior_strength,
        "controlnet_enabled": controlnet_enabled,
        "controlnet_model": args.controlnet_model,
        "controlnet_conditioning_scale": args.controlnet_conditioning_scale,
        "control_guidance_start": args.control_guidance_start,
        "control_guidance_end": args.control_guidance_end,
        "control_image_source": args.control_image_source,
        "control_canny_low_threshold": args.control_canny_low_threshold,
        "control_canny_high_threshold": args.control_canny_high_threshold,
        "image_name": args.image_name,
        "ip_adapter_enabled": ip_adapter_enabled,
        "reference_image": args.reference_image,
        "ip_adapter_repo": args.ip_adapter_repo,
        "ip_adapter_subfolder": args.ip_adapter_subfolder,
        "ip_adapter_weight_name": args.ip_adapter_weight_name,
        "ip_adapter_scale": args.ip_adapter_scale,
        "lora_enabled": bool(args.lora_path),
        "lora_path": args.lora_path,
        "lora_scale": args.lora_scale,
        "lora_adapter_name": args.lora_adapter_name,
        "records": summary_records,
    }
    (output_dir / "batch_manifest_oriented.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
