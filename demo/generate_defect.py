from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

try:
    from demo.project_boot import DemoConfig, build_demo_config
except ModuleNotFoundError:
    from project_boot import DemoConfig, build_demo_config


def extract_canny_feature(image):
    import cv2
    import numpy as np
    from PIL import Image

    image_np = np.array(image)
    edges = cv2.Canny(image_np, 100, 200)
    edges = edges[:, :, None]
    edges = np.concatenate([edges, edges, edges], axis=2)
    return Image.fromarray(edges)


def ensure_demo_assets(config: DemoConfig) -> None:
    from PIL import Image, ImageDraw

    config.input_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    if not config.image_path.exists():
        dummy_img = Image.new("RGB", (512, 512), color=(200, 200, 200))
        draw = ImageDraw.Draw(dummy_img)
        for i in range(0, 512, 40):
            draw.line((i, 0, i, 512), fill=(180, 180, 180), width=1)
            draw.line((0, i, 512, i), fill=(180, 180, 180), width=1)
        dummy_img.save(config.image_path)

    if not config.mask_path.exists():
        dummy_mask = Image.new("L", (512, 512), color=0)
        draw = ImageDraw.Draw(dummy_mask)
        draw.line((100, 100, 300, 400), fill=255, width=20)
        draw.line((300, 400, 450, 450), fill=255, width=15)
        dummy_mask.save(config.mask_path)

    if not config.reference_image_path.exists():
        with Image.open(config.image_path) as image:
            reference = image.convert("RGB").crop((156, 156, 356, 356)).resize((512, 512))
        reference.save(config.reference_image_path)


def print_dry_run(config: DemoConfig) -> None:
    payload = {
        "mode": "dry-run",
        "input_dir": str(config.input_dir),
        "output_dir": str(config.output_dir),
        "image_path": str(config.image_path),
        "mask_path": str(config.mask_path),
        "reference_image_path": str(config.reference_image_path),
        "pipeline_kind": config.pipeline_kind,
        "base_model": config.base_model,
        "controlnet_model": config.controlnet_model,
        "ip_adapter_repo": config.ip_adapter_repo,
        "ip_adapter_subfolder": config.ip_adapter_subfolder,
        "ip_adapter_weight_name": config.ip_adapter_weight_name,
        "ip_adapter_scale": config.ip_adapter_scale,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def build_sdxl_load_kwargs(device: str, dtype) -> dict[str, object]:
    load_kwargs: dict[str, object] = {"torch_dtype": dtype}
    if device == "cuda":
        load_kwargs["variant"] = "fp16"
        load_kwargs["use_safetensors"] = True
    return load_kwargs


def run_generation(config: DemoConfig) -> Path:
    import torch
    from diffusers import AutoPipelineForInpainting
    from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    if config.pipeline_kind == "sdxl-inpaint":
        pipe = AutoPipelineForInpainting.from_pretrained(
            config.base_model,
            **build_sdxl_load_kwargs(device, dtype),
        ).to(device)
    else:
        controlnet = ControlNetModel.from_pretrained(
            config.controlnet_model,
            torch_dtype=dtype,
        )

        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            config.base_model,
            controlnet=controlnet,
            torch_dtype=dtype,
        ).to(device)
    pipe.safety_checker = None

    if config.ip_adapter_repo:
        if config.pipeline_kind != "sdxl-inpaint":
            raise ValueError("IP-Adapter is only supported for the sdxl-inpaint path.")
        if not config.ip_adapter_weight_name:
            raise ValueError("IP-Adapter weight name is required when --ip-adapter-repo is set.")
        pipe.load_ip_adapter(
            config.ip_adapter_repo,
            subfolder=config.ip_adapter_subfolder,
            weight_name=config.ip_adapter_weight_name,
        )
        pipe.set_ip_adapter_scale(config.ip_adapter_scale)

    original_image = Image.open(config.image_path).convert("RGB").resize((512, 512))
    mask_image = Image.open(config.mask_path).convert("L").resize((512, 512))
    reference_image = Image.open(config.reference_image_path).convert("RGB").resize((512, 512))
    control_image = extract_canny_feature(original_image)

    prompt = "photorealistic concrete crack, dark moss, weathered building facade, high detail, 8k uhd, coherent cracks"
    negative_prompt = "flat color, monochromatic, gray, smooth, clean, cartoon, illustration, blur, low quality"
    generator = torch.Generator(device=device).manual_seed(42)

    pipe_kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_inference_steps": 30,
        "image": original_image,
        "mask_image": mask_image,
        "generator": generator,
        "strength": 0.9,
    }
    if config.pipeline_kind != "sdxl-inpaint":
        pipe_kwargs["control_image"] = control_image
    if config.ip_adapter_repo:
        pipe_kwargs["ip_adapter_image"] = reference_image

    result = pipe(**pipe_kwargs)

    output_image_path = config.output_dir / "augmented_crack_sample_01.jpg"
    control_image_path = config.output_dir / "debug_control_canny.png"
    result.images[0].save(output_image_path)
    if config.pipeline_kind != "sdxl-inpaint":
        control_image.save(control_image_path)
    return output_image_path


def main(argv: Sequence[str] | None = None) -> int:
    config = build_demo_config(argv, repo_root=Path(__file__).resolve().parents[1])
    ensure_demo_assets(config)

    if config.dry_run:
        print_dry_run(config)
        return 0

    output_path = run_generation(config)
    print(f"Generated sample saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
