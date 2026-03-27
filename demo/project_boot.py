from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


DEFAULT_PIPELINE_KIND = "sd15-controlnet"
DEFAULT_BASE_MODEL = "runwayml/stable-diffusion-inpainting"
DEFAULT_CONTROLNET_MODEL = "lllyasviel/control_v11p_sd15_canny"
DEFAULT_SDXL_BASE_MODEL = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
DEFAULT_IP_ADAPTER_SCALE = 0.6
DEFAULT_IP_ADAPTER_SUBFOLDER = "sdxl_models"


@dataclass(frozen=True)
class DemoConfig:
    repo_root: Path
    input_dir: Path
    output_dir: Path
    image_path: Path
    mask_path: Path
    pipeline_kind: str
    base_model: str
    controlnet_model: str
    ip_adapter_repo: str
    ip_adapter_subfolder: str
    ip_adapter_weight_name: str
    ip_adapter_scale: float
    reference_image_path: Path
    dry_run: bool


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a defect demo sample.")
    parser.add_argument("--input-dir", default="data/sg/inputs")
    parser.add_argument("--output-dir", default="data/sg/outputs")
    parser.add_argument("--image-name", default="drone_healthy_facade.jpg")
    parser.add_argument("--mask-name", default="crack_position_mask.png")
    parser.add_argument(
        "--pipeline-kind",
        choices=("sd15-controlnet", "sdxl-inpaint"),
        default="",
    )
    parser.add_argument("--base-model", default="")
    parser.add_argument("--controlnet-model", default="")
    parser.add_argument("--reference-image-name", default="reference_subject.jpg")
    parser.add_argument("--ip-adapter-repo", default="")
    parser.add_argument("--ip-adapter-subfolder", default="")
    parser.add_argument("--ip-adapter-weight-name", default="")
    parser.add_argument("--ip-adapter-scale", type=float, default=-1.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _default_base_model_for_pipeline(pipeline_kind: str) -> str:
    if pipeline_kind == "sdxl-inpaint":
        return DEFAULT_SDXL_BASE_MODEL
    return DEFAULT_BASE_MODEL


def _default_controlnet_for_pipeline(pipeline_kind: str) -> str:
    if pipeline_kind == "sdxl-inpaint":
        return ""
    return DEFAULT_CONTROLNET_MODEL


def build_demo_config(
    argv: Sequence[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    repo_root: Path | None = None,
) -> DemoConfig:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    env_map = dict(os.environ if env is None else env)
    root = Path.cwd() if repo_root is None else Path(repo_root)

    input_dir = root / args.input_dir
    output_dir = root / args.output_dir
    image_path = input_dir / args.image_name
    mask_path = input_dir / args.mask_name
    reference_image_path = input_dir / args.reference_image_name

    pipeline_kind = args.pipeline_kind or env_map.get(
        "SD_PIPELINE_KIND",
        DEFAULT_PIPELINE_KIND,
    )
    base_model = args.base_model or env_map.get(
        "SD_BASE_MODEL",
        _default_base_model_for_pipeline(pipeline_kind),
    )
    controlnet_model = args.controlnet_model or env_map.get(
        "SD_CONTROLNET_MODEL",
        _default_controlnet_for_pipeline(pipeline_kind),
    )
    ip_adapter_repo = args.ip_adapter_repo or env_map.get("SD_IP_ADAPTER_REPO", "")
    ip_adapter_subfolder = args.ip_adapter_subfolder or env_map.get(
        "SD_IP_ADAPTER_SUBFOLDER",
        DEFAULT_IP_ADAPTER_SUBFOLDER,
    )
    ip_adapter_weight_name = args.ip_adapter_weight_name or env_map.get(
        "SD_IP_ADAPTER_WEIGHT_NAME",
        "",
    )
    ip_adapter_scale = args.ip_adapter_scale
    if ip_adapter_scale < 0:
        ip_adapter_scale = float(env_map.get("SD_IP_ADAPTER_SCALE", DEFAULT_IP_ADAPTER_SCALE))

    return DemoConfig(
        repo_root=root,
        input_dir=input_dir,
        output_dir=output_dir,
        image_path=image_path,
        mask_path=mask_path,
        pipeline_kind=pipeline_kind,
        base_model=base_model,
        controlnet_model=controlnet_model,
        ip_adapter_repo=ip_adapter_repo,
        ip_adapter_subfolder=ip_adapter_subfolder,
        ip_adapter_weight_name=ip_adapter_weight_name,
        ip_adapter_scale=ip_adapter_scale,
        reference_image_path=reference_image_path,
        dry_run=args.dry_run,
    )
