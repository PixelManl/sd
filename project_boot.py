from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


DEFAULT_BASE_MODEL = "runwayml/stable-diffusion-inpainting"
DEFAULT_CONTROLNET_MODEL = "lllyasviel/control_v11p_sd15_canny"


@dataclass(frozen=True)
class DemoConfig:
    repo_root: Path
    input_dir: Path
    output_dir: Path
    image_path: Path
    mask_path: Path
    base_model: str
    controlnet_model: str
    dry_run: bool


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a defect demo sample.")
    parser.add_argument("--input-dir", default="data/sg/inputs")
    parser.add_argument("--output-dir", default="data/sg/outputs")
    parser.add_argument("--image-name", default="drone_healthy_facade.jpg")
    parser.add_argument("--mask-name", default="crack_position_mask.png")
    parser.add_argument("--base-model", default="")
    parser.add_argument("--controlnet-model", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser


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

    base_model = args.base_model or env_map.get("SD_BASE_MODEL", DEFAULT_BASE_MODEL)
    controlnet_model = args.controlnet_model or env_map.get(
        "SD_CONTROLNET_MODEL",
        DEFAULT_CONTROLNET_MODEL,
    )

    return DemoConfig(
        repo_root=root,
        input_dir=input_dir,
        output_dir=output_dir,
        image_path=image_path,
        mask_path=mask_path,
        base_model=base_model,
        controlnet_model=controlnet_model,
        dry_run=args.dry_run,
    )
