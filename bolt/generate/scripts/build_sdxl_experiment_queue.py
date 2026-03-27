from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_IMAGE_NAMES = (
    "1766374553.644332.jpg",
    "1766374537.309335.jpg",
    "1766374537.2991467.jpg",
    "1766374537.3700714.jpg",
    "1766374537.6485233.jpg",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a staged SDXL experiment queue package without launching inference."
    )
    parser.add_argument("--batch-manifest", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--core-mask-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--queue-dir", required=True)
    parser.add_argument("--image-name", action="append", default=None)
    parser.add_argument("--base-model", default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--strength", type=float, default=0.92)
    parser.add_argument("--target-size", type=int, default=1024)
    parser.add_argument("--seed-base", type=int, default=9000)
    parser.add_argument("--adaptive-min-side", type=int, default=256)
    parser.add_argument("--geometry-prior-strength", type=float, default=0.55)
    return parser.parse_args()


def _task(
    *,
    task_name: str,
    stage: str,
    output_root: str,
    batch_manifest: str,
    image_dir: str,
    core_mask_dir: str,
    image_names: list[str],
    geometry_prior: str,
    adaptive_target_occupancy: float,
    adaptive_min_side: int,
    base_model: str,
    steps: int,
    guidance_scale: float,
    strength: float,
    target_size: int,
    seed_base: int,
    limit: int,
    mask_mode: str,
) -> dict[str, object]:
    return {
        "task_name": task_name,
        "stage": stage,
        "args": {
            "batch_manifest": batch_manifest,
            "image_dir": image_dir,
            "core_mask_dir": core_mask_dir,
            "output_dir": f"{output_root}/{task_name}",
            "base_model": base_model,
            "image_names": image_names,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "strength": strength,
            "target_size": target_size,
            "seed_base": seed_base,
            "limit": limit,
            "adaptive_target_occupancy": adaptive_target_occupancy,
            "adaptive_min_side": adaptive_min_side,
            "mask_mode": mask_mode,
            "geometry_prior": geometry_prior,
        },
    }


def build_default_queue(
    *,
    batch_manifest: str,
    image_dir: str,
    core_mask_dir: str,
    output_root: str,
    image_names: list[str],
    base_model: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    steps: int = 30,
    guidance_scale: float = 6.0,
    strength: float = 0.92,
    target_size: int = 1024,
    seed_base: int = 9000,
    adaptive_min_side: int = 256,
) -> dict[str, object]:
    probes = [
        ("exp_a_oriented_axis", "structure_probe_a", "oriented", "axis", 0.22, strength, seed_base),
        (
            "exp_a_oriented_envelope",
            "structure_probe_a",
            "oriented",
            "envelope",
            0.22,
            strength,
            seed_base + 100,
        ),
        ("exp_a_root_axis", "structure_probe_a", "root_contact", "axis", 0.22, strength, seed_base + 200),
        (
            "exp_a_root_envelope",
            "structure_probe_a",
            "root_contact",
            "envelope",
            0.22,
            strength,
            seed_base + 300,
        ),
        (
            "exp_b_oriented_axis",
            "structure_probe_b",
            "oriented",
            "axis",
            0.24,
            min(round(strength + 0.04, 2), 1.0),
            seed_base + 400,
        ),
        (
            "exp_b_oriented_envelope",
            "structure_probe_b",
            "oriented",
            "envelope",
            0.24,
            min(round(strength + 0.04, 2), 1.0),
            seed_base + 500,
        ),
        (
            "exp_b_root_axis",
            "structure_probe_b",
            "root_contact",
            "axis",
            0.24,
            min(round(strength + 0.04, 2), 1.0),
            seed_base + 600,
        ),
        (
            "exp_b_root_envelope",
            "structure_probe_b",
            "root_contact",
            "envelope",
            0.24,
            min(round(strength + 0.04, 2), 1.0),
            seed_base + 700,
        ),
    ]

    tasks = [
        _task(
            task_name=task_name,
            stage=stage,
            output_root=output_root,
            batch_manifest=batch_manifest,
            image_dir=image_dir,
            core_mask_dir=core_mask_dir,
            image_names=image_names,
            geometry_prior=geometry_prior,
            adaptive_target_occupancy=occupancy,
            adaptive_min_side=adaptive_min_side,
            base_model=base_model,
            steps=steps,
            guidance_scale=guidance_scale,
            strength=task_strength,
            target_size=target_size,
            seed_base=task_seed,
            limit=0,
            mask_mode=mask_mode,
        )
        for task_name, stage, mask_mode, geometry_prior, occupancy, task_strength, task_seed in probes
    ]

    return {
        "queue_name": "sdxl_targeted_remove_queue_v2",
        "estimated_runtime_hours": 2,
        "tasks": tasks,
    }


def _build_command(task: dict[str, object], shell: str) -> str:
    args = task["args"]
    script = "bolt/generate/scripts/run_sdxl_oriented_batch.py"
    if shell == "powershell":
        prefix = "python -m uv run python"
        cont = " `\n  "
    else:
        prefix = "python -m uv run python"
        cont = " \\\n  "

    parts = [
        f"{prefix} {script}",
        f"--batch-manifest {args['batch_manifest']}",
        f"--image-dir {args['image_dir']}",
        f"--core-mask-dir {args['core_mask_dir']}",
        f"--output-dir {args['output_dir']}",
        f"--base-model {args['base_model']}",
        f"--steps {args['steps']}",
        f"--guidance-scale {args['guidance_scale']}",
        f"--strength {args['strength']}",
        f"--target-size {args['target_size']}",
        f"--seed-base {args['seed_base']}",
        f"--limit {args['limit']}",
        f"--adaptive-target-occupancy {args['adaptive_target_occupancy']}",
        f"--adaptive-min-side {args['adaptive_min_side']}",
        f"--mask-mode {args['mask_mode']}",
        f"--geometry-prior {args['geometry_prior']}",
    ]
    for image_name in args["image_names"]:
        parts.append(f"--image-name {image_name}")
    return cont.join(parts)


def write_queue_package(queue: dict[str, object], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "queue_manifest.json"
    bash_path = output_dir / "run_queue.sh"
    powershell_path = output_dir / "run_queue.ps1"

    manifest_path.write_text(json.dumps(queue, ensure_ascii=False, indent=2), encoding="utf-8")

    bash_lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    ps_lines = ["$ErrorActionPreference = 'Stop'", ""]
    for task in queue["tasks"]:
        bash_lines.append(f"# {task['task_name']} [{task['stage']}]")
        bash_lines.append(_build_command(task, "bash"))
        bash_lines.append("")

        ps_lines.append(f"# {task['task_name']} [{task['stage']}]")
        ps_lines.append(_build_command(task, "powershell"))
        ps_lines.append("")

    bash_path.write_text("\n".join(bash_lines).rstrip() + "\n", encoding="utf-8")
    powershell_path.write_text("\n".join(ps_lines).rstrip() + "\n", encoding="utf-8")
    return {
        "manifest_path": str(manifest_path),
        "bash_path": str(bash_path),
        "powershell_path": str(powershell_path),
    }


def main() -> int:
    args = parse_args()
    image_names = list(args.image_name or DEFAULT_IMAGE_NAMES)
    queue = build_default_queue(
        batch_manifest=args.batch_manifest,
        image_dir=args.image_dir,
        core_mask_dir=args.core_mask_dir,
        output_root=args.output_root,
        image_names=image_names,
        base_model=args.base_model,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        strength=args.strength,
        target_size=args.target_size,
        seed_base=args.seed_base,
        adaptive_min_side=args.adaptive_min_side,
    )
    package = write_queue_package(queue, Path(args.queue_dir))
    print(json.dumps(package, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
