from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _copy_backend(*, source_image: Path, output_image: Path, record: dict[str, Any]) -> dict[str, Any]:
    output_image.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_image, output_image)
    return {
        "backend_mode": "placeholder-copy",
        "source_image": str(source_image.resolve()),
        "edited_image": str(output_image.resolve()),
        "target_id": str(record.get("target_id") or ""),
    }


def _resolve_box(record: dict[str, Any], key: str = "mask_box") -> list[int]:
    raw_box = record.get(key) or record.get("bbox")
    if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
        raise ValueError(f"Record must contain {key} or bbox with 4 elements")
    return [int(round(float(value))) for value in raw_box]


def _resolve_mask_path(record: dict[str, Any], key: str = "mask_path") -> Path | None:
    raw_path = str(record.get(key) or "").strip()
    if not raw_path:
        return None
    mask_path = Path(raw_path).resolve()
    if not mask_path.exists():
        raise FileNotFoundError(f"PowerPaint {key} not found: {mask_path}")
    return mask_path


def _resolve_optional_dir(record: dict[str, Any], key: str) -> Path | None:
    raw_path = str(record.get(key) or "").strip()
    if not raw_path:
        return None
    resolved = Path(raw_path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{key} not found: {resolved}")
    return resolved


def _run_powerpaint_v2_1_offline(
    *,
    source_image: Path,
    output_image: Path,
    record: dict[str, Any],
) -> dict[str, Any]:
    checkpoint_dir_text = str(record.get("powerpaint_checkpoint_dir") or "").strip()
    conda_prefix_text = str(record.get("powerpaint_conda_prefix") or "").strip()
    if not checkpoint_dir_text:
        raise ValueError("powerpaint_checkpoint_dir is required for powerpaint-v2-1-offline")
    if not conda_prefix_text:
        raise ValueError("powerpaint_conda_prefix is required for powerpaint-v2-1-offline")

    checkpoint_dir = Path(checkpoint_dir_text).resolve()
    conda_prefix = Path(conda_prefix_text).resolve()
    powerpaint_repo_dir = _resolve_optional_dir(record, "powerpaint_repo_dir")
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"PowerPaint checkpoint_dir not found: {checkpoint_dir}")
    if not conda_prefix.exists():
        raise FileNotFoundError(f"PowerPaint conda prefix not found: {conda_prefix}")

    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / ".context" / "scratch" / "powerpaint_backend_infer.py"
    python_executable = conda_prefix / ("python.exe" if os.name == "nt" else "bin/python")
    if not python_executable.exists():
        raise FileNotFoundError(f"PowerPaint python executable not found: {python_executable}")
    output_image.parent.mkdir(parents=True, exist_ok=True)

    mask_path = _resolve_mask_path(record, "mask_path")
    protect_mask_path = _resolve_mask_path(record, "protect_mask_path")
    paste_mask_path = _resolve_mask_path(record, "paste_mask_path")
    env = os.environ.copy()
    env["POWERPAINT_SOURCE_IMAGE"] = str(source_image.resolve())
    env["POWERPAINT_OUTPUT_IMAGE"] = str(output_image.resolve())
    env["POWERPAINT_CHECKPOINT_DIR"] = str(checkpoint_dir)
    if powerpaint_repo_dir is not None:
        env["POWERPAINT_REPO_DIR"] = str(powerpaint_repo_dir)
    if protect_mask_path is not None:
        env["POWERPAINT_PROTECT_MASK_PATH"] = str(protect_mask_path)
    else:
        env.pop("POWERPAINT_PROTECT_MASK_PATH", None)
    if paste_mask_path is not None:
        env["POWERPAINT_PASTE_MASK_PATH"] = str(paste_mask_path)
    else:
        env.pop("POWERPAINT_PASTE_MASK_PATH", None)
    if mask_path is not None:
        env["POWERPAINT_MASK_PATH"] = str(mask_path)
        env.pop("POWERPAINT_MASK_BOX", None)
    else:
        mask_box = _resolve_box(record)
        env["POWERPAINT_MASK_BOX"] = ",".join(str(value) for value in mask_box)
        env.pop("POWERPAINT_MASK_PATH", None)
    env["POWERPAINT_STEPS"] = str(int(record.get("steps", 4)))
    env["POWERPAINT_GUIDANCE_SCALE"] = str(float(record.get("guidance_scale", 7.5)))
    env["POWERPAINT_SEED"] = str(int(record.get("seed", 123)))
    env["POWERPAINT_STRICT_PASTE_SEAM_PX"] = str(int(record.get("strict_paste_seam_px", 2)))
    env["POWERPAINT_STRICT_PASTE_BLUR_PX"] = str(int(record.get("strict_paste_blur_px", 0)))
    env["POWERPAINT_CONTEXT_RING_PX"] = str(int(record.get("context_ring_px", 12)))

    completed = subprocess.run(
        [str(python_executable), str(script_path)],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "PowerPaint offline backend failed.\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    if not output_image.exists():
        raise FileNotFoundError(f"PowerPaint did not create output image: {output_image}")

    return {
        "backend_mode": "powerpaint-v2-1-offline",
        "source_image": str(source_image.resolve()),
        "edited_image": str(output_image.resolve()),
        "target_id": str(record.get("target_id") or ""),
        "mask_path": str(mask_path) if mask_path is not None else None,
        "protect_mask_path": str(protect_mask_path) if protect_mask_path is not None else None,
        "paste_mask_path": str(paste_mask_path) if paste_mask_path is not None else None,
        "powerpaint_repo_dir": str(powerpaint_repo_dir) if powerpaint_repo_dir is not None else None,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def run_backend(
    *,
    source_image: Path,
    output_image: Path,
    record: dict[str, Any],
    backend_mode: str,
) -> dict[str, Any]:
    normalized = backend_mode.strip().lower()
    if normalized == "placeholder-copy":
        return _copy_backend(source_image=source_image, output_image=output_image, record=record)
    if normalized == "powerpaint-v2-1-offline":
        return _run_powerpaint_v2_1_offline(
            source_image=source_image,
            output_image=output_image,
            record=record,
        )
    raise ValueError(f"Unsupported backend_mode: {backend_mode}")
