from __future__ import annotations

import os
import threading
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


DEFAULT_SAM2_CONFIG = os.environ.get(
    "GOOD_BOLT_SAM2_CONFIG",
    "configs/sam2.1/sam2.1_hiera_t.yaml",
)
DEFAULT_SAM2_CHECKPOINT = os.environ.get(
    "GOOD_BOLT_SAM2_CHECKPOINT",
    "/root/sam2-local/checkpoints/sam2.1_hiera_tiny.pt",
)
DEFAULT_EDIT_DILATE_PX = int(os.environ.get("GOOD_BOLT_SAM2_EDIT_DILATE_PX", "18"))
DEFAULT_EDIT_DILATE_RATIO = float(os.environ.get("GOOD_BOLT_SAM2_EDIT_DILATE_RATIO", "0.10"))
DEFAULT_EDIT_DILATE_MAX_PX = int(os.environ.get("GOOD_BOLT_SAM2_EDIT_DILATE_MAX_PX", "64"))

_RUNTIME: dict[str, object] | None = None
_RUNTIME_LOCK = threading.Lock()


def clamp_box_xyxy(box_xyxy: list[int], width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = [int(round(float(value))) for value in box_xyxy]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def mask_array_to_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")


def mask_bbox(mask_image: Image.Image) -> tuple[int, int, int, int] | None:
    mask = np.asarray(mask_image.convert("L")) > 0
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def fit_mask_geometry(mask_image: Image.Image) -> dict[str, object] | None:
    mask = np.asarray(mask_image.convert("L")) > 0
    ys, xs = np.nonzero(mask)
    if xs.size < 3:
        return None

    points = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    center = points.mean(axis=0)
    centered = points - center
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    major_axis = eigenvectors[:, order[0]].astype(np.float32)
    norm = float(np.linalg.norm(major_axis))
    if norm <= 1e-8:
        major_axis = np.array([1.0, 0.0], dtype=np.float32)
    else:
        major_axis /= norm

    if abs(float(major_axis[1])) >= abs(float(major_axis[0])):
        if float(major_axis[1]) < 0.0:
            major_axis *= -1.0
    elif float(major_axis[0]) < 0.0:
        major_axis *= -1.0

    minor_axis = np.array([-major_axis[1], major_axis[0]], dtype=np.float32)
    major_projection = centered @ major_axis
    minor_projection = centered @ minor_axis
    return {
        "center": center,
        "major_axis": major_axis,
        "minor_axis": minor_axis,
        "contact_offset": float(np.min(major_projection)),
        "far_offset": float(np.max(major_projection)),
        "minor_radius": float(np.max(np.abs(minor_projection))),
    }


def resolve_edit_dilate_px(
    mask_image: Image.Image,
    *,
    min_px: int = DEFAULT_EDIT_DILATE_PX,
    ratio: float = DEFAULT_EDIT_DILATE_RATIO,
    max_px: int = DEFAULT_EDIT_DILATE_MAX_PX,
) -> int:
    bbox = mask_bbox(mask_image)
    if bbox is None:
        return max(0, int(min_px))

    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    diagonal = math.hypot(width, height)
    resolved = int(math.ceil(diagonal * max(0.0, float(ratio))))
    resolved = max(int(min_px), resolved)
    if max_px > 0:
        resolved = min(int(max_px), resolved)
    return max(0, resolved)


def expand_binary_mask(mask_image: Image.Image, expand_px: int) -> Image.Image:
    if expand_px <= 0:
        return mask_image.copy()
    size = expand_px * 2 + 1
    return mask_image.filter(ImageFilter.MaxFilter(size=size))


def build_directional_edit_mask(mask_image: Image.Image, expand_px: int) -> Image.Image:
    if expand_px <= 0:
        return mask_image.copy()

    geometry = fit_mask_geometry(mask_image)
    if geometry is None:
        return expand_binary_mask(mask_image, expand_px)

    mask = np.asarray(mask_image.convert("L")) > 0
    height, width = mask.shape
    grid_y, grid_x = np.indices((height, width), dtype=np.float32)
    center = geometry["center"]
    major_axis = geometry["major_axis"]
    minor_axis = geometry["minor_axis"]

    centered_x = grid_x - float(center[0])
    centered_y = grid_y - float(center[1])
    axis_coord = centered_x * float(major_axis[0]) + centered_y * float(major_axis[1])
    transverse_coord = centered_x * float(minor_axis[0]) + centered_y * float(minor_axis[1])

    contact_offset = float(geometry["contact_offset"])
    far_offset = float(geometry["far_offset"])
    minor_radius = float(geometry["minor_radius"])

    contact_extra = float(expand_px)
    far_extra = max(8.0, float(expand_px) * 0.35)
    transverse_extra = max(12.0, float(expand_px) * 0.75)
    transverse_limit = minor_radius + transverse_extra

    body = (
        (axis_coord >= contact_offset - contact_extra)
        & (axis_coord <= far_offset + far_extra)
        & (np.abs(transverse_coord) <= transverse_limit)
    )
    top_cap = (
        ((axis_coord - (contact_offset - contact_extra)) ** 2 + transverse_coord**2)
        <= transverse_limit**2
    )
    bottom_cap = (
        ((axis_coord - (far_offset + far_extra)) ** 2 + transverse_coord**2)
        <= transverse_limit**2
    )

    directional = np.zeros((height, width), dtype=np.uint8)
    directional[body | top_cap | bottom_cap] = 255

    # Keep a small isotropic shell so very thin structures do not get clipped.
    isotropic = np.asarray(expand_binary_mask(mask_image, max(6, int(round(expand_px * 0.35)))).convert("L"))
    directional = np.maximum(directional, isotropic)
    directional = np.maximum(directional, np.asarray(mask_image.convert("L")))
    return Image.fromarray(directional.astype(np.uint8), mode="L")


def get_tool_version() -> str:
    return f"{Path(DEFAULT_SAM2_CONFIG).stem}:{Path(DEFAULT_SAM2_CHECKPOINT).stem}"


def get_source_image_mtime_ns(source_image: Path) -> int | None:
    try:
        return source_image.stat().st_mtime_ns
    except OSError:
        return None


def get_runtime() -> dict[str, object]:
    global _RUNTIME
    with _RUNTIME_LOCK:
        if _RUNTIME is not None:
            return _RUNTIME

        try:
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = build_sam2(
                DEFAULT_SAM2_CONFIG,
                DEFAULT_SAM2_CHECKPOINT,
                device=device,
            )
            predictor = SAM2ImagePredictor(model)
        except Exception as exc:  # pragma: no cover - depends on runtime environment
            raise RuntimeError(
                "failed to initialize SAM2 runtime "
                f"(config={DEFAULT_SAM2_CONFIG}, checkpoint={DEFAULT_SAM2_CHECKPOINT})"
            ) from exc

        _RUNTIME = {
            "device": device,
            "predictor": predictor,
            "cached_source_image": None,
            "cached_source_image_mtime_ns": None,
            "cached_image_size": None,
        }
        return _RUNTIME


def predictor(source_image: Path, asset_context: dict[str, object]) -> dict[str, object]:
    runtime = get_runtime()
    sam2_predictor = runtime["predictor"]
    source_image = source_image.expanduser().resolve()
    box_xyxy = asset_context.get("box_xyxy")
    if not isinstance(box_xyxy, list) or len(box_xyxy) != 4:
        raise ValueError(f"asset_context missing valid box_xyxy for {source_image}")

    source_rgb = asset_context.get("source_rgb")
    source_size = asset_context.get("source_size")
    source_key = str(source_image)
    source_mtime_ns = get_source_image_mtime_ns(source_image)

    import torch

    with _RUNTIME_LOCK:
        try:
            if (
                runtime["cached_source_image"] != source_key
                or runtime["cached_source_image_mtime_ns"] != source_mtime_ns
            ):
                if isinstance(source_rgb, np.ndarray):
                    rgb_array = np.array(source_rgb, copy=True)
                    if rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
                        raise ValueError(
                            f"asset_context source_rgb must be an HxWx3 array for {source_image.name}"
                        )
                    height, width = rgb_array.shape[:2]
                else:
                    with Image.open(source_image) as image:
                        rgb = image.convert("RGB")
                        width, height = rgb.size
                        rgb_array = np.array(rgb, copy=True)

                if isinstance(source_size, tuple) and len(source_size) == 2:
                    expected_size = (int(source_size[0]), int(source_size[1]))
                    if expected_size != (width, height):
                        raise ValueError(
                            f"asset_context source_size mismatch for {source_image.name}: "
                            f"expected {expected_size}, got {(width, height)}"
                        )

                if runtime["device"] == "cuda":
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        sam2_predictor.set_image(rgb_array)
                else:
                    with torch.inference_mode():
                        sam2_predictor.set_image(rgb_array)
                runtime["cached_source_image"] = source_key
                runtime["cached_source_image_mtime_ns"] = source_mtime_ns
                runtime["cached_image_size"] = (width, height)

            width, height = runtime["cached_image_size"]
            box = clamp_box_xyxy(box_xyxy, width, height)
            if runtime["device"] == "cuda":
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    masks, scores, _ = sam2_predictor.predict(box=box, multimask_output=False)
            else:
                with torch.inference_mode():
                    masks, scores, _ = sam2_predictor.predict(box=box, multimask_output=False)
        except Exception as exc:
            raise RuntimeError(f"SAM2 prediction failed for {source_image.name}") from exc

    if len(masks) == 0:
        raise RuntimeError(f"SAM2 returned no masks for {source_image.name}")

    core_mask = mask_array_to_image(np.asarray(masks[0], dtype=np.float32))
    edit_dilate_px = resolve_edit_dilate_px(core_mask)
    edit_mask = build_directional_edit_mask(core_mask, edit_dilate_px)
    score = float(scores[0]) if len(scores) > 0 else None
    return {
        "core_mask": core_mask,
        "edit_mask": edit_mask,
        "tool_name": "sam2-box-prompt",
        "tool_version": get_tool_version(),
        "qa_state": "draft",
        "qa_notes": (
            f"sam2_box_score={score:.4f};edit_dilate_px={edit_dilate_px}"
            if score is not None
            else f"sam2_box_score=na;edit_dilate_px={edit_dilate_px}"
        ),
    }
