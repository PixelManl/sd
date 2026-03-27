from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageFilter


def _to_mask_array(mask: Any) -> np.ndarray:
    if isinstance(mask, Image.Image):
        array = np.asarray(mask.convert("L"))
    else:
        array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape={array.shape!r}")
    return np.where(array > 0, 255, 0).astype(np.uint8)


def _dilate_mask(mask: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px <= 0:
        return mask.copy()
    size = radius_px * 2 + 1
    pil_mask = Image.fromarray(mask, mode="L")
    return np.asarray(pil_mask.filter(ImageFilter.MaxFilter(size=size)), dtype=np.uint8)


def _blur_mask(mask: np.ndarray, blur_px: int) -> np.ndarray:
    if blur_px <= 0:
        return mask.copy()
    pil_mask = Image.fromarray(mask, mode="L")
    blurred = pil_mask.filter(ImageFilter.GaussianBlur(radius=float(blur_px)))
    return np.asarray(blurred, dtype=np.uint8)


def sanitize_remove_mask(remove_mask: Any, protect_mask: Any | None = None) -> np.ndarray:
    remove = _to_mask_array(remove_mask)
    if protect_mask is None:
        return remove
    protect = _to_mask_array(protect_mask)
    return np.where(protect > 0, 0, remove).astype(np.uint8)


def _fit_mask_geometry(mask: np.ndarray) -> dict[str, Any] | None:
    ys, xs = np.nonzero(mask > 0)
    if xs.size < 3:
        return None
    points = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    center = points.mean(axis=0)
    centered = points - center
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, order]
    major_axis = eigenvectors[:, 0].astype(np.float32)
    norm = float(np.linalg.norm(major_axis))
    if norm <= 1e-8:
        major_axis = np.array([1.0, 0.0], dtype=np.float32)
    else:
        major_axis = major_axis / norm
    minor_axis = np.array([-major_axis[1], major_axis[0]], dtype=np.float32)
    major_projection = centered @ major_axis
    minor_projection = centered @ minor_axis
    major_radius = float(np.max(np.abs(major_projection)))
    minor_radius = float(np.max(np.abs(minor_projection)))
    return {
        "center": (float(center[0]), float(center[1])),
        "major_axis": major_axis,
        "minor_axis": minor_axis,
        "major_radius": major_radius,
        "minor_radius": minor_radius,
    }


def build_stud_keep_hard_mask(
    remove_mask: Any,
    *,
    length_scale: float = 1.45,
    width_scale: float = 0.32,
    min_length_radius: float = 8.0,
    min_width_radius: float = 2.0,
) -> np.ndarray:
    remove = _to_mask_array(remove_mask)
    geometry = _fit_mask_geometry(remove)
    protect = np.zeros_like(remove, dtype=np.uint8)
    if geometry is None:
        ys, xs = np.nonzero(remove > 0)
        if xs.size == 0:
            return protect
        cx = int(round(float(xs.mean())))
        y1 = max(0, int(ys.min()) - 4)
        y2 = min(remove.shape[0], int(ys.max()) + 5)
        x1 = max(0, cx - 1)
        x2 = min(remove.shape[1], cx + 2)
        protect[y1:y2, x1:x2] = 255
        return protect

    stud_axis = np.array(geometry["minor_axis"], dtype=np.float32)
    normal_axis = np.array([-stud_axis[1], stud_axis[0]], dtype=np.float32)
    center_x, center_y = geometry["center"]
    length_radius = max(float(min_length_radius), float(geometry["major_radius"]) * float(length_scale))
    width_radius = max(float(min_width_radius), float(geometry["minor_radius"]) * float(width_scale))

    grid_y, grid_x = np.indices(remove.shape, dtype=np.float32)
    centered_x = grid_x - float(center_x)
    centered_y = grid_y - float(center_y)
    axial_coord = centered_x * float(stud_axis[0]) + centered_y * float(stud_axis[1])
    transverse_coord = centered_x * float(normal_axis[0]) + centered_y * float(normal_axis[1])

    body = (
        (np.abs(axial_coord) <= length_radius)
        & (np.abs(transverse_coord) <= width_radius)
    )
    cap_a = (axial_coord - length_radius) ** 2 + transverse_coord**2 <= width_radius**2
    cap_b = (axial_coord + length_radius) ** 2 + transverse_coord**2 <= width_radius**2
    protect[body | cap_a | cap_b] = 255
    return protect


def build_three_zone_masks(
    remove_mask: Any,
    *,
    protect_mask: Any | None = None,
    seam_px: int = 2,
    context_px: int = 12,
    blur_px: int = 0,
) -> dict[str, np.ndarray]:
    remove = sanitize_remove_mask(remove_mask, protect_mask)
    keep_hard = _to_mask_array(protect_mask) if protect_mask is not None else np.zeros_like(remove, dtype=np.uint8)

    context_outer = _dilate_mask(remove, max(int(context_px), 0))
    context = np.where(remove > 0, 0, context_outer).astype(np.uint8)
    context = np.where(keep_hard > 0, 0, context).astype(np.uint8)

    paste = _dilate_mask(remove, max(int(seam_px), 0))
    paste = np.where(keep_hard > 0, 0, paste).astype(np.uint8)
    if blur_px > 0:
        paste = _blur_mask(paste, int(blur_px))
        paste = np.where(remove > 0, 255, paste).astype(np.uint8)
        paste = np.where(keep_hard > 0, 0, paste).astype(np.uint8)

    return {
        "keep_hard": keep_hard,
        "remove": remove,
        "context": context,
        "paste": paste,
    }


def composite_generated_patch(
    *,
    source_rgb: Any,
    generated_rgb: Any,
    paste_mask: Any,
) -> np.ndarray:
    source = np.asarray(source_rgb, dtype=np.uint8)
    generated = np.asarray(generated_rgb, dtype=np.uint8)
    if source.shape != generated.shape:
        raise ValueError(f"source and generated shapes must match, got {source.shape!r} vs {generated.shape!r}")
    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError(f"source and generated must be HxWx3, got {source.shape!r}")

    paste = _to_mask_array(paste_mask).astype(np.float32) / 255.0
    alpha = paste[:, :, None]
    composited = generated.astype(np.float32) * alpha + source.astype(np.float32) * (1.0 - alpha)
    return np.clip(composited, 0, 255).astype(np.uint8)
