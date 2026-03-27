from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from bolt.generate.mask_geometry import fit_mask_geometry


@dataclass(frozen=True)
class StudGeometryPrior:
    axis_vector: tuple[float, float]
    stud_half_width: float
    contact_offset: float
    tail_offset: float
    axis_mask: np.ndarray
    envelope_mask: np.ndarray
    preserve_tail_mask: np.ndarray


def _ensure_odd(value: int) -> int:
    if value <= 1:
        return 0
    return value if value % 2 == 1 else value + 1


def _orient_axis(axis: tuple[float, float]) -> np.ndarray:
    vector = np.array(axis, dtype=np.float32)
    if vector.shape != (2,):
        raise ValueError(f"axis must have shape (2,), got {vector.shape!r}")

    if abs(float(vector[1])) >= abs(float(vector[0])):
        if float(vector[1]) < 0.0:
            vector *= -1.0
    elif float(vector[0]) < 0.0:
        vector *= -1.0
    return vector


def _mask_from_band(
    shape: tuple[int, int],
    *,
    center: tuple[float, float],
    axis_vector: np.ndarray,
    start_offset: float,
    end_offset: float,
    half_width: float,
    blur_ksize: int,
) -> np.ndarray:
    grid_y, grid_x = np.indices(shape, dtype=np.float32)
    centered_x = grid_x - float(center[0])
    centered_y = grid_y - float(center[1])
    axis_coord = centered_x * float(axis_vector[0]) + centered_y * float(axis_vector[1])
    transverse = centered_x * float(-axis_vector[1]) + centered_y * float(axis_vector[0])

    mask = np.zeros(shape, dtype=np.uint8)
    band = (
        (axis_coord >= float(start_offset))
        & (axis_coord <= float(end_offset))
        & (np.abs(transverse) <= float(max(1.0, half_width)))
    )
    mask[band] = 255

    blur_ksize = _ensure_odd(blur_ksize)
    if blur_ksize > 1:
        mask = cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), 0)
    return mask


def build_stud_geometry_prior(
    core_mask: np.ndarray,
    *,
    prior_mode: str = "envelope",
    nut_width_ratio: float = 2.4,
    tail_keep_ratio: float = 0.25,
    min_envelope_length: float = 12.0,
    axis_thickness_ratio: float = 0.8,
    blur_ksize: int = 15,
) -> StudGeometryPrior | None:
    binary_mask = np.asarray(core_mask)
    if binary_mask.ndim != 2:
        raise ValueError(f"core_mask must be 2D, got shape {binary_mask.shape!r}")
    if prior_mode not in {"axis", "envelope"}:
        raise ValueError(f"prior_mode must be 'axis' or 'envelope', got {prior_mode!r}")

    geometry = fit_mask_geometry(binary_mask)
    if geometry is None:
        return None

    ys, xs = np.nonzero(binary_mask > 0)
    points = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    axis_vector = _orient_axis(geometry.major_axis)
    centered = points - np.array(geometry.center, dtype=np.float32)
    projection = centered @ axis_vector

    contact_offset = float(np.min(projection))
    bottom_offset = float(np.max(projection))
    span = max(1.0, bottom_offset - contact_offset)
    tail_length = max(4.0, span * float(tail_keep_ratio))
    tail_offset = max(contact_offset + float(min_envelope_length), bottom_offset - tail_length)

    stud_half_width = max(2.0, float(geometry.minor_radius))
    axis_half_width = max(1.0, stud_half_width * float(axis_thickness_ratio))
    nut_half_width = max(axis_half_width + 1.0, stud_half_width * float(nut_width_ratio))

    axis_mask = _mask_from_band(
        binary_mask.shape,
        center=geometry.center,
        axis_vector=axis_vector,
        start_offset=contact_offset,
        end_offset=tail_offset,
        half_width=axis_half_width,
        blur_ksize=blur_ksize,
    )
    envelope_half_width = nut_half_width if prior_mode == "envelope" else axis_half_width
    envelope_mask = _mask_from_band(
        binary_mask.shape,
        center=geometry.center,
        axis_vector=axis_vector,
        start_offset=contact_offset,
        end_offset=tail_offset,
        half_width=envelope_half_width,
        blur_ksize=blur_ksize,
    )
    preserve_tail_mask = _mask_from_band(
        binary_mask.shape,
        center=geometry.center,
        axis_vector=axis_vector,
        start_offset=tail_offset,
        end_offset=bottom_offset,
        half_width=stud_half_width * 1.2,
        blur_ksize=blur_ksize,
    )

    return StudGeometryPrior(
        axis_vector=(float(axis_vector[0]), float(axis_vector[1])),
        stud_half_width=stud_half_width,
        contact_offset=contact_offset,
        tail_offset=tail_offset,
        axis_mask=axis_mask,
        envelope_mask=envelope_mask,
        preserve_tail_mask=preserve_tail_mask,
    )


def seed_roi_with_geometry_prior(
    roi_rgb: np.ndarray,
    prior_mask: np.ndarray,
    *,
    fill_rgb: tuple[int, int, int] = (160, 160, 160),
    alpha: float = 0.55,
) -> np.ndarray:
    roi = np.asarray(roi_rgb)
    mask = np.asarray(prior_mask)
    if roi.ndim != 3 or roi.shape[2] != 3:
        raise ValueError(f"roi_rgb must have shape (H, W, 3), got {roi.shape!r}")
    if mask.shape != roi.shape[:2]:
        raise ValueError(
            f"prior_mask must match roi spatial shape, got mask {mask.shape!r} vs roi {roi.shape!r}"
        )

    alpha = float(max(0.0, min(1.0, alpha)))
    overlay = np.empty_like(roi, dtype=np.float32)
    overlay[:, :] = np.array(fill_rgb, dtype=np.float32)
    blend = (mask.astype(np.float32) / 255.0)[:, :, None] * alpha

    seeded = roi.astype(np.float32) * (1.0 - blend) + overlay * blend
    return np.clip(seeded, 0, 255).astype(np.uint8)
