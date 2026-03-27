from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class MaskGeometry:
    center: tuple[float, float]
    major_axis: tuple[float, float]
    minor_axis: tuple[float, float]
    major_radius: float
    minor_radius: float


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return np.array([1.0, 0.0], dtype=np.float32)
    return (vector / norm).astype(np.float32)


def _orient_major_axis(vector: np.ndarray) -> np.ndarray:
    oriented = _normalize(vector)
    if abs(float(oriented[1])) >= abs(float(oriented[0])):
        if float(oriented[1]) < 0.0:
            oriented *= -1.0
    elif float(oriented[0]) < 0.0:
        oriented *= -1.0
    return oriented


def fit_mask_geometry(mask: np.ndarray) -> MaskGeometry | None:
    binary_mask = np.asarray(mask)
    if binary_mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {binary_mask.shape!r}")

    ys, xs = np.nonzero(binary_mask > 0)
    if xs.size < 3:
        return None

    points = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    center = points.mean(axis=0)
    centered = points - center

    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, order]

    major_axis = _normalize(eigenvectors[:, 0])
    minor_axis = np.array([-major_axis[1], major_axis[0]], dtype=np.float32)

    major_projection = centered @ major_axis
    minor_projection = centered @ minor_axis
    major_radius = float(np.max(np.abs(major_projection)))
    minor_radius = float(np.max(np.abs(minor_projection)))

    if major_radius < minor_radius:
        major_axis, minor_axis = minor_axis, major_axis
        major_radius, minor_radius = minor_radius, major_radius

    return MaskGeometry(
        center=(float(center[0]), float(center[1])),
        major_axis=(float(major_axis[0]), float(major_axis[1])),
        minor_axis=(float(minor_axis[0]), float(minor_axis[1])),
        major_radius=major_radius,
        minor_radius=minor_radius,
    )


def build_oriented_focus_mask(
    core_mask: np.ndarray,
    *,
    axial_scale: float = 0.8,
    transverse_scale: float = 1.8,
    min_axial_radius: float = 8.0,
    min_transverse_radius: float = 10.0,
    contact_bias: float = 0.0,
    blur_ksize: int = 17,
) -> np.ndarray:
    binary_mask = np.asarray(core_mask)
    if binary_mask.ndim != 2:
        raise ValueError(f"core_mask must be 2D, got shape {binary_mask.shape!r}")

    geometry = fit_mask_geometry(binary_mask)
    focus_mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    if geometry is None:
        return focus_mask

    axial_radius = max(min_axial_radius, geometry.major_radius * axial_scale)
    transverse_radius = max(min_transverse_radius, geometry.minor_radius * transverse_scale)
    contact_bias = max(0.0, min(float(contact_bias), 1.0))

    grid_y, grid_x = np.indices(binary_mask.shape, dtype=np.float32)
    major_axis = _orient_major_axis(np.array(geometry.major_axis, dtype=np.float32))
    minor_axis = np.array(geometry.minor_axis, dtype=np.float32)
    if float(major_axis[0] * minor_axis[0] + major_axis[1] * minor_axis[1]) > 1e-5:
        minor_axis = np.array([-major_axis[1], major_axis[0]], dtype=np.float32)

    center_shift = major_axis * float(geometry.major_radius) * contact_bias
    centered_x = grid_x - (geometry.center[0] - center_shift[0])
    centered_y = grid_y - (geometry.center[1] - center_shift[1])

    axial_coord = centered_x * major_axis[0] + centered_y * major_axis[1]
    transverse_coord = centered_x * minor_axis[0] + centered_y * minor_axis[1]

    ellipse = (
        (axial_coord / max(axial_radius, 1.0)) ** 2
        + (transverse_coord / max(transverse_radius, 1.0)) ** 2
    ) <= 1.0
    focus_mask[ellipse] = 255

    if blur_ksize > 1:
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        focus_mask = cv2.GaussianBlur(focus_mask, (blur_ksize, blur_ksize), 0)
    return focus_mask


def build_root_contact_focus_mask(
    core_mask: np.ndarray,
    *,
    axial_scale: float = 0.5,
    transverse_scale: float = 2.4,
    min_axial_radius: float = 8.0,
    min_transverse_radius: float = 10.0,
    contact_bias: float = 0.25,
    blur_ksize: int = 17,
) -> np.ndarray:
    binary_mask = np.asarray(core_mask)
    if binary_mask.ndim != 2:
        raise ValueError(f"core_mask must be 2D, got shape {binary_mask.shape!r}")

    geometry = fit_mask_geometry(binary_mask)
    focus_mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    if geometry is None:
        return focus_mask

    ys, xs = np.nonzero(binary_mask > 0)
    points = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    major_axis = _orient_major_axis(np.array(geometry.major_axis, dtype=np.float32))
    minor_axis = np.array([-major_axis[1], major_axis[0]], dtype=np.float32)

    centered_points = points - np.array(geometry.center, dtype=np.float32)
    projection = centered_points @ major_axis
    contact_offset = float(np.min(projection))
    bottom_offset = float(np.max(projection))
    span = max(1.0, bottom_offset - contact_offset)

    downward_extent = max(min_axial_radius, span * float(axial_scale))
    upward_pad = max(0.0, min(float(contact_bias), 1.0)) * downward_extent
    transverse_radius = max(min_transverse_radius, geometry.minor_radius * transverse_scale)

    grid_y, grid_x = np.indices(binary_mask.shape, dtype=np.float32)
    centered_x = grid_x - float(geometry.center[0])
    centered_y = grid_y - float(geometry.center[1])
    axis_coord = centered_x * float(major_axis[0]) + centered_y * float(major_axis[1])
    transverse_coord = centered_x * float(minor_axis[0]) + centered_y * float(minor_axis[1])
    local_axis = axis_coord - contact_offset

    body = (
        (local_axis >= -upward_pad)
        & (local_axis <= downward_extent)
        & (np.abs(transverse_coord) <= transverse_radius)
    )
    top_cap = (local_axis + upward_pad) ** 2 + transverse_coord**2 <= transverse_radius**2
    bottom_cap = (local_axis - downward_extent) ** 2 + transverse_coord**2 <= transverse_radius**2

    focus_mask[body | top_cap | bottom_cap] = 255

    if blur_ksize > 1:
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        focus_mask = cv2.GaussianBlur(focus_mask, (blur_ksize, blur_ksize), 0)
    return focus_mask
