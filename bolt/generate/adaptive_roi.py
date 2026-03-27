from __future__ import annotations

import math

import numpy as np

from bolt.generate.distance_ladder import mask_bbox
from bolt.generate.mask_geometry import fit_mask_geometry


def _clamp_square_interval(center: float, side: int, limit: int) -> tuple[int, int]:
    side = min(max(1, int(side)), int(limit))
    start = int(math.floor(center - side / 2.0))
    end = start + side

    if start < 0:
        end -= start
        start = 0
    if end > limit:
        start -= end - limit
        end = limit
    if start < 0:
        start = 0
    return start, end


def compute_square_crop_box_from_bbox(
    box: list[int] | tuple[int, int, int, int],
    *,
    image_width: int,
    image_height: int,
    target_occupancy: float,
    min_side: int = 0,
) -> list[int]:
    if target_occupancy <= 0.0 or target_occupancy > 1.0:
        raise ValueError(f"target_occupancy must be in (0, 1], got {target_occupancy}")

    x1, y1, x2, y2 = [int(v) for v in box]
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    dominant_side = max(width, height)
    desired_side = max(int(min_side), int(math.ceil(dominant_side / target_occupancy)))
    desired_side = min(desired_side, int(image_width), int(image_height))

    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    crop_x1, crop_x2 = _clamp_square_interval(center_x, desired_side, int(image_width))
    crop_y1, crop_y2 = _clamp_square_interval(center_y, desired_side, int(image_height))
    side = min(crop_x2 - crop_x1, crop_y2 - crop_y1)

    crop_x1, crop_x2 = _clamp_square_interval(center_x, side, int(image_width))
    crop_y1, crop_y2 = _clamp_square_interval(center_y, side, int(image_height))
    return [crop_x1, crop_y1, crop_x2, crop_y2]


def compute_square_crop_box_from_mask(
    mask: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    target_occupancy: float,
    min_side: int = 0,
    root_bias: float = 0.0,
) -> list[int] | None:
    bbox = mask_bbox(np.asarray(mask))
    if bbox is None:
        return None
    if root_bias <= 0.0:
        return compute_square_crop_box_from_bbox(
            bbox,
            image_width=image_width,
            image_height=image_height,
            target_occupancy=target_occupancy,
            min_side=min_side,
        )

    geometry = fit_mask_geometry(np.asarray(mask))
    if geometry is None:
        return compute_square_crop_box_from_bbox(
            bbox,
            image_width=image_width,
            image_height=image_height,
            target_occupancy=target_occupancy,
            min_side=min_side,
        )

    x1, y1, x2, y2 = [int(v) for v in bbox]
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    dominant_side = max(width, height)
    desired_side = max(int(min_side), int(math.ceil(dominant_side / target_occupancy)))
    desired_side = min(desired_side, int(image_width), int(image_height))

    bias = max(0.0, min(float(root_bias), 1.0))
    axis = np.array(geometry.major_axis, dtype=np.float32)
    if abs(float(axis[1])) >= abs(float(axis[0])):
        if float(axis[1]) < 0.0:
            axis *= -1.0
    elif float(axis[0]) < 0.0:
        axis *= -1.0

    base_center_x = (x1 + x2) / 2.0
    base_center_y = (y1 + y2) / 2.0
    center_x = float(base_center_x) - float(axis[0]) * float(geometry.major_radius) * bias
    center_y = float(base_center_y) - float(axis[1]) * float(geometry.major_radius) * bias
    crop_x1, crop_x2 = _clamp_square_interval(center_x, desired_side, int(image_width))
    crop_y1, crop_y2 = _clamp_square_interval(center_y, desired_side, int(image_height))
    side = min(crop_x2 - crop_x1, crop_y2 - crop_y1)

    crop_x1, crop_x2 = _clamp_square_interval(center_x, side, int(image_width))
    crop_y1, crop_y2 = _clamp_square_interval(center_y, side, int(image_height))
    return [crop_x1, crop_y1, crop_x2, crop_y2]
