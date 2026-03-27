from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class DistanceVariant:
    name: str
    dilate_ratio: float


DEFAULT_DISTANCE_VARIANTS = (
    DistanceVariant("d18", 0.18),
    DistanceVariant("d26", 0.26),
    DistanceVariant("d34", 0.34),
    DistanceVariant("d42", 0.42),
)


def parse_distance_variants(values: list[str] | None) -> list[DistanceVariant]:
    if not values:
        return list(DEFAULT_DISTANCE_VARIANTS)

    variants: list[DistanceVariant] = []
    for value in values:
        name, _, ratio_text = value.partition(":")
        if not name or not ratio_text:
            raise ValueError(f"variant must look like <name>:<ratio>, got {value!r}")
        variants.append(DistanceVariant(name=name, dilate_ratio=float(ratio_text)))
    return variants


def resolve_base_crop_box(record: dict[str, object]) -> list[int]:
    raw_box = record.get("crop_box")
    if raw_box is None:
        raw_box = record.get("base_crop_box")
    if raw_box is None:
        raise KeyError("record must contain crop_box or base_crop_box")
    return [int(v) for v in raw_box]


def mask_bbox(mask: np.ndarray) -> list[int] | None:
    binary_mask = np.asarray(mask)
    if binary_mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {binary_mask.shape!r}")

    ys, xs = np.nonzero(binary_mask > 0)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def expand_box(
    box: list[int],
    *,
    pad_x: int,
    pad_y: int,
    limit_width: int,
    limit_height: int,
) -> list[int]:
    x1, y1, x2, y2 = box
    return [
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(limit_width, x2 + pad_x),
        min(limit_height, y2 + pad_y),
    ]


def expand_crop_box(
    crop_box: list[int],
    *,
    image_width: int,
    image_height: int,
    crop_scale: float,
) -> list[int]:
    if crop_scale <= 0:
        raise ValueError(f"crop_scale must be positive, got {crop_scale}")

    x1, y1, x2, y2 = crop_box
    width = x2 - x1
    height = y2 - y1
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    half_width = width * crop_scale / 2.0
    half_height = height * crop_scale / 2.0

    return [
        max(0, int(math.floor(center_x - half_width))),
        max(0, int(math.floor(center_y - half_height))),
        min(image_width, int(math.ceil(center_x + half_width))),
        min(image_height, int(math.ceil(center_y + half_height))),
    ]


def _ensure_odd(value: int) -> int:
    if value <= 1:
        return 0
    return value if value % 2 == 1 else value + 1


def build_edit_mask(
    core_mask: np.ndarray,
    *,
    dilate_ratio: float,
    min_pad: int,
    blur_ksize: int,
) -> tuple[np.ndarray, list[int] | None]:
    binary_mask = np.asarray(core_mask)
    if binary_mask.ndim != 2:
        raise ValueError(f"core_mask must be 2D, got shape {binary_mask.shape!r}")

    bbox = mask_bbox(binary_mask)
    edit_mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    if bbox is None:
        return edit_mask, None

    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    diagonal = math.hypot(width, height)
    pad = max(int(min_pad), int(math.ceil(diagonal * dilate_ratio)))

    expanded_box = expand_box(
        bbox,
        pad_x=pad,
        pad_y=pad,
        limit_width=binary_mask.shape[1],
        limit_height=binary_mask.shape[0],
    )
    kernel_size = max(1, pad * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate((binary_mask > 0).astype(np.uint8) * 255, kernel, iterations=1)
    edit_mask[:, :] = dilated

    ex1, ey1, ex2, ey2 = expanded_box
    edit_mask[:ey1, :] = 0
    edit_mask[ey2:, :] = 0
    edit_mask[:, :ex1] = 0
    edit_mask[:, ex2:] = 0

    blur_ksize = _ensure_odd(blur_ksize)
    if blur_ksize > 1:
        edit_mask = cv2.GaussianBlur(edit_mask, (blur_ksize, blur_ksize), 0)

    return edit_mask, expanded_box
