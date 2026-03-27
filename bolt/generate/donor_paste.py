from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter


@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)


def alpha_bbox(alpha: Image.Image) -> Box:
    bbox = alpha.convert("L").getbbox()
    if bbox is None:
        raise ValueError("alpha mask is empty")
    return Box(*[int(v) for v in bbox])


def fit_content_box(
    source_content_box: Box,
    target_box: Box,
    *,
    width_ratio: float = 0.72,
    top_offset: int = 0,
) -> Box:
    if source_content_box.width <= 0 or source_content_box.height <= 0:
        raise ValueError("source content box must be non-empty")
    if target_box.width <= 0 or target_box.height <= 0:
        raise ValueError("target box must be non-empty")

    desired_width = max(1, int(round(target_box.width * width_ratio)))
    scale = desired_width / float(source_content_box.width)
    desired_height = max(1, int(round(source_content_box.height * scale)))

    center_x = (target_box.x1 + target_box.x2) / 2.0
    x1 = int(round(center_x - desired_width / 2.0))
    y1 = target_box.y1 + int(top_offset)
    x2 = x1 + desired_width
    y2 = y1 + desired_height
    return Box(x1, y1, x2, y2)


def composite_rgba_at(
    background_rgb: Image.Image,
    donor_rgba: Image.Image,
    placement: Box,
    *,
    feather_radius: float = 2.0,
) -> Image.Image:
    background = background_rgb.convert("RGBA")
    resized = donor_rgba.resize((placement.width, placement.height), Image.Resampling.LANCZOS)

    if feather_radius > 0:
        r, g, b, a = resized.split()
        a = a.filter(ImageFilter.GaussianBlur(radius=feather_radius))
        resized = Image.merge("RGBA", (r, g, b, a))

    layer = Image.new("RGBA", background.size, (0, 0, 0, 0))
    layer.alpha_composite(resized, (placement.x1, placement.y1))
    composited = Image.alpha_composite(background, layer)
    return composited.convert("RGB")


def crop_box_from_mask_array(mask_array: np.ndarray) -> Box:
    ys, xs = np.nonzero(mask_array > 0)
    if len(xs) == 0:
        raise ValueError("mask is empty")
    return Box(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
