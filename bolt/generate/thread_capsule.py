from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageDraw

from bolt.generate.donor_paste import Box, crop_box_from_mask_array


@dataclass(frozen=True)
class ThreadCapsuleDebug:
    target_box: Box
    probe_box: Box
    source_box: Box
    visible_box: Box
    placement_box: Box
    stud_width: int
    center_x: int
    source_mode: str
    visible_top: int
    visible_fade_rows: int
    texture_gray_mean: float
    texture_gray_std: float
    texture_gate_triggered: bool


def _to_binary_mask(mask: Any) -> np.ndarray:
    if isinstance(mask, Image.Image):
        array = np.asarray(mask.convert("L"))
    else:
        array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape={array.shape!r}")
    return (array > 0).astype(np.uint8)


def _resolve_probe_box(image_shape: tuple[int, int, int], target_box: Box, probe_height_ratio: float) -> Box:
    height, width = image_shape[:2]
    target_h = target_box.height
    probe_y1 = min(height - 2, target_box.y2 + max(2, int(target_h * 0.01)))
    probe_y2 = min(height, probe_y1 + max(24, int(target_h * float(probe_height_ratio))))
    if probe_y2 <= probe_y1:
        probe_y1 = max(0, target_box.y1 - max(24, int(target_h * float(probe_height_ratio))))
        probe_y2 = target_box.y1
    if probe_y2 <= probe_y1:
        probe_y1 = max(0, target_box.y1)
        probe_y2 = min(height, target_box.y2)
    return Box(target_box.x1, probe_y1, target_box.x2, probe_y2)


def _pick_lower_component_center(
    mask_bin: np.ndarray, target_box: Box
) -> tuple[float | None, int | None, int | None, int | None]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, 8)
    split_y = target_box.y1 + int(target_box.height * 0.35)
    best_center_x: float | None = None
    best_width: int | None = None
    best_top_y: int | None = None
    best_height: int | None = None
    best_score: float | None = None
    for idx in range(1, count):
        x, y, width, height, area = [int(v) for v in stats[idx]]
        if area < 30 or y < split_y:
            continue
        score = float(y + height * 0.5 + area * 0.001)
        if best_score is None or score > best_score:
            best_score = score
            best_center_x = float(x + width / 2.0)
            best_width = width
            best_top_y = y
            best_height = height
    return best_center_x, best_width, best_top_y, best_height


def _pick_bottom_component(mask_bin: np.ndarray) -> Box | None:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask_bin, 8)
    best_box: Box | None = None
    best_bottom: int | None = None
    best_area: int | None = None
    for idx in range(1, count):
        x, y, width, height, area = [int(v) for v in stats[idx]]
        if area < 30:
            continue
        bottom = y + height
        if (
            best_bottom is None
            or bottom > best_bottom
            or (bottom == best_bottom and (best_area is None or area > best_area))
        ):
            best_bottom = bottom
            best_area = area
            best_box = Box(x, y, x + width, y + height)
    return best_box


def _pick_dark_column_center(image_rgb: np.ndarray, probe_box: Box) -> int:
    probe = image_rgb[probe_box.y1:probe_box.y2, probe_box.x1:probe_box.x2]
    if probe.size == 0:
        return int(round((probe_box.x1 + probe_box.x2) / 2.0))
    gray = cv2.cvtColor(probe, cv2.COLOR_RGB2GRAY)
    col_scores = gray.mean(axis=0)
    if col_scores.size == 0:
        return int(round((probe_box.x1 + probe_box.x2) / 2.0))
    return probe_box.x1 + int(np.argmin(col_scores))


def _build_capsule_alpha(width: int, height: int, blur_radius: float) -> Image.Image:
    alpha = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(alpha)
    radius = max(3, width // 2)
    draw.rounded_rectangle((0, 0, width - 1, height - 1), radius=radius, fill=255)
    if blur_radius > 0:
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))
    return alpha


def _resolve_visible_segment_box(
    target_box: Box,
    *,
    full_height_threshold: int,
    visible_height_ratio: float,
    min_visible_height: int,
) -> Box:
    if target_box.height <= int(full_height_threshold):
        return target_box

    visible_height = max(int(min_visible_height), int(round(target_box.height * float(visible_height_ratio))))
    visible_height = max(1, min(target_box.height, visible_height))
    return Box(target_box.x1, target_box.y2 - visible_height, target_box.x2, target_box.y2)


def _build_vertical_gate(height: int, visible_top: int, fade_rows: int) -> np.ndarray:
    gate = np.zeros((height,), dtype=np.float32)
    start = int(np.clip(visible_top, 0, max(0, height - 1)))
    fade = max(1, int(fade_rows))
    stop = min(height, start + fade)
    if stop > start:
        gate[start:stop] = np.linspace(0.0, 1.0, stop - start, endpoint=False, dtype=np.float32)
    gate[stop:] = 1.0
    return gate


def _build_segment_alpha(width: int, height: int, blur_radius: float, top_fade_ratio: float) -> Image.Image:
    fade_rows = max(1, int(round(height * float(top_fade_ratio))))
    base_alpha = np.asarray(_build_capsule_alpha(width, height, blur_radius=0.0), dtype=np.float32)
    vertical_gate = _build_vertical_gate(height, 0, fade_rows)[:, None]
    alpha = Image.fromarray(np.clip(base_alpha * vertical_gate, 0, 255).astype(np.uint8), mode="L")
    if blur_radius > 0:
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))
    return alpha


def _measure_texture_gray(texture: np.ndarray) -> tuple[float, float]:
    if texture.ndim != 3 or texture.shape[2] != 3:
        raise ValueError(f"texture must have shape (H, W, 3), got {texture.shape!r}")
    gray = cv2.cvtColor(texture, cv2.COLOR_RGB2GRAY)
    return float(gray.mean()), float(gray.std())


def _should_gate_texture(
    texture: np.ndarray,
    *,
    bright_mean_threshold: float = 220.0,
    low_std_threshold: float = 8.0,
) -> tuple[bool, float, float]:
    mean_gray, std_gray = _measure_texture_gray(texture)
    should_gate = mean_gray >= float(bright_mean_threshold) and std_gray <= float(low_std_threshold)
    return should_gate, mean_gray, std_gray


def _resolve_visible_box(
    target_box: Box,
    bottom_component_box: Box | None,
    *,
    tall_mask_min_height: int,
    visible_height_ratio: float,
    lower_anchor_extend_ratio: float,
    lower_anchor_min_pixels: int,
) -> Box:
    base_box = _resolve_visible_segment_box(
        target_box,
        full_height_threshold=tall_mask_min_height,
        visible_height_ratio=visible_height_ratio,
        min_visible_height=max(32, int(round(target_box.height * 0.18))),
    )
    if bottom_component_box is None or target_box.height < int(tall_mask_min_height):
        return base_box

    extend_rows = max(
        int(lower_anchor_min_pixels),
        int(round(bottom_component_box.height * float(lower_anchor_extend_ratio))),
    )
    visible_y1 = min(base_box.y1, max(target_box.y1, bottom_component_box.y1 - extend_rows))
    return Box(target_box.x1, visible_y1, target_box.x2, target_box.y2)


def repair_mask_with_thread_capsule(
    image_rgb: Any,
    target_mask: Any,
    *,
    inpaint_radius: int = 5,
    probe_height_ratio: float = 0.20,
    stud_width_ratio: float = 0.22,
    capsule_blur_radius: float = 2.0,
    tall_mask_min_height: int = 120,
    visible_height_ratio: float = 0.42,
    lower_anchor_extend_ratio: float = 0.45,
    lower_anchor_min_pixels: int = 18,
    vertical_fade_ratio: float = 0.35,
) -> tuple[np.ndarray, ThreadCapsuleDebug]:
    source = np.asarray(image_rgb, dtype=np.uint8)
    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError(f"image_rgb must have shape (H, W, 3), got {source.shape!r}")

    mask_bin = _to_binary_mask(target_mask)
    target_box = crop_box_from_mask_array(mask_bin * 255)
    probe_box = _resolve_probe_box(source.shape, target_box, probe_height_ratio)
    visible_box = _resolve_visible_box(
        target_box,
        _pick_bottom_component(mask_bin),
        tall_mask_min_height=tall_mask_min_height,
        visible_height_ratio=visible_height_ratio,
        lower_anchor_extend_ratio=lower_anchor_extend_ratio,
        lower_anchor_min_pixels=lower_anchor_min_pixels,
    )

    center_x, component_width, _, _ = _pick_lower_component_center(mask_bin, target_box)
    source_mode = "lower_component"
    if center_x is None:
        center_x = float(_pick_dark_column_center(source, probe_box))
        component_width = None
        source_mode = "dark_column"

    target_width = target_box.width
    if component_width is None:
        stud_width = max(16, int(round(target_width * float(stud_width_ratio))))
    else:
        width_cap = max(0.18, float(stud_width_ratio))
        stud_width = max(
            16,
            min(
                int(round(target_width * width_cap)),
                int(round(max(component_width, target_width * 0.18))),
            ),
        )

    source_x1 = max(0, int(round(center_x - stud_width / 2.0)))
    source_x2 = min(source.shape[1], source_x1 + stud_width)
    source_x1 = max(0, source_x2 - stud_width)
    source_box = Box(source_x1, probe_box.y1, source_x2, probe_box.y2)

    texture = source[source_box.y1:source_box.y2, source_box.x1:source_box.x2]
    if texture.size == 0:
        raise ValueError("thread source texture is empty")
    texture_gate_triggered, texture_gray_mean, texture_gray_std = _should_gate_texture(texture)

    placement_y1 = visible_box.y1
    placement_y2 = visible_box.y2

    placement_box = Box(
        max(0, int(round(center_x - stud_width / 2.0))),
        placement_y1,
        0,
        placement_y2,
    )
    placement_x2 = min(source.shape[1], placement_box.x1 + stud_width)
    placement_x1 = max(0, placement_x2 - stud_width)
    placement_box = Box(placement_x1, placement_y1, placement_x2, placement_y2)

    repeats = int(np.ceil(placement_box.height / max(1, texture.shape[0])))
    tiled = np.tile(texture, (repeats, 1, 1))[: placement_box.height, :, :]

    visible_fade_rows = max(6, int(round(placement_box.height * float(vertical_fade_ratio))))
    alpha = _build_segment_alpha(
        stud_width,
        placement_box.height,
        blur_radius=capsule_blur_radius,
        top_fade_ratio=vertical_fade_ratio,
    )
    inpaint_mask = (mask_bin * 255).astype(np.uint8)
    background_bgr = cv2.inpaint(
        cv2.cvtColor(source, cv2.COLOR_RGB2BGR),
        inpaint_mask,
        max(1, int(inpaint_radius)),
        cv2.INPAINT_TELEA,
    )
    background = cv2.cvtColor(background_bgr, cv2.COLOR_BGR2RGB)

    if texture_gate_triggered:
        merged = Image.fromarray(background, mode="RGB")
    else:
        donor = Image.fromarray(tiled, mode="RGB").convert("RGBA")
        donor.putalpha(alpha)
        result = Image.fromarray(background, mode="RGB").convert("RGBA")
        layer = Image.new("RGBA", result.size, (0, 0, 0, 0))
        layer.alpha_composite(donor, (placement_box.x1, placement_box.y1))
        merged = Image.alpha_composite(result, layer).convert("RGB")

    debug = ThreadCapsuleDebug(
        target_box=target_box,
        probe_box=probe_box,
        source_box=source_box,
        visible_box=visible_box,
        placement_box=placement_box,
        stud_width=stud_width,
        center_x=int(round(center_x)),
        source_mode=source_mode,
        visible_top=max(0, placement_box.y1 - target_box.y1),
        visible_fade_rows=visible_fade_rows,
        texture_gray_mean=texture_gray_mean,
        texture_gray_std=texture_gray_std,
        texture_gate_triggered=texture_gate_triggered,
    )
    return np.asarray(merged, dtype=np.uint8), debug
