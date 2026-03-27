from __future__ import annotations

from PIL import Image, ImageFilter

from bolt.generate.donor_paste import Box


def expand_box(box: Box, *, padding: int, image_size: tuple[int, int]) -> Box:
    width, height = image_size
    return Box(
        max(0, box.x1 - padding),
        max(0, box.y1 - padding),
        min(width, box.x2 + padding),
        min(height, box.y2 + padding),
    )


def build_feather_alpha(size: tuple[int, int], *, feather_radius: float) -> Image.Image:
    alpha = Image.new("L", size, 255)
    if feather_radius > 0:
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=feather_radius))
    return alpha


def transfer_rect_patch(
    image_rgb: Image.Image,
    *,
    donor_rect: Box,
    target_rect: Box,
    feather_radius: float,
) -> Image.Image:
    base = image_rgb.convert("RGBA")
    donor_patch = image_rgb.crop((donor_rect.x1, donor_rect.y1, donor_rect.x2, donor_rect.y2))
    donor_patch = donor_patch.resize((target_rect.width, target_rect.height), Image.Resampling.LANCZOS)
    donor_rgba = donor_patch.convert("RGBA")
    donor_rgba.putalpha(build_feather_alpha(donor_patch.size, feather_radius=feather_radius))

    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    layer.alpha_composite(donor_rgba, (target_rect.x1, target_rect.y1))
    return Image.alpha_composite(base, layer).convert("RGB")


def transfer_mask_patch(
    image_rgb: Image.Image,
    *,
    donor_rect: Box,
    target_rect: Box,
    target_alpha: Image.Image,
    feather_radius: float,
) -> Image.Image:
    if target_alpha.size != (target_rect.width, target_rect.height):
        raise ValueError("target_alpha size must match target_rect size")

    base = image_rgb.convert("RGBA")
    donor_patch = image_rgb.crop((donor_rect.x1, donor_rect.y1, donor_rect.x2, donor_rect.y2))
    donor_patch = donor_patch.resize((target_rect.width, target_rect.height), Image.Resampling.LANCZOS)
    donor_rgba = donor_patch.convert("RGBA")

    alpha = target_alpha.convert("L")
    if feather_radius > 0:
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=feather_radius))
    donor_rgba.putalpha(alpha)

    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    layer.alpha_composite(donor_rgba, (target_rect.x1, target_rect.y1))
    return Image.alpha_composite(base, layer).convert("RGB")
