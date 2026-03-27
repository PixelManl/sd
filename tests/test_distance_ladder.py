from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bolt.generate.distance_ladder import (
    build_edit_mask,
    expand_crop_box,
    mask_bbox,
    parse_distance_variants,
    resolve_base_crop_box,
)


class DistanceLadderTests(unittest.TestCase):
    def test_expand_crop_box_scales_around_center_and_clamps(self) -> None:
        expanded = expand_crop_box(
            [100, 120, 200, 220],
            image_width=260,
            image_height=240,
            crop_scale=1.75,
        )

        self.assertEqual(expanded, [62, 82, 238, 240])

    def test_build_edit_mask_grows_when_dilate_ratio_increases(self) -> None:
        core_mask = np.zeros((120, 120), dtype=np.uint8)
        core_mask[50:70, 54:66] = 255

        small_mask, small_box = build_edit_mask(
            core_mask,
            dilate_ratio=0.18,
            min_pad=8,
            blur_ksize=0,
        )
        large_mask, large_box = build_edit_mask(
            core_mask,
            dilate_ratio=0.42,
            min_pad=8,
            blur_ksize=0,
        )

        self.assertEqual(small_mask.shape, core_mask.shape)
        self.assertEqual(large_mask.shape, core_mask.shape)
        self.assertGreater(int(np.count_nonzero(large_mask)), int(np.count_nonzero(small_mask)))
        self.assertGreaterEqual(small_box[0], large_box[0])
        self.assertGreaterEqual(small_box[1], large_box[1])
        self.assertGreaterEqual(small_box[2], small_box[0])
        self.assertGreaterEqual(large_box[2], large_box[0])
        self.assertLess(large_box[0], small_box[0])
        self.assertLess(large_box[1], small_box[1])
        self.assertGreater(large_box[2], small_box[2])
        self.assertGreater(large_box[3], small_box[3])

    def test_mask_bbox_returns_none_for_empty_mask(self) -> None:
        empty_mask = np.zeros((64, 64), dtype=np.uint8)
        self.assertIsNone(mask_bbox(empty_mask))

    def test_parse_distance_variants_supports_custom_values(self) -> None:
        variants = parse_distance_variants(["d18:0.18", "d34:0.34"])
        self.assertEqual([variant.name for variant in variants], ["d18", "d34"])
        self.assertEqual([variant.dilate_ratio for variant in variants], [0.18, 0.34])

    def test_resolve_base_crop_box_supports_two_manifest_shapes(self) -> None:
        self.assertEqual(
            resolve_base_crop_box({"crop_box": [1, 2, 3, 4]}),
            [1, 2, 3, 4],
        )
        self.assertEqual(
            resolve_base_crop_box({"base_crop_box": [5, 6, 7, 8]}),
            [5, 6, 7, 8],
        )


if __name__ == "__main__":
    unittest.main()
