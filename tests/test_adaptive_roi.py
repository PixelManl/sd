from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bolt.generate.adaptive_roi import (
    compute_square_crop_box_from_bbox,
    compute_square_crop_box_from_mask,
)


class AdaptiveRoiTests(unittest.TestCase):
    def test_compute_square_crop_box_matches_target_occupancy(self) -> None:
        crop_box = compute_square_crop_box_from_bbox(
            [190, 160, 210, 240],
            image_width=400,
            image_height=400,
            target_occupancy=0.25,
            min_side=128,
        )

        self.assertEqual(crop_box, [40, 40, 360, 360])
        self.assertEqual(crop_box[2] - crop_box[0], crop_box[3] - crop_box[1])

    def test_compute_square_crop_box_clamps_to_image_edges(self) -> None:
        crop_box = compute_square_crop_box_from_bbox(
            [18, 24, 38, 104],
            image_width=180,
            image_height=180,
            target_occupancy=0.4,
            min_side=96,
        )

        self.assertEqual(crop_box[0], 0)
        self.assertEqual(crop_box[1], 0)
        self.assertEqual(crop_box[2] - crop_box[0], crop_box[3] - crop_box[1])
        self.assertLessEqual(crop_box[2], 180)
        self.assertLessEqual(crop_box[3], 180)

    def test_compute_square_crop_box_from_mask_returns_none_for_empty_mask(self) -> None:
        empty_mask = np.zeros((64, 64), dtype=np.uint8)
        self.assertIsNone(
            compute_square_crop_box_from_mask(
                empty_mask,
                image_width=64,
                image_height=64,
                target_occupancy=0.3,
                min_side=32,
            )
        )

    def test_compute_square_crop_box_from_mask_can_bias_toward_root_side(self) -> None:
        core_mask = np.zeros((400, 400), dtype=np.uint8)
        core_mask[160:240, 190:210] = 255

        crop_box = compute_square_crop_box_from_mask(
            core_mask,
            image_width=400,
            image_height=400,
            target_occupancy=0.25,
            min_side=128,
            root_bias=1.0,
        )

        self.assertEqual(crop_box, [40, 0, 360, 320])
        self.assertEqual(crop_box[2] - crop_box[0], crop_box[3] - crop_box[1])


if __name__ == "__main__":
    unittest.main()
