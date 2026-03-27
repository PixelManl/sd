from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bolt.generate.donor_paste import Box, alpha_bbox, crop_box_from_mask_array, fit_content_box


class DonorPasteTests(unittest.TestCase):
    def test_alpha_bbox_raises_for_empty_mask(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            alpha_bbox(Image.new("L", (16, 16), 0))

    def test_crop_box_from_mask_array_matches_nonzero_extent(self) -> None:
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[4:11, 7:13] = 255
        box = crop_box_from_mask_array(mask)
        self.assertEqual(box, Box(7, 4, 13, 11))

    def test_fit_content_box_centers_and_scales_to_target_width_ratio(self) -> None:
        source = Box(12, 12, 236, 198)
        target = Box(347, 312, 678, 713)
        placed = fit_content_box(source, target, width_ratio=0.72, top_offset=6)
        self.assertEqual(placed.width, int(round(target.width * 0.72)))
        self.assertEqual(placed.y1, target.y1 + 6)
        self.assertAlmostEqual((placed.x1 + placed.x2) / 2.0, (target.x1 + target.x2) / 2.0, delta=1.0)


if __name__ == "__main__":
    unittest.main()
