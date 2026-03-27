from __future__ import annotations

import sys
import unittest
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bolt.generate.donor_paste import Box
from bolt.generate.missing_donor_patch import build_feather_alpha, expand_box, transfer_rect_patch


class MissingDonorPatchTests(unittest.TestCase):
    def test_expand_box_clamps_to_image_bounds(self) -> None:
        expanded = expand_box(Box(10, 12, 20, 25), padding=8, image_size=(24, 30))
        self.assertEqual(expanded, Box(2, 4, 24, 30))

    def test_build_feather_alpha_returns_requested_size(self) -> None:
        alpha = build_feather_alpha((32, 20), feather_radius=6.0)
        self.assertEqual(alpha.size, (32, 20))

    def test_transfer_rect_patch_keeps_image_size(self) -> None:
        image = Image.new("RGB", (40, 40), (255, 255, 255))
        for x in range(20, 30):
            for y in range(5, 15):
                image.putpixel((x, y), (0, 0, 0))
        output = transfer_rect_patch(
            image,
            donor_rect=Box(20, 5, 30, 15),
            target_rect=Box(5, 20, 15, 30),
            feather_radius=0.0,
        )
        self.assertEqual(output.size, image.size)
        self.assertEqual(output.getpixel((10, 25)), (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
