import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


def load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = (
        repo_root / "bolt" / "mask" / "scripts" / "good_bolt_sam2_box_prompt_backend.py"
    )
    spec = importlib.util.spec_from_file_location("good_bolt_sam2_box_prompt_backend", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class GoodBoltSam2BoxPromptBackendTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_clamp_box_xyxy_keeps_valid_extent(self):
        box = self.module.clamp_box_xyxy([-5, 2, 40, 18], width=32, height=24)

        self.assertTrue(np.array_equal(box, np.asarray([0, 2, 32, 18], dtype=np.float32)))

    def test_expand_binary_mask_grows_foreground_region(self):
        mask = Image.new("L", (9, 9), 0)
        mask.putpixel((4, 4), 255)

        expanded = self.module.expand_binary_mask(mask, expand_px=1)

        self.assertEqual(expanded.getpixel((4, 4)), 255)
        self.assertEqual(expanded.getpixel((3, 4)), 255)
        self.assertEqual(expanded.getpixel((5, 4)), 255)
        self.assertEqual(expanded.getpixel((0, 0)), 0)

    def test_resolve_edit_dilate_px_scales_with_mask_size_and_respects_bounds(self):
        small = Image.new("L", (64, 64), 0)
        for x in range(20, 30):
            for y in range(18, 34):
                small.putpixel((x, y), 255)

        large = Image.new("L", (320, 320), 0)
        for x in range(60, 220):
            for y in range(40, 260):
                large.putpixel((x, y), 255)

        small_px = self.module.resolve_edit_dilate_px(small, min_px=18, ratio=0.10, max_px=64)
        large_px = self.module.resolve_edit_dilate_px(large, min_px=18, ratio=0.10, max_px=64)

        self.assertEqual(small_px, 18)
        self.assertGreater(large_px, small_px)
        self.assertLessEqual(large_px, 64)

    def test_resolve_edit_dilate_px_caps_at_max_px(self):
        huge = Image.new("L", (1024, 1024), 0)
        for x in range(200, 700):
            for y in range(100, 900):
                huge.putpixel((x, y), 255)

        resolved = self.module.resolve_edit_dilate_px(huge, min_px=18, ratio=0.25, max_px=64)

        self.assertEqual(resolved, 64)

    def test_build_directional_edit_mask_biases_toward_contact_side(self):
        mask = Image.new("L", (128, 128), 0)
        for x in range(58, 70):
            for y in range(30, 98):
                mask.putpixel((x, y), 255)

        directional = self.module.build_directional_edit_mask(mask, expand_px=24)
        isotropic = self.module.expand_binary_mask(mask, expand_px=24)

        def bbox(image):
            return self.module.mask_bbox(image)

        mx1, my1, mx2, my2 = bbox(mask)
        dx1, dy1, dx2, dy2 = bbox(directional)
        ix1, iy1, ix2, iy2 = bbox(isotropic)

        self.assertLess(dy1, my1)
        self.assertGreater(dy2, my2)
        self.assertLessEqual(dy1, iy1)
        directional_area = int(np.count_nonzero(np.asarray(directional.convert("L"))))
        isotropic_area = int(np.count_nonzero(np.asarray(isotropic.convert("L"))))
        self.assertLess(directional_area, isotropic_area)

    def test_mask_array_to_image_thresholds_boolean_foreground(self):
        array = np.asarray(
            [
                [0.0, 1.0],
                [-0.2, 3.5],
            ],
            dtype=np.float32,
        )

        image = self.module.mask_array_to_image(array)

        self.assertEqual(image.mode, "L")
        self.assertEqual(image.getpixel((0, 0)), 0)
        self.assertEqual(image.getpixel((1, 0)), 255)
        self.assertEqual(image.getpixel((0, 1)), 0)
        self.assertEqual(image.getpixel((1, 1)), 255)


if __name__ == "__main__":
    unittest.main()
