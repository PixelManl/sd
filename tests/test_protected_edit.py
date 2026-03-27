import unittest

import numpy as np

from bolt.generate.protected_edit import (
    build_stud_keep_hard_mask,
    build_three_zone_masks,
    composite_generated_patch,
    sanitize_remove_mask,
)


class ProtectedEditTests(unittest.TestCase):
    def test_build_stud_keep_hard_mask_builds_centerline_protection(self) -> None:
        remove = np.zeros((41, 41), dtype=np.uint8)
        remove[14:27, 8:33] = 255

        protect = build_stud_keep_hard_mask(remove)

        self.assertGreater(int(protect.sum()), 0)
        self.assertEqual(int(protect[20, 20]), 255)
        self.assertEqual(int(protect[20, 3]), 0)

    def test_sanitize_remove_mask_clears_protected_overlap(self) -> None:
        remove = np.zeros((9, 9), dtype=np.uint8)
        protect = np.zeros((9, 9), dtype=np.uint8)
        remove[2:7, 2:7] = 255
        protect[4, 2:7] = 255

        sanitized = sanitize_remove_mask(remove, protect)

        self.assertEqual(int(sanitized[4, 4]), 0)
        self.assertEqual(int(sanitized[3, 4]), 255)

    def test_build_three_zone_masks_creates_context_and_paste_without_touching_keep_hard(self) -> None:
        remove = np.zeros((11, 11), dtype=np.uint8)
        protect = np.zeros((11, 11), dtype=np.uint8)
        remove[4:7, 4:7] = 255
        protect[5, 5] = 255

        zones = build_three_zone_masks(remove, protect_mask=protect, seam_px=1, context_px=2, blur_px=0)

        self.assertEqual(int(zones["keep_hard"][5, 5]), 255)
        self.assertEqual(int(zones["remove"][5, 5]), 0)
        self.assertEqual(int(zones["paste"][5, 5]), 0)
        self.assertGreater(int(zones["context"].sum()), 0)
        self.assertGreater(int(zones["paste"].sum()), int(zones["remove"].sum()))

    def test_composite_generated_patch_only_updates_masked_region(self) -> None:
        source = np.full((6, 6, 3), 10, dtype=np.uint8)
        generated = np.full((6, 6, 3), 200, dtype=np.uint8)
        paste = np.zeros((6, 6), dtype=np.uint8)
        paste[2:4, 2:4] = 255

        output = composite_generated_patch(source_rgb=source, generated_rgb=generated, paste_mask=paste)

        self.assertEqual(int(output[0, 0, 0]), 10)
        self.assertEqual(int(output[2, 2, 0]), 200)


if __name__ == "__main__":
    unittest.main()
