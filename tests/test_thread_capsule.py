from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bolt.generate.donor_paste import Box
from bolt.generate.thread_capsule import (
    _build_segment_alpha,
    _resolve_visible_segment_box,
    repair_mask_with_thread_capsule,
)


class ThreadCapsuleTests(unittest.TestCase):
    def test_resolve_visible_segment_box_keeps_small_target_full_height(self) -> None:
        target = Box(10, 20, 30, 60)
        segment = _resolve_visible_segment_box(
            target,
            full_height_threshold=72,
            visible_height_ratio=0.42,
            min_visible_height=32,
        )
        self.assertEqual(segment, target)

    def test_resolve_visible_segment_box_bottom_anchors_tall_target(self) -> None:
        target = Box(100, 200, 160, 620)
        segment = _resolve_visible_segment_box(
            target,
            full_height_threshold=72,
            visible_height_ratio=0.42,
            min_visible_height=32,
        )
        self.assertEqual(segment.x1, target.x1)
        self.assertEqual(segment.x2, target.x2)
        self.assertEqual(segment.y2, target.y2)
        self.assertGreater(segment.y1, target.y1)
        self.assertEqual(segment.height, 176)

    def test_build_segment_alpha_fades_in_from_top(self) -> None:
        alpha = _build_segment_alpha(width=32, height=120, blur_radius=0.0, top_fade_ratio=0.30)
        alpha_np = np.asarray(alpha, dtype=np.uint8)
        self.assertLess(int(alpha_np[0, 16]), 10)
        self.assertGreater(int(alpha_np[-1, 16]), 200)

    def test_repair_mask_with_thread_capsule_gates_bright_flat_texture(self) -> None:
        image = np.full((120, 120, 3), 96, dtype=np.uint8)
        image[94:118, 50:70] = 245

        mask = np.zeros((120, 120), dtype=np.uint8)
        mask[20:28, 50:70] = 255
        mask[60:92, 54:66] = 255

        output, debug = repair_mask_with_thread_capsule(
            image,
            mask,
            inpaint_radius=3,
            probe_height_ratio=0.25,
            stud_width_ratio=0.24,
            capsule_blur_radius=0.0,
        )

        self.assertTrue(debug.texture_gate_triggered)
        self.assertLess(int(output[70, 60, 0]), 140)
        self.assertLess(int(output[70, 60, 1]), 140)
        self.assertLess(int(output[70, 60, 2]), 140)


if __name__ == "__main__":
    unittest.main()
