from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bolt.generate.geometry_prior import (
    build_stud_geometry_prior,
    seed_roi_with_geometry_prior,
)


class GeometryPriorTests(unittest.TestCase):
    def test_build_stud_geometry_prior_points_axis_downward_and_preserves_tail(self) -> None:
        core_mask = np.zeros((128, 128), dtype=np.uint8)
        core_mask[30:94, 60:68] = 255

        prior = build_stud_geometry_prior(
            core_mask,
            prior_mode="envelope",
            nut_width_ratio=2.4,
            tail_keep_ratio=0.25,
            blur_ksize=0,
        )

        self.assertIsNotNone(prior)
        assert prior is not None
        self.assertGreater(prior.axis_vector[1], 0.0)
        self.assertGreater(int(np.count_nonzero(prior.envelope_mask)), 0)
        self.assertGreater(int(np.count_nonzero(prior.axis_mask)), 0)
        self.assertGreater(int(np.count_nonzero(prior.preserve_tail_mask)), 0)
        self.assertGreater(int(np.count_nonzero(prior.envelope_mask[32:48, :])), 0)
        self.assertEqual(int(np.count_nonzero(prior.envelope_mask[90:98, :])), 0)
        self.assertGreater(int(np.count_nonzero(prior.preserve_tail_mask[84:98, :])), 0)

    def test_seed_roi_with_geometry_prior_only_changes_masked_area(self) -> None:
        roi_rgb = np.full((64, 64, 3), 100, dtype=np.uint8)
        envelope_mask = np.zeros((64, 64), dtype=np.uint8)
        envelope_mask[20:40, 28:36] = 255

        seeded = seed_roi_with_geometry_prior(
            roi_rgb,
            envelope_mask,
            fill_rgb=(180, 180, 180),
            alpha=0.5,
        )

        self.assertEqual(int(seeded[8, 8, 0]), 100)
        self.assertGreater(int(seeded[30, 32, 0]), 100)


if __name__ == "__main__":
    unittest.main()
