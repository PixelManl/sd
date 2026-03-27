from __future__ import annotations

import math
import unittest

import cv2
import numpy as np

from bolt.generate.mask_geometry import (
    build_oriented_focus_mask,
    build_root_contact_focus_mask,
    fit_mask_geometry,
)


def draw_rotated_mask(
    height: int,
    width: int,
    *,
    center: tuple[float, float],
    size: tuple[float, float],
    angle_deg: float,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    rect = (center, size, angle_deg)
    points = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillConvexPoly(mask, points, 255)
    return mask


class GenerateMaskGeometryTests(unittest.TestCase):
    def test_build_oriented_focus_mask_preserves_orientation_and_changes_aspect(self) -> None:
        core_mask = draw_rotated_mask(
            160,
            160,
            center=(80, 80),
            size=(18, 72),
            angle_deg=38,
        )

        core_geometry = fit_mask_geometry(core_mask)
        self.assertIsNotNone(core_geometry)

        focus_mask = build_oriented_focus_mask(
            core_mask,
            axial_scale=0.72,
            transverse_scale=1.85,
            min_axial_radius=8.0,
            min_transverse_radius=10.0,
            blur_ksize=0,
        )
        focus_geometry = fit_mask_geometry(focus_mask)
        self.assertIsNotNone(focus_geometry)

        assert core_geometry is not None
        assert focus_geometry is not None

        alignment = abs(
            core_geometry.major_axis[0] * focus_geometry.major_axis[0]
            + core_geometry.major_axis[1] * focus_geometry.major_axis[1]
        )
        self.assertGreater(alignment, 0.98)
        self.assertLess(focus_geometry.major_radius, core_geometry.major_radius)
        self.assertGreater(focus_geometry.minor_radius, core_geometry.minor_radius)

    def test_build_oriented_focus_mask_returns_zeros_for_empty_mask(self) -> None:
        empty_mask = np.zeros((64, 64), dtype=np.uint8)
        focus_mask = build_oriented_focus_mask(empty_mask, blur_ksize=0)
        self.assertEqual(int(focus_mask.max()), 0)
        self.assertIsNone(fit_mask_geometry(focus_mask))

    def test_build_oriented_focus_mask_can_shift_toward_contact_side(self) -> None:
        core_mask = draw_rotated_mask(
            160,
            160,
            center=(80, 80),
            size=(18, 72),
            angle_deg=90,
        )

        baseline_focus = build_oriented_focus_mask(
            core_mask,
            axial_scale=0.72,
            transverse_scale=1.85,
            min_axial_radius=8.0,
            min_transverse_radius=10.0,
            contact_bias=0.0,
            blur_ksize=0,
        )
        root_biased_focus = build_oriented_focus_mask(
            core_mask,
            axial_scale=0.72,
            transverse_scale=1.85,
            min_axial_radius=8.0,
            min_transverse_radius=10.0,
            contact_bias=0.75,
            blur_ksize=0,
        )

        baseline_geometry = fit_mask_geometry(baseline_focus)
        root_biased_geometry = fit_mask_geometry(root_biased_focus)
        core_geometry = fit_mask_geometry(core_mask)
        self.assertIsNotNone(baseline_geometry)
        self.assertIsNotNone(root_biased_geometry)
        self.assertIsNotNone(core_geometry)

        assert baseline_geometry is not None
        assert root_biased_geometry is not None
        assert core_geometry is not None

        direction = np.array(core_geometry.major_axis, dtype=np.float32)
        if abs(float(direction[1])) >= abs(float(direction[0])):
            if float(direction[1]) < 0.0:
                direction *= -1.0
        elif float(direction[0]) < 0.0:
            direction *= -1.0

        baseline_projection = float(np.dot(np.array(baseline_geometry.center), direction))
        root_biased_projection = float(np.dot(np.array(root_biased_geometry.center), direction))
        self.assertLess(root_biased_projection, baseline_projection)

    def test_build_root_contact_focus_mask_stays_near_contact_side(self) -> None:
        core_mask = draw_rotated_mask(
            160,
            160,
            center=(80, 80),
            size=(18, 72),
            angle_deg=90,
        )

        oriented_focus = build_oriented_focus_mask(
            core_mask,
            axial_scale=0.72,
            transverse_scale=1.85,
            min_axial_radius=8.0,
            min_transverse_radius=10.0,
            contact_bias=0.0,
            blur_ksize=0,
        )
        root_contact_focus = build_root_contact_focus_mask(
            core_mask,
            axial_scale=0.50,
            transverse_scale=2.45,
            min_axial_radius=8.0,
            min_transverse_radius=10.0,
            contact_bias=0.35,
            blur_ksize=0,
        )

        oriented_geometry = fit_mask_geometry(oriented_focus)
        root_contact_geometry = fit_mask_geometry(root_contact_focus)
        core_geometry = fit_mask_geometry(core_mask)
        self.assertIsNotNone(oriented_geometry)
        self.assertIsNotNone(root_contact_geometry)
        self.assertIsNotNone(core_geometry)

        assert oriented_geometry is not None
        assert root_contact_geometry is not None
        assert core_geometry is not None

        direction = np.array(core_geometry.major_axis, dtype=np.float32)
        if abs(float(direction[1])) >= abs(float(direction[0])):
            if float(direction[1]) < 0.0:
                direction *= -1.0
        elif float(direction[0]) < 0.0:
            direction *= -1.0

        oriented_projection = float(np.dot(np.array(oriented_geometry.center), direction))
        root_contact_projection = float(np.dot(np.array(root_contact_geometry.center), direction))

        self.assertLess(root_contact_projection, oriented_projection)


    def test_fit_mask_geometry_reports_expected_angle_for_diagonal_target(self) -> None:
        core_mask = draw_rotated_mask(
            128,
            128,
            center=(64, 64),
            size=(16, 64),
            angle_deg=45,
        )
        geometry = fit_mask_geometry(core_mask)
        self.assertIsNotNone(geometry)
        assert geometry is not None

        angle = math.degrees(math.atan2(geometry.major_axis[1], geometry.major_axis[0]))
        angle = abs(((angle + 90) % 180) - 90)
        self.assertGreater(angle, 35)
        self.assertLess(angle, 55)


if __name__ == "__main__":
    unittest.main()
