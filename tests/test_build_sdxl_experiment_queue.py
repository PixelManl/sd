from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bolt.generate.scripts.build_sdxl_experiment_queue import (
    build_default_queue,
    write_queue_package,
)


class BuildSdxlExperimentQueueTests(unittest.TestCase):
    def test_build_default_queue_emits_targeted_structure_probe_tasks(self) -> None:
        queue = build_default_queue(
            batch_manifest="runs/base_manifest.json",
            image_dir="data/images",
            core_mask_dir="data/masks",
            output_root="runs/queue",
            image_names=["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg"],
        )

        task_names = [task["task_name"] for task in queue["tasks"]]
        self.assertEqual(len(task_names), 8)
        self.assertEqual(
            task_names,
            [
                "exp_a_oriented_axis",
                "exp_a_oriented_envelope",
                "exp_a_root_axis",
                "exp_a_root_envelope",
                "exp_b_oriented_axis",
                "exp_b_oriented_envelope",
                "exp_b_root_axis",
                "exp_b_root_envelope",
            ],
        )

    def test_build_default_queue_sets_structure_constrained_variants(self) -> None:
        queue = build_default_queue(
            batch_manifest="runs/base_manifest.json",
            image_dir="data/images",
            core_mask_dir="data/masks",
            output_root="runs/queue",
            image_names=["only.jpg"],
        )

        first = queue["tasks"][0]
        self.assertEqual(first["args"]["mask_mode"], "oriented")
        self.assertEqual(first["args"]["geometry_prior"], "axis")
        self.assertEqual(first["args"]["adaptive_target_occupancy"], 0.22)
        self.assertEqual(first["args"]["strength"], 0.92)

        root_envelope = queue["tasks"][3]
        self.assertEqual(root_envelope["args"]["mask_mode"], "root_contact")
        self.assertEqual(root_envelope["args"]["geometry_prior"], "envelope")
        self.assertEqual(root_envelope["args"]["adaptive_target_occupancy"], 0.22)

        higher_probe = queue["tasks"][7]
        self.assertEqual(higher_probe["args"]["mask_mode"], "root_contact")
        self.assertEqual(higher_probe["args"]["geometry_prior"], "envelope")
        self.assertEqual(higher_probe["args"]["adaptive_target_occupancy"], 0.24)
        self.assertEqual(higher_probe["args"]["strength"], 0.96)
        self.assertEqual(higher_probe["args"]["limit"], 0)

    def test_write_queue_package_creates_manifest_and_shell_wrappers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            queue = build_default_queue(
                batch_manifest="runs/base_manifest.json",
                image_dir="data/images",
                core_mask_dir="data/masks",
                output_root="runs/queue",
                image_names=["a.jpg", "b.jpg"],
            )

            package = write_queue_package(queue, output_dir)

            manifest_path = Path(package["manifest_path"])
            sh_path = Path(package["bash_path"])
            ps1_path = Path(package["powershell_path"])

            self.assertTrue(manifest_path.exists())
            self.assertTrue(sh_path.exists())
            self.assertTrue(ps1_path.exists())

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["tasks"]), 8)
            self.assertEqual(manifest["tasks"][0]["args"]["mask_mode"], "oriented")
            self.assertEqual(manifest["tasks"][2]["args"]["mask_mode"], "root_contact")
            self.assertIn("run_sdxl_oriented_batch.py", sh_path.read_text(encoding="utf-8"))
            self.assertIn("run_sdxl_oriented_batch.py", ps1_path.read_text(encoding="utf-8"))
            self.assertIn("--mask-mode root_contact", sh_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
