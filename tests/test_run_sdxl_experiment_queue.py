from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bolt.generate.scripts.run_sdxl_experiment_queue import (
    build_subprocess_command,
    run_queue,
)


class RunSdxlExperimentQueueTests(unittest.TestCase):
    def test_build_subprocess_command_includes_repeated_image_name_flags(self) -> None:
        task = {
            "task_name": "exp_a_oriented_axis",
            "args": {
                "batch_manifest": "manifest.json",
                "image_dir": "images",
                "core_mask_dir": "masks",
                "output_dir": "outputs",
                "base_model": "model-id",
                "image_names": ["a.jpg", "b.jpg"],
                "steps": 30,
                "guidance_scale": 6.0,
                "strength": 0.92,
                "target_size": 1024,
                "seed_base": 9000,
                "limit": 0,
                "adaptive_target_occupancy": 0.25,
                "adaptive_min_side": 256,
                "mask_mode": "oriented",
                "geometry_prior": "none",
            },
        }

        cmd = build_subprocess_command(task)

        self.assertIn("--image-name", cmd)
        self.assertEqual(cmd.count("--image-name"), 2)
        self.assertIn("a.jpg", cmd)
        self.assertIn("b.jpg", cmd)

    def test_build_subprocess_command_prefers_uv_executable_override(self) -> None:
        task = {
            "task_name": "exp_a_oriented_axis",
            "args": {
                "batch_manifest": "manifest.json",
                "image_dir": "images",
                "core_mask_dir": "masks",
                "output_dir": "outputs",
                "base_model": "model-id",
                "image_names": [],
                "steps": 30,
                "guidance_scale": 6.0,
                "strength": 0.92,
                "target_size": 1024,
                "seed_base": 9000,
                "limit": 0,
                "adaptive_target_occupancy": 0.25,
                "adaptive_min_side": 256,
                "mask_mode": "oriented",
                "geometry_prior": "none",
            },
        }

        with mock.patch.dict(os.environ, {"UV_EXECUTABLE": r"C:\tools\uv.exe"}, clear=False):
            cmd = build_subprocess_command(task)

        self.assertEqual(cmd[:3], [r"C:\tools\uv.exe", "run", "python"])
        self.assertNotIn("-m", cmd[:4])

    def test_build_subprocess_command_forwards_mask_mode_when_present(self) -> None:
        task = {
            "task_name": "exp_a_root_envelope",
            "args": {
                "batch_manifest": "manifest.json",
                "image_dir": "images",
                "core_mask_dir": "masks",
                "output_dir": "outputs",
                "base_model": "model-id",
                "image_names": [],
                "steps": 30,
                "guidance_scale": 6.0,
                "strength": 0.92,
                "target_size": 1024,
                "seed_base": 9000,
                "limit": 0,
                "adaptive_target_occupancy": 0.22,
                "adaptive_min_side": 256,
                "mask_mode": "root_contact",
                "geometry_prior": "envelope",
            },
        }

        cmd = build_subprocess_command(task)

        self.assertIn("--mask-mode", cmd)
        self.assertIn("root_contact", cmd)

    def test_run_queue_dry_run_writes_journal_without_executing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_path = Path(tmpdir) / "queue_manifest.json"
            log_path = Path(tmpdir) / "queue_run_log.json"
            queue_path.write_text(
                json.dumps(
                    {
                        "queue_name": "demo",
                        "tasks": [
                            {
                                "task_name": "exp_a_none",
                                "args": {
                                    "batch_manifest": "manifest.json",
                                    "image_dir": "images",
                                    "core_mask_dir": "masks",
                                    "output_dir": "outputs",
                                    "base_model": "model-id",
                                    "image_names": ["a.jpg"],
                                    "steps": 30,
                                    "guidance_scale": 6.0,
                                    "strength": 0.92,
                                    "target_size": 1024,
                                    "seed_base": 9000,
                                    "limit": 0,
                                    "adaptive_target_occupancy": 0.25,
                                    "adaptive_min_side": 256,
                                    "mask_mode": "oriented",
                                    "geometry_prior": "none",
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            summary = run_queue(queue_path, log_path=log_path, dry_run=True)

            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["executed_tasks"], 1)
            log_payload = json.loads(log_path.read_text(encoding="utf-8"))
            self.assertEqual(log_payload["records"][0]["status"], "dry-run")

    def test_run_queue_stops_on_first_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            ok_script = workspace / "ok.py"
            fail_script = workspace / "fail.py"
            ok_script.write_text("print('ok')\n", encoding="utf-8")
            fail_script.write_text("import sys\nsys.exit(3)\n", encoding="utf-8")

            queue_path = workspace / "queue_manifest.json"
            log_path = workspace / "queue_run_log.json"
            queue_path.write_text(
                json.dumps(
                    {
                        "queue_name": "demo",
                        "tasks": [
                            {"task_name": "ok", "command": [sys.executable, str(ok_script)]},
                            {"task_name": "fail", "command": [sys.executable, str(fail_script)]},
                            {"task_name": "skipped", "command": [sys.executable, str(ok_script)]},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            summary = run_queue(queue_path, log_path=log_path, dry_run=False)

            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["executed_tasks"], 2)
            self.assertEqual(summary["failed_task"], "fail")
            log_payload = json.loads(log_path.read_text(encoding="utf-8"))
            self.assertEqual([r["task_name"] for r in log_payload["records"]], ["ok", "fail"])


if __name__ == "__main__":
    unittest.main()
