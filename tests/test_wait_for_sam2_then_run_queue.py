from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bolt.generate.scripts.wait_for_sam2_then_run_queue import (
    build_queue_runner_command,
    wait_then_run_queue,
)


class WaitForSam2ThenRunQueueTests(unittest.TestCase):
    def test_marker_trigger_dry_run_writes_watcher_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            marker_path = workspace / "sam2.done"
            queue_manifest = workspace / "queue_manifest.json"
            queue_log = workspace / "queue_log.json"
            watcher_log = workspace / "watcher_log.json"
            marker_path.write_text("done\n", encoding="utf-8")
            queue_manifest.write_text('{"queue_name":"demo","tasks":[]}', encoding="utf-8")

            summary = wait_then_run_queue(
                queue_manifest=queue_manifest,
                queue_log_path=queue_log,
                watcher_log_path=watcher_log,
                marker_path=marker_path,
                dry_run=True,
                poll_seconds=0.01,
            )

            self.assertEqual(summary["status"], "dry-run")
            self.assertEqual(summary["trigger"]["mode"], "marker")
            self.assertIn("run_sdxl_experiment_queue.py", " ".join(summary["queue_command"]))
            watcher_payload = json.loads(watcher_log.read_text(encoding="utf-8"))
            self.assertEqual(watcher_payload["status"], "dry-run")

    def test_pid_trigger_waits_until_process_exits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            queue_manifest = workspace / "queue_manifest.json"
            queue_log = workspace / "queue_log.json"
            watcher_log = workspace / "watcher_log.json"
            queue_manifest.write_text('{"queue_name":"demo","tasks":[]}', encoding="utf-8")

            pid_states = iter([True, True, False])
            sleeps: list[float] = []

            summary = wait_then_run_queue(
                queue_manifest=queue_manifest,
                queue_log_path=queue_log,
                watcher_log_path=watcher_log,
                wait_pid=4321,
                dry_run=True,
                poll_seconds=3.0,
                pid_alive_fn=lambda pid: next(pid_states),
                sleep_fn=sleeps.append,
            )

            self.assertEqual(summary["status"], "dry-run")
            self.assertEqual(summary["trigger"]["mode"], "pid")
            self.assertEqual(summary["trigger"]["pid"], 4321)
            self.assertEqual(sleeps, [3.0, 3.0])

    def test_command_pattern_trigger_uses_process_snapshot_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            queue_manifest = workspace / "queue_manifest.json"
            queue_log = workspace / "queue_log.json"
            watcher_log = workspace / "watcher_log.json"
            queue_manifest.write_text('{"queue_name":"demo","tasks":[]}', encoding="utf-8")

            snapshots = iter(
                [
                    [{"pid": 9001, "command": "python bolt/mask/scripts/run_sam2_pilot.py --execute"}],
                    [],
                ]
            )

            summary = wait_then_run_queue(
                queue_manifest=queue_manifest,
                queue_log_path=queue_log,
                watcher_log_path=watcher_log,
                wait_command_pattern="run_sam2_pilot.py",
                dry_run=True,
                poll_seconds=5.0,
                process_snapshot_fn=lambda: next(snapshots),
                sleep_fn=lambda seconds: None,
            )

            self.assertEqual(summary["status"], "dry-run")
            self.assertEqual(summary["trigger"]["mode"], "command-pattern")
            self.assertEqual(summary["trigger"]["pattern"], "run_sam2_pilot.py")
            self.assertEqual(summary["trigger"]["matched_pids"], [9001])

    def test_timeout_returns_without_launching_queue(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            marker_path = workspace / "sam2.done"
            queue_manifest = workspace / "queue_manifest.json"
            queue_log = workspace / "queue_log.json"
            watcher_log = workspace / "watcher_log.json"
            queue_manifest.write_text('{"queue_name":"demo","tasks":[]}', encoding="utf-8")

            clock = {"now": 0.0}

            def fake_time() -> float:
                return clock["now"]

            def fake_sleep(seconds: float) -> None:
                clock["now"] += seconds

            summary = wait_then_run_queue(
                queue_manifest=queue_manifest,
                queue_log_path=queue_log,
                watcher_log_path=watcher_log,
                marker_path=marker_path,
                dry_run=True,
                poll_seconds=4.0,
                max_wait_seconds=9.0,
                time_fn=fake_time,
                sleep_fn=fake_sleep,
            )

            self.assertEqual(summary["status"], "timeout")
            self.assertEqual(summary["queue_launched"], False)
            watcher_payload = json.loads(watcher_log.read_text(encoding="utf-8"))
            self.assertEqual(watcher_payload["status"], "timeout")

    def test_build_queue_runner_command_keeps_uv_entrypoint(self) -> None:
        command = build_queue_runner_command(
            queue_manifest=Path("queue_manifest.json"),
            queue_log_path=Path("queue_log.json"),
            dry_run=False,
        )

        self.assertEqual(command[:5], [sys.executable, "-m", "uv", "run", "python"])
        self.assertIn("bolt/generate/scripts/run_sdxl_experiment_queue.py", command)
        self.assertNotIn("--dry-run", command)


if __name__ == "__main__":
    unittest.main()
