from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a queued set of SDXL experiment tasks sequentially."
    )
    parser.add_argument("--queue-manifest", required=True)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_queue(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_uv_launcher() -> list[str]:
    override = os.environ.get("UV_EXECUTABLE")
    if override:
        return [override, "run", "python"]

    which_uv = shutil.which("uv")
    if which_uv:
        return [which_uv, "run", "python"]

    roaming_uv = (
        Path.home() / "AppData" / "Roaming" / "Python" / f"Python{sys.version_info.major}{sys.version_info.minor}" / "Scripts" / "uv.exe"
    )
    if roaming_uv.exists():
        return [str(roaming_uv), "run", "python"]

    return [sys.executable, "-m", "uv", "run", "python"]


def build_subprocess_command(task: dict[str, object]) -> list[str]:
    if "command" in task:
        return [str(v) for v in task["command"]]

    args = dict(task["args"])
    command = [
        *resolve_uv_launcher(),
        "bolt/generate/scripts/run_sdxl_oriented_batch.py",
        "--batch-manifest",
        str(args["batch_manifest"]),
        "--image-dir",
        str(args["image_dir"]),
        "--core-mask-dir",
        str(args["core_mask_dir"]),
        "--output-dir",
        str(args["output_dir"]),
        "--base-model",
        str(args["base_model"]),
        "--steps",
        str(args["steps"]),
        "--guidance-scale",
        str(args["guidance_scale"]),
        "--strength",
        str(args["strength"]),
        "--target-size",
        str(args["target_size"]),
        "--seed-base",
        str(args["seed_base"]),
        "--limit",
        str(args["limit"]),
        "--adaptive-target-occupancy",
        str(args["adaptive_target_occupancy"]),
        "--adaptive-min-side",
        str(args["adaptive_min_side"]),
    ]
    mask_mode = args.get("mask_mode")
    if mask_mode:
        command.extend(["--mask-mode", str(mask_mode)])
    geometry_prior = args.get("geometry_prior")
    if geometry_prior:
        command.extend(["--geometry-prior", str(geometry_prior)])
    for image_name in args.get("image_names", []):
        command.extend(["--image-name", str(image_name)])
    return command


def _timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def run_queue(
    queue_path: Path,
    *,
    log_path: Path,
    dry_run: bool,
) -> dict[str, object]:
    queue = load_queue(queue_path)
    records: list[dict[str, object]] = []

    for task in queue["tasks"]:
        command = build_subprocess_command(task)
        entry = {
            "task_name": str(task["task_name"]),
            "stage": str(task.get("stage", "")),
            "command": command,
            "started_at": _timestamp(),
        }

        if dry_run:
            entry["status"] = "dry-run"
            entry["returncode"] = 0
            entry["finished_at"] = _timestamp()
            records.append(entry)
            continue

        completed = subprocess.run(
            command,
            cwd=str(Path.cwd()),
            capture_output=True,
            text=True,
        )
        entry["returncode"] = int(completed.returncode)
        entry["stdout"] = completed.stdout
        entry["stderr"] = completed.stderr
        entry["finished_at"] = _timestamp()
        entry["status"] = "completed" if completed.returncode == 0 else "failed"
        records.append(entry)

        if completed.returncode != 0:
            summary = {
                "queue_name": queue.get("queue_name", ""),
                "status": "failed",
                "executed_tasks": len(records),
                "failed_task": task["task_name"],
                "records": records,
            }
            log_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            return summary

    summary = {
        "queue_name": queue.get("queue_name", ""),
        "status": "completed",
        "executed_tasks": len(records),
        "failed_task": "",
        "records": records,
    }
    log_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    summary = run_queue(
        Path(args.queue_manifest),
        log_path=Path(args.log_path),
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
