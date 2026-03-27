from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wait for a SAM2 completion signal, then launch the queued SDXL experiment runner."
    )
    parser.add_argument("--queue-manifest", required=True)
    parser.add_argument("--queue-log-path", required=True)
    parser.add_argument("--watcher-log-path", required=True)
    trigger = parser.add_mutually_exclusive_group(required=True)
    trigger.add_argument("--marker-path")
    trigger.add_argument("--wait-pid", type=int)
    trigger.add_argument("--wait-command-pattern")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--max-wait-seconds", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def build_queue_runner_command(
    *,
    queue_manifest: Path,
    queue_log_path: Path,
    dry_run: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "uv",
        "run",
        "python",
        "bolt/generate/scripts/run_sdxl_experiment_queue.py",
        "--queue-manifest",
        str(queue_manifest),
        "--log-path",
        str(queue_log_path),
    ]
    if dry_run:
        command.append("--dry-run")
    return command


def is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def list_process_snapshots() -> list[dict[str, object]]:
    if os.name == "nt":
        command = [
            "powershell",
            "-NoProfile",
            "-Command",
            (
                "Get-CimInstance Win32_Process | "
                "Select-Object ProcessId,CommandLine | "
                "ConvertTo-Json -Compress"
            ),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=True)
        payload = completed.stdout.strip()
        if not payload:
            return []
        rows = json.loads(payload)
        if isinstance(rows, dict):
            rows = [rows]
        return [
            {
                "pid": int(row.get("ProcessId", 0)),
                "command": str(row.get("CommandLine") or ""),
            }
            for row in rows
        ]

    completed = subprocess.run(
        ["ps", "-eo", "pid=,args="],
        capture_output=True,
        text=True,
        check=True,
    )
    snapshots: list[dict[str, object]] = []
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_text, _, command = stripped.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        snapshots.append({"pid": pid, "command": command.strip()})
    return snapshots


def _resolve_trigger(
    *,
    marker_path: Path | None,
    wait_pid: int | None,
    wait_command_pattern: str | None,
    pid_alive_fn: Callable[[int], bool],
    process_snapshot_fn: Callable[[], list[dict[str, object]]],
) -> tuple[bool, dict[str, object]]:
    if marker_path is not None:
        ready = marker_path.exists()
        return ready, {"mode": "marker", "marker_path": str(marker_path)}

    if wait_pid is not None:
        running = pid_alive_fn(wait_pid)
        return (not running), {"mode": "pid", "pid": wait_pid}

    if wait_command_pattern is None:
        raise ValueError("one trigger source is required")

    pattern = re.compile(wait_command_pattern)
    matched_pids: list[int] = []
    for row in process_snapshot_fn():
        command = str(row.get("command") or "")
        if pattern.search(command):
            matched_pids.append(int(row.get("pid", 0)))
    return (
        len(matched_pids) == 0,
        {
            "mode": "command-pattern",
            "pattern": wait_command_pattern,
            "matched_pids": matched_pids,
        },
    )


def write_summary(path: Path, summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def wait_then_run_queue(
    *,
    queue_manifest: Path,
    queue_log_path: Path,
    watcher_log_path: Path,
    marker_path: Path | None = None,
    wait_pid: int | None = None,
    wait_command_pattern: str | None = None,
    dry_run: bool = False,
    poll_seconds: float = 60.0,
    max_wait_seconds: float = 0.0,
    pid_alive_fn: Callable[[int], bool] = is_pid_alive,
    process_snapshot_fn: Callable[[], list[dict[str, object]]] = list_process_snapshots,
    sleep_fn: Callable[[float], None] = time.sleep,
    time_fn: Callable[[], float] = time.time,
) -> dict[str, object]:
    started_at = _timestamp()
    started_clock = time_fn()
    queue_command = build_queue_runner_command(
        queue_manifest=queue_manifest,
        queue_log_path=queue_log_path,
        dry_run=dry_run,
    )
    last_trigger: dict[str, object] = {}

    while True:
        ready, trigger_details = _resolve_trigger(
            marker_path=marker_path,
            wait_pid=wait_pid,
            wait_command_pattern=wait_command_pattern,
            pid_alive_fn=pid_alive_fn,
            process_snapshot_fn=process_snapshot_fn,
        )

        if trigger_details.get("matched_pids"):
            last_trigger = trigger_details
        elif not last_trigger:
            last_trigger = trigger_details
        else:
            merged = {**last_trigger, **trigger_details}
            if (
                last_trigger.get("mode") == "command-pattern"
                and "matched_pids" in last_trigger
                and not trigger_details.get("matched_pids")
            ):
                merged["matched_pids"] = last_trigger["matched_pids"]
            last_trigger = merged

        waited_seconds = max(0.0, time_fn() - started_clock)
        if ready:
            break

        if max_wait_seconds > 0 and waited_seconds >= max_wait_seconds:
            summary = {
                "status": "timeout",
                "started_at": started_at,
                "finished_at": _timestamp(),
                "waited_seconds": waited_seconds,
                "queue_launched": False,
                "queue_command": queue_command,
                "trigger": last_trigger,
            }
            write_summary(watcher_log_path, summary)
            return summary

        sleep_fn(poll_seconds)

    waited_seconds = max(0.0, time_fn() - started_clock)
    if dry_run:
        summary = {
            "status": "dry-run",
            "started_at": started_at,
            "finished_at": _timestamp(),
            "waited_seconds": waited_seconds,
            "queue_launched": False,
            "queue_command": queue_command,
            "trigger": last_trigger,
        }
        write_summary(watcher_log_path, summary)
        return summary

    completed = subprocess.run(
        queue_command,
        cwd=str(Path.cwd()),
        capture_output=True,
        text=True,
    )
    summary = {
        "status": "completed" if completed.returncode == 0 else "failed",
        "started_at": started_at,
        "finished_at": _timestamp(),
        "waited_seconds": waited_seconds,
        "queue_launched": True,
        "queue_command": queue_command,
        "trigger": last_trigger,
        "returncode": int(completed.returncode),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    write_summary(watcher_log_path, summary)
    return summary


def main() -> int:
    args = parse_args()
    summary = wait_then_run_queue(
        queue_manifest=Path(args.queue_manifest),
        queue_log_path=Path(args.queue_log_path),
        watcher_log_path=Path(args.watcher_log_path),
        marker_path=Path(args.marker_path) if args.marker_path else None,
        wait_pid=args.wait_pid,
        wait_command_pattern=args.wait_command_pattern,
        dry_run=args.dry_run,
        poll_seconds=args.poll_seconds,
        max_wait_seconds=args.max_wait_seconds,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] in {"completed", "dry-run"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
