from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a thread-capsule nut-removal batch.")
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runner-python", default=sys.executable)
    parser.add_argument(
        "--single-runner",
        type=Path,
        default=REPO_ROOT / "bolt" / "generate" / "scripts" / "run_thread_capsule_single.py",
    )
    parser.add_argument("--inpaint-radius", type=int, default=5)
    parser.add_argument("--probe-height-ratio", type=float, default=0.20)
    parser.add_argument("--stud-width-ratio", type=float, default=0.22)
    parser.add_argument("--capsule-blur-radius", type=float, default=2.0)
    parser.add_argument("--tall-mask-min-height", type=int, default=220)
    parser.add_argument("--lower-anchor-extend-ratio", type=float, default=0.45)
    parser.add_argument("--lower-anchor-min-pixels", type=int, default=18)
    parser.add_argument("--vertical-fade-ratio", type=float, default=0.35)
    parser.add_argument("--preview-pad", type=int, default=140)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("manifest.records must be a list")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    has_failure = False

    for record in records:
        if not isinstance(record, dict):
            raise ValueError("manifest record must be an object")
        asset_id = str(record.get("asset_id") or "").strip()
        if not asset_id:
            raise ValueError("asset_id missing in manifest record")
        output_subdir = args.output_dir / asset_id
        output_subdir.mkdir(parents=True, exist_ok=True)
        cmd = [
            args.runner_python,
            str(args.single_runner.resolve()),
            "--image",
            str(record["image_path"]),
            "--target-mask",
            str(record["target_mask_path"]),
            "--output-dir",
            str(output_subdir.resolve()),
            "--inpaint-radius",
            str(args.inpaint_radius),
            "--probe-height-ratio",
            str(args.probe_height_ratio),
            "--stud-width-ratio",
            str(args.stud_width_ratio),
            "--capsule-blur-radius",
            str(args.capsule_blur_radius),
            "--tall-mask-min-height",
            str(args.tall_mask_min_height),
            "--lower-anchor-extend-ratio",
            str(args.lower_anchor_extend_ratio),
            "--lower-anchor-min-pixels",
            str(args.lower_anchor_min_pixels),
            "--vertical-fade-ratio",
            str(args.vertical_fade_ratio),
            "--preview-pad",
            str(args.preview_pad),
        ]
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if completed.returncode != 0:
            has_failure = True
        results.append(
            {
                "asset_id": asset_id,
                "returncode": completed.returncode,
                "preview_path": str((output_subdir / "preview_thread_capsule.png").resolve()),
                "stdout": completed.stdout[-2000:],
                "stderr": completed.stderr[-2000:],
            }
        )

    summary = {
        "task": "thread_capsule_batch",
        "status": "completed_with_errors" if has_failure else "completed",
        "manifest_path": str(args.manifest_path.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "record_count": len(results),
        "inpaint_radius": args.inpaint_radius,
        "probe_height_ratio": args.probe_height_ratio,
        "stud_width_ratio": args.stud_width_ratio,
        "capsule_blur_radius": args.capsule_blur_radius,
        "tall_mask_min_height": args.tall_mask_min_height,
        "lower_anchor_extend_ratio": args.lower_anchor_extend_ratio,
        "lower_anchor_min_pixels": args.lower_anchor_min_pixels,
        "vertical_fade_ratio": args.vertical_fade_ratio,
        "preview_pad": args.preview_pad,
        "records": results,
    }
    (args.output_dir / "run_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not has_failure else 1


if __name__ == "__main__":
    raise SystemExit(main())
