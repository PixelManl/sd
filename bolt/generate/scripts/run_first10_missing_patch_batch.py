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
    parser = argparse.ArgumentParser(description="Run the strict first-10 same-image missing patch batch.")
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runner-python", default=sys.executable)
    parser.add_argument(
        "--single-runner",
        type=Path,
        default=REPO_ROOT / "bolt" / "generate" / "scripts" / "run_same_image_missing_donor_patch.py",
    )
    parser.add_argument("--donor-pad", type=int, default=24)
    parser.add_argument("--target-pad", type=int, default=32)
    parser.add_argument("--feather-radius", type=float, default=12.0)
    parser.add_argument("--preview-pad", type=int, default=180)
    parser.add_argument("--blend-mode", choices=["rect", "mask", "top_band"], default="rect")
    parser.add_argument("--mask-dilate", type=int, default=0)
    parser.add_argument("--preserve-bottom-ratio", type=float, default=0.45)
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
        donor_box = ",".join(str(v) for v in record.get("donor_box", []))
        cmd = [
            args.runner_python,
            str(args.single_runner.resolve()),
            "--image",
            str(record["image_path"]),
            "--target-mask",
            str(record["target_mask_path"]),
            "--donor-box",
            donor_box,
            "--output-dir",
            str(output_subdir.resolve()),
            "--donor-pad",
            str(args.donor_pad),
            "--target-pad",
            str(args.target_pad),
            "--feather-radius",
            str(args.feather_radius),
            "--preview-pad",
            str(args.preview_pad),
            "--blend-mode",
            args.blend_mode,
            "--mask-dilate",
            str(args.mask_dilate),
            "--preserve-bottom-ratio",
            str(args.preserve_bottom_ratio),
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
                "preview_path": str((output_subdir / "preview_same_image_missing_patch.png").resolve()),
                "stdout": completed.stdout[-2000:],
                "stderr": completed.stderr[-2000:],
            }
        )

    summary = {
        "task": "first10_same_image_missing_patch_batch",
        "status": "completed_with_errors" if has_failure else "completed",
        "manifest_path": str(args.manifest_path.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "record_count": len(results),
        "donor_pad": args.donor_pad,
        "target_pad": args.target_pad,
        "feather_radius": args.feather_radius,
        "preview_pad": args.preview_pad,
        "blend_mode": args.blend_mode,
        "mask_dilate": args.mask_dilate,
        "preserve_bottom_ratio": args.preserve_bottom_ratio,
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
