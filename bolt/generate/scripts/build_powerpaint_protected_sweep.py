from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bolt.generate.scripts.build_powerpaint_protected_manifest import build_protected_manifest


def parse_csv_floats(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_csv_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a protected-manifest sweep for PowerPaint.")
    parser.add_argument("--source-manifest-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seam-values", default="0,1,2")
    parser.add_argument("--keep-hard-width-values", default="0.16,0.22,0.28")
    parser.add_argument("--keep-hard-length-values", default="1.10,1.25")
    parser.add_argument("--context-ring-px", type=int, default=12)
    parser.add_argument("--strict-paste-blur-px", type=int, default=0)
    parser.add_argument("--limit", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seam_values = parse_csv_ints(str(args.seam_values))
    width_values = parse_csv_floats(str(args.keep_hard_width_values))
    length_values = parse_csv_floats(str(args.keep_hard_length_values))
    tasks: list[dict[str, object]] = []

    for length_scale in length_values:
        for width_scale in width_values:
            for seam_px in seam_values:
                variant_name = f"protect_l{length_scale:.2f}_w{width_scale:.2f}_s{seam_px}".replace(".", "p")
                output_dir = args.output_root / variant_name
                summary = build_protected_manifest(
                    source_manifest_path=args.source_manifest_path,
                    output_dir=output_dir,
                    variant_name=variant_name,
                    seam_px=seam_px,
                    context_ring_px=args.context_ring_px,
                    strict_paste_blur_px=args.strict_paste_blur_px,
                    keep_hard_length_scale=length_scale,
                    keep_hard_width_scale=width_scale,
                    limit=args.limit,
                )
                tasks.append(summary)

    payload = {
        "task": "powerpaint_protected_sweep",
        "source_manifest_path": str(args.source_manifest_path.resolve()),
        "task_count": len(tasks),
        "tasks": tasks,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "sweep_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
