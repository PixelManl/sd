from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the local-only workspace layout for the bolt mainline."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/bolt"),
        help="Local-only root directory for bolt datasets and generated assets.",
    )
    parser.add_argument(
        "--round-tag",
        default="2026-03-27",
        help="Merge tag for the next detector training round.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the layout plan without creating directories.",
    )
    return parser.parse_args()


def build_layout(root: Path, round_tag: str) -> dict[str, list[str]]:
    merged_round = round_tag.replace("-", "")
    return {
        "seed_source": [
            "source/seed_round/images",
            "source/seed_round/annotations",
            "source/seed_round/metadata",
        ],
        "sdxl_flow": [
            "generate/sdxl/incoming/images",
            "generate/sdxl/incoming/annotations",
            "generate/sdxl/repaired/images",
            "generate/sdxl/repaired/annotations",
            "generate/sdxl/review",
            "generate/sdxl/accepted/images",
            "generate/sdxl/accepted/annotations",
        ],
        "detect_flow": [
            "detect/current/images",
            "detect/current/annotations",
            f"detect/merged_{merged_round}/images",
            f"detect/merged_{merged_round}/annotations",
            "detect/metadata",
        ],
        "ascend_flow": [
            "ascend/background/incoming",
            "ascend/background/accepted",
        ],
    }


def materialize_layout(root: Path, layout: dict[str, list[str]], dry_run: bool) -> list[str]:
    created: list[str] = []
    for paths in layout.values():
        for relative in paths:
            target = root / relative
            created.append(str(target.resolve()))
            if not dry_run:
                target.mkdir(parents=True, exist_ok=True)
    return created


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    layout = build_layout(root, args.round_tag)
    created = materialize_layout(root, layout, args.dry_run)
    payload = {
        "task": "bootstrap_local_workspace",
        "status": "dry-run" if args.dry_run else "executed",
        "root": str(root),
        "round_tag": args.round_tag,
        "layout": layout,
        "path_count": len(created),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
