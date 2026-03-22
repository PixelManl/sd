from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a local SAM2 pilot manifest from per-asset metadata JSON files."
    )
    parser.add_argument(
        "--metadata-root",
        required=True,
        help="Directory containing per-asset metadata JSON files.",
    )
    parser.add_argument(
        "--output",
        help="Optional output path. Defaults to stdout when omitted.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "jsonl"),
        default="json",
        help="Manifest output format.",
    )
    parser.add_argument(
        "--asset-id",
        help="Optional asset_id filter.",
    )
    parser.add_argument(
        "--qa-state",
        help="Optional qa_state filter.",
    )
    return parser.parse_args()


def fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 2


def validate_directory(path_text: str, label: str) -> Path:
    path = Path(path_text).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"{label} does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"{label} is not a directory: {path}")
    return path


def validate_output(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text).expanduser().resolve()
    if path.exists() and path.is_dir():
        raise ValueError(f"output must be a file path, not a directory: {path}")
    parent = path.parent
    if not parent.exists():
        raise ValueError(f"output parent does not exist: {parent}")
    return path


def load_records(metadata_root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(metadata_root.rglob("*.json")):
        if "manifests" in path.parts:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"metadata file must contain a JSON object: {path}")
        records.append(payload)
    return records


def filter_records(
    records: list[dict[str, object]],
    *,
    asset_id: str | None,
    qa_state: str | None,
) -> list[dict[str, object]]:
    filtered: list[dict[str, object]] = []
    for record in records:
        if asset_id and record.get("asset_id") != asset_id:
            continue
        if qa_state and record.get("qa_state") != qa_state:
            continue
        filtered.append(record)
    return filtered


def build_json_manifest(records: list[dict[str, object]], metadata_root: Path) -> dict[str, object]:
    return {
        "status": "placeholder",
        "mode": "sam2-manifest-export",
        "sam2_dependency_required": False,
        "contract_version": "sam2_asset_contract/v1",
        "asset_line": "sam2_pilot",
        "metadata_root": str(metadata_root),
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "record_count": len(records),
        "records": records,
    }


def emit_manifest_jsonl(records: list[dict[str, object]]) -> str:
    return "\n".join(json.dumps(record, ensure_ascii=False) for record in records)


def main() -> int:
    args = parse_args()

    try:
        metadata_root = validate_directory(args.metadata_root, "metadata root")
        output = validate_output(args.output)
        records = load_records(metadata_root)
    except ValueError as exc:
        return fail(str(exc))

    filtered = filter_records(
        records,
        asset_id=args.asset_id,
        qa_state=args.qa_state,
    )

    if args.format == "json":
        content = json.dumps(
            build_json_manifest(filtered, metadata_root),
            ensure_ascii=False,
            indent=2,
        )
    else:
        content = emit_manifest_jsonl(filtered)

    if output is None:
        print(content)
    else:
        output.write_text(content, encoding="utf-8")
        print(
            json.dumps(
                {
                    "status": "placeholder",
                    "mode": "sam2-manifest-export",
                    "output": str(output),
                    "format": args.format,
                    "record_count": len(filtered),
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
