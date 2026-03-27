from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image

SUPPORTED_IMAGE_SUFFIXES = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
)

MODE_SYMBOLS = {
    "normal": "0",
    "screen": "4",
    "glow": "4",
    "overlay": "5",
    "addition": "7",
    "lighten": "10",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build GIMP XCF files from paired images and masks."
    )
    parser.add_argument("--image-dir", type=Path, required=True, help="Directory containing source images.")
    parser.add_argument("--mask-dir", type=Path, required=True, help="Directory containing mask files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write .xcf outputs into.")
    parser.add_argument(
        "--mask-suffix",
        default="_mask",
        help="Mask filename suffix before extension, for example sample_a_mask.png -> _mask.",
    )
    parser.add_argument(
        "--background-layer-name",
        default="background",
        help="Layer name for the source image layer inside the XCF.",
    )
    parser.add_argument(
        "--mask-layer-name",
        default="mask",
        help="Layer name for the mask layer inside the XCF.",
    )
    parser.add_argument(
        "--background-mode",
        choices=tuple(sorted(MODE_SYMBOLS)),
        default="lighten",
        help="Background layer blend mode inside the XCF.",
    )
    parser.add_argument(
        "--mask-mode",
        choices=tuple(sorted(MODE_SYMBOLS)),
        default="lighten",
        help="Top layer blend mode.",
    )
    parser.add_argument(
        "--background-opacity",
        type=float,
        default=100.0,
        help="Opacity percent for the background layer inside the XCF.",
    )
    parser.add_argument(
        "--mask-opacity",
        type=float,
        default=55.0,
        help="Opacity percent for the top mask layer inside the XCF.",
    )
    parser.add_argument(
        "--gimp-exe",
        type=Path,
        help="Optional path to the GIMP executable. If omitted, the script tries common Windows names and PATH.",
    )
    parser.add_argument(
        "--resize-mask-to-image",
        action="store_true",
        help="Resize mismatched masks to image size with nearest-neighbor before building the XCF.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .xcf files in the output directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only prepare the batch manifest and Scheme script without calling GIMP.",
    )
    return parser.parse_args()


def fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 2


def resolve_mode_symbol(mask_mode: str) -> str:
    try:
        return MODE_SYMBOLS[mask_mode]
    except KeyError as exc:
        raise ValueError(f"unsupported mask mode: {mask_mode!r}") from exc


def validate_dir(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise ValueError(f"{label} does not exist: {resolved}")
    if not resolved.is_dir():
        raise ValueError(f"{label} is not a directory: {resolved}")
    return resolved


def list_image_paths(image_dir: Path) -> list[Path]:
    return sorted(
        path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )


def find_mask_path(image_path: Path, mask_dir: Path, mask_suffix: str) -> Path:
    candidates: list[Path] = []
    for suffix in SUPPORTED_IMAGE_SUFFIXES:
        candidates.append(mask_dir / f"{image_path.stem}{mask_suffix}{suffix}")
    for suffix in SUPPORTED_IMAGE_SUFFIXES:
        candidates.append(mask_dir / f"{image_path.stem}{suffix}")

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    raise ValueError(f"mask not found for image: {image_path.name}")


def prepare_mask_path(
    image_path: Path,
    mask_path: Path,
    output_dir: Path,
    resize_mask_to_image: bool,
) -> tuple[Path, int, int]:
    with Image.open(image_path) as image:
        image_width, image_height = image.size
    with Image.open(mask_path) as mask_image:
        mask_width, mask_height = mask_image.size
        if (mask_width, mask_height) == (image_width, image_height):
            return mask_path.resolve(), image_width, image_height
        if not resize_mask_to_image:
            raise ValueError(
                "mask size mismatch for "
                f"{image_path.name}: image={image_width}x{image_height}, mask={mask_width}x{mask_height}"
            )
        staging_dir = output_dir / "_mask_stage"
        staging_dir.mkdir(parents=True, exist_ok=True)
        staged_mask_path = staging_dir / f"{image_path.stem}_mask_resized.png"
        resized = mask_image.convert("L").resize((image_width, image_height), resample=Image.NEAREST)
        resized.save(staged_mask_path)
        return staged_mask_path.resolve(), image_width, image_height


def collect_batch_jobs(
    image_dir: Path,
    mask_dir: Path,
    output_dir: Path,
    mask_suffix: str,
    overwrite: bool,
    resize_mask_to_image: bool,
) -> list[dict[str, object]]:
    image_dir = validate_dir(image_dir, "image dir")
    mask_dir = validate_dir(mask_dir, "mask dir")
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list_image_paths(image_dir)
    if not image_paths:
        raise ValueError(f"no supported images found in: {image_dir}")

    jobs: list[dict[str, object]] = []
    for image_path in image_paths:
        mask_path = find_mask_path(image_path, mask_dir, mask_suffix)
        prepared_mask_path, width, height = prepare_mask_path(
            image_path=image_path,
            mask_path=mask_path,
            output_dir=output_dir,
            resize_mask_to_image=resize_mask_to_image,
        )
        xcf_path = output_dir / f"{image_path.stem}.xcf"
        if xcf_path.exists() and not overwrite:
            raise ValueError(f"output already exists, pass --overwrite to replace it: {xcf_path}")
        jobs.append(
            {
                "stem": image_path.stem,
                "image_name": image_path.name,
                "mask_name": mask_path.name,
                "image_path": str(image_path.resolve()),
                "mask_path": str(prepared_mask_path),
                "xcf_path": str(xcf_path.resolve()),
                "width": width,
                "height": height,
            }
        )
    return jobs


def scheme_quote(text: str) -> str:
    return text.replace("\\", "/").replace("\"", "\\\"")


def build_scheme_content(
    jobs: list[dict[str, object]],
    background_layer_name: str,
    mask_layer_name: str,
    background_opacity: float,
    background_mode_symbol: str,
    mask_opacity: float,
    mask_mode_symbol: str,
) -> str:
    lines = [
        ";;; Auto-generated by bolt/scripts/build_gimp_xcf_batch.py",
        "",
        "(define (sd-build-one image-path mask-path output-path background-name mask-name background-opacity mask-opacity)",
        "  (script-fu-use-v3)",
        "  (let* (",
        "      (image (gimp-file-load RUN-NONINTERACTIVE image-path))",
        "      (background-layer (vector-ref (gimp-image-get-layers image) 0))",
        "      (mask-layer (gimp-file-load-layer RUN-NONINTERACTIVE image mask-path)))",
        "    (gimp-item-set-name background-layer background-name)",
        f"    (gimp-layer-set-mode background-layer {background_mode_symbol})",
        "    (gimp-layer-set-opacity background-layer background-opacity)",
        "    (gimp-image-insert-layer image mask-layer 0 0)",
        "    (gimp-item-set-name mask-layer mask-name)",
        f"    (gimp-layer-set-mode mask-layer {mask_mode_symbol})",
        "    (gimp-layer-set-opacity mask-layer mask-opacity)",
        "    (gimp-image-set-selected-layers image (vector mask-layer))",
        "    (gimp-file-save RUN-NONINTERACTIVE image output-path (vector mask-layer))",
        "    (gimp-image-delete image)))",
        "",
        "(define (sd-build-xcf-batch)",
    ]
    for job in jobs:
        lines.append(
            '  (sd-build-one "{image_path}" "{mask_path}" "{xcf_path}" "{background_name}" "{mask_name}" {background_opacity} {mask_opacity})'.format(
                image_path=scheme_quote(str(job["image_path"])),
                mask_path=scheme_quote(str(job["mask_path"])),
                xcf_path=scheme_quote(str(job["xcf_path"])),
                background_name=scheme_quote(background_layer_name),
                mask_name=scheme_quote(mask_layer_name),
                background_opacity=f"{background_opacity:.2f}",
                mask_opacity=f"{mask_opacity:.2f}",
            )
        )
    lines.extend(["  TRUE)", ""])
    return "\n".join(lines)


def write_batch_scheme(
    output_dir: Path,
    jobs: list[dict[str, object]],
    background_layer_name: str,
    mask_layer_name: str,
    background_opacity: float,
    background_mode_symbol: str,
    mask_opacity: float,
    mask_mode_symbol: str,
) -> Path:
    scheme_path = output_dir / "build_gimp_xcf_batch.scm"
    scheme_path.write_text(
        build_scheme_content(
            jobs=jobs,
            background_layer_name=background_layer_name,
            mask_layer_name=mask_layer_name,
            background_opacity=background_opacity,
            background_mode_symbol=background_mode_symbol,
            mask_opacity=mask_opacity,
            mask_mode_symbol=mask_mode_symbol,
        ),
        encoding="utf-8",
    )
    return scheme_path.resolve()


def write_manifest(output_dir: Path, jobs: list[dict[str, object]]) -> Path:
    manifest_path = output_dir / "gimp_xcf_jobs.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(job, ensure_ascii=False) + "\n" for job in jobs),
        encoding="utf-8",
    )
    return manifest_path.resolve()


def resolve_gimp_executable(explicit_path: Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(explicit_path.expanduser())

    for command in (
        "gimp-console-3.0.exe",
        "gimp-3.0.exe",
        "gimp-console-2.10.exe",
        "gimp-2.10.exe",
        "gimp.exe",
    ):
        resolved = shutil.which(command)
        if resolved:
            candidates.append(Path(resolved))

    candidates.extend(
        [
            Path(r"C:\Program Files\GIMP 3\bin\gimp-console-3.0.exe"),
            Path(r"C:\Program Files\GIMP 3\bin\gimp-3.0.exe"),
            Path(r"C:\Program Files\GIMP 2\bin\gimp-console-2.10.exe"),
            Path(r"C:\Program Files\GIMP 2\bin\gimp-2.10.exe"),
        ]
    )

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists() and resolved.is_file():
            return resolved
    raise ValueError("unable to locate GIMP executable, pass --gimp-exe explicitly")


def build_gimp_command(gimp_exe: Path, scheme_path: Path) -> list[str]:
    batch_expr = f'(begin (load "{scheme_quote(str(scheme_path))}") (sd-build-xcf-batch) (gimp-quit 0))'
    return [
        str(gimp_exe),
        "--batch-interpreter=plug-in-script-fu-eval",
        f"--batch={batch_expr}",
    ]


def run_gimp_batch(gimp_exe: Path, scheme_path: Path) -> list[str]:
    command = build_gimp_command(gimp_exe, scheme_path)
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip() or "unknown GIMP batch failure"
        raise RuntimeError(stderr)
    return command


def build_xcf_batch(args: argparse.Namespace) -> dict[str, object]:
    background_mode_symbol = resolve_mode_symbol(args.background_mode)
    mask_mode_symbol = resolve_mode_symbol(args.mask_mode)
    jobs = collect_batch_jobs(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        output_dir=args.output_dir,
        mask_suffix=args.mask_suffix,
        overwrite=args.overwrite,
        resize_mask_to_image=args.resize_mask_to_image,
    )
    output_dir = args.output_dir.expanduser().resolve()
    manifest_path = write_manifest(output_dir, jobs)
    scheme_path = write_batch_scheme(
        output_dir=output_dir,
        jobs=jobs,
        background_layer_name=args.background_layer_name,
        mask_layer_name=args.mask_layer_name,
        background_opacity=args.background_opacity,
        background_mode_symbol=background_mode_symbol,
        mask_opacity=args.mask_opacity,
        mask_mode_symbol=mask_mode_symbol,
    )

    summary: dict[str, object] = {
        "mode": "build-gimp-xcf-batch",
        "pair_count": len(jobs),
        "background_mode": args.background_mode,
        "background_mode_symbol": background_mode_symbol,
        "background_opacity": args.background_opacity,
        "mask_mode": args.mask_mode,
        "mask_mode_symbol": mask_mode_symbol,
        "mask_opacity": args.mask_opacity,
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "scheme_path": str(scheme_path),
    }

    if args.dry_run:
        summary["dry_run"] = True
        summary["gimp_executed"] = False
        return summary

    gimp_exe = resolve_gimp_executable(args.gimp_exe)
    command = run_gimp_batch(gimp_exe, scheme_path)
    summary["dry_run"] = False
    summary["gimp_executed"] = True
    summary["gimp_exe"] = str(gimp_exe)
    summary["gimp_command"] = command
    return summary


def main() -> int:
    args = parse_args()
    try:
        summary = build_xcf_batch(args)
    except (OSError, ValueError, RuntimeError) as exc:
        return fail(str(exc))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
