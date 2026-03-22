#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def find_images(image_dir: Path):
    files = []
    for p in sorted(image_dir.rglob('*')):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            files.append(p)
    return files


def stem_to_candidate_json(image_path: Path, candidate_json_dir: Path):
    return candidate_json_dir / f"{image_path.stem}.json"


def build_cmd(args, image_path: Path, candidate_json: Path | None, manual_box: str | None):
    cmd = [
        sys.executable,
        args.single_infer_script,
        '--image', str(image_path),
        '--base_model', args.base_model,
        '--controlnet_model', args.controlnet_model,
        '--lora_path', args.lora_path,
        '--output_dir', args.output_dir,
        '--prompt', args.prompt,
        '--negative_prompt', args.negative_prompt,
        '--mask_scale', str(args.mask_scale),
        '--controlnet_scale', str(args.controlnet_scale),
        '--strength', str(args.strength),
        '--lora_scale', str(args.lora_scale),
        '--steps', str(args.steps),
        '--seed', str(args.seed),
    ]

    if args.device:
        cmd += ['--device', args.device]
    if args.width:
        cmd += ['--width', str(args.width)]
    if args.height:
        cmd += ['--height', str(args.height)]
    if args.guidance_scale is not None:
        cmd += ['--guidance_scale', str(args.guidance_scale)]
    if args.class_name:
        cmd += ['--class_name', args.class_name]
    if args.box_expand is not None:
        cmd += ['--box_expand', str(args.box_expand)]
    if args.use_ellipse_mask:
        cmd += ['--use_ellipse_mask']
    if args.save_debug:
        cmd += ['--save_debug']

    if candidate_json is not None:
        cmd += ['--candidate_json', str(candidate_json)]
    elif manual_box is not None:
        cmd += ['--box', manual_box]
    else:
        raise ValueError('Either candidate_json or manual_box must be provided.')

    return cmd


def main():
    parser = argparse.ArgumentParser(description='Batch inference for missing fastener defect generation.')
    parser.add_argument('--single_infer_script', default='/mnt/data/infer_missing_fastener_controlnet_lora.py')
    parser.add_argument('--image_dir', required=True, help='Directory of healthy images.')
    parser.add_argument('--candidate_json_dir', default='', help='Directory of candidate json files. Uses <stem>.json matching.')
    parser.add_argument('--manual_box_json', default='', help='Optional JSON mapping image stem to box string "x1,y1,x2,y2".')
    parser.add_argument('--base_model', required=True)
    parser.add_argument('--controlnet_model', required=True)
    parser.add_argument('--lora_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--prompt', default='missing nut defect, power grid equipment, realistic metal fastener, missing fastener')
    parser.add_argument('--negative_prompt', default='deformed, blurry, extra parts, duplicate, bad anatomy, unrealistic metal texture')
    parser.add_argument('--mask_scale', type=float, default=1.18)
    parser.add_argument('--controlnet_scale', type=float, default=0.8)
    parser.add_argument('--strength', type=float, default=0.78)
    parser.add_argument('--lora_scale', type=float, default=0.9)
    parser.add_argument('--steps', type=int, default=28)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--guidance_scale', type=float, default=7.0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--width', type=int, default=0)
    parser.add_argument('--height', type=int, default=0)
    parser.add_argument('--class_name', default='missing_fastener')
    parser.add_argument('--box_expand', type=float, default=1.0)
    parser.add_argument('--use_ellipse_mask', action='store_true')
    parser.add_argument('--save_debug', action='store_true')
    parser.add_argument('--skip_missing_json', action='store_true', help='Skip images whose candidate json is missing.')
    parser.add_argument('--max_images', type=int, default=0, help='Process at most N images; 0 means all.')
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--continue_on_error', action='store_true')
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_json_dir = Path(args.candidate_json_dir) if args.candidate_json_dir else None
    manual_boxes = {}
    if args.manual_box_json:
        with open(args.manual_box_json, 'r', encoding='utf-8') as f:
            manual_boxes = json.load(f)

    images = find_images(image_dir)
    if args.max_images > 0:
        images = images[:args.max_images]

    print(f'[INFO] Found {len(images)} healthy images.')

    processed = 0
    skipped = 0
    failed = 0

    for image_path in images:
        candidate_json = None
        manual_box = None

        if candidate_json_dir is not None:
            cj = stem_to_candidate_json(image_path, candidate_json_dir)
            if cj.exists():
                candidate_json = cj
            else:
                if args.skip_missing_json:
                    print(f'[SKIP] Missing candidate json: {cj.name}')
                    skipped += 1
                    continue

        if candidate_json is None:
            manual_box = manual_boxes.get(image_path.stem)
            if manual_box is None:
                print(f'[SKIP] No candidate json or manual box for: {image_path.name}')
                skipped += 1
                continue

        cmd = build_cmd(args, image_path, candidate_json, manual_box)
        print('[RUN]', ' '.join(cmd))

        if args.dry_run:
            processed += 1
            continue

        try:
            subprocess.run(cmd, check=True)
            processed += 1
        except subprocess.CalledProcessError as e:
            failed += 1
            print(f'[FAIL] {image_path.name}: returncode={e.returncode}')
            if not args.continue_on_error:
                raise

    print(f'[DONE] processed={processed}, skipped={skipped}, failed={failed}')


if __name__ == '__main__':
    main()
