import argparse
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image
import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetInpaintPipeline,
    UniPCMultistepScheduler,
)


"""
核心主流程（单张推理测试）
"""

"""
自动取框
python infer_missing_fastener_controlnet_lora.py \
  --image healthy_images/h001.jpg \
  --candidate_json match_results/json/h001.json \
  --base_model /path/to/sd-v1-5-inpainting \
  --controlnet_model /path/to/control_v11p_sd15_canny \
  --lora_path /path/to/output/missing_fastener_lora.safetensors \
  --output_dir infer_out
"""

"""
python infer_missing_fastener_controlnet_lora.py \
  --image healthy_images/h001.jpg \
  --candidate_json match_results/json/h001.json \
  --base_model /path/to/sd-v1-5-inpainting \
  --controlnet_model /path/to/control_v11p_sd15_canny \
  --lora_path /path/to/output/missing_fastener_lora.safetensors \
  --prompt "missing nut defect, power grid equipment, realistic metal fastener, missing fastener" \
  --mask_scale 1.18 \
  --controlnet_scale 0.8 \
  --strength 0.78 \
  --lora_scale 0.9 \
  --steps 28 \
  --output_dir infer_out
"""

# ------------------------------
# Utility functions
# ------------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def read_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def pil_to_bgr(img_pil: Image.Image) -> np.ndarray:
    rgb = np.array(img_pil)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def clamp(v: int, low: int, high: int) -> int:
    return max(low, min(v, high))


def make_square_roi(box: List[int], img_w: int, img_h: int, context_scale: float = 3.0) -> List[int]:
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    side = int(round(max(bw, bh) * context_scale))
    side = max(side, max(bw, bh) + 16)

    rx1 = int(round(cx - side / 2))
    ry1 = int(round(cy - side / 2))
    rx2 = rx1 + side
    ry2 = ry1 + side

    if rx1 < 0:
        rx2 -= rx1
        rx1 = 0
    if ry1 < 0:
        ry2 -= ry1
        ry1 = 0
    if rx2 > img_w:
        shift = rx2 - img_w
        rx1 -= shift
        rx2 = img_w
    if ry2 > img_h:
        shift = ry2 - img_h
        ry1 -= shift
        ry2 = img_h

    rx1 = clamp(rx1, 0, img_w - 1)
    ry1 = clamp(ry1, 0, img_h - 1)
    rx2 = clamp(rx2, 1, img_w)
    ry2 = clamp(ry2, 1, img_h)

    return [rx1, ry1, rx2, ry2]


def box_global_to_local(box: List[int], roi: List[int]) -> List[int]:
    rx1, ry1, _, _ = roi
    x1, y1, x2, y2 = box
    return [x1 - rx1, y1 - ry1, x2 - rx1, y2 - ry1]


def build_mask(
    roi_shape: Tuple[int, int],
    local_box: List[int],
    box_scale: float = 1.15,
    blur_ksize: int = 21,
    shape: str = "ellipse",
) -> np.ndarray:
    h, w = roi_shape[:2]
    x1, y1, x2, y2 = local_box
    bw = x2 - x1
    bh = y2 - y1
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    mw = bw * box_scale
    mh = bh * box_scale
    mx1 = int(round(cx - mw / 2))
    my1 = int(round(cy - mh / 2))
    mx2 = int(round(cx + mw / 2))
    my2 = int(round(cy + mh / 2))

    mx1 = clamp(mx1, 0, w - 1)
    my1 = clamp(my1, 0, h - 1)
    mx2 = clamp(mx2, 1, w)
    my2 = clamp(my2, 1, h)

    mask = np.zeros((h, w), dtype=np.uint8)
    if shape == "rect":
        mask[my1:my2, mx1:mx2] = 255
    else:
        center = ((mx1 + mx2) // 2, (my1 + my2) // 2)
        axes = (max(1, (mx2 - mx1) // 2), max(1, (my2 - my1) // 2))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

    if blur_ksize > 1:
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        mask = cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), 0)
    return mask


def build_canny_control(roi_bgr: np.ndarray, low_thresh: int = 80, high_thresh: int = 160) -> np.ndarray:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, low_thresh, high_thresh)
    edges_3 = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return edges_3


def resize_to_multiple_of_8(img_bgr: np.ndarray, mask: np.ndarray, control_bgr: np.ndarray, target: int = 512):
    h, w = img_bgr.shape[:2]
    side = max(h, w)
    scale = target / float(side)
    nh = max(64, int(round(h * scale / 8) * 8))
    nw = max(64, int(round(w * scale / 8) * 8))

    img_r = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_CUBIC)
    mask_r = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_LINEAR)
    control_r = cv2.resize(control_bgr, (nw, nh), interpolation=cv2.INTER_CUBIC)
    return img_r, mask_r, control_r


def paste_roi_back(full_bgr: np.ndarray, roi_box: List[int], gen_roi_bgr: np.ndarray, feather_mask: np.ndarray) -> np.ndarray:
    rx1, ry1, rx2, ry2 = roi_box
    out = full_bgr.copy()
    orig_patch = out[ry1:ry2, rx1:rx2].astype(np.float32)

    gh, gw = gen_roi_bgr.shape[:2]
    ph, pw = orig_patch.shape[:2]
    if (gh, gw) != (ph, pw):
        gen_roi_bgr = cv2.resize(gen_roi_bgr, (pw, ph), interpolation=cv2.INTER_CUBIC)
        feather_mask = cv2.resize(feather_mask, (pw, ph), interpolation=cv2.INTER_LINEAR)

    alpha = feather_mask.astype(np.float32) / 255.0
    if alpha.ndim == 2:
        alpha = alpha[:, :, None]
    gen_patch = gen_roi_bgr.astype(np.float32)
    blend = gen_patch * alpha + orig_patch * (1.0 - alpha)
    out[ry1:ry2, rx1:rx2] = np.clip(blend, 0, 255).astype(np.uint8)
    return out


def save_voc_xml(xml_path: str, image_filename: str, image_size: Tuple[int, int, int], bbox: List[int], label: str):
    h, w, c = image_size
    x1, y1, x2, y2 = [int(v) for v in bbox]

    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "filename").text = image_filename

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(w)
    ET.SubElement(size, "height").text = str(h)
    ET.SubElement(size, "depth").text = str(c)

    obj = ET.SubElement(annotation, "object")
    ET.SubElement(obj, "name").text = label
    bnd = ET.SubElement(obj, "bndbox")
    ET.SubElement(bnd, "xmin").text = str(x1)
    ET.SubElement(bnd, "ymin").text = str(y1)
    ET.SubElement(bnd, "xmax").text = str(x2)
    ET.SubElement(bnd, "ymax").text = str(y2)

    tree = ET.ElementTree(annotation)
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)


def save_yolo_txt(txt_path: str, image_size: Tuple[int, int], bbox: List[int], class_id: int = 0):
    h, w = image_size
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) / 2.0) / w
    cy = ((y1 + y2) / 2.0) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def pick_best_box(candidate_json_path: str, min_score: float = 0.0) -> List[int]:
    with open(candidate_json_path, "r", encoding="utf-8") as f:
        cands = json.load(f)
    if not cands:
        raise ValueError(f"No candidates found in {candidate_json_path}")
    cands = sorted(cands, key=lambda x: x.get("final_score", x.get("score", 0.0)), reverse=True)
    best = cands[0]
    score = best.get("final_score", best.get("score", 0.0))
    if score < min_score:
        raise ValueError(f"Best candidate score {score:.4f} < min_score {min_score:.4f}")
    return [int(v) for v in best["pred_box"]]


# ------------------------------
# Main inference
# ------------------------------

def run(args):
    ensure_dir(args.output_dir)
    ensure_dir(os.path.join(args.output_dir, "vis"))
    ensure_dir(os.path.join(args.output_dir, "images"))
    ensure_dir(os.path.join(args.output_dir, "labels_voc"))
    ensure_dir(os.path.join(args.output_dir, "labels_yolo"))
    ensure_dir(os.path.join(args.output_dir, "debug"))

    full_bgr = read_bgr(args.image)
    H, W = full_bgr.shape[:2]

    if args.candidate_json:
        target_box = pick_best_box(args.candidate_json, args.min_candidate_score)
    elif args.box:
        target_box = [int(v) for v in args.box.split(",")]
        if len(target_box) != 4:
            raise ValueError("--box format must be x1,y1,x2,y2")
    else:
        raise ValueError("Provide either --candidate_json or --box")

    roi_box = make_square_roi(target_box, W, H, context_scale=args.context_scale)
    local_box = box_global_to_local(target_box, roi_box)
    rx1, ry1, rx2, ry2 = roi_box
    roi_bgr = full_bgr[ry1:ry2, rx1:rx2].copy()

    mask = build_mask(
        roi_bgr.shape,
        local_box,
        box_scale=args.mask_scale,
        blur_ksize=args.mask_blur,
        shape=args.mask_shape,
    )
    control_bgr = build_canny_control(roi_bgr, args.canny_low, args.canny_high)

    roi_r, mask_r, control_r = resize_to_multiple_of_8(roi_bgr, mask, control_bgr, target=args.infer_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    controlnet = ControlNetModel.from_pretrained(args.controlnet_model, torch_dtype=dtype)
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        args.base_model,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    if args.lora_path:
        adapter_name = "missing_fastener_lora"
        pipe.load_lora_weights(args.lora_path, adapter_name=adapter_name)
        pipe.set_adapters([adapter_name], adapter_weights=[args.lora_scale])

    if device == "cuda":
        pipe = pipe.to(device)
        if args.enable_xformers:
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
    else:
        pipe = pipe.to(device)

    generator = None
    if args.seed is not None and args.seed >= 0:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    image_pil = bgr_to_pil(roi_r)
    mask_pil = Image.fromarray(mask_r)
    control_pil = bgr_to_pil(control_r)

    negative_prompt = args.negative_prompt or (
        "extra parts, duplicated parts, deformed metal, broken structure, "
        "blurry, unrealistic, severe artifacts"
    )

    with torch.inference_mode():
        result = pipe(
            prompt=args.prompt,
            negative_prompt=negative_prompt,
            image=image_pil,
            mask_image=mask_pil,
            control_image=control_pil,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            controlnet_conditioning_scale=args.controlnet_scale,
            strength=args.strength,
            generator=generator,
        ).images[0]

    gen_roi_bgr = pil_to_bgr(result)
    final_bgr = paste_roi_back(full_bgr, roi_box, gen_roi_bgr, mask)

    stem = Path(args.image).stem
    out_img_name = f"{stem}_missing_fastener.png"
    out_img_path = os.path.join(args.output_dir, "images", out_img_name)
    cv2.imwrite(out_img_path, final_bgr)

    vis = final_bgr.copy()
    x1, y1, x2, y2 = target_box
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(vis, args.label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(args.output_dir, "vis", f"{stem}_vis.jpg"), vis)

    debug_roi = roi_bgr.copy()
    debug_mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    debug_overlay = cv2.addWeighted(debug_roi, 0.7, debug_mask_color, 0.3, 0)
    cv2.rectangle(debug_overlay, (local_box[0], local_box[1]), (local_box[2], local_box[3]), (0, 255, 0), 2)
    cv2.imwrite(os.path.join(args.output_dir, "debug", f"{stem}_roi.jpg"), debug_roi)
    cv2.imwrite(os.path.join(args.output_dir, "debug", f"{stem}_mask.jpg"), mask)
    cv2.imwrite(os.path.join(args.output_dir, "debug", f"{stem}_overlay.jpg"), debug_overlay)
    cv2.imwrite(os.path.join(args.output_dir, "debug", f"{stem}_control.jpg"), control_bgr)

    save_voc_xml(
        os.path.join(args.output_dir, "labels_voc", f"{stem}_missing_fastener.xml"),
        out_img_name,
        final_bgr.shape,
        target_box,
        args.label,
    )
    save_yolo_txt(
        os.path.join(args.output_dir, "labels_yolo", f"{stem}_missing_fastener.txt"),
        final_bgr.shape[:2],
        target_box,
        args.class_id,
    )

    meta = {
        "source_image": args.image,
        "candidate_json": args.candidate_json,
        "target_box": target_box,
        "roi_box": roi_box,
        "prompt": args.prompt,
        "negative_prompt": negative_prompt,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "controlnet_scale": args.controlnet_scale,
        "strength": args.strength,
        "seed": args.seed,
        "label": args.label,
    }
    with open(os.path.join(args.output_dir, "debug", f"{stem}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[DONE] saved image: {out_img_path}")
    print(f"[DONE] target box: {target_box}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Healthy image + auto box + mask + ControlNet + LoRA inference")
    parser.add_argument("--image", required=True, help="Path to healthy image")
    parser.add_argument("--candidate_json", default="", help="JSON produced by the context matching script")
    parser.add_argument("--box", default="", help="Fallback manual box: x1,y1,x2,y2")

    parser.add_argument("--base_model", required=True, help="SD1.5 inpainting base model path or repo")
    parser.add_argument("--controlnet_model", required=True, help="ControlNet model path or repo")
    parser.add_argument("--lora_path", default="", help="LoRA .safetensors or directory")
    parser.add_argument("--lora_scale", type=float, default=0.9)

    parser.add_argument("--prompt", default="missing nut defect, power grid equipment, realistic metal fastener, missing fastener")
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--label", default="missing_fastener")
    parser.add_argument("--class_id", type=int, default=0)

    parser.add_argument("--context_scale", type=float, default=3.2)
    parser.add_argument("--mask_scale", type=float, default=1.18)
    parser.add_argument("--mask_blur", type=int, default=21)
    parser.add_argument("--mask_shape", choices=["ellipse", "rect"], default="ellipse")

    parser.add_argument("--canny_low", type=int, default=80)
    parser.add_argument("--canny_high", type=int, default=160)
    parser.add_argument("--infer_size", type=int, default=512)

    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=6.5)
    parser.add_argument("--controlnet_scale", type=float, default=0.8)
    parser.add_argument("--strength", type=float, default=0.78)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_candidate_score", type=float, default=0.0)
    parser.add_argument("--enable_xformers", action="store_true")

    parser.add_argument("--output_dir", default="infer_out")
    args = parser.parse_args()
    run(args)
