import torch
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel
from PIL import Image
import numpy as np
import cv2
import os
import glob
import argparse

def extract_canny_feature(image):
    image = np.array(image)
    edges = cv2.Canny(image, 100, 200)
    edges = edges[:, :, None]
    edges = np.concatenate([edges, edges, edges], axis=2)
    return Image.fromarray(edges)

def main():
    parser = argparse.ArgumentParser(description="Batch Generate Synthetic Cracks")
    parser.add_argument("--num", type=int, default=5, help="Number of images to process")
    parser.add_argument("--dir", type=str, default="test", choices=["test", "train", "val"], help="Source directory (test/train/val)")
    args = parser.parse_args()

    # 1. Path Configuration
    base_dir = r"/home/jrx/cygy/zf/sd"
    source_img_dir = rf"/home/jrx/cygy/zf/data/crack/{args.dir}_img"
    source_lab_dir = rf"/home/jrx/cygy/zf/data/crack/{args.dir}_lab"
    output_dir = os.path.join(base_dir, 'data', f'augmented_{args.dir}')
    
    os.makedirs(output_dir, exist_ok=True)

    print(f"--- Batch Processing Config ---")
    print(f"Source: {source_img_dir}")
    print(f"Count: {args.num}")
    print(f"Output: {output_dir}")
    print(f"-------------------------------")

    print("Checking CUDA availability...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    # 2. Load Models
    print("Loading models (this may take a moment on first run)...")
    controlnet = ControlNetModel.from_pretrained(
        # "lllyasviel/control_v11p_sd15_canny",
        "/home/jrx/cygy/zf/sd/models/canny",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )

    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        # "runwayml/stable-diffusion-inpainting",
        "/home/jrx/cygy/zf/sd/models/inpainting",
        controlnet=controlnet,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    pipe.safety_checker = None

    # 3. Batch Process
    image_paths = glob.glob(os.path.join(source_img_dir, "*.jpg"))
    if not image_paths:
        print(f"Error: No images found in {source_img_dir}")
        return

    # Process requested number
    test_files = image_paths[:args.num]
    
    prompt = "photorealistic concrete crack, weathering, building facade disease, high resolution"
    negative_prompt = "flat color, monochromatic, cartoon, blur, low quality, clean wall"

    for i, img_path in enumerate(test_files):
        filename = os.path.basename(img_path)
        basename = os.path.splitext(filename)[0]
        mask_path = os.path.join(source_lab_dir, basename + ".png")
        
        if not os.path.exists(mask_path):
            print(f"[{i+1}/{args.num}] Skipping {filename}: Mask not found at {mask_path}")
            continue

        print(f"[{i+1}/{args.num}] Generating: {filename}...")
        
        # Load and resize
        original_image = Image.open(img_path).convert("RGB").resize((512, 512))
        mask_image = Image.open(mask_path).convert("L").resize((512, 512))
        
        # Guide the structure
        control_image = extract_canny_feature(original_image)

        # Generate
        generator = torch.Generator(device).manual_seed(42 + i) # Vary seed slightly for each image
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=30,
            image=original_image,
            mask_image=mask_image,
            control_image=control_image,
            generator=generator,
            strength=0.95
        ).images[0]

        # Save result
        result.save(os.path.join(output_dir, f"aug_{filename}"))

    print(f"\nBatch process complete. Results saved in: {output_dir}")

if __name__ == "__main__":
    main()
