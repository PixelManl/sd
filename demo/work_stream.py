"""
合成缺陷数据集生成脚本
流程：
1. 生成健康背景图像（使用普通 SD）
2. 对每张背景图像，分析合理区域，生成随机掩码
3. 使用 inpainting 模型（+ LoRA + ControlNet）修复掩码区域
4. 保存修复图像和 YOLO 标注
"""

import os
import random
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionInpaintPipeline,
    ControlNetModel,
)
from diffusers.utils import load_image
import torch

TIME_STAMP = os.times

# ======================== 配置参数 ========================
# 路径设置
BASE_DIR = "/home/jrx/cygy/zf/sd"
HEALTHY_BACKGROUND_DIR = f"{BASE_DIR}/data/my/healthy_backgrounds"   # 存放生成/已有的健康背景图
OUTPUT_DIR = f"{BASE_DIR}/data/my/mask_output"          # 输出目录
LORA_WEIGHTS_PATH = ""  # 可选，留空则不加载./output/defect_lora.safetensors
USE_CONTROLNET = True                               # 是否使用 ControlNet 控制缺陷形状
CONTROLNET_TYPE = "canny"                          # 目前支持 canny

# 模型加载路径
CANNY_DIR = "/home/jrx/cygy/zf/sd/models/canny"
INPAINTING_DIR = "/home/jrx/cygy/zf/sd/models/inpainting"
SD15_DIR = "/home/jrx/cygy/sd15"


# 生成参数
NUM_BACKGROUNDS_TO_GENERATE = 2                  # 要生成的背景图数量（如果背景目录已存在，则跳过生成）
IMAGE_SIZE = 512                                   # 生成图像尺寸（正方形）
BATCH_SIZE = 2                                      # 生成背景图时的 batch size
BG_PROMPTS=[
    # "a photo of a healthy building facade",
    "a photo of a wall surface"
]

BG_N_PROMPTS=[
    "a photo of a damaged building facade",
    "a photo of a building surface with damage"
]

# 缺陷生成参数
DEFECT_CLASS_ID = 0                                 # YOLO 类别 ID
DEFECT_PROMPTS = [                                  # 缺陷描述列表
    # "a scratch on metal surface",
    # "a dent on metal surface",
    # "a crack on metal surface",
    # "a rust spot on metal surface",
    # "a scratch on building surface",
    # "a crack on building surface",
    "photorealistic concrete crack, dark moss, weathered building facade, high detail, 8k uhd, coherent crack"
]
MASKS_PER_IMAGE = 1                                 # 每张图生成的缺陷数量（避免重叠）
MASK_MIN_RATIO = 0.05                               # 掩码最小尺寸比例（相对于图像边长）
MASK_MAX_RATIO = 0.3                                # 掩码最大尺寸比例

# 模型加载选项
USE_CPU_OFFLOAD = False                             # 是否启用CPU，如果显存不足，开启此选项（会降低速度）
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# ======================== 辅助函数 ========================
def generate_backgrounds(num_images, output_dir):
    """使用普通 SD 生成健康背景图像"""
    print(f"正在生成 {num_images} 张健康背景图像...")
    pipe = StableDiffusionPipeline.from_pretrained(
        SD15_DIR,
        safety_checker=None,            #禁用安全检查
        requires_safety_checker=False,
        torch_dtype=DTYPE
    )

    pipe.safety_checker = None

    if USE_CPU_OFFLOAD:
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to(DEVICE)

    # 预定义提示词列表（可自行扩展）
    prompts = BG_PROMPTS           # 例如 ["a photo of a healthy building facade", "a photo of a building surface"]
    negative_prompts = BG_N_PROMPTS  # 例如 ["a photo of a damaged building facade", "a photo of a building surface with damage"]

    os.makedirs(output_dir, exist_ok=True)
    generated = 0
    while generated < num_images:
        # 确定本次批次数目
        batch_size = min(BATCH_SIZE, num_images - generated)
        # 随机选取 batch_size 个提示词（允许重复）
        batch_prompts = random.choices(prompts, k=batch_size)
        batch_negative = random.choices(negative_prompts, k=batch_size)

        with torch.no_grad():
            images = pipe(
                prompt=batch_prompts,
                negative_prompt=batch_negative,
                height=IMAGE_SIZE,
                width=IMAGE_SIZE,
                num_inference_steps=50   # 可根据需要调整
            ).images   # 返回 list of PIL.Image

        for j, img in enumerate(images):
            img.save(os.path.join(output_dir, f"bg_{generated+j:06d}.png"))
        generated += batch_size
        print(f"已生成 {generated}/{num_images} 张背景图")

    del pipe
    torch.cuda.empty_cache()
    print("背景图生成完成。")

def compute_probability_map(image):
    """
    基于图像边缘密度生成概率图，边缘区域概率低，平坦区域概率高。
    返回归一化的概率图（numpy 数组，shape=(H,W)，值在 0~1 之间）。
    """
    # 转换为灰度图
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    # Canny 边缘检测
    edges = cv2.Canny(gray, 50, 150)
    # 膨胀边缘区域，扩大抑制范围
    kernel = np.ones((20,20), np.uint8)
    edge_region = cv2.dilate(edges, kernel, iterations=1)
    # 概率图：平坦区域概率高，边缘区域概率低
    prob = np.ones_like(edges, dtype=np.float32)
    prob[edge_region > 0] = 0.2
    # 平滑一下，避免边界太锐利
    prob = cv2.GaussianBlur(prob, (15,15), 0)
    # 归一化
    prob = prob / prob.sum()
    return prob

def sample_center_from_prob_map(prob_map):
    """根据概率图随机采样一个中心点 (x, y)"""
    h, w = prob_map.shape
    # 将概率图展平为一维
    flat = prob_map.flatten()
    # 随机选择一个索引
    idx = np.random.choice(len(flat), p=flat)
    y = idx // w
    x = idx % w
    return x, y

def generate_mask(image_size, center, size_ratio):
    """生成矩形掩码，返回掩码图像（PIL）和边界框 (x1,y1,x2,y2)"""
    w, h = image_size
    rect_w = int(w * random.uniform(*size_ratio))
    rect_h = int(h * random.uniform(*size_ratio))
    x1 = center[0] - rect_w // 2
    y1 = center[1] - rect_h // 2
    # 确保不超出边界
    x1 = max(0, min(x1, w - rect_w))
    y1 = max(0, min(y1, h - rect_h))
    x2 = x1 + rect_w
    y2 = y1 + rect_h
    
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    mask_pil = Image.fromarray(mask)
    return mask_pil, (x1, y1, x2, y2)

def save_yolo_annotation(bbox, image_size, output_path, class_id=0):
    """保存 YOLO 格式标注文件"""
    img_w, img_h = image_size
    x1, y1, x2, y2 = bbox
    x_center = (x1 + x2) / 2 / img_w
    y_center = (y1 + y2) / 2 / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h
    with open(output_path, 'w') as f:
        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

def get_control_image(image, mask, defect_type):
    """
    根据缺陷类型生成 ControlNet 输入图像。
    目前简单实现：对于 scratch/crack 返回随机线段图；其他返回边缘图。
    """
    if not USE_CONTROLNET:
        return None
    if CONTROLNET_TYPE == "canny":
        # 使用原图的 Canny 边缘作为控制
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return Image.fromarray(edges)
    elif CONTROLNET_TYPE == "scratch_line":
        # 生成随机线段图，模拟划痕
        h, w = image.size[1], image.size[0]
        line_img = np.zeros((h, w), dtype=np.uint8)
        # 随机生成一些线段
        num_lines = random.randint(1, 5)
        for _ in range(num_lines):
            x1 = random.randint(0, w)
            y1 = random.randint(0, h)
            x2 = random.randint(0, w)
            y2 = random.randint(0, h)
            cv2.line(line_img, (x1, y1), (x2, y2), 255, random.randint(1, 3))
        return Image.fromarray(line_img)
    else:
        return None

# ======================== 主流程 ========================
def main():
    # 1. 准备健康背景图（同原代码）
    if not os.path.exists(HEALTHY_BACKGROUND_DIR) or len(os.listdir(HEALTHY_BACKGROUND_DIR)) == 0:
        generate_backgrounds(NUM_BACKGROUNDS_TO_GENERATE, HEALTHY_BACKGROUND_DIR)
    else:
        print(f"使用已有背景图目录: {HEALTHY_BACKGROUND_DIR}")

    # 2. 加载模型（同原代码）
    print("加载 inpainting 模型...")
    if USE_CONTROLNET:
        controlnet = ControlNetModel.from_pretrained(CANNY_DIR)
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            INPAINTING_DIR, controlnet=controlnet, torch_dtype=DTYPE
        )
    else:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            INPAINTING_DIR, torch_dtype=DTYPE
        )
    if LORA_WEIGHTS_PATH and os.path.exists(LORA_WEIGHTS_PATH):
        print(f"加载 LoRA 权重: {LORA_WEIGHTS_PATH}")
        pipe.load_lora_weights(LORA_WEIGHTS_PATH)

    pipe.safty_checker = None
    
    if USE_CPU_OFFLOAD:
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to(DEVICE)

    # 3. 创建输出目录（新增 masks 和 controls）
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    images_dir = os.path.join(OUTPUT_DIR, "images")
    labels_dir = os.path.join(OUTPUT_DIR, "labels")
    masks_dir = os.path.join(OUTPUT_DIR, "masks")
    controls_dir = os.path.join(OUTPUT_DIR, "controls")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(controls_dir, exist_ok=True)

    # 4. 遍历所有背景图
    bg_files = sorted(Path(HEALTHY_BACKGROUND_DIR).glob("*.png")) + sorted(Path(HEALTHY_BACKGROUND_DIR).glob("*.jpg"))
    print(f"找到 {len(bg_files)} 张背景图，开始生成缺陷数据...")

    for idx, bg_path in enumerate(bg_files):
        print(f"处理 {idx+1}/{len(bg_files)}: {bg_path.name}")
        bg_img = Image.open(bg_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        prob_map = compute_probability_map(bg_img)

        # 生成所有掩码信息
        masks_info = []
        for _ in range(MASKS_PER_IMAGE):
            center = sample_center_from_prob_map(prob_map)
            mask, bbox = generate_mask((IMAGE_SIZE, IMAGE_SIZE), center, (MASK_MIN_RATIO, MASK_MAX_RATIO))
            masks_info.append((mask, bbox))

        prompt = random.choice(DEFECT_PROMPTS)

        current_image = bg_img
        for defect_idx, (mask, bbox) in enumerate(masks_info):
            # 为每个缺陷生成控制图
            control_image = get_control_image(current_image, mask, prompt)

            # 保存掩码
            mask_filename = f"{bg_path.stem}_mask_{defect_idx}.png"
            mask.save(os.path.join(masks_dir, mask_filename))

            # 保存控制图
            if control_image is not None:
                control_filename = f"{bg_path.stem}_control_{defect_idx}.png"
                control_image.save(os.path.join(controls_dir, control_filename))

            # 修复
            with torch.no_grad():
                result = pipe(
                    prompt=prompt,
                    image=current_image,
                    mask_image=mask,
                    control_image=control_image,
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    height=IMAGE_SIZE,
                    width=IMAGE_SIZE,
                    strength=0.75,
                ).images[0]
            current_image = result

            # 保存 YOLO 标注
            label_filename = f"{bg_path.stem}_defect_{defect_idx}.txt"
            save_yolo_annotation(bbox, (IMAGE_SIZE, IMAGE_SIZE),
                                 os.path.join(labels_dir, label_filename),
                                 DEFECT_CLASS_ID)

        # 保存最终图像
        result_image = current_image
        result_image.save(os.path.join(images_dir, f"{bg_path.stem}.png"))

        if (idx+1) % 10 == 0:
            torch.cuda.empty_cache()

    print(f"所有数据生成完成！输出目录：{OUTPUT_DIR}")

if __name__ == "__main__":
    main()