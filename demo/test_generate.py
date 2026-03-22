import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os

# 设置设备（自动选择 GPU 或 CPU）
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载模型（使用 float16 加速，也可用 float32）
model_path = "/home/jrx/cygy/sd2-inpainting"
pipe = StableDiffusionPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    safety_checker=None,  # 可选：关闭安全检查器以加速
)
pipe = pipe.to(device)

# 启用内存优化（如果显存不足）
pipe.enable_attention_slicing()
# pipe.enable_xformers_memory_efficient_attention()  # 可选，需安装 xformers

# 提示词：描述目标场景
prompt = (
    "a close-up realistic photo of an electrical wire connection point, "
    "a rusty steel bolt without a nut, missing nut on the bolt, exposed threads, "
    "industrial environment, high detail, 8k, sharp focus, natural lighting"
)

# 负面提示词：避免不想要的元素
negative_prompt = (
    "cartoon, illustration, painting, blurry, low quality, distorted, "
    "nut present, multiple nuts, wires disconnected, unrealistic"
)

# 生成参数
num_inference_steps = 50   # 推理步数，越高越细致
guidance_scale = 7.5       # 引导比例，越高越贴近提示词
seed = 42                  # 固定随机种子，确保可复现
generator = torch.Generator(device=device).manual_seed(seed)

# 生成图像
with torch.autocast(device):  # 自动混合精度加速
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        height=512,   # SD1.5 默认 512x512
        width=512,
    ).images[0]

# 保存图像
output_path = "missing_nut_bolt.png"
image.save(output_path)
print(f"图像已保存至: {output_path}")