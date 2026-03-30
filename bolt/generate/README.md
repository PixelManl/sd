# Generate Stage

这一层负责健康图匹配、缺陷注入和图像生成。

## Current Shared Mainline

当前可共享、可复述的 `SDXL inpainting` 主线，不是“删除螺帽”，而是“在缺失螺栓场景里补绘一个正常螺帽”。

当前默认入口：

- `bolt/generate/scripts/run_sdxl_nut_mainline.py`

它会把当前主线包装成一个固定默认值的调用，再转发到：

- `bolt/generate/scripts/run_sdxl_oriented_batch.py`

当前默认共享口径如下：

- task：在局部 ROI 内补绘 `exactly one` 正常六角螺帽
- base model：`diffusers/stable-diffusion-xl-1.0-inpainting-0.1`
- default SAM2 asset source：`data/sam2_box_prompt_tiny_full_20260322/`
- default manifest：`data/sam2_box_prompt_tiny_full_20260322/manifest.json`
- default mask source：`core_mask`
- optional LoRA：`nut_semantic_lora`
- optional ControlNet：通过 `--controlnet-model` 显式传入；默认不强绑

默认 prompt 的语义目标是：

- 只补一个略微风化的灰色钢制六角螺帽
- 螺帽要紧贴金属板下表面
- 保留已有螺杆、金属板、透视、光照和背景
- 明确排斥 `tall spacer`、`sleeve`、重复五金件、空心套筒等错误形态

这条线当前服务的是“结构修复原型”和“可复核的局部增强实验”，不是最终官方标注生产线。

建议本地私有目录：

- `outputs/`：生成结果
- `cache/`：中间缓存

这些目录默认不入 Git。

## Current SDXL Nut-Mainline Example

当需要把当前“补绘螺帽”主线发给别人复现时，优先给这一条：

```bash
python -m uv run python bolt/generate/scripts/run_sdxl_nut_mainline.py \
  --batch-manifest data/sam2_box_prompt_tiny_full_20260322/manifest.json \
  --image-dir <image_dir> \
  --core-mask-dir data/sam2_box_prompt_tiny_full_20260322 \
  --output-dir <output_dir> \
  --dry-run
```

如果要真跑，再改为：

```bash
python -m uv run python bolt/generate/scripts/run_sdxl_nut_mainline.py \
  --batch-manifest data/sam2_box_prompt_tiny_full_20260322/manifest.json \
  --image-dir <image_dir> \
  --core-mask-dir data/sam2_box_prompt_tiny_full_20260322 \
  --output-dir <output_dir> \
  --execute
```

补充说明：

- `run_sdxl_nut_mainline.py` 代表“当前共享默认值”
- `run_sdxl_oriented_batch.py` 代表“底层批处理执行器”
- `run_sdxl_distance_ladder.py` 用于只扫 `edit mask` 外扩距离，不代表默认主线
- `thread_capsule`、`donor_patch`、`PowerPaint V2` 属于并行实验支线，不应混同为当前共享的 SDXL 主线

## Distance Ladder

当需要固定 ROI 尺度、只对 `edit mask` 外扩距离做单变量对照时，使用：

```bash
python -m uv run python bolt/generate/scripts/run_sdxl_distance_ladder.py \
  --batch-manifest <batch_manifest.json> \
  --image-dir <image_dir> \
  --core-mask-dir <core_mask_dir> \
  --output-dir <output_dir> \
  --crop-scale 1.75 \
  --variant d18:0.18 \
  --variant d26:0.26 \
  --variant d34:0.34 \
  --variant d42:0.42
```

这条脚本会输出每一档的 `roi_input`、`roi_mask_core`、`roi_mask_edit`、`roi_output`、`full_output`，并生成 `distance_ladder_manifest.json`。
