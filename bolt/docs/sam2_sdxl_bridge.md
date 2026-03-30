# SAM2 -> SDXL Bridge

本文件只回答一个问题：当前仓库里，`SAM2` 资产线如何接到 `SDXL inpainting` 的补绘螺帽主线。

它不是新的算法设计文档，而是当前共享口径下的“衔接说明”。

## One-Line Summary

当前默认链路是：

`box / reviewed ROI -> SAM2.1 tiny -> core_mask / edit_mask / manifest -> SDXL nut inpainting -> 局部补绘螺帽结果`

也就是说，`SAM2` 负责把“缺失螺帽的局部结构”沉淀成可复用 mask 资产，`SDXL` 负责消费这些资产，在局部 ROI 里补绘一个合理的正常螺帽。

## Why The Bridge Exists

单独看两条线，很容易各说各话：

- `SAM2` 这边在说 `core_mask`、`edit_mask`、metadata
- `SDXL` 这边在说 ROI、prompt、base model、LoRA、ControlNet

真正能跑起来，关键不在于两边都“做了很多事”，而在于中间接口稳定：

- `SAM2` 输出的 mask 要能被 `SDXL` 直接消费
- `SDXL` 需要明确自己默认吃的是哪一种 mask
- 共享给外部同学时，必须说清当前默认脚本、默认模型和默认资产来源

## Current Default Connection

当前仓库里的共享默认值，应当按下面理解：

- `SAM2` 默认运行口径：`SAM2.1 hiera tiny`
- `SAM2` 默认后端入口：`bolt/mask/scripts/good_bolt_sam2_box_prompt_backend.py`
- `SDXL` 默认主线入口：`bolt/generate/scripts/run_sdxl_nut_mainline.py`
- `SDXL` 实际批处理执行器：`bolt/generate/scripts/run_sdxl_oriented_batch.py`
- `SDXL` 默认 base model：`diffusers/stable-diffusion-xl-1.0-inpainting-0.1`
- 当前共享默认资产来源：`data/sam2_box_prompt_tiny_full_20260322/`
- 当前共享默认 manifest：`data/sam2_box_prompt_tiny_full_20260322/manifest.json`
- 当前共享默认 mask source：`core_mask`

这意味着当前主线不是“先手工画一个 inpaint mask 再自由发挥”，而是“先用 SAM2 资产线把局部结构固定下来，再交给 SDXL 做受控补绘”。

## What SAM2 Must Hand Over

对下游 `SDXL` 来说，`SAM2` 线最重要的不是“分割模型名字”，而是下面这些资产是否稳定：

- `asset_id`
- 原图定位信息
- `core_mask`
- 可选的 `edit_mask`
- 对应的 ROI / crop 信息
- metadata / manifest

最关键的两层语义是：

- `core_mask`：尽量只覆盖缺失证据本身，保守、干净、少碰背景
- `edit_mask`：在 `core_mask` 外再给一点编辑余量，服务更宽松的 inpainting 或距离扫参

如果这两层没有分开，下游就会反复混淆“结构锚点”和“允许改动区域”。

## What SDXL Consumes By Default

当前 `SDXL` 补绘螺帽主线默认吃的是 `core_mask`，不是 `edit_mask`。

原因很直接：

- 当前任务是“补绘一个正常螺帽”
- 不是大面积重画整段结构
- 也不是把螺杆一起覆盖掉重生

因此默认策略更偏保守：

- 让 `core_mask` 锚定真正缺失的位置
- 让 ROI 保留螺杆、金属板、背景、光照这些上下文
- 让 prompt 明确要求“只补一个正常六角螺帽”

只有在做 `distance ladder` 或更宽松实验时，才会刻意放大到 `edit_mask` 级别去观察上下文变化。

## Current End-to-End Path

当前共享链路可以按下面理解：

1. 上游提供框、审阅 ROI 或已确认的缺失目标。
2. `SAM2` 线基于当前默认后端产出 `core_mask`、`edit_mask`、overlay 和 metadata。
3. 资产导出为下游可消费的 manifest。
4. `run_sdxl_nut_mainline.py` 读取 manifest、原图目录和 mask 目录。
5. 脚本把默认参数包装后，转发给 `run_sdxl_oriented_batch.py`。
6. `SDXL` 在局部 ROI 内补绘 `exactly one` 正常六角螺帽。
7. 输出局部结果、整图回贴结果和复核记录，继续人工筛选。

核心逻辑不是“SAM2 直接生成结果”，而是：

`SAM2` 负责把问题区域说清楚，`SDXL` 负责在这个受控区域里完成结构修补。

## Minimal Handoff Command Shape

如果要给别人解释当前怎么接，最小命令形态就是：

```bash
python -m uv run python bolt/generate/scripts/run_sdxl_nut_mainline.py \
  --batch-manifest data/sam2_box_prompt_tiny_full_20260322/manifest.json \
  --image-dir <image_dir> \
  --core-mask-dir data/sam2_box_prompt_tiny_full_20260322 \
  --output-dir <output_dir> \
  --dry-run
```

这里可以直接看出 `SDXL` 的三个核心输入：

- 原图
- 来自 `SAM2` 资产线的 manifest
- 来自 `SAM2` 资产线的 `core_mask`

这就是两条线当前最关键的连接面。

## Recommended Shared Wording

对外复述时，建议统一成下面这句话：

“当前流程是先用 `SAM2.1 tiny` 把缺失螺帽区域沉淀成 `core_mask / edit_mask` 资产，再用 `SDXL inpainting` 在保留螺杆和背景上下文的前提下，局部补绘一个正常六角螺帽。”

这句话比“我们在做 SAM2 + SDXL”更准确，因为它明确了：

- 先后顺序
- 资产交接面
- 当前任务是补绘，不是删除
- 当前要保留螺杆和外部上下文

## Boundaries

当前桥接关系有几个边界不要说错：

- `SAM2` 不是官方标注替代
- `SDXL` 当前共享主线不是“去螺帽”，而是“补螺帽”
- 当前默认共享 mask source 是 `core_mask`
- `thread_capsule`、`donor_patch`、`PowerPaint V2` 仍属于并行实验支线，不应混写成默认主线

## Read Together

如果对方需要继续追细节，按下面顺序读：

- [../mask/README.md](/E:/Repository/Project/sd/bolt/mask/README.md)
- [sam2_asset_contract.md](/E:/Repository/Project/sd/bolt/docs/sam2_asset_contract.md)
- [../generate/README.md](/E:/Repository/Project/sd/bolt/generate/README.md)

