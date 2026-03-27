# Round1 Parallel Workflow

当前交付按三条线并行推进，但目标只有一个：在 **2026-03-31** 前完成初赛可提交版本。

## Why This Split Is Reasonable

这个切法是合理的，原因很直接：

- 你负责 `SDXL` 修复和后续增强样本，是新的正向增益来源。
- 一位同学负责原始 `100` 张图的修图与数据增强，可以先在旧数据上持续产出，不阻塞检测训练。
- 一位同学负责 `YOLO11n` 训练，先用当前真实框数据稳定 baseline，等新数据合入后再切第二轮训练。
- 昇腾背景处理独立成并行支线，不应该卡住 `SDXL -> detector` 这条主线。

关键不是所有人做所有事，而是每个人有固定入口、固定产出目录、固定交接时间点。

## Fixed Timeline

- `2026-03-24` 到 `2026-03-26`
  - 你继续做 `SDXL` 修复与样本筛选。
  - 图像增强同学基于原始 `100` 张图做第一轮增强。
  - 检测同学用当前真实数据和现有标注先跑 `YOLO11n` baseline。
- `2026-03-27`
  - 合并第一轮新增样本。
  - 统一整理标注、元数据和训练入口。
  - 检测同学切到“合并版数据集”开始新一轮训练。
- `2026-03-28` 到 `2026-03-30`
  - 昇腾侧做背景处理或补充增强。
  - 检测侧做第二轮训练、误检漏检复核、导出 best 权重和样例结果。
- `2026-03-31`
  - 冻结提交版数据集。
  - 冻结检测模型和说明文档。

## Local-Only Workspace Layout

下面这些目录是固定入口，全部属于本地私有目录，不入 Git：

```text
data/bolt/
├─ source/
│  └─ seed_round/
│     ├─ images/
│     ├─ annotations/
│     └─ metadata/
├─ generate/
│  └─ sdxl/
│     ├─ incoming/
│     │  ├─ images/
│     │  └─ annotations/
│     ├─ repaired/
│     │  ├─ images/
│     │  └─ annotations/
│     ├─ review/
│     └─ accepted/
│        ├─ images/
│        └─ annotations/
├─ detect/
│  ├─ current/
│  │  ├─ images/
│  │  └─ annotations/
│  ├─ merged_20260327/
│  │  ├─ images/
│  │  └─ annotations/
│  └─ metadata/
└─ ascend/
   └─ background/
      ├─ incoming/
      └─ accepted/
```

## Data Handoff Checklist

你现在给大家发数据，直接按下面这张表发：

- 共同底座：
  - `data/bolt/source/seed_round/images`
  - `data/bolt/source/seed_round/annotations`
- 发给 `SDXL` 修复线：
  - `data/bolt/generate/sdxl/incoming/images`
  - `data/bolt/generate/sdxl/incoming/annotations`
- 发给 `YOLO11n` 训练线：
  - `data/bolt/detect/current/images`
  - `data/bolt/detect/current/annotations`
- 暂存通过人工复核的新图：
  - `data/bolt/generate/sdxl/accepted/images`
  - `data/bolt/generate/sdxl/accepted/annotations`
- `2026-03-27` 统一合流到：
  - `data/bolt/detect/merged_20260327/images`
  - `data/bolt/detect/merged_20260327/annotations`

## Role Contract

### 1. SDXL 修复主线

- 输入：
  - `data/bolt/source/seed_round/images`
  - `data/bolt/source/seed_round/annotations`
- 中间结果：
  - `data/bolt/generate/sdxl/repaired`
- 最终交付到：
  - `data/bolt/generate/sdxl/accepted`

要求：

- 只把通过人工复核的图放进 `accepted`。
- 图像和标注必须同名配套。

### 2. 原始图像增强同学

- 基于原始 `100` 张图工作。
- 产出优先汇入：
  - `data/bolt/generate/sdxl/review`
  - 通过复核后再进 `data/bolt/generate/sdxl/accepted`

要求：

- 不直接覆盖 seed 图。
- 每一批增强图要保留来源说明。

### 3. YOLO11n 训练同学

- 当前训练入口：
  - `data/bolt/detect/current`
- `2026-03-27` 后切换到：
  - `data/bolt/detect/merged_20260327`

推荐命令：

```powershell
python -m uv run python bolt/detect/scripts/run_detection_pipeline.py `
  --images-dir data/bolt/detect/current/images `
  --annotations data/bolt/detect/current/annotations `
  --prepared-root data/bolt/detect/prepared/current_round `
  --run-root bolt/detect/runs/current_round `
  --include-label faultScrew `
  --group-field sample_id `
  --copy-mode copy `
  --epochs 20 `
  --batch-size 4 `
  --imgsz 640 `
  --device 0
```

`2026-03-27` 合并新数据后，只替换 `--images-dir`、`--annotations` 和 `--prepared-root` / `--run-root`。

## Bootstrap Command

本地目录可以直接用下面这条命令初始化：

```powershell
python -m uv run python bolt/scripts/bootstrap_local_workspace.py --round-tag 2026-03-27
```

如果只想预览目录计划：

```powershell
python -m uv run python bolt/scripts/bootstrap_local_workspace.py --round-tag 2026-03-27 --dry-run
```
