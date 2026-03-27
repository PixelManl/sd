# PowerPaint V2 Batch Workflow

## Status

当前仓库已经接入 `PowerPaint V2` 批处理骨架，并且已经验证 `PowerPaint v2-1` 本地离线真实推理 smoke。

现在可用的是：

- manifest 驱动的批处理总控
- 串行处理同一张图上的 `K` 个标记
- 每步成功后同步改写标注
- `VOC XML` / `COCO JSON` 双格式支持
- `placeholder-copy` 占位后端
- `powerpaint-v2-1-offline` 真实离线后端
- `--dry-run` 路径校验与输出计划

现在还没有做的是：

- 服务器部署说明固化
- 更大批次的长期稳定性验证
- 与私有健康图批处理清单的正式合流

## Entry

主入口：

- `bolt/generate/scripts/run_powerpaint_v2_batch.py`

当前推荐先跑 dry-run：

```powershell
python -m uv run python bolt/generate/scripts/run_powerpaint_v2_batch.py `
  --manifest-path data/bolt/generate/powerpaint_v2/example_manifest.json `
  --output-dir data/bolt/generate/powerpaint_v2/runs/demo `
  --dry-run
```

如果本机已经装好真实 `PowerPaint v2-1` 环境，也可以直接走真实后端：

```powershell
python -m uv run python bolt/generate/scripts/run_powerpaint_v2_batch.py `
  --manifest-path data/bolt/generate/powerpaint_v2/example_manifest.json `
  --output-dir data/bolt/generate/powerpaint_v2/runs/demo_real `
  --backend-mode powerpaint-v2-1-offline
```

## Manifest Shape

manifest 支持两种 JSON 形状：

1. 顶层为数组
2. 顶层为对象，且包含 `records`

每条 record 表示“一张图上的一个待处理目标”。如果一张图有 `K` 个标记，就要展开成 `K` 条 record，并按处理顺序排列。

最小字段：

- `image_path`
- `annotation_format`：`voc` 或 `coco`
- `annotation_path`
- `target_id`
- `output_stem`

目标定位字段：

- VOC：
  - 推荐：`bbox` + `class_name`
  - 备选：`object_index`
- COCO：
  - 推荐：`annotation_id`
  - 备选：`image_id` + `bbox`
  - 可附加：`category_id` 或 `class_name`

示例：

```json
{
  "records": [
    {
      "image_path": "E:/data/bolt/healthy/sample.jpg",
      "annotation_format": "voc",
      "annotation_path": "E:/data/bolt/healthy/sample.xml",
      "target_id": "sample-target-001",
      "output_stem": "sample_step_001",
      "bbox": [10, 12, 30, 36],
      "class_name": "missing_fastener"
    },
    {
      "image_path": "E:/data/bolt/healthy/sample.jpg",
      "annotation_format": "voc",
      "annotation_path": "E:/data/bolt/healthy/sample.xml",
      "target_id": "sample-target-002",
      "output_stem": "sample_step_002",
      "bbox": [48, 52, 80, 92],
      "class_name": "missing_fastener"
    }
  ]
}
```

## Sequential Processing Rule

这是一个串行批处理任务：

- manifest 中第 `1` 条先处理
- 成功后输出新的图片和新的标注
- 同一张图的第 `2` 条 record 会以上一步输出作为输入继续处理
- 直到这张图的 `K` 个标记全部处理完

这就是“序列任务，并不阻塞主流程”的实现方式：

- 它不是检测主线的一部分
- 但你可以单独启动它，让它在本地按顺序把一个批次做完

## Output Layout

输出目录完全由 `--output-dir` 控制，默认应放在本地私有目录。

运行后会写出：

```text
<output-dir>/
├─ images/
├─ annotations/
└─ manifest_results.json
```

每条 record 至少会记录：

- `target_id`
- `status`
- `backend_mode`
- `source_image`
- `edited_image`
- `annotation_before`
- `annotation_after`
- `error_message`

## Annotation Rewrite Rule

### VOC XML

- 删除匹配目标框对应的 `<object>`
- 其他对象保持不变
- 原始 XML 不覆盖，只写新文件

### COCO JSON

- 删除匹配的 annotation item
- `images` 和 `categories` 保留
- 原始 JSON 不覆盖，只写新文件

## Backend Modes

当前实现了两个后端模式：

### `placeholder-copy`

行为：

- 不做真实图像编辑
- 只把当前图片复制为下一步输出图片
- 用于验证批处理顺序、产物路径、标注改写和错误恢复

### `powerpaint-v2-1-offline`

行为：

- 使用本地 `PowerPaint v2-1` checkpoint 做真实对象移除推理
- 不依赖官方 `app.py` 的在线模型下载路径
- 直接调用目标 Conda 环境里的 `python.exe`
- 适合当前“先本机跑通、后续再迁移服务器”的接法

record 里需要额外提供：

- `powerpaint_checkpoint_dir`
- `powerpaint_conda_prefix`
- `mask_path` 或 `mask_box`

可选参数：

- `powerpaint_repo_dir`
- `steps`
- `guidance_scale`
- `seed`

当前推荐的本机路径示例：

- `powerpaint_checkpoint_dir=E:/Repository/Project/sd/external/PowerPaint/checkpoints/ppt-v2-1`
- `powerpaint_conda_prefix=E:/Repository/Project/sd/external/PowerPaint/.conda-ppt`

补充说明：

- 如果已经有 `SAM2 edit_mask`，优先直接传 `mask_path`，不要退化回矩形 `mask_box`。
- 当 checkpoint 目录不是标准 `repo/checkpoints/ppt-v2-1` 结构时，可以显式补 `powerpaint_repo_dir`。
- 对高分辨率原图，当前服务器实测不建议直接整图推理；应先做 ROI crop，再在局部区域执行对象移除。

## Focused Verification

- `python -m uv run python -m unittest discover -s tests -p "test_powerpaint_v2_annotations.py" -v`
- `python -m uv run python -m unittest discover -s tests -p "test_run_powerpaint_v2_batch.py" -v`

如果要打开真实后端 smoke：

```powershell
$env:POWERPAINT_SMOKE="1"
$env:POWERPAINT_CHECKPOINT_DIR="E:\Repository\Project\sd\external\PowerPaint\checkpoints\ppt-v2-1"
$env:POWERPAINT_CONDA_PREFIX="E:\Repository\Project\sd\external\PowerPaint\.conda-ppt"
python -m uv run python -m unittest discover -s tests -p "test_run_powerpaint_v2_batch.py" -v
```

## Code Map

- `bolt/generate/powerpaint_v2_manifest.py`
- `bolt/generate/powerpaint_v2_backend.py`
- `bolt/generate/powerpaint_v2_annotations.py`
- `bolt/generate/scripts/run_powerpaint_v2_batch.py`
