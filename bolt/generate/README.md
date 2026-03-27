# Generate Stage

这一层负责健康图匹配、缺陷注入和图像生成。

建议本地私有目录：

- `outputs/`：生成结果
- `cache/`：中间缓存

这些目录默认不入 Git。

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
