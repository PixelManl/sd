# 2026-03-24 Detection Mainline

## Goal

把检测主线从“分散脚本”收敛成一条可以直接执行的闭环：

- 数据准备
- 训练
- 评估
- 推理导出

## Decisions

- 初赛当前以检测提交为第一优先级。
- 单类目标仍固定为 `missing_fastener`。
- 当源数据是多类别 VOC / COCO 时，必须先用源标签筛出真正的缺失螺栓正类，再统一映射到 `missing_fastener`。
- 如果没有稳定元数据，首轮先退化到 `sample_id` 级切分，优先把训练闭环跑通。

## Implemented

- `bolt/detect/scripts/prepare_detection_dataset.py`
  - 支持 `VOC XML dir` 和 `COCO JSON`
  - 支持 `--include-label`
  - 支持 `group-aware split`
- `bolt/detect/scripts/run_detection_pipeline.py`
  - 串联 `prepare -> train -> eval -> infer`
  - 在非 dry-run 下写出 `pipeline_summary.json`

## Current Local Training Target

- 候选数据：`C:\Users\21941\Desktop\2511bwb_5\2511bwb_5`
- 结构：`JPEGImages/` + `Annotations/`
- 已观察到源标签包含：
  - `faultScrew`
  - `normScrew`
  - 以及其他正常/故障部件类
- 首轮训练策略：
  - `--include-label faultScrew`
  - `class_name = missing_fastener`
  - `group_field = sample_id`

## Notes

- 本轮训练结果、权重、推理图、metrics 仍然只允许留在本地私有目录。
- 后续如果用户补全 `scene_id / capture_group_id` 元数据，应优先切换到组级切分。
- 当前首轮真实基线指标：
  - precision: `0.7405`
  - recall: `0.5764`
  - mAP50: `0.6225`
  - mAP50-95: `0.2534`
- 更早一版未对齐尺度的评估记录仍保留在本地，不能再当作“当前基线”引用。
- 当前待收敛点：
  - 评估与推理必须跟随训练的 `imgsz/device` 覆盖，避免尺度不一致。
  - `pipeline_summary.json` 只保留压缩摘要，详细预测列表继续单独写在推理目录内。
