# Project Memory

- date: 2026-03-22
- branch: main
- current_focus: missing_fastener_mainline
- dataset_status: detector line now prioritizes real box data first; healthy bolt assets are auxiliary and currently tracked separately
- competition_goal: initial round score is detection-first, with generation reduced to a minimal runnable companion package
- practical_route: freeze detector-facing dataset, run YOLO11n baseline, keep SAM2 and healthy-bolt assets as support lines
- layout_status: demo isolated under demo/, bolt mainline organized under bolt/
- privacy_rule: datasets, docs/规范, docs/规范_md, data outputs, and .env must never be committed
- detect_pipeline_status: prepare_detection_dataset supports VOC/COCO, group-aware split, and source-label filtering; run_detection_pipeline wires prepare/train/eval/infer end-to-end
- first_detector_run: on local GPU env with `faultScrew -> missing_fastener`, an early eval pass reached precision 0.7682, recall 0.5714, mAP50 0.6587, mAP50-95 0.2995
- latest_detector_eval: after wiring imgsz/device through eval and infer, the aligned `imgsz=640` val run reached precision 0.7405, recall 0.5764, mAP50 0.6225, mAP50-95 0.2534; treat this as the current baseline reference
