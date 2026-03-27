# Project Test Registry

这个文件记录“仓库级”测试入口，也就是代码和文档修改后应该优先执行的自动化校验。

## Smoke

- `python -m uv run python demo/generate_defect.py --dry-run`
  - 用途：确认 demo 入口、配置解析和默认模型解析没有被破坏。

## Detection

- `python -m uv run python -m unittest discover -s tests -p "test_prepare_detection_dataset.py" -v`
  - 用途：确认检测数据准备脚本仍能处理 VOC / COCO、split 和标签筛选。
- `python -m uv run python -m unittest discover -s tests -p "test_detection_baseline_cli.py" -v`
  - 用途：确认 train / eval / infer CLI 合约没有破坏。
- `python -m uv run python -m unittest discover -s tests -p "test_run_detection_pipeline.py" -v`
  - 用途：确认 `prepare -> train -> eval -> infer` 总控透传没有破坏。

## Workspace Governance

- `python -m uv run python -m unittest discover -s tests -p "test_bootstrap_local_workspace.py" -v`
  - 用途：确认本地三人协作目录脚手架仍可创建。

## Generate

- `python -m uv run python -m unittest discover -s tests -p "test_powerpaint_v2_annotations.py" -v`
  - 用途：确认 PowerPaint V2 批处理的 VOC / COCO 标注改写仍然正确。
- `python -m uv run python -m unittest discover -s tests -p "test_run_powerpaint_v2_batch.py" -v`
  - 用途：确认 PowerPaint V2 批处理的 manifest 解析、dry-run 和同图串行处理没有破坏。

## Notes

- 不是每次都要跑全量测试，但至少要覆盖与你修改范围对应的测试。
- 训练、推理、评测真实产物都属于本地私有资产，不属于“项目测试注册表”的 Git 追踪对象。
