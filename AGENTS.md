# Agent Guide

开始任何深度阅读、实现修改、代码审计或大范围检索前，先读 [.context/index.md](./.context/index.md)。

## Core Rules

- 私有数据永不入库：禁止提交 `data/`、`docs/规范/`、`docs/规范_md/`、`.env`、任何数据集原图、标注、mask、训练产物、评测产物。
- 当前主线聚焦“挂点金具螺栓缺失”方向，不再扩散到其他缺陷类型，除非用户明确切换。
- 当前工程优先级已经切到“检测优先、生成并行支撑”。
- 当前最优先资产化任务是：稳定检测数据入口、维持 `YOLO11n` baseline 可复现，并并行沉淀 `SAM2` mask 资产。
- “SAM2 mask 资产化”是当前治理目标，不代表仓库已经具备现成 SAM2 实现。
- 没有健康图时，不直接推进模板匹配和缺陷合成全流水线；先把可沉淀的标注资产、mask 资产、数据说明资产做扎实。
- 不自动 `git push`；只有用户明确要求时才推送远端。
- 格式以 [.editorconfig](./.editorconfig) 为唯一来源：`charset=utf-8`、`end_of_line=lf`、`indent_style=space`、`insert_final_newline=true`。

## Read Order

1. [.context/index.md](./.context/index.md)
2. [.context/filesystem_policy.md](./.context/filesystem_policy.md)
3. [docs/README.md](./docs/README.md)
4. [docs/repo_tree.md](./docs/repo_tree.md)
5. [README.md](./README.md)

## Repository Routes

- 公共文档入口：[docs/README.md](./docs/README.md)
- Demo 启动入口：[demo/generate_defect.py](./demo/generate_defect.py)
- Demo 启动配置：[demo/project_boot.py](./demo/project_boot.py)
- 主线脚本集合：[bolt/scripts](./bolt/scripts)
- 检测闭环入口：[bolt/detect/scripts/run_detection_pipeline.py](./bolt/detect/scripts/run_detection_pipeline.py)
- 主线文档入口：[bolt/docs/README.md](./bolt/docs/README.md)
- 测试入口：[tests/test_demo_boot.py](./tests/test_demo_boot.py)
- Demo 历史实验脚本：[demo/batch_augment.py](./demo/batch_augment.py)、[demo/work_stream.py](./demo/work_stream.py)
- 私有规范源目录：`docs/规范/`，只允许本地读取，不允许入库
- 私有规范转换目录：`docs/规范_md/`，只允许本地读取，不允许入库

## Mainline Focus

- 输入资源现状：当前仅有 100 张挂点金具螺栓缺失样本及对应标注。
- 关键短板：没有健康图，因此无法直接稳定完成“健康图匹配 -> 缺陷注入”主流水线。
- 当前建议路线：
  - 先稳住检测数据入口与 `YOLO11n` baseline。
  - 并行推进 `SDXL` 修复、原图增强和 `SAM2` 资产。
  - 在统一合流日期后再并入 detector 训练数据。
- 当前仓库现实：
  - `demo/generate_defect.py` 是最小 smoke 路径。
  - `bolt/scripts/` 是缺失紧固件主线实现区。
  - `bolt/detect/scripts/run_detection_pipeline.py` 是当前检测主线。
  - `demo/batch_augment.py` 和 `demo/work_stream.py` 默认视为 legacy。

## Verification

- Python 单测：`python -m unittest discover -s tests -p "test_*.py" -v`
- 最小 dry-run：`python -m uv run python demo/generate_defect.py --dry-run`
- 做治理类文档修改时，不需要跑训练；但提交前至少确认上述两条命令不被破坏。
- 检测脚本改动优先补跑：
  - `python -m uv run python -m unittest discover -s tests -p "test_detection_baseline_cli.py" -v`
  - `python -m uv run python -m unittest discover -s tests -p "test_run_detection_pipeline.py" -v`

## Context Tracks

- 项目入口索引：[.context/index.md](./.context/index.md)
- 文件系统策略：[.context/filesystem_policy.md](./.context/filesystem_policy.md)
- 项目记忆：[.context/memory.md](./.context/memory.md)
- 角色映射：[.context/agents/mapping.md](./.context/agents/mapping.md)
- 交接约定：[.context/contracts/handoffs.md](./.context/contracts/handoffs.md)
- 会话轨道：[.context/sessions](./.context/sessions)
- 流程轨道：[.context/flows](./.context/flows)
- 草稿轨道：[.context/scratch](./.context/scratch)
- 归档轨道：[.context/archive](./.context/archive)
- 项目测试注册表：[docs/project_test_registry.md](./docs/project_test_registry.md)
- 环境回归注册表：[docs/env_regression_registry.md](./docs/env_regression_registry.md)

## Governance Defaults

- `max_flows_default`: 12
- `expert_prompt_mode_default=summary`
- `daily_token_budget`: local session managed, avoid full-repo bulk reads
- `flow_iteration_cap`: 3

## Current Docs

- 总文档门户：[docs/README.md](./docs/README.md)
- 初赛技术路线：[docs/初赛技术路线README.md](./docs/初赛技术路线README.md)
- 初赛执行指南：[docs/初赛详细执行指南README.md](./docs/初赛详细执行指南README.md)
- 并行协作文档：[bolt/docs/round1_parallel_workflow.md](./bolt/docs/round1_parallel_workflow.md)
