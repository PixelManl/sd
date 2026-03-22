# Agent Guide

开始任何深度阅读、实现修改、代码审计或大范围检索前，先读 [.context/index.md](./.context/index.md)。

## Core Rules

- 私有数据永不入库：禁止提交 `data/`、`docs/规范/`、`docs/规范_md/`、`.env`、任何数据集原图、标注、mask、训练产物、评测产物。
- 当前主线聚焦“挂点金具螺栓缺失”方向，不再扩散到其他缺陷类型，除非用户明确切换。
- 当前最优先资产化任务是：基于现有 100 张缺陷图，补全像素级 mask 资产；优先考虑 SAM2。
- “SAM2 mask 资产化”是当前治理目标，不代表仓库已经具备现成 SAM2 实现。
- 没有健康图时，不直接推进模板匹配和缺陷合成全流水线；先把可沉淀的标注资产、mask 资产、数据说明资产做扎实。
- 不自动 `git push`；只有用户明确要求时才推送远端。

## Read Order

1. [.context/index.md](./.context/index.md)
2. [.context/filesystem_policy.md](./.context/filesystem_policy.md)
3. [docs/repo_tree.md](./docs/repo_tree.md)
4. [README.md](./README.md)

## Repository Routes

- Demo 启动入口：[demo/generate_defect.py](./demo/generate_defect.py)
- Demo 启动配置：[demo/project_boot.py](./demo/project_boot.py)
- 主线脚本集合：[bolt/scripts](./bolt/scripts)
- 测试入口：[tests/test_demo_boot.py](./tests/test_demo_boot.py)
- Demo 历史实验脚本：[demo/batch_augment.py](./demo/batch_augment.py)、[demo/work_stream.py](./demo/work_stream.py)
- 私有规范源目录：`docs/规范/`，只允许本地读取，不允许入库
- 私有规范转换目录：`docs/规范_md/`，只允许本地读取，不允许入库

## Mainline Focus

- 输入资源现状：当前仅有 100 张挂点金具螺栓缺失样本及对应标注。
- 关键短板：没有健康图，因此无法直接稳定完成“健康图匹配 -> 缺陷注入”主流水线。
- 当前建议路线：
  - 先用 SAM2 生成像素级 mask，形成可复用资产。
  - 再整理数据集说明文档与提交目录规范。
  - 最后再评估是否补健康图或改走识别模型主线。
- 当前仓库现实：
  - `demo/generate_defect.py` 是最小 smoke 路径。
  - `bolt/scripts/` 是缺失紧固件主线实现区。
  - `demo/batch_augment.py` 和 `demo/work_stream.py` 默认视为 legacy。

## Verification

- Python 单测：`python -m unittest discover -s tests -p "test_*.py" -v`
- 最小 dry-run：`python -m uv run python demo/generate_defect.py --dry-run`
- 做治理类文档修改时，不需要跑训练；但提交前至少确认上述两条命令不被破坏。

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

## Missing Pieces

- `.context` 是本次刚建立的最小骨架，后续可按需要补细。
- 目前还没有独立的项目测试注册表与环境回归注册表；需要时再补。
