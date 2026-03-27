# Docs Portal

这是仓库的公共文档入口。默认只放可提交、可共享、可治理的文档，不放任何私有数据、私有规范原文或训练产物。

## Read First

1. [../.context/index.md](../.context/index.md)
2. [repo_tree.md](./repo_tree.md)
3. [../README.md](../README.md)

## Current Mainline

当前主线已经切到“检测优先，生成并行支撑”：

- 检测闭环入口：[../bolt/detect/scripts/run_detection_pipeline.py](../bolt/detect/scripts/run_detection_pipeline.py)
- 主线文档入口：[../bolt/docs/README.md](../bolt/docs/README.md)
- 并行协作说明：[../bolt/docs/round1_parallel_workflow.md](../bolt/docs/round1_parallel_workflow.md)

## Data Handoff

团队发数据时，直接看根 README 的“当前数据位置”：

- [../README.md](../README.md)

当前最关键的三个入口：

- 原始 `100` 张图与标注：`data/bolt/source/seed_round/`
- 当前训练入口：`data/bolt/detect/current/`
- `2026-03-27` 合流入口：`data/bolt/detect/merged_20260327/`

## Competition Docs

- 初赛技术路线：[初赛技术路线README.md](./初赛技术路线README.md)
- 初赛详细执行指南：[初赛详细执行指南README.md](./初赛详细执行指南README.md)

## Governance Docs

- 项目树说明：[repo_tree.md](./repo_tree.md)
- 项目测试注册表：[project_test_registry.md](./project_test_registry.md)
- 环境回归注册表：[env_regression_registry.md](./env_regression_registry.md)

## Private Zones

下面两个目录只允许本地读取，不允许入库：

- `docs/规范/`
- `docs/规范_md/`

不要把这两个目录里的原文、截图、解析结果复制进公共 README、提交记录或 Git 路径。
