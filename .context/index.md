# Context Index

## Project Snapshot

- 项目：扩散模型与缺陷数据资产原型仓库
- 当前主线：挂点金具螺栓缺失
- 当前资源：100 张缺陷图，已具备框标注；暂无健康图
- 当前优先级：先把像素级 mask 做成资产，再决定生成链路或识别链路的推进顺序
- 当前事实边界：仓库里还没有现成的 SAM2 实现，SAM2 是主线目标而不是既成模块

## Private Data Rules

- `docs/规范/` 与 `docs/规范_md/` 是本地私有资料目录，不入 Git。
- `data/` 下所有原图、标注、mask、输出结果都视为私有资产，不入 Git。
- `.env` 只允许本地读取，不允许复制到文档、日志、提交记录或外部服务。

## Current Competition Understanding

- 初赛要交三类成果：生图模型、高质量数据集、数据集说明文档。
- 复赛要交两类成果：缺陷识别模型、高质量数据集。
- 本项目当前最有把握沉淀的资产是“高质量数据集”部分，尤其是高质量标注和像素级 mask。
- 挂点金具螺栓缺失的原始标注规则当前只支持矩形框；像素级 mask 属于我们自行增强的资产能力。

## Repo Map

- 文档树：参见 [../docs/repo_tree.md](../docs/repo_tree.md)
- Demo 启动入口：[../demo/generate_defect.py](../demo/generate_defect.py)
- Demo 配置入口：[../demo/project_boot.py](../demo/project_boot.py)
- 缺失紧固件主线脚本：[../bolt/scripts](../bolt/scripts)
- 测试：[../tests/test_demo_boot.py](../tests/test_demo_boot.py)
- Demo 历史实验脚本：[../demo/batch_augment.py](../demo/batch_augment.py)、[../demo/work_stream.py](../demo/work_stream.py)

## Mainline Plan

1. 基于现有 100 张缺陷图，用 SAM2 获取像素级 mask。
2. 将 mask、可视化叠图、质量检查记录保存为本地资产。
3. 整理高质量数据集说明文档需要的统计项与样例。
4. 再判断是补健康图推进生成链路，还是先转向识别模型主线。
