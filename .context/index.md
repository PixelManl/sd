# Context Index

## Project Snapshot

- 项目：扩散模型与缺陷数据资产原型仓库
- 当前主线：挂点金具螺栓缺失
- 当前资源：100 张缺陷图，已具备框标注；暂无健康图
- 当前优先级：以检测为最终交付主线，同步推进健康图种子集与 SAM2 资产 pilot
- 当前事实边界：仓库里还没有现成的 SAM2 实现，SAM2 是主线目标而不是既成模块
- 当前工程状态：`bolt/detect/scripts/run_detection_pipeline.py` 已可串联 `prepare -> train -> eval -> infer`

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

1. 整理现有 100 张缺陷图与框标注，形成检测可用的数据集。
2. 拉起单类 `missing_fastener` 检测 baseline，先拿到真实误检漏检形态。
3. 同步收集小规模高匹配健康图种子集，用作负样本与后续生成母图。
4. 用 SAM2 做小批量资产 pilot，沉淀 `core_mask`、`edit_mask`、overlay 和 QA 记录。
5. 用检测结果反向指导健康图补集与 SAM2 资产优先级。

## Current Detector Baseline

- 已完成一轮真实本地训练闭环，源数据为 `2511bwb_5` 中的 `faultScrew` 单类筛选视图。
- 当前按训练同尺度 `imgsz=640` 对齐后的基线结果可作为下一轮补数和误检分析起点：`precision 0.7405 / recall 0.5764 / mAP50 0.6225 / mAP50-95 0.2534`。
- 训练产物、权重、推理图与评测 JSON 只允许保留在本地私有目录。
