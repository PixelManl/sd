# Dataset Stage

`bolt/dataset/` 是挂点金具螺栓缺失主线里的数据阶段工作区，目标不是把真实数据入库，而是把本地私有数据整理成检测主线可消费、可复核、可追溯的数据资产。

当前约束：

- 检测是比赛与仓库的最终交付主线。
- 官方监督口径是矩形框，不是 mask。
- 当前单类主线类别名固定为 `missing_fastener`。
- `bolt/dataset/raw/`、`bolt/dataset/annotations/`、`bolt/dataset/derived/` 默认视为本地私有目录，不提交 Git。

## 这一层负责什么

数据阶段当前只做三件事：

1. 整理现有缺陷图与框标注，形成检测可用的单类数据集。
2. 补齐最小必要的元数据、来源记录和划分信息，避免后续训练与评审时失真。
3. 给 `bolt/detect/` 输出统一的数据清单与派生产物，而不是在这里堆积实验结果。

不在这一层做的事：

- 不把 SAM2 mask 当作官方标注替代。
- 不在仓库里保存真实图像、真实标注文件、切片结果或训练产物。
- 不因为健康图暂时不足而阻塞缺陷检测主线。

## 本地私有目录建议

下面的目录是推荐布局，用于本地治理；目录名可以固定，真实内容不得提交：

```text
bolt/dataset/
├─ README.md
├─ raw/                    # 本地私有原始图像与原始收集记录
│  ├─ defect/
│  ├─ healthy_candidates/
│  └─ intake_notes/
├─ annotations/            # 本地私有官方框标注与人工修订结果
│  ├─ source/
│  ├─ reviewed/
│  └─ manifests/
└─ derived/                # 本地私有派生数据
   ├─ detector/
   ├─ stats/
   └─ review_exports/
```

建议理解如下：

- `raw/` 保存未加工原图、来源批次记录、采集说明。
- `annotations/` 保存矩形框标注原件、复核后版本、样本级元数据清单。
- `derived/` 只保存从私有原始资产派生出来的 detector-ready 结构、统计表和复核导出。

## 推荐工作流

推荐按下面顺序推进：

1. 把原始缺陷图放入 `raw/defect/`，按批次或采集来源记录来源信息。
2. 把原始框标注放入 `annotations/source/`，先不做格式大改。
3. 依据 [dataset_contract.md](/E:/Repository/Project/sd/bolt/docs/dataset_contract.md) 补齐样本级元数据，统一类名为 `missing_fastener`。
4. 先做人工复核，再做 train/val/test 划分，避免先切分后发现泄漏。
5. 将 detector 可直接消费的图像、标签、split manifest 和统计结果导出到 `derived/detector/`。
6. 用 [dataset_review_checklist.md](/E:/Repository/Project/sd/bolt/docs/dataset_review_checklist.md) 做抽检，确认图像质量、框质量与隐私边界。

## 与检测主线的接口

这一层对 `bolt/detect/` 的最小交付应该是：

- 单类 `missing_fastener` 的统一标签集。
- 稳定的 `train/val/test` 划分结果。
- 每张图像可追溯的样本 ID、场景 ID、来源信息。
- 面向训练脚本的 detector-ready 图像目录、标签目录和 split manifest。
- 一份简要统计说明，至少包含样本数、类别数、空图比例和基本框分布。

如果这些资产还没有准备好，不要先扩展检测结构；先把数据契约和复核闭环站稳。

## 当前并行协作入口

为了让 `SDXL` 修复、原图增强和 `YOLO11n` 训练三条线并行推进，当前本地目录统一按
[round1_parallel_workflow.md](/E:/Repository/Project/sd/bolt/docs/round1_parallel_workflow.md) 执行。

可以直接运行：

```powershell
python -m uv run python bolt/scripts/bootstrap_local_workspace.py --round-tag 2026-03-27
```

这条命令只会创建本地私有目录，不会生成任何需要提交的真实数据。
