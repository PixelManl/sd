# 螺栓缺失数据集契约

## 目标

本文档定义 `bolt/dataset/` 阶段的最小私有数据契约，用于把当前“挂点金具螺栓缺失”样本整理成检测主线可直接消费的数据资产。

适用范围：

- 当前主线只覆盖单类缺陷 `missing_fastener`。
- 官方监督口径是矩形框。
- mask、overlay、SAM2 metadata 属于增强资产，不是本文档里的官方标注主体。
- 文档描述的是本地私有布局与字段约定，不代表任何真实数据应入库。

## 基本原则

1. 真实图像、真实标注、review 导出、统计表、切分结果都默认保留在本地私有目录，不提交 Git。
2. 所有 detector-facing 标签都统一映射到单一类别名 `missing_fastener`，禁止同义词混用。
3. 划分规则优先避免场景泄漏，其次才考虑数量均衡。
4. 派生输出必须可回溯到原始图像、原始标注和人工复核记录。

## 私有目录布局

推荐使用下面的本地私有布局：

```text
bolt/dataset/
├─ raw/
│  ├─ defect/                          # 缺陷原图
│  ├─ healthy_candidates/              # 健康候选原图
│  └─ intake_notes/                    # 来源说明、采集批次、授权记录
├─ annotations/
│  ├─ source/                          # 原始框标注
│  ├─ reviewed/                        # 人工复核后的标注
│  └─ manifests/                       # 样本级元数据、split 清单
└─ derived/
   ├─ detector/
   │  ├─ images/
   │  │  ├─ train/
   │  │  ├─ val/
   │  │  └─ test/
   │  ├─ labels/
   │  │  ├─ train/
   │  │  ├─ val/
   │  │  └─ test/
   │  ├─ manifests/
   │  └─ stats/
   ├─ review_exports/
   └─ snapshots/
```

约定说明：

- `raw/` 只接受未经训练脚本改写的原始输入。
- `annotations/source/` 保留原标注来源；`annotations/reviewed/` 才是下游训练的标注基线。
- `derived/detector/` 是提供给检测脚本的统一导出区，不反向覆盖原始资产。

## 规范对象与命名

### 规范类别名

当前 canonical class name 固定为：

```text
missing_fastener
```

执行要求：

- detector 数据导出时只允许一个正类：`missing_fastener`。
- 中文名称如“螺栓缺失”“紧固件缺失”只用于说明，不进入训练标签文件。
- 如果原始标注里出现其他别名，必须在导出前完成统一映射。

### 样本 ID

每张图像至少要有一个稳定的 `sample_id`，建议格式如下：

```text
mf_<source_batch>_<scene_id>_<frame_or_index>
```

要求：

- 同一图像在不同阶段使用同一个 `sample_id`。
- 文件名变化不应导致 `sample_id` 变化。
- `sample_id` 必须能映射回原始图像路径和标注来源。

## 必需元数据

下面字段是当前阶段的最小必填集。可以存成 JSON、CSV 或 JSONL，但字段语义必须一致。

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `sample_id` | 是 | 样本唯一 ID |
| `image_relpath` | 是 | 相对私有数据根目录的图像路径 |
| `annotation_relpath` | 是 | 相对私有数据根目录的标注路径；无框时也要显式记录 |
| `class_name` | 是 | 固定为 `missing_fastener` 或空样本约定值 |
| `is_positive` | 是 | 是否包含缺失目标 |
| `scene_id` | 是 | 同一拍摄场景或同一挂点的稳定分组 ID |
| `capture_group_id` | 是 | 同一连续拍摄批次、视频片段或相邻帧组 ID |
| `source_batch` | 是 | 采集批次、外部来源批次或整理批次 |
| `image_width` | 是 | 图像宽度 |
| `image_height` | 是 | 图像高度 |
| `bbox_count` | 是 | 框数量；空图填 `0` |
| `review_status` | 是 | 如 `pending`、`approved`、`needs_fix`、`rejected` |
| `reviewer` | 否 | 复核人 |
| `review_time` | 否 | 复核时间 |
| `privacy_flag` | 是 | 是否含敏感信息；默认需要显式判断 |
| `split` | 否 | `train`、`val`、`test`，建议在复核完成后再填 |
| `notes` | 否 | 特殊说明，如遮挡、模糊、可疑误标 |

补充要求：

- `is_positive = false` 的样本也要记录 `scene_id` 和 `capture_group_id`，否则无法做防泄漏切分。
- `bbox_count > 0` 时，`class_name` 必须为 `missing_fastener`。
- 负样本或空框样本不得人为引入第二个类别名。
- `privacy_flag` 不能省略；不确定时先标为需要复核。

## 标注要求

当前官方可交监督为矩形框，因此这里的标注要求也围绕框展开：

- 框必须覆盖可见的“缺失证据区域”，不要用大框把无关背景一起吞掉。
- 一个缺失实例对应一个框；不允许多个独立缺失目标共用一个框。
- 框坐标必须落在图像范围内。
- 模糊、遮挡、远距离样本可以保留，但要通过 `review_status` 和 `notes` 标出风险。
- SAM2 产生的 mask 只能作为补充审阅材料，不能回写成官方框标注真值。

## 防泄漏划分规则

train/val/test 划分必须遵守下面规则：

1. 以 `scene_id` 作为最小隔离单元，同一 `scene_id` 只能出现在一个 split。
2. 如果缺少稳定 `scene_id`，退化到 `capture_group_id`；同一连续拍摄组不能跨 split。
3. 同一设备、同一杆塔、同一挂点、同一视频相邻帧、同一连拍序列，默认视为同组样本。
4. 同一原图裁切出的 patch、不同尺寸重采样图、仅做轻微增强的副本，不得跨 split。
5. 先完成人工复核与去重，再输出 split；不能一边补标一边随意改 split。
6. 健康候选样本如果进入 detector 负样本，也必须服从同样的 `scene_id` / `capture_group_id` 隔离规则。

建议执行方式：

- 先生成去重后的样本清单。
- 再按组级别划分 train/val/test。
- 最后导出 split manifest，并冻结版本号或日期标签。

## Detector-facing 输出

数据阶段对检测主线的最小输出应包括以下内容：

### 1. 标准图像目录

- `derived/detector/images/train/`
- `derived/detector/images/val/`
- `derived/detector/images/test/`

### 2. 标准标签目录

- `derived/detector/labels/train/`
- `derived/detector/labels/val/`
- `derived/detector/labels/test/`

标签要求：

- 只包含 `missing_fastener` 一个类别。
- 坐标格式在单次导出内必须保持一致，不允许混用两套格式。
- 每个标签文件都能通过 `sample_id` 或文件名回溯到元数据清单。

### 3. 样本清单与划分清单

至少提供：

- `all_samples`：全量样本元数据表
- `train_manifest`
- `val_manifest`
- `test_manifest`

每条记录至少带上：

- `sample_id`
- `scene_id`
- `capture_group_id`
- `is_positive`
- `bbox_count`
- `split`
- `review_status`

### 4. 最小统计产物

建议至少输出：

- 总图像数、正样本数、空图数
- train/val/test 样本数
- 平均每图框数
- 框尺寸或面积占比的基本分布
- 被拒样本数与主要原因

## 不应提交到仓库的内容

下面内容必须留在本地私有空间，不进入 Git：

- 原始图像
- 原始标注文件
- reviewed 标注文件
- split manifest 真数据
- 任何包含真实路径、真实设备编号、真实采集批次详情的清单
- 统计导出、可视化导出、审阅截图

仓库里允许提交的只应是：

- 目录说明
- 契约文档
- 脚本骨架
- 字段定义
- 不含真实数据的模板示例
