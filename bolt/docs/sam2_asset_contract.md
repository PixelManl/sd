# SAM2 Asset Contract

## Status

本文件定义 `挂点金具螺栓缺失` 方向的 SAM2 资产试点合同。

当前范围是：

- 统一输出物语义
- 统一 metadata 字段
- 统一 QA 状态
- 统一 pilot 工作流

当前不包含：

- 真实 SAM2 推理接入
- 官方标注替代方案
- 大规模批处理生产规范

## Current Runtime Note

虽然本合同本身不强绑定权重，但当前仓库里已经存在一个可接真实推理的默认后端约定：

- model family：`SAM2.1`
- default variant：`hiera tiny`
- default config：`configs/sam2.1/sam2.1_hiera_t.yaml`
- default checkpoint：`/root/sam2-local/checkpoints/sam2.1_hiera_tiny.pt`

对应代码入口：

- `bolt/mask/scripts/good_bolt_sam2_box_prompt_backend.py`

因此当前共享口径应当写成“默认是 `SAM2.1 tiny`，可被环境变量覆盖”，而不是泛称 “SAM2.1”。

## Contract Scope

SAM2 在本仓库里被视作内部资产线，而不是官方 annotation replacement。

它的目标是把代表性缺陷 ROI 沉淀为本地私有资产，支撑后续：

- 局部编辑
- ROI 审核
- 结构先验沉淀
- 受控增强实验

因此所有合同字段都围绕“本地私有资产 bundle”设计，而不是围绕比赛提交格式设计。

## Output Bundle

一个最小资产 bundle 由以下对象组成：

| Object | Required | Description |
| --- | --- | --- |
| `core_mask` | yes | 仅覆盖可见缺失证据的保守二值 mask |
| `edit_mask` | yes | 大于等于 `core_mask` 的局部编辑区 mask |
| `overlay` | yes | 供人工复核的可视化叠图 |
| `metadata` | yes | 逐资产 JSON 元数据 |
| `manifest_entry` | optional | 导出到聚合 manifest 时的标准记录 |

### `core_mask` Semantics

`core_mask` 必须满足：

- 只圈定缺失螺栓或直接缺失证据
- 优先保守，不为了“好看”扩大背景
- 不把大面积结构上下文并入核心证据区
- 如果存在不确定边缘，应宁可偏紧，不应偏松

### `edit_mask` Semantics

`edit_mask` 必须满足：

- 完全覆盖 `core_mask`
- 覆盖后续局部编辑所需的上下文缓冲区
- 允许包含结构边界附近的少量背景
- 目标是提供可编辑区域，而不是精确证据边界

一句话区分：

- `core_mask` 回答“缺失证据在哪里”
- `edit_mask` 回答“允许编辑到哪里”

## Recommended Local Layout

以下路径是推荐的本地私有布局，不代表必须在仓库内持久化：

```text
bolt/mask/
├─ assets/
│  └─ <asset_id>/
│     ├─ core_mask.png
│     └─ edit_mask.png
├─ overlays/
│  └─ <asset_id>/
│     └─ review_overlay.png
└─ metadata/
   ├─ <asset_id>.json
   └─ manifests/
      ├─ sam2_pilot_manifest.json
      └─ sam2_pilot_manifest.jsonl
```

这些目录都是私有资产目录，默认不应纳入版本控制。

## Metadata Schema

每个资产至少应对应一份 JSON metadata。字段合同如下：

| Field | Required | Type | Meaning |
| --- | --- | --- | --- |
| `contract_version` | yes | string | 当前合同版本，例如 `sam2_asset_contract/v1` |
| `asset_line` | yes | string | 固定标识内部资产线，例如 `sam2_pilot` |
| `asset_id` | yes | string | 单资产唯一 ID |
| `pilot_run_id` | yes | string | 试点批次 ID |
| `defect_type` | yes | string | 当前应为 `missing_fastener` 或等价受控值 |
| `source_image` | yes | string | 原图路径或相对路径 |
| `source_image_sha256` | no | string | 原图摘要，用于去重和追溯 |
| `roi_id` | no | string | ROI 或候选框 ID |
| `core_mask_path` | yes | string | `core_mask` 文件路径 |
| `edit_mask_path` | yes | string | `edit_mask` 文件路径 |
| `overlay_path` | yes | string | review overlay 文件路径 |
| `qa_state` | yes | string | 资产当前 QA 状态 |
| `qa_notes` | no | string | 审核说明 |
| `reviewer` | no | string | 审核人标识 |
| `lineage_parent` | no | string | 父资产或来源资产 ID |
| `tool_name` | no | string | 生产工具名，例如 `sam2-placeholder` |
| `tool_version` | no | string | 工具版本 |
| `created_at` | yes | string | ISO 8601 时间 |
| `updated_at` | yes | string | ISO 8601 时间 |

### Field Rules

- `asset_id` 在 pilot 范围内必须唯一。
- `defect_type` 不应扩散到当前主线之外的缺陷类型。
- `core_mask_path`、`edit_mask_path`、`overlay_path` 应指向本地私有资产，不应指向 Git 跟踪文档目录。
- `qa_state` 变更时必须同步更新 `updated_at`。
- `lineage_parent` 用于记录二次编辑或返工资产的来源。

## QA States

试点阶段统一使用以下状态：

| State | Meaning |
| --- | --- |
| `planned` | 已进入 pilot 计划，但尚未产出资产 |
| `draft` | 已有初稿资产，未完成人工复核 |
| `needs_review` | 资产已提交待审，等待 QA 判断 |
| `accepted` | 通过本轮 QA，可进入下游本地使用 |
| `rejected` | 明确不合格，需要重做或废弃 |
| `superseded` | 已被更新资产替代，不再作为主版本使用 |

状态转换建议：

- `planned -> draft`
- `draft -> needs_review`
- `needs_review -> accepted`
- `needs_review -> rejected`
- `accepted -> superseded`
- `rejected -> draft`

## Pilot Workflow

当前试点流程按以下五步执行：

1. 选取小批量代表性样本，而不是一开始全量资产化。
2. 运行 `run_sam2_pilot.py` 做路径校验、样本收集和输出规划。
3. 在本地私有环境中补齐 `core_mask`、`edit_mask`、overlay 和 metadata。
4. 运行 `review_sam2_assets.py` 汇总 QA 状态、检查缺失路径和 metadata 完整度。
5. 运行 `export_sam2_manifest.py` 生成聚合 manifest，供下游本地流程读取。

### Pilot Principles

- 先把合同和审阅机制跑通，再考虑真实模型接入。
- 先做代表性 ROI，不做无差别铺量。
- 任何自动化结果都要经过人工 QA，而不是默认可信。
- manifest 只是本地资产目录的映射，不是官方标注产物。

## Manifest Entry Shape

导出 manifest 时，每条记录至少包含：

```json
{
  "contract_version": "sam2_asset_contract/v1",
  "asset_line": "sam2_pilot",
  "asset_id": "example-asset-001",
  "pilot_run_id": "pilot-20260322-01",
  "defect_type": "missing_fastener",
  "source_image": "data/private/source/example.jpg",
  "core_mask_path": "bolt/mask/assets/example-asset-001/core_mask.png",
  "edit_mask_path": "bolt/mask/assets/example-asset-001/edit_mask.png",
  "overlay_path": "bolt/mask/overlays/example-asset-001/review_overlay.png",
  "qa_state": "needs_review",
  "updated_at": "2026-03-22T12:00:00Z"
}
```

manifest 可以导出为：

- 单个 JSON 文档
- JSONL 逐行记录

两者都属于本地私有资产索引，不应误认为仓库文档。

## Current Non-Goals

本轮明确不做以下内容：

- 不绑定 SAM2 官方仓库或权重
- 不引入 GPU、CUDA 或交互 UI 依赖
- 不承诺自动生成任何可直接训练的高质量像素标注
- 不把私有资产目录纳入 Git

当前的成功标准只有两个：

- 合同定义清楚
- CLI 骨架可在无 SAM2 依赖时安全运行
