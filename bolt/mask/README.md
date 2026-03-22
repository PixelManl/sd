# Mask Stage

`bolt/mask/` 是挂点金具螺栓缺失方向的像素级资产阶段，用来沉淀内部可复用 mask 资产。

这里的 SAM2 线有两个明确边界：

- 它是内部资产线，不是比赛官方标注替代。
- 它服务于后续局部编辑、ROI 审核和受控增强，不直接替代框标注主线。

## Purpose

这一层当前只做 `SAM2 asset-contract + pilot skeleton`，目标是先把资产格式、元数据、质检状态和最小 CLI 骨架定下来，再决定后续是否接入真实 SAM2 推理或交互流程。

当前约定的两类核心输出：

- `core_mask`：紧贴可见缺失证据的保守二值 mask，优先保证语义准确，不主动向背景扩张。
- `edit_mask`：覆盖 `core_mask` 且略大一圈的局部编辑区域，用于后续 inpainting、局部替换和更宽松的结构修补。

## Recommended Private Layout

以下目录建议仅作为本地私有资产目录使用：

- `assets/`：保存 `core_mask`、`edit_mask` 等二值 mask。
- `overlays/`：保存叠图、轮廓图、局部审阅图。
- `metadata/`：保存逐资产 JSON 元数据、QA 记录和导出的 manifest。
- `scripts/`：保存试点阶段命令行骨架，不依赖真实 SAM2 安装。

这些目录中的资产默认不应入 Git；尤其是 `assets/`、`overlays/`、`metadata/` 下的运行产物必须保留在本地。

## Outputs

一次完整的本地资产 bundle 通常包含：

- 一个 `core_mask`
- 一个 `edit_mask`
- 至少一个 review overlay
- 一份逐资产 metadata JSON
- 可选的导出 manifest，用于下游本地消费

推荐关系如下：

- `edit_mask` 必须完全覆盖 `core_mask`
- overlay 只用于人工复核，不作为正式标注源
- metadata 负责说明来源、版本、QA 状态和 lineage

详细字段与工作流定义见 [sam2_asset_contract.md](../docs/sam2_asset_contract.md)。

## Pilot Scripts

当前只提供安全占位脚本，不绑定任何真实 SAM2 依赖：

- `scripts/run_sam2_pilot.py`：校验输入路径、收集样本、输出 pilot plan
- `scripts/review_sam2_assets.py`：读取 metadata，汇总 QA 状态和路径缺口
- `scripts/export_sam2_manifest.py`：把 metadata 汇总导出为 JSON 或 JSONL manifest

这些脚本的设计原则是：

- 没有真实 SAM2 也能运行
- 默认只做计划、校验、汇总和导出
- 不把“已经有 CLI”误解成“已经完成 SAM2 集成”

## Boundaries

这一层当前不承担以下职责：

- 不生成官方比赛标注替代物
- 不承诺自动生成高质量 mask
- 不在本轮接入模型权重、GPU 推理或交互式分割
- 不把私有资产写入受版本控制路径

如果后续接入真实 SAM2，也必须继续遵守这里的合同，而不是反过来让工具实现绑架资产格式。
