# Bolt Docs

这里放螺栓缺失主线专属文档。当前默认按“检测优先、生成并行支撑”组织。

## Current Entry

- 主线架构：[mainline_architecture.md](/E:/Repository/Project/sd/bolt/docs/mainline_architecture.md)
- 并行协作：[round1_parallel_workflow.md](/E:/Repository/Project/sd/bolt/docs/round1_parallel_workflow.md)
- 数据契约：[dataset_contract.md](/E:/Repository/Project/sd/bolt/docs/dataset_contract.md)
- 数据复核清单：[dataset_review_checklist.md](/E:/Repository/Project/sd/bolt/docs/dataset_review_checklist.md)
- SAM2 资产契约：[sam2_asset_contract.md](/E:/Repository/Project/sd/bolt/docs/sam2_asset_contract.md)
- 生成主线说明：[../generate/README.md](/E:/Repository/Project/sd/bolt/generate/README.md)
- SAM2 资产阶段说明：[../mask/README.md](/E:/Repository/Project/sd/bolt/mask/README.md)
- PowerPaint V2 批处理：[powerpaint_v2_batch_workflow.md](/E:/Repository/Project/sd/bolt/docs/powerpaint_v2_batch_workflow.md)

## Usage

- `detect/` 同学优先看数据契约和并行协作文档。
- `SDXL` / 增强同学优先看并行协作文档和主线架构。
- 如果要对外说明“当前 SAM2 用哪一个模型、当前 SDXL inpainting 在做什么”，优先引用 `mask/README.md` 和 `generate/README.md`。
- `PowerPaint V2` 接入和健康图转缺陷批处理，优先看 PowerPaint V2 批处理文档。
- 资产治理和提交说明继续沉淀在这个目录，不写进私有数据目录。
