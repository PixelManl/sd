# Bolt Mainline

`bolt/` 是挂点金具螺栓缺失主线工作区。

当前组织方式按任务阶段拆分：

- `dataset/`
- `mask/`
- `generate/`
- `detect/`
- `package/`
- `docs/`
- `scripts/`

当前执行原则：

- `detect/` 是最终交付主线
- `dataset/` 负责缺陷数据治理和健康图种子集
- `mask/` 负责 SAM2 像素级资产 pilot

详细关系参见 `bolt/docs/mainline_architecture.md`。

当前三人并行协作入口参见 [round1_parallel_workflow.md](/E:/Repository/Project/sd/bolt/docs/round1_parallel_workflow.md)。
本地私有目录可以通过 [bootstrap_local_workspace.py](/E:/Repository/Project/sd/bolt/scripts/bootstrap_local_workspace.py) 一键初始化。
