# Package Stage

这一层现在不再只是占位说明，而是初赛提交物的本地 staging 区。

核心原则只有两条：

- 所有打包与上传检查都先落到本地 `bolt/package/build/`
- `build/` 永远不入 Git，只服务本地交付和上传彩排

## 当前用途

`bolt/package/` 负责四件事：

1. 生成本地提交骨架
2. 生成私有资产盘点清单
3. 生成上传清单与打包检查单
4. 约束最终对外只导出三件成果物

## 推荐本地目录

```text
bolt/package/build/<tag>/
├─ model_pkg/
│  ├─ model/core/
│  ├─ code/
│  ├─ env/
│  ├─ ascend_image/
│  └─ docs/
├─ dataset_pkg/
│  ├─ DataFiles/
│  └─ Annotations/
├─ docs/
└─ checks/
   ├─ asset_inventory.json
   ├─ asset_inventory.md
   ├─ upload_manifest.csv
   └─ submission_checklist.md
```

## 推荐命令

先生成一轮本地 build 骨架和资产清单：

```powershell
python -m uv run python bolt/package/scripts/materialize_submission_scaffold.py `
  --tag 2026-03-24-upload-round-01 `
  --team-name your_team_name
```

这条命令会：

- 建立 `bolt/package/build/<tag>/`
- 扫描本地 `data/bolt_parallel/good_bolt_assets/`
- 生成资产统计、上传清单和打包检查单

## 与比赛主线的关系

- `checks/asset_inventory.md`
  给你看“这一轮到底有哪些资产已经整理到位”
- `checks/upload_manifest.csv`
  给你做云盘上传时的顺序和记录
- `checks/submission_checklist.md`
  给最终封板前做目录、命名、内容检查

## 对外最终成果物

对外始终只保留三件：

1. `团队名_生图模型成果物.zip`
2. `团队名_高质量数据集成果物.zip`
3. `团队名_高质量数据集说明文档.docx`

不要把下面这些东西直接混进最终包：

- mask
- overlay
- 中间日志
- review 截图
- 本地 build 清单
- 任意私有路径记录
