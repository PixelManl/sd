# Filesystem Policy

## Read Policy

- 先读 `.context/index.md`，再决定是否进入代码或文档细节。
- 默认优先读小文件、入口文件、测试文件。
- 对私有目录 `docs/规范/`、`docs/规范_md/`、`data/` 只做最小必要读取。

## Write Policy

- 可以写入治理文档、代码、测试。
- 不把任何私有数据、转换文档、训练产物纳入 Git 跟踪。
- 与主线相关的实验产物优先写到本地私有目录，不写到受版本控制路径。

## Ignore Zones

- `data/`
- `docs/规范/`
- `docs/规范_md/`
- `bolt/dataset/raw/`
- `bolt/dataset/annotations/`
- `bolt/dataset/derived/`
- `bolt/mask/assets/`
- `bolt/mask/overlays/`
- `bolt/mask/metadata/`
- `bolt/generate/outputs/`
- `bolt/generate/cache/`
- `bolt/detect/runs/`
- `bolt/package/build/`
- `.env`
- 模型缓存、训练输出、推理结果
