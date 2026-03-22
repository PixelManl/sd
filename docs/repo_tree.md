# Repository Tree

```text
sd/
├─ AGENTS.md
├─ .context/
│  ├─ index.md
│  ├─ filesystem_policy.md
│  ├─ memory.md
│  ├─ agents/
│  │  └─ mapping.md
│  ├─ contracts/
│  │  └─ handoffs.md
│  ├─ flows/
│  ├─ sessions/
│  ├─ scratch/
│  └─ archive/
├─ bolt/
│  ├─ dataset/              # 主线数据阶段
│  ├─ mask/                 # SAM2 mask 资产阶段
│  ├─ generate/             # 健康图匹配与缺陷生成阶段
│  ├─ detect/               # 缺陷识别阶段
│  ├─ package/              # 比赛提交物阶段
│  ├─ docs/                 # 主线文档
│  └─ scripts/              # 缺失紧固件主线脚本
├─ demo/
│  ├─ generate_defect.py    # demo smoke 入口
│  ├─ project_boot.py       # demo 配置解析
│  ├─ batch_augment.py      # demo/legacy 脚本
│  └─ work_stream.py        # demo/legacy 脚本
├─ docs/
│  ├─ repo_tree.md
│  ├─ superpowers/
│  ├─ 规范/                 # 私有源文档，不入库
│  └─ 规范_md/              # 私有转换文档，不入库
├─ tests/                   # 最小自动化校验
├─ pyproject.toml           # uv 项目配置
├─ uv.lock
└─ data/                    # 本地私有数据，不入库
```

## Current Mainline

- 目标缺陷：挂点金具螺栓缺失
- 当前资源：100 张缺陷图
- 当前短板：缺少健康图
- 当前建议：先做 SAM2 像素级 mask 资产化
- 当前结构：`bolt/` 是主线，`demo/` 只保留最小演示与 smoke 路径
