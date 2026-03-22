# SD Defect Project

一个面向缺陷数据增强与缺陷数据资产化的原型仓库。当前仓库已经拆成两条线：

- `bolt/`：挂点金具螺栓缺失主线
- `demo/`：裂缝 smoke demo，只负责最小可运行验证

## 当前状态

- 已完成 `uv` 环境管理接入。
- 已完成 demo 最小启动入口整理：`demo/generate_defect.py`。
- 已完成主线目录收口：`bolt/`。
- 已支持 `--dry-run`，可在不加载模型的情况下验证路径、配置和默认模型源。
- 已补最小测试，验证启动配置解析和 dry-run 行为。
- 真实推理路径仍依赖本机模型下载、网络和 GPU/CPU 环境，当前仓库里还没有把所有旧脚本统一到同一套工程规范。

## 项目目标

这个仓库的核心思路是：

1. 用真实或近真实的健康背景图作为底图。
2. 用 mask 指定缺陷发生区域。
3. 用 ControlNet / Inpainting 在局部区域生成缺陷纹理。
4. 输出增强图、调试图和后续可转成标注的数据。

当前优先支持的是“螺栓缺失三线架构”：

- 检测是最终交付主线
- 健康数据是负样本与生成母图支撑线
- SAM2 是像素级资产支撑线

裂缝 demo 仅作为最小启动与环境烟测。

## 仓库结构

```text
sd/
├─ bolt/                      # 螺栓缺失主线
│  ├─ dataset/
│  ├─ mask/
│  ├─ generate/
│  ├─ detect/
│  ├─ package/
│  ├─ docs/
│  └─ scripts/
├─ demo/                      # demo / smoke 路径
│  ├─ generate_defect.py
│  ├─ project_boot.py
│  ├─ batch_augment.py
│  └─ work_stream.py
├─ data/                      # 本地运行时目录；默认不纳入 Git
├─ tests/
│  └─ test_demo_boot.py       # demo 最小启动测试
├─ pyproject.toml             # uv 项目配置
└─ uv.lock                    # uv 锁文件
```

## 推荐入口

### 主线入口

主线工作统一在 `bolt/` 下推进：

- `bolt/dataset/`
- `bolt/mask/`
- `bolt/generate/`
- `bolt/detect/`
- `bolt/package/`
- `bolt/scripts/`

当前主线建议按以下关系推进：

- `bolt/detect/`：最终交付主线
- `bolt/dataset/`：缺陷数据治理与健康图种子集
- `bolt/mask/`：SAM2 资产 pilot

详细说明参见 `bolt/docs/mainline_architecture.md`。

### Demo 入口

当前推荐只使用下面这条路径启动项目：

```powershell
python -m uv run python demo/generate_defect.py --dry-run
```

这条命令会验证：

- 输入目录是否存在
- 输出目录是否可创建
- demo 图片和 mask 路径是否正确
- 当前默认模型源是否已解析

## 数据安全约定

- 仓库只提交代码、配置和文档，不提交任何真实数据集、标注文件、生成结果或中间产物。
- `data/`、`defect_images/`、`defect_xml/`、`healthy_images/`、`train_data_*`、`templates/`、`uploads/` 默认都被 `.gitignore` 排除。
- `bolt/dataset/raw/`、`bolt/dataset/annotations/`、`bolt/mask/assets/`、`bolt/mask/overlays/`、`bolt/mask/metadata/`、`bolt/generate/outputs/`、`bolt/detect/runs/`、`bolt/package/build/` 也默认不入 Git。
- `demo/generate_defect.py` 首次运行时会在本地自动创建最小 demo 输入，不依赖仓库内置图片。

## 环境准备

### 1. 安装 uv

如果你的 shell 里 `uv` 不在 PATH 上，直接用下面这种方式最稳：

```powershell
python -m pip install --user uv
python -m uv --version
```

### 2. 创建虚拟环境

```powershell
python -m uv venv
```

### 3. 安装依赖

```powershell
python -m uv sync
```

## 启动方式

### A. 先做 dry-run

```powershell
python -m uv run python demo/generate_defect.py --dry-run
```

预期输出是一段 JSON，包含：

- `mode: "dry-run"`
- `input_dir`
- `output_dir`
- `image_path`
- `mask_path`
- `base_model`
- `controlnet_model`

### B. 再尝试真实生成

```powershell
python -m uv run python demo/generate_defect.py
```

默认会使用：

- `runwayml/stable-diffusion-inpainting`
- `lllyasviel/control_v11p_sd15_canny`

首次运行通常会很慢，因为会涉及模型下载和缓存构建。

## 环境变量覆盖

如果你不想使用默认模型源，可以通过环境变量覆盖：

```powershell
$env:SD_BASE_MODEL="your/inpaint/model"
$env:SD_CONTROLNET_MODEL="your/controlnet/model"
python -m uv run python demo/generate_defect.py --dry-run
```

也可以直接通过命令行传参：

```powershell
python -m uv run python demo/generate_defect.py `
  --base-model your/inpaint/model `
  --controlnet-model your/controlnet/model `
  --dry-run
```

## 运行结果

默认输入输出位置：

- 输入目录：`data/sg/inputs`
- 输出目录：`data/sg/outputs`

默认输出文件：

- `data/sg/outputs/augmented_crack_sample_01.jpg`
- `data/sg/outputs/debug_control_canny.png`

这些文件都是本地运行时产物，不会被提交到仓库。

## 测试

当前最小测试命令：

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

当前测试覆盖的内容：

- 默认启动配置是否使用仓库相对路径
- CLI 和环境变量覆盖是否生效
- `demo/generate_defect.py --dry-run` 是否可以在不加载模型时启动成功

## 当前限制

这几个点要明确：

- 目前“可用”指的是 `uv + dry-run + 最小入口` 已经整理完成。
- `demo/generate_defect.py` 的真实生成链路已经接好，但是否在你的机器上稳定跑完，还取决于模型下载、网络、显卡和驱动环境。
- `demo/batch_augment.py`、`demo/work_stream.py` 仍保留较多旧路径和实验性写法。
- `bolt/scripts/` 已经是主线脚本区，但当前仍处在“数据资产优先、生成链路次之”的阶段。

## 主要脚本说明

### `demo/generate_defect.py`

当前唯一推荐启动入口。

职责：

- 创建 demo 所需的默认输入资源
- 支持 dry-run 配置检查
- 在真实运行时加载 Inpainting + ControlNet 做局部生成

### `demo/project_boot.py`

负责解析启动配置：

- repo 相对路径
- 输入输出目录
- 默认模型
- 环境变量覆盖

### `demo/batch_augment.py`

历史裂缝批量增强脚本。保留参考价值，但还没有完成当前阶段的路径治理。

### `demo/work_stream.py`

更早期的实验脚本，包含背景生成、随机掩码、缺陷注入和 YOLO 标注导出等逻辑。目前仍属于实验性实现。

### `bolt/scripts/`

主要是“缺失紧固件”方向的脚本，包括：

- ROI 裁剪
- 模板构建
- 模板匹配
- 单图推理
- 批量推理
- LoRA 训练脚本

这些脚本说明了项目的第二条路线，但不是当前最小启动目标。

## 下一步建议

当前最合理的推进顺序是：

1. 在 `bolt/dataset/` 下把缺陷图和框标注整理成检测可用数据集。
2. 在 `bolt/detect/` 下先跑一个轻量单类 detector baseline。
3. 同步收集小规模高匹配健康图种子集，用于负样本与后续生成母图。
4. 在 `bolt/mask/` 下做 SAM2 小批量 pilot，沉淀 `core_mask`、`edit_mask`、overlay 和 QA 记录。
5. 再决定何时把健康图和 SAM2 资产接入 `bolt/generate/`。

## 维护说明

如果只是想验证项目已经“能启动”：

```powershell
python -m uv sync
python -m uv run python demo/generate_defect.py --dry-run
python -m unittest discover -s tests -p "test_*.py" -v
```

如果这三步都通过，就说明当前 demo smoke 路径和 `uv` 管理仍然正常。
