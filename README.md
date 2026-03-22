# SD Defect Demo

一个面向缺陷数据增强的扩散模型原型仓库。当前先以“建筑裂缝 demo”作为最小启动入口，并保留“缺失紧固件”方向的脚本雏形。

## 当前状态

- 已完成 `uv` 环境管理接入。
- 已完成最小启动入口整理：`generate_defect.py`。
- 已支持 `--dry-run`，可在不加载模型的情况下验证路径、配置和默认模型源。
- 已补最小测试，验证启动配置解析和 dry-run 行为。
- 真实推理路径仍依赖本机模型下载、网络和 GPU/CPU 环境，当前仓库里还没有把所有旧脚本统一到同一套工程规范。

## 项目目标

这个仓库的核心思路是：

1. 用真实或近真实的健康背景图作为底图。
2. 用 mask 指定缺陷发生区域。
3. 用 ControlNet / Inpainting 在局部区域生成缺陷纹理。
4. 输出增强图、调试图和后续可转成标注的数据。

当前优先支持的是“裂缝 demo 启动”，不是全量流水线的一次性工程化。

## 仓库结构

```text
sd/
├─ data/                      # 本地运行时目录；默认不纳入 Git
├─ scripts/                   # 缺失紧固件方向脚本与训练脚本
├─ tests/
│  └─ test_project_boot.py    # 最小启动测试
├─ generate_defect.py         # 当前推荐启动入口
├─ project_boot.py            # 启动配置解析
├─ pyproject.toml             # uv 项目配置
└─ uv.lock                    # uv 锁文件
```

## 推荐入口

当前推荐只使用下面这条路径启动项目：

```powershell
python -m uv run python generate_defect.py --dry-run
```

这条命令会验证：

- 输入目录是否存在
- 输出目录是否可创建
- demo 图片和 mask 路径是否正确
- 当前默认模型源是否已解析

## 数据安全约定

- 仓库只提交代码、配置和文档，不提交任何真实数据集、标注文件、生成结果或中间产物。
- `data/`、`defect_images/`、`defect_xml/`、`healthy_images/`、`train_data_*`、`templates/`、`uploads/` 默认都被 `.gitignore` 排除。
- `generate_defect.py` 首次运行时会在本地自动创建最小 demo 输入，不依赖仓库内置图片。

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
python -m uv run python generate_defect.py --dry-run
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
python -m uv run python generate_defect.py
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
python -m uv run python generate_defect.py --dry-run
```

也可以直接通过命令行传参：

```powershell
python -m uv run python generate_defect.py `
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
- `generate_defect.py --dry-run` 是否可以在不加载模型时启动成功

## 当前限制

这几个点要明确：

- 目前“可用”指的是 `uv + dry-run + 最小入口` 已经整理完成。
- `generate_defect.py` 的真实生成链路已经接好，但是否在你的机器上稳定跑完，还取决于模型下载、网络、显卡和驱动环境。
- 历史脚本如 `batch_augment.py`、`work_stream.py` 仍保留较多旧路径和实验性写法，还没有统一重构。
- `scripts/` 目录下的“缺失紧固件”流程是下一阶段工程化对象，目前不建议直接当作开箱即用流程。

## 主要脚本说明

### `generate_defect.py`

当前唯一推荐启动入口。

职责：

- 创建 demo 所需的默认输入资源
- 支持 dry-run 配置检查
- 在真实运行时加载 Inpainting + ControlNet 做局部生成

### `project_boot.py`

负责解析启动配置：

- repo 相对路径
- 输入输出目录
- 默认模型
- 环境变量覆盖

### `batch_augment.py`

历史裂缝批量增强脚本。保留参考价值，但还没有完成当前阶段的路径治理。

### `work_stream.py`

更早期的实验脚本，包含背景生成、随机掩码、缺陷注入和 YOLO 标注导出等逻辑。目前仍属于实验性实现。

### `scripts/`

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

1. 在本机完整跑通一次 `generate_defect.py` 真实生成。
2. 给真实推理补日志和下载/缓存提示。
3. 再逐步清理 `batch_augment.py` 和 `work_stream.py` 的硬编码路径。
4. 最后再收敛 `scripts/` 里的缺失紧固件流程。

## 维护说明

如果只是想验证项目已经“能启动”：

```powershell
python -m uv sync
python -m uv run python generate_defect.py --dry-run
python -m unittest discover -s tests -p "test_*.py" -v
```

如果这三步都通过，就说明当前仓库的最小入口和 `uv` 管理已经正常。
