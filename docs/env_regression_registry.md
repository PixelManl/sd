# Environment Regression Registry

这个文件记录“环境级”事实，避免后续因为 Python、Torch、CUDA 或运行入口不一致导致重复踩坑。

## uv Environment

- 管理方式：`uv`
- 当前事实：仓库 `uv` 环境中的 `torch` 是 CPU 版
- 影响：可跑单测和 dry-run，不适合真实 GPU 检测训练

## GPU Training Environment

- Python 解释器：`E:\APP\Miniconda3\envs\pytorch-py310-cu126\python.exe`
- Torch：`2.6.0+cu126`
- CUDA：可用
- 当前用途：`YOLO11n` 真实训练、评估、推理

## Rules

- 文档、测试、脚手架优先用 `uv` 环境验证。
- 真实 GPU 训练必须明确写出解释器或环境名，避免误落到 CPU 环境。
- 新增训练环境时，要把解释器路径、Torch 版本、CUDA 可用性补到这里。
