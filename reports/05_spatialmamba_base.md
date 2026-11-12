# Spatial‑Mamba‑Base LoveDA 官方检查点核验报告

日期：2026-11-10  
论文：A Controlled Benchmark of Visual State-Space Backbones with Domain-Shift and Boundary Analysis for Remote-Sensing Segmentation  
论文地址：https://arxiv.org/abs/2604.18721  
官方代码：https://github.com/Dineth14/Mamba-Segmentation  
官方检查点：https://huggingface.co/dineth18/Mamba-Segmentation  

## 1. 核验范围

本报告记录 Spatial-Mamba-Base 的官方检查点评估：

- LoveDA All→All
- Urban→Rural
- Rural→Urban

本次没有重新训练模型。

## 2. 运行环境

- GPU：NVIDIA GeForce RTX 4090 24 GB
- Python：3.10.8
- PyTorch：2.1.2+cu118
- torchvision：0.16.2+cu118
- timm：0.4.12
- mamba_ssm：1.1.1
- Cython：0.29.37
- CUDA：11.8
- 输入尺寸：512×512
- 模型参数量：100.47 M
- 验证集：1669 张

## 3. 自定义 CUDA 扩展

Spatial-Mamba 依赖两个自定义扩展：

- `selective_scan_cuda_oflex_rh`
- `dwconv2d`

为当前环境完成以下兼容修复：

1. 将 setuptools 固定为 `69.5.1`，恢复 PyTorch 2.1 所需的 `pkg_resources`。
2. selective_scan 仅针对 RTX 4090 的 `sm_89` 编译。
3. dwconv2d 源码移除了推理不需要的旧版 `at::NoGradGuard`。
4. 使用 `TORCH_CUDA_ARCH_LIST=8.9` 避免编译无关 GPU 架构。

随机张量前向测试成功，输出 `(1, 7, 512, 512)`。

## 4. All→All

检查点：

`spatialmamba_base_512/checkpoints/best.pth`

检查点内部记录：

- iteration：47500
- mIoU：0.4803124368
- mF1：0.6419355869
- OA：0.6635324955

重新评估：

| 类别 | IoU | F1 |
|---|---:|---:|
| Background | 50.53 | 67.13 |
| Building | 57.68 | 73.16 |
| Road | 54.03 | 70.16 |
| Water | 59.85 | 74.88 |
| Barren | 30.10 | 46.27 |
| Forest | 35.93 | 52.87 |
| Agricultural | 49.22 | 65.97 |
| **Mean** | **48.19** | **64.35** |
| Overall Accuracy | 66.51 |  |

与论文比较：

| 指标 | 本次结果 | 论文 | 差距 |
|---|---:|---:|---:|
| mIoU | 48.19 | 48.03 | +0.16 |
| OA | 66.51 | 66.35 | +0.16 |
| mF1 | 64.35 | 64.19 | +0.16 |

## 5. Urban→Rural

检查点：

`spatialmamba_base_urbantrain_512/checkpoints/best.pth`

| 类别 | IoU | F1 |
|---|---:|---:|
| Background | 51.95 | 68.38 |
| Building | 28.26 | 44.07 |
| Road | 31.58 | 48.00 |
| Water | 38.17 | 55.25 |
| Barren | 8.37 | 15.45 |
| Forest | 15.75 | 27.21 |
| Agricultural | 42.08 | 59.23 |
| **Mean** | **30.88** | **45.37** |
| Overall Accuracy | 60.48 |  |

与论文比较：

| 指标 | 本次结果 | 论文 | 差距 |
|---|---:|---:|---:|
| U→R mIoU | 30.88 | 35.23 | -4.35 |

## 6. Rural→Urban

检查点：

`spatialmamba_base_ruraltrain_512/checkpoints/best.pth`

| 类别 | IoU | F1 |
|---|---:|---:|
| Background | 35.10 | 51.97 |
| Building | 41.15 | 58.31 |
| Road | 46.67 | 63.64 |
| Water | 66.04 | 79.55 |
| Barren | 41.04 | 58.19 |
| Forest | 34.33 | 51.11 |
| Agricultural | 47.97 | 64.84 |
| **Mean** | **44.62** | **61.09** |
| Overall Accuracy | 60.06 |  |

与论文比较：

| 指标 | 本次结果 | 论文 | 差距 |
|---|---:|---:|---:|
| R→U mIoU | 44.62 | 46.55 | -1.93 |

## 7. 总结

| 实验 | 本次结果 | 论文 | 差距 |
|---|---:|---:|---:|
| All→All | 48.19 | 48.03 | +0.16 |
| U→R | 30.88 | 35.23 | -4.35 |
| R→U | 44.62 | 46.55 | -1.93 |

结论：

1. All→All 与论文接近，差距 0.16 个百分点。
2. 两个跨域结果均明显低于论文。
3. 发布源域检查点与论文跨域表格很可能不是同一版本。
4. 跨域方向性仍成立：Rural→Urban 优于 Urban→Rural。

## 8. 结果位置

结果目录：

`/root/autodl-tmp/results/spatialmamba_base_official/`

检查点目录：

`/root/autodl-tmp/hf-mamba-seg/Comparison_Experiments/spatialmamba_base_*`

