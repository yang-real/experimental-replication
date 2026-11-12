# MambaVision‑Base LoveDA 官方检查点核验报告

日期：2026-11-10  
论文：A Controlled Benchmark of Visual State-Space Backbones with Domain-Shift and Boundary Analysis for Remote-Sensing Segmentation  
论文地址：https://arxiv.org/abs/2604.18721  
官方代码：https://github.com/Dineth14/Mamba-Segmentation  
官方检查点：https://huggingface.co/dineth18/Mamba-Segmentation  

## 1. 核验范围

本报告记录 MambaVision‑Base 在 LoveDA 上的官方检查点重新评估结果，包括：

- LoveDA All→All
- 真正的 Urban→Rural 跨域评估
- 真正的 Rural→Urban 跨域评估

本次没有重新训练模型，只验证作者发布的检查点。

## 2. 运行环境

- GPU：NVIDIA GeForce RTX 4090 24 GB
- 独立环境：`/root/autodl-tmp/venvs/mambavision`
- Python：3.10.8
- PyTorch：2.1.2+cu118
- torchvision：0.16.2+cu118
- timm：1.0.15
- torch：2.1.2+cu118
- mamba_ssm：1.1.1
- 输入尺寸：512×512
- 验证集：LoveDA Urban 677 张 + Rural 992 张，共 1669 张
- 配置：`configs/mambavision.yaml`
- 模型规模：MambaVision‑Base，约 104.18 M 参数

MambaVision 的 setup.py 指定 `mamba-ssm==2.2.4`，但现场编译失败。使用已有 `mamba-ssm 1.1.1` 后，MambaVision 随机张量前向测试通过，随后官方检查点评估也与论文一致。

## 3. All→All

检查点：

`mambavision_base_512/checkpoints/best.pth`

检查点内部记录：

- iteration：15000
- mIoU：0.5516692996
- mF1：0.7042873502
- OA：0.7242124081

重新评估：

| 类别 | IoU | F1 |
|---|---:|---:|
| Background | 55.94 | 71.75 |
| Building | 65.97 | 79.50 |
| Road | 56.70 | 72.37 |
| Water | 70.26 | 82.53 |
| Barren | 35.32 | 52.20 |
| Forest | 44.95 | 62.02 |
| Agricultural | 56.99 | 72.60 |
| **Mean** | **55.16** | **70.42** |
| Overall Accuracy | 72.42 |  |

与论文比较：

| 指标 | 本次结果 | 论文 | 差距 |
|---|---:|---:|---:|
| mIoU | 55.16 | 55.17 | 0.01 |
| OA | 72.42 | 72.42 | 0.00 |
| mF1 | 70.42 | 70.43 | 0.01 |

## 4. Urban→Rural

检查点：

`mambavision_base_urbantrain_512/checkpoints/best.pth`

Urban-only 模型直接在 Rural 验证集评估，不使用目标域数据。

| 类别 | IoU | F1 |
|---|---:|---:|
| Background | 47.78 | 64.66 |
| Building | 50.93 | 67.49 |
| Road | 38.07 | 55.15 |
| Water | 52.82 | 69.13 |
| Barren | 16.74 | 28.68 |
| Forest | 27.68 | 43.36 |
| Agricultural | 47.65 | 64.55 |
| **Mean** | **40.24** | **56.14** |
| Overall Accuracy | 61.11 |  |

与论文比较：

| 指标 | 本次结果 | 论文 | 差距 |
|---|---:|---:|---:|
| U→R mIoU | 40.24 | 40.24 | 0.00 |

## 5. Rural→Urban

检查点：

`mambavision_base_ruraltrain_512/checkpoints/best.pth`

Rural-only 模型直接在 Urban 验证集评估，不使用目标域数据。

| 类别 | IoU | F1 |
|---|---:|---:|
| Background | 39.67 | 56.80 |
| Building | 60.89 | 75.69 |
| Road | 53.17 | 69.43 |
| Water | 55.38 | 71.29 |
| Barren | 45.36 | 62.41 |
| Forest | 54.22 | 70.32 |
| Agricultural | 55.06 | 71.02 |
| **Mean** | **51.97** | **68.14** |
| Overall Accuracy | 67.08 |  |

与论文比较：

| 指标 | 本次结果 | 论文 | 差距 |
|---|---:|---:|---:|
| R→U mIoU | 51.97 | 51.96 | 0.01 |

## 6. 结论

MambaVision‑Base 的三项主要指标均与论文高度一致：

| 实验 | 本次结果 | 论文 | 差距 |
|---|---:|---:|---:|
| All→All | 55.16 | 55.17 | 0.01 |
| U→R | 40.24 | 40.24 | 0.00 |
| R→U | 51.97 | 51.96 | 0.01 |

结论：

1. 作者发布的 MambaVision‑Base 检查点可复现论文结果。
2. 使用 `mamba-ssm 1.1.1` 足以完成 MambaVision 推理。
3. 跨域不对称结论继续成立：Rural→Urban 明显优于 Urban→Rural。
4. 官方检查点、LoveDA 数据和评估流程均正常。

## 7. 结果文件

云端结果目录：

`/root/autodl-tmp/results/mambavision_base_official/`

检查点目录：

`/root/autodl-tmp/hf-mamba-seg/Comparison_Experiments/mambavision_base_*`

主要结果文件：

- `all.json`
- `urban_to_rural.json`
- `rural_to_urban.json`
- `eval.csv`
- `cross_domain_eval.csv`

## 8. 可复现表述

> 使用作者发布的 MambaVision‑Base 检查点，在 LoveDA 官方验证集上得到 All→All 55.16、Urban→Rural 40.24、Rural→Urban 51.97 mIoU。三项分别与论文的 55.17、40.24、51.96 相差 0.01、0.00、0.01 个百分点，属于四舍五入误差范围。
