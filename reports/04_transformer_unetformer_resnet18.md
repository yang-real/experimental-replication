# Transformer / UNetFormer / ResNet‑18 LoveDA 核验报告

日期：2026-11-10  
论文：A Controlled Benchmark of Visual State-Space Backbones with Domain-Shift and Boundary Analysis for Remote-Sensing Segmentation  
论文地址：https://arxiv.org/abs/2604.18721  
官方代码：https://github.com/Dineth14/Mamba-Segmentation  
官方检查点：https://huggingface.co/dineth18/Mamba-Segmentation  

## 1. 核验范围

本报告记录论文所谓 Transformer 基线在 LoveDA 上的官方检查点评估：

- All→All
- Urban→Rural
- Rural→Urban

## 2. 重要架构说明

配置文件 `configs/transformer_unetformer.yaml` 实际内容为：

```yaml
backbone: cnn
variant: resnet18
```

仓库中搜索不到 UNetFormer 的模型实现，只存在配置中的名称和注释。因此，作者发布的“Transformer / UNetFormer”检查点实际由：

```text
ResNet-18 编码器
+ 固定轻量 U-Net 解码器
```

构成，并不是真正的 Transformer 编码器。

本报告复现的是作者发布检查点的数值行为，不代表真实 UNetFormer 架构。

## 3. 运行环境

- GPU：NVIDIA GeForce RTX 4090 24 GB
- Python：3.10.8
- PyTorch：2.1.2+cu118
- torchvision：0.16.2+cu118
- timm：0.4.12
- 输入尺寸：512×512
- 验证集：1669 张
- 配置：`configs/transformer_unetformer.yaml`

## 4. All→All

检查点：

`transformer_unetformer_r18_512/checkpoints/best.pth`

检查点内部记录：

- iteration：40000
- mIoU：0.4861190319
- mF1：0.6399134398
- OA：0.6503611207

| 类别 | IoU | F1 |
|---|---:|---:|
| Background | 47.70 | 64.59 |
| Building | 63.15 | 77.42 |
| Road | 54.79 | 70.79 |
| Water | 66.64 | 79.98 |
| Barren | 20.10 | 33.48 |
| Forest | 38.24 | 55.32 |
| Agricultural | 49.65 | 66.36 |
| **Mean** | **48.61** | **63.99** |
| Overall Accuracy | 65.04 |  |

与论文完全一致：

| 指标 | 本次结果 | 论文 |
|---|---:|---:|
| mIoU | 48.61 | 48.61 |
| OA | 65.04 | 65.04 |
| mF1 | 63.99 | 63.99 |

## 5. Urban→Rural

检查点：

`transformerunetformer_resnet18_urbantrain_512/checkpoints/best.pth`

| 类别 | IoU | F1 |
|---|---:|---:|
| Background | 47.02 | 63.96 |
| Building | 39.05 | 56.16 |
| Road | 30.49 | 46.73 |
| Water | 45.24 | 62.29 |
| Barren | 4.71 | 9.00 |
| Forest | 5.06 | 9.63 |
| Agricultural | 43.86 | 60.97 |
| **Mean** | **30.77** | **44.11** |
| Overall Accuracy | 55.96 |  |

| 指标 | 本次结果 | 论文 | 差距 |
|---|---:|---:|---:|
| U→R mIoU | 30.77 | 34.56 | -3.79 |

## 6. Rural→Urban

检查点：

`transformerunetformer_resnet18_ruraltrain_512/checkpoints/best.pth`

| 类别 | IoU | F1 |
|---|---:|---:|
| Background | 38.39 | 55.48 |
| Building | 55.80 | 71.63 |
| Road | 50.76 | 67.34 |
| Water | 67.23 | 80.40 |
| Barren | 46.56 | 63.54 |
| Forest | 48.73 | 65.53 |
| Agricultural | 52.90 | 69.19 |
| **Mean** | **51.48** | **67.59** |
| Overall Accuracy | 65.70 |  |

| 指标 | 本次结果 | 论文 | 差距 |
|---|---:|---:|---:|
| R→U mIoU | 51.48 | 44.84 | +6.64 |

## 7. 总结

| 实验 | 本次结果 | 论文 | 差距 |
|---|---:|---:|---:|
| All→All | 48.61 | 48.61 | 0.00 |
| U→R | 30.77 | 34.56 | -3.79 |
| R→U | 51.48 | 44.84 | +6.64 |

结论：

1. 发布检查点的 All→All 数值与论文完全一致。
2. 两个跨域结果与论文不一致，说明跨域检查点与论文表格不是同一版本。
3. 发布配置实际使用 ResNet-18 CNN，而不是 Transformer UNetFormer。
4. 不能把当前结果表述为“真实 UNetFormer 复现”。
5. 跨域方向性仍成立：Rural→Urban 明显优于 Urban→Rural。

## 8. 云端结果文件

结果目录：

`/root/autodl-tmp/results/transformer_unetformer_r18_official/`

主要文件：

- `all.json`
- `urban_to_rural.json`
- `rural_to_urban.json`
- `eval.csv`
- `cross_domain_eval.csv`

