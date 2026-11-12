# CNN DeepLabv3 / ResNet-50 官方检查点核验报告

日期：2026-11-08  
论文：A Controlled Benchmark of Visual State-Space Backbones with Domain-Shift and Boundary Analysis for Remote-Sensing Segmentation  
论文地址：https://arxiv.org/abs/2604.18721  
官方代码：https://github.com/Dineth14/Mamba-Segmentation  
官方检查点：https://huggingface.co/dineth18/Mamba-Segmentation  
检查点仓库版本：`0df6efb255155c92a7fb5fd70731516fef4bd9dd`

## 1. 核验目的

本报告记录使用作者发布的 CNN / DeepLabv3 / ResNet-50 分割检查点，在 LoveDA 官方验证集上的重新评估结果。

本次没有重新训练 CNN。目的是判断论文主表中的 CNN 基线能否通过发布检查点复现。

## 2. 运行环境

- GPU：NVIDIA GeForce RTX 4090 24 GB
- Python：3.10.8
- PyTorch：2.1.2+cu118
- torchvision：0.16.2+cu118
- 输入尺寸：512 × 512
- 验证集：LoveDA Urban 677 张 + Rural 992 张，共 1669 张
- 评估配置：`configs/cnn_deeplabv3p.yaml`
- 模型：CNN / ResNet-50 / DeepLabv3-style encoder

## 3. 使用的官方检查点

### 3.1 All→All

`cnn_deeplabv3p_r50_512/checkpoints/best.pth`

检查点内部记录的元数据：

- iteration：10000
- mIoU：0.520231664
- mF1：0.676658392
- OA：0.693973243

### 3.2 Urban-only

`cnn_deeplabv3p_resnet50_urbantrain_512/checkpoints/best.pth`

用于 Urban→Rural 跨域评估。

### 3.3 Rural-only

`cnn_deeplabv3p_resnet50_ruraltrain_512/checkpoints/best.pth`

用于 Rural→Urban 跨域评估。

## 4. All→All 结果

评估样本：1669。

| 类别 | IoU | F1 |
|---|---:|---:|
| Background | 52.98 | 69.26 |
| Building | 60.63 | 75.49 |
| Road | 55.30 | 71.21 |
| Water | 69.86 | 82.26 |
| Barren | 32.13 | 48.63 |
| Forest | 41.01 | 58.17 |
| Agricultural | 52.25 | 68.64 |
| **Mean** | **52.02** | **67.67** |
| Overall Accuracy | 69.40 |  |

### 与论文比较

| 指标 | 官方检查点评估 | 论文 | 差异 |
|---|---:|---:|---:|
| mIoU | 52.02 | 43.01 | +9.01 |
| OA | 69.40 | 59.14 | +10.26 |
| mF1 | 67.67 | 64.20 | +3.47 |

## 5. Urban→Rural 结果

Urban-only 模型直接在 Rural 验证集上评估，不使用目标域标签或适配。

评估样本：992。

| 类别 | IoU | F1 |
|---|---:|---:|
| Background | 46.41 | 63.39 |
| Building | 36.91 | 53.91 |
| Road | 29.72 | 45.83 |
| Water | 40.48 | 57.63 |
| Barren | 19.20 | 32.22 |
| Forest | 15.00 | 26.09 |
| Agricultural | 38.61 | 55.71 |
| **Mean** | **32.33** | **47.83** |
| Overall Accuracy | 54.30 |  |

### 与论文比较

| 指标 | 本次结果 | 论文 | 差异 |
|---|---:|---:|---:|
| U→R mIoU | 32.33 | 30.36 | +1.97 |

## 6. Rural→Urban 结果

Rural-only 模型直接在 Urban 验证集上评估，不使用目标域标签或适配。

评估样本：677。

| 类别 | IoU | F1 |
|---|---:|---:|
| Background | 36.89 | 53.89 |
| Building | 59.33 | 74.47 |
| Road | 50.98 | 67.53 |
| Water | 59.99 | 74.99 |
| Barren | 48.23 | 65.07 |
| Forest | 50.46 | 67.08 |
| Agricultural | 53.79 | 69.95 |
| **Mean** | **51.38** | **67.57** |
| Overall Accuracy | 65.81 |  |

### 与论文比较

| 指标 | 本次结果 | 论文 | 差异 |
|---|---:|---:|---:|
| R→U mIoU | 51.38 | 39.98 | +11.40 |

## 7. 与论文表格的总体差异

| 实验 | 官方检查点结果 | 论文 | 差异 |
|---|---:|---:|---:|
| CNN All→All | 52.02 | 43.01 | +9.01 |
| CNN Urban→Rural | 32.33 | 30.36 | +1.97 |
| CNN Rural→Urban | 51.38 | 39.98 | +11.40 |

官方检查点内部保存的 All→All 指标也是 52.02，因此差异不是本次评估脚本造成的。

## 8. 另一个 CNN 检查点

同时检查了：

`cnn_unet_r50_512/checkpoints/best.pth`

其内部记录的指标为：

- iteration：7500
- mIoU：0.502592921
- mF1：0.660858989
- OA：0.680185497

该检查点也不是论文表格中的 43.01 mIoU。

因此，`cnn_deeplabv3p_r50_512` 和 `cnn_unet_r50_512` 都不能解释论文中的 43.01。

## 9. 结论

可以确认：

1. 发布 CNN 检查点可以在当前代码和 LoveDA 官方验证集上正常评估。
2. 官方 CNN 检查点的绝对结果明显高于论文表格。
3. 发布 CNN 检查点与论文表格很可能不是同一代码或训练版本。
4. CNN 的跨域方向性结论仍然成立：Rural→Urban 明显优于 Urban→Rural。
5. 不能用本次官方 CNN 结果直接复现论文中的 `43.01`。

与 VMamba-Small 的比较：

| 指标 | VMamba-Small | CNN 官方检查点 | 官方检查点差距 |
|---|---:|---:|---:|
| All→All | 55.66 | 52.02 | +3.64 |
| U→R | 40.61 | 32.33 | +8.28 |
| R→U | 53.51 | 51.38 | +2.13 |

论文表格中的 VMamba 相对优势更大：

| 指标 | VMamba 论文优势 |
|---|---:|
| All→All | +12.65 |
| U→R | +10.26 |
| R→U | +13.54 |

因此，当前官方发布检查点支持“VMamba 优于该 CNN 检查点”，但优势幅度小于论文表格报告的幅度。

## 10. 结果文件

云端结果目录：

`/root/autodl-tmp/results/cnn_deeplabv3p_official/`

主要文件：

- `all.json`
- `urban_to_rural.json`
- `rural_to_urban.json`
- `eval.csv`
- `cross_domain_eval.csv`

检查点目录：

`/root/autodl-tmp/hf-mamba-seg/Comparison_Experiments/`

## 11. 复现结论

本报告记录的不是 CNN 训练复现，而是对作者发布 CNN 检查点的核验。

最准确的表述是：

> 使用作者发布的 DeepLabv3 / ResNet-50 检查点，LoveDA All→All、Urban→Rural 和 Rural→Urban 分别得到 52.02、32.33 和 51.38 mIoU。三者均高于论文表格中的 43.01、30.36 和 39.98。检查点内部保存的指标与外部评估一致，说明差异来自论文记录与发布检查点版本不一致，而不是本次评估流程。
