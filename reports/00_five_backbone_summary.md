# 五骨干 LoveDA 复现总报告

日期：2026-11-10  
论文：A Controlled Benchmark of Visual State-Space Backbones with Domain-Shift and Boundary Analysis for Remote-Sensing Segmentation  
论文地址：https://arxiv.org/abs/2604.18721  
官方代码：https://github.com/Dineth14/Mamba-Segmentation  
官方检查点：https://huggingface.co/dineth18/Mamba-Segmentation  

## 1. 总体结论

本次复现覆盖论文表中的五个模型条目：

1. VMamba-Small
2. CNN / DeepLabv3 / ResNet-50
3. MambaVision-Base
4. “Transformer / UNetFormer” / ResNet-18
5. Spatial-Mamba-Base

主要结论：

- VMamba-Small 与 MambaVision-Base 的官方检查点结果能够高度复现。
- CNN 官方检查点与论文 All→All 和跨域数值不一致。
- 所谓 Transformer/UNetFormer 实际由 ResNet-18 CNN 实现。
- Spatial-Mamba-Base 的 All→All 接近论文，但跨域结果不一致。
- 所有模型的跨域方向性都一致：Rural→Urban 明显优于 Urban→Rural。

## 2. All→All 结果

| 模型 | 本次 mIoU | 论文 mIoU | 差距 | 结论 |
|---|---:|---:|---:|---|
| VMamba-Small | 55.66 | 55.66 | 0.00 | 精确 |
| CNN / DeepLabv3 | 52.02 | 43.01 | +9.01 | 不一致 |
| MambaVision-Base | 55.16 | 55.17 | -0.01 | 近似 |
| “Transformer” / ResNet-18 | 48.61 | 48.61 | 0.00 | 数值精确，架构错误 |
| Spatial-Mamba-Base | 48.19 | 48.03 | +0.16 | 接近 |

## 3. 跨域结果

### Urban→Rural

| 模型 | 本次 mIoU | 论文 mIoU | 差距 |
|---|---:|---:|---:|
| VMamba-Small | 40.61 | 40.62 | -0.01 |
| CNN / DeepLabv3 | 32.33 | 30.36 | +1.97 |
| MambaVision-Base | 40.24 | 40.24 | 0.00 |
| “Transformer” / ResNet-18 | 30.77 | 34.56 | -3.79 |
| Spatial-Mamba-Base | 30.88 | 35.23 | -4.35 |

### Rural→Urban

| 模型 | 本次 mIoU | 论文 mIoU | 差距 |
|---|---:|---:|---:|
| VMamba-Small | 53.51 | 53.52 | -0.01 |
| CNN / DeepLabv3 | 51.38 | 39.98 | +11.40 |
| MambaVision-Base | 51.97 | 51.96 | +0.01 |
| “Transformer” / ResNet-18 | 51.48 | 44.84 | +6.64 |
| Spatial-Mamba-Base | 44.62 | 46.55 | -1.93 |

## 4. VMamba 三种子

已发布 All→All 检查点：

| 检查点 | mIoU |
|---|---:|
| `Vmamb_small_512` | 55.66 |
| `Vmamb_small_512_2` | 54.87 |
| `Vmamb_small_512_3` | 54.40 |

统计：

- 均值：54.98
- 样本标准差：0.64
- 论文文字：55.66 ± 0.23

论文主表值可以由 Seed 1 精确复现，但三种子统计无法由当前三个已发布检查点复现。

## 5. 架构与版本问题

### CNN

`cnn_deeplabv3p_r50_512` 的检查点内部指标为 52.02，作者旧仓库分析文件也记录 52.02；论文表格却写 43.01。

### Transformer

`configs/transformer_unetformer.yaml` 实际为：

```yaml
backbone: cnn
variant: resnet18
```

论文提交前的 `TransformerUNetFormer/model.py` 同样使用 ResNet-18 和普通 LightUNetDecoder，没有真正的 Transformer 编码器实现。

### Spatial-Mamba

All→All 接近论文，但两个跨域结果与论文相差 1.93–4.35 个百分点，说明跨域检查点与论文表格不是同一版本。

## 6. 跨域方向性

五个条目都满足：

```text
Rural→Urban > Urban→Rural
```

这支持论文的主要定性结论：跨域迁移存在明显不对称性。

但绝对跨域数值并非都能复现。

## 7. 复现等级

| 条目 | 复现等级 |
|---|---|
| VMamba-Small | 高：主结果、跨域和 Boundary 分析均可复现 |
| MambaVision-Base | 高：三项主要结果均近似精确 |
| Spatial-Mamba-Base | 部分：All→All 接近，跨域不一致 |
| CNN / DeepLabv3 | 低：官方检查点与论文数值明显不一致 |
| Transformer / UNetFormer | 不可认定：发布配置实际为 ResNet-18 CNN |

## 8. 最终表述

> 使用作者发布的官方检查点，在统一 LoveDA 验证流程下，VMamba-Small 和 MambaVision-Base 的主要结果得到高度复现。五个条目的跨域不对称性方向一致，但 CNN、Transformer 和 Spatial-Mamba 的公开检查点与论文表格存在明显的数值或架构版本差异。因此，本次工作完成了对公开代码和权重的独立复现审计，但不能声称整篇论文的所有表格数值均已严格复现。

