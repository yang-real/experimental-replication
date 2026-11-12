# Boundary Loss 15k 短预算消融报告

日期：2026-11-12  
论文：A Controlled Benchmark of Visual State-Space Backbones with Domain-Shift and Boundary Analysis for Remote-Sensing Segmentation  
论文地址：https://arxiv.org/abs/2604.18721  

## 1. 实验目的

验证 TriBraid Loss 中的 Boundary Loss 是否能够改善 VMamba-Small 在 LoveDA 上的总体分割性能和边界区域性能。

这次实验属于短预算消融，不是 50k 完整训练协议的严格重复。

## 2. 受控条件

两组实验使用完全相同的：

- 骨干：VMamba-Small
- LoveDA 训练集和验证集
- 随机种子：42
- 输入尺寸：512×512
- batch size：4
- 最大迭代：15000
- 验证间隔：7500
- Boundary warmup：2000
- AdamW、学习率策略、数据增强和评估流程

唯一变量：

| 实验 | Boundary 权重 |
|---|---:|
| A：完整 TriBraid | 0.5 |
| B：无 Boundary Loss | 0 |

## 3. 总体指标

| 指标 | A：完整 TriBraid | B：无 Boundary | A−B |
|---|---:|---:|---:|
| 最佳迭代 | 7500 | 15000 | — |
| Best mIoU | **54.30** | 53.84 | **+0.46** |
| Best mF1 | 69.50 | 68.82 | +0.68 |
| Best OA | 72.20 | 71.68 | +0.52 |

逐类别 IoU：

| 类别 | A | B | A−B |
|---|---:|---:|---:|
| Background | 55.62 | 55.49 | +0.13 |
| Building | 64.66 | 66.73 | -2.07 |
| Road | 57.68 | 58.10 | -0.42 |
| Water | 71.54 | 72.97 | -1.43 |
| Barren | 33.26 | 28.35 | +4.91 |
| Forest | 39.98 | 40.06 | -0.08 |
| Agricultural | 57.35 | 55.17 | +2.18 |

## 4. Boundary 与 Interior 分析

使用最佳检查点，在 LoveDA 全部 1669 张验证图上计算：

- Boundary 半径：2 像素
- Boundary 区域：3×3 形态学梯度后膨胀
- Interior 区域：有效像素中不属于 Boundary 的部分

| 指标 | A：完整 TriBraid | B：无 Boundary | A−B |
|---|---:|---:|---:|
| Boundary mIoU | **29.12** | 28.38 | **+0.74** |
| Interior mIoU | **57.46** | 57.07 | +0.39 |
| Interior−Boundary Gap | **28.33** | 28.68 | **-0.35** |

Gap 越小表示边界和内部分割性能的差距越小。

## 5. 结果解释

短预算单种子实验显示：

1. 加入 Boundary Loss 后，Best mIoU 提高约 0.46 个百分点。
2. Boundary mIoU 提高约 0.74 个百分点。
3. Interior mIoU 同时提高约 0.39 个百分点。
4. Interior−Boundary Gap 缩小约 0.35 个百分点。
5. Barren 提高约 4.91，Agricultural 提高约 2.18。
6. 去掉 Boundary Loss 后，Building 和 Water 略好。

结论：

> Boundary Loss 对总体指标和边界指标均产生了正向作用，并略微缩小了边界与内部的性能差距；但其作用不只局限于边界，因为 Interior mIoU 也有所提高。

## 6. 特征可视化与定量分析

使用固定的随机种子，从 Urban 和 Rural 验证集中各抽取 50 张图片。两个模型使用完全相同的图片，并分别在 F1–F4 上比较 Boundary 与 Interior 区域的平均特征向量。

### 6.1 Boundary–Interior 余弦相似度

合并 Urban 和 Rural 后，各层两组平均余弦相似度为：

| 特征层 | A：完整 TriBraid | B：无 Boundary | A−B |
|---|---:|---:|---:|
| F1 | 0.9791 | 0.9781 | +0.0010 |
| F2 | 0.9805 | 0.9836 | −0.0031 |
| **F3** | **0.8911** | **0.9063** | **−0.0152** |
| F4 | 0.9897 | 0.9894 | +0.0003 |

余弦相似度越低，表示 Boundary 与 Interior 区域的特征方向越不同。

分域结果：

| 域 | 模型 | F3 余弦相似度 | 样本数 |
|---|---|---:|---:|
| Urban | A：完整 TriBraid | 0.8938 | 48 |
| Urban | B：无 Boundary | 0.9110 | 48 |
| Rural | A：完整 TriBraid | 0.8884 | 49 |
| Rural | B：无 Boundary | 0.9017 | 49 |

Urban 和 Rural 均显示，完整损失模型的 F3 Boundary/Interior 特征更不相似。

### 6.2 特征幅值

Boundary/Interior 的特征范数比在所有层均约为 1.000：

| 模型与域 | F1 | F2 | F3 | F4 |
|---|---:|---:|---:|---:|
| A Urban | 1.0009 | 0.9998 | 1.0002 | 0.9999 |
| A Rural | 1.0003 | 0.9993 | 1.0003 | 0.9998 |
| B Urban | 1.0010 | 0.9997 | 1.0001 | 0.9999 |
| B Rural | 1.0005 | 0.9992 | 1.0004 | 0.9998 |

这表示 Boundary Loss 没有简单提高边界区域的激活幅值，而更可能改变了中层特征的表示方向。

### 6.3 可视化文件

已生成 A/B 在 Urban 与 Rural 样本上的 F1–F4 PCA 图，以及：

```text
feature_quantitative.json
feature_similarity_comparison.png
```

可视化结果位于：

`/root/autodl-tmp/runs/ablation_feature_viz/`

定量结果位于：

`/root/autodl-tmp/runs/ablation_feature_quant/`

### 6.4 特征结论

100 张验证图片的统计显示：

1. F1、F2 和 F4 没有表现出一致且明显的 A/B 差异。
2. F3 是唯一在两域中均显示 A 模型相似度更低的特征层。
3. Boundary/Interior 特征幅值比约为 1，说明主要变化来自特征方向而非幅值。
4. 这一现象与 F3 同时包含局部结构和较大空间上下文的性质相符。

因此，更谨慎的表述是：

> Boundary Loss 主要增强了中层 F3 特征对边界和内部区域的区分能力，而非简单增强边界幅值或改变深层全局语义特征。

当前每个域随机抽取 50 张图片。F3 的余弦相似度标准差约为 0.06–0.09，因此该差异虽然方向一致，但幅度较小，仍需要多种子或更大样本分析才能进行更强推断。

## 7. 限制

- 只完成单随机种子。
- 只训练 15000 次，不是论文的 50000 次完整协议。
- 只在 VMamba-Small 上验证，不能直接推广到全部骨干。
- A、B 选择的是各自最佳验证检查点，最佳迭代分别为 7500 和 15000。
- 结果应表述为探索性短预算消融，不能替换论文的正式实验。

## 8. 文件位置

A 组：

`/root/autodl-tmp/runs/ablation15k_full_s42/`

B 组：

`/root/autodl-tmp/runs/ablation15k_noboundary_s42/`

边界指标：

`/root/autodl-tmp/runs/ablation_boundary_metrics.json`

