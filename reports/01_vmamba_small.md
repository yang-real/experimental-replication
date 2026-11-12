# VMamba‑Small LoveDA 复现报告

日期：2026-11-08  
论文：A Controlled Benchmark of Visual State-Space Backbones with Domain-Shift and Boundary Analysis for Remote-Sensing Segmentation  
论文地址：https://arxiv.org/abs/2604.18721  
官方代码：https://github.com/Dineth14/Mamba-Segmentation  
官方检查点：https://huggingface.co/dineth18/Mamba-Segmentation

## 1. 复现范围

本次实验复现论文中的 VMamba‑Small / LoveDA 部分，包括：

- LoveDA All→All 总体分割结果
- 三个已发布 All→All 检查点的重复性检查
- 真正的 Urban→Rural 和 Rural→Urban 跨域结果
- Boundary 与 Interior 分割性能比较
- Structural 与 Natural 类别的跨域分组指标

未完成：

- MambaVision、Spatial‑Mamba、CNN、Transformer 全部变体
- ISPRS Potsdam
- Params、FLOPs、FPS 和峰值显存
- Rotation robustness
- 定性结果

## 2. 运行环境

- GPU：NVIDIA GeForce RTX 4090
- Python：3.10.8
- PyTorch：2.1.2+cu118
- torchvision：0.16.2+cu118
- timm：0.4.12
- mamba_ssm：1.1.1
- 训练输入：512×512
- 官方评估验证集：1669 张图像

## 3. 自行训练的 All→All 结果

实验目录：

`/root/autodl-tmp/runs/vmamba_small_512_full_v2`

训练设置：

- VMamba‑Small
- LoveDA Urban + Rural
- 50000 次迭代
- batch size 8
- 每 2500 次迭代验证
- AMP 开启
- 最佳检查点来自第 10000 次迭代

结果：

| 指标 | 本次训练 | 论文 |
|---|---:|---:|
| mIoU | 55.52 | 55.66 |
| OA | 73.10 | 73.49 |
| mF1 | 70.41 | 70.69 |
| 最佳迭代 | 10000 | 未说明 |
| Final mIoU | 54.99 | 不适用 |

自行训练的 All→All 模型在 Urban 和 Rural 验证子集上的域内结果：

| 验证子集 | 样本数 | mIoU | OA |
|---|---:|---:|---:|
| Urban | 677 | 58.90 | 72.59 |
| Rural | 992 | 47.95 | 73.44 |
| All | 1669 | 55.52 | 73.10 |

注意：由于训练同时使用 Urban 和 Rural，以上 Urban/Rural 子集指标不能解释为真正的跨域迁移结果。

## 4. 官方检查点复现

### 4.1 All→All

| 检查点 | mIoU | OA | mF1 |
|---|---:|---:|---:|
| `Vmamb_small_512` | 55.66 | 73.49 | 70.69 |
| `Vmamb_small_512_2` | 54.87 | 71.92 | 69.95 |
| `Vmamb_small_512_3` | 54.40 | 72.36 | 69.21 |

Seed 1 与论文主表完全一致：

| 指标 | 本次评估 | 论文 |
|---|---:|---:|
| mIoU | 55.66 | 55.66 |
| OA | 73.49 | 73.49 |
| mF1 | 70.69 | 70.69 |

三个已发布检查点的统计：

- 均值：54.98
- 样本标准差：0.64
- 总体标准差：0.52

论文正文给出的 VMamba‑Small 三种子统计为 `55.66 ± 0.23`。当前三个已发布检查点与该统计不一致，原因尚未确认。可能涉及检查点与论文种子定义不同、种子组合不同或论文与发布版本的统计方式不同。

### 4.2 真正的跨域评估

Urban→Rural 使用仅在 Urban 上训练的 `vmamba_small_urbantrain_512`，直接在 Rural 验证集评估，不进行目标域适配。

Rural→Urban 使用仅在 Rural 上训练的 `vmamba_small_ruraltrain_512`，直接在 Urban 验证集评估，不进行目标域适配。

| 实验 | 样本数 | 本次 mIoU | 论文 mIoU | 差距 |
|---|---:|---:|---:|---:|
| Urban→Rural | 992 | 40.61 | 40.62 | 0.01 |
| Rural→Urban | 677 | 53.51 | 53.52 | 0.01 |

结论：

- 真正的跨域方向差异被精确复现。
- Rural→Urban 明显优于 Urban→Rural。
- 当前跨域结果来自官方源域检查点，没有目标域数据泄漏。

### 4.3 跨域类别组指标

| 指标 | 本次结果 | 论文 |
|---|---:|---:|
| U→R Structural（Building、Road） | 42.25 | 42.3 |
| U→R Natural（Water、Forest、Agricultural） | 46.24 | 46.2 |
| R→U Structural | 55.98 | 56.0 |
| R→U Natural | 58.24 | 58.2 |

## 5. Boundary 与 Interior 分析

使用官方 `Vmamb_small_512` 检查点，在 LoveDA All 验证集上执行：

- Boundary 半径：2 像素
- Boundary mask：3×3 形态学梯度，然后按半径 2 膨胀
- 分别累计 Boundary 和 Interior 混淆矩阵

| 指标 | 本次结果 | 论文 |
|---|---:|---:|
| Boundary mIoU | 28.23 | 28.2 |
| Interior mIoU | 59.08 | 59.1 |
| Interior−Boundary | 30.85 | 30.8 |

结论：

- 边界性能显著低于内部区域。
- 约 30.85 个 mIoU 点的差距与论文报告高度一致。
- 边界细化确实是该模型和该数据设置下需要重点关注的问题。

## 6. 代码修复记录

### 6.1 AMP 下 Boundary Loss 精度不匹配

文件：`core/losses.py`

问题：

验证阶段出现 FP16 输入与 FP32 Laplacian 卷积核不匹配。

修复：

```python
self.laplacian_kernel.to(
    device=max_probs.device,
    dtype=max_probs.dtype,
)
```

原文件备份：

`core/losses.py.before_amp_fix`

### 6.2 `eval_domain.py` 路径重复

原脚本错误使用 `Val/Val/...`。LoveDA 实际目录为 `Val/...`，现已修正。

### 6.3 指标类别名称缺失

`format_metrics_table()` 现在接收 YAML 中的 `class_names`，不再访问不存在的 `Config.CLASS_NAMES`。

### 6.4 官方检查点严格加载失败

官方检查点包含 THOP 生成的 `total_ops`、`total_params` 属性，不是模型参数。严格加载失败后使用 `strict=False`，不会影响实际模型权重。

## 7. 论文与仓库需要注意的差异

1. 论文正文写 batch size 4；当前 README 和 `base.yaml` 写 batch size 8。自行训练使用了 batch 8，因此不属于与论文逐项相同的训练复现。
2. 论文正文列出的每域验证数量与当前代码使用的官方 LoveDA 验证集数量不一致。当前运行使用 Urban 677、Rural 992、总计 1669 张。
3. 官方 Seed 1 精确复现主表，但另外两个已发布检查点的三种子统计与论文 `55.66 ± 0.23` 不一致。
4. 作者提供的部分旧分析脚本依赖已经不存在的 `VMamba/config.py`，需要适配当前统一模型接口。Boundary 分析已通过兼容脚本完成。

## 8. 结论

本次工作不是“重新训练了整篇论文”，而是完成了论文中 VMamba‑Small / LoveDA 核心结果的独立核验。

已达到的高置信度复现：

- 官方 Seed 1 All→All：精确匹配
- Urban→Rural：与论文相差 0.01 个百分点
- Rural→Urban：与论文相差 0.01 个百分点
- Boundary/Interior 分析：与论文相差不超过 0.05 个百分点
- 跨域 Structural/Natural 分组趋势：一致

尚未严格复现：

- 论文声称的三种子均值和标准差
- 与论文逐项一致的 batch size 4 训练
- 其他骨干、Potsdam、效率与旋转分析

因此，最合适的表述是：

> 使用作者发布的官方检查点，VMamba‑Small 在 LoveDA 上的主结果、跨域不对称性和边界失败现象均得到高精度复现；自行完成的一次 batch size 8 训练也达到 55.52% mIoU，接近论文结果。由于 batch size、种子统计和发布检查点之间存在差异，本次工作不声称完成了整篇论文的严格端到端复现。

## 9. 结果文件位置

自行训练结果：

`/root/autodl-tmp/results/vmamba_small/`

官方检查点评估：

`/root/autodl-tmp/results/vmamba_small_official/`

Boundary 分析：

`/root/autodl-tmp/results/vmamba_small_official/boundary_analysis/`

自行训练最佳权重：

`/root/autodl-tmp/results/vmamba_small/best.pth`

官方检查点：

`/root/autodl-tmp/hf-mamba-seg/Comparison_Experiments/`

关键 SHA256：

```text
908e74e9e989601c7bc2fa5584ea08500297caac1abc159f9cfa7a7417aec559  best.pth
7a1e23fb1fca9a021d57271b175d64062c9199e1f1d9f69d0259bcd19a85dc1f  final.pth
566cb59c48d55c7097d2522508b845e3d4c588df3d9cdf0c430528c9a61578a9  train_20261107_094337.log
791c6c7b0ca652400e935b7e3a5a3aff8fd70673609833620e0a4f31c502e21b  domain_eval_clean.csv
```
