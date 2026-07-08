# SoM-MIMO + WiFo 论文当前状态与最新改进进展

本文档用于说明当前文章的研究状态、已经验证有效的基础方法、最近失败的尝试、最新有效改进方法，以及后续还需要补哪些实验才能支撑论文。

## 1. 文档整体介绍

这份文档不是代码说明书，而是论文推进状态总结。它回答四个问题：

1. 这篇文章现在的主线是什么。
2. 当前已经有效的改进是什么。
3. 哪些最近尝试过的方法不适合作为主线。
4. 最新真正带来 AP 提升的方法是什么，后面应该怎么继续验证。

文档最核心的结论是：

```text
当前文章应从“让 CSI 数值更准”转向“让预测 CSI 条件下的语义特征更接近真实 CSI 条件下的理想语义特征”。
```

也就是说，论文重点不应该只写 WiFo 预测 CSI，而应该写：

```text
CSI aging 下的 mismatch-aware semantic transmission。
```

## 2. 当前文章主线

原始 SoM-MIMO 主要关注 MIMO 语义特征传输。你的改进把它扩展到更真实的 CSI aging 场景：

```text
车辆或移动终端在高速移动时，系统当前可用 CSI 已经过时。
如果继续用过时 CSI 做 MCE 和特征传输，接收端语义特征会退化，最终 instance segmentation AP 下降。
```

因此，文章主线可以概括为：

```text
利用 WiFo 的历史 CSI 预测能力，对 SoM-MIMO 的过时 CSI 进行残差补偿，并在预测 CSI 与真实未来信道存在 mismatch 的条件下提升语义任务 AP。
```

这个主线比“把 WiFo 接进 SoM-MIMO”更准确，因为真正的问题不是模型拼接，而是：

```text
预测 CSI 不完美时，语义通信系统怎样保持鲁棒。
```

## 3. 基础有效方法：WiFo residual CSI compensation

当前已经验证有效的基础方法是 direct residual：

```python
h_pred_som = h_outdated + delta_h
```

其中：

- `h_outdated` 是过时 CSI。
- `delta_h` 是 WiFo 分支和 adapter 预测出的残差修正量。
- `h_pred_som` 是系统用于 MCE 的预测 CSI。
- `h_true` 是真实未来 CSI，用于真实信道传播和训练监督。

基础结果为：

| 版本 | 方法 | AP | AP50 | 说明 |
|---|---|---:|---:|---|
| 0616 baseline | WiFo residual CSI compensation | 27.2213 | 49.8665 | 当前最重要的基础模型 |
| 低学习率普通微调 | 不加新 loss，只低 LR 微调 5k | 约 27.4 | - | 说明低 LR 微调不会破坏模型 |

这个方法的意义是：

```text
系统不直接知道未来真实 CSI，而是使用 WiFo 预测得到的 h_pred_som；真实传播仍然通过 h_true，因此实验模拟了预测 CSI 与真实信道 mismatch。
```

## 4. 最近失败或不适合作主线的方案

最近几条路线已经通过实验说明不适合作为主线。

### 4.1 semantic CSI gate

semantic gate 的思路是：

```python
h_pred_som = h_outdated + gate * delta_h
```

它希望模型根据语义特征或信道状态决定是否相信 WiFo residual：

```text
gate 接近 1：多用 WiFo residual。
gate 接近 0：少用 WiFo residual，更多保留 outdated CSI。
```

但实验结果较差：

| 版本 | 方法 | AP | AP50 |
|---|---|---:|---:|
| direct residual | `h_outdated + delta_h` | 27.2213 | 49.8665 |
| semantic gate | `h_outdated + gate * delta_h` | 24.3488 | 47.5447 |

失败原因是 gate 改变了原本稳定的 residual 输出路径。它虽然想做可信度控制，但实际直接扰动了最终 CSI 分布，导致 AP 明显下降。

### 4.2 双路传输

双路传输的想法是同时保留原始路径和预测 CSI 路径，再做融合。但这会带来两个问题：

- 两路信道后的特征分布不一致，融合反而增加检测头负担。
- 它没有直接解决 predicted CSI 与 true CSI mismatch 对语义特征造成的影响。

因此双路传输不适合作为当前文章主线。

### 4.3 delta loss 和冻结 WiFo

后来尝试过显式 residual 监督：

```python
delta_gt = h_true - h_outdated
loss_wifo_delta = L1(delta_h, delta_gt)
```

但实验显示，即使 CSI 数值指标可能变好，AP 不一定提升。例如 `WIFO_DELTA_LOSS_WEIGHT=0.005` 时 AP 下降到约 26.9，`0.1` 也不理想。

冻结 WiFo 主体也不适合当前模型，`FREEZE_WIFO=True` 时 AP 约为 25.6。说明当前系统需要 WiFo/adapter 与 SoM-MIMO 保持联合适应，而不是简单冻结。

这些结果说明：

```text
让 CSI residual 数值更接近 h_true - h_outdated，不等于最终语义 AP 更高。
```

## 5. 最新有效改进：True-CSI Guided Semantic Feature Distillation

最新真正带来明显提升的方法是：

```text
True-CSI Guided Semantic Feature Distillation
```

中文可以写成：

```text
真实 CSI 引导的语义特征蒸馏。
```

它的核心思想是：不要只监督 CSI 本身，而是直接监督预测 CSI 条件下的接收语义特征。

训练时构造两条分支：

```text
student branch:
  使用 h_pred_som 做 MCE
  真实传播仍使用 h_true
  得到预测 CSI 条件下的接收特征 p_rx

teacher branch:
  使用 h_true 做 MCE
  真实传播也使用 h_true
  得到理想 true CSI 条件下的接收特征 p_rx_teacher
```

然后加入蒸馏损失：

```text
L_distill = SmoothL1(p_rx, stopgrad(p_rx_teacher))
```

总训练目标变为：

```text
L = L_detection + L_RPN + L_mask + lambda_csi * L_csi + lambda_distill * L_distill
```

推理时不需要 teacher branch，仍然只使用：

```python
h_pred_som = h_outdated + delta_h
```

所以这个方法的优势是：

- 推理结构不变。
- 推理计算量不增加。
- 不强迫 CSI 数值最接近真实 CSI。
- 直接让预测 CSI 下的语义接收特征对齐 true CSI 的理想接收特征。

## 6. 最新实验结果

目前最新结果如下：

| 实验 | 训练设置 | AP | 结论 |
|---|---|---:|---|
| 0616 baseline | 直接 eval | 约 27.3 | 原始最佳基础模型 |
| 低 LR 普通微调 | `delta=0`, `lr=2e-6`, 5k | 约 27.4 | 低学习率微调稳定 |
| True-CSI distill | `lambda_distill=0.01`, `lr=2e-6`, 5k | 约 28.0 | 当前最有效改进 |
| 从 28.0 checkpoint 继续训 | `lr=5e-7`, 再 5k | 约 27.8 | 继续训练会轻微回落 |
| delta loss | `WIFO_DELTA_LOSS_WEIGHT=0.005` | 约 26.9 | 不适合当前路线 |
| semantic gate | gate 控制 residual | 约 24.35 | 明显失败 |

这说明：

```text
True-CSI feature distillation 是当前唯一明确把 AP 从 27.3 附近提升到 28.0 左右的方法。
```

这个提升虽然不是特别大，但已经有论文价值，因为它验证了一个关键观点：

```text
CSI 数值误差不是唯一目标，语义特征层面对齐更能服务最终 AP。
```

## 7. 当前方法为什么更有深度

原来的 WiFo residual 方法解决的是：

```text
如何预测更好的 CSI。
```

最新蒸馏方法进一步解决：

```text
预测 CSI 不完美时，如何让接收端语义特征接近理想 true CSI 条件下的语义特征。
```

这比继续优化 CSI NMSE 更贴近语义通信，因为最终任务不是重建 CSI，而是提升 instance segmentation AP。

论文中可以这样写：

```text
Directly minimizing CSI prediction error does not necessarily improve semantic AP. Therefore, we introduce a true-CSI guided feature distillation objective, which aligns the received semantic features under predicted CSI with those under ideal true CSI during training.
```

中文：

```text
直接最小化 CSI 预测误差并不一定提升语义 AP。因此，我们提出真实 CSI 引导的特征蒸馏目标，在训练阶段使预测 CSI 条件下的接收语义特征对齐理想 true CSI 条件下的接收语义特征。
```

## 8. 后续最重要的实验

接下来不建议继续盲目加新模块，而应该围绕当前有效方法补实验。

### 8.1 训练步数 sweep

当前 5k 已经达到约 28.0，但继续训会回落，因此要做短程训练长度 sweep：

| 训练步数 | 目的 |
|---:|---|
| 3k | 看是否更早达到最佳 |
| 4k | 检查 5k 前是否已最优 |
| 5k | 当前最佳参考 |
| 6k | 看是否略长更好 |
| 7k | 检查是否开始过拟合或回落 |

不要直接用 50k 的中间 5k checkpoint 和单独 5k 比，因为学习率 schedule 不同。单独 5k 使用的是：

```text
MAX_ITER=5000, STEPS=(4000, 4500)
```

而 50k 中的 5k checkpoint 此时还没有进入学习率衰减阶段，所以结果不可直接对比。

### 8.2 蒸馏权重消融

需要测试：

```text
lambda_distill = 0.005, 0.01, 0.02
```

如果 `0.01` 最好，就将其作为主设置。

### 8.3 不同 Doppler 下验证

为了证明方法不是只在 120 Hz 调出来，需要测试：

```text
Doppler = 60, 120, 240, 360, 480 Hz
```

如果高 Doppler 下提升更明显，论文说服力会更强。应用场景可以设为 3.5 GHz 车载语义通信，此时大致对应：

| Doppler | 速度，3.5 GHz |
|---:|---:|
| 60 Hz | 18.5 km/h |
| 120 Hz | 37 km/h |
| 240 Hz | 74 km/h |
| 360 Hz | 111 km/h |
| 480 Hz | 148 km/h |

这覆盖城市道路、快速路和高速公路等正常车辆移动场景。

### 8.4 必要消融

至少需要比较：

| 方法 | 作用 |
|---|---|
| outdated CSI | 证明 CSI aging 会损害性能 |
| direct residual | 基础 WiFo residual 方法 |
| delta loss | 证明单纯 CSI residual 监督不等于 AP 提升 |
| semantic gate | 证明直接门控 CSI 会破坏稳定路径 |
| true-CSI distill | 证明语义特征对齐有效 |
| true CSI upper bound | 显示理想 CSI 条件下的性能上限 |

## 9. 文章贡献点建议

基于当前结果，文章可以形成三个贡献：

1. 提出面向 CSI aging 的 SoM-MIMO 语义传输框架，在预测 CSI 与真实信道 mismatch 条件下评估 instance segmentation。
2. 引入 WiFo-assisted residual CSI compensation，用历史 CSI 预测未来信道残差，构造 `h_pred = h_outdated + delta_h`。
3. 提出 True-CSI Guided Semantic Feature Distillation，在训练阶段用 true CSI teacher 引导 predicted CSI student，使接收语义特征更接近理想信道条件，从而提升 AP。

其中第三点是现在最有深度、最有实验支撑的新贡献。

## 10. 当前一句话总结

当前文章已经从“WiFo 接入 SoM-MIMO”推进到：

```text
在 CSI aging 和 predicted-CSI mismatch 下，用 true-CSI guided semantic feature distillation 提升接收语义特征质量。
```

当前最重要结果是：

```text
baseline AP 约 27.3，true-CSI distillation 后 AP 约 28.0。
```

后续重点不是继续堆模块，而是完成训练步数、蒸馏权重、Doppler/SNR/CSI_LAG 和消融实验，证明该方法在不同 CSI aging 强度下稳定有效。
