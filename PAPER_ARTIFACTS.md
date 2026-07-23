# SoM-MIMO + WiFo 论文代码、模型与实验结果索引

本仓库保存论文当前版本实际使用的核心代码、模型权重、实验原始结果和论文图表。实验脚本和论文 Word 文档不包含在本次上传中。当前论文主线为：

> 原始 SoM-MIMO 在高速移动环境中使用过时 CSI，导致 MIMO 决策与真实传播信道失配。本文利用 WiFo 根据历史 CSI 预测信道变化，通过残差 CSI 校正和面向实例分割任务的联合细化，提高 CSI 老化条件下的语义传输性能。

## 1. 核心代码

| 文件 | 作用 |
| --- | --- |
| `custom_mapper.py` | 联合生成 WiFo 与 SoM 分支使用的历史 CSI、未来真实 CSI、训练 SNR 和多普勒条件 |
| `wifo_wrapper.py` | WiFo 模型封装、CSI 预测输出适配和残差预测接口 |
| `SoM_MIMO/som_mimo_backbone.py` | 构造 `h_outdated`、WiFo 残差 `delta_h` 和 `h_pred=h_outdated+delta_h`，并执行 MIMO 语义特征传输 |
| `SoM_MIMO/som_mimo_rcnn.py` | 实例分割主流程、预测 CSI/真实 CSI 分支和任务损失衔接 |
| `SoM_MIMO/config_city.py` | WiFo、CSI 模式、损失权重和训练配置项 |
| `configs/MIMO/SoM_MIMO_Cityscapes_WiFo_stage2.yaml` | 当前有效的 Stage 2 训练与测试配置 |
| `train.py` | Detectron2 训练、测试和参数组学习率设置 |

## 2. 论文模型

模型权重通过 Git LFS 管理。

| 路径 | 论文中的角色 |
| --- | --- |
| `model/SoM_MIMO_C48.pth` | 原始 SoM-MIMO C48 基线 |
| `model/wifo_base.pkl` | WiFo 预训练基础模型 |
| `model/0616/model_final.pth` | 残差 CSI 校正模型，用于论文三阶段曲线中的中间阶段 |
| `model/0616/model_0019999.pth` | `delta_adapter_ft` 对照模型，用于 CSI 与语义特征机制比较 |
| `output/ft0616_delta0_lr2e6_5k/model_final.pth` | 最终无蒸馏任务联合细化模型 |

最终模型从 `model/0616/model_final.pth` 初始化，以基础学习率 `2e-6` 继续训练 5,000 次；训练中关闭额外残差监督和全局真实 CSI 蒸馏，由实例分割任务及当前混合 CSI 辅助目标共同细化。

## 3. 实验结果

| 目录 | 内容 |
| --- | --- |
| `output/paper_main_newhz/` | 40--400 Hz 多普勒主性能原始日志、汇总表和三阶段结果 |
| `output/paper_final_figures/` | 多普勒主实验与方法阶段消融的论文图表 |
| `output/paper_snr_task_adapted/` | 120/280 Hz、不同 SNR 下的鲁棒性结果和论文图表 |
| `output/paper_mechanism_nodistill/` | 最终无蒸馏模型的 CSI、FPN、ROI、类别 ROI 和 Feature Swapping 机制实验 |

这些目录保留 CSV、日志、PNG、PDF 等原始或可直接用于论文的结果文件。数据集、完整训练输出、缓存和无关旧实验不上传。
