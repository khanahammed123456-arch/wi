# Semantic-aware CSI reliability gate

This change keeps the original single-path SoM-MIMO transmission logic and adds a LECLN-inspired reliability gate to the WiFo residual CSI path.

Reference idea: PKU-PCNI/CSI-Learning-LECLN fuses wireless features with sensing features through a bottleneck MLP and sigmoid modulation. Here the same principle is adapted without importing LiDAR data or changing the SoM-MIMO channel interface.

## Core logic

1. WiFo predicts a residual CSI update `delta_h`.
2. The raw prediction is still `h_raw_pred_som = h_outdated + delta_h`.
3. FPN semantic features are global-average-pooled from `p2` to `p5` into a 256-D semantic vector.
4. The gate MLP receives semantic vector, outdated CSI, residual CSI, raw predicted CSI, and SNR.
5. The final CSI used by MCE/channel is:

```python
h_pred_som = h_outdated + semantic_gate * delta_h
```

This makes the model learn when to trust WiFo residual correction and when to fall back closer to outdated CSI.

## Files changed locally

- `SoM_MIMO/som_mimo_backbone.py`
  - Added `SemanticCSIGate`.
  - Added `_build_semantic_vector`.
  - Changed WiFo residual path from direct residual addition to gated residual addition.
  - Added `loss_semantic_csi_gate` and event scalars `csi/semantic_gate`, `csi/semantic_gate_target`.

- `configs/MIMO/SoM_MIMO_Cityscapes_WiFo_stage2.yaml`
  - Added semantic gate config keys.
  - Changed `WIFO_AUX_LOSS_WEIGHT` from `1.0` to `0.3` so the CSI auxiliary loss does not dominate detection and mask losses.

- `SoM_MIMO/config_city.py`
  - Added default config keys for the semantic CSI gate.
  - Changed default `WIFO_AUX_LOSS_WEIGHT` to `0.3`.

- `train.py`
  - Cleaned corrupted comments.
  - Restored `torch.backends.cudnn.benchmark = False` as executable code.
  - Added `semantic_csi_gate` to trainability logging and adapter LR group.

## Important training note

Because `SemanticCSIGate` adds new parameters, train from `model/SoM_MIMO_C48.pth` or a checkpoint created after this module exists. Do not resume an old optimizer state from the previous architecture.

Recommended start:

```bash
python train.py \
  --config-file configs/MIMO/SoM_MIMO_Cityscapes_WiFo_stage2.yaml \
  --num-gpus 1
```

Expected logs should include:

- `loss_wifo_csi`
- `loss_semantic_csi_gate`
- `csi/semantic_gate`
- `csi/semantic_gate_target`
