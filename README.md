# SoM-MIMO WiFo

This repository stores the current SoM-MIMO/WiFo experiment code snapshot.

The active config is:

```text
configs/MIMO/SoM_MIMO_Cityscapes_WiFo_stage2.yaml
```

Current main idea:

- Keep the original single-path SoM-MIMO image/feature transmission pipeline.
- Use WiFo to predict a CSI residual instead of directly trusting outdated CSI.
- Use the residual CSI compensation path:

```python
h_pred_som = h_outdated + delta_h
```

The previous semantic CSI gate experiment has been reverted because it did not improve AP in the tested setting.

Large artifacts such as datasets, model weights, outputs, and generated experiment assets are intentionally not committed.
