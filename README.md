# SoM-MIMO WiFo

This repository stores the current SoM-MIMO/WiFo experiment code snapshot and the semantic-aware CSI reliability gate changes.

The active config is:

```text
configs/MIMO/SoM_MIMO_Cityscapes_WiFo_stage2.yaml
```

Main idea:

- Keep the original single-path SoM-MIMO image/feature transmission pipeline.
- Use WiFo to predict a CSI residual instead of directly trusting outdated CSI.
- Add a LECLN-inspired semantic CSI gate: FPN semantic features + CSI residual features + SNR predict a sigmoid reliability gate.
- Final predicted CSI is `h_outdated + semantic_gate * delta_h`.

Large artifacts such as datasets, model weights, outputs, and generated experiment assets are intentionally not committed.
