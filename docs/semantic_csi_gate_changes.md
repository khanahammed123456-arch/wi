# Semantic CSI Gate Experiment Reverted

This document is kept only as an experiment note.

The semantic-aware CSI reliability gate was tested and then reverted because it reduced AP in the tested setting. The current active code path should not use semantic gate modules or semantic gate config keys.

Current active logic:

```python
h_pred_som = h_outdated + delta_h
```

Current active config:

```text
configs/MIMO/SoM_MIMO_Cityscapes_WiFo_stage2.yaml
```

The restored stage2 config uses:

```yaml
WIFO_AUX_LOSS_WEIGHT: 1.0
```

Do not treat the old semantic gate patch as the main method.
