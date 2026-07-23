# Experiment 3 Feature Sensitivity Summary

| mode | layer | rel_l2 | mae | mse | cosine | samples |
|---|---:|---:|---:|---:|---:|---:|
| outdated | p2 | 0.299844 | 0.385467 | 0.320863 | 0.939713 | 500 |
| outdated | p3 | 0.322418 | 0.331358 | 0.233565 | 0.929974 | 500 |
| outdated | p4 | 0.366853 | 0.335984 | 0.232363 | 0.911590 | 500 |
| outdated | p5 | 0.421645 | 0.370276 | 0.275475 | 0.881581 | 500 |
| pred | p2 | 0.304811 | 0.392275 | 0.330066 | 0.937684 | 500 |
| pred | p3 | 0.327572 | 0.336907 | 0.239460 | 0.928035 | 500 |
| pred | p4 | 0.373360 | 0.341942 | 0.238312 | 0.909368 | 500 |
| pred | p5 | 0.428488 | 0.376335 | 0.282129 | 0.878955 | 500 |

Interpretation guide:

- Higher rel_l2/mae/mse means the received semantic feature is farther from the true-CSI feature.
- Lower cosine means the semantic feature direction is less consistent with the true-CSI feature.
- If p2/p3 errors are larger than p4/p5, high-resolution semantic features are more sensitive to CSI mismatch.
- If pred errors are smaller than outdated errors, WiFo prediction reduces semantic feature distortion.