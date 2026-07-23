# Experiment 3 Feature Sensitivity Summary

| mode | layer | rel_l2 | mae | mse | cosine | samples |
|---|---:|---:|---:|---:|---:|---:|
| outdated | p2 | 0.464222 | 0.599265 | 0.712958 | 0.863126 | 500 |
| outdated | p3 | 0.497554 | 0.510394 | 0.512971 | 0.840115 | 500 |
| outdated | p4 | 0.577787 | 0.534819 | 0.544803 | 0.786666 | 500 |
| outdated | p5 | 0.651500 | 0.611418 | 0.709524 | 0.713924 | 500 |
| pred | p2 | 0.468140 | 0.604378 | 0.724857 | 0.860369 | 500 |
| pred | p3 | 0.501362 | 0.514443 | 0.520378 | 0.837179 | 500 |
| pred | p4 | 0.581783 | 0.538438 | 0.552029 | 0.783118 | 500 |
| pred | p5 | 0.653796 | 0.613790 | 0.715612 | 0.711043 | 500 |

Interpretation guide:

- Higher rel_l2/mae/mse means the received semantic feature is farther from the true-CSI feature.
- Lower cosine means the semantic feature direction is less consistent with the true-CSI feature.
- If p2/p3 errors are larger than p4/p5, high-resolution semantic features are more sensitive to CSI mismatch.
- If pred errors are smaller than outdated errors, WiFo prediction reduces semantic feature distortion.