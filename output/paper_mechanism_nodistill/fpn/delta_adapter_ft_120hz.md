# Experiment 3 Feature Sensitivity Summary

| mode | layer | rel_l2 | mae | mse | cosine | samples |
|---|---:|---:|---:|---:|---:|---:|
| outdated | p2 | 0.306365 | 0.389643 | 0.330630 | 0.937388 | 500 |
| outdated | p3 | 0.327624 | 0.331515 | 0.236036 | 0.927627 | 500 |
| outdated | p4 | 0.389141 | 0.356184 | 0.260758 | 0.900444 | 500 |
| outdated | p5 | 0.444885 | 0.412859 | 0.351283 | 0.865334 | 500 |
| pred | p2 | 0.313901 | 0.399296 | 0.345159 | 0.934319 | 500 |
| pred | p3 | 0.335176 | 0.339294 | 0.245841 | 0.924292 | 500 |
| pred | p4 | 0.396915 | 0.363484 | 0.269742 | 0.896752 | 500 |
| pred | p5 | 0.450981 | 0.419276 | 0.359654 | 0.861908 | 500 |

Interpretation guide:

- Higher rel_l2/mae/mse means the received semantic feature is farther from the true-CSI feature.
- Lower cosine means the semantic feature direction is less consistent with the true-CSI feature.
- If p2/p3 errors are larger than p4/p5, high-resolution semantic features are more sensitive to CSI mismatch.
- If pred errors are smaller than outdated errors, WiFo prediction reduces semantic feature distortion.