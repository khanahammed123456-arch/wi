# Experiment 3 Feature Sensitivity Summary

| mode | layer | rel_l2 | mae | mse | cosine | samples |
|---|---:|---:|---:|---:|---:|---:|
| outdated | p2 | 0.514254 | 0.666874 | 0.856977 | 0.834756 | 500 |
| outdated | p3 | 0.551990 | 0.568614 | 0.618325 | 0.806529 | 500 |
| outdated | p4 | 0.637064 | 0.591416 | 0.650215 | 0.743991 | 500 |
| outdated | p5 | 0.712622 | 0.670961 | 0.834039 | 0.659348 | 500 |
| pred | p2 | 0.517265 | 0.670570 | 0.864806 | 0.832970 | 500 |
| pred | p3 | 0.555259 | 0.571980 | 0.624044 | 0.804414 | 500 |
| pred | p4 | 0.640831 | 0.594778 | 0.657082 | 0.740899 | 500 |
| pred | p5 | 0.715017 | 0.673294 | 0.840435 | 0.656783 | 500 |

Interpretation guide:

- Higher rel_l2/mae/mse means the received semantic feature is farther from the true-CSI feature.
- Lower cosine means the semantic feature direction is less consistent with the true-CSI feature.
- If p2/p3 errors are larger than p4/p5, high-resolution semantic features are more sensitive to CSI mismatch.
- If pred errors are smaller than outdated errors, WiFo prediction reduces semantic feature distortion.