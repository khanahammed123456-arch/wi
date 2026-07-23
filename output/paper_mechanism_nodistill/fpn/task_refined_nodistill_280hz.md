# Experiment 3 Feature Sensitivity Summary

| mode | layer | rel_l2 | mae | mse | cosine | samples |
|---|---:|---:|---:|---:|---:|---:|
| outdated | p2 | 0.459444 | 0.599813 | 0.710193 | 0.862917 | 500 |
| outdated | p3 | 0.491174 | 0.512272 | 0.510770 | 0.842608 | 500 |
| outdated | p4 | 0.548150 | 0.508333 | 0.491588 | 0.807946 | 500 |
| outdated | p5 | 0.623832 | 0.552866 | 0.571811 | 0.742432 | 500 |
| pred | p2 | 0.462259 | 0.603736 | 0.718057 | 0.861164 | 500 |
| pred | p3 | 0.493797 | 0.515125 | 0.515281 | 0.841033 | 500 |
| pred | p4 | 0.552063 | 0.511802 | 0.497329 | 0.805556 | 500 |
| pred | p5 | 0.628277 | 0.556699 | 0.579166 | 0.739133 | 500 |

Interpretation guide:

- Higher rel_l2/mae/mse means the received semantic feature is farther from the true-CSI feature.
- Lower cosine means the semantic feature direction is less consistent with the true-CSI feature.
- If p2/p3 errors are larger than p4/p5, high-resolution semantic features are more sensitive to CSI mismatch.
- If pred errors are smaller than outdated errors, WiFo prediction reduces semantic feature distortion.