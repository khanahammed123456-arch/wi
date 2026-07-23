# ROI-aware Feature Distortion

| mode | layer | region | rel_l2 | mae | mse | cosine | samples |
|---|---:|---|---:|---:|---:|---:|---:|
| outdated | p2 | roi | 0.524910 | 0.518658 | 0.576426 | 0.793906 | 492 |
| outdated | p2 | background | 0.454739 | 0.607812 | 0.723470 | 0.867071 | 500 |
| outdated | p2 | global | 0.459444 | 0.599813 | 0.710193 | 0.862917 | 500 |
| outdated | p3 | roi | 0.557100 | 0.481486 | 0.495308 | 0.763969 | 492 |
| outdated | p3 | background | 0.483677 | 0.513953 | 0.508620 | 0.849550 | 500 |
| outdated | p3 | global | 0.491174 | 0.512272 | 0.510770 | 0.842608 | 500 |
| outdated | p4 | roi | 0.580031 | 0.499097 | 0.526358 | 0.744367 | 492 |
| outdated | p4 | background | 0.541209 | 0.505810 | 0.479006 | 0.816473 | 500 |
| outdated | p4 | global | 0.548150 | 0.508333 | 0.491588 | 0.807946 | 500 |
| outdated | p5 | roi | 0.617051 | 0.501487 | 0.525529 | 0.721144 | 492 |
| outdated | p5 | background | 0.623584 | 0.555984 | 0.567841 | 0.746976 | 500 |
| outdated | p5 | global | 0.623832 | 0.552866 | 0.571811 | 0.742432 | 500 |
| pred | p2 | roi | 0.526639 | 0.520524 | 0.582062 | 0.790846 | 492 |
| pred | p2 | background | 0.457452 | 0.611662 | 0.730788 | 0.865469 | 500 |
| pred | p2 | global | 0.462259 | 0.603736 | 0.718057 | 0.861164 | 500 |
| pred | p3 | roi | 0.558633 | 0.483037 | 0.500483 | 0.760548 | 492 |
| pred | p3 | background | 0.486193 | 0.516760 | 0.512634 | 0.848163 | 500 |
| pred | p3 | global | 0.493797 | 0.515125 | 0.515281 | 0.841033 | 500 |
| pred | p4 | roi | 0.580530 | 0.499694 | 0.530096 | 0.741849 | 492 |
| pred | p4 | background | 0.545285 | 0.509482 | 0.484716 | 0.814107 | 500 |
| pred | p4 | global | 0.552063 | 0.511802 | 0.497329 | 0.805556 | 500 |
| pred | p5 | roi | 0.615419 | 0.500589 | 0.526008 | 0.721118 | 492 |
| pred | p5 | background | 0.628635 | 0.560420 | 0.576022 | 0.743313 | 500 |
| pred | p5 | global | 0.628277 | 0.556699 | 0.579166 | 0.739133 | 500 |

Interpretation:

- ROI region is built from ground-truth boxes projected to each FPN feature map.
- Background region is the complement of the ROI mask.
- If ROI distortion is more aligned with AP changes than background distortion, semantic effectiveness depends on task-relevant feature preservation rather than global CSI accuracy.