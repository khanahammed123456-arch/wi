# ROI-aware Feature Distortion

| mode | layer | region | rel_l2 | mae | mse | cosine | samples |
|---|---:|---|---:|---:|---:|---:|---:|
| outdated | p2 | roi | 0.586708 | 0.610071 | 0.763939 | 0.749398 | 492 |
| outdated | p2 | background | 0.509135 | 0.671407 | 0.863651 | 0.839893 | 500 |
| outdated | p2 | global | 0.514254 | 0.666874 | 0.856977 | 0.834756 | 500 |
| outdated | p3 | roi | 0.628809 | 0.563566 | 0.651924 | 0.705347 | 492 |
| outdated | p3 | background | 0.543293 | 0.566753 | 0.608831 | 0.815459 | 500 |
| outdated | p3 | global | 0.551990 | 0.568614 | 0.618325 | 0.806529 | 500 |
| outdated | p4 | roi | 0.652573 | 0.594377 | 0.716370 | 0.676255 | 492 |
| outdated | p4 | background | 0.632516 | 0.586759 | 0.631277 | 0.753494 | 500 |
| outdated | p4 | global | 0.637064 | 0.591416 | 0.650215 | 0.743991 | 500 |
| outdated | p5 | roi | 0.679565 | 0.585014 | 0.692302 | 0.658198 | 492 |
| outdated | p5 | background | 0.716047 | 0.678956 | 0.842851 | 0.661203 | 500 |
| outdated | p5 | global | 0.712622 | 0.670961 | 0.834039 | 0.659348 | 500 |
| pred | p2 | roi | 0.589443 | 0.612446 | 0.767935 | 0.747180 | 492 |
| pred | p2 | background | 0.512431 | 0.675554 | 0.872590 | 0.837831 | 500 |
| pred | p2 | global | 0.517265 | 0.670570 | 0.864806 | 0.832970 | 500 |
| pred | p3 | roi | 0.630778 | 0.565011 | 0.653690 | 0.703590 | 492 |
| pred | p3 | background | 0.547053 | 0.570663 | 0.615837 | 0.812897 | 500 |
| pred | p3 | global | 0.555259 | 0.571980 | 0.624044 | 0.804414 | 500 |
| pred | p4 | roi | 0.653450 | 0.594818 | 0.716198 | 0.676144 | 492 |
| pred | p4 | background | 0.637113 | 0.590790 | 0.639960 | 0.749567 | 500 |
| pred | p4 | global | 0.640831 | 0.594778 | 0.657082 | 0.740899 | 500 |
| pred | p5 | roi | 0.679341 | 0.584733 | 0.691365 | 0.659245 | 492 |
| pred | p5 | background | 0.719106 | 0.681833 | 0.850895 | 0.657736 | 500 |
| pred | p5 | global | 0.715017 | 0.673294 | 0.840435 | 0.656783 | 500 |

Interpretation:

- ROI region is built from ground-truth boxes projected to each FPN feature map.
- Background region is the complement of the ROI mask.
- If ROI distortion is more aligned with AP changes than background distortion, semantic effectiveness depends on task-relevant feature preservation rather than global CSI accuracy.