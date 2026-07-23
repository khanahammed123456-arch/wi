# ROI-aware Feature Distortion

| mode | layer | region | rel_l2 | mae | mse | cosine | samples |
|---|---:|---|---:|---:|---:|---:|---:|
| outdated | p2 | roi | 0.317589 | 0.311665 | 0.235218 | 0.919900 | 492 |
| outdated | p2 | background | 0.298442 | 0.393058 | 0.329496 | 0.941167 | 500 |
| outdated | p2 | global | 0.299844 | 0.385467 | 0.320863 | 0.939713 | 500 |
| outdated | p3 | roi | 0.332811 | 0.285428 | 0.196571 | 0.910307 | 492 |
| outdated | p3 | background | 0.320374 | 0.335588 | 0.235937 | 0.932182 | 500 |
| outdated | p3 | global | 0.322418 | 0.331358 | 0.233565 | 0.929974 | 500 |
| outdated | p4 | roi | 0.350557 | 0.299461 | 0.212400 | 0.901269 | 492 |
| outdated | p4 | background | 0.366773 | 0.338536 | 0.231213 | 0.913700 | 500 |
| outdated | p4 | global | 0.366853 | 0.335984 | 0.232363 | 0.911590 | 500 |
| outdated | p5 | roi | 0.387894 | 0.313672 | 0.223860 | 0.886557 | 492 |
| outdated | p5 | background | 0.426122 | 0.376431 | 0.278933 | 0.881362 | 500 |
| outdated | p5 | global | 0.421645 | 0.370276 | 0.275475 | 0.881581 | 500 |
| pred | p2 | roi | 0.322381 | 0.316675 | 0.242207 | 0.917267 | 492 |
| pred | p2 | background | 0.303255 | 0.399886 | 0.338626 | 0.939235 | 500 |
| pred | p2 | global | 0.304811 | 0.392275 | 0.330066 | 0.937684 | 500 |
| pred | p3 | roi | 0.337226 | 0.289408 | 0.201669 | 0.907753 | 492 |
| pred | p3 | background | 0.325421 | 0.341192 | 0.241768 | 0.930329 | 500 |
| pred | p3 | global | 0.327572 | 0.336907 | 0.239460 | 0.928035 | 500 |
| pred | p4 | roi | 0.354779 | 0.303228 | 0.217784 | 0.898658 | 492 |
| pred | p4 | background | 0.373294 | 0.344643 | 0.237095 | 0.911531 | 500 |
| pred | p4 | global | 0.373360 | 0.341942 | 0.238312 | 0.909368 | 500 |
| pred | p5 | roi | 0.390955 | 0.316590 | 0.227801 | 0.885029 | 492 |
| pred | p5 | background | 0.433245 | 0.382891 | 0.285916 | 0.878601 | 500 |
| pred | p5 | global | 0.428488 | 0.376335 | 0.282129 | 0.878955 | 500 |

Interpretation:

- ROI region is built from ground-truth boxes projected to each FPN feature map.
- Background region is the complement of the ROI mask.
- If ROI distortion is more aligned with AP changes than background distortion, semantic effectiveness depends on task-relevant feature preservation rather than global CSI accuracy.