# ROI-aware Feature Distortion

| mode | layer | region | rel_l2 | mae | mse | cosine | samples |
|---|---:|---|---:|---:|---:|---:|---:|
| outdated | p2 | roi | 0.516104 | 0.535043 | 0.611813 | 0.801281 | 492 |
| outdated | p2 | background | 0.460618 | 0.604811 | 0.721005 | 0.866841 | 500 |
| outdated | p2 | global | 0.464222 | 0.599265 | 0.712958 | 0.863126 | 500 |
| outdated | p3 | roi | 0.550678 | 0.492181 | 0.518455 | 0.768080 | 492 |
| outdated | p3 | background | 0.491281 | 0.510380 | 0.507643 | 0.846637 | 500 |
| outdated | p3 | global | 0.497554 | 0.510394 | 0.512971 | 0.840115 | 500 |
| outdated | p4 | roi | 0.575020 | 0.522408 | 0.575207 | 0.743793 | 492 |
| outdated | p4 | background | 0.575975 | 0.532769 | 0.532435 | 0.792973 | 500 |
| outdated | p4 | global | 0.577787 | 0.534819 | 0.544803 | 0.786666 | 500 |
| outdated | p5 | roi | 0.608013 | 0.522310 | 0.567233 | 0.724368 | 492 |
| outdated | p5 | background | 0.656647 | 0.620702 | 0.721029 | 0.713704 | 500 |
| outdated | p5 | global | 0.651500 | 0.611418 | 0.709524 | 0.713924 | 500 |
| pred | p2 | roi | 0.518517 | 0.536935 | 0.614611 | 0.798996 | 492 |
| pred | p2 | background | 0.464897 | 0.610501 | 0.734407 | 0.863823 | 500 |
| pred | p2 | global | 0.468140 | 0.604378 | 0.724857 | 0.860369 | 500 |
| pred | p3 | roi | 0.551999 | 0.492955 | 0.519028 | 0.766499 | 492 |
| pred | p3 | background | 0.495634 | 0.515048 | 0.516476 | 0.843279 | 500 |
| pred | p3 | global | 0.501362 | 0.514443 | 0.520378 | 0.837179 | 500 |
| pred | p4 | roi | 0.575249 | 0.522175 | 0.573829 | 0.743722 | 492 |
| pred | p4 | background | 0.580816 | 0.537102 | 0.541498 | 0.788642 | 500 |
| pred | p4 | global | 0.581783 | 0.538438 | 0.552029 | 0.783118 | 500 |
| pred | p5 | roi | 0.607343 | 0.521546 | 0.565167 | 0.725323 | 492 |
| pred | p5 | background | 0.659597 | 0.623641 | 0.728771 | 0.709976 | 500 |
| pred | p5 | global | 0.653796 | 0.613790 | 0.715612 | 0.711043 | 500 |

Interpretation:

- ROI region is built from ground-truth boxes projected to each FPN feature map.
- Background region is the complement of the ROI mask.
- If ROI distortion is more aligned with AP changes than background distortion, semantic effectiveness depends on task-relevant feature preservation rather than global CSI accuracy.