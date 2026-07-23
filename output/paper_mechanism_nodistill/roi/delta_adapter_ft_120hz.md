# ROI-aware Feature Distortion

| mode | layer | region | rel_l2 | mae | mse | cosine | samples |
|---|---:|---|---:|---:|---:|---:|---:|
| outdated | p2 | roi | 0.312185 | 0.320518 | 0.244318 | 0.924680 | 492 |
| outdated | p2 | background | 0.306070 | 0.396355 | 0.338707 | 0.938305 | 500 |
| outdated | p2 | global | 0.306365 | 0.389643 | 0.330630 | 0.937388 | 500 |
| outdated | p3 | roi | 0.327924 | 0.290013 | 0.199087 | 0.915187 | 492 |
| outdated | p3 | background | 0.326789 | 0.335104 | 0.238235 | 0.929202 | 500 |
| outdated | p3 | global | 0.327624 | 0.331515 | 0.236036 | 0.927627 | 500 |
| outdated | p4 | roi | 0.347265 | 0.312382 | 0.225753 | 0.905387 | 492 |
| outdated | p4 | background | 0.393046 | 0.359679 | 0.261464 | 0.900620 | 500 |
| outdated | p4 | global | 0.389141 | 0.356184 | 0.260758 | 0.900444 | 500 |
| outdated | p5 | roi | 0.385115 | 0.328135 | 0.240357 | 0.890242 | 492 |
| outdated | p5 | background | 0.452656 | 0.423438 | 0.364119 | 0.862275 | 500 |
| outdated | p5 | global | 0.444885 | 0.412859 | 0.351283 | 0.865334 | 500 |
| pred | p2 | roi | 0.317140 | 0.325491 | 0.251060 | 0.921958 | 492 |
| pred | p2 | background | 0.314007 | 0.406741 | 0.354613 | 0.935071 | 500 |
| pred | p2 | global | 0.313901 | 0.399296 | 0.345159 | 0.934319 | 500 |
| pred | p3 | roi | 0.332163 | 0.293608 | 0.203683 | 0.912448 | 492 |
| pred | p3 | background | 0.334855 | 0.343559 | 0.249072 | 0.925634 | 500 |
| pred | p3 | global | 0.335176 | 0.339294 | 0.245841 | 0.924292 | 500 |
| pred | p4 | roi | 0.351134 | 0.315602 | 0.229891 | 0.902870 | 492 |
| pred | p4 | background | 0.401558 | 0.367689 | 0.271563 | 0.896491 | 500 |
| pred | p4 | global | 0.396915 | 0.363484 | 0.269742 | 0.896752 | 500 |
| pred | p5 | roi | 0.388929 | 0.331330 | 0.243916 | 0.888182 | 492 |
| pred | p5 | background | 0.459348 | 0.430498 | 0.373700 | 0.858323 | 500 |
| pred | p5 | global | 0.450981 | 0.419276 | 0.359654 | 0.861908 | 500 |

Interpretation:

- ROI region is built from ground-truth boxes projected to each FPN feature map.
- Background region is the complement of the ROI mask.
- If ROI distortion is more aligned with AP changes than background distortion, semantic effectiveness depends on task-relevant feature preservation rather than global CSI accuracy.