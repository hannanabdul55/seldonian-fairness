# Safety-test false-pass probability for unsafe classifiers

A fixed classifier with true group TPRs (p_a, p_b) violates the constraint |TPR_a - TPR_b| <= 0.2. The safety set has n_a and n_b positives in the two groups. The table gives the exact probability that the safety test (delta = 0.05) nevertheless certifies the classifier; a Seldonian method must stay <= 0.05.

| method | p=(0.75,1.0) n=(8,40) | p=(0.75,1.0) n=(15,60) | p=(0.7,0.95) n=(15,60) | p=(0.6,0.85) n=(30,100) | p=(0.5,0.75) n=(60,60) | p=(0.02,0.27) n=(40,40) | p=(0.0,0.3) n=(10,100) |
|---|---|---|---|---|---|---|---|
| ttest | 0.1001 | 0.0134 | 0.0046 | 0.0000 | 0.0000 | 0.0012 | 0.0000 |
| hoeffdings | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| clopper_pearson | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0004 | 0.0000 |
| bentkus | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| betting_mixture | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| bentkus_diff | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| convex_order_diff | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0001 | 0.0000 |
