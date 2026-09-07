# End-to-end Seldonian classification

`LogisticRegressionSeldonianGD` on `make_synthetic` (d=5, group base rates (0.4, 0.8)), constraint |TPR_a - TPR_b| <= 0.2 at delta = 0.05, 40 seeds per cell. The true gap of each returned model is measured on 200k fresh samples. "failure" = a solution was returned whose true gap exceeds 0.2 (must stay <= delta for a Seldonian method); "solution rate" = fraction of runs that passed the safety test.

## Solution rate

| method | n=150 | n=300 | n=600 | n=1200 | n=2500 | n=5000 |
|---|---|---|---|---|---|---|
| ttest | 0.425 | 0.800 | 0.850 | 0.950 | 1.000 | 1.000 |
| hoeffdings | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| clopper_pearson | 0.000 | 0.675 | 1.000 | 1.000 | 1.000 | 1.000 |
| bentkus | 0.000 | 0.675 | 1.000 | 1.000 | 1.000 | 1.000 |
| betting_mixture | 0.000 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 |
| bentkus_diff | 0.000 | 0.050 | 0.975 | 1.000 | 1.000 | 1.000 |
| convex_order_diff | 0.000 | 0.125 | 1.000 | 1.000 | 1.000 | 1.000 |

## Failure rate (unsafe solution returned)

| method | n=150 | n=300 | n=600 | n=1200 | n=2500 | n=5000 |
|---|---|---|---|---|---|---|
| ttest | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| hoeffdings | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| clopper_pearson | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| bentkus | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| betting_mixture | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| bentkus_diff | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| convex_order_diff | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Accuracy of returned solutions (200k test)

| method | n=150 | n=300 | n=600 | n=1200 | n=2500 | n=5000 |
|---|---|---|---|---|---|---|
| ttest | 0.606 | 0.633 | 0.639 | 0.655 | 0.670 | 0.688 |
| hoeffdings | nan | nan | nan | nan | 0.600 | 0.600 |
| clopper_pearson | nan | 0.600 | 0.600 | 0.623 | 0.665 | 0.686 |
| bentkus | nan | 0.593 | 0.600 | 0.619 | 0.659 | 0.676 |
| betting_mixture | nan | 0.600 | 0.600 | 0.610 | 0.653 | 0.663 |
| bentkus_diff | nan | 0.600 | 0.600 | 0.600 | 0.651 | 0.692 |
| convex_order_diff | nan | 0.599 | 0.600 | 0.603 | 0.664 | 0.697 |

## True TPR gap of returned solutions

| method | n=150 | n=300 | n=600 | n=1200 | n=2500 | n=5000 |
|---|---|---|---|---|---|---|
| ttest | 0.023 | 0.024 | 0.010 | 0.009 | 0.012 | 0.014 |
| hoeffdings | nan | nan | nan | nan | 0.000 | 0.000 |
| clopper_pearson | nan | 0.000 | 0.000 | 0.003 | 0.011 | 0.013 |
| bentkus | nan | 0.000 | 0.000 | 0.002 | 0.009 | 0.010 |
| betting_mixture | nan | 0.000 | 0.000 | 0.001 | 0.007 | 0.007 |
| bentkus_diff | nan | 0.000 | 0.000 | 0.000 | 0.007 | 0.015 |
| convex_order_diff | nan | 0.000 | 0.000 | 0.000 | 0.010 | 0.017 |

## Unconstrained logistic regression reference

| n | accuracy | true gap | P(gap > threshold) |
|---|---|---|---|
| 150 | 0.751 | 0.378 | 0.975 |
| 300 | 0.754 | 0.383 | 1.000 |
| 600 | 0.757 | 0.414 | 1.000 |
| 1200 | 0.758 | 0.433 | 1.000 |
| 2500 | 0.759 | 0.427 | 1.000 |
| 5000 | 0.759 | 0.434 | 1.000 |
