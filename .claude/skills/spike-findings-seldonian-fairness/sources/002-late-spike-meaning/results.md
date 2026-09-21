# Spike 002 results (60 seeds per setting, 300 steps, predictors on the first 200)

## Where the late spikes come from

| setting | spikes / run | late share | spikes in 3 steps after a lambda move (base rate) | corr(abs delta, lambda) | corr(abs delta, policy step) | corr(abs true adv, policy step) |
|---|---|---|---|---|---|---|
| grpo p1 | 13.0 | 0.21 | 0.00 (0.00) | nan | 0.77 | 0.91 |
| grpo p4 | 19.2 | 0.13 | 0.00 (0.00) | nan | 0.84 | 0.91 |
| lag p1 | 32.1 | 0.40 | 0.14 (0.07) | 0.85 | 0.29 | 0.74 |
| lag p4 | 30.4 | 0.50 | 0.14 (0.10) | 0.80 | -0.31 | 0.26 |
| lag p4 floor5 | 11.3 | 0.49 | 0.03 (0.02) | 0.37 | 0.15 | 0.82 |

## Valence (mean agent TD error) around multiplier moves

| setting | mean valence, 3 steps after lambda up | after lambda down | steps to re-adapt after an increase | late mean valence | late mean negative part |
|---|---|---|---|---|---|
| grpo p1 | +nan | +nan | nan | -0.004 | 0.209 |
| grpo p4 | +nan | +nan | nan | -0.004 | 0.224 |
| lag p1 | -0.428 | +0.245 | 4.4 | +0.008 | 0.389 |
| lag p4 | -0.771 | +0.273 | 4.4 | -0.013 | 0.922 |
| lag p4 floor5 | -0.183 | +0.088 | 2.9 | -0.012 | 0.525 |

## Do late spikes predict a breach if training continued (steps 201-300)?

AUC of each step-<=200 statistic for `max true rate over 201-300 > threshold` (0.5 = no information; > 0.5 = higher value, more breaches).

| setting | future breaches | late spikes (abs delta) | late spikes (true adv) | late share | late abs delta | late policy step | lambda at 200 | late lambda moves | margin at 200 (neg.) |
|---|---|---|---|---|---|---|---|---|---|
| grpo p1 | 57/60 | 0.61 | 0.29 | 0.67 | 0.57 | 0.37 | 0.50 | 0.50 | 1.00 |
| grpo p4 | 60/60 | nan | nan | nan | nan | nan | nan | nan | nan |
| lag p1 | 22/60 | 0.47 | 0.49 | 0.46 | 0.49 | 0.62 | 0.39 | 0.50 | 0.70 |
| lag p4 | 43/60 | 0.28 | 0.25 | 0.31 | 0.30 | 0.64 | 0.23 | 0.49 | 0.60 |
| lag p4 floor5 | 0/60 | nan | nan | nan | nan | nan | nan | nan | nan |

## Incremental value over the run state (5x10-fold CV logistic regression, AUC for future breach)

| features | lag p1 | lag p4 | pooled lag p1 + p4 |
|---|---|---|---|
| state (lambda at 200, margin at 200) | 0.66 | 0.75 | 0.62 |
| state + late policy step | 0.66 | 0.75 | 0.62 |
| state + late TD spikes (count, share, mean abs delta) | 0.66 | 0.75 | 0.76 |
| state + late TD spikes in the learnable part | 0.67 | 0.74 | 0.67 |
| state + late valence (mean, negative part) | 0.67 | 0.76 | 0.75 |
| TD spikes alone | 0.34 | 0.67 | 0.63 |

## E3: Spearman(late share of spikes, fraction of feasible predicted tests in steps 1-200)

| setting | raw | partial, controlling multiplier moves and total abs change | partial, also lambda at 200 |
|---|---|---|---|
| lag p1 | -0.06 | +0.11 | +0.17 |
| lag p4 | -0.46 | -0.44 | -0.02 |
| pooled | -0.30 | -0.21 | +0.12 |
