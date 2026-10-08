# The robot benchmark's sampling unit (paper section 10.1)

`scripts/robodojo_sampling.py`, from `results/spikes/019/trials.csv` (360 trials). A safety stop is the harness ending the trial for overheating.

## 1. Layout: 6 tasks by 20 trials per model

| task | Claude Opus 5 | Claude Opus 5.5 | GPT-6 Astra |
|---|---|---|---|
| Cap Pen | 2 / 20 | 0 / 20 | 0 / 20 |
| Classify Objects | 1 / 20 | 0 / 20 | 0 / 20 |
| Pack And Pour Fruit | 2 / 20 | 1 / 20 | 2 / 20 |
| Stack Bowls | 0 / 20 | 0 / 20 | 0 / 20 |
| Stand Up Bottles | 1 / 20 | 0 / 20 | 0 / 20 |
| Store In Safe | 4 / 20 | 1 / 20 | 0 / 20 |
| all | 10 / 120 | 2 / 120 | 2 / 120 |

At most 6 trials ran at once and never two of the same task (largest number of one task's trials running together: 1), which is what one station per task would give (the report's run paths name six rigs, one per task, so task and rig cannot be told apart). On each task the three models ran one after another (0 overlaps between one model's last trial and the next model's first), in the same order on all six: GPT-6 Astra, then Claude Opus 5.5, then Claude Opus 5.

| model | first trial started (UTC) | last trial started | hours into the session |
|---|---|---|---|
| Claude Opus 5 | 2026-09-22T20:40 | 2026-09-23T03:19 | 4.4 to 11.1 |
| Claude Opus 5.5 | 2026-09-22T18:18 | 2026-09-23T00:14 | 2.1 to 8.0 |
| GPT-6 Astra | 2026-09-22T16:13 | 2026-09-22T21:56 | 0.0 to 5.7 |

## 2. Clustering by task

| model | stops | exact permutation p, equal rates across tasks | ICC by task | design effect |
|---|---|---|---|---|
| Claude Opus 5 | 10 | 0.370 | 0.011 | 1.22 |
| Claude Opus 5.5 | 2 | 1.000 | -0.010 | 0.81 |
| GPT-6 Astra | 2 | 0.158 | 0.053 | 2.03 |

The permutation test deals each model's stops to its 120 trials at random (a Monte Carlo permutation test, 200,000 deals; statistic: the sum of squared task counts). The design effect is section 6.3's cluster variance over the binomial variance; with six clusters it is itself a noisy number.

## 3. Tasks taken as fixed: the pooled Clopper-Pearson limit

If the six tasks are the population (the rate over this task mix, each task weighted equally) and trials within a task are independent, the count of stops is a sum of six binomials with their own rates. The table gives the exact probability that the pooled Clopper-Pearson limit falls below the mean rate, for task rates with that mean spread three ways.

| model | mean rate | shape | miss at delta 0.05 | miss at delta 0.10 |
|---|---|---|---|---|
| Claude Opus 5 | 0.083 | even over tasks | 0.0246 | 0.0592 |
| Claude Opus 5 | 0.083 | in the observed shares | 0.0220 | 0.0546 |
| Claude Opus 5 | 0.083 | all on one task | 0.0059 | 0.0207 |
| Claude Opus 5 | 0.050 | even over tasks | 0.0155 | 0.0575 |
| Claude Opus 5 | 0.050 | in the observed shares | 0.0145 | 0.0550 |
| Claude Opus 5 | 0.050 | all on one task | 0.0076 | 0.0355 |
| Claude Opus 5.5 | 0.017 | even over tasks | 0.0000 | 0.0000 |
| Claude Opus 5.5 | 0.017 | in the observed shares | 0.0000 | 0.0000 |
| Claude Opus 5.5 | 0.017 | all on one task | 0.0000 | 0.0000 |
| Claude Opus 5.5 | 0.050 | even over tasks | 0.0155 | 0.0575 |
| Claude Opus 5.5 | 0.050 | in the observed shares | 0.0121 | 0.0486 |
| Claude Opus 5.5 | 0.050 | all on one task | 0.0076 | 0.0355 |
| GPT-6 Astra | 0.017 | even over tasks | 0.0000 | 0.0000 |
| GPT-6 Astra | 0.017 | in the observed shares | 0.0000 | 0.0000 |
| GPT-6 Astra | 0.017 | all on one task | 0.0000 | 0.0000 |
| GPT-6 Astra | 0.050 | even over tasks | 0.0155 | 0.0575 |
| GPT-6 Astra | 0.050 | in the observed shares | 0.0076 | 0.0355 |
| GPT-6 Astra | 0.050 | all on one task | 0.0076 | 0.0355 |

Largest miss over mean rates of 0.005 to 0.160 in steps of 0.0005: delta 0.05: even 0.0500, all on two tasks 0.0442, all on one task 0.0388; delta 0.1: even 0.0999, all on two tasks 0.0897, all on one task 0.0830. Over these rates the miss stays under delta and unequal task rates only lower it (Hoeffding, 1956), so for this population the pooled limit is valid and no clustering correction applies.

## 4. Tasks taken as a sample of tasks like them

The six task rates are then six draws in [0, 1] whose mean is the rate of interest. Upper limits:

**delta 0.05**

| model | pooled CP (tasks fixed) | Hoeffding (exact) | Anderson (exact) | Bentkus (exact) | betting mixture (exact) | Student-t on task rates (approximate) | cluster bootstrap-t (approximate) |
|---|---|---|---|---|---|---|---|
| Claude Opus 5 | 0.137 | 0.583 | 0.566 | 0.552 | 0.480 | 0.140 | 0.159 |
| Claude Opus 5.5 | 0.052 | 0.516 | 0.516 | 0.440 | 0.441 | 0.038 | 1.000 |
| GPT-6 Astra | 0.052 | 0.516 | 0.516 | 0.440 | 0.442 | 0.050 | 1.000 |

**delta 0.1**

| model | pooled CP (tasks fixed) | Hoeffding (exact) | Anderson (exact) | Bentkus (exact) | betting mixture (exact) | Student-t on task rates (approximate) | cluster bootstrap-t (approximate) |
|---|---|---|---|---|---|---|---|
| Claude Opus 5 | 0.125 | 0.521 | 0.508 | 0.486 | 0.411 | 0.124 | 0.134 |
| Claude Opus 5.5 | 0.044 | 0.455 | 0.455 | 0.368 | 0.367 | 0.032 | 0.027 |
| GPT-6 Astra | 0.044 | 0.455 | 0.455 | 0.368 | 0.367 | 0.041 | 1.000 |

With all six task rates at zero no valid bound can return less than `1 - delta^(1/6)`: 0.393 at delta 0.05 and 0.319 at 0.10 (a population in which a share q of tasks always stops and the rest never do shows six clean tasks with probability (1 - q)^6). The pooled limit with nothing observed is 0.025 and 0.019. Neither approximate limit is usable with six clusters: the Student-t limit on six task rates comes out below the pooled limit for the two models with two stops, and the cluster bootstrap-t returns 1 (no limit) when resamples of six tasks with no stop among them are common (200,000 draws).

## 5. Independence of trials within a task

| model | stops in the first / second half of its block | rank-sum p, start time of stops against the rest | adjacent stops within a task's sequence | permutation p |
|---|---|---|---|---|
| Claude Opus 5 | 3 / 7 | 0.23 | 0 | 1.00 |
| Claude Opus 5.5 | 2 / 0 | 0.33 | 0 | 1.00 |
| GPT-6 Astra | 0 / 2 | 0.52 | 0 | 1.00 |

Because the models ran one after another on each task, a difference between models is also a difference in the hours the session had been running. Within a model's block the data do not show stops arriving later or in runs, with 2 to 10 events to show it.
