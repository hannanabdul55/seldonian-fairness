# Spike 001 results (40 seeds per setting)

| setting | corr(A, delta) | corr(r - group mean, delta) | corr(A, true adv) | corr(critic delta, delta) | within-group ranks equal | max abs A (bound) | group-mean share of var(delta) | learnable share early / late |
|---|---|---|---|---|---|---|---|---|
| grpo p1 | 0.892 | 0.935 | 0.291 | 0.981 | 1.000 | 2.461 (2.475) | 0.126 | 0.255 / 0.053 |
| seldonian_lag p1 | 0.741 | 0.936 | 0.249 | 0.953 | 1.000 | 2.474 (2.475) | 0.124 | 0.185 / 0.067 |
| grpo p4 | 0.867 | 0.935 | 0.349 | 0.929 | 1.000 | 2.467 (2.475) | 0.126 | 0.449 / 0.113 |
| seldonian_lag p4 | 0.686 | 0.936 | 0.190 | 0.969 | 1.000 | 2.475 (2.475) | 0.125 | 0.070 / 0.049 |

Per-step magnitude: can a step-level spike in mean |delta| (robust z > 2 on the step-to-step jump) be seen in the GRPO-side series?

| setting | spikes per run | series | corr with mean abs delta | recall | precision |
|---|---|---|---|---|---|
| grpo p1 | 11.8 | mean|adv| | 0.137 | 0.05 | 0.05 |
| grpo p1 | 11.8 | max|adv| | -0.043 | 0.04 | 0.05 |
| grpo p1 | 11.8 | mean group sd | 0.946 | 0.62 | 0.62 |
| grpo p1 | 11.8 | mean|centred| | 0.954 | 0.65 | 0.64 |
| grpo p1 | 11.8 | mean|delta_critic| | 0.944 | 0.67 | 0.58 |
| grpo p1 | 11.8 | mean|adv_true| | 0.756 | 0.35 | 0.21 |
| seldonian_lag p1 | 34.8 | mean|adv| | -0.207 | 0.13 | 0.42 |
| seldonian_lag p1 | 34.8 | max|adv| | 0.549 | 0.14 | 0.18 |
| seldonian_lag p1 | 34.8 | mean group sd | 0.980 | 0.78 | 0.75 |
| seldonian_lag p1 | 34.8 | mean|centred| | 0.986 | 0.82 | 0.80 |
| seldonian_lag p1 | 34.8 | mean|delta_critic| | 0.980 | 0.77 | 0.81 |
| seldonian_lag p1 | 34.8 | mean|adv_true| | 0.658 | 0.30 | 0.42 |
| grpo p4 | 15.6 | mean|adv| | 0.160 | 0.07 | 0.11 |
| grpo p4 | 15.6 | max|adv| | 0.033 | 0.04 | 0.10 |
| grpo p4 | 15.6 | mean group sd | 0.971 | 0.68 | 0.67 |
| grpo p4 | 15.6 | mean|centred| | 0.977 | 0.70 | 0.69 |
| grpo p4 | 15.6 | mean|delta_critic| | 0.924 | 0.58 | 0.45 |
| grpo p4 | 15.6 | mean|adv_true| | 0.908 | 0.52 | 0.43 |
| seldonian_lag p4 | 33.5 | mean|adv| | 0.074 | 0.10 | 0.31 |
| seldonian_lag p4 | 33.5 | max|adv| | 0.409 | 0.22 | 0.13 |
| seldonian_lag p4 | 33.5 | mean group sd | 0.969 | 0.76 | 0.72 |
| seldonian_lag p4 | 33.5 | mean|centred| | 0.978 | 0.79 | 0.77 |
| seldonian_lag p4 | 33.5 | mean|delta_critic| | 0.985 | 0.79 | 0.86 |
| seldonian_lag p4 | 33.5 | mean|adv_true| | 0.469 | 0.28 | 0.38 |

sd across steps of mean |A| (per run, averaged): grpo p1 0.0214, seldonian_lag p1 0.0289, grpo p4 0.0223, seldonian_lag p4 0.0332

coefficient of variation across steps of mean |delta|: grpo p1 0.149, seldonian_lag p1 0.480, grpo p4 0.236, seldonian_lag p4 0.558

Per-step correlation with the multiplier lambda (20 seeds):

| setting | corr(mean abs delta, lambda) | corr(mean abs noise part, lambda) | corr(mean abs learnable part, lambda) |
|---|---|---|---|
| seldonian_lag p1 | 0.863 | 0.863 | 0.569 |
| seldonian_lag p4 | 0.814 | 0.818 | 0.414 |
