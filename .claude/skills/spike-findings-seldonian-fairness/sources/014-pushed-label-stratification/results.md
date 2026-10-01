# Spike 014 stage 2-3: Granite over-refusal under the Lagrangian

## Rate, ICC and rho across the run

Reference (013's 8 `cov` samples per C1 prompt): rate 0.186, ICC_ref 0.72. Threshold 0.206. Safety test on D_s (one response per prompt, n 500): rate 0.168, Clopper-Pearson upper 0.198, passed True; selected step 175 (best feasible predicted test).

Predicted-test trajectory during training (D_c subsample, `predict_every` steps):

| step | rate | upper | feasible | lambda | reward |
|---|---|---|---|---|---|
| 25 | 0.113 | 0.162 | True | 0.6 | 3.33 |
| 50 | 0.262 | 0.323 | False | 12.3 | 4.52 |
| 75 | 0.129 | 0.179 | True | 9.7 | 4.28 |
| 100 | 0.102 | 0.148 | True | 5.0 | 4.35 |
| 125 | 0.129 | 0.179 | True | 5.0 | 4.57 |
| 150 | 0.125 | 0.175 | True | 5.0 | 4.72 |
| 175 | 0.141 | 0.192 | True | 5.0 | 4.90 |
| 200 | 0.160 | 0.214 | False | 5.8 | 5.00 |

Checkpoints sampled on the pool (all K samples; rho disattenuated for both sides' sampling noise, as 013's pre-flight does):

| run | step | K | rate | moved | ICC_cand | ICC_cand/ICC_ref | r_obs | rho | pred meas | pred naive |
|---|---|---|---|---|---|---|---|---|---|---|
| 014 pushed (Lagrangian, C4) | 100 | 4 | 0.151 | -0.034 | 0.73 | 1.03 | 0.85 | 0.91 | 2.27 | 1.72 |
| 014 pushed (Lagrangian, C4) | 200 | 12 | 0.182 | -0.004 | 0.72 | 1.01 | 0.88 | 0.92 | 2.26 | 1.72 |
| 013 side-effect (GRPO, C1) | 0 | 16 | 0.181 | -0.005 | 0.71 | 0.99 | 0.96 | 1.00 | 2.83 | 1.72 |
| 013 side-effect (GRPO, C1) | 100 | 16 | 0.162 | -0.023 | 0.72 | 1.01 | 0.96 | 1.00 | 2.92 | 1.72 |
| 013 side-effect (GRPO, C1) | 200 | 16 | 0.166 | -0.019 | 0.71 | 0.99 | 0.96 | 0.99 | 2.82 | 1.72 |

## Realised ESS of 013's rule (S2, k 8, H 8, n_s 200, `b1w`) and the predictions

`bandit` = the gate's cells whose ICC_cand/ICC_ref is within 0.15 of the real checkpoint's, their rho carried into the formula at the real ICC_ref; `meas` = the formula at the real checkpoint's measured ICC_cand and rho (all K samples); `naive` = ICC_ref, rho 0.8. Miss against delta; valid = miss <= delta + 2 MC se.

| run | step | delta | miss R | miss S2 | valid | width R | width S2 | realised ESS | bandit (rho, cells) | meas | naive |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 014 pushed (Lagrangian, C4) | 100 | 0.05 | 0.020 | 0.011 | True | 0.0461 | 0.0322 | **2.05** | 2.03 (0.87, no cell within 0.15; nearest ratio 1.21) | 2.27 | 1.72 |
| 014 pushed (Lagrangian, C4) | 100 | 0.1 | 0.057 | 0.040 | True | 0.0352 | 0.0242 | **2.11** | 2.03 (0.87, no cell within 0.15; nearest ratio 1.21) | 2.27 | 1.72 |
| 014 pushed (Lagrangian, C4) | 200 | 0.05 | 0.017 | 0.023 | True | 0.0491 | 0.0334 | **2.16** | 2.00 (0.87, no cell within 0.15; nearest ratio 1.21) | 2.26 | 1.72 |
| 014 pushed (Lagrangian, C4) | 200 | 0.1 | 0.051 | 0.064 | True | 0.0377 | 0.0253 | **2.21** | 2.00 (0.87, no cell within 0.15; nearest ratio 1.21) | 2.26 | 1.72 |
| 013 side-effect (GRPO, C1) | 100 | 0.05 | 0.016 | 0.024 | True | 0.0468 | 0.0301 | **2.41** | 2.00 (0.87, no cell within 0.15; nearest ratio 1.21) | 2.92 | 1.72 |
| 013 side-effect (GRPO, C1) | 100 | 0.1 | 0.048 | 0.068 | True | 0.0358 | 0.0227 | **2.49** | 2.00 (0.87, no cell within 0.15; nearest ratio 1.21) | 2.92 | 1.72 |
| 013 side-effect (GRPO, C1) | 200 | 0.05 | 0.028 | 0.022 | True | 0.0475 | 0.0310 | **2.35** | 1.97 (0.87, no cell within 0.15; nearest ratio 1.21) | 2.82 | 1.72 |
| 013 side-effect (GRPO, C1) | 200 | 0.1 | 0.075 | 0.069 | True | 0.0363 | 0.0233 | **2.42** | 1.97 (0.87, no cell within 0.15; nearest ratio 1.21) | 2.82 | 1.72 |

## H3 and H4

- **H3** (bandit cells matching the real compression 1.01 +- 0.15, constrained, eta 400, 200 steps: 4 cells): realised bandit ESS 3.35-3.35, stratified miss 0.073-0.093 at delta 0.1. H3 asks ESS >= 1.2 and valid: **holds**.
- **H4** at delta 0.1: realised ESS 2.21 at step 200, bandit-calibrated prediction 2.00 (|error| 0.21; asks <= 0.3), `b1w` miss 0.064 (valid): **holds**. Formula at the measured moderators 2.26 (|error| 0.04); naive 1.72 (|error| 0.49).
- **H4** at delta 0.05: realised ESS 2.16 at step 200, bandit-calibrated prediction 2.00 (|error| 0.16; asks <= 0.3), `b1w` miss 0.023 (valid): **holds**. Formula at the measured moderators 2.26 (|error| 0.09); naive 1.72 (|error| 0.44).
- Like for like at delta 0.1, step 200: pushed ESS 2.21 against 013's side-effect ESS 2.42 on the same prompts and reference labels.
