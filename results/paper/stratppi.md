# StratPPI beside this project's bounds, same resamples

- miss: share of draws with the bound under the truth; it should be at most delta.
- ESS: (mean bound minus truth, for the block's first arm) / (the same for this arm), squared: how many
  times more safety samples the first arm would need for the same bound. Given only where the arm is valid.
- pass: share of draws certifying tau = truth + 0.02.

## A. The reference rate as stratifier and predictor, delta 0.05

Largest miss over the label's checkpoints; ESS and pass at its last checkpoint.

| label (rate) | n_s | random + pooled Wilson (R) | S2 + b1w (this paper) | S2 + Wald-t b1 | S2 + StratPPI | S2 + StratPPI, bootstrap-t | random + PPI++ normal | random + PPI++ bootstrap-t |
|---|---|---|---|---|---|---|---|---|
| C1:refusal (0.16) | 100 | 0.033; ESS 1.00; pass 0.12 | 0.023; ESS 2.39; pass 0.13 | 0.017; ESS 2.13; pass 0.11 | 0.066; **invalid** | 0.042; ESS 2.58; pass 0.19 | 0.050; ESS 3.38; pass 0.21 | 0.020; ESS 2.10; pass 0.12 |
| C1:refusal (0.16) | 200 | 0.028; ESS 1.00; pass 0.11 | 0.024; ESS 2.30; pass 0.23 | 0.026; ESS 2.32; pass 0.24 | 0.052; ESS 3.46; pass 0.36 | 0.042; ESS 3.10; pass 0.33 | 0.021; ESS 2.19; pass 0.22 | 0.009; ESS 1.62; pass 0.14 |
| C2:gated (0.02) | 100 | 0.240; **invalid** | 0.238; **invalid** | 0.000; ESS 0.77; pass 0.13 | 0.248; **invalid** | 0.010; ESS 0.00; pass 0.03 | 0.261; **invalid** | 0.010; ESS 0.00; pass 0.05 |
| C2:gated (0.02) | 200 | 0.048; ESS 1.00; pass 0.44 | 0.038; ESS 1.04; pass 0.42 | 0.027; ESS 1.07; pass 0.42 | 0.180; **invalid** | 0.002; ESS 0.01; pass 0.22 | 0.161; **invalid** | 0.002; ESS 0.00; pass 0.19 |
| C2:refusal (0.93) | 100 | 0.057; **invalid** | 0.054; ESS 0.96; pass 0.24 | 0.012; ESS 0.43; pass 0.07 | 0.032; ESS 0.78; pass 0.18 | 0.031; ESS 0.83; pass 0.20 | 0.013; ESS 0.59; pass 0.12 | 0.035; ESS 0.83; pass 0.21 |
| C2:refusal (0.93) | 200 | 0.043; ESS 1.00; pass 0.42 | 0.045; ESS 1.12; pass 0.40 | 0.013; ESS 0.65; pass 0.24 | 0.026; ESS 0.92; pass 0.34 | 0.033; ESS 1.03; pass 0.38 | 0.012; ESS 0.72; pass 0.24 | 0.027; ESS 1.02; pass 0.37 |
| C2:unsafe (0.09) | 100 | 0.038; ESS 1.00; pass 0.16 | 0.034; ESS 1.38; pass 0.13 | 0.036; ESS 1.37; pass 0.13 | 0.086; **invalid** | 0.045; ESS 1.43; pass 0.16 | 0.092; **invalid** | 0.023; ESS 0.90; pass 0.11 |
| C2:unsafe (0.09) | 200 | 0.025; ESS 1.00; pass 0.20 | 0.027; ESS 1.36; pass 0.23 | 0.031; ESS 1.49; pass 0.25 | 0.063; **invalid** | 0.041; ESS 1.59; pass 0.27 | 0.059; **invalid** | 0.023; ESS 1.24; pass 0.19 |
| C3:refusal (0.65) | 100 | 0.042; ESS 1.00; pass 0.09 | 0.047; ESS 4.75; pass 0.21 | 0.011; ESS 2.62; pass 0.08 | 0.056; **invalid** | 0.033; ESS 1.77; pass 0.17 | 0.004; ESS 2.46; pass 0.05 | 0.007; ESS 2.73; pass 0.08 |
| C3:refusal (0.65) | 200 | 0.026; ESS 1.00; pass 0.08 | 0.040; ESS 5.19; pass 0.41 | 0.017; ESS 3.66; pass 0.27 | 0.045; ESS 6.29; pass 0.46 | 0.034; ESS 5.35; pass 0.41 | 0.001; ESS 1.81; pass 0.04 | 0.002; ESS 1.94; pass 0.06 |
| C3:unsafe (0.01) | 100 | 0.448; **invalid** | 0.443; **invalid** | 0.000; ESS 0.42; pass 0.00 | 0.445; **invalid** | 0.002; ESS 0.00; pass 0.04 | 0.461; **invalid** | 0.001; ESS 0.00; pass 0.05 |
| C3:unsafe (0.01) | 200 | 0.178; **invalid** | 0.174; **invalid** | 0.000; ESS 0.85; pass 0.66 | 0.176; **invalid** | 0.001; ESS 0.00; pass 0.05 | 0.194; **invalid** | 0.001; ESS 0.00; pass 0.05 |
| C1:refusal pushed (014) (0.18) | 100 | 0.043; ESS 1.00; pass 0.07 | 0.024; ESS 2.19; pass 0.13 | 0.021; ESS 2.00; pass 0.11 | 0.056; ESS 3.24; pass 0.22 | 0.036; ESS 2.43; pass 0.16 | 0.051; ESS 2.84; pass 0.19 | 0.022; ESS 1.80; pass 0.10 |
| C1:refusal pushed (014) (0.18) | 200 | 0.020; ESS 1.00; pass 0.12 | 0.023; ESS 2.13; pass 0.21 | 0.025; ESS 2.17; pass 0.22 | 0.045; ESS 2.85; pass 0.30 | 0.037; ESS 2.49; pass 0.26 | 0.022; ESS 1.97; pass 0.20 | 0.011; ESS 1.48; pass 0.13 |

## A. The reference rate as stratifier and predictor, delta 0.1

Largest miss over the label's checkpoints; ESS and pass at its last checkpoint.

| label (rate) | n_s | random + pooled Wilson (R) | S2 + b1w (this paper) | S2 + Wald-t b1 | S2 + StratPPI | S2 + StratPPI, bootstrap-t | random + PPI++ normal | random + PPI++ bootstrap-t |
|---|---|---|---|---|---|---|---|---|
| C1:refusal (0.16) | 100 | 0.066; ESS 1.00; pass 0.19 | 0.076; ESS 2.52; pass 0.26 | 0.046; ESS 2.06; pass 0.22 | 0.114; **invalid** | 0.088; ESS 2.83; pass 0.32 | 0.103; ESS 3.41; pass 0.34 | 0.054; ESS 2.09; pass 0.23 |
| C1:refusal (0.16) | 200 | 0.075; ESS 1.00; pass 0.23 | 0.069; ESS 2.35; pass 0.41 | 0.067; ESS 2.24; pass 0.40 | 0.096; ESS 3.29; pass 0.50 | 0.089; ESS 3.05; pass 0.48 | 0.057; ESS 2.16; pass 0.38 | 0.033; ESS 1.57; pass 0.29 |
| C2:gated (0.02) | 100 | 0.240; **invalid** | 0.238; **invalid** | 0.000; ESS 0.64; pass 0.42 | 0.266; **invalid** | 0.018; ESS 0.00; pass 0.08 | 0.293; **invalid** | 0.017; ESS 0.00; pass 0.08 |
| C2:gated (0.02) | 200 | 0.079; ESS 1.00; pass 0.66 | 0.067; ESS 1.03; pass 0.64 | 0.067; ESS 0.90; pass 0.64 | 0.204; **invalid** | 0.015; ESS 0.01; pass 0.49 | 0.234; **invalid** | 0.011; ESS 0.01; pass 0.42 |
| C2:refusal (0.93) | 100 | 0.114; **invalid** | 0.097; ESS 1.05; pass 0.41 | 0.040; ESS 0.49; pass 0.19 | 0.069; ESS 0.87; pass 0.32 | 0.076; ESS 0.95; pass 0.34 | 0.043; ESS 0.63; pass 0.24 | 0.084; ESS 0.92; pass 0.35 |
| C2:refusal (0.93) | 200 | 0.087; ESS 1.00; pass 0.55 | 0.091; ESS 1.18; pass 0.54 | 0.044; ESS 0.71; pass 0.40 | 0.066; ESS 0.99; pass 0.50 | 0.078; ESS 1.12; pass 0.54 | 0.038; ESS 0.76; pass 0.41 | 0.073; ESS 1.13; pass 0.54 |
| C2:unsafe (0.09) | 100 | 0.088; ESS 1.00; pass 0.27 | 0.085; ESS 1.40; pass 0.26 | 0.074; ESS 1.26; pass 0.25 | 0.144; **invalid** | 0.094; ESS 1.47; pass 0.29 | 0.152; **invalid** | 0.067; ESS 1.06; pass 0.24 |
| C2:unsafe (0.09) | 200 | 0.084; ESS 1.00; pass 0.29 | 0.080; ESS 1.36; pass 0.38 | 0.080; ESS 1.39; pass 0.38 | 0.114; **invalid** | 0.089; ESS 1.53; pass 0.42 | 0.101; ESS 1.87; pass 0.46 | 0.060; ESS 1.21; pass 0.35 |
| C3:refusal (0.65) | 100 | 0.091; ESS 1.00; pass 0.18 | 0.092; ESS 5.06; pass 0.34 | 0.039; ESS 2.74; pass 0.21 | 0.097; ESS 6.43; pass 0.43 | 0.073; ESS 2.15; pass 0.32 | 0.019; ESS 2.46; pass 0.16 | 0.028; ESS 2.81; pass 0.21 |
| C3:refusal (0.65) | 200 | 0.072; ESS 1.00; pass 0.20 | 0.093; ESS 5.40; pass 0.56 | 0.054; ESS 3.78; pass 0.44 | 0.093; ESS 6.41; pass 0.61 | 0.082; ESS 5.79; pass 0.58 | 0.005; ESS 1.83; pass 0.17 | 0.007; ESS 2.01; pass 0.21 |
| C3:unsafe (0.01) | 100 | 0.448; **invalid** | 0.443; **invalid** | 0.000; ESS 0.34; pass 0.34 | 0.454; **invalid** | 0.015; ESS 0.00; pass 0.05 | 0.478; **invalid** | 0.007; ESS 0.00; pass 0.06 |
| C3:unsafe (0.01) | 200 | 0.178; **invalid** | 0.174; **invalid** | 0.000; ESS 0.69; pass 0.66 | 0.185; **invalid** | 0.001; ESS 0.00; pass 0.16 | 0.217; **invalid** | 0.002; ESS 0.00; pass 0.16 |
| C1:refusal pushed (014) (0.18) | 100 | 0.084; ESS 1.00; pass 0.20 | 0.067; ESS 2.28; pass 0.25 | 0.055; ESS 1.94; pass 0.23 | 0.108; ESS 3.09; pass 0.34 | 0.085; ESS 2.45; pass 0.30 | 0.099; ESS 2.86; pass 0.32 | 0.056; ESS 1.79; pass 0.21 |
| C1:refusal pushed (014) (0.18) | 200 | 0.057; ESS 1.00; pass 0.24 | 0.064; ESS 2.17; pass 0.37 | 0.062; ESS 2.10; pass 0.36 | 0.085; ESS 2.74; pass 0.44 | 0.078; ESS 2.46; pass 0.41 | 0.059; ESS 1.95; pass 0.35 | 0.035; ESS 1.44; pass 0.26 |

## A, summary over the mid-rate labels (5-95%)

| arm | delta | largest miss | cells valid | median ESS where valid |
|---|---|---|---|---|
| random + pooled Wilson (R) | 0.05 | 0.057 | 9 of 10 | 1.00 |
| random + pooled Wilson (R) | 0.1 | 0.114 | 9 of 10 | 1.00 |
| S2 + b1w (this paper) | 0.05 | 0.054 | 10 of 10 | 2.16 |
| S2 + b1w (this paper) | 0.1 | 0.097 | 10 of 10 | 2.23 |
| S2 + Wald-t b1 | 0.05 | 0.036 | 10 of 10 | 2.06 |
| S2 + Wald-t b1 | 0.1 | 0.080 | 10 of 10 | 2.00 |
| S2 + StratPPI | 0.05 | 0.086 | 6 of 10 | 3.04 |
| S2 + StratPPI | 0.1 | 0.144 | 7 of 10 | 3.09 |
| S2 + StratPPI, bootstrap-t | 0.05 | 0.045 | 10 of 10 | 2.10 |
| S2 + StratPPI, bootstrap-t | 0.1 | 0.094 | 10 of 10 | 2.30 |
| random + PPI++ normal | 0.05 | 0.092 | 8 of 10 | 2.08 |
| random + PPI++ normal | 0.1 | 0.152 | 9 of 10 | 1.95 |
| random + PPI++ bootstrap-t | 0.05 | 0.035 | 10 of 10 | 1.55 |
| random + PPI++ bootstrap-t | 0.1 | 0.084 | 10 of 10 | 1.51 |

## B. A judge's logit as the predictor, delta 0.05

| pool | rate | n of N | labels alone, Clopper-Pearson | PPI++ normal | PPI++ bootstrap-t (this paper) | StratPPI, K=5 | StratPPI, K=10 | StratPPI, bootstrap-t, K=5 | StratPPI, bootstrap-t, K=10 | judge strata + b1w, K=5 | judge strata + b1w, K=10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| refusal|raw|0 | 0.013 | 225 of 4000 | 0.000; ESS 1.00 | 0.199; **invalid** | 0.000; ESS 0.00 | 0.187; **invalid** | 0.182; **invalid** | 0.001; ESS 0.00 | 0.003; ESS 0.00 | 0.047; ESS 1.22 | 0.040; ESS 1.24 |
| refusal|raw|0 | 0.013 | 1000 of 4000 | 0.023; ESS 1.00 | 0.088; **invalid** | 0.031; ESS 1.10 | 0.078; **invalid** | 0.091; **invalid** | 0.030; ESS 1.09 | 0.033; ESS 1.20 | 0.041; ESS 1.04 | 0.047; ESS 1.13 |
| refusal|raw|0 | 0.050 | 225 of 4000 | 0.024; ESS 1.00 | 0.086; **invalid** | 0.029; ESS 1.23 | 0.092; **invalid** | 0.093; **invalid** | 0.037; ESS 1.67 | 0.037; ESS 1.78 | 0.021; ESS 1.24 | 0.036; ESS 1.43 |
| refusal|raw|0 | 0.050 | 1000 of 4000 | 0.047; ESS 1.00 | 0.069; **invalid** | 0.050; ESS 1.29 | 0.074; **invalid** | 0.066; **invalid** | 0.053; ESS 1.85 | 0.048; ESS 1.84 | 0.046; ESS 1.32 | 0.033; ESS 1.42 |
| refusal|raw|0 | 0.200 | 100 of 2000 | 0.050; ESS 1.00 | 0.070; **invalid** | 0.046; ESS 2.12 | 0.074; **invalid** | 0.086; **invalid** | 0.051; ESS 2.32 | 0.049; ESS 2.48 | 0.026; ESS 2.33 | 0.029; ESS 2.57 |
| refusal|raw|0 | 0.200 | 225 of 2000 | 0.040; ESS 1.00 | 0.068; **invalid** | 0.052; ESS 1.96 | 0.062; **invalid** | 0.068; **invalid** | 0.048; ESS 2.70 | 0.049; ESS 2.58 | 0.035; ESS 2.24 | 0.041; ESS 2.41 |
| refusal|raw|0 | 0.200 | 500 of 2000 | 0.043; ESS 1.00 | 0.061; **invalid** | 0.050; ESS 1.73 | 0.056; ESS 2.94 | 0.061; **invalid** | 0.049; ESS 2.68 | 0.053; ESS 2.77 | 0.040; ESS 2.25 | 0.043; ESS 2.52 |
| refusal|rubric|0 | 0.013 | 225 of 4000 | 0.000; ESS 1.00 | 0.232; **invalid** | 0.008; ESS 0.00 | 0.232; **invalid** | 0.239; **invalid** | 0.010; ESS 0.00 | 0.012; ESS 0.00 | 0.046; ESS 1.23 | 0.053; ESS 1.29 |
| refusal|rubric|0 | 0.013 | 1000 of 4000 | 0.023; ESS 1.00 | 0.103; **invalid** | 0.029; ESS 1.32 | 0.109; **invalid** | 0.109; **invalid** | 0.040; ESS 1.53 | 0.039; ESS 1.19 | 0.049; ESS 1.10 | 0.048; ESS 1.17 |
| refusal|rubric|0 | 0.050 | 225 of 4000 | 0.027; ESS 1.00 | 0.121; **invalid** | 0.033; ESS 1.60 | 0.130; **invalid** | 0.100; **invalid** | 0.042; ESS 1.79 | 0.033; ESS 1.99 | 0.033; ESS 1.28 | 0.031; ESS 1.47 |
| refusal|rubric|0 | 0.050 | 1000 of 4000 | 0.041; ESS 1.00 | 0.079; **invalid** | 0.043; ESS 1.49 | 0.082; **invalid** | 0.072; **invalid** | 0.049; ESS 1.72 | 0.050; ESS 1.91 | 0.045; ESS 1.16 | 0.038; ESS 1.36 |
| refusal|rubric|0 | 0.200 | 100 of 2000 | 0.051; ESS 1.00 | 0.112; **invalid** | 0.042; ESS 1.61 | 0.079; **invalid** | 0.084; **invalid** | 0.056; ESS 2.13 | 0.052; ESS 2.11 | 0.034; ESS 2.25 | 0.038; ESS 2.40 |
| refusal|rubric|0 | 0.200 | 225 of 2000 | 0.036; ESS 1.00 | 0.078; **invalid** | 0.040; ESS 1.61 | 0.066; **invalid** | 0.064; **invalid** | 0.046; ESS 2.50 | 0.050; ESS 2.56 | 0.035; ESS 2.22 | 0.038; ESS 2.41 |
| refusal|rubric|0 | 0.200 | 500 of 2000 | 0.041; ESS 1.00 | 0.066; **invalid** | 0.042; ESS 1.50 | 0.059; **invalid** | 0.053; ESS 2.87 | 0.048; ESS 2.44 | 0.044; ESS 2.52 | 0.040; ESS 2.16 | 0.039; ESS 2.32 |

Largest miss over the cells of B: labels alone, Clopper-Pearson 0.051; PPI++ normal 0.232; PPI++ bootstrap-t (this paper) 0.052; StratPPI, K=5 0.232; StratPPI, K=10 0.239; StratPPI, bootstrap-t, K=5 0.056; StratPPI, bootstrap-t, K=10 0.053; judge strata + b1w, K=5 0.049; judge strata + b1w, K=10 0.053.
