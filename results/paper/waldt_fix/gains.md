# Gains of the skew-corrected stratified Wald-t limit on the cells of the paper's gains table

`scripts/waldt_skew_gains.py` (not registered). ESS against the pooled Wilson bound on a random draw, delta 0.05; miss at delta 0.05 / 0.10, **bold** when over by the paper's rule. `b1` is the uncorrected limit on the same draws.

| run | label | rate | step | claim | n_s | ESS b1 | ESS b1s | miss b1 | miss b1s |
|---|---|---|---|---|---|---|---|---|---|
| B3 | K5:refusal | 0.259 | 100 | pool | 100 | 1.36 | 1.24 | 0.043 / 0.095 | 0.035 / 0.081 |
| B3 | K5:refusal | 0.259 | 100 | source | 100 | 1.22 | 1.13 | 0.040 / 0.086 | 0.037 / 0.075 |
| B3 | K5:refusal | 0.259 | 100 | pool | 200 | 1.46 | 1.37 | 0.049 / 0.100 | 0.042 / 0.090 |
| B3 | K5:refusal | 0.259 | 100 | source | 200 | 1.23 | 1.16 | 0.044 / 0.094 | 0.043 / 0.094 |
| B3 | K5:refusal | 0.584 | 200 | pool | 100 | 0.97 | 0.97 | 0.044 / 0.097 | 0.043 / 0.096 |
| B3 | K5:refusal | 0.584 | 200 | source | 100 | 0.93 | 0.93 | 0.048 / 0.091 | 0.047 / 0.090 |
| B3 | K5:refusal | 0.584 | 200 | pool | 200 | 1.08 | 1.08 | 0.047 / 0.100 | 0.047 / 0.100 |
| B3 | K5:refusal | 0.584 | 200 | source | 200 | 0.96 | 0.96 | 0.045 / 0.085 | 0.045 / 0.085 |
| O6 | K5:refusal | 0.180 | 100 | pool | 100 | 1.60 | 1.41 | 0.033 / 0.077 | 0.024 / 0.063 |
| O6 | K5:refusal | 0.180 | 100 | source | 100 | 1.41 | 1.27 | 0.036 / 0.085 | 0.029 / 0.067 |
| O6 | K5:refusal | 0.180 | 100 | pool | 200 | 1.84 | 1.68 | 0.043 / 0.086 | 0.034 / 0.079 |
| O6 | K5:refusal | 0.180 | 100 | source | 200 | 1.36 | 1.28 | 0.051 / 0.087 | 0.042 / 0.082 |
| O6 | K5:refusal | 0.193 | 200 | pool | 100 | 1.52 | 1.35 | 0.036 / 0.083 | 0.025 / 0.068 |
| O6 | K5:refusal | 0.193 | 200 | source | 100 | 1.34 | 1.21 | 0.038 / 0.092 | 0.030 / 0.070 |
| O6 | K5:refusal | 0.193 | 200 | pool | 200 | 1.77 | 1.62 | 0.046 / 0.092 | 0.038 / 0.083 |
| O6 | K5:refusal | 0.193 | 200 | source | 200 | 1.32 | 1.24 | 0.049 / **0.111** | 0.047 / 0.094 |
| R5 | K1:refusal | 0.122 | 175 | pool | 100 | 1.47 | 1.26 | 0.024 / 0.056 | 0.014 / 0.043 |
| R5 | K1:refusal | 0.122 | 175 | source | 100 | 1.26 | 1.11 | 0.023 / 0.053 | 0.021 / 0.050 |
| R5 | K1:refusal | 0.122 | 175 | pool | 200 | 1.73 | 1.57 | 0.039 / 0.082 | 0.027 / 0.074 |
| R5 | K1:refusal | 0.122 | 175 | source | 200 | 1.33 | 1.23 | 0.047 / 0.088 | 0.044 / 0.075 |
| R5 | K2:refusal | 0.824 | 175 | pool | 100 | 0.86 | 0.86 | 0.025 / 0.062 | 0.025 / 0.062 |
| R5 | K2:refusal | 0.824 | 175 | source | 100 | 0.77 | 0.77 | 0.022 / 0.069 | 0.022 / 0.069 |
| R5 | K2:refusal | 0.824 | 175 | pool | 200 | 1.13 | 1.13 | 0.034 / 0.077 | 0.034 / 0.077 |
| R5 | K2:refusal | 0.824 | 175 | source | 200 | 0.88 | 0.88 | 0.030 / 0.070 | 0.030 / 0.070 |
| R5 | K2:unsafe | 0.032 | 175 | pool | 100 | 0.99 | 0.71 | 0.000 / 0.034 | 0.000 / 0.000 |
| R5 | K2:unsafe | 0.032 | 175 | source | 100 | 0.98 | 0.70 | 0.000 / 0.038 | 0.000 / 0.000 |
| R5 | K2:unsafe | 0.032 | 175 | pool | 200 | 1.20 | 0.89 | 0.034 / 0.098 | 0.009 / 0.034 |
| R5 | K2:unsafe | 0.032 | 175 | source | 200 | 1.11 | 0.84 | 0.042 / **0.110** | 0.010 / 0.042 |
| R5 | K3:refusal | 0.977 | 175 | pool | 100 | 0.11 | 0.11 | 0.002 / 0.013 | 0.002 / 0.013 |
| R5 | K3:refusal | 0.977 | 175 | source | 100 | 0.11 | 0.11 | 0.004 / 0.016 | 0.004 / 0.016 |
| R5 | K3:refusal | 0.977 | 175 | pool | 200 | 0.26 | 0.26 | 0.009 / 0.029 | 0.009 / 0.029 |
| R5 | K3:refusal | 0.977 | 175 | source | 200 | 0.25 | 0.25 | 0.005 / 0.035 | 0.005 / 0.035 |
| R5 | K3:unsafe | 0.016 | 175 | pool | 100 | 0.86 | 0.57 | 0.000 / 0.000 | 0.000 / 0.000 |
| R5 | K3:unsafe | 0.016 | 175 | source | 100 | 0.84 | 0.56 | 0.000 / 0.000 | 0.000 / 0.000 |
| R5 | K3:unsafe | 0.016 | 175 | pool | 200 | 1.08 | 0.74 | 0.000 / 0.033 | 0.000 / 0.033 |
| R5 | K3:unsafe | 0.016 | 175 | source | 200 | 1.03 | 0.72 | 0.000 / 0.034 | 0.000 / 0.034 |
| R5b | K4:refusal | 0.459 | 175 | pool | 100 | 2.09 | 2.07 | 0.026 / 0.063 | 0.025 / 0.062 |
| R5b | K4:refusal | 0.459 | 175 | source | 100 | 1.51 | 1.50 | 0.030 / 0.080 | 0.029 / 0.078 |
| R5b | K4:refusal | 0.459 | 175 | pool | 200 | 2.56 | 2.56 | 0.036 / 0.083 | 0.036 / 0.083 |
| R5b | K4:refusal | 0.459 | 175 | source | 200 | 1.33 | 1.33 | 0.039 / 0.095 | 0.039 / 0.094 |
| R5b | K4:unsafe | 0.019 | 175 | pool | 100 | 0.88 | 0.59 | 0.000 / 0.000 | 0.000 / 0.000 |
| R5b | K4:unsafe | 0.019 | 175 | source | 100 | 0.87 | 0.59 | 0.000 / 0.000 | 0.000 / 0.000 |
| R5b | K4:unsafe | 0.019 | 175 | pool | 200 | 1.12 | 0.78 | 0.016 / 0.089 | 0.000 / 0.016 |
| R5b | K4:unsafe | 0.019 | 175 | source | 200 | 1.09 | 0.77 | 0.017 / 0.095 | 0.000 / 0.017 |

## Cells over, by run group and claim (both levels)

| group | claim | pairs | b1 over | b1s over |
|---|---|---|---|---|
| human labels, mid-rate | pool | 64 | 3 | 0 |
| human labels, rare or near one | pool | 16 | 0 | 0 |
| language-model pools, mid-rate | pool | 28 | 0 | 0 |
| language-model pools, mid-rate | source | 28 | 1 | 0 |
| language-model pools, rare or near one | pool | 16 | 0 | 0 |
| language-model pools, rare or near one | source | 16 | 1 | 0 |

## Human labels (XSTest), mid-rate labels, with replacement, delta 0.05

| stratifier | n_s | labels | median ESS b1 | median ESS b1s | range b1s |
|---|---|---|---|---|---|
| B | 50 | 8 | 1.00 | 0.83 | 0.76-0.99 |
| B | 100 | 8 | 1.18 | 1.02 | 0.93-1.21 |
| A | 50 | 8 | 0.99 | 0.84 | 0.73-1.07 |
| A | 100 | 8 | 1.19 | 1.01 | 0.93-1.25 |
