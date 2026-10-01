# Spike 017, stage D: the harm certificate on the real sheet

Population 4800 responses, all scored; 225 human labels, 3 positive. delta = 0.05, bounds one-sided.

## 0. The GPU pass reproduces 015 on the sheet items

| wording | largest abs difference in p | label flips at 0.5 (of 225) |
|---|---|---|
| 0 | 0.1846 | 3 |
| 2 | 0.3584 | 5 |
| 4 | 0.1548 | 0 |

## 1. The judge on the population and on the sheet

| wording | feature | population mean (exact) | sheet, unweighted | sheet, design-weighted (se) | rho^2 with the human label (sheet) |
|---|---|---|---|---|---|
| 0 | f01 | 0.224 | 0.218 | 0.227 (0.029) | 0.016 |
| 0 | p | 0.236 | 0.233 | 0.241 (0.025) | 0.019 |
| 0 | logit | 0.322 | 0.326 | 0.318 (0.016) | 0.022 |
| 2 | f01 | 0.311 | 0.462 | 0.329 (0.021) | 0.002 |
| 2 | p | 0.325 | 0.465 | 0.340 (0.018) | 0.009 |
| 2 | logit | 0.369 | 0.456 | 0.384 (0.014) | 0.025 |
| 4 | f01 | 0.055 | 0.093 | 0.086 (0.017) | 0.001 |
| 4 | p | 0.065 | 0.095 | 0.082 (0.013) | 0.000 |
| 4 | logit | 0.124 | 0.165 | 0.139 (0.011) | 0.021 |

Compiled judge (wording 0) flag rate on the whole population, by the guard's refusal field: answered 0.126 (1593 responses), refused 0.272 (3207); by the 0.6B flag: flagged 0.145, cleared 0.249.

## 2. Estimates of the population's human harm rate

| wording | feature | route | estimate | 95% upper bound | lam |
|---|---|---|---|---|---|
| 0 | f01 | sheet as i.i.d. | 0.0133 | 0.0341 |  |
| 0 | f01 | weighted labels, normal | 0.0134 | 0.0292 |  |
| 0 | f01 | weighted labels, b1w | 0.0134 | 0.0393 |  |
| 0 | f01 | PPI as i.i.d. | 0.0191 | 0.0646 |  |
| 0 | f01 | weighted PPI | 0.0098 | 0.0627 |  |
| 0 | f01 | weighted PPI++ | 0.0136 | 0.0291 | -0.07 |
| 0 | f01 | judge alone | 0.2235 | | |
| 0 | p | PPI as i.i.d. | 0.0166 | 0.0576 |  |
| 0 | p | weighted PPI | 0.0088 | 0.0557 |  |
| 0 | p | weighted PPI++ | 0.0138 | 0.0292 | -0.09 |
| 0 | p | judge alone | 0.2363 | | |
| 0 | logit | PPI as i.i.d. | 0.0092 | 0.0358 |  |
| 0 | logit | weighted PPI | 0.0172 | 0.0505 |  |
| 0 | logit | weighted PPI++ | 0.0130 | 0.0287 | -0.09 |
| 0 | logit | judge alone | 0.3217 | | |
| 2 | f01 | PPI as i.i.d. | -0.1378 | -0.0822 |  |
| 2 | f01 | weighted PPI | -0.0046 | 0.0330 |  |
| 2 | f01 | weighted PPI++ | 0.0134 | 0.0292 | 0.00 |
| 2 | f01 | judge alone | 0.3110 | | |
| 2 | p | PPI as i.i.d. | -0.1264 | -0.0776 |  |
| 2 | p | weighted PPI | -0.0019 | 0.0289 |  |
| 2 | p | weighted PPI++ | 0.0119 | 0.0275 | 0.09 |
| 2 | p | judge alone | 0.3248 | | |
| 2 | logit | PPI as i.i.d. | -0.0740 | -0.0449 |  |
| 2 | logit | weighted PPI | -0.0016 | 0.0255 |  |
| 2 | logit | weighted PPI++ | 0.0124 | 0.0282 | 0.06 |
| 2 | logit | judge alone | 0.3690 | | |
| 4 | f01 | PPI as i.i.d. | -0.0246 | 0.0102 |  |
| 4 | f01 | weighted PPI | -0.0170 | 0.0146 |  |
| 4 | f01 | weighted PPI++ | 0.0134 | 0.0292 | 0.00 |
| 4 | f01 | judge alone | 0.0554 | | |
| 4 | p | PPI as i.i.d. | -0.0171 | 0.0129 |  |
| 4 | p | weighted PPI | -0.0036 | 0.0218 |  |
| 4 | p | weighted PPI++ | 0.0125 | 0.0283 | 0.05 |
| 4 | p | judge alone | 0.0646 | | |
| 4 | logit | PPI as i.i.d. | -0.0276 | -0.0040 |  |
| 4 | logit | weighted PPI | -0.0021 | 0.0198 |  |
| 4 | logit | weighted PPI++ | 0.0114 | 0.0270 | 0.13 |
| 4 | logit | judge alone | 0.1239 | | |

## 3. Validity under the real sampling rule (planted labels)

Labels planted on the real population (harm only among answered responses, the wording's own 0/1 judge catching 2/3 of it), the sheet re-drawn 4,000 times (20 plantings x 200 draws) by the real rule. Miss = bound below the planted population rate (target <= 0.05); mean bound in brackets.

| wording | planted rate | feature | sheet as i.i.d. | weighted labels, normal | weighted labels, b1w | PPI as i.i.d. | weighted PPI | weighted PPI++ |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.013 | f01 | 0.001 (0.050) | 0.198 (0.024) | 0.001 (0.037) | 0.000 (0.095) | 0.029 (0.069) | 0.197 (0.024) |
| 0 | 0.013 | logit | 0.003 (0.048) | 0.216 (0.024) | 0.003 (0.037) | 0.005 (0.051) | 0.043 (0.046) | 0.216 (0.024) |
| 0 | 0.05 | f01 | 0.000 (0.134) | 0.118 (0.072) | 0.024 (0.080) | 0.000 (0.165) | 0.026 (0.108) | 0.112 (0.072) |
| 0 | 0.05 | logit | 0.000 (0.134) | 0.117 (0.072) | 0.025 (0.080) | 0.000 (0.130) | 0.055 (0.088) | 0.118 (0.072) |
| 0 | 0.2 | f01 | 0.000 (0.243) | 0.099 (0.140) | 0.052 (0.144) | 0.000 (0.272) | 0.039 (0.170) | 0.101 (0.139) |
| 0 | 0.2 | logit | 0.000 (0.243) | 0.090 (0.140) | 0.043 (0.144) | 0.000 (0.239) | 0.052 (0.154) | 0.090 (0.140) |
| 2 | 0.013 | f01 | 0.004 (0.046) | 0.188 (0.024) | 0.004 (0.037) | 0.984 (-0.043) | 0.059 (0.060) | 0.188 (0.024) |
| 2 | 0.013 | logit | 0.004 (0.047) | 0.192 (0.024) | 0.004 (0.037) | 0.989 (-0.026) | 0.070 (0.043) | 0.194 (0.024) |
| 2 | 0.05 | f01 | 0.000 (0.125) | 0.127 (0.072) | 0.028 (0.079) | 0.797 (0.025) | 0.065 (0.098) | 0.127 (0.071) |
| 2 | 0.05 | logit | 0.000 (0.124) | 0.115 (0.073) | 0.022 (0.080) | 0.589 (0.045) | 0.070 (0.085) | 0.115 (0.072) |
| 2 | 0.2 | f01 | 0.000 (0.404) | 0.077 (0.240) | 0.059 (0.241) | 0.003 (0.286) | 0.069 (0.254) | 0.085 (0.239) |
| 2 | 0.2 | logit | 0.000 (0.405) | 0.069 (0.241) | 0.049 (0.242) | 0.000 (0.318) | 0.062 (0.247) | 0.074 (0.240) |
| 4 | 0.013 | f01 | 0.005 (0.045) | 0.183 (0.024) | 0.005 (0.037) | 0.093 (0.031) | 0.026 (0.038) | 0.181 (0.024) |
| 4 | 0.013 | logit | 0.008 (0.044) | 0.174 (0.024) | 0.008 (0.037) | 0.698 (0.007) | 0.058 (0.032) | 0.175 (0.024) |
| 4 | 0.05 | f01 | 0.000 (0.117) | 0.117 (0.071) | 0.018 (0.079) | 0.003 (0.087) | 0.036 (0.073) | 0.105 (0.068) |
| 4 | 0.05 | logit | 0.000 (0.118) | 0.117 (0.071) | 0.019 (0.079) | 0.093 (0.072) | 0.073 (0.073) | 0.118 (0.069) |
| 4 | 0.2 | f01 | 0.000 (0.214) | 0.092 (0.131) | 0.041 (0.136) | 0.000 (0.187) | 0.060 (0.134) | 0.090 (0.130) |
| 4 | 0.2 | logit | 0.000 (0.215) | 0.090 (0.131) | 0.039 (0.136) | 0.001 (0.170) | 0.084 (0.132) | 0.095 (0.130) |
