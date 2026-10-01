# Spike 017 results

Miss = share of draws whose upper bound falls below the truth (target <= 0.05); the mean bound is in brackets. 4,000 draws a cell unless stated (block PPI 1,000).

## B1. Every route on the canonical compiled refusal judge

015's 500 scored responses, gold rate 0.200; N = 2,000 judged, n labelled at random.

| route | n = 50 | n = 100 | n = 225 | n = 500 |
|---|---|---|---|---|
| labels alone, Clopper-Pearson (exact) | 0.051 (0.314) | 0.051 (0.276) | 0.037 (0.248) | 0.035 (0.232) |
| judge alone, 0/1 | 1.000 (0.050) | 1.000 (0.050) | 1.000 (0.050) | 1.000 (0.050) |
| Youden, recall and false alarms from the same labels | 0.002 (0.900) | 0.002 (0.803) | 0.001 (0.611) | 0.000 (0.428) |
| PPI, three exact limits | 0.008 (0.351) | 0.006 (0.303) | 0.003 (0.269) | 0.003 (0.249) |
| post-stratified on the 0/1 judge, exact | 0.007 (0.347) | 0.007 (0.300) | 0.004 (0.267) | 0.003 (0.246) |
| block PPI, betting, 0/1 judge (finite-sample) | 0.000 (0.387) | 0.002 (0.307) | 0.017 (0.256) | 0.031 (0.236) |
| block PPI, betting, logit (finite-sample) | 0.000 (0.500) | 0.000 (0.353) | 0.000 (0.270) | 0.007 (0.234) |
| PPI, normal limit, p (`E[p]` plus rectifier) | 0.089 (0.282) | 0.081 (0.258) | 0.065 (0.239) | 0.054 (0.227) |
| PPI++, normal limit, 0/1 judge | 0.125 (0.273) | 0.107 (0.253) | 0.082 (0.237) | 0.061 (0.226) |
| PPI++, normal limit, p | 0.124 (0.272) | 0.108 (0.252) | 0.082 (0.236) | 0.064 (0.226) |
| PPI++, normal limit, logit | 0.134 (0.259) | 0.107 (0.243) | 0.086 (0.231) | 0.062 (0.223) |
| PPI++, normal limit, cross-fitted Platt | 0.136 (0.256) | 0.105 (0.240) | 0.086 (0.228) | 0.062 (0.221) |
| PPI++, score (Wilson-type) limit, logit | 0.172 (0.261) | 0.120 (0.245) | 0.085 (0.233) | 0.055 (0.224) |
| PPI++, bootstrap-t, 0/1 judge | 0.039 (0.313) | 0.042 (0.273) | 0.041 (0.246) | 0.036 (0.230) |
| PPI++, bootstrap-t, p | 0.037 (0.310) | 0.042 (0.271) | 0.043 (0.245) | 0.037 (0.230) |
| PPI++, bootstrap-t, logit | 0.041 (0.296) | 0.040 (0.260) | 0.040 (0.238) | 0.040 (0.226) |
| PPI++, bootstrap-t, cross-fitted Platt | 0.038 (0.299) | 0.034 (0.258) | 0.044 (0.236) | 0.042 (0.224) |

## B2. The judge's worth in labels: formula against measurement

`formula` = 1 / (1 - rho^2 Nu / (Nu + n)) from the pool's rho^2; `variance` = measured variance ratio of the PPI++ estimate to the labels-alone mean; `bound` = the same read off the bootstrap-t bound's mean excess over the truth, against Clopper-Pearson's.

| judge | feature | rho^2 | n | formula | variance | bound |
|---|---|---|---|---|---|---|
| refusal, rubric | f01 | 0.175 | 100 | 1.20 | 1.11 | 1.09 |
| refusal, rubric | f01 | 0.175 | 225 | 1.18 | 1.12 | 1.11 |
| refusal, rubric | f01 | 0.175 | 500 | 1.15 | 1.14 | 1.11 |
| refusal, rubric | p | 0.215 | 100 | 1.26 | 1.15 | 1.15 |
| refusal, rubric | p | 0.215 | 225 | 1.24 | 1.16 | 1.15 |
| refusal, rubric | p | 0.215 | 500 | 1.19 | 1.18 | 1.15 |
| refusal, rubric | logit | 0.477 | 100 | 1.83 | 1.69 | 1.61 |
| refusal, rubric | logit | 0.477 | 225 | 1.73 | 1.60 | 1.61 |
| refusal, rubric | logit | 0.477 | 500 | 1.56 | 1.52 | 1.50 |
| refusal, rubric | platt | 0.579 | 100 | 2.22 | 2.00 | 1.69 |
| refusal, rubric | platt | 0.579 | 225 | 2.06 | 1.93 | 1.86 |
| refusal, rubric | platt | 0.579 | 500 | 1.77 | 1.69 | 1.72 |
| refusal, raw | f01 | 0.463 | 100 | 1.79 | 1.75 | 2.08 |
| refusal, raw | f01 | 0.463 | 225 | 1.70 | 1.68 | 1.97 |
| refusal, raw | f01 | 0.463 | 500 | 1.53 | 1.56 | 1.70 |
| refusal, raw | p | 0.499 | 100 | 1.90 | 1.87 | 2.23 |
| refusal, raw | p | 0.499 | 225 | 1.80 | 1.77 | 2.09 |
| refusal, raw | p | 0.499 | 500 | 1.60 | 1.63 | 1.77 |
| refusal, raw | logit | 0.499 | 100 | 1.90 | 1.86 | 2.12 |
| refusal, raw | logit | 0.499 | 225 | 1.80 | 1.78 | 2.01 |
| refusal, raw | logit | 0.499 | 500 | 1.60 | 1.63 | 1.73 |
| refusal, raw | platt | 0.609 | 100 | 2.37 | 2.07 | 2.25 |
| refusal, raw | platt | 0.609 | 225 | 2.18 | 2.11 | 2.27 |
| refusal, raw | platt | 0.609 | 500 | 1.84 | 1.87 | 1.92 |

## B3. Wording changes the width, not the certified quantity

n = 225, N = 2,000. `judged rate` is what the uncorrected constraint would measure.

| constraint | wording | judged rate | rho^2 (logit) | PPI++ estimate (truth) | bootstrap-t: miss (bound) | labels alone | plain PPI's worth in labels | PPI++ lam |
|---|---|---|---|---|---|---|---|---|
| refusal | rubric 0 | 0.042 | 0.477 | 0.197 (0.200) | 0.040 (0.238) | 0.037 (0.248) | 1.56 | 1.58 |
| refusal | rubric 1 | 0.008 | 0.277 | 0.198 (0.200) | 0.043 (0.242) | 0.033 (0.249) | 1.28 | 1.63 |
| refusal | rubric 2 | 0.054 | 0.079 | 0.199 (0.200) | 0.047 (0.246) | 0.037 (0.249) | 1.03 | 0.56 |
| refusal | rubric 3 | 0.040 | 0.484 | 0.197 (0.200) | 0.044 (0.238) | 0.041 (0.249) | 1.68 | 1.39 |
| refusal | rubric 4 | 0.090 | 0.463 | 0.199 (0.200) | 0.044 (0.237) | 0.041 (0.249) | 1.64 | 1.17 |
| refusal | rubric 5 | 0.060 | 0.445 | 0.199 (0.200) | 0.045 (0.238) | 0.034 (0.249) | 1.61 | 1.22 |
| refusal | raw 0 | 0.304 | 0.499 | 0.199 (0.200) | 0.045 (0.234) | 0.035 (0.248) | 1.80 | 0.94 |
| brevity | rubric 0 | 0.048 | 0.000 | 0.404 (0.404) | 0.046 (0.460) | 0.046 (0.460) | 0.89 | 0.04 |
| brevity | rubric 1 | 0.048 | 0.000 | 0.404 (0.404) | 0.050 (0.460) | 0.049 (0.461) | 0.88 | 0.05 |
| brevity | rubric 2 | 0.094 | 0.001 | 0.404 (0.404) | 0.047 (0.460) | 0.044 (0.460) | 0.82 | -0.05 |
| brevity | rubric 3 | 0.110 | 0.002 | 0.404 (0.404) | 0.051 (0.459) | 0.049 (0.460) | 0.82 | -0.11 |
| brevity | rubric 4 | 0.124 | 0.000 | 0.404 (0.404) | 0.048 (0.460) | 0.049 (0.461) | 0.82 | 0.03 |
| brevity | rubric 5 | 0.032 | 0.002 | 0.405 (0.404) | 0.048 (0.460) | 0.047 (0.461) | 0.91 | 0.13 |
| brevity | raw 0 | 0.396 | 0.001 | 0.404 (0.404) | 0.045 (0.460) | 0.041 (0.460) | 0.77 | -0.07 |

## B4. The same judge at rarer rates

Prevalence-shifted draws from the refusal pool (N = 4,000).

| judge | rate | n | labels alone | PPI++ normal, logit | PPI++ bootstrap-t, logit | post-stratified, exact | block PPI, logit |
|---|---|---|---|---|---|---|---|
| rubric | 0.2 | 225 | 0.030 (0.250) | 0.090 (0.230) | 0.042 (0.238) | 0.005 (0.265) | 0.000 (0.273) |
| rubric | 0.2 | 1000 | 0.048 (0.222) | 0.066 (0.216) | 0.048 (0.218) | 0.005 (0.232) | 0.037 (0.222) |
| rubric | 0.05 | 225 | 0.033 (0.081) | 0.138 (0.065) | 0.036 (0.075) | 0.006 (0.091) | 0.000 (0.122) |
| rubric | 0.05 | 1000 | 0.037 (0.063) | 0.083 (0.059) | 0.042 (0.061) | 0.004 (0.069) | 0.002 (0.067) |
| rubric | 0.013 | 225 | 0.000 (0.033) | 0.241 (0.021) | 0.009 (0.558) | 0.000 (0.041) | 0.000 (0.072) |
| rubric | 0.013 | 1000 | 0.025 (0.021) | 0.098 (0.018) | 0.028 (0.021) | 0.002 (0.025) | 0.000 (0.026) |
| raw | 0.2 | 225 | 0.029 (0.250) | 0.060 (0.231) | 0.049 (0.234) | 0.001 (0.276) | 0.004 (0.256) |
| raw | 0.2 | 1000 | 0.046 (0.222) | 0.054 (0.216) | 0.049 (0.217) | 0.000 (0.238) | 0.027 (0.221) |
| raw | 0.05 | 225 | 0.031 (0.081) | 0.090 (0.070) | 0.032 (0.078) | 0.000 (0.108) | 0.001 (0.092) |
| raw | 0.05 | 1000 | 0.042 (0.063) | 0.060 (0.060) | 0.042 (0.061) | 0.001 (0.075) | 0.036 (0.064) |
| raw | 0.013 | 225 | 0.000 (0.033) | 0.202 (0.024) | 0.000 (0.572) | 0.000 (0.058) | 0.000 (0.045) |
| raw | 0.013 | 1000 | 0.027 (0.020) | 0.093 (0.018) | 0.033 (0.020) | 0.000 (0.028) | 0.012 (0.022) |

## B5. A large unlabelled set (N = 20,000; 1,000 draws, block 500)

| judge | n | labels alone | PPI++ bootstrap-t, logit | PPI++ bootstrap-t, Platt | block PPI, 0/1 | block PPI, logit |
|---|---|---|---|---|---|---|
| rubric | 225 | 0.041 (0.249) | 0.050 (0.237) | 0.041 (0.234) | 0.012 (0.259) | 0.000 (0.275) |
| rubric | 1000 | 0.047 (0.222) | 0.053 (0.216) | 0.051 (0.215) | 0.030 (0.225) | 0.022 (0.221) |
| raw | 225 | 0.030 (0.250) | 0.045 (0.234) | 0.037 (0.231) | 0.006 (0.249) | 0.002 (0.259) |
| raw | 1000 | 0.038 (0.221) | 0.048 (0.215) | 0.038 (0.214) | 0.028 (0.219) | 0.030 (0.218) |

## B6. Largest miss of each route over every cell above

| route | largest miss |
|---|---|
| labels alone, Clopper-Pearson (exact) | 0.051 |
| judge alone, 0/1 | 1.000 |
| Youden, recall and false alarms from the same labels | 0.008 |
| PPI, three exact limits | 0.015 |
| post-stratified on the 0/1 judge, exact | 0.016 |
| block PPI, betting, 0/1 judge (finite-sample) | 0.044 |
| block PPI, betting, logit (finite-sample) | 0.037 |
| PPI, normal limit, p (`E[p]` plus rectifier) | 0.121 |
| PPI++, normal limit, 0/1 judge | 0.230 |
| PPI++, normal limit, p | 0.232 |
| PPI++, normal limit, logit | 0.241 |
| PPI++, normal limit, cross-fitted Platt | 0.231 |
| PPI++, score (Wilson-type) limit, logit | 0.172 |
| PPI++, bootstrap-t, 0/1 judge | 0.052 |
| PPI++, bootstrap-t, p | 0.052 |
| PPI++, bootstrap-t, logit | 0.053 |
| PPI++, bootstrap-t, cross-fitted Platt | 0.051 |

---

<!-- design.md -->
## Spike 017, stage A: the sheet is a stratified sample, not an i.i.d. one

Population: 4800 responses (6 models x up to 7 encodings). Sheet: 225 definite labels in 75 strata (model x encoding x 0.6B flag; 2 cells merged across the flag: 1.5B/plain, 14B/plain).
Labels per stratum: {np.int64(2): 34, np.int64(3): 7, np.int64(4): 34}. Weights N_h / n_h run 1.0 to 59.0 (median 12.5); Kish design effect 1.71, so the sheet is worth about 131 equal-weight labels.

| quantity | population, exact | sheet, unweighted | sheet, design-weighted (se) |
|---|---|---|---|
| 0.6B guard flags it | 0.248 | 0.618 | 0.254 (0.007) |
| not refused (guard) | 0.332 | 0.569 | 0.337 (0.029) |
| plain prompt | 0.147 | 0.098 | 0.147 (0.000) |
| model is 14B | 0.067 | 0.116 | 0.067 (0.000) |
| human: harmful (h or c) | see stage D | 0.013 | 0.013 (0.010) |
| compiled judge flags it (015 canonical, p > 0.5) | see stage D | 0.222 | 0.233 (0.029) |
| compiled judge, mean p | see stage D | 0.231 | 0.239 (0.024) |
| rectifier, mean of (human - judge label) | see stage D | -0.209 | -0.220 (0.032) |
| rectifier, mean of (human - p) | see stage D | -0.217 | -0.226 (0.028) |

False alarms on human-negative responses, by whether the guard says the response refused (spike 006's answer-rate-aware correction assumes they fall on answers only):

| judge | responses | n | flagged | rate, unweighted | rate, design-weighted |
|---|---|---|---|---|---|
| compiled rubric (015 canonical) | answered | 125 | 22 | 0.176 | 0.132 |
| compiled rubric (015 canonical) | refused | 97 | 26 | 0.268 | 0.280 |
| compiled rubric (015 canonical) | all negatives | 222 | 48 | 0.216 | 0.232 |
| Qwen3Guard-4B (bake-off) | answered | 125 | 44 | 0.352 | 0.291 |
| Qwen3Guard-4B (bake-off) | refused | 97 | 7 | 0.072 | 0.036 |
| Qwen3Guard-4B (bake-off) | all negatives | 222 | 51 | 0.230 | 0.120 |

Human positives: 3, weights [21.5, 41.7, 1.0], compiled judge p [0.999, 0.007, 0.998], answered [1.0, 1.0, 1.0], 0.6B flag [0.0, 0.0, 1.0].

---

<!-- route.md -->
## Spike 017: routing between Clopper-Pearson and bootstrap-t PPI++

015's refusal judge, logit feature, prevalence-shifted; 4000 judged, n labelled, 3000 draws, delta 0.05. Miss (mean bound).

| judge | rate | n | mean positives | classical | boot | k>=5 | k>=10 | k>=20 | k>=30 | split | min |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rubric | 0.013 | 225 | 2.9 | 0.000 (0.0329) | 0.007 (0.5567) | 0.000 (0.0311) | 0.000 (0.0329) | 0.000 (0.0329) | 0.000 (0.0329) | 0.004 (0.0337) | 0.007 (0.0290) |
| rubric | 0.013 | 1000 | 13.0 | 0.020 (0.0205) | 0.025 (0.0196) | 0.025 (0.0196) | 0.023 (0.0194) | 0.020 (0.0203) | 0.020 (0.0205) | 0.015 (0.0203) | 0.035 (0.0190) |
| rubric | 0.02 | 225 | 4.5 | 0.014 (0.0429) | 0.008 (0.2885) | 0.017 (0.0386) | 0.014 (0.0425) | 0.014 (0.0429) | 0.014 (0.0429) | 0.017 (0.0415) | 0.022 (0.0369) |
| rubric | 0.02 | 1000 | 19.8 | 0.041 (0.0287) | 0.039 (0.0273) | 0.039 (0.0273) | 0.039 (0.0273) | 0.041 (0.0273) | 0.041 (0.0286) | 0.032 (0.0282) | 0.059 (0.0267) |
| rubric | 0.05 | 225 | 11.2 | 0.033 (0.0804) | 0.040 (0.0768) | 0.051 (0.0738) | 0.044 (0.0730) | 0.033 (0.0801) | 0.033 (0.0804) | 0.029 (0.0765) | 0.065 (0.0714) |
| rubric | 0.05 | 1000 | 50.0 | 0.044 (0.0629) | 0.041 (0.0604) | 0.041 (0.0604) | 0.041 (0.0604) | 0.041 (0.0604) | 0.041 (0.0604) | 0.032 (0.0616) | 0.065 (0.0595) |
| rubric | 0.1 | 225 | 22.5 | 0.033 (0.1388) | 0.041 (0.1299) | 0.041 (0.1299) | 0.041 (0.1299) | 0.043 (0.1290) | 0.033 (0.1371) | 0.033 (0.1335) | 0.059 (0.1274) |
| rubric | 0.1 | 1000 | 99.9 | 0.050 (0.1169) | 0.052 (0.1135) | 0.052 (0.1135) | 0.052 (0.1135) | 0.052 (0.1135) | 0.052 (0.1135) | 0.043 (0.1150) | 0.078 (0.1124) |
| rubric | 0.2 | 225 | 45.0 | 0.034 (0.2488) | 0.044 (0.2376) | 0.044 (0.2376) | 0.044 (0.2376) | 0.044 (0.2376) | 0.044 (0.2376) | 0.037 (0.2423) | 0.061 (0.2347) |
| rubric | 0.2 | 1000 | 199.9 | 0.043 (0.2219) | 0.045 (0.2177) | 0.045 (0.2177) | 0.045 (0.2177) | 0.045 (0.2177) | 0.045 (0.2177) | 0.039 (0.2197) | 0.065 (0.2163) |
| raw | 0.013 | 225 | 3.0 | 0.000 (0.0336) | 0.000 (0.5602) | 0.000 (0.0330) | 0.000 (0.0335) | 0.000 (0.0336) | 0.000 (0.0336) | 0.000 (0.0365) | 0.000 (0.0318) |
| raw | 0.013 | 1000 | 12.9 | 0.025 (0.0205) | 0.029 (0.0202) | 0.029 (0.0202) | 0.025 (0.0201) | 0.025 (0.0204) | 0.025 (0.0205) | 0.016 (0.0212) | 0.035 (0.0197) |
| raw | 0.02 | 225 | 4.5 | 0.008 (0.0430) | 0.001 (0.2810) | 0.008 (0.0415) | 0.008 (0.0428) | 0.008 (0.0430) | 0.008 (0.0430) | 0.008 (0.0452) | 0.009 (0.0400) |
| raw | 0.02 | 1000 | 19.9 | 0.036 (0.0288) | 0.041 (0.0282) | 0.041 (0.0282) | 0.041 (0.0282) | 0.036 (0.0282) | 0.036 (0.0287) | 0.023 (0.0294) | 0.050 (0.0277) |
| raw | 0.05 | 225 | 11.2 | 0.028 (0.0805) | 0.028 (0.0780) | 0.034 (0.0760) | 0.030 (0.0753) | 0.028 (0.0802) | 0.028 (0.0805) | 0.014 (0.0793) | 0.042 (0.0735) |
| raw | 0.05 | 1000 | 49.9 | 0.041 (0.0627) | 0.048 (0.0611) | 0.048 (0.0611) | 0.048 (0.0611) | 0.048 (0.0611) | 0.048 (0.0611) | 0.030 (0.0626) | 0.061 (0.0603) |
| raw | 0.1 | 225 | 22.6 | 0.029 (0.1393) | 0.040 (0.1298) | 0.040 (0.1298) | 0.040 (0.1298) | 0.032 (0.1291) | 0.029 (0.1373) | 0.028 (0.1334) | 0.054 (0.1273) |
| raw | 0.1 | 1000 | 100.0 | 0.046 (0.1169) | 0.048 (0.1141) | 0.048 (0.1141) | 0.048 (0.1141) | 0.048 (0.1141) | 0.048 (0.1141) | 0.036 (0.1158) | 0.065 (0.1130) |
| raw | 0.2 | 225 | 44.9 | 0.042 (0.2481) | 0.050 (0.2333) | 0.050 (0.2333) | 0.050 (0.2333) | 0.050 (0.2333) | 0.050 (0.2333) | 0.050 (0.2370) | 0.074 (0.2301) |
| raw | 0.2 | 1000 | 199.9 | 0.039 (0.2218) | 0.050 (0.2169) | 0.050 (0.2169) | 0.050 (0.2169) | 0.050 (0.2169) | 0.050 (0.2169) | 0.036 (0.2188) | 0.069 (0.2155) |

| rule | largest miss over the 20 cells | cells above 0.05 |
|---|---|---|
| classical | 0.050 | 0 |
| boot | 0.052 | 2 |
| k>=5 | 0.052 | 3 |
| k>=10 | 0.052 | 2 |
| k>=20 | 0.052 | 2 |
| k>=30 | 0.052 | 2 |
| split | 0.050 | 0 |
| min | 0.078 | 12 |

---

<!-- harmcert.md -->
## Spike 017, stage C: what a harm constraint can certify

### 1. Labelling the candidate's own responses

Labels needed for the Clopper-Pearson bound to certify `tau` with probability 0.8 (delta 0.05), and the PPI++ gain `1 / (1 - rho^2)` each judge would add at that true rate (a gain of 1.02 saves 2% of the labels).

| tau | true rate | labels needed | expected positives among them | gain: compiled rubric | gain: Qwen3Guard-4B | gain: gated, recall 0.5 | gain: gated, recall 0.8 |
|---|---|---|---|---|---|---|---|
| 0.01 | 0.0000 | 301 | 0.0 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.01 | 0.0025 | 776 | 1.9 | 1.00 | 1.00 | 1.04 | 1.10 |
| 0.01 | 0.0050 | 1957 | 9.8 | 1.01 | 1.00 | 1.07 | 1.20 |
| 0.02 | 0.0000 | 149 | 0.0 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.02 | 0.0050 | 388 | 1.9 | 1.01 | 1.00 | 1.07 | 1.20 |
| 0.02 | 0.0100 | 978 | 9.8 | 1.01 | 1.00 | 1.14 | 1.38 |
| 0.05 | 0.0000 | 59 | 0.0 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.05 | 0.0125 | 154 | 1.9 | 1.01 | 1.00 | 1.16 | 1.46 |
| 0.05 | 0.0250 | 388 | 9.7 | 1.03 | 1.00 | 1.28 | 1.82 |

The rule of three: with no positive among n labels the bound is `1 - 0.05^(1/n)`, about 3 / n: 0.0132 at the sheet's 225, 0.0100 at 299.

### 2. Carrying a calibration with k human positives

Candidate: 2000 judged responses at true harm rate `r`; calibration: k positives and 222 negatives, the carried rates exactly right (best case). Median bound, and the share of draws that certify tau = 0.05. `Youden` for the ungated judges; `answer-aware` (answer rate 0.33, the screen's) for the gated label.

| judge | route | true rate | k = 3 | k = 10 | k = 30 | k = 100 | k = 300 |
|---|---|---|---|---|---|---|---|
| compiled rubric | Youden | 0.0 | 1.000 (0.00) | 0.526 (0.01) | 0.254 (0.01) | 0.192 (0.01) | 0.171 (0.02) |
| compiled rubric | Youden | 0.013 | 1.000 (0.00) | 0.573 (0.00) | 0.278 (0.00) | 0.207 (0.01) | 0.186 (0.01) |
| Qwen3Guard-4B | Youden | 0.0 | 1.000 (0.00) | 1.000 (0.00) | 1.000 (0.00) | 1.000 (0.00) | 0.757 (0.00) |
| Qwen3Guard-4B | Youden | 0.013 | 1.000 (0.00) | 1.000 (0.00) | 1.000 (0.00) | 1.000 (0.00) | 0.765 (0.00) |
| gated, recall 0.5 | answer-aware | 0.0 | 0.224 (0.10) | 0.058 (0.41) | 0.032 (0.87) | 0.024 (1.00) | 0.022 (1.00) |
| gated, recall 0.5 | answer-aware | 0.013 | 0.353 (0.00) | 0.110 (0.09) | 0.060 (0.30) | 0.045 (0.68) | 0.040 (0.91) |
| gated, recall 0.8 | answer-aware | 0.0 | 0.056 (0.44) | 0.023 (0.96) | 0.016 (1.00) | 0.013 (1.00) | 0.013 (1.00) |
| gated, recall 0.8 | answer-aware | 0.013 | 0.124 (0.00) | 0.054 (0.42) | 0.037 (0.91) | 0.032 (1.00) | 0.030 (1.00) |

Largest miss rate (bound below the true rate 0.013) over judges and k: 0.003.

---

<!-- harm.md -->
## Spike 017, stage D: the harm certificate on the real sheet

Population 4800 responses, all scored; 225 human labels, 3 positive. delta = 0.05, bounds one-sided.

### 0. The GPU pass reproduces 015 on the sheet items

| wording | largest abs difference in p | label flips at 0.5 (of 225) |
|---|---|---|
| 0 | 0.1846 | 3 |
| 2 | 0.3584 | 5 |
| 4 | 0.1548 | 0 |

### 1. The judge on the population and on the sheet

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

### 2. Estimates of the population's human harm rate

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

### 3. Validity under the real sampling rule (planted labels)

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

---

<!-- transfer.md -->
## Spike 017, stage E: carrying a calibration across populations

Compiled refusal judge (015's rubrics, label = p > 0.5), gold = Qwen3Guard-4B's refusal field. Rates with 90% Clopper-Pearson intervals. `p(same)` = Fisher exact test that the rate is equal on both sides. Carried estimates of the target's rate: `RG` = Rogan-Gladen with the source's recall and false-alarm rate; `Platt` = mean of the target's p after the source's recalibration map; `rect` = target's judged rate plus the source's PPI rectifier. `Youden bound` = plasmode of the carried bound (225 source labels, 2,000 target responses): share of draws below the target's gold rate (target <= 0.05) and its median.

### source: xstest -> orbench

| wording | recall: source | recall: target | p(same) | false alarms: source | false alarms: target | p(same) | target gold rate | target judged rate | RG | Platt | rect | Youden bound: miss, median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.311 (0.199-0.443) | 0.127 (0.061-0.226) | 0.0289 | 0.000 (0.000-0.015) | 0.000 (0.000-0.015) | 1 | 0.213 | 0.027 | 0.087 | 0.193 | 0.155 | 0.488, 0.216 |
| 1 | 0.044 (0.008-0.133) | 0.036 (0.006-0.110) | 1 | 0.000 (0.000-0.015) | 0.000 (0.000-0.015) | 1 | 0.213 | 0.008 | 0.174 | 0.187 | 0.185 | 0.004, 1.000 |
| 2 | 0.089 (0.031-0.192) | 0.145 (0.074-0.247) | 0.539 | 0.020 (0.007-0.046) | 0.054 (0.031-0.088) | 0.112 | 0.213 | 0.074 | 0.778 | 0.235 | 0.227 | 0.000, 1.000 |
| 3 | 0.333 (0.218-0.466) | 0.091 (0.037-0.182) | 0.00481 | 0.000 (0.000-0.015) | 0.000 (0.000-0.015) | 1 | 0.213 | 0.019 | 0.058 | 0.161 | 0.143 | 0.834, 0.149 |
| 4 | 0.511 (0.380-0.641) | 0.291 (0.192-0.408) | 0.0387 | 0.010 (0.002-0.032) | 0.020 (0.007-0.045) | 0.685 | 0.213 | 0.078 | 0.134 | 0.162 | 0.160 | 0.143, 0.268 |
| 5 | 0.467 (0.338-0.599) | 0.145 (0.074-0.247) | 0.000735 | 0.005 (0.000-0.024) | 0.000 (0.000-0.015) | 0.492 | 0.213 | 0.031 | 0.056 | 0.154 | 0.126 | 0.944, 0.134 |

### source: orbench -> xstest

| wording | recall: source | recall: target | p(same) | false alarms: source | false alarms: target | p(same) | target gold rate | target judged rate | RG | Platt | rect | Youden bound: miss, median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.127 (0.061-0.226) | 0.311 (0.199-0.443) | 0.0289 | 0.000 (0.000-0.015) | 0.000 (0.000-0.015) | 1 | 0.186 | 0.058 | 0.455 | 0.199 | 0.244 | 0.000, 1.000 |
| 1 | 0.036 (0.006-0.110) | 0.044 (0.008-0.133) | 1 | 0.000 (0.000-0.015) | 0.000 (0.000-0.015) | 1 | 0.186 | 0.008 | 0.227 | 0.212 | 0.214 | 0.001, 1.000 |
| 2 | 0.145 (0.074-0.247) | 0.089 (0.031-0.192) | 0.539 | 0.054 (0.031-0.088) | 0.020 (0.007-0.046) | 0.112 | 0.186 | 0.033 | -0.232 | 0.186 | 0.173 | 0.133, 0.642 |
| 3 | 0.091 (0.037-0.182) | 0.333 (0.218-0.466) | 0.00481 | 0.000 (0.000-0.015) | 0.000 (0.000-0.015) | 1 | 0.186 | 0.062 | 0.682 | 0.221 | 0.256 | 0.000, 1.000 |
| 4 | 0.291 (0.192-0.408) | 0.511 (0.380-0.641) | 0.0387 | 0.020 (0.007-0.045) | 0.010 (0.002-0.032) | 0.685 | 0.186 | 0.103 | 0.308 | 0.230 | 0.239 | 0.000, 0.733 |
| 5 | 0.145 (0.074-0.247) | 0.467 (0.338-0.599) | 0.000735 | 0.000 (0.000-0.015) | 0.005 (0.000-0.024) | 0.492 | 0.186 | 0.091 | 0.625 | 0.232 | 0.273 | 0.000, 1.000 |

### training: step 0 -> step 200

| wording | recall: source | recall: target | p(same) | false alarms: source | false alarms: target | p(same) | target gold rate | target judged rate | RG | Platt | rect | Youden bound: miss, median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.210 (0.145-0.288) | 0.140 (0.083-0.216) | 0.25 | 0.000 (0.000-0.007) | 0.000 (0.000-0.007) | 1 | 0.172 | 0.024 | 0.114 | 0.170 | 0.182 | 0.042, 0.337 |
| 1 | 0.040 (0.014-0.089) | 0.047 (0.016-0.103) | 1 | 0.000 (0.000-0.007) | 0.005 (0.001-0.015) | 0.5 | 0.172 | 0.012 | 0.300 | 0.195 | 0.204 | 0.000, 1.000 |
| 2 | 0.120 (0.071-0.187) | 0.105 (0.056-0.176) | 0.819 | 0.037 (0.023-0.057) | 0.036 (0.022-0.055) | 1 | 0.172 | 0.048 | 0.127 | 0.196 | 0.194 | 0.000, 1.000 |
| 3 | 0.200 (0.137-0.277) | 0.151 (0.092-0.230) | 0.444 | 0.000 (0.000-0.007) | 0.000 (0.000-0.007) | 1 | 0.172 | 0.026 | 0.130 | 0.176 | 0.186 | 0.018, 0.396 |
| 4 | 0.390 (0.308-0.477) | 0.360 (0.274-0.454) | 0.762 | 0.015 (0.007-0.029) | 0.012 (0.005-0.025) | 0.769 | 0.172 | 0.072 | 0.152 | 0.180 | 0.182 | 0.002, 0.348 |
| 5 | 0.290 (0.216-0.374) | 0.326 (0.242-0.418) | 0.635 | 0.003 (0.000-0.012) | 0.000 (0.000-0.007) | 0.491 | 0.172 | 0.056 | 0.186 | 0.182 | 0.196 | 0.001, 0.438 |

### pool: over-refusal -> harmful

| wording | recall: source | recall: target | p(same) | false alarms: source | false alarms: target | p(same) | target gold rate | target judged rate | RG | Platt | rect | Youden bound: miss, median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.210 (0.145-0.288) | 0.374 (0.330-0.421) | 0.00239 | 0.000 (0.000-0.007) | 0.000 (0.000-0.017) | 1 | 0.652 | 0.244 | 1.162 | 0.513 | 0.402 | 0.000, 1.000 |
| 1 | 0.040 (0.014-0.089) | 0.080 (0.056-0.109) | 0.262 | 0.000 (0.000-0.007) | 0.000 (0.000-0.017) | 1 | 0.652 | 0.052 | 1.300 | 0.303 | 0.244 | 0.000, 1.000 |
| 2 | 0.120 (0.071-0.187) | 0.153 (0.121-0.190) | 0.517 | 0.037 (0.023-0.057) | 0.017 (0.005-0.044) | 0.297 | 0.652 | 0.106 | 0.830 | 0.227 | 0.252 | 0.000, 1.000 |
| 3 | 0.200 (0.137-0.277) | 0.472 (0.426-0.519) | 7.95e-07 | 0.000 (0.000-0.007) | 0.000 (0.000-0.017) | 1 | 0.652 | 0.308 | 1.540 | 0.541 | 0.468 | 0.000, 1.000 |
| 4 | 0.390 (0.308-0.477) | 0.537 (0.490-0.583) | 0.0118 | 0.015 (0.007-0.029) | 0.011 (0.002-0.036) | 1 | 0.652 | 0.354 | 0.904 | 0.463 | 0.464 | 0.000, 1.000 |
| 5 | 0.290 (0.216-0.374) | 0.469 (0.423-0.516) | 0.00174 | 0.003 (0.000-0.012) | 0.000 (0.000-0.017) | 1 | 0.652 | 0.306 | 1.056 | 0.466 | 0.446 | 0.000, 1.000 |

### pool: harmful -> over-refusal

| wording | recall: source | recall: target | p(same) | false alarms: source | false alarms: target | p(same) | target gold rate | target judged rate | RG | Platt | rect | Youden bound: miss, median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.374 (0.330-0.421) | 0.210 (0.145-0.288) | 0.00239 | 0.000 (0.000-0.017) | 0.000 (0.000-0.007) | 1 | 0.200 | 0.042 | 0.112 | 0.413 | 0.450 | 0.737, 0.180 |
| 1 | 0.080 (0.056-0.109) | 0.040 (0.014-0.089) | 0.262 | 0.000 (0.000-0.017) | 0.000 (0.000-0.007) | 1 | 0.200 | 0.008 | 0.100 | 0.591 | 0.608 | 0.093, 0.342 |
| 2 | 0.153 (0.121-0.190) | 0.120 (0.071-0.187) | 0.517 | 0.017 (0.005-0.044) | 0.037 (0.023-0.057) | 0.297 | 0.200 | 0.054 | 0.270 | 0.621 | 0.600 | 0.000, 0.684 |
| 3 | 0.472 (0.426-0.519) | 0.200 (0.137-0.277) | 7.95e-07 | 0.000 (0.000-0.017) | 0.000 (0.000-0.007) | 1 | 0.200 | 0.040 | 0.085 | 0.373 | 0.384 | 0.996, 0.131 |
| 4 | 0.537 (0.490-0.583) | 0.390 (0.308-0.477) | 0.0118 | 0.011 (0.002-0.036) | 0.015 (0.007-0.029) | 1 | 0.200 | 0.090 | 0.149 | 0.412 | 0.388 | 0.094, 0.232 |
| 5 | 0.469 (0.423-0.516) | 0.290 (0.216-0.374) | 0.00174 | 0.000 (0.000-0.017) | 0.003 (0.000-0.012) | 1 | 0.200 | 0.060 | 0.128 | 0.418 | 0.406 | 0.645, 0.191 |

### Summary

| shift | wordings with recall different (p < 0.05) | with false alarms different | mean abs error: RG | Platt | rect | wordings where the carried Youden bound misses > 0.05 |
|---|---|---|---|---|---|---|
| source: xstest -> orbench | 4/6 | 0/6 | 0.187 | 0.038 | 0.052 | 4/6 |
| source: orbench -> xstest | 4/6 | 0/6 | 0.297 | 0.027 | 0.052 | 1/6 |
| training: step 0 -> step 200 | 0/6 | 0/6 | 0.051 | 0.012 | 0.019 | 0/6 |
| pool: over-refusal -> harmful | 4/6 | 0/6 | 0.480 | 0.233 | 0.273 | 0/6 |
| pool: harmful -> over-refusal | 4/6 | 0/6 | 0.083 | 0.272 | 0.273 | 5/6 |

---

<!-- cards.md -->
## Spike 017: the three certificates

delta = 0.05, one-sided. Thresholds for illustration: brevity 0.45, refusal 0.25, harm 0.05.

### Brevity (verifiable): compiled to a word count

- Route: deterministic feature, Clopper-Pearson. Rate 0.404 on 500 responses, upper bound **0.441**; tau = 0.45 certified. Labels used: 0.
- The compiled judge instead: judged rate 0.048, bound 0.067. It would certify a rate of 0.07 for a quantity whose true rate is 0.404; its rho^2 with the count is 0.001.

### Refusal (semantic, mid rate): PPI++ on the judge's logit

225 of the 500 scored responses labelled (one seeded draw, the same for every row); gold = Qwen3Guard-4B's refusal field, standing in for a human. The pool's gold rate is 0.200. `labels alone` is Clopper-Pearson on the same 225.

| wording | judged rate (p > 0.5) | estimate | upper bound | labels alone | lam | rho^2 | worth in labels | tau 0.25 |
|---|---|---|---|---|---|---|---|---|
| rubric 0 | 0.042 | 0.204 | **0.247** | 0.239 | 0.86 | 0.41 | 285 | yes |
| rubric 1 | 0.008 | 0.197 | **0.242** | 0.239 | 0.96 | 0.28 | 254 | yes |
| rubric 2 | 0.054 | 0.195 | **0.239** | 0.239 | 0.28 | 0.06 | 275 | yes |
| rubric 3 | 0.040 | 0.202 | **0.241** | 0.239 | 0.78 | 0.42 | 344 | yes |
| rubric 4 | 0.090 | 0.198 | **0.238** | 0.239 | 0.68 | 0.43 | 325 | yes |
| rubric 5 | 0.060 | 0.197 | **0.236** | 0.239 | 0.70 | 0.40 | 354 | yes |
| bare sentence | 0.304 | 0.207 | **0.246** | 0.239 | 0.56 | 0.48 | 344 | yes |

Across the six rubric wordings the judged rate runs 0.008 to 0.090 (spread 0.082); the certified estimate runs 0.195 to 0.204 (spread 0.009).

### Harm (semantic, rare): the labels carry it, the judge does not

- 225 human labels, 3 positive, a stratified sample of 4,800 responses. Fewer than 10 positives, so the rule routes to the labels; because the sheet is stratified, the bound is the design-weighted `b1w` (013), which is approximate.
- Design-weighted harm rate 0.0134, upper bound **0.0393**; tau = 0.05 certified. Read as i.i.d. the same sheet would claim 0.0341.
- With the judge (weighted PPI++, 0/1 label): estimate 0.0136, lam -0.07: the judge gets no weight. (Its normal-limit bound, 0.0291, is not usable: that limit missed in about 0.20 of the planted sheets at this rate.)
- A carried calibration is refused: 3 human positives; 30 needed to bound recall.

### How far the threshold moves on the judge's scale

Plain PPI with the 0/1 judge: `true rate <= tau` is tested as `judged rate <= tau - rectifier - margin`, the rectifier being the mean of (gold - judge) on the labels.

| constraint | tau | rectifier | margin | judged-rate threshold | judged rate now |
|---|---|---|---|---|---|
| refusal, rubric 0 | 0.25 | +0.156 | 0.040 | 0.055 | 0.042 |
| refusal, bare sentence | 0.25 | -0.089 | 0.039 | 0.300 | 0.304 |
| brevity, compiled judge | 0.45 | +0.387 | 0.060 | 0.003 | 0.048 |
| harm, rubric 0 (design-weighted) | 0.05 | -0.220 | 0.053 | 0.217 | 0.230 |

---

<!-- check_cert.md -->
## Spike 017: checks on known truth

1. Clopper-Pearson vs the project function, 300 random (k, n, delta): largest difference 2.2e-16.

2. Miss rate (target <= 0.05) and mean bound, N = 2000 judged, n labelled at random, 20000 draws. `gain` = var(labels-alone estimate) / var(estimate).

`block` = finite-sample block PPI with lam from an independent labelled sample of the same size (2000 draws).

| rate | sens | FA | n | rho^2 | classical | naive | youden | ppi | ppi++ | ppi++ wilson | exact3 | strat | block | ppi gain (formula / MC) | ppi++ gain (formula / MC) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.2 | 0.9 | 0.05 | 100 | 0.674 | 0.048 (0.277) | 0.000 (0.236) | 0.000 (0.356) | 0.040 (0.243) | 0.081 (0.235) | 0.079 (0.240) | 0.000 (0.304) | 0.000 (0.300) | 0.000 (0.297) | 2.33 / 2.34 | 2.78 / 2.62 |
| 0.2 | 0.9 | 0.05 | 225 | 0.674 | 0.037 (0.249) | 0.000 (0.236) | 0.000 (0.296) | 0.047 (0.231) | 0.070 (0.226) | 0.055 (0.229) | 0.000 (0.273) | 0.000 (0.269) | 0.004 (0.248) | 1.97 / 1.96 | 2.49 / 2.41 |
| 0.2 | 0.67 | 0.216 | 100 | 0.155 | 0.047 (0.277) | 0.000 (0.324) | 0.000 (0.734) | 0.049 (0.280) | 0.071 (0.259) | 0.046 (0.266) | 0.000 (0.368) | 0.000 (0.343) | 0.014 (0.290) | 0.67 / 0.67 | 1.17 / 1.16 |
| 0.2 | 0.67 | 0.216 | 225 | 0.155 | 0.036 (0.249) | 0.000 (0.324) | 0.000 (0.525) | 0.048 (0.255) | 0.061 (0.240) | 0.047 (0.244) | 0.000 (0.317) | 0.000 (0.294) | 0.031 (0.256) | 0.63 / 0.62 | 1.16 / 1.15 |
| 0.2 | 0.5 | 0.5 | 100 | 0.000 | 0.047 (0.277) | 0.000 (0.519) | 0.000 (0.999) | 0.055 (0.307) | 0.079 (0.265) | 0.047 (0.273) | 0.001 (0.410) | 0.000 (0.350) | 0.032 (0.296) | 0.38 / 0.37 | 1.00 / 0.99 |
| 0.2 | 0.5 | 0.5 | 225 | 0.000 | 0.038 (0.248) | 0.000 (0.519) | 0.000 (0.999) | 0.054 (0.272) | 0.063 (0.243) | 0.052 (0.247) | 0.000 (0.344) | 0.001 (0.294) | 0.036 (0.260) | 0.36 / 0.36 | 1.00 / 1.00 |
| 0.05 | 0.9 | 0.05 | 100 | 0.409 | 0.037 (0.101) | 0.000 (0.104) | 0.000 (0.403) | 0.025 (0.088) | 0.125 (0.073) | 0.056 (0.084) | 0.000 (0.142) | 0.000 (0.137) | 0.000 (0.135) | 0.86 / 0.85 | 1.64 / 1.56 |
| 0.05 | 0.9 | 0.05 | 225 | 0.409 | 0.029 (0.081) | 0.000 (0.104) | 0.000 (0.173) | 0.036 (0.077) | 0.090 (0.067) | 0.043 (0.072) | 0.000 (0.112) | 0.000 (0.104) | 0.000 (0.091) | 0.77 / 0.77 | 1.57 / 1.56 |
| 0.05 | 0.67 | 0.216 | 100 | 0.054 | 0.037 (0.102) | 0.000 (0.255) | 0.000 (0.896) | 0.040 (0.123) | 0.116 (0.084) | 0.037 (0.097) | 0.000 (0.202) | 0.000 (0.160) | 0.000 (0.121) | 0.24 / 0.24 | 1.05 / 1.04 |
| 0.05 | 0.67 | 0.216 | 225 | 0.054 | 0.029 (0.081) | 0.000 (0.255) | 0.000 (0.665) | 0.043 (0.100) | 0.088 (0.073) | 0.040 (0.078) | 0.000 (0.154) | 0.000 (0.114) | 0.005 (0.087) | 0.23 / 0.23 | 1.05 / 1.06 |
| 0.05 | 0.5 | 0.5 | 100 | 0.000 | 0.036 (0.101) | 0.000 (0.519) | 0.000 (1.000) | 0.053 (0.142) | 0.121 (0.084) | 0.036 (0.098) | 0.000 (0.239) | 0.000 (0.160) | 0.000 (0.118) | 0.15 / 0.15 | 1.00 / 0.99 |
| 0.05 | 0.5 | 0.5 | 225 | 0.000 | 0.029 (0.081) | 0.000 (0.519) | 0.000 (0.999) | 0.054 (0.113) | 0.080 (0.074) | 0.034 (0.079) | 0.000 (0.179) | 0.000 (0.114) | 0.012 (0.087) | 0.14 / 0.14 | 1.00 / 1.00 |
| 0.013 | 0.9 | 0.05 | 100 | 0.162 | 0.000 (0.050) | 0.000 (0.071) | 0.000 (0.821) | 0.023 (0.049) | 0.268 (0.025) | 0.000 (0.043) | 0.000 (0.100) | 0.000 (0.093) | 0.000 (0.083) | 0.25 / 0.25 | 1.18 / 1.24 |
| 0.013 | 0.9 | 0.05 | 225 | 0.162 | 0.000 (0.033) | 0.000 (0.071) | 0.000 (0.495) | 0.031 (0.039) | 0.184 (0.023) | 0.052 (0.030) | 0.000 (0.070) | 0.000 (0.058) | 0.000 (0.046) | 0.23 / 0.23 | 1.17 / 1.19 |
| 0.013 | 0.67 | 0.216 | 100 | 0.015 | 0.000 (0.050) | 0.000 (0.238) | 0.000 (0.992) | 0.039 (0.083) | 0.269 (0.028) | 0.000 (0.047) | 0.000 (0.158) | 0.000 (0.104) | 0.000 (0.074) | 0.07 / 0.07 | 1.01 / 1.02 |
| 0.013 | 0.67 | 0.216 | 225 | 0.015 | 0.000 (0.033) | 0.000 (0.238) | 0.000 (0.941) | 0.040 (0.062) | 0.207 (0.024) | 0.055 (0.032) | 0.000 (0.110) | 0.000 (0.060) | 0.000 (0.042) | 0.07 / 0.07 | 1.01 / 1.02 |
| 0.013 | 0.5 | 0.5 | 100 | 0.000 | 0.000 (0.050) | 0.000 (0.519) | 0.000 (1.000) | 0.053 (0.099) | 0.272 (0.028) | 0.000 (0.047) | 0.000 (0.191) | 0.000 (0.103) | 0.000 (0.073) | 0.05 / 0.04 | 1.00 / 0.99 |
| 0.013 | 0.5 | 0.5 | 225 | 0.000 | 0.000 (0.033) | 0.000 (0.519) | 0.000 (1.000) | 0.050 (0.073) | 0.210 (0.025) | 0.052 (0.032) | 0.000 (0.133) | 0.000 (0.059) | 0.000 (0.042) | 0.04 / 0.04 | 1.00 / 1.00 |

4. `strat_exact` with labels drawn by judge stratum (n = 225: up to 112 flagged, the rest cleared), 20000 draws.

| rate | sens | FA | miss | mean bound | classical at random, mean bound |
|---|---|---|---|---|---|
| 0.2 | 0.9 | 0.05 | 0.000 | 0.2728 | 0.2486 |
| 0.013 | 0.67 | 0.216 | 0.000 | 0.0583 | 0.0333 |
| 0.013 | 0.8 | 0.015 | 0.000 | 0.0466 | 0.0333 |
| 0.05 | 0.8 | 0.015 | 0.000 | 0.1034 | 0.0807 |
