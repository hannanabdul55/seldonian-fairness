Paired against `random` on the same 2000 seeds: solution-rate difference (paired se), and sd of (d_s - d_c) / se_s, the disagreement between the candidate and safety sets.

| setting | split | d solution | paired se | sd((d_s - d_c)/se) |
|---|---|---|---|---|
| base inf1 | random | +0.000 | 0.000 | 1.239 |
| base inf1 | strat_y | -0.008 | 0.016 | 1.232 |
| base inf1 | strat_Ay | -0.007 | 0.016 | 1.238 |
| base inf1 | alg1_orig_bo30 | +0.000 | 0.006 | 1.227 |
| base inf1 | alg1_gauss_bo30 | -0.013 | 0.016 | 1.125 |
| base inf1 | alg1_gauss_thr | -0.001 | 0.015 | 1.171 |
| base inf1 | maha_design | -0.004 | 0.015 | 1.238 |
| base inf1 | adv_own_score | -0.022 | 0.015 | 1.077 |
| base inf1 | adv_grid | -0.004 | 0.016 | 0.824 |
| base inf1 | strat_cells | -0.009 | 0.016 | 1.226 |
| base inf1 | adv_bins | -0.015 | 0.016 | 1.113 |
| base inf2 | random | +0.000 | 0.000 | 1.249 |
| base inf2 | strat_y | +0.009 | 0.011 | 1.230 |
| base inf2 | strat_Ay | +0.022 | 0.011 | 1.236 |
| base inf2 | alg1_orig_bo30 | +0.003 | 0.004 | 1.238 |
| base inf2 | alg1_gauss_bo30 | +0.026 | 0.010 | 1.144 |
| base inf2 | alg1_gauss_thr | +0.013 | 0.010 | 1.192 |
| base inf2 | maha_design | +0.013 | 0.010 | 1.252 |
| base inf2 | adv_own_score | +0.044 | 0.010 | 1.075 |
| base inf2 | adv_grid | +0.102 | 0.009 | 0.817 |
| base inf2 | strat_cells | +0.002 | 0.011 | 1.252 |
| base inf2 | adv_bins | +0.044 | 0.010 | 1.100 |
| binned inf1 | random | +0.000 | 0.000 | 1.242 |
| binned inf1 | strat_y | +0.006 | 0.013 | 1.257 |
| binned inf1 | strat_cells | +0.025 | 0.014 | 1.219 |
| binned inf1 | alg1_orig_bo30 | +0.002 | 0.006 | 1.233 |
| binned inf1 | alg1_gauss_bo30 | -0.018 | 0.013 | 1.129 |
| binned inf1 | alg1_gauss_thr | -0.011 | 0.013 | 1.171 |
| binned inf1 | maha_design | +0.022 | 0.013 | 1.238 |
| binned inf1 | adv_own_score | -0.016 | 0.013 | 1.076 |
| binned inf1 | adv_grid | -0.068 | 0.013 | 0.890 |
| binned inf1 | adv_bins | -0.004 | 0.014 | 1.104 |
| binned inf2 | random | +0.000 | 0.000 | 1.235 |
| binned inf2 | strat_y | +0.004 | 0.014 | 1.241 |
| binned inf2 | strat_cells | +0.027 | 0.014 | 1.232 |
| binned inf2 | alg1_orig_bo30 | -0.003 | 0.006 | 1.229 |
| binned inf2 | alg1_gauss_bo30 | +0.006 | 0.014 | 1.153 |
| binned inf2 | alg1_gauss_thr | +0.004 | 0.014 | 1.197 |
| binned inf2 | maha_design | +0.007 | 0.013 | 1.245 |
| binned inf2 | adv_own_score | +0.040 | 0.014 | 1.062 |
| binned inf2 | adv_grid | +0.087 | 0.014 | 0.883 |
| binned inf2 | adv_bins | +0.051 | 0.014 | 1.087 |
