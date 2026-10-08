# The labels-alone rows of Table 4, by enumeration

`scripts/binomial_rows.py`. Labels drawn with replacement from a 0/1 pool are a binomial count, so each bound's miss probability is a finite sum over counts. Nothing is simulated and the lost pool is not needed, only its rate.

## 1. The training paper's printed table (delta 0.10, 5,000 draws) against the exact values

The pool held 3,600 labels at a printed rate of 0.034, which is a count of 121 to 124. The count is not recorded, so it is chosen here by its fit to the 25 printed values. Agreement by count (z = printed minus exact, in Monte Carlo standard errors of a 5,000-draw estimate):

| pool count | rate | cells within 2 se | largest abs z | sum of z squared |
|---|---|---|---|---|
| 121 | 0.03361 | 14 of 25 | 8.71 | 270.6 |
| 122 | 0.03389 | 16 of 25 | 9.67 | 220.0 |
| 123 | 0.03417 | 19 of 25 | 3.62 | 60.3 |
| 124 | 0.03444 | 25 of 25 | 1.27 | 8.0 |

At the best-fitting count, 124 of 3,600:

| bound | n 200: printed / exact | n 400: printed / exact | n 800: printed / exact | n 1200: printed / exact | n 2400: printed / exact |
|---|---|---|---|---|---|
| Student-t | 0.184 / 0.1786 | 0.115 / 0.1161 | 0.121 / 0.1174 | 0.101 / 0.1046 | 0.101 / 0.1037 |
| Clopper-Pearson | 0.084 / 0.0840 | 0.067 / 0.0658 | 0.083 / 0.0808 | 0.074 / 0.0770 | 0.084 / 0.0841 |
| Bentkus | 0.029 / 0.0302 | 0.035 / 0.0335 | 0.034 / 0.0335 | 0.023 / 0.0259 | 0.032 / 0.0319 |
| betting mixture | 0.007 / 0.0073 | 0.016 / 0.0150 | 0.011 / 0.0113 | 0.009 / 0.0107 | 0.010 / 0.0097 |
| Hoeffding | 0.000 / 0.0000 | 0.000 / 0.0000 | 0.000 / 0.0000 | 0.000 / 0.0000 | 0.000 / 0.0000 |
| Anderson | 0.000 / 0.0000 | 0.000 / 0.0000 | 0.000 / 0.0000 | 0.000 / 0.0000 | 0.000 / 0.0000 |

## 2. Largest miss probability over rates of 0.5% to 20% (steps of 0.25 points)

Each cell: the largest exact miss, the rate at which it occurs, and the share of the grid's rates at which the miss exceeds delta.

**delta 0.05**

| bound | n 200 | n 400 | n 800 | n 1200 | n 2400 |
|---|---|---|---|---|---|
| Student-t | 0.367 at 0.50% (100%) | 0.198 at 0.75% (100%) | 0.150 at 0.75% (100%) | 0.151 at 0.50% (100%) | 0.091 at 0.75% (100%) |
| Clopper-Pearson | 0.050 at 19.25% (0%) | 0.049 at 12.00% (0%) | 0.050 at 17.50% (0%) | 0.050 at 17.50% (0%) | 0.050 at 5.75% (0%) |
| Bentkus | 0.049 at 1.50% (0%) | 0.049 at 0.75% (0%) | 0.018 at 18.25% (0%) | 0.018 at 13.75% (0%) | 0.018 at 17.50% (0%) |
| betting mixture | 0.029 at 1.75% (0%) | 0.018 at 1.00% (0%) | 0.018 at 0.50% (0%) | 0.007 at 1.00% (0%) | 0.007 at 0.50% (0%) |
| Hoeffding | 0.001 at 19.75% (0%) | 0.001 at 20.00% (0%) | 0.001 at 20.00% (0%) | 0.001 at 20.00% (0%) | 0.001 at 20.00% (0%) |
| Anderson | 0.001 at 19.75% (0%) | 0.001 at 20.00% (0%) | 0.001 at 20.00% (0%) | 0.001 at 20.00% (0%) | 0.001 at 20.00% (0%) |

**delta 0.1**

| bound | n 200 | n 400 | n 800 | n 1200 | n 2400 |
|---|---|---|---|---|---|
| Student-t | 0.367 at 0.50% (97%) | 0.237 at 1.00% (99%) | 0.237 at 0.50% (96%) | 0.154 at 1.00% (99%) | 0.154 at 0.50% (97%) |
| Clopper-Pearson | 0.099 at 15.50% (0%) | 0.100 at 16.75% (0%) | 0.100 at 14.00% (0%) | 0.100 at 12.25% (0%) | 0.100 at 15.25% (0%) |
| Bentkus | 0.081 at 1.25% (0%) | 0.049 at 0.75% (0%) | 0.036 at 12.50% (0%) | 0.037 at 10.25% (0%) | 0.037 at 19.50% (0%) |
| betting mixture | 0.049 at 1.50% (0%) | 0.049 at 0.75% (0%) | 0.020 at 1.50% (0%) | 0.020 at 1.00% (0%) | 0.015 at 0.75% (0%) |
| Hoeffding | 0.003 at 19.75% (0%) | 0.003 at 20.00% (0%) | 0.003 at 20.00% (0%) | 0.003 at 20.00% (0%) | 0.003 at 20.00% (0%) |
| Anderson | 0.003 at 19.75% (0%) | 0.003 at 20.00% (0%) | 0.003 at 20.00% (0%) | 0.003 at 20.00% (0%) | 0.003 at 20.00% (0%) |

