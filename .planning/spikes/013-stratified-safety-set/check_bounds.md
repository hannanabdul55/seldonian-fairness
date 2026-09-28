## 1. pooled (one stratum), n = 200
p 0.05: mean 0.060  b1 0.0820  b2 0.1424 (13 ms)  project betting 0.1073  CP 0.0876
p 0.3: mean 0.260  b1 0.3000  b2 0.3791 (12 ms)  project betting 0.3369  CP 0.3040
p 0.5: mean 0.455  b1 0.5004  b2 0.5568 (12 ms)  project betting 0.5385  CP 0.5028

## 2-3. coverage and width, proportional allocation, 2000 reps each
H4 spread n_s 100: b1 miss 0.095 w 0.0493  b2 miss 0.000 w 0.1490  b1_pool miss 0.051 w 0.0614  b2_pool miss 0.000 w 0.1477  | ESS b1 1.55 b2 0.98
H4 spread n_s 400: b1 miss 0.112 w 0.0239  b2 miss 0.000 w 0.0818  b1_pool miss 0.072 w 0.0305  b2_pool miss 0.000 w 0.0818  | ESS b1 1.62 b2 1.00
H4 flat n_s 100: b1 miss 0.114 w 0.0597  b2 miss 0.000 w 0.1807  b1_pool miss 0.114 w 0.0592  b2_pool miss 0.000 w 0.1482  | ESS b1 0.98 b2 0.67
H4 flat n_s 400: b1 miss 0.107 w 0.0295  b2 miss 0.000 w 0.1071  b1_pool miss 0.107 w 0.0294  b2_pool miss 0.001 w 0.0799  | ESS b1 1.00 b2 0.56
H8 spread n_s 100: b1 miss 0.078 w 0.0555  b2 miss 0.000 w 0.1701  b1_pool miss 0.063 w 0.0657  b2_pool miss 0.000 w 0.1399  | ESS b1 1.40 b2 0.68
H8 spread n_s 400: b1 miss 0.104 w 0.0260  b2 miss 0.000 w 0.0968  b1_pool miss 0.061 w 0.0319  b2_pool miss 0.000 w 0.0774  | ESS b1 1.51 b2 0.64
H2 low n_s 100: b1 miss 0.106 w 0.0294  b2 miss 0.007 w 0.1015  b1_pool miss 0.106 w 0.0287  b2_pool miss 0.007 w 0.1153  | ESS b1 0.95 b2 1.29
H2 low n_s 400: b1 miss 0.154 w 0.0140  b2 miss 0.000 w 0.0621  b1_pool miss 0.154 w 0.0141  b2_pool miss 0.000 w 0.0706  | ESS b1 1.03 b2 1.29

## two-phase: pool of N from a population, target = population mean
N 1000 n_s 400: b1 miss 0.111  b2 miss 0.000
N 5000 n_s 200: b1 miss 0.080  b2 miss 0.000
