# Spike 023 results

60 paired evaluation seeds (0-59) per arm and policy; utility = true reward of the returned policy minus the reference's, No Solution Found = 0. Paired differences are against `lagrangian` on the same seeds.

## linear

| arm | hp | solutions | over threshold | utility (se) | % of lagrangian | paired diff (se) | train ckpts over | pred pass, safety fail | grid median / worst (tuning) |
|---|---|---|---|---|---|---|---|---|---|
| `lagrangian` | `{"eta": 100.0, "lam0": 5.0, "lr": 0.4}` | 52/60 | 0 | 0.523 (0.030) | 100% | +0.000 (0.000) | 8.1% | 8 | 0.535 / 0.409 (30) |
| `select` | `{"lr": 0.8}` | 6/60 | 0 | 0.064 (0.025) | 12% | -0.458 (0.035) | 82.5% | 2 | 0.140 / 0.136 (5) |
| `feas_first` | `{"lr": 0.05}` | 48/60 | 0 | 0.490 (0.033) | 94% | -0.033 (0.042) | 1.5% | 12 | 0.478 / 0.458 (5) |
| `projected` | `{"eta_p": 10.0, "lr": 0.05}` | 57/60 | 0 | 0.536 (0.019) | 103% | +0.014 (0.031) | 0.0% | 3 | 0.495 / 0.284 (15) |
| `smooth_pen` | `{"T": 0.02, "c0": 20.0, "lr": 0.4}` | 59/60 | 0 | 0.591 (0.016) | 113% | +0.068 (0.030) | 0.0% | 1 | 0.561 / 0.446 (30) |
| `es_direct` | `{"lr": 0.2, "m": 8, "pop": 16, "sigma": 1.0}` | 60/60 | 0 | 0.445 (0.013) | 85% | -0.078 (0.027) | 0.2% | 0 | not tuned |

## mlp

| arm | hp | solutions | over threshold | utility (se) | % of lagrangian | paired diff (se) | train ckpts over | pred pass, safety fail | grid median / worst (tuning) |
|---|---|---|---|---|---|---|---|---|---|
| `lagrangian` | `{"eta": 30.0, "lam0": 1.0, "lr": 0.003}` | 49/60 | 0 | 0.518 (0.034) | 100% | +0.000 (0.000) | 10.6% | 11 | 0.523 / 0.365 (24) |
| `select` | `{"lr": 0.001}` | 6/60 | 0 | 0.066 (0.026) | 13% | -0.451 (0.039) | 78.5% | 4 | 0.207 / 0.140 (4) |
| `feas_first` | `{"lr": 0.001}` | 45/60 | 0 | 0.461 (0.036) | 89% | -0.056 (0.046) | 1.9% | 15 | 0.461 / 0.319 (4) |
| `projected` | `{"eta_p": 100.0, "lr": 0.001}` | 55/60 | 0 | 0.527 (0.024) | 102% | +0.009 (0.036) | 0.0% | 5 | 0.430 / 0.250 (12) |
| `smooth_pen` | `{"T": 0.02, "c0": 20.0, "lr": 0.001}` | 60/60 | 0 | 0.609 (0.012) | 118% | +0.091 (0.031) | 0.0% | 0 | 0.559 / 0.358 (24) |
| `es_direct` | `{"lr": 0.01, "m": 8, "pop": 4, "sigma": 0.1}` | 59/60 | 0 | 0.364 (0.013) | 70% | -0.153 (0.032) | 0.0% | 1 | not tuned |

