# Spike 024 results

60 paired evaluation seeds (0-59) per arm and policy; utility = true reward of the returned policy minus the reference's (No Solution Found = 0). Paired differences are against `lagrangian` unless stated.

## linear

| Arm | settings | solutions | over | utility (se) | paired vs lagrangian (se) | ckpts over | pred-pass safety-fail | grid median | spread share | severe share | EB C'(0)<0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `lagrangian` | {'eta': 100.0, 'lam0': 5.0, 'lr': 0.1} | 35/60 | 0 | 0.235 (0.029) |  | 16.9% | 16 | 0.169 | 0.58 | 0.43 |  |
| `pi_mean` | {'c0': 10.0, 'eta_i': 30.0, 'lr': 0.1} | 47/60 | 0 | 0.283 (0.024) | +0.048 (0.025) | 0.2% | 13 | 0.266 | 0.65 | 0.41 |  |
| `pi_full` | {'c0': 10.0, 'eta_i': 30.0, 'lr': 0.1} | 43/60 | 0 | 0.288 (0.027) | +0.053 (0.039) | 2.1% | 15 | 0.258 | 0.62 | 0.44 | 0.0% |
| `pi_mean_sep` | {'c0': 10.0, 'eta_i': 10.0, 'lr': 0.1} | 50/60 | 0 | 0.299 (0.021) | +0.064 (0.031) | 0.0% | 10 | 0.192 | 0.61 | 0.43 |  |
| `pi_full_sep` | {'c0': 10.0, 'eta_i': 10.0, 'lr': 0.1} | 44/60 | 0 | 0.284 (0.026) | +0.049 (0.033) | 0.2% | 16 | 0.224 | 0.52 | 0.44 | 0.0% |

Pairs (paired difference, se):

- `pi_full` minus `pi_mean`: +0.004 (0.032)
- `pi_full_sep` minus `pi_mean_sep`: -0.015 (0.030)
- `pi_mean_sep` minus `pi_mean`: +0.015 (0.032)
- `pi_full_sep` minus `pi_full`: -0.004 (0.034)

## mlp

| Arm | settings | solutions | over | utility (se) | paired vs lagrangian (se) | ckpts over | pred-pass safety-fail | grid median | spread share | severe share | EB C'(0)<0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `lagrangian` | {'eta': 300.0, 'lam0': 5.0, 'lr': 0.003} | 35/60 | 0 | 0.193 (0.024) |  | 24.0% | 13 | 0.181 | 0.58 | 0.42 |  |
| `pi_mean` | {'c0': 10.0, 'eta_i': 3.0, 'lr': 0.001} | 42/60 | 0 | 0.291 (0.027) | +0.099 (0.037) | 0.8% | 17 | 0.233 | 0.60 | 0.43 |  |
| `pi_full` | {'c0': 30.0, 'eta_i': 30.0, 'lr': 0.001} | 43/60 | 0 | 0.254 (0.026) | +0.061 (0.033) | 0.0% | 17 | 0.242 | 0.64 | 0.45 | 0.0% |
| `pi_mean_sep` | {'c0': 10.0, 'eta_i': 10.0, 'lr': 0.001} | 52/60 | 0 | 0.295 (0.021) | +0.103 (0.027) | 0.0% | 8 | 0.155 | 0.56 | 0.44 |  |
| `pi_full_sep` | {'c0': 10.0, 'eta_i': 3.0, 'lr': 0.001} | 46/60 | 0 | 0.290 (0.025) | +0.097 (0.030) | 0.0% | 13 | 0.181 | 0.54 | 0.42 | 0.0% |

Pairs (paired difference, se):

- `pi_full` minus `pi_mean`: -0.037 (0.034)
- `pi_full_sep` minus `pi_mean_sep`: -0.005 (0.027)
- `pi_mean_sep` minus `pi_mean`: +0.004 (0.031)
- `pi_full_sep` minus `pi_full`: +0.036 (0.032)

