# Spike 023b results

Paired on seeds 0-59 with spike 023's rows.

| policy | arm | solutions | over threshold | utility (se) | ckpts over | smooth_pen minus this (se) | grid median / worst |
|---|---|---|---|---|---|---|---|
| linear | `lagrangian` | 52/60 | 0 | 0.523 (0.030) | 8.1% | +0.068 (0.030) | see 023 |
| linear | `lagrangian_step` | 50/60 | 0 | 0.515 (0.032) | 10.4% | +0.076 (0.033) | 0.513 / 0.375 |
| linear | `smooth_pen` | 59/60 | 0 | 0.591 (0.016) | 0.0% | +0.000 (0.000) | see 023 |
| mlp | `lagrangian` | 49/60 | 0 | 0.518 (0.034) | 10.6% | +0.091 (0.031) | see 023 |
| mlp | `lagrangian_step` | 50/60 | 0 | 0.520 (0.032) | 5.0% | +0.089 (0.031) | 0.490 / 0.296 |
| mlp | `smooth_pen` | 60/60 | 0 | 0.609 (0.012) | 0.0% | +0.000 (0.000) | see 023 |

Chosen: `{"linear": {"eta_s": 1.0, "lam0": 5.0, "lr": 0.2}, "mlp": {"eta_s": 10.0, "lam0": 1.0, "lr": 0.001}}`
