# Spike 023c results

Linear policy, seeds 0-59. pi's eta_i = 10.0 (tuning means {0.3: 0.592, 1.0: 0.626, 3.0: 0.624, 10.0: 0.635}).

| reward scale | arm | solutions | over threshold | utility (se) | ckpts over | mean penalty weight | final multiplier |
|---|---|---|---|---|---|---|---|
| 1 | `lagrangian` | 52/60 | 0 | 0.523 (0.030) | 8.1% | nan | 2.43 |
| 1 | `smooth_pen` | 59/60 | 0 | 0.591 (0.016) | 0.0% | 3.73 | nan |
| 1 | `pi` | 59/60 | 0 | 0.594 (0.016) | 0.0% | 3.74 | nan |
| 3 | `lagrangian` | 53/60 | 0 | 0.565 (0.029) | 12.1% | nan | 4.05 |
| 3 | `smooth_pen` | 54/60 | 0 | 0.560 (0.027) | 1.2% | 6.33 | nan |
| 3 | `pi` | 51/60 | 0 | 0.525 (0.031) | 0.8% | 6.58 | nan |
| 10 | `lagrangian` | 46/60 | 0 | 0.485 (0.036) | 16.9% | nan | 10.76 |
| 10 | `smooth_pen` | 40/60 | 0 | 0.428 (0.040) | 9.2% | 11.09 | nan |
| 10 | `pi` | 49/60 | 0 | 0.521 (0.034) | 3.8% | 13.57 | nan |
