# Spike 022a results

Seeds 0-59, paired across arms. Utility = true reward of the returned policy minus
the reference policy's, No Solution Found = 0. `se` is the standard error over seeds;
the last two columns are the paired difference from GRPO.

## linear policy (36 parameters)

| arm | solutions | over threshold | utility (se) | checkpoints over | final multiplier | KL to ref | utility - GRPO (se) | % of GRPO |
|---|---|---|---|---|---|---|---|---|
| grpo | 52/60 | 0/52 | 0.5226 (0.0296) | 0.081 | 2.4 | 1.2732 | | |
| es | 46/60 | 0/46 | 0.4054 (0.0308) | 0.029 | 3.3 | 1.1254 | -0.1172 (0.0387) | 78% |
| es_x8 | 42/60 | 0/42 | 0.4234 (0.0375) | 0.012 | 2.6 | 1.2644 | -0.0992 (0.0470) | 81% |
| cma | 54/60 | 0/54 | 0.3396 (0.0215) | 0.040 | 1.7 | 1.3076 | -0.1830 (0.0311) | 65% |
| es_direct | 60/60 | 0/60 | 0.4448 (0.0127) | 0.002 | - | 1.1420 | -0.0778 (0.0271) | 85% |

Tuned settings: `grpo` {'lr': 0.4}; `es` {'lr': 0.2, 'm': 8, 'pop': 16, 'sigma': 1.0}; `es_x8` {'lr': 0.2, 'm': 8, 'pop': 16, 'sigma': 1.0}; `cma` {'m': 16, 'pop': 8, 'sigma0': 2.0}; `es_direct` {'lr': 0.2, 'm': 8, 'pop': 16, 'sigma': 1.0}

## mlp policy (94,204 parameters)

| arm | solutions | over threshold | utility (se) | checkpoints over | final multiplier | KL to ref | utility - GRPO (se) | % of GRPO |
|---|---|---|---|---|---|---|---|---|
| grpo | 50/60 | 0/50 | 0.4956 (0.0307) | 0.010 | 1.8 | 0.8439 | | |
| es | 46/60 | 0/46 | 0.3401 (0.0266) | 0.002 | 1.8 | 0.7458 | -0.1555 (0.0361) | 69% |
| es_x8 | 45/60 | 0/45 | 0.4299 (0.0337) | 0.007 | 1.8 | 1.1430 | -0.0657 (0.0486) | 87% |
| es_direct | 59/60 | 0/59 | 0.3642 (0.0129) | 0.000 | - | 0.8919 | -0.1314 (0.0308) | 73% |

Tuned settings: `grpo` {'lr': 0.0003}; `es` {'lr': 0.01, 'm': 8, 'pop': 8, 'sigma': 0.1}; `es_x8` {'lr': 0.01, 'm': 8, 'pop': 8, 'sigma': 0.1}; `es_direct` {'lr': 0.01, 'm': 8, 'pop': 4, 'sigma': 0.1}

