# Spike 022a results

Seeds 0-59, paired across arms. Utility = true reward of the returned policy minus
the reference policy's, No Solution Found = 0. `se` is the standard error over seeds;
the last two columns are the paired difference from GRPO.

## linear policy (36 parameters)

| arm | solutions | over threshold | utility (se) | checkpoints over | final multiplier | KL to ref | utility - GRPO (se) | % of GRPO |
|---|---|---|---|---|---|---|---|---|
| grpo | 48/60 | 0/48 | 0.4951 (0.0335) | 0.085 | 2.3 | 1.0753 | | |
| es | 53/60 | 0/53 | 0.3569 (0.0218) | 0.000 | 0.9 | 0.7645 | -0.1382 (0.0408) | 72% |
| es_x8 | 47/60 | 0/47 | 0.4619 (0.0331) | 0.002 | 1.6 | 1.0638 | -0.0331 (0.0404) | 93% |
| cma | 49/60 | 0/49 | 0.4151 (0.0278) | 0.033 | 1.6 | 1.0955 | -0.0799 (0.0459) | 84% |
| es_direct | 59/60 | 0/59 | 0.3520 (0.0151) | 0.000 | - | 0.8644 | -0.1430 (0.0362) | 71% |

Tuned settings: `grpo` {'lr': 0.1}; `es` {'lr': 0.1, 'm': 32, 'pop': 8, 'sigma': 0.3}; `es_x8` {'lr': 0.1, 'm': 32, 'pop': 8, 'sigma': 0.3}; `cma` {'m': 16, 'pop': 30, 'sigma0': 1.0}; `es_direct` {'lr': 0.1, 'm': 8, 'pop': 16, 'sigma': 0.3}

## mlp policy (94,204 parameters)

| arm | solutions | over threshold | utility (se) | checkpoints over | final multiplier | KL to ref | utility - GRPO (se) | % of GRPO |
|---|---|---|---|---|---|---|---|---|
| grpo | 49/60 | 0/49 | 0.5167 (0.0335) | 0.087 | 2.1 | 1.0687 | | |
| es | 47/60 | 0/47 | 0.2370 (0.0211) | 0.004 | 1.5 | 0.6075 | -0.2797 (0.0369) | 46% |
| es_x8 | 41/60 | 0/41 | 0.3101 (0.0289) | 0.007 | 2.7 | 1.0525 | -0.2066 (0.0441) | 60% |
| es_direct | 51/60 | 0/51 | 0.2088 (0.0154) | 0.002 | - | 0.5881 | -0.3078 (0.0341) | 40% |

Tuned settings: `grpo` {'lr': 0.001}; `es` {'lr': 0.01, 'm': 8, 'pop': 8, 'sigma': 0.03}; `es_x8` {'lr': 0.01, 'm': 8, 'pop': 8, 'sigma': 0.03}; `es_direct` {'lr': 0.01, 'm': 8, 'pop': 8, 'sigma': 0.03}

