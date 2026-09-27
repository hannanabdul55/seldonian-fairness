# Spike 011 results

Jackpot on 10% of contexts (gain 1.0, loss 1.0 elsewhere); reference uses it ~1%. Means over seeds; `best safe` is the reward of the best safe action per context: 0.695; reference reward 0.509.

| arm | solution | violates | true_rate | true_reward | jackpot_use_on | jackpot_use_off | tv_share | bonus_early | bonus_late | lam_end |
|---|---|---|---|---|---|---|---|---|---|---|
| none | 0.750 | 0.017 | 0.105 | 1.065 | 0.084 | 0.009 | 0.145 | 0.000 | 0.000 | 6.616 |
| lp1 | 0.867 | 0.000 | 0.102 | 1.048 | 0.052 | 0.008 | 0.153 | 0.169 | 0.142 | 7.345 |
| lp2 | 0.767 | 0.000 | 0.099 | 1.007 | 0.078 | 0.018 | 0.146 | 0.361 | 0.277 | 9.190 |
| abs1 | 0.750 | 0.033 | 0.103 | 0.929 | 0.040 | 0.003 | 0.538 | 2.155 | 2.417 | 6.086 |
| random2 | 0.767 | 0.017 | 0.102 | 0.998 | 0.020 | 0.005 | 0.181 | 3.124 | 4.027 | 3.965 |
