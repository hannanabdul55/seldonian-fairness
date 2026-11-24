# Q6: the carried-bound test across the targeted update, with Claude's label as a second gold

2,000 responses (500 per set) labelled once by agents under the human guideline. Source: `results/paper/claude_labels_carried/carried.json`. Pass B batches without labels: none.

## The labels, per set

| set | guard's flag rate | Claude r | Claude r or h | Claude's labels (r, a, h, u) | of the guard's flags: r, h, a | Claude r that the guard flags | kappa, guard against Claude r |
|---|---|---|---|---|---|---|---|
| before the update | 0.200 | 0.152 | 0.182 | 76, 409, 15, 0 | 74, 11, 15 of 100 | 74 of 76 | 0.808 |
| after, seed 0 | 0.186 | 0.146 | 0.174 | 73, 413, 14, 0 | 69, 10, 14 of 93 | 69 of 73 | 0.798 |
| after, seed 1 | 0.216 | 0.164 | 0.194 | 82, 403, 15, 0 | 78, 12, 18 of 108 | 78 of 82 | 0.780 |
| after, seed 2 | 0.194 | 0.162 | 0.186 | 81, 407, 12, 0 | 76, 8, 13 of 97 | 76 of 81 | 0.823 |

## Gold: the guard's flag (the paper's test, recomputed here on a new random stream)

Carried bound over its level in 8 of 18 cells.

| target | wording | recall before -> after (Fisher p) | false-alarm rate before -> after (p) | target rate | carried bound misses | median bound |
|---|---|---|---|---|---|---|
| after, seed 0 | 0 | 21/100 -> 3/93 (0.000141) | 0/400 -> 0/407 (1) | 0.186 | 0.810 OVER | 0.114 |
| after, seed 0 | 1 | 4/100 -> 0/93 (0.122) | 0/400 -> 0/407 (1) | 0.186 | 0.224 OVER | 0.531 |
| after, seed 0 | 2 | 12/100 -> 0/93 (0.000373) | 15/400 -> 12/407 (0.562) | 0.186 | 0.079 OVER | 0.821 |
| after, seed 0 | 3 | 20/100 -> 14/93 (0.45) | 0/400 -> 1/407 (1) | 0.186 | 0.010 | 0.446 |
| after, seed 0 | 4 | 39/100 -> 29/93 (0.292) | 6/400 -> 5/407 (0.771) | 0.186 | 0.008 | 0.332 |
| after, seed 0 | 5 | 29/100 -> 18/93 (0.133) | 1/400 -> 1/407 (1) | 0.186 | 0.051 | 0.310 |
| after, seed 1 | 0 | 21/100 -> 8/108 (0.00514) | 0/400 -> 0/392 (1) | 0.216 | 0.402 OVER | 0.242 |
| after, seed 1 | 1 | 4/100 -> 0/108 (0.0518) | 0/400 -> 1/392 (0.495) | 0.216 | 0.065 OVER | 1.000 |
| after, seed 1 | 2 | 12/100 -> 3/108 (0.014) | 15/400 -> 6/392 (0.0748) | 0.216 | 0.232 OVER | 0.520 |
| after, seed 1 | 3 | 20/100 -> 17/108 (0.471) | 0/400 -> 0/392 (1) | 0.216 | 0.014 | 0.497 |
| after, seed 1 | 4 | 39/100 -> 36/108 (0.47) | 6/400 -> 1/392 (0.123) | 0.216 | 0.016 | 0.357 |
| after, seed 1 | 5 | 29/100 -> 24/108 (0.27) | 1/400 -> 1/392 (1) | 0.216 | 0.021 | 0.392 |
| after, seed 2 | 0 | 21/100 -> 5/97 (0.00126) | 0/400 -> 0/403 (1) | 0.194 | 0.611 OVER | 0.167 |
| after, seed 2 | 1 | 4/100 -> 1/97 (0.369) | 0/400 -> 0/403 (1) | 0.194 | 0.046 | 1.000 |
| after, seed 2 | 2 | 12/100 -> 5/97 (0.127) | 15/400 -> 5/403 (0.0244) | 0.194 | 0.149 OVER | 0.627 |
| after, seed 2 | 3 | 20/100 -> 15/97 (0.458) | 0/400 -> 0/403 (1) | 0.194 | 0.014 | 0.447 |
| after, seed 2 | 4 | 39/100 -> 32/97 (0.458) | 6/400 -> 0/403 (0.015) | 0.194 | 0.027 | 0.314 |
| after, seed 2 | 5 | 29/100 -> 20/97 (0.19) | 1/400 -> 0/403 (0.498) | 0.194 | 0.054 | 0.319 |

## Gold: Claude's strict refusal (r)

Carried bound over its level in 10 of 18 cells.

| target | wording | recall before -> after (Fisher p) | false-alarm rate before -> after (p) | target rate | carried bound misses | median bound |
|---|---|---|---|---|---|---|
| after, seed 0 | 0 | 21/76 -> 3/73 (8.75e-05) | 0/424 -> 0/427 (1) | 0.146 | 0.835 OVER | 0.085 |
| after, seed 0 | 1 | 4/76 -> 0/73 (0.12) | 0/424 -> 0/427 (1) | 0.146 | 0.226 OVER | 0.405 |
| after, seed 0 | 2 | 12/76 -> 0/73 (0.000311) | 15/424 -> 12/427 (0.564) | 0.146 | 0.088 OVER | 0.525 |
| after, seed 0 | 3 | 20/76 -> 12/73 (0.165) | 0/424 -> 3/427 (0.249) | 0.146 | 0.011 | 0.339 |
| after, seed 0 | 4 | 37/76 -> 27/73 (0.186) | 8/424 -> 7/427 (0.801) | 0.146 | 0.009 | 0.258 |
| after, seed 0 | 5 | 27/76 -> 16/73 (0.0733) | 3/424 -> 3/427 (1) | 0.146 | 0.038 | 0.253 |
| after, seed 1 | 0 | 21/76 -> 5/82 (0.000412) | 0/424 -> 3/418 (0.122) | 0.164 | 0.408 OVER | 0.182 |
| after, seed 1 | 1 | 4/76 -> 1/82 (0.196) | 0/424 -> 0/418 (1) | 0.164 | 0.052 | 1.000 |
| after, seed 1 | 2 | 12/76 -> 4/82 (0.0331) | 15/424 -> 5/418 (0.0391) | 0.164 | 0.255 OVER | 0.340 |
| after, seed 1 | 3 | 20/76 -> 14/82 (0.179) | 0/424 -> 3/418 (0.122) | 0.164 | 0.013 | 0.372 |
| after, seed 1 | 4 | 37/76 -> 33/82 (0.337) | 8/424 -> 4/418 (0.384) | 0.164 | 0.015 | 0.279 |
| after, seed 1 | 5 | 27/76 -> 21/82 (0.226) | 3/424 -> 4/418 (0.724) | 0.164 | 0.010 | 0.320 |
| after, seed 2 | 0 | 21/76 -> 5/81 (0.000417) | 0/424 -> 0/419 (1) | 0.162 | 0.691 OVER | 0.124 |
| after, seed 2 | 1 | 4/76 -> 1/81 (0.199) | 0/424 -> 0/419 (1) | 0.162 | 0.059 OVER | 1.000 |
| after, seed 2 | 2 | 12/76 -> 5/81 (0.0716) | 15/424 -> 5/419 (0.0391) | 0.162 | 0.189 OVER | 0.395 |
| after, seed 2 | 3 | 20/76 -> 12/81 (0.0789) | 0/424 -> 3/419 (0.122) | 0.162 | 0.032 | 0.334 |
| after, seed 2 | 4 | 37/76 -> 29/81 (0.109) | 8/424 -> 3/419 (0.224) | 0.162 | 0.057 OVER | 0.243 |
| after, seed 2 | 5 | 27/76 -> 19/81 (0.115) | 3/424 -> 1/419 (0.624) | 0.162 | 0.062 OVER | 0.267 |

## Gold: Claude's r or h

Carried bound over its level in 10 of 18 cells.

| target | wording | recall before -> after (Fisher p) | false-alarm rate before -> after (p) | target rate | carried bound misses | median bound |
|---|---|---|---|---|---|---|
| after, seed 0 | 0 | 21/91 -> 3/87 (0.000112) | 0/409 -> 0/413 (1) | 0.174 | 0.822 OVER | 0.102 |
| after, seed 0 | 1 | 4/91 -> 0/87 (0.121) | 0/409 -> 0/413 (1) | 0.174 | 0.231 OVER | 0.489 |
| after, seed 0 | 2 | 12/91 -> 0/87 (0.000341) | 15/409 -> 12/413 (0.563) | 0.174 | 0.084 OVER | 0.685 |
| after, seed 0 | 3 | 20/91 -> 12/87 (0.175) | 0/409 -> 3/413 (0.249) | 0.174 | 0.015 | 0.409 |
| after, seed 0 | 4 | 38/91 -> 28/87 (0.215) | 7/409 -> 6/413 (0.788) | 0.174 | 0.010 | 0.303 |
| after, seed 0 | 5 | 28/91 -> 17/87 (0.12) | 2/409 -> 2/413 (1) | 0.174 | 0.047 | 0.294 |
| after, seed 1 | 0 | 21/91 -> 5/97 (0.000511) | 0/409 -> 3/403 (0.122) | 0.194 | 0.395 OVER | 0.220 |
| after, seed 1 | 1 | 4/91 -> 1/97 (0.2) | 0/409 -> 0/403 (1) | 0.194 | 0.058 OVER | 1.000 |
| after, seed 1 | 2 | 12/91 -> 4/97 (0.0352) | 15/409 -> 5/403 (0.0391) | 0.194 | 0.229 OVER | 0.455 |
| after, seed 1 | 3 | 20/91 -> 14/97 (0.19) | 0/409 -> 3/403 (0.122) | 0.194 | 0.014 | 0.451 |
| after, seed 1 | 4 | 38/91 -> 34/97 (0.37) | 7/409 -> 3/403 (0.341) | 0.194 | 0.010 | 0.335 |
| after, seed 1 | 5 | 28/91 -> 21/97 (0.184) | 2/409 -> 4/403 (0.448) | 0.194 | 0.011 | 0.377 |
| after, seed 2 | 0 | 21/91 -> 5/93 (0.000589) | 0/409 -> 0/407 (1) | 0.186 | 0.652 OVER | 0.153 |
| after, seed 2 | 1 | 4/91 -> 1/93 (0.209) | 0/409 -> 0/407 (1) | 0.186 | 0.057 OVER | 1.000 |
| after, seed 2 | 2 | 12/91 -> 5/93 (0.0786) | 15/409 -> 5/407 (0.039) | 0.186 | 0.167 OVER | 0.532 |
| after, seed 2 | 3 | 20/91 -> 12/93 (0.122) | 0/409 -> 3/407 (0.124) | 0.186 | 0.021 | 0.404 |
| after, seed 2 | 4 | 38/91 -> 29/93 (0.168) | 7/409 -> 3/407 (0.341) | 0.186 | 0.039 | 0.291 |
| after, seed 2 | 5 | 28/91 -> 19/93 (0.129) | 2/409 -> 1/407 (1) | 0.186 | 0.058 OVER | 0.308 |

Second labelling of 250 items: agreement 0.992, kappa 0.972. Unsure: 0.

## Registered predictions

- G1 Claude's strict-refusal rate is below the guard's flag rate after the update, in all three seeds: kept
- G2 the share of the guard's flags Claude labels h is higher after the update than before, in all three seeds: REFUTED
- G3 with Claude's strict label as gold, wording 0's recall does not differ across the update (Fisher p >= 0.05) in at least two of three seeds: REFUTED
- G4 the carried bound is over its level in fewer of the 18 cells with Claude's strict gold than with the guard's: REFUTED
- G5 with Claude's strict gold the carried bound is over its level in at most 2 of 18 cells: REFUTED
- G6 with Claude's broad gold (r or h) it is over its level in at least 6 of 18 cells: kept
- G7 second labelling of 250: four-way kappa >= 0.80: kept
- G8 unsure on at most 40 of 2,000: kept

Registered reading (strict gold; at most 2 of 18 over: depends on the gold label; at least 6: holds under a second gold label; otherwise undecided): the failure across the targeted update holds under a second gold label.
