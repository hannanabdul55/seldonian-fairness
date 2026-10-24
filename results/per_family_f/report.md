# Per-family guarantee, rule F (finish one family at a time)

Registration: `.planning/harm-twin/REGISTRATION-PER-FAMILY-F.md`. Plasmode on judge labels, 1000 runs per rule and cell.

**Verdict: GO**

| ID | value | verdict |
|---|---|---|
| V F misclassifies <= 0.1 | worst 0.003 | holds |
| G F vs U families resolved at 600 / 1,000 | 2.8 vs 1.1 / 4.6 vs 3.7 | GO |
| N n80(F) / n80(A) | 2436 / 2444 = 1.00 | GO |

## Families resolved (of 7) when labelling stops at a budget

| label | tau | rule | 300 | 600 | 1000 | 1500 | 2500 | n80 | P(mis) |
|---|---|---|---|---|---|---|---|---|---|
| gated | 0.02 | F | 0.4 | 1.0 | 1.4 | 2.4 | 4.4 | 5858 | 0.000 |
| gated | 0.02 | A | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 5847 | 0.000 |
| gated | 0.02 | U | 0.2 | 0.6 | 1.0 | 1.3 | 3.9 | (fixed n) | 0.000 |
| gated | 0.05 | F | 1.0 | 2.8 | 4.6 | 6.0 | 6.8 | 2436 | 0.000 |
| gated | 0.05 | A | 0.0 | 0.0 | 0.0 | 3.4 | 6.8 | 2444 | 0.000 |
| gated | 0.05 | U | 0.7 | 1.1 | 3.7 | 6.2 | 6.4 | (fixed n) | 0.001 |
| gated | 0.1 | F | 3.0 | 5.9 | 5.9 | 6.1 | 6.6 | 2758 | 0.000 |
| gated | 0.1 | A | 0.0 | 0.0 | 3.6 | 6.1 | 6.7 | 2751 | 0.000 |
| gated | 0.1 | U | 1.0 | 5.7 | 6.0 | 6.0 | 6.1 | (fixed n) | 0.002 |
| ungated | 0.02 | F | 1.6 | 2.1 | 2.7 | 3.1 | 3.7 | 14799 | 0.003 |
| ungated | 0.02 | A | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 14636 | 0.003 |
| ungated | 0.02 | U | 1.4 | 1.9 | 2.7 | 3.4 | 4.4 | (fixed n) | 0.007 |
| ungated | 0.05 | F | 1.1 | 1.8 | 2.2 | 2.4 | 2.6 | 18631 | 0.000 |
| ungated | 0.05 | A | 0.7 | 0.7 | 0.7 | 0.7 | 0.7 | 18882 | 0.001 |
| ungated | 0.05 | U | 1.7 | 2.1 | 2.2 | 2.4 | 2.8 | (fixed n) | 0.013 |
| ungated | 0.1 | F | 2.1 | 3.1 | 3.8 | 4.5 | 5.8 | 3517 | 0.001 |
| ungated | 0.1 | A | 0.2 | 0.2 | 0.6 | 2.8 | 5.4 | 3536 | 0.000 |
| ungated | 0.1 | U | 2.0 | 2.7 | 3.4 | 4.2 | 5.1 | (fixed n) | 0.002 |
