### Coverage of the recommended setting: S2, random-tie equal strata, H = 8, k = 8, b1w (max over 3 checkpoints) vs R

| pool:label | n_s | delta | R | S2 H8 |
|---|---|---|---|---|
| C1:refusal | 100 | 0.05 | 0.033 | 0.023 |
| C1:refusal | 100 | 0.1 | 0.066 | 0.076 |
| C1:refusal | 200 | 0.05 | 0.028 | 0.024 |
| C1:refusal | 200 | 0.1 | 0.075 | 0.069 |
| C2:unsafe | 100 | 0.05 | 0.038 | 0.034 |
| C2:unsafe | 100 | 0.1 | 0.088 | 0.085 |
| C2:unsafe | 200 | 0.05 | 0.025 | 0.027 |
| C2:unsafe | 200 | 0.1 | 0.084 | 0.080 |
| C2:gated | 100 | 0.05 | 0.000 | 0.000 |
| C2:gated | 100 | 0.1 | 0.126 | 0.126 |
| C2:gated | 200 | 0.05 | 0.048 | 0.038 |
| C2:gated | 200 | 0.1 | 0.079 | 0.067 |
| C2:refusal | 100 | 0.05 | 0.057 | 0.054 |
| C2:refusal | 100 | 0.1 | 0.114 | 0.097 |
| C2:refusal | 200 | 0.05 | 0.043 | 0.045 |
| C2:refusal | 200 | 0.1 | 0.087 | 0.091 |
| C3:unsafe | 100 | 0.05 | 0.000 | 0.000 |
| C3:unsafe | 100 | 0.1 | 0.000 | 0.000 |
| C3:unsafe | 200 | 0.05 | 0.000 | 0.000 |
| C3:unsafe | 200 | 0.1 | 0.102 | 0.093 |
| C3:refusal | 100 | 0.05 | 0.042 | 0.047 |
| C3:refusal | 100 | 0.1 | 0.091 | 0.092 |
| C3:refusal | 200 | 0.05 | 0.026 | 0.040 |
| C3:refusal | 200 | 0.1 | 0.072 | 0.093 |

Regenerated on 2026-10-07 from `results/spikes/013/real_plasmode_randomties.json.xz` after the correction of `b1w` at zero positives (`reports/b1w_fix_and_audit_2026-10-06.md`). Before it the two rare labels read 0.238-0.448 at n_s 100 and 0.174-0.178 for C3:unsafe at n_s 200, at both deltas; every other row is unchanged. These draws take 20-40% of a 500-prompt pool without replacement; `results/paper/replacement_check.md` redraws the same cells with replacement.
