# Confirmation of Monte-Carlo-affected cells

Worst-case one-sided coverage on Bernoulli laws (99-point p grid) with more Monte Carlo effort inside the bound / the enumeration. The last row is a plain Monte Carlo estimate of the two-sided coverage at the previously reported worst case (so its target is 1 - 2 delta = 0.90).

| method | setting | n | min coverage |
|---|---|---|---|
| learned_miller_thomas | draws=2000 | 20 | 0.9409 |
| learned_miller_thomas | draws=20000 | 20 | 0.9510 |
| learned_miller_thomas | draws=2000 | 100 | 0.9441 |
| learned_miller_thomas | draws=20000 | 100 | 0.9485 |
| betting | orderings=32 | 50 | 0.9540 |
| betting | orderings=256 | 50 | 0.9619 |
| betting | orderings=32 | 100 | 0.9473 |
| betting | orderings=256 | 100 | 0.9605 |
| betting | MC 20000 reps, p=0.152, both sides | 100 | 0.9498 |
