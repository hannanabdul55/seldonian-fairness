# Order dependence of the sequential betting bound

An adversary re-shuffles the safety data until the lower bound clears the threshold. `miss_single` is the honest one-sided miscoverage (one ordering); `miss_best_of_tries` is the miscoverage of the best of `tries` orderings; `mean_spread` is the average max - min of the lower bound across orderings.

| method | dist | n | tries | miss_single | miss_best_of_tries | mean_spread |
|---|---|---|---|---|---|---|
| betting | beta(2,8) | 50 | 20 | 0.0167 | 0.0300 | 0.0174 |
| betting(permutations=16) | beta(2,8) | 50 | 20 | 0.0000 | 0.0000 | 0.0033 |
| betting_mixture | beta(2,8) | 50 | 5 | 0.0000 | 0.0000 | 0.0000 |
| betting | beta(2,8) | 200 | 20 | 0.0233 | 0.1067 | 0.0102 |
| betting(permutations=16) | beta(2,8) | 200 | 20 | 0.0000 | 0.0000 | 0.0010 |
| betting_mixture | beta(2,8) | 200 | 5 | 0.0000 | 0.0000 | 0.0000 |
| betting | spike-slab | 50 | 20 | 0.0033 | 0.0167 | 0.0048 |
| betting(permutations=16) | spike-slab | 50 | 20 | 0.0000 | 0.0000 | 0.0008 |
| betting_mixture | spike-slab | 50 | 5 | 0.0000 | 0.0000 | 0.0000 |
| betting | spike-slab | 200 | 20 | 0.0000 | 0.0400 | 0.0090 |
| betting(permutations=16) | spike-slab | 200 | 20 | 0.0000 | 0.0000 | 0.0011 |
| betting_mixture | spike-slab | 200 | 5 | 0.0000 | 0.0000 | 0.0000 |
| betting | uniform | 50 | 20 | 0.0167 | 0.0967 | 0.0509 |
| betting(permutations=16) | uniform | 50 | 20 | 0.0000 | 0.0000 | 0.0031 |
| betting_mixture | uniform | 50 | 5 | 0.0000 | 0.0000 | 0.0000 |
| betting | uniform | 200 | 20 | 0.0300 | 0.1867 | 0.0296 |
| betting(permutations=16) | uniform | 200 | 20 | 0.0000 | 0.0000 | 0.0011 |
| betting_mixture | uniform | 200 | 5 | 0.0100 | 0.0100 | 0.0000 |
