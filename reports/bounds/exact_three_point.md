# Exact coverage on three-point laws

delta = 0.05; atoms [(0.0, 0.5, 1.0), (0.0, 0.1, 1.0), (0.0, 0.9, 1.0), (0.0, 0.05, 0.2)]; 66 probability vectors on a 0.1 simplex grid; exact multinomial enumeration. Minimum one-sided coverage:

| method | n=10 | n=25 | n=40 |
|---|---|---|---|
| ttest | 0.6513 | 0.8769 | 0.8918 |
| hoeffdings | 0.9877 | 0.9927 | 0.9917 |
| bentkus | 0.9718 | 0.9825 | 0.9844 |
| empirical_bentkus | 0.9936 | 0.9905 | 0.9914 |
| chernoff_kl | 0.9718 | 0.9905 | 0.9852 |
| empirical_bernstein | 1.0000 | 1.0000 | 1.0000 |
| anderson | 0.9877 | 0.9927 | 0.9917 |
| learned_miller_thomas | 0.9527 | 0.9558 | 0.9680 |
| betting | 0.9632 | 0.9641 | 0.9516 |
| betting_mixture | 0.9718 | 0.9884 | 0.9852 |
