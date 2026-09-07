# Fairness-constraint bound on |rate_a - rate_b|

The constraint objective certifies an upper bound on the absolute TPR gap at delta = 0.05. One-sample methods build a rectangle from two intervals (delta/4 per tail); `bentkus_diff` bounds the difference directly (delta/2 per tail). Coverage is exact (enumeration of both binomial counts) and minimised over a 25x25 grid of (p_a, p_b).

## Minimum coverage of the certified upper bound

| method | n=(20,20) | n=(50,50) | n=(100,100) | n=(200,200) | n=(30,150) |
|---|---|---|---|---|---|
| ttest | 0.8063 | 0.9835 | 0.9948 | 0.9968 | 0.9342 |
| hoeffdings | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| clopper_pearson | 0.9928 | 0.9949 | 0.9961 | 0.9965 | 0.9931 |
| bentkus | 0.9954 | 0.9981 | 0.9987 | 0.9990 | 0.9974 |
| betting_mixture | 0.9984 | 0.9993 | 0.9997 | 0.9998 | 0.9974 |
| bentkus_diff | 0.9949 | 0.9949 | 0.9946 | 0.9947 | 0.9969 |
| convex_order_diff | 0.9876 | 0.9897 | 0.9902 | n/a | 0.9917 |

## Expected slack E[upper - |p_a - p_b|] at (p_a, p_b) = (0.5, 0.5)

| method | n=(20,20) | n=(50,50) | n=(100,100) | n=(200,200) | n=(30,150) |
|---|---|---|---|---|---|
| ttest | 0.6691 | 0.4066 | 0.2839 | 0.1996 | 0.3881 |
| hoeffdings | 0.7873 | 0.4983 | 0.3524 | 0.2492 | 0.4710 |
| clopper_pearson | 0.5976 | 0.3967 | 0.2835 | 0.2009 | 0.3710 |
| bentkus | 0.6523 | 0.4408 | 0.3173 | 0.2257 | 0.4103 |
| betting_mixture | 0.6974 | 0.4874 | 0.3569 | 0.2576 | 0.4515 |
| bentkus_diff | 0.5139 | 0.3346 | 0.2381 | 0.1686 | 0.3442 |
| convex_order_diff | 0.4750 | 0.3133 | 0.2246 | n/a | 0.3134 |

## Expected slack E[upper - |p_a - p_b|] at (p_a, p_b) = (0.4, 0.6)

| method | n=(20,20) | n=(50,50) | n=(100,100) | n=(200,200) | n=(30,150) |
|---|---|---|---|---|---|
| ttest | 0.5467 | 0.3218 | 0.2230 | 0.1565 | 0.3034 |
| hoeffdings | 0.6766 | 0.4202 | 0.2961 | 0.2093 | 0.3927 |
| clopper_pearson | 0.4609 | 0.3032 | 0.2173 | 0.1549 | 0.2801 |
| bentkus | 0.5102 | 0.3437 | 0.2487 | 0.1782 | 0.3166 |
| betting_mixture | 0.5509 | 0.3864 | 0.2860 | 0.2080 | 0.3548 |
| bentkus_diff | 0.3847 | 0.2457 | 0.1748 | 0.1243 | 0.2562 |
| convex_order_diff | 0.3481 | 0.2254 | 0.1620 | n/a | 0.2270 |

## Expected slack E[upper - |p_a - p_b|] at (p_a, p_b) = (0.1, 0.3)

| method | n=(20,20) | n=(50,50) | n=(100,100) | n=(200,200) | n=(30,150) |
|---|---|---|---|---|---|
| ttest | 0.4017 | 0.2457 | 0.1718 | 0.1208 | 0.2083 |
| hoeffdings | 0.6668 | 0.4189 | 0.2960 | 0.2093 | 0.3913 |
| clopper_pearson | 0.3567 | 0.2343 | 0.1685 | 0.1200 | 0.1706 |
| bentkus | 0.3991 | 0.2659 | 0.1926 | 0.1379 | 0.1905 |
| betting_mixture | 0.4294 | 0.2966 | 0.2194 | 0.1601 | 0.2125 |
| bentkus_diff | 0.3715 | 0.2320 | 0.1601 | 0.1101 | 0.1832 |
| convex_order_diff | 0.3271 | 0.2066 | 0.1441 | n/a | 0.1602 |

## Expected slack E[upper - |p_a - p_b|] at (p_a, p_b) = (0.05, 0.05)

| method | n=(20,20) | n=(50,50) | n=(100,100) | n=(200,200) | n=(30,150) |
|---|---|---|---|---|---|
| ttest | 0.2373 | 0.1667 | 0.1210 | 0.0861 | 0.1554 |
| hoeffdings | 0.7132 | 0.4526 | 0.3203 | 0.2266 | 0.4258 |
| clopper_pearson | 0.3108 | 0.1865 | 0.1296 | 0.0905 | 0.1991 |
| bentkus | 0.3452 | 0.2100 | 0.1456 | 0.1017 | 0.2262 |
| betting_mixture | 0.3659 | 0.2270 | 0.1599 | 0.1132 | 0.2473 |
| bentkus_diff | 0.3999 | 0.2273 | 0.1454 | 0.0941 | 0.2329 |
| convex_order_diff | 0.3411 | 0.1935 | 0.1265 | n/a | 0.1984 |

## Expected slack E[upper - |p_a - p_b|] at (p_a, p_b) = (0.02, 0.5)

| method | n=(20,20) | n=(50,50) | n=(100,100) | n=(200,200) | n=(30,150) |
|---|---|---|---|---|---|
| ttest | 0.3153 | 0.1991 | 0.1425 | 0.1014 | 0.1325 |
| hoeffdings | 0.6620 | 0.4187 | 0.2960 | 0.2093 | 0.3911 |
| clopper_pearson | 0.2672 | 0.1799 | 0.1317 | 0.0958 | 0.1124 |
| bentkus | 0.2980 | 0.2035 | 0.1497 | 0.1094 | 0.1272 |
| betting_mixture | 0.3228 | 0.2279 | 0.1706 | 0.1265 | 0.1452 |
| bentkus_diff | 0.2992 | 0.2001 | 0.1424 | 0.1001 | 0.1463 |
| convex_order_diff | 0.2667 | 0.1801 | 0.1288 | n/a | 0.1268 |

## Cost (ms per constraint evaluation)

| method | ms/call |
|---|---|
| ttest | 0.14 |
| hoeffdings | 0.04 |
| clopper_pearson | 0.08 |
| bentkus | 4.58 |
| betting_mixture | 4.43 |
| bentkus_diff | 11.98 |
| convex_order_diff | 49.88 |
