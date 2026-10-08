# The one-sided Wilson upper limit: exact miss probability

`scripts/wilson_exact.py`. Enumeration over binomial counts; nothing is simulated. The miss probability at a true rate p is P(U(X) < p). It is largest just above a limit U(k), where it equals P(X <= k) at p = U(k); the table gives those suprema. `b1w` of spike 013 at one stratum equals this limit at every count for n in 25, 100, 200, 400 (largest gap 0.00025, the step of its grid).

At zero positives the limit is z^2 / (n + z^2) and the miss just above it is (n / (n + z^2))^n, which is close to exp(-z^2) at any n and tends to it as n grows: 0.0668 at delta 0.05 and 0.1935 at delta 0.10. At the other end, just above the limit at n - 1 positives, the miss is 1 - U(n - 1)^n, the largest in the table: about 0.20 at delta 0.05 and 0.26 at delta 0.10.

| delta | n | limit at 0 positives | miss just above it | largest miss, any rate | at rate | counts k with a miss over delta above U(k) | highest such rate | Clopper-Pearson, largest miss |
|---|---|---|---|---|---|---|---|---|
| 0.05 | 59 | 0.0438 | 0.0710 | 0.2007 | 0.9962 | 59 of 59 | 0.996 | 0.0500 |
| 0.05 | 100 | 0.0263 | 0.0693 | 0.2004 | 0.9978 | 100 of 100 | 0.998 | 0.0500 |
| 0.05 | 149 | 0.0178 | 0.0685 | 0.2003 | 0.9985 | 149 of 149 | 0.999 | 0.0500 |
| 0.05 | 200 | 0.0133 | 0.0681 | 0.2002 | 0.9989 | 200 of 200 | 0.999 | 0.0500 |
| 0.05 | 299 | 0.0090 | 0.0677 | 0.2001 | 0.9993 | 299 of 299 | 0.999 | 0.0500 |
| 0.05 | 400 | 0.0067 | 0.0674 | 0.2001 | 0.9994 | 400 of 400 | 0.999 | 0.0500 |
| 0.05 | 1000 | 0.0027 | 0.0671 | 0.2000 | 0.9998 | 1000 of 1000 | 1.000 | 0.0500 |
| 0.1 | 59 | 0.0271 | 0.1979 | 0.2597 | 0.9949 | 59 of 59 | 0.995 | 0.1000 |
| 0.1 | 100 | 0.0162 | 0.1961 | 0.2592 | 0.9970 | 100 of 100 | 0.997 | 0.1000 |
| 0.1 | 149 | 0.0109 | 0.1953 | 0.2590 | 0.9980 | 149 of 149 | 0.998 | 0.1000 |
| 0.1 | 200 | 0.0081 | 0.1948 | 0.2589 | 0.9985 | 200 of 200 | 0.999 | 0.1000 |
| 0.1 | 299 | 0.0055 | 0.1944 | 0.2588 | 0.9990 | 299 of 299 | 0.999 | 0.1000 |
| 0.1 | 400 | 0.0041 | 0.1942 | 0.2587 | 0.9993 | 400 of 400 | 0.999 | 0.1000 |
| 0.1 | 1000 | 0.0016 | 0.1938 | 0.2586 | 0.9997 | 1000 of 1000 | 1.000 | 0.1000 |

## By band of the true rate

For true rates in each band: the mean miss probability over a uniform grid of rates / its supremum / the share of the band where it is over delta. The mean shows the systematic part: under delta at rates below one half and over it above, where the limit is too short.

| delta | n | 0.005-0.01 | 0.01-0.02 | 0.02-0.05 | 0.05-0.2 | 0.2-0.5 | 0.5-0.8 | 0.8-0.95 | 0.95-0.99 |
|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 100 | 0.000 / 0.000 / 0% | 0.000 / 0.000 / 0% | 0.029 / 0.069 / 20% | 0.042 / 0.063 / 26% | 0.048 / 0.061 / 40% | 0.052 / 0.068 / 57% | 0.056 / 0.085 / 69% | 0.064 / 0.121 / 71% |
| 0.05 | 200 | 0.000 / 0.000 / 0% | 0.025 / 0.068 / 15% | 0.036 / 0.063 / 18% | 0.045 / 0.059 / 27% | 0.049 / 0.058 / 41% | 0.051 / 0.062 / 58% | 0.055 / 0.074 / 69% | 0.061 / 0.100 / 74% |
| 0.05 | 400 | 0.025 / 0.067 / 15% | 0.035 / 0.063 / 20% | 0.041 / 0.059 / 20% | 0.046 / 0.056 / 27% | 0.049 / 0.055 / 41% | 0.051 / 0.059 / 58% | 0.053 / 0.068 / 70% | 0.058 / 0.087 / 74% |
| 0.1 | 100 | 0.000 / 0.000 / 0% | 0.062 / 0.196 / 38% | 0.088 / 0.156 / 37% | 0.095 / 0.138 / 40% | 0.099 / 0.122 / 47% | 0.101 / 0.125 / 52% | 0.104 / 0.146 / 57% | 0.109 / 0.181 / 58% |
| 0.1 | 200 | 0.060 / 0.195 / 37% | 0.090 / 0.155 / 42% | 0.090 / 0.143 / 34% | 0.097 / 0.126 / 41% | 0.099 / 0.115 / 46% | 0.101 / 0.118 / 53% | 0.103 / 0.133 / 57% | 0.107 / 0.167 / 58% |
| 0.1 | 400 | 0.090 / 0.154 / 42% | 0.090 / 0.142 / 37% | 0.094 / 0.129 / 38% | 0.098 / 0.118 / 41% | 0.099 / 0.110 / 46% | 0.101 / 0.112 / 53% | 0.102 / 0.123 / 57% | 0.105 / 0.148 / 58% |
