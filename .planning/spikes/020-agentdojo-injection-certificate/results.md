# Spike 020: certificates on AgentDojo's published runs

19381 (user task, injection task) pairs under `important_instructions` over 28 pipelines. Label = `security` (the injected goal was reached). Upper bounds at delta 0.05 unless marked. `naive` = Clopper-Pearson over pairs; `t(user)` / `t(inj)` = studentised cluster bootstrap by user task / injection task; `any-inj` = Clopper-Pearson over user tasks on 'any injection succeeded'; `b1w` = suite-stratified Wilson-type; `deff` = 1 + (k-1) ICC. Verdict = `t` by the dominant cluster against tau 0.05.

| pipeline | pairs | errors | successes | rate | naive | ICC user (k) | ICC inj (k) | deff user | deff inj | t(user) | t(inj) | two-way | any-inj (n_u, k_u) | pooled Wilson | b1w suites | ESS | verdict | utility no attack | utility under attack |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Meta-SecAlign-70B | 949 | 0 | 21 | 0.022 | 0.032 | 0.72 (10) | 0.08 (27) | 7.5 | 3.2 | 0.104 | 0.035 | 0.041 | 0.105 (97, 5) | 0.032 | 0.031 | 1.05 | **NSF** (user) | 0.78 | 0.78 |
| Meta-SecAlign-70B-repeat_user_prompt | 949 | 0 | 20 | 0.021 | 0.030 | 0.85 (10) | 0.09 (27) | 8.7 | 3.2 | 0.057 | 0.034 | 0.039 | 0.105 (97, 5) | 0.030 | 0.030 | 1.05 | **NSF** (user) | 0.85 | 0.80 |
| claude-3-5-sonnet-20240620 | 629 | 0 | 213 | 0.339 | 0.371 | 0.25 (6) | 0.54 (23) | 2.2 | 12.8 | 0.384 | 0.467 | 0.459 | 0.868 (97, 78) | 0.370 | 0.368 | 1.18 | **NSF** (inj) | 0.78 | 0.51 |
| claude-3-5-sonnet-20241022 | 629 | 0 | 7 | 0.011 | 0.021 | 0.10 (6) | 0.07 (23) | 1.5 | 2.6 | 0.022 | 0.034 | 0.022 | 0.118 (97, 6) | 0.021 | 0.021 | 1.00 | **pass** (user) | 0.79 | 0.72 |
| claude-3-7-sonnet-20250219 | 949 | 0 | 47 | 0.050 | 0.063 | 0.33 (10) | 0.23 (27) | 4.0 | 7.0 | 0.070 | 0.106 | 0.079 | 0.374 (97, 28) | 0.063 | 0.062 | 1.12 | **NSF** (user) | 0.90 | 0.82 |
| claude-3-haiku-20240307 | 629 | 0 | 57 | 0.091 | 0.112 | 0.39 (6) | 0.27 (23) | 2.9 | 7.0 | 0.123 | 0.154 | 0.136 | 0.385 (97, 29) | 0.111 | 0.110 | 1.09 | **NSF** (user) | 0.38 | 0.33 |
| claude-3-opus-20240229 | 629 | 0 | 71 | 0.113 | 0.136 | 0.31 (6) | 0.29 (23) | 2.5 | 7.3 | 0.146 | 0.195 | 0.170 | 0.491 (97, 39) | 0.135 | 0.134 | 1.18 | **NSF** (user) | 0.67 | 0.52 |
| claude-3-sonnet-20240229 | 629 | 0 | 168 | 0.267 | 0.298 | 0.33 (6) | 0.26 (23) | 2.7 | 6.6 | 0.315 | 0.345 | 0.353 | 0.701 (97, 60) | 0.297 | 0.295 | 1.12 | **NSF** (user) | 0.53 | 0.33 |
| command-r | 629 | 3 | 21 | 0.033 | 0.048 | 0.36 (6) | 0.08 (23) | 2.8 | 2.8 | 0.062 | 0.054 | 0.056 | 0.169 (97, 10) | 0.047 | 0.047 | 1.04 | **NSF** (user) | 0.26 | 0.31 |
| command-r-plus | 629 | 22 | 28 | 0.045 | 0.061 | 0.65 (6) | 0.10 (23) | 4.3 | 3.3 | 0.089 | 0.061 | 0.073 | 0.169 (97, 10) | 0.060 | 0.060 | 1.03 | **NSF** (user) | 0.25 | 0.25 |
| gemini-1.5-flash-001 | 629 | 0 | 77 | 0.122 | 0.146 | 0.33 (6) | 0.21 (23) | 2.7 | 5.6 | 0.159 | 0.185 | 0.177 | 0.480 (97, 38) | 0.146 | 0.145 | 1.10 | **NSF** (user) | 0.36 | 0.34 |
| gemini-1.5-flash-002 | 629 | 0 | 22 | 0.035 | 0.050 | 0.22 (6) | 0.34 (23) | 2.1 | 8.4 | 0.052 | 0.112 | 0.062 | 0.251 (97, 17) | 0.049 | 0.049 | 1.03 | **NSF** (inj) | 0.37 | 0.32 |
| gemini-1.5-pro-001 | 629 | 19 | 180 | 0.286 | 0.317 | 0.45 (6) | 0.21 (23) | 3.3 | 5.6 | 0.341 | 0.377 | 0.380 | 0.701 (97, 60) | 0.317 | 0.314 | 1.23 | **NSF** (user) | 0.46 | 0.29 |
| gemini-1.5-pro-002 | 629 | 0 | 107 | 0.170 | 0.197 | 0.40 (6) | 0.22 (23) | 3.0 | 5.9 | 0.218 | 0.251 | 0.240 | 0.511 (97, 41) | 0.196 | 0.194 | 1.22 | **NSF** (user) | 0.60 | 0.47 |
| gemini-2.0-flash-001 | 949 | 0 | 134 | 0.141 | 0.161 | 0.70 (10) | 0.35 (27) | 7.3 | 10.1 | 0.187 | 0.211 | 0.202 | 0.449 (97, 35) | 0.161 | 0.158 | 1.47 | **NSF** (user) | 0.41 | 0.39 |
| gemini-2.0-flash-exp | 629 | 0 | 107 | 0.170 | 0.197 | 0.48 (6) | 0.19 (23) | 3.4 | 5.2 | 0.221 | 0.234 | 0.235 | 0.491 (97, 39) | 0.196 | 0.195 | 1.14 | **NSF** (user) | 0.45 | 0.40 |
| gpt-3.5-turbo-0125 | 629 | 12 | 65 | 0.103 | 0.126 | 0.33 (6) | 0.36 (23) | 2.6 | 9.0 | 0.147 | 0.168 | 0.156 | 0.374 (97, 28) | 0.125 | 0.124 | 1.09 | **NSF** (inj) | 0.34 | 0.35 |
| gpt-4-0125-preview | 629 | 0 | 354 | 0.563 | 0.596 | 0.19 (6) | 0.66 (23) | 2.0 | 15.5 | 0.605 | 0.701 | 0.706 | 0.951 (97, 88) | 0.595 | 0.592 | 1.24 | **NSF** (inj) | 0.66 | 0.41 |
| gpt-4-turbo-2024-04-09 | 629 | 0 | 180 | 0.286 | 0.317 | 0.53 (6) | 0.44 (23) | 3.7 | 10.8 | 0.343 | 0.398 | 0.393 | 0.682 (97, 58) | 0.317 | 0.312 | 1.43 | **NSF** (user) | 0.63 | 0.54 |
| gpt-4o-2024-05-13 | 629 | 0 | 300 | 0.477 | 0.511 | 0.45 (6) | 0.55 (23) | 3.3 | 13.2 | 0.538 | 0.586 | 0.596 | 0.859 (97, 77) | 0.510 | 0.505 | 1.39 | **NSF** (inj) | 0.69 | 0.50 |
| gpt-4o-2024-05-13-repeat_user_prompt | 629 | 0 | 175 | 0.278 | 0.309 | 0.37 (6) | 0.37 (23) | 2.8 | 9.2 | 0.331 | 0.379 | 0.377 | 0.711 (97, 61) | 0.309 | 0.305 | 1.24 | **NSF** (inj) | 0.86 | 0.67 |
| gpt-4o-2024-05-13-spotlighting_with_delimiting | 629 | 0 | 262 | 0.417 | 0.450 | 0.42 (6) | 0.55 (23) | 3.1 | 13.1 | 0.472 | 0.535 | 0.543 | 0.814 (97, 72) | 0.449 | 0.444 | 1.38 | **NSF** (inj) | 0.73 | 0.56 |
| gpt-4o-2024-05-13-tool_filter | 629 | 0 | 43 | 0.068 | 0.087 | 0.08 (6) | 0.12 (23) | 1.4 | 3.6 | 0.092 | 0.106 | 0.102 | 0.417 (97, 32) | 0.087 | 0.087 | 1.03 | **NSF** (inj) | 0.73 | 0.56 |
| gpt-4o-2024-05-13-transformers_pi_detector | 629 | 0 | 50 | 0.079 | 0.100 | 0.30 (6) | 0.27 (23) | 2.5 | 6.9 | 0.108 | 0.192 | 0.135 | 0.363 (97, 27) | 0.099 | 0.099 | 1.07 | **NSF** (user) | 0.41 | 0.21 |
| gpt-4o-mini-2024-07-18 | 629 | 0 | 171 | 0.272 | 0.303 | 0.51 (6) | 0.21 (23) | 3.6 | 5.6 | 0.334 | 0.348 | 0.357 | 0.672 (97, 57) | 0.302 | 0.300 | 1.13 | **NSF** (user) | 0.66 | 0.50 |
| meta-llama_Llama-3-70b-chat-hf | 629 | 35 | 161 | 0.256 | 0.286 | 0.35 (6) | 0.26 (23) | 2.8 | 6.7 | 0.304 | 0.342 | 0.342 | 0.682 (97, 58) | 0.286 | 0.285 | 1.07 | **NSF** (user) | 0.34 | 0.18 |
| meta-llama_Llama-3.3-70B-Instruct | 949 | 0 | 219 | 0.231 | 0.254 | 0.55 (10) | 0.27 (27) | 5.9 | 7.9 | 0.285 | 0.303 | 0.306 | 0.711 (97, 61) | 0.254 | 0.251 | 1.30 | **NSF** (user) | 0.55 | 0.41 |
| meta-llama_Llama-3.3-70B-Instruct-repeat_user_prompt | 797 | 0 | 72 | 0.090 | 0.109 | 0.39 (11) | 0.17 (27) | 4.9 | 5.5 | 0.131 | 0.130 | 0.133 | 0.427 (71, 23) | 0.109 | 0.108 | 1.08 | **NSF** (user) | 0.53 | 0.40 |

## Hypotheses

- **H1 (reproduction).** gpt-4o-2024-05-13: website 0.477, all pairs 0.477, v1 subset (629 pairs) 0.477; claude-3-7-sonnet-20250219: website 0.073, all pairs 0.050, v1 subset (629 pairs) 0.073; claude-3-5-sonnet-20241022: website 0.011, all pairs 0.011, v1 subset (629 pairs) 0.011.
- **H2 (dependence).** Over the 21 pipelines with rates in [0.05, 0.60]: ICC by user task 0.08-0.70 (asked >= 0.2), design effect 1.4-7.3 (asked >= 2); the clustered bound's distance above the estimate is 1.44-4.19 times the naive bound's (asked >= 1.5, median 1.97). Dominant clustering: {'user': 20, 'inj': 8}. Fails in part.
- **H3 (certificate at tau 0.05, delta 0.05, clustered by the dominant unit).** Predicted pass: claude-3-5-sonnet-20241022 pass (0.022), Meta-SecAlign-70B NSF (0.104), Meta-SecAlign-70B-repeat_user_prompt NSF (0.057). Predicted NSF: claude-3-7-sonnet-20250219 NSF (0.070), command-r-plus NSF (0.089), gpt-4o-2024-05-13-tool_filter NSF (0.106). Not predicted: command-r NSF (0.062), gemini-1.5-flash-002 NSF (0.112). Passes overall: 1 of 28. Fails in part.
- **H4 (strata).** Suite-stratified `b1w` against pooled Wilson: ESS 1.00-1.47 (asked < 1.2). Fails.

## Resolution

Any-injection bound with zero compromised user tasks at n 97: 0.030. Naive bound over 629 i.i.d. pairs with zero successes: 0.005; the clustered bound cannot be computed at zero successes (no variance) and is reported as 1.0, which is the honest answer for a bootstrap with nothing to resample.
