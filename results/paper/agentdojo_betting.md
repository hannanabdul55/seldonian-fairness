# AgentDojo: an exact betting bound and suite-blocked resampling (paper O4)

`scripts/agentdojo_betting.py`. 28 pipelines, 4,000 replications per pipeline and scheme, 4,000 inner bootstrap draws, delta 0.05, seed 20261014 (per cell: crc32 of pipeline and scheme). A cell is over its level above a miss of 0.0569. A NaN bound counts as a miss. Truth: the observed table's pair rate; `bet_a1@uavg` scores the same bound against the observed user-task-averaged rate, the quantity it bounds.

## Registered predictions

- **Z1** the betting bound (a1) against the user-task average is over in no pipeline under (a) or (a_s): **kept**
- **Z2** the betting bound (a2) against the pair rate is over in no pipeline under (a_s): **kept**
- **Z3** the cluster bootstrap-t by user task is over in no pipeline under (a_s): **kept**
- **Z4** the cluster bootstrap-t by user task is over in at least 15 of 28 pipelines under (b_s): **refuted**
- **Z5** the quadrature bound is over in at most 2 of 28 pipelines under each of (a_s), (b_s), (c_s): **kept**
- **Z6** on the real tables the betting bound (a1) certifies 5% for at most one pipeline, and (a2) for none: **kept**

## 1. Cells over / unresolved / at or under, by arm and scheme

| arm | (a) | (b) | (c) | (a_s) | (b_s) | (c_s) | largest miss a; b; c; a_s; b_s; c_s |
|---|---|---|---|---|---|---|---|
| betting mixture on user-task rates (a1), truth: pair rate | 0 / 0 / 28 | 2 / 0 / 26 | 3 / 3 / 22 | 0 / 0 / 28 | 1 / 0 / 27 | 2 / 0 / 26 | 0.018; 0.077; 0.104; 0.009; 0.064; 0.069 |
| betting mixture on user-task rates (a1), truth: user-task-averaged rate | 0 / 0 / 28 | 4 / 0 / 24 | 7 / 1 / 20 | 0 / 0 / 28 | 3 / 0 / 25 | 4 / 1 / 23 | 0.007; 0.101; 0.131; 0.005; 0.082; 0.102 |
| betting mixture, pair rate (a2) | 0 / 0 / 28 | 4 / 2 / 22 | 8 / 3 / 17 | 0 / 0 / 28 | 2 / 0 / 26 | 2 / 0 / 26 | 0.006; 0.125; 0.130; 0.003; 0.098; 0.105 |
| cluster bootstrap-t by user task (issued) | 0 / 2 / 26 | 21 / 0 / 7 | 24 / 0 / 4 | 0 / 0 / 28 | 11 / 1 / 16 | 19 / 2 / 7 | 0.052; 0.257; 0.273; 0.042; 0.231; 0.249 |
| two clustered margins in quadrature | 0 / 0 / 28 | 0 / 0 / 28 | 0 / 1 / 27 | 0 / 0 / 28 | 0 / 0 / 28 | 0 / 0 / 28 | 0.017; 0.046; 0.051; 0.016; 0.037; 0.040 |
| larger of the two clustered bounds | 0 / 0 / 28 | 4 / 1 / 23 | 17 / 1 / 10 | 0 / 0 / 28 | 1 / 0 / 27 | 1 / 1 / 26 | 0.029; 0.067; 0.087; 0.029; 0.067; 0.062 |
| pigeonhole bootstrap-t (extra; returned by all_bounds) | 0 / 0 / 28 | 0 / 0 / 28 | 0 / 0 / 28 | 0 / 0 / 28 | 0 / 0 / 28 | 0 / 0 / 28 | 0.006; 0.030; 0.041; 0.003; 0.012; 0.020 |
| multiway variance, t quantile (extra; returned by all_bounds) | 5 / 0 / 23 | 8 / 2 / 18 | 25 / 2 / 1 | 5 / 0 / 23 | 4 / 0 / 24 | 12 / 2 / 14 | 0.123; 0.175; 0.202; 0.122; 0.154; 0.173 |

Schemes: (a) user tasks redrawn, injection tasks fixed; (b) injection tasks redrawn, user tasks fixed; (c) both redrawn; (a_s) user tasks redrawn within suite; (b_s) injection tasks redrawn within suite; (c_s) both redrawn within suite.

## 2. Suites lost by unblocked redraws

Share of replications in which a whole suite vanished from the redrawn table.

| scheme | largest share | mean share | pipelines with any |
|---|---|---|---|
| (a) | 0.0000 | 0.0000 | 0 |
| (b) | 0.0075 | 0.0050 | 28 |
| (c) | 0.0085 | 0.0053 | 27 |
| (a_s) | 0.0000 | 0.0000 | 0 |
| (b_s) | 0.0000 | 0.0000 | 0 |
| (c_s) | 0.0000 | 0.0000 | 0 |

## 3. Bounds on the published tables (delta 0.05)

`bet_a1` bounds the user-task-averaged rate, `bet_a2` the pair rate. The issued bound is the recheck's cluster bootstrap-t by user task (paper seed, 4,000 draws; and 200,000 draws, Table 7), read from `agentdojo_recheck.json`, not recomputed.

| pipeline | pairs | user tasks | suites | rate | user-task avg | n_max / n_bar | bet_a1 | bet_a2 | issued t_user | t_user 200k |
|---|---|---|---|---|---|---|---|---|---|---|
| Meta-SecAlign-70B | 949 | 97 | 4 | 0.022 | 0.030 | 14 / 9.78 | 0.092 | 0.078 | 0.104 | 0.062 |
| Meta-SecAlign-70B-repeat_user_prompt | 949 | 97 | 4 | 0.021 | 0.029 | 14 / 9.78 | 0.090 | 0.076 | 0.057 | 0.058 |
| claude-3-5-sonnet-20240620 | 629 | 97 | 4 | 0.339 | 0.371 | 9 / 6.48 | 0.451 | 0.409 | 0.384 | 0.385 |
| claude-3-5-sonnet-20241022 | 629 | 97 | 4 | 0.011 | 0.012 | 9 / 6.48 | 0.048 | 0.060 | 0.022 | 0.022 |
| claude-3-7-sonnet-20250219 | 949 | 97 | 4 | 0.050 | 0.071 | 14 / 9.78 | 0.119 | 0.105 | 0.070 | 0.070 |
| claude-3-haiku-20240307 | 629 | 97 | 4 | 0.091 | 0.098 | 9 / 6.48 | 0.157 | 0.150 | 0.123 | 0.123 |
| claude-3-opus-20240229 | 629 | 97 | 4 | 0.113 | 0.128 | 9 / 6.48 | 0.192 | 0.174 | 0.146 | 0.146 |
| claude-3-sonnet-20240229 | 629 | 97 | 4 | 0.267 | 0.275 | 9 / 6.48 | 0.362 | 0.353 | 0.315 | 0.318 |
| command-r | 629 | 97 | 4 | 0.033 | 0.029 | 9 / 6.48 | 0.071 | 0.087 | 0.062 | 0.063 |
| command-r-plus | 629 | 97 | 4 | 0.045 | 0.044 | 9 / 6.48 | 0.112 | 0.108 | 0.089 | 0.090 |
| gemini-1.5-flash-001 | 629 | 97 | 4 | 0.122 | 0.127 | 9 / 6.48 | 0.188 | 0.189 | 0.159 | 0.160 |
| gemini-1.5-flash-002 | 629 | 97 | 4 | 0.035 | 0.038 | 9 / 6.48 | 0.078 | 0.084 | 0.052 | 0.052 |
| gemini-1.5-pro-001 | 629 | 97 | 4 | 0.286 | 0.300 | 9 / 6.48 | 0.397 | 0.375 | 0.341 | 0.341 |
| gemini-1.5-pro-002 | 629 | 97 | 4 | 0.170 | 0.184 | 9 / 6.48 | 0.267 | 0.247 | 0.218 | 0.218 |
| gemini-2.0-flash-001 | 949 | 97 | 4 | 0.141 | 0.219 | 14 / 9.78 | 0.322 | 0.211 | 0.187 | 0.188 |
| gemini-2.0-flash-exp | 629 | 97 | 4 | 0.170 | 0.178 | 9 / 6.48 | 0.264 | 0.253 | 0.221 | 0.221 |
| gpt-3.5-turbo-0125 | 629 | 97 | 4 | 0.103 | 0.094 | 9 / 6.48 | 0.159 | 0.174 | 0.147 | 0.145 |
| gpt-4-0125-preview | 629 | 97 | 4 | 0.563 | 0.569 | 9 / 6.48 | 0.648 | 0.641 | 0.605 | 0.604 |
| gpt-4-turbo-2024-04-09 | 629 | 97 | 4 | 0.286 | 0.305 | 9 / 6.48 | 0.406 | 0.378 | 0.343 | 0.343 |
| gpt-4o-2024-05-13 | 629 | 97 | 4 | 0.477 | 0.493 | 9 / 6.48 | 0.593 | 0.580 | 0.538 | 0.537 |
| gpt-4o-2024-05-13-repeat_user_prompt | 629 | 97 | 4 | 0.278 | 0.282 | 9 / 6.48 | 0.368 | 0.368 | 0.331 | 0.329 |
| gpt-4o-2024-05-13-spotlighting_with_delimiting | 629 | 97 | 4 | 0.417 | 0.427 | 9 / 6.48 | 0.519 | 0.515 | 0.472 | 0.472 |
| gpt-4o-2024-05-13-tool_filter | 629 | 97 | 4 | 0.068 | 0.066 | 9 / 6.48 | 0.107 | 0.122 | 0.092 | 0.092 |
| gpt-4o-2024-05-13-transformers_pi_detector | 629 | 97 | 4 | 0.079 | 0.090 | 9 / 6.48 | 0.143 | 0.136 | 0.108 | 0.109 |
| gpt-4o-mini-2024-07-18 | 629 | 97 | 4 | 0.272 | 0.282 | 9 / 6.48 | 0.379 | 0.378 | 0.334 | 0.333 |
| meta-llama_Llama-3-70b-chat-hf | 629 | 97 | 4 | 0.256 | 0.266 | 9 / 6.48 | 0.349 | 0.336 | 0.304 | 0.304 |
| meta-llama_Llama-3.3-70B-Instruct | 949 | 97 | 4 | 0.231 | 0.307 | 14 / 9.78 | 0.407 | 0.308 | 0.285 | 0.284 |
| meta-llama_Llama-3.3-70B-Instruct-repeat_user_prompt | 797 | 71 | 3 | 0.090 | 0.114 | 14 / 11.23 | 0.197 | 0.161 | 0.131 | 0.131 |

Pipelines certifying an attack success rate of at most 0.05:

- bet_a1: 1 (claude-3-5-sonnet-20241022)
- bet_a2: 0
- issued_t_user: 1 (claude-3-5-sonnet-20241022)
- issued_t_user_200k: 1 (claude-3-5-sonnet-20241022)
