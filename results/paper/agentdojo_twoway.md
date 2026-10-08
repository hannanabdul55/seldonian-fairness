# Two-way bounds on AgentDojo's crossed design (paper section 10.2)

`scripts/agentdojo_twoway.py`, registered in `.planning/paper-certification/R2_registration.md` before the run. 28 pipelines, 4,000 resampled tables per pipeline and scheme, 4,000 inner bootstrap draws, delta 0.05, seed 20261008. A cell is over its level above a miss of 0.0569.

## 1. Cells over / unresolved / at or under, by scheme

| bound | (a) user tasks resampled | (b) injection tasks resampled | (c) both | largest miss (a); (b); (c) |
|---|---|---|---|---|
| per-pair Clopper-Pearson | 27 / 1 / 0 | 28 / 0 / 0 | 28 / 0 / 0 | 0.272; 0.335; 0.362 |
| cluster bootstrap-t by user task | 0 / 3 / 25 | 21 / 0 / 7 | 24 / 0 / 4 | 0.051; 0.262; 0.269 |
| cluster bootstrap-t by injection task | 3 / 0 / 25 | 5 / 0 / 23 | 25 / 0 / 3 | 0.165; 0.098; 0.189 |
| multiway variance, t quantile (registered) | 5 / 0 / 23 | 8 / 3 / 17 | 26 / 1 / 1 | 0.136; 0.175; 0.179 |
| pigeonhole bootstrap-t (registered) | 0 / 0 / 28 | 0 / 0 / 28 | 0 / 0 / 28 | 0.004; 0.037; 0.042 |
| one-way margins in quadrature (registered) | 0 / 0 / 28 | 0 / 0 / 28 | 1 / 1 / 26 | 0.015; 0.046; 0.058 |

## 2. Width on the published tables

Limits from 200,000 inner draws. The margin is the limit minus the rate; the ratio is to the margin of the bound clustered by user task.

| bound | median margin ratio | range | pipelines certifying 5% | no limit returned |
|---|---|---|---|---|
| per-pair Clopper-Pearson | 0.58 | 0.25 to 0.87 | 5 (Meta-SecAlign-70B, Meta-SecAlign-70B-repeat_user_prompt, claude-3-5-sonnet-20241022, command-r, gemini-1.5-flash-002) | 0 |
| cluster bootstrap-t by user task | 1.00 | 1.00 to 1.00 | 1 (claude-3-5-sonnet-20241022) | 0 |
| cluster bootstrap-t by injection task | 1.71 | 0.34 to 4.61 | 3 (Meta-SecAlign-70B, Meta-SecAlign-70B-repeat_user_prompt, claude-3-5-sonnet-20241022) | 0 |
| multiway variance, t quantile | 1.56 | 0.55 to 3.53 | 3 (Meta-SecAlign-70B, Meta-SecAlign-70B-repeat_user_prompt, claude-3-5-sonnet-20241022) | 0 |
| pigeonhole bootstrap-t | 2.05 | 1.17 to 87.49 | 0 | 1 |
| one-way margins in quadrature | 1.98 | 1.06 to 4.72 | 1 (claude-3-5-sonnet-20241022) | 0 |

## 3. Per pipeline

Miss under scheme (c), and the limit on the published table. **Bold** is over the level.

| pipeline | pairs | rate | cgm_t miss (c) | pig_t miss (c) | quad miss (c) | t_user limit | cgm_t limit | pig_t limit | quad limit |
|---|---|---|---|---|---|---|---|---|---|
| Meta-SecAlign-70B | 949 | 0.022 | **0.154** | 0.000 | 0.001 | 0.061 | 0.043 | 0.091 | 0.063 |
| Meta-SecAlign-70B-repeat_user_prompt | 949 | 0.021 | **0.152** | 0.000 | 0.002 | 0.057 | 0.041 | 0.089 | 0.059 |
| claude-3-5-sonnet-20240620 | 629 | 0.339 | **0.071** | 0.033 | 0.033 | 0.385 | 0.462 | 0.477 | 0.477 |
| claude-3-5-sonnet-20241022 | 629 | 0.011 | **0.166** | 0.000 | 0.000 | 0.022 | 0.022 | 1.000 | 0.038 |
| claude-3-7-sonnet-20250219 | 949 | 0.050 | **0.143** | 0.015 | 0.048 | 0.070 | 0.082 | 0.107 | 0.108 |
| claude-3-haiku-20240307 | 629 | 0.091 | **0.129** | 0.037 | **0.058** | 0.122 | 0.140 | 0.165 | 0.161 |
| claude-3-opus-20240229 | 629 | 0.113 | **0.122** | 0.037 | 0.051? | 0.146 | 0.174 | 0.208 | 0.207 |
| claude-3-sonnet-20240229 | 629 | 0.267 | **0.060** | 0.032 | 0.038 | 0.317 | 0.354 | 0.363 | 0.361 |
| command-r | 629 | 0.033 | **0.144** | 0.000 | 0.013 | 0.063 | 0.056 | 0.078 | 0.069 |
| command-r-plus | 629 | 0.045 | **0.143** | 0.000 | 0.020 | 0.090 | 0.074 | 0.098 | 0.093 |
| gemini-1.5-flash-001 | 629 | 0.122 | **0.094** | 0.036 | 0.042 | 0.160 | 0.179 | 0.199 | 0.197 |
| gemini-1.5-flash-002 | 629 | 0.035 | **0.179** | 0.000 | 0.025 | 0.052 | 0.066 | 0.123 | 0.114 |
| gemini-1.5-pro-001 | 629 | 0.286 | **0.065** | 0.036 | 0.042 | 0.341 | 0.381 | 0.392 | 0.390 |
| gemini-1.5-pro-002 | 629 | 0.170 | **0.091** | 0.035 | 0.040 | 0.217 | 0.244 | 0.266 | 0.263 |
| gemini-2.0-flash-001 | 949 | 0.141 | **0.092** | 0.036 | 0.042 | 0.188 | 0.211 | 0.233 | 0.227 |
| gemini-2.0-flash-exp | 629 | 0.170 | **0.082** | 0.036 | 0.043 | 0.221 | 0.239 | 0.255 | 0.252 |
| gpt-3.5-turbo-0125 | 629 | 0.103 | **0.096** | 0.016 | 0.021 | 0.145 | 0.158 | 0.181 | 0.179 |
| gpt-4-0125-preview | 629 | 0.563 | 0.040 | 0.032 | 0.032 | 0.604 | 0.708 | 0.701 | 0.702 |
| gpt-4-turbo-2024-04-09 | 629 | 0.286 | **0.077** | 0.041 | 0.047 | 0.343 | 0.397 | 0.416 | 0.410 |
| gpt-4o-2024-05-13 | 629 | 0.477 | 0.051? | 0.033 | 0.036 | 0.537 | 0.600 | 0.604 | 0.604 |
| gpt-4o-2024-05-13-repeat_user_prompt | 629 | 0.278 | **0.075** | 0.035 | 0.038 | 0.329 | 0.381 | 0.396 | 0.393 |
| gpt-4o-2024-05-13-spotlighting_with_delimiting | 629 | 0.417 | **0.063** | 0.040 | 0.046 | 0.472 | 0.545 | 0.551 | 0.548 |
| gpt-4o-2024-05-13-tool_filter | 629 | 0.068 | **0.097** | 0.009 | 0.030 | 0.092 | 0.102 | 0.115 | 0.112 |
| gpt-4o-2024-05-13-transformers_pi_detector | 629 | 0.079 | **0.175** | 0.004 | 0.046 | 0.108 | 0.141 | 0.215 | 0.199 |
| gpt-4o-mini-2024-07-18 | 629 | 0.272 | **0.072** | 0.040 | 0.043 | 0.333 | 0.358 | 0.369 | 0.367 |
| meta-llama_Llama-3-70b-chat-hf | 629 | 0.256 | **0.072** | 0.037 | 0.033 | 0.304 | 0.340 | 0.355 | 0.354 |
| meta-llama_Llama-3.3-70B-Instruct | 949 | 0.231 | **0.075** | 0.042 | 0.050 | 0.284 | 0.313 | 0.324 | 0.321 |
| meta-llama_Llama-3.3-70B-Instruct-repeat_user_prompt | 797 | 0.090 | **0.107** | 0.024 | 0.044 | 0.131 | 0.136 | 0.153 | 0.148 |

## 4. Spread over bootstrap seeds on the published tables

Largest range of a limit over 30 seeds at 4,000 inner draws: cluster bootstrap-t by user task 0.061; cluster bootstrap-t by injection task 0.017; pigeonhole bootstrap-t 0.026; one-way margins in quadrature 0.059.
