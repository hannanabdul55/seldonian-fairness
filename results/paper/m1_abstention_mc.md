# M1: abstention and clustered Monte Carlo error (not registered)

`scripts/m1_abstention_mc.py`. Written after every result it reads was seen; no draw or bound is changed. A draw *returns no limit* when its upper limit is 1 or more (no certificate). As published, such a draw is not a miss; read conservatively it is. Counts are over / unresolved / at or under by the paper's rule (se from the draw count). Replayed rows reproduce the published miss rate of every cell exactly (asserted).

## 1. Abstention, by Table 4 row

| row | group | cells | largest share with no limit | cells with any | counts as published | counts, no limit = miss | largest miss, no limit = miss |
|---|---|---|---|---|---|---|---|
| 9 | 017_boot | 168 | 0.5537 (refusal raw 0, rate 0.013, n 225 of 4000, f01) | 36 | 0 / 13 / 155 | 10 / 13 / 145 | 0.5537 |
| 10 | ad_t_user_a | 28 | 0.1467 (Meta-SecAlign-70B) | 5 | 0 / 2 / 26 | 3 / 2 / 23 | 0.1600 |
| 11 | p14_a_sboot_mid_0.05 | 28 | 0.0616 (C3:refusal, n_s 100, step 200) | 16 | 0 / 0 / 28 | 4 / 0 / 24 | 0.0948 |
| 11 | p14_b_sboot | 28 | 0.5190 (refusal rubric 0, rate 0.013, n 225 of 4000, K=5) | 15 | 0 / 5 / 23 | 5 / 5 / 18 | 0.5288 |
| 12 | rep_sboot_0.05 | 28 | 0.0855 (C2:refusal, n_s 100, step 0) | 21 | 0 / 0 / 28 | 7 / 2 / 19 | 0.1151 |
| 12 | rep_sboot_0.1 | 28 | 0.0508 (C3:refusal, n_s 100, step 200) | 20 | 0 / 1 / 27 | 3 / 4 / 21 | 0.1201 |
| 14 | tw_pig_t_a | 28 | 0.8173 (claude-3-5-sonnet-20241022) | 6 | 0 / 0 / 28 | 3 / 0 / 25 | 0.8173 |
| 14 | tw_pig_t_b | 28 | 0.7023 (claude-3-5-sonnet-20241022) | 9 | 0 / 0 / 28 | 4 / 0 / 24 | 0.7023 |
| 14 | tw_pig_t_c | 28 | 0.7738 (claude-3-5-sonnet-20241022) | 11 | 0 / 0 / 28 | 7 / 0 / 21 | 0.7738 |
| 24 | ad_naive_a | 28 | 0.0000 (Meta-SecAlign-70B) | 0 | 27 / 0 / 1 | 27 / 0 / 1 | 0.2682 |
| 24 | ad_naive_b | 28 | 0.0000 (Meta-SecAlign-70B) | 0 | 28 / 0 / 0 | 28 / 0 / 0 | 0.3397 |
| 24 | ad_naive_c | 28 | 0.0000 (Meta-SecAlign-70B) | 0 | 28 / 0 / 0 | 28 / 0 / 0 | 0.3500 |
| 25 | ad_t_user_b | 28 | 0.2115 (claude-3-5-sonnet-20241022) | 6 | 21 / 0 / 7 | 24 / 0 / 4 | 0.2577 |
| 25 | ad_t_user_c | 28 | 0.3295 (claude-3-5-sonnet-20241022) | 8 | 24 / 0 / 4 | 28 / 0 / 0 | 0.3508 |
| 26 | ad_max_a | 28 | 0.2427 (claude-3-5-sonnet-20241022) | 6 | 0 / 0 / 28 | 3 / 0 / 25 | 0.2440 |
| 26 | ad_max_b | 28 | 0.3085 (claude-3-5-sonnet-20241022) | 7 | 4 / 0 / 24 | 8 / 0 / 20 | 0.3085 |
| 26 | ad_max_c | 28 | 0.4993 (claude-3-5-sonnet-20241022) | 10 | 16 / 3 / 9 | 21 / 4 / 3 | 0.4998 |
| 27 | ad_twoway_a | 28 | 0.0000 (Meta-SecAlign-70B) | 0 | 5 / 0 / 23 | 5 / 0 / 23 | 0.1745 |
| 27 | ad_twoway_b | 28 | 0.0000 (Meta-SecAlign-70B) | 0 | 14 / 1 / 13 | 14 / 1 / 13 | 0.1777 |
| 27 | ad_twoway_c | 28 | 0.0000 (Meta-SecAlign-70B) | 0 | 27 / 0 / 1 | 27 / 0 / 1 | 0.2777 |
| 30 | tw_cgm_t_a | 28 | 0.0063 (Meta-SecAlign-70B-repeat_user_prompt) | 3 | 5 / 0 / 23 | 5 / 0 / 23 | 0.1423 |
| 30 | tw_cgm_t_b | 28 | 0.0105 (claude-3-5-sonnet-20241022) | 2 | 8 / 3 / 17 | 8 / 3 / 17 | 0.1750 |
| 30 | tw_cgm_t_c | 28 | 0.0600 (claude-3-5-sonnet-20241022) | 7 | 26 / 1 / 1 | 26 / 1 / 1 | 0.2255 |
| 31 | tw_quad_a | 28 | 0.2487 (claude-3-5-sonnet-20241022) | 6 | 0 / 0 / 28 | 3 / 0 / 25 | 0.2487 |
| 31 | tw_quad_b | 28 | 0.2963 (claude-3-5-sonnet-20241022) | 6 | 0 / 0 / 28 | 4 / 0 / 24 | 0.2963 |
| 31 | tw_quad_c | 28 | 0.4935 (claude-3-5-sonnet-20241022) | 9 | 1 / 1 / 26 | 6 / 1 / 21 | 0.4935 |
| 40 | o4_bet_a1@uavg_a | 28 | 0.0000 (Meta-SecAlign-70B) | 0 | 0 / 0 / 28 | 0 / 0 / 28 | 0.0070 |
| 40 | o4_bet_a1@uavg_a_s | 28 | 0.0000 (Meta-SecAlign-70B) | 0 | 0 / 0 / 28 | 0 / 0 / 28 | 0.0050 |
| 40 | o4_bet_a2_a | 28 | 0.0000 (Meta-SecAlign-70B) | 0 | 0 / 0 / 28 | 0 / 0 / 28 | 0.0065 |
| 40 | o4_bet_a2_a_s | 28 | 0.0000 (Meta-SecAlign-70B) | 0 | 0 / 0 / 28 | 0 / 0 / 28 | 0.0030 |
| 41 | o4_t_user_a_s | 28 | 0.1465 (Meta-SecAlign-70B-repeat_user_prompt) | 5 | 0 / 0 / 28 | 3 / 0 / 25 | 0.1588 |
| 42 | o4_t_user_b_s | 28 | 0.1673 (claude-3-5-sonnet-20241022) | 5 | 11 / 1 / 16 | 14 / 1 / 13 | 0.2310 |
| 42 | o4_t_user_c_s | 28 | 0.2993 (claude-3-5-sonnet-20241022) | 8 | 19 / 2 / 7 | 23 / 2 / 3 | 0.3352 |
| 43 | o4_quad_a_s | 28 | 0.2107 (claude-3-5-sonnet-20241022) | 6 | 0 / 0 / 28 | 3 / 0 / 25 | 0.2107 |
| 43 | o4_quad_b_s | 28 | 0.2560 (claude-3-5-sonnet-20241022) | 7 | 0 / 0 / 28 | 3 / 0 / 25 | 0.2560 |
| 43 | o4_quad_c_s | 28 | 0.4600 (claude-3-5-sonnet-20241022) | 9 | 0 / 0 / 28 | 4 / 1 / 23 | 0.4600 |
| 49 | b6_t_user_P_a | 28 | 0.0635 (Meta-SecAlign-70B) | 4 | 0 / 0 / 28 | 2 / 0 / 26 | 0.0867 |
| 50 | b6_t_user_P_c | 28 | 0.0638 (Meta-SecAlign-70B) | 5 | 18 / 3 / 7 | 20 / 3 / 5 | 0.2467 |
| 51 | b6_bet_a1@uavg_P_a | 28 | 0.0000 (Meta-SecAlign-70B) | 0 | 0 / 0 / 28 | 0 / 0 / 28 | 0.0045 |
| 51 | b6_bet_a2_P_a | 28 | 0.0000 (Meta-SecAlign-70B) | 0 | 0 / 0 / 28 | 0 / 0 / 28 | 0.0032 |
| 52 | b6_pig_t_P_a | 28 | 0.1653 (Meta-SecAlign-70B-repeat_user_prompt) | 6 | 0 / 0 / 28 | 3 / 0 / 25 | 0.1653 |
| 52 | b6_pig_t_P_c | 28 | 0.2600 (claude-3-5-sonnet-20241022) | 8 | 0 / 0 / 28 | 3 / 0 / 25 | 0.2600 |
| 52 | b6_quad_P_a | 28 | 0.0643 (Meta-SecAlign-70B-repeat_user_prompt) | 5 | 0 / 0 / 28 | 2 / 0 / 26 | 0.0805 |
| 52 | b6_quad_P_c | 28 | 0.0640 (Meta-SecAlign-70B) | 5 | 0 / 0 / 28 | 2 / 1 / 25 | 0.0777 |
| - | p14_a_sboot_mid_0.1 | 28 | 0.0400 (C3:refusal, n_s 100, step 200) | 15 | 0 / 0 / 28 | 0 / 1 / 27 | 0.1008 |
| - | p14_a_sboot_rare_0.05 | 12 | 0.9304 (C3:unsafe, n_s 100, step 0) | 12 | 0 / 0 / 12 | 12 / 0 / 0 | 0.9314 |
| - | p14_a_sboot_rare_0.1 | 12 | 0.9040 (C3:unsafe, n_s 100, step 0) | 12 | 0 / 0 / 12 | 12 / 0 / 0 | 0.9118 |
| - | tw_t_user_a | 28 | 0.1573 (Meta-SecAlign-70B-repeat_user_prompt) | 5 | 0 / 3 / 25 | 3 / 3 / 22 | 0.1720 |
| - | tw_t_user_b | 28 | 0.2023 (claude-3-5-sonnet-20241022) | 6 | 21 / 0 / 7 | 24 / 0 / 4 | 0.2615 |
| - | tw_t_user_c | 28 | 0.3125 (claude-3-5-sonnet-20241022) | 9 | 24 / 0 / 4 | 28 / 0 / 0 | 0.3468 |

## 2. The sheet rows with plantings as clusters

20 plantings x 200 sheets a cell. se(binomial) treats the 4,000 sheets as independent; se(cluster) is the sd of the 20 per-planting miss rates over sqrt(20). Classified with the clustered se (never below the binomial within a planting).

| row | cell | miss | se binomial | se cluster | design effect | class, binomial | class, clustered |
|---|---|---|---|---|---|---|---|
| 8 | wording 0, planted rate 0.013, f01 | 0.0000 | 0.0034 | 0.0000 | 0.0 | at or under | at or under |
| 8 | wording 0, planted rate 0.013, logit | 0.0000 | 0.0034 | 0.0000 | 0.0 | at or under | at or under |
| 8 | wording 0, planted rate 0.05, f01 | 0.0238 | 0.0034 | 0.0023 | 0.5 | at or under | at or under |
| 8 | wording 0, planted rate 0.05, logit | 0.0247 | 0.0034 | 0.0027 | 0.6 | at or under | at or under |
| 8 | wording 0, planted rate 0.2, f01 | 0.0518 | 0.0034 | 0.0048 | 1.9 | unresolved | unresolved |
| 8 | wording 0, planted rate 0.2, logit | 0.0425 | 0.0034 | 0.0038 | 1.2 | at or under | at or under |
| 8 | wording 2, planted rate 0.013, f01 | 0.0000 | 0.0034 | 0.0000 | 0.0 | at or under | at or under |
| 8 | wording 2, planted rate 0.013, logit | 0.0000 | 0.0034 | 0.0000 | 0.0 | at or under | at or under |
| 8 | wording 2, planted rate 0.05, f01 | 0.0275 | 0.0034 | 0.0028 | 0.6 | at or under | at or under |
| 8 | wording 2, planted rate 0.05, logit | 0.0220 | 0.0034 | 0.0031 | 0.8 | at or under | at or under |
| 8 | wording 2, planted rate 0.2, f01 | 0.0595 | 0.0034 | 0.0040 | 1.3 | over | over |
| 8 | wording 2, planted rate 0.2, logit | 0.0485 | 0.0034 | 0.0039 | 1.3 | at or under | at or under |
| 8 | wording 4, planted rate 0.013, f01 | 0.0000 | 0.0034 | 0.0000 | 0.0 | at or under | at or under |
| 8 | wording 4, planted rate 0.013, logit | 0.0000 | 0.0034 | 0.0000 | 0.0 | at or under | at or under |
| 8 | wording 4, planted rate 0.05, f01 | 0.0175 | 0.0034 | 0.0012 | 0.1 | at or under | at or under |
| 8 | wording 4, planted rate 0.05, logit | 0.0195 | 0.0034 | 0.0019 | 0.3 | at or under | at or under |
| 8 | wording 4, planted rate 0.2, f01 | 0.0408 | 0.0034 | 0.0036 | 1.1 | at or under | at or under |
| 8 | wording 4, planted rate 0.2, logit | 0.0388 | 0.0034 | 0.0034 | 1.0 | at or under | at or under |
| 8 | summary | | | | median 0.9, max 1.9 | 1 / 1 / 16 | 1 / 1 / 16 |
| 28 | wording 0, planted rate 0.013, f01 | 0.0000 | 0.0034 | 0.0000 | 0.0 | at or under | at or under |
| 28 | wording 0, planted rate 0.013, logit | 0.0055 | 0.0034 | 0.0014 | 0.2 | at or under | at or under |
| 28 | wording 0, planted rate 0.05, f01 | 0.0000 | 0.0034 | 0.0000 | 0.0 | at or under | at or under |
| 28 | wording 0, planted rate 0.05, logit | 0.0000 | 0.0034 | 0.0000 | 0.0 | at or under | at or under |
| 28 | wording 0, planted rate 0.2, f01 | 0.0000 | 0.0034 | 0.0000 | 0.0 | at or under | at or under |
| 28 | wording 0, planted rate 0.2, logit | 0.0000 | 0.0034 | 0.0000 | 0.0 | at or under | at or under |
| 28 | wording 2, planted rate 0.013, f01 | 0.9838 | 0.0034 | 0.0028 | 0.6 | over | over |
| 28 | wording 2, planted rate 0.013, logit | 0.9892 | 0.0034 | 0.0033 | 0.9 | over | over |
| 28 | wording 2, planted rate 0.05, f01 | 0.7968 | 0.0034 | 0.0143 | 17.2 | over | over |
| 28 | wording 2, planted rate 0.05, logit | 0.5885 | 0.0034 | 0.0191 | 30.6 | over | over |
| 28 | wording 2, planted rate 0.2, f01 | 0.0033 | 0.0034 | 0.0010 | 0.1 | at or under | at or under |
| 28 | wording 2, planted rate 0.2, logit | 0.0000 | 0.0034 | 0.0000 | 0.0 | at or under | at or under |
| 28 | wording 4, planted rate 0.013, f01 | 0.0928 | 0.0034 | 0.0062 | 3.2 | over | over |
| 28 | wording 4, planted rate 0.013, logit | 0.6983 | 0.0034 | 0.0129 | 14.0 | over | over |
| 28 | wording 4, planted rate 0.05, f01 | 0.0030 | 0.0034 | 0.0008 | 0.0 | at or under | at or under |
| 28 | wording 4, planted rate 0.05, logit | 0.0927 | 0.0034 | 0.0092 | 7.1 | over | over |
| 28 | wording 4, planted rate 0.2, f01 | 0.0000 | 0.0034 | 0.0000 | 0.0 | at or under | at or under |
| 28 | wording 4, planted rate 0.2, logit | 0.0010 | 0.0034 | 0.0005 | 0.0 | at or under | at or under |
| 28 | summary | | | | median 0.9, max 30.6 | 7 / 0 / 11 | 7 / 0 / 11 |
