# Refusal sheet: analysis

Annotators: ah (220); adjudicated: 0

## Agreement on the shared items
1 annotator so far (40 of 40 shared items labelled); agreement needs a second.

## Rates by population (220 items with a gold label)

Estimate, then the design standard error, the prompt-clustered one, and a conservative 95% interval.

| population | n | refuses, strict (primary) | refuse-then-answer | refuses or hybrid (secondary) | answers |
|---|---|---|---|---|---|
| step 0 (reference) | 109 | 0.107 (se 0.016; clustered 0.029; 0.047 to 0.215) | 0.034 (se 0.013; clustered 0.014; 0.004 to 0.155) | 0.141 (se 0.014; clustered 0.033; 0.076 to 0.233) | 0.859 (se 0.014; clustered 0.033; 0.767 to 0.924) |
| step 200 (pushed policy) | 111 | 0.106 (se 0.017; clustered 0.031; 0.052 to 0.212) | 0.003 (se 0.002; clustered 0.003; 0.000 to 0.110) | 0.110 (se 0.017; clustered 0.031; 0.053 to 0.215) | 0.890 (se 0.017; clustered 0.031; 0.785 to 0.947) |
- refuses, strict (primary), pushed policy minus reference: -0.001 (se 0.023; clustered over shared prompts 0.039)
- refuse-then-answer, pushed policy minus reference: -0.030 (se 0.013; clustered over shared prompts 0.014)
- refuses or hybrid (secondary), pushed policy minus reference: -0.031 (se 0.022; clustered over shared prompts 0.042)

## Items without a gold label
{'unlabelled': 220}. Unsure and tied items, counted either way:
- step 0: strict refusal rate between 0.107 (none of them refusals) and 0.107 (all of them)
- step 200: strict refusal rate between 0.106 (none of them refusals) and 0.106 (all of them)

## Each annotator alone (strict refusal rate from that person's labels only)
- ah: step 0: 0.107 (se 0.016, n 109); step 200: 0.106 (se 0.017, n 111)

## Cut-off responses
A response that ends without closing punctuation probably hit the 128-token cap; a hybrid whose answer starts after the cap reads as a refusal, so the hybrid share is a lower estimate.
- step 0: 0.12 of strict refusals are cut off (weighted; 3 of 18 items)
- step 200: 0.23 of strict refusals are cut off (weighted; 4 of 18 items)

## Qwen3Guard-4B's refusal flag against the human label, by population

| population | human event | recall (95% interval) | false-alarm rate | precision | missed refusals in the pool: estimate, upper limit |
|---|---|---|---|---|---|
| step 0 | refuses, strict (primary) | 0.990 (0.438 to 1.000) | 0.081 | 0.593 | 4, 237 |
| step 0 | refuses or hybrid (secondary) | 0.985 (0.553 to 0.999) | 0.047 | 0.775 | 8, 240 |
| step 200 | refuses, strict (primary) | 0.984 (0.454 to 0.999) | 0.077 | 0.603 | 10, 368 |
| step 200 | refuses or hybrid (secondary) | 0.977 (0.457 to 0.998) | 0.075 | 0.616 | 15, 372 |

- step 0 raw counts: guard says refusal {'h': 6, 'r': 17, 'a': 7}; guard says not {'r': 1, 'a': 77, 'h': 1}
- step 200 raw counts: guard says refusal {'r': 16, 'a': 14, 'h': 1}; guard says not {'a': 77, 'r': 2, 'h': 1}
- refuse-then-answer items: 9 of 220 labelled; the guard flags 0.78 of them (raw share)

## rho^2 between the human label and the guard's feature (design-weighted)

What PPI++ can use: a judge is worth about 1 / (1 - rho^2) labels. The rule for spike 018: under 0.3 on the logit, an own judge has a target; at 0.5 the guard already halves the labels.

| population | strict: flag | strict: p | strict: logit | broad: flag | broad: p | broad: logit |
|---|---|---|---|---|---|---|
| step 0 | 0.54 | 0.55 | 0.61 | 0.73 | 0.72 | 0.73 |
| step 200 | 0.54 | 0.57 | 0.60 | 0.55 | 0.58 | 0.61 |
