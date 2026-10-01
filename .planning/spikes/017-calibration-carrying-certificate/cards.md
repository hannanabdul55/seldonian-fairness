# Spike 017: the three certificates

delta = 0.05, one-sided. Thresholds for illustration: brevity 0.45, refusal 0.25, harm 0.05.

## Brevity (verifiable): compiled to a word count

- Route: deterministic feature, Clopper-Pearson. Rate 0.404 on 500 responses, upper bound **0.441**; tau = 0.45 certified. Labels used: 0.
- The compiled judge instead: judged rate 0.048, bound 0.067. It would certify a rate of 0.07 for a quantity whose true rate is 0.404; its rho^2 with the count is 0.001.

## Refusal (semantic, mid rate): PPI++ on the judge's logit

225 of the 500 scored responses labelled (one seeded draw, the same for every row); gold = Qwen3Guard-4B's refusal field, standing in for a human. The pool's gold rate is 0.200. `labels alone` is Clopper-Pearson on the same 225.

| wording | judged rate (p > 0.5) | estimate | upper bound | labels alone | lam | rho^2 | worth in labels | tau 0.25 |
|---|---|---|---|---|---|---|---|---|
| rubric 0 | 0.042 | 0.204 | **0.247** | 0.239 | 0.86 | 0.41 | 285 | yes |
| rubric 1 | 0.008 | 0.197 | **0.242** | 0.239 | 0.96 | 0.28 | 254 | yes |
| rubric 2 | 0.054 | 0.195 | **0.239** | 0.239 | 0.28 | 0.06 | 275 | yes |
| rubric 3 | 0.040 | 0.202 | **0.241** | 0.239 | 0.78 | 0.42 | 344 | yes |
| rubric 4 | 0.090 | 0.198 | **0.238** | 0.239 | 0.68 | 0.43 | 325 | yes |
| rubric 5 | 0.060 | 0.197 | **0.236** | 0.239 | 0.70 | 0.40 | 354 | yes |
| bare sentence | 0.304 | 0.207 | **0.246** | 0.239 | 0.56 | 0.48 | 344 | yes |

Across the six rubric wordings the judged rate runs 0.008 to 0.090 (spread 0.082); the certified estimate runs 0.195 to 0.204 (spread 0.009).

## Harm (semantic, rare): the labels carry it, the judge does not

- 225 human labels, 3 positive, a stratified sample of 4,800 responses. Fewer than 10 positives, so the rule routes to the labels; because the sheet is stratified, the bound is the design-weighted `b1w` (013), which is approximate.
- Design-weighted harm rate 0.0134, upper bound **0.0393**; tau = 0.05 certified. Read as i.i.d. the same sheet would claim 0.0341.
- With the judge (weighted PPI++, 0/1 label): estimate 0.0136, lam -0.07: the judge gets no weight. (Its normal-limit bound, 0.0291, is not usable: that limit missed in about 0.20 of the planted sheets at this rate.)
- A carried calibration is refused: 3 human positives; 30 needed to bound recall.

## How far the threshold moves on the judge's scale

Plain PPI with the 0/1 judge: `true rate <= tau` is tested as `judged rate <= tau - rectifier - margin`, the rectifier being the mean of (gold - judge) on the labels.

| constraint | tau | rectifier | margin | judged-rate threshold | judged rate now |
|---|---|---|---|---|---|
| refusal, rubric 0 | 0.25 | +0.156 | 0.040 | 0.055 | 0.042 |
| refusal, bare sentence | 0.25 | -0.089 | 0.039 | 0.300 | 0.304 |
| brevity, compiled judge | 0.45 | +0.387 | 0.060 | 0.003 | 0.048 |
| harm, rubric 0 (design-weighted) | 0.05 | -0.220 | 0.053 | 0.217 | 0.230 |
