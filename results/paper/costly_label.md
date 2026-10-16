# O3: a costly label stratified on a cheap one (XSTest, 250 safe prompts)

Registered in `.planning/paper-certification/O3_registration.md`. Over: the miss exceeds delta by more than 2 Monte Carlo SE.

## Labels

| target | event | rate | class |
|---|---|---|---|
| gpt4 | broad | 0.084 | mid |
| gpt4 | strict | 0.064 | mid |
| llama2new | broad | 0.296 | mid |
| llama2new | strict | 0.140 | mid |
| llama2orig | broad | 0.596 | mid |
| llama2orig | strict | 0.380 | mid |
| mistralguard | broad | 0.188 | mid |
| mistralguard | strict | 0.096 | mid |
| mistralinstruct | broad | 0.016 | rare |
| mistralinstruct | strict | 0.008 | rare |

## Predictions

- **Y1** Wald-t on strata, labels at 10% or more (5), n_s 100, with replacement: over in no cell (20 cells): **refuted**
- **Y2** Wald-t on strata, mid-rate labels (8), without replacement: over in no cell (64 cells): **kept**
- **Y3** stratifier B, n_s 100, with replacement, delta 0.05: median gain over the mid-rate labels at least 1.1: **kept**
- **Y4** the same: no mid-rate label gains more than 2.0: **kept**
- **Y5** stratifier A gains more than B for at least 6 of the mid-rate labels: **refuted**
- **Y6** Clopper-Pearson on a random draw, rare labels (2), with replacement: over in no cell: **kept**

## Gains (Wald-t on strata against the pooled Wilson bound; with replacement)

| target | event | stratifier | n_s | delta | Wald-t | b1w |
|---|---|---|---|---|---|---|
| gpt4 | broad | B | 50 | 0.05 | 1.05 | 1.31 |
| gpt4 | broad | B | 50 | 0.1 | 0.95 | 1.37 |
| gpt4 | broad | B | 100 | 0.05 | 1.33 | 1.34 |
| gpt4 | broad | B | 100 | 0.1 | 1.22 | 1.37 |
| gpt4 | broad | A | 50 | 0.05 | 1.01 | 1.23 |
| gpt4 | broad | A | 50 | 0.1 | 0.91 | 1.27 |
| gpt4 | broad | A | 100 | 0.05 | 1.27 | 1.26 |
| gpt4 | broad | A | 100 | 0.1 | 1.17 | 1.29 |
| gpt4 | strict | B | 50 | 0.05 | 0.99 | 1.27 |
| gpt4 | strict | B | 50 | 0.1 | 0.87 | 1.33 |
| gpt4 | strict | B | 100 | 0.05 | 1.28 | 1.32 |
| gpt4 | strict | B | 100 | 0.1 | 1.16 | 1.36 |
| gpt4 | strict | A | 50 | 0.05 | 0.96 | 1.21 |
| gpt4 | strict | A | 50 | 0.1 | 0.84 | 1.25 |
| gpt4 | strict | A | 100 | 0.05 | 1.22 | 1.24 |
| gpt4 | strict | A | 100 | 0.1 | 1.10 | 1.27 |
| llama2new | broad | B | 50 | 0.05 | 1.06 | 1.31 |
| llama2new | broad | B | 50 | 0.1 | 1.05 | 1.33 |
| llama2new | broad | B | 100 | 0.05 | 1.20 | 1.26 |
| llama2new | broad | B | 100 | 0.1 | 1.19 | 1.27 |
| llama2new | broad | A | 50 | 0.05 | 1.14 | 1.46 |
| llama2new | broad | A | 50 | 0.1 | 1.14 | 1.49 |
| llama2new | broad | A | 100 | 0.05 | 1.31 | 1.40 |
| llama2new | broad | A | 100 | 0.1 | 1.30 | 1.41 |
| llama2new | strict | B | 50 | 0.05 | 1.01 | 1.16 |
| llama2new | strict | B | 50 | 0.1 | 0.95 | 1.17 |
| llama2new | strict | B | 100 | 0.05 | 1.16 | 1.12 |
| llama2new | strict | B | 100 | 0.1 | 1.10 | 1.13 |
| llama2new | strict | A | 50 | 0.05 | 1.00 | 1.15 |
| llama2new | strict | A | 50 | 0.1 | 0.94 | 1.16 |
| llama2new | strict | A | 100 | 0.05 | 1.17 | 1.13 |
| llama2new | strict | A | 100 | 0.1 | 1.10 | 1.13 |
| llama2orig | broad | B | 50 | 0.05 | 0.82 | 1.26 |
| llama2orig | broad | B | 50 | 0.1 | 0.87 | 1.28 |
| llama2orig | broad | B | 100 | 0.05 | 0.97 | 1.21 |
| llama2orig | broad | B | 100 | 0.1 | 0.99 | 1.21 |
| llama2orig | broad | A | 50 | 0.05 | 0.84 | 1.32 |
| llama2orig | broad | A | 50 | 0.1 | 0.89 | 1.34 |
| llama2orig | broad | A | 100 | 0.05 | 1.03 | 1.32 |
| llama2orig | broad | A | 100 | 0.1 | 1.06 | 1.32 |
| llama2orig | strict | B | 50 | 0.05 | 0.95 | 1.22 |
| llama2orig | strict | B | 50 | 0.1 | 0.96 | 1.22 |
| llama2orig | strict | B | 100 | 0.05 | 1.07 | 1.16 |
| llama2orig | strict | B | 100 | 0.1 | 1.07 | 1.16 |
| llama2orig | strict | A | 50 | 0.05 | 0.93 | 1.19 |
| llama2orig | strict | A | 50 | 0.1 | 0.94 | 1.20 |
| llama2orig | strict | A | 100 | 0.05 | 1.05 | 1.13 |
| llama2orig | strict | A | 100 | 0.1 | 1.05 | 1.14 |
| mistralguard | broad | B | 50 | 0.05 | 1.13 | 1.38 |
| mistralguard | broad | B | 50 | 0.1 | 1.08 | 1.41 |
| mistralguard | broad | B | 100 | 0.05 | 1.34 | 1.36 |
| mistralguard | broad | B | 100 | 0.1 | 1.29 | 1.38 |
| mistralguard | broad | A | 50 | 0.05 | 1.12 | 1.35 |
| mistralguard | broad | A | 50 | 0.1 | 1.08 | 1.38 |
| mistralguard | broad | A | 100 | 0.05 | 1.29 | 1.29 |
| mistralguard | broad | A | 100 | 0.1 | 1.24 | 1.30 |
| mistralguard | strict | B | 50 | 0.05 | 0.97 | 1.13 |
| mistralguard | strict | B | 50 | 0.1 | 0.88 | 1.16 |
| mistralguard | strict | B | 100 | 0.05 | 1.14 | 1.10 |
| mistralguard | strict | B | 100 | 0.1 | 1.05 | 1.10 |
| mistralguard | strict | A | 50 | 0.05 | 0.97 | 1.13 |
| mistralguard | strict | A | 50 | 0.1 | 0.89 | 1.16 |
| mistralguard | strict | A | 100 | 0.05 | 1.15 | 1.11 |
| mistralguard | strict | A | 100 | 0.1 | 1.06 | 1.11 |
| mistralinstruct | broad | B | 50 | 0.05 | 0.69 | 0.97 |
| mistralinstruct | broad | B | 50 | 0.1 | 0.51 | 0.98 |
| mistralinstruct | broad | B | 100 | 0.05 | 0.87 | 1.00 |
| mistralinstruct | broad | B | 100 | 0.1 | 0.68 | 1.01 |
| mistralinstruct | broad | A | 50 | 0.05 | 0.70 | 1.00 |
| mistralinstruct | broad | A | 50 | 0.1 | 0.53 | 1.02 |
| mistralinstruct | broad | A | 100 | 0.05 | 0.88 | 1.02 |
| mistralinstruct | broad | A | 100 | 0.1 | 0.69 | 1.03 |
| mistralinstruct | strict | B | 50 | 0.05 | 0.64 | 0.96 |
| mistralinstruct | strict | B | 50 | 0.1 | 0.45 | 0.97 |
| mistralinstruct | strict | B | 100 | 0.05 | 0.78 | 1.00 |
| mistralinstruct | strict | B | 100 | 0.1 | 0.57 | 1.01 |
| mistralinstruct | strict | A | 50 | 0.05 | 0.64 | 0.96 |
| mistralinstruct | strict | A | 50 | 0.1 | 0.45 | 0.98 |
| mistralinstruct | strict | A | 100 | 0.05 | 0.77 | 0.99 |
| mistralinstruct | strict | A | 100 | 0.1 | 0.56 | 1.00 |

## Cells

| target | event | stratifier | n_s | draw | delta | arm | R | miss | cell |
|---|---|---|---|---|---|---|---|---|---|
| gpt4 | strict | B | 50 | fixed | 0.05 | Wald-t b1 | 20000 | 0.0000 | at or under |
| gpt4 | strict | B | 50 | fixed | 0.05 | b1w | 20000 | 0.0119 | at or under |
| gpt4 | strict | B | 50 | fixed | 0.05 | pooled Wilson | 20000 | 0.0251 | at or under |
| gpt4 | strict | B | 50 | fixed | 0.05 | Clopper-Pearson | 20000 | 0.0251 | at or under |
| gpt4 | strict | B | 50 | fixed | 0.1 | Wald-t b1 | 20000 | 0.0119 | at or under |
| gpt4 | strict | B | 50 | fixed | 0.1 | b1w | 20000 | 0.0119 | at or under |
| gpt4 | strict | B | 50 | fixed | 0.1 | pooled Wilson | 20000 | 0.0251 | at or under |
| gpt4 | strict | B | 50 | fixed | 0.1 | Clopper-Pearson | 20000 | 0.0251 | at or under |
| gpt4 | strict | B | 50 | large | 0.05 | Wald-t b1 | 40000 | 0.0000 | at or under |
| gpt4 | strict | B | 50 | large | 0.05 | b1w | 40000 | 0.0210 | at or under |
| gpt4 | strict | B | 50 | large | 0.05 | pooled Wilson | 40000 | 0.0377 | at or under |
| gpt4 | strict | B | 50 | large | 0.05 | Clopper-Pearson | 40000 | 0.0377 | at or under |
| gpt4 | strict | B | 50 | large | 0.1 | Wald-t b1 | 40000 | 0.0210 | at or under |
| gpt4 | strict | B | 50 | large | 0.1 | b1w | 40000 | 0.0210 | at or under |
| gpt4 | strict | B | 50 | large | 0.1 | pooled Wilson | 40000 | 0.0377 | at or under |
| gpt4 | strict | B | 50 | large | 0.1 | Clopper-Pearson | 40000 | 0.0377 | at or under |
| gpt4 | strict | B | 100 | fixed | 0.05 | Wald-t b1 | 20000 | 0.0076 | at or under |
| gpt4 | strict | B | 100 | fixed | 0.05 | b1w | 20000 | 0.0076 | at or under |
| gpt4 | strict | B | 100 | fixed | 0.05 | pooled Wilson | 20000 | 0.0150 | at or under |
| gpt4 | strict | B | 100 | fixed | 0.05 | Clopper-Pearson | 20000 | 0.0150 | at or under |
| gpt4 | strict | B | 100 | fixed | 0.1 | Wald-t b1 | 20000 | 0.0303 | at or under |
| gpt4 | strict | B | 100 | fixed | 0.1 | b1w | 20000 | 0.0411 | at or under |
| gpt4 | strict | B | 100 | fixed | 0.1 | pooled Wilson | 20000 | 0.0587 | at or under |
| gpt4 | strict | B | 100 | fixed | 0.1 | Clopper-Pearson | 20000 | 0.0150 | at or under |
| gpt4 | strict | B | 100 | large | 0.05 | Wald-t b1 | 40000 | 0.0262 | at or under |
| gpt4 | strict | B | 100 | large | 0.05 | b1w | 40000 | 0.0262 | at or under |
| gpt4 | strict | B | 100 | large | 0.05 | pooled Wilson | 40000 | 0.0407 | at or under |
| gpt4 | strict | B | 100 | large | 0.05 | Clopper-Pearson | 40000 | 0.0407 | at or under |
| gpt4 | strict | B | 100 | large | 0.1 | Wald-t b1 | 40000 | 0.0629 | at or under |
| gpt4 | strict | B | 100 | large | 0.1 | b1w | 40000 | 0.0838 | at or under |
| gpt4 | strict | B | 100 | large | 0.1 | pooled Wilson | 40000 | 0.1091 | over |
| gpt4 | strict | B | 100 | large | 0.1 | Clopper-Pearson | 40000 | 0.0407 | at or under |
| gpt4 | strict | A | 50 | fixed | 0.05 | Wald-t b1 | 20000 | 0.0000 | at or under |
| gpt4 | strict | A | 50 | fixed | 0.05 | b1w | 20000 | 0.0157 | at or under |
| gpt4 | strict | A | 50 | fixed | 0.05 | pooled Wilson | 20000 | 0.0234 | at or under |
| gpt4 | strict | A | 50 | fixed | 0.05 | Clopper-Pearson | 20000 | 0.0234 | at or under |
| gpt4 | strict | A | 50 | fixed | 0.1 | Wald-t b1 | 20000 | 0.0157 | at or under |
| gpt4 | strict | A | 50 | fixed | 0.1 | b1w | 20000 | 0.0157 | at or under |
| gpt4 | strict | A | 50 | fixed | 0.1 | pooled Wilson | 20000 | 0.0234 | at or under |
| gpt4 | strict | A | 50 | fixed | 0.1 | Clopper-Pearson | 20000 | 0.0234 | at or under |
| gpt4 | strict | A | 50 | large | 0.05 | Wald-t b1 | 40000 | 0.0000 | at or under |
| gpt4 | strict | A | 50 | large | 0.05 | b1w | 40000 | 0.0239 | at or under |
| gpt4 | strict | A | 50 | large | 0.05 | pooled Wilson | 40000 | 0.0379 | at or under |
| gpt4 | strict | A | 50 | large | 0.05 | Clopper-Pearson | 40000 | 0.0379 | at or under |
| gpt4 | strict | A | 50 | large | 0.1 | Wald-t b1 | 40000 | 0.0239 | at or under |
| gpt4 | strict | A | 50 | large | 0.1 | b1w | 40000 | 0.0239 | at or under |
| gpt4 | strict | A | 50 | large | 0.1 | pooled Wilson | 40000 | 0.0379 | at or under |
| gpt4 | strict | A | 50 | large | 0.1 | Clopper-Pearson | 40000 | 0.0379 | at or under |
| gpt4 | strict | A | 100 | fixed | 0.05 | Wald-t b1 | 20000 | 0.0098 | at or under |
| gpt4 | strict | A | 100 | fixed | 0.05 | b1w | 20000 | 0.0098 | at or under |
| gpt4 | strict | A | 100 | fixed | 0.05 | pooled Wilson | 20000 | 0.0153 | at or under |
| gpt4 | strict | A | 100 | fixed | 0.05 | Clopper-Pearson | 20000 | 0.0153 | at or under |
| gpt4 | strict | A | 100 | fixed | 0.1 | Wald-t b1 | 20000 | 0.0316 | at or under |
| gpt4 | strict | A | 100 | fixed | 0.1 | b1w | 20000 | 0.0445 | at or under |
| gpt4 | strict | A | 100 | fixed | 0.1 | pooled Wilson | 20000 | 0.0580 | at or under |
| gpt4 | strict | A | 100 | fixed | 0.1 | Clopper-Pearson | 20000 | 0.0153 | at or under |
| gpt4 | strict | A | 100 | large | 0.05 | Wald-t b1 | 40000 | 0.0294 | at or under |
| gpt4 | strict | A | 100 | large | 0.05 | b1w | 40000 | 0.0294 | at or under |
| gpt4 | strict | A | 100 | large | 0.05 | pooled Wilson | 40000 | 0.0418 | at or under |
| gpt4 | strict | A | 100 | large | 0.05 | Clopper-Pearson | 40000 | 0.0418 | at or under |
| gpt4 | strict | A | 100 | large | 0.1 | Wald-t b1 | 40000 | 0.0704 | at or under |
| gpt4 | strict | A | 100 | large | 0.1 | b1w | 40000 | 0.0911 | at or under |
| gpt4 | strict | A | 100 | large | 0.1 | pooled Wilson | 40000 | 0.1096 | over |
| gpt4 | strict | A | 100 | large | 0.1 | Clopper-Pearson | 40000 | 0.0418 | at or under |
| gpt4 | broad | B | 50 | fixed | 0.05 | Wald-t b1 | 20000 | 0.0028 | at or under |
| gpt4 | broad | B | 50 | fixed | 0.05 | b1w | 20000 | 0.0028 | at or under |
| gpt4 | broad | B | 50 | fixed | 0.05 | pooled Wilson | 20000 | 0.0060 | at or under |
| gpt4 | broad | B | 50 | fixed | 0.05 | Clopper-Pearson | 20000 | 0.0060 | at or under |
| gpt4 | broad | B | 50 | fixed | 0.1 | Wald-t b1 | 20000 | 0.0309 | at or under |
| gpt4 | broad | B | 50 | fixed | 0.1 | b1w | 20000 | 0.0309 | at or under |
| gpt4 | broad | B | 50 | fixed | 0.1 | pooled Wilson | 20000 | 0.0506 | at or under |
| gpt4 | broad | B | 50 | fixed | 0.1 | Clopper-Pearson | 20000 | 0.0506 | at or under |
| gpt4 | broad | B | 50 | large | 0.05 | Wald-t b1 | 40000 | 0.0054 | at or under |
| gpt4 | broad | B | 50 | large | 0.05 | b1w | 40000 | 0.0054 | at or under |
| gpt4 | broad | B | 50 | large | 0.05 | pooled Wilson | 40000 | 0.0115 | at or under |
| gpt4 | broad | B | 50 | large | 0.05 | Clopper-Pearson | 40000 | 0.0115 | at or under |
| gpt4 | broad | B | 50 | large | 0.1 | Wald-t b1 | 40000 | 0.0443 | at or under |
| gpt4 | broad | B | 50 | large | 0.1 | b1w | 40000 | 0.0443 | at or under |
| gpt4 | broad | B | 50 | large | 0.1 | pooled Wilson | 40000 | 0.0684 | at or under |
| gpt4 | broad | B | 50 | large | 0.1 | Clopper-Pearson | 40000 | 0.0684 | at or under |
| gpt4 | broad | B | 100 | fixed | 0.05 | Wald-t b1 | 20000 | 0.0031 | at or under |
| gpt4 | broad | B | 100 | fixed | 0.05 | b1w | 20000 | 0.0097 | at or under |
| gpt4 | broad | B | 100 | fixed | 0.05 | pooled Wilson | 20000 | 0.0094 | at or under |
| gpt4 | broad | B | 100 | fixed | 0.05 | Clopper-Pearson | 20000 | 0.0094 | at or under |
| gpt4 | broad | B | 100 | fixed | 0.1 | Wald-t b1 | 20000 | 0.0190 | at or under |
| gpt4 | broad | B | 100 | fixed | 0.1 | b1w | 20000 | 0.0316 | at or under |
| gpt4 | broad | B | 100 | fixed | 0.1 | pooled Wilson | 20000 | 0.0326 | at or under |
| gpt4 | broad | B | 100 | fixed | 0.1 | Clopper-Pearson | 20000 | 0.0326 | at or under |
| gpt4 | broad | B | 100 | large | 0.05 | Wald-t b1 | 40000 | 0.0144 | at or under |
| gpt4 | broad | B | 100 | large | 0.05 | b1w | 40000 | 0.0269 | at or under |
| gpt4 | broad | B | 100 | large | 0.05 | pooled Wilson | 40000 | 0.0269 | at or under |
| gpt4 | broad | B | 100 | large | 0.05 | Clopper-Pearson | 40000 | 0.0269 | at or under |
| gpt4 | broad | B | 100 | large | 0.1 | Wald-t b1 | 40000 | 0.0487 | at or under |
| gpt4 | broad | B | 100 | large | 0.1 | b1w | 40000 | 0.0673 | at or under |
| gpt4 | broad | B | 100 | large | 0.1 | pooled Wilson | 40000 | 0.0713 | at or under |
| gpt4 | broad | B | 100 | large | 0.1 | Clopper-Pearson | 40000 | 0.0713 | at or under |
| gpt4 | broad | A | 50 | fixed | 0.05 | Wald-t b1 | 20000 | 0.0037 | at or under |
| gpt4 | broad | A | 50 | fixed | 0.05 | b1w | 20000 | 0.0037 | at or under |
| gpt4 | broad | A | 50 | fixed | 0.05 | pooled Wilson | 20000 | 0.0076 | at or under |
| gpt4 | broad | A | 50 | fixed | 0.05 | Clopper-Pearson | 20000 | 0.0076 | at or under |
| gpt4 | broad | A | 50 | fixed | 0.1 | Wald-t b1 | 20000 | 0.0392 | at or under |
| gpt4 | broad | A | 50 | fixed | 0.1 | b1w | 20000 | 0.0392 | at or under |
| gpt4 | broad | A | 50 | fixed | 0.1 | pooled Wilson | 20000 | 0.0502 | at or under |
| gpt4 | broad | A | 50 | fixed | 0.1 | Clopper-Pearson | 20000 | 0.0502 | at or under |
| gpt4 | broad | A | 50 | large | 0.05 | Wald-t b1 | 40000 | 0.0080 | at or under |
| gpt4 | broad | A | 50 | large | 0.05 | b1w | 40000 | 0.0080 | at or under |
| gpt4 | broad | A | 50 | large | 0.05 | pooled Wilson | 40000 | 0.0126 | at or under |
| gpt4 | broad | A | 50 | large | 0.05 | Clopper-Pearson | 40000 | 0.0126 | at or under |
| gpt4 | broad | A | 50 | large | 0.1 | Wald-t b1 | 40000 | 0.0553 | at or under |
| gpt4 | broad | A | 50 | large | 0.1 | b1w | 40000 | 0.0553 | at or under |
| gpt4 | broad | A | 50 | large | 0.1 | pooled Wilson | 40000 | 0.0693 | at or under |
| gpt4 | broad | A | 50 | large | 0.1 | Clopper-Pearson | 40000 | 0.0693 | at or under |
| gpt4 | broad | A | 100 | fixed | 0.05 | Wald-t b1 | 20000 | 0.0052 | at or under |
| gpt4 | broad | A | 100 | fixed | 0.05 | b1w | 20000 | 0.0089 | at or under |
| gpt4 | broad | A | 100 | fixed | 0.05 | pooled Wilson | 20000 | 0.0078 | at or under |
| gpt4 | broad | A | 100 | fixed | 0.05 | Clopper-Pearson | 20000 | 0.0078 | at or under |
| gpt4 | broad | A | 100 | fixed | 0.1 | Wald-t b1 | 20000 | 0.0254 | at or under |
| gpt4 | broad | A | 100 | fixed | 0.1 | b1w | 20000 | 0.0295 | at or under |
| gpt4 | broad | A | 100 | fixed | 0.1 | pooled Wilson | 20000 | 0.0302 | at or under |
| gpt4 | broad | A | 100 | fixed | 0.1 | Clopper-Pearson | 20000 | 0.0302 | at or under |
| gpt4 | broad | A | 100 | large | 0.05 | Wald-t b1 | 40000 | 0.0200 | at or under |
| gpt4 | broad | A | 100 | large | 0.05 | b1w | 40000 | 0.0279 | at or under |
| gpt4 | broad | A | 100 | large | 0.05 | pooled Wilson | 40000 | 0.0274 | at or under |
| gpt4 | broad | A | 100 | large | 0.05 | Clopper-Pearson | 40000 | 0.0274 | at or under |
| gpt4 | broad | A | 100 | large | 0.1 | Wald-t b1 | 40000 | 0.0586 | at or under |
| gpt4 | broad | A | 100 | large | 0.1 | b1w | 40000 | 0.0682 | at or under |
| gpt4 | broad | A | 100 | large | 0.1 | pooled Wilson | 40000 | 0.0701 | at or under |
| gpt4 | broad | A | 100 | large | 0.1 | Clopper-Pearson | 40000 | 0.0701 | at or under |
| llama2orig | strict | B | 50 | fixed | 0.05 | Wald-t b1 | 20000 | 0.0282 | at or under |
| llama2orig | strict | B | 50 | fixed | 0.05 | b1w | 20000 | 0.0467 | at or under |
| llama2orig | strict | B | 50 | fixed | 0.05 | pooled Wilson | 20000 | 0.0323 | at or under |
| llama2orig | strict | B | 50 | fixed | 0.05 | Clopper-Pearson | 20000 | 0.0146 | at or under |
| llama2orig | strict | B | 50 | fixed | 0.1 | Wald-t b1 | 20000 | 0.0713 | at or under |
| llama2orig | strict | B | 50 | fixed | 0.1 | b1w | 20000 | 0.0998 | at or under |
| llama2orig | strict | B | 50 | fixed | 0.1 | pooled Wilson | 20000 | 0.0675 | at or under |
| llama2orig | strict | B | 50 | fixed | 0.1 | Clopper-Pearson | 20000 | 0.0675 | at or under |
| llama2orig | strict | B | 50 | large | 0.05 | Wald-t b1 | 40000 | 0.0404 | at or under |
| llama2orig | strict | B | 50 | large | 0.05 | b1w | 40000 | 0.0602 | over |
| llama2orig | strict | B | 50 | large | 0.05 | pooled Wilson | 40000 | 0.0503 | unresolved |
| llama2orig | strict | B | 50 | large | 0.05 | Clopper-Pearson | 40000 | 0.0252 | at or under |
| llama2orig | strict | B | 50 | large | 0.1 | Wald-t b1 | 40000 | 0.0904 | at or under |
| llama2orig | strict | B | 50 | large | 0.1 | b1w | 40000 | 0.1197 | over |
| llama2orig | strict | B | 50 | large | 0.1 | pooled Wilson | 40000 | 0.0924 | at or under |
| llama2orig | strict | B | 50 | large | 0.1 | Clopper-Pearson | 40000 | 0.0924 | at or under |
| llama2orig | strict | B | 100 | fixed | 0.05 | Wald-t b1 | 20000 | 0.0167 | at or under |
| llama2orig | strict | B | 100 | fixed | 0.05 | b1w | 20000 | 0.0181 | at or under |
| llama2orig | strict | B | 100 | fixed | 0.05 | pooled Wilson | 20000 | 0.0233 | at or under |
| llama2orig | strict | B | 100 | fixed | 0.05 | Clopper-Pearson | 20000 | 0.0123 | at or under |
| llama2orig | strict | B | 100 | fixed | 0.1 | Wald-t b1 | 20000 | 0.0588 | at or under |
| llama2orig | strict | B | 100 | fixed | 0.1 | b1w | 20000 | 0.0602 | at or under |
| llama2orig | strict | B | 100 | fixed | 0.1 | pooled Wilson | 20000 | 0.0410 | at or under |
| llama2orig | strict | B | 100 | fixed | 0.1 | Clopper-Pearson | 20000 | 0.0410 | at or under |
| llama2orig | strict | B | 100 | large | 0.05 | Wald-t b1 | 40000 | 0.0453 | at or under |
| llama2orig | strict | B | 100 | large | 0.05 | b1w | 40000 | 0.0489 | at or under |
| llama2orig | strict | B | 100 | large | 0.05 | pooled Wilson | 40000 | 0.0579 | over |
| llama2orig | strict | B | 100 | large | 0.05 | Clopper-Pearson | 40000 | 0.0364 | at or under |
| llama2orig | strict | B | 100 | large | 0.1 | Wald-t b1 | 40000 | 0.1052 | over |
| llama2orig | strict | B | 100 | large | 0.1 | b1w | 40000 | 0.1066 | over |
| llama2orig | strict | B | 100 | large | 0.1 | pooled Wilson | 40000 | 0.0872 | at or under |
| llama2orig | strict | B | 100 | large | 0.1 | Clopper-Pearson | 40000 | 0.0872 | at or under |
| llama2orig | strict | A | 50 | fixed | 0.05 | Wald-t b1 | 20000 | 0.0253 | at or under |
| llama2orig | strict | A | 50 | fixed | 0.05 | b1w | 20000 | 0.0445 | at or under |
| llama2orig | strict | A | 50 | fixed | 0.05 | pooled Wilson | 20000 | 0.0353 | at or under |
| llama2orig | strict | A | 50 | fixed | 0.05 | Clopper-Pearson | 20000 | 0.0152 | at or under |
| llama2orig | strict | A | 50 | fixed | 0.1 | Wald-t b1 | 20000 | 0.0670 | at or under |
| llama2orig | strict | A | 50 | fixed | 0.1 | b1w | 20000 | 0.0995 | at or under |
| llama2orig | strict | A | 50 | fixed | 0.1 | pooled Wilson | 20000 | 0.0707 | at or under |
| llama2orig | strict | A | 50 | fixed | 0.1 | Clopper-Pearson | 20000 | 0.0707 | at or under |
| llama2orig | strict | A | 50 | large | 0.05 | Wald-t b1 | 40000 | 0.0401 | at or under |
| llama2orig | strict | A | 50 | large | 0.05 | b1w | 40000 | 0.0633 | over |
| llama2orig | strict | A | 50 | large | 0.05 | pooled Wilson | 40000 | 0.0520 | unresolved |
| llama2orig | strict | A | 50 | large | 0.05 | Clopper-Pearson | 40000 | 0.0263 | at or under |
| llama2orig | strict | A | 50 | large | 0.1 | Wald-t b1 | 40000 | 0.0889 | at or under |
| llama2orig | strict | A | 50 | large | 0.1 | b1w | 40000 | 0.1232 | over |
| llama2orig | strict | A | 50 | large | 0.1 | pooled Wilson | 40000 | 0.0933 | at or under |
| llama2orig | strict | A | 50 | large | 0.1 | Clopper-Pearson | 40000 | 0.0933 | at or under |
| llama2orig | strict | A | 100 | fixed | 0.05 | Wald-t b1 | 20000 | 0.0174 | at or under |
| llama2orig | strict | A | 100 | fixed | 0.05 | b1w | 20000 | 0.0180 | at or under |
| llama2orig | strict | A | 100 | fixed | 0.05 | pooled Wilson | 20000 | 0.0217 | at or under |
| llama2orig | strict | A | 100 | fixed | 0.05 | Clopper-Pearson | 20000 | 0.0118 | at or under |
| llama2orig | strict | A | 100 | fixed | 0.1 | Wald-t b1 | 20000 | 0.0566 | at or under |
| llama2orig | strict | A | 100 | fixed | 0.1 | b1w | 20000 | 0.0576 | at or under |
| llama2orig | strict | A | 100 | fixed | 0.1 | pooled Wilson | 20000 | 0.0407 | at or under |
| llama2orig | strict | A | 100 | fixed | 0.1 | Clopper-Pearson | 20000 | 0.0407 | at or under |
| llama2orig | strict | A | 100 | large | 0.05 | Wald-t b1 | 40000 | 0.0475 | at or under |
| llama2orig | strict | A | 100 | large | 0.05 | b1w | 40000 | 0.0503 | unresolved |
| llama2orig | strict | A | 100 | large | 0.05 | pooled Wilson | 40000 | 0.0593 | over |
| llama2orig | strict | A | 100 | large | 0.05 | Clopper-Pearson | 40000 | 0.0375 | at or under |
| llama2orig | strict | A | 100 | large | 0.1 | Wald-t b1 | 40000 | 0.1079 | over |
| llama2orig | strict | A | 100 | large | 0.1 | b1w | 40000 | 0.1091 | over |
| llama2orig | strict | A | 100 | large | 0.1 | pooled Wilson | 40000 | 0.0889 | at or under |
| llama2orig | strict | A | 100 | large | 0.1 | Clopper-Pearson | 40000 | 0.0889 | at or under |
| llama2orig | broad | B | 50 | fixed | 0.05 | Wald-t b1 | 20000 | 0.0240 | at or under |
| llama2orig | broad | B | 50 | fixed | 0.05 | b1w | 20000 | 0.0424 | at or under |
| llama2orig | broad | B | 50 | fixed | 0.05 | pooled Wilson | 20000 | 0.0447 | at or under |
| llama2orig | broad | B | 50 | fixed | 0.05 | Clopper-Pearson | 20000 | 0.0203 | at or under |
| llama2orig | broad | B | 50 | fixed | 0.1 | Wald-t b1 | 20000 | 0.0684 | at or under |
| llama2orig | broad | B | 50 | fixed | 0.1 | b1w | 20000 | 0.0891 | at or under |
| llama2orig | broad | B | 50 | fixed | 0.1 | pooled Wilson | 20000 | 0.0870 | at or under |
| llama2orig | broad | B | 50 | fixed | 0.1 | Clopper-Pearson | 20000 | 0.0447 | at or under |
| llama2orig | broad | B | 50 | large | 0.05 | Wald-t b1 | 40000 | 0.0391 | at or under |
| llama2orig | broad | B | 50 | large | 0.05 | b1w | 40000 | 0.0654 | over |
| llama2orig | broad | B | 50 | large | 0.05 | pooled Wilson | 40000 | 0.0650 | over |
| llama2orig | broad | B | 50 | large | 0.05 | Clopper-Pearson | 40000 | 0.0363 | at or under |
| llama2orig | broad | B | 50 | large | 0.1 | Wald-t b1 | 40000 | 0.0914 | at or under |
| llama2orig | broad | B | 50 | large | 0.1 | b1w | 40000 | 0.1185 | over |
| llama2orig | broad | B | 50 | large | 0.1 | pooled Wilson | 40000 | 0.1098 | over |
| llama2orig | broad | B | 50 | large | 0.1 | Clopper-Pearson | 40000 | 0.0650 | at or under |
| llama2orig | broad | B | 100 | fixed | 0.05 | Wald-t b1 | 20000 | 0.0140 | at or under |
| llama2orig | broad | B | 100 | fixed | 0.05 | b1w | 20000 | 0.0242 | at or under |
| llama2orig | broad | B | 100 | fixed | 0.05 | pooled Wilson | 20000 | 0.0175 | at or under |
| llama2orig | broad | B | 100 | fixed | 0.05 | Clopper-Pearson | 20000 | 0.0094 | at or under |
| llama2orig | broad | B | 100 | fixed | 0.1 | Wald-t b1 | 20000 | 0.0497 | at or under |
| llama2orig | broad | B | 100 | fixed | 0.1 | b1w | 20000 | 0.0529 | at or under |
| llama2orig | broad | B | 100 | fixed | 0.1 | pooled Wilson | 20000 | 0.0592 | at or under |
| llama2orig | broad | B | 100 | fixed | 0.1 | Clopper-Pearson | 20000 | 0.0321 | at or under |
| llama2orig | broad | B | 100 | large | 0.05 | Wald-t b1 | 40000 | 0.0440 | at or under |
| llama2orig | broad | B | 100 | large | 0.05 | b1w | 40000 | 0.0621 | over |
| llama2orig | broad | B | 100 | large | 0.05 | pooled Wilson | 40000 | 0.0499 | at or under |
| llama2orig | broad | B | 100 | large | 0.05 | Clopper-Pearson | 40000 | 0.0335 | at or under |
| llama2orig | broad | B | 100 | large | 0.1 | Wald-t b1 | 40000 | 0.0993 | at or under |
| llama2orig | broad | B | 100 | large | 0.1 | b1w | 40000 | 0.1073 | over |
| llama2orig | broad | B | 100 | large | 0.1 | pooled Wilson | 40000 | 0.1076 | over |
| llama2orig | broad | B | 100 | large | 0.1 | Clopper-Pearson | 40000 | 0.0753 | at or under |
| llama2orig | broad | A | 50 | fixed | 0.05 | Wald-t b1 | 20000 | 0.0226 | at or under |
| llama2orig | broad | A | 50 | fixed | 0.05 | b1w | 20000 | 0.0472 | at or under |
| llama2orig | broad | A | 50 | fixed | 0.05 | pooled Wilson | 20000 | 0.0454 | at or under |
| llama2orig | broad | A | 50 | fixed | 0.05 | Clopper-Pearson | 20000 | 0.0227 | at or under |
| llama2orig | broad | A | 50 | fixed | 0.1 | Wald-t b1 | 20000 | 0.0663 | at or under |
| llama2orig | broad | A | 50 | fixed | 0.1 | b1w | 20000 | 0.0973 | at or under |
| llama2orig | broad | A | 50 | fixed | 0.1 | pooled Wilson | 20000 | 0.0847 | at or under |
| llama2orig | broad | A | 50 | fixed | 0.1 | Clopper-Pearson | 20000 | 0.0454 | at or under |
| llama2orig | broad | A | 50 | large | 0.05 | Wald-t b1 | 40000 | 0.0350 | at or under |
| llama2orig | broad | A | 50 | large | 0.05 | b1w | 40000 | 0.0653 | over |
| llama2orig | broad | A | 50 | large | 0.05 | pooled Wilson | 40000 | 0.0652 | over |
| llama2orig | broad | A | 50 | large | 0.05 | Clopper-Pearson | 40000 | 0.0356 | at or under |
| llama2orig | broad | A | 50 | large | 0.1 | Wald-t b1 | 40000 | 0.0840 | at or under |
| llama2orig | broad | A | 50 | large | 0.1 | b1w | 40000 | 0.1208 | over |
| llama2orig | broad | A | 50 | large | 0.1 | pooled Wilson | 40000 | 0.1105 | over |
| llama2orig | broad | A | 50 | large | 0.1 | Clopper-Pearson | 40000 | 0.0652 | at or under |
| llama2orig | broad | A | 100 | fixed | 0.05 | Wald-t b1 | 20000 | 0.0126 | at or under |
| llama2orig | broad | A | 100 | fixed | 0.05 | b1w | 20000 | 0.0231 | at or under |
| llama2orig | broad | A | 100 | fixed | 0.05 | pooled Wilson | 20000 | 0.0169 | at or under |
| llama2orig | broad | A | 100 | fixed | 0.05 | Clopper-Pearson | 20000 | 0.0084 | at or under |
| llama2orig | broad | A | 100 | fixed | 0.1 | Wald-t b1 | 20000 | 0.0462 | at or under |
| llama2orig | broad | A | 100 | fixed | 0.1 | b1w | 20000 | 0.0567 | at or under |
| llama2orig | broad | A | 100 | fixed | 0.1 | pooled Wilson | 20000 | 0.0554 | at or under |
| llama2orig | broad | A | 100 | fixed | 0.1 | Clopper-Pearson | 20000 | 0.0316 | at or under |
| llama2orig | broad | A | 100 | large | 0.05 | Wald-t b1 | 40000 | 0.0420 | at or under |
| llama2orig | broad | A | 100 | large | 0.05 | b1w | 40000 | 0.0609 | over |
| llama2orig | broad | A | 100 | large | 0.05 | pooled Wilson | 40000 | 0.0522 | unresolved |
| llama2orig | broad | A | 100 | large | 0.05 | Clopper-Pearson | 40000 | 0.0336 | at or under |
| llama2orig | broad | A | 100 | large | 0.1 | Wald-t b1 | 40000 | 0.0945 | at or under |
| llama2orig | broad | A | 100 | large | 0.1 | b1w | 40000 | 0.1116 | over |
| llama2orig | broad | A | 100 | large | 0.1 | pooled Wilson | 40000 | 0.1105 | over |
| llama2orig | broad | A | 100 | large | 0.1 | Clopper-Pearson | 40000 | 0.0779 | at or under |
| llama2new | strict | B | 50 | fixed | 0.05 | Wald-t b1 | 20000 | 0.0153 | at or under |
| llama2new | strict | B | 50 | fixed | 0.05 | b1w | 20000 | 0.0267 | at or under |
| llama2new | strict | B | 50 | fixed | 0.05 | pooled Wilson | 20000 | 0.0133 | at or under |
| llama2new | strict | B | 50 | fixed | 0.05 | Clopper-Pearson | 20000 | 0.0133 | at or under |
| llama2new | strict | B | 50 | fixed | 0.1 | Wald-t b1 | 20000 | 0.0563 | at or under |
| llama2new | strict | B | 50 | fixed | 0.1 | b1w | 20000 | 0.0646 | at or under |
| llama2new | strict | B | 50 | fixed | 0.1 | pooled Wilson | 20000 | 0.0457 | at or under |
| llama2new | strict | B | 50 | fixed | 0.1 | Clopper-Pearson | 20000 | 0.0457 | at or under |
| llama2new | strict | B | 50 | large | 0.05 | Wald-t b1 | 40000 | 0.0227 | at or under |
| llama2new | strict | B | 50 | large | 0.05 | b1w | 40000 | 0.0372 | at or under |
| llama2new | strict | B | 50 | large | 0.05 | pooled Wilson | 40000 | 0.0225 | at or under |
| llama2new | strict | B | 50 | large | 0.05 | Clopper-Pearson | 40000 | 0.0225 | at or under |
| llama2new | strict | B | 50 | large | 0.1 | Wald-t b1 | 40000 | 0.0701 | at or under |
| llama2new | strict | B | 50 | large | 0.1 | b1w | 40000 | 0.0816 | at or under |
| llama2new | strict | B | 50 | large | 0.1 | pooled Wilson | 40000 | 0.0677 | at or under |
| llama2new | strict | B | 50 | large | 0.1 | Clopper-Pearson | 40000 | 0.0677 | at or under |
| llama2new | strict | B | 100 | fixed | 0.05 | Wald-t b1 | 20000 | 0.0186 | at or under |
| llama2new | strict | B | 100 | fixed | 0.05 | b1w | 20000 | 0.0186 | at or under |
| llama2new | strict | B | 100 | fixed | 0.05 | pooled Wilson | 20000 | 0.0193 | at or under |
| llama2new | strict | B | 100 | fixed | 0.05 | Clopper-Pearson | 20000 | 0.0193 | at or under |
| llama2new | strict | B | 100 | fixed | 0.1 | Wald-t b1 | 20000 | 0.0498 | at or under |
| llama2new | strict | B | 100 | fixed | 0.1 | b1w | 20000 | 0.0498 | at or under |
| llama2new | strict | B | 100 | fixed | 0.1 | pooled Wilson | 20000 | 0.0463 | at or under |
| llama2new | strict | B | 100 | fixed | 0.1 | Clopper-Pearson | 20000 | 0.0463 | at or under |
| llama2new | strict | B | 100 | large | 0.05 | Wald-t b1 | 40000 | 0.0485 | at or under |
| llama2new | strict | B | 100 | large | 0.05 | b1w | 40000 | 0.0485 | at or under |
| llama2new | strict | B | 100 | large | 0.05 | pooled Wilson | 40000 | 0.0490 | at or under |
| llama2new | strict | B | 100 | large | 0.05 | Clopper-Pearson | 40000 | 0.0490 | at or under |
| llama2new | strict | B | 100 | large | 0.1 | Wald-t b1 | 40000 | 0.0935 | at or under |
| llama2new | strict | B | 100 | large | 0.1 | b1w | 40000 | 0.0937 | at or under |
| llama2new | strict | B | 100 | large | 0.1 | pooled Wilson | 40000 | 0.0915 | at or under |
| llama2new | strict | B | 100 | large | 0.1 | Clopper-Pearson | 40000 | 0.0915 | at or under |
| llama2new | strict | A | 50 | fixed | 0.05 | Wald-t b1 | 20000 | 0.0147 | at or under |
| llama2new | strict | A | 50 | fixed | 0.05 | b1w | 20000 | 0.0264 | at or under |
| llama2new | strict | A | 50 | fixed | 0.05 | pooled Wilson | 20000 | 0.0135 | at or under |
| llama2new | strict | A | 50 | fixed | 0.05 | Clopper-Pearson | 20000 | 0.0135 | at or under |
| llama2new | strict | A | 50 | fixed | 0.1 | Wald-t b1 | 20000 | 0.0518 | at or under |
| llama2new | strict | A | 50 | fixed | 0.1 | b1w | 20000 | 0.0587 | at or under |
| llama2new | strict | A | 50 | fixed | 0.1 | pooled Wilson | 20000 | 0.0450 | at or under |
| llama2new | strict | A | 50 | fixed | 0.1 | Clopper-Pearson | 20000 | 0.0450 | at or under |
| llama2new | strict | A | 50 | large | 0.05 | Wald-t b1 | 40000 | 0.0221 | at or under |
| llama2new | strict | A | 50 | large | 0.05 | b1w | 40000 | 0.0375 | at or under |
| llama2new | strict | A | 50 | large | 0.05 | pooled Wilson | 40000 | 0.0223 | at or under |
| llama2new | strict | A | 50 | large | 0.05 | Clopper-Pearson | 40000 | 0.0223 | at or under |
| llama2new | strict | A | 50 | large | 0.1 | Wald-t b1 | 40000 | 0.0688 | at or under |
| llama2new | strict | A | 50 | large | 0.1 | b1w | 40000 | 0.0784 | at or under |
| llama2new | strict | A | 50 | large | 0.1 | pooled Wilson | 40000 | 0.0674 | at or under |
| llama2new | strict | A | 50 | large | 0.1 | Clopper-Pearson | 40000 | 0.0674 | at or under |
| llama2new | strict | A | 100 | fixed | 0.05 | Wald-t b1 | 20000 | 0.0201 | at or under |
| llama2new | strict | A | 100 | fixed | 0.05 | b1w | 20000 | 0.0201 | at or under |
| llama2new | strict | A | 100 | fixed | 0.05 | pooled Wilson | 20000 | 0.0193 | at or under |
| llama2new | strict | A | 100 | fixed | 0.05 | Clopper-Pearson | 20000 | 0.0193 | at or under |
| llama2new | strict | A | 100 | fixed | 0.1 | Wald-t b1 | 20000 | 0.0512 | at or under |
| llama2new | strict | A | 100 | fixed | 0.1 | b1w | 20000 | 0.0512 | at or under |
| llama2new | strict | A | 100 | fixed | 0.1 | pooled Wilson | 20000 | 0.0464 | at or under |
| llama2new | strict | A | 100 | fixed | 0.1 | Clopper-Pearson | 20000 | 0.0464 | at or under |
| llama2new | strict | A | 100 | large | 0.05 | Wald-t b1 | 40000 | 0.0491 | at or under |
| llama2new | strict | A | 100 | large | 0.05 | b1w | 40000 | 0.0491 | at or under |
| llama2new | strict | A | 100 | large | 0.05 | pooled Wilson | 40000 | 0.0480 | at or under |
| llama2new | strict | A | 100 | large | 0.05 | Clopper-Pearson | 40000 | 0.0480 | at or under |
| llama2new | strict | A | 100 | large | 0.1 | Wald-t b1 | 40000 | 0.0956 | at or under |
| llama2new | strict | A | 100 | large | 0.1 | b1w | 40000 | 0.0956 | at or under |
| llama2new | strict | A | 100 | large | 0.1 | pooled Wilson | 40000 | 0.0912 | at or under |
| llama2new | strict | A | 100 | large | 0.1 | Clopper-Pearson | 40000 | 0.0912 | at or under |
| llama2new | broad | B | 50 | fixed | 0.05 | Wald-t b1 | 20000 | 0.0326 | at or under |
| llama2new | broad | B | 50 | fixed | 0.05 | b1w | 20000 | 0.0369 | at or under |
| llama2new | broad | B | 50 | fixed | 0.05 | pooled Wilson | 20000 | 0.0318 | at or under |
| llama2new | broad | B | 50 | fixed | 0.05 | Clopper-Pearson | 20000 | 0.0318 | at or under |
| llama2new | broad | B | 50 | fixed | 0.1 | Wald-t b1 | 20000 | 0.0748 | at or under |
| llama2new | broad | B | 50 | fixed | 0.1 | b1w | 20000 | 0.0823 | at or under |
| llama2new | broad | B | 50 | fixed | 0.1 | pooled Wilson | 20000 | 0.0690 | at or under |
| llama2new | broad | B | 50 | fixed | 0.1 | Clopper-Pearson | 20000 | 0.0690 | at or under |
| llama2new | broad | B | 50 | large | 0.05 | Wald-t b1 | 40000 | 0.0446 | at or under |
| llama2new | broad | B | 50 | large | 0.05 | b1w | 40000 | 0.0525 | over |
| llama2new | broad | B | 50 | large | 0.05 | pooled Wilson | 40000 | 0.0454 | at or under |
| llama2new | broad | B | 50 | large | 0.05 | Clopper-Pearson | 40000 | 0.0454 | at or under |
| llama2new | broad | B | 50 | large | 0.1 | Wald-t b1 | 40000 | 0.0930 | at or under |
| llama2new | broad | B | 50 | large | 0.1 | b1w | 40000 | 0.1040 | over |
| llama2new | broad | B | 50 | large | 0.1 | pooled Wilson | 40000 | 0.0888 | at or under |
| llama2new | broad | B | 50 | large | 0.1 | Clopper-Pearson | 40000 | 0.0888 | at or under |
| llama2new | broad | B | 100 | fixed | 0.05 | Wald-t b1 | 20000 | 0.0182 | at or under |
| llama2new | broad | B | 100 | fixed | 0.05 | b1w | 20000 | 0.0183 | at or under |
| llama2new | broad | B | 100 | fixed | 0.05 | pooled Wilson | 20000 | 0.0210 | at or under |
| llama2new | broad | B | 100 | fixed | 0.05 | Clopper-Pearson | 20000 | 0.0088 | at or under |
| llama2new | broad | B | 100 | fixed | 0.1 | Wald-t b1 | 20000 | 0.0469 | at or under |
| llama2new | broad | B | 100 | fixed | 0.1 | b1w | 20000 | 0.0488 | at or under |
| llama2new | broad | B | 100 | fixed | 0.1 | pooled Wilson | 20000 | 0.0401 | at or under |
| llama2new | broad | B | 100 | fixed | 0.1 | Clopper-Pearson | 20000 | 0.0401 | at or under |
| llama2new | broad | B | 100 | large | 0.05 | Wald-t b1 | 40000 | 0.0525 | over |
| llama2new | broad | B | 100 | large | 0.05 | b1w | 40000 | 0.0529 | over |
| llama2new | broad | B | 100 | large | 0.05 | pooled Wilson | 40000 | 0.0564 | over |
| llama2new | broad | B | 100 | large | 0.05 | Clopper-Pearson | 40000 | 0.0336 | at or under |
| llama2new | broad | B | 100 | large | 0.1 | Wald-t b1 | 40000 | 0.0965 | at or under |
| llama2new | broad | B | 100 | large | 0.1 | b1w | 40000 | 0.1007 | unresolved |
| llama2new | broad | B | 100 | large | 0.1 | pooled Wilson | 40000 | 0.0894 | at or under |
| llama2new | broad | B | 100 | large | 0.1 | Clopper-Pearson | 40000 | 0.0894 | at or under |
| llama2new | broad | A | 50 | fixed | 0.05 | Wald-t b1 | 20000 | 0.0261 | at or under |
| llama2new | broad | A | 50 | fixed | 0.05 | b1w | 20000 | 0.0293 | at or under |
| llama2new | broad | A | 50 | fixed | 0.05 | pooled Wilson | 20000 | 0.0307 | at or under |
| llama2new | broad | A | 50 | fixed | 0.05 | Clopper-Pearson | 20000 | 0.0307 | at or under |
| llama2new | broad | A | 50 | fixed | 0.1 | Wald-t b1 | 20000 | 0.0627 | at or under |
| llama2new | broad | A | 50 | fixed | 0.1 | b1w | 20000 | 0.0709 | at or under |
| llama2new | broad | A | 50 | fixed | 0.1 | pooled Wilson | 20000 | 0.0636 | at or under |
| llama2new | broad | A | 50 | fixed | 0.1 | Clopper-Pearson | 20000 | 0.0636 | at or under |
| llama2new | broad | A | 50 | large | 0.05 | Wald-t b1 | 40000 | 0.0382 | at or under |
| llama2new | broad | A | 50 | large | 0.05 | b1w | 40000 | 0.0447 | at or under |
| llama2new | broad | A | 50 | large | 0.05 | pooled Wilson | 40000 | 0.0452 | at or under |
| llama2new | broad | A | 50 | large | 0.05 | Clopper-Pearson | 40000 | 0.0452 | at or under |
| llama2new | broad | A | 50 | large | 0.1 | Wald-t b1 | 40000 | 0.0827 | at or under |
| llama2new | broad | A | 50 | large | 0.1 | b1w | 40000 | 0.0945 | at or under |
| llama2new | broad | A | 50 | large | 0.1 | pooled Wilson | 40000 | 0.0868 | at or under |
| llama2new | broad | A | 50 | large | 0.1 | Clopper-Pearson | 40000 | 0.0868 | at or under |
| llama2new | broad | A | 100 | fixed | 0.05 | Wald-t b1 | 20000 | 0.0177 | at or under |
| llama2new | broad | A | 100 | fixed | 0.05 | b1w | 20000 | 0.0179 | at or under |
| llama2new | broad | A | 100 | fixed | 0.05 | pooled Wilson | 20000 | 0.0210 | at or under |
| llama2new | broad | A | 100 | fixed | 0.05 | Clopper-Pearson | 20000 | 0.0103 | at or under |
| llama2new | broad | A | 100 | fixed | 0.1 | Wald-t b1 | 20000 | 0.0421 | at or under |
| llama2new | broad | A | 100 | fixed | 0.1 | b1w | 20000 | 0.0440 | at or under |
| llama2new | broad | A | 100 | fixed | 0.1 | pooled Wilson | 20000 | 0.0401 | at or under |
| llama2new | broad | A | 100 | fixed | 0.1 | Clopper-Pearson | 20000 | 0.0401 | at or under |
| llama2new | broad | A | 100 | large | 0.05 | Wald-t b1 | 40000 | 0.0479 | at or under |
| llama2new | broad | A | 100 | large | 0.05 | b1w | 40000 | 0.0484 | at or under |
| llama2new | broad | A | 100 | large | 0.05 | pooled Wilson | 40000 | 0.0553 | over |
| llama2new | broad | A | 100 | large | 0.05 | Clopper-Pearson | 40000 | 0.0344 | at or under |
| llama2new | broad | A | 100 | large | 0.1 | Wald-t b1 | 40000 | 0.0838 | at or under |
| llama2new | broad | A | 100 | large | 0.1 | b1w | 40000 | 0.0903 | at or under |
| llama2new | broad | A | 100 | large | 0.1 | pooled Wilson | 40000 | 0.0872 | at or under |
| llama2new | broad | A | 100 | large | 0.1 | Clopper-Pearson | 40000 | 0.0872 | at or under |
| mistralinstruct | strict | B | 50 | fixed | 0.05 | Wald-t b1 | 20000 | 0.0000 | at or under |
| mistralinstruct | strict | B | 50 | fixed | 0.05 | b1w | 20000 | 0.0000 | at or under |
| mistralinstruct | strict | B | 50 | fixed | 0.05 | pooled Wilson | 20000 | 0.0000 | at or under |
| mistralinstruct | strict | B | 50 | fixed | 0.05 | Clopper-Pearson | 20000 | 0.0000 | at or under |
| mistralinstruct | strict | B | 50 | fixed | 0.1 | Wald-t b1 | 20000 | 0.0000 | at or under |
| mistralinstruct | strict | B | 50 | fixed | 0.1 | b1w | 20000 | 0.0000 | at or under |
| mistralinstruct | strict | B | 50 | fixed | 0.1 | pooled Wilson | 20000 | 0.0000 | at or under |
| mistralinstruct | strict | B | 50 | fixed | 0.1 | Clopper-Pearson | 20000 | 0.0000 | at or under |
| mistralinstruct | strict | B | 50 | large | 0.05 | Wald-t b1 | 40000 | 0.0000 | at or under |
| mistralinstruct | strict | B | 50 | large | 0.05 | b1w | 40000 | 0.0000 | at or under |
| mistralinstruct | strict | B | 50 | large | 0.05 | pooled Wilson | 40000 | 0.0000 | at or under |
| mistralinstruct | strict | B | 50 | large | 0.05 | Clopper-Pearson | 40000 | 0.0000 | at or under |
| mistralinstruct | strict | B | 50 | large | 0.1 | Wald-t b1 | 40000 | 0.0000 | at or under |
| mistralinstruct | strict | B | 50 | large | 0.1 | b1w | 40000 | 0.0000 | at or under |
| mistralinstruct | strict | B | 50 | large | 0.1 | pooled Wilson | 40000 | 0.0000 | at or under |
| mistralinstruct | strict | B | 50 | large | 0.1 | Clopper-Pearson | 40000 | 0.0000 | at or under |
| mistralinstruct | strict | B | 100 | fixed | 0.05 | Wald-t b1 | 20000 | 0.0000 | at or under |
| mistralinstruct | strict | B | 100 | fixed | 0.05 | b1w | 20000 | 0.0000 | at or under |
| mistralinstruct | strict | B | 100 | fixed | 0.05 | pooled Wilson | 20000 | 0.0000 | at or under |
| mistralinstruct | strict | B | 100 | fixed | 0.05 | Clopper-Pearson | 20000 | 0.0000 | at or under |
| mistralinstruct | strict | B | 100 | fixed | 0.1 | Wald-t b1 | 20000 | 0.0000 | at or under |
| mistralinstruct | strict | B | 100 | fixed | 0.1 | b1w | 20000 | 0.0000 | at or under |
| mistralinstruct | strict | B | 100 | fixed | 0.1 | pooled Wilson | 20000 | 0.0000 | at or under |
| mistralinstruct | strict | B | 100 | fixed | 0.1 | Clopper-Pearson | 20000 | 0.0000 | at or under |
| mistralinstruct | strict | B | 100 | large | 0.05 | Wald-t b1 | 40000 | 0.0000 | at or under |
| mistralinstruct | strict | B | 100 | large | 0.05 | b1w | 40000 | 0.0000 | at or under |
| mistralinstruct | strict | B | 100 | large | 0.05 | pooled Wilson | 40000 | 0.0000 | at or under |
| mistralinstruct | strict | B | 100 | large | 0.05 | Clopper-Pearson | 40000 | 0.0000 | at or under |
| mistralinstruct | strict | B | 100 | large | 0.1 | Wald-t b1 | 40000 | 0.0000 | at or under |
| mistralinstruct | strict | B | 100 | large | 0.1 | b1w | 40000 | 0.0000 | at or under |
| mistralinstruct | strict | B | 100 | large | 0.1 | pooled Wilson | 40000 | 0.0000 | at or under |
| mistralinstruct | strict | B | 100 | large | 0.1 | Clopper-Pearson | 40000 | 0.0000 | at or under |
| mistralinstruct | strict | A | 50 | fixed | 0.05 | Wald-t b1 | 20000 | 0.0000 | at or under |
| mistralinstruct | strict | A | 50 | fixed | 0.05 | b1w | 20000 | 0.0000 | at or under |
| mistralinstruct | strict | A | 50 | fixed | 0.05 | pooled Wilson | 20000 | 0.0000 | at or under |
| mistralinstruct | strict | A | 50 | fixed | 0.05 | Clopper-Pearson | 20000 | 0.0000 | at or under |
| mistralinstruct | strict | A | 50 | fixed | 0.1 | Wald-t b1 | 20000 | 0.0000 | at or under |
| mistralinstruct | strict | A | 50 | fixed | 0.1 | b1w | 20000 | 0.0000 | at or under |
| mistralinstruct | strict | A | 50 | fixed | 0.1 | pooled Wilson | 20000 | 0.0000 | at or under |
| mistralinstruct | strict | A | 50 | fixed | 0.1 | Clopper-Pearson | 20000 | 0.0000 | at or under |
| mistralinstruct | strict | A | 50 | large | 0.05 | Wald-t b1 | 40000 | 0.0000 | at or under |
| mistralinstruct | strict | A | 50 | large | 0.05 | b1w | 40000 | 0.0000 | at or under |
| mistralinstruct | strict | A | 50 | large | 0.05 | pooled Wilson | 40000 | 0.0000 | at or under |
| mistralinstruct | strict | A | 50 | large | 0.05 | Clopper-Pearson | 40000 | 0.0000 | at or under |
| mistralinstruct | strict | A | 50 | large | 0.1 | Wald-t b1 | 40000 | 0.0000 | at or under |
| mistralinstruct | strict | A | 50 | large | 0.1 | b1w | 40000 | 0.0000 | at or under |
| mistralinstruct | strict | A | 50 | large | 0.1 | pooled Wilson | 40000 | 0.0000 | at or under |
| mistralinstruct | strict | A | 50 | large | 0.1 | Clopper-Pearson | 40000 | 0.0000 | at or under |
| mistralinstruct | strict | A | 100 | fixed | 0.05 | Wald-t b1 | 20000 | 0.0000 | at or under |
| mistralinstruct | strict | A | 100 | fixed | 0.05 | b1w | 20000 | 0.0000 | at or under |
| mistralinstruct | strict | A | 100 | fixed | 0.05 | pooled Wilson | 20000 | 0.0000 | at or under |
| mistralinstruct | strict | A | 100 | fixed | 0.05 | Clopper-Pearson | 20000 | 0.0000 | at or under |
| mistralinstruct | strict | A | 100 | fixed | 0.1 | Wald-t b1 | 20000 | 0.0000 | at or under |
| mistralinstruct | strict | A | 100 | fixed | 0.1 | b1w | 20000 | 0.0000 | at or under |
| mistralinstruct | strict | A | 100 | fixed | 0.1 | pooled Wilson | 20000 | 0.0000 | at or under |
| mistralinstruct | strict | A | 100 | fixed | 0.1 | Clopper-Pearson | 20000 | 0.0000 | at or under |
| mistralinstruct | strict | A | 100 | large | 0.05 | Wald-t b1 | 40000 | 0.0000 | at or under |
| mistralinstruct | strict | A | 100 | large | 0.05 | b1w | 40000 | 0.0000 | at or under |
| mistralinstruct | strict | A | 100 | large | 0.05 | pooled Wilson | 40000 | 0.0000 | at or under |
| mistralinstruct | strict | A | 100 | large | 0.05 | Clopper-Pearson | 40000 | 0.0000 | at or under |
| mistralinstruct | strict | A | 100 | large | 0.1 | Wald-t b1 | 40000 | 0.0000 | at or under |
| mistralinstruct | strict | A | 100 | large | 0.1 | b1w | 40000 | 0.0000 | at or under |
| mistralinstruct | strict | A | 100 | large | 0.1 | pooled Wilson | 40000 | 0.0000 | at or under |
| mistralinstruct | strict | A | 100 | large | 0.1 | Clopper-Pearson | 40000 | 0.0000 | at or under |
| mistralinstruct | broad | B | 50 | fixed | 0.05 | Wald-t b1 | 20000 | 0.0000 | at or under |
| mistralinstruct | broad | B | 50 | fixed | 0.05 | b1w | 20000 | 0.0000 | at or under |
| mistralinstruct | broad | B | 50 | fixed | 0.05 | pooled Wilson | 20000 | 0.0000 | at or under |
| mistralinstruct | broad | B | 50 | fixed | 0.05 | Clopper-Pearson | 20000 | 0.0000 | at or under |
| mistralinstruct | broad | B | 50 | fixed | 0.1 | Wald-t b1 | 20000 | 0.0000 | at or under |
| mistralinstruct | broad | B | 50 | fixed | 0.1 | b1w | 20000 | 0.0000 | at or under |
| mistralinstruct | broad | B | 50 | fixed | 0.1 | pooled Wilson | 20000 | 0.0000 | at or under |
| mistralinstruct | broad | B | 50 | fixed | 0.1 | Clopper-Pearson | 20000 | 0.0000 | at or under |
| mistralinstruct | broad | B | 50 | large | 0.05 | Wald-t b1 | 40000 | 0.0000 | at or under |
| mistralinstruct | broad | B | 50 | large | 0.05 | b1w | 40000 | 0.0000 | at or under |
| mistralinstruct | broad | B | 50 | large | 0.05 | pooled Wilson | 40000 | 0.0000 | at or under |
| mistralinstruct | broad | B | 50 | large | 0.05 | Clopper-Pearson | 40000 | 0.0000 | at or under |
| mistralinstruct | broad | B | 50 | large | 0.1 | Wald-t b1 | 40000 | 0.0000 | at or under |
| mistralinstruct | broad | B | 50 | large | 0.1 | b1w | 40000 | 0.0000 | at or under |
| mistralinstruct | broad | B | 50 | large | 0.1 | pooled Wilson | 40000 | 0.0000 | at or under |
| mistralinstruct | broad | B | 50 | large | 0.1 | Clopper-Pearson | 40000 | 0.0000 | at or under |
| mistralinstruct | broad | B | 100 | fixed | 0.05 | Wald-t b1 | 20000 | 0.0000 | at or under |
| mistralinstruct | broad | B | 100 | fixed | 0.05 | b1w | 20000 | 0.0000 | at or under |
| mistralinstruct | broad | B | 100 | fixed | 0.05 | pooled Wilson | 20000 | 0.0000 | at or under |
| mistralinstruct | broad | B | 100 | fixed | 0.05 | Clopper-Pearson | 20000 | 0.0000 | at or under |
| mistralinstruct | broad | B | 100 | fixed | 0.1 | Wald-t b1 | 20000 | 0.0000 | at or under |
| mistralinstruct | broad | B | 100 | fixed | 0.1 | b1w | 20000 | 0.0000 | at or under |
| mistralinstruct | broad | B | 100 | fixed | 0.1 | pooled Wilson | 20000 | 0.0000 | at or under |
| mistralinstruct | broad | B | 100 | fixed | 0.1 | Clopper-Pearson | 20000 | 0.0000 | at or under |
| mistralinstruct | broad | B | 100 | large | 0.05 | Wald-t b1 | 40000 | 0.0000 | at or under |
| mistralinstruct | broad | B | 100 | large | 0.05 | b1w | 40000 | 0.0000 | at or under |
| mistralinstruct | broad | B | 100 | large | 0.05 | pooled Wilson | 40000 | 0.0000 | at or under |
| mistralinstruct | broad | B | 100 | large | 0.05 | Clopper-Pearson | 40000 | 0.0000 | at or under |
| mistralinstruct | broad | B | 100 | large | 0.1 | Wald-t b1 | 40000 | 0.0000 | at or under |
| mistralinstruct | broad | B | 100 | large | 0.1 | b1w | 40000 | 0.0000 | at or under |
| mistralinstruct | broad | B | 100 | large | 0.1 | pooled Wilson | 40000 | 0.0000 | at or under |
| mistralinstruct | broad | B | 100 | large | 0.1 | Clopper-Pearson | 40000 | 0.0000 | at or under |
| mistralinstruct | broad | A | 50 | fixed | 0.05 | Wald-t b1 | 20000 | 0.0000 | at or under |
| mistralinstruct | broad | A | 50 | fixed | 0.05 | b1w | 20000 | 0.0000 | at or under |
| mistralinstruct | broad | A | 50 | fixed | 0.05 | pooled Wilson | 20000 | 0.0000 | at or under |
| mistralinstruct | broad | A | 50 | fixed | 0.05 | Clopper-Pearson | 20000 | 0.0000 | at or under |
| mistralinstruct | broad | A | 50 | fixed | 0.1 | Wald-t b1 | 20000 | 0.0000 | at or under |
| mistralinstruct | broad | A | 50 | fixed | 0.1 | b1w | 20000 | 0.0000 | at or under |
| mistralinstruct | broad | A | 50 | fixed | 0.1 | pooled Wilson | 20000 | 0.0000 | at or under |
| mistralinstruct | broad | A | 50 | fixed | 0.1 | Clopper-Pearson | 20000 | 0.0000 | at or under |
| mistralinstruct | broad | A | 50 | large | 0.05 | Wald-t b1 | 40000 | 0.0000 | at or under |
| mistralinstruct | broad | A | 50 | large | 0.05 | b1w | 40000 | 0.0000 | at or under |
| mistralinstruct | broad | A | 50 | large | 0.05 | pooled Wilson | 40000 | 0.0000 | at or under |
| mistralinstruct | broad | A | 50 | large | 0.05 | Clopper-Pearson | 40000 | 0.0000 | at or under |
| mistralinstruct | broad | A | 50 | large | 0.1 | Wald-t b1 | 40000 | 0.0000 | at or under |
| mistralinstruct | broad | A | 50 | large | 0.1 | b1w | 40000 | 0.0000 | at or under |
| mistralinstruct | broad | A | 50 | large | 0.1 | pooled Wilson | 40000 | 0.0000 | at or under |
| mistralinstruct | broad | A | 50 | large | 0.1 | Clopper-Pearson | 40000 | 0.0000 | at or under |
| mistralinstruct | broad | A | 100 | fixed | 0.05 | Wald-t b1 | 20000 | 0.0000 | at or under |
| mistralinstruct | broad | A | 100 | fixed | 0.05 | b1w | 20000 | 0.0000 | at or under |
| mistralinstruct | broad | A | 100 | fixed | 0.05 | pooled Wilson | 20000 | 0.0000 | at or under |
| mistralinstruct | broad | A | 100 | fixed | 0.05 | Clopper-Pearson | 20000 | 0.0000 | at or under |
| mistralinstruct | broad | A | 100 | fixed | 0.1 | Wald-t b1 | 20000 | 0.0000 | at or under |
| mistralinstruct | broad | A | 100 | fixed | 0.1 | b1w | 20000 | 0.0000 | at or under |
| mistralinstruct | broad | A | 100 | fixed | 0.1 | pooled Wilson | 20000 | 0.0000 | at or under |
| mistralinstruct | broad | A | 100 | fixed | 0.1 | Clopper-Pearson | 20000 | 0.0000 | at or under |
| mistralinstruct | broad | A | 100 | large | 0.05 | Wald-t b1 | 40000 | 0.0000 | at or under |
| mistralinstruct | broad | A | 100 | large | 0.05 | b1w | 40000 | 0.0000 | at or under |
| mistralinstruct | broad | A | 100 | large | 0.05 | pooled Wilson | 40000 | 0.0000 | at or under |
| mistralinstruct | broad | A | 100 | large | 0.05 | Clopper-Pearson | 40000 | 0.0000 | at or under |
| mistralinstruct | broad | A | 100 | large | 0.1 | Wald-t b1 | 40000 | 0.0000 | at or under |
| mistralinstruct | broad | A | 100 | large | 0.1 | b1w | 40000 | 0.0000 | at or under |
| mistralinstruct | broad | A | 100 | large | 0.1 | pooled Wilson | 40000 | 0.0000 | at or under |
| mistralinstruct | broad | A | 100 | large | 0.1 | Clopper-Pearson | 40000 | 0.0000 | at or under |
| mistralguard | strict | B | 50 | fixed | 0.05 | Wald-t b1 | 20000 | 0.0277 | at or under |
| mistralguard | strict | B | 50 | fixed | 0.05 | b1w | 20000 | 0.0277 | at or under |
| mistralguard | strict | B | 50 | fixed | 0.05 | pooled Wilson | 20000 | 0.0262 | at or under |
| mistralguard | strict | B | 50 | fixed | 0.05 | Clopper-Pearson | 20000 | 0.0262 | at or under |
| mistralguard | strict | B | 50 | fixed | 0.1 | Wald-t b1 | 20000 | 0.0277 | at or under |
| mistralguard | strict | B | 50 | fixed | 0.1 | b1w | 20000 | 0.1087 | over |
| mistralguard | strict | B | 50 | fixed | 0.1 | pooled Wilson | 20000 | 0.0994 | at or under |
| mistralguard | strict | B | 50 | fixed | 0.1 | Clopper-Pearson | 20000 | 0.0262 | at or under |
| mistralguard | strict | B | 50 | large | 0.05 | Wald-t b1 | 40000 | 0.0376 | at or under |
| mistralguard | strict | B | 50 | large | 0.05 | b1w | 40000 | 0.0376 | at or under |
| mistralguard | strict | B | 50 | large | 0.05 | pooled Wilson | 40000 | 0.0415 | at or under |
| mistralguard | strict | B | 50 | large | 0.05 | Clopper-Pearson | 40000 | 0.0415 | at or under |
| mistralguard | strict | B | 50 | large | 0.1 | Wald-t b1 | 40000 | 0.0376 | at or under |
| mistralguard | strict | B | 50 | large | 0.1 | b1w | 40000 | 0.1282 | over |
| mistralguard | strict | B | 50 | large | 0.1 | pooled Wilson | 40000 | 0.1310 | over |
| mistralguard | strict | B | 50 | large | 0.1 | Clopper-Pearson | 40000 | 0.0415 | at or under |
| mistralguard | strict | B | 100 | fixed | 0.05 | Wald-t b1 | 20000 | 0.0110 | at or under |
| mistralguard | strict | B | 100 | fixed | 0.05 | b1w | 20000 | 0.0110 | at or under |
| mistralguard | strict | B | 100 | fixed | 0.05 | pooled Wilson | 20000 | 0.0088 | at or under |
| mistralguard | strict | B | 100 | fixed | 0.05 | Clopper-Pearson | 20000 | 0.0088 | at or under |
| mistralguard | strict | B | 100 | fixed | 0.1 | Wald-t b1 | 20000 | 0.0378 | at or under |
| mistralguard | strict | B | 100 | fixed | 0.1 | b1w | 20000 | 0.0378 | at or under |
| mistralguard | strict | B | 100 | fixed | 0.1 | pooled Wilson | 20000 | 0.0315 | at or under |
| mistralguard | strict | B | 100 | fixed | 0.1 | Clopper-Pearson | 20000 | 0.0315 | at or under |
| mistralguard | strict | B | 100 | large | 0.05 | Wald-t b1 | 40000 | 0.0307 | at or under |
| mistralguard | strict | B | 100 | large | 0.05 | b1w | 40000 | 0.0311 | at or under |
| mistralguard | strict | B | 100 | large | 0.05 | pooled Wilson | 40000 | 0.0319 | at or under |
| mistralguard | strict | B | 100 | large | 0.05 | Clopper-Pearson | 40000 | 0.0319 | at or under |
| mistralguard | strict | B | 100 | large | 0.1 | Wald-t b1 | 40000 | 0.0760 | at or under |
| mistralguard | strict | B | 100 | large | 0.1 | b1w | 40000 | 0.0762 | at or under |
| mistralguard | strict | B | 100 | large | 0.1 | pooled Wilson | 40000 | 0.0734 | at or under |
| mistralguard | strict | B | 100 | large | 0.1 | Clopper-Pearson | 40000 | 0.0734 | at or under |
| mistralguard | strict | A | 50 | fixed | 0.05 | Wald-t b1 | 20000 | 0.0276 | at or under |
| mistralguard | strict | A | 50 | fixed | 0.05 | b1w | 20000 | 0.0276 | at or under |
| mistralguard | strict | A | 50 | fixed | 0.05 | pooled Wilson | 20000 | 0.0258 | at or under |
| mistralguard | strict | A | 50 | fixed | 0.05 | Clopper-Pearson | 20000 | 0.0258 | at or under |
| mistralguard | strict | A | 50 | fixed | 0.1 | Wald-t b1 | 20000 | 0.0276 | at or under |
| mistralguard | strict | A | 50 | fixed | 0.1 | b1w | 20000 | 0.1076 | over |
| mistralguard | strict | A | 50 | fixed | 0.1 | pooled Wilson | 20000 | 0.0993 | at or under |
| mistralguard | strict | A | 50 | fixed | 0.1 | Clopper-Pearson | 20000 | 0.0258 | at or under |
| mistralguard | strict | A | 50 | large | 0.05 | Wald-t b1 | 40000 | 0.0403 | at or under |
| mistralguard | strict | A | 50 | large | 0.05 | b1w | 40000 | 0.0403 | at or under |
| mistralguard | strict | A | 50 | large | 0.05 | pooled Wilson | 40000 | 0.0402 | at or under |
| mistralguard | strict | A | 50 | large | 0.05 | Clopper-Pearson | 40000 | 0.0402 | at or under |
| mistralguard | strict | A | 50 | large | 0.1 | Wald-t b1 | 40000 | 0.0403 | at or under |
| mistralguard | strict | A | 50 | large | 0.1 | b1w | 40000 | 0.1299 | over |
| mistralguard | strict | A | 50 | large | 0.1 | pooled Wilson | 40000 | 0.1295 | over |
| mistralguard | strict | A | 50 | large | 0.1 | Clopper-Pearson | 40000 | 0.0402 | at or under |
| mistralguard | strict | A | 100 | fixed | 0.05 | Wald-t b1 | 20000 | 0.0127 | at or under |
| mistralguard | strict | A | 100 | fixed | 0.05 | b1w | 20000 | 0.0127 | at or under |
| mistralguard | strict | A | 100 | fixed | 0.05 | pooled Wilson | 20000 | 0.0115 | at or under |
| mistralguard | strict | A | 100 | fixed | 0.05 | Clopper-Pearson | 20000 | 0.0115 | at or under |
| mistralguard | strict | A | 100 | fixed | 0.1 | Wald-t b1 | 20000 | 0.0403 | at or under |
| mistralguard | strict | A | 100 | fixed | 0.1 | b1w | 20000 | 0.0403 | at or under |
| mistralguard | strict | A | 100 | fixed | 0.1 | pooled Wilson | 20000 | 0.0332 | at or under |
| mistralguard | strict | A | 100 | fixed | 0.1 | Clopper-Pearson | 20000 | 0.0332 | at or under |
| mistralguard | strict | A | 100 | large | 0.05 | Wald-t b1 | 40000 | 0.0321 | at or under |
| mistralguard | strict | A | 100 | large | 0.05 | b1w | 40000 | 0.0322 | at or under |
| mistralguard | strict | A | 100 | large | 0.05 | pooled Wilson | 40000 | 0.0309 | at or under |
| mistralguard | strict | A | 100 | large | 0.05 | Clopper-Pearson | 40000 | 0.0309 | at or under |
| mistralguard | strict | A | 100 | large | 0.1 | Wald-t b1 | 40000 | 0.0766 | at or under |
| mistralguard | strict | A | 100 | large | 0.1 | b1w | 40000 | 0.0767 | at or under |
| mistralguard | strict | A | 100 | large | 0.1 | pooled Wilson | 40000 | 0.0729 | at or under |
| mistralguard | strict | A | 100 | large | 0.1 | Clopper-Pearson | 40000 | 0.0729 | at or under |
| mistralguard | broad | B | 50 | fixed | 0.05 | Wald-t b1 | 20000 | 0.0159 | at or under |
| mistralguard | broad | B | 50 | fixed | 0.05 | b1w | 20000 | 0.0278 | at or under |
| mistralguard | broad | B | 50 | fixed | 0.05 | pooled Wilson | 20000 | 0.0177 | at or under |
| mistralguard | broad | B | 50 | fixed | 0.05 | Clopper-Pearson | 20000 | 0.0177 | at or under |
| mistralguard | broad | B | 50 | fixed | 0.1 | Wald-t b1 | 20000 | 0.0433 | at or under |
| mistralguard | broad | B | 50 | fixed | 0.1 | b1w | 20000 | 0.0825 | at or under |
| mistralguard | broad | B | 50 | fixed | 0.1 | pooled Wilson | 20000 | 0.0510 | at or under |
| mistralguard | broad | B | 50 | fixed | 0.1 | Clopper-Pearson | 20000 | 0.0510 | at or under |
| mistralguard | broad | B | 50 | large | 0.05 | Wald-t b1 | 40000 | 0.0243 | at or under |
| mistralguard | broad | B | 50 | large | 0.05 | b1w | 40000 | 0.0397 | at or under |
| mistralguard | broad | B | 50 | large | 0.05 | pooled Wilson | 40000 | 0.0295 | at or under |
| mistralguard | broad | B | 50 | large | 0.05 | Clopper-Pearson | 40000 | 0.0295 | at or under |
| mistralguard | broad | B | 50 | large | 0.1 | Wald-t b1 | 40000 | 0.0585 | at or under |
| mistralguard | broad | B | 50 | large | 0.1 | b1w | 40000 | 0.1003 | unresolved |
| mistralguard | broad | B | 50 | large | 0.1 | pooled Wilson | 40000 | 0.0735 | at or under |
| mistralguard | broad | B | 50 | large | 0.1 | Clopper-Pearson | 40000 | 0.0735 | at or under |
| mistralguard | broad | B | 100 | fixed | 0.05 | Wald-t b1 | 20000 | 0.0142 | at or under |
| mistralguard | broad | B | 100 | fixed | 0.05 | b1w | 20000 | 0.0142 | at or under |
| mistralguard | broad | B | 100 | fixed | 0.05 | pooled Wilson | 20000 | 0.0179 | at or under |
| mistralguard | broad | B | 100 | fixed | 0.05 | Clopper-Pearson | 20000 | 0.0179 | at or under |
| mistralguard | broad | B | 100 | fixed | 0.1 | Wald-t b1 | 20000 | 0.0358 | at or under |
| mistralguard | broad | B | 100 | fixed | 0.1 | b1w | 20000 | 0.0409 | at or under |
| mistralguard | broad | B | 100 | fixed | 0.1 | pooled Wilson | 20000 | 0.0406 | at or under |
| mistralguard | broad | B | 100 | fixed | 0.1 | Clopper-Pearson | 20000 | 0.0406 | at or under |
| mistralguard | broad | B | 100 | large | 0.05 | Wald-t b1 | 40000 | 0.0390 | at or under |
| mistralguard | broad | B | 100 | large | 0.05 | b1w | 40000 | 0.0390 | at or under |
| mistralguard | broad | B | 100 | large | 0.05 | pooled Wilson | 40000 | 0.0482 | at or under |
| mistralguard | broad | B | 100 | large | 0.05 | Clopper-Pearson | 40000 | 0.0482 | at or under |
| mistralguard | broad | B | 100 | large | 0.1 | Wald-t b1 | 40000 | 0.0782 | at or under |
| mistralguard | broad | B | 100 | large | 0.1 | b1w | 40000 | 0.0874 | at or under |
| mistralguard | broad | B | 100 | large | 0.1 | pooled Wilson | 40000 | 0.0824 | at or under |
| mistralguard | broad | B | 100 | large | 0.1 | Clopper-Pearson | 40000 | 0.0824 | at or under |
| mistralguard | broad | A | 50 | fixed | 0.05 | Wald-t b1 | 20000 | 0.0167 | at or under |
| mistralguard | broad | A | 50 | fixed | 0.05 | b1w | 20000 | 0.0267 | at or under |
| mistralguard | broad | A | 50 | fixed | 0.05 | pooled Wilson | 20000 | 0.0194 | at or under |
| mistralguard | broad | A | 50 | fixed | 0.05 | Clopper-Pearson | 20000 | 0.0194 | at or under |
| mistralguard | broad | A | 50 | fixed | 0.1 | Wald-t b1 | 20000 | 0.0456 | at or under |
| mistralguard | broad | A | 50 | fixed | 0.1 | b1w | 20000 | 0.0761 | at or under |
| mistralguard | broad | A | 50 | fixed | 0.1 | pooled Wilson | 20000 | 0.0520 | at or under |
| mistralguard | broad | A | 50 | fixed | 0.1 | Clopper-Pearson | 20000 | 0.0520 | at or under |
| mistralguard | broad | A | 50 | large | 0.05 | Wald-t b1 | 40000 | 0.0253 | at or under |
| mistralguard | broad | A | 50 | large | 0.05 | b1w | 40000 | 0.0378 | at or under |
| mistralguard | broad | A | 50 | large | 0.05 | pooled Wilson | 40000 | 0.0298 | at or under |
| mistralguard | broad | A | 50 | large | 0.05 | Clopper-Pearson | 40000 | 0.0298 | at or under |
| mistralguard | broad | A | 50 | large | 0.1 | Wald-t b1 | 40000 | 0.0642 | at or under |
| mistralguard | broad | A | 50 | large | 0.1 | b1w | 40000 | 0.0996 | at or under |
| mistralguard | broad | A | 50 | large | 0.1 | pooled Wilson | 40000 | 0.0725 | at or under |
| mistralguard | broad | A | 50 | large | 0.1 | Clopper-Pearson | 40000 | 0.0725 | at or under |
| mistralguard | broad | A | 100 | fixed | 0.05 | Wald-t b1 | 20000 | 0.0138 | at or under |
| mistralguard | broad | A | 100 | fixed | 0.05 | b1w | 20000 | 0.0138 | at or under |
| mistralguard | broad | A | 100 | fixed | 0.05 | pooled Wilson | 20000 | 0.0177 | at or under |
| mistralguard | broad | A | 100 | fixed | 0.05 | Clopper-Pearson | 20000 | 0.0177 | at or under |
| mistralguard | broad | A | 100 | fixed | 0.1 | Wald-t b1 | 20000 | 0.0370 | at or under |
| mistralguard | broad | A | 100 | fixed | 0.1 | b1w | 20000 | 0.0390 | at or under |
| mistralguard | broad | A | 100 | fixed | 0.1 | pooled Wilson | 20000 | 0.0361 | at or under |
| mistralguard | broad | A | 100 | fixed | 0.1 | Clopper-Pearson | 20000 | 0.0361 | at or under |
| mistralguard | broad | A | 100 | large | 0.05 | Wald-t b1 | 40000 | 0.0421 | at or under |
| mistralguard | broad | A | 100 | large | 0.05 | b1w | 40000 | 0.0421 | at or under |
| mistralguard | broad | A | 100 | large | 0.05 | pooled Wilson | 40000 | 0.0475 | at or under |
| mistralguard | broad | A | 100 | large | 0.05 | Clopper-Pearson | 40000 | 0.0475 | at or under |
| mistralguard | broad | A | 100 | large | 0.1 | Wald-t b1 | 40000 | 0.0795 | at or under |
| mistralguard | broad | A | 100 | large | 0.1 | b1w | 40000 | 0.0842 | at or under |
| mistralguard | broad | A | 100 | large | 0.1 | pooled Wilson | 40000 | 0.0828 | at or under |
| mistralguard | broad | A | 100 | large | 0.1 | Clopper-Pearson | 40000 | 0.0828 | at or under |
