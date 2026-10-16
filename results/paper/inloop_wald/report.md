# O2: the recommended design inside the training loop

Registered in `.planning/paper-certification/O2_registration.md`. 64000 runs; cell rule: over when the miss exceeds delta by more than 2 Monte Carlo SE.

## Predictions

- **Q1** Wald-t, 8 strata, pool rate: over in at most 1 of 8 cells: **kept**
- **Q2** Wald-t, 8 strata, sampled-pool term, source rate: over in at most 2 of 8 cells: **kept**
- **Q3** the same at icc05, delta 0.10: over (the 4-strata in-loop cell was over): **kept**
- **Q4** pooled Wilson on a random split, source rate: over in at most 2 of 8 cells: **kept**
- **Q5** gain of the Wald-t source-rate limit against the pooled Wilson bound, delta 0.10: at least 1.6 at icc75: **kept**
- **Q6** the same gain at icc05 between 0.9 and 1.15: **kept**

## Cells

| env | delta | bound | R | misses | miss | cell | mean width |
|---|---|---|---|---|---|---|---|
| icc05 | 0.05 | Wald-t, 8 strata, pool rate | 8000 | 417 | 0.0521 | unresolved | 0.0282 |
| icc05 | 0.05 | Wald-t, 8 strata, sampled-pool term, source rate | 8000 | 453 | 0.0566 | over | 0.0283 |
| icc05 | 0.05 | b1w, 8 strata, pool rate | 8000 | 336 | 0.0420 | under | 0.0298 |
| icc05 | 0.05 | b1w, 8 strata, sampled-pool term, source rate | 8000 | 374 | 0.0467 | under | 0.0299 |
| icc05 | 0.05 | pooled Wilson, random split, source rate | 8000 | 363 | 0.0454 | under | 0.0302 |
| icc05 | 0.05 | pooled Wilson, random split, pool rate | 8000 | 309 | 0.0386 | under | 0.0302 |
| icc05 | 0.05 | t-test, random split, source rate | 8000 | 508 | 0.0635 | over | 0.0277 |
| icc05 | 0.05 | t-test, 8 strata, source rate | 8000 | 497 | 0.0621 | over | 0.0277 |
| icc05 | 0.1 | Wald-t, 8 strata, pool rate | 8000 | 802 | 0.1003 | unresolved | 0.0219 |
| icc05 | 0.1 | Wald-t, 8 strata, sampled-pool term, source rate | 8000 | 857 | 0.1071 | over | 0.0221 |
| icc05 | 0.1 | b1w, 8 strata, pool rate | 8000 | 736 | 0.0920 | under | 0.0228 |
| icc05 | 0.1 | b1w, 8 strata, sampled-pool term, source rate | 8000 | 775 | 0.0969 | under | 0.0229 |
| icc05 | 0.1 | pooled Wilson, random split, source rate | 8000 | 788 | 0.0985 | under | 0.0231 |
| icc05 | 0.1 | pooled Wilson, random split, pool rate | 8000 | 726 | 0.0907 | under | 0.0231 |
| icc05 | 0.1 | t-test, random split, source rate | 8000 | 933 | 0.1166 | over | 0.0216 |
| icc05 | 0.1 | t-test, 8 strata, source rate | 8000 | 919 | 0.1149 | over | 0.0215 |
| icc26 | 0.05 | Wald-t, 8 strata, pool rate | 8000 | 343 | 0.0429 | under | 0.0304 |
| icc26 | 0.05 | Wald-t, 8 strata, sampled-pool term, source rate | 8000 | 423 | 0.0529 | unresolved | 0.0316 |
| icc26 | 0.05 | b1w, 8 strata, pool rate | 8000 | 299 | 0.0374 | under | 0.0316 |
| icc26 | 0.05 | b1w, 8 strata, sampled-pool term, source rate | 8000 | 381 | 0.0476 | under | 0.0328 |
| icc26 | 0.05 | pooled Wilson, random split, source rate | 8000 | 384 | 0.0480 | under | 0.0344 |
| icc26 | 0.05 | pooled Wilson, random split, pool rate | 8000 | 291 | 0.0364 | under | 0.0344 |
| icc26 | 0.05 | t-test, random split, source rate | 8000 | 482 | 0.0602 | over | 0.0324 |
| icc26 | 0.05 | t-test, 8 strata, source rate | 8000 | 407 | 0.0509 | unresolved | 0.0324 |
| icc26 | 0.1 | Wald-t, 8 strata, pool rate | 8000 | 705 | 0.0881 | under | 0.0237 |
| icc26 | 0.1 | Wald-t, 8 strata, sampled-pool term, source rate | 8000 | 799 | 0.0999 | under | 0.0246 |
| icc26 | 0.1 | b1w, 8 strata, pool rate | 8000 | 652 | 0.0815 | under | 0.0243 |
| icc26 | 0.1 | b1w, 8 strata, sampled-pool term, source rate | 8000 | 756 | 0.0945 | under | 0.0252 |
| icc26 | 0.1 | pooled Wilson, random split, source rate | 8000 | 779 | 0.0974 | under | 0.0265 |
| icc26 | 0.1 | pooled Wilson, random split, pool rate | 8000 | 625 | 0.0781 | under | 0.0265 |
| icc26 | 0.1 | t-test, random split, source rate | 8000 | 885 | 0.1106 | over | 0.0252 |
| icc26 | 0.1 | t-test, 8 strata, source rate | 8000 | 758 | 0.0948 | under | 0.0253 |
| icc55 | 0.05 | Wald-t, 8 strata, pool rate | 8000 | 278 | 0.0348 | under | 0.0285 |
| icc55 | 0.05 | Wald-t, 8 strata, sampled-pool term, source rate | 8000 | 390 | 0.0488 | under | 0.0327 |
| icc55 | 0.05 | b1w, 8 strata, pool rate | 8000 | 254 | 0.0318 | under | 0.0289 |
| icc55 | 0.05 | b1w, 8 strata, sampled-pool term, source rate | 8000 | 368 | 0.0460 | under | 0.0332 |
| icc55 | 0.05 | pooled Wilson, random split, source rate | 8000 | 390 | 0.0488 | under | 0.0386 |
| icc55 | 0.05 | pooled Wilson, random split, pool rate | 8000 | 208 | 0.0260 | under | 0.0386 |
| icc55 | 0.05 | t-test, random split, source rate | 8000 | 459 | 0.0574 | over | 0.0373 |
| icc55 | 0.05 | t-test, 8 strata, source rate | 8000 | 252 | 0.0315 | under | 0.0373 |
| icc55 | 0.1 | Wald-t, 8 strata, pool rate | 8000 | 614 | 0.0767 | under | 0.0221 |
| icc55 | 0.1 | Wald-t, 8 strata, sampled-pool term, source rate | 8000 | 762 | 0.0953 | under | 0.0254 |
| icc55 | 0.1 | b1w, 8 strata, pool rate | 8000 | 598 | 0.0747 | under | 0.0223 |
| icc55 | 0.1 | b1w, 8 strata, sampled-pool term, source rate | 8000 | 743 | 0.0929 | under | 0.0256 |
| icc55 | 0.1 | pooled Wilson, random split, source rate | 8000 | 799 | 0.0999 | under | 0.0298 |
| icc55 | 0.1 | pooled Wilson, random split, pool rate | 8000 | 533 | 0.0666 | under | 0.0298 |
| icc55 | 0.1 | t-test, random split, source rate | 8000 | 848 | 0.1060 | unresolved | 0.0290 |
| icc55 | 0.1 | t-test, 8 strata, source rate | 8000 | 552 | 0.0690 | under | 0.0291 |
| icc75 | 0.05 | Wald-t, 8 strata, pool rate | 8000 | 231 | 0.0289 | under | 0.0227 |
| icc75 | 0.05 | Wald-t, 8 strata, sampled-pool term, source rate | 8000 | 372 | 0.0465 | under | 0.0309 |
| icc75 | 0.05 | b1w, 8 strata, pool rate | 8000 | 238 | 0.0297 | under | 0.0224 |
| icc75 | 0.05 | b1w, 8 strata, sampled-pool term, source rate | 8000 | 354 | 0.0442 | under | 0.0311 |
| icc75 | 0.05 | pooled Wilson, random split, source rate | 8000 | 403 | 0.0504 | unresolved | 0.0401 |
| icc75 | 0.05 | pooled Wilson, random split, pool rate | 8000 | 179 | 0.0224 | under | 0.0401 |
| icc75 | 0.05 | t-test, random split, source rate | 8000 | 439 | 0.0549 | over | 0.0393 |
| icc75 | 0.05 | t-test, 8 strata, source rate | 8000 | 129 | 0.0161 | under | 0.0393 |
| icc75 | 0.1 | Wald-t, 8 strata, pool rate | 8000 | 572 | 0.0715 | under | 0.0176 |
| icc75 | 0.1 | Wald-t, 8 strata, sampled-pool term, source rate | 8000 | 744 | 0.0930 | under | 0.0241 |
| icc75 | 0.1 | b1w, 8 strata, pool rate | 8000 | 613 | 0.0766 | under | 0.0173 |
| icc75 | 0.1 | b1w, 8 strata, sampled-pool term, source rate | 8000 | 741 | 0.0926 | under | 0.0240 |
| icc75 | 0.1 | pooled Wilson, random split, source rate | 8000 | 826 | 0.1032 | unresolved | 0.0311 |
| icc75 | 0.1 | pooled Wilson, random split, pool rate | 8000 | 477 | 0.0596 | under | 0.0311 |
| icc75 | 0.1 | t-test, random split, source rate | 8000 | 869 | 0.1086 | over | 0.0306 |
| icc75 | 0.1 | t-test, 8 strata, source rate | 8000 | 395 | 0.0494 | under | 0.0306 |

## Gain (ESS against the pooled Wilson bound on a random split, by mean width)

| env | delta | rho | source rate | pool rate |
|---|---|---|---|---|
| icc05 | 0.05 | 0.43 | 1.13 | 1.15 |
| icc05 | 0.1 | 0.43 | 1.10 | 1.11 |
| icc26 | 0.05 | 0.72 | 1.19 | 1.28 |
| icc26 | 0.1 | 0.72 | 1.16 | 1.25 |
| icc55 | 0.05 | 0.88 | 1.39 | 1.84 |
| icc55 | 0.1 | 0.88 | 1.38 | 1.82 |
| icc75 | 0.05 | 0.98 | 1.68 | 3.13 |
| icc75 | 0.1 | 0.98 | 1.68 | 3.12 |
