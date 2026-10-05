# The rubric judge against human refusal labels, before and after constrained training

220 labelled responses (one annotator), scored with spike 015's judge (Qwen3-8B, 4-bit) under its six refusal wordings; a response is judged a refusal when p > 0.5. Gold: the annotator's strict refusal (`r`). Rates are design-weighted; counts are unweighted, with 90% Clopper-Pearson intervals on the counts.

## Recall and false alarms against human strict refusal

| wording | recall, reference (step 0) | recall, trained (step 200) | Fisher p (counts) | false alarms, reference | false alarms, trained |
|---|---|---|---|---|---|
| 0 | 0.21; 3/18 (0.05-0.38) | 0.00; 0/18 (0.00-0.15) | 0.229 | 0.000; 0/91 | 0.000; 0/93 |
| 1 | 0.21; 3/18 (0.05-0.38) | 0.00; 0/18 (0.00-0.15) | 0.229 | 0.000; 0/91 | 0.000; 0/93 |
| 2 | 0.21; 3/18 (0.05-0.38) | 0.00; 0/18 (0.00-0.15) | 0.229 | 0.025; 2/91 | 0.013; 1/93 |
| 3 | 0.32; 5/18 (0.12-0.50) | 0.07; 1/18 (0.00-0.24) | 0.177 | 0.008; 1/91 | 0.027; 3/93 |
| 4 | 0.66; 11/18 (0.39-0.80) | 0.40; 6/18 (0.16-0.55) | 0.181 | 0.027; 4/91 | 0.029; 4/93 |
| 5 | 0.43; 7/18 (0.20-0.61) | 0.40; 6/18 (0.16-0.55) | 1.000 | 0.027; 4/91 | 0.030; 5/93 |

## Recall and false alarms against the guard's flag (same responses)

| wording | recall, reference (step 0) | recall, trained (step 200) | Fisher p (counts) | false alarms, reference | false alarms, trained |
|---|---|---|---|---|---|
| 0 | 0.13; 3/30 (0.03-0.24) | 0.00; 0/31 (0.00-0.09) | 0.113 | 0.000; 0/79 | 0.000; 0/80 |
| 1 | 0.13; 3/30 (0.03-0.24) | 0.00; 0/31 (0.00-0.09) | 0.113 | 0.000; 0/79 | 0.000; 0/80 |
| 2 | 0.13; 3/30 (0.03-0.24) | 0.00; 0/31 (0.00-0.09) | 0.113 | 0.028; 2/79 | 0.014; 1/80 |
| 3 | 0.24; 6/30 (0.09-0.36) | 0.18; 4/31 (0.05-0.27) | 0.508 | 0.000; 0/79 | 0.000; 0/80 |
| 4 | 0.53; 15/30 (0.34-0.66) | 0.39; 10/31 (0.19-0.49) | 0.198 | 0.000; 0/79 | 0.000; 0/80 |
| 5 | 0.39; 11/30 (0.22-0.53) | 0.39; 10/31 (0.19-0.49) | 0.791 | 0.000; 0/79 | 0.001; 1/80 |

## The guard's flag against human strict refusal

| | reference (step 0) | trained (step 200) |
|---|---|---|
| recall | 0.99; 17/18 | 0.98; 16/18 |
| precision | 0.59; 17/30 | 0.60; 16/31 |
