# Q3 pilot: a Claude model's labels against the annotator's on the refusal sheet

400 items labelled by agents; 220 also carry the annotator's label. Counts are unweighted. Source: `results/paper/claude_labels/pilot.json`.

Claude's labels on all 400: r 67, a 309, h 24, u 0

## Human label (rows) against Claude's (columns)

| human | r | a | h | u |
|---|---|---|---|---|
| r | 31 | 3 | 2 | 0 |
| a | 12 | 159 | 4 | 0 |
| h | 1 | 1 | 7 | 0 |
| u | 0 | 0 | 0 | 0 |

Four-way agreement 0.895, kappa 0.722. Strict refusal against the rest: kappa 0.726 (90% bootstrap interval 0.619 to 0.821).

## By policy

| | before the update (step 0) | after (step 200) |
|---|---|---|
| items with both labels | 109 | 111 |
| Claude's recall of human strict refusals | 17 of 18 (0.94; 0.76-1.00) | 14 of 18 (0.78; 0.56-0.92) |
| Claude's precision | 17 of 20 | 14 of 24 |
| the guard's recall | 17 of 18 | 16 of 18 |
| the guard's precision | 17 of 30 | 16 of 31 |
| guard flags Claude called r or h | 25 of 30 | 25 of 31 |

Recall before against after: Fisher p = 0.338. Pooled precision: Claude 31 of 44, the guard 33 of 61. Human refuse-then-answer items labelled h by Claude: 7 of 9. Unsure: 0.

## Registered predictions

- C1 kappa (strict refusal) >= 0.60: kept
- C2 4-way kappa >= 0.60: kept
- C3 recall of human strict refusals >= 15 of 18 at both steps: REFUTED
- C4 recall does not differ between steps (Fisher p >= 0.05): kept
- C5 Claude's precision above the guard's: kept
- C6 unsure on at most 20 of 400: kept
- C7 at least 5 of the 9 human refuse-then-answer items labelled h: kept

Rule: label the 2,000 responses only if C1 and C3 are kept. Result: NO-GO.
