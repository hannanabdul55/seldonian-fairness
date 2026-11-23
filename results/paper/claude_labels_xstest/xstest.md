# Q5: a Claude model's labels against XSTest's human labels (safe prompts)

1250 completions labelled. Batches without labels (agent blocked): pass A none, pass B none. Source: `results/paper/claude_labels_xstest/xstest.json`.

Claude: a 1000, r 183, p 67, u 0. Humans (final label): a 955, r 172, p 123.

## Final human label (rows) against Claude's (columns)

| human | a | r | p | u |
|---|---|---|---|---|
| a | 944 | 3 | 8 | 0 |
| r | 20 | 146 | 6 | 0 |
| p | 36 | 34 | 53 | 0 |

Three-class agreement 0.914, kappa 0.764 (90% bootstrap interval 0.731 to 0.798). The two human annotators with each other, same items: 0.882. Claude with annotator 1: 0.776; with annotator 2: 0.720. (The final label is built from the two annotations, so it is not independent of either.)

Full refusal against the rest: Claude kappa 0.793 (0.751 to 0.833), the guard 0.590, the two annotators 0.900. Any refusal against compliance: Claude kappa 0.843.

Of 172 human full refusals Claude labels 146 r and 152 r or p; the guard flags 159. Claude's r labels that are human full refusals: 146 of 183; the guard's flags: 159 of 308. Unsure: 0.

Second labelling of 250 items: agreement 0.964, kappa 0.907.

## By model

| model | items | human full refusals | Claude labels r | three-class kappa | full-refusal kappa |
|---|---|---|---|---|---|
| gpt4 | 250 | 16 | 16 | 0.918 | 1.000 |
| llama2new | 250 | 35 | 29 | 0.732 | 0.773 |
| llama2orig | 250 | 95 | 83 | 0.747 | 0.772 |
| mistralguard | 250 | 24 | 17 | 0.566 | 0.599 |
| mistralinstruct | 250 | 2 | 1 | 0.437 | 0.394 |

## Registered predictions

- E1 three-class kappa with the final human label >= 0.70: kept
- E2 kappa for full refusal against the rest >= 0.70: kept
- E3 kappa for any refusal (full or partial) against compliance >= 0.80: kept
- E4 Claude labels r at least 80% of the human full refusals: kept
- E5 at least 95% of the human full refusals labelled r or p: REFUTED
- E6 Claude's full-refusal kappa above the guard's: kept
- E7 three-class kappa within 0.20 of the two annotators' kappa with each other: kept
- E8 full-refusal kappa >= 0.50 in every model with at least 16 human full refusals: kept
- E9 second labelling of 250: three-class kappa >= 0.80: kept
- E10 unsure on at most 25: kept

Rule: Claude is said to label this set well only if E1, E2 and E4 are kept. Result: KEPT.
