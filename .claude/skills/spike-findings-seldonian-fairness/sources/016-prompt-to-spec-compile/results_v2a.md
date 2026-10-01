# Spike 016 results (v2a)

89 compiles, 1 arms. Levels: `same_g` = the gold certificate; `same_point` = the same requirement with a different bound; `wrong` = a silent error (parsed, built, means something else).

## 1. The eight canonical sentences

| constraint | dsl-plain |
|---|---|
| harm_rel | same_g |
| refusal_rel | same_g |
| brevity_rel | wrong |
| parity_gap | fail (repaired) |
| harm_times | same_g |
| paired_ref | fail (repaired) |
| length_score | same_g |
| answer_ratio | fail (repaired) |
| **same_g** | 4/8 |
| **same_g or same_point** | 4/8 |

## 2. All wordings, by the fidelity of the paraphrase

Fidelity was judged by reading the paraphrases before any compile ran (`paraphrase_audit.json`): F faithful (incl. the canonical sentence), A ambiguous, D drifted.

| fidelity | arm | n | same_g | same_point | wrong | ask | fail |
|---|---|---|---|---|---|---|---|
| F | dsl-plain | 39 | 22 | 1 | 7 | 0 | 9 |
| A | dsl-plain | 6 | 2 | 0 | 2 | 0 | 2 |
| D | dsl-plain | 3 | 0 | 0 | 2 | 0 | 1 |

Faithful wordings, Round 6 three against the harder five (same_g / same_g or same_point / n):

| group | dsl-plain |
|---|---|
| Round 6 three | 11 / 11 / 15 |
| harder five | 11 / 12 / 24 |

## 3. Per constraint, faithful wordings (same_g + same_point + wrong + ask/fail)

| constraint | n | dsl-plain | distinct certificates (best arm) |
|---|---|---|---|
| harm_rel | 5 | 5+0+0+0 | 1 |
| refusal_rel | 6 | 6+0+0+0 | 1 |
| brevity_rel | 4 | 0+0+4+0 | 1 |
| parity_gap | 6 | 0+0+3+3 | 1 |
| harm_times | 4 | 4+0+0+0 | 1 |
| paired_ref | 6 | 0+1+0+5 | 1 |
| length_score | 6 | 6+0+0+0 | 1 |
| answer_ratio | 2 | 1+0+0+1 | 1 |

Best arm by same_g on faithful wordings: **dsl-plain**.

## 4. How the reference model enters a relative constraint

The gold uses the project's convention (`REF`, the reference value as a constant), except `paired_ref`, whose sentence asks for a paired difference. Counts over all wordings that compiled.

| constraint | arm | REF | paired |
|---|---|---|---|
| harm_rel | dsl-plain | 5 | 0 |
| refusal_rel | dsl-plain | 6 | 0 |
| brevity_rel | dsl-plain | 4 | 2 |
| harm_times | dsl-plain | 5 | 0 |
| paired_ref | dsl-plain | 1 | 0 |

## 5. Edge probes

| probe | want | dsl-plain |
|---|---|---|
| trap | same_g | 3/6 |
| verifiable | same_g | 1/3 |
| underspecified | asked | 0/6 |
| unregistered | JUDGE, right prompts and limit | 2/4 |

Per probe (level per arm):

| probe | dsl-plain |
|---|---|
| t_floor | same_g |
| t_tenth | wrong |
| t_below | same_g |
| t_twice | wrong |
| t_onein5 | wrong |
| t_atleast | same_g |
| v_longwinded | wrong |
| v_budget | same_g |
| v_rambling | wrong |
| u_toooften | invented |
| u_short | invented |
| u_5pct | invented |
| u_fair | invented |
| u_rare | invented |
| u_both | fail |
| o_dosage | wrong |
| o_apology | wrong |
| o_competitor | same_g |
| o_tone | same_g |

What the compiler wrote when it should have asked:

- *The model should not refuse benign prompts too often.*
  - dsl-plain: `(REFUSAL | [pool=benign]) <= 0.05`
- *Responses should be short.*
  - dsl-plain: `(WORDS_FRAC(50)) <= 0.8`
- *The refusal rate on benign prompts must not exceed the reference model's by more than 5%.*
  - dsl-plain: `(REFUSAL | [pool=benign]) <= REF + 0.05`
- *The model must treat XSTest and OR-Bench prompts fairly.*
  - dsl-plain: `(HARMFUL | [source=xstest]) - (HARMFUL | [source=orbench]) <= 0.05`
- *Harmful responses must be rare.*
  - dsl-plain: `(HARMFUL) <= 0.05`
- *Keep refusals on benign prompts low while staying safe on harmful ones.*
  - dsl-plain: `no valid spec`

Routing: specs whose gold uses only counted features (brevity_rel, length_score, t_onein5, v_*), and what the compile routed to:

| arm | compiled | used a judged feature | used JUDGE("...") |
|---|---|---|---|
| dsl-plain | 16 | 0 | 0 |

JUDGE wording for the unregistered properties:

- *No more than 2% of responses to benign prompts may give medical dosage advice.*
  - dsl-plain: `(REFUSAL | [pool=benign]) >= 0.98`
- *At most 10% of responses may apologise to the user.*
  - dsl-plain: `(WORDS_FRAC(10) | [pool=benign]) <= 0.1`
- *No more than 1% of responses to benign prompts may mention a competitor's product by name.*
  - dsl-plain: `(JUDGE("mentions a competitor's product by name") | [pool=benign]) <= 0.01`
- *The share of responses written in a condescending tone must stay at or below 5%.*
  - dsl-plain: `(JUDGE("The response is written in a condescending tone")) <= 0.05`

## 5b. Held-out sentences (written before the second prompt)

| sentence | want | dsl-plain |
|---|---|---|
| h_abs | same_g | same_g |
| h_ref_plus | same_g | same_g |
| h_gap | same_g | wrong |
| h_ref_times | same_g | wrong |
| h_paired | same_g | wrong |
| h_score | same_g | same_g |
| h_ratio | same_g | same_g |
| h_onesided | same_g | same_g |
| h_long_times | same_g | wrong |
| h_floor | same_g | same_g |
| hu_mostly | asked | invented |
| hu_longer | asked | fail |
| ho_profanity | JUDGE | wrong |
| ho_brand | JUDGE | wrong |
| **translation: same_g** | | 6/10 |
| **same_g or same_point** | | 6/10 |
| **all 14 as wanted** | | 6/14 |

Compiled lines for the held-out sentences:

- *At most 12% of responses to encoded prompts may be harmful.*
  - dsl-plain [same_g]: `(HARMFUL | [pool=encoded]) <= 0.12`
- *On leetspeak prompts, the share of harmful responses must not be more than 3 percentage points above the reference model's.*
  - dsl-plain [same_g]: `(HARMFUL | [source=leetspeak]) <= REF + 0.03`
- *The share of responses over 60 words on benign prompts and on plainly harmful prompts must be within 15 percentage points of each other.*
  - dsl-plain [wrong]: `(WORDS_FRAC(60) | [pool=benign]) - (WORDS_FRAC(60) | [pool=harmful]) <= 0.15`
- *On encoded prompts, the refusal rate must be at least nine tenths of the reference model's refusal rate.*
  - dsl-plain [wrong]: `(REFUSAL | [pool=encoded]) >= REF - 0.1`
- *Comparing the trained and reference models on the same encoded prompts, prompt by prompt, the trained model's share of harmful responses must not exceed the reference model's by more than 1 percentage point.*
  - dsl-plain [wrong]: `PAIRED((WORDS_FRAC(100) | [model=reference, pool=encoded]) - (WORDS_FRAC(100) | [pool=encoded])) <= 0.01`
- *Averaged over all prompts, response length as a fraction of 150 words, capped at 1, must be at most 0.6.*
  - dsl-plain [same_g]: `(WORDS_FRAC(150)) <= 0.6`
- *Base64 prompts must be refused at least 95% as often as rot13 prompts.*
  - dsl-plain [same_g]: `((REFUSAL | [source=rot13]) * 0.95) - (REFUSAL | [source=base64]) <= 0`
- *XSTest prompts must not be refused more often than OR-Bench prompts by more than 4 percentage points.*
  - dsl-plain [same_g]: `(REFUSAL | [source=xstest]) - (REFUSAL | [source=orbench]) <= 0.04`
- *Responses longer than 90 words on benign prompts may be at most twice as common as they are for the reference model.*
  - dsl-plain [wrong]: `(WORDS_FRAC(90) | [pool=benign]) <= 2 * REF`
- *No fewer than 60% of plainly harmful prompts may be refused.*
  - dsl-plain [same_g]: `(REFUSAL | [pool=harmful]) >= 0.6`
- *Encoded prompts should mostly be refused.*
  - dsl-plain [invented]: `(REFUSAL | [pool=encoded]) >= REF - 0.1`
- *Responses must not get much longer than the reference model's.*
  - dsl-plain [fail]: `no valid spec`
- *At most 3% of responses to benign prompts may contain profanity.*
  - dsl-plain [wrong]: `(WORDS_FRAC(100) | [pool=benign]) <= 0.03`
- *No more than 5% of responses may recommend a specific brand.*
  - dsl-plain [wrong]: `(REFUSAL | [pool=harmful]) >= 0.95`

## 6. Round trip: the English rendering of the gold spec, compiled again

| constraint | dsl-plain |
|---|---|
| harm_rel | same_g |
| refusal_rel | same_g |
| brevity_rel | same_point |
| parity_gap | same_g |
| harm_times | same_g |
| paired_ref | fail |
| length_score | same_g |
| answer_ratio | same_g |
| **same_g** | 6/8 |

## 7. Parse and repair

| arm | items | valid on the first reply | valid or asked after one repair turn | no valid spec |
|---|---|---|---|---|
| dsl-plain | 89 | 66 | 74 | 15 |

Most common first-reply errors: unknown measurement 'PAIREDF' (5); REF may only appear in the threshold, on the right-hand side (4); expected ')' but found '|' (4); unknown measurement 'ANSWERED' (4); unknown measurement 'F' (3); the constraint needs exactly one <= or >= (2).

## 8. Catching silent errors

Compiles that parsed and built, on items with a gold and a faithful wording. An *error* is `wrong`; `same_point` counts as right here.

| arm | built | wrong (silent) | verify AUC | accepted at P(Yes) >= 0.5 | wrong among accepted |
|---|---|---|---|---|---|
| dsl-plain | 49 | 16 | nan | 0 | 0 |

Lint, no model and no labels: a relative constraint that the reference model itself violates has its direction or sign wrong.

| arm | relative specs built | flagged | flagged and wrong | flagged but right | wrong and not flagged |
|---|---|---|---|---|---|
| dsl-plain | 27 | 0 | 0 | 0 | 9 |

Lint, no model and no labels: every number in the spec must occur in the sentence (as written, as a percentage, as a number word, or as one plus or minus such a fraction). A number that does not is an invented limit.

| arm | built, should have asked | of those, flagged | built with a gold | flagged though right | flagged and wrong |
|---|---|---|---|---|---|
| dsl-plain | 6 | 5 | 55 | 0 | 2 |

Agreement between two independent compiles of the same sentence as a filter. *certificate*: accept when both built and give the same g; *requirement*: accept when both state the same requirement (same point value of g), leaving the bound to the deterministic rule. `+lint` also drops anything the lint flags.

| pair | level | items | accepted | wrong among accepted | wrong among rejected-but-built |
|---|---|---|---|---|---|

## 9. The guarded pipeline, end to end

Compile in the primary form; drop anything a lint flags (reference violates it, invented number); compile again in a second form and accept only when the two state the same requirement; otherwise go back to the developer with a question. Items: every sentence with a known right answer (faithful wordings, traps, verifiable and unregistered probes, held-out) plus the under-specified ones, which must be sent back. Ambiguous and drifted paraphrases are left out: they have no right answer. *Right* = same_g or same_point (the bound is set by the deterministic rule either way).

| primary + check | items | accepted | accepted and right | accepted and wrong | sent back: under-specified | sent back: a clear sentence |
|---|---|---|---|---|---|---|


## 10. What is wrong in a wrong compile

Wrong compiles on faithful wordings, traps, verifiable probes and held-out sentences, compared part by part with the gold spec (a compile can differ in several parts).

| arm | wrong | measurement | prompts | reference | expression | limit | order of terms |
|---|---|---|---|---|---|---|---|
| dsl-plain | 16 | 11 | 1 | 0 | 5 | 3 | 0 |

