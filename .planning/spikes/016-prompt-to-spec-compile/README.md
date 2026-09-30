---
spike: 016
idea: prompted-ghat
name: prompt-to-spec-compile
type: standard
validates: "Given the three Round 6 constraints plus five harder ones in English, when a local LLM compiles each to a Seldonian-toolkit-style constraint string and JSON spec, rendered back to English, then the compiled g equals the hand-written g on cached responses and paraphrases compile to the same spec"
verdict: PARTIAL
related: [013, 015]
tags: [compiler, constraint-dsl, expression-constraint, gpu]
---

# Spike 016: Can a local LLM compile an English constraint into a spec?

## What This Validates
Spike 015 covered the measurement half of `prompted-ghat` (can a judge be compiled from a
sentence). This spike covers the other half: the **statistic and threshold**. Given a
requirement in English, when Qwen3-8B (4-bit) writes it as one line of a constraint language
or as a JSON spec, and deterministic code builds the project's own constraint classes from
it, then (a) the built `g` equals the hand-written `g` on cached responses, (b) paraphrases
of one requirement compile to the same constraint, (c) the spec renders back to English the
developer can confirm, and (d) verifiable properties go to code, never to a judge (015).

## Pre-registered expectations
Written before any compile ran (the deterministic stage 0 had run).

- **P1** The three Round 6 sentences compile to a g-identical spec in every arm.
- **P2** At least 4 of the 5 harder sentences compile correctly in the best arm. Likeliest
  failures: the paired difference (PAIRED dropped, leaving a two-sample difference) and the
  ratio (direction, or refusal used where the sentence says answered).
- **P3** At least 85% of faithful paraphrases compile to the same certificate in the best
  arm; errors concentrate in the threshold form (points against percent, plus against times).
- **P4** The one-line DSL is at least as accurate as JSON (smaller surface); thinking helps
  on the harder five.
- **P5** No length property is ever routed to `JUDGE`.
- **P6** Without thinking the compiler asks for clarification on fewer than half of the
  under-specified requirements (it invents a limit), and reads "5%" as 5 points silently.
- **P7** Properties outside the registry mostly route to `JUDGE("...")`, with some mapped
  onto `HARMFUL` or `REFUSAL` instead.
- **P8** A wrong compile that parses and builds is the dangerous case. The model's own
  equivalence check of the English rendering will rank such errors (AUC > 0.7) but not
  separate them; agreement between the two surface forms is the better filter.
- **Kill rule:** fewer than 6 of 8 canonical sentences right in the best arm, or silent
  errors on more than 20% of faithful paraphrases with no filter that catches them, means the
  compile step needs a human to write the spec.

## Research
- **Seldonian Toolkit constraint strings** (seldonian.cs.umass.edu, Tutorial D): behavioural
  constraints are written as strings such as
  `abs((FPR | [M]) - (FPR | [F])) + abs((FNR | [M]) - (FNR | [F])) <= 0.2`, parsed into a tree
  over `+ - * /` and `min max abs exp`, with confidence intervals propagated from the leaves.
  The DSL here follows that style, and this repo's own `seldonian/parser.py::parse_ghat` is an
  unfinished attempt at the same thing (it raises `NotImplementedError`).
- **Req2LTL** (arXiv:2512.17334): natural-language requirements to temporal logic through a
  hierarchical intermediate representation, with the LLM producing the intermediate form and
  deterministic rules producing the formula; 88.4% semantic accuracy and 100% syntactic
  correctness on aerospace requirements (author-reported). Same architecture as here: the
  model fills a small inspectable form, code does the rest.
- **Language to Rewards** (arXiv:2306.08647): the inspectable intermediate layer between
  language and an optimiser, which the MANIFEST already requires (render back to English).
- No published work compiles a sentence into a *statistical* constraint with a confidence
  bound (the 2026-09-29 research pass; still true of what this spike's search found).

| Approach | Pros | Cons | Status |
|---|---|---|---|
| LLM writes one DSL line, a parser builds the spec | small surface, one line to read, the parser's errors feed a repair turn | needs a grammar the model can learn from a prompt | **built, arm `dsl`** |
| LLM fills a JSON spec with typed slots | explicit slots (threshold form is an enum) | longer, more places to be inconsistent | **built, arm `json`** |
| LLM writes Python that builds the constraint | no grammar | unverifiable, unsafe to execute, no canonical form | rejected |
| Grammar-constrained decoding | syntactically valid by construction | new dependency; syntax was not the problem (see Results 7) | not built |
| Typed questions, one slot at a time, scored by logits | a confidence per slot | fixed template family; bigger build | follow-up candidate |

## How to Run
    cd .planning/spikes/016-prompt-to-spec-compile
    ../../../.venv/bin/python check_builder.py            # stage 0, CPU, 10 s
    ./run.sh --stage paraphrase                            # GPU, 3 min
    ./run.sh --stage compile,verify --version v1,v2 --arms dsl-plain,json-plain   # GPU, 12 min
    ./run.sh --stage compile --version v2 --arms dsl-think # GPU, 37 to 73 min per thinking arm
    ./run.sh --stage compile --version v2a,v1b --arms dsl-plain   # the ablation, GPU, 5 min
    ../../../.venv/bin/python analyze016.py                # results.md      (prompt v1)
    ../../../.venv/bin/python analyze016.py --tag _v2      # results_v2.md   (also _v2a, _v1b)
    ../../../.venv/bin/python summary016.py                # summary.md, one table for all arms
    ../../../.venv/bin/python make_viewer.py               # then open viewer.html
    ./try.sh                                               # type a sentence, see the constraint

## What to Expect
`check_builder.py` prints a worst difference of `0.00e+00` between the built and the
hand-written g. `viewer.html` shows every sentence with what each arm wrote, the English
restatement, and a badge (same certificate / same requirement / silent error / asked).
`./try.sh` loads the model once and compiles whatever you type in both surface forms, with
the lints and the pass/fail on 013's cached step-200 responses.

## Method
- **Deterministic half (`speclab.py`).** DSL and JSON parse into one canonical spec, always
  `expr <= threshold`. The spec builds `ExpressionConstraint` or `PairedDifferenceConstraint`
  from the project's own classes (`monotone` is derived from the expression, never asked of
  the model), renders to English and back to the DSL, and gets its bound from spike 013's
  use-case map through `013/preflight.py` on the step-0 covariate samples.
- **Data.** Spike 013's cached Granite-3.3-2B responses: benign (XSTest + OR-Bench), plainly
  harmful and encoded harmful prompts, 500 each, 8 responses per prompt at steps 0, 100 and
  200, with Qwen3Guard-4B's cached refusal and harm labels. No judge is loaded. Episodes are
  clustered in prompts, so g here is a fixture for comparing code paths, not a certificate.
- **Hand-written g (`gold.py`).** Each of the eight constraints as a developer writes it
  today: the Round 6 three through `policy.Constraint` and
  `SeldonianLLMPolicy.constraint_values`, the rest through `ExpressionConstraint` and
  `PairedDifferenceConstraint` with a `group` column and a lambda. It shares no code with the
  parser or the builder.
- **Scoring by meaning.** A compiled spec is evaluated on five data sets (steps 100 and 200,
  three 30% prompt sub-samples) beside the gold: `same_g` (identical g, the same
  certificate), `same_point` (identical statistic minus threshold, another bound), `wrong`
  (it parsed and built and means something else: a silent error). Two algebraically
  equivalent golds are accepted for the ratio constraints.
- **Compiler.** Qwen3-8B, 4-bit, as in 015. Arms: `dsl` or `json` surface form, thinking off
  (greedy) or on (sampled at the model card's settings, seed 0, 1800 thinking tokens). One
  repair turn carrying the parser's error message.
- **Items.** 8 constraints x (canonical + 5 model-written paraphrases), audited for fidelity
  before any compile ran (31 faithful, 6 ambiguous, 3 drifted; `paraphrase_audit.json`);
  19 edge probes (6 unit/direction traps, 3 verifiable-in-disguise, 6 under-specified,
  4 outside the registry); the 8 round trips; 14 held-out sentences.
- **Two prompts.** `v1` is the pre-registered one: `[attribute=value]` restrictions over two
  overlapping attributes (`pool`, `source`), three ways to involve the reference model
  (`REF`, `model=reference`, `PAIRED(a - b)`), three examples. `v2` was written after reading
  v1's plain arms: one flat vocabulary of prompt groups, the reference only through `REF`
  (`PAIRED REF + m` for a paired comparison), one example per construct. The 14 held-out
  sentences were written after v1 was read and before v2 was written, and no v2 example uses
  a group, number or measurement pairing that any test sentence uses.

## Investigation Trail
1. **Stage 0, no model.** The builder reproduced the hand-written g to the last bit on all
   8 constraints x 5 data sets (worst difference 0.00e+00), every gold line round-trips
   through `render_dsl`, and the bound rule matched 013's map (exact Clopper-Pearson with no
   strata for the 1% harm rate; reference-rate strata with `b1w` for the mid-rate refusal and
   brevity labels; the betting mixture for the bounded score and the paired difference).
2. **Two side results from stage 0** (Results 2). A ratio and its linear rewrite give
   different g but never a different pass/fail (0 of 300 over a sweep of limits). "Relative
   to the reference" has three formalisations with three different certificates.
3. **Smoke test on 7 items changed the parser, not the prompt.** The model's first instinct
   for a relative limit was `X - REF <= 0.045`, which the first parser rejected. The parser
   now moves a REF term to the right (`E - REF <= m`, `E / REF <= f`); a strict `<` or `>` is
   read as `<=` or `>=`; `model=trained` is dropped as the default. The smoke rows are kept in
   `results/spikes/016/smoke/` and are not part of any table.
4. **The first thinking run was killed after 66 minutes** with nothing saved (one write per
   arm, 2,500 thinking tokens, batch 6 at 10.7 of 12 GB). Thinking arms now save every 12
   items, think for at most 1,800 tokens and run at batch 4: about 40 minutes per arm.
5. **v1's plain arms were read, and the kill rule had fired** (Results 3). Reading the wrong
   compiles showed three causes that belong to the interface rather than to the model:
   `pool` and `source` overlap (`source=plain` is `pool=harmful`), so the model added
   restrictions nobody asked for and wrote `pool=plain`; the reference model could enter
   three ways and the model mixed them (`a - a_ref <= REF + 0.045`); and with three examples
   it anchored on the one two-group difference it had been shown.
6. **Held-out sentences were written next, then prompt v2.** 14 fresh sentences first, so the
   revised prompt is scored on sentences it was not revised against; then v2 (flat prompt
   groups, REF only, one example per construct). Two v2 examples were reworded before any v2
   run because they shared a phrase with a test sentence.
7. **One more parser freeze after v2's first plain run.** `>= PAIRED REF - 0.02` had been
   rejected with a misleading message (the paired conversion ran after the `>=` flip), and
   the JSON arm's `abs(t - r)` over two identical measures built a constraint that is
   identically zero. The conversion now runs first, and a measurement minus itself is a
   validation error. All four plain arms (v1 and v2) were then re-run from scratch with the
   frozen parser; the pre-freeze rows are in `smoke/`.
8. **Three guards were added after reading v2's residual errors**, all deterministic and
   all scored on the same compiles: a lint that a relative constraint must be one the
   reference model itself satisfies; a lint that every number in the spec must occur in the
   sentence; and agreement between two independent compiles at the level of the requirement.
9. **Ablation.** v2 changed the registry and the examples together, so the DSL plain arm was
   also run with v2's registry and v1's three examples (`v2a`) and with v1's registry and
   v2's ten examples (`v1b`).

## Results
**Verdict: PARTIAL.** The deterministic half is exact and ready to build. The LLM half
failed as first designed (the kill rule fired) and works after the interface was redesigned,
but not alone: it needs three deterministic guards and a second compile, and with them it
still sends about a quarter of clear sentences back to the developer. Full tables:
`summary.md`, `results.md` (v1), `results_v2.md`, `results_v1b.md`, `results_v2a.md`.

### 1. The deterministic half is exact
Built from the gold line, every one of the 8 constraints reproduces the hand-written g on all
5 data sets with a worst difference of 0.00e+00: the three Round 6 constraints through
`policy.Constraint`, the parity gap, the multiplicative threshold, the paired difference, the
bounded score and the ratio. The model never chooses `monotone`, the delta split or the
bound; all three are derived. So everything the LLM has to get right is one line.

### 2. Two facts about the constraint language itself (no model involved)
- **An algebraic rewrite changes g but never the decision.** `a / b >= 0.8` and
  `a - 0.8 b >= 0` gave g of -0.166 and -0.141 on the same data, and the same pass/fail in
  300 of 300 cases over a sweep of limits across the boundary. Two compiles that differ only
  in this way certify the same thing; only the scale a Lagrangian would see differs.
- **"Relative to the reference" is three different certificates.** Benign refusal, "at most
  2 points above the reference", step 200, one response per prompt (n = 500):

  | formalisation | bound width | g | passes |
  |---|---|---|---|
  | reference value as a constant (`REF + 0.02`, what Round 6 did) | 0.029 | -0.017 | yes |
  | two-sample difference | 0.077 | +0.031 | no |
  | paired difference on the same prompts | 0.039 | -0.008 | yes |

  The first ignores the reference's own sampling error, so its pass is not a certificate of
  the sentence; the paired form covers it at 1.3x the width, the two-sample form at 2.6x.
  Which one is meant is a statistical design decision, and the sentence rarely says.

### 3. Headline: eight arms
`same_g` = the gold certificate; `req` = same_g or same_point (the same requirement, bound
left to the rule); `silent` = parsed, built, and wrong.

| prompt | arm | canonical 8: same_g / req | faithful wordings (39): same_g / req / silent / no spec | held-out 10 | traps 6 | under-specified 8: asked | outside registry 6 | valid first reply (89) |
|---|---|---|---|---|---|---|---|---|
| v1 | dsl-plain | 3 / 4 | 13 / 17 / 20 / 2 | 5 | 1 | 0 | 1 | 58 |
| v1 | json-plain | 1 / 1 | 8 / 13 / 20 / 6 | 6 | 1 | 0 | 1 | 59 |
| v1 | dsl-think | 5 / 6 | 24 / 27 / 7 / 5 | 9 | 5 | 5 | 5 | 59 |
| v1b (v1 registry, 10 examples) | dsl-plain | 5 / 6 | 20 / 25 / 13 / 1 | 6 | 5 | 3 | 3 | 74 |
| v2a (v2 registry, 3 examples) | dsl-plain | 4 / 4 | 22 / 23 / 7 / 9 | 6 | 3 | 0 | 2 | 66 |
| v2 | dsl-plain | 6 / 7 | 29 / 31 / 4 / 4 | 9 | 6 | 2 | 3 | 81 |
| v2 | json-plain | 4 / 7 | 23 / 33 / 6 / 0 | 6 | 3 | 3 | 5 | 74 |
| **v2** | **dsl-think** | **7 / 8** | **34 / 36 / 0 / 3** | 8 | 5 | **7** | 4 | 79 |

- **v1, as pre-registered, is unusable**: 3 of 8 canonical sentences, and a silent error on
  20 of 39 faithful wordings in both plain arms. The kill rule fired.
- **v2 with thinking**: 7 of 8 canonical sentences give the gold certificate and all 8 the
  gold requirement; 34 of 39 faithful wordings the gold certificate, none silently wrong.
- **Both changes matter, for different things.** The registry alone (v2a) cut silent errors
  from 20 to 7 and did nothing for asking (0 of 8). The examples alone (v1b) fixed the traps
  (1 to 5 of 6) and produced the first clarifying questions (3 of 8) but left 13 silent
  errors. Neither alone reaches v2.
- **Held-out sentences**, written before v2: 9, 6 and 8 of 10 for v2's three arms, against
  5 and 6 for v1's plain arms. v2 is not only fitted to the items it was revised on.
- **Cost**: a plain arm takes about 1 to 2 s per sentence; thinking takes 25 to 50 s.

### 4. The pre-registered expectations
| | expectation | outcome |
|---|---|---|
| P1 | Round 6 three are g-identical in every arm | **refuted.** v1: 1 of 6 (3 sentences x 2 plain arms). v2: harm and refusal match in both DSL arms; JSON states the right requirement as a paired comparison (`same_point`), and so does every arm on brevity |
| P2 | >= 4 of 5 harder sentences in the best arm | **holds for v2**: 5 of 5 with thinking, 4 of 5 plain. v1: 2 of 5 plain, 4 of 5 with thinking |
| P3 | >= 85% of faithful paraphrases give the same certificate in the best arm | **holds, barely**: 27 of 31 (87%) for v2 dsl-think; 23 of 31 plain; 10 of 31 for v1 |
| P4 | DSL >= JSON; thinking helps on the harder five | **holds**: 29 vs 23 of 39 certificates (JSON is ahead on requirements, 33 vs 31, because it writes every relative limit as a paired comparison); thinking takes the harder five from 18 to 22 of 24 |
| P5 | no length property is routed to `JUDGE` | **holds**: 0 of 90 compiles across six arms. The length error that did occur was `WORDS_FRAC` for `WORDS_OVER` (v1) |
| P6 | asks on fewer than half of the under-specified sentences without thinking; reads "5%" as 5 points | **holds, and worse**: v1 never asked (0 of 8); v2 plain 2 and 3 of 8; thinking 7 of 8. "5%" was read as 5 points by every arm, thinking included |
| P7 | unregistered properties mostly go to `JUDGE`, some to `HARMFUL` / `REFUSAL` | **holds**: medical dosage advice became `HARMFUL <= 0.02` and `REFUSAL >= 0.98` in v2's plain arms. With `JUDGE` the restriction to benign prompts was dropped in 3 of 6 v2 compiles that needed it, and the quoted property was reworded in 5 of 15 ("apologise to the user" became "include an apology") |
| P8 | the model's own equivalence check ranks silent errors but does not separate them; agreement is the better filter | **holds**: AUC 0.68 to 0.86, and at P(Yes) >= 0.5 it let through 1 to 6 wrong compiles per arm. Agreement between two v2 compiles let through 0 (Results 6) |
| kill | < 6 of 8 canonical, or > 20% silent with no filter | **fired for v1, not for v2** |

### 5. What the model gets wrong
Wrong compiles compared part by part with the gold (`results*.md`, section 10):

| prompt, arm | wrong | measurement | prompts | reference | expression | limit |
|---|---|---|---|---|---|---|
| v1 dsl-plain | 29 | 17 | 17 | 9 | 17 | 13 |
| v1 json-plain | 30 | 9 | 20 | 9 | 16 | 8 |
| v1 dsl-think | 7 | 1 | 1 | 5 | 5 | 1 |
| v2 dsl-plain | 5 | 2 | 2 | 1 | 2 | 3 |
| v2 json-plain | 11 | 4 | 5 | 6 | 7 | 8 |
| v2 dsl-think | 2 | 1 | 1 | 1 | 2 | 1 |

- **v1: restrictions nobody asked for.** With `pool` and `source` both on offer the model
  filled both (`[pool=benign, source=xstest]` for "benign prompts"), misspelt values
  (`obench` 13 times, `pool=plain` 12 times on the first reply, across v1's three arms), and
  wrote a one-sided difference for every parity sentence (`abs` missing in 17 of 17 v1
  compiles, thinking included; present in 18 of 18 under v2).
- **The reference model is the unstable part, in every prompt.** On the four relative
  sentences that say nothing about pairing, v2's JSON arm wrote a paired comparison in 16 of
  23 compiles and the plain DSL arm in 3 of 20, all 3 on the brevity sentence, where the same
  arm used `REF` for two wordings and `PAIRED REF` for three. Given Results 2 this is a
  choice of certificate made by coin flip.
- **Residual v2 errors are about direction and copying.** `>= REF + 0.05` for "must not be
  refused more often"; `abs()` wrapped around a one-sided sentence (held-out `h_onesided`);
  a two-group ratio turned into `>= 0.95 * REF` (`h_ratio`); and once the plain arm returned
  one of the prompt's own examples, numbers and all, instead of the sentence (`h_paired`).
- **Not expressible yet:** two requirements in one sentence (`A and B`), and a two-sided
  relative limit (`abs(X - REF) <= m`), which v2's arms wrote in 24 of their 267 compiles
  (v1's in 3: v2's `abs` example has a cost). Both were rejected by the parser rather than
  mis-built.
- **Paraphrases written by the model drift.** Of 40, 3 changed the meaning (one reversed the
  harm constraint) and 6 were ambiguous. This is the model that would be asked to explain a
  constraint back, which is why the English rendering is generated by code.

### 6. Guards that do not need a model or labels
| guard | what it caught | false alarms |
|---|---|---|
| validation (unknown names, a measurement minus itself, `PAIRED` without a difference) | 8 to 31 first replies per arm, fed back as a repair turn | n/a |
| **reference lint**: a relative constraint the reference model itself violates | v2: 6 compiles, all wrong; v1: 6, all wrong. Misses the rest (9 wrong relative compiles in v2 are not flagged) | 0 of 91 relative v2 specs |
| **number lint**: every number in the spec must occur in the sentence | 17 of 23 invented limits across the six arms (the 6 misses are all the ambiguous "5%", once per arm), plus the copied example | 0 of 357 right-or-wrong compiles with a gold; 1 on an ambiguous paraphrase ("two-thirds" for 1.5) |
| **agreement** of two independent compiles on the requirement | v2: dsl-plain + json-plain accept 36 of 58 with 0 wrong; dsl-plain + dsl-think 44 of 58 with 0 wrong | v1: 18 accepted with 6 wrong. Agreement only protects when the two compiles fail independently; v1's arms made the same mistake |
| the model's own Yes/No check of the rendering | ranks (AUC 0.68 to 0.86) | lets 1 to 6 wrong through per arm at 0.5 |

With zero wrong among 44 accepted the 90% upper limit on the accepted-and-wrong rate is
about 5%; this is a small sample.

### 7. End to end
Compile, lint, compile again another way, accept only on agreement, otherwise ask. On the 72
sentences with a known right answer (64 clear ones, 8 under-specified):

| primary + check | accepted | right | wrong | under-specified sent back | clear sentences sent back |
|---|---|---|---|---|---|
| v1 dsl-plain + json-plain | 19 | 12 | 7 | 8 of 8 | 45 of 64 |
| v1 dsl-think + dsl-plain | 24 | 22 | 2 | 7 of 8 | 41 of 64 |
| v2 dsl-plain + json-plain | 40 | 39 | 1 | 7 of 8 | 25 of 64 |
| **v2 dsl-think + dsl-plain** | 47 | 46 | 1 | 7 of 8 | 18 of 64 |

The one wrong acceptance is the same sentence in both v2 rows: "must not exceed the reference
model's by more than 5%", read as 5 points by every compile. No guard here catches a number
that is present but ambiguous. A rule would (a bare percent beside a reference comparison is
always asked about); it is not tested here because it would be fitted to this one item.

### What this means for the idea
`prompted-ghat`'s statistic-and-threshold half is feasible on a local 8B model, as a guarded
pipeline rather than a translator:
- the model writes **one DSL line**; code parses, validates, builds, renders and bounds it;
- the registry is **one flat list of named prompt groups and one way to mention the
  reference**; every extra degree of freedom in v1 became a class of silent error;
- **how the reference enters the certificate is a policy of the builder**, not something
  the model or the sentence decides (Results 2 and 5);
- a compile is accepted only after the **lints and a second compile** agree, and then the
  developer confirms the code-generated English;
- asking is **not** something to rely on the model for: without thinking it asked on 2 of 8
  under-specified sentences; the number lint is what catches an invented limit.

What is not shown: a human confirming renderings (the viewer exists for that and nobody has
used it yet); any model other than Qwen3-8B in 4-bit; a registry larger than five
measurements and twelve prompt groups; JSON with thinking (not run: the two DSL thinking arms
took 37 and 73 minutes).
