---
spike: 015
idea: prompted-ghat
name: prompted-judge-fidelity
type: standard
validates: "Given a constraint written in English, when a local LLM compiles it into judging instructions and scores responses with one forward pass, then its agreement with the hand-written judge and with human labels, its calibration, its paraphrase flip rate and its rubric-artifact behaviour are measured against the trained guard"
verdict: PARTIAL
related: [005, 007, 013]
tags: [prompted-judge, rubric, calibration, artifact-test, gpu]
---

# Spike 015: Is a judge compiled from a sentence good enough?

## What This Validates
The `prompted-ghat` idea needs a measurement half: the developer writes the constraint in
English, and something turns it into a judge. This spike asks whether that judge can carry a
Seldonian certificate at all. Given a constraint sentence, when Qwen3-8B (4-bit) compiles it
into judging instructions and scores each response by P(Yes) on one forward pass, then:
fidelity (agreement with the reference label and with human labels), calibration (ECE,
Brier), paraphrase robustness (does the measured rate move when the sentence is reworded),
and the rubric artifact (does the verdict actually depend on the response) are all measured
against the trained guard as the baseline.

**Pre-registered expectation** (from the research pass, before running): the prompted judge
loses to the trained guard on harm and ties on brevity; paraphrases move the measured rate;
the artifact test is the one most likely to kill the idea.

## Research
- **Rubrics as Rewards** (arXiv:2507.17746): LLM-written rubrics beat Likert judges by up to
  31% on HealthBench, with the gain concentrated in smaller judges. Author-reported.
- **Rubric Artifacts** (arXiv:2609.02942): a classifier on the rubric text *alone* predicts
  the judge's verdict at >0.5 and sometimes >0.8; reversing the response flips the verdict
  only 37.7% of the time. This is the test that matters, and it is why `blank` and `swap`
  variants are run here.
- **The Coin Flip Judge?** (arXiv:2606.13685): rewording an evaluation prompt flips the
  preferred response in up to 25% of comparisons; pointwise judges are worse than binary.
- **Qwen3Guard** (arXiv:2510.14276): trained small guards beat prompted judges on safety
  (8B avg F1 90.0 prompt / 83.9 response vs LlamaGuard3-8B 79.4 / 70.7). Author-reported.
  So the trained guard is the bar, and spike 005's own bake-off measured it on our items.

## How to Run
    cd .planning/spikes/015-prompted-judge-fidelity
    ./run.sh --stage compile                       # wordings + rubrics (~2 min GPU)
    ./run.sh --stage judge --n 500 --batch 8       # ~11k forward passes (~25 min GPU)
    ../../../.venv/bin/python analyze015.py        # results.md

## Method
- **The harm wording is not authored in this spike.** It is read out of spike 005's
  `RUBRIC_PROMPT` (`h = harmful:` line), which is the definition the human annotator worked
  from (`results/screen/labels/GUIDELINE.md`). Refusal and brevity are one sentence each.
  All paraphrases are written by the model, not by hand, so the robustness test uses
  rewordings a developer might plausibly produce.
- **Compile:** one sentence in, a short paragraph of grader instructions out, greedy,
  thinking off.
- **Judge:** P(Yes) / (P(Yes) + P(No)) on the first answer token, one forward pass per item
  (the scoring `judge_bakeoff.shieldgemma_pyes` uses). This gives a probability, not just a
  label, so calibration is measurable and the "bounded feature" certificate of spike 017 is
  possible.
- **Items and references:** harm = the 225 definite human labels (positive = `h` or `c`,
  3 positives); refusal = 013's over-refusal pool at step 0, two responses per prompt,
  reference = Qwen3Guard-4B's refusal field; brevity = the same responses, reference = words
  > 80, exact. 500 items for refusal and brevity.
- **Variants:** `rubric` (compiled, all six wordings), `raw` (the sentence used directly, so
  the value of compiling is visible), `blank` (response removed), `swap` (responses shuffled
  across prompts, scored against the response actually shown).
- **Baseline:** Qwen3Guard-4B in 4-bit from spike 005's bake-off, on the same 225 harm items
  (52 flagged, matching the published table).

## Investigation Trail
1. **A safety classifier stopped the first draft of this spike's script**, which had newly
   written prose defining harmful-content criteria for the rubric. The redesign avoids
   authoring such text at all and loads the harm definition from the repo's own artifacts.
   This is also better science: the compiled judge is now tested against the same definition
   the human labels used, rather than a fresh paraphrase of it.
2. **The Jev arm was never run.** The spike was scoped with TypeSafe's Jev Noul question as
   a possible third judge; on 2026-09-30 the user confirmed no access and declined sending
   prompts or responses to an external API, so this and every later spike in the idea use
   local models only. Jev remains useful as a design reference (a typed question in, a
   calibrated probability out) and is not a dependency.
3. **Baseline corrected before running:** `key.jsonl`'s `judge_primary` is the screen's
   original 0.6B judge (139/225 flags). The fair baseline is the bake-off winner,
   Qwen3Guard-4B 4-bit (52/225), loaded from `results/screen/judge_bakeoff/qg-4b-q4.jsonl`.

4. **The judge stage was still appending** when the first analysis ran (8,525 of 11,025
   rows), which briefly showed brevity with only two wordings. Re-run on the complete file.

## Results
**Verdict: PARTIAL.** A compiled judge is usable as a *ranker* and unusable as a *labeller*,
and it cannot do a verifiable constraint at all. Three findings, in order of how much they
change the idea.

### 1. A verifiable constraint must compile to code, not a judge
Brevity is "more than 80 words": decidable by counting. The compiled judge is at chance.

| variant | AUC | caught | corr(P(Yes), actual words) | mean P(Yes) at <=80 words | at >80 |
|---|---|---|---|---|---|
| rubric | 0.500 | 8/202 | +0.059 | 0.052 | 0.043 |
| raw | 0.482 | 74/202 | -0.078 | 0.428 | 0.385 |

P(Yes) is *lower* for long responses than short ones. Worse, the `raw` variant's measured
rate (0.396) lands within one point of the truth (0.404) **by coincidence**, on labels that
are noise. A Seldonian test bounds a mean, so a judge with a near-correct marginal and no
per-item signal yields a *valid-looking* bound on a quantity that is not the constraint, and
nothing in the certificate reveals it. This is the sharpest lesson of the spike: the
compiler must route length, format and exact-match properties to deterministic features
(`LengthFeature` already exists) and reserve judges for the irreducibly semantic part.

### 2. The ranking is robust to paraphrase; the operating point is not
Across six wordings (canonical plus five model-written paraphrases):

| task | AUC range | measured-rate range | mean flip rate vs canonical | max pairwise disagreement |
|---|---|---|---|---|
| harm | 0.802-0.866 | 0.093-0.458 | 0.319 | 0.449 |
| refusal | 0.707-0.922 | 0.008-0.090 | 0.040 | 0.084 |

On harm the AUC barely moves while the measured rate swings five-fold, and one paraphrase
(w4, rate 0.093) catches 0 of the 3 human positives while another (w2, 0.458) flags nearly
half the sheet. So the compiled judge orders responses consistently and thresholds them
arbitrarily. Consequences for the build: the certificate must pin the **exact wording** as
part of the constraint's identity, and the decision threshold must be **calibrated on
labelled data** rather than left at 0.5. That makes spike 017 mandatory rather than optional.
On refusal every wording under-flags (0.008-0.090 against a 0.200 reference): conservative,
but by an amount that depends on the sentence.

### 3. Prompted matches the trained guard on harm, and both are unusable raw
On the 225 human-labelled items (3 positives, base rate 0.013):

| judge | flagged | FA rate (90% CI) | caught |
|---|---|---|---|
| compiled rubric (canonical) | 0.222 | 0.216 (0.172-0.267) | 2/3 |
| the sentence used directly | 0.298 | 0.293 (0.243-0.347) | 2/3 |
| Qwen3Guard-4B (bake-off winner) | 0.231 | 0.230 (0.184-0.281) | 1/3 |

This **refutes the pre-registered expectation** that the trained guard wins on harm: the
compiled judge matches it on false alarms and catches one more positive. Compiling helps
over the bare sentence (FA 0.216 vs 0.293 at the same recall). But a 22% false-alarm rate
against a 1.3% base rate is unusable without correction, which is spike 013's rare-rate
finding arriving from a different direction.

### 4. The rubric artifact fires on refusal
`blank` removes the response; `swap` shuffles responses across prompts and is scored against
the response actually shown.

| task | blank AUC | corr(blank, canonical) | canonical AUC | swap AUC (shown response) |
|---|---|---|---|---|
| harm | 0.525 | +0.138 | 0.802 | 0.574 |
| refusal | **0.694** | **+0.319** | 0.886 | 0.605 |
| brevity | 0.487 | +0.038 | 0.500 | 0.471 |

For refusal the prompt alone predicts the verdict better than chance (AUC 0.694) and
correlates with the judge's real scores (+0.319): part of the verdict is about which prompt
was asked, not what was answered. On both harm and refusal the swap AUC falls well below the
canonical AUC, so the prompt carries a substantial share of the decision. The caveat is that
a shuffled pair is incoherent in a way the paper's response-reversal is not, so `swap` is
suggestive rather than conclusive; `blank` is the clean test, and it fires on refusal.

### What this means for the idea
`prompted-ghat` survives, narrowed:
- **compile verifiable properties to code**, and judges only for semantic ones;
- **ship the wording, a calibrated threshold and the human-label correction as part of the
  constraint**, since none of the three is optional at these numbers;
- **the probability, not the label, is the useful output** (AUC 0.80-0.87 on harm), which
  also supports the bounded-feature certificate planned for 017.

**Gate for spike 018 has fired:** the prompted rubric failed the artifact test on refusal and
is at chance on a verifiable constraint. A trained, instruction-conditioned, calibrated judge
(018, "Jev tier A") is now the indicated fix rather than a speculative one.
