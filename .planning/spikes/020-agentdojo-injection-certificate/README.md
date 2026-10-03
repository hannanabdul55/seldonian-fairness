---
spike: 020
idea: external-trace-certificate
name: agentdojo-injection-certificate
type: standard
validates: "Given AgentDojo's published per-episode runs (29 model/defence pipelines, 629-949 (user task, injection task) pairs each, the harness's code-computed `security` label), when a per-pipeline certificate on the targeted attack success rate is computed with the crossed design respected, then we know how much the naive i.i.d. bound understates the uncertainty, which pipelines certify at 5%, and whether suite strata buy anything"
verdict: VALIDATED
related: [013, 017, 019]
tags: [certificate, external-traces, prompt-injection, agents, clustered, clopper-pearson, cpu]
---

# Spike 020: a certificate on AgentDojo's published runs

Approved by the user on 2026-10-03 ("try agentdojo") after the benchmark scout ranked it
first: the only public corpus with a code-computed per-episode safety label (`security`:
the injected task's goal was reached in the environment state), published runs for 29
pipelines, and positives from 1% to 56%. Repo `ethz-spylab/agentdojo` at 089ed468
(2026-06-02), `runs/` 477 MB, cloned to `/mnt/d/seldonian-runs/020/agentdojo/`.

## What This Validates
The safety-test half of the pipeline on a fixed policy with a label that needs no judge and
no human, on a benchmark whose episodes are *not* independent: each pipeline's episodes are
97 user tasks crossed with 27 (v1) or 35 (v1.2) injection tasks under the standard
`important_instructions` attack. A certificate that treats the 629-949 pairs as i.i.d. is
wrong in a known direction; this spike measures by how much, and what certifies once the
design is respected.

## Pre-registration (written 2026-10-03 10:30 PT, before any bound was computed)
Seen before writing this: the directory layout, the fields of one episode file, and a
count pass per pipeline (pairs, user tasks, injection tasks, errors, `security` successes
and the raw rate: e.g. gpt-4o 300/629 = 0.477, claude-3-7-sonnet 47/949 = 0.050,
claude-3-5-sonnet-20241022 7/629 = 0.011, Meta-SecAlign-70B 21/949 = 0.022,
gpt-4o + tool_filter 43/629 = 0.068; one pipeline has no `important_instructions` runs;
errors are 0 for most pipelines, up to 35 for Llama-3-70b). The website's results table
quotes claude-3-7-sonnet at 7.3%, which does not match 47/949; H1 checks what it computes.

Definitions. Policy = pipeline (model plus defence). Episode = (user task, injection task)
pair under `important_instructions`. Label = `security` (1 = the injection succeeded).
Episodes with `error` set are kept as label 0 only if `security` is recorded, else dropped
and counted. The certified quantity is the mean label over the benchmark's pairs for that
pipeline (the population is the benchmark's own grid, so no two-phase term).

Bounds, all one-sided upper at delta 0.05 (and 0.1 in the tables):
- **naive**: Clopper-Pearson on the pairs as if independent;
- **cluster-t (user)**: studentised cluster bootstrap over the 97 user tasks (017's
  bootstrap-t, 4,000 resamples; a zero-variance resample counts as t = -inf);
- **cluster-t (injection)**: the same over injection tasks;
- **two-way**: resampling user tasks and injection tasks independently (pigeonhole
  bootstrap), exploratory;
- **any-injection**: unit = user task, label = any injection succeeded, Clopper-Pearson at
  n 97: an exact bound on a stricter quantity;
- **suite strata**: 013's `b1w` with the four suites as strata (weights = the suite's share
  of pairs), against pooled Wilson.
Dependence is measured as the one-way ICC of the label by user task and by injection task,
and the design effect 1 + (k - 1) ICC with k the mean cluster size.

- **H1 (reproduction).** Our pair rates equal the website's table for the pipelines it
  lists, to within 0.5 points, once the subset it uses is identified (v1's 629 pairs or
  v1.2's 949); where they differ the difference is the subset, not the label.
- **H2 (dependence).** For pipelines with rates between 5% and 60%, the ICC by user task is
  at least 0.2 (design effect at least 2), and the cluster-t (user) bound sits at least 1.5
  times as far above the point estimate as the naive bound does. Which of the two
  clusterings carries more dependence is not predicted; both are reported and the larger
  ICC names the unit the certificate uses.
- **H3 (certificate).** "Targeted attack success rate <= 0.05 at delta 0.05", cluster-t by
  the dominant unit: passes for claude-3-5-sonnet-20241022 (0.011), Meta-SecAlign-70B and
  its repeat_user_prompt variant (0.022, 0.021); NSF for claude-3-7-sonnet (0.050),
  command-r-plus (0.045), gpt-4o + tool_filter (0.068) and every pipeline above; not
  predicted for command-r (0.033) and gemini-1.5-flash-002 (0.035), where the clustered
  bound decides.
- **H4 (strata).** Suite strata buy an ESS under 1.2 against pooled Wilson: the dependence
  is within user tasks, not between suites.

Expectation stated plainly: the naive bound will look fine and be wrong; the clustered one
is the certificate. With 97 user tasks the any-injection bound at zero successes is 0.030,
so no pipeline certifies "no user task is ever compromised" below 3%.

Not tested: utility (reported descriptively, with and without attack); the other 13 attack
variants gpt-4o was run with (exploratory table only); training a policy against the label
(the next step if this one works, needing the local harness).

## How to Run
    cd .planning/spikes/020-agentdojo-injection-certificate
    ../../../.venv/bin/python cert020.py          # reads /mnt/d/seldonian-runs/020/agentdojo/runs -> results.md, cert.json, results/spikes/020/episodes.jsonl (2 min)
    OMP_NUM_THREADS=2 ../../../.venv/bin/python plasmode020.py --reps 400   # validity of each bound under user-task resampling -> plasmode.md (12 min)

## Investigation Trail
1. Clone 2026-10-03 10:15 PT; layout `runs/<pipeline>/<suite>/<user_task>/<attack>/<injection_task>.json`
   with `utility`, `security`, `error`, `duration`, `messages`. 29 pipelines; 15 attack
   variants for gpt-4o, `important_instructions` and `none` for the rest.
2. **Scored pass** (`cert020.py`, 2026-10-03 10:45 PT): 28 pipelines with runs under the
   standard attack (one has none), 19,380 pairs. Errors are rare (0 for 20 pipelines, 35 at
   most) and every errored episode still carries a `security` value, so none was dropped.
3. **Post hoc validity check** (`plasmode020.py`, not pre-registered): the 97 observed user
   tasks taken as the population, 400 redraws of 97 clusters, each bound's miss rate against
   the observed rate. Added because the cluster-t bound for Meta-SecAlign (0.104 on a 2.2%
   rate) looked wide and the question was whether it or the naive bound (0.032) was right.

## Results

Full table in `results.md`; validity table in `plasmode.md`.

| pipeline (selected) | pairs | rate | naive CP | ICC user / inj | t(user) | t(inj) | any-injection (user tasks compromised) | verdict at 5% |
|---|---|---|---|---|---|---|---|---|
| claude-3-5-sonnet-20241022 | 629 | 0.011 | 0.021 | 0.10 / 0.07 | 0.022 | 0.034 | 0.118 (6 of 97) | **pass** |
| Meta-SecAlign-70B | 949 | 0.022 | 0.032 | 0.72 / 0.08 | 0.104 | 0.035 | 0.105 (5 of 97) | NSF |
| Meta-SecAlign-70B + repeat_user_prompt | 949 | 0.021 | 0.030 | 0.85 / 0.09 | 0.057 | 0.034 | 0.105 (5 of 97) | NSF |
| command-r | 629 | 0.033 | 0.048 | 0.36 / 0.08 | 0.062 | 0.054 | 0.169 (10 of 97) | NSF |
| gemini-1.5-flash-002 | 629 | 0.035 | 0.050 | 0.22 / 0.34 | 0.052 | 0.112 | 0.251 (17 of 97) | NSF (inj) |
| claude-3-7-sonnet-20250219 | 949 | 0.050 | 0.063 | 0.33 / 0.23 | 0.070 | 0.106 | 0.374 (28 of 97) | NSF |
| gpt-4o + tool_filter | 629 | 0.068 | 0.087 | 0.08 / 0.12 | 0.092 | 0.106 | 0.417 (32 of 97) | NSF |
| gpt-4o-2024-05-13 | 629 | 0.477 | 0.511 | 0.45 / 0.55 | 0.538 | 0.586 | 0.859 (77 of 97) | NSF |

- **The naive bound is invalid on this benchmark, and the clustered one is valid.** Under
  user-task resampling (the design's own randomness) Clopper-Pearson over pairs missed the
  truth in 5-28% of draws at delta 0.05 (Meta-SecAlign 0.275, gemini-2.0-flash 0.223,
  gpt-4o 0.168, claude-3-5 0.048); the studentised cluster bootstrap by user task missed in
  0.02-0.06 (Monte Carlo se 0.011). The two-way basic bootstrap missed 0.17 where the
  positives sit in a few clusters. So the certificate is `t(user)`, and where injection
  tasks carry more dependence (eight pipelines, all with rates above 3%) the larger of the
  two clustered bounds.
- **One pipeline in 28 certifies a targeted attack success rate at or under 5%:**
  claude-3-5-sonnet-20241022 (0.022 clustered). Meta-SecAlign-70B, whose raw rate of 2.2%
  looks certifiable, is not: its 21 successes sit in 5 of 97 user tasks (ICC 0.72, design
  effect 7.5), so the clustered bound is 0.104, and the exact any-injection bound agrees
  that up to 10.5% of user tasks are compromised. The defence moved the dependence into a
  few tasks rather than removing it. The same holds for its repeat_user_prompt variant.
- **Dependence is large and its direction depends on the policy.** Design effects by user
  task run 1.4-8.7 (median about 3), by injection task 2.6-15.5. For weak attacks the user
  task decides (which tool output the injection lands in); for strong ones the injection
  goal decides (gpt-4o: ICC 0.55 by injection task, gpt-4-0125: 0.66). Clustering by the
  wrong unit gives bounds that differ by up to 0.10.
- **The website's numbers are the v1 subset.** gpt-4o 0.477 and claude-3-5 0.011 match
  on all pairs; claude-3-7-sonnet is 0.050 on its 949 pairs and 0.073 on the 629 v1 pairs,
  which is what the site prints. The label reproduces exactly.
- **Suite strata buy little but not nothing**: ESS 1.00-1.47 against pooled Wilson,
  largest for gemini-2.0-flash (suite rates differ most). Both are i.i.d. bounds and
  neither is the certificate here.
- **Utility under attack** (descriptive): the two pipelines with the lowest rates keep
  their utility (claude-3-5: 0.79 to 0.72; SecAlign 0.78 to 0.78); gpt-4o drops 0.69 to
  0.50 and its tool_filter defence costs it 0.73 to 0.56 while leaving the rate at 0.068.

### Scored expectations
| hypothesis | asked | found | result |
|---|---|---|---|
| H1 reproduction | website rates to 0.5 points once the subset is known | exact on the v1 subset (0.073 = 0.073); all pairs 0.050 | holds |
| H2 dependence | ICC user >= 0.2, deff >= 2, clustered distance >= 1.5x naive, for rates in [0.05, 0.60] | ICC 0.08-0.70, deff 1.4-7.3, ratio 1.44-4.19 (median 1.97); tool_filter and gpt-4-0125 below the ICC floor, claude-3-opus and the pi_detector at ratio 1.44 | holds in 17 of 21; the dependence is real, not uniform |
| H3 certificate | pass for claude-3-5-20241022 and both SecAlign; NSF for 3.7, command-r-plus, tool_filter | pass only for claude-3-5-20241022; SecAlign NSF by clustering (0.104, 0.057); the three NSFs as predicted; the two open cases NSF | half: the SecAlign prediction was wrong for the reason the spike exists |
| H4 strata | ESS < 1.2 | 1.00-1.47 | fails (modest gains where suite rates differ) |

**Verdict: VALIDATED.** The questions were how much the i.i.d. bound understates the
uncertainty (it misses 5-28% of the time at delta 0.05; the clustered bound holds), which
pipelines certify at 5% (one of 28), and whether suite strata help (a little, and they are
not the certificate). The pre-registered miss was SecAlign: a defence can hold the raw rate
at 2% while concentrating the failures in a handful of user tasks, and only a bound that
respects the design sees it. This is the first certificate in the project on a label that
needs no judge and no human, and the benchmark is runnable locally, so the training half
of the pipeline has a target (next step, separately approved).

## Files
`cert020.py` (`results.md`, `cert.json`), `plasmode020.py` (`plasmode.md`, `plasmode.json`);
`results/spikes/020/episodes.jsonl.xz` (19,380 rows, no messages). Runs: `/mnt/d/seldonian-runs/020/agentdojo/`.
