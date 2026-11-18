# Q1 registration: closed-form limits on AgentDojo, leave one injection task out, first-sentence cuts

registered_at: 2026-11-18T01:01:05Z (UTC, written from the clock by the job that makes this commit)

Three checks asked for on 2026-11-17. Nothing below had been run when this file was written, with the
exceptions listed under "Seen before registration". The scripts are committed with this file:
`scripts/agentdojo_closed.py`, `scripts/trunc_judge.py`, `scripts/run_trunc.sh`.

A fourth check that was proposed, resampling user tasks to test the cluster bootstrap-t for all 28
pipelines, is not run: `scripts/agentdojo_recheck.py` already holds it (4,000 resamples, over in none).

## A. Closed-form limits by user task (`agentdojo_closed.py`, part A)

The paper's AgentDojo limit is a cluster bootstrap-t by user task, which returned no limit in up to 16% of
tables. Two closed forms, defined in the script's header: the cluster t-limit `ct` and its one-term skew
correction `cts` (the correction of `scripts/waldt_skew.py`, with the clusters' third moment). Scheme: user
tasks drawn with replacement, injection tasks fixed, each pipeline's table the population; 28 pipelines,
20,000 resamples each, delta 0.05, seed 20261117. A resample with no variance gives no limit and is not a
miss. Over: miss > delta + 2 se.

- C1. `ct` is over its level for at least 8 of 28 pipelines.
- C2. `cts` is over for fewer pipelines than `ct`, and for at most 4.
- C3. The largest miss rate of `cts` is at most 0.08.
- C4. On the real tables, claude-3-5-sonnet-20241022 is the only pipeline with `cts` at or under 0.05.
- C5. Its `cts` lies between 0.020 and 0.030 (the bootstrap-t gives 0.022).

Rule. `cts` over for no pipeline: the paper reports it as a closed form that held on this benchmark, beside
the bootstrap-t, with this single check. `cts` over for any pipeline: the paper reports that the closed form
does not replace the bootstrap-t there. C4 refuted: the "one pipeline of 28" sentence gains the closed
form's count beside it; the bootstrap-t's count stays the headline because it is the checked limit.

## B. Leave one injection task out (`agentdojo_closed.py`, part B)

For each pipeline, suite and injection task of that suite: the cluster bootstrap-t by user task (4,000 inner
draws, delta 0.05) on the suite's other injection tasks, against the left-out task's success rate over the
same user tasks. A limit bounds a mean over tasks, so a single task's rate above it is not a miss of the
limit; the count describes how little the limit says about an unseen task.

- L1. Among task-and-pipeline pairs with a limit, the left-out rate is above the limit in more than 15%.
- L2. At least 20 of 28 pipelines have at least one such task.

Rule. Both kept: the paper's sentence that the certificate does not extend to new injection tasks gains this
count. L1 refuted: the count is reported and the sentence rests on the resampling result alone.

## C. First-sentence cuts (`trunc_judge.py`)

Section 5.1 says the compiled judge lost recall because trained refusals carry a helpful-sounding body, and
that no experiment isolates this. Every response of the pre-update set and of the three seeds' step-200 sets
(500 each) is cut to its first sentence and scored again by the same judge under the same six wordings.
Gold stays the guard's flag on the full response. Hybrid shape is Appendix B's rule (refusal opener and
more than 40 words). Carried bound: `transfer017.youden_plasmode`, 4,000 draws, rng 1117; arms: full
responses, target cut (A), both sides cut (B). The first-sentence rule is crude (it stops at "U.S." and at
"Sure!"); it is not tuned.

- M1. Pooled over the three seeds, among hybrid-shaped guard flags at step 200, wording 0 flags more cut
  responses than full ones: exact McNemar p < 0.05 and a rise of at least 0.20 in the flagged share.
- M2. Wording 0's recall on the guard's refusals at step 200, cut, is at least 0.15 in each seed (full:
  0.03, 0.07, 0.05; before the update 0.21).
- M3. With both sides cut, the carried bound is over for at most 1 of six wordings in each seed (full: 3, 3, 2).
- M4. At step 0, cutting changes wording 0's false-alarm rate on the guard's non-refusals by less than 0.03.
- M5. With only the target cut, the carried bound is over for fewer wordings than with full responses in at
  least two of three seeds.

Rule. M1 and M2 kept: the paper replaces "which no experiment isolates" with this result. M1 kept and M3
refuted: the paper says the body explains part of the lost recall and not the failure of the carried bound.
M1 refuted: the paper says the one experiment that tested the account did not support it.

## Seen before registration

- The per-injection-task success counts of claude-3-5-sonnet-20241022 (7 successes: 2, 1, 3 and 1 on four
  tasks; the largest rate 3 of 21). No prediction above is about that pipeline's task-level rates.
- `results/paper/agentdojo_recheck.md` (the bootstrap-t's miss rates and the real-data certificates).
- The median length of the cut responses (17 to 19 words) and that no cut is empty. No judge score on a cut
  response has been computed.
- `--selftest`, `--time` and `score --dry` outputs (formula checks, timings, item counts).

## What may not change after the run

Seeds, replication counts, the formulas, the first-sentence rule, the hybrid rule, the over rule, the sets
and the wordings. A crash is reported; a retry of the GPU stage resumes the same file and is logged.
