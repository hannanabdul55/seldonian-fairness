# O4 registration: an exact bound on AgentDojo and suite-blocked resampling

Written 2026-10-14, before any run of `scripts/agentdojo_betting.py` (a smoke test printed
seconds and key names only). This file and the script are committed and pushed together before
the run. Open item O4 of PLAN.md section 10 (blind review W8).

## Why

The one AgentDojo pipeline that certifies an attack success rate under 5% does so with the
cluster bootstrap-t by user task, an approximate bound, at a rate of 1.1%, where the paper's
own rule sends rare labels to an exact bound. Under the paper's sampling assumption the user
tasks are i.i.d. draws, so the betting bound used for the robot tasks applies. Second, AgentDojo's
injection tasks are suite-specific: each pipeline's table is four suite blocks (banking 16 x 9,
slack 21 x 5, travel 20 x 7, workspace 40 x 6 or 40 x 14; one pipeline has three), fully
crossed within a block, and the earlier checks redrew tasks across the whole table, which can
lose a suite. Menzel's bootstrap is not run (it would be new code for a block design); the paper
keeps citing it as not run.

## What is fixed

- **Data.** `results/spikes/020/episodes.jsonl.xz`, the 28 pipelines of section 10.2, after
  `agentdojo_recheck.prep`.
- **Exact bounds on the real tables, delta 0.05.** (a1) the betting-mixture bound on the
  per-user-task rates `s_c / n_c`, a bound on the user-task-averaged rate; (a2)
  `n_max U(s / n_max) / n_bar`, a bound on the pair rate when the suite sizes are fixed by the
  benchmark. A pipeline certifies when the bound is at most 0.05.
- **Resampling.** Schemes (a), (b), (c) as `agentdojo_recheck.run_cell` draws them, and
  suite-blocked versions (a_s), (b_s), (c_s) that redraw user and/or injection tasks with
  replacement within each suite, keeping every block. 4,000 replications a cell, seeds from
  (pipeline, scheme). Arms: (a1), (a2), the cluster bootstrap-t by user task (4,000 inner draws,
  as the recheck), the quadrature bound, the larger of the two clustered bounds, the pigeonhole
  bootstrap-t and the multiway-variance t (`agentdojo_twoway.all_bounds`, unchanged). Truth: the
  observed table's pair rate for every arm, and also the user-task average for (a1). A miss is
  truth above the bound, a missing bound counted as a miss; over when the miss exceeds 0.05 by
  more than 2 Monte Carlo SE.

## Predictions

Scored by `agentdojo_betting.py report` as committed (`PRED`).

- **Z1.** The betting bound (a1) against the user-task average is over in no pipeline under (a)
  or (a_s).
- **Z2.** The betting bound (a2) against the pair rate is over in no pipeline under (a_s).
- **Z3.** The cluster bootstrap-t by user task is over in no pipeline under (a_s).
- **Z4.** The cluster bootstrap-t by user task is over in at least 15 of 28 pipelines under (b_s).
- **Z5.** The quadrature bound is over in at most 2 of 28 pipelines under each of (a_s), (b_s),
  (c_s).
- **Z6.** On the real tables (a1) certifies 5% for at most one pipeline, and (a2) for none.

Reported without a prediction: every other arm x scheme count, the share of unblocked
redraws that lose a suite, the real-table values beside Table 7.

## Seen before writing this

Table 4's AgentDojo rows and Table 7 (the issued bounds and their checks under (a), (b), (c));
the suite structure (counted by the scout from the data and the benchmark's code). No betting
bound on any real pipeline and no blocked-scheme miss was computed.

## Not allowed after the run

Changing pipelines, schemes, replication counts, seeds, arms, the cell rule or the scoring code.
A forced deviation is reported as one.
