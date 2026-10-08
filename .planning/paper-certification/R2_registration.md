# R2 registration: two standard two-way bounds on AgentDojo's crossed design

Written 2026-10-07, before `scripts/agentdojo_twoway.py` was run on any of the benchmark's
tables. The script was tested on random tables only (`--selftest`). This file and the script
are committed together before the run; the commit is the timestamp.

## Question

Section 10.2 of the paper says that no bound specified in advance held when AgentDojo's
injection tasks are treated as sampled along with its user tasks, and sets aside one that
did hold in a single check (the two clustered margins added in quadrature) because it was
not specified in advance. A reviewer asked for the standard two-way methods. This run asks
whether either of two standard bounds, fixed here, holds its level under scheme (c), and
whether the quadrature bound's earlier result repeats on fresh draws.

## What is already known and seen

- `results/paper/agentdojo_recheck.md` (seed 20261004): under scheme (c) the per-pair bound is
  over its level for 28 of 28 pipelines, the bootstrap-t by user task for 24, by injection
  task for 23, the larger of the two for 16, the spike's "two-way bootstrap" for 27, and the
  quadrature bound for none.
- The spike's "two-way bootstrap" (`cert020.twoway_t`) is the pigeonhole bootstrap of Owen
  (2007) with a basic (reflected percentile) limit. So one standard method is already in the
  paper, and fails. The pigeonhole candidate below differs from it in one respect: it is
  studentised.
- Neither `cgm_t` nor `pig_t` below has been computed on any of the 28 tables, on a resample
  of them, or on any other AgentDojo data.

## Data, schemes, sizes

- The 28 pipelines' (user task by injection task) tables from
  `results/spikes/020/episodes.jsonl.xz`, built by `agentdojo_recheck.table`. Each table's
  own rate is the truth.
- Schemes as in `agentdojo_recheck.py`: (a) user tasks redrawn with replacement, injection
  tasks fixed; (b) the reverse; (c) both. Draws are over all of a pipeline's tasks, a redrawn
  task is a new cluster.
- 4,000 resampled tables per pipeline and scheme; 4,000 inner bootstrap draws; delta 0.05;
  master seed 20261008 (not used before).
- The rule is the paper's (section 6.1): a cell is over its level when its miss exceeds
  `0.05 + 2 sqrt(0.05 * 0.95 / 4000)` = 0.0569, unresolved between 0.05 and that, at or under
  otherwise.

## The bounds

With `n` valid pairs, `k` successes, `est = k / n`, and `m_u`, `m_i` the numbers of user and
injection tasks:

```
V_u  = m_u / (m_u - 1) * sum_u (s_u - est n_u)^2 / n^2        (section 6.3's cluster variance)
V_i  = the same over injection tasks
V_p  = est (1 - est) / (n - 1)                                 (each pair its own cluster)
V2   = V_u + V_i - V_p,  and max(V_u, V_i) if that is not positive
```

1. **`cgm_t`, the multiway variance with a t quantile** (Cameron, Gelbach and Miller, 2011):
   `est + t_{0.95, min(m_u, m_i) - 1} sqrt(V2)`. With no success in the table the variance is
   zero and the bound returns 1 (no limit), the convention of section 6.3's bound.
2. **`pig_t`, the pigeonhole bootstrap-t**: resample user tasks and injection tasks
   independently with replacement (Owen, 2007), compute `est*` and `V2*` on each resampled
   table (a task drawn twice is two clusters), `t* = (est* - est) / sqrt(V2*)`, and take
   `est - q sqrt(V2)` with `q` the lower 0.05 quantile of `t*`. A resample with zero variance
   below the estimate counts as minus infinity; a quantile that is not finite returns 1.
3. **`quad`, a replication**: `est + sqrt((U_u - est)^2 + (U_i - est)^2)` with `U_u`, `U_i`
   the one-way cluster bootstrap-t limits. It was first computed on these tables after
   their results were seen, so new draws test it against Monte Carlo luck and not against
   selection on these 28 tables. It is reported with that caveat whatever it shows.

## Reading, fixed now

- A bound **holds under two-way sampling** if it is over its level for none of the 28
  pipelines under scheme (c). Counts under (a) and (b) are reported beside it.
- If `cgm_t` or `pig_t` holds, section 10.2 offers it as the certificate that covers new
  injection tasks, with its cost: the ratio of its margin (limit minus rate) to the
  user-task-clustered margin on the 28 published tables (limits at 200,000 inner draws),
  and the number of pipelines that certify "at most 5%" under it.
- If neither holds, section 10.2 says that the two standard bounds do not hold here, with
  their counts, and the paper's certificate stays as it is (injection tasks fixed).
- `quad` stays a bound that was not specified before the tables were seen. If it is over for
  none again, the paper may report it as holding on two independent sets of draws, with
  that caveat, and give its cost in width; it does not become the headline certificate.

## Expectation (to be scored)

`cgm_t` is over its level for some pipelines under (c): a symmetric limit on a rate whose
successes sit in a handful of tasks. `pig_t` is over for fewer than `cgm_t`. `quad` is over
for none.

## Not allowed after the run

Changing a bound's definition, the fallback rules, the seed, the sizes or the reading. A
deviation, if one is forced (a crash, a numerical failure), is reported as a deviation with
both versions of the result.
