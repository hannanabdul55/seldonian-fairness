# B6 registration: AgentDojo's bounds against a fitted crossed model

Written 2026-10-16, before any bound was computed on a simulated table. This file and
`scripts/agentdojo_parametric.py` are committed and pushed together before the run. Open item B6 of PLAN.md
section 11 (blind review of v0.9.10, M9).

## Why

Section 10.2's validity checks redraw rows and columns of each pipeline's observed table and score the
bounds against that table's own rate. That is the dependence structure the pigeonhole bootstrap assumes, so
the check is partly circular. Here the truth is a fitted model's rate, which no table shows.

## What is fixed

- **Model, per pipeline.** P(success) = expit(mu_s + u_i + v_j) inside each suite block, with
  u_i ~ N(0, su^2) and v_j ~ N(0, sv^2). The table's pattern of pairs is kept.
- **Fit, by simulated moments.**
  - Grid: su and sv each in 0, 0.25, ..., 6.
  - mu_s is set by quadrature so that the model's suite rate equals the observed rate, Jeffreys-smoothed.
  - The grid point kept is the one whose simulated within-suite variances of user-task and injection-task
    rates (64 tables, common random numbers) are closest on the log scale to the observed ones.
- **Schemes.** 4,000 tables each, seed 20261016.
  - `P_a`: injection tasks fixed (v drawn once per pipeline); user tasks drawn afresh. Truth: the model's rate
    given those injection tasks.
  - `P_c`: both kinds of task drawn afresh. Truth: the model's population rate.
- **Truths.**
  - The pair rate, for every arm.
  - Also the user-task average, for the betting bound on user-task rates (`bet_a1@uavg`).
- **Arms.** `agentdojo_betting.one_table`, unchanged: `bet_a1`, `bet_a2`, `t_user`, `t_inj`, `quad`, `max`,
  `pig_t` and `cgm_t`, with 4,000 inner bootstrap draws.
- **Scoring.**
  - A NaN bound counts as a miss.
  - Delta is 0.05.
  - A cell is over if its miss exceeds delta by more than 2 Monte Carlo SE.
- **Not run.** Menzel's (2021) bootstrap.

## Predictions

Scored by the script's `report` as committed.

- **W1.** Under P_a, `bet_a1@uavg` and `bet_a2` are over for no pipeline.
- **W2.** Under P_a, `t_user` (the issued certificate's bound) is over for at most 2 of 28 pipelines.
- **W3.** Under P_c, `t_user` is over for at least 8 of 28 pipelines.
- **W4.** Under P_c, `quad` is over for at most 2 of 28 pipelines.
- **W5.** Under P_c, `bet_a1@uavg` is over for at least 2 of 28 pipelines.

Reported without a prediction:
- Every other arm and scheme.
- The probability that `t_user` and `bet_a1` certify 5%, per pipeline.
- The truths against the tables' rates.

## Seen before writing this

- O4's results: under (b_s), `t_user` was over for 11 pipelines and the betting bounds for 3 and 2; under
  (c_s), `quad` was over for none.
- The fitted (su, sv) of every pipeline (`results/paper/agentdojo_parametric_fit.json`) and its fit error.
  Two pipelines sit at the grid's edge of 6: Meta-SecAlign-70B's su and gpt-3.5-turbo-0125's su.
- A timing run of 5 tables a scheme on one pipeline, which printed shapes and seconds only.

## Run

    OMP_NUM_THREADS=1 .venv/bin/python scripts/agentdojo_parametric.py run --workers 14
