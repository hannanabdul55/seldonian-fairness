# Recovery of the LLM post-training work (2026-09-19)

On 2026-09-19 the WSL distro holding this repo was unregistered, which deleted its
disk image. The branch `llm-seldonian-rl` had never been pushed, so its code, its
`results/` and the local Claude Code transcripts were lost. This branch is a rebuild.

## How the branch was rebuilt

The main working session (https://claude.ai/code/session_01PS6cEqLggjroZPamwdJZZC,
2026-09-02 to 2026-09-12) was downloaded with `claude --teleport`. Its transcript holds
every file write, edit and patch script the session ran, so the branch was rebuilt by
replaying them in order from `cbbdb55` (the base the session pulled), including the
2026-09-07 rebase onto `254a3d7`, and recommitting at the same points with the original
messages and dates. The replay was checked against full file dumps the session printed
(identical) and against every later patch's `assert anchor in s` (all apply).

Three batches were written by subagents whose transcripts were not recoverable, and were
**reconstructed** from their original task prompts, their final reports, the main
session's later views of the files (verbatim excerpts, argparse listings, line numbers),
and the anchors of later patches:

| batch | files | how close |
|---|---|---|
| Round 4 track 1 (2026-09-07) | `seldonian/llm/synthetic.py`, `scripts/synthetic_calibration.py`, `tests/test_llm_synthetic.py` | argparse block verbatim; sweeps reproduce the paper's section 6.3 numbers within seed noise (grpo unsafe 0.830 vs 0.830, lag true rate 0.128 vs 0.128) |
| Round 4 track 2 (2026-09-07) | `seldonian/llm/constraints.py`, `tests/test_llm_constraints.py`, `scripts/resample_calibration.py`, `cache_only` in `judges.py` | class/def layout and the paired-constraint code verbatim; 26 tests as in the original report |
| Round 6 stage 0 (2026-09-09) | floor / eta_down in `rewards.py`, `next_token_probs`, `YesProbabilityFeature`, driver flags, `train_log`, multi-dir `summarize_llm.py`, synthetic drift, `tests/test_llm_round6.py` | `LagrangianReward` and `YesProbabilityFeature` verbatim; Stage A table reproduced (drift 0.063 / 0.012 vs 0.065 / 0.012) |

Everything else in `seldonian/llm/`, `scripts/` and `reports/llm_*.md`,
`reports/paper_seldonian_llm.md` is the session's own text, replayed. The test suite
count at each point matches the original (310 + 1 skipped before stage 0, 317 + 1 after).

`uv.lock` was regenerated with `uv lock --exclude-newer 2026-09-02T05:15:00Z`, the time of
the session's `uv sync --extra rl`; it resolves trl 1.12.0 and torch 2.13.0 as the paper states.

## Commit hashes

The session and the reports cite the original hashes; the rebuilt commits map as follows. The first two were
rebased onto `254a3d7` on 2026-09-07, after which the session cites `e22d3c3` as `82cf288`.

| original | rebuilt | subject |
|---|---|---|
| 928d668 | acaf8ae | Add Seldonian RL post-training for LLMs and Round 1 pilot |
| e22d3c3 | 17323d3 | Correct the predicted safety test and add Round 1b-3 drivers and report |
| 5be849f | 98a13ca | Add open-ended constraints, a synthetic Seldonian environment and pressure tasks |
| 300456d | c733467 | Add Round 4-5 drivers: pressure tasks, discrim task, calibration harnesses |
| 668b9ad | 81d1b46 | Add the Round 4-5 evaluation-design report and calibration tables |
| 24ca1a9 | dc30dae | Add the Round 6 plan: dual dynamics, attribution, over-refusal, DiscrimEval |
| c6a4a03 | 8482448 | Add the working paper on Seldonian post-training of language models (draft v0.1) |
| fca64a1 | 49fe6f2 | Round 6 stage 0: multiplier floor, harm group, yes-probability feature, train logs |
| d367fee | 352b962 | Fold Stage 0 and Stage A results into the Round 6 plan and the paper |
| 35ee493 | 49d3219 | Round 6 stage E: judge calibration tooling, and the second-half GPU queue |
| 7ed402f | 52cf1f7 | summarize_llm: drift column (last predicted rate minus the checkpoint minimum, with the final multiplier) |
| b49c5b8 | 5651c06 | Round 6 stage E: provisional (Claude-labelled) judge calibration |
| 62678a7 | 2097739 | Lagrangian multiplier: floor-from-start option (--lam-floor-always, synthetic --floor-always) |
| cecd223 | 87422ad | Round 6 plan: B1 bonus-8 result (floor halves the drift, first excursion unchanged) |
| 9a1f1f0 | f3a9ad6 | Round 6: always-on floor for the brevity stages (synthetic sweep), B1b rerun queued first |
| b6fd146 | 4cfe93d | Round 6 plan: B1 bonus-16 result (a floor of 5 against a pressure of 16 changes nothing) |
| 70085ad | d2b8a1b | Paper: sections 6.8 (the floor on brevity) and 6.9 (provisional judge calibration) |
| 5bf04dc | 1f9e257 | Round 6 stage C: harm margin 0.045 (adversarial-group width 0.039 at n_s 1200) |
| d661bcf | 7c5f556 | Round 6 queue: composite penalty 4 arms for stage C in their own directory (skipped by the first queue) |
| f19ae22 | b6ae381 | Round 6 plan: stage C seed 0 (grpo and penalty 1 breach, Seldonian certified at 88% of the gain) |
| c07312a | 30e36b6 | Round 6 plan: stage C seed 1 (grpo and penalty 1 breach; Seldonian NSF on a 0.05 winner's-curse gap) |
| 3479b39 | 4928713 | Round 6 plan: stage C three-seed table (grpo and penalty 1 breach 3/3; Seldonian 0/3 breaches, 2/3 solutions, 82% of the gain) |
| cffda76 | 366fc94 | Stage C complete: penalty 4 breaches 1/3; paper section 6.10 and attribution (draft v0.3) |
| 11d4667 | 192eb64 | Paper: section 6.10 (over-refusal attribution), abstract item (v), draft v0.3 |

## Still missing

- `results/` (every run's `result.json`, episodes, logs, the synthetic and calibration
  tables the commits included) and `.cache/judges`. The numbers survive only as tables in
  the reports and the paper. The CPU-only results (`results/synthetic/`) can be regenerated
  with `scripts/synthetic_calibration.py` using the commands in the reports.
- Work after 2026-09-12 01:01. The published paper (`llm_working_paper_v0.3_2026-09-15`
  here) already contains Round 6 B1b, B2, B4 and D results and the "hardening the judge" /
  "moving landscape" analysis, so another session ran after this one. If its link is found,
  it can be replayed the same way.
- The old project memory was restored into Claude Code's memory folder for this project.

## Resuming the session

The transcript was placed at
`~/.claude/projects/-home-hannanabdul-seldonian-fairness/47ff14c6-c6bd-495e-bffc-414e1ceedd6e.jsonl`,
so `claude --resume 47ff14c6-c6bd-495e-bffc-414e1ceedd6e` from the repo root continues it.
Its last message says the brevity rerun with the always-on floor is training, then B2, B3,
D and B4 (`scripts/run_round6b.sh`); none of those runs survive, so the queue has to be
restarted (it skips finished runs by `result.json`, and all are missing). Rounds 1-6
thresholds are per-seed reference files under `results/`, so reference runs must be redone first.

## Published write-ups kept here

| file | source artifact |
|---|---|
| `llm_working_paper_v0.3_2026-09-15` | https://claude.ai/artifact/GV4ECwqSvz7bqdypAVFUNR |
| `llm_round4_round5_2026-09-09` | https://claude.ai/artifact/QPAkiPhiQUAnDcoY3ksALy |
| `llm_plan_rounds1-3_2026-09-05` | https://claude.ai/artifact/2ctgm1Qc7ZMPU4UvNsVbRF |

Each `.html` is the page as published; each `.md` is a searchable text extraction.
