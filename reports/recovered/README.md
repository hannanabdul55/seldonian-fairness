# Recovered LLM post-training write-ups

On 2026-09-19 the WSL distro holding this repo was unregistered, which deleted its
disk image. The branch `llm-seldonian-rl` was never pushed, so its code and every
`results/` file from Rounds 1-6 were lost. Local Claude Code transcripts were stored
on the same disk and were lost as well.

The files here are the write-ups that had been published as claude.ai artifacts,
downloaded again on 2026-09-19. Each `.html` is the page exactly as published; each
`.md` is a text extraction of it that can be searched with grep.

| file | source artifact | covers |
|---|---|---|
| `llm_working_paper_v0.3_2026-09-15` | https://claude.ai/artifact/GV4ECwqSvz7bqdypAVFUNR | **Most complete.** Method, tasks, setup, all results Rounds 1-6, analysis, limitations, next steps, reproduction commands |
| `llm_round4_round5_2026-09-09` | https://claude.ai/artifact/QPAkiPhiQUAnDcoY3ksALy | Round 4 evaluation design + Round 5 brevity task, with extra per-step detail |
| `llm_plan_rounds1-3_2026-09-05` | https://claude.ai/artifact/2ctgm1Qc7ZMPU4UvNsVbRF | Original plan, Round 1 pilot gates, Round 1b/1c/2, GSM8K control |

Already in git (not lost): `reports/bounds/` (bounds study, also at
https://claude.ai/artifact/XcsLGbBWhVLg722EKXwNub) and everything on `master`
up to commit 254a3d7 (2026-09-07).

## What is still lost

The per-run `result.json` files and training logs cannot be recovered; the
write-ups hold the tables derived from them. The missing code is:

- `seldonian/llm/`: `policy.py`, `backend.py`, `judges.py`, `rewards.py`,
  `constraints.py`, `data.py`, `discrim.py`, `synthetic.py`, `calibration.py`
- `scripts/`: `run_llm_rl.py`, `summarize_llm.py`, `plot_seldonian.py`,
  `resample_calibration.py`, `synthetic_calibration.py`, `judge_calibration.py`,
  `run_round1.sh` ... `run_round6b.sh`
- `tests/`: `test_llm_policy.py`, `test_llm_constraints.py`, `test_llm_discrim.py`,
  `test_llm_synthetic.py`
- `reports/`: `llm_round1_pilot.md`, `llm_round2.md`,
  `llm_round4_evaluation_design.md`, `llm_round6_plan.md`
- `results/`: `llm_r4/`, `llm_r5/`, `llm_r6/`, `synthetic/`, `calibration/`

The working paper (sections 3, 5 and Appendix B) records the design, the
hyperparameters and the CLI flags in enough detail to rebuild these.
