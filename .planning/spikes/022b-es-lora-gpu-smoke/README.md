---
spike: 022b
idea: es-post-training
name: es-lora-gpu-smoke
type: standard
validates: "Given Granite-3.3-2B with 014's LoRA adapter and 014's reward stack on the 12 GB GPU, when an ES backend perturbs the adapter from an fp32 master copy, then it fits in memory, a seed replays a perturbation exactly, and we know its completions per GPU-hour against GRPO (022a assumed 8x)"
verdict: PARTIAL
related: [022a, 014]
tags: [evolution-strategies, lora, granite, gpu, throughput]
---

# Spike 022b: ES on the LoRA adapter, GPU smoke

Status 2026-11-11: done. Seed replay is exact on the GPU. ES makes 0.5x GRPO's completions
per GPU-hour at 8 prompts per member and 1.5x at 32, not the 8x that 022a assumed. **The
kill rule for 022c fired; the decision is the user's.**

## What This Validates

Three measurements, none of them about learning (that is 022c):

1. peak GPU memory of ES with 014's stack resident (Granite-2B, Qwen3Guard-4B 4-bit, the
   Skywork reward model);
2. completions per GPU-hour, ES against GRPO, on the same machine in the same session;
3. whether a seed reproduces a perturbation, and the (seeds, fitness) log an update, exactly.

## Research

No new literature (`.planning/notes/es-vs-grpo-lora.md`). Library facts read from the
installed code (torch 2.13, peft 0.20, transformers 5.16, trl 1.12):

- PEFT casts a bf16 adapter to fp32 (`autocast_adapter_dtype`, on by default). Under the
  project's GPU loading path the base is bf16 and **the adapter is already fp32** (check
  C1b). The note's worry, that a sigma of 1e-3 would be lost in bf16, does not apply to
  LoRA; it would apply to full-parameter ES.
- TRL's KL term is `exp(d) - d - 1`, `d = log pi_ref - log pi`, per sampled token. The
  backend uses the same term in the fitness.
- 014's adapter: 560 tensors, 28.2 million weights, 107.5 MB in fp32.

| Choice | Taken | Why |
|---|---|---|
| Where the adapter's value lives | fp32 master copy; members are written from it | in-place add-then-subtract fails to restore 37% of weights exactly, even in fp32 (C6) |
| Noise | regenerated from an integer seed on the adapter's device | nothing but seeds and scores need storing; streams differ between CPU and CUDA |
| Policy log-probs for the KL term | read off the generation scores | no second forward pass per member; equal to a forward pass to 1e-5 (C3) |
| Reward scoring | all members' completions in one call | one multiplier for the whole population; the judge batches across members |

## How to Run

    ../../../.venv/bin/python check.py        # CPU, about 1 minute, writes check.json
    ./run.sh                                  # GPU, 24 minutes, takes the GPU lock, 40-minute cap
                                              # writes results/spikes/022b/smoke.json

## CPU checks (2026-11-11, 10 of 10 pass)

A random two-layer model with Granite's architecture and tokenizer stands in for Granite.

| Check | Result |
|---|---|
| C1 a seed regenerates the same noise | bit-identical; another seed differs |
| C1b dtypes under the GPU loading path | base bf16, adapter fp32 |
| C2 perturb and put back 200 times | adapter hash unchanged, with fp32 and bf16 base |
| C3 generation scores against a forward pass | largest gap 1.2e-5, the same alone and in a padded batch |
| C3b, C3c KL term | 1e-11 while B = 0; 5.2e-5 once the adapter acts |
| C4 ES estimate against backprop | cosine 0.509, theory 0.518 (1,500 pairs, 4,096 weights) |
| C5 the Seldonian loop on the backend | 20 equivalent steps, 4 predicted tests, checkpoint round-trip, the log file rebuilds the trained adapter bit for bit |
| C6 in-place perturbation on 014's adapter | not restored exactly: 37% of weights (sigma 1e-3), 64% (1e-2), fp32 |

## Investigation Trail

1. **First run: 8 of 10.** Both failures were fp32 arithmetic, not the backend's logic.
   - The KL term came out exactly 0 with a working adapter: `exp(d) - d - 1` cancels to
     zero in fp32 once `|d|` is under about 3e-4. The backend now computes it in fp64.
     TRL computes the same expression in the training dtype, so GRPO's KL penalty is
     probably also zero early in training; not checked.
   - The ES estimate was noisier than theory (cosine 0.40, expected 0.52) at sigma 1e-3:
     the fitness difference of a pair was near the rounding error of an fp32 forward
     pass. At sigma 2e-2 it matches. This matters for 022c: Qiu's sigma 1e-3 with
     a deterministic fitness is near that floor. With sampled completions the fitness is
     far noisier than rounding, so the floor is not the binding problem there.
2. **Second run: 10 of 10.**
3. **GPU run, one attempt, no errors.** Surprises: memory at 32 prompts and above (stored
   scores), GRPO's first step being the fastest, and TRL's "ref" adapter. The timings at
   64 prompts are spoiled by memory spill and are reported as such.

## Plan for the GPU run (written before it)

One process under `flock /tmp/claude-gpu.lock`, `timeout` 40 minutes, results to
`results/spikes/022b/`, 014's checkpoint `s0/checkpoints/feasible-step100` loaded so B is
not zero.

- Generation time against batch size (8, 16, 32, 64 prompts per member).
- One full ES generation at (population 30, 8 prompts) and one at (8, 32), with 014's
  Lagrangian reward; per-part timing (perturb, generate, reference pass, reward, update).
- C1, C2 and the replay of C5 repeated on CUDA; whether two runs of one member with the
  same sampling seed give the same completions.
- GRPO reference: 5 steps of 014's configuration, `step_time` read from the TRL log.

Expectations:

1. Peak memory under 11 GB; ES needs no more than GRPO did (11.98 GB at 014's spill).
2. Replay is exact on CUDA for noise, weights and the update. Completions repeat for the
   same member and seed; I am less sure of this one.
3. ES gives **well under 8x** GRPO's completions per GPU-hour, and at 8 prompts per
   member possibly under 2x: a member generates its own small batch, and a batch of 8
   takes nearly as long as a batch of 64. GRPO in 014 made 64 completions in about 62 s.

**Kill rule for 022c.** If ES at its best batch size gives under 2x GRPO's completions per
GPU-hour, the equal-GPU-hours arm of 022c is the equal-generations arm, where 022a has ES
at 69% of GRPO's reward; the decision goes to the user before any DESIGN.md is written.

## Results

GPU run 2026-11-11, 24 minutes, `results/spikes/022b/smoke.json`. Each ES configuration
is one generation and the GRPO reference is five steps, so every rate below is one
measurement with visible noise (GRPO's steps took 38 to 83 s).

### Verdict: PARTIAL

Exactness holds. Memory holds only at 8 prompts per member. Throughput is far below the
assumption, and the kill rule fired.

### Throughput (expectation 3: met, and worse than feared)

| Arm | Completions | Seconds | Completions per GPU-hour | Against GRPO | Peak memory |
|---|---|---|---|---|---|
| GRPO, 014's configuration, steps 2-5 | 64 per step | 63.0 per step | 3,658 | 1 | 11.27 GB |
| ES, population 30, 8 prompts each | 240 | 468 | 1,847 | 0.50x | 10.98 GB |
| ES, population 8, 32 prompts each | 256 | 165 | 5,569 | 1.52x | 12.65 GB |

The script drops GRPO's first step as warm-up, which was written before the run; the first
step was in fact the fastest. With all five steps GRPO is 3,944 per hour and the ratios
are 0.47x and 1.41x. 014 measured about 62 s per step, so the reference agrees with it.

Where an ES generation's time goes:

| Part | Population 30, 8 prompts | Population 8, 32 prompts |
|---|---|---|
| Generating | 422 s (90%) | 117 s (71%) |
| Reward model and judge | 38 s | 41 s (25%) |
| Reference pass for the KL term | 6 s | 6 s |
| Writing members into the model | 2.0 s | 0.6 s |
| Update | 0.6 s | 0.2 s |

One member's batch takes about 14 s whether it holds 8, 16 or 32 prompts (curve: 12.7,
15.3, 14.6 s), so the cost is per decoding step, not per completion. Each member needs its
own batch because its weights differ. Perturbing and updating cost nothing that matters.

**Why 8x was never available.** ES only saves GRPO's backward pass. Both arms pay for
generation and for the reward model and judge, and at population 8 with 32 prompts the
reward scoring is already a quarter of ES's time. A batch of 64 was not measured cleanly
(below). If it also took about 15 s, ES would reach roughly 2.4x; that is an estimate, not
a measurement.

### Memory (expectation 1: not met at 32 prompts and above)

The stack at rest holds 8.93 GB (Granite, the 4-bit judge, the reward model, the ES
copies). ES at 8 prompts per member peaks at 10.98 GB, below GRPO's 11.27 GB. At 32
prompts it peaks at 12.65 GB, over the card's 12 GB, and at 64 at 16.2 GB; both ran only
because WSL spills into shared memory, which is slow.

The cause is the backend keeping every decoding step's full-vocabulary scores to read the
log-probs for the KL term: without them, 32 prompts peak at 9.56 GB and 64 at 10.43 GB.
The fix is to pick out the sampled token's log-prob at each step and keep only that. Not
done here. The two timings at 64 prompts (35 s and 65 s) were taken while spilling and are
not usable.

### Exactness on CUDA (expectation 2: met)

- The same seed gives bit-identical noise; another seed differs.
- 200 perturb-and-restore rounds leave the adapter's hash unchanged (49 ms per round).
- One member with one sampling seed gives the same tokens twice, and again after the
  member is restored and rewritten (8 and 32 prompts).
- For both configurations the update rebuilt from the logged seeds and fitness, passed
  through JSON, equals the update that was applied, bit for bit.

### Two things 022c would need to know

- **Generation scores are a noisy stand-in for a forward pass under bf16.** Per token they
  differ from a teacher-forced pass of the same model by 0.017 on average and up to 0.43.
  The KL term of a model against itself comes out at 7e-4 instead of 0. Against the
  reference the term is 0.08-0.10 at this checkpoint, so the floor is under 1% of it here,
  but it would matter near the start of training, where 014's GRPO logged 9e-4.
- **TRL's KL reference is not always the base model.** When training starts from an
  adapter that is already trained, TRL copies it as a "ref" adapter and measures KL
  against that copy: the GRPO reference here logged 7e-4 at its first step, while the same
  checkpoint is 0.09 from the base model. 014 started from a fresh adapter, so its
  reference was the base model, as is the ES backend's.

At sigma 1e-3 a member's noise has norm 5.3 over the adapter, against 1.04 for the
checkpoint's B matrices. Members' KL from the base (0.08-0.10) is about what 014 logged
for the unperturbed adapter at this step (0.095), so the noise adds little KL.

### Kill rule

Fired: the best measured configuration gives 1.52x, under 2x. For 022c the equal-GPU-hours
arm is then close to the equal-generations arm, where 022a has ES at 69-78% of GRPO's
reward. Not settled by this spike: a clean batch of 64 after the memory fix (estimate
2.4x), and whether a population of 8 is enough for ES to learn at all. No DESIGN.md for
022c is written until the user decides.
