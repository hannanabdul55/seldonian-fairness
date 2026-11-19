# Q2 registration: a rubric that names the long refusal

registered_at: 2026-11-19T15:31:11Z (UTC, from the clock when this file was written, minutes before its commit)

Question: is Section 5.1's failure of a carried calibration a property of the compiled judge we used, which misses
refusals followed by a helpful-sounding body, or does a rubric judge told about that form fail too?

One new developer sentence, fixed in `scripts/rubric7_judge.py` (SENTENCE), is compiled by the same model (Qwen3-8B)
with spike 015's compile prompt, greedy, and used with the same judge prompt on the full responses of the pre-update set
and the three seeds' step-200 sets (500 each; 2,000 forward passes). Gold is the guard's flag. Carried bound:
`transfer017.youden_plasmode`, 4,000 draws, rng 1119, over if miss > delta + 2 se. The rubric has not been compiled and
no score under it exists.

- N1. Recall on the guard's refusals before the update is at least 0.50 (wording 0: 0.21).
- N2. Recall at step 200 is at least 0.50 in each of the three seeds.
- N3. Recall before and after does not differ (Fisher p >= 0.05) in at least two of three seeds.
- N4. The carried bound is over its level in no seed.
- N5. The false-alarm rate on the guard's non-refusals before the update is at most 0.05.

Rule. N2 and N4 kept: the paper says the failure across the update belongs to a judge that misses long refusals, that a
rubric naming them carried on the same sets, and narrows the abstract and Section 5.1 to that. N4 refuted: the paper says
a rubric judge told about the form still did not carry, so the failure is not explained by that blind spot alone. In
both cases the paper states the limits below.

Limits fixed now. The sentence was written after the Q1 results, on the same four sets, knowing which responses the old
judge missed; it is one wording with no paraphrases; a pass shows that a rubric designed with hindsight carried across
this update, not that a rubric written before an update would. The sentence is not revised after the run; if the
compiled rubric is malformed (empty, or not instructions), that is reported and the run stops.

Seen before registration: everything in Q1's results, the six compiled rubrics of spike 015 (several say a response that
gives relevant information is not a refusal), and `score --dry`'s item count.
