# Per-family guarantee, rule F: finish one family before starting the next (zero-GPU)

**Registered 2026-10-23, before this simulation is run.** Follows
`REGISTRATION-PER-FAMILY.md` (GO, results `results/per_family/`). A planning check after that
run (judge-label plasmode, not registered) found the registered adaptive rule A is
all-or-nothing: it spreads labels evenly over the open families, so they resolve together.
Stopped at a fixed budget on the primary cell (gated, tau = 0.05), A had resolved 0 of 7
families at 300, 600 and 1,000 labels, while uniform sampling had resolved 0.7, 1.1 and 3.8.
Human labellers stop early, so the human run needs a rule that certifies families one at a time.
Thresholds do not change after the run starts; changes are dated amendments below.

## Rule F

Same population, labels, decision rule, cells, bounds (anytime-valid beta-binomial mixture at
delta/K) and burn-in (10 labels per family) as rule A in `REGISTRATION-PER-FAMILY.md`. Only the
choice of the next family changes:

- `n*(m)`: the smallest label count `t` at which a family whose observed rate stays at `m`
  (`s = m t`) would be resolved under the decision rule, precomputed on a grid of `m` for each
  cell (capped at 10^6).
- Next family: the open family with the smallest estimated remaining labels
  `max(n*(mean_k) - T_k, 1)`; ties go to the family with more labels, then the lower index.

Once one zero-rate family pulls ahead it has the fewest labels left, so it is finished before
the next is started. Families near `tau` have large `n*` and come last. The bounds stay valid
because each family's labels are still i.i.d. draws from that family and the choice of family
uses only past labels.

## Compared at the same cells (1,000 runs each)

- **F** (this rule) and **A** (the registered APT rule), both sequential, mixture bounds.
- **U** (uniform, Clopper-Pearson at delta/K at a fixed n), as before.

Metrics: `n80` (as before), and **families resolved at a budget** of 300, 600, 1,000, 1,500 and
2,500 labels: mean number of the 7 families resolved when labelling stops at that count (U at
a fixed n equal to the budget).

## Predictions

| ID | Prediction | GO | NO-GO |
|---|---|---|---|
| V | P(any misclassification) <= 0.10 for F, every cell and budget | holds | fails anywhere |
| G | primary cell: F resolves at least as many families as U at budgets 600 and 1,000 | both hold | F < U at 1,000 |
| N | primary cell: `n80(F) / n80(A)` | <= 1.10 | > 1.50 |

Between GO and NO-GO is INCONCLUSIVE. **Verdict: GO if V holds and G and N are GO.** NO-GO if V
fails, or G or N is NO-GO. A GO makes F the rule for the human-label run (a separate
registration, with the guideline, page and labellers fixed before any label).

Pre-run expectation: a zero-rate family needs about 185 labels under the mixture bound at
tau = 0.05, so F should resolve about one family per 185-250 labels after the 70-label
burn-in: roughly 1 at 300, 3 at 600 and 5 at 1,000, while needing about as many labels as A
to resolve all 7.

## Amendments

(none)
