# TeamBench-first optimization protocol

Status: frozen before the first missing development-baseline run and before
training a benchmark strategy selector.

## Objective

MultiTown's current optimization target is the public TeamBench task suite.
Synthetic-town and tabletop-RPG work is downstream and must not be used as
evidence that the benchmark method is better.

The immediate target is a quality/cost Pareto improvement over five frozen
conditions run with the same strong and weak models:

- `Solo-TB`: one strong, full-access agent;
- `PlanExecute-TB`: strong Planner followed by weak Executor;
- `ExecuteReview-TB`: weak Executor followed by strong Verifier;
- `FixedTeam-TB`: fixed Planner–Executor–Verifier organization;
- `MultiTown-TB`: the existing deterministic selective organization.

The existing 89-task, seed-0 matrix has already been inspected. It is therefore
discovery evidence, not an untouched test set. No controller selected after
2026-08-24 may be described as preregistered or independently confirmed from
that matrix alone.

## Data partitions

1. **Selection set:** the frozen 30-task TeamBench development split at seed 0.
   Before fitting a selector, complete all five strategy outcomes on these same
   tasks. Model choice, temperature, sandbox, role permissions and turn caps
   must match the frozen comparison.
2. **Public discovery set:** the existing 89 evaluable TeamBench-90 tasks at
   seed 0. It may be used for transparent post-hoc evaluation and failure
   analysis, but not for tuning followed by a held-out claim.
3. **Confirmation set:** a new generated seed frozen before candidate outcomes
   are opened. At minimum, compare the frozen candidate with `Solo-TB` and
   `PlanExecute-TB`, the two current Pareto-front methods. A five-condition
   confirmation remains preferable when compute permits.
4. **Official hidden track:** if submitted later, TeamBench's rotating hidden
   seeds are the strongest independent confirmation. Hidden results must never
   be inferred from public-seed results.

Task IDs, seeds, split hashes, source revision, model aliases, model file hashes,
Docker image ID and controller hash are recorded for every formal run.

## Candidate ladder

The optimization proceeds in increasing methodological strength:

1. `MT-Selector`: a pre-task contextual policy that chooses a frozen
   organization from observable task features. This is a strategy router or
   contextual bandit, **not Agentic RL**.
2. `MT-Sequential`: a sequential policy whose state includes observable task
   features, consumed budget, runtime checks, role messages and independent
   review status. Its action space contains `stop`, `delegate`, `escalate`,
   `review` and `human/abstain`. Only a policy actually optimized from
   trajectories may be called the Agentic-RL candidate.

The sequential candidate must be compared on TeamBench before it is used to
justify another product branch. Offline replay numbers and oracle routing are
diagnostics, not final benchmark results.

## Optimization target and constraints

The quality-first reward used for training is declared before fitting:

```text
reward = partial_score + 0.25 * passed
         - 0.05 * log1p(total_tokens / reference_median_tokens)
         - 0.25 * invocation_error
```

Training may use only selection-set outcomes. The hidden grader, test outcome,
task-specific expected output and test-set strategy label are never policy
inputs. Features must be available before the corresponding action.

All methods use the same task instance, temperature zero, local model aliases,
maximum response-token setting, sandbox and deterministic grader. Organization
cost is part of the treatment, so actual input/output tokens, latency and
energy are reported rather than equalized away. Budgeted success/goodput is
also reported at shared token caps.

## Required report

For every fixed baseline and candidate, report:

- passes and pass rate;
- mean partial score;
- mean input, output and total tokens;
- clean median and p95 end-to-end latency;
- monitored energy where measurement is valid;
- request errors, failure modes and action/route counts;
- paired bootstrap confidence intervals for partial score, tokens and latency;
- exact paired pass comparison (McNemar) and a Pareto-front table;
- budgeted pass/partial-score curves at common token caps.

An oracle selector may be shown only as an explicitly unattainable upper bound.

## Claim gates

- **Benchmark winner:** the frozen candidate exceeds every completed fixed
  baseline on the declared primary quality endpoint on the confirmation set.
- **Pareto winner:** no completed baseline is at least as good in quality while
  using no more tokens, and uncertainty is reported rather than hidden.
- **Agentic RL:** allowed only for the trained sequential policy, never for a
  hand-written controller, retrospective oracle or one-step contextual router.
- A failure to beat `Solo-TB` or `PlanExecute-TB` is retained as a negative
  result. No synthetic-town score may be substituted for the missing public
  benchmark evidence.
