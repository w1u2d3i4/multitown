# TeamBench mainstream strategy matrix protocol

Status: frozen before the first formal `PlanExecute` or `ExecuteReview` test run.

Protocol date: 2026-08-24.

## Question

Does MultiTown's selective team (`A8-TB`) provide a useful quality/cost trade-off
against representative single-agent and fixed multi-agent organizations when every
method is evaluated on the same public tasks, model tiers, sandbox and grader?

This comparison tests **agent organization**, not just different prompts. The direct
matrix deliberately follows TeamBench's role-isolation protocol so that the Planner,
Executor and Verifier do not silently collapse into one full-access agent.

## Frozen methods

| Public label | Runner method | Organization | Role isolation | TeamBench analogue |
| --- | --- | --- | --- | --- |
| Solo-TB | `Solo` | one strong full-access agent implements and self-certifies | No | Oracle / Solo |
| PlanExecute-TB | `PlanExecute` | strong Planner -> weak Executor; no independent review | Yes | Team No Verify |
| ExecuteReview-TB | `ExecuteReview` | weak Executor -> strong Verifier; no planning or remediation | Yes | Team No Plan |
| FixedTeam-TB | `A4` | strong Planner -> weak Executor -> strong Verifier, with at most one remediation/reverification loop | Yes | Full fixed team |
| MultiTown-TB | `A8` | weak-first execution with deterministic selective planning/review/remediation | Yes | Proposed dynamic controller |

`PlanExecute-TB` receives a controller-written passing attestation after execution,
exactly because no independent Verifier is present. This only satisfies TeamBench's
submission protocol; the hidden grader still determines task quality.

`ExecuteReview-TB` stops after independent verification. Adding a repair loop would
change the official no-planner ablation and is therefore excluded from this frozen
baseline.

## Frozen execution controls

- TeamBench public test split: all 89 unique tasks in the frozen split file.
- Strong tier: `qwen-game` / Qwen3.5-35B-A3B, temperature 0.
- Weak tier: `qwen-mm-backup` / Qwen3-VL-4B-Instruct, temperature 0.
- Maximum output tokens per model turn: 2,048.
- Same Docker image, task setup, hidden grader and filesystem policy.
- Same role-turn caps used by the existing formal A4/A8 run:
  Planner 6, Executor 12, Verifier 8; method-specific absent roles consume zero.
- One method runs at a time. Monitoring begins before the first task and ends after
  the final task, so concurrent inference does not contaminate latency or energy.
- Successful task rows are never rerun. An infrastructure-error row may be retried
  only under the existing archive-and-resume policy and remains auditable.

## Frozen metrics and comparisons

Report per method:

- exact pass count/rate and deterministic mean partial score;
- mean input/output/total tokens;
- median and p95 task end-to-end latency;
- system monitoring summaries including CPU, RAM and GPU power/energy where available;
- role activation counts and route distribution;
- invocation-error count.

Report paired `MultiTown-TB` comparisons against every other method:

- paired mean partial-score difference with a 95% bootstrap interval;
- better/tie/worse task counts;
- exact McNemar test on pass/fail outcomes;
- mean token and latency reduction;
- five-method quality/token Pareto frontier.

No non-inferiority claim is allowed unless its margin was frozen before examining the
new result. External paper results are related-work context only and are not mixed into
the local numeric table.

## Interpretation constraints

- A role-isolated method may be valuable for governance even if unrestricted Solo-TB
  has higher raw quality.
- MultiTown-TB is an efficiency result only if it improves cost at an explicitly
  reported quality trade-off; it is not automatically a quality winner.
- If MultiTown-TB is dominated by any direct baseline on both partial score and tokens,
  that negative result must remain in the report.
- These deterministic rules are not a trained Agentic RL policy.
