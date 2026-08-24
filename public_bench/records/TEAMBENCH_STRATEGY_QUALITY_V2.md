# TeamBench mainstream-strategy comparison

Status: completed paired quality/token comparison on 89 public tasks. Raw model
messages, task workspaces and system logs remain private.

## What was compared

The five rows are concrete organizations in the same TeamBench role-isolation
harness, not cross-paper leaderboard numbers:

- **Solo-TB:** one strong, unrestricted agent;
- **PlanExecute-TB:** strong Planner -> weak Executor, with no Verifier;
- **ExecuteReview-TB:** weak Executor -> strong Verifier, with no Planner or
  repair loop;
- **FixedTeam-TB:** strong Planner -> weak Executor -> strong Verifier, with at
  most one repair/reverification loop;
- **MultiTown-TB:** weak-first execution with deterministic selective planning,
  review and remediation.

All rows use the same 89 task IDs, task hash, deterministic graders, Docker
image, models, temperature 0 and 2,048-token per-call output cap. The first
three were rerun after the sandbox lifecycle repair. FixedTeam-TB and
MultiTown-TB use the already frozen v1.2 task-quality/token rows. Their runner
source revision differs, so the five-way table intentionally excludes latency,
energy and other system measurements.

## Five-way quality/token result

| Strategy | Fully passed | Mean partial score | Mean tokens/task |
| --- | ---: | ---: | ---: |
| Solo-TB | 16 / 89 | **0.64180** | 82,869 |
| PlanExecute-TB | **18 / 89** | 0.62434 | **49,166** |
| ExecuteReview-TB | 10 / 89 | 0.54940 | 67,011 |
| FixedTeam-TB | 14 / 89 | 0.63375 | 108,381 |
| MultiTown-TB | 11 / 89 | 0.58251 | 68,218 |

The quality/token Pareto frontier is **Solo-TB and PlanExecute-TB**. The result
does not show universal multi-agent superiority. It shows that organization
choice matters:

- PlanExecute-TB achieved the largest pass count and used **40.67% fewer
  tokens** than Solo-TB. Its mean partial-score difference versus Solo-TB was
  -0.01745 (95% paired bootstrap CI [-0.05573, +0.01861]); the pass difference
  was not significant (exact McNemar p=0.7744). This supports an efficiency
  trade-off, not a quality-superiority claim.
- ExecuteReview-TB was worse than PlanExecute-TB by **-0.07494 partial-score
  points** (95% CI [-0.11236, -0.04090]), produced 10 rather than 18 passes and
  used **36.30% more tokens**. Adding a strong read-only Verifier without a
  repair loop was harmful in this harness.
- FixedTeam-TB used **120.44% more tokens** than PlanExecute-TB. Its partial
  score was only +0.00940 higher (95% CI [-0.02745, +0.04940]) and it produced
  fewer passes (14 vs 18). Always activating every role was not cost-effective.
- MultiTown-TB beat ExecuteReview-TB in mean partial score by **+0.03311** (95%
  CI [+0.00161, +0.06592]), but it did not beat PlanExecute-TB or Solo-TB. The
  current dynamic controller is therefore a negative/diagnostic result, not the
  winning method.

## Clean post-fix three-way runtime result

The following three rows share the repaired runner source revision and may be
compared on runtime and monitored energy:

| Strategy | Median latency | p95 latency | Monitored energy | Request errors |
| --- | ---: | ---: | ---: | ---: |
| Solo-TB | 48.12 s | 96.86 s | 78.67 Wh | 0 |
| PlanExecute-TB | **37.49 s** | **85.16 s** | **59.30 Wh** | 0 |
| ExecuteReview-TB | 35.96 s | 124.18 s | 63.00 Wh | 0 |

PlanExecute-TB reduced mean latency by 17.17% and monitored energy by 24.63%
relative to Solo-TB. ExecuteReview-TB has a slightly smaller median but a much
worse p95; it is not preferred because quality and tail latency both degrade.

## Claim boundary

This branch can claim a controlled mainstream-strategy study, a Pareto result,
and a useful negative finding: more roles are not automatically better, and a
Verifier without a repair path can add cost while reducing quality. It cannot
claim that MultiTown-TB beats MetaGPT, DyLAN, AgentPrune or TeamBench leaderboard
systems, because those exact implementations were not rerun under this local
contract. It also cannot claim that the current dynamic controller is the best
of the five; it is not.

The machine-readable compact summary is
[`teambench-strategy-quality-v2-summary.json`](teambench-strategy-quality-v2-summary.json).
The report generator now rejects cross-revision runtime comparisons by default
and requires an explicit `--quality-only` mode for this type of bridge table.
