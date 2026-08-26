# TeamBench capacity-route v1 protocol

Status: frozen before the first generator-seed-5 invocation.

Protocol date: 2026-08-26.

## Hypothesis and development boundary

The completed uniform strong-Executor experiment is negative overall, but its
task-level development evidence is structured rather than random. Across the
seven development tasks in `Distributed Systems`, `Operations` and `Security`,
the strong Executor improves three tasks, ties four and loses none. A
counterfactual reconstruction that changes only those seven task routes adds
one full pass, raises mean partial score by 0.02156 and uses 45 fewer tokens per
task. Strong execution outside those categories causes every observed quality
loss in the uniform candidate.

This reconstruction is **development evidence only**: it selected the routing
categories and cannot confirm the method. No task ID, hidden grader result,
workspace outcome or runtime failure signal is available to the router.

## Frozen method

`MTCapacityRoute-v1` is a deterministic non-RL capacity router:

- a Qwen3.5-35B-A3B Planner reads the full specification and sends a plan;
- the Executor remains brief-only and OS-isolated from the full specification;
- the Executor uses Qwen3.5-35B-A3B only when the public TeamBench category is
  `Distributed Systems`, `Operations` or `Security`;
- all other categories use Qwen3.5-4B;
- no Verifier, remediation, hidden grader signal or post-execution selection is
  used.

The route is an interpretable benchmark-metadata policy, not a learned policy
and not Agentic RL. Its benchmark-category dependence is a limitation and must
be reported.

## Fresh confirmation

The confirmation uses all 89 public test task IDs regenerated with unseen
generator seed `5`. `PlanExecute-TB` and `MTCapacityRoute-v1` run from task 1
with sampling seed `20260841`, the same task order, runner revision, Docker
image, temperature 0 and 2,048-token per-call cap. Generator seeds 0-4 and the
30-task development comparison are discovery evidence and are excluded from
confirmation.

Required outputs are full passes, deterministic mean partial score, paired
bootstrap intervals for partial score/tokens/latency, exact McNemar test,
win/tie/loss counts, route mix, p95 latency, monitored energy, invocation
errors and grader timeouts.

The candidate passes the confirmation gate only if it has at least as many
full passes as `PlanExecute-TB`, strictly higher mean partial score, zero
invocation errors and no grader-timeout increase. A quality-superiority claim
requires the paired partial-score interval to exclude zero. A Pareto claim
additionally requires no increase in mean tokens. Any failure or null result is
retained.

Even if this gate passes, “best reported TeamBench method” is not claimed until
the candidate is compared under the same seed-5 harness with every method that
could remain non-dominated. Cross-paper numbers with different models, task
instances or harnesses are related-work context, not a local ranking.

Raw prompts, workspaces, request logs and telemetry remain private. Aggregate
statistics, hashes, protocol code and non-sensitive plots may be published.

## Completed confirmation result

The frozen run completed all 89 paired tasks with zero invocation errors and
zero `grader_timeout` failure modes. The hard provenance fields match across
PlanExecute, Solo and MT-CapacityRoute: split and task-instance hashes,
generator and sampling seeds, upstream and runner revisions, Docker image,
model endpoints, temperature, output cap and controller hash.

| Method | Passes | Mean partial | Mean tokens | P95 latency | Energy |
| --- | ---: | ---: | ---: | ---: | ---: |
| PlanExecute | 16/89 | 0.61603 | 49,579 | 151.96 s | 67.89 Wh |
| Solo | 14/89 | 0.63989 | 84,085 | 237.06 s | 90.94 Wh |
| MT-CapacityRoute | 20/89 | 0.64180 | 48,296 | 91.06 s | 63.32 Wh |

The frozen confirmation gate passes. MT-CapacityRoute point-dominates both
same-seed anchors on full passes, mean partial score and mean tokens. Versus
PlanExecute, the paired partial difference is +0.02577 (95% bootstrap CI
[-0.00603, +0.06142]); versus Solo it is +0.00191 (95% CI [-0.03524,
+0.03948]). Neither interval supports partial-score superiority.

The 14 action-changing strong routes contribute +0.11096 partial on average
(95% CI [-0.00737, +0.23976]), with 6 wins, 7 ties and 1 loss. The 75 weak
routes are a repeat-control audit and contribute +0.00987 with a confidence
interval spanning zero. Combining candidate observations only on strong routes
with baseline observations elsewhere gives an 18-pass, 0.63348 audit estimate;
it is not a separately executed method.

The allowed claim is “strongest tested local operating point under the frozen
seed-5 harness.” “Statistically superior mean partial score,” “best reported
TeamBench method,” and “literature SOTA” remain unsupported.
