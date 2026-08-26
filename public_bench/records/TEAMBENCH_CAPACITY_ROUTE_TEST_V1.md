# TeamBench MT-CapacityRoute-v1 formal record

Status: completed positive local operating-point result.

Protocol frozen: 2026-08-26, before the first generator-seed-5 invocation.

## Question

Can a deterministic capacity-aware Planner→Executor organization improve the
quality/cost point over both the efficient role-separated PlanExecute baseline
and TeamBench's full-access strong Solo anchor?

The route uses the strong Executor only for `Distributed Systems`, `Operations`
and `Security`. Those categories were selected from development evidence. No
test task ID, hidden grader result, workspace outcome or runtime failure signal
is available to the router. This is a non-RL metadata policy.

## Matched result

All methods completed the same 89 generator-seed-5 test instances with sampling
seed 20260841, temperature 0, 2,048 maximum output tokens per request, pinned
models, pinned Docker image, identical task order and zero invocation errors.

| Method | Passes | Mean partial | Mean tokens | Mean latency | P95 latency | Energy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PlanExecute | 16/89 | 0.61603 | 49,579 | 69.21 s | 151.96 s | 67.89 Wh |
| Solo | 14/89 | 0.63989 | 84,085 | 75.60 s | 237.06 s | 90.94 Wh |
| **MT-CapacityRoute** | **20/89** | **0.64180** | **48,296** | **68.24 s** | **91.06 s** | **63.32 Wh** |

MT-CapacityRoute has the highest pass count and mean partial score and the
lowest mean token use. Relative to Solo it adds six passes and loses none
(two-sided exact McNemar p=0.03125), reduces tokens by 42.56%, mean latency by
9.73% and monitored energy by 30.37%. Relative to PlanExecute it adds five
passes and loses one, reduces tokens by 2.59%, mean latency by 1.40% and energy
by 6.74%.

Paired mean partial-score differences:

- versus PlanExecute: +0.02577, 95% bootstrap CI [-0.00603, +0.06142];
- versus Solo: +0.00191, 95% bootstrap CI [-0.03524, +0.03948].

## Route-effect audit

The candidate selected 14 strong and 75 weak Executor routes. Because local
generation can vary even at temperature 0, action-changing and unchanged tasks
are reported separately:

| Subset | N | Partial difference vs PlanExecute | Wins / ties / losses | Token change |
| --- | ---: | ---: | ---: | ---: |
| Strong route (action-changing) | 14 | +0.11096, CI [-0.00737, +0.23976] | 6 / 7 / 1 | -2,690/task |
| Weak route (repeat control) | 75 | +0.00987, CI [-0.01880, +0.04453] | 4 / 67 / 4 | -1,021/task |

A baseline-anchored audit that takes MT-CapacityRoute observations only for the
14 strong routes and PlanExecute observations for the other 75 tasks gives
18/89 passes, 0.63348 mean partial and 49,156 tokens/task. It estimates the
route action separately from repeat noise; it is not a fourth executed method.

## Decision and claim boundary

The frozen candidate gate passes and MT-CapacityRoute point-dominates both
same-seed anchors on passes, mean partial score and mean tokens. It is the
**strongest tested local operating point under this frozen harness**.

The evidence does not establish statistically superior mean partial score: both
paired confidence intervals include zero. It also does not establish “best
reported TeamBench method” or literature SOTA, because published systems use
different models, prompts, budgets and task instances. The benchmark-category
router is interpretable but benchmark-dependent.

## Integrity

- Upstream TeamBench revision: `2f060a33501b19fbc8d26f8ccdad7580e3b04635`
- MultiTown runner revision: `523de00870d8932f5abb98164fc2b5efa603e02b`
- Docker image ID: `sha256:d218bef3a99b863f630a66bca9a8d8b7bd0e6218078bcf856bf374a14ad06397`
- Split SHA-256: `376c01ba6c78819bc22b53b9ba6561b5cae5e4db9133b7f307e834755b964809`
- Task instances SHA-256: `7923eddc70ce9739e887ce472afb070f420eb89d2e83b441878c290b0f14ed37`
- PlanExecute results SHA-256: `32a772510a925d09cd4177a74d47f900006d0b079c0dafd2d4a58a38876c6128`
- Solo results SHA-256: `4ac2c25a40ad8c7e9001d58e5476778489f9c3191a06ac61dba6bf5a672a32a5`
- MT-CapacityRoute results SHA-256: `6e2b652ea3cd33003fe2c66cf83fda813b7c29eb57c0cece9d0d508adb1676a1`
- Invocation errors: 0 across all three methods
- `grader_timeout` failure modes: 0 across all three methods

Raw prompts, generated workspaces, request logs, task rows and telemetry are not
included in the public repository.
