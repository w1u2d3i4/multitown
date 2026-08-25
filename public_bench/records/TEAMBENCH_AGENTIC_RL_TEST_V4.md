# TeamBench Agentic RL v4 public-test result

Status: completed negative result. The phase-specific value-of-information
controller satisfies its frozen cost, request-budget and reliability
constraints, but it does not improve the 89-task public test. It must not be
presented as better than PlanExecute or as benchmark-best.

Raw prompts, messages, requests and workspaces remain private.

## Frozen change

V4 retains the v3 pessimistic bootstrap-Q ensemble and p90 action reserves, but
learns separate lower-confidence-bound intervention margins after execution,
replanning and recovery. The selected margins are 0.01, 0.01 and 0,
respectively. Selection used only 85 complete development episodes from
generator seeds 0, 1 and 2. The already consumed seed-4 result was diagnostic,
not independent confirmation.

The policy and a single 89-task public-test run were frozen in the
[v4 protocol](../docs/TEAMBENCH_AGENTIC_RL_V4_PROTOCOL.md) before the first
test request. The primary baseline grades the exact PlanExecute prefix from the
same generated trajectory, so task-level changes measure the controller rather
than a second stochastic generation.

## Public-test result

| Method | Passes | Mean partial | Mean tokens | Median latency | p95 latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Same-trajectory PlanExecute | 19 / 89 | **0.63738** | **47,942** | **36.294 s** | 108.994 s |
| MT-Agentic-RL-v4 | 19 / 89 | 0.63592 | 53,564 | 41.215 s | 109.004 s |

V4 changes quality on only one task: `LH2_budgeted_workflow` falls from 0.83
to 0.70 after a recovery is kept. The paired mean partial-score difference is
-0.00146 (95% paired bootstrap interval [-0.00438, 0.00000]), with 0 wins, 88
ties and 1 loss. There are no new or lost full passes.

The controller adds 5,623 tokens/task (+11.728%; 95% paired interval +3,997 to
+7,344) and 4.40 seconds/task (95% paired interval +3.16 to +5.73). It directly
or conservatively stops on 51 tasks, keeps 25 recovery outputs, restores the
exact PlanExecute prefix on 13 tasks and calls no LLM Verifier.

## Budget and integrity result

The conservative request guard blocks a request on 13 tasks and caps a
completion on 4. No task exceeds 90,000 provider-reported tokens, the maximum
is 75,939, and there are zero provider overruns, invocation errors or selected
prefix hash mismatches. Across the whole run, 1,306 model requests are logged.

The system monitor records 6,204.28 seconds and 64.44 Wh for the complete run.
This energy number includes both the candidate path and same-trajectory shadow
grading; it is not attributable to either arm.

## Gate decision

The frozen basic gate fails because mean partial score is not strictly higher.
All other basic conditions pass: full passes do not decrease, mean token
overhead is below 20%, no task exceeds 90k and no invocation, sandbox or
provider-overrun error occurs. The strong gate also fails because there is no
new pass and the quality interval is not above zero.

This falsifies the v4 hypothesis that phase-specific value-of-information
margins generalize the seed-4 repairs to the public test. The observed result
supports budget control and deterministic rollback integrity, not performance
superiority. Further work needs new development trajectories and explicit
recoverability/tool-failure features; the public test must not be reused for
threshold selection.

Exact aggregate values and artifact hashes are in the
[summary JSON](teambench-agentic-rl-test-v4-summary.json).
