# TeamBench seed-1 selector confirmation

Status: completed negative confirmation on 89 public generator-seed-1 tasks.
Raw workspaces, prompts, request logs and system telemetry remain private.

## Question

Does the frozen `MT-Efficient-v2` contextual router improve on the matched
`PlanExecute-TB` baseline?  The router was selected on the 30-task development
set and frozen before either seed-1 run.  It selects `Solo-TB` for the three
`Adversarial / hard` tasks and `PlanExecute-TB` for the other 86 tasks.  It is a
contextual router, not Agentic RL.

Both runs use the same 89 task IDs, generated seed 1, per-request sampling seed
20260824, strong and weak local model aliases, temperature 0, 2,048-token
per-call cap, Docker image, runner revision and deterministic graders.  A
hanging upstream grader is failed closed as `grader_timeout`; neither run has
an invocation error.

## Whole-run observation

| Method | Fully passed | Mean partial score | Mean tokens/task | Median latency | p95 latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| PlanExecute-TB | 14 / 89 | 0.61221 | **48,870** | **38.67 s** | 215.47 s |
| MT-Efficient-v2 | **15 / 89** | **0.62232** | 49,788 | 40.14 s | **182.81 s** |

The whole-run partial-score difference is +0.01011 with a 95% paired bootstrap
CI of [-0.00742, +0.03011].  The candidate uses 918 more tokens per task on
average (95% CI [-344, +2,712]).  Pass discordance is one PlanExecute-only pass
and two candidate-only passes (exact McNemar p=1.0).  This does not establish a
quality or pass-rate improvement.

## Strategy-effect audit

The apparent whole-run increase is not attributable to the router.  Separating
the three tasks where the policy changed the organization from the 86
same-action controls gives:

| Subset | Candidate minus PlanExecute partial score | Passes | Mean token difference |
| --- | ---: | ---: | ---: |
| 3 changed-action tasks | **-0.13333** | 0 vs 1 | **+29,488** |
| 86 unchanged-action controls | +0.01512 | 15 vs 13 | -78 |

On the actual treatment subset, routing to Solo loses quality, loses one pass
and costs 63.30% more tokens.  On the 86 tasks where both runs execute
PlanExecute, backend repeat variability creates all of the apparent aggregate
gain, including the two candidate-only passes.  A request seed reduces but does
not eliminate local quantized-backend nondeterminism.

## Decision

`MT-Efficient-v2` fails the confirmation gate and is retained as a transparent
negative result.  It must not be described as the best TeamBench method or as
Agentic RL.  The next candidate replaces static category routing with a
sequential policy over observable runtime state, including failed-command
count, repeated-action evidence, remaining budget and validation status.  It
must be selected on development data and confirmed on a fresh generator seed.

## Provenance

- runner revision: `e24447112e4c2349a197d0d3b00af15f1ea32d5e`
- task-instance SHA-256: `3dff00c1eca440e084faf4a7295632a5beb4cf6e0fae2389e5531bb4c78ea72b`
- PlanExecute results SHA-256: `aa67b2ca17896a9709544378d932ea87a69294b827a849c3c4191c440c145f59`
- MT-Efficient-v2 results SHA-256: `7aee0df818d1532f1222fc206ba16f81b164861f2b040a68e279efec1af601ba`
- frozen policy SHA-256: `0d68800dd76a2c89ae68e12c9e0120b2de70f5a636cf65b93462ab8197a1f7c5`

