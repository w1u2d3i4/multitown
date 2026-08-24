# MT-Sequential-v1 TeamBench protocol

Status: frozen before the first `MTSequential` development run and before any
public generator-seed-2 result is opened.

## Hypothesis

The fixed PlanExecute organization is a strong quality/token baseline, but its
Executor sometimes exposes observable evidence of being stuck. Conditional
review and targeted remediation should improve those cases without paying for
an always-on Verifier or changing successful PlanExecute cases.

This is a deterministic sequential controller, **not Agentic RL**. A future
learned controller may reuse its environment and trace schema, but it must be
trained from trajectories and compared with this hand-written policy.

## Frozen prefix and decision state

Every task first executes the unchanged `PlanExecute-TB` prefix: a strong
Planner receives the full specification and delegates to a weak Executor that
receives the public brief plus the plan. The first controller decision may use
only:

- workspace-change and runtime-validator status;
- successful, failed, timed-out and repeated command counts;
- turns used and turn-budget exhaustion;
- tokens consumed and remaining controller budget.

It cannot use the hidden grader, expected output, final score, test-set label or
task-specific allowlist.

## Frozen policy

The action space is `stop`, `delegate`, `escalate`, `review` and
`human/abstain`. Version 1 uses only the following deterministic path:

1. `review` if the PlanExecute prefix has at least one failed command, one
   command timeout, two repeated commands, or exhausts its turn budget;
   otherwise `stop`.
2. The strong, read-only Verifier reads the full specification. If it passes,
   `stop`. If it fails, supplies Executor feedback and budget remains,
   `delegate` one six-turn targeted-remediation phase to the weak Executor.
3. Keep a remediation that changes the workspace. If no effective change is
   made, restore the byte-for-byte PlanExecute snapshot. The controller never
   consults the grader when selecting a candidate.
4. Stop when 120,000 observed tokens have already been consumed. Version 1
   declares but does not autonomously choose `escalate` or `human/abstain`.

The frozen machine-readable controller is
[`../configs/mt-sequential-teambench-v1.json`](../configs/mt-sequential-teambench-v1.json).

## Selection evidence and limits

On the 30-task development split, a retrospective composition of independently
run PlanExecute and FixedTeam outcomes suggested that conditional review at
`failed_commands >= 1` could improve mean partial score from 0.61932 to
0.68856 while activating review on 17/30 tasks. That value is an optimistic
diagnostic, not a result: the two source runs were subject to local-backend
variation and did not share the exact intermediate workspace.

The first formal development run therefore measures the actual sequential
trajectory. No threshold is changed after opening its outcome. If it fails, a
new named version and a new protocol are required.

## Evaluation gates

1. Run all 30 frozen development tasks and compare paired outcomes with all
   five completed development baselines. Report the action-changing subset and
   unchanged PlanExecute controls separately.
2. If development quality is competitive, freeze the exact controller and
   runner revision, then run a matched `PlanExecute-TB` and candidate pair on
   untouched public generator seed 2.
3. A benchmark-best claim requires the candidate to exceed every completed
   fixed baseline on mean partial score on seed 2. A Pareto claim additionally
   follows the quality/token dominance rule in the parent protocol. Confidence
   intervals, pass discordance, latency and token-budget curves remain visible
   even when the point estimate wins.

Raw workspaces, prompts and telemetry remain local. Public records contain only
aggregate results, configuration/provenance hashes and the policy definition.
