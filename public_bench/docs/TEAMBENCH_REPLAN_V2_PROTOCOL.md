# MT-Replan-v2 TeamBench protocol

Status: implementation and thresholds frozen before the first `MTReplan`
development run. Public generator seed 2 is a discovery set because it exposed
the failure pattern of MT-Sequential-v1; it is not a confirmation set for this
version. The next untouched confirmation seed is 3.

## Why this version exists

MT-Sequential-v1 added a strong post-execution Verifier whenever the weak
Executor showed broad runtime risk. Its development run was competitive in
quality but expensive, and its seed-2 run exposed three structural problems:

1. ordinary failing tests triggered many false-positive reviews;
2. review happened after the Executor had already paid for repeated or timed-out
   commands;
3. a read-only Verifier often spent several turns inspecting the same workspace
   without producing actionable feedback.

This agrees with TeamBench's report that verifier approval is poorly calibrated
and that removing the verifier can improve mean partial score. Version 2
therefore preserves the strong `Planner -> Executor` baseline and adds a short
execution-feedback loop instead of an always-on or post-hoc review loop.

This is a deterministic sequential controller, **not Agentic RL**. It creates
the state/action traces and counterfactual baseline needed for a later learned
policy, but no policy parameters are trained here.

## Frozen organization

Every task begins with the matched `PlanExecute-TB` roles and model tiers:

- a strong Planner receives the full specification and sends a plan;
- a weak Executor receives only the public brief plus team messages and edits
  the workspace;
- no Verifier is activated.

If the first Planner fails to send a plan, the controller gives it one
format-constrained retry before execution. Executor shell commands use a
30-second sandbox timeout because the sandbox has no network and longer
dependency-download waits cannot succeed.

## Frozen state and policy

The controller may observe only plan delivery, workspace hashes, command exit
codes, timeout text, normalized command repetition counts, turns and observed
token usage. It cannot read the hidden grader, expected output, task score or a
task-specific allowlist.

The declared action space is `stop`, `delegate`, `escalate`, `review` and
`human/abstain`. Version 2 deterministically uses this path:

1. Interrupt before the next model request after one command timeout or two
   identical failed command invocations. Successful re-runs after an edit do
   not trigger intervention.
2. After execution, `escalate` to the same strong Planner only if the plan is
   missing, a timeout/repetition occurred, the workspace is unchanged, or at
   least two commands failed with no successful command. Otherwise `stop` with
   the PlanExecute candidate.
3. The escalated Planner gets read-only workspace access and sends one targeted
   recovery plan. If the 90,000-token observed budget still permits, `delegate`
   at most six recovery turns to the weak Executor.
4. Keep recovery only when it changes the workspace and its observable runtime
   reliability is no worse than the prefix. Otherwise restore the byte-for-byte
   PlanExecute snapshot, then `stop`.

The machine-readable controller is
[`../configs/mt-replan-teambench-v2.json`](../configs/mt-replan-teambench-v2.json).

## Evaluation gates

1. Smoke-test plan-missing, timeout, repeat and clean-stop paths.
2. Run all 30 frozen development tasks. Compare to matched PlanExecute first,
   then the five fixed baselines. Report routed and untouched tasks separately.
3. Advance only if the point estimate is non-dominated in pass/partial quality
   versus tokens and latency. Freeze code, controller and hashes.
4. Confirm on all 89 public tasks with generator seed 3 and identical model,
   sampling seed, image, task order and limits for candidate and baselines.
5. A benchmark-best claim requires a quality win over every completed matched
   baseline. A Pareto claim additionally requires non-domination under the
   repository's quality/token rule and paired uncertainty reporting.

Raw prompts, workspaces and telemetry remain local. Public records contain only
aggregate results, paired statistics, provenance hashes and the frozen policy.
