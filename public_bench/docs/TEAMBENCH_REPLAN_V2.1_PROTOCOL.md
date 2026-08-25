# MT-Replan-v2.1 TeamBench protocol

Status: the policy was frozen after offline trigger preflight and before its
first model invocation. A one-task development smoke run then confirmed the
execution and shadow-grading path but showed that an upstream grader may mutate
the mounted candidate while grading. Before the formal development run, the
evaluator was therefore amended to record selected workspace and reports hashes
before grading. No policy state, trigger, action, prompt, role, budget or model
setting changed. Public generator seed 2 is discovery evidence; generator seed
3 is reserved for confirmation if development gates pass.

## Change from the unrun v2 design

The v2 design would escalate 44/89 completed seed-2 discovery traces. Its broad
`workspace_unchanged` signal included report-only tasks, and two identical
failed commands included three trajectories that eventually passed.

Version 2.1 makes two preregistered changes without using a hidden score as a
policy input or a task-specific allowlist:

1. unchanged workspace triggers only when the Executor has no successful shell
   command;
2. live repetition interruption requires three identical failed commands.

The same offline audit projects 16/89 escalations, of which 15 occur on failed
trajectories and one on a passing trajectory. This is an activation diagnostic,
not a v2.1 quality result: no v2 or v2.1 model trajectory existed when the
policy was frozen.

## Frozen organization and actions

Every task begins with the matched strong-Planner/weak-Executor
`PlanExecute-TB` organization. A missing Planner message receives one mandatory
format retry. Executor commands run in the network-disabled sandbox with a
30-second timeout.

The declared action space is `stop`, `delegate`, `escalate`, `review` and
`human/abstain`. The deterministic path is:

1. interrupt before the next model request after one command timeout or three
   identical failed command invocations;
2. after execution, `escalate` to the same strong Planner only for a missing
   plan, timeout/repetition, unchanged workspace with zero successful commands,
   or at least two failed commands with zero successful commands;
3. give the Planner read-only workspace access plus at most three bounded
   command-failure summaries; require one new recovery-plan message;
4. if the 90,000-token observed budget permits, `delegate` at most six targeted
   recovery turns to the weak Executor;
5. keep recovery only if it changes the workspace and its public runtime
   reliability is no worse than the prefix; otherwise restore the byte-for-byte
   PlanExecute workspace-and-reports snapshot, then `stop`.

After the controller has stopped, the deterministic grader scores both the
selected candidate and the saved PlanExecute prefix. The prefix score is a
paired, evaluation-only counterfactual: it is never exposed to the controller,
Planner or Executor. This separates recovery's actual task-level effect from
local-model repeat variation in a separately generated baseline. Extra shadow
grading time is recorded separately and excluded from policy end-to-end
latency. A separately rerun matched PlanExecute arm remains required as a
reproducibility anchor.

Both selected and post-grader hashes are retained. Equality checks against the
shadow prefix use the selected, pre-grader hashes because the grader itself may
create caches or other files in its mounted workspace.

No Verifier is activated. The policy never sees a hidden grader result, expected
output, task score or task-specific route. It is a deterministic controller,
**not Agentic RL**.

The machine-readable controller is
[`../configs/mt-replan-teambench-v2.1.json`](../configs/mt-replan-teambench-v2.1.json).

## Evaluation gates

1. Smoke-test missing-plan, timeout, repeated-failure, report-only clean-stop and
   successful-rerun paths.
2. Run all 30 frozen development tasks and compare against matched
   PlanExecute first, then the five fixed development baselines. Report
   escalated and untouched subsets separately. For every escalation, report
   the paired selected-minus-prefix shadow effect.
3. Advance only if the candidate is non-dominated in pass/partial quality,
   tokens and latency. Freeze the exact runner and controller hashes.
4. Confirm on all 89 public tasks with generator seed 3 using matched Solo and
   PlanExecute anchors, task order, models, sampling seed, sandbox and limits.
5. A benchmark-best claim requires a quality win over every completed matched
   baseline plus paired uncertainty and action-effect evidence. A point estimate
   alone is insufficient.

Raw prompts, workspaces and telemetry remain local. Only aggregate results,
paired statistics, provenance hashes and the frozen policy may be published.
