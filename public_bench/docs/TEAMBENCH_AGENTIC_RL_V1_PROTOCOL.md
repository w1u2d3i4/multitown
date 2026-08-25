# TeamBench Agentic RL v1 development protocol

Status: frozen before the first `MTAgenticRLExplore` model request.

One-task preflight on `DIST4_clock_skew` found that the inherited result timer
stopped after hidden grading. The evaluator was corrected before the formal
30-task collection so `latency_s` stops before grading. The preflight is not a
training episode and no controller state, prompt, action, candidate or score
was changed by this evaluator-only correction.

## Scope

This protocol trains the orchestration controller, not the Planner, Executor or
Verifier language models. It uses only the frozen 30-task TeamBench development
split. Seed 3 remains untouched unless the learned policy passes the development
gate.

The development sequence is:

1. sampling seed 20260825 collects one full PlanExecute prefix, one bounded
   replan/recovery candidate and one independent review decision per task;
2. deterministic graders score the prefix, recovery and review-selected
   candidate only after all policy actions end;
3. finite-horizon fitted Q iteration trains a three-stage policy;
4. sampling seed 20260826 compares the frozen policy with PlanExecute on the
   same task order, models, prompts, sandbox and request seeds.

Raw prompts, workspaces, messages and requests remain private. A policy JSON,
aggregate record and hashes may be published.

## State, actions and leakage boundary

The policy observes only pre-action fields: public category and difficulty,
workspace-change status, successful/failed/timed-out command counts, repeated
failure count, turn use, runtime reliability, plan-delivery status, consumed
tokens, remaining token budget and elapsed policy time.

Task ID, expected output, hidden grader output, hidden tests and test-set labels
are excluded from features. Counterfactual grades are training rewards only.

The action masks are:

- post execution: `stop`, `escalate`, `human/abstain`;
- post replan: `stop`, `delegate`, `human/abstain`;
- post recovery: `stop`, `review`, `human/abstain`.

The declared global action space is `stop`, `delegate`, `escalate`, `review`
and `human/abstain`.

## Reward and training

The reward is frozen as:

```text
partial_score + 0.25 * passed
- 0.05 * log1p(total_tokens / reference_median_tokens)
- 0.02 * log1p(latency / reference_median_latency)
- 0.25 * invocation_error
- 0.50 * safety_violation
- 0.10 * human_abstain
```

A safety violation is an invocation/sandbox failure or grader timeout. The
TeamBench OS isolation remains enforced for every role.

Fitted Q iteration uses ridge regression. The predeclared alpha grid is
`[0.1, 1, 10, 100]`; the conservative intervention-margin grid is
`[0, 0.01, 0.02, 0.05]`. Selection uses leave-one-task-out predictions and
first requires no full-pass regression against stopping at the PlanExecute
prefix, then maximizes pass count, reward and partial score, followed by lower
tokens and latency. The final weights are refit on all development episodes and
content-hashed before evaluation.

## Advancement gate

The learned controller advances only if the matched development run has no
full-pass regression and either:

- establishes a quality improvement without higher mean tokens; or
- establishes a token reduction while satisfying the existing quality
  non-inferiority rule.

Offline replay is training evidence, not a benchmark result. Failure remains a
negative result and does not consume seed 3.
