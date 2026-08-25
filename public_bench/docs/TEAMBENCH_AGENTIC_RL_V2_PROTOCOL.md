# TeamBench Agentic RL v2 protocol

Status: frozen before the first seed-3 `MTAgenticRL` model request.

## Question and scope

V1 learned to stop on every task. V2 asks whether a controller can recover a
small, quality-improving subset without using an LLM reviewer as the value
function. The Planner, Executor and model weights remain frozen. Only the
three-stage orchestration policy is trained.

This is a quality-first experiment with a bounded cost overhead. It is not a
cost-reduction experiment and it is not a claim of global or TeamBench
state-of-the-art performance.

## Training data and leakage boundary

The training set contains 85 complete paired episodes collected on the frozen
30-task development manifest with generator seeds 0, 1 and 2. Five incomplete
episodes are excluded rather than assigned invented outcomes. Each episode
contains an exact PlanExecute prefix and a bounded replan/recovery
counterfactual scored only after the action sequence ends.

Policy features are restricted to public pre-action information:

- public category and difficulty;
- plan-delivery, token-budget and elapsed-time state;
- executor command, failure, repetition and public test-command signals;
- a 16-bin signed hash of the public brief;
- workspace file-count and language-presence features.

Task ID, full specification, expected outputs, hidden tests, hidden grader
results and labels are excluded at inference time. Counterfactual grader scores
are training rewards only.

## Policy

The action masks remain:

- after execution: `stop`, `escalate`, `human/abstain`;
- after replanning: `stop`, `delegate`, `human/abstain`;
- after recovery: `stop`, `review`, `human/abstain`.

V2 fits a bootstrap ensemble of paired finite-horizon advantages and takes an
intervention only when `mean - beta * standard_deviation` exceeds a
non-negative margin. A learned action is also rejected when the remaining
token budget is below the 75th-percentile reserve observed in training.

At the terminal stage, `review` is an evidence-based controller action that
restores the exact saved PlanExecute prefix. It does not call an LLM Verifier.
This definition follows the v1 negative result, where the Verifier rejected all
three quality-improving recoveries.

The frozen reward is:

```text
partial_score + 0.25 * passed
- 0.05 * log1p(total_tokens / reference_median_tokens)
- 0.02 * log1p(latency / reference_median_latency)
- 0.25 * invocation_error
- 0.50 * safety_violation
```

Model selection is leave-one-generator-seed-out. It first rejects candidates
with fewer full passes than the matched prefix baseline, then maximizes passes,
reward and partial score, followed by lower token and latency cost. The search
grid is ridge alpha `[10, 100]`, uncertainty beta `[0.5, 1, 2]` and margin
`[0, 0.01]`. Beta zero is deliberately excluded.

## Frozen candidate and development gate

The candidate policy SHA-256 is
`2598816885d9e6177fc37491abb018b5ef696ab5d5987dae3b82a35108a43b30`.
It selects alpha 100, beta 0.5 and margin 0. The learned token reserves are
26,775 after execution, 23,455 after replanning and zero after recovery.

Across leave-one-seed-out predictions, the candidate records 15 versus 13
prefix passes, 0.66607 versus 0.65140 mean partial score, and 58,401 versus
47,368 mean tokens (+23.3%). This is model-selection evidence, not an online
benchmark result.

The candidate advances to the untouched generator seed 3 because this
development result meets the pre-seed-3 quality-first gate: more passes,
positive partial-score delta, token overhead below 30%, and no recorded
invocation errors.

Seed 3 is the single independent confirmation run. Sampling seed 20260829,
task order, models, prompts, OS isolation, turn budgets and the 90,000-token
cap are frozen. The exact pre-policy PlanExecute workspace is graded after all
actions, providing a same-trajectory baseline without a separate stochastic
model call.

The independent result passes the confirmation gate only if all conditions
hold:

1. candidate full passes are not below its same-trajectory prefix;
2. candidate mean partial score is strictly higher;
3. mean token overhead is at most 30%;
4. there are zero invocation or sandbox errors.

A stronger `validated quality improvement` claim additionally requires at
least one new pass and no lost passes. Paired bootstrap intervals are reported
as uncertainty summaries, not substituted for the frozen gate. Any failure is
retained as a negative result; no policy or threshold is retuned on seed 3.

## Claim boundary

If the gate passes, this supports a trained offline sequential orchestration
policy on this frozen TeamBench development protocol. It does not establish
superiority to all multi-agent systems, a public leaderboard result, language
model weight training, or a generally optimal Agentic RL algorithm.
