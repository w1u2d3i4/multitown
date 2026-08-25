# TeamBench Agentic RL v1 development result

Status: completed. This is a genuinely trained offline sequential controller,
but it converges to the PlanExecute fallback and does not pass the benchmark
advancement gate. Seed 3 remains untouched.

Raw prompts, messages, requests and workspaces remain private.

## What was trained

The language models are frozen. The learned component is the orchestration
policy over three decisions:

1. after execution: `stop`, `escalate` or `human/abstain`;
2. after replanning: `stop`, `delegate` or `human/abstain`;
3. after recovery: `stop`, `review` or `human/abstain`.

The global action space therefore contains `stop`, `delegate`, `escalate`,
`review` and `human/abstain`. Finite-horizon fitted Q iteration uses only
pre-action runtime state. Task ID, expected output, hidden tests and grader
outcomes are excluded from policy features. Grader scores are terminal training
rewards only. The exact reward and model-selection rule were frozen in the
[protocol](../docs/TEAMBENCH_AGENTIC_RL_V1_PROTOCOL.md) before collection.

## Counterfactual collection

Sampling seed 20260825 completed 30 development tasks with zero invocation
errors. Twenty-eight tasks produced complete prefix, recovery and review
counterfactuals. Two tasks did not deliver a usable recovery plan and were
excluded rather than assigned an invented reward.

On the 28 complete episodes:

| Candidate | Passes | Mean partial | Mean tokens | Mean latency |
| --- | ---: | ---: | ---: | ---: |
| Exact PlanExecute prefix | 5 / 28 | 0.64284 | 47,034 | 61.17 s |
| Forced recovery | 6 / 28 | 0.66536 | 69,781 | 86.97 s |
| Review-selected candidate | 5 / 28 | 0.64284 | 86,365 | 101.02 s |

Recovery minus prefix is +0.02251 mean partial score with a 95% paired
bootstrap interval of [0.00000, +0.05399]. Three tasks improve and 25 tie; one
recovery creates a new full pass. This gain costs +22,747 tokens/task and
+25.80 seconds/task.

The independent Verifier rejects all three quality-improving recoveries. Its
selected candidate is therefore exactly equal to the prefix in quality while
adding 39,331 tokens/task and 39.86 seconds/task. This is a concrete negative
finding: a read-only reviewer is not a reliable value estimator for recovery
selection in this harness.

## Training result

Leave-one-task-out selection chooses ridge alpha 100 and a conservative Q-value
margin of 0.05. It preserves the five prefix passes and 0.64284 partial score,
but routes 27 tasks directly to `stop` and one to `escalate -> stop`. The latter
adds cost without changing quality. Refitting on all 28 episodes chooses
`stop` for every observed training state.

This is a trained Agentic RL policy under the declared finite-horizon offline
RL definition. It is not evidence of performance superiority.

## Fresh online development evaluation

Sampling seed 20260826 evaluates the frozen policy on all 30 development tasks.
Every policy-visible state is generated again by the frozen models. The exact
pre-policy PlanExecute workspace and reports are saved and graded after routing,
so the comparison does not rely on a separate stochastic model run.

| Method | Passes | Mean partial | Mean tokens | Median latency | p95 latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Same-trajectory PlanExecute | 6 / 30 | 0.62299 | 45,682 | 51.037 s | 99.673 s |
| MT-Agentic-RL-v1 | 6 / 30 | 0.62299 | 45,682 | 51.040 s | 99.675 s |

The learned policy chooses `stop` on all 30 tasks. Every selected workspace and
reports hash matches its PlanExecute prefix; scores and token counts also match
exactly. Partial-score and token differences are exactly zero. Policy inference
adds 1.454 ms/task (95% paired bootstrap interval 1.166 to 1.794 ms). There are
zero invocation errors.

## Decision

V1 demonstrates a real training and deployment path and successfully learns to
avoid the expensive false-positive interventions of the hand-written v2.1
controller. It does not learn a generalizable profitable intervention, so it
does not improve quality or cost over PlanExecute. It fails the advancement
gate, is not a benchmark winner and does not advance to seed 3.

The useful next target is recovery-value data diversity, not a more aggressive
policy on the same 28 episodes. In particular, future collection must preserve
the three positive recoveries and add independently varied recovery attempts or
new development tasks before retraining.

## Provenance

- runner revision: `add29cac26922fbf2b92e7b9168c892f906c435a`;
- source revision: `2f060a33501b19fbc8d26f8ccdad7580e3b04635`;
- fitted-Q policy parameter SHA-256:
  `18ebd78ae9dca90522e46bc3280c399e253c8955e70e39550e49aaab679b1942`;
- controller SHA-256:
  `daf4a8c6e4563d483fa1530c556d9b611848be20ba6e3368bab449d8319fe157`;
- exact aggregate values and artifact hashes are in
  [the summary JSON](teambench-agentic-rl-dev-v1-summary.json).
