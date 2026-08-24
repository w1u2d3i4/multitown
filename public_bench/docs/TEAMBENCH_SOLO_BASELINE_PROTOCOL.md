# TeamBench Solo-TB post-hoc baseline protocol

Protocol frozen on 2026-08-24 before the first Solo-TB test invocation.

## Question

Does MultiTown's selective A8-TB organization offer a useful quality/resource
trade-off relative to the canonical single-agent baseline, rather than only
relative to a fixed Planner-Executor-Verifier team?

This is a post-hoc contextual baseline added after the frozen A4-TB/A8-TB v1.2
comparison. Its results must not be used to modify A8-TB and then be reported
against the same 89 test tasks.

## Solo-TB definition

Solo-TB follows TeamBench's canonical `oracle` condition:

1. one strong agent receives the complete specification and public brief;
2. the same agent can read, edit and execute inside the isolated task workspace;
3. the agent writes its own final attestation;
4. the deterministic hidden task grader runs only after the submission is frozen.

The agent cannot read the grader, expected values, reference outcome or any
previous A4-TB/A8-TB result.

## Frozen evaluation controls

- Tasks: the same 89 evaluable tasks and order from the frozen TeamBench-90
  test list used for A4-TB/A8-TB v1.2.
- Model: local Qwen3.5-35B-A3B GGUF endpoint, alias `qwen-game`.
- Temperature: 0.
- Maximum generated tokens per request: 2,048.
- Maximum agent turns per task: 20, matching TeamBench's canonical Solo
  default.
- Sandbox, setup, deterministic graders, context compaction and system monitor:
  unchanged from the v1.2 protocol.
- No A8-TB controller setting may be changed after observing Solo-TB results.

## Required reporting

Report all three systems on the same task set:

- Solo-TB: canonical single strong agent with full access;
- A4-TB: fixed strong Planner, weak Executor and strong Verifier;
- A8-TB: weak-first selective organization.

For every system report pass rate, mean partial score, tokens, latency, energy,
model/role activations and invocation failures. Report paired task-level quality
differences with bootstrap intervals and exact McNemar tests. Compare Pareto
trade-offs; do not turn unlike model tokens into a monetary-cost claim.

External TeamBench leaderboard rows may be shown only as separately labelled
published context because they use different models and harness conditions.
