# TeamBench strong-executor v1 protocol

Status: completed negative development result; independent confirmation was
not opened.

Protocol date: 2026-08-26.

## Hypothesis

The completed strategy matrix shows that `PlanExecute-TB` is the strongest
quality/token baseline in the current same-harness comparison, while broad
Verifier and remediation stages usually add cost without a causal repair. The
next non-RL candidate therefore changes one factor only: the Executor uses the
same Qwen3.5-35B-A3B tier as the Planner instead of the Qwen3.5-4B tier.

`StrongPlanExecute-TB` keeps the same OS-enforced role boundary. The Planner can
read the full specification but cannot edit; the Executor receives the public
brief and Planner message, can edit and run public checks, but cannot read the
full specification directly; no Verifier or remediation loop is activated.
It is a fixed two-agent workflow, not Agentic RL.

## Development comparison

- Frozen 30-task TeamBench development split, generator seed from the frozen
  split and sampling seed `20260840`.
- Matched `PlanExecute-TB` and `StrongPlanExecute-TB` runs from task 1 under the
  same runner revision, task order, Docker image, temperature 0, 2,048-token
  per-call cap and local model files.
- Strong model: Qwen3.5-35B-A3B (`qwen-game`).
- Weak baseline Executor: Qwen3.5-4B (`qwen-mm-backup`).
- Primary quality metrics: full passes and deterministic mean partial score.
- Resource metrics: tokens, policy latency, p95 latency and monitored energy.
- Reliability metrics: invocation errors, grader timeouts and stale sandbox
  containers.

The candidate advances only if it produces at least as many full passes as the
matched baseline, has strictly higher mean partial score, has zero invocation
errors and no grader-timeout increase. Token and latency costs remain reported
and may prevent a Pareto claim even when quality improves.

## Independent confirmation

If the development gate passes, freeze the exact code and use generator seed 5
for a single 89-task matched confirmation against `PlanExecute-TB`, with a new
sampling seed declared before either run. Generator seeds 0-4 and the original
public matrix are discovery evidence and cannot independently confirm this
candidate. A benchmark-best claim additionally requires the candidate to beat
the strongest completed same-harness quality result rather than only the old
`MultiTown-TB` A8 row.

Raw prompts, workspaces, model messages and telemetry stay private. Only
aggregate results, artifact hashes, protocol code and non-sensitive plots may
be published.

## Frozen decision

Both development runs completed all 30 unique tasks with zero invocation
errors. `StrongPlanExecute-TB` tied the baseline at 6 passes, but reduced mean
partial score from 0.68732 to 0.67688. Candidate-minus-baseline partial score
was -0.01044 (95% paired-bootstrap CI [-0.06011, +0.03556]). The candidate
also used 4.13% more tokens, took 20.67% longer on mean task latency and used
21.97% more monitored energy. It failed the predeclared development gate and
therefore was not evaluated on generator seed 5.

The full result and provenance boundary are recorded in
[`../records/TEAMBENCH_STRONG_EXECUTOR_DEV_V1.md`](../records/TEAMBENCH_STRONG_EXECUTOR_DEV_V1.md).
