# TeamBench formal test record v1.2

## Scope boundary

This record belongs only to the public, general-purpose multi-agent benchmark
workstream in `public_bench/`. It does not contain MultiTown's synthetic town
tasks, metrics or claims; the two evaluation tracks remain separate.

For this public branch, machine-local paths in the replacement manifest were
rewritten as repository-relative logical paths. Metric values and the SHA-256
digests of the excluded raw result files were not changed.

## Frozen protocol

- Benchmark: 89 currently evaluable tasks from the public TeamBench test split.
- Split SHA-256: `376c01ba6c78819bc22b53b9ba6561b5cae5e4db9133b7f307e834755b964809`.
- A4-TB: fixed strong Planner, weak Executor, strong Verifier, and at most one
  remediation/reverification loop.
- A8-TB: weak-first execution, public runtime validation, selective strong
  Planner/Verifier activation, and frozen rollback/remediation logic.
- Strong model: local `qwen-game`; weak model: local `qwen-mm-backup`.
- Temperature: `0`; maximum output per model request: `2048` tokens.
- Sandbox image ID:
  `sha256:d218bef3a99b863f630a66bca9a8d8b7bd0e6218078bcf856bf374a14ad06397`.
- Bootstrap: 10,000 paired resamples, seed `20260810`.
- Preregistered gate: paired partial-score CI lower bound at least `-0.02` and
  mean token reduction at least `30%`.

The policy was frozen after the 30-task development split. No test-time
controller or prompt tuning was performed.

## Formal result

| Metric | A4-TB | A8-TB |
|---|---:|---:|
| Fully passed | 14 / 89 | 11 / 89 |
| Pass rate | 15.73% | 12.36% |
| Mean partial score | 0.63375 | 0.58251 |
| Mean tokens/task | 108,381 | 68,218 |
| Median tokens/task | 114,438 | 73,465 |
| Mean latency/task | 93.37 s | 85.21 s |
| Median latency/task | 72.20 s | 58.15 s |
| P95 latency/task | 134.98 s | 165.11 s |
| Monitored energy | 101.04 Wh | 86.41 Wh |
| Monitored wall duration | 8,309.55 s | 7,764.36 s |

Paired A8−A4 partial-score difference was **−0.05124**, with paired bootstrap
95% CI **[−0.08951, −0.01678]**. A8 was better on 12 tasks, tied on 57, and
worse on 20. Exact McNemar comparison of full-pass outcomes produced
`p=0.5078125` (6 A4-only passes and 3 A8-only passes).

A8 reduced mean tokens by **37.06%**, monitored energy by **14.48%**, and
monitored wall duration by approximately **6.56%**. Its median latency was
lower, but its P95 latency was higher because failed weak-first attempts and
strong fallback created a long tail.

The cost gate passed; the quality non-inferiority gate failed. The combined
preregistered gate therefore **failed**. The evidence does not support replacing
A4-TB with the current A8-TB controller.

## Infrastructure correction

The original A8 run completed 89 tasks but one row, `INC6_deadlock`, was invalid
because Python's `TimeoutExpired` supplied byte output that the sandbox wrapper
concatenated with text. That row had no meaningful model usage or score. The
wrapper now normalizes byte/text timeout payloads and has a regression test.

The affected task alone was rerun with the same split, models, controller,
container, and temperature. The retry completed with zero request errors,
partial score `0.75`, and `99,295` tokens. The original run, retry, and merged
result remain separate; `teambench-test-v1.2-replacement.json` records their
SHA-256 values. The formal 89-row A8 result has no duplicate IDs, request
errors, or error rows.

## Diagnostic findings and next experiment

- `weak_early_stop` used only about 24.2k tokens/task on average, but its mean
  A8−A4 quality difference was still −0.051. `PIPE3_stream_processing` was the
  clearest false-positive early stop (1.00 to 0.30).
- Failed strong review followed by fallback also regressed: mean differences
  were about −0.071 for fallback-to-initial and −0.053 for
  fallback-to-planned. Candidate selection needs stronger public evidence.
- Largest category regressions appeared in Code Review, Adversarial, Pipeline,
  Multi-language, and Specification tasks.
- The next controller iteration should be developed without reusing this test
  set for tuning. Use development tasks to add semantic/public acceptance
  validators, deterministic candidate ranking, and an escalation budget; then
  preregister and evaluate on a new holdout or a different public benchmark.

## Files

- `teambench-test-v1.2-summary.json`: full compact statistics, provenance, input
  hashes, route counts, and monitoring summaries.
- `teambench-test-v1.2-paired.csv`: all 89 paired task scores, token counts,
  latencies, and routes.
- `teambench-test-v1.2-replacement.json`: infrastructure retry audit trail.
- Raw report, A4/A8 request logs, workspaces and model outputs are excluded from
  the public branch; their frozen SHA-256 references remain in the compact
  records.
