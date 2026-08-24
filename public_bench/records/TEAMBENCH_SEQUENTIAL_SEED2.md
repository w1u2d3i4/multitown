# MT-Sequential-v1 seed-2 paired confirmation

Status: the matched PlanExecute/candidate pair is complete on all 89 public
generator-seed-2 tasks. It is a negative controller result. The protocol's
same-seed Solo rank anchor remains required before closing the full three-way
confirmation; no benchmark-best claim is made.

Raw workspaces, prompts, request logs and system telemetry remain private.

## Method and controls

Both runs use the same 89 task IDs, generated seed 2, per-request sampling seed
20260824, strong and weak local model aliases, temperature 0, 2,048-token
per-call cap, Docker image, task order, runner revision and deterministic
graders. There are 89 unique results in each run, zero invocation errors and
zero grader-timeout rows.

`MT-Sequential-v1` preserves the PlanExecute prefix, then may `review` with a
strong read-only Verifier and `delegate` one weak-Executor remediation. It is a
frozen deterministic controller, **not Agentic RL**.

## Whole-run observation

| Method | Fully passed | Mean partial | Mean tokens | Median latency | p95 latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| PlanExecute-TB | **17 / 89** | 0.61670 | **48,117** | **36.73 s** | **98.29 s** |
| MT-Sequential-v1 | 16 / 89 | **0.62805** | 73,288 | 51.23 s | 138.23 s |

The candidate-minus-baseline partial-score difference is +0.01135 with a 95%
paired-bootstrap CI of [-0.01247, +0.03955]. It is better on 7 tasks, tied on
75 and worse on 7. Pass discordance is two PlanExecute-only passes and one
candidate-only pass (exact McNemar p=1.0). Quality or pass-rate superiority is
not established.

The candidate adds 25,171 tokens/task (95% CI [+21,778, +28,357]), a 52.31%
increase. It adds 20.81 seconds/task (95% CI [+10.98, +33.99]), a 33.02%
increase. Its p95 token count is 102,081 versus 65,953 for PlanExecute.

## Action-effect audit

The controller reviews 80/89 tasks. Seventy-one reviews roll back, eight end in
a Verifier-pass attestation without changing the candidate workspace, and only
one review delegates a remediation. Nine tasks take the runtime-clear stop.

| Subset | Tasks | Candidate vs baseline passes | Mean partial delta | Mean token delta |
| --- | ---: | ---: | ---: | ---: |
| Accepted remediation | 1 | 0 vs 0 | **0.00000** | **+48,841** |
| Review, no remediation | 79 | 11 vs 12 | +0.01278 | +28,800 |
| Runtime-clear stop | 9 | 5 vs 5 | 0.00000 | -9,315 |

The only action that changes a final workspace is remediation of
`INC10_rollback_plan`; its deterministic score stays 0.40 while tokens increase
from 48,765 to 97,606. All 14 non-zero task-score differences occur on
non-remediated runs and therefore cannot be attributed to a downstream
workspace treatment. They are consistent with residual local-backend repeat
variation despite matched request seeds.

## Decision

`MT-Sequential-v1` fails the causal confirmation gate: the only actual
treatment has zero quality benefit, the pass count decreases, the quality
interval includes zero, and review adds substantial token and latency cost.
It must not be described as the best TeamBench method or as Agentic RL.

Because the whole-run partial point estimate is above PlanExecute, the frozen
protocol still requires the same-seed Solo anchor before the three-way rank
record is closed. Independently of that pending rank anchor, v1 is retired as a
controller. The next version must intervene before repeated long-tail commands,
use a high-precision failure trigger, and prefer Planner feedback over broad
post-hoc verification.

## Provenance

- runner revision: `36e7ea30da3f3702c18376f2ccb6cfa08dea1361` (clean)
- project-source revision: `2f060a33501b19fbc8d26f8ccdad7580e3b04635` (clean)
- controller SHA-256: `cdfb8fe55829e7a4455426195f588fc98d46c9e87aa36a12eb51df79f1d79d40`
- task-instance SHA-256: `07d5e4ff0db139d1e10bb7ac15ee78954822e186d6d0c71f6780925141bb23a3`
- Docker image ID: `sha256:d218bef3a99b863f630a66bca9a8d8b7bd0e6218078bcf856bf374a14ad06397`
- PlanExecute config/results/request/monitor SHA-256: `50148fbb...`,
  `fdf7d942...`, `a6a92069...`, `87c9b7e6...`
- candidate config/results/request/monitor SHA-256: `3311b6e7...`,
  `1b7c760d...`, `adcaddbf...`, `c70b4500...`

Exact complete hashes and machine-readable statistics are in
[`teambench-sequential-seed2-summary.json`](teambench-sequential-seed2-summary.json).
