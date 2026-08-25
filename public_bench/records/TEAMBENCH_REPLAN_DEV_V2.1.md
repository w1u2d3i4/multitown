# MT-Replan-v2.1 development result

Status: completed on the frozen 30-task TeamBench development split. The
candidate has one positive causal partial-score repair but fails the advancement
gate. It does not advance to seed-3 confirmation and is not Agentic RL.

Raw prompts, workspaces, requests and system telemetry remain private.

## Matched comparison

Both arms use the same task order, source revision, runner revision, controller
configuration, strong/weak model aliases, sampling seed 20260824, temperature,
token cap, Docker image and deterministic graders. Each has 30 unique results,
zero invocation errors and zero grader timeouts.

| Method | Fully passed | Mean partial | Mean tokens | Median latency | p95 latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| PlanExecute-TB | **6 / 30** | 0.61187 | **45,334** | **53.69 s** | **107.97 s** |
| MT-Replan-v2.1 | 4 / 30 | **0.63343** | 49,451 | 54.94 s | 138.92 s |

Candidate minus baseline partial score is +0.02156 with a 95% paired-bootstrap
CI of [-0.05466, +0.10145]. It is better on four tasks, tied on 23 and worse on
three. Pass discordance is two PlanExecute-only and zero candidate-only passes
(exact McNemar p=0.5). The candidate adds 4,117 tokens/task (95% CI
[+812, +7,879]), or 9.08%, and 6.79 seconds/task (95% CI [+1.38, +13.25]), or
12.66%.

## Same-trajectory action audit

Every candidate task saves and grades its exact pre-replan PlanExecute
workspace-and-reports prefix after the controller stops. The hidden shadow
score is evaluation-only. All 24 untouched and four rollback candidates match
their shadow hashes exactly before grading.

Six tasks escalate. Four recoveries are rejected and rolled back; two changed
workspaces are retained:

| Task | Prefix partial | Selected partial | Delta | Extra tokens |
| --- | ---: | ---: | ---: | ---: |
| `O7_capacity_planning` | 0.9000 | 0.9000 | 0.0000 | +28,879 |
| `DIST4_clock_skew` | 0.5833 | 0.9167 | **+0.3334** | +18,237 |

Across all 30 tasks, selected minus shadow prefix is +0.01111 mean partial
(95% CI [0.00000, +0.03334]) and +4,436 tokens/task (95% CI
[+1,443, +7,910]). It creates no additional full pass. The six escalation
attempts add 133,083 tokens in total; only one changes deterministic quality.

This is stronger causal evidence than a separate-run point estimate: recovery
can repair a hard trajectory, but v2.1 activates too often relative to its
success rate and does not improve the primary full-pass outcome.

## Decision

V2.1 fails the benchmark-best gate: its full-pass count decreases, independent
quality superiority is not established, and both tokens and latency increase.
It does not advance to seed 3. The positive DIST4 transition and five zero-gain
escalations are retained as development trajectories for a budgeted learned
policy. No learned policy exists yet, so neither v2.1 nor this trajectory set is
called Agentic RL.

The next candidate must predict recovery value before spending the extra
Planner/Executor calls, retain the same shadow counterfactual, and beat this
PlanExecute anchor under the same development budget before any public-seed
confirmation.

## Provenance

- runner revision: `158e3a174f47cab6ef71c855db4f0b95cff1d62b` (clean)
- controller SHA-256: `83d08c2b4991e6118bdb66653fb58e28f5f9bd342d9438aff6a42a2dbec1d016`
- source revision: `2f060a33501b19fbc8d26f8ccdad7580e3b04635` (clean)
- Docker image ID: `sha256:d218bef3a99b863f630a66bca9a8d8b7bd0e6218078bcf856bf374a14ad06397`
- exact artifact hashes are in
  [`teambench-replan-dev-v2.1-summary.json`](teambench-replan-dev-v2.1-summary.json)
