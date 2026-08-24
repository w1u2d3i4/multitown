# MT-Sequential-v1 development result

Status: completed on the frozen 30-task TeamBench development split. The
candidate advances to fresh-seed confirmation but is not a benchmark winner.
Raw workspaces, prompts, request logs and system telemetry remain private.

## Method

`MT-Sequential-v1` preserves the fixed PlanExecute prefix. Observable runtime
signals can then select `stop` or `review`; a failed review with concrete
feedback can `delegate` one targeted remediation to the weak Executor. A
byte-for-byte PlanExecute snapshot is restored when remediation does not change
the workspace. The declared action space also contains `escalate` and
`human/abstain`, but v1 does not choose them.

This is a frozen deterministic controller, **not Agentic RL**. Its complete
pre-run contract is in
[`../docs/TEAMBENCH_SEQUENTIAL_V1_PROTOCOL.md`](../docs/TEAMBENCH_SEQUENTIAL_V1_PROTOCOL.md).

## Development comparison

| Strategy | Fully passed | Mean partial score | Mean tokens/task |
| --- | ---: | ---: | ---: |
| Solo-TB | 4 / 30 | 0.62232 | 76,253 |
| PlanExecute-TB | **6 / 30** | 0.61932 | **45,629** |
| ExecuteReview-TB | 1 / 30 | 0.51832 | 58,363 |
| FixedTeam-TB | 4 / 30 | 0.65267 | 99,658 |
| MultiTown-TB | 4 / 30 | 0.62311 | 66,509 |
| **MT-Sequential-v1** | **6 / 30** | **0.65433** | 66,370 |

At the point estimate, PlanExecute-TB and MT-Sequential-v1 form the
quality/token Pareto frontier. The candidate is 0.00167 above FixedTeam-TB in
mean partial score while using 33.40% fewer tokens, and it is slightly better
and cheaper than the previous MultiTown-TB controller. These are development
selection observations, not independent confirmation.

Relative to PlanExecute-TB, the paired mean partial-score difference is
+0.03501 (95% paired-bootstrap CI [-0.02699, +0.10224]). Mean tokens increase
by 20,741/task (CI [+13,749, +27,909]) and mean latency increases by 27.31 s
(CI [+12.69, +47.55]). Pass discordance is two candidate-only and two
PlanExecute-only passes. Quality superiority is not established.

The historical development baselines predate per-request seed logging and have
no sampling seed in their configurations. They match task split, model aliases,
temperature, response cap and Docker image, but local-backend repeat variation
remains a material limitation. The fresh seed-2 pair will use matched request
seeding and runner provenance.

## Action audit

| Route outcome | Tasks | Candidate minus PlanExecute partial score | Mean token difference |
| --- | ---: | ---: | ---: |
| Accepted remediation | 2 | +0.22355 | +38,650 |
| Review without remediation | 22 | -0.00212 | +26,515 |
| Runtime-clear stop | 6 | +0.10833 | -6,399 |

The two accepted remediations are `CROSS4_auth_federation` and
`D2_data_quality`; together they add one pass. Of 24 review actions, 18 rolled
back to the PlanExecute snapshot, four ended with a passing Verifier
attestation and only two produced an accepted workspace remediation. The 8.33%
remediation rate explains why v1 adds substantial review cost.

The stop subset's apparent +0.10833 gain cannot be caused by a downstream
controller action because those tasks execute only the PlanExecute prefix. It
is evidence of backend repeat variation. The remediation subset is the most
relevant directional evidence, but it is still compared with a separately run,
unseeded historical baseline and is not a clean causal estimate.

## Decision

The candidate passes the development competitiveness gate because it has the
highest mean partial-score point estimate and remains on the point-estimate
Pareto frontier. Its exact controller and runner revision advance unchanged to
a matched PlanExecute/Solo/candidate comparison on previously unopened public
generator seed 2.

No seed-2 result may be used to tune v1. If it fails confirmation, it remains a
negative result. A v2 redesign should make decisions at tool-turn granularity,
raise structured Verifier-feedback reliability and reduce unnecessary review.
No learned-policy or Agentic-RL claim follows from this run.

## Provenance

- runner revision: `67ad64a1b231c1bba5124393220e34f6f203945d`
- controller SHA-256: `cdfb8fe55829e7a4455426195f588fc98d46c9e87aa36a12eb51df79f1d79d40`
- split SHA-256: `376c01ba6c78819bc22b53b9ba6561b5cae5e4db9133b7f307e834755b964809`
- Docker image ID: `sha256:d218bef3a99b863f630a66bca9a8d8b7bd0e6218078bcf856bf374a14ad06397`
- results SHA-256: `d2e45f641e1865777a10c546cf8886756246c52dd6ec783c5c6dce6dd0f0abb4`
- request-log SHA-256: `05f8248f4c88ccd6c958b1bcb9851d300a632d131566eca4ba5ecac808108a91`
- system-metrics SHA-256: `af4996c9795e7a802e33aa67f2855ebf016e320483c43310a364f529c886b457`
