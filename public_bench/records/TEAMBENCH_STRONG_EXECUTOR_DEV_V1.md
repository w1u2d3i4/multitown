# StrongPlanExecute-TB development result

Status: completed 30-task negative result. The candidate is retired as a
uniform policy and was not opened on the fresh generator-seed-5 confirmation.

Raw workspaces, prompts, request payloads and system telemetry remain private.

## Controlled comparison

`PlanExecute-TB` uses the Qwen3.5-35B-A3B Planner and Qwen3.5-4B Executor.
`StrongPlanExecute-TB` changes only the Executor to the same 35B-A3B model.
Both preserve TeamBench's OS-enforced Planner/Executor role boundary, use the
same 30 tasks and order, sampling seed `20260840`, temperature 0, 2,048-token
per-call cap and Docker image, and omit Verifier/remediation stages. This is a
fixed workflow, not Agentic RL.

| Method | Passes | Mean partial | Mean tokens | Mean latency | p95 latency | Energy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PlanExecute-TB | 6/30 | **0.68732** | **46,207** | **53.58 s** | **89.93 s** | **21.44 Wh** |
| StrongPlanExecute-TB | 6/30 | 0.67688 | 48,117 | 64.66 s | 127.67 s | 26.14 Wh |

Candidate minus baseline mean partial score is **-0.01044**, with 95% paired
bootstrap CI **[-0.06011, +0.03556]**. The candidate is better on 3 tasks,
tied on 23 and worse on 4. Pass discordance is one baseline-only and one
candidate-only pass (exact McNemar p=1.0).

The candidate adds 1,910 tokens/task (95% CI [-1,641, +5,731]), a 4.13%
increase, and 11.08 seconds/task (95% CI [-1.91, +28.63]), a 20.67% increase.
Monitored energy increases by 21.97%. Both runs contain 30 unique results, zero
invocation errors and no stale sandbox containers.

## Interpretation and decision

Uniformly scaling the Executor does not improve this benchmark under the
frozen protocol. It repairs `O7_capacity_planning` from 0.90 to a full pass and
improves `EA1_security_scan` and `DIST3_idempotency`, but loses the full pass on
`SQL1_query_repair` and regresses three other tasks. The development gate
requires no pass loss in aggregate and a strictly higher mean partial score;
the candidate fails the second requirement and is retired.

The task-level pattern motivates a separately frozen, capacity-aware routing
hypothesis. That follow-up must be confirmed on an unseen generator seed and
must not reuse this same development run as confirmation.

## Provenance

- runner revision used for both invocations:
  `686357b6e7af95d05ebdf771e6da07684ffcd910` (clean)
- TeamBench project-source revision:
  `2f060a33501b19fbc8d26f8ccdad7580e3b04635` (clean)
- task split SHA-256:
  `376c01ba6c78819bc22b53b9ba6561b5cae5e4db9133b7f307e834755b964809`
- Docker image ID:
  `sha256:d218bef3a99b863f630a66bca9a8d8b7bd0e6218078bcf856bf374a14ad06397`
- baseline config/results/request/monitor SHA-256:
  `d7520daf111e5ec31a56bc2e23f4f6791c242086088a24dd5e9576d5682a1bd4`,
  `28b34df8b84872c8691854b95c28fdc701d405b5516b6e827138bce1094ae2bb`,
  `5b8c7930967e9b120a5bb59dd4631ea2d2d153e71f539f284c3be31d7aedf875`,
  `a2f844575082986f2747dc758aacde8738cbabb0a1d67e58df3e610938c45ad4`
- candidate config/results/request/monitor SHA-256:
  `47ec45fc7fcd71a6abc28ac615bfb5fe071730d35af1b0675d28b9c5baed4fa5`,
  `9dda01dc5c63f06a3147e7eb18f9661208fa40bfa2c0dfa4fcb4ecbee2dff8cc`,
  `8c31c09b31f9a825462be932139d8371695528ad2e389b0263a9705be092726a`,
  `378e2270f6a3b1d415b4719a764c4a2abfeac72026bafd2c0d0912d68795b162`
- report summary/paired CSV SHA-256:
  `e8d3579830ca79067d9416f43820454fe87aee0d22db0955b447417cb920a129`,
  `5e57210e824926e2727b7cfa895664d387828088878804696d684eb4a42110eb`
