# TeamBench Agentic RL v3 tail-budget confirmation

Status: completed. V3 creates one new full pass without losing a pass and
eliminates the observed token-budget overruns, but misses its frozen mean-cost
gate by 0.223 percentage points. It does not advance as a replacement for
PlanExecute and is not benchmark-best.

Raw prompts, messages, requests and workspaces remain private.

## Frozen change

V3 keeps the v2 pessimistic bootstrap-Q controller but raises learned action
reserves from p75 to p90. A conservative request guard uses the remaining task
budget, UTF-8 request bytes and a fixed chat-template margin to cap or block the
next generation. The policy, reserve scan and seed-4 gate were frozen in the
[v3 protocol](../docs/TEAMBENCH_AGENTIC_RL_V3_PROTOCOL.md).

The p90 candidate was selected only because it was the highest scanned reserve
that retained the best leave-one-seed-out pass count. P95 was rejected because
it removed both cross-validated new passes. Seed 3 was diagnostic evidence and
was not reused as independent confirmation.

## Independent seed-4 result

| Method | Passes | Mean partial | Mean tokens | Median latency | p95 latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Same-trajectory PlanExecute | 5 / 30 | 0.64510 | 46,155 | 47.147 s | 88.036 s |
| MT-Agentic-RL-v3 | **6 / 30** | **0.67667** | 55,489 | 58.348 s | 96.687 s |

V3 improves mean partial score by +0.03157 (95% paired bootstrap interval
[0.00000, +0.09171]). It wins on 2 tasks, ties on 28 and loses on none.
`CROSS4_auth_federation` improves from 0.1429 to 1.0 and creates the new full
pass; `D2_data_quality` improves from 0.73 to 0.82. No LLM Verifier is called.

The candidate adds 9,334 tokens/task (+20.223%; 95% paired interval +6,370 to
+12,340) and 11.00 seconds/task (95% paired interval +7.22 to +15.01). It stops
directly on 10 tasks, stops after replanning on 5, keeps 10 recoveries and rolls
5 recoveries back to the exact prefix.

## Budget result

The request guard blocks a request on 8 tasks and caps a completion on 5. No
task exceeds 90,000 provider-reported tokens, the maximum is 74,734, and there
are zero provider overruns or invocation errors. This supports observed budget
compliance for the pinned llama.cpp endpoints. It is not a universal hard-cap
guarantee for arbitrary providers or tokenizers.

## Gate decision

The basic gate fails solely because +20.223% mean token overhead is above the
frozen +20% limit. The threshold is not changed after seeing the result. The
strong gate also fails because the partial-score interval touches zero, despite
the new pass and absence of losses.

V3 is the strongest causal repair evidence in this Agentic RL line so far: a
frozen learned policy produces a new pass against an exact same-trajectory
baseline while respecting the observed per-task budget. It still cannot be
called superior to PlanExecute. The next iteration should learn an explicit
value-of-information gate to remove tie-only replan and rollback paths, using
expanded development data rather than another threshold adjustment.

Exact aggregate values and artifact hashes are in the
[summary JSON](teambench-agentic-rl-confirm-v3-summary.json).
