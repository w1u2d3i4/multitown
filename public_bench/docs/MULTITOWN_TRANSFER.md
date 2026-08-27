# From MultiTown's cyber town to TeamBench

This document separates three questions that are easy to conflate:

1. which MultiTown runtime ideas are implemented by the TeamBench adapter;
2. which ideas are active in the confirmed `MT-CapacityRoute-v1` result; and
3. which ideas remain research infrastructure or negative results.

It does not introduce a new measured method or change any frozen metric.

## Transfer map

| MultiTown idea | Public adapter mechanism | Confirmed v1 | Research-only status |
| --- | --- | ---: | --- |
| Role-specific information and authority | Full-spec Planner, brief-only Executor, optional independent Verifier, OS-isolated workspace | Yes | — |
| Economical-first capacity allocation | Weak Executor by default; strong Executor for the frozen `Distributed Systems`, `Operations` and `Security` categories | Yes | — |
| Deterministic runtime evidence | Workspace/write/test/timeout/repetition/turn-budget validators | Record-only | Used by sequential and learned controllers |
| Hard budget guard | Request-bound total-token adapter with reserved completion capacity | No | Implemented and tested in replan/RL variants |
| Sequential actions | `stop`, `delegate`, `escalate`, `review`, `human/abstain` | No | Implemented; learned public candidates have not passed all gates |
| Snapshot recovery | Candidate snapshots, bounded remediation and deterministic PlanExecute rollback | No | Implemented in sequential/replan variants |
| Comparable telemetry | Tokens, E2E latency, p95, failures, route/action counts and compatible monitored energy | Yes | — |

## Why v1 does not blindly escalate after a validator warning

MultiTown's synthetic A8 result supports validation-triggered selective
delegation inside its frozen town environment. That causal result does not
automatically transfer to TeamBench.

The controlled 30-task TeamBench development comparison found that replacing
the economical Executor with the strong Executor on every task kept passes at
6/30, reduced mean partial score from 0.68732 to 0.67688, increased mean tokens
by 4.13%, increased mean latency by 20.67%, and increased monitored energy by
21.97%. The strong Executor improved three tasks, tied 23 and worsened four.
Accordingly, a generic “warning means use the stronger model” rule is not
supported by the development evidence.

The actionable pattern was narrower: strong execution was directionally useful
in three public categories and harmful elsewhere. That observation was frozen
as `MT-CapacityRoute-v1` and then tested once on a fresh generator seed. The
confirmed result is 20/89 passes and 0.64180 mean partial score at 48,296 mean
tokens/task, versus PlanExecute's 16/89, 0.61603 and 49,579.

See:

- [`TEAMBENCH_STRONG_EXECUTOR_DEV_V1.md`](../records/TEAMBENCH_STRONG_EXECUTOR_DEV_V1.md)
- [`TEAMBENCH_CAPACITY_ROUTE_V1_PROTOCOL.md`](TEAMBENCH_CAPACITY_ROUTE_V1_PROTOCOL.md)
- [`TEAMBENCH_CAPACITY_ROUTE_TEST_V1.md`](../records/TEAMBENCH_CAPACITY_ROUTE_TEST_V1.md)

## Controls available for the next iteration

The public runner already exposes the remaining cyber-town controls, but a new
headline result requires a separately frozen protocol and a fresh confirmation:

- validator-state routing without hidden-grader fields;
- request-level hard token budgets;
- bounded review or replanning;
- snapshot-bound rollback;
- explicit human/abstain termination;
- paired quality, token, latency and energy gates.

Until such a candidate passes its frozen quality and cost gates, the public
headline remains `MT-CapacityRoute-v1`. Infrastructure completeness is not
reported as benchmark superiority.
