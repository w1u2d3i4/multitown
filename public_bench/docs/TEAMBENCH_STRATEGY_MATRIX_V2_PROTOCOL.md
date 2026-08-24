# TeamBench five-strategy matrix v2 protocol

Status: frozen before the first formal v2 invocation.

Protocol date: 2026-08-24.

## Why v2 is necessary

The first `PlanExecute` formal attempt exposed a sandbox lifecycle defect. A
timed-out `docker run` client returned exit code 124 to the agent, but the
unnamed container could continue running its command. Repeated offline
`npm install` attempts therefore leaked containers and contaminated later
latency and system-energy samples.

Read-only audit found the same timeout signature in the historical Solo, A4 and
A8 raw logs. Their deterministic task scores remain historical records, but
their latency and energy are not mixed with post-fix methods in a new five-way
claim. The interrupted 34-row PlanExecute attempt is retained privately for
audit and excluded from all formal statistics.

The infrastructure repair gives each command container a unique scoped name
and unconditionally issues an idempotent `docker rm -f <exact-name>` after
normal completion, timeout, exception or operator interrupt. Unit and live
Docker timeout tests must pass, and no `general-mas-*` command container may be
running before or after a formal method.

## Frozen rerun matrix

All five methods are rerun from task 1 under the repaired harness. No v1 result
row is substituted into v2.

| Public label | Runner method | Organization |
| --- | --- | --- |
| Solo-TB | `Solo` | one strong full-access agent |
| PlanExecute-TB | `PlanExecute` | strong Planner -> weak Executor; no review |
| ExecuteReview-TB | `ExecuteReview` | weak Executor -> strong Verifier; no planning or repair |
| FixedTeam-TB | `A4` | strong Planner -> weak Executor -> strong Verifier, at most one repair/reverification |
| MultiTown-TB | `A8` | weak-first execution with deterministic selective planning/review/remediation |

## Controls

- All 89 unique currently evaluable TeamBench public test tasks, frozen order.
- Strong model: Qwen3.5-35B-A3B (`qwen-game`).
- Weak model: Qwen3.5-VL-4B-Instruct (`qwen-mm-backup`).
- Temperature 0; maximum 2,048 output tokens per model turn.
- Same Docker image ID, task setup, hidden deterministic graders, filesystem
  policy and role-turn caps.
- One formal method runs at a time. Both model services remain resident across
  methods when possible.
- System monitoring interval: 5 seconds.
- Formal output directories end in `-v2-t0`.
- A successful result row is never rerun. An invocation-error row may only be
  retried through the archive-and-resume path and remains auditable.
- No stale command container may exist at method start or end.

## Outputs and claims

The metrics, paired tests, quality/token Pareto analysis and claim boundaries
are unchanged from `TEAMBENCH_STRATEGY_MATRIX_PROTOCOL.md`. Every direct
five-way number must come exclusively from v2 directories. Existing v1.2 A4/A8
records remain published as historical results and are never silently
overwritten.
