# Frozen TeamBench A4-TB/A8-TB protocol

## Separation from the synthetic town benchmark

This protocol belongs to `public_bench/`. Its tasks, metrics and release tags
are not merged with MultiTown's synthetic town benchmark. The suffix `-TB`
means an architecture adapted to TeamBench's engineering role contract.

## Data boundary

- Dev: 30 deterministic non-leaderboard tasks selected by SHA-ranked category
  round robin.
- Test: the public TeamBench-90 list minus `GH120_redis-py_3863`, which upstream
  currently declares under re-curation. Formal test size is 89.
- Dev and test source directories are disjoint.
- Test graders, expected values, pass/fail and partial scores are unavailable to
  the Controller and all model roles until the final submission is frozen.

## A4-TB

1. A strong Planner sees the full specification and sends a plan.
2. A weak Executor sees the brief, plan and isolated workspace, then edits and
   runs local checks.
3. An independent strong Verifier sees the full specification and a read-only
   workspace, writes an attestation and may trigger one remediation cycle.

Planner, Executor and Verifier are always activated. This is the fixed
planner-worker-verifier baseline.

## A8-TB

1. A weak Executor attempts the task from the public brief before any other
   role is activated.
2. A deterministic runtime validator observes only workspace change, write
   calls, command exit codes, public category and public difficulty.
3. A safe, sufficiently reliable attempt stops early. Otherwise a strong
   Planner is activated and the weak Executor performs a targeted revision.
4. Low-risk revised attempts may stop after runtime validation. Hard failures
   and high-risk tasks activate an independent strong Verifier.
5. The initial weak candidate and planned candidate are snapshotted. If strong
   review fails, the Controller selects between them using only public runtime
   evidence, breaking ties in favour of the earlier candidate. It never uses
   the TeamBench grader to select a candidate.

The v2 development policy treats Adversarial, Cross-System Integration, Data
Engineering, Distributed Systems, Long-Horizon, Multi-language, Operations and
Security tasks (plus expert tasks) as high risk. When a strong Verifier emits a
valid failing attestation plus concrete feedback, A8 permits one weak guided
remediation and accepts it only when public runtime evidence is no worse than
the preserved candidates. It does not spend a second strong-model review. A4
retains its single remediation and reverification cycle as part of the fixed
baseline.

The runtime validator never reads `grade.sh`, `expected.json`,
`check_solution.py`, post-hoc scores or reference outcomes.

## Models and budgets

- Strong roles: local Qwen3.5-35B-A3B GGUF endpoint, alias `qwen-game`.
- Weak roles: local Qwen3.5-4B GGUF endpoint, alias `qwen-mm-backup`.
- Temperature: 0; maximum 2,048 generated tokens per call. Development runs
  at temperature 0.2 are retained as diagnostics but are not the frozen
  comparison because sampling variance materially changed task outcomes.
- Phase turn caps: Planner 6, Executor 12, Verifier 8; remediation phases 6.
- A phase stops before another model request once its protocol artifact exists:
  Planner message or valid Verifier attestation. This deterministic rule is
  identical for A4-TB and A8-TB and never reads a grader outcome.
- Both methods use the same deterministic rolling context policy for the local
  16,384-token endpoints: preserve the full initial task prompt and newest
  turns, and omit oldest intermediate tool transcripts once message content
  exceeds 20,000 characters. Compaction counts are logged per request; no
  grader, reference answer or model-generated summary is used.
- A4 and A8 receive the same task version, seed, models, tool declarations and
  final deterministic grader.

## Security

Agent commands, setup scripts and graders execute in `general-mas-runner:0.1`
with no network, a read-only root filesystem, dropped Linux capabilities,
process/memory/CPU limits and only the current task workspace mounted. Trusted
upstream graders additionally receive a project-local npm cache with npm forced
offline; that cache is never mounted into an agent role. Grader containers are
identified by cidfile and forcibly reclaimed after a timeout. Paths used by
read/write tools are canonicalized and must remain under role-specific roots.
No model-generated shell command executes directly on the host.

## Primary metrics and preregistered gate

- Pass rate and mean deterministic partial score.
- Paired bootstrap 95% interval over task-level partial-score differences.
- Exact McNemar test over pass/fail outcomes.
- Total tokens, latency, GPU energy, role activation and route mix.

A8-TB passes the primary transfer gate if its paired partial-score 95% lower
bound versus A4-TB is at least -2 percentage points and its mean tokens per task
are at least 30% lower. All results are reported even if the gate fails.

The frozen 30-task development run passed the token target but failed the
quality non-inferiority target. The policy is nevertheless evaluated once on
the untouched 89-task test split as a held-out negative/mixed result; test
outcomes are never used to revise this policy.
