# MultiTown Public Bench

[返回中文说明](../README_ZH.md) | [Back to MultiTown](../README.md)

This standalone subproject evaluates fixed and dynamic multi-agent
organizations on public, general-purpose tasks. It is intentionally isolated
from MultiTown's synthetic town benchmark: task sets, metrics and claims are
never merged across the two tracks.

This is an **evidence branch**, not a leaderboard shortcut. Its core question is
whether selective multi-agent control occupies a better measured trade-off than
both a standard single agent and an always-on multi-agent team when tools,
tasks and accounting are held fixed.

The multi-agent motivation here is **separation of duties**, not “more agents
must be smarter.” In all four role-separated strategies, no one role can
simultaneously read the full specification, modify the workspace and certify
the result. Solo-TB is the
full-access quality anchor but does not provide that governance boundary. The
method question is therefore whether MultiTown can preserve role isolation
while avoiding the cost of activating every specialist on every task.

## Current scope

- Primary benchmark: [TeamBench](https://github.com/ybkim95/TeamBench), using
  deterministic task graders and 89 currently evaluable tasks from its public
  TeamBench-90 test list.
- Secondary data lock: MultiAgentBench/MARBLE, five domains and 500 released
  configurations. No MARBLE headline result is claimed here.

The five-method comparison follows the patterns used by TeamBench and related
agent-scaling work: include a strong single-agent anchor, hold task/tool access
and execution accounting fixed, report paired task-level statistics, and show
the quality/resource Pareto trade-off instead of only the best accuracy cell.
See [`docs/RELATED_WORK_AND_EVIDENCE.md`](docs/RELATED_WORK_AND_EVIDENCE.md).

## What the names mean

`A4` and `A8` are experiment IDs, not model names, version numbers or rankings.
The `-TB` suffix means that an organization has been adapted to TeamBench's
software-engineering role contract. These scores are therefore not the
synthetic MultiTown A4/A8 scores.

In that role contract, the **Planner** writes a task plan, the **Executor** edits
code and runs tools, and the **Verifier** independently checks the result and
may request one repair:

- **Solo-TB — canonical single-agent anchor:** one strong model sees the full
  specification, edits and tests the workspace, and certifies its own result.
- **PlanExecute-TB — plan then act:** call a strong Planner and a weak Executor,
  but no independent Verifier. This mirrors TeamBench's no-verifier ablation.
- **ExecuteReview-TB — act then review:** call a weak Executor and an independent
  strong Verifier, but no Planner or repair loop. This mirrors TeamBench's
  no-planner ablation.
- **FixedTeam-TB (`A4`) — fixed full team:** always call a strong Planner, a
  weak Executor and an independent strong Verifier, with a frozen one-loop
  remediation budget.
- **MultiTown-TB (`A8`) — selective team:** start with the weak Executor, check the candidate
  with a deterministic public runtime validator, and activate the strong
  Planner or Verifier only when that evidence says more work is needed. A failed
  review can roll back to the best valid candidate.

No system can inspect hidden grader outcomes when making a decision. The
Solo-TB protocol is frozen separately in
[`docs/TEAMBENCH_SOLO_BASELINE_PROTOCOL.md`](docs/TEAMBENCH_SOLO_BASELINE_PROTOCOL.md).
The full five-method contract is frozen in
[`docs/TEAMBENCH_STRATEGY_MATRIX_PROTOCOL.md`](docs/TEAMBENCH_STRATEGY_MATRIX_PROTOCOL.md).
An infrastructure audit found that timed-out Docker commands could outlive the
client process, so the clean five-way rerun is governed by the superseding
[`v2 protocol`](docs/TEAMBENCH_STRATEGY_MATRIX_V2_PROTOCOL.md).

These local strategies cover the parts of mainstream manager/worker,
executor/reviewer, SOP pipeline and dynamic-selection designs that can be
compared fairly under TeamBench role isolation. Debate, independent voting and
free-form peer-chat systems are documented as related families, not falsely
reported as reproduced experiments.

## Mainstream-strategy result

| Strategy | Fully passed | Mean partial score | Mean tokens/task |
| --- | ---: | ---: | ---: |
| Solo-TB | 16 / 89 | **0.64180** | 82,869 |
| PlanExecute-TB | **18 / 89** | 0.62434 | **49,166** |
| ExecuteReview-TB | 10 / 89 | 0.54940 | 67,011 |
| FixedTeam-TB | 14 / 89 | 0.63375 | 108,381 |
| MultiTown-TB | 11 / 89 | 0.58251 | 68,218 |

**Solo-TB and PlanExecute-TB form the quality/token Pareto frontier.**
PlanExecute-TB used 40.67% fewer tokens than Solo-TB; its partial-score
difference was -0.01745 (95% CI [-0.05573, +0.01861]), so this is an efficiency
result rather than a quality-superiority result. ExecuteReview-TB was worse than
PlanExecute-TB by -0.07494 partial-score points (95% CI [-0.11236, -0.04090])
while using 36.30% more tokens. FixedTeam-TB used 120.44% more tokens than
PlanExecute-TB without a clear partial-score improvement.

The current MultiTown-TB controller did not beat Solo-TB or PlanExecute-TB. The
publishable contribution is therefore the controlled strategy matrix, Pareto
evidence and negative finding that **more roles—and especially review without a
repair path—can hurt**. See the
[`formal record`](records/TEAMBENCH_STRATEGY_QUALITY_V2.md).

The next frozen static router also failed causal confirmation.  Although its
seed-1 whole-run mean was +0.01011 above PlanExecute-TB, the three tasks where
it actually changed organization were -0.13333 worse, lost one pass and used
63.30% more tokens.  The apparent aggregate gain came from local-backend
variation on 86 unchanged controls.  The
[`seed-1 selector record`](records/TEAMBENCH_SELECTOR_SEED1.md) therefore marks
it as a negative result and motivates runtime sequential control.

## Historical v1.2 A4/A8 result

| Metric | A4-TB — fixed full team | A8-TB — selective team |
| --- | ---: | ---: |
| Fully passed | 14 / 89 | 11 / 89 |
| Pass rate | 15.73% | 12.36% |
| Mean partial score | 0.63375 | 0.58251 |
| Mean tokens/task | 108,381 | 68,218 |
| Median latency/task | 72.20 s | 58.15 s |
| p95 latency/task | 134.98 s | 165.11 s |
| Monitored energy | 101.04 Wh | 86.41 Wh |

A8-TB reduced mean tokens by **37.06%** and monitored energy by **14.48%**.
Its paired partial-score difference was **−0.05124**, with 95% paired bootstrap
CI **[−0.08951, −0.01678]**. It was better on 12 tasks, tied on 57 and worse on
20. The preregistered cost gate passed; the quality non-inferiority gate failed.

This is a mixed/negative transfer result. It does not support replacing A4-TB
with the current A8-TB controller. The complete interpretation is in
[`records/TEAMBENCH_TEST_V1.2.md`](records/TEAMBENCH_TEST_V1.2.md).

These deterministic task scores remain an auditable historical record. Because
the later sandbox audit found that a timed-out command could leave a container
running, v1.2 latency and energy are not mixed with post-fix methods in a new
five-way runtime claim. The current five-way bridge is explicitly quality/token
only. Clean runtime and energy are reported for the three post-fix methods;
completing post-fix A4/A8 reruns remains required for a clean five-way runtime
claim.

## What is published

- `benchmarks/`: frozen task IDs, public metadata and source hashes;
- `configs/`: frozen A4-TB and A8-TB controller configurations;
- `general_mas_bench/`: upstream audit, runner, role isolation, monitoring,
  report and deterministic grader integration;
- `docker/`: network-disabled execution image;
- `records/`: compact formal summary, all 89 paired scores and retry audit;
- `tests/`: unit and regression tests for the released surface.

Not published: third-party checkouts, generated task workspaces, prompts and
model messages, raw request/system logs, model weights, credentials or failed
run directories. TeamBench remains owned and licensed by its authors; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Install

From this directory:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Docker is required to run model-generated commands and deterministic graders.
The runner never executes model-generated shell commands directly on the host.

## Acquire the pinned TeamBench source

```bash
mkdir -p third_party
git clone https://github.com/ybkim95/TeamBench.git third_party/TeamBench
git -C third_party/TeamBench checkout d185aef1916fd86a9ba554d581fd256319a973af

general-mas-audit --project-root "$PWD" --upstream TeamBench
./bin/build-sandbox.sh
```

The upstream revision, license and dataset hashes are pinned in
[`upstream-lock.json`](upstream-lock.json). `third_party/` stays ignored and is
never vendored into this repository.

## Run a development comparison

Start OpenAI-compatible strong and weak model endpoints, then run:

```bash
general-mas-run-teambench --method Solo --split dev \
  --project-root "$PWD" --temperature 0 \
  --output-dir artifacts/teambench-dev-solo

general-mas-run-teambench --method PlanExecute --split dev \
  --project-root "$PWD" --temperature 0 \
  --output-dir artifacts/teambench-dev-plan-execute

general-mas-run-teambench --method ExecuteReview --split dev \
  --project-root "$PWD" --temperature 0 \
  --output-dir artifacts/teambench-dev-execute-review

general-mas-run-teambench --method A4 --split dev \
  --project-root "$PWD" --controller-config configs/a4-teambench-v1.json \
  --temperature 0 --output-dir artifacts/teambench-dev-a4

general-mas-run-teambench --method A8 --split dev \
  --project-root "$PWD" --controller-config configs/a8-teambench-v1.json \
  --temperature 0 --output-dir artifacts/teambench-dev-a8

general-mas-report --a4-dir artifacts/teambench-dev-a4 \
  --a8-dir artifacts/teambench-dev-a8 --expected-task-count 30 \
  --output-dir artifacts/teambench-dev-report

general-mas-baseline-report \
  --solo-dir artifacts/teambench-dev-solo \
  --a4-dir artifacts/teambench-dev-a4 \
  --a8-dir artifacts/teambench-dev-a8 \
  --expected-task-count 30 \
  --output-dir artifacts/teambench-dev-baseline-report

general-mas-strategy-report \
  --solo-dir artifacts/teambench-dev-solo \
  --plan-execute-dir artifacts/teambench-dev-plan-execute \
  --execute-review-dir artifacts/teambench-dev-execute-review \
  --a4-dir artifacts/teambench-dev-a4 \
  --a8-dir artifacts/teambench-dev-a8 \
  --expected-task-count 30 \
  --output-dir artifacts/teambench-dev-strategy-report
```

The strategy report rejects mismatched source revisions for runtime claims. Use
`--quality-only` only for an intentional bridge across revisions that still
matches the frozen task hash, sandbox image, models, temperature and token cap;
that mode removes latency and energy from the report.

Use `--strong-endpoint`, `--strong-model`, `--weak-endpoint` and `--weak-model`
to select local endpoints. Formal test results must not be used to tune another
controller iteration; use development tasks or a newly preregistered holdout.

## Test

```bash
pytest -q
```

The frozen protocol is [`docs/TEAMBENCH_PROTOCOL.md`](docs/TEAMBENCH_PROTOCOL.md).
