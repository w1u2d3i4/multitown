# MultiTown Public Bench

[返回中文说明](../README_ZH.md) | [Back to MultiTown](../README.md)

This standalone subproject evaluates fixed and dynamic multi-agent
organizations on public, general-purpose tasks. It is intentionally isolated
from MultiTown's synthetic town benchmark: task sets, metrics and claims are
never merged across the two tracks.

## Current scope

- Primary benchmark: [TeamBench](https://github.com/ybkim95/TeamBench), using
  deterministic task graders and 89 currently evaluable tasks from its public
  TeamBench-90 test list.
- Secondary data lock: MultiAgentBench/MARBLE, five domains and 500 released
  configurations. No MARBLE headline result is claimed here.

## What the names mean

`A4` and `A8` are experiment IDs, not model names, version numbers or rankings.
The `-TB` suffix means that an organization has been adapted to TeamBench's
software-engineering role contract. These scores are therefore not the
synthetic MultiTown A4/A8 scores.

In that role contract, the **Planner** writes a task plan, the **Executor** edits
code and runs tools, and the **Verifier** independently checks the result and
may request one repair:

- **A4-TB — fixed full team:** always call a strong Planner, a weak Executor and
  an independent strong Verifier, with a frozen one-loop remediation budget.
- **A8-TB — selective team:** start with the weak Executor, check the candidate
  with a deterministic public runtime validator, and activate the strong
  Planner or Verifier only when that evidence says more work is needed. A failed
  review can roll back to the best valid candidate.

Neither controller can inspect hidden grader outcomes when making a decision.

## Frozen formal result

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
general-mas-run-teambench --method A4 --split dev \
  --project-root "$PWD" --controller-config configs/a4-teambench-v1.json \
  --temperature 0 --output-dir artifacts/teambench-dev-a4

general-mas-run-teambench --method A8 --split dev \
  --project-root "$PWD" --controller-config configs/a8-teambench-v1.json \
  --temperature 0 --output-dir artifacts/teambench-dev-a8

general-mas-report --a4-dir artifacts/teambench-dev-a4 \
  --a8-dir artifacts/teambench-dev-a8 --expected-task-count 30 \
  --output-dir artifacts/teambench-dev-report
```

Use `--strong-endpoint`, `--strong-model`, `--weak-endpoint` and `--weak-model`
to select local endpoints. Formal test results must not be used to tune another
controller iteration; use development tasks or a newly preregistered holdout.

## Test

```bash
pytest -q
```

The frozen protocol is [`docs/TEAMBENCH_PROTOCOL.md`](docs/TEAMBENCH_PROTOCOL.md).
