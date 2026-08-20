# MultiTown

[English](README.md) | [简体中文](README_ZH.md)

MultiTown is a Python runtime toolkit for studying cost-aware multi-agent
organization and sequential control. It contains deterministic environments,
organization controllers, routing and safety components, trace tooling, and
machine-readable schemas.

This repository is a code-only public surface with concise, audited result
summaries. It intentionally excludes raw experiment records, generated result
files, model checkpoints, benchmark datasets, internal process notes,
interrupted runs, and the private development Git history.

## Selected benchmark results

The strongest frozen MultiTown result is the deterministic A8 execution-time
controller on the 180-scenario held-out split:

| Controller | Success | Tokens per decision | Mean E2E latency | p95 E2E latency |
| --- | ---: | ---: | ---: | ---: |
| A8 | 142/180 (78.89%) | 621.5 | 1.104 s | 2.275 s |

Under the preregistered paired comparison:

- A8 improved success over A4 by **11.67 percentage points** (95% paired
  bootstrap CI: **+4.44 to +18.89**) while using **76.54% fewer tokens**.
- A8 improved success over A7 by **11.11 percentage points** (95% paired
  bootstrap CI: **+4.44 to +17.78**) while using **54.58% fewer tokens**.
- The earlier A6 budget router matched the observed A4 accuracy level while
  reducing tokens per decision by **24.63%** and mean latency by **31.79%**;
  its accuracy interval crossed zero, so this is an efficiency result rather
  than evidence that A6 is more accurate.

A8 is a deterministic heuristic controller, not a trained RL policy. These
results are local Qwen/llama.cpp measurements on the frozen synthetic
MultiTown population; they do not establish general multi-agent superiority.

## Agentic RL and safety findings

MultiTown retains both positive optimization results and safety failures:

| Study | Main result | Required interpretation |
| --- | --- | --- |
| A9-v1 offline fitted-Q | 75.00% success vs 72.22% for the matched A8 baseline | Difference interval crossed zero; no significant advantage claim |
| A9-v2 masked PPO | Success 23.83% → 34.00%; +10.17 pp (95% CI +7.87 to +12.47); tokens −25.41% | Train-only controller result; unsafe episodes rose 15.73% → 66.00% |
| A9-v3 hard-shield diagnostic | Unsafe episodes 66.00% → 5.84% | Autonomous success fell 34.00% → 0%; safety–utility negative result |
| A22 constrained-PPO follow-up | 60 fits, 345,600 rollouts, 36,000 calibration rows and 9,000 outer rows | Measured safety margins recovered, but success noninferiority was not stable and tokens increased |

The historical A10 long-horizon run is not listed as a positive result because
a later audit found that policy-visible fields deterministically revealed the
correct action. A23 was invalidated by snapshot-binding failure, and the Stage
W CR axis was inert. These failures are preserved in the research evidence but
their raw records are intentionally outside this code-only repository.

## Install

MultiTown requires Linux and Python 3.12 or newer.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Optional controller-training dependencies are installed separately:

```bash
python -m pip install -e '.[rl]'
```

Public benchmark adapters use a separate optional dependency set:

```bash
python -m pip install -e '.[reproduction]'
```

## Explore the runtime

```bash
multitown-bench --help
multitown-validate-serving-trace --help
```

After installing `.[reproduction]`, the routing and A8 commands are available;
after installing `.[rl]`, the PPO commands are available:

```bash
multitown-run-a8 --help
multitown-a10-ppo --help
```

Model-backed commands expect user-supplied OpenAI-compatible endpoints and
model identifiers. The package does not download weights and no credentials
belong in tracked files.

## Test

```bash
pytest -q \
  tests/test_contracts.py \
  tests/test_a8_controller.py \
  tests/test_stateful_ops.py \
  tests/test_stateful_behavior.py \
  tests/test_stateful_groups.py \
  tests/test_stateful_pomdp.py \
  tests/test_serving_trace.py
```

The tests in this repository exercise only the published runtime surface.
They do not reproduce or validate private experiments.

## Repository boundary

Included:

- `multitown/`: runtime implementation;
- `schemas/`: public machine-readable contracts;
- `tests/`: selected runtime tests.

Excluded by design:

- experiment records and monitoring curves;
- raw result tables and generated reports;
- datasets, trajectories and model outputs;
- model or controller checkpoints;
- research diaries, plans and review-process documents;
- local endpoints, credentials and machine-specific provenance.

## License

MultiTown is released under the MIT License. See [LICENSE](LICENSE).
