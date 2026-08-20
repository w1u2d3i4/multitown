# MultiTown

[English](README.md) | [简体中文](README_ZH.md)

MultiTown is a Python runtime toolkit for studying cost-aware multi-agent
organization and sequential control. It contains deterministic environments,
organization controllers, routing and safety components, trace tooling, and
machine-readable schemas.

This repository is a code-only public surface. It intentionally excludes
experiment records, result tables, generated artifacts, model checkpoints,
benchmark datasets, internal process notes, interrupted runs, and the private
development Git history.

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
- datasets, trajectories and model outputs;
- model or controller checkpoints;
- research diaries, plans and review-process documents;
- local endpoints, credentials and machine-specific provenance.

## License

MultiTown is released under the MIT License. See [LICENSE](LICENSE).
