# MultiTown

[English](README.md) | [简体中文](README_ZH.md)

<p align="center">
  <img src="demo/assets/multitown-arena.gif" alt="MultiTown Arena: A4 and A8 AI organizations solve the same task" width="960" />
</p>

<p align="center">
  <strong>A visual cyber town for AI agents — build an AI company, watch it work, spend, validate, escalate, and adapt.</strong>
</p>

MultiTown is a **visual cyber town**, an AI-organization digital twin, and a
Python runtime for cost-aware multi-agent control. Planner, Worker, Specialist,
Validator and Router agents become buildings and residents with different
information, permissions and costs. The Arena turns otherwise invisible model
calls, hand-offs, validation, escalation, token use and outcomes into a replay
you can watch and compare.

## What the cyber town can do

| Capability | What it looks like in MultiTown |
| --- | --- |
| Visual organization replay | Watch task packets move through two towns, see active buildings, queues, alerts, validation and final delivery in the browser Arena. |
| Role and permission isolation | Give Planner, Executor and Verifier different context, tools and authority instead of letting one agent propose, modify and certify its own work. |
| Cost-aware dispatch | Start with an economical worker and activate a strong specialist or review only when the controller's evidence and budget allow it. |
| Validation and recovery | Inspect observable runtime evidence, trigger review or escalation, enforce token guards, and preserve deterministic fallback/rollback paths. |
| Comparable experiments | Replay fixed and adaptive organizations on frozen tasks while recording success, tokens, latency, energy and safety outcomes. |
| Product path | Use the same town abstraction as the foundation for future characters, plots, missions and player intervention on `agentic-rpg`. |

The current Arena is a deterministic visualization and comparison surface, not
yet a free-form game. The RPG branch is where the cyber town becomes playable.

## Why Multi-Agent?

MultiTown does not assume that adding agents automatically improves accuracy.
It treats a multi-agent system as an **organization and governance problem**:
different roles can receive different information and permissions, while a
controller decides when extra planning, execution or review is worth its cost.

A single full-access agent is simple but can read requirements, change the work
and certify itself. An always-on team separates those duties but pays the full
coordination cost on every task. MultiTown studies the middle ground: start with
the least expensive valid organization, validate observable work, and activate
specialists only when needed. The `public-bench` branch tests this proposition
against both single-agent and fixed-team anchors on public tasks.

## Project branches

Start here on `main`. Each long-lived branch has one job, and experimental
claims stay on the branch that owns their evidence:

| Branch | Purpose | What to expect |
| --- | --- | --- |
| [`main`](https://github.com/w1u2d3i4/multitown/tree/main) | Stable project showcase | The current validated MultiTown result, Arena demo, public runtime and conservative headline claims. |
| [`public-bench`](https://github.com/w1u2d3i4/multitown/tree/public-bench) | External evidence track | Tests the method on public general-purpose benchmarks against a canonical single agent and a fixed Planner–Executor–Verifier team; reports quality, tokens, latency and energy, including negative results. |
| [`agentic-rl`](https://github.com/w1u2d3i4/multitown/tree/agentic-rl) | Learned-control research | Experiments with replacing deterministic delegation rules by trained sequential policies under quality, budget and safety constraints. It is also the control research foundation for future role-playing agents. |
| [`agentic-rpg`](https://github.com/w1u2d3i4/multitown/tree/agentic-rpg) | Tabletop-RPG product direction | Evolves the town from an experiment viewer into a playable multi-agent role-playing prototype with characters, plots, tasks and player intervention. |

The branches are not a performance ranking. `public-bench` asks whether the
current method transfers; `agentic-rl` asks whether control can be learned;
`agentic-rpg` explores how that control becomes gameplay.

This repository is a code-only public surface with concise, audited result
summaries. It intentionally excludes raw experiment records, generated result
files, model checkpoints, benchmark datasets, internal process notes,
interrupted runs, and the private development Git history.

## Launch the Arena

The bundled replay runs without a model, API key, build step, or network call:

```bash
git clone https://github.com/w1u2d3i4/multitown.git
cd multitown
python3 -m http.server 8000 --directory demo
```

Open <http://127.0.0.1:8000>. The displayed benchmark aggregates are measured
frozen results. The animated work order is a deterministic explanatory scenario,
not a raw experimental episode. See [`demo/`](demo/) for the automatic GIF
generation command.

## How to read the experiment labels

The `A` numbers are organization experiment IDs, not model versions or a
ranking. The result section mainly uses four designs:

| ID | Plain-language meaning |
| --- | --- |
| A4 | Fixed full team: always call a strong Planner, three weak Workers and an independent strong Verifier. |
| A6 | Statistical pre-task router: choose one complete organization before execution using cross-fitted scenario statistics and a budget. |
| A7 | Learned pre-task router: predict quality, tokens and latency from safe task features, then choose an organization before execution. |
| A8 | Execution-time adaptive controller: start with an economical agent and delegate, escalate or review only when validation evidence requires it. |

A8 is a deterministic selective-delegation controller, not a trained RL
policy. Learned-controller attempts are isolated on the `agentic-rl` branch.

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

## Public transfer result: TeamBench

The `public-bench` branch transfers MultiTown's organization ideas to 89 public
TeamBench tasks with OS-level role isolation and matched model, task, seed,
container and token settings:

| Strategy | Fully passed | Mean partial | Tokens/task | p95 latency | Energy |
| --- | ---: | ---: | ---: | ---: | ---: |
| PlanExecute-TB | 16/89 | 0.61603 | 49,579 | 151.96 s | 67.89 Wh |
| Solo-TB | 14/89 | 0.63989 | 84,085 | 237.06 s | 90.94 Wh |
| **MT-CapacityRoute-v1** | **20/89** | **0.64180** | **48,296** | **91.06 s** | **63.32 Wh** |

MT-CapacityRoute keeps Planner–Executor separation, routes strong execution to
three development-selected task categories and uses the economical Executor
elsewhere. It is the strongest tested point under this local frozen harness:
relative to Solo it adds six passes with no lost Solo passes, cuts tokens by
42.56%, and cuts monitored energy by 30.37%. Mean-partial confidence intervals
against both anchors still include zero, so this is not a literature-wide SOTA
claim. Protocols, negative results and transferable runtime controls live on
[`public-bench`](https://github.com/w1u2d3i4/multitown/tree/public-bench).

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
