# Agentic RL in MultiTown

[简体中文](agentic-rl.zh-CN.md)

> Research status: this branch exposes controller-learning code and audited
> findings. It does not claim that a learned policy is ready to replace A8.

## The idea in one sentence

MultiTown does not use reinforcement learning to retrain the language model. It
trains the **organization controller around the models**: when to gather more
evidence, call another agent, escalate to a stronger model, request review,
execute, stop, or hand the task to a human.

The objective is therefore not accuracy alone. A useful controller must trade
off task success, token cost, latency, human effort, and unsafe execution.

## Controller formulation

MultiTown contains two related sequential-control surfaces:

| Surface | Public state | Actions | Objective |
| --- | --- | --- | --- |
| A9 offline fitted-Q | Current candidate and validation state, tokens and latency used, prior delegation/escalation/review, weak disagreement | `stop`, `delegate`, `escalate`, `review`, `human` | Success minus token, latency, safety, and human penalties |
| Long-horizon POMDP | A 47-dimensional public observation, remaining budgets, incident progress, tool/review state, and action mask | `observe`, `delegate`, `escalate`, `connect`, `review`, `execute`, `human`, `stop` | Multi-incident completion with action cost, safety, budget, delegation, and human penalties |

The policy does not receive the private correct action. Legal-action masks stop
the actor from sampling transitions that are unavailable in the current state.
Safety experiments additionally compare learned constraints and a public-state
review shield.

For the offline controller, the terminal reward is represented by:

```text
success
− token_penalty × tokens / 1000
− latency_penalty × latency
− safety_penalty × unsafe
− human_penalty × human_escalation
```

The long-horizon environment expands this contract with subgoal progress, tool
failure recovery, invalid-action and budget penalties, and unnecessary
delegation costs. Exact coefficients live in the versioned environment code,
not in an untracked experiment configuration.

## Evidence ladder

The deterministic A8 controller remains the strongest frozen reference on the
published 180-scenario comparison. The RL work should be read as a sequence of
mechanism and safety findings:

| Study | What it tested | Audited result |
| --- | --- | --- |
| A9-v1 | Offline fitted-Q over counterfactual controller transitions | 75.00% success vs 72.22% for matched A8; interval crossed zero |
| A9-v2 | Train-only masked PPO | Success 23.83% → 34.00% and tokens −25.41%; unsafe episodes rose 15.73% → 66.00% |
| A9-v3 | Hard review shield diagnostic | Unsafe episodes fell 66.00% → 5.84%, but autonomous success fell 34.00% → 0% |
| A22 | Constrained-PPO follow-up across 60 fits | Safety margins recovered in the measured protocol; success noninferiority was unstable and tokens increased |
| A26 | Non-leaking risk-calibrated router | Negative: 18.9% vs 18.7% success; unsafe ceiling missed by 0.1 pp |
| A28 | Agreement-gated specialist-first router | 30.9% vs 18.6% success (paired 95% CI for difference +9.1 to +15.5 pp); tokens −3.17%; all frozen gates passed |

These results show why cost-only policy improvement is not enough: an agentic
controller can become cheaper and apparently more successful while learning an
unsafe stopping or execution policy.

## Code map

| Path | Role |
| --- | --- |
| `multitown/a9_fitted_q.py` | Five-action offline fitted-Q controller |
| `multitown/long_horizon_env.py` | Deterministic 20–50 step POMDP and reward contract |
| `multitown/a9_long_horizon_env.py` | Leakage-resistant train-only episode generator and audit |
| `multitown/ppo_controller.py` | Masked actor-critic PPO implementation |
| `multitown/a9_ppo_oof.py` | Out-of-fold training protocol |
| `multitown/a9_safety_development.py` | Review-shield safety diagnostic |
| `multitown/a22_constrained_ppo.py` | Lagrangian and shield mechanism primitives |
| `multitown/a25_method_conformance.py` | Method and claim-conformance checks |
| `multitown/a26_safe_router.py` | Non-leaking fixture and risk-calibrated policy-improvement baseline |
| `multitown/a28_conservative_router.py` | Confirmed specialist-first controller and next-RL fallback |

The next frozen protocol is [A26 risk-calibrated policy improvement](A26_SAFE_POLICY_IMPROVEMENT.md).
It first establishes a safer learned routing baseline; it is explicitly not
called full Agentic RL.

A26 subsequently failed its OOD gates. Its frozen successor,
[A28 agreement-gated specialist-first routing](A28_CONSERVATIVE_ROUTER.md),
changed the unsafe disagreement path, used a new confirmation bank, and passed
all pre-registered gates. A28 is still not full Agentic RL.

## External benchmark track

The next phase no longer evaluates only MultiTown's private town distribution.
It has a separate, frozen external benchmark protocol for AgentConductor,
MAGRPO, and MATTRL. See
[External Agentic Benchmark Protocol](EXTERNAL_AGENTIC_BENCHMARKS.md).

This track does not reuse TeamBench as an RL environment. TeamBench remains the
non-RL public evidence on the `public-bench` branch; the three method-family
benchmarks here measure training or test-time experience mechanisms under their
own task rewards.

## Explore the public entry points

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev,rl,reproduction]'

multitown-run-a9-offline --help
multitown-a10-ppo --help
multitown-a9-ppo-oof --help
multitown-a9-review-shield --help
multitown-a22-adaptive --help
multitown-a22-report --help
multitown-a26-safe-router --help
multitown-a28-conservative-router --help
```

The repository intentionally excludes raw experiment records, private episode
banks, checkpoints, and model outputs. The commands expose and test the public
mechanics; they do not reproduce the private headline runs by themselves.

## Branch contract

- `main` is the stable public runtime and Arena branch.
- `agentic-rl` is the research surface for learned controller code,
  documentation, and reproducible public fixtures.
- `agentic-rpg` is the future playable narrative surface. Learned delegation,
  budget, review and human-handoff mechanics developed here may feed that
  branch only after they pass their evidence and safety gates.
- A learned-policy result should move to `main` only after frozen held-out
  evaluation, equal-budget comparison against A8, leakage checks, safety
  accounting, and tests that can run from public inputs.
- Negative results remain first-class evidence. An experiment is not described
  as an improvement when its paired interval crosses zero or when safety gains
  destroy autonomous utility.

## Next public milestone

The next publishable milestone is a small public episode bank plus a one-command
A8-versus-RL replay that emits the same trace schema consumed by MultiTown
Arena. Until that exists, this branch is best described as an **Agentic RL
research implementation**, not a generally superior trained controller.
