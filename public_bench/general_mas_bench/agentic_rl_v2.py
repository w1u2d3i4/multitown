"""Train a pessimistic counterfactual ensemble for TeamBench orchestration.

The controller combines paired execution/recovery outcomes with public execution
feedback.  It uses bootstrap disagreement as an uncertainty penalty and never
consults task IDs, specifications, hidden graders, or expected outputs at
inference time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .common import read_json, write_json

POLICY_SCHEMA = "multitown-teambench-pessimistic-ensemble-q-v2"
DATASET_SCHEMA = "multitown-teambench-counterfactual-v2"
ACTION_SPACE = ("stop", "delegate", "escalate", "review", "human/abstain")
PHASE_ACTIONS = {
    "post_execution": ("stop", "escalate", "human/abstain"),
    "post_replan": ("stop", "delegate", "human/abstain"),
    "post_recovery": ("stop", "review", "human/abstain"),
}
INTERVENTION_ACTION = {
    "post_execution": "escalate",
    "post_replan": "delegate",
    "post_recovery": "stop",
}
BASELINE_ACTION = {
    "post_execution": "stop",
    "post_replan": "stop",
    "post_recovery": "review",
}

STATE_NUMERIC = (
    "plan_delivered",
    "plan_retry_used",
    "recovery_plan_delivered",
    "consumed_tokens",
    "remaining_token_budget",
    "consumed_latency_s",
    "brief_word_count",
    *(f"brief_hash_{index:02d}" for index in range(16)),
    "workspace_file_count",
    "workspace_test_file_count",
    "workspace_language_count",
    "workspace_has_python",
    "workspace_has_javascript",
    "workspace_has_go",
    "workspace_has_rust",
    "workspace_has_java",
    "workspace_has_shell",
    "workspace_has_data",
)
VALIDATOR_NUMERIC = (
    "workspace_changed",
    "successful_commands",
    "failed_commands",
    "timed_out_commands",
    "repeated_commands",
    "max_failed_command_repetitions",
    "turns_used",
    "turn_budget_exhausted",
    "reliability_score",
    "hard_fail",
    "test_commands",
    "successful_test_commands",
    "failed_test_commands",
    "last_test_exit_code",
)
FEATURE_NAMES = (
    *STATE_NUMERIC,
    *(f"current_{name}" for name in VALIDATOR_NUMERIC),
    *(f"execution_{name}" for name in VALIDATOR_NUMERIC),
    *(f"validator_delta_{name}" for name in VALIDATOR_NUMERIC),
)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _state_values(state: dict[str, Any]) -> dict[str, Any]:
    current = state.get("validator") or {}
    execution = state.get("execution_validator") or current
    values: dict[str, Any] = {
        "category": str(state.get("category") or current.get("category") or "Other"),
        "difficulty": str(
            state.get("difficulty") or current.get("difficulty") or "unknown"
        ),
    }
    for name in STATE_NUMERIC:
        values[name] = _number(state.get(name))
    for name in VALIDATOR_NUMERIC:
        default = -1.0 if name == "last_test_exit_code" else 0.0
        current_value = _number(current.get(name), default)
        execution_value = _number(execution.get(name), default)
        values[f"current_{name}"] = current_value
        values[f"execution_{name}"] = execution_value
        values[f"validator_delta_{name}"] = current_value - execution_value
    return values


def _feature_spec(states: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [_state_values(state) for state in states]
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    for name in FEATURE_NAMES:
        column = np.asarray([row[name] for row in rows], dtype=float)
        means[name] = float(column.mean())
        scale = float(column.std())
        scales[name] = scale if scale > 1e-9 else 1.0
    return {
        "numeric_features": list(FEATURE_NAMES),
        "numeric_means": means,
        "numeric_scales": scales,
        "categories": sorted({row["category"] for row in rows}),
        "difficulties": sorted({row["difficulty"] for row in rows}),
    }


def encode_state(state: dict[str, Any], spec: dict[str, Any]) -> np.ndarray:
    values = _state_values(state)
    encoded = [1.0]
    for name in spec["numeric_features"]:
        encoded.append(
            (float(values[name]) - float(spec["numeric_means"][name]))
            / float(spec["numeric_scales"][name])
        )
    encoded.extend(
        float(values["category"] == category) for category in spec["categories"]
    )
    encoded.extend(
        float(values["difficulty"] == difficulty)
        for difficulty in spec["difficulties"]
    )
    return np.asarray(encoded, dtype=float)


def _ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    penalty = np.eye(x.shape[1], dtype=float) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def _outcome_reward(
    outcome: dict[str, Any],
    *,
    reference_tokens: float,
    reference_latency_s: float,
) -> float:
    return (
        float(outcome["partial_score"])
        + 0.25 * float(bool(outcome["passed"]))
        - 0.05
        * math.log1p(float(outcome["total_tokens"]) / max(reference_tokens, 1.0))
        - 0.02
        * math.log1p(float(outcome["latency_s"]) / max(reference_latency_s, 1e-6))
        - 0.25 * float(bool(outcome.get("invocation_error", False)))
        - 0.50 * float(bool(outcome.get("safety_violation", False)))
    )


def _with_sunk_cost(outcome: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    return {
        **outcome,
        "total_tokens": int(state.get("consumed_tokens", outcome["total_tokens"])),
        "latency_s": float(state.get("consumed_latency_s", outcome["latency_s"])),
    }


def _advantages(
    episode: dict[str, Any],
    *,
    reference_tokens: float,
    reference_latency_s: float,
) -> dict[str, float]:
    states = episode["states"]
    prefix = episode["outcomes"]["prefix"]
    recovery = episode["outcomes"]["recovery"]

    def reward(outcome: dict[str, Any]) -> float:
        return _outcome_reward(
            outcome,
            reference_tokens=reference_tokens,
            reference_latency_s=reference_latency_s,
        )

    prefix_post_replan = _with_sunk_cost(prefix, states["post_replan"])
    prefix_post_recovery = _with_sunk_cost(prefix, states["post_recovery"])
    recovery_post_recovery = _with_sunk_cost(recovery, states["post_recovery"])
    best_after_recovery = max(reward(recovery), reward(prefix_post_recovery))
    best_after_replan = max(reward(prefix_post_replan), best_after_recovery)
    return {
        "post_execution": best_after_replan - reward(prefix),
        "post_replan": best_after_recovery - reward(prefix_post_replan),
        "post_recovery": reward(recovery_post_recovery) - reward(prefix_post_recovery),
    }


def _fit_ensemble(
    episodes: list[dict[str, Any]],
    *,
    spec: dict[str, Any],
    alpha: float,
    ensemble_size: int,
    seed: int,
    reference_tokens: float,
    reference_latency_s: float,
) -> dict[str, list[list[float]]]:
    targets = [
        _advantages(
            episode,
            reference_tokens=reference_tokens,
            reference_latency_s=reference_latency_s,
        )
        for episode in episodes
    ]
    rng = np.random.default_rng(seed)
    models: dict[str, list[list[float]]] = {phase: [] for phase in PHASE_ACTIONS}
    for phase in PHASE_ACTIONS:
        x = np.stack([encode_state(episode["states"][phase], spec) for episode in episodes])
        y = np.asarray([target[phase] for target in targets], dtype=float)
        for _ in range(ensemble_size):
            indices = rng.integers(0, len(episodes), size=len(episodes))
            models[phase].append(_ridge(x[indices], y[indices], alpha).tolist())
    return models


def _estimate(
    state: dict[str, Any],
    *,
    phase: str,
    spec: dict[str, Any],
    models: dict[str, list[list[float]]],
    uncertainty_beta: float,
) -> dict[str, float]:
    x = encode_state(state, spec)
    values = np.asarray(
        [float(x @ np.asarray(weights, dtype=float)) for weights in models[phase]],
        dtype=float,
    )
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "lcb": mean - uncertainty_beta * std,
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _decision(
    phase: str,
    estimate: dict[str, float],
    *,
    margin: float,
) -> str:
    if estimate["lcb"] > margin:
        return INTERVENTION_ACTION[phase]
    return BASELINE_ACTION[phase]


def _budget_allows(
    phase: str, state: dict[str, Any], reserves: dict[str, int]
) -> bool:
    required = int(reserves.get(phase, 0))
    return int(state.get("remaining_token_budget", 0)) >= required


def select_action(policy: dict[str, Any], phase: str, state: dict[str, Any]) -> dict[str, Any]:
    validate_policy(policy)
    if phase not in PHASE_ACTIONS:
        raise ValueError(f"unsupported policy phase: {phase}")
    estimate = _estimate(
        state,
        phase=phase,
        spec=policy["feature_spec"],
        models=policy["advantage_ensemble"],
        uncertainty_beta=float(policy["uncertainty_beta"]),
    )
    action = _decision(
        phase, estimate, margin=float(policy["conservative_margin"])
    )
    budget_allowed = _budget_allows(
        phase, state, policy.get("budget_reserve_tokens", {})
    )
    if action == INTERVENTION_ACTION[phase] and not budget_allowed:
        action = BASELINE_ACTION[phase]
    scores = {candidate: -1.0 for candidate in PHASE_ACTIONS[phase]}
    scores[BASELINE_ACTION[phase]] = 0.0
    scores[INTERVENTION_ACTION[phase]] = estimate["lcb"]
    return {
        "schema_version": "multitown-agentic-rl-decision-v2",
        "policy_source": "trained_pessimistic_counterfactual_ensemble",
        "policy_sha256": policy["policy_sha256"],
        "phase": phase,
        "action_space": list(ACTION_SPACE),
        "valid_actions": list(PHASE_ACTIONS[phase]),
        "state": state,
        "q_values": scores,
        "advantage": estimate,
        "action": action,
        "reason": (
            "positive_pessimistic_advantage"
            if action == INTERVENTION_ACTION[phase]
            else (
                "insufficient_reserved_budget"
                if not budget_allowed
                else "uncertainty_or_cost_gate"
            )
        ),
    }


def validate_policy(policy: dict[str, Any]) -> None:
    if policy.get("schema_version") != POLICY_SCHEMA:
        raise ValueError("unsupported Agentic RL v2 policy schema")
    supplied_hash = policy.get("policy_sha256")
    unhashed = {key: value for key, value in policy.items() if key != "policy_sha256"}
    if supplied_hash != _canonical_sha256(unhashed):
        raise ValueError("Agentic RL v2 policy hash mismatch")
    if tuple(policy.get("action_space", [])) != ACTION_SPACE:
        raise ValueError("Agentic RL v2 action space mismatch")
    if float(policy.get("uncertainty_beta", 0.0)) <= 0.0:
        raise ValueError("Agentic RL v2 requires a positive uncertainty penalty")
    if float(policy.get("conservative_margin", -1.0)) < 0.0:
        raise ValueError("Agentic RL v2 requires a non-negative action margin")
    reserves = policy.get("budget_reserve_tokens")
    if not isinstance(reserves, dict) or set(reserves) != set(PHASE_ACTIONS):
        raise ValueError("Agentic RL v2 budget reserve phases mismatch")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in reserves.values()
    ):
        raise ValueError("Agentic RL v2 budget reserves must be non-negative integers")
    ensemble_size = policy.get("ensemble_size")
    if isinstance(ensemble_size, bool) or not isinstance(ensemble_size, int):
        raise TypeError("Agentic RL v2 ensemble size must be an integer")
    if ensemble_size < 2:
        raise ValueError("Agentic RL v2 requires at least two ensemble members")
    width = (
        1
        + len(policy["feature_spec"]["numeric_features"])
        + len(policy["feature_spec"]["categories"])
        + len(policy["feature_spec"]["difficulties"])
    )
    for phase in PHASE_ACTIONS:
        weights = policy.get("advantage_ensemble", {}).get(phase, [])
        if len(weights) != ensemble_size or any(len(row) != width for row in weights):
            raise ValueError(f"Agentic RL v2 ensemble mismatch for {phase}")


def _selected_outcome(episode: dict[str, Any], actions: list[str]) -> dict[str, Any]:
    states = episode["states"]
    prefix = episode["outcomes"]["prefix"]
    recovery = episode["outcomes"]["recovery"]
    if actions[0] != "escalate":
        return prefix
    if actions[1] != "delegate":
        return _with_sunk_cost(prefix, states["post_replan"])
    if actions[2] == "stop":
        return recovery
    return _with_sunk_cost(prefix, states["post_recovery"])


def _rollout(
    episode: dict[str, Any],
    *,
    spec: dict[str, Any],
    models: dict[str, list[list[float]]],
    uncertainty_beta: float,
    margin: float,
    budget_reserve_tokens: dict[str, int],
) -> tuple[list[str], dict[str, Any]]:
    actions: list[str] = []
    for phase in PHASE_ACTIONS:
        estimate = _estimate(
            episode["states"][phase],
            phase=phase,
            spec=spec,
            models=models,
            uncertainty_beta=uncertainty_beta,
        )
        action = _decision(phase, estimate, margin=margin)
        if action == INTERVENTION_ACTION[phase] and not _budget_allows(
            phase, episode["states"][phase], budget_reserve_tokens
        ):
            action = BASELINE_ACTION[phase]
        actions.append(action)
        if phase != "post_recovery" and action == "stop":
            break
    return actions, _selected_outcome(episode, actions)


def _folds(episodes: list[dict[str, Any]]) -> list[tuple[list[int], list[int]]]:
    seeds = sorted({int(episode["seed"]) for episode in episodes})
    if len(seeds) >= 2:
        return [
            (
                [index for index, row in enumerate(episodes) if int(row["seed"]) != seed],
                [index for index, row in enumerate(episodes) if int(row["seed"]) == seed],
            )
            for seed in seeds
        ]
    return [
        ([other for other in range(len(episodes)) if other != index], [index])
        for index in range(len(episodes))
    ]


def _cross_validate(
    episodes: list[dict[str, Any]],
    *,
    alpha: float,
    uncertainty_beta: float,
    margin: float,
    reference_tokens: float,
    reference_latency_s: float,
    budget_reserve_quantile: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for fold_index, (train_indices, test_indices) in enumerate(_folds(episodes)):
        training = [episodes[index] for index in train_indices]
        spec = _feature_spec([
            episode["states"][phase]
            for episode in training
            for phase in PHASE_ACTIONS
        ])
        models = _fit_ensemble(
            training,
            spec=spec,
            alpha=alpha,
            ensemble_size=16,
            seed=20260827 + fold_index,
            reference_tokens=reference_tokens,
            reference_latency_s=reference_latency_s,
        )
        budget_reserve_tokens = _budget_reserves(
            training, quantile=budget_reserve_quantile
        )
        for index in test_indices:
            episode = episodes[index]
            actions, outcome = _rollout(
                episode,
                spec=spec,
                models=models,
                uncertainty_beta=uncertainty_beta,
                margin=margin,
                budget_reserve_tokens=budget_reserve_tokens,
            )
            rows.append({
                "instance_id": episode["instance_id"],
                "actions": actions,
                **outcome,
                "reward": _outcome_reward(
                    outcome,
                    reference_tokens=reference_tokens,
                    reference_latency_s=reference_latency_s,
                ),
            })
    return {
        "alpha": alpha,
        "uncertainty_beta": uncertainty_beta,
        "margin": margin,
        "budget_reserve_quantile": budget_reserve_quantile,
        "passes": sum(bool(row["passed"]) for row in rows),
        "mean_partial_score": float(np.mean([row["partial_score"] for row in rows])),
        "mean_total_tokens": float(np.mean([row["total_tokens"] for row in rows])),
        "mean_latency_s": float(np.mean([row["latency_s"] for row in rows])),
        "mean_reward": float(np.mean([row["reward"] for row in rows])),
        "routes": dict(Counter("/".join(row["actions"]) for row in rows)),
        "rows": rows,
    }


def validate_dataset(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    if dataset.get("schema_version") != DATASET_SCHEMA:
        raise ValueError("unsupported Agentic RL v2 dataset schema")
    episodes = dataset.get("episodes")
    if not isinstance(episodes, list) or len(episodes) < 3:
        raise ValueError("Agentic RL v2 dataset needs at least three episodes")
    instance_ids = [str(episode.get("instance_id")) for episode in episodes]
    if len(instance_ids) != len(set(instance_ids)):
        raise ValueError("Agentic RL v2 dataset contains duplicate instances")
    for episode in episodes:
        if set(episode.get("states", {})) != set(PHASE_ACTIONS):
            raise ValueError(f"incomplete states for {episode.get('instance_id')}")
        if set(episode.get("outcomes", {})) != {"prefix", "recovery"}:
            raise ValueError(f"incomplete outcomes for {episode.get('instance_id')}")
    return episodes


def _budget_reserves(
    episodes: list[dict[str, Any]], *, quantile: float = 0.75
) -> dict[str, int]:
    if not 0.5 <= quantile <= 1.0:
        raise ValueError("budget reserve quantile must be between 0.5 and 1.0")
    escalation = [
        max(
            0,
            int(episode["outcomes"]["recovery"]["total_tokens"])
            - int(episode["outcomes"]["prefix"]["total_tokens"]),
        )
        for episode in episodes
    ]
    delegation = [
        max(
            0,
            int(episode["outcomes"]["recovery"]["total_tokens"])
            - int(episode["states"]["post_replan"].get("consumed_tokens", 0)),
        )
        for episode in episodes
    ]
    return {
        "post_execution": int(np.quantile(escalation, quantile, method="higher")),
        "post_replan": int(np.quantile(delegation, quantile, method="higher")),
        "post_recovery": 0,
    }


def _public_features(brief: str, workspace: Path) -> dict[str, float]:
    bins = [0.0] * 16
    words = re.findall(r"[a-z0-9_+#.-]+", brief.lower())
    for word in words:
        digest = hashlib.sha256(word.encode("utf-8")).digest()
        bins[digest[0] % len(bins)] += 1.0 if digest[1] & 1 else -1.0
    norm = math.sqrt(sum(value * value for value in bins)) or 1.0
    files = [path for path in workspace.rglob("*") if path.is_file()]
    suffixes = {path.suffix.lower() for path in files}
    return {
        "brief_word_count": float(len(words)),
        **{f"brief_hash_{index:02d}": value / norm for index, value in enumerate(bins)},
        "workspace_file_count": float(len(files)),
        "workspace_test_file_count": float(sum(
            "test" in path.name.lower() or "tests" in path.parts for path in files
        )),
        "workspace_language_count": float(len(suffixes)),
        "workspace_has_python": float(".py" in suffixes),
        "workspace_has_javascript": float(bool({".js", ".ts", ".tsx"} & suffixes)),
        "workspace_has_go": float(".go" in suffixes),
        "workspace_has_rust": float(".rs" in suffixes),
        "workspace_has_java": float(bool({".java", ".kt"} & suffixes)),
        "workspace_has_shell": float(".sh" in suffixes),
        "workspace_has_data": float(bool({".csv", ".json", ".jsonl", ".sql"} & suffixes)),
    }


def _is_test_command(command: str) -> bool:
    value = " ".join(command.lower().split())
    return any(marker in value for marker in (
        "pytest", "unittest", "npm test", "npm run test", "yarn test",
        "pnpm test", "go test", "cargo test", "mvn test", "gradle test",
        "./test", " test_", "/tests/", " tests/",
    ))


def _test_signals(task_root: Path, phase: str) -> dict[str, int]:
    pairs: list[int] = []
    for path in sorted((task_root / "logs" / phase / "executor").glob("turn_*.json")):
        row = read_json(path)
        for call, result in zip(row.get("tool_calls", []), row.get("tool_results", [])):
            command = str(call.get("args", {}).get("cmd", ""))
            if call.get("name") == "run" and _is_test_command(command):
                pairs.append(int(result.get("exit_code", 1)))
    return {
        "test_commands": len(pairs),
        "successful_test_commands": sum(code == 0 for code in pairs),
        "failed_test_commands": sum(code != 0 for code in pairs),
        "last_test_exit_code": pairs[-1] if pairs else -1,
    }


def build_dataset(run_dirs: list[Path]) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    skipped: dict[str, str] = {}
    sources: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        config = read_json(run_dir / "config.json")
        seed = int(config.get("seed_override", 0))
        result_path = run_dir / "results.jsonl"
        sources.append({
            "run": run_dir.name,
            "seed": seed,
            "results_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
            "config_sha256": hashlib.sha256((run_dir / "config.json").read_bytes()).hexdigest(),
        })
        rows = [
            json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for row in rows:
            task_id = str(row["task_id"])
            instance_id = f"{task_id}::seed={seed}"
            states = row.get("training_states") or {}
            prefix = row.get("shadow_plan_execute")
            recovery = row.get("shadow_recovery")
            if set(states) != set(PHASE_ACTIONS):
                skipped[instance_id] = "incomplete_sequential_states"
                continue
            if not isinstance(prefix, dict) or not isinstance(recovery, dict):
                skipped[instance_id] = "missing_counterfactual_grade"
                continue
            task_root = run_dir / "tasks" / task_id
            plan_workspace = task_root / "candidates" / "plan_execute" / "workspace"
            brief = (task_root / "task" / "brief.md").read_text(encoding="utf-8")
            public = _public_features(brief, plan_workspace)
            execution_signals = _test_signals(task_root, "execution")
            recovery_signals = _test_signals(task_root, "recovery")
            execution_validator = {
                **(states["post_execution"].get("validator") or {}),
                **execution_signals,
            }
            enriched: dict[str, dict[str, Any]] = {}
            for phase, state in states.items():
                validator = dict(state.get("validator") or {})
                validator.update(
                    recovery_signals if phase == "post_recovery" else execution_signals
                )
                enriched[phase] = {
                    **state,
                    **public,
                    "category": str(row.get("category") or "Other"),
                    "difficulty": str(row.get("difficulty") or "unknown"),
                    "validator": validator,
                    "execution_validator": execution_validator,
                }
            episodes.append({
                "instance_id": instance_id,
                "task_id": task_id,
                "seed": seed,
                "category": row.get("category"),
                "difficulty": row.get("difficulty"),
                "states": enriched,
                "outcomes": {"prefix": prefix, "recovery": recovery},
            })
    dataset = {
        "schema_version": DATASET_SCHEMA,
        "source": sources,
        "episodes": episodes,
        "skipped": skipped,
    }
    validate_dataset(dataset)
    return dataset


def train_policy(
    dataset: dict[str, Any], *, budget_reserve_quantile: float = 0.75
) -> dict[str, Any]:
    episodes = validate_dataset(dataset)
    reference_tokens = float(np.median([
        episode["outcomes"]["prefix"]["total_tokens"] for episode in episodes
    ]))
    reference_latency_s = float(np.median([
        episode["outcomes"]["prefix"]["latency_s"] for episode in episodes
    ]))
    baseline_passes = sum(bool(row["outcomes"]["prefix"]["passed"]) for row in episodes)
    candidates = [
        _cross_validate(
            episodes,
            alpha=alpha,
            uncertainty_beta=beta,
            margin=margin,
            reference_tokens=reference_tokens,
            reference_latency_s=reference_latency_s,
            budget_reserve_quantile=budget_reserve_quantile,
        )
        for alpha in (10.0, 100.0)
        for beta in (0.5, 1.0, 2.0)
        for margin in (0.0, 0.01)
    ]
    feasible = [row for row in candidates if row["passes"] >= baseline_passes]
    selected = max(
        feasible or candidates,
        key=lambda row: (
            row["passes"], row["mean_reward"], row["mean_partial_score"],
            -row["mean_total_tokens"], -row["mean_latency_s"],
            row["uncertainty_beta"], row["margin"], row["alpha"],
        ),
    )
    spec = _feature_spec([
        episode["states"][phase] for episode in episodes for phase in PHASE_ACTIONS
    ])
    ensemble = _fit_ensemble(
        episodes,
        spec=spec,
        alpha=float(selected["alpha"]),
        ensemble_size=64,
        seed=20260827,
        reference_tokens=reference_tokens,
        reference_latency_s=reference_latency_s,
    )
    budget_reserve_tokens = _budget_reserves(
        episodes, quantile=budget_reserve_quantile
    )
    policy: dict[str, Any] = {
        "schema_version": POLICY_SCHEMA,
        "claim_class": "trained_offline_sequential_pessimistic_q_candidate",
        "action_space": list(ACTION_SPACE),
        "phase_actions": {phase: list(actions) for phase, actions in PHASE_ACTIONS.items()},
        "post_recovery_review_mode": "deterministic_rollback_to_prefix",
        "reward": {
            "formula": (
                "partial_score + 0.25*passed - 0.05*log1p(tokens/ref_tokens) "
                "- 0.02*log1p(latency/ref_latency) - 0.25*invocation_error "
                "- 0.50*safety_violation"
            ),
            "reference_tokens": reference_tokens,
            "reference_latency_s": reference_latency_s,
        },
        "training": {
            "algorithm": (
                "paired finite-horizon fitted advantage with bootstrap Q ensemble "
                "and lower-confidence-bound action gating"
            ),
            "dataset_sha256": _canonical_sha256(dataset),
            "episodes": len(episodes),
            "seeds": sorted({int(row["seed"]) for row in episodes}),
            "feature_leakage_boundary": (
                "task_id, full specification, expected output, hidden grader output "
                "and labels excluded; public brief is represented only by signed hash bins"
            ),
            "model_selection": (
                "leave-one-seed-out when multiple seeds exist; no-pass-regression then "
                "maximize pass count and cost-aware reward"
            ),
            "budget_reserve_quantile": budget_reserve_quantile,
            "selected_cross_validation": selected,
            "all_cross_validation": [
                {key: value for key, value in row.items() if key != "rows"}
                for row in candidates
            ],
        },
        "feature_spec": spec,
        "alpha": float(selected["alpha"]),
        "uncertainty_beta": float(selected["uncertainty_beta"]),
        "conservative_margin": float(selected["margin"]),
        "budget_reserve_tokens": budget_reserve_tokens,
        "ensemble_size": 64,
        "advantage_ensemble": ensemble,
    }
    policy["policy_sha256"] = _canonical_sha256(policy)
    return policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--dataset-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--budget-reserve-quantile", type=float, default=0.75
    )
    args = parser.parse_args()
    if bool(args.run_dir) == bool(args.dataset):
        parser.error("provide either one or more --run-dir values or --dataset")
    if args.run_dir:
        dataset = build_dataset([path.resolve() for path in args.run_dir])
        if args.dataset_output is None:
            parser.error("--run-dir requires --dataset-output")
        write_json(args.dataset_output.resolve(), dataset)
    else:
        dataset = read_json(args.dataset.resolve())
    policy = train_policy(
        dataset, budget_reserve_quantile=args.budget_reserve_quantile
    )
    write_json(args.output.resolve(), policy)
    selected = policy["training"]["selected_cross_validation"]
    print(json.dumps({
        "policy_sha256": policy["policy_sha256"],
        "episodes": policy["training"]["episodes"],
        "seeds": policy["training"]["seeds"],
        "alpha": policy["alpha"],
        "uncertainty_beta": policy["uncertainty_beta"],
        "conservative_margin": policy["conservative_margin"],
        "cross_validation": {
            key: selected[key] for key in (
                "passes", "mean_partial_score", "mean_total_tokens",
                "mean_latency_s", "mean_reward", "routes",
            )
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
