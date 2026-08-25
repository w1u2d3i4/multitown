from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .common import read_json, write_json

POLICY_SCHEMA = "multitown-teambench-fitted-q-v1"
DATASET_SCHEMA = "multitown-teambench-counterfactual-v1"
ACTION_SPACE = ("stop", "delegate", "escalate", "review", "human/abstain")
PHASE_ACTIONS = {
    "post_execution": ("stop", "escalate", "human/abstain"),
    "post_replan": ("stop", "delegate", "human/abstain"),
    "post_recovery": ("stop", "review", "human/abstain"),
}
NUMERIC_FEATURES = (
    "workspace_changed",
    "successful_commands",
    "failed_commands",
    "timed_out_commands",
    "max_failed_command_repetitions",
    "turns_used",
    "turn_budget_exhausted",
    "reliability_score",
    "hard_fail",
    "plan_delivered",
    "plan_retry_used",
    "recovery_plan_delivered",
    "consumed_tokens",
    "remaining_token_budget",
    "consumed_latency_s",
)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def outcome_reward(
    outcome: dict[str, Any],
    *,
    reference_tokens: float,
    reference_latency_s: float,
    reward_config: dict[str, float],
) -> float:
    token_ratio = float(outcome["total_tokens"]) / max(reference_tokens, 1.0)
    latency_ratio = float(outcome["latency_s"]) / max(reference_latency_s, 1e-6)
    return (
        float(outcome["partial_score"])
        + reward_config["pass_bonus"] * float(bool(outcome["passed"]))
        - reward_config["token_penalty"] * math.log1p(token_ratio)
        - reward_config["latency_penalty"] * math.log1p(latency_ratio)
        - reward_config["invocation_error_penalty"]
        * float(bool(outcome.get("invocation_error", False)))
        - reward_config["safety_penalty"]
        * float(bool(outcome.get("safety_violation", False)))
        - reward_config["human_penalty"]
        * float(bool(outcome.get("human_abstain", False)))
    )


def _state_values(state: dict[str, Any]) -> dict[str, Any]:
    validator = state.get("validator") or {}
    values = {
        "category": str(state.get("category") or validator.get("category") or "Other"),
        "difficulty": str(
            state.get("difficulty") or validator.get("difficulty") or "unknown"
        ),
    }
    for name in NUMERIC_FEATURES:
        raw = state.get(name, validator.get(name, 0.0))
        values[name] = float(bool(raw)) if isinstance(raw, bool) else float(raw or 0.0)
    return values


def _feature_spec(states: list[dict[str, Any]]) -> dict[str, Any]:
    values = [_state_values(state) for state in states]
    categories = sorted({str(value["category"]) for value in values})
    difficulties = sorted({str(value["difficulty"]) for value in values})
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    for name in NUMERIC_FEATURES:
        column = np.asarray([float(value[name]) for value in values], dtype=float)
        means[name] = float(column.mean())
        scale = float(column.std())
        scales[name] = scale if scale > 1e-9 else 1.0
    return {
        "numeric_features": list(NUMERIC_FEATURES),
        "numeric_means": means,
        "numeric_scales": scales,
        "categories": categories,
        "difficulties": difficulties,
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


def _human_outcome(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "passed": False,
        "partial_score": 0.0,
        "total_tokens": int(state.get("consumed_tokens", 0)),
        "latency_s": float(state.get("consumed_latency_s", 0.0)),
        "invocation_error": False,
        "safety_violation": False,
        "human_abstain": True,
    }


def _with_cost(outcome: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    return {
        **outcome,
        "total_tokens": int(state.get("consumed_tokens", outcome["total_tokens"])),
        "latency_s": float(state.get("consumed_latency_s", outcome["latency_s"])),
    }


def _episode_targets(
    episode: dict[str, Any],
    *,
    reference_tokens: float,
    reference_latency_s: float,
    reward_config: dict[str, float],
) -> dict[str, dict[str, float]]:
    states = episode["states"]
    outcomes = episode["outcomes"]

    def reward(outcome: dict[str, Any]) -> float:
        return outcome_reward(
            outcome,
            reference_tokens=reference_tokens,
            reference_latency_s=reference_latency_s,
            reward_config=reward_config,
        )

    post_recovery = {
        "stop": reward(outcomes["recovery"]),
        "review": reward(outcomes["reviewed"]),
        "human/abstain": reward(_human_outcome(states["post_recovery"])),
    }
    post_replan_stop = _with_cost(outcomes["prefix"], states["post_replan"])
    post_replan = {
        "stop": reward(post_replan_stop),
        "delegate": max(post_recovery.values()),
        "human/abstain": reward(_human_outcome(states["post_replan"])),
    }
    post_execution = {
        "stop": reward(outcomes["prefix"]),
        "escalate": max(post_replan.values()),
        "human/abstain": reward(_human_outcome(states["post_execution"])),
    }
    return {
        "post_execution": post_execution,
        "post_replan": post_replan,
        "post_recovery": post_recovery,
    }


def _fit_models(
    episodes: list[dict[str, Any]],
    *,
    alpha: float,
    reward_config: dict[str, float],
    reference_tokens: float,
    reference_latency_s: float,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    states = [
        episode["states"][phase]
        for episode in episodes
        for phase in PHASE_ACTIONS
    ]
    spec = _feature_spec(states)
    targets = [
        _episode_targets(
            episode,
            reference_tokens=reference_tokens,
            reference_latency_s=reference_latency_s,
            reward_config=reward_config,
        )
        for episode in episodes
    ]
    models: dict[str, dict[str, np.ndarray]] = {}
    for phase, actions in PHASE_ACTIONS.items():
        x = np.stack([encode_state(episode["states"][phase], spec) for episode in episodes])
        models[phase] = {}
        for action in actions:
            y = np.asarray([target[phase][action] for target in targets], dtype=float)
            models[phase][action] = _ridge(x, y, alpha)
    return spec, models


def _predict(
    state: dict[str, Any],
    *,
    phase: str,
    spec: dict[str, Any],
    models: dict[str, dict[str, np.ndarray]],
) -> dict[str, float]:
    x = encode_state(state, spec)
    return {action: float(x @ weights) for action, weights in models[phase].items()}


def _choose(scores: dict[str, float], margin: float) -> str:
    baseline = float(scores["stop"])
    best = max(scores, key=lambda action: (scores[action], action == "stop"))
    if best != "stop" and float(scores[best]) < baseline + margin:
        return "stop"
    return best


def select_action(policy: dict[str, Any], phase: str, state: dict[str, Any]) -> dict[str, Any]:
    validate_policy(policy)
    if phase not in PHASE_ACTIONS:
        raise ValueError(f"unsupported policy phase: {phase}")
    models = {
        model_phase: {
            action: np.asarray(weights, dtype=float)
            for action, weights in action_models.items()
        }
        for model_phase, action_models in policy["q_weights"].items()
    }
    scores = _predict(
        state,
        phase=phase,
        spec=policy["feature_spec"],
        models=models,
    )
    action = _choose(scores, float(policy["conservative_margin"]))
    return {
        "schema_version": "multitown-agentic-rl-decision-v1",
        "policy_source": "trained_offline_fitted_q",
        "policy_sha256": policy["policy_sha256"],
        "phase": phase,
        "action_space": list(ACTION_SPACE),
        "valid_actions": list(PHASE_ACTIONS[phase]),
        "state": state,
        "q_values": scores,
        "action": action,
    }


def validate_policy(policy: dict[str, Any]) -> None:
    if policy.get("schema_version") != POLICY_SCHEMA:
        raise ValueError("unsupported Agentic RL policy schema")
    supplied_hash = policy.get("policy_sha256")
    unhashed = {key: value for key, value in policy.items() if key != "policy_sha256"}
    if supplied_hash != _canonical_sha256(unhashed):
        raise ValueError("Agentic RL policy hash mismatch")
    if tuple(policy.get("action_space", [])) != ACTION_SPACE:
        raise ValueError("Agentic RL policy action space mismatch")
    if {
        phase: tuple(actions)
        for phase, actions in policy.get("phase_actions", {}).items()
    } != PHASE_ACTIONS:
        raise ValueError("Agentic RL policy phase actions mismatch")
    feature_count = (
        1
        + len(policy["feature_spec"]["numeric_features"])
        + len(policy["feature_spec"]["categories"])
        + len(policy["feature_spec"]["difficulties"])
    )
    for phase, actions in PHASE_ACTIONS.items():
        phase_weights = policy.get("q_weights", {}).get(phase, {})
        if set(phase_weights) != set(actions):
            raise ValueError(f"Agentic RL policy weights missing for {phase}")
        if any(len(weights) != feature_count for weights in phase_weights.values()):
            raise ValueError(f"Agentic RL policy feature width mismatch for {phase}")


def _selected_outcome(episode: dict[str, Any], actions: list[str]) -> dict[str, Any]:
    states = episode["states"]
    outcomes = episode["outcomes"]
    first = actions[0]
    if first == "stop":
        return outcomes["prefix"]
    if first == "human/abstain":
        return _human_outcome(states["post_execution"])
    second = actions[1]
    if second == "stop":
        return _with_cost(outcomes["prefix"], states["post_replan"])
    if second == "human/abstain":
        return _human_outcome(states["post_replan"])
    third = actions[2]
    if third == "stop":
        return outcomes["recovery"]
    if third == "review":
        return outcomes["reviewed"]
    return _human_outcome(states["post_recovery"])


def _rollout_model(
    episode: dict[str, Any],
    *,
    spec: dict[str, Any],
    models: dict[str, dict[str, np.ndarray]],
    margin: float,
) -> tuple[list[str], dict[str, Any]]:
    actions: list[str] = []
    for phase in PHASE_ACTIONS:
        scores = _predict(
            episode["states"][phase], phase=phase, spec=spec, models=models
        )
        action = _choose(scores, margin)
        actions.append(action)
        if action in {"stop", "human/abstain"}:
            break
    return actions, _selected_outcome(episode, actions)


def _cross_validate(
    episodes: list[dict[str, Any]],
    *,
    alpha: float,
    margin: float,
    reward_config: dict[str, float],
    reference_tokens: float,
    reference_latency_s: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index, held_out in enumerate(episodes):
        training = episodes[:index] + episodes[index + 1 :]
        spec, models = _fit_models(
            training,
            alpha=alpha,
            reward_config=reward_config,
            reference_tokens=reference_tokens,
            reference_latency_s=reference_latency_s,
        )
        actions, outcome = _rollout_model(
            held_out, spec=spec, models=models, margin=margin
        )
        rows.append({
            "task_id": held_out["task_id"],
            "actions": actions,
            **outcome,
            "reward": outcome_reward(
                outcome,
                reference_tokens=reference_tokens,
                reference_latency_s=reference_latency_s,
                reward_config=reward_config,
            ),
        })
    return {
        "alpha": alpha,
        "margin": margin,
        "passes": sum(bool(row["passed"]) for row in rows),
        "mean_partial_score": _mean([float(row["partial_score"]) for row in rows]),
        "mean_total_tokens": _mean([float(row["total_tokens"]) for row in rows]),
        "mean_latency_s": _mean([float(row["latency_s"]) for row in rows]),
        "mean_reward": _mean([float(row["reward"]) for row in rows]),
        "routes": dict(Counter("/".join(row["actions"]) for row in rows)),
        "rows": rows,
    }


def validate_dataset(value: dict[str, Any]) -> list[dict[str, Any]]:
    if value.get("schema_version") != DATASET_SCHEMA:
        raise ValueError("unsupported counterfactual dataset schema")
    episodes = value.get("episodes")
    if not isinstance(episodes, list) or len(episodes) < 3:
        raise ValueError("counterfactual dataset needs at least three episodes")
    task_ids = [str(episode.get("task_id")) for episode in episodes]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("counterfactual dataset contains duplicate task IDs")
    for episode in episodes:
        if set(episode.get("states", {})) != set(PHASE_ACTIONS):
            raise ValueError(f"incomplete states for {episode.get('task_id')}")
        if set(episode.get("outcomes", {})) != {"prefix", "recovery", "reviewed"}:
            raise ValueError(f"incomplete outcomes for {episode.get('task_id')}")
    return episodes


def build_dataset(
    rows: list[dict[str, Any]], *, source: dict[str, Any]
) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    skipped: dict[str, str] = {}
    for row in rows:
        task_id = str(row.get("task_id"))
        states = row.get("training_states") or {}
        prefix = row.get("shadow_plan_execute")
        recovery = row.get("shadow_recovery")
        if set(states) != set(PHASE_ACTIONS):
            skipped[task_id] = "incomplete_sequential_states"
            continue
        if not isinstance(prefix, dict) or not isinstance(recovery, dict):
            skipped[task_id] = "missing_counterfactual_grade"
            continue
        reviewed = {
            "passed": bool(row.get("passed", False)),
            "partial_score": float(row.get("partial_score", 0.0)),
            "total_tokens": int(row.get("total_tokens", 0)),
            "latency_s": float(row.get("latency_s", 0.0)),
            "failure_modes": list(row.get("failure_modes", [])),
            "invocation_error": bool(row.get("request_errors", 0)),
            "safety_violation": "grader_timeout"
            in set(row.get("failure_modes", [])),
        }
        episodes.append({
            "task_id": task_id,
            "category": row.get("category"),
            "difficulty": row.get("difficulty"),
            "states": states,
            "outcomes": {
                "prefix": prefix,
                "recovery": recovery,
                "reviewed": reviewed,
            },
            "route": row.get("route"),
        })
    dataset = {
        "schema_version": DATASET_SCHEMA,
        "source": source,
        "episodes": episodes,
        "skipped": skipped,
    }
    validate_dataset(dataset)
    return dataset


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def train_policy(dataset: dict[str, Any]) -> dict[str, Any]:
    episodes = validate_dataset(dataset)
    reward_config = {
        "pass_bonus": 0.25,
        "token_penalty": 0.05,
        "latency_penalty": 0.02,
        "invocation_error_penalty": 0.25,
        "safety_penalty": 0.50,
        "human_penalty": 0.10,
    }
    reference_tokens = float(
        np.median([episode["outcomes"]["prefix"]["total_tokens"] for episode in episodes])
    )
    reference_latency_s = float(
        np.median([episode["outcomes"]["prefix"]["latency_s"] for episode in episodes])
    )
    baseline_passes = sum(
        bool(episode["outcomes"]["prefix"]["passed"]) for episode in episodes
    )
    candidates = [
        _cross_validate(
            episodes,
            alpha=alpha,
            margin=margin,
            reward_config=reward_config,
            reference_tokens=reference_tokens,
            reference_latency_s=reference_latency_s,
        )
        for alpha in (0.1, 1.0, 10.0, 100.0)
        for margin in (0.0, 0.01, 0.02, 0.05)
    ]
    feasible = [row for row in candidates if row["passes"] >= baseline_passes]
    pool = feasible or candidates
    selected = max(
        pool,
        key=lambda row: (
            row["passes"],
            row["mean_reward"],
            row["mean_partial_score"],
            -row["mean_total_tokens"],
            -row["mean_latency_s"],
            row["margin"],
            row["alpha"],
        ),
    )
    spec, models = _fit_models(
        episodes,
        alpha=float(selected["alpha"]),
        reward_config=reward_config,
        reference_tokens=reference_tokens,
        reference_latency_s=reference_latency_s,
    )
    policy: dict[str, Any] = {
        "schema_version": POLICY_SCHEMA,
        "claim_class": "trained_offline_sequential_fitted_q_candidate",
        "action_space": list(ACTION_SPACE),
        "phase_actions": {key: list(value) for key, value in PHASE_ACTIONS.items()},
        "reward": {
            "formula": (
                "partial_score + 0.25*passed - 0.05*log1p(tokens/ref_tokens) "
                "- 0.02*log1p(latency/ref_latency) - 0.25*invocation_error "
                "- 0.50*safety_violation - 0.10*human_abstain"
            ),
            "config": reward_config,
            "reference_tokens": reference_tokens,
            "reference_latency_s": reference_latency_s,
        },
        "training": {
            "algorithm": "finite-horizon fitted Q iteration with ridge regression",
            "dataset_sha256": _canonical_sha256(dataset),
            "episodes": len(episodes),
            "feature_leakage_boundary": (
                "task_id, hidden grader output, expected output and test labels excluded"
            ),
            "model_selection": (
                "leave-one-task-out; maximize pass then reward under no-pass-regression"
            ),
            "candidate_grid": {
                "alpha": [0.1, 1.0, 10.0, 100.0],
                "conservative_margin": [0.0, 0.01, 0.02, 0.05],
            },
            "selected_cross_validation": selected,
            "all_cross_validation": [
                {key: value for key, value in row.items() if key != "rows"}
                for row in candidates
            ],
        },
        "feature_spec": spec,
        "alpha": float(selected["alpha"]),
        "conservative_margin": float(selected["margin"]),
        "q_weights": {
            phase: {action: weights.tolist() for action, weights in phase_models.items()}
            for phase, phase_models in models.items()
        },
    }
    policy["policy_sha256"] = _canonical_sha256(policy)
    return policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", type=Path)
    source.add_argument("--results", type=Path)
    parser.add_argument("--run-config", type=Path)
    parser.add_argument("--dataset-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.results:
        if not args.run_config or not args.dataset_output:
            parser.error("--results requires --run-config and --dataset-output")
        run_config = read_json(args.run_config.resolve())
        dataset = build_dataset(
            _read_jsonl(args.results.resolve()),
            source={
                "results_sha256": hashlib.sha256(
                    args.results.resolve().read_bytes()
                ).hexdigest(),
                "run_config_sha256": hashlib.sha256(
                    args.run_config.resolve().read_bytes()
                ).hexdigest(),
                "sampling_seed": run_config.get("sampling_seed"),
                "task_instances_sha256": run_config.get("task_instances_sha256"),
                "controller_sha256": run_config.get("controller_sha256"),
            },
        )
        write_json(args.dataset_output.resolve(), dataset)
    else:
        dataset = read_json(args.dataset.resolve())
    policy = train_policy(dataset)
    write_json(args.output.resolve(), policy)
    summary = policy["training"]["selected_cross_validation"]
    print(json.dumps({
        "policy_sha256": policy["policy_sha256"],
        "episodes": policy["training"]["episodes"],
        "alpha": policy["alpha"],
        "conservative_margin": policy["conservative_margin"],
        "cross_validation": {
            key: summary[key]
            for key in (
                "passes", "mean_partial_score", "mean_total_tokens",
                "mean_latency_s", "mean_reward", "routes",
            )
        },
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
