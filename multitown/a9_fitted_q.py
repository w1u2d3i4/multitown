"""Train and evaluate an offline fitted-Q sequential budget controller."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_extraction import DictVectorizer

from .a7_policy import load_bundle, predict_arms, safe_context_features
from .a7_train import scenario_maps, validate_matrix, write_jsonl
from .a8_controller import validate_candidate
from .counterfactual_runner import read_jsonl
from .masbench_report import bootstrap_difference, exact_mcnemar_p
from .masbench_routing import git_state, utc_now, write_json
from .scenarios import Scenario


POLICY_VERSION = "multitown-a9-offline-fitted-q-v1"
ACTION_ORDER = ("stop", "delegate", "escalate", "review", "human")
TOKEN_PENALTIES = (0.005, 0.05, 0.1, 0.2, 0.4, 0.8)
MIN_LEAVES = (4, 12, 24)
MODEL_SEED = 20260812


@dataclass(frozen=True)
class Candidate:
    source: str
    action: str | None
    correct: bool
    parse_valid: bool
    hard_constraints_pass: bool
    issue_codes: tuple[str, ...]
    tokens: float
    latency_s: float


@dataclass(frozen=True)
class Episode:
    scenario: Scenario
    predicted_a0_accuracy: float
    initial: Candidate
    delegate: Candidate
    escalate: Candidate
    review: Candidate


@dataclass(frozen=True)
class State:
    current: Candidate
    tokens_used: float
    latency_used_s: float
    delegated: bool = False
    escalated: bool = False
    reviewed: bool = False
    weak_disagreement: bool = False


@dataclass(frozen=True)
class RewardConfig:
    token_penalty_per_1k: float
    latency_penalty_per_s: float = 0.0025
    safety_penalty: float = 1.0
    human_penalty: float = 0.05


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _request_index(path: Path, scenario_ids: set[str]) -> dict[tuple[str, str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in read_jsonl(path):
        scenario_id = str(row["scenario_id"])
        if scenario_id not in scenario_ids:
            continue
        key = (scenario_id, str(row["target_arm"]), str(row["role"]))
        if key in indexed:
            raise ValueError(f"duplicate counterfactual request: {key}")
        indexed[key] = row
    return indexed


def _candidate(
    scenario: Scenario,
    *,
    source: str,
    action: str | None,
    correct: bool,
    valid: bool,
    tokens: float,
    latency_s: float,
) -> Candidate:
    validation = validate_candidate(scenario, action)
    return Candidate(
        source=source,
        action=action,
        correct=bool(correct),
        parse_valid=bool(valid) and validation.parse_valid,
        hard_constraints_pass=validation.hard_constraints_pass,
        issue_codes=validation.issue_codes,
        tokens=float(tokens),
        latency_s=float(latency_s),
    )


def build_episodes(
    *,
    scenario_ids: list[str],
    scenarios: dict[str, Scenario],
    matrix: dict[tuple[str, str], dict[str, Any]],
    requests_path: Path,
    a7_bundle: dict[str, Any],
) -> list[Episode]:
    request_rows = _request_index(requests_path, set(scenario_ids))
    model_name = str(a7_bundle["selected_config"]["model_name"])
    episodes: list[Episode] = []
    for scenario_id in scenario_ids:
        scenario = scenarios[scenario_id]
        a0 = matrix[(scenario_id, "A0")]
        a1 = matrix[(scenario_id, "A1")]
        delegate_row = request_rows[(scenario_id, "A2", "weak_vote_member_1")]
        review_row = request_rows[(scenario_id, "A4", "independent_strong_verifier")]
        predictions = predict_arms(a7_bundle, scenario, model_name=model_name)
        episodes.append(Episode(
            scenario=scenario,
            predicted_a0_accuracy=float(predictions["A0"]["predicted_accuracy"]),
            initial=_candidate(
                scenario,
                source="initial_weak",
                action=a0.get("selected_action"),
                correct=bool(a0["correct"]),
                valid=bool(a0["valid"]),
                tokens=float(a0["total_tokens"]),
                latency_s=float(a0["decision_latency_s"]),
            ),
            delegate=_candidate(
                scenario,
                source="weak_delegate",
                action=delegate_row.get("action"),
                correct=bool(delegate_row.get("correct_individual")),
                valid=bool(delegate_row.get("valid")),
                tokens=float(delegate_row["total_tokens"]),
                latency_s=float(delegate_row["latency_s"]),
            ),
            escalate=_candidate(
                scenario,
                source="strong_specialist",
                action=a1.get("selected_action"),
                correct=bool(a1["correct"]),
                valid=bool(a1["valid"]),
                tokens=float(a1["total_tokens"]),
                latency_s=float(a1["decision_latency_s"]),
            ),
            review=_candidate(
                scenario,
                source="independent_reviewer_proxy",
                action=review_row.get("action"),
                correct=bool(review_row.get("correct_individual")),
                valid=bool(review_row.get("valid")),
                tokens=float(review_row["total_tokens"]),
                latency_s=float(review_row["latency_s"]),
            ),
        ))
    return episodes


def initial_state(episode: Episode) -> State:
    return State(
        current=episode.initial,
        tokens_used=episode.initial.tokens,
        latency_used_s=episode.initial.latency_s,
    )


def _next_candidate(episode: Episode, action: str) -> Candidate:
    return {
        "delegate": episode.delegate,
        "escalate": episode.escalate,
        "review": episode.review,
    }[action]


def valid_actions(episode: Episode, state: State, *, token_cap: float) -> tuple[str, ...]:
    actions = ["stop", "human"]
    if not state.delegated and not state.escalated and not state.reviewed:
        if state.tokens_used + episode.delegate.tokens <= token_cap:
            actions.append("delegate")
    if not state.escalated and not state.reviewed:
        if state.tokens_used + episode.escalate.tokens <= token_cap:
            actions.append("escalate")
    if not state.reviewed:
        if state.tokens_used + episode.review.tokens <= token_cap:
            actions.append("review")
    return tuple(action for action in ACTION_ORDER if action in actions)


def transition(episode: Episode, state: State, action: str) -> State:
    if action not in {"delegate", "escalate", "review"}:
        raise ValueError(f"non-transition action: {action}")
    candidate = _next_candidate(episode, action)
    current = candidate
    disagreement = False
    if action == "delegate":
        disagreement = candidate.action != state.current.action
        current = state.current
    return State(
        current=current,
        tokens_used=state.tokens_used + candidate.tokens,
        latency_used_s=state.latency_used_s + candidate.latency_s,
        delegated=state.delegated or action == "delegate",
        escalated=state.escalated or action == "escalate",
        reviewed=state.reviewed or action == "review",
        weak_disagreement=disagreement if action == "delegate" else False,
    )


def safety_violation(state: State) -> bool:
    return (
        not state.current.parse_valid
        or not state.current.hard_constraints_pass
        or state.weak_disagreement
    )


def terminal_reward(state: State, action: str, config: RewardConfig) -> float:
    if action == "human":
        return -config.human_penalty
    if action != "stop":
        raise ValueError(action)
    return float(state.current.correct) - config.safety_penalty * float(safety_violation(state))


def transition_cost(candidate: Candidate, config: RewardConfig) -> float:
    return (
        -config.token_penalty_per_1k * candidate.tokens / 1000.0
        -config.latency_penalty_per_s * candidate.latency_s
    )


def exact_q(
    episode: Episode,
    state: State,
    action: str,
    *,
    config: RewardConfig,
    token_cap: float,
) -> float:
    if action in {"stop", "human"}:
        return terminal_reward(state, action, config)
    next_state = transition(episode, state, action)
    candidate = _next_candidate(episode, action)
    future = max(
        exact_q(episode, next_state, next_action, config=config, token_cap=token_cap)
        for next_action in valid_actions(episode, next_state, token_cap=token_cap)
    )
    return transition_cost(candidate, config) + future


def reachable_states(episode: Episode, *, token_cap: float) -> list[State]:
    pending = [initial_state(episode)]
    result: list[State] = []
    seen: set[tuple[Any, ...]] = set()
    while pending:
        state = pending.pop()
        key = (
            state.current.source, state.tokens_used, state.delegated,
            state.escalated, state.reviewed, state.weak_disagreement,
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(state)
        for action in valid_actions(episode, state, token_cap=token_cap):
            if action not in {"stop", "human"}:
                pending.append(transition(episode, state, action))
    return result


def state_features(episode: Episode, state: State, *, token_cap: float) -> dict[str, float | str]:
    features = dict(safe_context_features(episode.scenario))
    candidate_position = (
        episode.scenario.allowed_actions.index(state.current.action)
        if state.current.action in episode.scenario.allowed_actions else -1
    )
    features.update({
        "controller_stage": (
            "reviewed" if state.reviewed else
            "escalated_after_delegate" if state.escalated and state.delegated else
            "escalated" if state.escalated else
            "delegated" if state.delegated else "initial"
        ),
        "candidate_source": state.current.source,
        "candidate_choice": state.current.action or "none",
        "candidate_position": float(candidate_position),
        "candidate_parse_valid": float(state.current.parse_valid),
        "candidate_hard_constraints_pass": float(state.current.hard_constraints_pass),
        "validator_issue_codes": "|".join(state.current.issue_codes) or "none",
        "weak_disagreement": float(state.weak_disagreement),
        "delegated": float(state.delegated),
        "escalated": float(state.escalated),
        "reviewed": float(state.reviewed),
        "tokens_used": state.tokens_used,
        "token_budget_remaining": token_cap - state.tokens_used,
        "latency_used_s": state.latency_used_s,
        "predicted_a0_accuracy": episode.predicted_a0_accuracy,
    })
    return features


def fit_policy(
    episodes: list[Episode],
    *,
    reward_config: RewardConfig,
    token_cap: float,
    min_samples_leaf: int,
) -> dict[str, Any]:
    feature_rows: list[dict[str, float | str]] = []
    actions: list[str] = []
    targets: list[float] = []
    for episode in episodes:
        for state in reachable_states(episode, token_cap=token_cap):
            features = state_features(episode, state, token_cap=token_cap)
            for action in valid_actions(episode, state, token_cap=token_cap):
                feature_rows.append(features)
                actions.append(action)
                targets.append(exact_q(
                    episode, state, action, config=reward_config, token_cap=token_cap,
                ))
    vectorizer = DictVectorizer(sparse=False, sort=True)
    matrix = vectorizer.fit_transform(feature_rows)
    action_array = np.asarray(actions)
    target_array = np.asarray(targets, dtype=float)
    models: dict[str, ExtraTreesRegressor] = {}
    action_counts: dict[str, int] = {}
    for action in ACTION_ORDER:
        mask = action_array == action
        action_counts[action] = int(mask.sum())
        if not mask.any():
            raise ValueError(f"no training samples for action {action}")
        models[action] = ExtraTreesRegressor(
            n_estimators=128,
            min_samples_leaf=min_samples_leaf,
            n_jobs=-1,
            random_state=MODEL_SEED + ACTION_ORDER.index(action),
        ).fit(matrix[mask], target_array[mask])
    return {
        "policy_version": POLICY_VERSION,
        "training_algorithm": "offline fitted-Q regression with full counterfactual transitions",
        "reward_config": reward_config.__dict__,
        "per_episode_token_cap": token_cap,
        "min_samples_leaf": min_samples_leaf,
        "training_episodes": len(episodes),
        "training_action_samples": action_counts,
        "feature_names": list(vectorizer.get_feature_names_out()),
        "vectorizer": vectorizer,
        "models": models,
    }


def choose_action(policy: dict[str, Any], episode: Episode, state: State) -> tuple[str, dict[str, float]]:
    token_cap = float(policy["per_episode_token_cap"])
    matrix = policy["vectorizer"].transform([
        state_features(episode, state, token_cap=token_cap)
    ])
    q_values = {
        action: float(policy["models"][action].predict(matrix)[0])
        for action in valid_actions(episode, state, token_cap=token_cap)
    }
    selected = max(
        q_values,
        key=lambda action: (q_values[action], -ACTION_ORDER.index(action)),
    )
    return selected, q_values


def evaluate_policy(policy: dict[str, Any], episodes: list[Episode], *, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    config = RewardConfig(**policy["reward_config"])
    for episode in episodes:
        state = initial_state(episode)
        trajectory: list[dict[str, Any]] = []
        while True:
            action, q_values = choose_action(policy, episode, state)
            trajectory.append({
                "turn_id": len(trajectory),
                "action": action,
                "q_values": q_values,
                "tokens_used": state.tokens_used,
                "latency_used_s": state.latency_used_s,
                "validator_pass": state.current.hard_constraints_pass,
                "weak_disagreement": state.weak_disagreement,
            })
            if action in {"stop", "human"}:
                violation = safety_violation(state) if action == "stop" else False
                correct = state.current.correct if action == "stop" else False
                reward = (
                    float(correct)
                    - config.token_penalty_per_1k * state.tokens_used / 1000.0
                    - config.latency_penalty_per_s * state.latency_used_s
                    - config.safety_penalty * float(violation)
                    - config.human_penalty * float(action == "human")
                )
                rows.append({
                    "schema_version": "multitown-a9-offline-evaluation-v1",
                    "scenario_id": episode.scenario.scenario_id,
                    "split": split,
                    "family": episode.scenario.family,
                    "terminal_action": action,
                    "selected_action": state.current.action if action == "stop" else None,
                    "correct": bool(correct),
                    "safety_violation": bool(violation),
                    "human_escalation": action == "human",
                    "total_tokens": state.tokens_used,
                    "decision_latency_s": state.latency_used_s,
                    "reward": reward,
                    "trajectory": trajectory,
                })
                break
            state = transition(episode, state, action)
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    actions = Counter(
        step["action"] for row in rows for step in row["trajectory"]
    )
    routes = Counter("->".join(step["action"] for step in row["trajectory"]) for row in rows)
    return {
        "scenario_count": count,
        "correct": sum(bool(row["correct"]) for row in rows),
        "accuracy": sum(bool(row["correct"]) for row in rows) / count,
        "tokens_per_decision": sum(float(row["total_tokens"]) for row in rows) / count,
        "total_tokens": sum(float(row["total_tokens"]) for row in rows),
        "latency_mean_s": sum(float(row["decision_latency_s"]) for row in rows) / count,
        "safety_violation_rate": sum(bool(row["safety_violation"]) for row in rows) / count,
        "human_escalation_rate": sum(bool(row["human_escalation"]) for row in rows) / count,
        "mean_reward": sum(float(row["reward"]) for row in rows) / count,
        "action_counts": dict(sorted(actions.items())),
        "route_counts": dict(sorted(routes.items())),
    }


def summarize_a8(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    return {
        "scenario_count": count,
        "correct": sum(bool(row["correct"]) for row in rows),
        "accuracy": sum(bool(row["correct"]) for row in rows) / count,
        "tokens_per_decision": sum(float(row["total_tokens"]) for row in rows) / count,
        "total_tokens": sum(float(row["total_tokens"]) for row in rows),
        "latency_mean_s": sum(float(row["decision_latency_s"]) for row in rows) / count,
    }


def paired_a9_a8(a9_rows: list[dict[str, Any]], a8_rows: list[dict[str, Any]]) -> dict[str, Any]:
    a9 = {str(row["scenario_id"]): row for row in a9_rows}
    a8 = {str(row["scenario_id"]): row for row in a8_rows}
    if set(a9) != set(a8):
        raise ValueError("A9 and A8 held-out scenario IDs differ")
    ids = sorted(a9)
    left_correct = np.asarray([float(bool(a9[key]["correct"])) for key in ids])
    right_correct = np.asarray([float(bool(a8[key]["correct"])) for key in ids])
    left_tokens = np.asarray([float(a9[key]["total_tokens"]) for key in ids])
    right_tokens = np.asarray([float(a8[key]["total_tokens"]) for key in ids])
    left_latency = np.asarray([float(a9[key]["decision_latency_s"]) for key in ids])
    right_latency = np.asarray([float(a8[key]["decision_latency_s"]) for key in ids])
    return {
        "left": "A9-offline-fitted-Q",
        "right": "A8-offline-simulation",
        "accuracy": bootstrap_difference(left_correct, right_correct, seed=MODEL_SEED),
        "tokens_per_decision": bootstrap_difference(left_tokens, right_tokens, seed=MODEL_SEED + 1),
        "latency_mean_s": bootstrap_difference(left_latency, right_latency, seed=MODEL_SEED + 2),
        "mcnemar_exact": exact_mcnemar_p(
            [bool(value) for value in left_correct], [bool(value) for value in right_correct]
        ),
        "equal_or_lower_total_token_budget": float(left_tokens.sum()) <= float(right_tokens.sum()),
        "token_budget_ratio": float(left_tokens.sum() / right_tokens.sum()),
    }


def _selection_score(summary: dict[str, Any]) -> tuple[float, ...]:
    common_utility = (
        float(summary["accuracy"])
        - 0.005 * float(summary["tokens_per_decision"]) / 1000.0
        - 0.0025 * float(summary["latency_mean_s"])
        - float(summary["safety_violation_rate"])
        - 0.05 * float(summary["human_escalation_rate"])
    )
    return (
        common_utility,
        float(summary["accuracy"]),
        -float(summary["safety_violation_rate"]),
        -float(summary["tokens_per_decision"]),
    )


def run(args: argparse.Namespace) -> int:
    project_root = Path(__file__).resolve().parents[1]
    revision, dirty = git_state(project_root)
    bank_path = Path(args.bank).resolve()
    matrix_dir = Path(args.matrix_dir).resolve()
    matrix_path = matrix_dir / "decisions.jsonl"
    requests_path = matrix_dir / "requests.jsonl"
    bundle_path = Path(args.a7_policy_bundle).resolve()
    a8_dev_path = Path(args.a8_dev).resolve()
    a8_test_path = Path(args.a8_test).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)

    scenarios, splits, hashes = scenario_maps(bank_path)
    matrix = validate_matrix(read_jsonl(matrix_path), splits=splits, hashes=hashes)
    a7_bundle = load_bundle(bundle_path)
    train_ids = sorted(key for key, value in splits.items() if value == "train")
    dev_ids = sorted(key for key, value in splits.items() if value == "dev")
    train_episodes = build_episodes(
        scenario_ids=train_ids, scenarios=scenarios, matrix=matrix,
        requests_path=requests_path, a7_bundle=a7_bundle,
    )
    dev_episodes = build_episodes(
        scenario_ids=dev_ids, scenarios=scenarios, matrix=matrix,
        requests_path=requests_path, a7_bundle=a7_bundle,
    )
    a8_dev_rows = read_jsonl(a8_dev_path)
    a8_dev_summary = summarize_a8(a8_dev_rows)
    dev_budget = float(a8_dev_summary["tokens_per_decision"]) * args.dev_budget_guard_ratio

    candidates: list[tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]] = []
    for token_penalty in TOKEN_PENALTIES:
        for min_leaf in MIN_LEAVES:
            policy = fit_policy(
                train_episodes,
                reward_config=RewardConfig(token_penalty_per_1k=token_penalty),
                token_cap=args.per_episode_token_cap,
                min_samples_leaf=min_leaf,
            )
            dev_rows = evaluate_policy(policy, dev_episodes, split="dev")
            dev_summary = summarize(dev_rows)
            dev_summary.update({
                "token_penalty_per_1k": token_penalty,
                "min_samples_leaf": min_leaf,
                "within_guarded_a8_dev_budget": dev_summary["tokens_per_decision"] <= dev_budget,
            })
            candidates.append((policy, dev_summary, dev_rows))
    guarded_eligible = [item for item in candidates if item[1]["within_guarded_a8_dev_budget"]]
    exact_budget_eligible = [
        item for item in candidates
        if item[1]["tokens_per_decision"] <= a8_dev_summary["tokens_per_decision"]
    ]
    if guarded_eligible:
        selection_pool = guarded_eligible
        budget_selection_mode = "guarded_a8_dev_budget"
    elif exact_budget_eligible:
        selection_pool = exact_budget_eligible
        budget_selection_mode = "exact_a8_dev_budget_fallback"
    else:
        raise RuntimeError("no A9 candidate satisfies the A8 dev token budget")
    selected_policy, selected_dev, selected_dev_rows = max(
        selection_pool, key=lambda item: _selection_score(item[1])
    )

    # Freeze the selected policy before constructing or reading any held-out A9 episode.
    output.mkdir(parents=True, exist_ok=False)
    joblib.dump(selected_policy, output / "policy.joblib")
    write_jsonl(output / "dev-decisions.jsonl", selected_dev_rows)
    pretest = {
        "schema_version": "multitown-a9-pretest-freeze-v1",
        "created_at_utc": utc_now(),
        "source_revision": revision,
        "source_dirty_at_start": dirty,
        "training_scenarios": len(train_ids),
        "dev_scenarios": len(dev_ids),
        "test_scenarios_used_for_fit_or_selection": 0,
        "algorithm": selected_policy["training_algorithm"],
        "state_contract": (
            "safe scenario context + observed candidate/validator/disagreement + "
            "tokens/latency used + remaining hard budget"
        ),
        "action_contract": list(ACTION_ORDER),
        "reward_contract": selected_policy["reward_config"],
        "per_episode_token_cap": args.per_episode_token_cap,
        "a8_dev_reference": a8_dev_summary,
        "dev_budget_guard_ratio": args.dev_budget_guard_ratio,
        "guarded_dev_budget_tokens_per_decision": dev_budget,
        "budget_selection_mode": budget_selection_mode,
        "selected_dev": selected_dev,
        "candidate_leaderboard": sorted(
            [item[1] for item in candidates], key=_selection_score, reverse=True,
        ),
    }
    write_json(output / "pretest-freeze.json", pretest)

    # Held-out episodes are constructed and evaluated exactly once after the freeze above.
    test_ids = sorted(key for key, value in splits.items() if value == "test")
    test_episodes = build_episodes(
        scenario_ids=test_ids, scenarios=scenarios, matrix=matrix,
        requests_path=requests_path, a7_bundle=a7_bundle,
    )
    test_rows = evaluate_policy(selected_policy, test_episodes, split="test")
    a8_test_rows = read_jsonl(a8_test_path)
    test_summary = summarize(test_rows)
    a8_test_summary = summarize_a8(a8_test_rows)
    comparison = paired_a9_a8(test_rows, a8_test_rows)
    write_jsonl(output / "test-decisions.jsonl", test_rows)
    result = {
        "schema_version": "multitown-a9-offline-result-v1",
        "created_at_utc": utc_now(),
        "source_revision": revision,
        "source_dirty_at_start": dirty,
        "evaluation_scope": "offline counterfactual simulator; not the formal A8 online run",
        "test_access_protocol": "policy frozen on train/dev before one held-out evaluation",
        "a9_test": test_summary,
        "a8_offline_test_reference": a8_test_summary,
        "paired_comparison": comparison,
        "claim_gates": {
            "equal_or_lower_test_token_budget": comparison["equal_or_lower_total_token_budget"],
            "accuracy_ci_excludes_zero": comparison["accuracy"]["ci95_low"] > 0,
            "online_agentic_rl_claim_allowed": False,
        },
        "limitations": [
            "This is trained offline fitted-Q regression over a deterministic counterfactual simulator.",
            "Delegate uses one A2 vote member and review uses the A4 independent-verifier request as proxies.",
            "The review proxy includes verifier context from A4 and is not a fresh online A9 call.",
            "Human is an abstention scored incorrect with a fixed penalty; no human outcome is imputed.",
            "This result must not be called an online Agentic-RL improvement.",
        ],
    }
    write_json(output / "result.json", result)
    results_md = (
        "# A9 offline fitted-Q result\n\n"
        f"A9: {test_summary['correct']}/{test_summary['scenario_count']} "
        f"({100*test_summary['accuracy']:.2f}%), {test_summary['tokens_per_decision']:.1f} "
        f"tokens/decision, {test_summary['latency_mean_s']:.3f} s mean latency.\n\n"
        f"A8 offline reference: {a8_test_summary['correct']}/{a8_test_summary['scenario_count']} "
        f"({100*a8_test_summary['accuracy']:.2f}%), "
        f"{a8_test_summary['tokens_per_decision']:.1f} tokens/decision.\n\n"
        f"Paired accuracy difference: {100*comparison['accuracy']['difference']:.2f} pp "
        f"(95% bootstrap CI {100*comparison['accuracy']['ci95_low']:.2f} to "
        f"{100*comparison['accuracy']['ci95_high']:.2f} pp). Equal/lower total token "
        f"budget: {comparison['equal_or_lower_total_token_budget']}.\n\n"
        "This is an offline counterfactual fitted-Q prototype, not an online Agentic-RL claim.\n"
    )
    (output / "RESULTS.md").write_text(results_md, encoding="utf-8")
    manifest = {
        "schema_version": "multitown-a9-artifact-manifest-v1",
        "inputs": {
            "a9_source_sha256": _sha256(Path(__file__)),
            "bank_sha256": _sha256(bank_path),
            "matrix_decisions_sha256": _sha256(matrix_path),
            "matrix_requests_sha256": _sha256(requests_path),
            "a7_policy_bundle_sha256": _sha256(bundle_path),
            "a8_dev_sha256": _sha256(a8_dev_path),
            "a8_test_sha256": _sha256(a8_test_path),
        },
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in sorted(output.iterdir()) if path.is_file()
        },
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", required=True)
    parser.add_argument("--a7-policy-bundle", required=True)
    parser.add_argument("--a8-dev", required=True)
    parser.add_argument("--a8-test", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bank", default="benchmarks/multitown-v0.2-1200/scenario-bank.jsonl")
    parser.add_argument("--per-episode-token-cap", type=float, default=1300.0)
    parser.add_argument("--dev-budget-guard-ratio", type=float, default=0.95)
    return parser


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
