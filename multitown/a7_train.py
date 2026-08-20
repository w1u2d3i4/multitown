"""Fit, tune and evaluate A7 on the frozen counterfactual arm matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from .a6_policy import ARM_ORDER
from .a7_policy import POLICY_VERSION, arm_feature_rows, choose_arm, predict_arms
from .counterfactual_runner import load_frozen_bank, read_jsonl
from .masbench_routing import git_state, utc_now, write_json
from .scenarios import Scenario


MODEL_SEED = 20260810
MODEL_NAMES = (
    "family_empirical",
    "logistic_ridge",
    "knn",
    "svm_rbf",
    "mlp",
    "hist_gradient_boosting",
    "extra_trees",
)
DEFAULT_BUDGETS = (300.0, 500.0, 750.0, 1000.0, 1500.0, 2500.0, 5000.0)
DEFAULT_TOKEN_PENALTIES = (0.0, 0.0025, 0.005, 0.01)
DEFAULT_LATENCY_PENALTIES = (0.0, 0.0025)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def scenario_maps(bank_path: Path) -> tuple[dict[str, Scenario], dict[str, str], dict[str, str]]:
    scenarios: dict[str, Scenario] = {}
    splits: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for scenario, split, digest in load_frozen_bank(bank_path):
        scenarios[scenario.scenario_id] = scenario
        splits[scenario.scenario_id] = split
        hashes[scenario.scenario_id] = digest
    return scenarios, splits, hashes


def validate_matrix(
    rows: list[dict[str, Any]],
    *,
    splits: dict[str, str],
    hashes: dict[str, str],
    allow_request_errors: bool = False,
) -> dict[tuple[str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        scenario_id = str(row["scenario_id"])
        arm = str(row["arm"])
        if scenario_id not in splits:
            raise ValueError(f"matrix contains unknown scenario: {scenario_id}")
        if arm not in ARM_ORDER:
            raise ValueError(f"matrix contains unknown arm: {arm}")
        if str(row["split"]) != splits[scenario_id]:
            raise ValueError(f"split mismatch for {scenario_id}")
        if str(row["scenario_sha256"]) != hashes[scenario_id]:
            raise ValueError(f"scenario hash mismatch for {scenario_id}")
        if not allow_request_errors and int(row.get("request_errors", 0)):
            raise ValueError(f"request errors in formal matrix cell: {(scenario_id, arm)}")
        key = (scenario_id, arm)
        if key in indexed:
            raise ValueError(f"duplicate matrix cell: {key}")
        indexed[key] = row
    expected = {(scenario_id, arm) for scenario_id in splits for arm in ARM_ORDER}
    missing = expected - set(indexed)
    extra = set(indexed) - expected
    if missing or extra:
        raise ValueError(f"incomplete matrix: missing={len(missing)}, extra={len(extra)}")
    return indexed


def _statistics(
    members: list[dict[str, Any]], *, smooth_accuracy: bool,
) -> dict[str, float | int]:
    count = len(members)
    correct = sum(bool(row["correct"]) for row in members)
    return {
        "count": count,
        "correct": correct,
        "accuracy": (correct + 1) / (count + 2) if smooth_accuracy else correct / count,
        "tokens": sum(float(row["total_tokens"]) for row in members) / count,
        "latency_s": sum(float(row["decision_latency_s"]) for row in members) / count,
    }


def family_statistics(
    matrix: dict[tuple[str, str], dict[str, Any]],
    *, scenarios: dict[str, Scenario], splits: dict[str, str], train_split: str = "train",
) -> tuple[dict[str, dict[str, float | int]], dict[str, dict[str, float | int]]]:
    by_family: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (scenario_id, arm), row in matrix.items():
        if splits[scenario_id] != train_split:
            continue
        by_family[(scenarios[scenario_id].family, arm)].append(row)
        by_arm[arm].append(row)
    family = {
        f"{name}|{arm}": _statistics(members, smooth_accuracy=True)
        for (name, arm), members in sorted(by_family.items())
    }
    arm = {
        name: _statistics(members, smooth_accuracy=True)
        for name, members in sorted(by_arm.items())
    }
    return family, arm


def _estimator_factories() -> dict[str, tuple[bool, Callable[[], Any], Callable[[], Any]]]:
    return {
        "logistic_ridge": (
            True,
            lambda: LogisticRegression(max_iter=2000, class_weight="balanced", random_state=MODEL_SEED),
            lambda: Ridge(alpha=1.0),
        ),
        "knn": (
            True,
            lambda: KNeighborsClassifier(n_neighbors=31, weights="distance"),
            lambda: KNeighborsRegressor(n_neighbors=31, weights="distance"),
        ),
        "svm_rbf": (
            True,
            lambda: CalibratedClassifierCV(
                SVC(C=1.0, gamma="scale", class_weight="balanced", random_state=MODEL_SEED),
                method="sigmoid", cv=3, ensemble=False,
            ),
            lambda: SVR(C=5.0, gamma="scale", epsilon=0.05),
        ),
        "mlp": (
            True,
            lambda: MLPClassifier(
                hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=500,
                random_state=MODEL_SEED,
            ),
            lambda: MLPRegressor(
                hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=500,
                random_state=MODEL_SEED,
            ),
        ),
        "hist_gradient_boosting": (
            False,
            lambda: HistGradientBoostingClassifier(
                max_iter=200, max_leaf_nodes=31, l2_regularization=0.1,
                random_state=MODEL_SEED,
            ),
            lambda: HistGradientBoostingRegressor(
                max_iter=200, max_leaf_nodes=31, l2_regularization=0.1,
                random_state=MODEL_SEED,
            ),
        ),
        "extra_trees": (
            False,
            lambda: ExtraTreesClassifier(
                n_estimators=256, min_samples_leaf=8, class_weight="balanced",
                n_jobs=-1, random_state=MODEL_SEED,
            ),
            lambda: ExtraTreesRegressor(
                n_estimators=256, min_samples_leaf=8, n_jobs=-1,
                random_state=MODEL_SEED,
            ),
        ),
    }


def fit_bundle(
    *,
    scenarios: dict[str, Scenario],
    splits: dict[str, str],
    matrix: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    train_ids = sorted(scenario_id for scenario_id, split in splits.items() if split == "train")
    raw_features: list[dict[str, float | str]] = []
    success: list[int] = []
    tokens: list[float] = []
    latency: list[float] = []
    for scenario_id in train_ids:
        scenario = scenarios[scenario_id]
        for arm, features in zip(ARM_ORDER, arm_feature_rows(scenario), strict=True):
            row = matrix[(scenario_id, arm)]
            raw_features.append(features)
            success.append(int(bool(row["correct"])))
            tokens.append(float(row["total_tokens"]))
            latency.append(float(row["decision_latency_s"]))
    vectorizer = DictVectorizer(sparse=False, sort=True)
    features = vectorizer.fit_transform(raw_features)
    scaler = StandardScaler().fit(features)
    y_success = np.asarray(success, dtype=int)
    y_tokens = np.log1p(np.asarray(tokens, dtype=float))
    y_latency = np.log1p(np.asarray(latency, dtype=float))
    learned: dict[str, dict[str, Any]] = {}
    for name, (scaled, classifier_factory, regressor_factory) in _estimator_factories().items():
        matrix_features = scaler.transform(features) if scaled else features
        classifier = classifier_factory().fit(matrix_features, y_success)
        token_model = regressor_factory().fit(matrix_features, y_tokens)
        latency_model = regressor_factory().fit(matrix_features, y_latency)
        learned[name] = {
            "scaled": scaled,
            "success": classifier,
            "tokens": token_model,
            "latency": latency_model,
        }
    family, arm = family_statistics(matrix, scenarios=scenarios, splits=splits)
    return {
        "policy_version": POLICY_VERSION,
        "feature_contract": "pre_execution_aggregate_context_v1",
        "vectorizer": vectorizer,
        "scaler": scaler,
        "learned_models": learned,
        "family_statistics": family,
        "arm_statistics": arm,
        "selected_config": None,
        "training_scenario_count": len(train_ids),
        "training_example_count": len(raw_features),
    }


def prediction_cache(
    bundle: dict[str, Any], *, scenarios: dict[str, Scenario], scenario_ids: list[str],
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    return {
        model_name: {
            scenario_id: predict_arms(bundle, scenarios[scenario_id], model_name=model_name)
            for scenario_id in scenario_ids
        }
        for model_name in MODEL_NAMES
    }


def evaluate_config(
    *,
    config: dict[str, Any],
    scenario_ids: list[str],
    cache: dict[str, dict[str, dict[str, dict[str, float]]]],
    matrix: dict[tuple[str, str], dict[str, Any]],
    objective_token_penalty_per_1k: float,
    objective_latency_penalty_per_s: float,
    include_rows: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selections: list[dict[str, Any]] = []
    for scenario_id in scenario_ids:
        choice = choose_arm(
            cache[str(config["model_name"])][scenario_id],
            per_decision_token_budget=float(config["per_decision_token_budget"]),
            token_penalty_per_1k=float(config["token_penalty_per_1k"]),
            latency_penalty_per_s=float(config["latency_penalty_per_s"]),
        )
        observed = matrix[(scenario_id, str(choice["selected_arm"]))]
        selections.append({
            "scenario_id": scenario_id,
            "family": observed["family"],
            "split": observed["split"],
            "selected_arm": choice["selected_arm"],
            "budget_fallback": choice["budget_fallback"],
            "eligible_arms": choice["eligible_arms"],
            "predicted_accuracy": choice["predicted_accuracy"],
            "predicted_tokens": choice["predicted_tokens"],
            "predicted_latency_s": choice["predicted_latency_s"],
            "predicted_utility": choice["predicted_utility"],
            "arm_predictions": choice["predictions"],
            "selected_action": observed["selected_action"],
            "oracle_action": observed["oracle_action"],
            "correct": observed["correct"],
            "valid": observed["valid"],
            "total_tokens": observed["total_tokens"],
            "decision_latency_s": observed["decision_latency_s"],
            "request_errors": observed["request_errors"],
        })
    count = len(selections)
    accuracy = sum(bool(row["correct"]) for row in selections) / count
    mean_tokens = sum(float(row["total_tokens"]) for row in selections) / count
    mean_latency = sum(float(row["decision_latency_s"]) for row in selections) / count
    summary = {
        **config,
        "scenario_count": count,
        "accuracy": accuracy,
        "tokens_per_decision": mean_tokens,
        "latency_mean_s": mean_latency,
        "objective_utility": (
            accuracy
            - objective_token_penalty_per_1k * mean_tokens / 1000.0
            - objective_latency_penalty_per_s * mean_latency
        ),
        "actual_budget_violation_rate": sum(
            float(row["total_tokens"]) > float(config["per_decision_token_budget"])
            for row in selections
        ) / count,
        "budget_fallback_rate": sum(bool(row["budget_fallback"]) for row in selections) / count,
        "selected_arm_counts": dict(sorted(Counter(row["selected_arm"] for row in selections).items())),
    }
    return summary, selections if include_rows else []


def fixed_arm_summaries(
    *, scenario_ids: list[str], matrix: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    summaries = []
    for arm in ARM_ORDER:
        rows = [matrix[(scenario_id, arm)] for scenario_id in scenario_ids]
        summaries.append({
            "arm": arm,
            "scenario_count": len(rows),
            "accuracy": sum(bool(row["correct"]) for row in rows) / len(rows),
            "tokens_per_decision": sum(float(row["total_tokens"]) for row in rows) / len(rows),
            "latency_mean_s": sum(float(row["decision_latency_s"]) for row in rows) / len(rows),
            "request_errors": sum(int(row["request_errors"]) for row in rows),
        })
    return summaries


def oracle_upper_bound(
    *, scenario_ids: list[str], matrix: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    for scenario_id in scenario_ids:
        rows = [matrix[(scenario_id, arm)] for arm in ARM_ORDER]
        correct = [row for row in rows if bool(row["correct"])]
        choices = correct or rows
        selected.append(min(choices, key=lambda row: (float(row["total_tokens"]), ARM_ORDER.index(row["arm"]))))
    return {
        "label": "unattainable_cheapest_correct_arm_or_cheapest_if_all_wrong",
        "scenario_count": len(selected),
        "accuracy": sum(bool(row["correct"]) for row in selected) / len(selected),
        "tokens_per_decision": sum(float(row["total_tokens"]) for row in selected) / len(selected),
        "latency_mean_s": sum(float(row["decision_latency_s"]) for row in selected) / len(selected),
        "selected_arm_counts": dict(sorted(Counter(row["arm"] for row in selected).items())),
    }


def calibration_metrics(
    rows: list[dict[str, Any]],
) -> dict[str, float]:
    observed = np.asarray([int(bool(row["correct"])) for row in rows])
    predicted = np.clip(np.asarray([float(row["predicted_accuracy"]) for row in rows]), 1e-8, 1 - 1e-8)
    return {
        "brier_score": float(brier_score_loss(observed, predicted)),
        "log_loss": float(log_loss(observed, predicted, labels=[0, 1])),
    }


def run(args: argparse.Namespace) -> int:
    project_root = Path(__file__).resolve().parents[1]
    revision, dirty = git_state(project_root)
    bank_path = Path(args.bank).resolve()
    matrix_dir = Path(args.matrix_dir).resolve()
    decision_path = matrix_dir / "decisions.jsonl"
    output = Path(args.output_dir).resolve()
    scenarios, splits, hashes = scenario_maps(bank_path)
    matrix = validate_matrix(
        read_jsonl(decision_path),
        splits=splits,
        hashes=hashes,
        allow_request_errors=args.allow_request_errors,
    )
    train_ids = sorted(key for key, value in splits.items() if value == "train")
    dev_ids = sorted(key for key, value in splits.items() if value == "dev")
    test_ids = sorted(key for key, value in splits.items() if value == "test")
    bundle = fit_bundle(scenarios=scenarios, splits=splits, matrix=matrix)
    dev_cache = prediction_cache(bundle, scenarios=scenarios, scenario_ids=dev_ids)
    leaderboard: list[dict[str, Any]] = []
    configs: list[dict[str, Any]] = []
    for model_name in MODEL_NAMES:
        for budget in DEFAULT_BUDGETS:
            for token_penalty in DEFAULT_TOKEN_PENALTIES:
                for latency_penalty in DEFAULT_LATENCY_PENALTIES:
                    configs.append({
                        "model_name": model_name,
                        "per_decision_token_budget": budget,
                        "token_penalty_per_1k": token_penalty,
                        "latency_penalty_per_s": latency_penalty,
                    })
    for config in configs:
        summary, _ = evaluate_config(
            config=config,
            scenario_ids=dev_ids,
            cache=dev_cache,
            matrix=matrix,
            objective_token_penalty_per_1k=args.objective_token_penalty_per_1k,
            objective_latency_penalty_per_s=args.objective_latency_penalty_per_s,
        )
        leaderboard.append(summary)
    eligible = [row for row in leaderboard if row["tokens_per_decision"] <= args.max_average_tokens]
    if not eligible:
        eligible = leaderboard
    selected_dev = max(
        eligible,
        key=lambda row: (
            row["objective_utility"], row["accuracy"],
            -row["tokens_per_decision"], -row["latency_mean_s"],
            str(row["model_name"]),
        ),
    )
    selected_config = {
        key: selected_dev[key]
        for key in (
            "model_name", "per_decision_token_budget",
            "token_penalty_per_1k", "latency_penalty_per_s",
        )
    }
    bundle["selected_config"] = selected_config
    _, dev_rows = evaluate_config(
        config=selected_config,
        scenario_ids=dev_ids,
        cache=dev_cache,
        matrix=matrix,
        objective_token_penalty_per_1k=args.objective_token_penalty_per_1k,
        objective_latency_penalty_per_s=args.objective_latency_penalty_per_s,
        include_rows=True,
    )
    # Test data are first accessed here, after the complete configuration is frozen by dev.
    test_cache = prediction_cache(bundle, scenarios=scenarios, scenario_ids=test_ids)
    test_summary, test_rows = evaluate_config(
        config=selected_config,
        scenario_ids=test_ids,
        cache=test_cache,
        matrix=matrix,
        objective_token_penalty_per_1k=args.objective_token_penalty_per_1k,
        objective_latency_penalty_per_s=args.objective_latency_penalty_per_s,
        include_rows=True,
    )
    test_summary["calibration"] = calibration_metrics(test_rows)
    output.mkdir(parents=True, exist_ok=False)
    bundle_path = output / "policy.joblib"
    joblib.dump(bundle, bundle_path, compress=3)
    write_jsonl(output / "dev-selections.jsonl", dev_rows)
    write_jsonl(output / "test-selections.jsonl", test_rows)
    ordered_leaderboard = sorted(
        leaderboard,
        key=lambda row: (-row["objective_utility"], -row["accuracy"], row["tokens_per_decision"]),
    )
    policy = {
        "policy_version": POLICY_VERSION,
        "created_at_utc": utc_now(),
        "source_revision": revision,
        "source_dirty_at_start": dirty,
        "split_protocol": {
            "train": "fit vectorizer, scaler, success and cost models",
            "dev": "select model, hard per-decision token budget, token penalty and latency penalty",
            "test": "single post-freeze evaluation; never used for fitting or selection",
            "counts": {"train": len(train_ids), "dev": len(dev_ids), "test": len(test_ids)},
        },
        "leakage_controls": {
            "oracle_visible_to_policy": False,
            "test_outcomes_used_for_fit_or_selection": False,
            "features": "pre-execution aggregate context only",
            "excluded": ["oracle", "correct", "selected action", "scenario ID", "split", "score", "scores", "feasible", "finish", "root_index"],
        },
        "input": {
            "scenario_bank": str(bank_path),
            "scenario_bank_sha256": _sha256(bank_path),
            "matrix_decisions": str(decision_path),
            "matrix_decisions_sha256": _sha256(decision_path),
        },
        "model_candidates": list(MODEL_NAMES),
        "feature_count": len(bundle["vectorizer"].feature_names_),
        "feature_names": list(bundle["vectorizer"].feature_names_),
        "selection_objective": {
            "formula": "accuracy - token_penalty*mean_tokens/1000 - latency_penalty*mean_latency",
            "token_penalty_per_1k": args.objective_token_penalty_per_1k,
            "latency_penalty_per_s": args.objective_latency_penalty_per_s,
            "max_dev_average_tokens": args.max_average_tokens,
        },
        "selected_config": selected_config,
        "selected_dev_result": selected_dev,
        "test_result": test_summary,
        "test_fixed_arms": fixed_arm_summaries(scenario_ids=test_ids, matrix=matrix),
        "test_oracle_upper_bound": oracle_upper_bound(scenario_ids=test_ids, matrix=matrix),
        "dev_leaderboard": ordered_leaderboard,
        "bundle_sha256": _sha256(bundle_path),
    }
    write_json(output / "policy.json", policy)
    manifest = {
        "schema_version": "multitown-a7-training-manifest-v1",
        "created_at_utc": utc_now(),
        "source_revision": revision,
        "source_dirty_at_start": dirty,
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in sorted(output.iterdir()) if path.is_file()
        },
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps({
        "selected_config": selected_config,
        "dev": selected_dev,
        "test": test_summary,
    }, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bank", default="benchmarks/multitown-v0.2-1200/scenario-bank.jsonl")
    parser.add_argument("--max-average-tokens", type=float, default=1500.0)
    parser.add_argument("--objective-token-penalty-per-1k", type=float, default=0.005)
    parser.add_argument("--objective-latency-penalty-per-s", type=float, default=0.0025)
    parser.add_argument("--allow-request-errors", action="store_true")
    return parser


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
