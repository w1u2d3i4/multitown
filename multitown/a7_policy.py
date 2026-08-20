"""Leakage-safe pre-execution features and policy execution for A7."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .a6_policy import ARM_ORDER
from .scenarios import Scenario


POLICY_VERSION = "multitown-a7-contextual-router-v1"
DERIVED_METADATA_KEYS = frozenset({"score", "scores", "feasible", "finish", "root_index"})
FORBIDDEN_FEATURE_FRAGMENTS = (
    "oracle",
    "correct",
    "selected_action",
    "scenario_id",
    "scenario_sha256",
    "split",
    "score",
    "scores",
    "feasible",
    "finish",
    "root_index",
)


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value).strip("_")


def _collect_metadata(
    value: Any,
    *,
    path: tuple[str, ...],
    numeric: dict[str, list[float]],
    categorical: dict[str, list[str]],
    list_lengths: dict[str, list[float]],
) -> None:
    if isinstance(value, dict):
        for raw_key, child in sorted(value.items()):
            key = _safe_name(str(raw_key))
            if key in DERIVED_METADATA_KEYS:
                continue
            _collect_metadata(
                child,
                path=(*path, key),
                numeric=numeric,
                categorical=categorical,
                list_lengths=list_lengths,
            )
        return
    if isinstance(value, (list, tuple)):
        list_lengths["_".join(path)].append(float(len(value)))
        for child in value:
            _collect_metadata(
                child,
                path=path,
                numeric=numeric,
                categorical=categorical,
                list_lengths=list_lengths,
            )
        return
    key = "_".join(path)
    if isinstance(value, bool):
        numeric[key].append(float(value))
    elif isinstance(value, (int, float)) and math.isfinite(float(value)):
        numeric[key].append(float(value))
    elif isinstance(value, str):
        categorical[key].append(value)


def safe_context_features(scenario: Scenario) -> dict[str, float | str]:
    """Return aggregate pre-execution context without outcome or derived-oracle fields."""
    features: dict[str, float | str] = {
        "family": scenario.family,
        "prompt_chars": float(len(scenario.prompt)),
        "prompt_lines": float(scenario.prompt.count("\n") + 1),
        "fact_lines": float(scenario.prompt.count("\n- ")),
        "allowed_action_count": float(len(scenario.allowed_actions)),
    }
    numeric: dict[str, list[float]] = defaultdict(list)
    categorical: dict[str, list[str]] = defaultdict(list)
    list_lengths: dict[str, list[float]] = defaultdict(list)
    _collect_metadata(
        scenario.metadata,
        path=("metadata",),
        numeric=numeric,
        categorical=categorical,
        list_lengths=list_lengths,
    )
    for key, values in sorted(numeric.items()):
        array = np.asarray(values, dtype=float)
        prefix = f"{key}_numeric"
        features[f"{prefix}_count"] = float(len(array))
        features[f"{prefix}_mean"] = float(array.mean())
        features[f"{prefix}_std"] = float(array.std())
        features[f"{prefix}_min"] = float(array.min())
        features[f"{prefix}_max"] = float(array.max())
        features[f"{prefix}_range"] = float(array.max() - array.min())
    for key, values in sorted(categorical.items()):
        counts = Counter(values)
        prefix = f"{key}_categorical"
        features[f"{prefix}_count"] = float(len(values))
        features[f"{prefix}_unique"] = float(len(counts))
        features[f"{prefix}_max_frequency"] = float(max(counts.values()))
        probabilities = np.asarray(list(counts.values()), dtype=float) / len(values)
        features[f"{prefix}_entropy"] = float(-(probabilities * np.log(probabilities)).sum())
    for key, values in sorted(list_lengths.items()):
        array = np.asarray(values, dtype=float)
        features[f"{key}_list_count"] = float(len(array))
        features[f"{key}_list_length_mean"] = float(array.mean())
        features[f"{key}_list_length_max"] = float(array.max())
    validate_safe_feature_names(features)
    return features


def arm_feature_rows(scenario: Scenario) -> list[dict[str, float | str]]:
    context = safe_context_features(scenario)
    return [{**context, "organization_arm": arm} for arm in ARM_ORDER]


def validate_safe_feature_names(features: dict[str, Any]) -> None:
    for name in features:
        lowered = name.lower()
        if any(fragment in lowered for fragment in FORBIDDEN_FEATURE_FRAGMENTS):
            raise ValueError(f"forbidden A7 feature name: {name}")


def _family_prediction(
    bundle: dict[str, Any], *, family: str, arm: str,
) -> dict[str, float]:
    key = f"{family}|{arm}"
    row = bundle["family_statistics"].get(key, bundle["arm_statistics"][arm])
    return {
        "predicted_accuracy": float(row["accuracy"]),
        "predicted_tokens": float(row["tokens"]),
        "predicted_latency_s": float(row["latency_s"]),
    }


def predict_arms(
    bundle: dict[str, Any], scenario: Scenario, *, model_name: str,
) -> dict[str, dict[str, float]]:
    if model_name == "family_empirical":
        return {
            arm: _family_prediction(bundle, family=scenario.family, arm=arm)
            for arm in ARM_ORDER
        }
    vectorizer = bundle["vectorizer"]
    rows = arm_feature_rows(scenario)
    matrix = vectorizer.transform(rows)
    pack = bundle["learned_models"][model_name]
    if pack["scaled"]:
        matrix = bundle["scaler"].transform(matrix)
    probabilities = pack["success"].predict_proba(matrix)
    positive_index = list(pack["success"].classes_).index(1)
    success = probabilities[:, positive_index]
    tokens = np.expm1(pack["tokens"].predict(matrix))
    latency = np.expm1(pack["latency"].predict(matrix))
    return {
        arm: {
            "predicted_accuracy": float(np.clip(success[index], 0.0, 1.0)),
            "predicted_tokens": float(max(0.0, tokens[index])),
            "predicted_latency_s": float(max(0.0, latency[index])),
        }
        for index, arm in enumerate(ARM_ORDER)
    }


def choose_arm(
    predictions: dict[str, dict[str, float]],
    *,
    per_decision_token_budget: float,
    token_penalty_per_1k: float,
    latency_penalty_per_s: float,
) -> dict[str, Any]:
    eligible = [
        arm for arm in ARM_ORDER
        if predictions[arm]["predicted_tokens"] <= per_decision_token_budget
    ]
    budget_fallback = False
    if not eligible:
        eligible = [min(ARM_ORDER, key=lambda arm: (predictions[arm]["predicted_tokens"], arm))]
        budget_fallback = True
    scored: dict[str, dict[str, float]] = {}
    for arm in ARM_ORDER:
        row = dict(predictions[arm])
        row["predicted_utility"] = (
            row["predicted_accuracy"]
            - token_penalty_per_1k * row["predicted_tokens"] / 1000.0
            - latency_penalty_per_s * row["predicted_latency_s"]
        )
        scored[arm] = row
    selected = max(
        eligible,
        key=lambda arm: (
            scored[arm]["predicted_utility"],
            scored[arm]["predicted_accuracy"],
            -scored[arm]["predicted_tokens"],
            -ARM_ORDER.index(arm),
        ),
    )
    return {
        "selected_arm": selected,
        "budget_fallback": budget_fallback,
        "eligible_arms": eligible,
        "predictions": scored,
        **scored[selected],
    }


def select_for_scenario(bundle: dict[str, Any], scenario: Scenario) -> dict[str, Any]:
    config = bundle["selected_config"]
    predictions = predict_arms(bundle, scenario, model_name=str(config["model_name"]))
    choice = choose_arm(
        predictions,
        per_decision_token_budget=float(config["per_decision_token_budget"]),
        token_penalty_per_1k=float(config["token_penalty_per_1k"]),
        latency_penalty_per_s=float(config["latency_penalty_per_s"]),
    )
    choice.update({
        "policy_version": POLICY_VERSION,
        "model_name": config["model_name"],
    })
    return choice


def load_bundle(path: Path) -> dict[str, Any]:
    bundle = joblib.load(path)
    if bundle.get("policy_version") != POLICY_VERSION:
        raise ValueError(f"unsupported A7 policy bundle: {bundle.get('policy_version')!r}")
    return bundle
