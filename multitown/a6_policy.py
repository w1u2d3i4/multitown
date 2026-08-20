"""Leakage-safe, budget-aware organization policy for A6.

The policy treats A0-A5 as six organization arms.  A scenario's family is the
only routing context.  For each held-out fold it estimates every arm's utility
from the other four folds, so no outcome from the routed scenario can influence
its selected arm.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .report import read_jsonl
from .scenarios import Scenario


POLICY_VERSION = "a6-crossfit-family-utility-v1"
ARM_ORDER = ("A0", "A1", "A2", "A3", "A4", "A5")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scenario_signature(scenario: Scenario) -> tuple[str, str, str, tuple[str, ...]]:
    return scenario.scenario_id, scenario.family, scenario.prompt, scenario.allowed_actions


def _load_and_validate_bank(path: Path, scenarios: list[Scenario]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("scenarios", [])
    expected = [_scenario_signature(item) for item in scenarios]
    actual = [
        (
            str(item["scenario_id"]), str(item["family"]), str(item["prompt"]),
            tuple(str(value) for value in item["allowed_actions"]),
        )
        for item in rows
    ]
    if actual != expected:
        raise RuntimeError(f"Baseline scenario bank does not match A6 bank: {path}")


def _scenario_level(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "scenario_id", "family", "correct", "total_tokens", "decision_latency_s",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Baseline decisions are missing columns: {sorted(missing)}")
    work = frame.copy()
    work["correct"] = work["correct"].astype(float)
    work["total_tokens"] = pd.to_numeric(work["total_tokens"], errors="raise")
    work["decision_latency_s"] = pd.to_numeric(work["decision_latency_s"], errors="raise")
    return work.groupby(["scenario_id", "family"], as_index=False).agg(
        accuracy=("correct", "mean"),
        tokens=("total_tokens", "mean"),
        latency_s=("decision_latency_s", "mean"),
        trials=("correct", "size"),
    )


def build_crossfit_policy(
    scenarios: Iterable[Scenario], *, baseline_dirs: dict[str, str | Path],
    folds: int = 5, token_penalty_per_1k: float = 0.005,
    latency_penalty_per_s: float = 0.0025,
) -> dict[str, Any]:
    """Build a deterministic cross-fitted full-information contextual policy."""
    scenario_list = list(scenarios)
    if folds < 2:
        raise ValueError("folds must be at least 2")
    if set(baseline_dirs) != set(ARM_ORDER):
        raise ValueError(f"baseline_dirs must contain exactly {ARM_ORDER}")
    if len({item.scenario_id for item in scenario_list}) != len(scenario_list):
        raise RuntimeError("A6 scenario IDs are not unique")

    fold_by_scenario = {
        scenario.scenario_id: index % folds for index, scenario in enumerate(scenario_list)
    }
    family_by_scenario = {scenario.scenario_id: scenario.family for scenario in scenario_list}
    source_manifest: dict[str, Any] = {}
    arm_stats: dict[str, pd.DataFrame] = {}
    expected_ids = set(fold_by_scenario)
    for arm in ARM_ORDER:
        run_dir = Path(baseline_dirs[arm]).resolve()
        decision_path = run_dir / "decisions.jsonl"
        bank_path = run_dir / "scenario_bank.json"
        status_path = run_dir / "status.json"
        for required in (decision_path, bank_path, status_path):
            if not required.exists():
                raise FileNotFoundError(required)
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("state") != "complete":
            raise RuntimeError(f"Baseline {arm} is not complete: {status_path}")
        _load_and_validate_bank(bank_path, scenario_list)
        frame = read_jsonl(decision_path)
        stats = _scenario_level(frame)
        if set(stats["scenario_id"]) != expected_ids:
            raise RuntimeError(f"Baseline {arm} does not cover the A6 scenario IDs")
        stats["fold"] = stats["scenario_id"].map(fold_by_scenario)
        arm_stats[arm] = stats
        source_manifest[arm] = {
            "run_dir": str(run_dir),
            "decisions": int(len(frame)),
            "scenario_ids": int(stats["scenario_id"].nunique()),
            "decisions_sha256": sha256_file(decision_path),
            "scenario_bank_sha256": sha256_file(bank_path),
            "status_elapsed_s": float(status.get("elapsed_s", 0)),
        }

    families = sorted(set(family_by_scenario.values()))
    fold_tables: dict[str, Any] = {}
    selections: dict[str, Any] = {}
    heldout_rows: list[dict[str, Any]] = []
    for fold in range(folds):
        fold_key = str(fold)
        fold_tables[fold_key] = {}
        for family in families:
            candidates: dict[str, Any] = {}
            for arm in ARM_ORDER:
                training = arm_stats[arm]
                training = training[(training["fold"] != fold) & (training["family"] == family)]
                if training.empty:
                    raise RuntimeError(f"No training rows for fold={fold}, family={family}, arm={arm}")
                rewards = (
                    training["accuracy"]
                    - token_penalty_per_1k * training["tokens"] / 1000.0
                    - latency_penalty_per_s * training["latency_s"]
                )
                candidates[arm] = {
                    "predicted_accuracy": float(training["accuracy"].mean()),
                    "predicted_tokens": float(training["tokens"].mean()),
                    "predicted_latency_s": float(training["latency_s"].mean()),
                    "predicted_utility": float(rewards.mean()),
                    "training_scenarios": int(len(training)),
                    "training_trials": int(training["trials"].sum()),
                }
            selected_arm = max(
                ARM_ORDER,
                key=lambda arm: (
                    candidates[arm]["predicted_utility"],
                    -candidates[arm]["predicted_tokens"],
                    -candidates[arm]["predicted_latency_s"],
                    -ARM_ORDER.index(arm),
                ),
            )
            fold_tables[fold_key][family] = {
                "selected_arm": selected_arm,
                "arms": candidates,
            }

        for scenario in scenario_list:
            if fold_by_scenario[scenario.scenario_id] != fold:
                continue
            choice = fold_tables[fold_key][scenario.family]
            selected_arm = choice["selected_arm"]
            predicted = choice["arms"][selected_arm]
            selection = {
                "scenario_id": scenario.scenario_id,
                "family": scenario.family,
                "fold": fold,
                "selected_arm": selected_arm,
                "predicted_accuracy": predicted["predicted_accuracy"],
                "predicted_tokens": predicted["predicted_tokens"],
                "predicted_latency_s": predicted["predicted_latency_s"],
                "predicted_utility": predicted["predicted_utility"],
                "arm_scores": {
                    arm: values["predicted_utility"] for arm, values in choice["arms"].items()
                },
            }
            selections[scenario.scenario_id] = selection
            observed = arm_stats[selected_arm]
            observed = observed[observed["scenario_id"] == scenario.scenario_id].iloc[0]
            heldout_rows.append({
                **selection,
                "heldout_baseline_accuracy": float(observed["accuracy"]),
                "heldout_baseline_tokens": float(observed["tokens"]),
                "heldout_baseline_latency_s": float(observed["latency_s"]),
                "heldout_baseline_utility": float(
                    observed["accuracy"]
                    - token_penalty_per_1k * observed["tokens"] / 1000.0
                    - latency_penalty_per_s * observed["latency_s"]
                ),
            })

    heldout = pd.DataFrame(heldout_rows)
    arm_counts = {arm: int((heldout["selected_arm"] == arm).sum()) for arm in ARM_ORDER}
    return {
        "policy_version": POLICY_VERSION,
        "policy_class": "five_fold_cross_fitted_full_information_contextual_utility_router",
        "context_features": ["scenario_family"],
        "arms": list(ARM_ORDER),
        "folds": folds,
        "fold_assignment": "stable scenario-bank index modulo folds",
        "training_rule": "for each held-out fold and family, select the arm with maximum mean utility on the other folds",
        "utility": {
            "formula": "accuracy - token_penalty_per_1k * tokens/1000 - latency_penalty_per_s * latency_seconds",
            "token_penalty_per_1k": token_penalty_per_1k,
            "latency_penalty_per_s": latency_penalty_per_s,
            "interpretation": "0.5 percentage point per 1K tokens and 0.25 point per second",
        },
        "leakage_controls": {
            "current_scenario_outcomes_used_for_routing": False,
            "current_scenario_oracle_used_for_routing": False,
            "self_reported_confidence_used_for_routing": False,
            "test_fold_size": int(len(scenario_list) / folds),
            "train_fold_size": int(len(scenario_list) - len(scenario_list) / folds),
        },
        "source_manifest": source_manifest,
        "fold_tables": fold_tables,
        "selections": selections,
        "crossfit_offline_estimate": {
            "scenario_count": int(len(heldout)),
            "accuracy": float(heldout["heldout_baseline_accuracy"].mean()),
            "tokens_per_decision": float(heldout["heldout_baseline_tokens"].mean()),
            "latency_mean_s": float(heldout["heldout_baseline_latency_s"].mean()),
            "utility": float(heldout["heldout_baseline_utility"].mean()),
            "selected_arm_counts": arm_counts,
        },
    }


def choice_for(policy: dict[str, Any], scenario_id: str) -> dict[str, Any]:
    try:
        return dict(policy["selections"][scenario_id])
    except KeyError as exc:
        raise KeyError(f"A6 policy has no selection for {scenario_id}") from exc
