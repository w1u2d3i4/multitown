"""Frozen A23 fixed-seed statistics and adaptive-development gates."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .a22_runner import _behavior_summary
from .a9_oof_protocol import DEFAULT_FOLDS
from .a9_ppo_oof import (
    BOOTSTRAP_SEED, RunSchedule, _metric_value, _paired_effects,
    fold_cluster_bootstrap, fold_cluster_ratio_bootstrap,
)


A23_STATISTICS_VERSION = "multitown-a23-fixed-seed-adaptive-statistics-v1"


@dataclass(frozen=True)
class A23StatisticsSchedule:
    mode: str
    seeds: tuple[int, ...]
    folds: tuple[int, ...]
    updates: int
    episodes_per_update: int
    outer_episodes_per_fold: int
    bootstrap_iterations: int
    threads: int


def _seeded_index(
    rows: Sequence[Mapping[str, Any]], seeds: Sequence[int], *, label: str,
) -> tuple[dict[tuple[str, int], Mapping[str, Any]], dict[int, list[str]]]:
    seed_tuple = tuple(seeds)
    index = {
        (str(row["episode_id"]), int(row["training_seed"])): row
        for row in rows
    }
    if len(index) != len(rows):
        raise ValueError(f"duplicate {label} rows")
    by_fold: dict[int, set[str]] = defaultdict(set)
    for (episode_id, training_seed), row in index.items():
        if training_seed not in seed_tuple:
            raise ValueError(f"unexpected {label} training seed")
        fold = int(row["outer_fold"])
        if fold not in range(DEFAULT_FOLDS):
            raise ValueError(f"invalid {label} fold")
        by_fold[fold].add(episode_id)
    if set(by_fold) != set(range(DEFAULT_FOLDS)):
        raise ValueError(f"{label} fold coverage mismatch")
    canonical = {fold: sorted(ids) for fold, ids in by_fold.items()}
    expected = {
        (episode_id, seed) for ids in canonical.values()
        for episode_id in ids for seed in seed_tuple
    }
    if set(index) != expected:
        raise ValueError(f"{label} is not an episode x seed product")
    return index, canonical


def fixed_seed_policy_bootstrap(
    left_rows: Sequence[Mapping[str, Any]],
    right_rows: Sequence[Mapping[str, Any]], *, metric: str,
    seeds: Sequence[int], iterations: int, seed: int,
    ratio_reduction: bool = False,
) -> dict[str, Any]:
    """Compare two policies after fixed-seed averaging within every episode."""

    if type(iterations) is not int or iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    seed_tuple = tuple(seeds)
    left_index, left_ids = _seeded_index(left_rows, seed_tuple, label="left policy")
    right_index, right_ids = _seeded_index(
        right_rows, seed_tuple, label="right policy",
    )
    if left_ids != right_ids:
        raise ValueError("policy bootstrap episode products differ")
    arrays: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for fold in range(DEFAULT_FOLDS):
        ids = left_ids[fold]
        left = np.asarray([
            np.mean([
                _metric_value(left_index[(episode_id, training_seed)], metric)
                for training_seed in seed_tuple
            ]) for episode_id in ids
        ], dtype=np.float64)
        right = np.asarray([
            np.mean([
                _metric_value(right_index[(episode_id, training_seed)], metric)
                for training_seed in seed_tuple
            ]) for episode_id in ids
        ], dtype=np.float64)
        arrays[fold] = left, right
    left_point = float(np.mean([item[0].mean() for item in arrays.values()]))
    right_point = float(np.mean([item[1].mean() for item in arrays.values()]))
    if ratio_reduction and right_point == 0.0:
        raise ValueError("undefined policy ratio denominator")
    point = (
        1.0 - left_point / right_point
        if ratio_reduction else left_point - right_point
    )
    rng = np.random.default_rng(seed)
    values = np.empty(iterations, dtype=np.float64)
    for start in range(0, iterations, 256):
        width = min(256, iterations - start)
        left_replicates = np.zeros(width, dtype=np.float64)
        right_replicates = np.zeros(width, dtype=np.float64)
        for fold in range(DEFAULT_FOLDS):
            left, right = arrays[fold]
            indices = rng.integers(0, len(left), size=(width, len(left)))
            left_replicates += left[indices].mean(axis=1) / DEFAULT_FOLDS
            right_replicates += right[indices].mean(axis=1) / DEFAULT_FOLDS
        if ratio_reduction:
            if np.any(right_replicates == 0.0):
                raise ValueError("undefined bootstrap policy ratio denominator")
            values[start:start + width] = 1.0 - left_replicates / right_replicates
        else:
            values[start:start + width] = left_replicates - right_replicates
    return {
        "schema_version": A23_STATISTICS_VERSION,
        "metric": metric,
        "estimand": (
            "A23_over_A22_ratio_reduction"
            if ratio_reduction else "A23_minus_A22"
        ),
        "point": float(point), "a23_mean": left_point,
        "a22_mean": right_point,
        "ci95_low": float(np.quantile(values, 0.025, method="linear")),
        "ci95_high": float(np.quantile(values, 0.975, method="linear")),
        "iterations": iterations, "rng_seed": seed,
        "percentile_interval": [0.025, 0.975],
        "numpy_quantile_method": "linear",
        "fold_resampling": {
            str(fold): len(arrays[fold][0]) for fold in range(DEFAULT_FOLDS)
        },
        "training_seeds_fixed_not_resampled": list(seed_tuple),
    }


def fixed_seed_policy_ratio_bootstrap(
    left_rows: Sequence[Mapping[str, Any]],
    right_rows: Sequence[Mapping[str, Any]], *, numerator: str,
    denominator: str, seeds: Sequence[int], iterations: int, seed: int,
) -> dict[str, Any]:
    """Compare equal-fold aggregate ratios and recompute each replicate."""

    if type(iterations) is not int or iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    seed_tuple = tuple(seeds)
    left_index, left_ids = _seeded_index(left_rows, seed_tuple, label="left ratio")
    right_index, right_ids = _seeded_index(
        right_rows, seed_tuple, label="right ratio",
    )
    if left_ids != right_ids:
        raise ValueError("policy ratio episode products differ")
    arrays: dict[int, tuple[np.ndarray, ...]] = {}
    for fold in range(DEFAULT_FOLDS):
        ids = left_ids[fold]
        arrays[fold] = tuple(np.asarray([
            np.mean([
                float(index[(episode_id, training_seed)][field])
                for training_seed in seed_tuple
            ]) for episode_id in ids
        ], dtype=np.float64) for index, field in (
            (left_index, numerator), (left_index, denominator),
            (right_index, numerator), (right_index, denominator),
        ))
    if any(item[1].sum() == 0.0 or item[3].sum() == 0.0 for item in arrays.values()):
        raise ValueError("policy ratio point denominator is zero")
    left_point = float(np.mean([
        item[0].sum() / item[1].sum() for item in arrays.values()
    ]))
    right_point = float(np.mean([
        item[2].sum() / item[3].sum() for item in arrays.values()
    ]))
    rng = np.random.default_rng(seed)
    values = np.empty(iterations, dtype=np.float64)
    for start in range(0, iterations, 256):
        width = min(256, iterations - start)
        differences = np.zeros(width, dtype=np.float64)
        for fold in range(DEFAULT_FOLDS):
            item = arrays[fold]
            indices = rng.integers(0, len(item[0]), size=(width, len(item[0])))
            sampled = [array[indices].sum(axis=1) for array in item]
            if np.any(sampled[1] == 0.0) or np.any(sampled[3] == 0.0):
                raise ValueError("policy ratio bootstrap denominator is zero")
            differences += (
                sampled[0] / sampled[1] - sampled[2] / sampled[3]
            ) / DEFAULT_FOLDS
        values[start:start + width] = differences
    return {
        "schema_version": A23_STATISTICS_VERSION,
        "metric": f"sum({numerator})/sum({denominator})",
        "estimand": "A23_minus_A22", "point": left_point - right_point,
        "a23_ratio": left_point, "a22_ratio": right_point,
        "ci95_low": float(np.quantile(values, 0.025, method="linear")),
        "ci95_high": float(np.quantile(values, 0.975, method="linear")),
        "iterations": iterations, "rng_seed": seed,
        "percentile_interval": [0.025, 0.975],
        "numpy_quantile_method": "linear",
        "fold_weighting": "equal; ratio recomputed within every fold replicate",
        "fold_resampling": {
            str(fold): len(arrays[fold][0]) for fold in range(DEFAULT_FOLDS)
        },
        "training_seeds_fixed_not_resampled": list(seed_tuple),
    }


def _a23_labels(value: Any) -> Any:
    if isinstance(value, dict):
        renamed = {"a9_mean": "a23_mean", "a9_ratio": "a23_ratio"}
        return {
            renamed.get(str(key), str(key)): _a23_labels(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_a23_labels(item) for item in value]
    if isinstance(value, str):
        if value == "A9_minus_A8":
            return "A23_minus_A8"
        if value.startswith("multitown-a9-"):
            return "multitown-a23-" + value.removeprefix("multitown-a9-")
    return value


def _a23_vs_a22(
    a23_rows: Sequence[Mapping[str, Any]],
    a22_rows: Sequence[Mapping[str, Any]], schedule: A23StatisticsSchedule,
) -> dict[str, Any]:
    arguments = {
        "seeds": schedule.seeds,
        "iterations": schedule.bootstrap_iterations,
    }
    metrics = {
        "episode_success": 20260813,
        "tokens_used": 20260814,
        "latency_used_s": 20260815,
        "human_escalations": 20260816,
        "assisted_episode_success": 20260817,
        "wrong_execution": 20260818,
        "return": 20260819,
        "safety_penalty_burden": 20260916,
    }
    result = {
        metric: fixed_seed_policy_bootstrap(
            a23_rows, a22_rows, metric=metric,
            seed=seed + 100_000, **arguments,
        ) for metric, seed in metrics.items()
    }
    result["token_reduction_fraction"] = fixed_seed_policy_bootstrap(
        a23_rows, a22_rows, metric="tokens_used", ratio_reduction=True,
        seed=20260913 + 100_000, **arguments,
    )
    result["subgoal_completion_rate"] = fixed_seed_policy_ratio_bootstrap(
        a23_rows, a22_rows, numerator="resolved", denominator="incidents",
        seed=20260914 + 100_000, **arguments,
    )
    result["wrong_executions_per_incident"] = fixed_seed_policy_ratio_bootstrap(
        a23_rows, a22_rows, numerator="wrong_executions",
        denominator="incidents", seed=20260915 + 100_000, **arguments,
    )
    return result


def _descriptive_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize empty A23 descriptive cell")
    incidents = sum(int(row["incidents"]) for row in rows)
    if incidents <= 0:
        raise ValueError("A23 descriptive incident denominator is zero")
    return {
        "rows": len(rows),
        "episodes": len({str(row["episode_id"]) for row in rows}),
        "autonomous_success_rate": float(np.mean([
            bool(row["episode_success"]) for row in rows
        ])),
        "tokens_per_episode": float(np.mean([
            float(row["tokens_used"]) for row in rows
        ])),
        "unsafe_episode_rate": float(np.mean([
            bool(row["had_wrong_execution"]) for row in rows
        ])),
        "wrong_executions_per_incident": (
            sum(int(row["wrong_executions"]) for row in rows) / incidents
        ),
        "latency_per_episode_s": float(np.mean([
            float(row["latency_used_s"]) for row in rows
        ])),
        "subgoal_completion_rate": (
            sum(int(row["resolved"]) for row in rows) / incidents
        ),
        "safety_penalty_burden_per_episode": float(np.mean([
            float(row["safety_penalty_burden"]) for row in rows
        ])),
        "human_escalations_per_episode": float(np.mean([
            float(row["human_escalations"]) for row in rows
        ])),
        "invalid_actions": sum(int(row["invalid_actions"]) for row in rows),
        "budget_violations": sum(int(row["budget_violations"]) for row in rows),
    }


def _descriptive_tables(
    a8_rows: Sequence[Mapping[str, Any]],
    a23_rows: Sequence[Mapping[str, Any]],
    a22_rows: Sequence[Mapping[str, Any]], schedule: A23StatisticsSchedule,
) -> dict[str, Any]:
    a8_by_fold = {
        str(fold): _descriptive_summary([
            row for row in a8_rows if int(row["outer_fold"]) == fold
        ]) for fold in schedule.folds
    }
    a23_by_fold_seed = {}
    a22_by_fold_seed = {}
    differences = {}
    metric_fields = (
        "autonomous_success_rate", "tokens_per_episode",
        "unsafe_episode_rate", "wrong_executions_per_incident",
        "latency_per_episode_s", "subgoal_completion_rate",
        "safety_penalty_burden_per_episode",
        "human_escalations_per_episode",
    )
    for fold in schedule.folds:
        for seed in schedule.seeds:
            key = f"{fold}:{seed}"
            a23_summary = _descriptive_summary([
                row for row in a23_rows
                if int(row["outer_fold"]) == fold
                and int(row["training_seed"]) == seed
            ])
            a22_summary = _descriptive_summary([
                row for row in a22_rows
                if int(row["outer_fold"]) == fold
                and int(row["training_seed"]) == seed
            ])
            a23_by_fold_seed[key] = a23_summary
            a22_by_fold_seed[key] = a22_summary
            baseline = a8_by_fold[str(fold)]
            differences[key] = {
                "a23_minus_a8": {
                    field: a23_summary[field] - baseline[field]
                    for field in metric_fields
                },
                "a23_minus_a22": {
                    field: a23_summary[field] - a22_summary[field]
                    for field in metric_fields
                },
            }
    behavior = {}
    for fold in schedule.folds:
        for seed in schedule.seeds:
            cells = [
                row for row in a23_rows
                if int(row["outer_fold"]) == fold
                and int(row["training_seed"]) == seed
            ]
            mechanisms = {str(row["mechanism"]) for row in cells}
            if len(mechanisms) != 1:
                raise ValueError("A23 selected behavior cell has multiple mechanisms")
            mechanism = next(iter(mechanisms))
            item = _behavior_summary(cells)
            episodes = len(cells)
            item["action_means_per_episode"] = {
                action: count / episodes
                for action, count in item["action_counts"].items()
            }
            item["shield_interventions_per_episode"] = (
                item["shield_interventions"] / episodes
            )
            behavior[f"{fold}:{seed}:{mechanism}"] = item
    return {
        "a8_by_fold": a8_by_fold,
        "a23_by_fold_seed": a23_by_fold_seed,
        "a22_by_fold_seed": a22_by_fold_seed,
        "differences_by_fold_seed": differences,
        "a23_behavior_by_fold_seed_mechanism": behavior,
        "comparison_intervals_attached_to_behavior_diagnostics": False,
    }


def _validate_result_products(
    a8_rows: Sequence[Mapping[str, Any]],
    a23_rows: Sequence[Mapping[str, Any]],
    a22_rows: Sequence[Mapping[str, Any]], schedule: A23StatisticsSchedule,
) -> None:
    if schedule.folds != tuple(range(DEFAULT_FOLDS)):
        raise ValueError("A23 statistics require the frozen five folds")
    a8_index = {str(row["episode_id"]): row for row in a8_rows}
    if (
        len(a8_index) != len(a8_rows)
        or any(
            "training_seed" not in row or row["training_seed"] is not None
            for row in a8_rows
        )
    ):
        raise ValueError("A23 statistics A8 product is duplicated or seeded")
    a8_ids: dict[int, list[str]] = {}
    for fold in schedule.folds:
        ids = sorted(
            episode_id for episode_id, row in a8_index.items()
            if int(row["outer_fold"]) == fold
        )
        if len(ids) != schedule.outer_episodes_per_fold:
            raise ValueError("A23 statistics A8 fold product is incomplete")
        a8_ids[fold] = ids
    _, a23_ids = _seeded_index(a23_rows, schedule.seeds, label="A23 result")
    _, a22_ids = _seeded_index(a22_rows, schedule.seeds, label="A22 result")
    if a23_ids != a8_ids or a22_ids != a8_ids:
        raise ValueError("A23 statistics paired episode products differ")


def result_statistics(
    a8_rows: Sequence[Mapping[str, Any]],
    a23_rows: Sequence[Mapping[str, Any]],
    a22_rows: Sequence[Mapping[str, Any]], schedule: A23StatisticsSchedule, *,
    gate_evaluable: bool,
) -> dict[str, Any]:
    """Compute the two frozen comparisons and keep smoke gates disabled."""

    _validate_result_products(a8_rows, a23_rows, a22_rows, schedule)

    compatible = RunSchedule(
        mode=schedule.mode, seeds=schedule.seeds, folds=schedule.folds,
        updates=schedule.updates,
        episodes_per_update=schedule.episodes_per_update,
        evaluation_episodes_per_fold=schedule.outer_episodes_per_fold,
        bootstrap_iterations=schedule.bootstrap_iterations,
        bootstrap_seed=BOOTSTRAP_SEED, threads=schedule.threads,
    )
    vs_a8 = _a23_labels(_paired_effects(a8_rows, a23_rows, compatible))
    vs_a22 = _a23_vs_a22(a23_rows, a22_rows, schedule)
    per_seed: dict[str, Any] = {}
    token_seeds = (20260913, 20260914, 20260915)
    unsafe_seeds = (20261013, 20261014, 20261015)
    wrong_seeds = (20261113, 20261114, 20261115)
    for index, training_seed in enumerate(schedule.seeds):
        selected = [
            row for row in a23_rows
            if int(row["training_seed"]) == training_seed
        ]
        success = _a23_labels(fold_cluster_bootstrap(
            a8_rows, selected, metric="episode_success", seeds=(training_seed,),
            iterations=schedule.bootstrap_iterations, seed=20260813,
        ))
        tokens = _a23_labels(fold_cluster_bootstrap(
            a8_rows, selected, metric="tokens_used", seeds=(training_seed,),
            iterations=schedule.bootstrap_iterations, seed=token_seeds[index],
        ))
        unsafe = _a23_labels(fold_cluster_bootstrap(
            a8_rows, selected, metric="wrong_execution", seeds=(training_seed,),
            iterations=schedule.bootstrap_iterations, seed=unsafe_seeds[index],
        ))
        wrong = _a23_labels(fold_cluster_ratio_bootstrap(
            a8_rows, selected, numerator="wrong_executions",
            denominator="incidents", seeds=(training_seed,),
            iterations=schedule.bootstrap_iterations, seed=wrong_seeds[index],
        ))
        per_fold_unsafe = {}
        for fold in schedule.folds:
            learned = [row for row in selected if int(row["outer_fold"]) == fold]
            baseline = [row for row in a8_rows if int(row["outer_fold"]) == fold]
            per_fold_unsafe[str(fold)] = float(np.mean([
                bool(row["had_wrong_execution"]) for row in learned
            ])) - float(np.mean([
                bool(row["had_wrong_execution"]) for row in baseline
            ]))
        invalid_actions = sum(int(row["invalid_actions"]) for row in selected)
        budget_violations = sum(int(row["budget_violations"]) for row in selected)
        gates = {
            "unsafe_within_a8_plus_0_02": (
                unsafe["a23_mean"] <= unsafe["a8_mean"] + 0.02
            ),
            "wrong_per_incident_within_a8_plus_0_01": (
                wrong["a23_ratio"] <= wrong["a8_ratio"] + 0.01
            ),
            "success_lower_bound_nonnegative": success["ci95_low"] >= 0.0,
            "tokens_no_greater_than_a8": tokens["a23_mean"] <= tokens["a8_mean"],
            "every_fold_unsafe_within_a8_plus_0_05": all(
                value <= 0.05 for value in per_fold_unsafe.values()
            ),
            "zero_invalid_actions": invalid_actions == 0,
            "zero_budget_violations": budget_violations == 0,
        }
        per_seed[str(training_seed)] = {
            "success": success, "tokens": tokens, "unsafe_episode": unsafe,
            "wrong_executions_per_incident": wrong,
            "per_fold_unsafe_difference": per_fold_unsafe,
            "invalid_actions": invalid_actions,
            "budget_violations": budget_violations,
            "gates": gates, "all_gates_passed": all(gates.values()),
        }
    raw_system = all(item["all_gates_passed"] for item in per_seed.values())
    utility_conjuncts = {
        "success_lower_bound_nonnegative": (
            vs_a22["episode_success"]["ci95_low"] >= 0.0
        ),
        "token_upper_bound_nonpositive": (
            vs_a22["tokens_used"]["ci95_high"] <= 0.0
        ),
        "a8_relative_safety_and_validity_preserved": all(
            item["gates"]["unsafe_within_a8_plus_0_02"]
            and item["gates"]["wrong_per_incident_within_a8_plus_0_01"]
            and item["gates"]["every_fold_unsafe_within_a8_plus_0_05"]
            and item["gates"]["zero_invalid_actions"]
            and item["gates"]["zero_budget_violations"]
            for item in per_seed.values()
        ),
    }
    raw_utility = all(utility_conjuncts.values())
    utility_passed = bool(gate_evaluable and raw_system and raw_utility)
    return {
        "schema_version": A23_STATISTICS_VERSION,
        "a23_minus_a8": vs_a8, "a23_minus_a22": vs_a22,
        "per_seed_system_recovery": per_seed,
        "selected_outer_behavior": _behavior_summary(a23_rows),
        "descriptive_tables": _descriptive_tables(
            a8_rows, a23_rows, a22_rows, schedule,
        ),
        "gate_evaluable": gate_evaluable,
        "raw_system_recovery_conjunction": bool(gate_evaluable and raw_system),
        "system_recovery_gate_passed": bool(gate_evaluable and raw_system),
        "utility_replacement_gate_evaluable": bool(gate_evaluable and raw_system),
        "utility_replacement_conjuncts": utility_conjuncts,
        "raw_utility_replacement_conjunction": bool(
            gate_evaluable and raw_system and raw_utility
        ),
        "utility_replacement_criterion_passed": utility_passed,
        "strictly_favorable_success_interval": bool(
            utility_passed and vs_a22["episode_success"]["ci95_low"] > 0.0
        ),
        "strictly_favorable_token_interval": bool(
            utility_passed and vs_a22["tokens_used"]["ci95_high"] < 0.0
        ),
    }
