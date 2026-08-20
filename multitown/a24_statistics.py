"""Frozen A24 pairing, bootstrap statistics, and ordered gates."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .a24_contract import (
    A24_FORMAL_THREADS,
    CALIBRATION_DIAGNOSTIC_VERSION,
    FORMAL_FOLDS,
    FORMAL_SEEDS,
    MECHANISM,
)


A24_STATISTICS_VERSION = "multitown-a24-fixed-seed-adaptive-statistics-v1"
A24_INTERVAL_VERSION = "multitown-a24-fold-cluster-percentile-v1"

A24_MINUS_A8_SEEDS = {
    "episode_success": 20261213,
    "tokens_used": 20261214,
    "latency_used_s": 20261215,
    "wrong_execution": 20261216,
    "wrong_executions_per_incident": 20261217,
    "assisted_episode_success": 20261218,
    "subgoal_completion_rate": 20261219,
    "return": 20261220,
    "safety_penalty_burden": 20261221,
    "human_escalations": 20261222,
}
A24_MINUS_A22_OFFSET = 100_000
SEED_SPECIFIC_SUCCESS_STREAMS = {
    20260812: 20261312,
    20260813: 20261313,
    20260814: 20261314,
}
SEED_SPECIFIC_TOKEN_STREAMS = {
    20260812: 20261412,
    20260813: 20261413,
    20260814: 20261414,
}

_BOOLEAN_METRICS = (
    "episode_success",
    "assisted_episode_success",
    "had_wrong_execution",
)
_INTEGER_METRICS = (
    "tokens_used",
    "human_escalations",
    "resolved",
    "incidents",
    "wrong_executions",
    "invalid_actions",
    "budget_violations",
    "shield_interventions",
)
_REAL_METRICS = (
    "latency_used_s",
    "return",
    "safety_penalty_burden",
)
_SHARED_DIGEST_FIELDS = (
    "episode_sha256",
    "train_bank_sha256",
    "resource_contract_sha256",
    "environment_source_sha256",
)


@dataclass(frozen=True)
class A24StatisticsSchedule:
    mode: str
    seeds: tuple[int, ...]
    folds: tuple[int, ...]
    updates: int
    episodes_per_update: int
    outer_episodes_per_fold: int
    bootstrap_iterations: int
    threads: int


_FORMAL_STATISTICS_SCHEDULE = A24StatisticsSchedule(
    mode="adaptive-same-bank-development",
    seeds=FORMAL_SEEDS,
    folds=FORMAL_FOLDS,
    updates=120,
    episodes_per_update=48,
    outer_episodes_per_fold=600,
    bootstrap_iterations=20_000,
    threads=A24_FORMAL_THREADS,
)


@dataclass(frozen=True)
class PairedPanel:
    a8: dict[str, Mapping[str, Any]]
    a24: dict[tuple[str, int], Mapping[str, Any]]
    a22: dict[tuple[str, int], Mapping[str, Any]]
    ids_by_fold: dict[int, tuple[str, ...]]
    seeds: tuple[int, ...]


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"invalid A24 SHA-256 binding: {label}")
    return value


def _row_sha256(row: Mapping[str, Any]) -> str:
    raw = json.dumps(
        dict(row),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _require_int(value: Any, *, label: str, nonnegative: bool = False) -> int:
    if type(value) is not int or (nonnegative and value < 0):
        raise ValueError(f"invalid A24 integer field: {label}")
    return value


def _require_real(value: Any, *, label: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"invalid A24 numeric field: {label}")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"non-finite A24 numeric field: {label}")
    return result


def _validate_metric_types(row: Mapping[str, Any], *, label: str) -> None:
    if type(row) is not dict:
        raise ValueError(f"invalid A24 row object: {label}")
    episode_id = row.get("episode_id")
    if type(episode_id) is not str or not episode_id:
        raise ValueError(f"invalid A24 episode id: {label}")
    for field in _BOOLEAN_METRICS:
        if type(row.get(field)) is not bool:
            raise ValueError(f"invalid A24 boolean field: {label}.{field}")
    for field in _INTEGER_METRICS:
        if field == "shield_interventions" and field not in row:
            continue
        _require_int(row.get(field), label=f"{label}.{field}", nonnegative=True)
    for field in _REAL_METRICS:
        _require_real(row.get(field), label=f"{label}.{field}")
    for field in _SHARED_DIGEST_FIELDS:
        _require_sha256(row.get(field), label=f"{label}.{field}")


def _validate_a8_row(
    row: Mapping[str, Any],
    *,
    label: str,
    expected_run_contract_sha256: str,
) -> None:
    _validate_metric_types(row, label=label)
    if (
        "training_seed" not in row
        or row["training_seed"] is not None
        or type(row.get("outer_fold")) is not int
        or row.get("run_contract_sha256") != expected_run_contract_sha256
        or row.get("system") != "A8-long-public-view"
    ):
        raise ValueError(f"A24 A8 row binding mismatch: {label}")


def _validate_policy_row(
    row: Mapping[str, Any],
    *,
    label: str,
    expected_run_contract_sha256: str,
    expected_outer_gate_sha256: str | None,
    expected_checkpoint_sha256: str,
    expected_system: str,
) -> None:
    _validate_metric_types(row, label=label)
    if (
        type(row.get("training_seed")) is not int
        or type(row.get("outer_fold")) is not int
        or type(row.get("design_outer_fold")) is not int
        or row.get("run_contract_sha256") != expected_run_contract_sha256
        or row.get("outer_gate_sha256") != expected_outer_gate_sha256
        or row.get("final_checkpoint_sha256") != expected_checkpoint_sha256
        or row.get("shield_interventions") != 0
        or row.get("system") != expected_system
    ):
        raise ValueError(f"A24 policy row binding mismatch: {label}")
    _require_sha256(row.get("a8_row_sha256"), label=f"{label}.a8_row_sha256")
    _require_sha256(
        row.get("final_checkpoint_sha256"),
        label=f"{label}.final_checkpoint_sha256",
    )
    _require_sha256(
        row.get("run_contract_sha256"),
        label=f"{label}.run_contract_sha256",
    )
    if expected_outer_gate_sha256 is not None:
        _require_sha256(
            row.get("outer_gate_sha256"),
            label=f"{label}.outer_gate_sha256",
        )


def _validate_checkpoint_map(
    checkpoints: Mapping[tuple[int, int], str],
    *,
    folds: tuple[int, ...],
    seeds: tuple[int, ...],
    label: str,
) -> None:
    expected = {(fold, seed) for fold in folds for seed in seeds}
    if type(checkpoints) is not dict or set(checkpoints) != expected:
        raise ValueError(f"A24 {label} checkpoint product mismatch")
    for key, digest in checkpoints.items():
        _require_sha256(digest, label=f"{label}.{key}")


def _validate_schedule(schedule: A24StatisticsSchedule) -> None:
    if (
        type(schedule.mode) is not str
        or type(schedule.seeds) is not tuple
        or type(schedule.folds) is not tuple
        or not schedule.seeds
        or not schedule.folds
        or len(set(schedule.seeds)) != len(schedule.seeds)
        or len(set(schedule.folds)) != len(schedule.folds)
    ):
        raise ValueError("invalid A24 statistics schedule")
    for name in (
        "updates",
        "episodes_per_update",
        "outer_episodes_per_fold",
        "bootstrap_iterations",
        "threads",
    ):
        _require_int(getattr(schedule, name), label=f"schedule.{name}")
        if getattr(schedule, name) <= 0:
            raise ValueError("invalid A24 statistics schedule")
    if any(seed not in SEED_SPECIFIC_SUCCESS_STREAMS for seed in schedule.seeds):
        raise ValueError("A24 has no frozen per-seed RNG stream")
    if (
        schedule.mode == "adaptive-same-bank-development"
        and schedule != _FORMAL_STATISTICS_SCHEDULE
    ):
        raise ValueError("A24 formal statistics schedule changed")


def validate_calibration_pairs(
    a8_rows: Sequence[Mapping[str, Any]],
    a24_rows: Sequence[Mapping[str, Any]],
    *,
    folds: Sequence[int],
    seeds: Sequence[int],
    episodes_per_fold: int,
    expected_a8_run_contract_sha256: str,
    expected_run_contract_sha256: str,
    expected_checkpoint_sha256: Mapping[tuple[int, int], str],
) -> tuple[
    dict[int, tuple[Mapping[str, Any], ...]],
    dict[tuple[int, int], tuple[Mapping[str, Any], ...]],
]:
    """Validate the exact five-fold A8 and 15-cell A24 calibration panel."""

    fold_tuple = tuple(folds)
    seed_tuple = tuple(seeds)
    _require_sha256(
        expected_a8_run_contract_sha256,
        label="calibration.expected_a8_run_contract_sha256",
    )
    _require_sha256(
        expected_run_contract_sha256,
        label="calibration.expected_run_contract_sha256",
    )
    _validate_checkpoint_map(
        expected_checkpoint_sha256,
        folds=fold_tuple,
        seeds=seed_tuple,
        label="calibration",
    )
    if type(episodes_per_fold) is not int or episodes_per_fold <= 0:
        raise ValueError("invalid A24 calibration episode count")
    a8_by_design: dict[int, list[Mapping[str, Any]]] = {
        fold: [] for fold in fold_tuple
    }
    seen_a8: set[str] = set()
    for row in a8_rows:
        _validate_a8_row(
            row,
            label="calibration.a8",
            expected_run_contract_sha256=expected_a8_run_contract_sha256,
        )
        episode_id = row["episode_id"]
        if row["outer_fold"] not in fold_tuple:
            raise ValueError("A24 calibration A8 fold is outside design")
        if episode_id in seen_a8:
            raise ValueError("duplicate A24 calibration A8 episode")
        seen_a8.add(episode_id)
        design_fold = (int(row["outer_fold"]) - 1) % len(fold_tuple)
        if design_fold not in a8_by_design:
            raise ValueError("A24 calibration A8 fold is outside design")
        a8_by_design[design_fold].append(row)
    if any(len(rows) != episodes_per_fold for rows in a8_by_design.values()):
        raise ValueError("A24 calibration A8 fold product is incomplete")
    a24_by_cell: dict[tuple[int, int], list[Mapping[str, Any]]] = {
        (fold, seed): [] for fold in fold_tuple for seed in seed_tuple
    }
    seen_a24: set[tuple[int, int, str]] = set()
    ids_by_design = {
        fold: {str(row["episode_id"]) for row in rows}
        for fold, rows in a8_by_design.items()
    }
    for row in a24_rows:
        if type(row) is not dict:
            raise ValueError("A24 calibration policy row is malformed")
        fold = row.get("design_outer_fold")
        seed = row.get("training_seed")
        if type(fold) is not int or type(seed) is not int:
            raise ValueError("A24 calibration policy row is malformed")
        checkpoint = expected_checkpoint_sha256.get((fold, seed))
        if checkpoint is None:
            raise ValueError("A24 calibration policy cell is outside design")
        _validate_policy_row(
            row,
            label="calibration.a24",
            expected_run_contract_sha256=expected_run_contract_sha256,
            expected_outer_gate_sha256=None,
            expected_checkpoint_sha256=checkpoint,
            expected_system="A24-cr-ppo-no-shield",
        )
        episode_id = row["episode_id"]
        key = (fold, seed, episode_id)
        if key in seen_a24:
            raise ValueError("duplicate A24 calibration policy row")
        seen_a24.add(key)
        if (
            (fold, seed) not in a24_by_cell
            or episode_id not in ids_by_design[fold]
            or row["outer_fold"] != (fold + 1) % len(fold_tuple)
            or row.get("mechanism") != MECHANISM
            or row.get("evaluation_phase") != "inner-calibration"
        ):
            raise ValueError("A24 calibration policy row provenance mismatch")
        a24_by_cell[(fold, seed)].append(row)
    if any(len(rows) != episodes_per_fold for rows in a24_by_cell.values()):
        raise ValueError("A24 calibration policy cell is incomplete")
    for fold, baseline_rows in a8_by_design.items():
        baseline = {str(row["episode_id"]): row for row in baseline_rows}
        for seed in seed_tuple:
            cell = a24_by_cell[(fold, seed)]
            if {str(row["episode_id"]) for row in cell} != set(baseline):
                raise ValueError("A24 calibration pairing differs")
            for row in cell:
                source = baseline[row["episode_id"]]
                if any(
                    row.get(field) != source.get(field)
                    for field in _SHARED_DIGEST_FIELDS
                ) or row["a8_row_sha256"] != _row_sha256(source):
                    raise ValueError("A24 calibration paired provenance differs")
    return (
        {
            fold: tuple(sorted(rows, key=lambda row: str(row["episode_id"])))
            for fold, rows in a8_by_design.items()
        },
        {
            key: tuple(sorted(rows, key=lambda row: str(row["episode_id"])))
            for key, rows in a24_by_cell.items()
        },
    )


def evaluate_calibration_gate(
    a8_rows: Sequence[Mapping[str, Any]],
    a24_rows: Sequence[Mapping[str, Any]],
    *,
    folds: Sequence[int],
    seeds: Sequence[int],
    episodes_per_fold: int,
    smoke: bool,
    expected_a8_run_contract_sha256: str,
    expected_run_contract_sha256: str,
    expected_checkpoint_sha256: Mapping[tuple[int, int], str],
) -> dict[str, Any]:
    if type(smoke) is not bool:
        raise ValueError("A24 calibration smoke flag is not boolean")
    if not smoke and (
        tuple(folds) != FORMAL_FOLDS
        or tuple(seeds) != FORMAL_SEEDS
        or episodes_per_fold != 600
    ):
        raise ValueError("A24 formal calibration product changed")
    baseline, cells = validate_calibration_pairs(
        a8_rows,
        a24_rows,
        folds=folds,
        seeds=seeds,
        episodes_per_fold=episodes_per_fold,
        expected_a8_run_contract_sha256=expected_a8_run_contract_sha256,
        expected_run_contract_sha256=expected_run_contract_sha256,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
    )
    results: dict[str, Any] = {}
    for fold in folds:
        a8 = baseline[int(fold)]
        a8_unsafe = float(
            np.mean([bool(row["had_wrong_execution"]) for row in a8])
        )
        a8_wrong = _point_ratio(a8, "wrong_executions", "incidents")
        thresholds = {
            "unsafe_episode": min(1.0, a8_unsafe + 0.02),
            "wrong_executions_per_incident": min(1.0, a8_wrong + 0.01),
        }
        for seed in seeds:
            rows = cells[(int(fold), int(seed))]
            unsafe = float(
                np.mean([bool(row["had_wrong_execution"]) for row in rows])
            )
            wrong = _point_ratio(rows, "wrong_executions", "incidents")
            invalid_actions = sum(row["invalid_actions"] for row in rows)
            budget_violations = sum(row["budget_violations"] for row in rows)
            conjuncts = {
                "unsafe_within_a8_plus_0_02": unsafe <= thresholds["unsafe_episode"],
                "wrong_per_incident_within_a8_plus_0_01": (
                    wrong <= thresholds["wrong_executions_per_incident"]
                ),
                "zero_invalid_actions": invalid_actions == 0,
                "zero_budget_violations": budget_violations == 0,
            }
            results[f"{int(fold)}:{int(seed)}"] = {
                "outer_fold": int(fold),
                "training_seed": int(seed),
                "rows": len(rows),
                "thresholds": thresholds,
                "a8": {
                    "unsafe_episode_rate": a8_unsafe,
                    "wrong_executions_per_incident": a8_wrong,
                },
                "a24": {
                    "unsafe_episode_rate": unsafe,
                    "wrong_executions_per_incident": wrong,
                    "invalid_actions": invalid_actions,
                    "budget_violations": budget_violations,
                },
                "conjuncts": conjuncts,
                "passed": all(conjuncts.values()),
            }
    raw = all(row["passed"] for row in results.values())
    return {
        "schema_version": "multitown-a24-calibration-gate-decision-v1",
        "cells": results,
        "expected_cells": len(tuple(folds)) * len(tuple(seeds)),
        "all_cells_present": len(results) == len(tuple(folds)) * len(tuple(seeds)),
        "raw_conjunction": raw,
        "formal_gate_evaluable": not smoke,
        "outer_gate_permitted": bool(not smoke and raw),
        "a22_calibration_enters_gate": False,
    }


def _value_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def calibration_comparator_diagnostic(
    a8_rows: Sequence[Mapping[str, Any]],
    a24_rows: Sequence[Mapping[str, Any]],
    a22_rows: Sequence[Mapping[str, Any]],
    *,
    folds: Sequence[int],
    seeds: Sequence[int],
    episodes_per_fold: int,
    expected_a8_run_contract_sha256: str,
    expected_a24_run_contract_sha256: str,
    expected_a22_run_contract_sha256: str,
    expected_a22_source_artifacts: Mapping[str, str],
    expected_a24_checkpoint_sha256: Mapping[tuple[int, int], str],
    comparator_ledger: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and summarize A22 calibration without entering any A24 gate."""

    fold_tuple = tuple(folds)
    seed_tuple = tuple(seeds)
    baseline, a24_cells = validate_calibration_pairs(
        a8_rows,
        a24_rows,
        folds=fold_tuple,
        seeds=seed_tuple,
        episodes_per_fold=episodes_per_fold,
        expected_a8_run_contract_sha256=expected_a8_run_contract_sha256,
        expected_run_contract_sha256=expected_a24_run_contract_sha256,
        expected_checkpoint_sha256=expected_a24_checkpoint_sha256,
    )
    if type(comparator_ledger) is not dict or set(comparator_ledger) != {
        "schema_version",
        "source_artifacts",
        "source_run_contract_sha256",
        "source_calibration_rows",
        "selected_calibration_rows",
        "entries",
        "ledger_sha256",
        "historical_initial_tensor_receipt_available",
    }:
        raise ValueError("A24 comparator ledger schema changed")
    entries = comparator_ledger["entries"]
    if (
        comparator_ledger["schema_version"]
        != "multitown-a24-a22-lagrangian-comparator-ledger-v1"
        or comparator_ledger["source_run_contract_sha256"]
        != expected_a22_run_contract_sha256
        or type(comparator_ledger["source_artifacts"]) is not dict
        or type(comparator_ledger["source_calibration_rows"]) is not int
        or comparator_ledger["source_calibration_rows"] != 36_000
        or type(comparator_ledger["selected_calibration_rows"]) is not int
        or comparator_ledger["selected_calibration_rows"] != 9_000
        or type(entries) is not list
        or len(entries) != len(FORMAL_FOLDS) * len(FORMAL_SEEDS)
        or type(comparator_ledger["ledger_sha256"]) is not str
        or comparator_ledger["ledger_sha256"] != _value_sha256(entries)
        or comparator_ledger["historical_initial_tensor_receipt_available"]
        is not False
    ):
        raise ValueError("A24 comparator ledger binding changed")
    source_artifacts = comparator_ledger["source_artifacts"]
    if (
        type(expected_a22_source_artifacts) is not dict
        or source_artifacts != expected_a22_source_artifacts
    ):
        raise ValueError("A24 comparator pinned source artifacts changed")
    if set(source_artifacts) != {
        "all-fits-complete.json",
        "all-selections-frozen.json",
        "artifact-manifest.json",
        "calibration-decisions.jsonl",
        "outer-decisions.jsonl",
        "result.json",
        "run-contract.json",
    }:
        raise ValueError("A24 comparator source artifact inventory changed")
    for name, digest in source_artifacts.items():
        _require_sha256(
            digest,
            label=f"calibration_diagnostic.source_artifacts.{name}",
        )
    _require_sha256(
        comparator_ledger["ledger_sha256"],
        label="calibration_diagnostic.comparator_ledger_sha256",
    )
    entry_index: dict[tuple[int, int], Mapping[str, Any]] = {}
    for entry in entries:
        if type(entry) is not dict or set(entry) != {
            "outer_fold",
            "training_seed",
            "mechanism",
            "paths",
            "raw_sha256",
            "sample_sequence_sha256",
            "calibration_subset_rows",
            "calibration_subset_sha256",
            "derived_expected_initialization",
        }:
            raise ValueError("A24 comparator ledger entry schema changed")
        fold = entry["outer_fold"]
        seed = entry["training_seed"]
        paths = entry["paths"]
        raw_sha = entry["raw_sha256"]
        initialization = entry["derived_expected_initialization"]
        prefix = f"fits/outer-fold-{fold}/seed-{seed}/lagrangian"
        if (
            type(fold) is not int
            or fold not in FORMAL_FOLDS
            or type(seed) is not int
            or seed not in FORMAL_SEEDS
            or entry["mechanism"] != "lagrangian"
            or type(paths) is not dict
            or paths
            != {
                "final.pt": f"{prefix}/final.pt",
                "fit-complete.json": f"{prefix}/fit-complete.json",
                "training-metrics.jsonl": f"{prefix}/training-metrics.jsonl",
            }
            or type(raw_sha) is not dict
            or set(raw_sha)
            != {"final.pt", "fit-complete.json", "training-metrics.jsonl"}
            or type(entry["calibration_subset_rows"]) is not int
            or entry["calibration_subset_rows"] != 600
            or type(initialization) is not dict
            or set(initialization)
            != {
                "algorithm",
                "model_sha256",
                "named_optimizer_sha256",
                "historical_initial_tensor_receipt",
            }
            or initialization["algorithm"]
            != "pinned-a22-source-runtime-seed-derived-expectation-v1"
            or initialization["historical_initial_tensor_receipt"] is not False
        ):
            raise ValueError("A24 comparator ledger entry binding changed")
        for label, digest in (
            *((name, raw_sha[name]) for name in sorted(raw_sha)),
            ("calibration_subset", entry["calibration_subset_sha256"]),
            ("sample_sequence", entry["sample_sequence_sha256"]),
            ("initial_model", initialization["model_sha256"]),
            ("initial_optimizer", initialization["named_optimizer_sha256"]),
        ):
            _require_sha256(digest, label=f"calibration_diagnostic.{fold}:{seed}.{label}")
        key = (fold, seed)
        if key in entry_index:
            raise ValueError("duplicate A24 comparator ledger cell")
        entry_index[key] = entry
    if set(entry_index) != {
        (fold, seed) for fold in FORMAL_FOLDS for seed in FORMAL_SEEDS
    }:
        raise ValueError("A24 comparator ledger cell product changed")

    a22_cells: dict[tuple[int, int], list[Mapping[str, Any]]] = {
        (fold, seed): [] for fold in fold_tuple for seed in seed_tuple
    }
    a8_by_design_and_id = {
        fold: {str(source["episode_id"]): source for source in baseline[fold]}
        for fold in fold_tuple
    }
    seen: set[tuple[int, int, str]] = set()
    for row in a22_rows:
        if type(row) is not dict:
            raise ValueError("A24 A22 calibration row is malformed")
        fold = row.get("design_outer_fold")
        seed = row.get("training_seed")
        if type(fold) is not int or type(seed) is not int:
            raise ValueError("A24 A22 calibration row key is malformed")
        entry = entry_index.get((fold, seed))
        if (fold, seed) not in a22_cells or entry is None:
            raise ValueError("A24 A22 calibration cell is outside schedule")
        _validate_policy_row(
            row,
            label="calibration.a22_lagrangian",
            expected_run_contract_sha256=expected_a22_run_contract_sha256,
            expected_outer_gate_sha256=None,
            expected_checkpoint_sha256=entry["raw_sha256"]["final.pt"],
            expected_system="A22-lagrangian",
        )
        episode_id = row["episode_id"]
        key = (fold, seed, episode_id)
        source = a8_by_design_and_id[fold].get(episode_id)
        if key in seen or source is None:
            raise ValueError("A24 A22 calibration pairing changed")
        seen.add(key)
        if (
            row["outer_fold"] != (fold + 1) % len(fold_tuple)
            or row.get("mechanism") != "lagrangian"
            or row.get("evaluation_phase") != "inner-calibration"
            or any(
                row.get(field) != source.get(field)
                for field in _SHARED_DIGEST_FIELDS
            )
            or row["a8_row_sha256"] != _row_sha256(source)
        ):
            raise ValueError("A24 A22 calibration provenance changed")
        a22_cells[(fold, seed)].append(row)
    if any(len(rows) != episodes_per_fold for rows in a22_cells.values()):
        raise ValueError("A24 A22 calibration cell product is incomplete")

    metric_fields = (
        "autonomous_success_rate",
        "assisted_success_rate",
        "tokens_per_episode",
        "unsafe_episode_rate",
        "wrong_executions_per_incident",
        "latency_per_episode_s",
        "subgoal_completion_rate",
        "return_per_episode",
        "safety_penalty_burden_per_episode",
        "human_escalations_per_episode",
        "invalid_actions",
        "budget_violations",
    )
    cells: dict[str, Any] = {}
    for fold in fold_tuple:
        for seed in seed_tuple:
            key = (fold, seed)
            a24_cell = tuple(a24_cells[key])
            a22_cell = tuple(
                sorted(a22_cells[key], key=lambda row: str(row["episode_id"]))
            )
            if [row["episode_id"] for row in a24_cell] != [
                row["episode_id"] for row in a22_cell
            ]:
                raise ValueError("A24/A22 calibration episode order differs")
            entry = entry_index[key]
            scheduled_digest = _value_sha256(list(a22_cell))
            full_subset = episodes_per_fold == entry["calibration_subset_rows"]
            if full_subset and scheduled_digest != entry["calibration_subset_sha256"]:
                raise ValueError("A24 A22 full calibration subset digest changed")
            a24_summary = _descriptive_summary(a24_cell)
            a22_summary = _descriptive_summary(a22_cell)
            cells[f"{fold}:{seed}"] = {
                "outer_fold": fold,
                "training_seed": seed,
                "rows": episodes_per_fold,
                "a22_checkpoint_sha256": entry["raw_sha256"]["final.pt"],
                "a22_full_calibration_subset_sha256": entry[
                    "calibration_subset_sha256"
                ],
                "a22_scheduled_subset_sha256": scheduled_digest,
                "scheduled_subset_is_full_ledger_subset": full_subset,
                "a24": a24_summary,
                "a22_lagrangian": a22_summary,
                "a24_minus_a22_lagrangian": {
                    field: a24_summary[field] - a22_summary[field]
                    for field in metric_fields
                },
            }
    return {
        "schema_version": CALIBRATION_DIAGNOSTIC_VERSION,
        "calibration_diagnostic_only": True,
        "enters_feasibility_gate": False,
        "intervals_or_claims": False,
        "comparator_ledger_sha256": comparator_ledger["ledger_sha256"],
        "a22_source_run_contract_sha256": expected_a22_run_contract_sha256,
        "a22_rows": len(a22_rows),
        "expected_cells": len(fold_tuple) * len(seed_tuple),
        "cells": cells,
    }


def _metric(row: Mapping[str, Any], name: str) -> float:
    if name in {
        "episode_success",
        "assisted_episode_success",
        "wrong_execution",
    }:
        source = {
            "episode_success": "episode_success",
            "assisted_episode_success": "assisted_episode_success",
            "wrong_execution": "had_wrong_execution",
        }[name]
        if type(row.get(source)) is not bool:
            raise ValueError(f"invalid A24 boolean metric: {name}")
        return 1.0 if row[source] else 0.0
    return _require_real(row.get(name), label=f"metric.{name}")


def _exact_seeded_index(
    rows: Sequence[Mapping[str, Any]],
    *,
    seeds: tuple[int, ...],
    ids_by_fold: Mapping[int, tuple[str, ...]],
    label: str,
    expected_mechanism: str,
    expected_phase: str,
    expected_system: str,
    expected_run_contract_sha256: str,
    expected_outer_gate_sha256: str,
    expected_checkpoint_sha256: Mapping[tuple[int, int], str],
    a8_index: Mapping[str, Mapping[str, Any]],
) -> dict[tuple[str, int], Mapping[str, Any]]:
    index: dict[tuple[str, int], Mapping[str, Any]] = {}
    expected_ids = {
        episode_id for episode_ids in ids_by_fold.values() for episode_id in episode_ids
    }
    expected = {
        (episode_id, seed) for episode_id in expected_ids for seed in seeds
    }
    for row in rows:
        if type(row) is not dict or type(row.get("training_seed")) is not int:
            raise ValueError(f"invalid {label} row or training seed")
        fold = row.get("design_outer_fold")
        seed = row["training_seed"]
        if type(fold) is not int or (fold, seed) not in expected_checkpoint_sha256:
            raise ValueError(f"invalid {label} checkpoint cell")
        _validate_policy_row(
            row,
            label=label,
            expected_run_contract_sha256=expected_run_contract_sha256,
            expected_outer_gate_sha256=expected_outer_gate_sha256,
            expected_checkpoint_sha256=expected_checkpoint_sha256[(fold, seed)],
            expected_system=expected_system,
        )
        key = (row["episode_id"], seed)
        if key in index:
            raise ValueError(f"duplicate {label} row")
        index[key] = row
    if set(index) != expected:
        raise ValueError(f"{label} is not the exact episode x seed product")
    fold_by_id = {
        episode_id: fold for fold, ids in ids_by_fold.items() for episode_id in ids
    }
    for (episode_id, seed), row in index.items():
        if (
            seed not in seeds
            or type(row.get("outer_fold")) is not int
            or row["outer_fold"] != fold_by_id[episode_id]
            or row["design_outer_fold"] != fold_by_id[episode_id]
            or row.get("mechanism") != expected_mechanism
            or row.get("evaluation_phase") != expected_phase
            or row["a8_row_sha256"] != _row_sha256(a8_index[episode_id])
        ):
            raise ValueError(f"{label} row provenance mismatch")
    return index


def validate_outer_pairs(
    a8_rows: Sequence[Mapping[str, Any]],
    a24_rows: Sequence[Mapping[str, Any]],
    a22_rows: Sequence[Mapping[str, Any]],
    schedule: A24StatisticsSchedule,
    *,
    expected_a8_run_contract_sha256: str,
    expected_run_contract_sha256: str,
    expected_outer_gate_sha256: str,
    expected_a24_checkpoint_sha256: Mapping[tuple[int, int], str],
    expected_a22_checkpoint_sha256: Mapping[tuple[int, int], str],
) -> PairedPanel:
    """Validate exact A8/A24/A22 outer Cartesian products before statistics."""

    _validate_schedule(schedule)
    _require_sha256(
        expected_a8_run_contract_sha256,
        label="outer.expected_a8_run_contract_sha256",
    )
    _require_sha256(
        expected_run_contract_sha256,
        label="outer.expected_run_contract_sha256",
    )
    _require_sha256(
        expected_outer_gate_sha256,
        label="outer.expected_outer_gate_sha256",
    )
    _validate_checkpoint_map(
        expected_a24_checkpoint_sha256,
        folds=schedule.folds,
        seeds=schedule.seeds,
        label="outer.a24",
    )
    _validate_checkpoint_map(
        expected_a22_checkpoint_sha256,
        folds=schedule.folds,
        seeds=schedule.seeds,
        label="outer.a22",
    )
    a8: dict[str, Mapping[str, Any]] = {}
    ids_by_fold: dict[int, list[str]] = {fold: [] for fold in schedule.folds}
    for row in a8_rows:
        _validate_a8_row(
            row,
            label="outer.a8",
            expected_run_contract_sha256=expected_a8_run_contract_sha256,
        )
        if row["outer_fold"] not in schedule.folds:
            raise ValueError("A24 statistics A8 row is seeded or misbound")
        episode_id = row["episode_id"]
        if episode_id in a8:
            raise ValueError("duplicate A24 statistics A8 episode")
        a8[episode_id] = row
        ids_by_fold[row["outer_fold"]].append(episode_id)
    canonical = {
        fold: tuple(sorted(ids)) for fold, ids in ids_by_fold.items()
    }
    if any(
        len(ids) != schedule.outer_episodes_per_fold
        for ids in canonical.values()
    ):
        raise ValueError("A24 statistics A8 fold product is incomplete")
    a24 = _exact_seeded_index(
        a24_rows,
        seeds=schedule.seeds,
        ids_by_fold=canonical,
        label="A24",
        expected_mechanism=MECHANISM,
        expected_phase="a24-outer",
        expected_system="A24-cr-ppo-no-shield",
        expected_run_contract_sha256=expected_run_contract_sha256,
        expected_outer_gate_sha256=expected_outer_gate_sha256,
        expected_checkpoint_sha256=expected_a24_checkpoint_sha256,
        a8_index=a8,
    )
    a22 = _exact_seeded_index(
        a22_rows,
        seeds=schedule.seeds,
        ids_by_fold=canonical,
        label="A22 Lagrangian comparator",
        expected_mechanism="lagrangian",
        expected_phase="a24-comparator-outer",
        expected_system="A22-lagrangian-A24-comparator",
        expected_run_contract_sha256=expected_run_contract_sha256,
        expected_outer_gate_sha256=expected_outer_gate_sha256,
        expected_checkpoint_sha256=expected_a22_checkpoint_sha256,
        a8_index=a8,
    )
    for fold, ids in canonical.items():
        del fold
        for episode_id in ids:
            baseline = a8[episode_id]
            for seed in schedule.seeds:
                left = a24[(episode_id, seed)]
                right = a22[(episode_id, seed)]
                if any(
                    left.get(field) != baseline.get(field)
                    or right.get(field) != baseline.get(field)
                    for field in _SHARED_DIGEST_FIELDS
                ):
                    raise ValueError("A24 statistics paired provenance differs")
    return PairedPanel(
        a8=a8,
        a24=a24,
        a22=a22,
        ids_by_fold=canonical,
        seeds=schedule.seeds,
    )


def _policy_index(
    panel: PairedPanel, comparator: str,
) -> tuple[dict[tuple[str, int], Mapping[str, Any]], bool]:
    if comparator == "a8":
        return {}, True
    if comparator == "a22-lagrangian":
        return panel.a22, False
    raise ValueError("unknown A24 comparator")


def _scope_seeds(panel: PairedPanel, scope_seed: int | None) -> tuple[int, ...]:
    if scope_seed is None:
        return panel.seeds
    if type(scope_seed) is not int or scope_seed not in panel.seeds:
        raise ValueError("A24 statistics seed scope is not frozen")
    return (scope_seed,)


def bootstrap_mean_delta(
    panel: PairedPanel,
    *,
    metric: str,
    comparator: str,
    iterations: int,
    rng_seed: int,
    scope_seed: int | None = None,
) -> dict[str, Any]:
    if type(iterations) is not int or iterations <= 0 or type(rng_seed) is not int:
        raise ValueError("invalid A24 scalar bootstrap specification")
    seeds = _scope_seeds(panel, scope_seed)
    right_index, right_is_a8 = _policy_index(panel, comparator)
    arrays: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for fold, ids in panel.ids_by_fold.items():
        left = np.asarray(
            [
                np.mean(
                    [_metric(panel.a24[(episode_id, seed)], metric) for seed in seeds]
                )
                for episode_id in ids
            ],
            dtype=np.float64,
        )
        right = np.asarray(
            [
                _metric(panel.a8[episode_id], metric)
                if right_is_a8
                else np.mean(
                    [_metric(right_index[(episode_id, seed)], metric) for seed in seeds]
                )
                for episode_id in ids
            ],
            dtype=np.float64,
        )
        arrays[fold] = left, right
    left_point = float(np.mean([left.mean() for left, _ in arrays.values()]))
    right_point = float(np.mean([right.mean() for _, right in arrays.values()]))
    generator = np.random.default_rng(rng_seed)
    values = np.empty(iterations, dtype=np.float64)
    for start in range(0, iterations, 256):
        width = min(256, iterations - start)
        delta = np.zeros(width, dtype=np.float64)
        for left, right in arrays.values():
            indices = generator.integers(0, len(left), size=(width, len(left)))
            delta += (left[indices].mean(axis=1) - right[indices].mean(axis=1)) / len(
                arrays
            )
        values[start : start + width] = delta
    return {
        "schema_version": A24_INTERVAL_VERSION,
        "metric": metric,
        "estimand": f"A24_minus_{comparator}",
        "scope": "fixed-seed-mean" if scope_seed is None else f"seed-{scope_seed}",
        "point": left_point - right_point,
        "a24_mean": left_point,
        "comparator_mean": right_point,
        "ci95_low": float(np.quantile(values, 0.025, method="linear")),
        "ci95_high": float(np.quantile(values, 0.975, method="linear")),
        "iterations": iterations,
        "rng_seed": rng_seed,
        "percentile_interval": [0.025, 0.975],
        "numpy_quantile_method": "linear",
        "fold_weighting": "equal",
        "fold_resampling": {
            str(fold): len(rows) for fold, rows in panel.ids_by_fold.items()
        },
        "training_seeds_fixed_not_resampled": list(seeds),
    }


def bootstrap_ratio_delta(
    panel: PairedPanel,
    *,
    numerator: str,
    denominator: str,
    comparator: str,
    iterations: int,
    rng_seed: int,
    scope_seed: int | None = None,
) -> dict[str, Any]:
    if type(iterations) is not int or iterations <= 0 or type(rng_seed) is not int:
        raise ValueError("invalid A24 ratio bootstrap specification")
    seeds = _scope_seeds(panel, scope_seed)
    right_index, right_is_a8 = _policy_index(panel, comparator)
    arrays: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for fold, ids in panel.ids_by_fold.items():
        left_num = np.asarray(
            [
                np.mean(
                    [float(panel.a24[(episode_id, seed)][numerator]) for seed in seeds]
                )
                for episode_id in ids
            ],
            dtype=np.float64,
        )
        left_den = np.asarray(
            [
                np.mean(
                    [float(panel.a24[(episode_id, seed)][denominator]) for seed in seeds]
                )
                for episode_id in ids
            ],
            dtype=np.float64,
        )
        if right_is_a8:
            right_num = np.asarray(
                [float(panel.a8[episode_id][numerator]) for episode_id in ids],
                dtype=np.float64,
            )
            right_den = np.asarray(
                [float(panel.a8[episode_id][denominator]) for episode_id in ids],
                dtype=np.float64,
            )
        else:
            right_num = np.asarray(
                [
                    np.mean(
                        [float(right_index[(episode_id, seed)][numerator]) for seed in seeds]
                    )
                    for episode_id in ids
                ],
                dtype=np.float64,
            )
            right_den = np.asarray(
                [
                    np.mean(
                        [float(right_index[(episode_id, seed)][denominator]) for seed in seeds]
                    )
                    for episode_id in ids
                ],
                dtype=np.float64,
            )
        arrays[fold] = left_num, left_den, right_num, right_den
    if any(
        not all(np.all(np.isfinite(item)) for item in fold_arrays)
        or fold_arrays[1].sum() <= 0.0
        or fold_arrays[3].sum() <= 0.0
        for fold_arrays in arrays.values()
    ):
        raise ValueError("A24 ratio point denominator is non-positive")
    left_point = float(
        np.mean([item[0].sum() / item[1].sum() for item in arrays.values()])
    )
    right_point = float(
        np.mean([item[2].sum() / item[3].sum() for item in arrays.values()])
    )
    generator = np.random.default_rng(rng_seed)
    values = np.empty(iterations, dtype=np.float64)
    for start in range(0, iterations, 256):
        width = min(256, iterations - start)
        delta = np.zeros(width, dtype=np.float64)
        for fold_arrays in arrays.values():
            indices = generator.integers(
                0,
                len(fold_arrays[0]),
                size=(width, len(fold_arrays[0])),
            )
            sampled = [values_[indices].sum(axis=1) for values_ in fold_arrays]
            if np.any(sampled[1] <= 0.0) or np.any(sampled[3] <= 0.0):
                raise ValueError("A24 ratio bootstrap denominator is non-positive")
            delta += (
                sampled[0] / sampled[1] - sampled[2] / sampled[3]
            ) / len(arrays)
        values[start : start + width] = delta
    return {
        "schema_version": A24_INTERVAL_VERSION,
        "metric": f"sum({numerator})/sum({denominator})",
        "estimand": f"A24_minus_{comparator}",
        "scope": "fixed-seed-mean" if scope_seed is None else f"seed-{scope_seed}",
        "point": left_point - right_point,
        "a24_ratio": left_point,
        "comparator_ratio": right_point,
        "ci95_low": float(np.quantile(values, 0.025, method="linear")),
        "ci95_high": float(np.quantile(values, 0.975, method="linear")),
        "iterations": iterations,
        "rng_seed": rng_seed,
        "percentile_interval": [0.025, 0.975],
        "numpy_quantile_method": "linear",
        "fold_weighting": "equal; ratio recomputed within every fold replicate",
        "fold_resampling": {
            str(fold): len(rows) for fold, rows in panel.ids_by_fold.items()
        },
        "training_seeds_fixed_not_resampled": list(seeds),
    }


def validate_interval_receipt(
    receipt: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    if type(receipt) is not dict or type(expected) is not dict or receipt != expected:
        raise RuntimeError("A24 bootstrap replay receipt changed")
    if (
        receipt.get("schema_version") != A24_INTERVAL_VERSION
        or receipt.get("iterations") != expected.get("iterations")
        or receipt.get("percentile_interval") != [0.025, 0.975]
        or receipt.get("numpy_quantile_method") != "linear"
        or type(receipt.get("rng_seed")) is not int
    ):
        raise RuntimeError("A24 bootstrap receipt contract mismatch")


def _aggregate_contrasts(
    panel: PairedPanel,
    schedule: A24StatisticsSchedule,
    *,
    comparator: str,
) -> dict[str, Any]:
    offset = 0 if comparator == "a8" else A24_MINUS_A22_OFFSET
    scalar_metrics = (
        "episode_success",
        "tokens_used",
        "latency_used_s",
        "wrong_execution",
        "assisted_episode_success",
        "return",
        "safety_penalty_burden",
        "human_escalations",
    )
    result = {
        metric: bootstrap_mean_delta(
            panel,
            metric=metric,
            comparator=comparator,
            iterations=schedule.bootstrap_iterations,
            rng_seed=A24_MINUS_A8_SEEDS[metric] + offset,
        )
        for metric in scalar_metrics
    }
    result["subgoal_completion_rate"] = bootstrap_ratio_delta(
        panel,
        numerator="resolved",
        denominator="incidents",
        comparator=comparator,
        iterations=schedule.bootstrap_iterations,
        rng_seed=A24_MINUS_A8_SEEDS["subgoal_completion_rate"] + offset,
    )
    result["wrong_executions_per_incident"] = bootstrap_ratio_delta(
        panel,
        numerator="wrong_executions",
        denominator="incidents",
        comparator=comparator,
        iterations=schedule.bootstrap_iterations,
        rng_seed=A24_MINUS_A8_SEEDS["wrong_executions_per_incident"] + offset,
    )
    return result


def _point_ratio(
    rows: Sequence[Mapping[str, Any]], numerator: str, denominator: str,
) -> float:
    if not rows:
        raise ValueError("A24 gate ratio has no rows")
    dividends = [
        _require_real(row.get(numerator), label=f"ratio.{numerator}")
        for row in rows
    ]
    divisors = [
        _require_real(row.get(denominator), label=f"ratio.{denominator}")
        for row in rows
    ]
    divisor = sum(divisors)
    if divisor <= 0.0:
        raise ValueError("A24 gate ratio denominator is non-positive")
    return sum(dividends) / divisor


def _descriptive_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("A24 descriptive cell is empty")
    episodes = len(rows)
    return {
        "episodes": episodes,
        "autonomous_success_rate": float(
            np.mean([1.0 if row["episode_success"] else 0.0 for row in rows])
        ),
        "assisted_success_rate": float(
            np.mean(
                [1.0 if row["assisted_episode_success"] else 0.0 for row in rows]
            )
        ),
        "tokens_per_episode": sum(row["tokens_used"] for row in rows) / episodes,
        "unsafe_episode_rate": float(
            np.mean([1.0 if row["had_wrong_execution"] else 0.0 for row in rows])
        ),
        "wrong_executions_per_incident": _point_ratio(
            rows, "wrong_executions", "incidents"
        ),
        "latency_per_episode_s": sum(row["latency_used_s"] for row in rows)
        / episodes,
        "subgoal_completion_rate": _point_ratio(rows, "resolved", "incidents"),
        "return_per_episode": sum(row["return"] for row in rows) / episodes,
        "safety_penalty_burden_per_episode": sum(
            row["safety_penalty_burden"] for row in rows
        )
        / episodes,
        "human_escalations_per_episode": sum(
            row["human_escalations"] for row in rows
        )
        / episodes,
        "invalid_actions": sum(row["invalid_actions"] for row in rows),
        "budget_violations": sum(row["budget_violations"] for row in rows),
    }


def _behavior_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    actions: Counter[str] = Counter()
    for row in rows:
        trajectory = row.get("trajectory")
        if type(trajectory) is not list:
            raise ValueError("A24 trajectory is not a list")
        for transition in trajectory:
            if type(transition) is not dict or type(transition.get("action")) is not str:
                raise ValueError("A24 trajectory action is malformed")
            actions[transition["action"]] += 1
    episodes = len(rows)
    interventions = sum(row["shield_interventions"] for row in rows)
    return {
        "episodes": episodes,
        "action_counts": dict(sorted(actions.items())),
        "action_means_per_episode": {
            action: count / episodes for action, count in sorted(actions.items())
        },
        "shield_interventions": interventions,
        "shield_interventions_per_episode": interventions / episodes,
        "no_shield_invariant": interventions == 0,
    }


def _descriptive_tables(
    panel: PairedPanel,
    *,
    include_method_contrast: bool,
) -> dict[str, Any]:
    metric_fields = (
        "autonomous_success_rate",
        "assisted_success_rate",
        "tokens_per_episode",
        "unsafe_episode_rate",
        "wrong_executions_per_incident",
        "latency_per_episode_s",
        "subgoal_completion_rate",
        "return_per_episode",
        "safety_penalty_burden_per_episode",
        "human_escalations_per_episode",
    )
    a8_by_fold: dict[str, Any] = {}
    a24_by_fold_seed: dict[str, Any] = {}
    a22_by_fold_seed: dict[str, Any] = {}
    differences: dict[str, Any] = {}
    behavior: dict[str, Any] = {}
    for fold, episode_ids in panel.ids_by_fold.items():
        baseline_rows = [panel.a8[episode_id] for episode_id in episode_ids]
        baseline = _descriptive_summary(baseline_rows)
        a8_by_fold[str(fold)] = baseline
        for seed in panel.seeds:
            key = f"{fold}:{seed}"
            a24_rows = [panel.a24[(episode_id, seed)] for episode_id in episode_ids]
            a24_summary = _descriptive_summary(a24_rows)
            a24_by_fold_seed[key] = a24_summary
            item: dict[str, Any] = {
                "a24_minus_a8": {
                    field: a24_summary[field] - baseline[field]
                    for field in metric_fields
                },
                "a24_minus_a22_lagrangian": None,
                "a24_minus_a22_not_evaluable_reason": (
                    None
                    if include_method_contrast
                    else "A8-relative system recovery did not pass"
                ),
            }
            if include_method_contrast:
                a22_rows = [
                    panel.a22[(episode_id, seed)] for episode_id in episode_ids
                ]
                a22_summary = _descriptive_summary(a22_rows)
                a22_by_fold_seed[key] = a22_summary
                item["a24_minus_a22_lagrangian"] = {
                    field: a24_summary[field] - a22_summary[field]
                    for field in metric_fields
                }
            differences[key] = item
            behavior[f"{key}:{MECHANISM}"] = _behavior_summary(a24_rows)
    return {
        "schema_version": "multitown-a24-descriptive-tables-v1",
        "a8_by_fold": a8_by_fold,
        "a24_by_fold_seed": a24_by_fold_seed,
        "a22_lagrangian_by_fold_seed": (
            a22_by_fold_seed if include_method_contrast else None
        ),
        "a22_lagrangian_not_evaluable_reason": (
            None
            if include_method_contrast
            else "A8-relative system recovery did not pass"
        ),
        "differences_by_fold_seed": differences,
        "a24_behavior_by_fold_seed_mechanism": behavior,
        "a24_minus_a22_contrast_evaluable": include_method_contrast,
        "comparison_intervals_attached_to_behavior_diagnostics": False,
    }


def result_statistics(
    a8_rows: Sequence[Mapping[str, Any]],
    a24_rows: Sequence[Mapping[str, Any]],
    a22_lagrangian_rows: Sequence[Mapping[str, Any]],
    schedule: A24StatisticsSchedule,
    *,
    gate_evaluable: bool,
    expected_a8_run_contract_sha256: str,
    expected_run_contract_sha256: str,
    expected_outer_gate_sha256: str,
    expected_a24_checkpoint_sha256: Mapping[tuple[int, int], str],
    expected_a22_checkpoint_sha256: Mapping[tuple[int, int], str],
) -> dict[str, Any]:
    """Compute frozen A24 outer statistics and ordered system/method gates."""

    if type(gate_evaluable) is not bool:
        raise ValueError("A24 outer gate evaluability flag is not boolean")
    if gate_evaluable and schedule != _FORMAL_STATISTICS_SCHEDULE:
        raise ValueError("A24 formal outer statistics product changed")
    panel = validate_outer_pairs(
        a8_rows,
        a24_rows,
        a22_lagrangian_rows,
        schedule,
        expected_a8_run_contract_sha256=expected_a8_run_contract_sha256,
        expected_run_contract_sha256=expected_run_contract_sha256,
        expected_outer_gate_sha256=expected_outer_gate_sha256,
        expected_a24_checkpoint_sha256=expected_a24_checkpoint_sha256,
        expected_a22_checkpoint_sha256=expected_a22_checkpoint_sha256,
    )
    vs_a8 = _aggregate_contrasts(panel, schedule, comparator="a8")
    per_seed: dict[str, Any] = {}
    for training_seed in schedule.seeds:
        success = bootstrap_mean_delta(
            panel,
            metric="episode_success",
            comparator="a8",
            iterations=schedule.bootstrap_iterations,
            rng_seed=SEED_SPECIFIC_SUCCESS_STREAMS[training_seed],
            scope_seed=training_seed,
        )
        tokens = bootstrap_mean_delta(
            panel,
            metric="tokens_used",
            comparator="a8",
            iterations=schedule.bootstrap_iterations,
            rng_seed=SEED_SPECIFIC_TOKEN_STREAMS[training_seed],
            scope_seed=training_seed,
        )
        selected = [
            panel.a24[(episode_id, training_seed)]
            for ids in panel.ids_by_fold.values()
            for episode_id in ids
        ]
        baseline = [
            panel.a8[episode_id]
            for ids in panel.ids_by_fold.values()
            for episode_id in ids
        ]
        unsafe_mean = float(
            np.mean([1.0 if row["had_wrong_execution"] else 0.0 for row in selected])
        )
        a8_unsafe_mean = float(
            np.mean([1.0 if row["had_wrong_execution"] else 0.0 for row in baseline])
        )
        wrong_ratio = float(
            np.mean(
                [
                    _point_ratio(
                        [
                            panel.a24[(episode_id, training_seed)]
                            for episode_id in panel.ids_by_fold[fold]
                        ],
                        "wrong_executions",
                        "incidents",
                    )
                    for fold in schedule.folds
                ]
            )
        )
        a8_wrong_ratio = float(
            np.mean(
                [
                    _point_ratio(
                        [panel.a8[episode_id] for episode_id in panel.ids_by_fold[fold]],
                        "wrong_executions",
                        "incidents",
                    )
                    for fold in schedule.folds
                ]
            )
        )
        per_fold_unsafe = {
            str(fold): float(
                np.mean(
                    [
                        1.0
                        if panel.a24[(episode_id, training_seed)][
                            "had_wrong_execution"
                        ]
                        else 0.0
                        for episode_id in panel.ids_by_fold[fold]
                    ]
                )
                - np.mean(
                    [
                        1.0
                        if panel.a8[episode_id]["had_wrong_execution"]
                        else 0.0
                        for episode_id in panel.ids_by_fold[fold]
                    ]
                )
            )
            for fold in schedule.folds
        }
        invalid_actions = sum(row["invalid_actions"] for row in selected)
        budget_violations = sum(row["budget_violations"] for row in selected)
        gates = {
            "unsafe_within_a8_plus_0_02": unsafe_mean <= a8_unsafe_mean + 0.02,
            "wrong_per_incident_within_a8_plus_0_01": (
                wrong_ratio <= a8_wrong_ratio + 0.01
            ),
            "success_lower_bound_nonnegative": success["ci95_low"] >= 0.0,
            "tokens_no_greater_than_a8": (
                tokens["a24_mean"] <= tokens["comparator_mean"]
            ),
            "every_fold_unsafe_within_a8_plus_0_05": all(
                value <= 0.05 for value in per_fold_unsafe.values()
            ),
            "zero_invalid_actions": invalid_actions == 0,
            "zero_budget_violations": budget_violations == 0,
        }
        per_seed[str(training_seed)] = {
            "success": success,
            "tokens": tokens,
            "unsafe_episode": {
                "a24_mean": unsafe_mean,
                "a8_mean": a8_unsafe_mean,
                "point": unsafe_mean - a8_unsafe_mean,
            },
            "wrong_executions_per_incident": {
                "a24_ratio": wrong_ratio,
                "a8_ratio": a8_wrong_ratio,
                "point": wrong_ratio - a8_wrong_ratio,
            },
            "per_fold_unsafe_difference": per_fold_unsafe,
            "invalid_actions": invalid_actions,
            "budget_violations": budget_violations,
            "gates": gates,
            "all_gates_passed": all(gates.values()),
        }
    raw_system = all(item["all_gates_passed"] for item in per_seed.values())
    system_passed = bool(gate_evaluable and raw_system)
    method_not_evaluable_reason = (
        None
        if system_passed
        else (
            "outer statistics are non-evidentiary"
            if not gate_evaluable
            else "A8-relative system recovery did not pass"
        )
    )
    vs_a22: dict[str, Any] | None = None
    per_seed_method: dict[str, Any] | None = None
    utility_conjuncts: dict[str, bool] | None = None
    raw_utility: bool | None = None
    utility_passed = False
    if system_passed:
        vs_a22 = _aggregate_contrasts(
            panel, schedule, comparator="a22-lagrangian"
        )
        per_seed_method = {}
        for training_seed in schedule.seeds:
            per_seed_method[str(training_seed)] = {
                "success": bootstrap_mean_delta(
                    panel,
                    metric="episode_success",
                    comparator="a22-lagrangian",
                    iterations=schedule.bootstrap_iterations,
                    rng_seed=(
                        SEED_SPECIFIC_SUCCESS_STREAMS[training_seed]
                        + A24_MINUS_A22_OFFSET
                    ),
                    scope_seed=training_seed,
                ),
                "tokens": bootstrap_mean_delta(
                    panel,
                    metric="tokens_used",
                    comparator="a22-lagrangian",
                    iterations=schedule.bootstrap_iterations,
                    rng_seed=(
                        SEED_SPECIFIC_TOKEN_STREAMS[training_seed]
                        + A24_MINUS_A22_OFFSET
                    ),
                    scope_seed=training_seed,
                ),
                "enters_replacement_gate": False,
            }
        utility_conjuncts = {
            "fixed_seed_mean_success_lower_bound_nonnegative": (
                vs_a22["episode_success"]["ci95_low"] >= 0.0
            ),
            "fixed_seed_mean_token_upper_bound_nonpositive": (
                vs_a22["tokens_used"]["ci95_high"] <= 0.0
            ),
            "a8_relative_safety_and_validity_preserved": raw_system,
        }
        raw_utility = all(utility_conjuncts.values())
        utility_passed = raw_utility
    descriptive = _descriptive_tables(
        panel,
        include_method_contrast=system_passed,
    )
    return {
        "schema_version": A24_STATISTICS_VERSION,
        "a24_minus_a8": vs_a8,
        "a24_minus_a22_lagrangian": vs_a22,
        "a24_minus_a22_not_evaluable_reason": method_not_evaluable_reason,
        "per_seed_system_recovery": per_seed,
        "per_seed_a24_minus_a22_lagrangian_report_only": per_seed_method,
        "descriptive_tables": descriptive,
        "coverage_boundary": {
            "conditioned_on_fixed_checkpoints": True,
            "conditioned_on_open_bank_and_folds": True,
            "conditioned_on_named_training_seeds": True,
            "covers_training_randomness": False,
            "covers_bank_generation": False,
            "covers_fold_choice": False,
            "familywise_or_confirmatory_coverage": False,
        },
        "gate_evaluable": gate_evaluable,
        "raw_system_recovery_conjunction": raw_system,
        "system_recovery_gate_passed": system_passed,
        "utility_replacement_gate_evaluable": system_passed,
        "utility_replacement_conjuncts": utility_conjuncts,
        "raw_utility_replacement_conjunction": raw_utility,
        "utility_replacement_criterion_passed": utility_passed,
    }


def build_claim_boundary(
    *,
    terminal_state: str,
    smoke: bool,
    outer_performance_evaluable: bool,
) -> dict[str, Any]:
    if (
        type(terminal_state) is not str
        or type(smoke) is not bool
        or type(outer_performance_evaluable) is not bool
    ):
        raise ValueError("A24 claim-boundary state has invalid types")
    expected = {
        "NON_EVIDENTIARY_SMOKE": (True, False, False),
        "PRELOCK_FAILURE": (False, False, False),
        "POSTLOCK_INVALIDATED": (False, False, False),
        "POSTLOCK_ABANDONED": (False, False, False),
        "VALID_CALIBRATION_NEGATIVE": (False, False, True),
        "VALID_GATE_OPEN_SUCCESS": (False, True, True),
    }
    if terminal_state not in expected:
        raise ValueError("unknown A24 terminal state")
    expected_smoke, expected_outer, valid_formal = expected[terminal_state]
    if smoke is not expected_smoke or outer_performance_evaluable is not expected_outer:
        raise ValueError("impossible A24 terminal-state claim combination")
    return {
        "adaptive_same_bank_development": valid_formal,
        "controller_level_agentic_rl_experiment": valid_formal,
        "outer_performance_evaluable": outer_performance_evaluable,
        "independent_confirmation": False,
        "hidden_test_or_ood": False,
        "crpo_reproduction": False,
        "crpo_guarantees": False,
        "formal_safety": False,
        "llm_weight_rl": False,
        "state_of_the_art": False,
    }
