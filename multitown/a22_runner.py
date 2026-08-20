"""Run the frozen same-bank A22 constrained-PPO adaptive development study."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .a22_constrained_ppo import (
    A22_PRIMITIVES_VERSION, DUAL_STEP_SIZE, MECHANISMS, DualState, Mechanism,
    constrained_rollout, constrained_update, deterministic_action,
    thresholds_from_inner_train,
)
from .a9_oof_protocol import (
    DEFAULT_FOLDS, FROZEN_TRAIN_PATH, assign_stratified_group_folds,
    fold_manifest_sha256, load_frozen_train_bank, resource_contract_sha256,
    shared_resource_contract,
)
from .a9_ppo_oof import (
    BOOTSTRAP_SEED, FORMAL_EPISODES_PER_UPDATE, FORMAL_SEEDS, FORMAL_THREADS,
    FORMAL_UPDATES, POLICY_VERSION as A9_POLICY_VERSION, RunSchedule, _digest,
    _evaluate_episode, _formal_ppo_config, _paired_effects, _save_checkpoint,
    _set_seed, _sha256, _write_json, _write_jsonl, fold_cluster_bootstrap,
    fold_cluster_ratio_bootstrap,
)
from .a9_safety_development import (
    EXPECTED_R2_MANIFEST_SHA256, FROZEN_R2_PATH, _validate_shared_stack_bindings,
    _verify_r2_artifacts,
)
from .long_horizon_env import (
    ACTION_COUNT, ACTION_NAMES, LongHorizonEpisode, MultiTownLongHorizonEnv,
)
from .ppo_controller import ActorCritic, PPOConfig, load_checkpoint


RUNNER_VERSION = "multitown-a22-adaptive-nested-constrained-ppo-runner-v2"
POLICY_VERSION = "multitown-a22-constrained-masked-ppo-policy-v1"
RESULT_VERSION = "multitown-a22-adaptive-nested-development-result-v2"
EXPECTED_R2_A8_SHA256 = (
    "9bf3ed9148d58c1c49f85807f063c19946de8e12d982d6d04b92865153c0ed40"
)
EXPECTED_R2_RUN_FILE_SHA256 = (
    "37d9a17319e0ad768c63afae05dbaf419faf7e509fb9679048a8fa0c473cfbea"
)
EXPECTED_R2_RUN_DIGEST = (
    "7694d2816abd31753cfa68eec5020ab6da37eb18e4f77beed5714c544a0e51e1"
)
FORMAL_ATTEMPT_LOCK = Path("artifacts/a22-adaptive-attempt-v1.lock")


@dataclass(frozen=True)
class A22Schedule:
    mode: str
    seeds: tuple[int, ...]
    folds: tuple[int, ...]
    updates: int
    episodes_per_update: int
    calibration_episodes_per_fold: int
    outer_episodes_per_fold: int
    bootstrap_iterations: int
    bootstrap_seed: int
    threads: int


def schedule(smoke: bool) -> A22Schedule:
    if smoke:
        return A22Schedule(
            mode="smoke", seeds=(FORMAL_SEEDS[0],),
            folds=tuple(range(DEFAULT_FOLDS)), updates=1,
            episodes_per_update=4, calibration_episodes_per_fold=8,
            outer_episodes_per_fold=8, bootstrap_iterations=200,
            bootstrap_seed=BOOTSTRAP_SEED, threads=2,
        )
    return A22Schedule(
        mode="adaptive-development", seeds=FORMAL_SEEDS,
        folds=tuple(range(DEFAULT_FOLDS)), updates=FORMAL_UPDATES,
        episodes_per_update=FORMAL_EPISODES_PER_UPDATE,
        calibration_episodes_per_fold=600, outer_episodes_per_fold=600,
        bootstrap_iterations=20_000, bootstrap_seed=BOOTSTRAP_SEED,
        threads=FORMAL_THREADS,
    )


def partition_ids(
    assignments: Sequence[Any], outer_fold: int,
) -> dict[str, Any]:
    calibration_fold = (outer_fold + 1) % DEFAULT_FOLDS
    outer = sorted(row.episode_id for row in assignments if row.fold == outer_fold)
    calibration = sorted(
        row.episode_id for row in assignments if row.fold == calibration_fold
    )
    train = sorted(
        row.episode_id for row in assignments
        if row.fold not in {outer_fold, calibration_fold}
    )
    sets = [set(train), set(calibration), set(outer)]
    if (
        [len(train), len(calibration), len(outer)] != [1800, 600, 600]
        or any(sets[left] & sets[right] for left, right in ((0, 1), (0, 2), (1, 2)))
        or len(set().union(*sets)) != 3000
    ):
        raise RuntimeError("A22 inner-train/calibration/outer partition invalid")
    return {
        "outer_fold": outer_fold,
        "calibration_fold": calibration_fold,
        "inner_train_ids": train,
        "inner_calibration_ids": calibration,
        "outer_ids": outer,
        "inner_train_ids_sha256": _digest(train),
        "inner_calibration_ids_sha256": _digest(calibration),
        "outer_ids_sha256": _digest(outer),
    }


def _source_state(root: Path, *, require_clean: bool) -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    if require_clean and status:
        raise RuntimeError("A22 adaptive run requires a clean source checkout")
    required = (
        "multitown/a22_constrained_ppo.py",
        "multitown/a22_runner.py",
        "multitown/a9_oof_protocol.py",
        "multitown/a9_ppo_oof.py",
        "multitown/a9_long_horizon_env.py",
        "multitown/long_horizon_env.py",
        "multitown/ppo_controller.py",
        "docs/A22_A9_SAFETY_FIRST_FOLLOWUP.md",
        "pyproject.toml",
    )
    hashes = {}
    for relative in required:
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"missing A22 source: {relative}")
        if require_clean:
            head = subprocess.run(
                ["git", "show", f"HEAD:{relative}"], cwd=root,
                capture_output=True, check=True,
            ).stdout
            if path.read_bytes() != head:
                raise RuntimeError(f"executed A22 source differs from HEAD: {relative}")
        hashes[relative] = _sha256(path)
    return {
        "revision": revision, "dirty": bool(status),
        "executed_source_sha256": hashes,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"missing A22 JSONL artifact: {path}")
    try:
        return [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid A22 JSONL artifact: {path}") from exc


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f"{path.name}.partial")
    if temporary.exists():
        raise FileExistsError(temporary)
    _write_json(temporary, value)
    os.replace(temporary, path)


def _fit_key(row: Mapping[str, Any]) -> tuple[int, int, str]:
    return (
        int(row["outer_fold"]), int(row["training_seed"]),
        str(row["mechanism"]),
    )


def _fit_path(
    output: Path, outer_fold: int, training_seed: int, mechanism: str,
) -> Path:
    return (
        output / "fits" / f"outer-fold-{outer_fold}"
        / f"seed-{training_seed}" / mechanism
    )


def _assert_exact_keys(
    actual: Sequence[tuple[Any, ...]], expected: set[tuple[Any, ...]], *, label: str,
) -> None:
    actual_set = set(actual)
    if len(actual) != len(expected) or actual_set != expected:
        missing = sorted(expected - actual_set, key=repr)[:5]
        extra = sorted(actual_set - expected, key=repr)[:5]
        duplicates = len(actual) - len(actual_set)
        raise RuntimeError(
            f"A22 {label} key coverage mismatch: missing={missing}, "
            f"extra={extra}, duplicate_count={duplicates}"
        )


def _formal_config(run_schedule: A22Schedule) -> PPOConfig:
    base = asdict(_formal_ppo_config())
    base.update({
        "updates": run_schedule.updates,
        "episodes_per_update": run_schedule.episodes_per_update,
    })
    return PPOConfig(**base)


def _expected_sample_sequence(
    train_ids: Sequence[str], *, seed: int, draws: int,
) -> list[str]:
    canonical = [str(item) for item in train_ids]
    if (
        canonical != sorted(canonical) or len(canonical) != 1800
        or len(set(canonical)) != len(canonical) or draws <= 0
    ):
        raise ValueError("invalid A22 canonical sampler input")
    generator = random.Random(seed)
    return [canonical[generator.randrange(len(canonical))] for _ in range(draws)]


def _validate_checkpoint(
    checkpoint: Path, *, fit: Mapping[str, Any], mechanism: Mechanism,
    outer_fold: int, training_seed: int, run_schedule: A22Schedule,
    run_contract_sha256: str,
) -> tuple[ActorCritic, dict[str, Any]]:
    if not checkpoint.is_file() or _sha256(checkpoint) != fit["checkpoint_sha256"]:
        raise RuntimeError("A22 final checkpoint disk hash mismatch")
    model, metadata = load_checkpoint(
        checkpoint, torch.device("cpu"), expected_policy_version=POLICY_VERSION,
    )
    expected = {
        "seed": training_seed,
        "update": run_schedule.updates,
        "run_contract_sha256": run_contract_sha256,
        "mechanism": asdict(mechanism),
        "outer_fold": outer_fold,
        "ppo_config": asdict(_formal_config(run_schedule)),
        "safety_thresholds": fit["thresholds"],
        "final_dual": fit["final_dual"],
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise RuntimeError("A22 final checkpoint provenance metadata mismatch")
    if any(
        not bool(torch.isfinite(parameter).all()) for parameter in model.parameters()
    ):
        raise RuntimeError("A22 final checkpoint contains non-finite parameters")
    return model, metadata


def _validate_fits(
    fits: Sequence[Mapping[str, Any]], *, output: Path,
    run_schedule: A22Schedule, partitions: Mapping[int, Mapping[str, Any]],
    a8_index: Mapping[str, Mapping[str, Any]],
    episode_index: Mapping[str, LongHorizonEpisode],
    run_contract_sha256: str,
) -> dict[tuple[int, int, str], Mapping[str, Any]]:
    mechanism_index = {item.name: item for item in MECHANISMS}
    expected_keys = {
        (fold, seed, mechanism.name)
        for fold in run_schedule.folds for seed in run_schedule.seeds
        for mechanism in MECHANISMS
    }
    _assert_exact_keys([_fit_key(row) for row in fits], expected_keys, label="fit")
    fit_index = {_fit_key(row): row for row in fits}
    sample_sequences: dict[tuple[int, int], set[str]] = {}
    for key in sorted(expected_keys):
        fold, training_seed, mechanism_name = key
        mechanism = mechanism_index[mechanism_name]
        fit = fit_index[key]
        partition = partitions[fold]
        expected_draws = run_schedule.updates * run_schedule.episodes_per_update
        train_ids_ordered = [str(item) for item in partition["inner_train_ids"]]
        expected_sequence = _expected_sample_sequence(
            train_ids_ordered, seed=training_seed, draws=expected_draws,
        )
        expected_thresholds = asdict(thresholds_from_inner_train([
            a8_index[episode_id] for episode_id in train_ids_ordered
        ]))
        expected_fields = {
            "dual_enabled": mechanism.dual_enabled,
            "shield_enabled": mechanism.shield_enabled,
            "final_update": run_schedule.updates,
            "training_episode_draws": expected_draws,
            "inner_train_ids_sha256": partition["inner_train_ids_sha256"],
            "calibration_ids_sha256": partition["inner_calibration_ids_sha256"],
            "outer_ids_sha256": partition["outer_ids_sha256"],
            "run_contract_sha256": run_contract_sha256,
            "calibration_evaluations_during_training": 0,
            "outer_evaluations_during_training": 0,
            "selected_checkpoint": "final",
            "thresholds": expected_thresholds,
        }
        if any(fit.get(field) != value for field, value in expected_fields.items()):
            raise RuntimeError("A22 fit completion contract mismatch")
        if (
            not math.isfinite(float(fit["training_seconds"]))
            or float(fit["training_seconds"]) < 0.0
        ):
            raise RuntimeError("A22 fit duration is invalid")
        fit_dir = _fit_path(output, fold, training_seed, mechanism_name)
        persisted = json.loads(
            (fit_dir / "fit-complete.json").read_text(encoding="utf-8")
        )
        if persisted != fit:
            raise RuntimeError("A22 fit-complete artifact differs from fit index")
        logs = _read_jsonl(fit_dir / "training-metrics.jsonl")
        if len(logs) != run_schedule.updates:
            raise RuntimeError("A22 training update log count mismatch")
        sampled_sequence: list[str] = []
        train_ids = set(partition["inner_train_ids"])
        carried_dual = DualState()
        total_steps = 0
        total_minibatches = 0
        for expected_update, row in enumerate(logs, start=1):
            sampled = [str(item) for item in row["sampled_episode_ids"]]
            update_incidents = sum(
                len(episode_index[episode_id].incidents) for episode_id in sampled
            )
            unsafe_events = float(row["rollout_unsafe_events"])
            wrong_executions = float(row["rollout_wrong_executions"])
            unsafe_rate = unsafe_events / run_schedule.episodes_per_update
            wrong_rate = (
                wrong_executions / run_schedule.episodes_per_update
                / expected_thresholds["mean_incidents"]
            )
            steps = int(row["environment_steps"])
            minibatches = (
                _formal_config(run_schedule).ppo_epochs
                * math.ceil(steps / _formal_config(run_schedule).minibatch_size)
            )
            scale_fields = (
                "reward_advantage_raw_mean", "reward_advantage_raw_std",
                "reward_advantage_raw_max_abs", "unsafe_cost_return_mean",
                "unsafe_cost_return_std", "unsafe_cost_return_max",
                "wrong_cost_return_mean", "wrong_cost_return_std",
                "wrong_cost_return_max", "combined_advantage_raw_mean",
                "combined_advantage_raw_std", "combined_advantage_raw_max_abs",
            )
            if (
                int(row["outer_fold"]) != fold
                or int(row["training_seed"]) != training_seed
                or str(row["mechanism"]) != mechanism_name
                or int(row["update"]) != expected_update
                or len(sampled) != run_schedule.episodes_per_update
                or not set(sampled) <= train_ids
                or row["sampled_episode_ids_sha256"] != _digest(sampled)
                or int(row["rollout_incidents"]) != update_incidents
                or unsafe_events != int(unsafe_events)
                or not 0 <= unsafe_events <= run_schedule.episodes_per_update
                or wrong_executions != int(wrong_executions)
                or wrong_executions < unsafe_events
                or float(row["rollout_episodes"])
                != float(run_schedule.episodes_per_update)
                or not math.isclose(
                    float(row["rollout_unsafe_rate"]), unsafe_rate,
                    rel_tol=1e-12, abs_tol=1e-12,
                )
                or not math.isclose(
                    float(row["rollout_wrong_per_fixed_mean_incident"]),
                    wrong_rate, rel_tol=1e-12, abs_tol=1e-12,
                )
                or int(row["optimizer_minibatches"]) != minibatches
                or steps <= 0
                or int(row["shield_interventions"]) < 0
                or row.get("cost_return_scale_version")
                != "raw-transition-cost-to-go-summary-v1"
                or any(
                    not math.isfinite(float(row[field])) for field in scale_fields
                )
                or any(
                    isinstance(value, (int, float))
                    and not math.isfinite(float(value))
                    for value in row.values()
                )
            ):
                raise RuntimeError("A22 training sample stream binding mismatch")
            dual_before = DualState(
                unsafe=float(row["lambda_unsafe_before"]),
                wrong_per_incident=float(row["lambda_wrong_before"]),
            )
            unsafe_violation = unsafe_rate - expected_thresholds["unsafe"]
            wrong_violation = (
                wrong_rate - expected_thresholds["wrong_per_incident"]
            )
            expected_after = (
                DualState(
                    unsafe=max(
                        0.0, carried_dual.unsafe
                        + DUAL_STEP_SIZE * unsafe_violation,
                    ),
                    wrong_per_incident=max(
                        0.0, carried_dual.wrong_per_incident
                        + DUAL_STEP_SIZE * wrong_violation,
                    ),
                )
                if mechanism.dual_enabled else DualState()
            )
            observed_after = DualState(
                unsafe=float(row["lambda_unsafe_after"]),
                wrong_per_incident=float(row["lambda_wrong_after"]),
            )
            if (
                dual_before != carried_dual
                or float(row["actor_dual_unsafe"]) != carried_dual.unsafe
                or float(row["actor_dual_wrong_per_incident"])
                != carried_dual.wrong_per_incident
                or not math.isclose(
                    observed_after.unsafe, expected_after.unsafe,
                    rel_tol=1e-12, abs_tol=1e-12,
                )
                or not math.isclose(
                    observed_after.wrong_per_incident,
                    expected_after.wrong_per_incident,
                    rel_tol=1e-12, abs_tol=1e-12,
                )
                or bool(row["dual_update_performed"]) != mechanism.dual_enabled
                or bool(row["dual_updated_after_ppo"]) != mechanism.dual_enabled
                or float(row["dual_step_size"]) != DUAL_STEP_SIZE
                or float(row["unsafe_threshold"])
                != expected_thresholds["unsafe"]
                or float(row["wrong_per_incident_threshold"])
                != expected_thresholds["wrong_per_incident"]
                or not math.isclose(
                    float(row["unsafe_violation"]), unsafe_violation,
                    rel_tol=1e-12, abs_tol=1e-12,
                )
                or not math.isclose(
                    float(row["wrong_per_incident_violation"]), wrong_violation,
                    rel_tol=1e-12, abs_tol=1e-12,
                )
            ):
                raise RuntimeError("A22 training dual-chain binding mismatch")
            carried_dual = observed_after
            total_steps += steps
            total_minibatches += minibatches
            sampled_sequence.extend(sampled)
        sequence_sha = _digest(sampled_sequence)
        if (
            sampled_sequence != expected_sequence
            or len(sampled_sequence) != expected_draws
            or fit["sample_sequence_sha256"] != sequence_sha
            or int(fit["training_episode_draws"]) != len(sampled_sequence)
            or int(fit["sampled_unique_episodes"]) != len(set(sampled_sequence))
            or int(fit["environment_steps"]) != total_steps
            or int(fit["optimizer_minibatches"]) != total_minibatches
            or fit["final_dual"] != asdict(carried_dual)
        ):
            raise RuntimeError("A22 fit sample-sequence hash mismatch")
        _validate_checkpoint(
            fit_dir / "final.pt", fit=fit, mechanism=mechanism,
            outer_fold=fold, training_seed=training_seed,
            run_schedule=run_schedule, run_contract_sha256=run_contract_sha256,
        )
        sample_sequences.setdefault((fold, training_seed), set()).add(sequence_sha)
    if any(len(hashes) != 1 for hashes in sample_sequences.values()):
        raise RuntimeError("A22 mechanisms did not share sample sequence")
    return fit_index


def _fit(
    episodes: Sequence[LongHorizonEpisode], *, a8_rows: Sequence[Mapping[str, Any]],
    mechanism: Mechanism, seed: int, outer_fold: int,
    run_schedule: A22Schedule, output: Path, run_contract_sha256: str,
    partition: Mapping[str, Any],
) -> dict[str, Any]:
    if len(episodes) != 1800 or [row.episode_id for row in episodes] != sorted(
        row.episode_id for row in episodes
    ):
        raise ValueError("A22 fit requires 1800 canonically ordered episodes")
    if {row.episode_id for row in episodes} != set(partition["inner_train_ids"]):
        raise ValueError("A22 fit episodes do not bind to inner-train partition")
    config = _formal_config(run_schedule)
    thresholds = thresholds_from_inner_train(a8_rows)
    expected_mean = sum(len(row.incidents) for row in episodes) / len(episodes)
    if thresholds.mean_incidents != expected_mean:
        raise RuntimeError("A8 threshold mean incidents does not bind to train bank")
    _set_seed(seed)
    model = ActorCritic(
        MultiTownLongHorizonEnv.observation_size, config.hidden_size, ACTION_COUNT,
    ).cpu()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, eps=1e-5,
    )
    sample_rng = random.Random(seed)
    tensor_generator = torch.Generator(device="cpu").manual_seed(seed)
    dual = DualState()
    logs: list[dict[str, Any]] = []
    sampled_ids: list[str] = []
    total_steps = 0
    total_minibatches = 0
    started = time.perf_counter()
    for update in range(1, config.updates + 1):
        transition_episodes = []
        rollout_rows = []
        update_ids = []
        for _ in range(config.episodes_per_update):
            episode = episodes[sample_rng.randrange(len(episodes))]
            update_ids.append(episode.episode_id)
            sampled_ids.append(episode.episode_id)
            transitions, row = constrained_rollout(
                model, episode, torch.device("cpu"),
                mean_incidents=thresholds.mean_incidents,
                shield_enabled=mechanism.shield_enabled,
            )
            transition_episodes.append(transitions)
            rollout_rows.append(row)
        dual_before = dual
        dual, metrics = constrained_update(
            model, optimizer, transition_episodes, config, torch.device("cpu"),
            tensor_generator, mechanism=mechanism, dual_before=dual_before,
            thresholds=thresholds,
        )
        if any(
            isinstance(value, (int, float, np.integer, np.floating))
            and not math.isfinite(float(value))
            for value in metrics.values()
        ):
            raise FloatingPointError("non-finite A22 update metric")
        steps = sum(len(row) for row in transition_episodes)
        minibatches = config.ppo_epochs * math.ceil(steps / config.minibatch_size)
        total_steps += steps
        total_minibatches += minibatches
        logs.append({
            "outer_fold": outer_fold, "training_seed": seed,
            "mechanism": mechanism.name, "update": update,
            "sampled_episode_ids": update_ids,
            "sampled_episode_ids_sha256": _digest(update_ids),
            "environment_steps": steps, "optimizer_minibatches": minibatches,
            "rollout_incidents": sum(int(row["incidents"]) for row in rollout_rows),
            "rollout_episode_success_rate": float(np.mean([
                bool(row["episode_success"]) for row in rollout_rows
            ])),
            "rollout_tokens_per_episode": float(np.mean([
                int(row["tokens_used"]) for row in rollout_rows
            ])),
            "shield_interventions": sum(
                int(row["shield_interventions"]) for row in rollout_rows
            ),
            **metrics,
        })
        _write_json(output / "progress.json", {
            "outer_fold": outer_fold, "training_seed": seed,
            "mechanism": mechanism.name, "current_update": update,
            "scheduled_updates": config.updates,
            "outer_evaluation_started": False,
        })
    checkpoint = output / "final.pt"
    _save_checkpoint(
        checkpoint, model, config, seed=seed, update=config.updates,
        policy_version=POLICY_VERSION,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload.update({
        "run_contract_sha256": run_contract_sha256,
        "mechanism": asdict(mechanism),
        "final_dual": asdict(dual),
        "safety_thresholds": asdict(thresholds),
        "outer_fold": outer_fold,
    })
    temporary = checkpoint.with_name("final.pt.partial")
    torch.save(payload, temporary)
    os.replace(temporary, checkpoint)
    _write_jsonl(output / "training-metrics.jsonl", logs)
    expected_draws = config.updates * config.episodes_per_update
    if (
        len(sampled_ids) != expected_draws
        or not set(sampled_ids) <= set(partition["inner_train_ids"])
    ):
        raise RuntimeError("A22 sampler escaped inner train")
    result = {
        "outer_fold": outer_fold, "training_seed": seed,
        "mechanism": mechanism.name, "dual_enabled": mechanism.dual_enabled,
        "shield_enabled": mechanism.shield_enabled,
        "final_update": config.updates, "training_episode_draws": len(sampled_ids),
        "sample_sequence_sha256": _digest(sampled_ids),
        "sampled_unique_episodes": len(set(sampled_ids)),
        "inner_train_ids_sha256": partition["inner_train_ids_sha256"],
        "outer_ids_sha256": partition["outer_ids_sha256"],
        "calibration_ids_sha256": partition["inner_calibration_ids_sha256"],
        "thresholds": asdict(thresholds), "final_dual": asdict(dual),
        "environment_steps": total_steps,
        "optimizer_minibatches": total_minibatches,
        "checkpoint_sha256": _sha256(checkpoint),
        "run_contract_sha256": run_contract_sha256,
        "training_seconds": time.perf_counter() - started,
        "calibration_evaluations_during_training": 0,
        "outer_evaluations_during_training": 0,
        "selected_checkpoint": "final",
    }
    _write_json(output / "fit-complete.json", result)
    return result


def _evaluate(
    model: ActorCritic, episodes: Sequence[LongHorizonEpisode], *,
    mechanism: Mechanism, training_seed: int, design_outer_fold: int,
    assignment_index: Mapping[str, Any], a8_index: Mapping[str, Mapping[str, Any]],
    bank_sha256: str, resource_sha256: str, environment_sha256: str,
    checkpoint_sha256: str, run_contract_sha256: str, phase: str,
    selection_manifest_sha256: str | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for episode in episodes:
        counters = {"interventions": 0}

        def action(observation: np.ndarray, mask: np.ndarray) -> int:
            selected, intervened = deterministic_action(
                model, observation, mask, shield_enabled=mechanism.shield_enabled,
            )
            counters["interventions"] += int(intervened)
            return selected

        assignment = assignment_index[episode.episode_id]
        baseline = a8_index[episode.episode_id]
        row = _evaluate_episode(
            episode, action, fold=assignment.fold,
            system=f"A22-{mechanism.name}", training_seed=training_seed,
            episode_sha256=assignment.episode_sha256,
            bank_sha256=bank_sha256, resource_sha256=resource_sha256,
            environment_sha256=environment_sha256,
            checkpoint_sha256=checkpoint_sha256,
            a8_row_sha256=_digest(baseline),
            run_contract_sha256=run_contract_sha256,
        )
        row.update({
            "evaluation_phase": phase,
            "design_outer_fold": design_outer_fold,
            "mechanism": mechanism.name,
            "shield_interventions": counters["interventions"],
            "selection_manifest_sha256": selection_manifest_sha256,
        })
        rows.append(row)
    return rows


def _validate_evaluation_binding(
    row: Mapping[str, Any], *, episode_id: str, design_outer_fold: int,
    training_seed: int, mechanism: Mechanism, phase: str,
    selection_manifest_sha256: str | None,
    assignment_index: Mapping[str, Any], a8_index: Mapping[str, Mapping[str, Any]],
    bank_sha256: str, resource_sha256: str, environment_sha256: str,
    checkpoint_sha256: str, run_contract_sha256: str,
) -> None:
    assignment = assignment_index[episode_id]
    baseline = a8_index[episode_id]
    expected = {
        "episode_id": episode_id,
        "episode_sha256": assignment.episode_sha256,
        "outer_fold": assignment.fold,
        "design_outer_fold": design_outer_fold,
        "training_seed": training_seed,
        "mechanism": mechanism.name,
        "system": f"A22-{mechanism.name}",
        "evaluation_phase": phase,
        "train_bank_sha256": bank_sha256,
        "resource_contract_sha256": resource_sha256,
        "environment_source_sha256": environment_sha256,
        "final_checkpoint_sha256": checkpoint_sha256,
        "a8_row_sha256": _digest(baseline),
        "run_contract_sha256": run_contract_sha256,
        "selection_manifest_sha256": selection_manifest_sha256,
    }
    mismatched = [
        field for field, value in expected.items() if row.get(field) != value
    ]
    if mismatched:
        raise RuntimeError(
            f"A22 decision row provenance binding mismatch: {mismatched}"
        )


def _validate_calibration_rows(
    rows: Sequence[Mapping[str, Any]], *, run_schedule: A22Schedule,
    partitions: Mapping[int, Mapping[str, Any]],
    fit_index: Mapping[tuple[int, int, str], Mapping[str, Any]],
    assignment_index: Mapping[str, Any], a8_index: Mapping[str, Mapping[str, Any]],
    bank_sha256: str, resource_sha256: str, environment_sha256: str,
    run_contract_sha256: str,
) -> None:
    ids_by_fold = {
        fold: tuple(partitions[fold]["inner_calibration_ids"][
            :run_schedule.calibration_episodes_per_fold
        ])
        for fold in run_schedule.folds
    }
    expected_keys = {
        (fold, seed, mechanism.name, episode_id)
        for fold in run_schedule.folds for seed in run_schedule.seeds
        for mechanism in MECHANISMS for episode_id in ids_by_fold[fold]
    }
    actual_keys = [
        (
            int(row["design_outer_fold"]), int(row["training_seed"]),
            str(row["mechanism"]), str(row["episode_id"]),
        )
        for row in rows
    ]
    _assert_exact_keys(actual_keys, expected_keys, label="calibration decision")
    mechanism_index = {item.name: item for item in MECHANISMS}
    for row, key in zip(rows, actual_keys, strict=True):
        fold, seed, mechanism_name, episode_id = key
        fit = fit_index[(fold, seed, mechanism_name)]
        _validate_evaluation_binding(
            row, episode_id=episode_id, design_outer_fold=fold,
            training_seed=seed, mechanism=mechanism_index[mechanism_name],
            phase="inner-calibration", selection_manifest_sha256=None,
            assignment_index=assignment_index, a8_index=a8_index,
            bank_sha256=bank_sha256, resource_sha256=resource_sha256,
            environment_sha256=environment_sha256,
            checkpoint_sha256=str(fit["checkpoint_sha256"]),
            run_contract_sha256=run_contract_sha256,
        )


def _validate_outer_rows(
    rows: Sequence[Mapping[str, Any]], *, selections: Sequence[Mapping[str, Any]],
    run_schedule: A22Schedule, partitions: Mapping[int, Mapping[str, Any]],
    fit_index: Mapping[tuple[int, int, str], Mapping[str, Any]],
    assignment_index: Mapping[str, Any], a8_index: Mapping[str, Mapping[str, Any]],
    bank_sha256: str, resource_sha256: str, environment_sha256: str,
    run_contract_sha256: str, selection_manifest_sha256: str,
) -> None:
    _assert_exact_keys(
        [(int(row["outer_fold"]),) for row in selections],
        {(fold,) for fold in run_schedule.folds}, label="selection",
    )
    selected_by_fold = {
        int(row["outer_fold"]): row["selected_mechanism"] for row in selections
    }
    all_feasible = all(value is not None for value in selected_by_fold.values())
    expected_keys = (
        {
            (fold, seed, str(selected_by_fold[fold]), episode_id)
            for fold in run_schedule.folds for seed in run_schedule.seeds
            for episode_id in partitions[fold]["outer_ids"][
                :run_schedule.outer_episodes_per_fold
            ]
        }
        if all_feasible else set()
    )
    actual_keys = [
        (
            int(row["design_outer_fold"]), int(row["training_seed"]),
            str(row["mechanism"]), str(row["episode_id"]),
        )
        for row in rows
    ]
    _assert_exact_keys(actual_keys, expected_keys, label="selected outer decision")
    mechanism_index = {item.name: item for item in MECHANISMS}
    for row, key in zip(rows, actual_keys, strict=True):
        fold, seed, mechanism_name, episode_id = key
        if mechanism_name != selected_by_fold[fold]:
            raise RuntimeError("A22 outer row does not bind to frozen selection")
        fit = fit_index[(fold, seed, mechanism_name)]
        _validate_evaluation_binding(
            row, episode_id=episode_id, design_outer_fold=fold,
            training_seed=seed, mechanism=mechanism_index[mechanism_name],
            phase="selected-outer",
            selection_manifest_sha256=selection_manifest_sha256,
            assignment_index=assignment_index, a8_index=a8_index,
            bank_sha256=bank_sha256, resource_sha256=resource_sha256,
            environment_sha256=environment_sha256,
            checkpoint_sha256=str(fit["checkpoint_sha256"]),
            run_contract_sha256=run_contract_sha256,
        )


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize empty A22 rows")
    return {
        "rows": len(rows),
        "episodes": len({str(row["episode_id"]) for row in rows}),
        "episode_success_rate": float(np.mean([
            bool(row["episode_success"]) for row in rows
        ])),
        "unsafe_episode_rate": float(np.mean([
            bool(row["had_wrong_execution"]) for row in rows
        ])),
        "wrong_executions_per_incident": (
            sum(int(row["wrong_executions"]) for row in rows)
            / sum(int(row["incidents"]) for row in rows)
        ),
        "tokens_per_episode": float(np.mean([
            int(row["tokens_used"]) for row in rows
        ])),
        "latency_per_episode_s": float(np.mean([
            float(row["latency_used_s"]) for row in rows
        ])),
        "invalid_actions": sum(int(row["invalid_actions"]) for row in rows),
        "budget_violations": sum(int(row["budget_violations"]) for row in rows),
        "shield_interventions": sum(
            int(row.get("shield_interventions", 0)) for row in rows
        ),
    }


def _action_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(
        str(step["action"]) for row in rows for step in row["trajectory"]
    )
    return {action: int(counts.get(action, 0)) for action in ACTION_NAMES}


def _behavior_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize empty A22 behavior rows")
    route_counts: Counter[tuple[str, ...]] = Counter()
    execute_after_review = 0
    for row in rows:
        route = tuple(str(step["action"]) for step in row["trajectory"])
        route_counts[route] += 1
        reviewed_incidents: set[int] = set()
        for step in row["trajectory"]:
            action = str(step["action"])
            incident = int(step["incident_index"])
            if action == "review":
                reviewed_incidents.add(incident)
            elif action == "execute" and incident in reviewed_incidents:
                execute_after_review += 1
    actions = _action_counts(rows)
    route_mix = [
        {"actions": list(route), "episodes": count}
        for route, count in sorted(
            route_counts.items(), key=lambda item: (-item[1], item[0]),
        )[:20]
    ]
    return {
        **_summary(rows),
        "action_counts": actions,
        "review_actions": actions["review"],
        "execute_actions": actions["execute"],
        "human_actions": actions["human"],
        "execute_after_prior_review_action": execute_after_review,
        "execute_without_prior_review_action": (
            actions["execute"] - execute_after_review
        ),
        "route_mix_top20": route_mix,
        "route_mix_unique": len(route_counts),
        "action_counts_by_seed": {
            str(seed): _action_counts([
                row for row in rows if int(row["training_seed"]) == seed
            ])
            for seed in sorted({int(row["training_seed"]) for row in rows})
        },
        "action_counts_by_fold": {
            str(fold): _action_counts([
                row for row in rows if int(row["outer_fold"]) == fold
            ])
            for fold in sorted({int(row["outer_fold"]) for row in rows})
        },
        "action_counts_by_mechanism": {
            mechanism: _action_counts([
                row for row in rows if str(row["mechanism"]) == mechanism
            ])
            for mechanism in sorted({str(row["mechanism"]) for row in rows})
        },
    }


def _a22_stat_labels(value: Any) -> Any:
    """Relabel reused numeric bootstrap machinery without A9 evidence semantics."""

    if isinstance(value, dict):
        renamed = {"a9_mean": "a22_mean", "a9_ratio": "a22_ratio"}
        return {
            renamed.get(str(key), str(key)): _a22_stat_labels(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_a22_stat_labels(item) for item in value]
    if isinstance(value, str):
        if value == "A9_minus_A8":
            return "A22_minus_A8"
        if value.startswith("multitown-a9-"):
            return "multitown-a22-" + value.removeprefix("multitown-a9-")
    return value


def _select(
    rows: Sequence[Mapping[str, Any]], *, outer_fold: int,
    seeds: Sequence[int], episode_ids: Sequence[str],
    a8_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    canonical_ids = tuple(str(item) for item in episode_ids)
    if len(canonical_ids) != len(set(canonical_ids)):
        raise RuntimeError("A22 selection episode IDs are not unique")
    _assert_exact_keys(
        [(str(row["episode_id"]),) for row in a8_rows],
        {(episode_id,) for episode_id in canonical_ids},
        label="selection A8 baseline",
    )
    expected_keys = {
        (mechanism.name, seed, episode_id)
        for mechanism in MECHANISMS for seed in seeds for episode_id in canonical_ids
    }
    actual_keys = [
        (
            str(row["mechanism"]), int(row["training_seed"]),
            str(row["episode_id"]),
        )
        for row in rows
    ]
    _assert_exact_keys(actual_keys, expected_keys, label="selection calibration")
    if any(
        int(row.get("design_outer_fold", -1)) != outer_fold
        or row.get("evaluation_phase") != "inner-calibration"
        or row.get("selection_manifest_sha256") is not None
        for row in rows
    ):
        raise RuntimeError("A22 selection received misbound calibration rows")
    a8_unsafe = float(np.mean([bool(row["had_wrong_execution"]) for row in a8_rows]))
    a8_wrong = sum(int(row["wrong_executions"]) for row in a8_rows) / sum(
        int(row["incidents"]) for row in a8_rows
    )
    thresholds = {
        "unsafe_episode": min(1.0, a8_unsafe + 0.02),
        "wrong_executions_per_incident": min(1.0, a8_wrong + 0.01),
    }
    candidates: dict[str, Any] = {}
    feasible: list[tuple[float, float, int, str]] = []
    for order, mechanism in enumerate(MECHANISMS):
        mechanism_rows = [row for row in rows if row["mechanism"] == mechanism.name]
        by_seed = {}
        seed_pass = []
        for training_seed in seeds:
            selected = [
                row for row in mechanism_rows
                if int(row["training_seed"]) == training_seed
            ]
            summary = _summary(selected)
            passed = bool(
                summary["unsafe_episode_rate"] <= thresholds["unsafe_episode"]
                and summary["wrong_executions_per_incident"]
                <= thresholds["wrong_executions_per_incident"]
                and summary["invalid_actions"] == 0
                and summary["budget_violations"] == 0
            )
            by_seed[str(training_seed)] = {**summary, "safety_feasible": passed}
            seed_pass.append(passed)
        overall = _summary(mechanism_rows)
        is_feasible = all(seed_pass)
        candidates[mechanism.name] = {
            "mechanism": asdict(mechanism), "by_seed": by_seed,
            "three_seed_summary": overall, "all_seeds_feasible": is_feasible,
        }
        if is_feasible:
            feasible.append((
                -overall["episode_success_rate"], overall["tokens_per_episode"],
                order, mechanism.name,
            ))
    selected_name = min(feasible)[3] if feasible else None
    selection = {
        "outer_fold": outer_fold,
        "selected_mechanism": selected_name,
        "status": "selected" if selected_name else "no_feasible_mechanism",
        "calibration_a8_summary": _summary(a8_rows),
        "calibration_thresholds": thresholds,
        "tie_order": [item.name for item in MECHANISMS],
        "selection_key": (
            "all-seed joint feasibility, max mean autonomous success, "
            "min mean tokens, fixed mechanism order"
        ),
    }
    return selection, candidates


def _result_statistics(
    a8_rows: Sequence[Mapping[str, Any]], outer_rows: Sequence[Mapping[str, Any]],
    run_schedule: A22Schedule, *, gate_evaluable: bool,
) -> dict[str, Any]:
    compatible = RunSchedule(
        mode=run_schedule.mode, seeds=run_schedule.seeds, folds=run_schedule.folds,
        updates=run_schedule.updates,
        episodes_per_update=run_schedule.episodes_per_update,
        evaluation_episodes_per_fold=run_schedule.outer_episodes_per_fold,
        bootstrap_iterations=run_schedule.bootstrap_iterations,
        bootstrap_seed=run_schedule.bootstrap_seed, threads=run_schedule.threads,
    )
    overall = _paired_effects(a8_rows, outer_rows, compatible)
    per_seed = {}
    for index, training_seed in enumerate(run_schedule.seeds):
        selected = [
            row for row in outer_rows if int(row["training_seed"]) == training_seed
        ]
        success = fold_cluster_bootstrap(
            a8_rows, selected, metric="episode_success", seeds=(training_seed,),
            iterations=run_schedule.bootstrap_iterations,
            seed=run_schedule.bootstrap_seed,
        )
        tokens = fold_cluster_bootstrap(
            a8_rows, selected, metric="tokens_used", seeds=(training_seed,),
            iterations=run_schedule.bootstrap_iterations,
            seed=run_schedule.bootstrap_seed + 100 + index,
        )
        unsafe = fold_cluster_bootstrap(
            a8_rows, selected, metric="wrong_execution", seeds=(training_seed,),
            iterations=run_schedule.bootstrap_iterations,
            seed=run_schedule.bootstrap_seed + 200 + index,
        )
        wrong = fold_cluster_ratio_bootstrap(
            a8_rows, selected, numerator="wrong_executions", denominator="incidents",
            seeds=(training_seed,), iterations=run_schedule.bootstrap_iterations,
            seed=run_schedule.bootstrap_seed + 300 + index,
        )
        per_fold_unsafe = {}
        for fold in run_schedule.folds:
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
            "unsafe_within_a8_plus_0_02": unsafe["a9_mean"] <= unsafe["a8_mean"] + 0.02,
            "wrong_per_incident_within_a8_plus_0_01": (
                wrong["a9_ratio"] <= wrong["a8_ratio"] + 0.01
            ),
            "success_noninferiority_lower_bound_nonnegative": success["ci95_low"] >= 0.0,
            "tokens_no_greater_than_a8": tokens["a9_mean"] <= tokens["a8_mean"],
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
    raw_gate_conjunction = all(
        row["all_gates_passed"] for row in per_seed.values()
    )
    result = {
        "overall_fixed_seed_mean": overall,
        "per_seed": per_seed,
        "selected_outer_behavior": _behavior_summary(outer_rows),
        "raw_gate_conjunction": raw_gate_conjunction,
        "gate_evaluable": gate_evaluable,
        "adaptive_development_gate_passed": bool(
            gate_evaluable and raw_gate_conjunction
        ),
    }
    return _a22_stat_labels(result)


def _manifest(output: Path, source_revision: str) -> dict[str, Any]:
    files = {
        str(path.relative_to(output)): {
            "bytes": path.stat().st_size, "sha256": _sha256(path),
        }
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name not in {"artifact-manifest.json", "RUNNING.json"}
    }
    return {
        "schema_version": "multitown-a22-adaptive-artifact-manifest-v1",
        "source_revision": source_revision, "files": files,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }


def _gate_scope(*, smoke: bool, all_feasible: bool) -> dict[str, bool]:
    return {
        "formal_selection_outcome_evaluable": not smoke,
        "formal_selection_negative": bool(not smoke and not all_feasible),
        "formal_development_gate_evaluable": bool(not smoke and all_feasible),
    }


def _isolate_valid_outputs(output: Path) -> None:
    for valid_name, invalid_name in (
        ("result.json", "INVALID_RESULT.json"),
        ("artifact-manifest.json", "INVALID_ARTIFACT_MANIFEST.json"),
    ):
        valid_path = output / valid_name
        invalid_path = output / invalid_name
        if valid_path.exists() and not invalid_path.exists():
            os.replace(valid_path, invalid_path)


def _revalidate_run_inputs(
    root: Path, *, output: Path, source: Mapping[str, Any], require_clean: bool,
    train_bank_sha256: str, fold_manifest_sha: str,
    resource_sha256: str, environment_sha256: str,
    training_contract: Mapping[str, Any], contract: Mapping[str, Any],
    run_contract_sha256: str,
) -> None:
    if _source_state(root, require_clean=require_clean) != dict(source):
        raise RuntimeError("A22 executed source changed during the run")
    frozen = _verify_r2_artifacts(root)
    raw = Path(frozen["path"])
    if (
        _sha256(raw / "a8-oof-decisions.jsonl") != EXPECTED_R2_A8_SHA256
        or _sha256(raw / "run-contract.json") != EXPECTED_R2_RUN_FILE_SHA256
        or _digest(frozen["run_contract"]) != EXPECTED_R2_RUN_DIGEST
    ):
        raise RuntimeError("A22 frozen r2 inputs changed during the run")
    bank = load_frozen_train_bank(FROZEN_TRAIN_PATH)
    assignments = assign_stratified_group_folds(bank)
    resource = shared_resource_contract(bank)
    _validate_shared_stack_bindings(
        root, frozen["run_contract"], bank=bank,
        assignments=assignments, resource=resource,
    )
    if (
        bank.payload_sha256 != train_bank_sha256
        or fold_manifest_sha256(assignments) != fold_manifest_sha
        or resource_contract_sha256(resource) != resource_sha256
        or str(resource["environment_source_sha256"]) != environment_sha256
    ):
        raise RuntimeError("A22 frozen bank/fold/resource binding changed during run")
    persisted_training = json.loads(
        (output / "training-contract.json").read_text(encoding="utf-8")
    )
    persisted_run = json.loads(
        (output / "run-contract.json").read_text(encoding="utf-8")
    )
    if (
        _digest(persisted_training) != _digest(training_contract)
        or _digest(persisted_run) != _digest(contract)
        or _digest(persisted_training) != contract["training_contract_sha256"]
        or _digest(persisted_run) != run_contract_sha256
    ):
        raise RuntimeError("A22 persisted run contract changed during the run")


def _validate_persisted_artifacts(
    *, output: Path, fits: Sequence[Mapping[str, Any]],
    calibration_rows: Sequence[Mapping[str, Any]],
    selection_payload: Mapping[str, Any], selection_manifest_sha256: str,
    outer_rows: Sequence[Mapping[str, Any]] | None, all_feasible: bool,
    run_schedule: A22Schedule, partitions: Mapping[int, Mapping[str, Any]],
    a8_index: Mapping[str, Mapping[str, Any]],
    episode_index: Mapping[str, LongHorizonEpisode],
    assignment_index: Mapping[str, Any], bank_sha256: str,
    resource_sha256: str, environment_sha256: str,
    run_contract_sha256: str,
) -> dict[tuple[int, int, str], Mapping[str, Any]]:
    fit_index = _validate_fits(
        fits, output=output, run_schedule=run_schedule, partitions=partitions,
        a8_index=a8_index, episode_index=episode_index,
        run_contract_sha256=run_contract_sha256,
    )
    persisted_fits = json.loads(
        (output / "all-fits-complete.json").read_text(encoding="utf-8")
    )
    if (
        _digest(persisted_fits.get("fits")) != _digest(list(fits))
        or int(persisted_fits.get("expected_fits", -1)) != len(fits)
        or not persisted_fits.get("exact_fit_key_product_verified")
        or not persisted_fits.get("training_log_and_checkpoint_provenance_verified")
    ):
        raise RuntimeError("A22 persisted all-fits artifact changed")
    persisted_calibration = _read_jsonl(output / "calibration-decisions.jsonl")
    if _digest(persisted_calibration) != _digest(list(calibration_rows)):
        raise RuntimeError("A22 persisted calibration rows changed")
    _validate_calibration_rows(
        persisted_calibration, run_schedule=run_schedule, partitions=partitions,
        fit_index=fit_index, assignment_index=assignment_index,
        a8_index=a8_index, bank_sha256=bank_sha256,
        resource_sha256=resource_sha256, environment_sha256=environment_sha256,
        run_contract_sha256=run_contract_sha256,
    )
    selection_path = output / "all-selections-frozen.json"
    _validate_selection_file(
        selection_path, expected_payload=selection_payload,
        expected_sha256=selection_manifest_sha256,
        run_contract_sha256=run_contract_sha256,
    )
    outer_path = output / "outer-decisions.jsonl"
    outer_gate_path = output / "OUTER_GATE_OPEN.json"
    if outer_rows is None:
        if outer_path.exists() or outer_gate_path.exists():
            raise RuntimeError("A22 outer rows existed before global gate")
        return fit_index
    if all_feasible:
        gate = json.loads(outer_gate_path.read_text(encoding="utf-8"))
        if (
            gate.get("selection_manifest_sha256") != selection_manifest_sha256
            or gate.get("all_five_selections_feasible") is not True
        ):
            raise RuntimeError("A22 outer gate does not bind to frozen selection")
        persisted_outer = _read_jsonl(outer_path)
        if _digest(persisted_outer) != _digest(list(outer_rows)):
            raise RuntimeError("A22 persisted outer rows changed")
    else:
        if outer_rows or outer_path.exists() or outer_gate_path.exists():
            raise RuntimeError("A22 outer rows exist after infeasible selection")
        persisted_outer = []
    _validate_outer_rows(
        persisted_outer, selections=selection_payload["selections"],
        run_schedule=run_schedule, partitions=partitions,
        fit_index=fit_index, assignment_index=assignment_index,
        a8_index=a8_index, bank_sha256=bank_sha256,
        resource_sha256=resource_sha256, environment_sha256=environment_sha256,
        run_contract_sha256=run_contract_sha256,
        selection_manifest_sha256=selection_manifest_sha256,
    )
    return fit_index


def _validate_selection_file(
    path: Path, *, expected_payload: Mapping[str, Any],
    expected_sha256: str, run_contract_sha256: str,
) -> None:
    persisted = json.loads(path.read_text(encoding="utf-8"))
    if (
        _sha256(path) != expected_sha256
        or _digest(persisted) != _digest(expected_payload)
        or persisted.get("run_contract_sha256") != run_contract_sha256
    ):
        raise RuntimeError("A22 frozen selection artifact changed")


def run(output: Path, *, smoke: bool) -> int:
    root = Path(__file__).resolve().parents[1]
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    run_schedule = schedule(smoke)
    if not smoke and output.parent != (root / "artifacts").resolve():
        raise ValueError("formal A22 output must be a direct child of artifacts/")
    source = _source_state(root, require_clean=not smoke)
    frozen = _verify_r2_artifacts(root)
    raw = Path(frozen["path"])
    if (
        _sha256(raw / "a8-oof-decisions.jsonl") != EXPECTED_R2_A8_SHA256
        or _sha256(raw / "run-contract.json") != EXPECTED_R2_RUN_FILE_SHA256
        or _digest(frozen["run_contract"]) != EXPECTED_R2_RUN_DIGEST
    ):
        raise RuntimeError("A22 frozen r2 A8/run inputs changed")
    bank = load_frozen_train_bank(FROZEN_TRAIN_PATH)
    assignments = assign_stratified_group_folds(bank)
    resource = shared_resource_contract(bank)
    _validate_shared_stack_bindings(
        root, frozen["run_contract"], bank=bank,
        assignments=assignments, resource=resource,
    )
    resource_sha = resource_contract_sha256(resource)
    environment_sha = str(resource["environment_source_sha256"])
    frozen_fold_manifest_sha = fold_manifest_sha256(assignments)
    episode_index = {row.episode_id: row for row in bank.episodes}
    assignment_index = {row.episode_id: row for row in assignments}
    a8_rows = _read_jsonl(raw / "a8-oof-decisions.jsonl")
    a8_index = {str(row["episode_id"]): row for row in a8_rows}
    if (
        len(a8_rows) != 3000 or len(a8_index) != len(a8_rows)
        or set(a8_index) != set(episode_index)
        or any(
            int(row["outer_fold"]) != assignment_index[str(row["episode_id"])].fold
            or row["episode_sha256"]
            != assignment_index[str(row["episode_id"])].episode_sha256
            or row["train_bank_sha256"] != bank.payload_sha256
            or row["resource_contract_sha256"] != resource_sha
            or row["environment_source_sha256"] != environment_sha
            for row in a8_rows
        )
    ):
        raise RuntimeError("A22 frozen A8 baseline incomplete")
    partitions = {fold: partition_ids(assignments, fold) for fold in run_schedule.folds}
    training_contract = {
        "schema_version": "multitown-a22-frozen-training-contract-v1",
        "protocol_revision": "0e6adab9fb31c643475db21736ff330dcf650df4",
        "runner_version": RUNNER_VERSION,
        "primitives_version": A22_PRIMITIVES_VERSION,
        "policy_version": POLICY_VERSION,
        "reference_policy_version": A9_POLICY_VERSION,
        "mechanisms": [asdict(item) for item in MECHANISMS],
        "schedule": asdict(run_schedule),
        "ppo": asdict(_formal_config(run_schedule)),
        "dual_step_size": DUAL_STEP_SIZE,
        "partitions": list(partitions.values()),
        "outer_gate": "all fits and calibrations before all-selections-frozen",
        "completeness_validation": (
            "exact fold x seed x mechanism x episode Cartesian products; "
            "training logs, sample hashes, checkpoints and decision provenance"
        ),
        "non_evidentiary_smoke": smoke,
        "adaptive_same_bank_development": True,
        "independent_confirmation": False,
    }
    contract = {
        "schema_version": "multitown-a22-adaptive-run-contract-v1",
        "source": source, "mode": run_schedule.mode,
        "train_bank_sha256": bank.payload_sha256,
        "fold_manifest_sha256": frozen_fold_manifest_sha,
        "resource_contract_sha256": resource_sha,
        "environment_source_sha256": environment_sha,
        "frozen_r2_manifest_sha256": EXPECTED_R2_MANIFEST_SHA256,
        "frozen_r2_a8_sha256": EXPECTED_R2_A8_SHA256,
        "frozen_r2_run_file_sha256": EXPECTED_R2_RUN_FILE_SHA256,
        "frozen_r2_run_digest": EXPECTED_R2_RUN_DIGEST,
        "training_contract_sha256": _digest(training_contract),
    }
    contract_sha = _digest(contract)
    output.mkdir(parents=True)
    try:
        if not smoke:
            lock = root / FORMAL_ATTEMPT_LOCK
            descriptor = {"output": str(output), "run_contract_sha256": contract_sha}
            fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(descriptor, indent=2) + "\n")
        _write_json(output / "RUNNING.json", {
            "schema_version": RUNNER_VERSION,
            "started_at_utc": datetime.now(UTC).isoformat(),
            "outer_evaluation_started": False,
        })
        _write_json(output / "training-contract.json", training_contract)
        _write_json(output / "run-contract.json", contract)
        torch.set_num_threads(run_schedule.threads)
        fits = []
        for fold in run_schedule.folds:
            partition = partitions[fold]
            train_episodes = [episode_index[item] for item in partition["inner_train_ids"]]
            train_a8 = [a8_index[item] for item in partition["inner_train_ids"]]
            for training_seed in run_schedule.seeds:
                sample_hashes = []
                for mechanism in MECHANISMS:
                    fit_dir = (
                        output / "fits" / f"outer-fold-{fold}"
                        / f"seed-{training_seed}" / mechanism.name
                    )
                    fit_dir.mkdir(parents=True)
                    fit = _fit(
                        train_episodes, a8_rows=train_a8, mechanism=mechanism,
                        seed=training_seed, outer_fold=fold,
                        run_schedule=run_schedule, output=fit_dir,
                        run_contract_sha256=contract_sha, partition=partition,
                    )
                    fits.append(fit)
                    sample_hashes.append(fit["sample_sequence_sha256"])
                if len(set(sample_hashes)) != 1:
                    raise RuntimeError("A22 mechanisms did not share sample sequence")
        expected_fits = len(run_schedule.folds) * len(run_schedule.seeds) * len(MECHANISMS)
        fit_index = _validate_fits(
            fits, output=output, run_schedule=run_schedule,
            partitions=partitions, a8_index=a8_index,
            episode_index=episode_index, run_contract_sha256=contract_sha,
        )
        _write_json(output / "all-fits-complete.json", {
            "fits": fits, "expected_fits": expected_fits,
            "all_reached_final_update": True,
            "exact_fit_key_product_verified": True,
            "training_log_and_checkpoint_provenance_verified": True,
            "outer_evaluation_started": False,
        })

        calibration_rows = []
        for fold in run_schedule.folds:
            ids = partitions[fold]["inner_calibration_ids"][
                :run_schedule.calibration_episodes_per_fold
            ]
            episodes = [episode_index[item] for item in ids]
            for training_seed in run_schedule.seeds:
                for mechanism in MECHANISMS:
                    fit = fit_index[(fold, training_seed, mechanism.name)]
                    checkpoint = (
                        output / "fits" / f"outer-fold-{fold}"
                        / f"seed-{training_seed}" / mechanism.name / "final.pt"
                    )
                    model, _ = _validate_checkpoint(
                        checkpoint, fit=fit, mechanism=mechanism,
                        outer_fold=fold, training_seed=training_seed,
                        run_schedule=run_schedule,
                        run_contract_sha256=contract_sha,
                    )
                    calibration_rows.extend(_evaluate(
                        model, episodes, mechanism=mechanism,
                        training_seed=training_seed, design_outer_fold=fold,
                        assignment_index=assignment_index, a8_index=a8_index,
                        bank_sha256=bank.payload_sha256,
                        resource_sha256=resource_sha,
                        environment_sha256=environment_sha,
                        checkpoint_sha256=fit["checkpoint_sha256"],
                        run_contract_sha256=contract_sha, phase="inner-calibration",
                    ))
        expected_calibration = (
            len(run_schedule.folds) * len(run_schedule.seeds) * len(MECHANISMS)
            * run_schedule.calibration_episodes_per_fold
        )
        _validate_calibration_rows(
            calibration_rows, run_schedule=run_schedule, partitions=partitions,
            fit_index=fit_index, assignment_index=assignment_index,
            a8_index=a8_index, bank_sha256=bank.payload_sha256,
            resource_sha256=resource_sha, environment_sha256=environment_sha,
            run_contract_sha256=contract_sha,
        )
        _write_jsonl(output / "calibration-decisions.jsonl", calibration_rows)
        selections = []
        candidate_summaries = {}
        for fold in run_schedule.folds:
            rows = [row for row in calibration_rows if int(row["design_outer_fold"]) == fold]
            ids = partitions[fold]["inner_calibration_ids"][
                :run_schedule.calibration_episodes_per_fold
            ]
            selection, candidates = _select(
                rows, outer_fold=fold, seeds=run_schedule.seeds,
                episode_ids=ids,
                a8_rows=[a8_index[item] for item in ids],
            )
            selections.append(selection)
            candidate_summaries[str(fold)] = candidates
        _assert_exact_keys(
            [(int(row["outer_fold"]),) for row in selections],
            {(fold,) for fold in run_schedule.folds}, label="selection",
        )
        _revalidate_run_inputs(
            root, output=output, source=source, require_clean=not smoke,
            train_bank_sha256=bank.payload_sha256,
            fold_manifest_sha=frozen_fold_manifest_sha,
            resource_sha256=resource_sha, environment_sha256=environment_sha,
            training_contract=training_contract, contract=contract,
            run_contract_sha256=contract_sha,
        )
        selection_payload = {
            "schema_version": "multitown-a22-all-selections-frozen-v1",
            "run_contract_sha256": contract_sha,
            "expected_fits": expected_fits,
            "calibration_rows": len(calibration_rows),
            "selections": selections, "candidate_summaries": candidate_summaries,
            "checkpoint_sha256": {
                f"{row['outer_fold']}:{row['training_seed']}:{row['mechanism']}":
                row["checkpoint_sha256"] for row in fits
            },
            "all_fits_and_calibrations_complete": True,
            "exact_fit_and_calibration_key_products_verified": True,
            "input_bindings_revalidated_before_outer_gate": True,
            "outer_evaluation_started": False,
            "frozen_at_utc": datetime.now(UTC).isoformat(),
        }
        selection_path = output / "all-selections-frozen.json"
        _write_json_atomic(selection_path, selection_payload)
        selection_sha = _sha256(selection_path)
        all_feasible = all(row["selected_mechanism"] is not None for row in selections)
        fit_index = _validate_persisted_artifacts(
            output=output, fits=fits, calibration_rows=calibration_rows,
            selection_payload=selection_payload,
            selection_manifest_sha256=selection_sha, outer_rows=None,
            all_feasible=all_feasible, run_schedule=run_schedule,
            partitions=partitions, a8_index=a8_index,
            episode_index=episode_index, assignment_index=assignment_index,
            bank_sha256=bank.payload_sha256, resource_sha256=resource_sha,
            environment_sha256=environment_sha,
            run_contract_sha256=contract_sha,
        )
        outer_rows = []
        if all_feasible:
            _write_json(output / "OUTER_GATE_OPEN.json", {
                "selection_manifest_sha256": selection_sha,
                "all_five_selections_feasible": True,
                "opened_at_utc": datetime.now(UTC).isoformat(),
            })
            for selection in selections:
                fold = int(selection["outer_fold"])
                mechanism = next(
                    row for row in MECHANISMS
                    if row.name == selection["selected_mechanism"]
                )
                ids = partitions[fold]["outer_ids"][:run_schedule.outer_episodes_per_fold]
                episodes = [episode_index[item] for item in ids]
                for training_seed in run_schedule.seeds:
                    fit = fit_index[(fold, training_seed, mechanism.name)]
                    checkpoint = (
                        output / "fits" / f"outer-fold-{fold}"
                        / f"seed-{training_seed}" / mechanism.name / "final.pt"
                    )
                    model, _ = _validate_checkpoint(
                        checkpoint, fit=fit, mechanism=mechanism,
                        outer_fold=fold, training_seed=training_seed,
                        run_schedule=run_schedule,
                        run_contract_sha256=contract_sha,
                    )
                    outer_rows.extend(_evaluate(
                        model, episodes, mechanism=mechanism,
                        training_seed=training_seed, design_outer_fold=fold,
                        assignment_index=assignment_index, a8_index=a8_index,
                        bank_sha256=bank.payload_sha256,
                        resource_sha256=resource_sha,
                        environment_sha256=environment_sha,
                        checkpoint_sha256=fit["checkpoint_sha256"],
                        run_contract_sha256=contract_sha, phase="selected-outer",
                        selection_manifest_sha256=selection_sha,
                    ))
            _write_jsonl(output / "outer-decisions.jsonl", outer_rows)
        _validate_outer_rows(
            outer_rows, selections=selections, run_schedule=run_schedule,
            partitions=partitions, fit_index=fit_index,
            assignment_index=assignment_index, a8_index=a8_index,
            bank_sha256=bank.payload_sha256, resource_sha256=resource_sha,
            environment_sha256=environment_sha,
            run_contract_sha256=contract_sha,
            selection_manifest_sha256=selection_sha,
        )
        selected_outer_a8 = []
        for fold in run_schedule.folds:
            ids = partitions[fold]["outer_ids"][:run_schedule.outer_episodes_per_fold]
            selected_outer_a8.extend(a8_index[item] for item in ids)
        statistics = (
            _result_statistics(
                selected_outer_a8, outer_rows, run_schedule,
                gate_evaluable=not smoke,
            )
            if all_feasible else None
        )
        fit_index = _validate_persisted_artifacts(
            output=output, fits=fits, calibration_rows=calibration_rows,
            selection_payload=selection_payload,
            selection_manifest_sha256=selection_sha, outer_rows=outer_rows,
            all_feasible=all_feasible, run_schedule=run_schedule,
            partitions=partitions, a8_index=a8_index,
            episode_index=episode_index, assignment_index=assignment_index,
            bank_sha256=bank.payload_sha256, resource_sha256=resource_sha,
            environment_sha256=environment_sha,
            run_contract_sha256=contract_sha,
        )
        _revalidate_run_inputs(
            root, output=output, source=source, require_clean=not smoke,
            train_bank_sha256=bank.payload_sha256,
            fold_manifest_sha=frozen_fold_manifest_sha,
            resource_sha256=resource_sha, environment_sha256=environment_sha,
            training_contract=training_contract, contract=contract,
            run_contract_sha256=contract_sha,
        )
        result = {
            "schema_version": RESULT_VERSION, "mode": run_schedule.mode,
            "evidence_scope": (
                "non-evidentiary implementation smoke"
                if smoke else "post-A9 adaptive same-bank nested development"
            ),
            "non_evidentiary_smoke": smoke,
            **_gate_scope(smoke=smoke, all_feasible=all_feasible),
            "raw_gate_conjunction": bool(
                statistics and statistics["raw_gate_conjunction"]
            ),
            "adaptive_development_gate_passed": bool(
                statistics and statistics["adaptive_development_gate_passed"]
            ),
            "adaptive_same_bank_development": True,
            "independent_confirmation": False,
            "may_claim_preregistered_or_hidden_test": False,
            "source_revision": source["revision"],
            "run_contract_sha256": contract_sha,
            "selection_manifest_sha256": selection_sha,
            "fits": len(fits), "calibration_rows": len(calibration_rows),
            "all_folds_feasible": all_feasible,
            "outer_rows": len(outer_rows), "selections": selections,
            "statistics": statistics,
            "validation": {
                "exact_fit_key_product": True,
                "training_logs_and_checkpoints": True,
                "exact_calibration_key_product": True,
                "exact_outer_key_product": True,
                "decision_provenance_bindings": True,
                "inputs_revalidated_before_outer_and_result": True,
                "persisted_artifacts_revalidated_before_outer_and_result": True,
            },
            "claim_boundary": {
                "complete_confirmatory_evidence": False,
                "independent_replication": False,
                "hidden_test_or_ood": False,
                "llm_weight_rl": False,
            },
        }
        _write_json(output / "result.json", result)
        (output / "RUNNING.json").unlink()
        _write_json(output / "artifact-manifest.json", _manifest(output, source["revision"]))
        print(json.dumps({
            "output": str(output), "mode": run_schedule.mode,
            "fits": len(fits), "calibration_rows": len(calibration_rows),
            "all_folds_feasible": all_feasible, "outer_rows": len(outer_rows),
            "selections": [row["selected_mechanism"] for row in selections],
            "adaptive_gate": (
                statistics["adaptive_development_gate_passed"]
                if statistics else False
            ),
            "raw_gate_conjunction": (
                statistics["raw_gate_conjunction"] if statistics else False
            ),
        }, ensure_ascii=False, indent=2))
        return 0
    except BaseException as exc:
        _isolate_valid_outputs(output)
        failure = {
            "schema_version": "multitown-a22-invalidated-attempt-v1",
            "invalidated": True, "error_type": type(exc).__name__,
            "error": str(exc), "traceback": traceback.format_exc(),
            "selective_retry_forbidden": True,
            "failed_at_utc": datetime.now(UTC).isoformat(),
        }
        try:
            _write_json(output / "INVALIDATED.json", failure)
        except Exception:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    raise SystemExit(run(args.output_dir, smoke=args.smoke))


if __name__ == "__main__":
    main()
