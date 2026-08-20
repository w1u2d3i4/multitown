"""Implement and smoke-test A24 without authorizing its formal attempt."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import signal
import stat
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .a22_constrained_ppo import (
    MECHANISMS as A22_MECHANISMS,
    constrained_rollout,
    deterministic_action,
    thresholds_from_inner_train,
)
from .a22_runner import (
    _validate_checkpoint as _validate_a22_checkpoint,
    partition_ids,
    schedule as a22_schedule,
)
from .a23_cr_ppo import mode_sequence_sha256, model_parameter_sha256
from .a24_artifact_state import (
    FormalLockCreatedError,
    FormalTerminationRequested,
    SUCCESS_TO_INVALID,
    acquire_formal_lock,
    invalidate_postlock_failure,
    isolate_success_shaped,
    lock_binding as formal_lock_binding_payload,
    lock_descriptor,
    manifest_payload,
    raw_snapshot,
    sha256_file,
    supervised_postlock_signals,
    validate_manifest,
    verify_formal_lock,
    verify_raw_snapshot,
)
from .a24_contract import (
    A24_FORMAL_THREADS,
    ALL_FITS_VERSION,
    CALIBRATION_GATE_VERSION,
    FIT_COMPLETE_VERSION,
    FORMAL_FOLDS,
    FORMAL_LOCK,
    FORMAL_SEEDS,
    LOCK_VERSION,
    MECHANISM,
    POLICY_VERSION,
    RESULT_VERSION,
    RUNNER_VERSION,
    RUN_CONTRACT_VERSION,
    TRAINING_CONTRACT_VERSION,
    UPDATE_LOG_VERSION,
    atomic_replace_json,
    atomic_write_json,
    atomic_write_jsonl,
    canonical_json_bytes,
    expected_managed_paths,
    fit_prefix,
    fsync_directory,
    same_typed_json,
    strict_read_json,
    strict_read_jsonl,
)
from .a24_inputs import (
    EXPECTED_A22,
    EXPECTED_A22_RUN_DIGEST,
    EXPECTED_R2,
    revalidate_inputs,
    verify_inputs,
)
from .a24_statistics import (
    A24StatisticsSchedule,
    build_claim_boundary,
    calibration_comparator_diagnostic,
    evaluate_calibration_gate,
    result_statistics,
)
from .a9_oof_protocol import DEFAULT_FOLDS
from .a9_ppo_oof import (
    FORMAL_EPISODES_PER_UPDATE,
    FORMAL_UPDATES,
    _digest,
    _evaluate_episode,
    _formal_ppo_config,
    _set_seed,
)
from .long_horizon_env import ACTION_COUNT, LongHorizonEpisode, MultiTownLongHorizonEnv
from .ppo_controller import ActorCritic, PPOConfig, _save_checkpoint
from .pq1_numerical_conformance import (
    PQ1_PRIMITIVES_VERSION,
    optimizer_state_sha256,
    pq1_cr_ppo_update,
    transition_episode_sha256,
)


_FORMAL_EXECUTION_CAPABILITY = object()


@dataclass(frozen=True)
class A24Schedule:
    mode: str
    seeds: tuple[int, ...]
    folds: tuple[int, ...]
    updates: int
    episodes_per_update: int
    calibration_episodes_per_fold: int
    outer_episodes_per_fold: int
    bootstrap_iterations: int
    threads: int


def schedule(smoke: bool) -> A24Schedule:
    if smoke:
        return A24Schedule(
            mode="non-evidentiary-smoke",
            seeds=(FORMAL_SEEDS[0],),
            folds=FORMAL_FOLDS,
            updates=1,
            episodes_per_update=4,
            calibration_episodes_per_fold=8,
            outer_episodes_per_fold=8,
            bootstrap_iterations=200,
            threads=2,
        )
    return A24Schedule(
        mode="adaptive-same-bank-development",
        seeds=FORMAL_SEEDS,
        folds=FORMAL_FOLDS,
        updates=FORMAL_UPDATES,
        episodes_per_update=FORMAL_EPISODES_PER_UPDATE,
        calibration_episodes_per_fold=600,
        outer_episodes_per_fold=600,
        bootstrap_iterations=20_000,
        threads=A24_FORMAL_THREADS,
    )


def expected_products(
    run_schedule: A24Schedule, *, gate_open: bool,
) -> dict[str, int]:
    fits = len(run_schedule.folds) * len(run_schedule.seeds)
    return {
        "fits": fits,
        "optimizer_updates": fits * run_schedule.updates,
        "training_episode_draws": (
            fits * run_schedule.updates * run_schedule.episodes_per_update
        ),
        "a24_calibration_rows": (
            fits * run_schedule.calibration_episodes_per_fold
        ),
        "a22_lagrangian_calibration_source_rows": (
            fits * run_schedule.calibration_episodes_per_fold
        ),
        "a24_outer_rows": (
            fits * run_schedule.outer_episodes_per_fold if gate_open else 0
        ),
        "a22_lagrangian_outer_rows": (
            fits * run_schedule.outer_episodes_per_fold if gate_open else 0
        ),
        "a8_outer_source_rows": (
            len(run_schedule.folds) * run_schedule.outer_episodes_per_fold
            if gate_open
            else 0
        ),
        "manifest_entries": len(
            expected_managed_paths(
                run_schedule.folds,
                run_schedule.seeds,
                gate_open=gate_open,
            )
        ),
    }


def _config(run_schedule: A24Schedule) -> PPOConfig:
    values = asdict(_formal_ppo_config())
    values.update(
        {
            "updates": run_schedule.updates,
            "episodes_per_update": run_schedule.episodes_per_update,
        }
    )
    return PPOConfig(**values)


def _fit_path(output: Path, fold: int, seed: int) -> Path:
    return output / fit_prefix(fold, seed)


def _make_private_tree(path: Path, *, root: Path) -> None:
    current = root
    for part in path.relative_to(root).parts:
        current = current / part
        if not current.exists():
            current.mkdir(mode=0o700)
            fsync_directory(current.parent)
        os.chmod(current, 0o700)
        metadata = current.lstat()
        if current.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError("A24 output tree contains non-directory")


def _checkpoint_atomic(
    path: Path,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    config: PPOConfig,
    *,
    seed: int,
    update: int,
    metadata: Mapping[str, Any],
) -> None:
    partial = path.with_name(path.name + ".partial")
    if path.exists() or partial.exists():
        raise FileExistsError(path if path.exists() else partial)
    _save_checkpoint(
        partial,
        model,
        config,
        seed=seed,
        update=update,
        policy_version=POLICY_VERSION,
    )
    payload = torch.load(partial, map_location="cpu", weights_only=False)
    payload["optimizer_state"] = optimizer.state_dict()
    payload.update(dict(metadata))
    torch.save(payload, partial)
    os.chmod(partial, 0o600)
    descriptor = os.open(partial, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    verified = torch.load(partial, map_location="cpu", weights_only=False)
    if any(not bool(torch.isfinite(value).all()) for value in verified["model_state"].values()):
        raise FloatingPointError("A24 checkpoint contains non-finite model state")
    if any(verified.get(key) != value for key, value in metadata.items()):
        raise RuntimeError("A24 checkpoint metadata changed before publication")
    os.replace(partial, path)
    fsync_directory(path.parent)


def _load_a24_checkpoint(
    path: Path,
    *,
    fit: Mapping[str, Any],
    run_schedule: A24Schedule,
    run_contract_sha256: str,
) -> tuple[ActorCritic, torch.optim.Optimizer, dict[str, Any]]:
    if (
        not path.is_file()
        or path.is_symlink()
        or sha256_file(path) != fit["checkpoint_sha256"]
    ):
        raise RuntimeError("A24 checkpoint disk binding changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = _config(run_schedule)
    expected_keys = {
        "policy_version",
        "observation_size",
        "action_count",
        "hidden_size",
        "seed",
        "update",
        "ppo_config",
        "model_state",
        "optimizer_state",
        "pq1_primitives_version",
        "run_contract_sha256",
        "mechanism",
        "outer_fold",
        "training_log_sha256",
        "sample_sequence_sha256",
        "mode_sequence_sha256",
        "initial_model_sha256",
        "initial_optimizer_sha256",
        "final_model_sha256",
        "final_optimizer_sha256",
    }
    if (
        type(payload) is not dict
        or set(payload) != expected_keys
        or payload["policy_version"] != POLICY_VERSION
        or payload["pq1_primitives_version"] != PQ1_PRIMITIVES_VERSION
        or payload["mechanism"] != MECHANISM
        or payload["run_contract_sha256"] != run_contract_sha256
        or payload["update"] != run_schedule.updates
        or not same_typed_json(payload["ppo_config"], asdict(config))
        or payload["observation_size"] != MultiTownLongHorizonEnv.observation_size
        or payload["action_count"] != ACTION_COUNT
        or payload["hidden_size"] != config.hidden_size
        or type(payload["seed"]) is not int
        or type(payload["outer_fold"]) is not int
        or type(payload["update"]) is not int
        or type(payload["observation_size"]) is not int
        or type(payload["action_count"]) is not int
        or type(payload["hidden_size"]) is not int
        or payload["seed"] != fit["training_seed"]
        or payload["outer_fold"] != fit["outer_fold"]
        or payload["initial_model_sha256"] != fit["initial_model_sha256"]
        or payload["initial_optimizer_sha256"]
        != fit["initial_optimizer_sha256"]
        or payload["final_model_sha256"] != fit["final_model_sha256"]
        or payload["final_optimizer_sha256"]
        != fit["final_optimizer_sha256"]
    ):
        raise RuntimeError("A24 checkpoint exact schema or contract mismatch")
    model = ActorCritic(
        MultiTownLongHorizonEnv.observation_size,
        config.hidden_size,
        ACTION_COUNT,
    ).cpu()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        eps=1e-5,
    )
    try:
        model.load_state_dict(payload["model_state"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state"])
    except Exception as exc:
        raise RuntimeError("A24 checkpoint model/optimizer cannot be reconstructed") from exc
    if (
        model_parameter_sha256(model) != payload["final_model_sha256"]
        or optimizer_state_sha256(optimizer, model)
        != payload["final_optimizer_sha256"]
    ):
        raise RuntimeError("A24 checkpoint reconstructed digest mismatch")
    return model, optimizer, payload


def fit_no_shield(
    episodes: Sequence[LongHorizonEpisode],
    *,
    a8_rows: Sequence[Mapping[str, Any]],
    seed: int,
    outer_fold: int,
    run_schedule: A24Schedule,
    output: Path,
    run_contract_sha256: str,
    partition: Mapping[str, Any],
    comparator_entry: Mapping[str, Any],
    expected_sample_ids: Sequence[str],
) -> dict[str, Any]:
    canonical_ids = [episode.episode_id for episode in episodes]
    if (
        len(canonical_ids) != 1800
        or canonical_ids != sorted(canonical_ids)
        or set(canonical_ids) != set(partition["inner_train_ids"])
    ):
        raise ValueError("A24 fit requires exact canonical inner-train partition")
    config = _config(run_schedule)
    thresholds = thresholds_from_inner_train(a8_rows)
    _set_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    model = ActorCritic(
        MultiTownLongHorizonEnv.observation_size,
        config.hidden_size,
        ACTION_COUNT,
    ).cpu()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        eps=1e-5,
    )
    initial_model = model_parameter_sha256(model)
    initial_optimizer = optimizer_state_sha256(optimizer, model)
    expected_initial = comparator_entry["derived_expected_initialization"]
    if (
        initial_model != expected_initial["model_sha256"]
        or initial_optimizer != expected_initial["named_optimizer_sha256"]
    ):
        raise RuntimeError("A24 initialization differs from derived A22 expectation")
    sample_rng = random.Random(seed)
    tensor_generator = torch.Generator(device="cpu").manual_seed(seed)
    logs: list[dict[str, Any]] = []
    sampled_ids: list[str] = []
    modes: list[str] = []
    mode_counts: Counter[str] = Counter()
    total_steps = 0
    total_minibatches = 0
    started = time.perf_counter()
    for update in range(1, config.updates + 1):
        pre_model = model_parameter_sha256(model)
        pre_optimizer = optimizer_state_sha256(optimizer, model)
        transition_episodes = []
        transition_hashes = []
        rollout_rows = []
        update_ids = []
        for _ in range(config.episodes_per_update):
            episode = episodes[sample_rng.randrange(len(episodes))]
            update_ids.append(episode.episode_id)
            transitions, rollout = constrained_rollout(
                model,
                episode,
                torch.device("cpu"),
                mean_incidents=thresholds.mean_incidents,
                shield_enabled=False,
            )
            digest = transition_episode_sha256(transitions)
            transition_episodes.append(transitions)
            transition_hashes.append(digest)
            rollout_rows.append(rollout)
        post_rollout_model = model_parameter_sha256(model)
        post_rollout_optimizer = optimizer_state_sha256(optimizer, model)
        if pre_model != post_rollout_model or pre_optimizer != post_rollout_optimizer:
            raise RuntimeError("A24 rollout mutated model or optimizer")
        decision, metrics = pq1_cr_ppo_update(
            model,
            optimizer,
            transition_episodes,
            config,
            torch.device("cpu"),
            tensor_generator,
            thresholds=thresholds,
            rollout_model_sha256=pre_model,
            rollout_optimizer_sha256=pre_optimizer,
            rollout_transition_sha256=transition_hashes,
        )
        if [transition_episode_sha256(rows) for rows in transition_episodes] != transition_hashes:
            raise RuntimeError("A24 transitions changed across optimizer boundary")
        snapshot = metrics.pop("snapshot_diagnostics")
        if (
            not all(snapshot["diagnostic_gates"].values())
            or snapshot["rowwise_max_batch_size"] != 1
            or snapshot["rowwise_forward_calls"] != snapshot["transition_count"]
        ):
            raise RuntimeError("A24 PQ numerical/provenance gate failed")
        mode_counts[decision.mode] += 1
        modes.append(decision.mode)
        sampled_ids.extend(update_ids)
        steps = sum(len(rows) for rows in transition_episodes)
        minibatches = config.ppo_epochs * math.ceil(steps / config.minibatch_size)
        total_steps += steps
        total_minibatches += minibatches
        row = {
            "schema_version": UPDATE_LOG_VERSION,
            "pq1_primitives_version": PQ1_PRIMITIVES_VERSION,
            "outer_fold": outer_fold,
            "training_seed": seed,
            "mechanism": MECHANISM,
            "update": update,
            "episodes_per_update": config.episodes_per_update,
            "sampled_episode_ids": update_ids,
            "sampled_episode_ids_sha256": _digest(update_ids),
            "transition_episode_sha256": transition_hashes,
            "pre_rollout_model_sha256": pre_model,
            "post_rollout_model_sha256": post_rollout_model,
            "pre_rollout_optimizer_sha256": pre_optimizer,
            "post_rollout_optimizer_sha256": post_rollout_optimizer,
            "post_update_model_sha256": model_parameter_sha256(model),
            "post_update_optimizer_sha256": optimizer_state_sha256(optimizer, model),
            "selected_actor_mode": decision.mode,
            "mode_counts": {
                name: mode_counts[name] for name in ("reward", "unsafe", "wrong")
            },
            "environment_steps": steps,
            "optimizer_minibatches": minibatches,
            "rollout_summary": {
                "episodes": len(rollout_rows),
                "incidents": sum(int(item["incidents"]) for item in rollout_rows),
                "unsafe_events": sum(bool(item["unsafe_episode"]) for item in rollout_rows),
                "wrong_executions": sum(int(item["wrong_executions"]) for item in rollout_rows),
                "shield_interventions": sum(int(item["shield_interventions"]) for item in rollout_rows),
            },
            "actor_mode_decision": asdict(decision),
            "snapshot_diagnostics": snapshot,
            "ppo_metrics": {
                key: value
                for key, value in metrics.items()
                if key
                not in {
                    "selected_actor_mode",
                    "normalized_advantage_sha256",
                    "selected_advantage_constant",
                    "reward_advantage_raw",
                    "unsafe_advantage_raw",
                    "wrong_advantage_raw",
                    "selected_advantage_raw",
                }
            },
            "advantage_diagnostics": {
                key: metrics[key]
                for key in (
                    "selected_actor_mode",
                    "normalized_advantage_sha256",
                    "selected_advantage_constant",
                    "reward_advantage_raw",
                    "unsafe_advantage_raw",
                    "wrong_advantage_raw",
                    "selected_advantage_raw",
                )
            },
        }
        if (
            row["rollout_summary"]["shield_interventions"] != 0
            or sum(row["mode_counts"].values()) != update
        ):
            raise RuntimeError("A24 no-shield/update-count invariant failed")
        logs.append(row)
        progress = {
            "schema_version": "multitown-a24-fit-progress-v1",
            "outer_fold": outer_fold,
            "training_seed": seed,
            "mechanism": MECHANISM,
            "current_update": update,
            "scheduled_updates": config.updates,
            "mode_counts": row["mode_counts"],
            "calibration_started": False,
            "outer_evaluation_started": False,
        }
        if update == 1:
            atomic_write_json(output / "progress.json", progress)
        else:
            atomic_replace_json(output / "progress.json", progress)
    if sampled_ids != list(expected_sample_ids):
        raise RuntimeError("A24 sample sequence differs from pinned A22")
    atomic_write_jsonl(output / "training-metrics.jsonl", logs)
    training_log_sha = sha256_file(output / "training-metrics.jsonl")
    final_model = model_parameter_sha256(model)
    final_optimizer = optimizer_state_sha256(optimizer, model)
    checkpoint = output / "final.pt"
    checkpoint_metadata = {
        "pq1_primitives_version": PQ1_PRIMITIVES_VERSION,
        "run_contract_sha256": run_contract_sha256,
        "mechanism": MECHANISM,
        "outer_fold": outer_fold,
        "training_log_sha256": training_log_sha,
        "sample_sequence_sha256": _digest(sampled_ids),
        "mode_sequence_sha256": mode_sequence_sha256(modes),
        "initial_model_sha256": initial_model,
        "initial_optimizer_sha256": initial_optimizer,
        "final_model_sha256": final_model,
        "final_optimizer_sha256": final_optimizer,
    }
    _checkpoint_atomic(
        checkpoint,
        model,
        optimizer,
        config,
        seed=seed,
        update=config.updates,
        metadata=checkpoint_metadata,
    )
    complete = {
        "schema_version": FIT_COMPLETE_VERSION,
        "pq1_primitives_version": PQ1_PRIMITIVES_VERSION,
        "outer_fold": outer_fold,
        "training_seed": seed,
        "mechanism": MECHANISM,
        "shield_enabled": False,
        "final_update": config.updates,
        "training_episode_draws": len(sampled_ids),
        "sample_sequence_sha256": _digest(sampled_ids),
        "sampled_unique_episodes": len(set(sampled_ids)),
        "initial_model_sha256": initial_model,
        "initial_optimizer_sha256": initial_optimizer,
        "final_model_sha256": final_model,
        "final_optimizer_sha256": final_optimizer,
        "mode_sequence_sha256": mode_sequence_sha256(modes),
        "mode_counts": {name: mode_counts[name] for name in ("reward", "unsafe", "wrong")},
        "training_thresholds": asdict(thresholds),
        "training_log_sha256": training_log_sha,
        "checkpoint_sha256": sha256_file(checkpoint),
        "inner_train_ids_sha256": partition["inner_train_ids_sha256"],
        "calibration_ids_sha256": partition["inner_calibration_ids_sha256"],
        "outer_ids_sha256": partition["outer_ids_sha256"],
        "environment_steps": total_steps,
        "optimizer_minibatches": total_minibatches,
        "run_contract_sha256": run_contract_sha256,
        "calibration_evaluations_during_training": 0,
        "outer_evaluations_during_training": 0,
        "training_seconds": time.perf_counter() - started,
        "selected_checkpoint": "final",
    }
    atomic_write_json(output / "fit-complete.json", complete)
    return complete


def _validate_actor_log(
    row: Mapping[str, Any],
    *,
    thresholds: Mapping[str, Any],
) -> None:
    decision = row.get("actor_mode_decision")
    diagnostics = row.get("advantage_diagnostics")
    if type(decision) is not dict or type(diagnostics) is not dict:
        raise RuntimeError("A24 actor diagnostics are not objects")
    if set(diagnostics) != {
        "selected_actor_mode",
        "normalized_advantage_sha256",
        "selected_advantage_constant",
        "reward_advantage_raw",
        "unsafe_advantage_raw",
        "wrong_advantage_raw",
        "selected_advantage_raw",
    } or type(diagnostics["selected_advantage_constant"]) is not bool:
        raise RuntimeError("A24 advantage diagnostics schema changed")
    expected_decision_keys = {
        "mode",
        "unsafe_cost",
        "wrong_cost",
        "unsafe_threshold",
        "wrong_threshold",
        "unsafe_violation",
        "wrong_violation",
        "unsafe_normalized_violation",
        "wrong_normalized_violation",
        "unsafe_eligible",
        "wrong_eligible",
        "unsafe_tie_break_used",
    }
    if set(decision) != expected_decision_keys:
        raise RuntimeError("A24 actor decision schema changed")
    mode = decision["mode"]
    float_fields = expected_decision_keys - {
        "mode",
        "unsafe_eligible",
        "wrong_eligible",
        "unsafe_tie_break_used",
    }
    if (
        type(mode) is not str
        or mode not in {"reward", "unsafe", "wrong"}
        or any(type(decision[name]) is not float for name in float_fields)
        or not all(math.isfinite(decision[name]) for name in float_fields)
        or type(decision["unsafe_eligible"]) is not bool
        or type(decision["wrong_eligible"]) is not bool
        or type(decision["unsafe_tie_break_used"]) is not bool
    ):
        raise RuntimeError("A24 actor decision types changed")
    unsafe_violation = decision["unsafe_cost"] - decision["unsafe_threshold"]
    wrong_violation = decision["wrong_cost"] - decision["wrong_threshold"]
    unsafe_normalized = unsafe_violation / max(
        decision["unsafe_threshold"], 1e-8
    )
    wrong_normalized = wrong_violation / max(decision["wrong_threshold"], 1e-8)
    unsafe_eligible = decision["unsafe_cost"] > decision["unsafe_threshold"]
    wrong_eligible = decision["wrong_cost"] > decision["wrong_threshold"]
    tie = bool(
        unsafe_eligible
        and wrong_eligible
        and unsafe_normalized == wrong_normalized
    )
    expected_mode = (
        "reward"
        if not unsafe_eligible and not wrong_eligible
        else (
            "unsafe"
            if unsafe_eligible
            and (not wrong_eligible or unsafe_normalized >= wrong_normalized)
            else "wrong"
        )
    )
    rollout = row["rollout_summary"]
    expected_unsafe_cost = rollout["unsafe_events"] / rollout["episodes"]
    expected_wrong_cost = (
        rollout["wrong_executions"]
        / rollout["episodes"]
        / thresholds["mean_incidents"]
    )
    if (
        decision["unsafe_cost"] != expected_unsafe_cost
        or decision["wrong_cost"] != expected_wrong_cost
        or decision["unsafe_threshold"] != thresholds["unsafe"]
        or decision["wrong_threshold"] != thresholds["wrong_per_incident"]
        or decision["unsafe_violation"] != unsafe_violation
        or decision["wrong_violation"] != wrong_violation
        or decision["unsafe_normalized_violation"] != unsafe_normalized
        or decision["wrong_normalized_violation"] != wrong_normalized
        or decision["unsafe_eligible"] is not unsafe_eligible
        or decision["wrong_eligible"] is not wrong_eligible
        or decision["unsafe_tie_break_used"] is not tie
        or mode != expected_mode
        or row.get("selected_actor_mode") != mode
        or diagnostics.get("selected_actor_mode") != mode
    ):
        raise RuntimeError("A24 actor decision arithmetic changed")
    summary_by_mode = {
        "reward": diagnostics.get("reward_advantage_raw"),
        "unsafe": diagnostics.get("unsafe_advantage_raw"),
        "wrong": diagnostics.get("wrong_advantage_raw"),
    }
    selected = diagnostics.get("selected_advantage_raw")
    source = summary_by_mode[mode]
    if (
        type(selected) is not dict
        or type(source) is not dict
        or set(selected) != {"mean", "std", "max_abs"}
        or set(source) != {"mean", "std", "max_abs"}
    ):
        raise RuntimeError("A24 advantage summary schema changed")
    summaries = tuple(summary_by_mode.values()) + (selected,)
    if any(type(summary) is not dict for summary in summaries) or any(
        set(summary) != {"mean", "std", "max_abs"} for summary in summaries
    ):
        raise RuntimeError("A24 advantage summary schema changed")
    if any(
        type(summary.get(name)) is not float
        or not math.isfinite(summary[name])
        for summary in summaries
        for name in ("mean", "std", "max_abs")
    ) or not _is_sha256(diagnostics.get("normalized_advantage_sha256")):
        raise RuntimeError("A24 advantage summary types changed")
    sign = 1.0 if mode == "reward" else -1.0
    if (
        selected["mean"] != sign * source["mean"]
        or selected["std"] != source["std"]
        or selected["max_abs"] != source["max_abs"]
    ):
        raise RuntimeError("A24 selected advantage sign/source changed")


def _is_sha256(value: Any) -> bool:
    return bool(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_snapshot_schema(snapshot: Mapping[str, Any]) -> None:
    if type(snapshot) is not dict or set(snapshot) != {
        "schema_version",
        "rowwise_log_probability_exact",
        "rowwise_value_exact",
        "rowwise_forward_calls",
        "rowwise_max_batch_size",
        "transition_count",
        "observation_sha256",
        "mask_sha256",
        "action_sha256",
        "full_batch_log_probability",
        "full_batch_value",
        "max_probability_ratio_drift",
        "diagnostic_gates",
        "first_legacy_exceed_transition_sha256",
        "ordered_transition_episode_sha256",
        "ordered_transition_batch_sha256",
    }:
        raise RuntimeError("A24 PQ snapshot schema changed")
    if (
        snapshot["schema_version"] != PQ1_PRIMITIVES_VERSION
        or snapshot["rowwise_log_probability_exact"] is not True
        or snapshot["rowwise_value_exact"] is not True
        or any(
            type(snapshot[name]) is not int
            for name in (
                "rowwise_forward_calls",
                "rowwise_max_batch_size",
                "transition_count",
            )
        )
        or any(
            not _is_sha256(snapshot[name])
            for name in ("observation_sha256", "mask_sha256", "action_sha256")
        )
        or type(snapshot["max_probability_ratio_drift"]) is not float
        or not math.isfinite(snapshot["max_probability_ratio_drift"])
        or (
            snapshot["first_legacy_exceed_transition_sha256"] is not None
            and not _is_sha256(
                snapshot["first_legacy_exceed_transition_sha256"]
            )
        )
    ):
        raise RuntimeError("A24 PQ snapshot receipt types changed")
    summary_keys = {
        "legacy_exceedances",
        "legacy_max_tolerance_ratio",
        "max_abs",
        "max_relative",
        "max_ulp",
        "p50_abs",
        "p95_abs",
        "p99_abs",
    }
    for name in ("full_batch_log_probability", "full_batch_value"):
        summary = snapshot[name]
        if (
            type(summary) is not dict
            or set(summary) != summary_keys
            or type(summary["legacy_exceedances"]) is not int
            or type(summary["max_ulp"]) is not int
            or any(
                type(summary[field]) is not float
                or not math.isfinite(summary[field])
                for field in summary_keys
                - {"legacy_exceedances", "max_ulp"}
            )
        ):
            raise RuntimeError("A24 PQ full-batch summary schema changed")


def validate_fit_artifacts(
    fit_dir: Path,
    *,
    expected_fit: Mapping[str, Any],
    run_schedule: A24Schedule,
    run_contract_sha256: str,
    expected_sample_ids: Sequence[str],
    expected_thresholds: Mapping[str, Any],
) -> tuple[ActorCritic, torch.optim.Optimizer]:
    actual_files = {
        path.name for path in fit_dir.iterdir() if path.is_file()
    }
    if actual_files != {
        "training-metrics.jsonl",
        "final.pt",
        "fit-complete.json",
        "progress.json",
    } or any(path.is_symlink() for path in fit_dir.iterdir()):
        raise RuntimeError("A24 fit exact physical inventory mismatch")
    persisted = strict_read_json(fit_dir / "fit-complete.json")
    if not same_typed_json(persisted, dict(expected_fit)):
        raise RuntimeError("A24 fit-complete changed")
    if expected_fit.get("schema_version") != FIT_COMPLETE_VERSION:
        raise RuntimeError("A24 fit-complete schema version changed")
    logs = strict_read_jsonl(fit_dir / "training-metrics.jsonl")
    progress = strict_read_json(fit_dir / "progress.json")
    if len(logs) != run_schedule.updates:
        raise RuntimeError("A24 fit update count mismatch")
    thresholds = expected_fit.get("training_thresholds")
    if (
        type(thresholds) is not dict
        or set(thresholds) != {"unsafe", "wrong_per_incident", "mean_incidents"}
        or any(type(value) is not float for value in thresholds.values())
        or not all(math.isfinite(value) for value in thresholds.values())
        or not 0.0 <= thresholds["unsafe"] <= 1.0
        or not 0.0 <= thresholds["wrong_per_incident"] <= 1.0
        or thresholds["mean_incidents"] <= 0.0
        or not same_typed_json(thresholds, expected_thresholds)
    ):
        raise RuntimeError("A24 fit training thresholds changed")
    sampled = [
        str(episode_id)
        for row in logs
        for episode_id in row["sampled_episode_ids"]
    ]
    modes = [str(row["selected_actor_mode"]) for row in logs]
    counts: Counter[str] = Counter()
    previous_model = expected_fit["initial_model_sha256"]
    previous_optimizer = expected_fit["initial_optimizer_sha256"]
    total_steps = 0
    total_minibatches = 0
    config = _config(run_schedule)
    for expected_update, row in enumerate(logs, start=1):
        if type(row) is not dict or type(row.get("selected_actor_mode")) is not str:
            raise RuntimeError("A24 update log row is malformed")
        counts[row["selected_actor_mode"]] += 1
        transition_hashes = row.get("transition_episode_sha256")
        snapshot = row.get("snapshot_diagnostics")
        rollout = row.get("rollout_summary")
        diagnostic_gates = (
            snapshot.get("diagnostic_gates") if type(snapshot) is dict else None
        )
        expected_batch_hash = (
            hashlib.sha256("".join(transition_hashes).encode("ascii")).hexdigest()
            if type(transition_hashes) is list
            and all(type(value) is str for value in transition_hashes)
            else None
        )
        expected_minibatches = (
            config.ppo_epochs
            * math.ceil(row["environment_steps"] / config.minibatch_size)
            if type(row.get("environment_steps")) is int
            else None
        )
        if (
            set(row)
            != {
                "schema_version",
                "pq1_primitives_version",
                "outer_fold",
                "training_seed",
                "mechanism",
                "update",
                "episodes_per_update",
                "sampled_episode_ids",
                "sampled_episode_ids_sha256",
                "transition_episode_sha256",
                "pre_rollout_model_sha256",
                "post_rollout_model_sha256",
                "pre_rollout_optimizer_sha256",
                "post_rollout_optimizer_sha256",
                "post_update_model_sha256",
                "post_update_optimizer_sha256",
                "selected_actor_mode",
                "mode_counts",
                "environment_steps",
                "optimizer_minibatches",
                "rollout_summary",
                "actor_mode_decision",
                "snapshot_diagnostics",
                "ppo_metrics",
                "advantage_diagnostics",
            }
            or row["schema_version"] != UPDATE_LOG_VERSION
            or row["pq1_primitives_version"] != PQ1_PRIMITIVES_VERSION
            or type(row["outer_fold"]) is not int
            or type(row["training_seed"]) is not int
            or type(row["update"]) is not int
            or row["update"] != expected_update
            or row["outer_fold"] != expected_fit["outer_fold"]
            or row["training_seed"] != expected_fit["training_seed"]
            or row["mechanism"] != MECHANISM
            or type(row["episodes_per_update"]) is not int
            or row["episodes_per_update"] != run_schedule.episodes_per_update
            or type(row["sampled_episode_ids"]) is not list
            or len(row["sampled_episode_ids"])
            != run_schedule.episodes_per_update
            or any(type(value) is not str for value in row["sampled_episode_ids"])
            or type(transition_hashes) is not list
            or len(transition_hashes) != run_schedule.episodes_per_update
            or any(
                not _is_sha256(value)
                for value in transition_hashes
            )
            or not _is_sha256(row["sampled_episode_ids_sha256"])
            or row["sampled_episode_ids_sha256"]
            != _digest(row["sampled_episode_ids"])
            or any(
                not _is_sha256(row[name])
                for name in (
                    "pre_rollout_model_sha256",
                    "post_rollout_model_sha256",
                    "pre_rollout_optimizer_sha256",
                    "post_rollout_optimizer_sha256",
                    "post_update_model_sha256",
                    "post_update_optimizer_sha256",
                )
            )
            or row["pre_rollout_model_sha256"] != previous_model
            or row["post_rollout_model_sha256"] != previous_model
            or row["pre_rollout_optimizer_sha256"] != previous_optimizer
            or row["post_rollout_optimizer_sha256"] != previous_optimizer
            or type(row["environment_steps"]) is not int
            or row["environment_steps"] <= 0
            or type(row["optimizer_minibatches"]) is not int
            or row["optimizer_minibatches"] != expected_minibatches
            or type(rollout) is not dict
            or set(rollout)
            != {
                "episodes",
                "incidents",
                "unsafe_events",
                "wrong_executions",
                "shield_interventions",
            }
            or any(type(rollout[name]) is not int for name in rollout)
            or any(rollout[name] < 0 for name in rollout)
            or rollout["episodes"] != run_schedule.episodes_per_update
            or rollout["unsafe_events"] > rollout["episodes"]
            or rollout["wrong_executions"] < rollout["unsafe_events"]
            or rollout["shield_interventions"] != 0
            or type(row["mode_counts"]) is not dict
            or set(row["mode_counts"]) != {"reward", "unsafe", "wrong"}
            or any(type(value) is not int for value in row["mode_counts"].values())
            or row["mode_counts"]
            != {name: counts[name] for name in ("reward", "unsafe", "wrong")}
            or sum(row["mode_counts"].values()) != expected_update
            or type(snapshot) is not dict
            or type(diagnostic_gates) is not dict
            or set(diagnostic_gates)
            != {
                "full_batch_log_within_frozen_tolerance",
                "full_batch_value_within_frozen_tolerance",
                "probability_ratio_drift_within_2e_5",
            }
            or any(type(value) is not bool or value is not True for value in diagnostic_gates.values())
            or snapshot.get("rowwise_log_probability_exact") is not True
            or snapshot.get("rowwise_value_exact") is not True
            or type(snapshot.get("rowwise_max_batch_size")) is not int
            or snapshot["rowwise_max_batch_size"] != 1
            or type(snapshot.get("rowwise_forward_calls")) is not int
            or type(snapshot.get("transition_count")) is not int
            or snapshot["rowwise_forward_calls"] != snapshot["transition_count"]
            or snapshot["transition_count"] != row["environment_steps"]
            or snapshot.get("ordered_transition_episode_sha256")
            != transition_hashes
            or snapshot.get("ordered_transition_batch_sha256")
            != expected_batch_hash
            or not _is_sha256(snapshot.get("ordered_transition_batch_sha256"))
            or type(row.get("ppo_metrics")) is not dict
            or any(
                type(value) not in (int, float) or not math.isfinite(float(value))
                for value in row["ppo_metrics"].values()
            )
            or row["ppo_metrics"].get("rollout_unsafe_events")
            != rollout["unsafe_events"]
            or row["ppo_metrics"].get("rollout_wrong_executions")
            != rollout["wrong_executions"]
        ):
            raise RuntimeError("A24 strict update-log binding mismatch")
        _validate_snapshot_schema(snapshot)
        _validate_actor_log(row, thresholds=thresholds)
        total_steps += row["environment_steps"]
        total_minibatches += row["optimizer_minibatches"]
        previous_model = row["post_update_model_sha256"]
        previous_optimizer = row["post_update_optimizer_sha256"]
    if (
        sampled != list(expected_sample_ids)
        or expected_fit["sample_sequence_sha256"] != _digest(sampled)
        or expected_fit["mode_sequence_sha256"] != mode_sequence_sha256(modes)
        or expected_fit["training_log_sha256"]
        != sha256_file(fit_dir / "training-metrics.jsonl")
        or expected_fit["checkpoint_sha256"] != sha256_file(fit_dir / "final.pt")
        or expected_fit["final_model_sha256"] != previous_model
        or expected_fit["final_optimizer_sha256"] != previous_optimizer
        or expected_fit["outer_fold"] not in run_schedule.folds
        or expected_fit["training_seed"] not in run_schedule.seeds
        or expected_fit["final_update"] != run_schedule.updates
        or expected_fit["training_episode_draws"]
        != run_schedule.updates * run_schedule.episodes_per_update
        or expected_fit["environment_steps"] != total_steps
        or expected_fit["optimizer_minibatches"] != total_minibatches
        or expected_fit["run_contract_sha256"] != run_contract_sha256
        or not same_typed_json(
            progress,
            {
            "schema_version": "multitown-a24-fit-progress-v1",
            "outer_fold": expected_fit["outer_fold"],
            "training_seed": expected_fit["training_seed"],
            "mechanism": MECHANISM,
            "current_update": run_schedule.updates,
            "scheduled_updates": run_schedule.updates,
            "mode_counts": expected_fit["mode_counts"],
            "calibration_started": False,
            "outer_evaluation_started": False,
            },
        )
    ):
        raise RuntimeError("A24 fit log/checkpoint/progress chain changed")
    model, optimizer, payload = _load_a24_checkpoint(
        fit_dir / "final.pt",
        fit=expected_fit,
        run_schedule=run_schedule,
        run_contract_sha256=run_contract_sha256,
    )
    if (
        payload["training_log_sha256"] != expected_fit["training_log_sha256"]
        or payload["sample_sequence_sha256"]
        != expected_fit["sample_sequence_sha256"]
        or payload["mode_sequence_sha256"] != expected_fit["mode_sequence_sha256"]
        or payload["initial_model_sha256"] != expected_fit["initial_model_sha256"]
        or payload["initial_optimizer_sha256"]
        != expected_fit["initial_optimizer_sha256"]
    ):
        raise RuntimeError("A24 checkpoint-to-fit provenance changed")
    return model, optimizer


def _evaluate(
    model: ActorCritic,
    episodes: Sequence[LongHorizonEpisode],
    *,
    system: str,
    mechanism: str,
    phase: str,
    training_seed: int,
    design_outer_fold: int,
    assignment_index: Mapping[str, Any],
    a8_index: Mapping[str, Mapping[str, Any]],
    bank_sha256: str,
    resource_sha256: str,
    environment_sha256: str,
    checkpoint_sha256: str,
    run_contract_sha256: str,
    outer_gate_sha256: str | None,
) -> list[dict[str, Any]]:
    rows = []
    for episode in episodes:
        interventions = 0

        def action(observation: np.ndarray, mask: np.ndarray) -> int:
            nonlocal interventions
            selected, intervened = deterministic_action(
                model,
                observation,
                mask,
                shield_enabled=False,
            )
            interventions += int(intervened)
            return selected

        assignment = assignment_index[episode.episode_id]
        baseline = a8_index[episode.episode_id]
        row = _evaluate_episode(
            episode,
            action,
            fold=assignment.fold,
            system=system,
            training_seed=training_seed,
            episode_sha256=assignment.episode_sha256,
            bank_sha256=bank_sha256,
            resource_sha256=resource_sha256,
            environment_sha256=environment_sha256,
            checkpoint_sha256=checkpoint_sha256,
            a8_row_sha256=_digest(baseline),
            run_contract_sha256=run_contract_sha256,
        )
        row.update(
            {
                "evaluation_phase": phase,
                "design_outer_fold": design_outer_fold,
                "mechanism": mechanism,
                "shield_interventions": interventions,
                "outer_gate_sha256": outer_gate_sha256,
            }
        )
        if interventions != 0:
            raise RuntimeError("A24 no-shield evaluation recorded intervention")
        rows.append(row)
    return rows


def _a22_fit_index(context: Mapping[str, Any]) -> dict[tuple[int, int], Mapping[str, Any]]:
    payload = json.loads(
        (Path(context["a22_root"]) / "all-fits-complete.json").read_text(
            encoding="utf-8"
        )
    )
    fits = [row for row in payload["fits"] if row["mechanism"] == "lagrangian"]
    index = {(int(row["outer_fold"]), int(row["training_seed"])): row for row in fits}
    if len(fits) != 15 or len(index) != 15:
        raise RuntimeError("A24 A22 Lagrangian comparator fit product changed")
    return index


def _scheduled_a22_calibration(
    context: Mapping[str, Any],
    *,
    run_schedule: A24Schedule,
    partitions: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in (Path(context["a22_root"]) / "calibration-decisions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    expected = {
        (fold, seed, episode_id)
        for fold in run_schedule.folds
        for seed in run_schedule.seeds
        for episode_id in partitions[fold]["inner_calibration_ids"][
            : run_schedule.calibration_episodes_per_fold
        ]
    }
    selected = [
        row
        for row in rows
        if row["mechanism"] == "lagrangian"
        and (
            int(row["design_outer_fold"]),
            int(row["training_seed"]),
            str(row["episode_id"]),
        )
        in expected
    ]
    actual = [
        (
            int(row["design_outer_fold"]),
            int(row["training_seed"]),
            str(row["episode_id"]),
        )
        for row in selected
    ]
    if len(actual) != len(expected) or set(actual) != expected:
        raise RuntimeError("A24 scheduled A22 calibration product changed")
    return selected


def _statistics_schedule(run_schedule: A24Schedule) -> A24StatisticsSchedule:
    return A24StatisticsSchedule(
        mode=run_schedule.mode,
        seeds=run_schedule.seeds,
        folds=run_schedule.folds,
        updates=run_schedule.updates,
        episodes_per_update=run_schedule.episodes_per_update,
        outer_episodes_per_fold=run_schedule.outer_episodes_per_fold,
        bootstrap_iterations=run_schedule.bootstrap_iterations,
        threads=run_schedule.threads,
    )


def _fault(point: str, requested: str | None) -> None:
    if requested == point:
        raise RuntimeError(f"injected A24 smoke fault: {point}")


def _execute(
    output: Path,
    *,
    smoke: bool,
    _force_smoke_outer: bool = False,
    _inject_failure: str | None = None,
    _root: Path | None = None,
    _formal_test_fault: str | None = None,
    _formal_test_schedule: A24Schedule | None = None,
    _formal_capability: object | None = None,
) -> int:
    """Execute the shared smoke/formal engine; public formal access stays blocked."""

    if type(smoke) is not bool:
        raise ValueError("A24 execution mode flag is not boolean")
    if not smoke and (_force_smoke_outer or _inject_failure is not None):
        raise ValueError("A24 formal execution rejects smoke-only controls")
    if smoke and _formal_test_fault is not None:
        raise ValueError("A24 smoke rejects formal test controls")
    if smoke and _formal_test_schedule is not None:
        raise ValueError("A24 smoke rejects formal test schedule injection")
    fault_request = _inject_failure if smoke else _formal_test_fault
    repository_root = Path(__file__).resolve().parents[1]
    root = (
        repository_root
        if _root is None
        else _root.resolve()
    )
    if not smoke and (
        _formal_test_fault is not None or _formal_test_schedule is not None
    ) and (_root is None or root == repository_root):
        raise ValueError("A24 formal test controls require an isolated test root")
    if (
        not smoke
        and root == repository_root
        and _formal_capability is not _FORMAL_EXECUTION_CAPABILITY
    ):
        raise PermissionError(
            "A24 internal formal engine requires a separate authorization capability"
        )
    formal_authorized = bool(
        not smoke
        and root == repository_root
        and _formal_capability is _FORMAL_EXECUTION_CAPABILITY
    )
    formal_lock = root / FORMAL_LOCK
    if smoke and formal_lock.exists():
        raise RuntimeError("A24 smoke refuses to run while a formal lock exists")
    if not smoke and formal_lock.exists():
        raise FileExistsError("A24 formal attempt lock already exists")
    output = output.resolve()
    if output.parent != (root / "artifacts").resolve():
        raise ValueError("A24 output must be a direct child of artifacts")
    if output.exists():
        raise FileExistsError(output)
    run_schedule = (
        _formal_test_schedule
        if _formal_test_schedule is not None
        else schedule(smoke)
    )
    if not smoke and _formal_test_schedule is None and run_schedule != A24Schedule(
        mode="adaptive-same-bank-development",
        seeds=FORMAL_SEEDS,
        folds=FORMAL_FOLDS,
        updates=FORMAL_UPDATES,
        episodes_per_update=FORMAL_EPISODES_PER_UPDATE,
        calibration_episodes_per_fold=600,
        outer_episodes_per_fold=600,
        bootstrap_iterations=20_000,
        threads=A24_FORMAL_THREADS,
    ):
        raise RuntimeError("A24 formal execution schedule changed")
    torch.set_num_threads(run_schedule.threads)
    torch.use_deterministic_algorithms(True, warn_only=False)
    context = verify_inputs(root, smoke=smoke, threads=run_schedule.threads)
    partitions = {
        fold: partition_ids(context["assignments"], fold)
        for fold in run_schedule.folds
    }
    formal_products = expected_products(schedule(False), gate_open=True)
    smoke_products_if_open = expected_products(run_schedule, gate_open=True)
    training_contract = {
        "schema_version": TRAINING_CONTRACT_VERSION,
        "protocol_revision": "e4c7d9aed9b5cee3e5cbd21cfe2c6a7da2f2e763",
        "runner_version": RUNNER_VERSION,
        "policy_version": POLICY_VERSION,
        "pq1_primitives_version": PQ1_PRIMITIVES_VERSION,
        "mechanism": MECHANISM,
        "shield_enabled": False,
        "schedule": asdict(run_schedule),
        "formal_schedule": asdict(schedule(False)),
        "ppo": asdict(_config(run_schedule)),
        "partitions": [partitions[fold] for fold in run_schedule.folds],
        "formal_products_if_gate_open": formal_products,
        "current_schedule_products_if_gate_open": smoke_products_if_open,
        "comparator_ledger_sha256": context["a22_comparator_ledger"][
            "ledger_sha256"
        ],
        "training_contract_is_canonical_protocol": True,
        "formal_execution_authorized": formal_authorized,
        "non_evidentiary_smoke": smoke,
    }
    protocol_sha = _digest(training_contract)
    formal_lock_projection = (
        None
        if smoke
        else {
            "schema_version": LOCK_VERSION,
            "attempt": 1,
            "path": str(formal_lock.resolve()),
            "output": str(output.resolve()),
            "source_revision": context["a24_source"]["revision"],
            "protocol_sha256": protocol_sha,
            "source_set_sha256": context["a24_source"]["source_set_sha256"],
        }
    )
    contract = {
        "schema_version": RUN_CONTRACT_VERSION,
        "mode": run_schedule.mode,
        "source": context["a24_source"],
        "runtime": context["a24_runtime"],
        "protocol_sha256": protocol_sha,
        "preflight_signature": context["a24_preflight_signature"],
        "pq1": context["pq1"],
        "a23_failure": context["a23_failure"],
        "bindings": context["bindings"],
        "a22_comparator_ledger": context["a22_comparator_ledger"],
        "training_contract_sha256": protocol_sha,
        "formal_lock": formal_lock_projection,
    }
    contract_sha = _digest(contract)
    formal_descriptor = (
        None
        if smoke
        else lock_descriptor(
            output=output,
            source_revision=context["a24_source"]["revision"],
            run_contract_sha256=contract_sha,
            protocol_sha256=protocol_sha,
            source_set_sha256=context["a24_source"]["source_set_sha256"],
        )
    )
    if not smoke:
        descriptor_projection = {
            "schema_version": formal_descriptor["schema_version"],
            "attempt": formal_descriptor["attempt"],
            "path": str(formal_lock.resolve()),
            "output": formal_descriptor["output"],
            "source_revision": formal_descriptor["source_revision"],
            "protocol_sha256": formal_descriptor["protocol_sha256"],
            "source_set_sha256": formal_descriptor["source_set_sha256"],
        }
        if not same_typed_json(descriptor_projection, formal_lock_projection):
            raise RuntimeError("A24 planned formal lock projection changed")
    running = {
        "schema_version": RUNNER_VERSION,
        "mode": run_schedule.mode,
        "formal_lock_acquired": not smoke,
        "calibration_started": False,
        "outer_evaluation_started": False,
    }
    formal_lock_binding = None
    lock_acquired = False
    signal_guard = None
    previous_signal_mask = None
    signals_unblocked = smoke
    if not smoke:
        previous_signal_mask = signal.pthread_sigmask(
            signal.SIG_BLOCK,
            {signal.SIGTERM, signal.SIGINT},
        )
        try:
            formal_lock_binding = acquire_formal_lock(
                formal_lock,
                formal_descriptor,
            )
            lock_acquired = True
            verify_formal_lock(formal_lock, formal_lock_binding)
            revalidate_inputs(
                root,
                smoke=False,
                threads=run_schedule.threads,
                expected_signature=context["a24_preflight_signature"],
            )
        except FormalLockCreatedError as exc:
            lock_acquired = formal_lock.exists()
            try:
                output.mkdir(mode=0o700)
                os.chmod(output, 0o700)
                fsync_directory(output.parent)
                atomic_write_json(output / "RUNNING.json", running)
                recovered_binding = formal_lock_binding_payload(
                    formal_lock,
                    formal_descriptor,
                )
                invalidate_postlock_failure(
                    output,
                    lock=formal_lock,
                    expected_binding=recovered_binding,
                    reason="formal_lock_publication_failure",
                    error=exc,
                )
            finally:
                signal.pthread_sigmask(
                    signal.SIG_SETMASK,
                    previous_signal_mask,
                )
            raise
        except BaseException as exc:
            try:
                if lock_acquired:
                    output.mkdir(mode=0o700)
                    os.chmod(output, 0o700)
                    fsync_directory(output.parent)
                    atomic_write_json(output / "RUNNING.json", running)
                    recovered_binding = formal_lock_binding_payload(
                        formal_lock,
                        formal_descriptor,
                    )
                    invalidate_postlock_failure(
                        output,
                        lock=formal_lock,
                        expected_binding=recovered_binding,
                        reason="postlock_preflight_failure",
                        error=exc,
                    )
            finally:
                signal.pthread_sigmask(
                    signal.SIG_SETMASK,
                    previous_signal_mask,
                )
            raise

    def revalidate_execution() -> None:
        revalidate_inputs(
            root,
            smoke=smoke,
            threads=run_schedule.threads,
            expected_signature=context["a24_preflight_signature"],
        )
        if not smoke:
            verify_formal_lock(formal_lock, formal_lock_binding)

    terminal_validated = False
    try:
        output.mkdir(mode=0o700)
        os.chmod(output, 0o700)
        fsync_directory(output.parent)
        if not smoke:
            signal_guard = supervised_postlock_signals()
            signal_guard.__enter__()
        atomic_write_json(output / "RUNNING.json", running)
        if not smoke:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_signal_mask)
            signals_unblocked = True
            if fault_request == "supervised-sigterm-after-running":
                os.kill(os.getpid(), signal.SIGTERM)
        atomic_write_json(output / "training-contract.json", training_contract)
        atomic_write_json(output / "run-contract.json", contract)
        comparator_index = {
            (int(row["outer_fold"]), int(row["training_seed"])): row
            for row in context["a22_comparator_ledger"]["entries"]
        }
        fits = []
        for fold in run_schedule.folds:
            partition = partitions[fold]
            train = [
                context["episode_index"][episode_id]
                for episode_id in partition["inner_train_ids"]
            ]
            train_a8 = [
                context["a8_index"][episode_id]
                for episode_id in partition["inner_train_ids"]
            ]
            for seed in run_schedule.seeds:
                fit_dir = _fit_path(output, fold, seed)
                _make_private_tree(fit_dir, root=output)
                expected_samples = context["a22_sample_sequences"][(fold, seed)][
                    : run_schedule.updates * run_schedule.episodes_per_update
                ]
                fit = fit_no_shield(
                    train,
                    a8_rows=train_a8,
                    seed=seed,
                    outer_fold=fold,
                    run_schedule=run_schedule,
                    output=fit_dir,
                    run_contract_sha256=contract_sha,
                    partition=partition,
                    comparator_entry=comparator_index[(fold, seed)],
                    expected_sample_ids=expected_samples,
                )
                validate_fit_artifacts(
                    fit_dir,
                    expected_fit=fit,
                    run_schedule=run_schedule,
                    run_contract_sha256=contract_sha,
                    expected_sample_ids=expected_samples,
                    expected_thresholds=asdict(
                        thresholds_from_inner_train(train_a8)
                    ),
                )
                fits.append(fit)
                revalidate_execution()
        products_negative = expected_products(run_schedule, gate_open=False)
        all_fits = {
            "schema_version": ALL_FITS_VERSION,
            "fits": fits,
            "expected_fits": products_negative["fits"],
            "expected_optimizer_updates": products_negative["optimizer_updates"],
            "expected_training_episode_draws": products_negative[
                "training_episode_draws"
            ],
            "all_reached_final_update": True,
            "exact_fit_key_product_verified": True,
            "pq1_provenance_and_checkpoint_chain_verified": True,
            "calibration_started_at_receipt": False,
            "outer_evaluation_started_at_receipt": False,
        }
        atomic_write_json(output / "all-fits-complete.json", all_fits)
        running["calibration_started"] = True
        atomic_replace_json(output / "RUNNING.json", running)
        fit_index = {
            (int(row["outer_fold"]), int(row["training_seed"])): row
            for row in fits
        }
        calibration_rows = []
        for fold in run_schedule.folds:
            ids = partitions[fold]["inner_calibration_ids"][
                : run_schedule.calibration_episodes_per_fold
            ]
            episodes = [context["episode_index"][episode_id] for episode_id in ids]
            for seed in run_schedule.seeds:
                fit = fit_index[(fold, seed)]
                model, _, _ = _load_a24_checkpoint(
                    _fit_path(output, fold, seed) / "final.pt",
                    fit=fit,
                    run_schedule=run_schedule,
                    run_contract_sha256=contract_sha,
                )
                calibration_rows.extend(
                    _evaluate(
                        model,
                        episodes,
                        system="A24-cr-ppo-no-shield",
                        mechanism=MECHANISM,
                        phase="inner-calibration",
                        training_seed=seed,
                        design_outer_fold=fold,
                        assignment_index=context["assignment_index"],
                        a8_index=context["a8_index"],
                        bank_sha256=context["bindings"]["train_bank"],
                        resource_sha256=context["bindings"]["resource"],
                        environment_sha256=context["bindings"]["environment"],
                        checkpoint_sha256=fit["checkpoint_sha256"],
                        run_contract_sha256=contract_sha,
                        outer_gate_sha256=None,
                    )
                )
        calibration_a8 = [
            context["a8_index"][episode_id]
            for fold in run_schedule.folds
            for episode_id in partitions[fold]["inner_calibration_ids"][
                : run_schedule.calibration_episodes_per_fold
            ]
        ]
        gate_decision = evaluate_calibration_gate(
            calibration_a8,
            calibration_rows,
            folds=run_schedule.folds,
            seeds=run_schedule.seeds,
            episodes_per_fold=run_schedule.calibration_episodes_per_fold,
            smoke=smoke,
            expected_a8_run_contract_sha256=EXPECTED_R2["run_digest"],
            expected_run_contract_sha256=contract_sha,
            expected_checkpoint_sha256={
                (fold, seed): fit_index[(fold, seed)]["checkpoint_sha256"]
                for fold in run_schedule.folds
                for seed in run_schedule.seeds
            },
        )
        comparator_calibration = _scheduled_a22_calibration(
            context,
            run_schedule=run_schedule,
            partitions=partitions,
        )
        comparator_diagnostic = calibration_comparator_diagnostic(
            calibration_a8,
            calibration_rows,
            comparator_calibration,
            folds=run_schedule.folds,
            seeds=run_schedule.seeds,
            episodes_per_fold=run_schedule.calibration_episodes_per_fold,
            expected_a8_run_contract_sha256=EXPECTED_R2["run_digest"],
            expected_a24_run_contract_sha256=contract_sha,
            expected_a22_run_contract_sha256=EXPECTED_A22_RUN_DIGEST,
            expected_a22_source_artifacts=EXPECTED_A22,
            expected_a24_checkpoint_sha256={
                (fold, seed): fit_index[(fold, seed)]["checkpoint_sha256"]
                for fold in run_schedule.folds
                for seed in run_schedule.seeds
            },
            comparator_ledger=context["a22_comparator_ledger"],
        )
        atomic_write_jsonl(
            output / "calibration-decisions.jsonl", calibration_rows
        )
        calibration_gate = {
            "schema_version": CALIBRATION_GATE_VERSION,
            "run_contract_sha256": contract_sha,
            "comparator_ledger_sha256": context["a22_comparator_ledger"][
                "ledger_sha256"
            ],
            "calibration_sha256": sha256_file(
                output / "calibration-decisions.jsonl"
            ),
            "a24_calibration_rows": len(calibration_rows),
            "a22_lagrangian_descriptive_rows": len(comparator_calibration),
            "a22_lagrangian_enters_gate": False,
            "a22_lagrangian_calibration_diagnostic": comparator_diagnostic,
            "decision": gate_decision,
            "smoke_forced_outer": _force_smoke_outer,
        }
        _fault("before-calibration-gate", fault_request)
        atomic_write_json(output / "calibration-gate.json", calibration_gate)
        _fault("after-calibration-gate", fault_request)
        revalidate_execution()
        gate_open = (
            bool(_force_smoke_outer and gate_decision["raw_conjunction"])
            if smoke
            else bool(gate_decision["outer_gate_permitted"])
        )
        outer_gate = None
        outer_gate_sha = None
        a24_outer: list[dict[str, Any]] = []
        a22_outer: list[dict[str, Any]] = []
        statistics = None
        if gate_open:
            running["outer_evaluation_started"] = True
            atomic_replace_json(output / "RUNNING.json", running)
            outer_gate = {
                "schema_version": "multitown-a24-outer-gate-v1",
                "run_contract_sha256": contract_sha,
                "calibration_gate_sha256": sha256_file(
                    output / "calibration-gate.json"
                ),
                "comparator_ledger_sha256": context["a22_comparator_ledger"][
                    "ledger_sha256"
                ],
                "a24_checkpoint_sha256": {
                    f"{fold}:{seed}": fit_index[(fold, seed)]["checkpoint_sha256"]
                    for fold in run_schedule.folds
                    for seed in run_schedule.seeds
                },
                "a22_lagrangian_checkpoint_sha256": {
                    f"{row['outer_fold']}:{row['training_seed']}": row[
                        "raw_sha256"
                    ]["final.pt"]
                    for row in context["a22_comparator_ledger"]["entries"]
                    if int(row["outer_fold"]) in run_schedule.folds
                    and int(row["training_seed"]) in run_schedule.seeds
                },
                "formal_gate": not smoke,
                "non_evidentiary_forced_smoke_path": smoke,
            }
            _fault("before-outer-gate", fault_request)
            atomic_write_json(output / "OUTER_GATE_OPEN.json", outer_gate)
            _fault("after-outer-gate", fault_request)
            revalidate_execution()
            outer_gate_sha = sha256_file(output / "OUTER_GATE_OPEN.json")
            a22_fit_index = _a22_fit_index(context)
            a22_mechanism = next(
                item for item in A22_MECHANISMS if item.name == "lagrangian"
            )
            for fold in run_schedule.folds:
                ids = partitions[fold]["outer_ids"][
                    : run_schedule.outer_episodes_per_fold
                ]
                episodes = [
                    context["episode_index"][episode_id] for episode_id in ids
                ]
                for seed in run_schedule.seeds:
                    fit = fit_index[(fold, seed)]
                    model, _, _ = _load_a24_checkpoint(
                        _fit_path(output, fold, seed) / "final.pt",
                        fit=fit,
                        run_schedule=run_schedule,
                        run_contract_sha256=contract_sha,
                    )
                    a24_outer.extend(
                        _evaluate(
                            model,
                            episodes,
                            system="A24-cr-ppo-no-shield",
                            mechanism=MECHANISM,
                            phase="a24-outer",
                            training_seed=seed,
                            design_outer_fold=fold,
                            assignment_index=context["assignment_index"],
                            a8_index=context["a8_index"],
                            bank_sha256=context["bindings"]["train_bank"],
                            resource_sha256=context["bindings"]["resource"],
                            environment_sha256=context["bindings"]["environment"],
                            checkpoint_sha256=fit["checkpoint_sha256"],
                            run_contract_sha256=contract_sha,
                            outer_gate_sha256=outer_gate_sha,
                        )
                    )
                    a22_fit = a22_fit_index[(fold, seed)]
                    a22_checkpoint = (
                        Path(context["a22_root"])
                        / "fits"
                        / f"outer-fold-{fold}"
                        / f"seed-{seed}"
                        / "lagrangian"
                        / "final.pt"
                    )
                    a22_model, _ = _validate_a22_checkpoint(
                        a22_checkpoint,
                        fit=a22_fit,
                        mechanism=a22_mechanism,
                        outer_fold=fold,
                        training_seed=seed,
                        run_schedule=a22_schedule(False),
                        run_contract_sha256=EXPECTED_A22_RUN_DIGEST,
                    )
                    a22_outer.extend(
                        _evaluate(
                            a22_model,
                            episodes,
                            system="A22-lagrangian-A24-comparator",
                            mechanism="lagrangian",
                            phase="a24-comparator-outer",
                            training_seed=seed,
                            design_outer_fold=fold,
                            assignment_index=context["assignment_index"],
                            a8_index=context["a8_index"],
                            bank_sha256=context["bindings"]["train_bank"],
                            resource_sha256=context["bindings"]["resource"],
                            environment_sha256=context["bindings"]["environment"],
                            checkpoint_sha256=a22_fit["checkpoint_sha256"],
                            run_contract_sha256=contract_sha,
                            outer_gate_sha256=outer_gate_sha,
                        )
                    )
                revalidate_execution()
            _fault("before-outer-rows", fault_request)
            atomic_write_jsonl(output / "a24-outer-decisions.jsonl", a24_outer)
            atomic_write_jsonl(
                output / "a22-lagrangian-outer-decisions.jsonl", a22_outer
            )
            _fault("after-outer-rows", fault_request)
            selected_a8 = [
                context["a8_index"][episode_id]
                for fold in run_schedule.folds
                for episode_id in partitions[fold]["outer_ids"][
                    : run_schedule.outer_episodes_per_fold
                ]
            ]
            _fault("before-statistics", fault_request)
            statistics = result_statistics(
                selected_a8,
                a24_outer,
                a22_outer,
                _statistics_schedule(run_schedule),
                gate_evaluable=not smoke,
                expected_a8_run_contract_sha256=EXPECTED_R2["run_digest"],
                expected_run_contract_sha256=contract_sha,
                expected_outer_gate_sha256=outer_gate_sha,
                expected_a24_checkpoint_sha256={
                    (fold, seed): fit_index[(fold, seed)]["checkpoint_sha256"]
                    for fold in run_schedule.folds
                    for seed in run_schedule.seeds
                },
                expected_a22_checkpoint_sha256={
                    (fold, seed): a22_fit_index[(fold, seed)][
                        "checkpoint_sha256"
                    ]
                    for fold in run_schedule.folds
                    for seed in run_schedule.seeds
                },
            )
            _fault("after-statistics", fault_request)
        if not same_typed_json(
            strict_read_jsonl(output / "calibration-decisions.jsonl"),
            calibration_rows,
        ) or not same_typed_json(
            strict_read_json(output / "calibration-gate.json"),
            calibration_gate,
        ):
            raise RuntimeError("A24 persisted calibration publication changed")
        if gate_open and (
            not same_typed_json(
                strict_read_json(output / "OUTER_GATE_OPEN.json"),
                outer_gate,
            )
            or not same_typed_json(
                strict_read_jsonl(output / "a24-outer-decisions.jsonl"),
                a24_outer,
            )
            or not same_typed_json(
                strict_read_jsonl(
                    output / "a22-lagrangian-outer-decisions.jsonl"
                ),
                a22_outer,
            )
        ):
            raise RuntimeError("A24 persisted conditional outer publication changed")
        if (
            (output / "training-contract.json").read_bytes()
            != canonical_json_bytes(training_contract)
            or (output / "run-contract.json").read_bytes()
            != canonical_json_bytes(contract)
            or (output / "all-fits-complete.json").read_bytes()
            != canonical_json_bytes(all_fits)
        ):
            raise RuntimeError("A24 persisted root training publication changed")
        for fit in fits:
            fold = int(fit["outer_fold"])
            seed = int(fit["training_seed"])
            train_a8 = [
                context["a8_index"][episode_id]
                for episode_id in partitions[fold]["inner_train_ids"]
            ]
            validate_fit_artifacts(
                _fit_path(output, fold, seed),
                expected_fit=fit,
                run_schedule=run_schedule,
                run_contract_sha256=contract_sha,
                expected_sample_ids=context["a22_sample_sequences"][(fold, seed)][
                    : run_schedule.updates * run_schedule.episodes_per_update
                ],
                expected_thresholds=asdict(
                    thresholds_from_inner_train(train_a8)
                ),
            )
        actual_products = expected_products(run_schedule, gate_open=gate_open)
        terminal_state = (
            "NON_EVIDENTIARY_SMOKE"
            if smoke
            else (
                "VALID_GATE_OPEN_SUCCESS"
                if gate_open
                else "VALID_CALIBRATION_NEGATIVE"
            )
        )
        result = {
            "schema_version": RESULT_VERSION,
            "mode": run_schedule.mode,
            "non_evidentiary_smoke": smoke,
            "formal_execution_authorized": formal_authorized,
            "formal_lock_acquired": not smoke,
            "source_revision": context["a24_source"]["revision"],
            "run_contract_sha256": contract_sha,
            "protocol_sha256": protocol_sha,
            "comparator_ledger_sha256": context["a22_comparator_ledger"][
                "ledger_sha256"
            ],
            "calibration_raw_conjunction": gate_decision["raw_conjunction"],
            "formal_calibration_gate_evaluable": not smoke,
            "formal_outer_gate_open": bool(not smoke and gate_open),
            "smoke_outer_path_exercised": bool(smoke and gate_open),
            "products": actual_products,
            "statistics": statistics,
            "claim_boundary": build_claim_boundary(
                terminal_state=terminal_state,
                smoke=smoke,
                outer_performance_evaluable=bool(not smoke and gate_open),
            ),
            "validation": {
                "pq1_functions_executed_every_update": True,
                "exact_fit_log_checkpoint_chain": True,
                "a22_calibration_enters_gate": False,
                "a8_rows_logically_paired_not_replicated": True,
                "formal_lock_untouched": smoke,
                "formal_lock_verified": not smoke,
            },
        }
        revalidate_execution()
        _fault("before-result", fault_request)
        atomic_write_json(output / "result.json", result)
        _fault("after-result", fault_request)
        if not same_typed_json(strict_read_json(output / "result.json"), result):
            raise RuntimeError("A24 persisted result publication changed")
        revalidate_execution()
        (output / "RUNNING.json").unlink()
        fsync_directory(output)
        manifest = manifest_payload(
            output,
            source_revision=context["a24_source"]["revision"],
            run_contract_sha256=contract_sha,
            folds=run_schedule.folds,
            seeds=run_schedule.seeds,
            gate_open=gate_open,
            smoke=smoke,
        )
        _fault("before-manifest", fault_request)
        atomic_write_json(output / "artifact-manifest.json", manifest)
        _fault("after-manifest", fault_request)
        for _ in range(2):
            revalidate_execution()
            validate_manifest(
                output,
                folds=run_schedule.folds,
                seeds=run_schedule.seeds,
                gate_open=gate_open,
                expected_source_revision=context["a24_source"]["revision"],
                expected_run_contract_sha256=contract_sha,
                expected_lock_descriptor=(
                    None if smoke else formal_descriptor
                ),
                smoke=smoke,
            )
        if smoke and formal_lock.exists():
            raise RuntimeError("A24 smoke unexpectedly created the formal lock")
        if not smoke:
            trusted_snapshot = raw_snapshot(
                output,
                formal_lock,
                folds=run_schedule.folds,
                seeds=run_schedule.seeds,
                gate_open=gate_open,
            )
            revalidate_execution()
            validate_manifest(
                output,
                folds=run_schedule.folds,
                seeds=run_schedule.seeds,
                gate_open=gate_open,
                expected_source_revision=context["a24_source"]["revision"],
                expected_run_contract_sha256=contract_sha,
                expected_lock_descriptor=formal_descriptor,
                smoke=False,
            )
            verify_raw_snapshot(
                trusted_snapshot,
                output,
                formal_lock,
                folds=run_schedule.folds,
                seeds=run_schedule.seeds,
                gate_open=gate_open,
            )
            terminal_validated = True
        if smoke:
            print(
                json.dumps(
                    {
                        "output": str(output),
                        "mode": run_schedule.mode,
                        **actual_products,
                        "calibration_raw_conjunction": gate_decision[
                            "raw_conjunction"
                        ],
                        "smoke_outer_path_exercised": bool(smoke and gate_open),
                        "formal_lock_created": False,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        return 0
    except BaseException as exc:
        if not smoke:
            if terminal_validated:
                raise
            if not lock_acquired:
                raise
            if not output.exists():
                output.mkdir(mode=0o700)
                os.chmod(output, 0o700)
                fsync_directory(output.parent)
            if not (output / "RUNNING.json").exists():
                atomic_write_json(output / "RUNNING.json", running)
            reason = (
                "supervised_sigterm"
                if isinstance(exc, FormalTerminationRequested)
                and exc.signum == signal.SIGTERM
                else "postlock_execution_failure"
            )
            invalidate_postlock_failure(
                output,
                lock=formal_lock,
                expected_binding=formal_lock_binding,
                reason=reason,
                error=exc,
            )
            raise
        isolated: dict[str, str] = {}
        if output.exists():
            isolated = isolate_success_shaped(output)
            if any((output / name).exists() for name in SUCCESS_TO_INVALID):
                raise RuntimeError(
                    "A24 smoke failure retains a success-shaped publication"
                ) from exc
            if not (output / "RUNNING.json").exists():
                atomic_write_json(output / "RUNNING.json", running)
        if output.exists() and not (output / "SMOKE_FAILED.json").exists():
            try:
                atomic_write_json(
                    output / "SMOKE_FAILED.json",
                    {
                        "schema_version": "multitown-a24-smoke-failure-v2",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "formal_lock_acquired": False,
                        "performance_evaluable": False,
                        "success_shaped_publications_quarantined": isolated,
                    },
                )
            except Exception:
                pass
        if formal_lock.exists():
            raise RuntimeError("A24 smoke failure coincided with a formal lock") from exc
        raise
    finally:
        if signal_guard is not None:
            signal_guard.__exit__(None, None, None)
        if not smoke and not signals_unblocked:
            signal.pthread_sigmask(
                signal.SIG_SETMASK,
                previous_signal_mask,
            )


def run(
    output: Path,
    *,
    smoke: bool,
    _force_smoke_outer: bool = False,
    _inject_failure: str | None = None,
) -> int:
    """Expose smoke only; formal execution still requires separate authorization."""

    if not smoke:
        raise PermissionError(
            "A24 formal execution is not authorized; implementation CLI is smoke-only"
        )
    return _execute(
        output,
        smoke=True,
        _force_smoke_outer=_force_smoke_outer,
        _inject_failure=_inject_failure,
    )


def prepare_formal(
    output: Path,
    *,
    _root: Path | None = None,
) -> dict[str, Any]:
    """Run a read-only clean-source preflight without authorizing the attempt."""

    root = (
        Path(__file__).resolve().parents[1]
        if _root is None
        else _root.resolve()
    )
    output = output.resolve()
    formal_lock = root / FORMAL_LOCK
    if output.parent != (root / "artifacts").resolve():
        raise ValueError("A24 formal output must be a direct child of artifacts")
    if output.exists():
        raise FileExistsError("A24 planned formal output already exists")
    if formal_lock.exists():
        raise FileExistsError("A24 formal attempt lock already exists")
    pending_locks = sorted(
        path.name
        for path in formal_lock.parent.glob(f".{formal_lock.name}.pending-*")
    )
    if pending_locks:
        raise RuntimeError(
            "A24 readiness found stale pending lock publications: "
            + ",".join(pending_locks)
        )
    run_schedule = schedule(False)
    expected_schedule = A24Schedule(
        mode="adaptive-same-bank-development",
        seeds=FORMAL_SEEDS,
        folds=FORMAL_FOLDS,
        updates=FORMAL_UPDATES,
        episodes_per_update=FORMAL_EPISODES_PER_UPDATE,
        calibration_episodes_per_fold=600,
        outer_episodes_per_fold=600,
        bootstrap_iterations=20_000,
        threads=A24_FORMAL_THREADS,
    )
    if run_schedule != expected_schedule:
        raise RuntimeError("A24 formal readiness schedule changed")
    torch.set_num_threads(run_schedule.threads)
    torch.use_deterministic_algorithms(True, warn_only=False)
    context = verify_inputs(root, smoke=False, threads=run_schedule.threads)
    if formal_lock.exists() or output.exists():
        raise RuntimeError("A24 read-only readiness preflight changed external state")
    return {
        "schema_version": "multitown-a24-formal-readiness-v1",
        "planned_output": str(output),
        "planned_lock": str(formal_lock.resolve()),
        "source_revision": context["a24_source"]["revision"],
        "source_set_sha256": context["a24_source"]["source_set_sha256"],
        "preflight_signature": context["a24_preflight_signature"],
        "schedule": asdict(run_schedule),
        "products_if_calibration_negative": expected_products(
            run_schedule,
            gate_open=False,
        ),
        "products_if_gate_open": expected_products(
            run_schedule,
            gate_open=True,
        ),
        "clean_source_verified": context["a24_source"]["dirty"] is False,
        "pinned_inputs_verified": True,
        "formal_lock_absent": True,
        "formal_output_absent": True,
        "formal_execution_authorized": False,
        "formal_lock_created": False,
        "formal_output_created": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--readiness-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.readiness_only:
        print(
            json.dumps(
                prepare_formal(arguments.output_dir),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    raise SystemExit(run(arguments.output_dir, smoke=True))


if __name__ == "__main__":
    main()
