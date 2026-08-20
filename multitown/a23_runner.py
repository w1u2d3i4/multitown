"""Run the frozen A23 CRPO-inspired CR-PPO adaptive development study."""

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
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .a22_constrained_ppo import (
    SafetyThresholds, constrained_rollout, deterministic_action,
    thresholds_from_inner_train,
)
from .a22_report import validate_raw as validate_a22_raw
from .a22_runner import (
    _assert_exact_keys, _behavior_summary, _expected_sample_sequence,
    _summary, partition_ids,
)
from .a23_cr_ppo import (
    A23_PRIMITIVES_VERSION, MECHANISMS, CRPPOMechanism, cr_ppo_update,
    initial_optimizer_sha256, mode_sequence_sha256, model_parameter_sha256,
    select_actor_mode,
)
from .a23_statistics import A23StatisticsSchedule, result_statistics
from .a9_oof_protocol import (
    DEFAULT_FOLDS, FROZEN_TRAIN_PATH, assign_stratified_group_folds,
    fold_manifest_sha256, load_frozen_train_bank, resource_contract_sha256,
    shared_resource_contract,
)
from .a9_ppo_oof import (
    BOOTSTRAP_SEED, FORMAL_EPISODES_PER_UPDATE, FORMAL_SEEDS, FORMAL_THREADS,
    FORMAL_UPDATES, POLICY_VERSION as A9_POLICY_VERSION, _digest,
    _evaluate_episode, _formal_ppo_config, _set_seed, _sha256,
)
from .a9_safety_development import (
    FROZEN_R2_PATH, _validate_shared_stack_bindings, _verify_r2_artifacts,
)
from .long_horizon_env import (
    ACTION_COUNT, LongHorizonEpisode, MultiTownLongHorizonEnv,
)
from .ppo_controller import ActorCritic, PPOConfig, _save_checkpoint, load_checkpoint


RUNNER_VERSION = "multitown-a23-cr-ppo-adaptive-runner-v1"
POLICY_VERSION = "multitown-a23-cr-ppo-policy-v1"
RESULT_VERSION = "multitown-a23-adaptive-development-result-v1"
UPDATE_LOG_VERSION = "multitown-a23-cr-ppo-update-log-v1"
FORMAL_ATTEMPT_LOCK = Path("artifacts/a23-cr-ppo-attempt-v1.lock")
A22_RAW_PATH = Path("artifacts/a22-adaptive-formal-20260814")

EXPECTED_A22 = {
    "artifact-manifest.json": "fd1898483012c4595b8536b2f7af146a03f347be083543f0b21a39c706ae0015",
    "result.json": "d8193ee28c2c81c87858408870c8af5d91b2f47ae49ca683d05512439154a719",
    "all-selections-frozen.json": "e39aff85db69ad446ad7b60f3cb745d5ed02a7461368d72ea86022cb61da9ead",
    "all-fits-complete.json": "dfd0baf000c0831aa3040b2c40ba69f6fb47e8ff3f96b4862f76fcafac596ae7",
    "calibration-decisions.jsonl": "60988e4ad8920defe604965d83163bf1c22263c518f6308c27951616cfe656d4",
    "outer-decisions.jsonl": "b49b6e9e696b584681ee8db208207c470707c086e6dd417586141220f19e9b23",
    "run-contract.json": "15966f1d40a162cf512b2c4476fc37610b5997ba1c267ede7729ba8dc8a9c601",
}
EXPECTED_A22_RUN_DIGEST = "716f8e3c2a4e968c5008cbb369ba92804d94b12c496b3376daa3a2d9f3648063"
EXPECTED_R2 = {
    "manifest": "62e0b5dc34219bf1816509deaf036f824ece96808eaa58a50a559dfa53497e3a",
    "a8": "9bf3ed9148d58c1c49f85807f063c19946de8e12d982d6d04b92865153c0ed40",
    "run_file": "37d9a17319e0ad768c63afae05dbaf419faf7e509fb9679048a8fa0c473cfbea",
    "run_digest": "7694d2816abd31753cfa68eec5020ab6da37eb18e4f77beed5714c544a0e51e1",
}
EXPECTED_BINDINGS = {
    "train_bank": "ee0c3d9677ea44694373405c2337b3e61cc5562f302976cab8af9fb143b7b777",
    "fold_manifest": "f73f51c0cb370db36a25393dca088b04add62665e07b3370795857ed68fd001e",
    "resource": "eeed603717262f96f40b95d5e602304698f9cb57d677d93f32db3d8ea52cece1",
    "environment": "e769cf989006ddcc2b51b2f5c0c1d0ee0b0dc638f7b0e09b5ae97f9d9ff82755",
}
PINNED_REUSED_SOURCES = {
    "multitown/a22_constrained_ppo.py": "d6920f55d0e97a1b0c6bdd615adf4ddc5df7a1cb3b67abde5dd8ae4e0f6d15e9",
    "multitown/a22_runner.py": "4c6455ce1cf718c6c093c27af9e86d7aab2e5c2ea70e598cb57d86da1d448127",
    "multitown/a9_oof_protocol.py": "174917933278d690cd27ee8a56fdd41008f6ebcfbf88156f0dda7da68b63a80f",
    "multitown/a9_ppo_oof.py": "892ebedb2e4d09282519b02b98299d6502aaf1b4a1b8306864b25e5b4a2f29e5",
    "multitown/a9_long_horizon_env.py": "a28a5f382a61389135d39babfef85e2e56a67371abc3164bdb58b4e0456d8718",
    "multitown/long_horizon_env.py": "e769cf989006ddcc2b51b2f5c0c1d0ee0b0dc638f7b0e09b5ae97f9d9ff82755",
    "multitown/ppo_controller.py": "c5862bbf594382bcba799d332ec2d4c427d1355d0f8924dc27bab16cf7868b92",
}
A22_COMPARATORS = ("lagrangian", "lagrangian-plus-shield", "shield")

UPDATE_LOG_STRING_FIELDS = {
    "schema_version", "primitives_version", "mechanism",
    "selected_actor_mode", "sampled_episode_ids_sha256",
    "normalized_advantage_sha256",
}
UPDATE_LOG_INTEGER_FIELDS = {
    "outer_fold", "training_seed", "update", "episodes_per_update",
    "rollout_episodes", "environment_steps", "optimizer_minibatches",
    "rollout_unsafe_events", "rollout_wrong_executions",
    "rollout_incidents", "shield_mask_active_decisions",
    "reward_mode_count", "unsafe_mode_count", "wrong_mode_count",
}
UPDATE_LOG_BOOLEAN_FIELDS = {
    "selected_advantage_constant", "unsafe_eligible", "wrong_eligible",
    "unsafe_tie_break_used",
}
UPDATE_LOG_FLOAT_FIELDS = {
    "rollout_episode_success_rate", "rollout_tokens_per_episode",
    "unsafe_cost", "wrong_cost", "unsafe_threshold", "wrong_threshold",
    "unsafe_violation", "wrong_violation",
    "unsafe_normalized_violation", "wrong_normalized_violation",
    "mean_incidents", "policy_loss", "value_loss", "entropy", "approx_kl",
    "clip_fraction",
    *{
        f"{prefix}_{suffix}"
        for prefix in (
            "reward_advantage_raw", "unsafe_cost_to_go_raw",
            "wrong_cost_to_go_raw", "selected_actor_advantage_raw",
        )
        for suffix in ("mean", "std", "max_abs")
    },
}

FIT_STRING_FIELDS = {
    "schema_version", "primitives_version", "mechanism",
    "sample_sequence_sha256", "initial_model_sha256",
    "initial_optimizer_sha256", "mode_sequence_sha256",
    "training_log_sha256", "checkpoint_sha256", "inner_train_ids_sha256",
    "calibration_ids_sha256", "outer_ids_sha256", "run_contract_sha256",
    "selected_checkpoint",
}
FIT_INTEGER_FIELDS = {
    "outer_fold", "training_seed", "final_update", "training_episode_draws",
    "sampled_unique_episodes", "environment_steps", "optimizer_minibatches",
    "calibration_evaluations_during_training",
    "outer_evaluations_during_training",
}


def _strict_json_object(
    value: Mapping[str, Any], *, label: str,
    strings: set[str] = frozenset(), integers: set[str] = frozenset(),
    floats: set[str] = frozenset(), booleans: set[str] = frozenset(),
    lists: set[str] = frozenset(), objects: set[str] = frozenset(),
) -> None:
    expected = strings | integers | floats | booleans | lists | objects
    if type(value) is not dict or set(value) != expected:
        raise RuntimeError(f"{label} exact key schema mismatch")
    checks = (
        (strings, str), (integers, int), (floats, float),
        (booleans, bool), (lists, list), (objects, dict),
    )
    for fields, expected_type in checks:
        if any(type(value[field]) is not expected_type for field in fields):
            raise RuntimeError(f"{label} strict JSON type mismatch")


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _exact_typed_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(
            _exact_typed_equal(left[key], right[key]) for key in left
        )
    if type(left) in {list, tuple}:
        return len(left) == len(right) and all(
            _exact_typed_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def _validate_update_log_schema(row: Mapping[str, Any]) -> None:
    _strict_json_object(
        row, label="A23 update log", strings=UPDATE_LOG_STRING_FIELDS,
        integers=UPDATE_LOG_INTEGER_FIELDS, floats=UPDATE_LOG_FLOAT_FIELDS,
        booleans=UPDATE_LOG_BOOLEAN_FIELDS, lists={"sampled_episode_ids"},
    )
    if (
        row["schema_version"] != UPDATE_LOG_VERSION
        or row["primitives_version"] != A23_PRIMITIVES_VERSION
        or any(type(item) is not str for item in row["sampled_episode_ids"])
        or not _is_sha256(row["sampled_episode_ids_sha256"])
        or not _is_sha256(row["normalized_advantage_sha256"])
        or any(not math.isfinite(row[field]) for field in UPDATE_LOG_FLOAT_FIELDS)
    ):
        raise RuntimeError("A23 update log typed provenance mismatch")


def _selected_actor_summary_matches(row: Mapping[str, Any]) -> bool:
    mode = str(row["selected_actor_mode"])
    source = {
        "reward": "reward_advantage_raw",
        "unsafe": "unsafe_cost_to_go_raw",
        "wrong": "wrong_cost_to_go_raw",
    }.get(mode)
    if source is None:
        return False
    selected = "selected_actor_advantage_raw"
    expected_mean = row[f"{source}_mean"]
    if mode != "reward":
        expected_mean = -expected_mean
    return bool(
        row[f"{selected}_mean"] == expected_mean
        and row[f"{selected}_std"] == row[f"{source}_std"]
        and row[f"{selected}_max_abs"] == row[f"{source}_max_abs"]
        and row["selected_advantage_constant"]
        is (row[f"{selected}_std"] == 0.0)
    )


@dataclass(frozen=True)
class A23Schedule:
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


def schedule(smoke: bool) -> A23Schedule:
    if smoke:
        return A23Schedule(
            mode="smoke", seeds=(FORMAL_SEEDS[0],),
            folds=tuple(range(DEFAULT_FOLDS)), updates=1,
            episodes_per_update=4, calibration_episodes_per_fold=8,
            outer_episodes_per_fold=8, bootstrap_iterations=200,
            bootstrap_seed=BOOTSTRAP_SEED, threads=2,
        )
    return A23Schedule(
        mode="adaptive-development", seeds=FORMAL_SEEDS,
        folds=tuple(range(DEFAULT_FOLDS)), updates=FORMAL_UPDATES,
        episodes_per_update=FORMAL_EPISODES_PER_UPDATE,
        calibration_episodes_per_fold=600, outer_episodes_per_fold=600,
        bootstrap_iterations=20_000, bootstrap_seed=BOOTSTRAP_SEED,
        threads=FORMAL_THREADS,
    )


def expected_products(run_schedule: A23Schedule, *, all_feasible: bool) -> dict[str, int]:
    fits = len(run_schedule.folds) * len(run_schedule.seeds) * len(MECHANISMS)
    draws = fits * run_schedule.updates * run_schedule.episodes_per_update
    calibration = fits * run_schedule.calibration_episodes_per_fold
    comparator = (
        len(run_schedule.folds) * len(run_schedule.seeds)
        * len(A22_COMPARATORS) * run_schedule.calibration_episodes_per_fold
    )
    outer = (
        len(run_schedule.folds) * len(run_schedule.seeds)
        * run_schedule.outer_episodes_per_fold if all_feasible else 0
    )
    return {
        "fits": fits, "training_episode_draws": draws,
        "a23_calibration_rows": calibration,
        "a22_comparator_rows": comparator,
        "method_table_rows": calibration + comparator,
        "outer_rows": outer,
        "manifest_entries": fits * 4 + (8 if all_feasible else 6),
    }


def _formal_config(run_schedule: A23Schedule) -> PPOConfig:
    values = asdict(_formal_ppo_config())
    values.update({
        "updates": run_schedule.updates,
        "episodes_per_update": run_schedule.episodes_per_update,
    })
    return PPOConfig(**values)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid A23 JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"A23 JSON artifact is not an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid A23 JSONL artifact: {path}") from exc
    if any(not isinstance(row, dict) for row in rows):
        raise RuntimeError(f"A23 JSONL artifact contains a non-object: {path}")
    return rows


def _json_payload(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def _jsonl_payload(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        for row in rows
    )


def _atomic_text(path: Path, payload: str, *, replace: bool = False) -> None:
    partial = path.with_name(path.name + ".partial")
    if partial.exists() or (path.exists() and not replace):
        raise FileExistsError(path if path.exists() else partial)
    partial.write_text(payload, encoding="utf-8")
    os.replace(partial, path)


def _atomic_json(path: Path, value: Any, *, replace: bool = False) -> None:
    payload = _json_payload(value)
    json.loads(payload)
    _atomic_text(path, payload, replace=replace)
    if _digest(_read_json(path)) != _digest(value):
        raise RuntimeError("A23 atomic JSON publication changed payload")


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = _jsonl_payload(rows)
    for line in payload.splitlines():
        json.loads(line)
    _atomic_text(path, payload)
    if _digest(_read_jsonl(path)) != _digest(list(rows)):
        raise RuntimeError("A23 atomic JSONL publication changed payload")


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
        raise RuntimeError("formal A23 run requires a clean source checkout")
    required = tuple(PINNED_REUSED_SOURCES) + (
        "multitown/a23_cr_ppo.py", "multitown/a23_statistics.py",
        "multitown/a23_runner.py", "docs/A23_CRPO_INSPIRED_CR_PPO.md",
        "pyproject.toml",
    )
    hashes = {}
    for relative in required:
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"missing A23 source: {relative}")
        if require_clean:
            committed = subprocess.run(
                ["git", "show", f"HEAD:{relative}"], cwd=root,
                capture_output=True, check=True,
            ).stdout
            if path.read_bytes() != committed:
                raise RuntimeError(f"executed A23 source differs from HEAD: {relative}")
        hashes[relative] = _sha256(path)
    if any(hashes[name] != digest for name, digest in PINNED_REUSED_SOURCES.items()):
        raise RuntimeError("A23 reused implementation source pin changed")
    return {
        "revision": revision, "dirty": bool(status),
        "executed_source_sha256": hashes,
        "runtime": {
            "python": sys.version, "numpy": np.__version__,
            "torch": torch.__version__, "platform": platform.platform(),
        },
    }


def _fit_path(output: Path, fold: int, seed: int, mechanism: str) -> Path:
    return output / "fits" / f"outer-fold-{fold}" / f"seed-{seed}" / mechanism


def _fit_key(row: Mapping[str, Any]) -> tuple[int, int, str]:
    return int(row["outer_fold"]), int(row["training_seed"]), str(row["mechanism"])


def _checkpoint_atomic(
    path: Path, model: ActorCritic, config: PPOConfig, *, seed: int,
    update: int, metadata: Mapping[str, Any],
) -> None:
    partial = path.with_name(path.name + ".partial")
    if path.exists() or partial.exists():
        raise FileExistsError(path if path.exists() else partial)
    _save_checkpoint(
        partial, model, config, seed=seed, update=update,
        policy_version=POLICY_VERSION,
    )
    payload = torch.load(partial, map_location="cpu", weights_only=False)
    payload.update(dict(metadata))
    if any(
        not bool(torch.isfinite(tensor).all())
        for tensor in payload["model_state"].values()
    ):
        raise FloatingPointError("A23 checkpoint contains non-finite tensor")
    torch.save(payload, partial)
    verified = torch.load(partial, map_location="cpu", weights_only=False)
    if any(verified.get(key) != value for key, value in metadata.items()):
        raise RuntimeError("A23 checkpoint metadata changed before publication")
    os.replace(partial, path)


def _initialization_oracle(seed: int, config: PPOConfig) -> tuple[str, str]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        _set_seed(seed)
        model = ActorCritic(
            MultiTownLongHorizonEnv.observation_size,
            config.hidden_size, ACTION_COUNT,
        ).cpu()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.learning_rate, eps=1e-5,
        )
        return model_parameter_sha256(model), initial_optimizer_sha256(
            optimizer, model,
        )
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def _fit(
    episodes: Sequence[LongHorizonEpisode], *, a8_rows: Sequence[Mapping[str, Any]],
    mechanism: CRPPOMechanism, seed: int, outer_fold: int,
    run_schedule: A23Schedule, output: Path, run_contract_sha256: str,
    partition: Mapping[str, Any], expected_a22_sample_ids: Sequence[str],
) -> dict[str, Any]:
    canonical_ids = [row.episode_id for row in episodes]
    if (
        len(episodes) != 1800 or canonical_ids != sorted(canonical_ids)
        or set(canonical_ids) != set(partition["inner_train_ids"])
    ):
        raise ValueError("A23 fit requires the exact canonical inner-train partition")
    config = _formal_config(run_schedule)
    thresholds = thresholds_from_inner_train(a8_rows)
    expected_mean = sum(len(row.incidents) for row in episodes) / len(episodes)
    if thresholds.mean_incidents != expected_mean:
        raise RuntimeError("A23 mean incident threshold does not bind to train bank")
    _set_seed(seed)
    model = ActorCritic(
        MultiTownLongHorizonEnv.observation_size, config.hidden_size, ACTION_COUNT,
    ).cpu()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, eps=1e-5,
    )
    initial_model_sha = model_parameter_sha256(model)
    initial_optimizer_sha = initial_optimizer_sha256(optimizer, model)
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
        transition_episodes = []
        rollout_rows = []
        update_ids = []
        for _ in range(config.episodes_per_update):
            episode = episodes[sample_rng.randrange(len(episodes))]
            update_ids.append(episode.episode_id)
            sampled_ids.append(episode.episode_id)
            transitions, rollout = constrained_rollout(
                model, episode, torch.device("cpu"),
                mean_incidents=thresholds.mean_incidents,
                shield_enabled=mechanism.shield_enabled,
            )
            if any(type(row.get("shield_intervened")) is not bool for row in transitions):
                raise ValueError("A23 rollout shield field is not strict bool")
            transition_episodes.append(transitions)
            rollout_rows.append(rollout)
        decision, metrics = cr_ppo_update(
            model, optimizer, transition_episodes, config, torch.device("cpu"),
            tensor_generator, thresholds=thresholds,
        )
        modes.append(decision.mode)
        mode_counts[decision.mode] += 1
        steps = sum(len(row) for row in transition_episodes)
        minibatches = config.ppo_epochs * math.ceil(steps / config.minibatch_size)
        incidents = sum(int(row["incidents"]) for row in rollout_rows)
        unsafe_events = sum(bool(row["unsafe_episode"]) for row in rollout_rows)
        wrong_executions = sum(int(row["wrong_executions"]) for row in rollout_rows)
        shield_decisions = sum(
            int(transition["shield_intervened"])
            for episode in transition_episodes for transition in episode
        )
        if (
            unsafe_events != int(metrics["rollout_unsafe_events"])
            or wrong_executions != int(metrics["rollout_wrong_executions"])
            or shield_decisions
            != sum(int(row["shield_interventions"]) for row in rollout_rows)
            or incidents != sum(
                len(next(item for item in episodes if item.episode_id == episode_id).incidents)
                for episode_id in update_ids
            )
        ):
            raise RuntimeError("A23 rollout summary does not bind to transitions/IDs")
        total_steps += steps
        total_minibatches += minibatches
        summaries = {
            "reward_advantage_raw": metrics.pop("reward_advantage_raw"),
            "unsafe_cost_to_go_raw": metrics.pop("unsafe_advantage_raw"),
            "wrong_cost_to_go_raw": metrics.pop("wrong_advantage_raw"),
            "selected_actor_advantage_raw": metrics.pop("selected_advantage_raw"),
        }
        row = {
            "schema_version": UPDATE_LOG_VERSION,
            "primitives_version": A23_PRIMITIVES_VERSION,
            "outer_fold": outer_fold, "training_seed": seed,
            "mechanism": mechanism.name, "update": update,
            "episodes_per_update": config.episodes_per_update,
            "rollout_episodes": config.episodes_per_update,
            "sampled_episode_ids": update_ids,
            "sampled_episode_ids_sha256": _digest(update_ids),
            "environment_steps": steps, "optimizer_minibatches": minibatches,
            "rollout_unsafe_events": unsafe_events,
            "rollout_wrong_executions": wrong_executions,
            "rollout_incidents": incidents,
            "shield_mask_active_decisions": shield_decisions,
            "rollout_episode_success_rate": float(np.mean([
                bool(item["episode_success"]) for item in rollout_rows
            ])),
            "rollout_tokens_per_episode": float(np.mean([
                int(item["tokens_used"]) for item in rollout_rows
            ])),
            "selected_actor_mode": decision.mode,
            "unsafe_cost": decision.unsafe_cost,
            "wrong_cost": decision.wrong_cost,
            "unsafe_threshold": decision.unsafe_threshold,
            "wrong_threshold": decision.wrong_threshold,
            "unsafe_violation": decision.unsafe_violation,
            "wrong_violation": decision.wrong_violation,
            "unsafe_normalized_violation": decision.unsafe_normalized_violation,
            "wrong_normalized_violation": decision.wrong_normalized_violation,
            "unsafe_eligible": decision.unsafe_eligible,
            "wrong_eligible": decision.wrong_eligible,
            "unsafe_tie_break_used": decision.unsafe_tie_break_used,
            "mean_incidents": thresholds.mean_incidents,
            "reward_mode_count": mode_counts["reward"],
            "unsafe_mode_count": mode_counts["unsafe"],
            "wrong_mode_count": mode_counts["wrong"],
            **{f"{name}_{field}": value for name, summary in summaries.items()
               for field, value in summary.items()},
            **metrics,
        }
        if (
            row["reward_mode_count"] + row["unsafe_mode_count"]
            + row["wrong_mode_count"] != update
            or any(
                isinstance(value, (int, float, np.integer, np.floating))
                and not math.isfinite(float(value)) for value in row.values()
            )
        ):
            raise FloatingPointError("invalid A23 update log row")
        logs.append(row)
        _atomic_json(output / "progress.json", {
            "schema_version": "multitown-a23-fit-progress-v1",
            "outer_fold": outer_fold, "training_seed": seed,
            "mechanism": mechanism.name, "current_update": update,
            "scheduled_updates": config.updates,
            "reward_mode_count": mode_counts["reward"],
            "unsafe_mode_count": mode_counts["unsafe"],
            "wrong_mode_count": mode_counts["wrong"],
            "outer_evaluation_started": False,
        }, replace=update > 1)
    _atomic_jsonl(output / "training-metrics.jsonl", logs)
    training_log_sha = _sha256(output / "training-metrics.jsonl")
    sample_sha = _digest(sampled_ids)
    mode_sha = mode_sequence_sha256(modes)
    if sampled_ids != list(expected_a22_sample_ids):
        raise RuntimeError("A23 sample sequence differs from pinned A22")
    checkpoint = output / "final.pt"
    checkpoint_metadata = {
        "primitives_version": A23_PRIMITIVES_VERSION,
        "run_contract_sha256": run_contract_sha256,
        "mechanism": asdict(mechanism), "outer_fold": outer_fold,
        "safety_thresholds": asdict(thresholds),
        "training_log_sha256": training_log_sha,
        "mode_sequence_sha256": mode_sha,
        "initial_model_sha256": initial_model_sha,
        "initial_optimizer_sha256": initial_optimizer_sha,
        "sample_sequence_sha256": sample_sha,
    }
    _checkpoint_atomic(
        checkpoint, model, config, seed=seed, update=config.updates,
        metadata=checkpoint_metadata,
    )
    result = {
        "schema_version": "multitown-a23-fit-complete-v1",
        "primitives_version": A23_PRIMITIVES_VERSION,
        "outer_fold": outer_fold, "training_seed": seed,
        "mechanism": mechanism.name,
        "shield_enabled": mechanism.shield_enabled,
        "final_update": config.updates,
        "training_episode_draws": len(sampled_ids),
        "sample_sequence_sha256": sample_sha,
        "sampled_unique_episodes": len(set(sampled_ids)),
        "initial_model_sha256": initial_model_sha,
        "initial_optimizer_sha256": initial_optimizer_sha,
        "mode_sequence_sha256": mode_sha,
        "training_log_sha256": training_log_sha,
        "checkpoint_sha256": _sha256(checkpoint),
        "inner_train_ids_sha256": partition["inner_train_ids_sha256"],
        "calibration_ids_sha256": partition["inner_calibration_ids_sha256"],
        "outer_ids_sha256": partition["outer_ids_sha256"],
        "thresholds": asdict(thresholds),
        "environment_steps": total_steps,
        "optimizer_minibatches": total_minibatches,
        "mode_counts": {
            mode: mode_counts[mode] for mode in ("reward", "unsafe", "wrong")
        },
        "run_contract_sha256": run_contract_sha256,
        "training_seconds": time.perf_counter() - started,
        "calibration_evaluations_during_training": 0,
        "outer_evaluations_during_training": 0,
        "selected_checkpoint": "final",
    }
    _atomic_json(output / "fit-complete.json", result)
    return result


def _evaluate(
    model: ActorCritic, episodes: Sequence[LongHorizonEpisode], *,
    mechanism: CRPPOMechanism, training_seed: int, design_outer_fold: int,
    assignment_index: Mapping[str, Any], a8_index: Mapping[str, Mapping[str, Any]],
    bank_sha256: str, resource_sha256: str, environment_sha256: str,
    checkpoint_sha256: str, run_contract_sha256: str, phase: str,
    selection_manifest_sha256: str | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for episode in episodes:
        counters = {"shield": 0}

        def action(observation: np.ndarray, mask: np.ndarray) -> int:
            selected, intervened = deterministic_action(
                model, observation, mask,
                shield_enabled=mechanism.shield_enabled,
            )
            counters["shield"] += int(intervened)
            return selected

        assignment = assignment_index[episode.episode_id]
        baseline = a8_index[episode.episode_id]
        row = _evaluate_episode(
            episode, action, fold=assignment.fold,
            system=f"A23-{mechanism.name}", training_seed=training_seed,
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
            "shield_interventions": counters["shield"],
            "selection_manifest_sha256": selection_manifest_sha256,
        })
        rows.append(row)
    return rows


def _validate_checkpoint(
    path: Path, *, fit: Mapping[str, Any], mechanism: CRPPOMechanism,
    fold: int, seed: int, run_schedule: A23Schedule,
    run_contract_sha256: str,
) -> ActorCritic:
    if not path.is_file() or _sha256(path) != fit["checkpoint_sha256"]:
        raise RuntimeError("A23 checkpoint disk hash mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _strict_json_object(
        payload, label="A23 checkpoint", strings={
            "policy_version", "primitives_version", "run_contract_sha256",
            "training_log_sha256", "mode_sequence_sha256",
            "initial_model_sha256", "initial_optimizer_sha256",
            "sample_sequence_sha256",
        }, integers={
            "observation_size", "action_count", "hidden_size", "seed",
            "update", "outer_fold",
        }, objects={
            "ppo_config", "model_state", "mechanism", "safety_thresholds",
        },
    )
    config = _formal_config(run_schedule)
    expected_metadata = {
        "policy_version": POLICY_VERSION,
        "primitives_version": A23_PRIMITIVES_VERSION,
        "observation_size": MultiTownLongHorizonEnv.observation_size,
        "action_count": ACTION_COUNT, "hidden_size": config.hidden_size,
        "seed": seed, "update": run_schedule.updates,
        "ppo_config": asdict(config),
        "run_contract_sha256": run_contract_sha256,
        "mechanism": asdict(mechanism), "outer_fold": fold,
        "safety_thresholds": fit["thresholds"],
        "training_log_sha256": fit["training_log_sha256"],
        "mode_sequence_sha256": fit["mode_sequence_sha256"],
        "initial_model_sha256": fit["initial_model_sha256"],
        "initial_optimizer_sha256": fit["initial_optimizer_sha256"],
        "sample_sequence_sha256": fit["sample_sequence_sha256"],
    }
    actual_metadata = {
        key: value for key, value in payload.items() if key != "model_state"
    }
    if not _exact_typed_equal(actual_metadata, expected_metadata):
        raise RuntimeError("A23 checkpoint strict metadata mismatch")
    expected_state = ActorCritic(
        MultiTownLongHorizonEnv.observation_size, config.hidden_size, ACTION_COUNT,
    ).state_dict()
    state = payload["model_state"]
    if set(state) != set(expected_state) or any(
        type(state[name]) is not torch.Tensor
        or state[name].shape != expected_state[name].shape
        or state[name].dtype != expected_state[name].dtype
        or state[name].device.type != "cpu"
        or not bool(torch.isfinite(state[name]).all())
        for name in expected_state
    ):
        raise RuntimeError("A23 checkpoint model-state schema mismatch")
    model, metadata = load_checkpoint(
        path, torch.device("cpu"), expected_policy_version=POLICY_VERSION,
    )
    loaded_metadata = {
        key: value for key, value in metadata.items() if key != "model_state"
    }
    if (
        _sha256(path) != fit["checkpoint_sha256"]
        or not _exact_typed_equal(loaded_metadata, expected_metadata)
    ):
        raise RuntimeError("A23 checkpoint loader/provenance mismatch")
    if any(not bool(torch.isfinite(item).all()) for item in model.state_dict().values()):
        raise RuntimeError("A23 checkpoint contains non-finite model state")
    return model


def _validate_fits(
    fits: Sequence[Mapping[str, Any]], *, output: Path,
    run_schedule: A23Schedule, partitions: Mapping[int, Mapping[str, Any]],
    run_contract_sha256: str,
    a22_sample_sequences: Mapping[tuple[int, int], Sequence[str]],
    a8_index: Mapping[str, Mapping[str, Any]],
) -> dict[tuple[int, int, str], Mapping[str, Any]]:
    expected_keys = {
        (fold, seed, mechanism.name) for fold in run_schedule.folds
        for seed in run_schedule.seeds for mechanism in MECHANISMS
    }
    _assert_exact_keys([_fit_key(row) for row in fits], expected_keys, label="A23 fit")
    fit_index = {_fit_key(row): row for row in fits}
    config = _formal_config(run_schedule)
    initialization_oracles = {
        seed: _initialization_oracle(seed, config) for seed in run_schedule.seeds
    }
    expected_thresholds = {
        fold: asdict(thresholds_from_inner_train([
            a8_index[episode_id]
            for episode_id in partitions[fold]["inner_train_ids"]
        ]))
        for fold in run_schedule.folds
    }
    for fold, seed, mechanism_name in sorted(expected_keys):
        fit = fit_index[(fold, seed, mechanism_name)]
        mechanism = next(item for item in MECHANISMS if item.name == mechanism_name)
        fit_dir = _fit_path(output, fold, seed, mechanism_name)
        actual_files = {path.name for path in fit_dir.iterdir() if path.is_file()}
        if actual_files != {
            "training-metrics.jsonl", "final.pt", "fit-complete.json",
            "progress.json",
        }:
            raise RuntimeError("A23 fit path inventory mismatch")
        persisted = _read_json(fit_dir / "fit-complete.json")
        logs = _read_jsonl(fit_dir / "training-metrics.jsonl")
        progress = _read_json(fit_dir / "progress.json")
        _strict_json_object(
            persisted, label="A23 fit completion", strings=FIT_STRING_FIELDS,
            integers=FIT_INTEGER_FIELDS, floats={"training_seconds"},
            booleans={"shield_enabled"}, objects={"thresholds", "mode_counts"},
        )
        _strict_json_object(
            persisted["thresholds"], label="A23 fit thresholds",
            floats={"mean_incidents", "unsafe", "wrong_per_incident"},
        )
        _strict_json_object(
            persisted["mode_counts"], label="A23 fit mode counts",
            integers={"reward", "unsafe", "wrong"},
        )
        _strict_json_object(
            progress, label="A23 fit progress",
            strings={"schema_version", "mechanism"},
            integers={
                "outer_fold", "training_seed", "current_update",
                "scheduled_updates", "reward_mode_count",
                "unsafe_mode_count", "wrong_mode_count",
            },
            booleans={"outer_evaluation_started"},
        )
        if persisted != dict(fit) or len(logs) != run_schedule.updates:
            raise RuntimeError("A23 fit completion/log mismatch")
        if (
            persisted["schema_version"] != "multitown-a23-fit-complete-v1"
            or persisted["primitives_version"] != A23_PRIMITIVES_VERSION
            or persisted["outer_fold"] != fold
            or persisted["training_seed"] != seed
            or persisted["mechanism"] != mechanism_name
            or persisted["shield_enabled"] is not mechanism.shield_enabled
            or persisted["thresholds"] != expected_thresholds[fold]
            or persisted["final_update"] != run_schedule.updates
            or persisted["training_episode_draws"]
            != run_schedule.updates * run_schedule.episodes_per_update
            or persisted["run_contract_sha256"] != run_contract_sha256
            or persisted["inner_train_ids_sha256"]
            != partitions[fold]["inner_train_ids_sha256"]
            or persisted["calibration_ids_sha256"]
            != partitions[fold]["inner_calibration_ids_sha256"]
            or persisted["outer_ids_sha256"]
            != partitions[fold]["outer_ids_sha256"]
            or persisted["selected_checkpoint"] != "final"
            or persisted["calibration_evaluations_during_training"] != 0
            or persisted["outer_evaluations_during_training"] != 0
            or not math.isfinite(persisted["training_seconds"])
            or persisted["training_seconds"] < 0.0
            or any(
                not _is_sha256(persisted[field])
                for field in FIT_STRING_FIELDS if field.endswith("sha256")
            )
            or progress != {
                "schema_version": "multitown-a23-fit-progress-v1",
                "outer_fold": fold, "training_seed": seed,
                "mechanism": mechanism_name,
                "current_update": run_schedule.updates,
                "scheduled_updates": run_schedule.updates,
                "reward_mode_count": persisted["mode_counts"]["reward"],
                "unsafe_mode_count": persisted["mode_counts"]["unsafe"],
                "wrong_mode_count": persisted["mode_counts"]["wrong"],
                "outer_evaluation_started": False,
            }
        ):
            raise RuntimeError("A23 fit/progress provenance mismatch")
        sampled: list[str] = []
        modes: list[str] = []
        steps = 0
        minibatches = 0
        counts = Counter()
        thresholds = SafetyThresholds(**fit["thresholds"])
        train_ids = set(partitions[fold]["inner_train_ids"])
        for expected_update, row in enumerate(logs, start=1):
            _validate_update_log_schema(row)
            update_ids = [str(item) for item in row["sampled_episode_ids"]]
            mode = str(row["selected_actor_mode"])
            counts[mode] += 1
            modes.append(mode)
            sampled.extend(update_ids)
            steps += int(row["environment_steps"])
            minibatches += int(row["optimizer_minibatches"])
            decision = select_actor_mode(
                unsafe_events=int(row["rollout_unsafe_events"]),
                wrong_executions=int(row["rollout_wrong_executions"]),
                episodes=run_schedule.episodes_per_update,
                thresholds=thresholds,
            )
            decision_fields = {
                "selected_actor_mode": decision.mode,
                "unsafe_cost": decision.unsafe_cost,
                "wrong_cost": decision.wrong_cost,
                "unsafe_threshold": decision.unsafe_threshold,
                "wrong_threshold": decision.wrong_threshold,
                "unsafe_violation": decision.unsafe_violation,
                "wrong_violation": decision.wrong_violation,
                "unsafe_normalized_violation": decision.unsafe_normalized_violation,
                "wrong_normalized_violation": decision.wrong_normalized_violation,
                "unsafe_eligible": decision.unsafe_eligible,
                "wrong_eligible": decision.wrong_eligible,
                "unsafe_tie_break_used": decision.unsafe_tie_break_used,
            }
            expected_incidents = sum(
                int(a8_index[episode_id]["incidents"])
                for episode_id in update_ids
            )
            digest = str(row["normalized_advantage_sha256"])
            summary_fields = (
                "reward_advantage_raw", "unsafe_cost_to_go_raw",
                "wrong_cost_to_go_raw", "selected_actor_advantage_raw",
            )
            if (
                int(row["update"]) != expected_update
                or int(row["outer_fold"]) != fold
                or int(row["training_seed"]) != seed
                or row["mechanism"] != mechanism_name
                or len(update_ids) != run_schedule.episodes_per_update
                or int(row["episodes_per_update"])
                != run_schedule.episodes_per_update
                or int(row["rollout_episodes"])
                != run_schedule.episodes_per_update
                or not set(update_ids) <= train_ids
                or row["sampled_episode_ids_sha256"] != _digest(update_ids)
                or row["mean_incidents"] != thresholds.mean_incidents
                or mode not in {"reward", "unsafe", "wrong"}
                or not _selected_actor_summary_matches(row)
                or int(row["reward_mode_count"]) != counts["reward"]
                or int(row["unsafe_mode_count"]) != counts["unsafe"]
                or int(row["wrong_mode_count"]) != counts["wrong"]
                or any(row.get(field) != value for field, value in decision_fields.items())
                or int(row["environment_steps"]) <= 0
                or int(row["optimizer_minibatches"])
                != config.ppo_epochs * math.ceil(
                    int(row["environment_steps"]) / config.minibatch_size
                )
                or not 0 <= int(row["shield_mask_active_decisions"])
                <= int(row["environment_steps"])
                or int(row["rollout_incidents"]) <= 0
                or int(row["rollout_incidents"]) != expected_incidents
                or not 0 <= int(row["rollout_unsafe_events"])
                <= run_schedule.episodes_per_update
                or not 0 <= int(row["rollout_wrong_executions"])
                <= int(row["rollout_incidents"])
                or not 0.0 <= row["rollout_episode_success_rate"] <= 1.0
                or row["rollout_tokens_per_episode"] <= 0.0
                or (
                    not mechanism.shield_enabled
                    and int(row["shield_mask_active_decisions"]) != 0
                )
                or any(
                    row[f"{prefix}_{suffix}"] < 0.0
                    for prefix in summary_fields
                    for suffix in ("std", "max_abs")
                )
                or any(
                    not math.isfinite(row[f"{prefix}_{suffix}"])
                    for prefix in summary_fields
                    for suffix in ("mean", "std", "max_abs")
                )
                or any(
                    not math.isfinite(row[field]) for field in (
                        "unsafe_cost", "wrong_cost", "unsafe_threshold",
                        "wrong_threshold", "unsafe_violation", "wrong_violation",
                        "unsafe_normalized_violation",
                        "wrong_normalized_violation", "policy_loss", "value_loss",
                        "entropy", "approx_kl", "clip_fraction",
                    )
                )
            ):
                raise RuntimeError("A23 training log binding mismatch")
        expected_sequence = _expected_sample_sequence(
            partitions[fold]["inner_train_ids"], seed=seed,
            draws=run_schedule.updates * run_schedule.episodes_per_update,
        )
        expected_a22_sequence = list(a22_sample_sequences[(fold, seed)])[
            :len(expected_sequence)
        ]
        if (
            sampled != expected_sequence
            or sampled != expected_a22_sequence
            or fit["sample_sequence_sha256"] != _digest(sampled)
            or fit["mode_sequence_sha256"] != mode_sequence_sha256(modes)
            or fit["training_log_sha256"]
            != _sha256(fit_dir / "training-metrics.jsonl")
            or int(fit["environment_steps"]) != steps
            or int(fit["optimizer_minibatches"]) != minibatches
            or int(fit["sampled_unique_episodes"]) != len(set(sampled))
            or progress.get("current_update") != run_schedule.updates
            or progress.get("reward_mode_count") != counts["reward"]
            or progress.get("unsafe_mode_count") != counts["unsafe"]
            or progress.get("wrong_mode_count") != counts["wrong"]
            or fit["mode_counts"] != {
                mode: counts[mode] for mode in ("reward", "unsafe", "wrong")
            }
            or (
                fit["initial_model_sha256"], fit["initial_optimizer_sha256"]
            ) != initialization_oracles[seed]
        ):
            raise RuntimeError("A23 fit sequence/progress mismatch")
        _validate_checkpoint(
            fit_dir / "final.pt", fit=fit, mechanism=mechanism,
            fold=fold, seed=seed, run_schedule=run_schedule,
            run_contract_sha256=run_contract_sha256,
        )
    for fold in run_schedule.folds:
        for seed in run_schedule.seeds:
            pair = [fit_index[(fold, seed, item.name)] for item in MECHANISMS]
            for field in (
                "initial_model_sha256", "initial_optimizer_sha256",
                "sample_sequence_sha256",
            ):
                if len({row[field] for row in pair}) != 1:
                    raise RuntimeError(f"A23 paired fit differs in {field}")
    return fit_index


def _validate_evaluation_rows(
    rows: Sequence[Mapping[str, Any]], *, phase: str,
    run_schedule: A23Schedule, partitions: Mapping[int, Mapping[str, Any]],
    fit_index: Mapping[tuple[int, int, str], Mapping[str, Any]],
    selections: Sequence[Mapping[str, Any]] | None,
    selection_manifest_sha256: str | None,
    assignment_index: Mapping[str, Any],
    a8_index: Mapping[str, Mapping[str, Any]], bank_sha256: str,
    resource_sha256: str, environment_sha256: str,
    run_contract_sha256: str,
) -> None:
    ids_by_fold = {
        fold: tuple(partitions[fold][
            "inner_calibration_ids" if phase == "inner-calibration" else "outer_ids"
        ][:run_schedule.calibration_episodes_per_fold if phase == "inner-calibration"
          else run_schedule.outer_episodes_per_fold])
        for fold in run_schedule.folds
    }
    if phase == "inner-calibration":
        mechanisms = {fold: tuple(item.name for item in MECHANISMS) for fold in run_schedule.folds}
    else:
        if selections is None:
            raise ValueError("A23 outer validation requires selections")
        mechanisms = {
            int(row["outer_fold"]): (str(row["selected_mechanism"]),)
            for row in selections
        }
    expected = {
        (fold, seed, mechanism, episode_id)
        for fold in run_schedule.folds for seed in run_schedule.seeds
        for mechanism in mechanisms[fold] for episode_id in ids_by_fold[fold]
    }
    actual = [
        (int(row["design_outer_fold"]), int(row["training_seed"]),
         str(row["mechanism"]), str(row["episode_id"])) for row in rows
    ]
    _assert_exact_keys(actual, expected, label=f"A23 {phase} row")
    for row, key in zip(rows, actual, strict=True):
        fold, seed, mechanism, episode_id = key
        fit = fit_index[(fold, seed, mechanism)]
        expected_physical = (
            (fold + 1) % DEFAULT_FOLDS if phase == "inner-calibration" else fold
        )
        assignment = assignment_index[episode_id]
        if (
            int(row["outer_fold"]) != expected_physical
            or row["evaluation_phase"] != phase
            or row["system"] != f"A23-{mechanism}"
            or row["final_checkpoint_sha256"] != fit["checkpoint_sha256"]
            or row["selection_manifest_sha256"] != selection_manifest_sha256
            or episode_id not in ids_by_fold[fold]
            or row["episode_sha256"] != assignment.episode_sha256
            or row["train_bank_sha256"] != bank_sha256
            or row["resource_contract_sha256"] != resource_sha256
            or row["environment_source_sha256"] != environment_sha256
            or row["a8_row_sha256"] != _digest(a8_index[episode_id])
            or row["run_contract_sha256"] != run_contract_sha256
        ):
            raise RuntimeError("A23 evaluation provenance mismatch")


def _a22_fit_bindings(
    a22_root: Path,
) -> tuple[
    dict[str, Any], dict[tuple[int, int], str],
    dict[tuple[int, int], tuple[str, ...]],
]:
    all_fits = _read_json(a22_root / "all-fits-complete.json")
    fits = all_fits.get("fits")
    if not isinstance(fits, list) or len(fits) != 60:
        raise RuntimeError("A23 pinned A22 fit product changed")
    checkpoint = {}
    samples: dict[tuple[int, int], set[str]] = defaultdict(set)
    sequences: dict[tuple[int, int], tuple[str, ...]] = {}
    for row in fits:
        key = (int(row["outer_fold"]), int(row["training_seed"]), str(row["mechanism"]))
        checkpoint[f"{key[0]}:{key[1]}:{key[2]}"] = str(row["checkpoint_sha256"])
        samples[(key[0], key[1])].add(str(row["sample_sequence_sha256"]))
        logs = _read_jsonl(
            a22_root / "fits" / f"outer-fold-{key[0]}" / f"seed-{key[1]}"
            / key[2] / "training-metrics.jsonl"
        )
        sequence = tuple(
            str(episode_id) for update in logs
            for episode_id in update["sampled_episode_ids"]
        )
        if len(logs) != FORMAL_UPDATES or _digest(list(sequence)) != row[
            "sample_sequence_sha256"
        ]:
            raise RuntimeError("A23 pinned A22 training sequence changed")
        pair_key = (key[0], key[1])
        if pair_key in sequences and sequences[pair_key] != sequence:
            raise RuntimeError("A23 pinned A22 mechanisms have different sequences")
        sequences[pair_key] = sequence
    if len(checkpoint) != 60 or any(len(values) != 1 for values in samples.values()):
        raise RuntimeError("A23 pinned A22 fit binding is incomplete")
    return (
        checkpoint,
        {key: next(iter(values)) for key, values in samples.items()},
        sequences,
    )


def _a22_calibration_comparators(
    a22_root: Path, *, run_schedule: A23Schedule,
    partitions: Mapping[int, Mapping[str, Any]],
    checkpoint_bindings: Mapping[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = _read_jsonl(a22_root / "calibration-decisions.jsonl")
    formal_expected = {
        (fold, seed, mechanism, episode_id)
        for fold in range(DEFAULT_FOLDS) for seed in FORMAL_SEEDS
        for mechanism in A22_COMPARATORS
        for episode_id in partitions[fold]["inner_calibration_ids"]
    }
    allowed = [row for row in raw if str(row["mechanism"]) in A22_COMPARATORS]
    keys = [
        (int(row["design_outer_fold"]), int(row["training_seed"]),
         str(row["mechanism"]), str(row["episode_id"])) for row in allowed
    ]
    _assert_exact_keys(keys, formal_expected, label="A23 pinned A22 comparator")
    for row, key in zip(allowed, keys, strict=True):
        fold, seed, mechanism, _ = key
        if (
            int(row["outer_fold"]) != (fold + 1) % DEFAULT_FOLDS
            or row["evaluation_phase"] != "inner-calibration"
            or row["selection_manifest_sha256"] is not None
            or row["run_contract_sha256"] != EXPECTED_A22_RUN_DIGEST
            or row["final_checkpoint_sha256"]
            != checkpoint_bindings[f"{fold}:{seed}:{mechanism}"]
        ):
            raise RuntimeError("A23 pinned A22 comparator provenance changed")
    chosen_keys = {
        (fold, seed, mechanism, episode_id)
        for fold in run_schedule.folds for seed in run_schedule.seeds
        for mechanism in A22_COMPARATORS
        for episode_id in partitions[fold]["inner_calibration_ids"][
            :run_schedule.calibration_episodes_per_fold
        ]
    }
    selected = [row for row, key in zip(allowed, keys, strict=True) if key in chosen_keys]
    _assert_exact_keys([
        (int(row["design_outer_fold"]), int(row["training_seed"]),
         str(row["mechanism"]), str(row["episode_id"])) for row in selected
    ], chosen_keys, label="A23 scheduled A22 comparator")
    binding = {
        "artifact_manifest_sha256": EXPECTED_A22["artifact-manifest.json"],
        "calibration_sha256": EXPECTED_A22["calibration-decisions.jsonl"],
        "selection_sha256": EXPECTED_A22["all-selections-frozen.json"],
        "run_contract_digest": EXPECTED_A22_RUN_DIGEST,
        "full_allowed_rows": len(allowed),
        "scheduled_rows": len(selected),
    }
    return selected, binding


def _select(
    rows: Sequence[Mapping[str, Any]], *, fold: int, seeds: Sequence[int],
    episode_ids: Sequence[str], a8_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    ids = tuple(str(item) for item in episode_ids)
    _assert_exact_keys(
        [(str(row["episode_id"]),) for row in a8_rows],
        {(episode_id,) for episode_id in ids}, label="A23 selection A8",
    )
    expected = {
        (mechanism.name, seed, episode_id) for mechanism in MECHANISMS
        for seed in seeds for episode_id in ids
    }
    actual = [
        (str(row["mechanism"]), int(row["training_seed"]), str(row["episode_id"]))
        for row in rows
    ]
    _assert_exact_keys(actual, expected, label="A23 selection calibration")
    if any(
        int(row["design_outer_fold"]) != fold
        or row["evaluation_phase"] != "inner-calibration"
        or row["selection_manifest_sha256"] is not None for row in rows
    ):
        raise RuntimeError("A23 selection received misbound calibration")
    unsafe = float(np.mean([bool(row["had_wrong_execution"]) for row in a8_rows]))
    wrong = sum(int(row["wrong_executions"]) for row in a8_rows) / sum(
        int(row["incidents"]) for row in a8_rows
    )
    thresholds = {
        "unsafe_episode": min(1.0, unsafe + 0.02),
        "wrong_executions_per_incident": min(1.0, wrong + 0.01),
    }
    candidates = {}
    feasible = []
    for order, mechanism in enumerate(MECHANISMS):
        mechanism_rows = [row for row in rows if row["mechanism"] == mechanism.name]
        by_seed = {}
        passes = []
        for seed in seeds:
            selected = [row for row in mechanism_rows if int(row["training_seed"]) == seed]
            summary = _summary(selected)
            passed = bool(
                summary["unsafe_episode_rate"] <= thresholds["unsafe_episode"]
                and summary["wrong_executions_per_incident"]
                <= thresholds["wrong_executions_per_incident"]
                and summary["invalid_actions"] == 0
                and summary["budget_violations"] == 0
            )
            by_seed[str(seed)] = {**summary, "safety_feasible": passed}
            passes.append(passed)
        overall = _summary(mechanism_rows)
        all_feasible = all(passes)
        candidates[mechanism.name] = {
            "mechanism": asdict(mechanism), "by_seed": by_seed,
            "scheduled_seed_summary": overall,
            "all_scheduled_seeds_feasible": all_feasible,
        }
        if all_feasible:
            feasible.append((
                -overall["episode_success_rate"], overall["tokens_per_episode"],
                order, mechanism.name,
            ))
    selected_name = min(feasible)[3] if feasible else None
    return {
        "outer_fold": fold, "selected_mechanism": selected_name,
        "status": "selected" if selected_name else "no_feasible_mechanism",
        "calibration_a8_summary": _summary(a8_rows),
        "calibration_thresholds": thresholds,
        "tie_order": [item.name for item in MECHANISMS],
        "selection_key": (
            "all-seed joint feasibility, max scheduled-seed mean autonomous "
            "success, min scheduled-seed mean tokens, fixed A23 order"
        ),
    }, candidates


def _validate_a8_rows(
    rows: Sequence[Mapping[str, Any]], *, episode_index: Mapping[str, Any],
    assignment_index: Mapping[str, Any], bank_sha256: str,
    resource_sha256: str, environment_sha256: str,
) -> dict[str, Mapping[str, Any]]:
    index = {str(row["episode_id"]): row for row in rows}
    if (
        len(rows) != 3000 or len(index) != len(rows)
        or set(index) != set(episode_index)
        or any("training_seed" not in row or row["training_seed"] is not None for row in rows)
        or any(
            int(row["outer_fold"])
            != assignment_index[str(row["episode_id"])].fold
            or row["episode_sha256"]
            != assignment_index[str(row["episode_id"])].episode_sha256
            or row["train_bank_sha256"] != bank_sha256
            or row["resource_contract_sha256"] != resource_sha256
            or row["environment_source_sha256"] != environment_sha256
            for row in rows
        )
    ):
        raise RuntimeError("A23 pinned A8 row product/binding changed")
    return index


def _preflight(root: Path, *, smoke: bool) -> dict[str, Any]:
    source = _source_state(root, require_clean=not smoke)
    a22_root = (root / A22_RAW_PATH).resolve()
    validate_a22_raw(a22_root)
    if any(_sha256(a22_root / name) != digest for name, digest in EXPECTED_A22.items()):
        raise RuntimeError("A23 pinned A22 artifact hash changed")
    a22_contract = _read_json(a22_root / "run-contract.json")
    if _digest(a22_contract) != EXPECTED_A22_RUN_DIGEST:
        raise RuntimeError("A23 pinned A22 run-contract digest changed")
    (
        a22_checkpoint_bindings, a22_sample_sha256, a22_sample_sequences,
    ) = _a22_fit_bindings(a22_root)
    a22_selection = _read_json(a22_root / "all-selections-frozen.json")
    if a22_selection.get("checkpoint_sha256") != a22_checkpoint_bindings:
        raise RuntimeError("A23 pinned A22 selection/checkpoint binding changed")

    frozen = _verify_r2_artifacts(root)
    r2_root = Path(frozen["path"]).resolve()
    if r2_root != (root / FROZEN_R2_PATH).resolve() or any((
        _sha256(r2_root / "artifact-manifest.json") != EXPECTED_R2["manifest"],
        _sha256(r2_root / "a8-oof-decisions.jsonl") != EXPECTED_R2["a8"],
        _sha256(r2_root / "run-contract.json") != EXPECTED_R2["run_file"],
        _digest(frozen["run_contract"]) != EXPECTED_R2["run_digest"],
    )):
        raise RuntimeError("A23 pinned R2 artifact binding changed")
    if any((
        a22_contract["frozen_r2_manifest_sha256"] != EXPECTED_R2["manifest"],
        a22_contract["frozen_r2_a8_sha256"] != EXPECTED_R2["a8"],
        a22_contract["frozen_r2_run_file_sha256"] != EXPECTED_R2["run_file"],
        a22_contract["frozen_r2_run_digest"] != EXPECTED_R2["run_digest"],
    )):
        raise RuntimeError("A23 A22-to-R2 binding changed")

    bank = load_frozen_train_bank(FROZEN_TRAIN_PATH)
    assignments = assign_stratified_group_folds(bank)
    resource = shared_resource_contract(bank)
    _validate_shared_stack_bindings(
        root, frozen["run_contract"], bank=bank,
        assignments=assignments, resource=resource,
    )
    bindings = {
        "train_bank": bank.payload_sha256,
        "fold_manifest": fold_manifest_sha256(assignments),
        "resource": resource_contract_sha256(resource),
        "environment": str(resource["environment_source_sha256"]),
    }
    if bindings != EXPECTED_BINDINGS or any((
        a22_contract["train_bank_sha256"] != bindings["train_bank"],
        a22_contract["fold_manifest_sha256"] != bindings["fold_manifest"],
        a22_contract["resource_contract_sha256"] != bindings["resource"],
        a22_contract["environment_source_sha256"] != bindings["environment"],
    )):
        raise RuntimeError("A23 bank/fold/resource binding changed")
    episode_index = {row.episode_id: row for row in bank.episodes}
    assignment_index = {row.episode_id: row for row in assignments}
    a8_rows = _read_jsonl(r2_root / "a8-oof-decisions.jsonl")
    a8_index = _validate_a8_rows(
        a8_rows, episode_index=episode_index,
        assignment_index=assignment_index, bank_sha256=bindings["train_bank"],
        resource_sha256=bindings["resource"],
        environment_sha256=bindings["environment"],
    )
    return {
        "source": source, "a22_root": a22_root,
        "a22_contract": a22_contract,
        "a22_selection": a22_selection,
        "a22_checkpoint_bindings": a22_checkpoint_bindings,
        "a22_sample_sha256": a22_sample_sha256,
        "a22_sample_sequences": a22_sample_sequences,
        "bank": bank, "assignments": assignments,
        "episode_index": episode_index, "assignment_index": assignment_index,
        "resource": resource, "bindings": bindings,
        "a8_rows": a8_rows, "a8_index": a8_index,
        "preflight_signature": _digest({
            "source": source, "a22_hashes": EXPECTED_A22,
            "a22_run_digest": EXPECTED_A22_RUN_DIGEST,
            "r2": EXPECTED_R2, "bindings": bindings,
            "a8_rows_sha256": EXPECTED_R2["a8"],
        }),
    }


def _revalidate_preflight(root: Path, *, smoke: bool, expected_signature: str) -> None:
    current = _preflight(root, smoke=smoke)
    if current["preflight_signature"] != expected_signature:
        raise RuntimeError("A23 preflight inputs changed during execution")


def _a22_selected_outer(
    context: Mapping[str, Any], *, run_schedule: A23Schedule,
    partitions: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    raw = _read_jsonl(context["a22_root"] / "outer-decisions.jsonl")
    selected_by_fold = {
        int(row["outer_fold"]): str(row["selected_mechanism"])
        for row in context["a22_selection"]["selections"]
    }
    expected_full = {
        (fold, seed, selected_by_fold[fold], episode_id)
        for fold in range(DEFAULT_FOLDS) for seed in FORMAL_SEEDS
        for episode_id in partitions[fold]["outer_ids"]
    }
    keys = [
        (int(row["design_outer_fold"]), int(row["training_seed"]),
         str(row["mechanism"]), str(row["episode_id"])) for row in raw
    ]
    _assert_exact_keys(keys, expected_full, label="A23 pinned A22 selected outer")
    for row, key in zip(raw, keys, strict=True):
        fold, seed, mechanism, _ = key
        if (
            int(row["outer_fold"]) != fold
            or row["evaluation_phase"] != "selected-outer"
            or row["run_contract_sha256"] != EXPECTED_A22_RUN_DIGEST
            or row["selection_manifest_sha256"]
            != EXPECTED_A22["all-selections-frozen.json"]
            or row["final_checkpoint_sha256"]
            != context["a22_checkpoint_bindings"][f"{fold}:{seed}:{mechanism}"]
        ):
            raise RuntimeError("A23 pinned A22 selected outer provenance changed")
    chosen = {
        (fold, seed, selected_by_fold[fold], episode_id)
        for fold in run_schedule.folds for seed in run_schedule.seeds
        for episode_id in partitions[fold]["outer_ids"][
            :run_schedule.outer_episodes_per_fold
        ]
    }
    rows = [row for row, key in zip(raw, keys, strict=True) if key in chosen]
    _assert_exact_keys([
        (int(row["design_outer_fold"]), int(row["training_seed"]),
         str(row["mechanism"]), str(row["episode_id"])) for row in rows
    ], chosen, label="A23 scheduled A22 selected outer")
    return rows


def _method_table(
    a23_rows: Sequence[Mapping[str, Any]],
    comparator_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [*a23_rows, *comparator_rows]
    result = {}
    for fold in sorted({int(row["design_outer_fold"]) for row in rows}):
        for seed in sorted({int(row["training_seed"]) for row in rows}):
            for mechanism in (
                *(item.name for item in MECHANISMS), *A22_COMPARATORS,
            ):
                selected = [
                    row for row in rows
                    if int(row["design_outer_fold"]) == fold
                    and int(row["training_seed"]) == seed
                    and row["mechanism"] == mechanism
                ]
                if not selected:
                    raise RuntimeError("A23 method table cell is empty")
                result[f"{fold}:{seed}:{mechanism}"] = _summary(selected)
    return result


def _training_mode_diagnostics(
    fits: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result = {}
    for fit in fits:
        counts = {
            mode: int(fit["mode_counts"][mode])
            for mode in ("reward", "unsafe", "wrong")
        }
        updates = int(fit["final_update"])
        if sum(counts.values()) != updates:
            raise RuntimeError("A23 training mode diagnostic count mismatch")
        key = (
            f"{int(fit['outer_fold'])}:{int(fit['training_seed'])}:"
            f"{fit['mechanism']}"
        )
        result[key] = {
            "outer_fold": int(fit["outer_fold"]),
            "training_seed": int(fit["training_seed"]),
            "mechanism": str(fit["mechanism"]),
            "updates": updates, "mode_counts": counts,
            "mode_fractions": {
                mode: count / updates for mode, count in counts.items()
            },
            "comparison_intervals_attached": False,
        }
    return result


def _expected_manifest_paths(
    run_schedule: A23Schedule, *, all_feasible: bool,
) -> set[str]:
    paths = {
        f"fits/outer-fold-{fold}/seed-{seed}/{mechanism.name}/{name}"
        for fold in run_schedule.folds for seed in run_schedule.seeds
        for mechanism in MECHANISMS for name in (
            "training-metrics.jsonl", "final.pt", "fit-complete.json",
            "progress.json",
        )
    }
    paths.update({
        "training-contract.json", "run-contract.json", "all-fits-complete.json",
        "calibration-decisions.jsonl", "all-selections-frozen.json", "result.json",
    })
    if all_feasible:
        paths.update({"OUTER_GATE_OPEN.json", "outer-decisions.jsonl"})
    return paths


def _manifest(
    output: Path, *, source_revision: str, run_schedule: A23Schedule,
    all_feasible: bool,
) -> dict[str, Any]:
    expected = _expected_manifest_paths(run_schedule, all_feasible=all_feasible)
    actual = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*") if path.is_file()
    }
    if actual != expected:
        raise RuntimeError(
            f"A23 manifest path mismatch: missing={sorted(expected-actual)[:5]}, "
            f"extra={sorted(actual-expected)[:5]}"
        )
    products = expected_products(run_schedule, all_feasible=all_feasible)
    if len(expected) != products["manifest_entries"]:
        raise RuntimeError("A23 manifest count differs from frozen product")
    return {
        "schema_version": "multitown-a23-artifact-manifest-v1",
        "source_revision": source_revision,
        "files": {
            name: {
                "bytes": (output / name).stat().st_size,
                "sha256": _sha256(output / name),
            } for name in sorted(expected)
        },
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }


def validate_manifest(output: Path, *, expected_entries: int) -> dict[str, Any]:
    manifest = _read_json(output / "artifact-manifest.json")
    files = manifest.get("files")
    if not isinstance(files, dict) or len(files) != expected_entries:
        raise RuntimeError("A23 manifest entry count mismatch")
    expected = set(files) | {"artifact-manifest.json"}
    actual = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*") if path.is_file()
    }
    if actual != expected or any(
        name.endswith(".partial") or name in {
            "RUNNING.json", "INVALIDATED.json", "INVALID_RESULT.json",
            "INVALID_ARTIFACT_MANIFEST.json",
        } for name in actual
    ):
        raise RuntimeError("A23 final manifest path set is invalid")
    for name, metadata in files.items():
        path = output / name
        if (
            path.stat().st_size != int(metadata["bytes"])
            or _sha256(path) != metadata["sha256"]
        ):
            raise RuntimeError(f"A23 manifest payload mismatch: {name}")
    return manifest


def _isolate_valid_outputs(output: Path) -> None:
    for valid, invalid in (
        ("result.json", "INVALID_RESULT.json"),
        ("artifact-manifest.json", "INVALID_ARTIFACT_MANIFEST.json"),
    ):
        source = output / valid
        target = output / invalid
        if source.exists() and not target.exists():
            os.replace(source, target)


def _fault(point: str, requested: str | None) -> None:
    if requested == point:
        raise RuntimeError(f"injected A23 failure: {point}")


class _FormalLockCreatedError(RuntimeError):
    """The persistent formal-attempt lock exists but could not be completed."""


def _acquire_formal_lock(lock: Path, descriptor: Mapping[str, Any]) -> None:
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(_json_payload(descriptor))
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException as exc:
        try:
            os.close(fd)
        except OSError:
            pass
        raise _FormalLockCreatedError(
            "A23 formal lock was created but its descriptor write failed"
        ) from exc


def _statistics_schedule(run_schedule: A23Schedule) -> A23StatisticsSchedule:
    return A23StatisticsSchedule(
        mode=run_schedule.mode, seeds=run_schedule.seeds,
        folds=run_schedule.folds, updates=run_schedule.updates,
        episodes_per_update=run_schedule.episodes_per_update,
        outer_episodes_per_fold=run_schedule.outer_episodes_per_fold,
        bootstrap_iterations=run_schedule.bootstrap_iterations,
        threads=run_schedule.threads,
    )


def _calibration_key_product(
    run_schedule: A23Schedule, partitions: Mapping[int, Mapping[str, Any]],
) -> list[list[Any]]:
    return sorted([
        [fold, seed, mechanism.name, episode_id]
        for fold in run_schedule.folds
        for seed in run_schedule.seeds
        for mechanism in MECHANISMS
        for episode_id in partitions[fold]["inner_calibration_ids"][
            :run_schedule.calibration_episodes_per_fold
        ]
    ])


def run(output: Path, *, smoke: bool, _inject_failure: str | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if not smoke and output.parent != (root / "artifacts").resolve():
        raise ValueError("formal A23 output must be a direct child of artifacts/")
    run_schedule = schedule(smoke)
    context = _preflight(root, smoke=smoke)
    partitions = {
        fold: partition_ids(context["assignments"], fold)
        for fold in run_schedule.folds
    }
    products = expected_products(run_schedule, all_feasible=True)
    training_contract = {
        "schema_version": "multitown-a23-frozen-training-contract-v1",
        "protocol_revision": "8c67613f86c044591aa51dba292608304cd244df",
        "runner_version": RUNNER_VERSION,
        "primitives_version": A23_PRIMITIVES_VERSION,
        "policy_version": POLICY_VERSION,
        "reference_policy_version": A9_POLICY_VERSION,
        "mechanisms": [asdict(item) for item in MECHANISMS],
        "schedule": asdict(run_schedule),
        "ppo": asdict(_formal_config(run_schedule)),
        "partitions": list(partitions.values()),
        "products_if_feasible": products,
        "new_actor_factor_only": "deterministic constraint-rectified advantage choice",
        "no_dual_state_or_cost_critic": True,
        "non_evidentiary_smoke": smoke,
    }
    contract = {
        "schema_version": "multitown-a23-run-contract-v1",
        "source": context["source"], "mode": run_schedule.mode,
        "bindings": context["bindings"],
        "pinned_a22": EXPECTED_A22,
        "pinned_a22_run_digest": EXPECTED_A22_RUN_DIGEST,
        "pinned_r2": EXPECTED_R2,
        "training_contract_sha256": _digest(training_contract),
    }
    contract_sha = _digest(contract)
    running_payload = {
        "schema_version": RUNNER_VERSION,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "outer_evaluation_started": False,
    }
    lock_acquired = False
    try:
        if not smoke:
            lock = root / FORMAL_ATTEMPT_LOCK
            descriptor = {
                "output": str(output),
                "source_revision": context["source"]["revision"],
                "run_contract_sha256": contract_sha,
            }
            _acquire_formal_lock(lock, descriptor)
            lock_acquired = True
        output.mkdir(parents=True)
        _atomic_json(output / "RUNNING.json", running_payload)
        _atomic_json(output / "training-contract.json", training_contract)
        _atomic_json(output / "run-contract.json", contract)
        torch.set_num_threads(run_schedule.threads)
        fits = []
        for fold in run_schedule.folds:
            partition = partitions[fold]
            train_episodes = [
                context["episode_index"][item] for item in partition["inner_train_ids"]
            ]
            train_a8 = [context["a8_index"][item] for item in partition["inner_train_ids"]]
            for seed in run_schedule.seeds:
                for mechanism in MECHANISMS:
                    fit_dir = _fit_path(output, fold, seed, mechanism.name)
                    fit_dir.mkdir(parents=True)
                    fits.append(_fit(
                        train_episodes, a8_rows=train_a8, mechanism=mechanism,
                        seed=seed, outer_fold=fold, run_schedule=run_schedule,
                        output=fit_dir, run_contract_sha256=contract_sha,
                        partition=partition,
                        expected_a22_sample_ids=context["a22_sample_sequences"][(fold, seed)][
                            :run_schedule.updates * run_schedule.episodes_per_update
                        ],
                    ))
        fit_index = _validate_fits(
            fits, output=output, run_schedule=run_schedule,
            partitions=partitions, run_contract_sha256=contract_sha,
            a22_sample_sequences=context["a22_sample_sequences"],
            a8_index=context["a8_index"],
        )
        all_fits_payload = {
            "schema_version": "multitown-a23-all-fits-complete-v1",
            "fits": fits, "expected_fits": products["fits"],
            "expected_training_episode_draws": products["training_episode_draws"],
            "all_reached_final_update": True,
            "exact_fit_key_product_verified": True,
            "training_log_checkpoint_initialization_and_sample_bindings_verified": True,
            "outer_evaluation_started": False,
        }
        _atomic_json(output / "all-fits-complete.json", all_fits_payload)

        calibration_rows = []
        for fold in run_schedule.folds:
            ids = partitions[fold]["inner_calibration_ids"][
                :run_schedule.calibration_episodes_per_fold
            ]
            episodes = [context["episode_index"][item] for item in ids]
            for seed in run_schedule.seeds:
                for mechanism in MECHANISMS:
                    fit = fit_index[(fold, seed, mechanism.name)]
                    model = _validate_checkpoint(
                        _fit_path(output, fold, seed, mechanism.name) / "final.pt",
                        fit=fit, mechanism=mechanism, fold=fold, seed=seed,
                        run_schedule=run_schedule, run_contract_sha256=contract_sha,
                    )
                    calibration_rows.extend(_evaluate(
                        model, episodes, mechanism=mechanism, training_seed=seed,
                        design_outer_fold=fold,
                        assignment_index=context["assignment_index"],
                        a8_index=context["a8_index"],
                        bank_sha256=context["bindings"]["train_bank"],
                        resource_sha256=context["bindings"]["resource"],
                        environment_sha256=context["bindings"]["environment"],
                        checkpoint_sha256=fit["checkpoint_sha256"],
                        run_contract_sha256=contract_sha, phase="inner-calibration",
                    ))
        _validate_evaluation_rows(
            calibration_rows, phase="inner-calibration",
            run_schedule=run_schedule, partitions=partitions,
            fit_index=fit_index, selections=None,
            selection_manifest_sha256=None,
            assignment_index=context["assignment_index"],
            a8_index=context["a8_index"],
            bank_sha256=context["bindings"]["train_bank"],
            resource_sha256=context["bindings"]["resource"],
            environment_sha256=context["bindings"]["environment"],
            run_contract_sha256=contract_sha,
        )
        _atomic_jsonl(output / "calibration-decisions.jsonl", calibration_rows)
        comparator_rows, comparator_binding = _a22_calibration_comparators(
            context["a22_root"], run_schedule=run_schedule,
            partitions=partitions,
            checkpoint_bindings=context["a22_checkpoint_bindings"],
        )
        selections = []
        candidates = {}
        for fold in run_schedule.folds:
            ids = partitions[fold]["inner_calibration_ids"][
                :run_schedule.calibration_episodes_per_fold
            ]
            selected, candidate = _select(
                [row for row in calibration_rows if int(row["design_outer_fold"]) == fold],
                fold=fold, seeds=run_schedule.seeds, episode_ids=ids,
                a8_rows=[context["a8_index"][item] for item in ids],
            )
            selections.append(selected)
            candidates[str(fold)] = candidate
        _revalidate_preflight(
            root, smoke=smoke,
            expected_signature=context["preflight_signature"],
        )
        selection_payload = {
            "schema_version": "multitown-a23-all-selections-frozen-v1",
            "run_contract_sha256": contract_sha,
            "scheduled_fit_keys": [list(key) for key in sorted(fit_index)],
            "fit_artifact_sha256": {
                f"{fold}:{seed}:{mechanism}": {
                    field: fit_index[(fold, seed, mechanism)][field]
                    for field in (
                        "checkpoint_sha256", "training_log_sha256",
                        "initial_model_sha256", "initial_optimizer_sha256",
                        "sample_sequence_sha256",
                    )
                } for fold, seed, mechanism in sorted(fit_index)
            },
            "calibration_rows": len(calibration_rows),
            "calibration_sha256": _sha256(output / "calibration-decisions.jsonl"),
            "calibration_key_product_entries": len(calibration_rows),
            "calibration_key_product_sha256": _digest(
                _calibration_key_product(run_schedule, partitions)
            ),
            "a22_comparator_binding": comparator_binding,
            "pinned_a22_artifacts": EXPECTED_A22,
            "pinned_a22_run_digest": EXPECTED_A22_RUN_DIGEST,
            "pinned_r2_and_a8": EXPECTED_R2,
            "a8_training_seed_field_verified_null": True,
            "method_table_rows": len(calibration_rows) + len(comparator_rows),
            "method_table": _method_table(calibration_rows, comparator_rows),
            "selections": selections, "candidate_summaries": candidates,
            "all_fits_and_calibrations_complete": True,
            "exact_fit_calibration_and_comparator_products_verified": True,
            "input_bindings_revalidated_before_outer_gate": True,
            "outer_evaluation_started": False,
            "frozen_at_utc": datetime.now(UTC).isoformat(),
        }
        _fault("before-selection-publish", _inject_failure)
        _atomic_json(output / "all-selections-frozen.json", selection_payload)
        _fault("after-selection-publish", _inject_failure)
        selection_sha = _sha256(output / "all-selections-frozen.json")
        all_feasible = all(row["selected_mechanism"] is not None for row in selections)
        outer_rows = []
        if all_feasible:
            _revalidate_preflight(
                root, smoke=smoke,
                expected_signature=context["preflight_signature"],
            )
            gate_payload = {
                "schema_version": "multitown-a23-outer-gate-v1",
                "selection_manifest_sha256": selection_sha,
                "all_five_selections_feasible": True,
                "opened_at_utc": datetime.now(UTC).isoformat(),
            }
            _fault("before-outer-gate-publish", _inject_failure)
            _atomic_json(output / "OUTER_GATE_OPEN.json", gate_payload)
            _fault("after-outer-gate-publish", _inject_failure)
            for selection in selections:
                fold = int(selection["outer_fold"])
                mechanism = next(
                    item for item in MECHANISMS
                    if item.name == selection["selected_mechanism"]
                )
                ids = partitions[fold]["outer_ids"][:run_schedule.outer_episodes_per_fold]
                episodes = [context["episode_index"][item] for item in ids]
                for seed in run_schedule.seeds:
                    fit = fit_index[(fold, seed, mechanism.name)]
                    model = _validate_checkpoint(
                        _fit_path(output, fold, seed, mechanism.name) / "final.pt",
                        fit=fit, mechanism=mechanism, fold=fold, seed=seed,
                        run_schedule=run_schedule, run_contract_sha256=contract_sha,
                    )
                    outer_rows.extend(_evaluate(
                        model, episodes, mechanism=mechanism, training_seed=seed,
                        design_outer_fold=fold,
                        assignment_index=context["assignment_index"],
                        a8_index=context["a8_index"],
                        bank_sha256=context["bindings"]["train_bank"],
                        resource_sha256=context["bindings"]["resource"],
                        environment_sha256=context["bindings"]["environment"],
                        checkpoint_sha256=fit["checkpoint_sha256"],
                        run_contract_sha256=contract_sha,
                        phase="selected-outer",
                        selection_manifest_sha256=selection_sha,
                    ))
            _validate_evaluation_rows(
                outer_rows, phase="selected-outer", run_schedule=run_schedule,
                partitions=partitions, fit_index=fit_index,
                selections=selections, selection_manifest_sha256=selection_sha,
                assignment_index=context["assignment_index"],
                a8_index=context["a8_index"],
                bank_sha256=context["bindings"]["train_bank"],
                resource_sha256=context["bindings"]["resource"],
                environment_sha256=context["bindings"]["environment"],
                run_contract_sha256=contract_sha,
            )
            _atomic_jsonl(output / "outer-decisions.jsonl", outer_rows)
        else:
            if (output / "OUTER_GATE_OPEN.json").exists() or (
                output / "outer-decisions.jsonl"
            ).exists():
                raise RuntimeError("A23 infeasible selection produced outer artifacts")

        statistics = None
        if all_feasible:
            selected_a8 = []
            for fold in run_schedule.folds:
                selected_a8.extend(
                    context["a8_index"][item]
                    for item in partitions[fold]["outer_ids"][
                        :run_schedule.outer_episodes_per_fold
                    ]
                )
            a22_outer = _a22_selected_outer(
                context, run_schedule=run_schedule, partitions=partitions,
            )
            statistics = result_statistics(
                selected_a8, outer_rows, a22_outer,
                _statistics_schedule(run_schedule), gate_evaluable=not smoke,
            )
        fit_index = _validate_fits(
            fits, output=output, run_schedule=run_schedule,
            partitions=partitions, run_contract_sha256=contract_sha,
            a22_sample_sequences=context["a22_sample_sequences"],
            a8_index=context["a8_index"],
        )
        root_publications = {
            "RUNNING.json": running_payload,
            "training-contract.json": training_contract,
            "run-contract.json": contract,
            "all-fits-complete.json": all_fits_payload,
            "all-selections-frozen.json": selection_payload,
        }
        if any(
            _digest(_read_json(output / name)) != _digest(payload)
            for name, payload in root_publications.items()
        ):
            raise RuntimeError("A23 published root artifact changed before result")
        if (
            _sha256(output / "all-selections-frozen.json") != selection_sha
            or _sha256(output / "calibration-decisions.jsonl")
            != selection_payload["calibration_sha256"]
        ):
            raise RuntimeError("A23 published byte-hash binding changed before result")
        if _digest(_read_jsonl(output / "calibration-decisions.jsonl")) != _digest(calibration_rows):
            raise RuntimeError("A23 persisted calibration rows changed")
        if all_feasible and _read_json(output / "OUTER_GATE_OPEN.json") != gate_payload:
            raise RuntimeError("A23 persisted outer gate changed")
        if all_feasible and _digest(_read_jsonl(output / "outer-decisions.jsonl")) != _digest(outer_rows):
            raise RuntimeError("A23 persisted outer rows changed")
        _revalidate_preflight(
            root, smoke=smoke,
            expected_signature=context["preflight_signature"],
        )
        actual_products = expected_products(run_schedule, all_feasible=all_feasible)
        result = {
            "schema_version": RESULT_VERSION, "mode": run_schedule.mode,
            "evidence_scope": (
                "non-evidentiary implementation smoke"
                if smoke else "adaptive same-bank A23 development"
            ),
            "non_evidentiary_smoke": smoke,
            "formal_selection_outcome_evaluable": not smoke,
            "formal_selection_negative": bool(not smoke and not all_feasible),
            "system_recovery_gate_evaluable": bool(not smoke and all_feasible),
            "utility_replacement_gate_evaluable": bool(
                statistics and statistics["utility_replacement_gate_evaluable"]
            ),
            "system_recovery_gate_passed": bool(
                statistics and statistics["system_recovery_gate_passed"]
            ),
            "utility_replacement_criterion_passed": bool(
                statistics and statistics["utility_replacement_criterion_passed"]
            ),
            "source_revision": context["source"]["revision"],
            "run_contract_sha256": contract_sha,
            "selection_manifest_sha256": selection_sha,
            "all_folds_feasible": all_feasible,
            "products": actual_products,
            "selections": selections, "statistics": statistics,
            "training_mode_diagnostics_by_fold_seed_mechanism": (
                _training_mode_diagnostics(fits)
            ),
            "validation": {
                "exact_fit_key_product": True,
                "training_logs_checkpoints_initialization_and_samples": True,
                "exact_calibration_and_comparator_products": True,
                "global_selection_frozen_before_outer": True,
                "exact_outer_key_product": True,
                "inputs_revalidated_before_selection_outer_and_result": True,
            },
            "claim_boundary": {
                "original_crpo_reproduction": False,
                "inherited_crpo_guarantees": False,
                "confirmatory_noninferiority": False,
                "independent_replication": False,
                "hidden_test_or_ood": False,
                "llm_weight_rl": False,
                "state_of_the_art": False,
            },
        }
        _atomic_json(output / "result.json", result)
        (output / "RUNNING.json").unlink()
        manifest = _manifest(
            output, source_revision=context["source"]["revision"],
            run_schedule=run_schedule, all_feasible=all_feasible,
        )
        _atomic_json(output / "artifact-manifest.json", manifest)
        validate_manifest(
            output, expected_entries=actual_products["manifest_entries"],
        )
        print(json.dumps({
            "output": str(output), "mode": run_schedule.mode,
            **actual_products, "all_folds_feasible": all_feasible,
            "selections": [row["selected_mechanism"] for row in selections],
            "system_recovery_gate_passed": result["system_recovery_gate_passed"],
            "utility_replacement_criterion_passed": result[
                "utility_replacement_criterion_passed"
            ],
        }, ensure_ascii=False, indent=2))
        return 0
    except BaseException as exc:
        if isinstance(exc, _FormalLockCreatedError):
            lock_acquired = True
        elif not smoke and not lock_acquired:
            raise
        if not output.exists():
            try:
                output.mkdir(parents=True)
            except Exception:
                pass
        _isolate_valid_outputs(output)
        if not (output / "RUNNING.json").exists():
            try:
                _atomic_json(output / "RUNNING.json", running_payload)
            except Exception:
                pass
        failure = {
            "schema_version": "multitown-a23-invalidated-attempt-v1",
            "invalidated": True, "error_type": type(exc).__name__,
            "error": str(exc), "traceback": traceback.format_exc(),
            "selective_retry_forbidden": True,
            "formal_lock_acquired": lock_acquired,
            "failed_at_utc": datetime.now(UTC).isoformat(),
        }
        try:
            _atomic_json(output / "INVALIDATED.json", failure)
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
