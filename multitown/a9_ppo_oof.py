"""Run the frozen train-only A9-v2 PPO out-of-fold experiment."""

from __future__ import annotations

import argparse
import hashlib
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
from typing import Any, Mapping, Protocol, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .a9_oof_protocol import (
    DEFAULT_FOLDS, EXPECTED_TRAIN_EPISODES, FOLD_VERSION, FROZEN_TRAIN_PATH,
    PROTOCOL_VERSION, assign_stratified_group_folds, fold_manifest_sha256,
    load_frozen_train_bank, resource_contract_sha256,
    shared_resource_contract, validate_fold_assignments,
)
from .long_horizon_env import (
    ACTION_COUNT, LongHorizonEpisode, MultiTownLongHorizonEnv,
    RLAction, summarize_results,
)
from .ppo_controller import (
    ActorCritic, PPOConfig, _advantages, _episode_rollout, _ppo_update,
    _save_checkpoint, _set_seed, exact_mcnemar_p, load_checkpoint,
)


RUNNER_VERSION = "multitown-a9-v2-train-only-ppo-oof-runner-v2-recovery"
POLICY_VERSION = "multitown-a9-v2-masked-ppo-policy-v1"
RESULT_VERSION = "multitown-a9-v2-train-only-ppo-oof-result-v2-recovery"
BOOTSTRAP_VERSION = "multitown-a9-fold-stratified-episode-cluster-percentile-v1"
MCNEMAR_STATISTIC_VERSION = "exact-binomial-two-sided-stable-fraction-v1"
FORMAL_SEEDS = (20260812, 20260813, 20260814)
FORMAL_UPDATES = 120
FORMAL_EPISODES_PER_UPDATE = 48
FORMAL_THREADS = 8
BOOTSTRAP_ITERATIONS = 20_000
BOOTSTRAP_SEED = 20260813
FORMAL_ATTEMPT_LOCK = Path("artifacts/a9-v2-formal-attempt-v2.lock")
INVALIDATED_PREDECESSOR_LOCK = Path("artifacts/a9-v2-formal-attempt.lock")
INVALIDATED_PREDECESSOR_OUTPUT = Path("artifacts/a9-v2-ppo-oof-20260813")
INVALIDATED_PREDECESSOR_LOCK_SHA256 = (
    "b6a18491126c386bfa723252eae42d3983280e02dd19aceeeac1735c38a1c051"
)
INVALIDATED_PREDECESSOR_FAILURE_SHA256 = (
    "de0d1c50b217f1654b3f69345814f154c5e38bf80c78a6c0f4cdf2f72468c96d"
)
INVALIDATED_PREDECESSOR_ALL_FITS_SHA256 = (
    "382f2c692ffe74758ece9dbaf45dc9cc7ad6bf47134e5ff3de35d643c5b4a51c"
)


@dataclass(frozen=True)
class A8PublicView:
    """The exact A8 runtime fields, excluding episode specs and private labels."""

    incident_active: bool
    last_tool_failed: bool
    last_action: int | None
    observed: bool
    current_candidate: int | None
    review_state: int
    weak_candidate: int | None
    connected: bool
    strong_candidate: int | None


@dataclass(frozen=True)
class RunSchedule:
    mode: str
    seeds: tuple[int, ...]
    folds: tuple[int, ...]
    updates: int
    episodes_per_update: int
    evaluation_episodes_per_fold: int
    bootstrap_iterations: int
    bootstrap_seed: int
    threads: int


def _canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, payload: str) -> None:
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def _write_json(path: Path, value: Any) -> None:
    _atomic_text(
        path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _atomic_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            for row in rows
        ),
    )


def _git_source_state(root: Path, *, require_clean: bool) -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=root, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    if require_clean and status:
        raise RuntimeError("formal A9 OOF requires a clean source checkout")
    listed = subprocess.run(
        [
            "git", "ls-files", "--", "multitown", "pyproject.toml",
            "docs/A21_A9_TRAIN_ONLY_PPO_OOF.md",
        ], cwd=root, text=True,
        capture_output=True, check=True,
    ).stdout.splitlines()
    sources = sorted(
        path for path in listed
        if path.endswith(".py") or path in {
            "pyproject.toml", "docs/A21_A9_TRAIN_ONLY_PPO_OOF.md",
        }
    )
    hashes: dict[str, str] = {}
    for relative in sources:
        disk = (root / relative).read_bytes()
        head = subprocess.run(
            ["git", "show", f"HEAD:{relative}"], cwd=root,
            capture_output=True, check=True,
        ).stdout
        if require_clean and disk != head:
            raise RuntimeError(f"executed source differs from HEAD: {relative}")
        hashes[relative] = hashlib.sha256(disk).hexdigest()
    required = {
        "multitown/a9_ppo_oof.py",
        "multitown/a9_oof_protocol.py",
        "multitown/a9_long_horizon_env.py",
        "multitown/long_horizon_env.py",
        "multitown/ppo_controller.py",
        "pyproject.toml",
        "docs/A21_A9_TRAIN_ONLY_PPO_OOF.md",
    }
    if not require_clean:
        for relative in sorted(required - set(hashes)):
            path = root / relative
            if path.is_file():
                hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not required <= set(hashes):
        raise RuntimeError("formal runner sources are not all tracked")
    return {
        "revision": revision,
        "tree": tree,
        "dirty": bool(status),
        "tracked_source_sha256": hashes,
    }


def _formal_ppo_config() -> PPOConfig:
    return PPOConfig(
        updates=FORMAL_UPDATES,
        episodes_per_update=FORMAL_EPISODES_PER_UPDATE,
        hidden_size=128,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        ppo_epochs=4,
        minibatch_size=512,
        value_coef=0.5,
        entropy_coef=0.02,
        max_grad_norm=0.5,
        dev_interval=0,
    )


def _schedule(smoke: bool) -> RunSchedule:
    if smoke:
        return RunSchedule(
            mode="smoke", seeds=(FORMAL_SEEDS[0],), folds=tuple(range(DEFAULT_FOLDS)),
            updates=1, episodes_per_update=4, evaluation_episodes_per_fold=8,
            bootstrap_iterations=500, bootstrap_seed=BOOTSTRAP_SEED, threads=2,
        )
    return RunSchedule(
        mode="formal", seeds=FORMAL_SEEDS, folds=tuple(range(DEFAULT_FOLDS)),
        updates=FORMAL_UPDATES, episodes_per_update=FORMAL_EPISODES_PER_UPDATE,
        evaluation_episodes_per_fold=EXPECTED_TRAIN_EPISODES // DEFAULT_FOLDS,
        bootstrap_iterations=BOOTSTRAP_ITERATIONS,
        bootstrap_seed=BOOTSTRAP_SEED, threads=FORMAL_THREADS,
    )


def canonical_training_contract(schedule: RunSchedule) -> dict[str, Any]:
    ppo = asdict(_formal_ppo_config())
    ppo["updates"] = schedule.updates
    ppo["episodes_per_update"] = schedule.episodes_per_update
    return {
        "schema_version": "multitown-a9-v2-canonical-training-contract-v2-recovery",
        "runner_version": RUNNER_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "policy_version": POLICY_VERSION,
        "algorithm": "online on-policy clipped PPO with masked discrete actions and GAE",
        "scope": "central organization controller; fixed environment workers and tools",
        "network": {
            "input_size": MultiTownLongHorizonEnv.observation_size,
            "hidden_layers": [128, 128],
            "activation": "tanh",
            "actor_outputs": ACTION_COUNT,
            "critic_outputs": 1,
            "action_mask": "invalid logits replaced by torch dtype minimum",
        },
        "ppo_config": ppo,
        "unused_ppo_config_fields": ["dev_interval"],
        "optimizer": {
            "name": "torch.optim.Adam", "lr": 3e-4,
            "betas": [0.9, 0.999], "eps": 1e-5, "weight_decay": 0.0,
            "amsgrad": False, "maximize": False, "foreach": None,
            "capturable": False, "differentiable": False, "fused": None,
        },
        "advantage": {
            "normalization": "global sampled-transition mean/std",
            "normalization_epsilon": 1e-8,
            "terminal_bootstrap_value": 0.0,
        },
        "initialization": {
            "linear": "orthogonal",
            "hidden_gain": math.sqrt(2),
            "actor_gain": 0.01,
            "critic_gain": 1.0,
            "actor_bias": {"stop": -1.5, "human": -0.5},
        },
        "sampler": {
            "episode_order": "episode_id ascending",
            "replacement": True,
            "rng": "python random.Random(training_seed)",
            "outer_training_episodes": 2400,
            "episodes_per_update": schedule.episodes_per_update,
        },
        "evaluation": {
            "phase": "after every scheduled fit reaches final update",
            "action": "deterministic masked argmax",
            "checkpoint": "final update only",
            "early_stopping": False,
            "outer_fold_selection": False,
            "retry_or_seed_replacement": False,
        },
        "randomness": {
            "fit_seeds": list(schedule.seeds),
            "python_numpy_torch_cuda_seeded": True,
            "torch_deterministic_algorithms": True,
            "warn_only": True,
        },
        "compute": {
            "device": "cpu",
            "torch_threads": schedule.threads,
            "fixed_transition_budget_claimed": False,
            "fixed_optimizer_step_budget_claimed": False,
        },
        "reporting_statistics": {
            "mcnemar_exploratory": MCNEMAR_STATISTIC_VERSION,
            "mcnemar_primary": False,
        },
        "schedule": asdict(schedule),
    }


def a8_public_view(env: MultiTownLongHorizonEnv) -> A8PublicView:
    return A8PublicView(
        incident_active=env.incident is not None,
        last_tool_failed=bool(env.last_tool_failed),
        last_action=env.last_action,
        observed=bool(env.observed),
        current_candidate=env.current_candidate,
        review_state=int(env.review_state),
        weak_candidate=env.weak_candidate,
        connected=bool(env.connected),
        strong_candidate=env.strong_candidate,
    )


def a8_public_action(view: A8PublicView, mask: np.ndarray) -> int:
    if not view.incident_active:
        return int(RLAction.STOP)
    if (
        view.last_tool_failed and view.last_action is not None
        and mask[view.last_action]
    ):
        return int(view.last_action)
    if not view.observed and mask[RLAction.OBSERVE]:
        return int(RLAction.OBSERVE)
    if (
        view.current_candidate is not None and view.review_state == 0
        and mask[RLAction.REVIEW]
    ):
        return int(RLAction.REVIEW)
    if view.review_state == 1 and mask[RLAction.EXECUTE]:
        return int(RLAction.EXECUTE)
    if view.weak_candidate is None and mask[RLAction.DELEGATE]:
        return int(RLAction.DELEGATE)
    if not view.connected and mask[RLAction.CONNECT]:
        return int(RLAction.CONNECT)
    if view.strong_candidate is None and mask[RLAction.ESCALATE]:
        return int(RLAction.ESCALATE)
    if not view.connected and mask[RLAction.CONNECT]:
        return int(RLAction.CONNECT)
    if view.review_state == 0 and mask[RLAction.REVIEW]:
        return int(RLAction.REVIEW)
    if mask[RLAction.HUMAN]:
        return int(RLAction.HUMAN)
    return int(RLAction.STOP)


class ActionFunction(Protocol):
    def __call__(self, observation: np.ndarray, mask: np.ndarray) -> int: ...


def _validate_policy_inputs(observation: np.ndarray, mask: np.ndarray) -> None:
    if (
        observation.shape != (MultiTownLongHorizonEnv.observation_size,)
        or observation.dtype != np.float32
        or not np.isfinite(observation).all()
    ):
        raise ValueError("invalid public observation tensor")
    if mask.shape != (ACTION_COUNT,) or mask.dtype != np.bool_ or not mask.any():
        raise ValueError("invalid public action-mask tensor")


def _model_action(model: ActorCritic, observation: np.ndarray, mask: np.ndarray) -> int:
    """Select an action from only public observation and action-mask tensors."""

    _validate_policy_inputs(observation, mask)
    obs = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
    valid = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)
    with torch.no_grad():
        logits, _ = model(obs)
        action = logits.masked_fill(
            ~valid, torch.finfo(logits.dtype).min,
        ).argmax(dim=-1)
    return int(action.item())


def _trajectory_safety(
    trajectory: Sequence[Mapping[str, Any]], *, assisted_episode_success: bool,
) -> dict[str, Any]:
    wrong = [
        step for step in trajectory
        if step["action"] == "execute"
        and float(step["reward"]["safety_penalty"]) < 0.0
    ]
    return {
        "wrong_executions": len(wrong),
        "had_wrong_execution": bool(wrong),
        "safety_penalty_burden": -sum(
            min(0.0, float(step["reward"]["safety_penalty"]))
            for step in trajectory
        ),
        "failed_stop_episode": bool(
            trajectory and trajectory[-1]["action"] == "stop"
            and not assisted_episode_success
        ),
        "truncated_episode": any(bool(step["truncated"]) for step in trajectory),
    }


def _evaluate_episode(
    episode: LongHorizonEpisode, action_function: ActionFunction, *,
    fold: int, system: str, training_seed: int | None,
    episode_sha256: str, bank_sha256: str, resource_sha256: str,
    environment_sha256: str, checkpoint_sha256: str,
    a8_row_sha256: str, run_contract_sha256: str,
) -> dict[str, Any]:
    env = MultiTownLongHorizonEnv(episode)
    observation, _ = env.reset()
    total_return = 0.0
    while not env.terminated:
        mask = env.action_mask()
        public_observation = observation.copy()
        public_mask = mask.copy()
        public_observation.flags.writeable = False
        public_mask.flags.writeable = False
        _validate_policy_inputs(public_observation, public_mask)
        action = int(action_function(public_observation, public_mask))
        observation, reward, _, _, _ = env.step(action)
        total_return += reward
    info = env.info()
    return {
        **info,
        "return": total_return,
        "trajectory": env.trajectory,
        **_trajectory_safety(
            env.trajectory,
            assisted_episode_success=bool(info["assisted_episode_success"]),
        ),
        "system": system,
        "training_seed": training_seed,
        "outer_fold": fold,
        "episode_sha256": episode_sha256,
        "train_bank_sha256": bank_sha256,
        "resource_contract_sha256": resource_sha256,
        "environment_source_sha256": environment_sha256,
        "final_checkpoint_sha256": checkpoint_sha256,
        "a8_row_sha256": a8_row_sha256,
        "run_contract_sha256": run_contract_sha256,
        "environment_class": "multitown.long_horizon_env.MultiTownLongHorizonEnv",
        "token_cap": episode.token_budget,
        "latency_cap_s": episode.latency_budget_s,
        "policy_input_contract": "observation+action_mask-only",
    }


def _evaluate_a8(
    episode: LongHorizonEpisode, *, fold: int, episode_sha256: str,
    bank_sha256: str, resource_sha256: str,
    environment_sha256: str, run_contract_sha256: str = "unbound-test-row",
) -> dict[str, Any]:
    env_holder: dict[str, MultiTownLongHorizonEnv] = {}

    def action(observation: np.ndarray, mask: np.ndarray) -> int:
        del observation
        return a8_public_action(a8_public_view(env_holder["env"]), mask)

    env = MultiTownLongHorizonEnv(episode)
    env_holder["env"] = env
    observation, _ = env.reset()
    total_return = 0.0
    while not env.terminated:
        mask = env.action_mask()
        public_observation = observation.copy()
        public_mask = mask.copy()
        public_observation.flags.writeable = False
        public_mask.flags.writeable = False
        _validate_policy_inputs(public_observation, public_mask)
        selected = int(action(public_observation, public_mask))
        observation, reward, _, _, _ = env.step(selected)
        total_return += reward
    info = env.info()
    return {
        **info, "return": total_return, "trajectory": env.trajectory,
        **_trajectory_safety(
            env.trajectory,
            assisted_episode_success=bool(info["assisted_episode_success"]),
        ),
        "system": "A8-long-public-view", "training_seed": None,
        "outer_fold": fold, "episode_sha256": episode_sha256,
        "train_bank_sha256": bank_sha256,
        "resource_contract_sha256": resource_sha256,
        "run_contract_sha256": run_contract_sha256,
        "environment_source_sha256": environment_sha256,
        "environment_class": "multitown.long_horizon_env.MultiTownLongHorizonEnv",
        "token_cap": episode.token_budget,
        "latency_cap_s": episode.latency_budget_s,
        "policy_input_contract": "A8PublicView+action_mask-only",
    }


def _training_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    prepared = [{**row, "trajectory": []} for row in rows]
    return summarize_results(prepared)


def _train_fit(
    train_episodes: Sequence[LongHorizonEpisode], *, seed: int, fold: int,
    held_episode_ids: Sequence[str], schedule: RunSchedule, output: Path,
    run_contract_sha256: str,
) -> dict[str, Any]:
    if len(train_episodes) != 2400:
        raise ValueError("every outer fit requires exactly 2400 training episodes")
    if tuple(sorted(item.episode_id for item in train_episodes)) != tuple(
        item.episode_id for item in train_episodes
    ):
        raise ValueError("outer training episodes must be canonically ordered")
    allowed_ids = {item.episode_id for item in train_episodes}
    held_ids = set(held_episode_ids)
    if allowed_ids & held_ids or len(held_ids) != 600:
        raise ValueError("outer train and held episode IDs must be disjoint")
    config = _formal_ppo_config()
    config = PPOConfig(
        **{
            **asdict(config),
            "updates": schedule.updates,
            "episodes_per_update": schedule.episodes_per_update,
        }
    )
    _set_seed(seed)
    model = ActorCritic(
        MultiTownLongHorizonEnv.observation_size, config.hidden_size, ACTION_COUNT,
    ).cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)
    sample_rng = random.Random(seed)
    tensor_generator = torch.Generator(device="cpu").manual_seed(seed)
    logs: list[dict[str, Any]] = []
    total_environment_steps = 0
    total_optimizer_minibatches = 0
    complete_sample_sequence: list[str] = []
    started = time.perf_counter()
    for update in range(1, config.updates + 1):
        transitions: list[list[dict[str, Any]]] = []
        rollout_rows: list[dict[str, Any]] = []
        sampled_episode_ids: list[str] = []
        for _ in range(config.episodes_per_update):
            episode = train_episodes[sample_rng.randrange(len(train_episodes))]
            sampled_episode_ids.append(episode.episode_id)
            complete_sample_sequence.append(episode.episode_id)
            episode_transitions, metrics = _episode_rollout(
                model, episode, torch.device("cpu"),
            )
            transitions.append(episode_transitions)
            rollout_rows.append(metrics)
        batch = _advantages(transitions, config)
        metrics = _ppo_update(
            model, optimizer, batch, config, torch.device("cpu"), tensor_generator,
        )
        environment_steps = sum(len(item) for item in transitions)
        optimizer_minibatches = config.ppo_epochs * math.ceil(
            len(batch["action"]) / config.minibatch_size
        )
        total_environment_steps += environment_steps
        total_optimizer_minibatches += optimizer_minibatches
        summary = _training_summary(rollout_rows)
        logs.append({
            "outer_fold": fold,
            "training_seed": seed,
            "update": update,
            "environment_steps": environment_steps,
            "optimizer_minibatches": optimizer_minibatches,
            "sampled_episode_ids": sampled_episode_ids,
            "rollout_episode_success_rate": summary["episode_success_rate"],
            "rollout_subgoal_completion_rate": summary["subgoal_completion_rate"],
            "rollout_mean_return": summary["mean_return"],
            "rollout_tokens_per_episode": summary["tokens_per_episode"],
            **metrics,
        })
        _write_json(output / "progress.json", {
            "outer_fold": fold,
            "training_seed": seed,
            "current_update": update,
            "scheduled_updates": config.updates,
            "environment_steps_so_far": total_environment_steps,
            "optimizer_minibatches_so_far": total_optimizer_minibatches,
            "latest_rollout_episode_success_rate": summary["episode_success_rate"],
            "oof_evaluation_started": False,
        })
    checkpoint = output / "final.pt"
    _save_checkpoint(
        checkpoint, model, config, seed=seed, update=config.updates,
        policy_version=POLICY_VERSION,
    )
    checkpoint_payload = torch.load(
        checkpoint, map_location="cpu", weights_only=False,
    )
    checkpoint_payload["run_contract_sha256"] = run_contract_sha256
    checkpoint_partial = checkpoint.with_name(checkpoint.name + ".partial")
    torch.save(checkpoint_payload, checkpoint_partial)
    os.replace(checkpoint_partial, checkpoint)
    _write_jsonl(output / "training-metrics.jsonl", logs)
    sampled_held_overlap = len(set(complete_sample_sequence) & held_ids)
    if sampled_held_overlap or not set(complete_sample_sequence) <= allowed_ids:
        raise RuntimeError("outer-held episode entered the training sample stream")
    result = {
        "outer_fold": fold,
        "training_seed": seed,
        "training_episodes": len(train_episodes),
        "final_update": config.updates,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "environment_steps": total_environment_steps,
        "optimizer_minibatches": total_optimizer_minibatches,
        "sampled_episode_draws": len(complete_sample_sequence),
        "sampled_unique_train_episodes": len(set(complete_sample_sequence)),
        "sample_sequence_sha256": _digest(complete_sample_sequence),
        "sampled_held_overlap": sampled_held_overlap,
        "outer_train_episode_ids_sha256": _digest(sorted(allowed_ids)),
        "outer_held_episode_ids_sha256": _digest(sorted(held_ids)),
        "run_contract_sha256": run_contract_sha256,
        "training_seconds": time.perf_counter() - started,
        "outer_evaluations_during_training": 0,
        "selected_checkpoint": "final",
    }
    _write_json(output / "fit-complete.json", result)
    return result


def _metric_value(row: Mapping[str, Any], metric: str) -> float:
    if metric == "episode_success":
        return float(bool(row["episode_success"]))
    if metric == "assisted_episode_success":
        return float(bool(row["assisted_episode_success"]))
    if metric == "wrong_execution":
        return float(bool(row["had_wrong_execution"]))
    return float(row[metric])


def fold_cluster_bootstrap(
    a8_rows: Sequence[Mapping[str, Any]],
    a9_rows: Sequence[Mapping[str, Any]], *, metric: str,
    folds: int = DEFAULT_FOLDS, seeds: Sequence[int] = FORMAL_SEEDS,
    iterations: int = BOOTSTRAP_ITERATIONS, seed: int = BOOTSTRAP_SEED,
    ratio_reduction: bool = False,
) -> dict[str, Any]:
    if type(iterations) is not int or iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    seed_tuple = tuple(seeds)
    a8_index = {str(row["episode_id"]): row for row in a8_rows}
    a9_index = {
        (str(row["episode_id"]), int(row["training_seed"])): row
        for row in a9_rows
    }
    if len(a8_index) != len(a8_rows) or len(a9_index) != len(a9_rows):
        raise ValueError("duplicate paired bootstrap rows")
    by_fold: dict[int, list[str]] = defaultdict(list)
    for episode_id, row in a8_index.items():
        by_fold[int(row["outer_fold"])].append(episode_id)
    if set(by_fold) != set(range(folds)):
        raise ValueError("bootstrap fold coverage mismatch")
    arrays: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for fold in range(folds):
        ids = sorted(by_fold[fold])
        left = np.asarray([
            np.mean([
                _metric_value(a9_index[(episode_id, training_seed)], metric)
                for training_seed in seed_tuple
            ])
            for episode_id in ids
        ], dtype=np.float64)
        right = np.asarray([
            _metric_value(a8_index[episode_id], metric) for episode_id in ids
        ], dtype=np.float64)
        arrays[fold] = left, right
    left_point = float(np.mean([left.mean() for left, _ in arrays.values()]))
    right_point = float(np.mean([right.mean() for _, right in arrays.values()]))
    point = (
        1.0 - left_point / right_point
        if ratio_reduction and right_point != 0.0 else left_point - right_point
    )
    rng = np.random.default_rng(seed)
    values = np.empty(iterations, dtype=np.float64)
    for start in range(0, iterations, 256):
        width = min(256, iterations - start)
        left_replicates = np.zeros(width, dtype=np.float64)
        right_replicates = np.zeros(width, dtype=np.float64)
        for fold in range(folds):
            left, right = arrays[fold]
            indices = rng.integers(0, len(left), size=(width, len(left)))
            left_replicates += left[indices].mean(axis=1) / folds
            right_replicates += right[indices].mean(axis=1) / folds
        if ratio_reduction:
            if np.any(right_replicates == 0.0):
                raise ValueError("undefined bootstrap ratio denominator")
            values[start : start + width] = 1.0 - left_replicates / right_replicates
        else:
            values[start : start + width] = left_replicates - right_replicates
    return {
        "schema_version": BOOTSTRAP_VERSION,
        "metric": metric,
        "estimand": "ratio_reduction" if ratio_reduction else "A9_minus_A8",
        "point": point,
        "a9_mean": left_point,
        "a8_mean": right_point,
        "ci95_low": float(np.quantile(values, 0.025, method="linear")),
        "ci95_high": float(np.quantile(values, 0.975, method="linear")),
        "iterations": iterations,
        "rng_seed": seed,
        "percentile_interval": [0.025, 0.975],
        "numpy_quantile_method": "linear",
        "fold_resampling": {
            str(fold): len(arrays[fold][0]) for fold in range(folds)
        },
        "training_seeds_fixed_not_resampled": list(seed_tuple),
    }


def fold_cluster_ratio_bootstrap(
    a8_rows: Sequence[Mapping[str, Any]],
    a9_rows: Sequence[Mapping[str, Any]], *, numerator: str, denominator: str,
    seeds: Sequence[int], iterations: int, seed: int,
) -> dict[str, Any]:
    """Compare equal-fold means of within-fold aggregate ratios."""

    if type(iterations) is not int or iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    seed_tuple = tuple(seeds)
    a8_index = {str(row["episode_id"]): row for row in a8_rows}
    a9_index = {
        (str(row["episode_id"]), int(row["training_seed"])): row
        for row in a9_rows
    }
    if len(a8_index) != len(a8_rows) or len(a9_index) != len(a9_rows):
        raise ValueError("duplicate ratio-bootstrap rows")
    expected_a9_keys = {
        (episode_id, training_seed)
        for episode_id in a8_index for training_seed in seed_tuple
    }
    if set(a9_index) != expected_a9_keys:
        raise ValueError("ratio bootstrap is not a full episode x seed product")
    by_fold: dict[int, list[str]] = defaultdict(list)
    for episode_id, row in a8_index.items():
        by_fold[int(row["outer_fold"])].append(episode_id)
    if set(by_fold) != set(range(DEFAULT_FOLDS)):
        raise ValueError("ratio bootstrap fold coverage mismatch")
    arrays: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for fold in range(DEFAULT_FOLDS):
        ids = sorted(by_fold[fold])
        a9_numerator = np.asarray([
            np.mean([
                float(a9_index[(episode_id, training_seed)][numerator])
                for training_seed in seed_tuple
            ]) for episode_id in ids
        ])
        a9_denominator = np.asarray([
            np.mean([
                float(a9_index[(episode_id, training_seed)][denominator])
                for training_seed in seed_tuple
            ]) for episode_id in ids
        ])
        a8_numerator = np.asarray([
            float(a8_index[episode_id][numerator]) for episode_id in ids
        ])
        a8_denominator = np.asarray([
            float(a8_index[episode_id][denominator]) for episode_id in ids
        ])
        arrays[fold] = (
            a9_numerator, a9_denominator, a8_numerator, a8_denominator,
        )
    if any(
        array[1].sum() == 0.0 or array[3].sum() == 0.0
        for array in arrays.values()
    ):
        raise ValueError("ratio point denominator is zero")
    a9_point = float(np.mean([
        array[0].sum() / array[1].sum() for array in arrays.values()
    ]))
    a8_point = float(np.mean([
        array[2].sum() / array[3].sum() for array in arrays.values()
    ]))
    rng = np.random.default_rng(seed)
    values = np.empty(iterations, dtype=np.float64)
    for start in range(0, iterations, 256):
        width = min(256, iterations - start)
        differences = np.zeros(width, dtype=np.float64)
        for fold in range(DEFAULT_FOLDS):
            fold_arrays = arrays[fold]
            indices = rng.integers(
                0, len(fold_arrays[0]), size=(width, len(fold_arrays[0])),
            )
            sampled = [array[indices].sum(axis=1) for array in fold_arrays]
            if np.any(sampled[1] == 0.0) or np.any(sampled[3] == 0.0):
                raise ValueError("ratio bootstrap denominator is zero")
            differences += (
                sampled[0] / sampled[1] - sampled[2] / sampled[3]
            ) / DEFAULT_FOLDS
        values[start : start + width] = differences
    return {
        "schema_version": BOOTSTRAP_VERSION,
        "metric": f"sum({numerator})/sum({denominator})",
        "estimand": "A9_minus_A8",
        "point": float(a9_point - a8_point),
        "a9_ratio": float(a9_point),
        "a8_ratio": float(a8_point),
        "ci95_low": float(np.quantile(values, 0.025, method="linear")),
        "ci95_high": float(np.quantile(values, 0.975, method="linear")),
        "iterations": iterations,
        "rng_seed": seed,
        "percentile_interval": [0.025, 0.975],
        "numpy_quantile_method": "linear",
        "fold_weighting": "equal; ratio recomputed within every fold replicate",
        "fold_resampling": {
            str(fold): len(arrays[fold][0]) for fold in range(DEFAULT_FOLDS)
        },
        "training_seeds_fixed_not_resampled": list(seed_tuple),
    }


def _paired_effects(
    a8_rows: Sequence[Mapping[str, Any]], a9_rows: Sequence[Mapping[str, Any]],
    schedule: RunSchedule,
) -> dict[str, Any]:
    primary = fold_cluster_bootstrap(
        a8_rows, a9_rows, metric="episode_success", seeds=schedule.seeds,
        iterations=schedule.bootstrap_iterations, seed=schedule.bootstrap_seed,
    )
    secondary = {
        metric: fold_cluster_bootstrap(
            a8_rows, a9_rows, metric=metric, seeds=schedule.seeds,
            iterations=schedule.bootstrap_iterations,
            seed=schedule.bootstrap_seed + index + 1,
        )
        for index, metric in enumerate((
            "tokens_used", "latency_used_s", "human_escalations",
            "assisted_episode_success", "wrong_execution", "return",
        ))
    }
    secondary["token_reduction_fraction"] = fold_cluster_bootstrap(
        a8_rows, a9_rows, metric="tokens_used", seeds=schedule.seeds,
        iterations=schedule.bootstrap_iterations,
        seed=schedule.bootstrap_seed + 100, ratio_reduction=True,
    )
    secondary["subgoal_completion_rate"] = fold_cluster_ratio_bootstrap(
        a8_rows, a9_rows, numerator="resolved", denominator="incidents",
        seeds=schedule.seeds, iterations=schedule.bootstrap_iterations,
        seed=schedule.bootstrap_seed + 101,
    )
    secondary["wrong_executions_per_incident"] = fold_cluster_ratio_bootstrap(
        a8_rows, a9_rows, numerator="wrong_executions", denominator="incidents",
        seeds=schedule.seeds, iterations=schedule.bootstrap_iterations,
        seed=schedule.bootstrap_seed + 102,
    )
    secondary["safety_penalty_burden"] = fold_cluster_bootstrap(
        a8_rows, a9_rows, metric="safety_penalty_burden",
        seeds=schedule.seeds, iterations=schedule.bootstrap_iterations,
        seed=schedule.bootstrap_seed + 103,
    )
    a8_index = {str(row["episode_id"]): row for row in a8_rows}
    per_seed: dict[str, Any] = {}
    for training_seed in schedule.seeds:
        rows = [row for row in a9_rows if int(row["training_seed"]) == training_seed]
        by_id = {str(row["episode_id"]): row for row in rows}
        ids = sorted(a8_index)
        per_seed[str(training_seed)] = {
            "episode_success_difference": float(np.mean([
                float(bool(by_id[item]["episode_success"]))
                - float(bool(a8_index[item]["episode_success"])) for item in ids
            ])),
            "mcnemar_exploratory": exact_mcnemar_p(
                [bool(by_id[item]["episode_success"]) for item in ids],
                [bool(a8_index[item]["episode_success"]) for item in ids],
            ),
        }
    per_fold = {}
    for fold in schedule.folds:
        baseline = [row for row in a8_rows if int(row["outer_fold"]) == fold]
        learned = [row for row in a9_rows if int(row["outer_fold"]) == fold]
        per_fold[str(fold)] = {
            "episodes": len(baseline),
            "episode_success_difference": (
                np.mean([bool(row["episode_success"]) for row in learned])
                - np.mean([bool(row["episode_success"]) for row in baseline])
            ),
        }
    return {
        "primary_episode_success": primary,
        "mcnemar_statistic_version": MCNEMAR_STATISTIC_VERSION,
        "secondary": secondary,
        "per_seed": per_seed,
        "per_fold": per_fold,
    }


def _routing_by_public_family(
    rows: Sequence[Mapping[str, Any]],
    episode_index: Mapping[str, LongHorizonEpisode],
) -> dict[str, Any]:
    accumulators: dict[str, dict[str, Any]] = {
        str(family): {"incidents": 0, "action_counts": Counter()}
        for family in range(4)
    }
    for row in rows:
        episode_id = str(row["episode_id"])
        if episode_id not in episode_index:
            raise ValueError("routing row does not bind to the frozen episode bank")
        episode = episode_index[episode_id]
        for incident in episode.incidents:
            accumulators[str(incident.family)]["incidents"] += 1
        for step in row["trajectory"]:
            incident_index = int(step["incident_index"])
            if incident_index < 0 or incident_index >= len(episode.incidents):
                if str(step["action"]) == "stop" and incident_index == len(episode.incidents):
                    continue
                raise ValueError("trajectory incident index is outside frozen episode")
            family = str(episode.incidents[incident_index].family)
            accumulators[family]["action_counts"][str(step["action"])] += 1
    result: dict[str, Any] = {}
    for family, accumulator in sorted(accumulators.items()):
        incidents = int(accumulator["incidents"])
        if incidents == 0:
            raise ValueError("public-family routing denominator is zero")
        counts = {
            action: int(accumulator["action_counts"].get(action, 0))
            for action in (
                "observe", "delegate", "escalate", "connect", "review",
                "execute", "human", "stop",
            )
        }
        result[family] = {
            "incidents": incidents,
            "action_counts": counts,
            **{
                f"{action}_per_incident": counts[action] / incidents
                for action in ("delegate", "escalate", "review", "execute", "human")
            },
        }
    return result


def _summarize(
    rows: Sequence[Mapping[str, Any]],
    episode_index: Mapping[str, LongHorizonEpisode],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty evaluation")
    result = summarize_results(list(rows))
    result.update({
        "policy_evaluation_rows": len(rows),
        "unique_episodes": len({str(row["episode_id"]) for row in rows}),
        "training_seeds": sorted({
            int(row["training_seed"]) for row in rows
            if row["training_seed"] is not None
        }),
        "wrong_executions": sum(int(row["wrong_executions"]) for row in rows),
        "episodes_with_wrong_execution": sum(
            bool(row["had_wrong_execution"]) for row in rows
        ),
        "safety_penalty_burden": sum(
            float(row["safety_penalty_burden"]) for row in rows
        ),
        "unsafe_episode_rate": sum(
            bool(row["had_wrong_execution"]) for row in rows
        ) / len(rows),
        "wrong_executions_per_incident": sum(
            int(row["wrong_executions"]) for row in rows
        ) / sum(int(row["incidents"]) for row in rows),
        "failed_stop_episodes": sum(
            bool(row["failed_stop_episode"]) for row in rows
        ),
        "truncated_episodes": sum(
            bool(row["truncated_episode"]) for row in rows
        ),
        "action_counts": dict(sorted(Counter(
            step["action"] for row in rows for step in row["trajectory"]
        ).items())),
        "routing_by_public_family": _routing_by_public_family(
            rows, episode_index,
        ),
    })
    return result


def validate_complete_result(
    a8_rows: Sequence[Mapping[str, Any]], a9_rows: Sequence[Mapping[str, Any]],
    fits: Sequence[Mapping[str, Any]], *, schedule: RunSchedule,
    resource_sha256: str, environment_sha256: str,
    expected_episode_bindings: Mapping[str, tuple[int, str]],
    expected_fit_partitions: Mapping[int, tuple[Sequence[str], Sequence[str]]] | None = None,
    fit_logs: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    output: Path | None = None,
    run_contract_sha256: str | None = None,
) -> dict[str, Any]:
    expected_episodes = len(schedule.folds) * schedule.evaluation_episodes_per_fold
    if len(a8_rows) != expected_episodes:
        raise ValueError("A8 OOF row count mismatch")
    if len(a9_rows) != expected_episodes * len(schedule.seeds):
        raise ValueError("A9 OOF row count mismatch")
    if len(fits) != len(schedule.folds) * len(schedule.seeds):
        raise ValueError("fit count mismatch")
    a8_ids = [str(row["episode_id"]) for row in a8_rows]
    a9_keys = [
        (str(row["episode_id"]), int(row["training_seed"])) for row in a9_rows
    ]
    if len(a8_ids) != len(set(a8_ids)) or len(a9_keys) != len(set(a9_keys)):
        raise ValueError("duplicate OOF decisions")
    if set(a8_ids) != set(expected_episode_bindings):
        raise ValueError("A8 rows do not exactly cover the scheduled OOF episodes")
    if any(
        (int(row["outer_fold"]), str(row["episode_sha256"]))
        != expected_episode_bindings[str(row["episode_id"])]
        for row in a8_rows
    ):
        raise ValueError("A8 row fold or episode hash does not match frozen folds")
    fold_counts = Counter(int(row["outer_fold"]) for row in a8_rows)
    if fold_counts != Counter({
        fold: schedule.evaluation_episodes_per_fold for fold in schedule.folds
    }):
        raise ValueError("OOF evaluation fold counts differ from schedule")
    expected_a9_keys = {
        (episode_id, training_seed)
        for episode_id in a8_ids for training_seed in schedule.seeds
    }
    if set(a9_keys) != expected_a9_keys:
        raise ValueError("A9 rows are not the full A8 episode x seed product")
    if {
        (int(row["outer_fold"]), int(row["training_seed"])) for row in fits
    } != {(fold, seed) for fold in schedule.folds for seed in schedule.seeds}:
        raise ValueError("scheduled fit coverage mismatch")
    if any(
        int(row["final_update"]) != schedule.updates
        or int(row["outer_evaluations_during_training"]) != 0
        or int(row["sampled_held_overlap"]) != 0
        or row["selected_checkpoint"] != "final"
        for row in fits
    ):
        raise ValueError("final-checkpoint/no-selection contract failed")
    fit_provenance_verified = False
    if any(item is not None for item in (
        expected_fit_partitions, fit_logs, output, run_contract_sha256,
    )):
        if (
            expected_fit_partitions is None or fit_logs is None
            or output is None or run_contract_sha256 is None
        ):
            raise ValueError("partial fit-provenance validation request")
        output = Path(output).resolve()
        if set(expected_fit_partitions) != set(schedule.folds):
            raise ValueError("fit partition fold coverage mismatch")
        expected_log_keys = {
            f"fold-{fold}-seed-{training_seed}"
            for fold in schedule.folds for training_seed in schedule.seeds
        }
        if set(fit_logs) != expected_log_keys:
            raise ValueError("training log fit coverage mismatch")
        for fit in fits:
            fold = int(fit["outer_fold"])
            training_seed = int(fit["training_seed"])
            train_ids = tuple(sorted(expected_fit_partitions[fold][0]))
            held_ids = tuple(sorted(expected_fit_partitions[fold][1]))
            if (
                len(train_ids) != 2400 or len(set(train_ids)) != 2400
                or len(held_ids) != 600 or len(set(held_ids)) != 600
                or set(train_ids) & set(held_ids)
            ):
                raise ValueError("invalid expected outer partition")
            if (
                fit["outer_train_episode_ids_sha256"] != _digest(train_ids)
                or fit["outer_held_episode_ids_sha256"] != _digest(held_ids)
            ):
                raise ValueError("fit train/held hash does not match frozen folds")
            if fit["run_contract_sha256"] != run_contract_sha256:
                raise ValueError("fit run-contract binding mismatch")
            key = f"fold-{fold}-seed-{training_seed}"
            logs = list(fit_logs[key])
            if len(logs) != schedule.updates:
                raise ValueError("training update log count mismatch")
            sample_sequence: list[str] = []
            train_set = set(train_ids)
            held_set = set(held_ids)
            for expected_update, row in enumerate(logs, start=1):
                sampled = [str(item) for item in row["sampled_episode_ids"]]
                if (
                    int(row["outer_fold"]) != fold
                    or int(row["training_seed"]) != training_seed
                    or int(row["update"]) != expected_update
                    or len(sampled) != schedule.episodes_per_update
                    or not set(sampled) <= train_set
                    or bool(set(sampled) & held_set)
                ):
                    raise ValueError("training sample stream violates frozen outer fold")
                sample_sequence.extend(sampled)
            if (
                fit["sample_sequence_sha256"] != _digest(sample_sequence)
                or int(fit["sampled_episode_draws"]) != len(sample_sequence)
            ):
                raise ValueError("fit sample-sequence hash mismatch")
            checkpoint = (
                output / "fits" / f"fold-{fold}"
                / f"seed-{training_seed}" / "final.pt"
            )
            if not checkpoint.is_file() or _sha256(checkpoint) != fit["checkpoint_sha256"]:
                raise ValueError("final checkpoint disk hash mismatch")
            _, metadata = load_checkpoint(
                checkpoint, torch.device("cpu"),
                expected_policy_version=POLICY_VERSION,
            )
            if (
                int(metadata["seed"]) != training_seed
                or int(metadata["update"]) != schedule.updates
                or metadata.get("run_contract_sha256") != run_contract_sha256
            ):
                raise ValueError("final checkpoint provenance metadata mismatch")
        fit_provenance_verified = True
    all_rows = [*a8_rows, *a9_rows]
    if any(row["resource_contract_sha256"] != resource_sha256 for row in all_rows):
        raise ValueError("resource binding mismatch")
    if any(row["environment_source_sha256"] != environment_sha256 for row in all_rows):
        raise ValueError("environment source binding mismatch")
    if run_contract_sha256 is not None and any(
        row.get("run_contract_sha256") != run_contract_sha256 for row in all_rows
    ):
        raise ValueError("decision row run-contract binding mismatch")
    if any(
        int(row["invalid_actions"]) != 0 or int(row["budget_violations"]) != 0
        for row in all_rows
    ):
        raise ValueError("invalid action or budget violation gate failed")
    a8_by_id = {str(row["episode_id"]): row for row in a8_rows}
    for row in a9_rows:
        baseline = a8_by_id[str(row["episode_id"])]
        for field in (
            "episode_sha256", "outer_fold", "train_bank_sha256",
            "resource_contract_sha256", "environment_source_sha256",
            "environment_class", "token_cap", "latency_cap_s", "incidents",
        ):
            if row[field] != baseline[field]:
                raise ValueError(f"paired resource field mismatch: {field}")
        if row["environment_source_sha256"] != environment_sha256:
            raise ValueError("environment source binding mismatch")
        if row["a8_row_sha256"] != _digest(baseline):
            raise ValueError("A8 paired-row hash mismatch")
    fit_checkpoint = {
        (int(row["outer_fold"]), int(row["training_seed"])):
        str(row["checkpoint_sha256"])
        for row in fits
    }
    if any(
        row["final_checkpoint_sha256"]
        != fit_checkpoint[(int(row["outer_fold"]), int(row["training_seed"]))]
        for row in a9_rows
    ):
        raise ValueError("final checkpoint row binding mismatch")
    return {
        "fits_complete": len(fits),
        "a8_rows_complete": len(a8_rows),
        "a9_rows_complete": len(a9_rows),
        "unique_a8_episodes": len(set(a8_ids)),
        "unique_a9_episode_seed_pairs": len(set(a9_keys)),
        "invalid_actions": 0,
        "budget_violations": 0,
        "shared_resource_binding": True,
        "all_checkpoints_final_update": True,
        "outer_evaluations_during_training": 0,
        "sampled_held_overlap": 0,
        "frozen_fold_episode_binding": True,
        "fit_sampling_logs_recomputed": fit_provenance_verified,
        "checkpoint_disk_hashes_recomputed": fit_provenance_verified,
        "run_contract_binding": fit_provenance_verified,
    }


def _plot_training(fit_logs: Mapping[str, Sequence[Mapping[str, Any]]], path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for label, rows in sorted(fit_logs.items()):
        updates = [int(row["update"]) for row in rows]
        axes[0, 0].plot(updates, [row["rollout_episode_success_rate"] for row in rows], alpha=0.65)
        axes[0, 1].plot(updates, [row["rollout_mean_return"] for row in rows], alpha=0.65)
        axes[1, 0].plot(updates, [row["rollout_tokens_per_episode"] for row in rows], alpha=0.65)
        axes[1, 1].plot(updates, [row["entropy"] for row in rows], alpha=0.65, label=label)
    titles = ("Training rollout success", "Training rollout return", "Training rollout tokens", "Policy entropy")
    for axis, title in zip(axes.flat, titles, strict=True):
        axis.set_title(title)
        axis.set_xlabel("PPO update")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _claim_result(
    validation: Mapping[str, Any], comparisons: Mapping[str, Any], *, formal: bool,
) -> dict[str, Any]:
    primary = comparisons["primary_episode_success"]
    validity = (
        validation["invalid_actions"] == 0
        and validation["budget_violations"] == 0
        and validation["shared_resource_binding"]
        and validation["all_checkpoints_final_update"]
        and validation["frozen_fold_episode_binding"]
        and validation["sampled_held_overlap"] == 0
        and validation["fit_sampling_logs_recomputed"]
        and validation["checkpoint_disk_hashes_recomputed"]
        and validation["run_contract_binding"]
    )
    return {
        "validity_gates_passed": validity,
        "primary_a9_beats_a8_success_ci": bool(
            formal and validity and primary["ci95_low"] > 0.0
        ),
        "formal_claim_evaluable": formal,
        "primary_claim_only": True,
        "secondary_metrics_substitutable": False,
        "may_claim_hidden_test_generalization": False,
        "may_claim_seed_population_stability": False,
        "may_claim_llm_weight_rl": False,
    }


def _verify_frozen_run_inputs(
    root: Path, *, source: Mapping[str, Any], train_bank_sha256: str,
    assignments: Sequence[Any], fold_manifest_sha: str,
    resource: Mapping[str, Any], resource_sha256: str,
    training_contract: Mapping[str, Any], training_contract_sha256: str,
    require_clean: bool,
) -> None:
    if _git_source_state(root, require_clean=require_clean) != source:
        raise RuntimeError("executed source changed during A9 OOF run")
    if _sha256(FROZEN_TRAIN_PATH) != train_bank_sha256:
        raise RuntimeError("frozen train bank changed during A9 OOF run")
    if fold_manifest_sha256(assignments) != fold_manifest_sha:
        raise RuntimeError("fold assignment changed during A9 OOF run")
    if resource_contract_sha256(resource) != resource_sha256:
        raise RuntimeError("resource contract changed during A9 OOF run")
    if _digest(training_contract) != training_contract_sha256:
        raise RuntimeError("training contract changed during A9 OOF run")


def _formal_lock(
    root: Path, output: Path, source: Mapping[str, Any],
    run_contract_sha256: str,
) -> None:
    path = root / FORMAL_ATTEMPT_LOCK
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "schema_version": "multitown-a9-v2-formal-attempt-lock-v2-recovery",
            "runner_version": RUNNER_VERSION,
            "output": str(output),
            "source_revision": source["revision"],
            "run_contract_sha256": run_contract_sha256,
            "policy": (
                "one formal attempt; failure invalidates the attempt and requires "
                "a new protocol version rather than seed/checkpoint cherry-picking"
            ),
        }, ensure_ascii=False, indent=2) + "\n")


def run(output: Path, *, smoke: bool) -> int:
    root = Path(__file__).resolve().parents[1]
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    schedule = _schedule(smoke)
    source = _git_source_state(root, require_clean=not smoke)
    bank = load_frozen_train_bank(FROZEN_TRAIN_PATH)
    assignments = assign_stratified_group_folds(bank)
    fold_validation = validate_fold_assignments(bank, assignments, folds=DEFAULT_FOLDS)
    assignment_index = {row.episode_id: row for row in assignments}
    episode_index = {row.episode_id: row for row in bank.episodes}
    resource = shared_resource_contract(bank)
    resource_sha = resource_contract_sha256(resource)
    environment_sha = str(resource["environment_source_sha256"])
    contract = canonical_training_contract(schedule)
    contract_sha = _digest(contract)
    frozen_fold_manifest_sha = fold_manifest_sha256(assignments)
    recovery_lineage: dict[str, Any] | None = None
    if not smoke:
        predecessor_lock = root / INVALIDATED_PREDECESSOR_LOCK
        predecessor_output = root / INVALIDATED_PREDECESSOR_OUTPUT
        predecessor_failure = predecessor_output / "INVALIDATED.json"
        predecessor_all_fits = predecessor_output / "all-fits-complete.json"
        if (
            not predecessor_lock.is_file()
            or not predecessor_failure.is_file()
            or not predecessor_all_fits.is_file()
            or (predecessor_output / "result.json").exists()
        ):
            raise RuntimeError(
                "r2 recovery requires the preserved invalidated r1 lock and output"
            )
        predecessor_lock_payload = json.loads(
            predecessor_lock.read_text(encoding="utf-8")
        )
        predecessor_failure_payload = json.loads(
            predecessor_failure.read_text(encoding="utf-8")
        )
        if (
            predecessor_lock_payload.get("runner_version")
            != "multitown-a9-v2-train-only-ppo-oof-runner-v1"
            or predecessor_failure_payload.get("error_type") != "OverflowError"
            or predecessor_failure_payload.get("invalidated") is not True
            or _sha256(predecessor_lock) != INVALIDATED_PREDECESSOR_LOCK_SHA256
            or _sha256(predecessor_failure) != INVALIDATED_PREDECESSOR_FAILURE_SHA256
            or _sha256(predecessor_all_fits)
            != INVALIDATED_PREDECESSOR_ALL_FITS_SHA256
        ):
            raise RuntimeError("r1 predecessor does not match frozen recovery lineage")
        recovery_lineage = {
            "predecessor_attempt": "A9-v2 formal r1",
            "predecessor_status": "technical invalidation",
            "predecessor_lock_path": str(INVALIDATED_PREDECESSOR_LOCK),
            "predecessor_lock_sha256": _sha256(predecessor_lock),
            "predecessor_failure_path": str(
                INVALIDATED_PREDECESSOR_OUTPUT / "INVALIDATED.json"
            ),
            "predecessor_failure_sha256": _sha256(predecessor_failure),
            "predecessor_all_fits_sha256": _sha256(predecessor_all_fits),
            "checkpoint_or_oof_reuse": False,
            "restart_all_fits_from_zero": True,
            "only_analysis_change": MCNEMAR_STATISTIC_VERSION,
        }
    run_contract = {
        "schema_version": "multitown-a9-v2-run-root-contract-v2-recovery",
        "runner_version": RUNNER_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "policy_version": POLICY_VERSION,
        "mode": schedule.mode,
        "source": source,
        "train_bank_sha256": bank.payload_sha256,
        "fold_manifest_sha256": frozen_fold_manifest_sha,
        "resource_contract_sha256": resource_sha,
        "environment_source_sha256": environment_sha,
        "training_contract_sha256": contract_sha,
        "schedule": asdict(schedule),
        "recovery_lineage": recovery_lineage,
    }
    run_contract_sha = _digest(run_contract)
    if not smoke:
        artifacts_root = (root / "artifacts").resolve()
        if output.parent != artifacts_root:
            raise ValueError(
                "formal output must be a direct child of the ignored artifacts directory"
            )
    if not smoke:
        _formal_lock(root, output, source, run_contract_sha)
    output.mkdir(parents=True)
    try:
        folds_path = output / "folds.jsonl"
        _write_jsonl(folds_path, [row.to_dict() for row in assignments])
        _write_json(output / "resource-contract.json", resource)
        _write_json(output / "run-contract.json", run_contract)
        pre_run = {
            "schema_version": "multitown-a9-v2-oof-pre-run-manifest-v2-recovery",
            "runner_version": RUNNER_VERSION,
            "mode": schedule.mode,
            "source": source,
            "train_bank": {
                "path": str(bank.path), "sha256": bank.payload_sha256,
                "episodes": len(bank.episodes),
                "manifest_sha256": _sha256(bank.path.with_name("manifest.json")),
                "leakage_audit_sha256": _sha256(
                    bank.path.with_name("precall-leakage-audit.json")
                ),
            },
            "fold_version": FOLD_VERSION,
            "fold_manifest_sha256": frozen_fold_manifest_sha,
            "fold_file_sha256": _sha256(folds_path),
            "fold_validation": fold_validation,
            "resource_contract_sha256": resource_sha,
            "training_contract": contract,
            "training_contract_sha256": contract_sha,
            "run_contract_sha256": run_contract_sha,
            "runtime": {
                "python": sys.version.split()[0], "platform": platform.platform(),
                "numpy": np.__version__, "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
            },
            "data_access": {
                "unique_physical_train_paths_opened": 1,
                "train_payload_content_reads_planned": 3,
                "development_files_opened": 0,
                "test_files_opened": 0,
                "combined_bank_filtering": False,
            },
            "technical_failure_policy": (
                "any failure invalidates the entire attempt; no fit, seed, checkpoint "
                "or poor valid outcome may be selectively retried"
            ),
            "started_at_utc": datetime.now(UTC).isoformat(),
        }
        _write_json(output / "pre-run-manifest.json", pre_run)
        torch.set_num_threads(schedule.threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            if not smoke:
                raise
        fits: list[dict[str, Any]] = []
        fit_logs: dict[str, list[dict[str, Any]]] = {}
        expected_fit_partitions: dict[int, tuple[list[str], list[str]]] = {}
        for fold in schedule.folds:
            train_ids = sorted(
                row.episode_id for row in assignments if row.fold != fold
            )
            held_ids = sorted(
                row.episode_id for row in assignments if row.fold == fold
            )
            if set(train_ids) & set(held_ids) or len(train_ids) != 2400 or len(held_ids) != 600:
                raise RuntimeError("outer train/evaluation isolation failed")
            expected_fit_partitions[fold] = (train_ids, held_ids)
            for training_seed in schedule.seeds:
                fit_dir = output / "fits" / f"fold-{fold}" / f"seed-{training_seed}"
                fit_dir.mkdir(parents=True)
                result = _train_fit(
                    [episode_index[item] for item in train_ids], seed=training_seed,
                    fold=fold, held_episode_ids=held_ids,
                    schedule=schedule, output=fit_dir,
                    run_contract_sha256=run_contract_sha,
                )
                fits.append(result)
                fit_logs[f"fold-{fold}-seed-{training_seed}"] = [
                    json.loads(line) for line in
                    (fit_dir / "training-metrics.jsonl").read_text(encoding="utf-8").splitlines()
                ]
        _verify_frozen_run_inputs(
            root, source=source, train_bank_sha256=bank.payload_sha256,
            assignments=assignments, fold_manifest_sha=frozen_fold_manifest_sha,
            resource=resource, resource_sha256=resource_sha,
            training_contract=contract, training_contract_sha256=contract_sha,
            require_clean=not smoke,
        )
        _write_json(output / "all-fits-complete.json", {
            "fits": fits,
            "evaluation_started": False,
            "all_scheduled_fits_reached_final_update": True,
            "run_contract_sha256": run_contract_sha,
        })
        a8_rows: list[dict[str, Any]] = []
        a9_rows: list[dict[str, Any]] = []
        expected_episode_bindings: dict[str, tuple[int, str]] = {}
        for fold in schedule.folds:
            ids = sorted(row.episode_id for row in assignments if row.fold == fold)
            ids = ids[:schedule.evaluation_episodes_per_fold]
            for episode_id in ids:
                assignment = assignment_index[episode_id]
                expected_episode_bindings[episode_id] = (
                    fold, assignment.episode_sha256,
                )
                episode = episode_index[episode_id]
                a8_rows.append(_evaluate_a8(
                    episode, fold=fold,
                    episode_sha256=assignment.episode_sha256,
                    bank_sha256=bank.payload_sha256, resource_sha256=resource_sha,
                    environment_sha256=environment_sha,
                    run_contract_sha256=run_contract_sha,
                ))
            for training_seed in schedule.seeds:
                checkpoint = output / "fits" / f"fold-{fold}" / f"seed-{training_seed}" / "final.pt"
                model, metadata = load_checkpoint(
                    checkpoint, torch.device("cpu"),
                    expected_policy_version=POLICY_VERSION,
                )
                if (
                    int(metadata["update"]) != schedule.updates
                    or int(metadata["seed"]) != training_seed
                    or metadata.get("run_contract_sha256") != run_contract_sha
                ):
                    raise RuntimeError("final checkpoint metadata mismatch")
                checkpoint_sha = _sha256(checkpoint)
                for episode_id in ids:
                    assignment = assignment_index[episode_id]
                    episode = episode_index[episode_id]
                    a9_rows.append(_evaluate_episode(
                        episode,
                        lambda observation, mask, model=model: _model_action(
                            model, observation, mask,
                        ),
                        fold=fold, system="A9-v2-PPO", training_seed=training_seed,
                        episode_sha256=assignment.episode_sha256,
                        bank_sha256=bank.payload_sha256, resource_sha256=resource_sha,
                        environment_sha256=environment_sha,
                        checkpoint_sha256=checkpoint_sha,
                        run_contract_sha256=run_contract_sha,
                        a8_row_sha256=_digest(next(
                            row for row in a8_rows
                            if row["episode_id"] == episode_id
                        )),
                    ))
        _write_jsonl(output / "a8-oof-decisions.jsonl", a8_rows)
        _write_jsonl(output / "a9-oof-decisions.jsonl", a9_rows)
        validation = validate_complete_result(
            a8_rows, a9_rows, fits, schedule=schedule,
            resource_sha256=resource_sha, environment_sha256=environment_sha,
            expected_episode_bindings=expected_episode_bindings,
            expected_fit_partitions=expected_fit_partitions, fit_logs=fit_logs,
            output=output, run_contract_sha256=run_contract_sha,
        )
        comparisons = _paired_effects(a8_rows, a9_rows, schedule)
        result = {
            "schema_version": RESULT_VERSION,
            "mode": schedule.mode,
            "evidence_scope": "train-only out-of-fold internal development evidence",
            "source_revision": source["revision"],
            "run_contract_sha256": run_contract_sha,
            "training_contract_sha256": contract_sha,
            "fold_manifest_sha256": frozen_fold_manifest_sha,
            "resource_contract_sha256": resource_sha,
            "validation": validation,
            "overall": {
                "A8-long-public-view": _summarize(a8_rows, episode_index),
                "A9-v2-PPO-fixed-seed-average": _summarize(a9_rows, episode_index),
            },
            "comparisons": comparisons,
            "claim_gates": _claim_result(validation, comparisons, formal=not smoke),
            "limitations": [
                "Train-only OOF evidence is not hidden-test or OOD generalization.",
                "The interval is conditional on the preregistered training seeds.",
                "PPO trains a lightweight controller, not Qwen or GLM weights.",
                "Workers, reviewer outcomes and tool failures remain simulator components.",
            ],
            "data_access": {
                "unique_physical_train_paths_opened": 1,
                "train_payload_content_reads_completed": 3,
                "development_files_opened": 0,
                "test_files_opened": 0,
            },
        }
        _plot_training(fit_logs, output / "training-curves.png")
        _verify_frozen_run_inputs(
            root, source=source, train_bank_sha256=bank.payload_sha256,
            assignments=assignments, fold_manifest_sha=frozen_fold_manifest_sha,
            resource=resource, resource_sha256=resource_sha,
            training_contract=contract, training_contract_sha256=contract_sha,
            require_clean=not smoke,
        )
        _write_json(output / "result.json", result)
        files = {
            str(path.relative_to(output)): {
                "bytes": path.stat().st_size, "sha256": _sha256(path),
            }
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name not in {"artifact-manifest.json", "RUNNING.json"}
        }
        _write_json(output / "artifact-manifest.json", {
            "schema_version": "multitown-a9-v2-oof-artifact-manifest-v2-recovery",
            "source_revision": source["revision"],
            "mode": schedule.mode,
            "run_contract_sha256": run_contract_sha,
            "files": files,
            "completed_at_utc": datetime.now(UTC).isoformat(),
        })
        print(json.dumps({
            "output": str(output), "mode": schedule.mode,
            "claim_gates": result["claim_gates"],
            "primary": result["comparisons"]["primary_episode_success"],
        }, ensure_ascii=False, indent=2))
        return 0
    except BaseException as exc:
        result_path = output / "result.json"
        invalid_result_path = output / "INVALID_RESULT.json"
        if result_path.exists() and not invalid_result_path.exists():
            os.replace(result_path, invalid_result_path)
        failure = {
            "schema_version": "multitown-a9-v2-invalidated-attempt-v2-recovery",
            "mode": schedule.mode,
            "invalidated": True,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "policy": "do not selectively retry a fit/seed/checkpoint from this attempt",
            "failed_at_utc": datetime.now(UTC).isoformat(),
        }
        try:
            _write_json(output / "INVALIDATED.json", failure)
        except Exception:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--smoke", action="store_true",
        help="run the fixed non-evidentiary five-fold one-update smoke schedule",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(run(args.output_dir, smoke=args.smoke))


if __name__ == "__main__":
    main()
