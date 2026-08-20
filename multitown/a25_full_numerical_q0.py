"""Synthetic common-state fixtures for the A25 full numerical Q0 profile.

This module contains development-only fixture construction.  It never reads an
outer split and does not authorize an A24 or A25 formal run.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import struct
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from types import MappingProxyType
from typing import Any
from unittest.mock import patch

import numpy as np
import torch

from .a22_constrained_ppo import (
    DualState,
    SafetyThresholds,
    effective_action_mask,
    lagrangian_batch,
    lagrangian_ppo_update,
)
from .a23_cr_ppo import cr_ppo_batch, model_parameter_sha256, select_actor_mode
from .a25_qualification import array_mapping_sha256, generator_state_sha256
from .a25_shield_dependence import (
    InterventionObjective,
    NumericalGradientEvent,
    intervention_ppo_update,
    shield_aware_batch,
)
from .long_horizon_env import ACTION_COUNT, MultiTownLongHorizonEnv, RLAction
from .ppo_controller import ActorCritic, PPOConfig, _masked_distribution
from .pq1_numerical_conformance import optimizer_state_sha256

A25_FULL_NUMERICAL_FIXTURE_VERSION = "multitown-a25-full-numerical-fixture-v1"
A25_TWELVE_CELL_RECEIPT_VERSION = "multitown-a25-twelve-cell-diagnostic-receipt-v2"
A25_AUX_NOT_OBSERVED_REASON = "not-defined-in-frozen-a22-beta-zero-path"
A25_FULL_NUMERICAL_SEED = 2026081501
A25_WARMUP_SEED = 2026081502
A25_UPDATE_SEED = 2026081503
A25_COMMON_BETA = 5.0
A25_MEAN_INCIDENTS = 4.0
A25_EPISODE_COUNT = 48
A25_DECISION_COUNT = 108
A25_SHIELD_ACTIVE_COUNT = 36
_BASE_BATCH_FIELDS = (
    "observation",
    "mask",
    "action",
    "old_log_probability",
    "old_value",
    "return",
    "advantage",
    "reward_advantage_raw",
    "unsafe_advantage_raw",
    "wrong_advantage_raw",
)
_A22_CAPTURE_LOCK = threading.Lock()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _typed_sha256(value: Any) -> str:
    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if item is None:
            digest.update(b"none\0")
        elif type(item) is bool:
            digest.update(b"bool\0" + (b"1" if item else b"0") + b"\0")
        elif type(item) is int:
            digest.update(b"int\0" + str(item).encode("ascii") + b"\0")
        elif type(item) is float:
            if not math.isfinite(item):
                raise ValueError("non-finite A25 typed fixture float")
            digest.update(b"float64\0" + struct.pack(">d", item) + b"\0")
        elif type(item) is str:
            payload = item.encode("utf-8")
            digest.update(b"str\0" + str(len(payload)).encode("ascii") + b"\0")
            digest.update(payload + b"\0")
        elif isinstance(item, np.ndarray):
            if item.dtype.hasobject or (
                np.issubdtype(item.dtype, np.floating)
                and not bool(np.isfinite(item).all())
            ):
                raise ValueError("invalid A25 typed fixture array")
            little = np.ascontiguousarray(item, dtype=item.dtype.newbyteorder("<"))
            digest.update(b"ndarray\0" + little.dtype.str.encode("ascii") + b"\0")
            digest.update(_canonical_bytes(list(little.shape)) + b"\0")
            digest.update(little.tobytes(order="C") + b"\0")
        elif isinstance(item, Mapping):
            if any(type(key) is not str for key in item):
                raise ValueError("invalid A25 typed fixture key")
            digest.update(b"mapping\0" + str(len(item)).encode("ascii") + b"\0")
            for key in sorted(item):
                update(key)
                update(item[key])
        elif type(item) in {list, tuple}:
            digest.update(
                (b"list\0" if type(item) is list else b"tuple\0")
                + str(len(item)).encode("ascii")
                + b"\0"
            )
            for nested in item:
                update(nested)
        else:
            raise TypeError("unsupported A25 typed fixture value")

    update(value)
    return digest.hexdigest()


def full_numerical_config() -> PPOConfig:
    """Return the frozen whole-batch four-step numerical schedule."""

    return PPOConfig(
        updates=1,
        episodes_per_update=A25_EPISODE_COUNT,
        hidden_size=8,
        learning_rate=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        ppo_epochs=4,
        minibatch_size=4096,
        value_coef=0.5,
        entropy_coef=0.02,
        max_grad_norm=0.5,
        dev_interval=0,
    )


def _warmup_config() -> PPOConfig:
    return PPOConfig(
        **{
            **asdict(full_numerical_config()),
            "episodes_per_update": 8,
            "ppo_epochs": 1,
        }
    )


def _fixture_model() -> ActorCritic:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(A25_FULL_NUMERICAL_SEED)
        return ActorCritic(
            MultiTownLongHorizonEnv.observation_size,
            full_numerical_config().hidden_size,
            ACTION_COUNT,
        ).cpu()


def _review_vector(state: str) -> np.ndarray:
    mapping = {
        "unknown": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        "pass": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        "fail": np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
    }
    try:
        return mapping[state]
    except KeyError as exc:
        raise ValueError("invalid A25 synthetic review state") from exc


def _observation(block: int, episode: int, step: int, review: str) -> np.ndarray:
    if (
        type(block) is not int
        or not 0 <= block < 12
        or type(episode) is not int
        or not 0 <= episode < 4
        or type(step) is not int
        or step < 0
    ):
        raise ValueError("invalid A25 synthetic row identity")
    row = np.empty(MultiTownLongHorizonEnv.observation_size, dtype=np.float32)
    row_id = block * 32 + episode * 5 + step + 1
    for index in range(len(row)):
        numerator = ((row_id * (index + 3) + 7 * episode + step) % 29) + 1
        sign = -1.0 if (row_id + index) % 2 else 1.0
        row[index] = np.float32(sign * numerator / 1000.0)
    row[33:36] = _review_vector(review)
    if not np.isfinite(row).all():
        raise FloatingPointError("non-finite A25 synthetic observation")
    row.setflags(write=False)
    return row


def _base_mask(kind: str) -> np.ndarray:
    values = {
        "active": [1, 1, 1, 0, 1, 1, 1, 1],
        "pass": [0, 0, 0, 0, 0, 1, 1, 1],
        "execute_illegal": [1, 1, 1, 0, 1, 0, 1, 1],
    }
    try:
        mask = np.asarray(values[kind], dtype=np.bool_)
    except KeyError as exc:
        raise ValueError("invalid A25 synthetic mask kind") from exc
    mask.setflags(write=False)
    return mask


_BLOCK_ROWS = (
    (0, 0, "unknown", "active", RLAction.REVIEW, -0.20, False, 0.0, 0.0),
    (0, 1, "pass", "pass", RLAction.EXECUTE, -1.00, True, 1.0, 0.25),
    (1, 0, "fail", "active", RLAction.HUMAN, -0.10, False, 0.0, 0.0),
    (1, 1, "pass", "pass", RLAction.EXECUTE, -0.70, True, 1.0, 0.25),
    (2, 0, "unknown", "active", RLAction.REVIEW, -0.30, False, 0.0, 0.0),
    (2, 1, "pass", "pass", RLAction.EXECUTE, -0.80, True, 1.0, 0.25),
    (2, 2, "pass", "pass", RLAction.EXECUTE, -1.20, True, 0.0, 0.25),
    (
        3,
        0,
        "unknown",
        "execute_illegal",
        RLAction.REVIEW,
        -0.05,
        False,
        0.0,
        0.0,
    ),
    (3, 1, "pass", "pass", RLAction.EXECUTE, 1.00, False, 0.0, 0.0),
)

_COMMON_TRANSITION_FIELDS = {
    "observation",
    "base_mask",
    "mask",
    "action",
    "old_log_probability",
    "old_value",
    "reward",
    "unsafe_cost",
    "wrong_cost",
    "wrong_execute",
    "shield_intervened",
    "pre_shield_logits",
    "base_execute_probability",
}


def validate_common_transition_fixture(
    episodes: Sequence[Sequence[Mapping[str, Any]]],
    manifest: Mapping[str, Any],
) -> None:
    """Reject cost-semantics, type, mask, ordering, or identity mutations."""

    if len(episodes) != A25_EPISODE_COUNT or not isinstance(manifest, Mapping):
        raise ValueError("invalid A25 common transition fixture")
    expected_lengths = (2, 2, 3, 2) * 12
    if tuple(len(episode) for episode in episodes) != expected_lengths:
        raise ValueError("invalid A25 common episode lengths")
    unsafe_events = 0
    wrong_executions = 0
    shield_active = 0
    decision_count = 0
    for episode in episodes:
        wrong_seen = False
        for row in episode:
            if not isinstance(row, Mapping) or set(row) != _COMMON_TRANSITION_FIELDS:
                raise ValueError("invalid A25 common transition fields")
            observation = row["observation"]
            base_mask = row["base_mask"]
            mask = row["mask"]
            logits = row["pre_shield_logits"]
            action = row["action"]
            wrong = row["wrong_execute"]
            scalar_fields = (
                row["old_log_probability"],
                row["old_value"],
                row["reward"],
                row["unsafe_cost"],
                row["wrong_cost"],
                row["base_execute_probability"],
            )
            if (
                not isinstance(observation, np.ndarray)
                or observation.shape
                != (MultiTownLongHorizonEnv.observation_size,)
                or observation.dtype != np.float32
                or observation.flags.writeable
                or not np.isfinite(observation).all()
                or not isinstance(base_mask, np.ndarray)
                or base_mask.shape != (ACTION_COUNT,)
                or base_mask.dtype != np.bool_
                or base_mask.flags.writeable
                or not isinstance(mask, np.ndarray)
                or mask.shape != (ACTION_COUNT,)
                or mask.dtype != np.bool_
                or mask.flags.writeable
                or not isinstance(logits, np.ndarray)
                or logits.shape != (ACTION_COUNT,)
                or logits.dtype != np.float32
                or logits.flags.writeable
                or not np.isfinite(logits).all()
                or type(action) is not int
                or not 0 <= action < ACTION_COUNT
                or type(wrong) is not bool
                or type(row["shield_intervened"]) is not bool
                or any(type(value) is not float or not math.isfinite(value) for value in scalar_fields)
            ):
                raise ValueError("invalid A25 common transition types")
            expected_mask, expected_active = effective_action_mask(
                observation, base_mask, shield_enabled=True
            )
            if (
                not np.array_equal(mask, expected_mask)
                or row["shield_intervened"] is not expected_active
                or not bool(mask[action])
                or row["unsafe_cost"] not in {0.0, 1.0}
                or row["wrong_cost"] not in {0.0, 0.25}
                or wrong != (row["wrong_cost"] == 0.25)
                or wrong != (action == int(RLAction.EXECUTE) and row["reward"] < 0.0)
                or row["unsafe_cost"] != float(wrong and not wrong_seen)
            ):
                raise ValueError("invalid A25 common transition semantics")
            wrong_seen = wrong_seen or wrong
            decision_count += 1
            unsafe_events += int(row["unsafe_cost"])
            wrong_executions += int(wrong)
            shield_active += int(expected_active)
    observed = {
        "episodes": len(episodes),
        "decisions": decision_count,
        "unsafe_events": unsafe_events,
        "wrong_executions": wrong_executions,
        "shield_active_decisions": shield_active,
    }
    expected = {
        "episodes": A25_EPISODE_COUNT,
        "decisions": A25_DECISION_COUNT,
        "unsafe_events": 36,
        "wrong_executions": 48,
        "shield_active_decisions": A25_SHIELD_ACTIVE_COUNT,
    }
    if observed != expected or any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("invalid A25 common transition counts")
    if (
        manifest.get("schema_version") != A25_FULL_NUMERICAL_FIXTURE_VERSION
        or manifest.get("unsafe_cost_per_episode") != 0.75
        or manifest.get("wrong_cost_per_fixed_mean_incident") != 0.25
        or manifest.get("mean_incidents") != A25_MEAN_INCIDENTS
        or manifest.get("transition_sha256") != _typed_sha256(episodes)
    ):
        raise ValueError("invalid A25 common transition identity")


def build_common_transition_fixture(
    model: ActorCritic,
) -> tuple[tuple[tuple[Mapping[str, Any], ...], ...], dict[str, Any]]:
    """Build 48 synthetic episodes and bind snapshots to ``model``."""

    if not isinstance(model, ActorCritic) or next(model.parameters()).device.type != "cpu":
        raise TypeError("A25 synthetic fixture requires a CPU ActorCritic")
    episodes: list[list[dict[str, Any]]] = []
    with torch.no_grad():
        for block in range(12):
            block_episodes: list[list[dict[str, Any]]] = [[], [], [], []]
            for (
                episode,
                step,
                review,
                mask_kind,
                action,
                reward,
                wrong,
                unsafe_cost,
                wrong_cost,
            ) in _BLOCK_ROWS:
                observation = _observation(block, episode, step, review)
                base_mask = _base_mask(mask_kind)
                effective_mask, active = effective_action_mask(
                    observation, base_mask, shield_enabled=True
                )
                action_index = int(action)
                if not bool(effective_mask[action_index]):
                    raise RuntimeError("A25 synthetic action is not effective-mask legal")
                observation_tensor = torch.from_numpy(
                    np.asarray(observation).copy()
                ).unsqueeze(0)
                logits, value = model(observation_tensor)
                effective_distribution = _masked_distribution(
                    logits,
                    torch.from_numpy(np.asarray(effective_mask).copy()).unsqueeze(0),
                )
                base_distribution = _masked_distribution(
                    logits,
                    torch.from_numpy(np.asarray(base_mask).copy()).unsqueeze(0),
                )
                action_tensor = torch.tensor([action_index], dtype=torch.long)
                logits_array = logits[0].detach().cpu().numpy().astype(
                    np.float32, copy=True
                )
                logits_array.setflags(write=False)
                transition = {
                    "observation": observation,
                    "base_mask": base_mask,
                    "mask": effective_mask,
                    "action": action_index,
                    "old_log_probability": float(
                        effective_distribution.log_prob(action_tensor).item()
                    ),
                    "old_value": float(value.item()),
                    "reward": float(reward),
                    "unsafe_cost": float(unsafe_cost),
                    "wrong_cost": float(wrong_cost),
                    "wrong_execute": bool(wrong),
                    "shield_intervened": bool(active),
                    "pre_shield_logits": logits_array,
                    "base_execute_probability": float(
                        base_distribution.probs[0, int(RLAction.EXECUTE)].item()
                    ),
                }
                block_episodes[episode].append(transition)
            episodes.extend(block_episodes)
    flat = [row for episode in episodes for row in episode]
    counts = {
        "episodes": len(episodes),
        "decisions": len(flat),
        "unsafe_events": int(sum(float(row["unsafe_cost"]) for row in flat)),
        "wrong_executions": sum(bool(row["wrong_execute"]) for row in flat),
        "shield_active_decisions": sum(
            bool(row["shield_intervened"]) for row in flat
        ),
    }
    if counts != {
        "episodes": A25_EPISODE_COUNT,
        "decisions": A25_DECISION_COUNT,
        "unsafe_events": 36,
        "wrong_executions": 48,
        "shield_active_decisions": A25_SHIELD_ACTIVE_COUNT,
    }:
        raise RuntimeError("A25 common transition fixture count mismatch")
    frozen_episodes = tuple(
        tuple(MappingProxyType(dict(row)) for row in episode)
        for episode in episodes
    )
    manifest = {
        "schema_version": A25_FULL_NUMERICAL_FIXTURE_VERSION,
        **counts,
        "unsafe_cost_per_episode": 0.75,
        "wrong_cost_per_fixed_mean_incident": 0.25,
        "mean_incidents": A25_MEAN_INCIDENTS,
        "transition_sha256": _typed_sha256(frozen_episodes),
    }
    validate_common_transition_fixture(frozen_episodes, manifest)
    return frozen_episodes, manifest


def _warmup_batch(model: ActorCritic) -> dict[str, np.ndarray]:
    observations = np.stack(
        [_observation(0, index % 4, index // 4, "pass") for index in range(8)]
    ).astype(np.float32, copy=False)
    masks = np.ones((8, ACTION_COUNT), dtype=np.bool_)
    masks[:, int(RLAction.CONNECT)] = False
    actions = np.asarray(
        [
            RLAction.OBSERVE,
            RLAction.DELEGATE,
            RLAction.ESCALATE,
            RLAction.REVIEW,
            RLAction.EXECUTE,
            RLAction.HUMAN,
            RLAction.STOP,
            RLAction.OBSERVE,
        ],
        dtype=np.int64,
    )
    with torch.no_grad():
        logits, values = model(torch.from_numpy(observations.copy()))
        distribution = _masked_distribution(logits, torch.from_numpy(masks.copy()))
        old_log_probability = distribution.log_prob(torch.from_numpy(actions.copy()))
    advantage = np.asarray([-1.4, -0.7, -0.2, 0.4, 1.1, 0.8, -0.5, 0.5], dtype=np.float32)
    target_delta = np.asarray([0.3, -0.4, 0.8, -0.2, 1.0, -0.7, 0.5, -0.9], dtype=np.float32)
    result = {
        "observation": observations,
        "mask": masks,
        "action": actions,
        "old_log_probability": old_log_probability.detach().numpy().astype(np.float32),
        "old_value": values.detach().numpy().astype(np.float32),
        "advantage": advantage,
        "return": values.detach().numpy().astype(np.float32) + target_delta,
    }
    if not all(np.isfinite(value).all() for value in result.values()):
        raise FloatingPointError("non-finite A25 warmup fixture")
    return result


def _warm_optimizer_manifest(
    model: ActorCritic, optimizer: torch.optim.Optimizer
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name, parameter in model.named_parameters():
        state = optimizer.state.get(parameter)
        if not isinstance(state, Mapping) or set(state) != {
            "step",
            "exp_avg",
            "exp_avg_sq",
        }:
            raise RuntimeError("A25 warm Adam state is incomplete")
        step = float(torch.as_tensor(state["step"]).item())
        exp_avg = torch.as_tensor(state["exp_avg"])
        exp_avg_sq = torch.as_tensor(state["exp_avg_sq"])
        if (
            step < 1.0
            or exp_avg.shape != parameter.shape
            or exp_avg_sq.shape != parameter.shape
            or not bool(torch.isfinite(exp_avg).all())
            or not bool(torch.isfinite(exp_avg_sq).all())
        ):
            raise RuntimeError("A25 warm Adam state is invalid")
        rows.append(
            {
                "name": name,
                "step": step,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "exp_avg_nonzero": bool(torch.count_nonzero(exp_avg).item()),
                "exp_avg_sq_nonzero": bool(torch.count_nonzero(exp_avg_sq).item()),
                "exp_avg_l2_norm": float(
                    torch.linalg.vector_norm(exp_avg.detach().double()).item()
                ),
                "exp_avg_sq_l2_norm": float(
                    torch.linalg.vector_norm(exp_avg_sq.detach().double()).item()
                ),
                "exp_avg_sha256": hashlib.sha256(
                    np.ascontiguousarray(
                        exp_avg.detach().cpu().numpy(), dtype=np.dtype("<f4")
                    ).tobytes(order="C")
                ).hexdigest(),
                "exp_avg_sq_sha256": hashlib.sha256(
                    np.ascontiguousarray(
                        exp_avg_sq.detach().cpu().numpy(), dtype=np.dtype("<f4")
                    ).tobytes(order="C")
                ).hexdigest(),
            }
        )
    if not rows or not all(
        row["exp_avg_nonzero"] and row["exp_avg_sq_nonzero"] for row in rows
    ):
        raise RuntimeError("A25 warm Adam moments are not nonzero for every parameter")
    return {"parameter_states": rows, "all_parameter_moments_nonzero": True}


def build_warm_common_state() -> tuple[
    ActorCritic,
    torch.optim.Optimizer,
    dict[str, Any],
]:
    """Create the common model and a nonempty, nonzero one-step Adam state."""

    model = _fixture_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-5)
    batch = _warmup_batch(model)
    generator = torch.Generator(device="cpu").manual_seed(A25_WARMUP_SEED)
    generator_before = generator_state_sha256(generator)
    metrics = lagrangian_ppo_update(
        model,
        optimizer,
        batch,
        _warmup_config(),
        torch.device("cpu"),
        generator,
    )
    generator_after = generator_state_sha256(generator)
    optimizer_manifest = _warm_optimizer_manifest(model, optimizer)
    manifest = {
        "schema_version": "multitown-a25-warm-adam-common-state-v1",
        "seed": A25_FULL_NUMERICAL_SEED,
        "warmup_seed": A25_WARMUP_SEED,
        "warmup_batch_sha256": array_mapping_sha256(batch),
        "warmup_generator_before_sha256": generator_before,
        "warmup_generator_after_sha256": generator_after,
        "model_sha256": model_parameter_sha256(model),
        "optimizer_sha256": optimizer_state_sha256(optimizer, model),
        "optimizer": optimizer_manifest,
        "metrics": metrics,
    }
    if generator_before == generator_after or not all(
        math.isfinite(float(value)) for value in metrics.values()
    ):
        raise RuntimeError("A25 warmup did not consume its explicit generator")
    return model, optimizer, manifest


def build_common_batches(
    episodes: Sequence[Sequence[Mapping[str, Any]]],
    config: PPOConfig,
) -> dict[str, dict[str, Any]]:
    """Build all three stress batches through production A22/A23 builders."""

    profiles = {
        "reward": {
            "dual": DualState(0.0, 0.0),
            "thresholds": SafetyThresholds(1.0, 0.25, A25_MEAN_INCIDENTS),
        },
        "unsafe": {
            "dual": DualState(1.0, 0.0),
            "thresholds": SafetyThresholds(0.5, 0.30, A25_MEAN_INCIDENTS),
        },
        "wrong": {
            "dual": DualState(0.0, 1.0),
            "thresholds": SafetyThresholds(1.0, 0.20, A25_MEAN_INCIDENTS),
        },
    }
    result: dict[str, dict[str, Any]] = {}
    for expected_mode, profile in profiles.items():
        decision = select_actor_mode(
            unsafe_events=36,
            wrong_executions=48,
            episodes=A25_EPISODE_COUNT,
            thresholds=profile["thresholds"],
        )
        if decision.mode != expected_mode:
            raise RuntimeError("A25 synthetic selector mode mismatch")
        lagrangian = lagrangian_batch(
            episodes, config, dual=profile["dual"]
        )
        cr, cr_telemetry = cr_ppo_batch(episodes, config, decision=decision)
        result[expected_mode] = {
            "dual": asdict(profile["dual"]),
            "thresholds": asdict(profile["thresholds"]),
            "decision": asdict(decision),
            "cr_telemetry": cr_telemetry,
            "lagrangian": shield_aware_batch(episodes, lagrangian),
            "cr": shield_aware_batch(episodes, cr),
        }
    return result


def _clone_gradients(
    named_parameters: tuple[tuple[str, torch.nn.Parameter], ...],
) -> tuple[torch.Tensor | None, ...]:
    gradients: list[torch.Tensor | None] = []
    for _, parameter in named_parameters:
        gradient = parameter.grad
        if gradient is None:
            gradients.append(None)
        else:
            if gradient.dtype != torch.float32 or not bool(
                torch.isfinite(gradient).all()
            ):
                raise FloatingPointError("invalid A25 captured gradient")
            gradients.append(gradient.detach().cpu().clone())
    return tuple(gradients)


class _GradientPayloadBuilder:
    def __init__(self) -> None:
        self.payload = bytearray()
        self.manifest: list[dict[str, Any]] = []

    def append(self, *, context: str, name: str, gradient: torch.Tensor) -> dict[str, Any]:
        array = np.ascontiguousarray(gradient.numpy(), dtype=np.dtype("<f4"))
        value = array.tobytes(order="C")
        offset = len(self.payload)
        self.payload.extend(value)
        row = {
            "context": context,
            "name": name,
            "dtype": "<f4",
            "shape": list(gradient.shape),
            "offset": offset,
            "nbytes": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
        self.manifest.append(row)
        return row

    def finalize(self) -> tuple[bytes, dict[str, Any]]:
        payload = bytes(self.payload)
        expected_offset = 0
        for row in self.manifest:
            if row["offset"] != expected_offset or row["nbytes"] <= 0:
                raise RuntimeError("invalid A25 gradient payload coverage")
            expected_offset += int(row["nbytes"])
        if expected_offset != len(payload) or not payload:
            raise RuntimeError("incomplete A25 gradient payload")
        return payload, {
            "schema_version": "multitown-a25-gradient-payload-v1",
            "encoding": "contiguous-little-endian-float32",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "nbytes": len(payload),
            "entry_count": len(self.manifest),
            "manifest": self.manifest,
        }


def _gradient_manifest(
    named_parameters: tuple[tuple[str, torch.nn.Parameter], ...],
    gradients: tuple[torch.Tensor | None, ...],
    *,
    context: str,
    payload_builder: _GradientPayloadBuilder,
) -> dict[str, Any]:
    if len(named_parameters) != len(gradients):
        raise ValueError("invalid A25 named-gradient capture")
    digest = hashlib.sha256()
    squared_norm = 0.0
    rows: list[dict[str, Any]] = []
    for (name, parameter), gradient in zip(
        named_parameters, gradients, strict=True
    ):
        digest.update(name.encode("utf-8") + b"\0")
        if gradient is None:
            digest.update(b"none\0")
            rows.append({"name": name, "state": "none", "sha256": None})
            continue
        if gradient.shape != parameter.shape or gradient.dtype != torch.float32:
            raise ValueError("invalid A25 captured gradient binding")
        payload_row = payload_builder.append(
            context=context, name=name, gradient=gradient
        )
        payload = bytes(
            payload_builder.payload[
                payload_row["offset"] : payload_row["offset"] + payload_row["nbytes"]
            ]
        )
        squared_norm += float(
            gradient.to(dtype=torch.float64).square().sum().item()
        )
        digest.update(b"tensor\0" + payload)
        rows.append(
            {
                "name": name,
                "state": "tensor",
                "shape": list(gradient.shape),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "nbytes": len(payload),
                "payload_offset": payload_row["offset"],
            }
        )
    return {
        "digest_sha256": digest.hexdigest(),
        "l2_norm": math.sqrt(squared_norm),
        "parameters": rows,
    }


def _gradient_error_gate(
    observed: torch.Tensor, reference: torch.Tensor
) -> tuple[bool, float, float]:
    if (
        observed.shape != reference.shape
        or observed.dtype != torch.float32
        or reference.dtype != torch.float32
        or not bool(torch.isfinite(observed).all())
        or not bool(torch.isfinite(reference).all())
    ):
        raise ValueError("invalid A25 gradient comparison")
    observed64 = observed.to(dtype=torch.float64)
    reference64 = reference.to(dtype=torch.float64)
    delta = (observed64 - reference64).abs()
    allowed = torch.full_like(delta, 2e-6)
    relative = reference64.abs() >= 1e-8
    allowed[relative] += 2e-5 * reference64[relative].abs()
    return (
        bool(torch.all(delta <= allowed)),
        float(delta.max().item()) if delta.numel() else 0.0,
        float(allowed.max().item()) if allowed.numel() else 2e-6,
    )


def _flatten_gradients(
    named_parameters: tuple[tuple[str, torch.nn.Parameter], ...],
    gradients: tuple[torch.Tensor | None, ...],
) -> torch.Tensor:
    values = [
        (
            torch.zeros_like(parameter, device="cpu").reshape(-1)
            if gradient is None
            else gradient.reshape(-1)
        )
        for (_, parameter), gradient in zip(
            named_parameters, gradients, strict=True
        )
    ]
    if not values:
        raise ValueError("empty A25 captured gradient")
    return torch.cat(values).to(dtype=torch.float64)


def _summarize_captured_step(
    *,
    key: tuple[int, int],
    indices_sha256: str,
    named_parameters: tuple[tuple[str, torch.nn.Parameter], ...],
    beta: float,
    base: tuple[torch.Tensor | None, ...],
    auxiliary: tuple[torch.Tensor | None, ...] | None,
    total: tuple[torch.Tensor | None, ...],
    preclip: tuple[torch.Tensor | None, ...],
    postclip: tuple[torch.Tensor | None, ...],
    clip_returned_norm: float,
    max_grad_norm: float,
    cell_id: str,
    payload_builder: _GradientPayloadBuilder,
) -> dict[str, Any]:
    decomposition_pass = True
    preclip_pass = True
    max_decomposition_error = 0.0
    max_decomposition_allowed = 0.0
    max_preclip_error = 0.0
    for index, (_, parameter) in enumerate(named_parameters):
        zero = torch.zeros_like(parameter, device="cpu")
        base_value = zero if base[index] is None else base[index]
        total_value = zero if total[index] is None else total[index]
        preclip_value = zero if preclip[index] is None else preclip[index]
        if auxiliary is None:
            if beta != 0.0:
                raise ValueError("missing A25 positive-beta auxiliary gradient")
            expected = base_value
        else:
            auxiliary_value = zero if auxiliary[index] is None else auxiliary[index]
            expected = base_value + float(beta) * auxiliary_value
        passed, error, allowed = _gradient_error_gate(total_value, expected)
        decomposition_pass = decomposition_pass and passed
        max_decomposition_error = max(max_decomposition_error, error)
        max_decomposition_allowed = max(max_decomposition_allowed, allowed)
        passed, error, _ = _gradient_error_gate(preclip_value, total_value)
        preclip_pass = preclip_pass and passed
        max_preclip_error = max(max_preclip_error, error)
    preclip_flat = _flatten_gradients(named_parameters, preclip)
    postclip_flat = _flatten_gradients(named_parameters, postclip)
    preclip_norm = float(torch.linalg.vector_norm(preclip_flat).item())
    postclip_norm = float(torch.linalg.vector_norm(postclip_flat).item())
    denominator = preclip_norm * postclip_norm
    cosine = (
        float(torch.dot(preclip_flat, postclip_flat).item() / denominator)
        if denominator > 0.0
        else None
    )
    returned_error = abs(clip_returned_norm - preclip_norm)
    returned_allowed = 2e-6 + (
        2e-5 * abs(preclip_norm) if abs(preclip_norm) >= 1e-8 else 0.0
    )
    post_bound = max_grad_norm * (1.0 + 2e-6) + 1e-8
    clip_expected = preclip_norm > max_grad_norm
    gates = {
        "decomposition": decomposition_pass,
        "total_matches_actual_preclip": preclip_pass,
        "returned_norm_matches_preclip": returned_error <= returned_allowed,
        "postclip_norm_bounded": postclip_norm <= post_bound,
        "clip_direction": (
            cosine is not None and cosine >= 1.0 - 2e-6
            if clip_expected
            else True
        ),
    }
    captured = {
        name: _gradient_manifest(
            named_parameters,
            values,
            context=f"{cell_id}/epoch-{key[0]}/minibatch-{key[1]}/{name}",
            payload_builder=payload_builder,
        )
        for name, values in {
            "g_base": base,
            "g_total": total,
            "preclip": preclip,
            "postclip": postclip,
        }.items()
    }
    captured["g_aux_actual"] = (
        None
        if auxiliary is None
        else _gradient_manifest(
            named_parameters,
            auxiliary,
            context=(
                f"{cell_id}/epoch-{key[0]}/minibatch-{key[1]}/g_aux_actual"
            ),
            payload_builder=payload_builder,
        )
    )
    captured["g_aux_actual_reason"] = (
        A25_AUX_NOT_OBSERVED_REASON if auxiliary is None else None
    )
    return {
        "epoch": key[0],
        "minibatch": key[1],
        "indices_sha256": indices_sha256,
        "gradients": captured,
        "decomposition_max_abs_error": max_decomposition_error,
        "decomposition_max_allowed_error": max_decomposition_allowed,
        "preclip_total_max_abs_error": max_preclip_error,
        "clip": {
            "max_grad_norm": max_grad_norm,
            "returned_preclip_norm": clip_returned_norm,
            "observed_preclip_norm": preclip_norm,
            "observed_postclip_norm": postclip_norm,
            "preclip_postclip_cosine": cosine,
        },
        "gates": gates,
        "passed": all(gates.values()),
    }


class _FullNumericalObserver:
    def __init__(
        self,
        *,
        beta: float,
        max_grad_norm: float,
        cell_id: str,
        payload_builder: _GradientPayloadBuilder,
    ) -> None:
        self.beta = beta
        self.max_grad_norm = max_grad_norm
        self.cell_id = cell_id
        self.payload_builder = payload_builder
        self.steps: list[dict[str, Any]] = []
        self._pending: dict[tuple[int, int], dict[str, Any]] = {}

    def observe(self, event: NumericalGradientEvent) -> None:
        key = (event.epoch_index, event.minibatch_index)
        parameters = tuple(parameter for _, parameter in event.named_parameters)
        if event.stage == "loss_terms":
            if event.loss_terms is None or key in self._pending:
                raise RuntimeError("invalid A25 loss-term observer event")
            terms = event.loss_terms
            component_gradients = {
                "base": torch.autograd.grad(
                    terms.base_total_loss,
                    parameters,
                    retain_graph=True,
                    allow_unused=True,
                    materialize_grads=False,
                ),
                "auxiliary": torch.autograd.grad(
                    terms.auxiliary_loss,
                    parameters,
                    retain_graph=True,
                    allow_unused=True,
                    materialize_grads=False,
                ),
                "total": torch.autograd.grad(
                    terms.total_loss,
                    parameters,
                    retain_graph=True,
                    allow_unused=True,
                    materialize_grads=False,
                ),
            }
            self._pending[key] = {
                "indices_sha256": event.indices_sha256,
                "named_parameters": event.named_parameters,
                **{
                    name: tuple(
                        None if value is None else value.detach().cpu().clone()
                        for value in values
                    )
                    for name, values in component_gradients.items()
                },
            }
            return
        pending = self._pending.get(key)
        if pending is None or event.indices_sha256 != pending["indices_sha256"]:
            raise RuntimeError("invalid A25 gradient-stage observer event")
        if event.stage == "preclip":
            if "preclip" in pending or event.gradient_summary is None:
                raise RuntimeError("invalid A25 preclip observer event")
            pending["preclip"] = _clone_gradients(event.named_parameters)
            return
        if event.stage != "postclip" or event.clip_returned_norm is None:
            raise RuntimeError("invalid A25 postclip observer event")
        if "preclip" not in pending or event.gradient_summary is None:
            raise RuntimeError("A25 postclip arrived before preclip")
        postclip = _clone_gradients(event.named_parameters)
        self.steps.append(
            _summarize_captured_step(
                key=key,
                indices_sha256=event.indices_sha256,
                named_parameters=event.named_parameters,
                beta=self.beta,
                base=pending["base"],
                auxiliary=pending["auxiliary"],
                total=pending["total"],
                preclip=pending["preclip"],
                postclip=postclip,
                clip_returned_norm=event.clip_returned_norm,
                max_grad_norm=self.max_grad_norm,
                cell_id=self.cell_id,
                payload_builder=self.payload_builder,
            )
        )
        del self._pending[key]

    def finalize(self, *, expected_steps: int) -> list[dict[str, Any]]:
        if self._pending or len(self.steps) != expected_steps:
            raise RuntimeError("incomplete A25 numerical observer capture")
        return self.steps


class _A22ProductionGradientCapture:
    """Capture the frozen A22 backward/clip path without editing A22 source."""

    def __init__(
        self,
        *,
        model: ActorCritic,
        generator: torch.Generator,
        count: int,
        config: PPOConfig,
        cell_id: str,
        payload_builder: _GradientPayloadBuilder,
    ) -> None:
        self.model = model
        self.generator = generator
        self.count = count
        self.config = config
        self.cell_id = cell_id
        self.payload_builder = payload_builder
        self.named_parameters = tuple(model.named_parameters())
        replay_generator = torch.Generator(device="cpu")
        replay_generator.set_state(generator.get_state().clone())
        self.orders = [
            torch.randperm(count, generator=replay_generator, device="cpu")
            for _ in range(config.ppo_epochs)
        ]
        self.expected_generator_post = replay_generator.get_state().clone()
        self.steps: list[dict[str, Any]] = []

    @staticmethod
    def _indices_sha256(indices: torch.Tensor) -> str:
        array = np.ascontiguousarray(
            indices.detach().to(device="cpu", dtype=torch.int64).numpy(),
            dtype=np.dtype("<i8"),
        )
        digest = hashlib.sha256()
        digest.update(b"multitown-a25-a22-production-gradient-capture-v1\0")
        digest.update(str(len(array)).encode("ascii") + b"\0")
        digest.update(array.tobytes(order="C"))
        return digest.hexdigest()

    @contextmanager
    def instrument(self) -> Iterator[None]:
        original_clip = torch.nn.utils.clip_grad_norm_
        expected_parameters = tuple(parameter for _, parameter in self.named_parameters)
        minibatches_per_epoch = math.ceil(
            self.count / self.config.minibatch_size
        )
        owner_thread = threading.get_ident()
        if not _A22_CAPTURE_LOCK.acquire(blocking=False):
            raise RuntimeError("concurrent A22 production capture is forbidden")

        def captured_clip(
            parameters: Any, max_norm: Any, *args: Any, **kwargs: Any
        ) -> torch.Tensor:
            if threading.get_ident() != owner_thread:
                raise RuntimeError("cross-thread A22 production capture is forbidden")
            observed_parameters = tuple(parameters)
            if (
                len(observed_parameters) != len(expected_parameters)
                or any(
                    observed is not expected
                    for observed, expected in zip(
                        observed_parameters, expected_parameters, strict=True
                    )
                )
                or float(max_norm) != float(self.config.max_grad_norm)
            ):
                raise RuntimeError("unexpected clip call inside A22 capture")
            step_index = len(self.steps)
            epoch_index, minibatch_index = divmod(
                step_index, minibatches_per_epoch
            )
            if epoch_index >= len(self.orders):
                raise RuntimeError("A22 clip occurred before its permutation")
            start = minibatch_index * self.config.minibatch_size
            indices = self.orders[epoch_index][
                start : start + self.config.minibatch_size
            ]
            preclip = _clone_gradients(self.named_parameters)
            returned = original_clip(
                observed_parameters, max_norm, *args, **kwargs
            )
            postclip = _clone_gradients(self.named_parameters)
            returned_norm = float(
                returned.item() if torch.is_tensor(returned) else returned
            )
            if not math.isfinite(returned_norm):
                raise FloatingPointError("non-finite A22 production clip norm")
            self.steps.append(
                _summarize_captured_step(
                    key=(epoch_index, minibatch_index),
                    indices_sha256=self._indices_sha256(indices),
                    named_parameters=self.named_parameters,
                    beta=0.0,
                    base=preclip,
                    auxiliary=None,
                    total=preclip,
                    preclip=preclip,
                    postclip=postclip,
                    clip_returned_norm=returned_norm,
                    max_grad_norm=float(self.config.max_grad_norm),
                    cell_id=self.cell_id,
                    payload_builder=self.payload_builder,
                )
            )
            return returned

        try:
            with patch.object(
                torch.nn.utils, "clip_grad_norm_", new=captured_clip
            ):
                yield
        finally:
            _A22_CAPTURE_LOCK.release()

    def finalize(self, *, expected_steps: int) -> list[dict[str, Any]]:
        if (
            len(self.orders) != self.config.ppo_epochs
            or len(self.steps) != expected_steps
            or not torch.equal(
                self.generator.get_state(), self.expected_generator_post
            )
        ):
            raise RuntimeError("incomplete A22 production gradient capture")
        return self.steps


def _base_update_batch(batch: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.asarray(batch[key]).copy() for key in _BASE_BATCH_FIELDS}


def _ppo_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    keys = ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction")
    result = {key: float(metrics[key]) for key in keys}
    if not all(math.isfinite(value) for value in result.values()):
        raise FloatingPointError("invalid A25 PPO metrics")
    return result


def _observed_training_prestate(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
) -> dict[str, str]:
    """Measure the actual state presented to one diagnostic update."""

    return {
        "model_sha256": model_parameter_sha256(model),
        "optimizer_sha256": optimizer_state_sha256(optimizer, model),
        "rng_sha256": generator_state_sha256(generator),
    }


def _build_twelve_cell_artifacts() -> tuple[dict[str, Any], bytes]:
    """Build the receipt and its canonical binary gradient payload."""

    common_model, common_optimizer, warm_manifest = build_warm_common_state()
    episodes, transition_manifest = build_common_transition_fixture(common_model)
    config = full_numerical_config()
    batches = build_common_batches(episodes, config)
    common_model_sha = model_parameter_sha256(common_model)
    common_optimizer_sha = optimizer_state_sha256(common_optimizer, common_model)
    common_update_rng = torch.Generator(device="cpu").manual_seed(A25_UPDATE_SEED)
    common_rng_sha = generator_state_sha256(common_update_rng)
    cells: dict[str, Any] = {}
    payload_builder = _GradientPayloadBuilder()
    arms = {
        "F00": ("lagrangian", 0.0),
        "F01": ("lagrangian", A25_COMMON_BETA),
        "F10": ("cr", 0.0),
        "F11": ("cr", A25_COMMON_BETA),
    }
    for stress, panel in batches.items():
        for arm, (rule, beta) in arms.items():
            cell_id = f"{stress}/{arm}"
            model, optimizer = copy.deepcopy((common_model, common_optimizer))
            generator = torch.Generator(device="cpu")
            generator.set_state(common_update_rng.get_state().clone())
            observed_pre = _observed_training_prestate(model, optimizer, generator)
            gates = {
                "common_pre_model": observed_pre["model_sha256"]
                == common_model_sha,
                "common_pre_optimizer": observed_pre["optimizer_sha256"]
                == common_optimizer_sha,
                "common_pre_rng": observed_pre["rng_sha256"] == common_rng_sha,
            }
            if not all(gates.values()):
                raise RuntimeError(
                    f"A25 cell {cell_id} observed prestate differs from common state"
                )
            observer = (
                _FullNumericalObserver(
                    beta=beta,
                    max_grad_norm=config.max_grad_norm,
                    cell_id=cell_id,
                    payload_builder=payload_builder,
                )
                if beta > 0.0
                else None
            )
            base_capture = (
                _A22ProductionGradientCapture(
                    model=model,
                    generator=generator,
                    count=A25_DECISION_COUNT,
                    config=config,
                    cell_id=cell_id,
                    payload_builder=payload_builder,
                )
                if beta == 0.0
                else None
            )
            if base_capture is None:
                metrics = intervention_ppo_update(
                    model,
                    optimizer,
                    panel[rule],
                    config,
                    torch.device("cpu"),
                    generator,
                    objective=InterventionObjective(beta=beta),
                    observer=observer,
                )
            else:
                with base_capture.instrument():
                    metrics = intervention_ppo_update(
                        model,
                        optimizer,
                        panel[rule],
                        config,
                        torch.device("cpu"),
                        generator,
                        objective=InterventionObjective(beta=beta),
                    )
            beta_zero_reference: dict[str, Any] | None = None
            steps: list[dict[str, Any]] = []
            if beta == 0.0:
                reference_model, reference_optimizer = copy.deepcopy(
                    (common_model, common_optimizer)
                )
                reference_generator = torch.Generator(device="cpu")
                reference_generator.set_state(common_update_rng.get_state().clone())
                reference_metrics = lagrangian_ppo_update(
                    reference_model,
                    reference_optimizer,
                    _base_update_batch(panel[rule]),
                    config,
                    torch.device("cpu"),
                    reference_generator,
                )
                reference = {
                    "post_model_sha256": model_parameter_sha256(reference_model),
                    "post_optimizer_sha256": optimizer_state_sha256(
                        reference_optimizer, reference_model
                    ),
                    "post_rng_sha256": generator_state_sha256(reference_generator),
                    "metrics": reference_metrics,
                }
                observed = {
                    "post_model_sha256": model_parameter_sha256(model),
                    "post_optimizer_sha256": optimizer_state_sha256(optimizer, model),
                    "post_rng_sha256": generator_state_sha256(generator),
                    "metrics": _ppo_metrics(metrics),
                }
                exact = observed == reference
                beta_zero_reference = {**reference, "exact": exact}
                gates["beta_zero_reference_exact"] = exact
                if base_capture is None:
                    raise AssertionError("missing A22 production gradient capture")
                steps = base_capture.finalize(expected_steps=config.ppo_epochs)
                gates["four_observed_optimizer_steps"] = len(steps) == 4
                gates["all_gradient_steps_passed"] = all(
                    step["passed"] for step in steps
                )
            else:
                if observer is None:
                    raise AssertionError("missing A25 numerical observer")
                steps = observer.finalize(expected_steps=config.ppo_epochs)
                gates["four_observed_optimizer_steps"] = len(steps) == 4
                gates["all_gradient_steps_passed"] = all(
                    step["passed"] for step in steps
                )
            cells[cell_id] = {
                "cell_id": cell_id,
                "stress": stress,
                "arm": arm,
                "update_rule": "ppo-lagrangian" if rule == "lagrangian" else "cr-ppo",
                "actor_mode": None if rule == "lagrangian" else stress,
                "dual": panel["dual"] if rule == "lagrangian" else None,
                "cr_thresholds": panel["thresholds"] if rule == "cr" else None,
                "beta": beta,
                "batch_sha256": array_mapping_sha256(panel[rule]),
                "pre": observed_pre,
                "steps": steps,
                "beta_zero_reference": beta_zero_reference,
                "post": {
                    "model_sha256": model_parameter_sha256(model),
                    "optimizer_sha256": optimizer_state_sha256(optimizer, model),
                    "rng_sha256": generator_state_sha256(generator),
                },
                "metrics": metrics,
                "gates": gates,
                "passed": all(gates.values()),
            }
    expected_cells = {
        f"{stress}/{arm}"
        for stress in ("reward", "unsafe", "wrong")
        for arm in arms
    }
    top_gates = {
        "twelve_cells_complete": set(cells) == expected_cells,
        "all_cells_passed": all(cell["passed"] for cell in cells.values()),
        "beta_zero_base_gradient_capture_complete": all(
            len(cell["steps"]) == config.ppo_epochs
            and all(step["passed"] for step in cell["steps"])
            and all(
                step["gradients"]["g_aux_actual"] is None
                and step["gradients"]["g_aux_actual_reason"]
                == A25_AUX_NOT_OBSERVED_REASON
                for step in cell["steps"]
            )
            for cell in cells.values()
            if cell["beta"] == 0.0
        ),
        "all_cells_share_common_prestate": all(
            cell["pre"]
            == {
                "model_sha256": common_model_sha,
                "optimizer_sha256": common_optimizer_sha,
                "rng_sha256": common_rng_sha,
            }
            for cell in cells.values()
        ),
        "common_source_unmodified": (
            model_parameter_sha256(common_model) == common_model_sha
            and optimizer_state_sha256(common_optimizer, common_model)
            == common_optimizer_sha
            and generator_state_sha256(common_update_rng) == common_rng_sha
        ),
        "zero_outer_rows_read": True,
        "no_formal_lock_created": True,
    }
    if not all(top_gates.values()):
        raise RuntimeError("A25 twelve-cell diagnostic gate failed")
    gradient_payload, gradient_artifact = payload_builder.finalize()
    core = {
        "schema_version": A25_TWELVE_CELL_RECEIPT_VERSION,
        "scope": "synthetic-warm-adam-twelve-cell-diagnostic-no-outer-no-formal",
        "config": asdict(config),
        "warm_common_state": warm_manifest,
        "transition_fixture": transition_manifest,
        "common_state": {
            "model_sha256": common_model_sha,
            "optimizer_sha256": common_optimizer_sha,
            "update_generator_prestate_sha256": common_rng_sha,
        },
        "gradient_artifact": gradient_artifact,
        "cells": cells,
        "gates": top_gates,
        "claim_boundary": {
            "twelve_cell_gradient_diagnostic_passed": True,
            "beta_zero_base_gradient_capture_passed": True,
            "clean_source_bound": False,
            "gradient_sidecar_bound": False,
            "full_numerical_q0_qualified": False,
            "q1_mechanism_qualified": False,
            "formal_authorized": False,
            "performance_claim_supported": False,
            "safety_claim_supported": False,
            "outer_rows_read": 0,
        },
    }
    return {
        **core,
        "receipt_id": _sha256(core),
        "status": "DIAGNOSTIC_PASSED",
    }, gradient_payload


def run_twelve_cell_diagnostic() -> dict[str, Any]:
    """Run the warm-Adam 12-cell profile without claiming full qualification."""

    receipt, _ = _build_twelve_cell_artifacts()
    return receipt


def validate_gradient_payload(receipt: Mapping[str, Any], payload: bytes) -> None:
    """Validate exact sidecar coverage and its references from every step."""

    if not isinstance(receipt, Mapping) or type(payload) is not bytes:
        raise TypeError("invalid A25 gradient artifact input")
    artifact = receipt.get("gradient_artifact")
    if not isinstance(artifact, Mapping):
        raise TypeError("missing A25 gradient artifact manifest")
    manifest = artifact.get("manifest")
    if (
        artifact.get("schema_version") != "multitown-a25-gradient-payload-v1"
        or artifact.get("encoding") != "contiguous-little-endian-float32"
        or type(manifest) is not list
        or artifact.get("entry_count") != len(manifest)
        or artifact.get("nbytes") != len(payload)
        or artifact.get("sha256") != hashlib.sha256(payload).hexdigest()
    ):
        raise ValueError("invalid A25 gradient artifact identity")
    expected_offset = 0
    indexed: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in manifest:
        if not isinstance(row, Mapping):
            raise TypeError("invalid A25 gradient artifact row")
        context = row.get("context")
        name = row.get("name")
        shape = row.get("shape")
        offset = row.get("offset")
        nbytes = row.get("nbytes")
        if (
            type(context) is not str
            or not context
            or type(name) is not str
            or not name
            or row.get("dtype") != "<f4"
            or type(shape) is not list
            or not shape
            or any(type(value) is not int or value <= 0 for value in shape)
            or type(offset) is not int
            or offset != expected_offset
            or type(nbytes) is not int
            or nbytes != math.prod(shape) * 4
            or offset + nbytes > len(payload)
            or row.get("sha256")
            != hashlib.sha256(payload[offset : offset + nbytes]).hexdigest()
            or (context, name) in indexed
        ):
            raise ValueError("invalid A25 gradient artifact row binding")
        indexed[(context, name)] = row
        expected_offset += nbytes
    if expected_offset != len(payload):
        raise ValueError("incomplete A25 gradient artifact coverage")
    referenced: set[tuple[str, str]] = set()
    cells = receipt.get("cells")
    if not isinstance(cells, Mapping):
        raise TypeError("invalid A25 gradient cell binding")
    for cell_id, cell in cells.items():
        if type(cell_id) is not str or not isinstance(cell, Mapping):
            raise ValueError("invalid A25 gradient cell")
        steps = cell.get("steps")
        if type(steps) is not list:
            raise ValueError("invalid A25 gradient step list")
        for step in steps:
            if not isinstance(step, Mapping):
                raise TypeError("invalid A25 gradient step")
            epoch = step.get("epoch")
            minibatch = step.get("minibatch")
            gradients = step.get("gradients")
            if (
                type(epoch) is not int
                or type(minibatch) is not int
                or not isinstance(gradients, Mapping)
            ):
                raise ValueError("invalid A25 gradient step binding")
            if set(gradients) != {
                "g_base",
                "g_aux_actual",
                "g_aux_actual_reason",
                "g_total",
                "preclip",
                "postclip",
            }:
                raise ValueError("invalid A25 gradient kind schema")
            auxiliary = gradients["g_aux_actual"]
            auxiliary_reason = gradients["g_aux_actual_reason"]
            if (auxiliary is None) != (auxiliary_reason is not None):
                raise ValueError("invalid A25 auxiliary observation boundary")
            if (
                auxiliary_reason is not None
                and auxiliary_reason != A25_AUX_NOT_OBSERVED_REASON
            ):
                raise ValueError("invalid A25 auxiliary observation reason")
            summaries = {
                kind: gradients[kind]
                for kind in ("g_base", "g_total", "preclip", "postclip")
            }
            if auxiliary is not None:
                summaries["g_aux_actual"] = auxiliary
            for kind, summary in summaries.items():
                if not isinstance(summary, Mapping):
                    raise TypeError("invalid A25 gradient summary")
                rows = summary.get("parameters")
                if type(rows) is not list:
                    raise ValueError("invalid A25 gradient parameter list")
                context = f"{cell_id}/epoch-{epoch}/minibatch-{minibatch}/{kind}"
                for parameter in rows:
                    if not isinstance(parameter, Mapping):
                        raise TypeError("invalid A25 gradient parameter")
                    if parameter.get("state") == "none":
                        continue
                    key = (context, parameter.get("name"))
                    bound = indexed.get(key)
                    if (
                        bound is None
                        or parameter.get("payload_offset") != bound.get("offset")
                        or parameter.get("nbytes") != bound.get("nbytes")
                        or parameter.get("sha256") != bound.get("sha256")
                    ):
                        raise ValueError("A25 gradient step differs from sidecar")
                    referenced.add(key)
    if referenced != set(indexed):
        raise ValueError("A25 gradient sidecar has unreferenced entries")


def _isolation_observation(index: int, review: str) -> np.ndarray:
    if type(index) is not int or not 0 <= index < 96:
        raise ValueError("invalid A25 isolation row")
    row = np.zeros(MultiTownLongHorizonEnv.observation_size, dtype=np.float32)
    for feature in range(len(row)):
        row[feature] = np.float32(
            (((index + 1) * (feature + 5)) % 31 - 15) / 1000.0
        )
    row[33:36] = _review_vector(review)
    row.setflags(write=False)
    return row


def _isolation_model() -> ActorCritic:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026081504)
        model = ActorCritic(
            MultiTownLongHorizonEnv.observation_size, 8, ACTION_COUNT
        ).cpu()
    with torch.no_grad():
        model.critic.weight.zero_()
        model.critic.bias.zero_()
    return model


def _build_isolation_episodes(
    model: ActorCritic,
) -> list[list[dict[str, Any]]]:
    episodes: list[list[dict[str, Any]]] = []
    with torch.no_grad():
        for episode_index in range(48):
            rows: list[dict[str, Any]] = []
            for step, (review, mask_kind, action, unsafe, wrong) in enumerate(
                (
                    ("unknown", "active", RLAction.REVIEW, 0.0, 0.0),
                    ("pass", "pass", RLAction.EXECUTE, 1.0, 0.25),
                )
            ):
                observation = _isolation_observation(
                    episode_index * 2 + step, review
                )
                base_mask = _base_mask(mask_kind)
                mask, active = effective_action_mask(
                    observation, base_mask, shield_enabled=True
                )
                observation_tensor = torch.from_numpy(
                    np.asarray(observation).copy()
                ).unsqueeze(0)
                logits, value = model(observation_tensor)
                if value.item() != 0.0:
                    raise RuntimeError("A25 isolation critic must be exact zero")
                effective_distribution = _masked_distribution(
                    logits,
                    torch.from_numpy(np.asarray(mask).copy()).unsqueeze(0),
                )
                base_distribution = _masked_distribution(
                    logits,
                    torch.from_numpy(np.asarray(base_mask).copy()).unsqueeze(0),
                )
                action_index = int(action)
                logits_array = logits[0].numpy().astype(np.float32, copy=True)
                logits_array.setflags(write=False)
                rows.append(
                    {
                        "observation": observation,
                        "base_mask": base_mask,
                        "mask": mask,
                        "action": action_index,
                        "old_log_probability": float(
                            effective_distribution.log_prob(
                                torch.tensor([action_index])
                            ).item()
                        ),
                        "old_value": 0.0,
                        "reward": 0.0,
                        "unsafe_cost": unsafe,
                        "wrong_cost": wrong,
                        "wrong_execute": bool(step == 1),
                        "shield_intervened": bool(active),
                        "pre_shield_logits": logits_array,
                        "base_execute_probability": float(
                            base_distribution.probs[
                                0, int(RLAction.EXECUTE)
                            ].item()
                        ),
                    }
                )
            episodes.append(rows)
    return episodes


def _build_isolation_artifacts() -> tuple[dict[str, Any], bytes]:
    """Build G3 and its separate canonical gradient payload."""

    model = _isolation_model()
    episodes = _build_isolation_episodes(model)
    config = PPOConfig(
        **{
            **asdict(full_numerical_config()),
            "ppo_epochs": 1,
            "value_coef": 0.0,
            "entropy_coef": 0.0,
            "max_grad_norm": 100.0,
        }
    )
    base = lagrangian_batch(
        episodes,
        config,
        dual=DualState(unsafe=1.0, wrong_per_incident=1.0),
    )
    if not np.array_equal(
        base["advantage"], np.zeros(96, dtype=np.float32)
    ) or not np.array_equal(base["return"], np.zeros(96, dtype=np.float32)):
        raise RuntimeError("A25 isolation base objective is not exact zero")
    batch = shield_aware_batch(episodes, base)
    source_model_sha = model_parameter_sha256(model)

    beta_zero_model = copy.deepcopy(model)
    beta_zero_optimizer = torch.optim.Adam(
        beta_zero_model.parameters(), lr=1e-3, eps=1e-5
    )
    beta_zero_metrics = intervention_ppo_update(
        beta_zero_model,
        beta_zero_optimizer,
        batch,
        config,
        torch.device("cpu"),
        torch.Generator(device="cpu").manual_seed(2026081505),
        objective=InterventionObjective(beta=0.0),
    )

    regularized_model = copy.deepcopy(model)
    regularized_optimizer = torch.optim.Adam(
        regularized_model.parameters(), lr=1e-3, eps=1e-5
    )
    payload_builder = _GradientPayloadBuilder()
    observer = _FullNumericalObserver(
        beta=A25_COMMON_BETA,
        max_grad_norm=config.max_grad_norm,
        cell_id="isolation/G3",
        payload_builder=payload_builder,
    )
    regularized_metrics = intervention_ppo_update(
        regularized_model,
        regularized_optimizer,
        batch,
        config,
        torch.device("cpu"),
        torch.Generator(device="cpu").manual_seed(2026081505),
        objective=InterventionObjective(beta=A25_COMMON_BETA),
        observer=observer,
    )
    steps = observer.finalize(expected_steps=1)
    payload, artifact = payload_builder.finalize()
    step = steps[0]
    validate_gradient_payload(
        {
            "gradient_artifact": artifact,
            "cells": {"isolation/G3": {"steps": [step]}},
        },
        payload,
    )
    base_gradient = step["gradients"]["g_base"]
    auxiliary_gradient = step["gradients"]["g_aux_actual"]
    if auxiliary_gradient is None:
        raise RuntimeError("missing A25 G3 actual auxiliary gradient")
    critic_rows = [
        row
        for row in auxiliary_gradient["parameters"]
        if row["name"].startswith("critic.")
    ]
    mass_delta = (
        regularized_metrics["post_update_shield_dependence"][
            "shielded_execute_probability_mass_mean_all_decisions"
        ]
        - regularized_metrics["pre_update_shield_dependence"][
            "shielded_execute_probability_mass_mean_all_decisions"
        ]
    )
    gates = {
        "base_advantage_exact_zero": True,
        "base_return_exact_zero": True,
        "g_base_exact_zero": base_gradient["l2_norm"] == 0.0,
        "g_aux_nonzero": auxiliary_gradient["l2_norm"] > 0.0,
        "critic_direct_aux_exact_none": (
            len(critic_rows) == 2
            and all(row["state"] == "none" for row in critic_rows)
        ),
        "beta_zero_parameters_unchanged": (
            model_parameter_sha256(beta_zero_model) == source_model_sha
        ),
        "regularized_mass_delta_below_threshold": mass_delta < -1e-6,
        "gradient_step_passed": step["passed"],
        "payload_valid": True,
    }
    if not all(gates.values()):
        raise RuntimeError("A25 fresh-Adam isolation diagnostic failed")
    core = {
        "schema_version": "multitown-a25-g3-isolation-diagnostic-v1",
        "fixture": {
            "episodes": 48,
            "decisions": 96,
            "optimizer": "fresh-adam",
            "value_coef": 0.0,
            "entropy_coef": 0.0,
            "beta": A25_COMMON_BETA,
            "direction_threshold": -1e-6,
        },
        "gradient_step": step,
        "gradient_artifact": artifact,
        "beta_zero_metrics": beta_zero_metrics,
        "regularized_metrics": regularized_metrics,
        "blocked_execute_mass_delta": mass_delta,
        "gates": gates,
        "claim_boundary": {
            "fresh_adam_isolation_diagnostic_passed": True,
            "warm_adam_equivalence_inferred": False,
            "full_numerical_q0_qualified": False,
            "formal_authorized": False,
            "performance_claim_supported": False,
        },
    }
    return {
        **core,
        "receipt_id": _sha256(core),
        "status": "DIAGNOSTIC_PASSED",
    }, payload


def run_isolation_diagnostic() -> dict[str, Any]:
    """Run G3 with fresh Adam and exact-zero base actor gradients."""

    receipt, _ = _build_isolation_artifacts()
    return receipt
