"""PQ-1 candidate primitives for rollout-shaped on-policy conformance."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from .a22_constrained_ppo import (
    SafetyThresholds,
    lagrangian_ppo_update,
    validate_thresholds,
)
from .a23_cr_ppo import (
    ActorModeDecision,
    _finite_value,
    _optimizer_parameter_groups,
    _validate_ppo_batch,
    _validate_pre_update_state,
    _validate_transition_costs,
    cr_ppo_batch,
    model_parameter_sha256,
    select_actor_mode,
)
from .long_horizon_env import ACTION_COUNT, MultiTownLongHorizonEnv
from .ppo_controller import ActorCritic, PPOConfig, _masked_distribution

PQ1_PRIMITIVES_VERSION = "multitown-pq1-rowwise-on-policy-primitives-v1"
FULL_BATCH_RTOL = 1.3e-6
FULL_BATCH_ATOL = 1e-5
MAX_PROBABILITY_RATIO_DRIFT = 2e-5
LEGACY_RTOL = 1e-6
LEGACY_ATOL = 1e-7
PQ1_TRANSITION_FIELDS = {
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
    "done",
}


def _hash_part(digest: Any, label: str, payload: bytes) -> None:
    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    digest.update(payload)
    digest.update(b"\0")


def _hash_value(digest: Any, label: str, value: Any) -> None:
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise FloatingPointError("PQ-1 optimizer tensor is non-finite")
        array = tensor.numpy()
        little = np.ascontiguousarray(array, dtype=array.dtype.newbyteorder("<"))
        _hash_part(digest, f"{label}:tensor-dtype", little.dtype.str.encode("ascii"))
        _hash_part(
            digest,
            f"{label}:tensor-shape",
            json.dumps(list(little.shape), separators=(",", ":")).encode("ascii"),
        )
        _hash_part(digest, f"{label}:tensor-bytes", little.tobytes(order="C"))
        return
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value, dtype=value.dtype.newbyteorder("<"))
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
            raise FloatingPointError("PQ-1 optimizer array is non-finite")
        _hash_part(digest, f"{label}:array-dtype", array.dtype.str.encode("ascii"))
        _hash_part(
            digest,
            f"{label}:array-shape",
            json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"),
        )
        _hash_part(digest, f"{label}:array-bytes", array.tobytes(order="C"))
        return
    if isinstance(value, Mapping):
        keys = sorted(value, key=lambda item: f"{type(item).__name__}:{item}")
        _hash_part(
            digest,
            f"{label}:mapping-keys",
            json.dumps(
                [f"{type(item).__name__}:{item}" for item in keys],
                separators=(",", ":"),
            ).encode("utf-8"),
        )
        for key in keys:
            _hash_value(digest, f"{label}.{type(key).__name__}:{key}", value[key])
        return
    if isinstance(value, (list, tuple)):
        _hash_part(
            digest,
            f"{label}:sequence-type",
            type(value).__name__.encode("ascii"),
        )
        for index, item in enumerate(value):
            _hash_value(digest, f"{label}[{index}]", item)
        return
    if value is None or type(value) in {bool, int, float, str}:
        if type(value) is float and not math.isfinite(value):
            raise FloatingPointError("PQ-1 optimizer scalar is non-finite")
        _hash_part(
            digest,
            f"{label}:{type(value).__name__}",
            json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf-8"),
        )
        return
    raise TypeError(f"unsupported PQ-1 digest value: {label}={type(value)!r}")


def optimizer_state_sha256(
    optimizer: torch.optim.Optimizer,
    model: ActorCritic,
) -> str:
    """Hash the complete finite optimizer state with named model parameters."""

    groups = _optimizer_parameter_groups(optimizer, model)
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    state = {
        names[id(parameter)]: values for parameter, values in optimizer.state.items()
    }
    digest = hashlib.sha256()
    _hash_value(
        digest,
        "optimizer",
        {
            "class": f"{optimizer.__class__.__module__}.{optimizer.__class__.__qualname__}",
            "defaults": dict(optimizer.defaults),
            "param_groups": groups,
            "state": state,
        },
    )
    return digest.hexdigest()


def transition_episode_sha256(
    transitions: Sequence[Mapping[str, Any]],
) -> str:
    """Hash the complete ordered rollout payload before any batch transform."""

    if not transitions or any(type(row) is not dict for row in transitions):
        raise ValueError("invalid PQ-1 transition episode")
    for index, row in enumerate(transitions):
        observation = row.get("observation")
        base_mask = row.get("base_mask")
        mask = row.get("mask")
        scalar_fields = (
            "old_log_probability",
            "old_value",
            "reward",
            "unsafe_cost",
            "wrong_cost",
        )
        if (
            isinstance(observation, np.ndarray)
            and observation.dtype == np.float32
            and observation.shape == (MultiTownLongHorizonEnv.observation_size,)
            and not bool(np.isfinite(observation).all())
        ) or (
            all(type(row.get(field)) is float for field in scalar_fields)
            and any(not math.isfinite(row[field]) for field in scalar_fields)
        ):
            raise FloatingPointError("non-finite PQ-1 transition episode")
        if (
            set(row) != PQ1_TRANSITION_FIELDS
            or not isinstance(observation, np.ndarray)
            or observation.dtype != np.float32
            or observation.shape != (MultiTownLongHorizonEnv.observation_size,)
            or not isinstance(base_mask, np.ndarray)
            or base_mask.dtype != np.bool_
            or base_mask.shape != (ACTION_COUNT,)
            or not bool(base_mask.any())
            or not isinstance(mask, np.ndarray)
            or mask.dtype != np.bool_
            or mask.shape != base_mask.shape
            or not bool(mask.any())
            or not bool(np.array_equal(base_mask, mask))
            or type(row.get("action")) is not int
            or not 0 <= row["action"] < len(mask)
            or not bool(mask[row["action"]])
            or any(type(row.get(field)) is not float for field in scalar_fields)
            or type(row.get("wrong_execute")) is not bool
            or type(row.get("shield_intervened")) is not bool
            or row["shield_intervened"] is not False
            or type(row.get("done")) is not bool
            or row["done"] is not (index == len(transitions) - 1)
        ):
            raise ValueError("invalid PQ-1 transition episode schema")
    digest = hashlib.sha256()
    try:
        _hash_value(digest, "ordered-transition-episode", list(transitions))
    except TypeError as exc:
        raise ValueError("invalid PQ-1 transition episode schema") from exc
    return digest.hexdigest()


def _array_sha256(values: Any, *, dtype: np.dtype[Any]) -> str:
    array = np.ascontiguousarray(values, dtype=dtype)
    digest = hashlib.sha256()
    _hash_part(digest, "dtype", array.dtype.str.encode("ascii"))
    _hash_part(
        digest,
        "shape",
        json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"),
    )
    _hash_part(digest, "bytes", array.tobytes(order="C"))
    return digest.hexdigest()


def _ordered_float32(values: np.ndarray) -> np.ndarray:
    bits = np.asarray(values, dtype="<f4").view("<u4")
    sign = (bits & np.uint32(0x80000000)) != 0
    return np.where(
        sign,
        np.bitwise_not(bits),
        bits | np.uint32(0x80000000),
    ).astype(np.uint64)


def _drift_summary(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    actual_array = actual.detach().cpu().numpy().astype("<f4", copy=False)
    expected_array = expected.detach().cpu().numpy().astype("<f4", copy=False)
    absolute = np.abs(
        actual_array.astype(np.float64) - expected_array.astype(np.float64)
    )
    relative = absolute / np.maximum(np.abs(expected_array), np.finfo(np.float32).tiny)
    ulp = np.abs(
        _ordered_float32(actual_array).astype(np.int64)
        - _ordered_float32(expected_array).astype(np.int64)
    )
    legacy_tolerance = LEGACY_ATOL + LEGACY_RTOL * np.abs(expected_array)
    return {
        "max_abs": float(absolute.max(initial=0.0)),
        "p50_abs": float(np.quantile(absolute, 0.50, method="linear")),
        "p95_abs": float(np.quantile(absolute, 0.95, method="linear")),
        "p99_abs": float(np.quantile(absolute, 0.99, method="linear")),
        "max_relative": float(relative.max(initial=0.0)),
        "max_ulp": int(ulp.max(initial=0)),
        "legacy_exceedances": int(np.sum(absolute > legacy_tolerance)),
        "legacy_max_tolerance_ratio": float(
            np.max(absolute / legacy_tolerance, initial=0.0)
        ),
    }


def full_batch_numerical_conformance(
    actual_log: torch.Tensor,
    expected_log: torch.Tensor,
    actual_value: torch.Tensor,
    expected_value: torch.Tensor,
) -> tuple[dict[str, bool], float]:
    """Apply the frozen secondary numerical-conformance acceptance gates."""

    values = (actual_log, expected_log, actual_value, expected_value)
    if (
        any(not torch.is_tensor(value) for value in values)
        or actual_log.shape != expected_log.shape
        or actual_value.shape != expected_value.shape
        or actual_log.shape != actual_value.shape
        or any(not bool(torch.isfinite(value).all()) for value in values)
    ):
        raise FloatingPointError("invalid PQ-1 full-batch conformance tensors")
    probability_ratio_drift = float(
        torch.max(torch.abs(torch.exp(actual_log - expected_log) - 1.0)).item()
    )
    gates = {
        "full_batch_log_within_frozen_tolerance": bool(
            torch.allclose(
                actual_log,
                expected_log,
                rtol=FULL_BATCH_RTOL,
                atol=FULL_BATCH_ATOL,
            )
        ),
        "full_batch_value_within_frozen_tolerance": bool(
            torch.allclose(
                actual_value,
                expected_value,
                rtol=FULL_BATCH_RTOL,
                atol=FULL_BATCH_ATOL,
            )
        ),
        "probability_ratio_drift_within_2e_5": (
            probability_ratio_drift <= MAX_PROBABILITY_RATIO_DRIFT
        ),
    }
    return gates, probability_ratio_drift


def rollout_shaped_snapshot_diagnostics(
    model: ActorCritic,
    batch: Mapping[str, np.ndarray],
    device: torch.device,
) -> dict[str, Any]:
    """Require exact batch=1 replay plus secondary full-batch conformance."""

    observation = torch.as_tensor(
        batch["observation"],
        dtype=torch.float32,
        device=device,
    )
    mask = torch.as_tensor(batch["mask"], dtype=torch.bool, device=device)
    action = torch.as_tensor(batch["action"], dtype=torch.long, device=device)
    stored_log = torch.as_tensor(
        batch["old_log_probability"],
        dtype=torch.float32,
        device=device,
    )
    stored_value = torch.as_tensor(
        batch["old_value"],
        dtype=torch.float32,
        device=device,
    )
    row_log: list[torch.Tensor] = []
    row_value: list[torch.Tensor] = []
    with torch.no_grad():
        for index in range(len(observation)):
            logits, value = model(observation[index : index + 1])
            distribution = _masked_distribution(
                logits,
                mask[index : index + 1],
            )
            log_probability = distribution.log_prob(action[index : index + 1])
            if not all(
                bool(torch.isfinite(item).all())
                for item in (
                    logits,
                    value,
                    log_probability,
                )
            ):
                raise FloatingPointError("non-finite PQ-1 row-wise replay")
            row_log.append(log_probability[0])
            row_value.append(value[0])
        replay_log = torch.stack(row_log)
        replay_value = torch.stack(row_value)
        batch_logits, batch_value = model(observation)
        batch_log = _masked_distribution(batch_logits, mask).log_prob(action)
    if not torch.equal(replay_log, stored_log) or not torch.equal(
        replay_value,
        stored_value,
    ):
        raise ValueError("PQ-1 rollout-shaped snapshot binding mismatch")
    if not all(
        bool(torch.isfinite(item).all())
        for item in (
            batch_logits,
            batch_value,
            batch_log,
        )
    ):
        raise FloatingPointError("non-finite PQ-1 full-batch diagnostic")
    batch_log_summary = _drift_summary(batch_log, stored_log)
    batch_value_summary = _drift_summary(batch_value, stored_value)
    diagnostic_gates, probability_ratio_drift = full_batch_numerical_conformance(
        batch_log,
        stored_log,
        batch_value,
        stored_value,
    )
    if not all(diagnostic_gates.values()):
        raise ValueError("PQ-1 full-batch numerical diagnostic gate failed")
    legacy_exceed = torch.abs(batch_log - stored_log) > (
        LEGACY_ATOL + LEGACY_RTOL * torch.abs(stored_log)
    )
    first_legacy_transition_sha256 = None
    if bool(legacy_exceed.any()):
        first = int(torch.nonzero(legacy_exceed, as_tuple=False)[0].item())
        first_legacy_transition_sha256 = hashlib.sha256(
            (
                _array_sha256(observation[first].cpu().numpy(), dtype=np.dtype("<f4"))
                + _array_sha256(mask[first].cpu().numpy(), dtype=np.dtype("?"))
                + _array_sha256(action[first].cpu().numpy(), dtype=np.dtype("<i8"))
            ).encode("ascii")
        ).hexdigest()
    return {
        "schema_version": PQ1_PRIMITIVES_VERSION,
        "transition_count": len(observation),
        "rowwise_forward_calls": len(observation),
        "rowwise_max_batch_size": 1,
        "rowwise_log_probability_exact": True,
        "rowwise_value_exact": True,
        "observation_sha256": _array_sha256(
            observation.cpu().numpy(),
            dtype=np.dtype("<f4"),
        ),
        "mask_sha256": _array_sha256(
            mask.cpu().numpy(),
            dtype=np.dtype("?"),
        ),
        "action_sha256": _array_sha256(
            action.cpu().numpy(),
            dtype=np.dtype("<i8"),
        ),
        "full_batch_log_probability": batch_log_summary,
        "full_batch_value": batch_value_summary,
        "max_probability_ratio_drift": probability_ratio_drift,
        "first_legacy_exceed_transition_sha256": first_legacy_transition_sha256,
        "diagnostic_gates": diagnostic_gates,
    }


def pq1_cr_ppo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    episodes: Sequence[Sequence[Mapping[str, Any]]],
    config: PPOConfig,
    device: torch.device,
    generator: torch.Generator,
    *,
    thresholds: SafetyThresholds,
    rollout_model_sha256: str,
    rollout_optimizer_sha256: str,
    rollout_transition_sha256: Sequence[str],
) -> tuple[ActorModeDecision, dict[str, Any]]:
    """Run one candidate update only after exact rollout-shaped validation."""

    thresholds = validate_thresholds(thresholds)
    if len(episodes) != config.episodes_per_update:
        raise ValueError("PQ-1 rollout batch does not match scheduled B")
    expected_transition_hashes = list(rollout_transition_sha256)
    actual_transition_hashes = [
        transition_episode_sha256(episode) for episode in episodes
    ]
    if (
        len(expected_transition_hashes) != len(episodes)
        or any(
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in expected_transition_hashes
        )
        or actual_transition_hashes != expected_transition_hashes
    ):
        raise ValueError("PQ-1 rollout transition digest changed")
    _validate_pre_update_state(model, optimizer)
    if (
        model_parameter_sha256(model) != rollout_model_sha256
        or optimizer_state_sha256(optimizer, model) != rollout_optimizer_sha256
    ):
        raise ValueError("PQ-1 rollout model/optimizer digest changed")
    unsafe_events, wrong_executions = _validate_transition_costs(
        episodes,
        thresholds=thresholds,
    )
    decision = select_actor_mode(
        unsafe_events=unsafe_events,
        wrong_executions=wrong_executions,
        episodes=len(episodes),
        thresholds=thresholds,
    )
    batch, batch_metrics = cr_ppo_batch(episodes, config, decision=decision)
    _validate_ppo_batch(batch)
    snapshot = rollout_shaped_snapshot_diagnostics(model, batch, device)
    ppo_metrics = lagrangian_ppo_update(
        model,
        optimizer,
        batch,
        config,
        device,
        generator,
    )
    if (
        any(
            not bool(torch.isfinite(parameter).all())
            for parameter in model.parameters()
        )
        or any(
            parameter.grad is not None
            and not bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        or not _finite_value(optimizer.state)
        or not _finite_value(
            [
                {key: value for key, value in group.items() if key != "params"}
                for group in optimizer.param_groups
            ]
        )
        or any(not math.isfinite(float(value)) for value in ppo_metrics.values())
    ):
        raise FloatingPointError("non-finite PQ-1 PPO state")
    return decision, {
        **batch_metrics,
        **ppo_metrics,
        "rollout_unsafe_events": unsafe_events,
        "rollout_wrong_executions": wrong_executions,
        "snapshot_diagnostics": {
            **snapshot,
            "ordered_transition_episode_sha256": actual_transition_hashes,
            "ordered_transition_batch_sha256": hashlib.sha256(
                "".join(actual_transition_hashes).encode("ascii")
            ).hexdigest(),
        },
    }
