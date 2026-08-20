"""CRPO-inspired constraint-rectified PPO primitives for frozen A23."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import torch

from .a22_constrained_ppo import (
    SafetyThresholds,
    _reward_and_cost_advantages,
    lagrangian_ppo_update,
    validate_thresholds,
)
from .a9_ppo_oof import _digest
from .long_horizon_env import ACTION_COUNT, MultiTownLongHorizonEnv, RLAction
from .ppo_controller import ActorCritic, PPOConfig, _masked_distribution


A23_PRIMITIVES_VERSION = "multitown-a23-cr-ppo-primitives-v1"
ActorMode = Literal["reward", "unsafe", "wrong"]


@dataclass(frozen=True)
class CRPPOMechanism:
    name: str
    shield_enabled: bool


MECHANISMS = (
    CRPPOMechanism("cr-ppo", False),
    CRPPOMechanism("cr-ppo-plus-shield", True),
)
MECHANISM_BY_NAME = {item.name: item for item in MECHANISMS}


@dataclass(frozen=True)
class ActorModeDecision:
    mode: ActorMode
    unsafe_cost: float
    wrong_cost: float
    unsafe_threshold: float
    wrong_threshold: float
    unsafe_violation: float
    wrong_violation: float
    unsafe_normalized_violation: float
    wrong_normalized_violation: float
    unsafe_eligible: bool
    wrong_eligible: bool
    unsafe_tie_break_used: bool


def select_actor_mode(
    *, unsafe_events: int, wrong_executions: int, episodes: int,
    thresholds: SafetyThresholds,
) -> ActorModeDecision:
    """Select the reward or one cost surrogate without consuming any RNG."""

    thresholds = validate_thresholds(thresholds)
    if (
        type(unsafe_events) is not int or unsafe_events < 0
        or type(wrong_executions) is not int or wrong_executions < 0
        or type(episodes) is not int or episodes <= 0
        or unsafe_events > episodes or wrong_executions < unsafe_events
    ):
        raise ValueError("invalid A23 batch cost counts")
    unsafe_cost = np.float64(unsafe_events) / np.float64(episodes)
    wrong_cost = (
        np.float64(wrong_executions) / np.float64(episodes)
        / np.float64(thresholds.mean_incidents)
    )
    unsafe_threshold = np.float64(thresholds.unsafe)
    wrong_threshold = np.float64(thresholds.wrong_per_incident)
    unsafe_violation = unsafe_cost - unsafe_threshold
    wrong_violation = wrong_cost - wrong_threshold
    unsafe_normalized = unsafe_violation / max(unsafe_threshold, np.float64(1e-8))
    wrong_normalized = wrong_violation / max(wrong_threshold, np.float64(1e-8))
    values = (
        unsafe_cost, wrong_cost, unsafe_threshold, wrong_threshold,
        unsafe_violation, wrong_violation, unsafe_normalized, wrong_normalized,
    )
    if not all(np.isfinite(value) for value in values):
        raise FloatingPointError("non-finite A23 selector value")
    unsafe_eligible = bool(unsafe_cost > unsafe_threshold)
    wrong_eligible = bool(wrong_cost > wrong_threshold)
    tie = bool(
        unsafe_eligible and wrong_eligible
        and unsafe_normalized == wrong_normalized
    )
    if not unsafe_eligible and not wrong_eligible:
        mode: ActorMode = "reward"
    elif unsafe_eligible and (
        not wrong_eligible or unsafe_normalized >= wrong_normalized
    ):
        mode = "unsafe"
    else:
        mode = "wrong"
    return ActorModeDecision(
        mode=mode,
        unsafe_cost=float(unsafe_cost),
        wrong_cost=float(wrong_cost),
        unsafe_threshold=float(unsafe_threshold),
        wrong_threshold=float(wrong_threshold),
        unsafe_violation=float(unsafe_violation),
        wrong_violation=float(wrong_violation),
        unsafe_normalized_violation=float(unsafe_normalized),
        wrong_normalized_violation=float(wrong_normalized),
        unsafe_eligible=unsafe_eligible,
        wrong_eligible=wrong_eligible,
        unsafe_tie_break_used=tie,
    )


def normalized_advantage_sha256(values: np.ndarray) -> str:
    """Hash the protocol's explicit little-endian float32 array encoding."""

    array = np.ascontiguousarray(values, dtype="<f4")
    if array.size == 0 or not np.isfinite(array).all():
        raise FloatingPointError("cannot hash empty or non-finite A23 advantage")
    shape = json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
    digest = hashlib.sha256()
    digest.update(b"float32\0")
    digest.update(shape)
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0 or not np.isfinite(array).all():
        raise FloatingPointError("empty or non-finite A23 advantage")
    widened = array.astype(np.float64)
    return {
        "mean": float(widened.mean()),
        "std": float(widened.std(ddof=0)),
        "max_abs": float(np.abs(widened).max()),
    }


def cr_ppo_batch(
    episodes: Sequence[Sequence[Mapping[str, Any]]], config: PPOConfig, *,
    decision: ActorModeDecision,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build one A23 batch through the pinned A22 reward/cost-return path."""

    flat, reward, unsafe, wrong = _reward_and_cost_advantages(episodes, config)
    raw_by_mode: dict[ActorMode, np.ndarray] = {
        "reward": reward,
        "unsafe": -unsafe,
        "wrong": -wrong,
    }
    selected_raw = np.asarray(raw_by_mode[decision.mode], dtype=np.float32)
    if selected_raw.size == 0 or not np.isfinite(selected_raw).all():
        raise FloatingPointError("invalid A23 selected actor advantage")
    selected_std = selected_raw.std()
    normalized = (
        (selected_raw - selected_raw.mean()) / (selected_std + 1e-8)
    )
    if normalized.dtype != np.float32 or not np.isfinite(normalized).all():
        raise FloatingPointError("invalid normalized A23 actor advantage")
    batch = {key: np.asarray(value) for key, value in flat.items()}
    batch.update({
        "advantage": normalized,
        "reward_advantage_raw": reward,
        "unsafe_advantage_raw": unsafe,
        "wrong_advantage_raw": wrong,
    })
    return batch, {
        "selected_actor_mode": decision.mode,
        "selected_advantage_constant": bool(selected_std == 0.0),
        "normalized_advantage_sha256": normalized_advantage_sha256(normalized),
        "reward_advantage_raw": _summary(reward),
        "unsafe_advantage_raw": _summary(unsafe),
        "wrong_advantage_raw": _summary(wrong),
        "selected_advantage_raw": _summary(selected_raw),
    }


def _finite_float32_scalar(value: Any) -> bool:
    with np.errstate(over="ignore", invalid="ignore"):
        return bool(np.isfinite(np.float32(value)))


def _validate_transition_costs(
    episodes: Sequence[Sequence[Mapping[str, Any]]], *,
    thresholds: SafetyThresholds,
) -> tuple[int, int]:
    unsafe_events = 0
    wrong_executions = 0
    for episode in episodes:
        if not episode:
            raise ValueError("empty A23 rollout episode")
        unsafe_seen = False
        episode_wrong = 0
        for transition in episode:
            if not isinstance(transition, Mapping):
                raise ValueError("invalid A23 transition schema")
            try:
                observation = transition["observation"]
                mask = transition["mask"]
                action = transition["action"]
                wrong_flag = transition["wrong_execute"]
                scalar_values = tuple(
                    transition[key] for key in (
                        "old_log_probability", "old_value", "reward",
                        "unsafe_cost", "wrong_cost",
                    )
                )
            except KeyError as exc:
                raise ValueError("invalid A23 transition schema") from exc
            if (
                not isinstance(observation, np.ndarray)
                or observation.shape
                != (MultiTownLongHorizonEnv.observation_size,)
                or observation.dtype != np.float32
                or not np.isfinite(observation).all()
                or not isinstance(mask, np.ndarray)
                or mask.shape != (ACTION_COUNT,)
                or mask.dtype != np.bool_
                or not bool(mask.any())
                or type(action) is not int
                or not 0 <= action < ACTION_COUNT
                or not bool(mask[action])
                or not isinstance(wrong_flag, (bool, np.bool_))
                or any(
                    isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, float, np.integer, np.floating))
                    or not math.isfinite(float(value))
                    or not _finite_float32_scalar(value)
                    for value in scalar_values
                )
            ):
                raise ValueError("invalid A23 transition schema")
            wrong_execute = bool(wrong_flag)
            if wrong_execute and action != int(RLAction.EXECUTE):
                raise ValueError("A23 wrong execute event does not bind to action")
            expected_wrong = float(wrong_execute) / thresholds.mean_incidents
            expected_unsafe = float(wrong_execute and not unsafe_seen)
            if (
                float(transition["wrong_cost"]) != expected_wrong
                or float(transition["unsafe_cost"]) != expected_unsafe
            ):
                raise ValueError("A23 transition safety costs do not bind to events")
            episode_wrong += int(wrong_execute)
            unsafe_seen = unsafe_seen or wrong_execute
        unsafe_events += int(unsafe_seen)
        wrong_executions += episode_wrong
    return unsafe_events, wrong_executions


def _finite_value(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, Mapping):
        return all(_finite_value(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_value(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    return True


def _optimizer_parameter_groups(
    optimizer: torch.optim.Optimizer, model: ActorCritic,
) -> list[dict[str, Any]]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    expected_parameters = [parameter for _, parameter in model.named_parameters()]
    observed_parameters: list[torch.nn.Parameter] = []
    groups: list[dict[str, Any]] = []
    for group in optimizer.param_groups:
        group_names = []
        for parameter in group["params"]:
            name = names.get(id(parameter))
            if name is None:
                raise ValueError("optimizer parameter does not bind to A23 model")
            group_names.append(name)
            observed_parameters.append(parameter)
        groups.append({
            **{key: value for key, value in group.items() if key != "params"},
            "params": group_names,
        })
    if (
        len(observed_parameters) != len(expected_parameters)
        or any(
            observed is not expected for observed, expected in zip(
                observed_parameters, expected_parameters, strict=True,
            )
        )
        or any(id(parameter) not in names for parameter in optimizer.state)
    ):
        raise ValueError("optimizer parameter order differs from A23 model")
    return groups


def validate_optimizer_model_binding(
    optimizer: torch.optim.Optimizer, model: ActorCritic,
) -> None:
    """Reject missing, duplicate, reordered, or foreign optimizer parameters."""

    _optimizer_parameter_groups(optimizer, model)


def _validate_pre_update_state(
    model: ActorCritic, optimizer: torch.optim.Optimizer,
) -> None:
    validate_optimizer_model_binding(optimizer, model)
    if any(
        parameter.dtype != torch.float32
        or not bool(torch.isfinite(parameter).all())
        or (
            parameter.grad is not None
            and not bool(torch.isfinite(parameter.grad).all())
        )
        for parameter in model.parameters()
    ):
        raise FloatingPointError("non-finite or non-float32 A23 model state")
    if any(
        tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all())
        for tensor in model.buffers()
    ):
        raise FloatingPointError("non-finite or non-float32 A23 model buffer")
    optimizer_options = [
        {key: value for key, value in group.items() if key != "params"}
        for group in optimizer.param_groups
    ]
    if not _finite_value(optimizer.state) or not _finite_value(optimizer_options):
        raise FloatingPointError("non-finite A23 optimizer state")


def _validate_ppo_batch(batch: Mapping[str, np.ndarray]) -> None:
    required = {
        "observation", "mask", "action", "old_log_probability", "old_value",
        "return", "advantage", "reward_advantage_raw",
        "unsafe_advantage_raw", "wrong_advantage_raw",
    }
    if set(batch) != required:
        raise ValueError("invalid A23 PPO batch fields")
    observation = np.asarray(batch["observation"])
    mask = np.asarray(batch["mask"])
    action = np.asarray(batch["action"])
    if (
        observation.ndim != 2
        or observation.shape[1] != MultiTownLongHorizonEnv.observation_size
        or observation.dtype != np.float32
        or not np.isfinite(observation).all()
        or mask.shape != (len(observation), ACTION_COUNT)
        or mask.dtype != np.bool_
        or not bool(mask.any(axis=1).all())
        or action.shape != (len(observation),)
        or not np.issubdtype(action.dtype, np.integer)
        or not bool(((action >= 0) & (action < ACTION_COUNT)).all())
        or not bool(mask[np.arange(len(action)), action].all())
    ):
        raise ValueError("invalid A23 PPO batch tensor schema")
    for key in (
        "old_log_probability", "old_value", "return", "advantage",
        "reward_advantage_raw", "unsafe_advantage_raw", "wrong_advantage_raw",
    ):
        values = np.asarray(batch[key])
        with np.errstate(over="ignore", invalid="ignore"):
            float32_values = np.asarray(values, dtype=np.float32)
        if (
            values.shape != (len(observation),)
            or not np.isfinite(values).all()
            or not np.isfinite(float32_values).all()
        ):
            raise FloatingPointError(f"non-finite A23 PPO batch field: {key}")
    for key in (
        "advantage", "reward_advantage_raw", "unsafe_advantage_raw",
        "wrong_advantage_raw",
    ):
        if np.asarray(batch[key]).dtype != np.float32:
            raise ValueError(f"non-float32 A23 advantage field: {key}")


def _validate_on_policy_snapshot(
    model: ActorCritic, batch: Mapping[str, np.ndarray], device: torch.device,
) -> None:
    observation = torch.as_tensor(
        batch["observation"], dtype=torch.float32, device=device,
    )
    mask = torch.as_tensor(batch["mask"], dtype=torch.bool, device=device)
    action = torch.as_tensor(batch["action"], dtype=torch.long, device=device)
    stored_log_probability = torch.as_tensor(
        batch["old_log_probability"], dtype=torch.float32, device=device,
    )
    stored_value = torch.as_tensor(
        batch["old_value"], dtype=torch.float32, device=device,
    )
    with torch.no_grad():
        logits, value = model(observation)
        distribution = _masked_distribution(logits, mask)
        log_probability = distribution.log_prob(action)
    if (
        not bool(torch.isfinite(logits).all())
        or not bool(torch.isfinite(value).all())
        or not bool(torch.isfinite(log_probability).all())
        or not torch.allclose(
            log_probability, stored_log_probability, rtol=1e-6, atol=1e-7,
        )
        or not torch.allclose(value, stored_value, rtol=1e-6, atol=1e-7)
    ):
        raise ValueError("A23 rollout does not bind to current on-policy model")


def cr_ppo_update(
    model: ActorCritic, optimizer: torch.optim.Optimizer,
    episodes: Sequence[Sequence[Mapping[str, Any]]], config: PPOConfig,
    device: torch.device, generator: torch.Generator, *,
    thresholds: SafetyThresholds,
) -> tuple[ActorModeDecision, dict[str, Any]]:
    """Select one surrogate, run pinned PPO, and fail on non-finite state."""

    thresholds = validate_thresholds(thresholds)
    if len(episodes) != config.episodes_per_update:
        raise ValueError("A23 rollout batch does not match scheduled B")
    _validate_pre_update_state(model, optimizer)
    unsafe_events, wrong_executions = _validate_transition_costs(
        episodes, thresholds=thresholds,
    )
    decision = select_actor_mode(
        unsafe_events=unsafe_events, wrong_executions=wrong_executions,
        episodes=len(episodes), thresholds=thresholds,
    )
    batch, batch_metrics = cr_ppo_batch(episodes, config, decision=decision)
    _validate_ppo_batch(batch)
    _validate_on_policy_snapshot(model, batch, device)
    ppo_metrics = lagrangian_ppo_update(
        model, optimizer, batch, config, device, generator,
    )
    if (
        any(not bool(torch.isfinite(parameter).all()) for parameter in model.parameters())
        or any(
            parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        or not _finite_value(optimizer.state)
        or not _finite_value([
            {key: value for key, value in group.items() if key != "params"}
            for group in optimizer.param_groups
        ])
        or any(not math.isfinite(float(value)) for value in ppo_metrics.values())
    ):
        raise FloatingPointError("non-finite A23 PPO state")
    return decision, {
        **batch_metrics,
        **ppo_metrics,
        "rollout_unsafe_events": unsafe_events,
        "rollout_wrong_executions": wrong_executions,
    }


def model_parameter_sha256(model: ActorCritic) -> str:
    """Hash the finite float32 model state independently of torch serialization."""

    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().numpy()
        if array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError("A23 initial model state must be finite float32")
        little = np.ascontiguousarray(array, dtype="<f4")
        for payload in (
            key.encode("utf-8"), b"float32",
            json.dumps(list(little.shape), separators=(",", ":")).encode("ascii"),
            little.tobytes(order="C"),
        ):
            digest.update(payload)
            digest.update(b"\0")
    return digest.hexdigest()


def initial_optimizer_sha256(
    optimizer: torch.optim.Optimizer, model: ActorCritic,
) -> str:
    """Hash an empty optimizer state with stable named parameter groups."""

    if optimizer.state:
        raise ValueError("A23 initial optimizer state is not empty")
    groups = _optimizer_parameter_groups(optimizer, model)
    return _digest({
        "optimizer_class": (
            f"{optimizer.__class__.__module__}.{optimizer.__class__.__qualname__}"
        ),
        "defaults": dict(optimizer.defaults),
        "param_groups": groups,
        "state": {},
    })


def mode_sequence_sha256(modes: Sequence[str]) -> str:
    if not modes or any(mode not in {"reward", "unsafe", "wrong"} for mode in modes):
        raise ValueError("invalid A23 actor mode sequence")
    return _digest(list(modes))
