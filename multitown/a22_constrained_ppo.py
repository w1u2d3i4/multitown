"""Constrained-PPO primitives for the frozen A22 adaptive protocol."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .long_horizon_env import (
    ACTION_COUNT, MultiTownLongHorizonEnv, RLAction,
)
from .ppo_controller import (
    ActorCritic, PPOConfig, _masked_distribution,
)


A22_PRIMITIVES_VERSION = "multitown-a22-constrained-ppo-primitives-v2"
REVIEW_STATE_SLICE = slice(33, 36)
DUAL_STEP_SIZE = 0.1
UNSAFE_MARGIN = 0.02
WRONG_PER_INCIDENT_MARGIN = 0.01


@dataclass(frozen=True)
class Mechanism:
    name: str
    dual_enabled: bool
    shield_enabled: bool


MECHANISMS = (
    Mechanism("reference", False, False),
    Mechanism("lagrangian", True, False),
    Mechanism("shield", False, True),
    Mechanism("lagrangian-plus-shield", True, True),
)
MECHANISM_BY_NAME = {item.name: item for item in MECHANISMS}


@dataclass(frozen=True)
class DualState:
    unsafe: float = 0.0
    wrong_per_incident: float = 0.0


@dataclass(frozen=True)
class SafetyThresholds:
    unsafe: float
    wrong_per_incident: float
    mean_incidents: float


def validate_thresholds(thresholds: SafetyThresholds) -> SafetyThresholds:
    """Fail before PPO on non-finite or out-of-range fit-local thresholds."""

    if (
        not math.isfinite(thresholds.unsafe)
        or not 0.0 <= thresholds.unsafe <= 1.0
        or not math.isfinite(thresholds.wrong_per_incident)
        or not 0.0 <= thresholds.wrong_per_incident <= 1.0
        or not math.isfinite(thresholds.mean_incidents)
        or thresholds.mean_incidents <= 0.0
    ):
        raise ValueError("invalid safety thresholds")
    return thresholds


def mechanism_dual(mechanism: Mechanism, dual: DualState) -> DualState:
    """Fail before actor construction if a dual-off arm is contaminated."""

    if not mechanism.dual_enabled and dual != DualState():
        raise ValueError("dual-off mechanism must keep zero multipliers before PPO")
    return dual


def public_review_state(observation: np.ndarray) -> int:
    """Decode only the public one-hot review state in the frozen observation."""

    if (
        observation.shape != (MultiTownLongHorizonEnv.observation_size,)
        or observation.dtype != np.float32
        or not np.isfinite(observation).all()
    ):
        raise ValueError("invalid public observation")
    values = observation[REVIEW_STATE_SLICE]
    index = int(np.argmax(values))
    if not np.array_equal(values, np.eye(3, dtype=np.float32)[index]):
        raise ValueError("review-state observation is not strict one-hot")
    return index


def effective_action_mask(
    observation: np.ndarray, base_mask: np.ndarray, *, shield_enabled: bool,
) -> tuple[np.ndarray, bool]:
    """Apply the public review shield before action sampling and log-probability."""

    if (
        base_mask.shape != (ACTION_COUNT,)
        or base_mask.dtype != np.bool_
        or not bool(base_mask.any())
    ):
        raise ValueError("invalid environment action mask")
    result = base_mask.copy()
    intervened = bool(
        shield_enabled
        and result[int(RLAction.EXECUTE)]
        and public_review_state(observation) != 1
    )
    if intervened:
        result[int(RLAction.EXECUTE)] = False
    if not result.any():
        raise RuntimeError("review shield removed every legal action")
    for action in (RLAction.STOP, RLAction.HUMAN):
        if result[int(action)] != base_mask[int(action)]:
            raise RuntimeError("review shield changed stop/human legality")
    result.flags.writeable = False
    return result, intervened


def constrained_rollout(
    model: ActorCritic, episode: Any, device: torch.device, *,
    mean_incidents: float, shield_enabled: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collect one stochastic on-policy rollout without reading private labels."""

    if not math.isfinite(mean_incidents) or mean_incidents <= 0.0:
        raise ValueError("mean incidents must be finite and positive")
    env = MultiTownLongHorizonEnv(episode)
    observation, _ = env.reset()
    transitions: list[dict[str, Any]] = []
    total_return = 0.0
    unsafe_seen = False
    shield_interventions = 0
    while not env.terminated:
        base_mask = env.action_mask()
        mask, intervened = effective_action_mask(
            observation, base_mask, shield_enabled=shield_enabled,
        )
        shield_interventions += int(intervened)
        obs_tensor = torch.from_numpy(observation).to(device).unsqueeze(0)
        mask_tensor = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, value = model(obs_tensor)
            distribution = _masked_distribution(logits, mask_tensor)
            action = distribution.sample()
            log_probability = distribution.log_prob(action)
        selected = int(action.item())
        next_observation, reward, done, _, _ = env.step(selected)
        latest_reward = env.trajectory[-1]["reward"]
        wrong_execute = bool(
            selected == int(RLAction.EXECUTE)
            and float(latest_reward["safety_penalty"]) < 0.0
        )
        first_unsafe = wrong_execute and not unsafe_seen
        unsafe_seen = unsafe_seen or wrong_execute
        transitions.append({
            "observation": observation,
            "base_mask": base_mask,
            "mask": mask,
            "action": selected,
            "old_log_probability": float(log_probability.item()),
            "old_value": float(value.item()),
            "reward": float(reward),
            "unsafe_cost": float(first_unsafe),
            "wrong_cost": float(wrong_execute) / mean_incidents,
            "wrong_execute": wrong_execute,
            "shield_intervened": intervened,
            "done": bool(done),
        })
        total_return += reward
        observation = next_observation
    wrong_executions = sum(int(row["wrong_execute"]) for row in transitions)
    info = env.info()
    return transitions, {
        **info,
        "return": total_return,
        "unsafe_episode": bool(wrong_executions),
        "wrong_executions": wrong_executions,
        "shield_interventions": shield_interventions,
        "episode_cost_unsafe": sum(row["unsafe_cost"] for row in transitions),
        "episode_cost_wrong": sum(row["wrong_cost"] for row in transitions),
    }


def deterministic_action(
    model: ActorCritic, observation: np.ndarray, base_mask: np.ndarray, *,
    shield_enabled: bool,
) -> tuple[int, bool]:
    """Apply the same pre-decision effective mask for calibration/outer argmax."""

    mask, intervened = effective_action_mask(
        observation, base_mask, shield_enabled=shield_enabled,
    )
    device = next(model.parameters()).device
    obs_tensor = torch.tensor(
        observation, dtype=torch.float32, device=device,
    ).unsqueeze(0)
    mask_tensor = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
    with torch.no_grad():
        logits, _ = model(obs_tensor)
        action = logits.masked_fill(
            ~mask_tensor, torch.finfo(logits.dtype).min,
        ).argmax(dim=-1)
    selected = int(action.item())
    if not bool(mask[selected]):
        raise RuntimeError("deterministic policy selected outside effective mask")
    return selected, intervened


def _reward_and_cost_advantages(
    episodes: Sequence[Sequence[Mapping[str, Any]]], config: PPOConfig,
) -> tuple[dict[str, list[Any]], np.ndarray, np.ndarray, np.ndarray]:
    flat: dict[str, list[Any]] = {
        "observation": [], "mask": [], "action": [],
        "old_log_probability": [], "old_value": [], "return": [],
    }
    reward_advantages: list[float] = []
    unsafe_advantages: list[float] = []
    wrong_advantages: list[float] = []
    for episode in episodes:
        if not episode:
            raise ValueError("empty rollout episode")
        next_reward_advantage = 0.0
        next_value = 0.0
        unsafe_return = 0.0
        wrong_return = 0.0
        episode_reward = [0.0] * len(episode)
        episode_return = [0.0] * len(episode)
        episode_unsafe = [0.0] * len(episode)
        episode_wrong = [0.0] * len(episode)
        for index in range(len(episode) - 1, -1, -1):
            transition = episode[index]
            delta = (
                float(transition["reward"])
                + config.gamma * next_value
                - float(transition["old_value"])
            )
            next_reward_advantage = (
                delta + config.gamma * config.gae_lambda * next_reward_advantage
            )
            unsafe_return = float(transition["unsafe_cost"]) + unsafe_return
            wrong_return = float(transition["wrong_cost"]) + wrong_return
            episode_reward[index] = next_reward_advantage
            episode_return[index] = (
                next_reward_advantage + float(transition["old_value"])
            )
            episode_unsafe[index] = unsafe_return
            episode_wrong[index] = wrong_return
            next_value = float(transition["old_value"])
        for transition, reward_advantage, target, unsafe, wrong in zip(
            episode, episode_reward, episode_return, episode_unsafe,
            episode_wrong, strict=True,
        ):
            for key in (
                "observation", "mask", "action", "old_log_probability",
                "old_value",
            ):
                flat[key].append(transition[key])
            flat["return"].append(target)
            reward_advantages.append(reward_advantage)
            unsafe_advantages.append(unsafe)
            wrong_advantages.append(wrong)
    return (
        flat,
        np.asarray(reward_advantages, dtype=np.float32),
        np.asarray(unsafe_advantages, dtype=np.float32),
        np.asarray(wrong_advantages, dtype=np.float32),
    )


def lagrangian_batch(
    episodes: Sequence[Sequence[Mapping[str, Any]]], config: PPOConfig, *,
    dual: DualState, mechanism: Mechanism | None = None,
) -> dict[str, np.ndarray]:
    """Build the frozen A22 reward/cost actor advantage and reward value target."""

    if mechanism is not None:
        dual = mechanism_dual(mechanism, dual)
    if any(
        not math.isfinite(value) or value < 0.0
        for value in (dual.unsafe, dual.wrong_per_incident)
    ):
        raise ValueError("dual multipliers must be finite and non-negative")
    flat, reward, unsafe, wrong = _reward_and_cost_advantages(episodes, config)
    combined = reward.copy()
    if dual.unsafe:
        combined -= np.float32(dual.unsafe) * unsafe
    if dual.wrong_per_incident:
        combined -= np.float32(dual.wrong_per_incident) * wrong
    combined = (combined - combined.mean()) / (combined.std() + 1e-8)
    result = {key: np.asarray(value) for key, value in flat.items()}
    result.update({
        "advantage": combined,
        "reward_advantage_raw": reward,
        "unsafe_advantage_raw": unsafe,
        "wrong_advantage_raw": wrong,
    })
    return result


def lagrangian_ppo_update(
    model: ActorCritic, optimizer: torch.optim.Optimizer,
    batch: Mapping[str, np.ndarray], config: PPOConfig, device: torch.device,
    generator: torch.Generator,
) -> dict[str, float]:
    """A22 PPO update; its reference path is tensor-equivalent to A9-v2."""

    tensors = {
        "observation": torch.as_tensor(
            batch["observation"], dtype=torch.float32, device=device,
        ),
        "mask": torch.as_tensor(batch["mask"], dtype=torch.bool, device=device),
        "action": torch.as_tensor(batch["action"], dtype=torch.long, device=device),
        "old_log_probability": torch.as_tensor(
            batch["old_log_probability"], dtype=torch.float32, device=device,
        ),
        "old_value": torch.as_tensor(
            batch["old_value"], dtype=torch.float32, device=device,
        ),
        "advantage": torch.as_tensor(
            batch["advantage"], dtype=torch.float32, device=device,
        ),
        "return": torch.as_tensor(
            batch["return"], dtype=torch.float32, device=device,
        ),
    }
    count = len(tensors["action"])
    metrics: dict[str, list[float]] = {
        "policy_loss": [], "value_loss": [], "entropy": [],
        "approx_kl": [], "clip_fraction": [],
    }
    for _ in range(config.ppo_epochs):
        order = torch.randperm(count, generator=generator, device=device)
        for start in range(0, count, config.minibatch_size):
            indices = order[start:start + config.minibatch_size]
            logits, values = model(tensors["observation"][indices])
            distribution = _masked_distribution(logits, tensors["mask"][indices])
            new_log_probability = distribution.log_prob(tensors["action"][indices])
            log_ratio = (
                new_log_probability - tensors["old_log_probability"][indices]
            )
            ratio = log_ratio.exp()
            advantage = tensors["advantage"][indices]
            unclipped = ratio * advantage
            clipped = ratio.clamp(
                1.0 - config.clip_ratio, 1.0 + config.clip_ratio,
            ) * advantage
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = 0.5 * (
                values - tensors["return"][indices]
            ).square().mean()
            entropy = distribution.entropy().mean()
            loss = (
                policy_loss + config.value_coef * value_loss
                - config.entropy_coef * entropy
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                metrics["policy_loss"].append(float(policy_loss.item()))
                metrics["value_loss"].append(float(value_loss.item()))
                metrics["entropy"].append(float(entropy.item()))
                metrics["approx_kl"].append(float(
                    ((ratio - 1.0) - log_ratio).mean().item()
                ))
                metrics["clip_fraction"].append(float(
                    ((ratio - 1.0).abs() > config.clip_ratio).float().mean().item()
                ))
    return {key: float(np.mean(values)) for key, values in metrics.items()}


def update_dual(
    state: DualState, *, unsafe_events: int, wrong_executions: int,
    episodes: int, thresholds: SafetyThresholds,
) -> tuple[DualState, dict[str, float]]:
    """Apply the post-PPO projected ascent update from one rollout batch."""

    if (
        type(unsafe_events) is not int or unsafe_events < 0
        or type(wrong_executions) is not int or wrong_executions < 0
        or type(episodes) is not int or episodes <= 0
        or unsafe_events > episodes
    ):
        raise ValueError("invalid dual update counts or step size")
    validate_thresholds(thresholds)
    if any(
        not math.isfinite(value) or value < 0.0
        for value in (
            state.unsafe, state.wrong_per_incident,
        )
    ):
        raise ValueError("invalid dual state")
    unsafe_rate = unsafe_events / episodes
    wrong_rate = wrong_executions / episodes / thresholds.mean_incidents
    unsafe_violation = unsafe_rate - thresholds.unsafe
    wrong_violation = wrong_rate - thresholds.wrong_per_incident
    after = DualState(
        unsafe=max(0.0, state.unsafe + DUAL_STEP_SIZE * unsafe_violation),
        wrong_per_incident=max(
            0.0,
            state.wrong_per_incident + DUAL_STEP_SIZE * wrong_violation,
        ),
    )
    if not math.isfinite(after.unsafe) or not math.isfinite(after.wrong_per_incident):
        raise FloatingPointError("non-finite dual update")
    return after, {
        "lambda_unsafe_before": state.unsafe,
        "lambda_wrong_before": state.wrong_per_incident,
        "lambda_unsafe_after": after.unsafe,
        "lambda_wrong_after": after.wrong_per_incident,
        "rollout_unsafe_events": float(unsafe_events),
        "rollout_wrong_executions": float(wrong_executions),
        "rollout_episodes": float(episodes),
        "rollout_unsafe_rate": unsafe_rate,
        "rollout_wrong_per_fixed_mean_incident": wrong_rate,
        "unsafe_threshold": thresholds.unsafe,
        "wrong_per_incident_threshold": thresholds.wrong_per_incident,
        "unsafe_violation": unsafe_violation,
        "wrong_per_incident_violation": wrong_violation,
        "dual_step_size": DUAL_STEP_SIZE,
        "dual_update_performed": True,
    }


def thresholds_from_inner_train(
    a8_rows: Sequence[Mapping[str, Any]],
) -> SafetyThresholds:
    """Compute fit-local thresholds from only frozen inner-train A8 rows."""

    if not a8_rows:
        raise ValueError("inner-train threshold input is empty or invalid")
    mean_incidents = sum(int(row["incidents"]) for row in a8_rows) / len(a8_rows)
    if not math.isfinite(mean_incidents) or mean_incidents <= 0.0:
        raise ValueError("inner-train mean incidents is invalid")
    unsafe = sum(bool(row["had_wrong_execution"]) for row in a8_rows) / len(a8_rows)
    wrong = (
        sum(int(row["wrong_executions"]) for row in a8_rows)
        / len(a8_rows) / mean_incidents
    )
    return SafetyThresholds(
        unsafe=min(1.0, unsafe + UNSAFE_MARGIN),
        wrong_per_incident=min(1.0, wrong + WRONG_PER_INCIDENT_MARGIN),
        mean_incidents=mean_incidents,
    )


def constrained_update(
    model: ActorCritic, optimizer: torch.optim.Optimizer,
    episodes: Sequence[Sequence[Mapping[str, Any]]], config: PPOConfig,
    device: torch.device, generator: torch.Generator, *, mechanism: Mechanism,
    dual_before: DualState, thresholds: SafetyThresholds,
) -> tuple[DualState, dict[str, Any]]:
    """Atomically enforce batch -> PPO -> same-rollout post-PPO dual ordering."""

    if not episodes:
        raise ValueError("rollout transitions are empty")
    thresholds = validate_thresholds(thresholds)
    unsafe_events = 0
    wrong_executions = 0
    for episode in episodes:
        unsafe_seen = False
        episode_wrong = 0
        for transition in episode:
            wrong_execute = bool(transition["wrong_execute"])
            expected_wrong = float(wrong_execute) / thresholds.mean_incidents
            expected_unsafe = float(wrong_execute and not unsafe_seen)
            if not math.isclose(
                float(transition["wrong_cost"]), expected_wrong,
                rel_tol=1e-12, abs_tol=1e-12,
            ) or float(transition["unsafe_cost"]) != expected_unsafe:
                raise ValueError(
                    "transition safety costs do not bind to rollout events"
                )
            episode_wrong += int(wrong_execute)
            unsafe_seen = unsafe_seen or wrong_execute
        unsafe_events += int(unsafe_seen)
        wrong_executions += episode_wrong
    dual_before = mechanism_dual(mechanism, dual_before)
    batch = lagrangian_batch(
        episodes, config, dual=dual_before, mechanism=mechanism,
    )
    reward_raw = np.asarray(batch["reward_advantage_raw"], dtype=np.float64)
    unsafe_return = np.asarray(batch["unsafe_advantage_raw"], dtype=np.float64)
    wrong_return = np.asarray(batch["wrong_advantage_raw"], dtype=np.float64)
    combined_raw = (
        reward_raw - dual_before.unsafe * unsafe_return
        - dual_before.wrong_per_incident * wrong_return
    )
    if any(
        values.size == 0 or not np.isfinite(values).all()
        for values in (reward_raw, unsafe_return, wrong_return, combined_raw)
    ):
        raise FloatingPointError("non-finite reward or cost-return scale")
    scale_metrics = {
        "cost_return_scale_version": "raw-transition-cost-to-go-summary-v1",
        "reward_advantage_raw_mean": float(reward_raw.mean()),
        "reward_advantage_raw_std": float(reward_raw.std()),
        "reward_advantage_raw_max_abs": float(np.abs(reward_raw).max()),
        "unsafe_cost_return_mean": float(unsafe_return.mean()),
        "unsafe_cost_return_std": float(unsafe_return.std()),
        "unsafe_cost_return_max": float(unsafe_return.max()),
        "wrong_cost_return_mean": float(wrong_return.mean()),
        "wrong_cost_return_std": float(wrong_return.std()),
        "wrong_cost_return_max": float(wrong_return.max()),
        "combined_advantage_raw_mean": float(combined_raw.mean()),
        "combined_advantage_raw_std": float(combined_raw.std()),
        "combined_advantage_raw_max_abs": float(np.abs(combined_raw).max()),
    }
    ppo_metrics = lagrangian_ppo_update(
        model, optimizer, batch, config, device, generator,
    )
    if mechanism.dual_enabled:
        dual_after, dual_metrics = update_dual(
            dual_before, unsafe_events=unsafe_events,
            wrong_executions=wrong_executions, episodes=len(episodes),
            thresholds=thresholds,
        )
    else:
        unsafe_rate = unsafe_events / len(episodes)
        wrong_rate = (
            wrong_executions / len(episodes) / thresholds.mean_incidents
        )
        dual_after = dual_before
        dual_metrics = {
            "lambda_unsafe_before": 0.0,
            "lambda_wrong_before": 0.0,
            "lambda_unsafe_after": 0.0,
            "lambda_wrong_after": 0.0,
            "rollout_unsafe_events": float(unsafe_events),
            "rollout_wrong_executions": float(wrong_executions),
            "rollout_episodes": float(len(episodes)),
            "rollout_unsafe_rate": unsafe_rate,
            "rollout_wrong_per_fixed_mean_incident": wrong_rate,
            "unsafe_threshold": thresholds.unsafe,
            "wrong_per_incident_threshold": thresholds.wrong_per_incident,
            "unsafe_violation": unsafe_rate - thresholds.unsafe,
            "wrong_per_incident_violation": (
                wrong_rate - thresholds.wrong_per_incident
            ),
            "dual_step_size": DUAL_STEP_SIZE,
            "dual_update_performed": False,
        }
    return dual_after, {
        **ppo_metrics, **dual_metrics, **scale_metrics,
        "actor_dual_unsafe": dual_before.unsafe,
        "actor_dual_wrong_per_incident": dual_before.wrong_per_incident,
        "dual_updated_after_ppo": mechanism.dual_enabled,
    }
