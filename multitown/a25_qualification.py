"""Replay-bound common-state qualification primitives for A25.

This module produces development-only Q0 evidence.  It does not authorize an
A25 formal run or read any outer evaluation rows.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import struct
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import torch

from .a22_constrained_ppo import (
    DualState,
    SafetyThresholds,
    effective_action_mask,
    lagrangian_batch,
    lagrangian_ppo_update,
    public_review_state,
    validate_thresholds,
)
from .a23_cr_ppo import (
    cr_ppo_batch,
    model_parameter_sha256,
    select_actor_mode,
    validate_optimizer_model_binding,
)
from .a25_shield_dependence import (
    A25_PRIMITIVES_VERSION,
    InterventionObjective,
    _generator_state_sha256,
    build_shield_dependence_ledger,
    intervention_ppo_update,
    shield_aware_batch,
    shield_aware_rollout,
)
from .long_horizon_env import (
    ACTION_COUNT,
    LongHorizonEpisode,
    MultiTownLongHorizonEnv,
    RLAction,
)
from .ppo_controller import ActorCritic, PPOConfig
from .pq1_numerical_conformance import optimizer_state_sha256

A25_QUALIFICATION_VERSION = "multitown-a25-common-state-q0-primitives-v2"
A25_REPLAY_VERSION = "multitown-a25-exact-policy-environment-replay-v2"
A25_COMMON_STATE_ARMS = ("F00", "F01", "F10", "F11")
A25_FORMAL_MINIBATCH_SIZE = 4096
A25_FORMAL_EPISODES_PER_UPDATE = 48
A25_MAX_EPISODE_STEPS = 50
A25_FORMAL_MAX_DECISIONS = (
    A25_FORMAL_EPISODES_PER_UPDATE * A25_MAX_EPISODE_STEPS
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_OBJECT_RE = re.compile(r"[0-9a-f]{40}")
CRMode = Literal["reward", "unsafe", "wrong"]


@dataclass(frozen=True)
class QualificationBindings:
    """Content identities shared by every arm in one common-state panel."""

    run_id: str
    source_revision: str
    source_tree: str
    train_bank_sha256: str
    environment_source_sha256: str
    fold_manifest_sha256: str
    episode_sha256: Mapping[str, str]
    source_sha256: Mapping[str, str]
    outer_fold: int
    training_seed: int
    rollout_seed: int
    update: int = 1


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def episode_spec_sha256(episode: LongHorizonEpisode) -> str:
    return canonical_sha256(episode.to_dict())


def generator_state_sha256(generator: torch.Generator) -> str:
    return _generator_state_sha256(generator)


def _exact_float(left: Any, right: float) -> bool:
    return bool(
        type(left) is float
        and type(right) is float
        and struct.pack(">d", left) == struct.pack(">d", right)
    )


def _exact_value_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if left is None or isinstance(left, (bool, int, str)):
        return left == right
    if isinstance(left, float):
        return _exact_float(left, right)
    if isinstance(left, np.ndarray):
        return bool(
            left.dtype == right.dtype
            and left.shape == right.shape
            and left.flags.c_contiguous
            and right.flags.c_contiguous
            and left.tobytes(order="C") == right.tobytes(order="C")
        )
    if isinstance(left, dict):
        return bool(
            set(left) == set(right)
            and all(type(key) is str for key in left)
            and all(_exact_value_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)):
        return bool(
            len(left) == len(right)
            and all(
                _exact_value_equal(left_item, right_item)
                for left_item, right_item in zip(left, right, strict=True)
            )
        )
    return False


def _typed_value_sha256(value: Any) -> str:
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
                raise ValueError("non-finite A25 trace float")
            digest.update(b"float64\0" + struct.pack(">d", item) + b"\0")
        elif type(item) is str:
            payload = item.encode("utf-8")
            digest.update(b"str\0" + str(len(payload)).encode("ascii") + b"\0")
            digest.update(payload + b"\0")
        elif isinstance(item, np.ndarray):
            if (
                item.dtype.hasobject
                or not item.flags.c_contiguous
                or (
                    np.issubdtype(item.dtype, np.floating)
                    and not bool(np.isfinite(item).all())
                )
            ):
                raise ValueError("invalid A25 trace array")
            little = np.ascontiguousarray(item, dtype=item.dtype.newbyteorder("<"))
            digest.update(b"ndarray\0" + little.dtype.str.encode("ascii") + b"\0")
            digest.update(_canonical_json(list(little.shape)) + b"\0")
            digest.update(little.tobytes(order="C") + b"\0")
        elif type(item) is dict:
            if any(type(key) is not str for key in item):
                raise ValueError("invalid A25 trace mapping key")
            digest.update(b"dict\0" + str(len(item)).encode("ascii") + b"\0")
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
            raise TypeError("unsupported A25 trace value")

    update(value)
    return digest.hexdigest()


def array_mapping_sha256(values: Mapping[str, np.ndarray]) -> str:
    if not values:
        raise ValueError("cannot hash empty A25 array mapping")
    digest = hashlib.sha256()
    for key in sorted(values):
        array = np.asarray(values[key])
        if array.dtype.hasobject or (
            np.issubdtype(array.dtype, np.floating)
            and not bool(np.isfinite(array).all())
        ):
            raise ValueError("invalid A25 array mapping")
        little = np.ascontiguousarray(array, dtype=array.dtype.newbyteorder("<"))
        for payload in (
            key.encode("utf-8"),
            little.dtype.str.encode("ascii"),
            _canonical_json(list(little.shape)),
            little.tobytes(order="C"),
        ):
            digest.update(payload)
            digest.update(b"\0")
    return digest.hexdigest()


def validate_bindings(
    bindings: QualificationBindings,
    *,
    episode_ids: Sequence[str],
) -> QualificationBindings:
    sha_values = (
        bindings.run_id,
        bindings.train_bank_sha256,
        bindings.environment_source_sha256,
        bindings.fold_manifest_sha256,
        *bindings.episode_sha256.values(),
        *bindings.source_sha256.values(),
    )
    if (
        _SHA256_RE.fullmatch(bindings.run_id) is None
        or _GIT_OBJECT_RE.fullmatch(bindings.source_revision) is None
        or _GIT_OBJECT_RE.fullmatch(bindings.source_tree) is None
        or any(_SHA256_RE.fullmatch(value) is None for value in sha_values)
        or not bindings.source_sha256
        or not episode_ids
        or len(set(episode_ids)) != len(episode_ids)
        or set(bindings.episode_sha256) != set(episode_ids)
        or type(bindings.outer_fold) is not int
        or not 0 <= bindings.outer_fold < 5
        or type(bindings.training_seed) is not int
        or bindings.training_seed < 0
        or type(bindings.rollout_seed) is not int
        or bindings.rollout_seed < 0
        or type(bindings.update) is not int
        or bindings.update <= 0
    ):
        raise ValueError("invalid A25 qualification bindings")
    return bindings


def _termination_reason(
    *, action: int, terminated: bool, truncated: bool, budget_violation: bool
) -> str:
    if not terminated:
        return "not-terminal"
    if truncated:
        return "max-steps"
    if budget_violation:
        return "budget-exhausted"
    if action == int(RLAction.STOP):
        return "agent-stop"
    raise RuntimeError("unclassified A25 replay termination")


def replay_and_verify_shield_episode(
    episode: LongHorizonEpisode,
    transitions: Sequence[Mapping[str, Any]],
    *,
    model: ActorCritic,
    policy_generator: torch.Generator,
    mean_incidents: float,
    expected_episode_sha256: str,
) -> dict[str, Any]:
    """Exactly replay policy/RNG and environment from the bound episode."""

    if (
        not isinstance(episode, LongHorizonEpisode)
        or not isinstance(model, ActorCritic)
        or not isinstance(policy_generator, torch.Generator)
        or torch.device(policy_generator.device) != torch.device("cpu")
        or _SHA256_RE.fullmatch(expected_episode_sha256) is None
        or episode_spec_sha256(episode) != expected_episode_sha256
        or type(mean_incidents) is not float
        or not math.isfinite(mean_incidents)
        or mean_incidents <= 0.0
        or not transitions
        or any(type(row) is not dict for row in transitions)
    ):
        raise ValueError("invalid A25 replay input binding")
    local_generator = torch.Generator(device="cpu")
    local_generator.set_state(policy_generator.get_state().clone())
    policy_rng_before = generator_state_sha256(local_generator)
    model_before = model_parameter_sha256(model)
    expected_transitions, expected_summary = shield_aware_rollout(
        model,
        episode,
        torch.device("cpu"),
        mean_incidents=mean_incidents,
        generator=local_generator,
    )
    if (
        model_parameter_sha256(model) != model_before
        or len(transitions) != len(expected_transitions)
        or any(
            not _exact_value_equal(recorded, expected)
            for recorded, expected in zip(
                transitions, expected_transitions, strict=True
            )
        )
    ):
        raise ValueError("A25 replay does not bind to exact policy/RNG trace")
    policy_rng_after = generator_state_sha256(local_generator)
    env = MultiTownLongHorizonEnv(episode)
    observation, _ = env.reset()
    unsafe_seen = False
    action_names: list[str] = []
    replay_rows: list[dict[str, Any]] = []
    for index, transition in enumerate(transitions):
        try:
            recorded_observation = transition["observation"]
            base_mask = transition["base_mask"]
            effective_mask = transition["mask"]
            action = transition["action"]
        except KeyError as exc:
            raise ValueError("missing A25 replay transition field") from exc
        actual_base_mask = env.action_mask()
        actual_effective_mask, actual_shield_active = effective_action_mask(
            observation, actual_base_mask, shield_enabled=True
        )
        review_state = public_review_state(observation)
        if (
            transition.get("episode_id") != episode.episode_id
            or transition.get("split") != episode.split
            or type(transition.get("step_index")) is not int
            or transition.get("step_index") != index
            or not isinstance(recorded_observation, np.ndarray)
            or recorded_observation.dtype != np.float32
            or not np.array_equal(recorded_observation, observation)
            or not isinstance(base_mask, np.ndarray)
            or base_mask.dtype != np.bool_
            or not np.array_equal(base_mask, actual_base_mask)
            or not isinstance(effective_mask, np.ndarray)
            or effective_mask.dtype != np.bool_
            or not np.array_equal(effective_mask, actual_effective_mask)
            or type(action) is not int
            or not 0 <= action < ACTION_COUNT
            or type(transition.get("sampled_action")) is not int
            or transition.get("sampled_action") != action
            or type(transition.get("executed_action")) is not int
            or transition.get("executed_action") != action
            or not bool(actual_effective_mask[action])
            or transition.get("shield_intervened") is not actual_shield_active
            or type(transition.get("review_state")) is not int
            or transition.get("review_state") != review_state
            or transition.get("prior_review_pass") is not (review_state == 1)
            or transition.get("review_before_execute")
            is not (review_state == 1 if action == int(RLAction.EXECUTE) else None)
            or transition.get("execute_without_prior_review")
            is not bool(action == int(RLAction.EXECUTE) and review_state != 1)
            or type(transition.get("incident_index")) is not int
            or transition.get("incident_index") != env.incident_index
            or type(transition.get("tokens_before")) is not int
            or transition.get("tokens_before") != env.tokens_used
            or not _exact_float(
                transition.get("latency_before_s"), float(env.latency_used_s)
            )
        ):
            raise ValueError("A25 replay does not bind to environment pre-state")
        next_observation, reward, terminated, truncated, _ = env.step(action)
        trajectory = env.trajectory[-1]
        reward_components = trajectory["reward"]
        wrong_execute = bool(
            action == int(RLAction.EXECUTE)
            and float(reward_components["safety_penalty"]) < 0.0
        )
        unsafe_cost = float(wrong_execute and not unsafe_seen)
        unsafe_seen = unsafe_seen or wrong_execute
        wrong_cost = float(wrong_execute) / mean_incidents
        invalid_action = bool(float(reward_components["invalid_action"]) < 0.0)
        budget_violation = bool(
            float(reward_components["budget_violation"]) < 0.0
        )
        termination_reason = _termination_reason(
            action=action,
            terminated=terminated,
            truncated=truncated,
            budget_violation=budget_violation,
        )
        scalar_checks = (
            _exact_float(transition.get("reward"), float(reward)),
            _exact_float(transition.get("unsafe_cost"), float(unsafe_cost)),
            _exact_float(transition.get("wrong_cost"), float(wrong_cost)),
            _exact_float(
                transition.get("latency_after_s"), float(env.latency_used_s)
            ),
        )
        if (
            not all(scalar_checks)
            or not _exact_value_equal(
                transition.get("next_observation"), next_observation
            )
            or not _exact_value_equal(
                transition.get("reward_components"), reward_components
            )
            or not _exact_value_equal(
                transition.get("environment_info"), env.info()
            )
            or transition.get("wrong_execute") is not wrong_execute
            or transition.get("invalid_action") is not invalid_action
            or transition.get("budget_violation") is not budget_violation
            or type(transition.get("tokens_after")) is not int
            or transition.get("tokens_after") != env.tokens_used
            or transition.get("tool_failed") is not bool(trajectory["tool_failed"])
            or transition.get("terminated") is not terminated
            or transition.get("truncated") is not truncated
            or transition.get("done") is not terminated
            or transition.get("termination_reason") != termination_reason
        ):
            raise ValueError("A25 replay does not bind to environment post-state")
        action_names.append(str(trajectory["action"]))
        replay_rows.append(
            {
                "step_index": index,
                "incident_index": int(trajectory["incident_index"]),
                "action": str(trajectory["action"]),
                "tokens_after": int(env.tokens_used),
                "latency_after_s": float(env.latency_used_s),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )
        observation = next_observation
    if not env.terminated or transitions[-1].get("terminated") is not True:
        raise ValueError("A25 replay episode is not physically complete")
    policy_generator.set_state(local_generator.get_state())
    return {
        "schema_version": A25_REPLAY_VERSION,
        "episode_id": episode.episode_id,
        "episode_sha256": expected_episode_sha256,
        "decision_count": len(transitions),
        "actions_sha256": canonical_sha256(action_names),
        "replay_sha256": canonical_sha256(replay_rows),
        "policy_trace_sha256": _typed_value_sha256(list(transitions)),
        "policy_summary_sha256": _typed_value_sha256(expected_summary),
        "policy_model_sha256": model_before,
        "policy_rng_before_sha256": policy_rng_before,
        "policy_rng_after_sha256": policy_rng_after,
        "tokens_used": int(env.tokens_used),
        "latency_used_s": float(env.latency_used_s),
        "invalid_actions": int(env.invalid_actions),
        "budget_violations": int(env.budget_violations),
        "terminated": bool(env.terminated),
    }


def _ppo_metric_subset(metrics: Mapping[str, Any]) -> dict[str, float]:
    return {
        key: float(metrics[key])
        for key in (
            "policy_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "clip_fraction",
        )
    }


def _arm_update(
    source_model: ActorCritic,
    source_optimizer: torch.optim.Optimizer,
    transition_episodes: Sequence[Sequence[Mapping[str, Any]]],
    enriched_batch: Mapping[str, np.ndarray],
    config: PPOConfig,
    *,
    beta: float,
    tensor_seed: int,
    bindings: QualificationBindings,
    arm: str,
) -> dict[str, Any]:
    model, optimizer = copy.deepcopy((source_model, source_optimizer))
    validate_optimizer_model_binding(optimizer, model)
    generator = torch.Generator(device="cpu").manual_seed(tensor_seed)
    pre_model = model_parameter_sha256(model)
    pre_optimizer = optimizer_state_sha256(optimizer, model)
    pre_rng = generator_state_sha256(generator)
    metrics = intervention_ppo_update(
        model,
        optimizer,
        enriched_batch,
        config,
        torch.device("cpu"),
        generator,
        objective=InterventionObjective(beta=beta),
    )
    post_model = model_parameter_sha256(model)
    post_optimizer = optimizer_state_sha256(optimizer, model)
    post_rng = generator_state_sha256(generator)
    ledger = build_shield_dependence_ledger(
        source_model,
        transition_episodes,
        torch.device("cpu"),
        episode_ids=[str(rows[0]["episode_id"]) for rows in transition_episodes],
        fold=bindings.outer_fold,
        training_seed=bindings.training_seed,
        update=bindings.update,
        checkpoint_sha256=model_parameter_sha256(source_model),
        post_update_model=model,
        post_update_checkpoint_sha256=post_model,
    )
    return {
        "arm": arm,
        "update_rule": "ppo-lagrangian" if arm in {"F00", "F01"} else "cr-ppo",
        "intervention_beta": beta,
        "pre_model_sha256": pre_model,
        "pre_optimizer_sha256": pre_optimizer,
        "pre_tensor_rng_sha256": pre_rng,
        "post_model_sha256": post_model,
        "post_optimizer_sha256": post_optimizer,
        "post_tensor_rng_sha256": post_rng,
        "batch_sha256": array_mapping_sha256(enriched_batch),
        "ledger_row_count": len(ledger),
        "ledger_sha256": canonical_sha256(ledger),
        "ledger": ledger,
        "ppo_metrics": _ppo_metric_subset(metrics),
        "intervention_loss": float(metrics["intervention_loss"]),
        "intervention_penalty": float(metrics["intervention_penalty"]),
        "pre_update_shield_dependence": metrics[
            "pre_update_shield_dependence"
        ],
        "post_update_shield_dependence": metrics[
            "post_update_shield_dependence"
        ],
    }


def _reference_update(
    source_model: ActorCritic,
    source_optimizer: torch.optim.Optimizer,
    batch: Mapping[str, np.ndarray],
    config: PPOConfig,
    *,
    tensor_seed: int,
) -> dict[str, Any]:
    model, optimizer = copy.deepcopy((source_model, source_optimizer))
    validate_optimizer_model_binding(optimizer, model)
    generator = torch.Generator(device="cpu").manual_seed(tensor_seed)
    metrics = lagrangian_ppo_update(
        model,
        optimizer,
        {key: np.asarray(value).copy() for key, value in batch.items()},
        config,
        torch.device("cpu"),
        generator,
    )
    return {
        "post_model_sha256": model_parameter_sha256(model),
        "post_optimizer_sha256": optimizer_state_sha256(optimizer, model),
        "post_tensor_rng_sha256": generator_state_sha256(generator),
        "ppo_metrics": _ppo_metric_subset(metrics),
    }


def common_state_update_panel(
    source_model: ActorCritic,
    source_optimizer: torch.optim.Optimizer,
    episodes: Sequence[LongHorizonEpisode],
    transition_episodes: Sequence[Sequence[Mapping[str, Any]]],
    config: PPOConfig,
    *,
    beta: float,
    cr_thresholds: SafetyThresholds,
    expected_cr_mode: CRMode,
    tensor_seed: int,
    bindings: QualificationBindings,
) -> dict[str, Any]:
    """Run four arms from identical state and emit a replay-bound Q0 panel."""

    episode_ids = [episode.episode_id for episode in episodes]
    validate_bindings(bindings, episode_ids=episode_ids)
    cr_thresholds = validate_thresholds(cr_thresholds)
    if (
        len(episodes) != len(transition_episodes)
        or len(episodes) != config.episodes_per_update
        or type(tensor_seed) is not int
        or tensor_seed < 0
        or type(beta) not in {int, float}
        or isinstance(beta, bool)
        or not math.isfinite(float(beta))
        or float(beta) <= 0.0
        or expected_cr_mode not in {"reward", "unsafe", "wrong"}
        or config.minibatch_size != A25_FORMAL_MINIBATCH_SIZE
    ):
        raise ValueError("invalid A25 common-state panel schedule")
    decision_count = sum(len(rows) for rows in transition_episodes)
    if (
        decision_count <= 0
        or decision_count > len(episodes) * A25_MAX_EPISODE_STEPS
        or decision_count > config.minibatch_size
    ):
        raise ValueError("A25 whole-rollout minibatch invariant failed")
    validate_optimizer_model_binding(source_optimizer, source_model)
    initial_model = model_parameter_sha256(source_model)
    initial_optimizer = optimizer_state_sha256(source_optimizer, source_model)
    replay_generator = torch.Generator(device="cpu").manual_seed(
        bindings.rollout_seed
    )
    replay_rng_before = generator_state_sha256(replay_generator)
    replay = [
        replay_and_verify_shield_episode(
            episode,
            transitions,
            model=source_model,
            policy_generator=replay_generator,
            mean_incidents=cr_thresholds.mean_incidents,
            expected_episode_sha256=bindings.episode_sha256[episode.episode_id],
        )
        for episode, transitions in zip(
            episodes, transition_episodes, strict=True
        )
    ]
    replay_rng_after = generator_state_sha256(replay_generator)
    if any(row["invalid_actions"] or row["budget_violations"] for row in replay):
        raise RuntimeError("A25 replay integrity gate failed")
    unsafe_events = sum(
        any(float(row["unsafe_cost"]) > 0.0 for row in transitions)
        for transitions in transition_episodes
    )
    wrong_executions = sum(
        int(row["wrong_execute"])
        for transitions in transition_episodes
        for row in transitions
    )
    cr_decision = select_actor_mode(
        unsafe_events=unsafe_events,
        wrong_executions=wrong_executions,
        episodes=len(episodes),
        thresholds=cr_thresholds,
    )
    if cr_decision.mode != expected_cr_mode:
        raise ValueError("A25 common-state CR mode differs from expected mode")
    lagrangian_base = lagrangian_batch(
        transition_episodes, config, dual=DualState()
    )
    cr_base, cr_diagnostics = cr_ppo_batch(
        transition_episodes, config, decision=cr_decision
    )
    lagrangian_enriched = shield_aware_batch(
        transition_episodes, lagrangian_base
    )
    cr_enriched = shield_aware_batch(transition_episodes, cr_base)
    arms = {
        "F00": _arm_update(
            source_model,
            source_optimizer,
            transition_episodes,
            lagrangian_enriched,
            config,
            beta=0.0,
            tensor_seed=tensor_seed,
            bindings=bindings,
            arm="F00",
        ),
        "F01": _arm_update(
            source_model,
            source_optimizer,
            transition_episodes,
            lagrangian_enriched,
            config,
            beta=float(beta),
            tensor_seed=tensor_seed,
            bindings=bindings,
            arm="F01",
        ),
        "F10": _arm_update(
            source_model,
            source_optimizer,
            transition_episodes,
            cr_enriched,
            config,
            beta=0.0,
            tensor_seed=tensor_seed,
            bindings=bindings,
            arm="F10",
        ),
        "F11": _arm_update(
            source_model,
            source_optimizer,
            transition_episodes,
            cr_enriched,
            config,
            beta=float(beta),
            tensor_seed=tensor_seed,
            bindings=bindings,
            arm="F11",
        ),
    }
    references = {
        "F00": _reference_update(
            source_model,
            source_optimizer,
            lagrangian_base,
            config,
            tensor_seed=tensor_seed,
        ),
        "F10": _reference_update(
            source_model,
            source_optimizer,
            cr_base,
            config,
            tensor_seed=tensor_seed,
        ),
    }
    reference_exact = {
        arm: all(
            arms[arm][key] == references[arm][key]
            for key in (
                "post_model_sha256",
                "post_optimizer_sha256",
                "post_tensor_rng_sha256",
                "ppo_metrics",
            )
        )
        for arm in ("F00", "F10")
    }
    pre_hashes_equal = all(
        row["pre_model_sha256"] == initial_model
        and row["pre_optimizer_sha256"] == initial_optimizer
        for row in arms.values()
    )
    rng_schedule_equal = len(
        {row["pre_tensor_rng_sha256"] for row in arms.values()}
    ) == 1 and len(
        {row["post_tensor_rng_sha256"] for row in arms.values()}
    ) == 1
    execute_mass_key = "shielded_execute_probability_mass_mean_all_decisions"
    lagrangian_direction = (
        arms["F01"]["post_update_shield_dependence"][execute_mass_key]
        < arms["F00"]["post_update_shield_dependence"][execute_mass_key]
    )
    cr_direction = (
        arms["F11"]["post_update_shield_dependence"][execute_mass_key]
        < arms["F10"]["post_update_shield_dependence"][execute_mass_key]
    )
    gates = {
        "environment_replay_bound": len(replay) == len(episodes),
        "whole_rollout_single_minibatch": decision_count <= config.minibatch_size,
        "common_initial_model_optimizer": pre_hashes_equal,
        "common_tensor_rng_schedule": rng_schedule_equal,
        "f00_beta_zero_reference_exact": reference_exact["F00"],
        "f10_beta_zero_reference_exact": reference_exact["F10"],
        "f01_reduces_blocked_execute_mass_vs_f00": lagrangian_direction,
        "f11_reduces_blocked_execute_mass_vs_f10": cr_direction,
        "zero_execute_without_prior_review": all(
            row["post_update_shield_dependence"]["execute_without_review"] == 0
            for row in arms.values()
        ),
    }
    identity = {
        "qualification_version": A25_QUALIFICATION_VERSION,
        "a25_primitives_version": A25_PRIMITIVES_VERSION,
        "bindings": {
            **asdict(bindings),
            "episode_sha256": dict(sorted(bindings.episode_sha256.items())),
            "source_sha256": dict(sorted(bindings.source_sha256.items())),
        },
        "config": asdict(config),
        "config_sha256": canonical_sha256(asdict(config)),
        "cr_thresholds": asdict(cr_thresholds),
        "expected_cr_mode": expected_cr_mode,
        "intervention_beta": float(beta),
        "tensor_seed": tensor_seed,
        "initial_model_sha256": initial_model,
        "initial_optimizer_sha256": initial_optimizer,
        "ordered_episode_ids": episode_ids,
        "ordered_episode_ids_sha256": canonical_sha256(episode_ids),
        "replay_sha256": canonical_sha256(replay),
        "replay_policy_rng_before_sha256": replay_rng_before,
        "replay_policy_rng_after_sha256": replay_rng_after,
        "lagrangian_batch_sha256": array_mapping_sha256(lagrangian_enriched),
        "cr_batch_sha256": array_mapping_sha256(cr_enriched),
    }
    return {
        "schema_version": "multitown-a25-common-state-q0-panel-v1",
        "identity": identity,
        "decision_count": decision_count,
        "unsafe_events": unsafe_events,
        "wrong_executions": wrong_executions,
        "cr_mode_decision": asdict(cr_decision),
        "cr_batch_diagnostics": cr_diagnostics,
        "environment_replay": replay,
        "arms": arms,
        "references": references,
        "gates": gates,
        "passed": all(gates.values()),
    }
