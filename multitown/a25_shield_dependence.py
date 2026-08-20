"""Hard-shield-on intervention-avoidance primitives for A25 qualification.

This module is development infrastructure, not a formal A25 result.  Behaviour
sampling and PPO log-probabilities always use the effective hard-shield mask.
The auxiliary objective only measures the current policy distribution under
the environment's pre-shield base mask.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch
from torch import nn
from torch.nn.modules import module as torch_nn_module
from torch.optim import optimizer as torch_optimizer_module

from .a22_constrained_ppo import (
    effective_action_mask,
    lagrangian_ppo_update,
    public_review_state,
)
from .a23_cr_ppo import model_parameter_sha256, validate_optimizer_model_binding
from .long_horizon_env import (
    ACTION_COUNT,
    ACTION_NAMES,
    MultiTownLongHorizonEnv,
    RLAction,
)
from .ppo_controller import ActorCritic, PPOConfig, _masked_distribution

A25_PRIMITIVES_VERSION = "multitown-a25-shield-dependence-primitives-v3"
A25_LEDGER_SCHEMA_VERSION = "multitown-a25-shield-dependence-ledger-row-v2"
NAMED_GRADIENT_SCHEMA_VERSION = "multitown-a25-named-gradient-summary-v1"
NUMERICAL_GRADIENT_EVENT_SCHEMA_VERSION = (
    "multitown-a25-numerical-gradient-event-v1"
)
REVIEW_STATE_NAMES = ("unknown", "pass", "fail")
COUNTERFACTUAL_ARGMAX_TIE_RULE = "lowest-action-index"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_BASE_BATCH_FIELDS = frozenset(
    {
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
    }
)
_A25_BATCH_FIELDS = frozenset(
    {
        "base_mask",
        "shield_active",
        "episode_index",
        "transition_index",
        "rollout_pre_shield_logits",
        "rollout_base_execute_probability",
    }
)
_MODULE_EXECUTION_REGISTRIES = (
    "_modules",
    "_parameters",
    "_buffers",
    "_backward_hooks",
    "_backward_pre_hooks",
    "_forward_hooks",
    "_forward_hooks_always_called",
    "_forward_hooks_with_kwargs",
    "_forward_pre_hooks",
    "_forward_pre_hooks_with_kwargs",
)
_MODULE_EXECUTION_FLAGS = ("_is_full_backward_hook",)
_TENSOR_HOOK_REGISTRIES = ("_backward_hooks", "_post_accumulate_grad_hooks")
_OPTIMIZER_HOOK_REGISTRIES = (
    "_optimizer_step_pre_hooks",
    "_optimizer_step_post_hooks",
    "_optimizer_state_dict_pre_hooks",
    "_optimizer_state_dict_post_hooks",
    "_optimizer_load_state_dict_pre_hooks",
    "_optimizer_load_state_dict_post_hooks",
)
_GLOBAL_OPTIMIZER_HOOKS = (
    "_global_optimizer_pre_hooks",
    "_global_optimizer_post_hooks",
)
_GLOBAL_EXECUTION_HOOKS = (
    "_global_backward_hooks",
    "_global_backward_pre_hooks",
    "_global_forward_hooks",
    "_global_forward_hooks_always_called",
    "_global_forward_hooks_with_kwargs",
    "_global_forward_pre_hooks",
    "_global_is_full_backward_hook",
)


def _generator_state_sha256(generator: torch.Generator) -> str:
    if not isinstance(generator, torch.Generator):
        raise TypeError("invalid A25 tensor generator")
    state = generator.get_state().detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(generator.device).encode("ascii"))
    digest.update(b"\0")
    digest.update(state.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(
        json.dumps(list(state.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(state.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class InterventionObjective:
    """Frozen local configuration for the auxiliary A25 actor objective."""

    beta: float
    normalization: str = "all-decisions"
    hard_shield_required: bool = True


@dataclass(frozen=True)
class InterventionAuxiliaryTelemetry:
    """Unscaled loss and the exact tensors that define its intervention mass."""

    loss: torch.Tensor
    active: torch.Tensor
    base_execute_probability: torch.Tensor


@dataclass(frozen=True)
class InterventionLossTerms:
    """Exact positive-beta minibatch terms from the production loss graph."""

    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy: torch.Tensor
    base_total_loss: torch.Tensor
    auxiliary_loss: torch.Tensor
    intervention_penalty: torch.Tensor
    total_loss: torch.Tensor
    log_ratio: torch.Tensor
    ratio: torch.Tensor


@dataclass(frozen=True)
class NumericalGradientEvent:
    """Live same-graph view for trusted internal numerical instrumentation."""

    schema_version: str
    stage: str
    epoch_index: int
    minibatch_index: int
    indices: tuple[int, ...]
    indices_sha256: str
    named_parameters: tuple[tuple[str, nn.Parameter], ...]
    loss_terms: InterventionLossTerms | None
    gradient_summary: dict[str, Any] | None
    clip_returned_norm: float | None


@runtime_checkable
class NumericalGradientObserver(Protocol):
    """Trusted internal-only observer for the production numerical audit.

    Events expose live tensors so an internal collector can use ``autograd.grad``
    on the production graph.  This callback boundary is not a Python security
    sandbox and must never be used for untrusted or third-party code.  The state
    guard is defense in depth for enumerated accidental mutations only.
    """

    def observe(self, event: NumericalGradientEvent) -> None:
        """Inspect one live stage without mutating its guarded training state."""


def validate_objective(objective: InterventionObjective) -> InterventionObjective:
    if (
        type(objective.beta) not in (int, float)
        or isinstance(objective.beta, bool)
        or not math.isfinite(float(objective.beta))
        or float(objective.beta) < 0.0
        or objective.normalization != "all-decisions"
        or objective.hard_shield_required is not True
    ):
        raise ValueError("invalid A25 intervention objective")
    return objective


def unscaled_intervention_auxiliary_loss(
    logits: torch.Tensor,
    base_mask: torch.Tensor,
    active: torch.Tensor,
) -> InterventionAuxiliaryTelemetry:
    """Return the production auxiliary loss before applying ``beta``.

    Normalization is over every supplied decision.  ``active`` identifies rows
    where the hard shield blocks pre-review EXECUTE, while the probability is
    always computed under the environment's pre-shield base mask.
    """

    if (
        not torch.is_tensor(logits)
        or not torch.is_tensor(base_mask)
        or not torch.is_tensor(active)
    ):
        raise TypeError("invalid A25 auxiliary tensor input")
    count = len(logits) if logits.ndim == 2 else 0
    if (
        count <= 0
        or logits.shape != (count, ACTION_COUNT)
        or not logits.is_floating_point()
        or not bool(torch.isfinite(logits).all())
        or base_mask.shape != logits.shape
        or base_mask.dtype != torch.bool
        or base_mask.device != logits.device
        or not bool(base_mask.any(dim=1).all())
        or active.shape != (count,)
        or active.dtype != torch.bool
        or active.device != logits.device
    ):
        raise ValueError("invalid A25 auxiliary tensor schema")
    base_distribution = _masked_distribution(logits, base_mask)
    base_execute_probability = base_distribution.probs[:, int(RLAction.EXECUTE)]
    loss = (active.to(dtype=logits.dtype) * base_execute_probability).mean()
    if not bool(torch.isfinite(base_execute_probability).all()) or not bool(
        torch.isfinite(loss)
    ):
        raise FloatingPointError("non-finite A25 auxiliary loss")
    return InterventionAuxiliaryTelemetry(
        loss=loss,
        active=active,
        base_execute_probability=base_execute_probability,
    )


def intervention_minibatch_loss_terms(
    *,
    new_log_probability: torch.Tensor,
    old_log_probability: torch.Tensor,
    advantage: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    entropy: torch.Tensor,
    auxiliary: InterventionAuxiliaryTelemetry,
    clip_ratio: float,
    value_coef: float,
    entropy_coef: float,
    beta: float,
) -> InterventionLossTerms:
    """Compose the production positive-beta loss without scaling ambiguity."""

    vectors = (
        new_log_probability,
        old_log_probability,
        advantage,
        values,
        returns,
    )
    if not all(torch.is_tensor(value) for value in vectors):
        raise TypeError("invalid A25 positive-beta loss tensor input")
    if not isinstance(auxiliary, InterventionAuxiliaryTelemetry):
        raise TypeError("invalid A25 auxiliary loss telemetry")
    count = len(new_log_probability) if new_log_probability.ndim == 1 else 0
    if (
        count <= 0
        or any(value.shape != (count,) for value in vectors)
        or any(value.device != new_log_probability.device for value in vectors)
        or any(not value.is_floating_point() for value in vectors)
        or any(not bool(torch.isfinite(value).all()) for value in vectors)
        or not torch.is_tensor(entropy)
        or entropy.ndim != 0
        or entropy.device != new_log_probability.device
        or not entropy.is_floating_point()
        or not bool(torch.isfinite(entropy))
        or auxiliary.loss.ndim != 0
        or auxiliary.loss.device != new_log_probability.device
        or not bool(torch.isfinite(auxiliary.loss))
        or not _finite_float(clip_ratio)
        or not 0.0 <= float(clip_ratio) < 1.0
        or not _finite_float(value_coef)
        or float(value_coef) < 0.0
        or not _finite_float(entropy_coef)
        or float(entropy_coef) < 0.0
        or not _finite_float(beta)
        or float(beta) <= 0.0
    ):
        raise ValueError("invalid A25 positive-beta loss terms")
    log_ratio = new_log_probability - old_log_probability
    ratio = log_ratio.exp()
    unclipped = ratio * advantage
    clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantage
    policy_loss = -torch.minimum(unclipped, clipped).mean()
    value_loss = 0.5 * (values - returns).square().mean()
    base_total_loss = (
        policy_loss + value_coef * value_loss - entropy_coef * entropy
    )
    intervention_penalty = beta * auxiliary.loss
    total_loss = base_total_loss + intervention_penalty
    if any(
        not bool(torch.isfinite(value))
        for value in (
            policy_loss,
            value_loss,
            base_total_loss,
            intervention_penalty,
            total_loss,
        )
    ):
        raise FloatingPointError("non-finite A25 positive-beta loss")
    return InterventionLossTerms(
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy=entropy,
        base_total_loss=base_total_loss,
        auxiliary_loss=auxiliary.loss,
        intervention_penalty=intervention_penalty,
        total_loss=total_loss,
        log_ratio=log_ratio,
        ratio=ratio,
    )


def _little_endian_gradient_bytes(gradient: torch.Tensor) -> tuple[str, bytes]:
    supported = {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    }
    if gradient.layout != torch.strided or gradient.dtype not in supported:
        raise TypeError("unsupported A25 gradient tensor")
    tensor = gradient.detach().cpu().contiguous()
    if not bool(torch.isfinite(tensor).all()):
        raise FloatingPointError("non-finite A25 named gradient")
    if tensor.dtype == torch.bfloat16:
        array = tensor.view(torch.uint16).numpy()
        little = np.ascontiguousarray(array, dtype=np.dtype("<u2"))
        return "<bf2", little.tobytes(order="C")
    array = tensor.numpy()
    little = np.ascontiguousarray(array, dtype=array.dtype.newbyteorder("<"))
    return little.dtype.str, little.tobytes(order="C")


def canonical_named_gradient_summary(model: nn.Module) -> dict[str, Any]:
    """Hash and summarize gradients in original ``named_parameters`` order."""

    if not isinstance(model, nn.Module):
        raise TypeError("A25 named-gradient source must be a module")
    named_parameters = list(model.named_parameters())
    if not named_parameters:
        raise ValueError("A25 named-gradient source has no parameters")
    digest = hashlib.sha256()
    digest.update(NAMED_GRADIENT_SCHEMA_VERSION.encode("ascii") + b"\0")
    entries: list[dict[str, Any]] = []
    groups: dict[str, dict[str, int | float]] = {}
    seen: set[str] = set()
    total_squared_norm = 0.0
    for name, parameter in named_parameters:
        if (
            type(name) is not str
            or not name
            or name in seen
            or not isinstance(parameter, nn.Parameter)
        ):
            raise ValueError("invalid A25 named-parameter sequence")
        seen.add(name)
        group = name.partition(".")[0]
        group_state = groups.setdefault(
            group,
            {
                "parameter_count": 0,
                "none_count": 0,
                "tensor_count": 0,
                "squared_l2_norm": 0.0,
            },
        )
        group_state["parameter_count"] += 1
        name_payload = name.encode("utf-8")
        digest.update(
            b"name\0"
            + str(len(name_payload)).encode("ascii")
            + b"\0"
            + name_payload
            + b"\0"
        )
        gradient = parameter.grad
        if gradient is None:
            digest.update(b"none\0")
            group_state["none_count"] += 1
            entries.append(
                {
                    "name": name,
                    "group": group,
                    "state": "none",
                    "dtype": None,
                    "little_endian_dtype": None,
                    "shape": None,
                    "little_endian_nbytes": None,
                    "little_endian_sha256": None,
                    "l2_norm": None,
                }
            )
            continue
        if (
            not torch.is_tensor(gradient)
            or gradient.shape != parameter.shape
            or gradient.dtype != parameter.dtype
            or gradient.device != parameter.device
        ):
            raise TypeError("invalid A25 named gradient tensor binding")
        little_dtype, payload = _little_endian_gradient_bytes(gradient)
        dtype = str(gradient.dtype)
        shape = list(gradient.shape)
        squared_norm = float(
            gradient.detach().to(device="cpu", dtype=torch.float64).square().sum()
        )
        if not math.isfinite(squared_norm):
            raise FloatingPointError("non-finite A25 gradient norm")
        group_state["tensor_count"] += 1
        group_state["squared_l2_norm"] += squared_norm
        total_squared_norm += squared_norm
        dtype_payload = dtype.encode("ascii")
        little_dtype_payload = little_dtype.encode("ascii")
        shape_payload = json.dumps(shape, separators=(",", ":")).encode("ascii")
        digest.update(b"tensor\0")
        for item in (dtype_payload, little_dtype_payload, shape_payload, payload):
            digest.update(str(len(item)).encode("ascii") + b"\0" + item + b"\0")
        entries.append(
            {
                "name": name,
                "group": group,
                "state": "tensor",
                "dtype": dtype,
                "little_endian_dtype": little_dtype,
                "shape": shape,
                "little_endian_nbytes": len(payload),
                "little_endian_sha256": hashlib.sha256(payload).hexdigest(),
                "l2_norm": math.sqrt(squared_norm),
            }
        )
    group_summaries = [
        {
            "group": name,
            "parameter_count": int(state["parameter_count"]),
            "none_count": int(state["none_count"]),
            "tensor_count": int(state["tensor_count"]),
            "l2_norm": math.sqrt(float(state["squared_l2_norm"])),
        }
        for name, state in groups.items()
    ]
    return {
        "schema_version": NAMED_GRADIENT_SCHEMA_VERSION,
        "digest_sha256": digest.hexdigest(),
        "parameter_count": len(entries),
        "none_count": sum(row["state"] == "none" for row in entries),
        "tensor_count": sum(row["state"] == "tensor" for row in entries),
        "total_l2_norm": math.sqrt(total_squared_norm),
        "parameters": entries,
        "groups": group_summaries,
    }


def _numerical_observer_indices(
    indices: torch.Tensor,
) -> tuple[tuple[int, ...], str]:
    if indices.ndim != 1 or indices.dtype != torch.int64:
        raise ValueError("invalid A25 numerical observer indices")
    array = indices.detach().to(device="cpu").contiguous().numpy()
    little = np.ascontiguousarray(array, dtype=np.dtype("<i8"))
    values = tuple(int(value) for value in little.tolist())
    digest = hashlib.sha256()
    digest.update(NUMERICAL_GRADIENT_EVENT_SCHEMA_VERSION.encode("ascii") + b"\0")
    digest.update(str(len(values)).encode("ascii") + b"\0")
    digest.update(little.tobytes(order="C"))
    return values, digest.hexdigest()


def _loss_term_tensor_sequence(
    terms: InterventionLossTerms | None,
) -> tuple[torch.Tensor, ...]:
    if terms is None:
        return ()
    return (
        terms.policy_loss,
        terms.value_loss,
        terms.entropy,
        terms.base_total_loss,
        terms.auxiliary_loss,
        terms.intervention_penalty,
        terms.total_loss,
        terms.log_ratio,
        terms.ratio,
    )


def _numpy_rng_state_equal(left: tuple[Any, ...], right: tuple[Any, ...]) -> bool:
    return bool(
        len(left) == len(right) == 5
        and left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def _observer_tensor_bytes(tensor: torch.Tensor) -> bytes:
    if tensor.layout != torch.strided:
        raise TypeError("unsupported A25 observer optimizer tensor")
    return (
        tensor.detach()
        .to(device="cpu")
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
        .numpy()
        .tobytes(order="C")
    )


def _observer_scalar_fingerprint(value: Any) -> tuple[Any, ...]:
    if isinstance(value, np.generic):
        return (type(value), value.dtype.str, value.tobytes())
    if isinstance(value, float):
        return (float, value.hex())
    if isinstance(value, complex):
        return (complex, value.real.hex(), value.imag.hex())
    if type(value) in {type(None), bool, int, str, bytes}:
        return (type(value), value)
    return (type(value), repr(value), id(value))


def _observer_optimizer_value_snapshot(value: Any) -> tuple[Any, ...]:
    if torch.is_tensor(value):
        return (
            "tensor",
            value,
            type(value),
            value.dtype,
            tuple(value.shape),
            value.device,
            value.layout,
            tuple(value.stride()),
            value.requires_grad,
            value._version,
            _observer_tensor_bytes(value),
            value.detach().clone(memory_format=torch.preserve_format),
        )
    if isinstance(value, Mapping):
        return (
            "mapping",
            value,
            type(value),
            tuple(
                (
                    key,
                    _observer_scalar_fingerprint(key),
                    _observer_optimizer_value_snapshot(item),
                )
                for key, item in value.items()
            ),
        )
    if isinstance(value, list):
        return (
            "list",
            value,
            type(value),
            tuple(_observer_optimizer_value_snapshot(item) for item in value),
        )
    if isinstance(value, tuple):
        return (
            "tuple",
            value,
            type(value),
            tuple(_observer_optimizer_value_snapshot(item) for item in value),
        )
    return ("scalar", _observer_scalar_fingerprint(value), value)


def _observer_optimizer_value_matches(
    value: Any, snapshot: tuple[Any, ...]
) -> bool:
    kind = snapshot[0]
    if kind == "tensor":
        return bool(
            torch.is_tensor(value)
            and value is snapshot[1]
            and type(value) is snapshot[2]
            and value.dtype == snapshot[3]
            and tuple(value.shape) == snapshot[4]
            and value.device == snapshot[5]
            and value.layout == snapshot[6]
            and tuple(value.stride()) == snapshot[7]
            and value.requires_grad is snapshot[8]
            and value._version == snapshot[9]
            and _observer_tensor_bytes(value) == snapshot[10]
        )
    if kind == "mapping":
        if value is not snapshot[1] or type(value) is not snapshot[2]:
            return False
        current_items = tuple(value.items())
        expected_items = snapshot[3]
        return bool(
            len(current_items) == len(expected_items)
            and all(
                _observer_scalar_fingerprint(current_key) == key_fingerprint
                and _observer_optimizer_value_matches(current_value, item_snapshot)
                for (current_key, current_value), (
                    _key,
                    key_fingerprint,
                    item_snapshot,
                ) in zip(current_items, expected_items, strict=True)
            )
        )
    if kind in {"list", "tuple"}:
        if value is not snapshot[1] or type(value) is not snapshot[2]:
            return False
        expected_items = snapshot[3]
        return bool(
            len(value) == len(expected_items)
            and all(
                _observer_optimizer_value_matches(item, item_snapshot)
                for item, item_snapshot in zip(value, expected_items, strict=True)
            )
        )
    return _observer_scalar_fingerprint(value) == snapshot[1]


def _restore_observer_optimizer_value(snapshot: tuple[Any, ...]) -> Any:
    kind = snapshot[0]
    if kind == "tensor":
        original = snapshot[1]
        saved = snapshot[11]
        with torch.no_grad():
            if original.shape != saved.shape:
                original.resize_(saved.shape)
            original.copy_(saved)
            original.requires_grad_(snapshot[8])
            torch._C._autograd._unsafe_set_version_counter(
                [original], [snapshot[9]]
            )
        return original
    if kind == "mapping":
        original = snapshot[1]
        original.clear()
        for key, _key_fingerprint, item_snapshot in snapshot[3]:
            original[key] = _restore_observer_optimizer_value(item_snapshot)
        return original
    if kind == "list":
        original = snapshot[1]
        original.clear()
        original.extend(
            _restore_observer_optimizer_value(item_snapshot)
            for item_snapshot in snapshot[3]
        )
        return original
    if kind == "tuple":
        return snapshot[1]
    return snapshot[2]


def _observer_optimizer_snapshot(
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    return {
        "state": _observer_optimizer_value_snapshot(optimizer.state),
        "param_groups": _observer_optimizer_value_snapshot(optimizer.param_groups),
        "defaults": _observer_optimizer_value_snapshot(optimizer.defaults),
        "instance_hooks": tuple(
            (
                registry,
                _observer_optimizer_value_snapshot(getattr(optimizer, registry)),
            )
            for registry in _OPTIMIZER_HOOK_REGISTRIES
        ),
        "global_hooks": tuple(
            (
                registry,
                _observer_optimizer_value_snapshot(
                    getattr(torch_optimizer_module, registry)
                ),
            )
            for registry in _GLOBAL_OPTIMIZER_HOOKS
        ),
    }


def _observer_optimizer_matches(
    optimizer: torch.optim.Optimizer,
    snapshot: Mapping[str, Any],
) -> bool:
    return bool(
        _observer_optimizer_value_matches(optimizer.state, snapshot["state"])
        and _observer_optimizer_value_matches(
            optimizer.param_groups, snapshot["param_groups"]
        )
        and _observer_optimizer_value_matches(
            optimizer.defaults, snapshot["defaults"]
        )
        and all(
            _observer_optimizer_value_matches(
                getattr(optimizer, registry), registry_snapshot
            )
            for registry, registry_snapshot in snapshot["instance_hooks"]
        )
        and all(
            _observer_optimizer_value_matches(
                getattr(torch_optimizer_module, registry), registry_snapshot
            )
            for registry, registry_snapshot in snapshot["global_hooks"]
        )
    )


def _restore_observer_optimizer(
    optimizer: torch.optim.Optimizer,
    snapshot: Mapping[str, Any],
) -> None:
    optimizer.state = _restore_observer_optimizer_value(snapshot["state"])
    optimizer.param_groups = _restore_observer_optimizer_value(
        snapshot["param_groups"]
    )
    optimizer.defaults = _restore_observer_optimizer_value(snapshot["defaults"])
    for registry, registry_snapshot in snapshot["instance_hooks"]:
        setattr(
            optimizer,
            registry,
            _restore_observer_optimizer_value(registry_snapshot),
        )
    for registry, registry_snapshot in snapshot["global_hooks"]:
        setattr(
            torch_optimizer_module,
            registry,
            _restore_observer_optimizer_value(registry_snapshot),
        )


def _observer_model_execution_snapshot(
    model: nn.Module,
    named_parameters: tuple[tuple[str, nn.Parameter], ...],
    terms: InterventionLossTerms | None,
) -> dict[str, Any]:
    named_modules = tuple(model.named_modules())
    return {
        "named_modules": named_modules,
        "modules": tuple(
            (
                name,
                module,
                module.training,
                tuple(
                    (
                        registry,
                        _observer_optimizer_value_snapshot(
                            getattr(module, registry)
                        ),
                    )
                    for registry in _MODULE_EXECUTION_REGISTRIES
                ),
                tuple(
                    (
                        flag,
                        _observer_optimizer_value_snapshot(getattr(module, flag)),
                    )
                    for flag in _MODULE_EXECUTION_FLAGS
                ),
            )
            for name, module in named_modules
        ),
        "named_buffers": tuple(
            (name, buffer, _observer_optimizer_value_snapshot(buffer))
            for name, buffer in model.named_buffers()
        ),
        "parameter_hooks": tuple(
            (
                name,
                parameter,
                tuple(
                    (
                        registry,
                        _observer_optimizer_value_snapshot(
                            getattr(parameter, registry)
                        ),
                    )
                    for registry in _TENSOR_HOOK_REGISTRIES
                ),
            )
            for name, parameter in named_parameters
        ),
        "loss_tensor_hooks": tuple(
            (
                tensor,
                tuple(
                    (
                        registry,
                        _observer_optimizer_value_snapshot(
                            getattr(tensor, registry)
                        ),
                    )
                    for registry in _TENSOR_HOOK_REGISTRIES
                ),
            )
            for tensor in _loss_term_tensor_sequence(terms)
        ),
        "global_hooks": tuple(
            (
                name,
                _observer_optimizer_value_snapshot(
                    getattr(torch_nn_module, name)
                ),
            )
            for name in _GLOBAL_EXECUTION_HOOKS
        ),
    }


def _observer_model_execution_matches(
    model: nn.Module,
    snapshot: Mapping[str, Any],
) -> bool:
    current_modules = tuple(model.named_modules())
    expected_modules = snapshot["named_modules"]
    if len(current_modules) != len(expected_modules) or any(
        current_name != expected_name or current_module is not expected_module
        for (current_name, current_module), (expected_name, expected_module) in zip(
            current_modules, expected_modules, strict=True
        )
    ):
        return False
    for name, module, training, registries, flags in snapshot["modules"]:
        del name
        if module.training is not training or any(
            not _observer_optimizer_value_matches(
                getattr(module, registry), registry_snapshot
            )
            for registry, registry_snapshot in registries
        ) or any(
            not _observer_optimizer_value_matches(
                getattr(module, flag), flag_snapshot
            )
            for flag, flag_snapshot in flags
        ):
            return False
    current_buffers = tuple(model.named_buffers())
    expected_buffers = snapshot["named_buffers"]
    if len(current_buffers) != len(expected_buffers) or any(
        current_name != expected_name
        or current_buffer is not expected_buffer
        or not _observer_optimizer_value_matches(current_buffer, buffer_snapshot)
        for (current_name, current_buffer), (
            expected_name,
            expected_buffer,
            buffer_snapshot,
        ) in zip(current_buffers, expected_buffers, strict=True)
    ):
        return False
    if any(
        not _observer_optimizer_value_matches(
            getattr(parameter, registry), registry_snapshot
        )
        for _name, parameter, registries in snapshot["parameter_hooks"]
        for registry, registry_snapshot in registries
    ) or any(
        not _observer_optimizer_value_matches(
            getattr(tensor, registry), registry_snapshot
        )
        for tensor, registries in snapshot["loss_tensor_hooks"]
        for registry, registry_snapshot in registries
    ):
        return False
    return all(
        _observer_optimizer_value_matches(
            getattr(torch_nn_module, name), hook_snapshot
        )
        for name, hook_snapshot in snapshot["global_hooks"]
    )


def _restore_observer_model_execution(snapshot: Mapping[str, Any]) -> None:
    for _name, module, training, registries, flags in snapshot["modules"]:
        for registry, registry_snapshot in registries:
            setattr(
                module,
                registry,
                _restore_observer_optimizer_value(registry_snapshot),
            )
        for flag, flag_snapshot in flags:
            setattr(
                module,
                flag,
                _restore_observer_optimizer_value(flag_snapshot),
            )
        module.training = training
    for _name, parameter, registries in snapshot["parameter_hooks"]:
        for registry, registry_snapshot in registries:
            setattr(
                parameter,
                registry,
                _restore_observer_optimizer_value(registry_snapshot),
            )
    for tensor, registries in snapshot["loss_tensor_hooks"]:
        for registry, registry_snapshot in registries:
            setattr(
                tensor,
                registry,
                _restore_observer_optimizer_value(registry_snapshot),
            )
    for name, hook_snapshot in snapshot["global_hooks"]:
        setattr(
            torch_nn_module,
            name,
            _restore_observer_optimizer_value(hook_snapshot),
        )


def _numerical_observer_protected_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    named_parameters: tuple[tuple[str, nn.Parameter], ...],
    terms: InterventionLossTerms | None,
    device: torch.device,
) -> dict[str, Any]:
    parameter_state = []
    for name, parameter in named_parameters:
        gradient = parameter.grad
        parameter_state.append(
            (
                name,
                parameter,
                parameter.detach().clone(),
                parameter._version,
                parameter.requires_grad,
                gradient,
                None if gradient is None else gradient.detach().clone(),
                None if gradient is None else gradient._version,
            )
        )
    loss_tensors = _loss_term_tensor_sequence(terms)
    return {
        "parameter_state": tuple(parameter_state),
        "loss_tensors": tuple(
            (tensor, tensor.detach().clone(), tensor._version)
            for tensor in loss_tensors
        ),
        "generator": generator.get_state().clone(),
        "torch_rng": torch.random.get_rng_state().clone(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        "cuda_rng": (
            torch.cuda.get_rng_state(device).clone()
            if device.type == "cuda"
            else None
        ),
        "optimizer": _observer_optimizer_snapshot(optimizer),
        "model_execution": _observer_model_execution_snapshot(
            model, named_parameters, terms
        ),
    }


def _numerical_observer_state_unchanged(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    named_parameters: tuple[tuple[str, nn.Parameter], ...],
    terms: InterventionLossTerms | None,
    device: torch.device,
    protected: Mapping[str, Any],
) -> bool:
    try:
        current_named = tuple(model.named_parameters())
        if len(current_named) != len(named_parameters) or any(
            current_name != expected_name or current_parameter is not expected_parameter
            for (current_name, current_parameter), (
                expected_name,
                expected_parameter,
            ) in zip(current_named, named_parameters, strict=True)
        ):
            return False
        for (
            name,
            parameter,
            value,
            version,
            requires_grad,
            gradient,
            gradient_value,
            gradient_version,
        ) in protected["parameter_state"]:
            del name
            if (
                parameter._version != version
                or parameter.requires_grad is not requires_grad
                or not torch.equal(parameter.detach(), value)
            ):
                return False
            if gradient is None:
                if parameter.grad is not None:
                    return False
            elif (
                parameter.grad is not gradient
                or gradient_value is None
                or gradient._version != gradient_version
                or not torch.equal(parameter.grad.detach(), gradient_value)
            ):
                return False
        current_loss_tensors = _loss_term_tensor_sequence(terms)
        if len(current_loss_tensors) != len(protected["loss_tensors"]):
            return False
        if any(
            current is not expected
            or current._version != version
            or not torch.equal(current.detach(), value)
            for current, (expected, value, version) in zip(
                current_loss_tensors,
                protected["loss_tensors"],
                strict=True,
            )
        ):
            return False
        if (
            not torch.equal(generator.get_state(), protected["generator"])
            or not torch.equal(torch.random.get_rng_state(), protected["torch_rng"])
            or not _numpy_rng_state_equal(
                np.random.get_state(), protected["numpy_rng"]
            )
            or random.getstate() != protected["python_rng"]
        ):
            return False
        if protected["cuda_rng"] is not None and not torch.equal(
            torch.cuda.get_rng_state(device), protected["cuda_rng"]
        ):
            return False
        if not _observer_optimizer_matches(optimizer, protected["optimizer"]):
            return False
        if not _observer_model_execution_matches(
            model, protected["model_execution"]
        ):
            return False
    except (RuntimeError, TypeError, ValueError):
        return False
    return True


def _restore_numerical_observer_protected_state(
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    device: torch.device,
    protected: Mapping[str, Any],
) -> None:
    _restore_observer_optimizer(optimizer, protected["optimizer"])
    with torch.no_grad():
        for (
            _name,
            parameter,
            value,
            _version,
            requires_grad,
            gradient,
            gradient_value,
            gradient_version,
        ) in protected["parameter_state"]:
            if parameter.shape == value.shape:
                parameter.copy_(value)
            else:
                parameter.data = value.clone()
            parameter.requires_grad_(requires_grad)
            if gradient is None:
                parameter.grad = None
            else:
                if gradient.shape == gradient_value.shape:
                    gradient.copy_(gradient_value)
                else:
                    gradient.data = gradient_value.clone()
                torch._C._autograd._unsafe_set_version_counter(
                    [gradient], [gradient_version]
                )
                parameter.grad = gradient
        for tensor, value, _version in protected["loss_tensors"]:
            if tensor.shape == value.shape:
                tensor.copy_(value)
            else:
                tensor.data = value.clone()
            torch._C._autograd._unsafe_set_version_counter(
                [tensor], [_version]
            )
        torch._C._autograd._unsafe_set_version_counter(
            [row[1] for row in protected["parameter_state"]],
            [row[3] for row in protected["parameter_state"]],
        )
    _restore_observer_model_execution(protected["model_execution"])
    generator.set_state(protected["generator"])
    torch.random.set_rng_state(protected["torch_rng"])
    np.random.set_state(protected["numpy_rng"])
    random.setstate(protected["python_rng"])
    if protected["cuda_rng"] is not None:
        torch.cuda.set_rng_state(protected["cuda_rng"], device)


def _notify_numerical_gradient_observer(
    observer: NumericalGradientObserver,
    event: NumericalGradientEvent,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    named_parameters: tuple[tuple[str, nn.Parameter], ...],
    device: torch.device,
) -> None:
    protected = _numerical_observer_protected_state(
        model,
        optimizer,
        generator,
        named_parameters,
        event.loss_terms,
        device,
    )
    try:
        observer.observe(event)
    finally:
        if not _numerical_observer_state_unchanged(
            model,
            optimizer,
            generator,
            named_parameters,
            event.loss_terms,
            device,
            protected,
        ):
            _restore_numerical_observer_protected_state(
                optimizer, generator, device, protected
            )
            raise RuntimeError(
                "A25 numerical gradient observer mutated protected state"
            )


def _finite_float(value: Any) -> bool:
    return bool(
        not isinstance(value, (bool, np.bool_))
        and isinstance(value, (int, float, np.integer, np.floating))
        and math.isfinite(float(value))
    )


def _finite_state(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, Mapping):
        return all(_finite_state(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_state(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    return True


def _validate_model_optimizer(
    model: nn.Module, optimizer: torch.optim.Optimizer
) -> None:
    validate_optimizer_model_binding(optimizer, model)  # type: ignore[arg-type]
    if any(
        tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all())
        for tensor in model.state_dict().values()
    ) or any(
        parameter.grad is not None
        and not bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    ):
        raise FloatingPointError("invalid A25 model state")
    optimizer_options = [
        {key: value for key, value in group.items() if key != "params"}
        for group in optimizer.param_groups
    ]
    if not _finite_state(optimizer.state) or not _finite_state(optimizer_options):
        raise FloatingPointError("invalid A25 optimizer state")


def _policy_snapshot(
    model: nn.Module,
    observation: torch.Tensor,
    base_mask: torch.Tensor,
    effective_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    logits, values = model(observation)
    if (
        logits.shape != (len(observation), ACTION_COUNT)
        or values.shape != (len(observation),)
        or not bool(torch.isfinite(logits).all())
        or not bool(torch.isfinite(values).all())
    ):
        raise FloatingPointError("invalid A25 policy output")
    base_distribution = _masked_distribution(logits, base_mask)
    effective_distribution = _masked_distribution(logits, effective_mask)
    base_probability = base_distribution.probs[:, int(RLAction.EXECUTE)]
    counterfactual_argmax = logits.masked_fill(
        ~base_mask, torch.finfo(logits.dtype).min
    ).argmax(dim=-1)
    masked_logits = logits.masked_fill(~base_mask, torch.finfo(logits.dtype).min)
    counterfactual_maximum = masked_logits.max(dim=-1, keepdim=True).values
    counterfactual_ties = base_mask & (masked_logits == counterfactual_maximum)
    if not bool(torch.isfinite(base_probability).all()) or not bool(
        torch.isfinite(effective_distribution.probs).all()
    ):
        raise FloatingPointError("non-finite A25 action probability")
    return {
        "logits": logits,
        "values": values,
        "base_execute_probability": base_probability,
        "counterfactual_argmax": counterfactual_argmax,
        "counterfactual_ties": counterfactual_ties,
        "effective_probabilities": effective_distribution.probs,
    }


def shield_aware_rollout(
    model: ActorCritic,
    episode: Any,
    device: torch.device,
    *,
    mean_incidents: float,
    generator: torch.Generator | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collect an on-policy rollout with hard shield and pre-shield telemetry."""

    if not math.isfinite(mean_incidents) or mean_incidents <= 0.0:
        raise ValueError("mean incidents must be finite and positive")
    if generator is not None and not isinstance(generator, torch.Generator):
        raise TypeError("invalid A25 rollout generator")
    if generator is not None and torch.device(generator.device) != device:
        raise ValueError("A25 rollout generator/device mismatch")
    env = MultiTownLongHorizonEnv(episode)
    observation, _ = env.reset()
    transitions: list[dict[str, Any]] = []
    total_return = 0.0
    unsafe_seen = False
    while not env.terminated:
        base_mask = env.action_mask()
        mask, shield_active = effective_action_mask(
            observation, base_mask, shield_enabled=True
        )
        obs_tensor = torch.from_numpy(observation).to(device).unsqueeze(0)
        base_tensor = torch.tensor(
            base_mask, dtype=torch.bool, device=device
        ).unsqueeze(0)
        mask_tensor = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
        with torch.no_grad():
            snapshot = _policy_snapshot(model, obs_tensor, base_tensor, mask_tensor)
            effective_distribution = _masked_distribution(
                snapshot["logits"], mask_tensor
            )
            policy_rng_before = (
                _generator_state_sha256(generator)
                if generator is not None
                else None
            )
            if generator is None:
                action = effective_distribution.sample()
            else:
                action = torch.multinomial(
                    effective_distribution.probs,
                    num_samples=1,
                    replacement=True,
                    generator=generator,
                ).squeeze(-1)
            policy_rng_after = (
                _generator_state_sha256(generator)
                if generator is not None
                else None
            )
            old_log_probability = effective_distribution.log_prob(action)
        selected = int(action.item())
        review_state = public_review_state(observation)
        counterfactual_argmax = int(snapshot["counterfactual_argmax"].item())
        tokens_before = int(env.tokens_used)
        latency_before = float(env.latency_used_s)
        next_observation, reward, terminated, truncated, step_info = env.step(
            selected
        )
        trajectory_row = env.trajectory[-1]
        reward_components = trajectory_row["reward"]
        wrong_execute = bool(
            selected == int(RLAction.EXECUTE)
            and float(reward_components["safety_penalty"]) < 0.0
        )
        first_unsafe = wrong_execute and not unsafe_seen
        unsafe_seen = unsafe_seen or wrong_execute
        if not terminated:
            termination_reason = "not-terminal"
        elif truncated:
            termination_reason = "max-steps"
        elif float(reward_components["budget_violation"]) < 0.0:
            termination_reason = "budget-exhausted"
        elif selected == int(RLAction.STOP):
            termination_reason = "agent-stop"
        else:
            raise RuntimeError("unclassified A25 termination")
        transition = {
            "episode_id": str(episode.episode_id),
            "split": str(episode.split),
            "observation": observation,
            "next_observation": next_observation.astype(np.float32, copy=True),
            "base_mask": base_mask,
            "mask": mask,
            "action": selected,
            "sampled_action": selected,
            "executed_action": selected,
            "old_log_probability": float(old_log_probability.item()),
            "old_value": float(snapshot["values"].item()),
            "reward": float(reward),
            "reward_components": {
                key: float(value) for key, value in reward_components.items()
            },
            "unsafe_cost": float(first_unsafe),
            "wrong_cost": float(wrong_execute) / mean_incidents,
            "wrong_execute": wrong_execute,
            "invalid_action": bool(float(reward_components["invalid_action"]) < 0.0),
            "budget_violation": bool(
                float(reward_components["budget_violation"]) < 0.0
            ),
            "shield_intervened": shield_active,
            "review_state": review_state,
            "pre_shield_logits": (
                snapshot["logits"]
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=True)
            ),
            "base_execute_probability": float(
                snapshot["base_execute_probability"].item()
            ),
            "counterfactual_unshielded_argmax": counterfactual_argmax,
            "counterfactual_unshielded_ties": (
                snapshot["counterfactual_ties"]
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
                .astype(np.bool_, copy=True)
            ),
            "counterfactual_argmax_tie_rule": COUNTERFACTUAL_ARGMAX_TIE_RULE,
            "would_intervene_argmax": bool(
                shield_active and counterfactual_argmax == int(RLAction.EXECUTE)
            ),
            "policy_rng_before_sha256": policy_rng_before,
            "policy_rng_after_sha256": policy_rng_after,
            "policy_rng_draw_index": len(transitions),
            "tokens_before": tokens_before,
            "tokens_after": int(env.tokens_used),
            "latency_before_s": latency_before,
            "latency_after_s": float(env.latency_used_s),
            "incident_index": int(trajectory_row["incident_index"]),
            "step_index": int(trajectory_row["step_index"]),
            "tool_failed": bool(trajectory_row["tool_failed"]),
            "prior_review_pass": review_state == 1,
            "review_before_execute": (
                review_state == 1 if selected == int(RLAction.EXECUTE) else None
            ),
            "execute_without_prior_review": bool(
                selected == int(RLAction.EXECUTE) and review_state != 1
            ),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "termination_reason": termination_reason,
            "done": bool(terminated),
            "environment_info": dict(step_info),
        }
        transitions.append(transition)
        total_return += reward
        observation = next_observation
    wrong_executions = sum(int(row["wrong_execute"]) for row in transitions)
    shield_active_decisions = sum(int(row["shield_intervened"]) for row in transitions)
    would_intervene = sum(int(row["would_intervene_argmax"]) for row in transitions)
    base_mass = sum(
        float(row["base_execute_probability"]) * int(row["shield_intervened"])
        for row in transitions
    ) / len(transitions)
    return transitions, {
        **env.info(),
        "return": total_return,
        "unsafe_episode": bool(wrong_executions),
        "wrong_executions": wrong_executions,
        "shield_active_decisions": shield_active_decisions,
        "counterfactual_argmax_interventions": would_intervene,
        "shielded_execute_probability_mass_per_decision": base_mass,
        "episode_cost_unsafe": sum(row["unsafe_cost"] for row in transitions),
        "episode_cost_wrong": sum(row["wrong_cost"] for row in transitions),
    }


def shield_aware_batch(
    episodes: Sequence[Sequence[Mapping[str, Any]]],
    base_batch: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Bind a frozen A22/A23 PPO batch to its base and effective masks."""

    if set(base_batch) != _BASE_BATCH_FIELDS:
        raise ValueError("A25 base PPO batch fields do not match frozen schema")
    flat = [transition for episode in episodes for transition in episode]
    if not flat or len(flat) != len(np.asarray(base_batch["action"])):
        raise ValueError("A25 episode-to-batch cardinality mismatch")
    base_masks: list[np.ndarray] = []
    shield_active_values: list[bool] = []
    episode_indices: list[int] = []
    transition_indices: list[int] = []
    rollout_logits: list[np.ndarray] = []
    rollout_base_probabilities: list[float] = []
    cursor = 0
    for episode_index, episode in enumerate(episodes):
        if not episode:
            raise ValueError("empty A25 rollout episode")
        for transition_index, transition in enumerate(episode):
            try:
                observation = transition["observation"]
                base_mask = transition["base_mask"]
                mask = transition["mask"]
                action = transition["action"]
                recorded_active = transition["shield_intervened"]
                recorded_logits = transition["pre_shield_logits"]
                recorded_base_probability = transition["base_execute_probability"]
            except KeyError as exc:
                raise ValueError("missing A25 shield transition field") from exc
            if (
                not isinstance(observation, np.ndarray)
                or observation.shape != (MultiTownLongHorizonEnv.observation_size,)
                or observation.dtype != np.float32
                or not np.isfinite(observation).all()
                or not isinstance(base_mask, np.ndarray)
                or base_mask.shape != (ACTION_COUNT,)
                or base_mask.dtype != np.bool_
                or not bool(base_mask.any())
                or not isinstance(mask, np.ndarray)
                or mask.shape != (ACTION_COUNT,)
                or mask.dtype != np.bool_
                or type(action) is not int
                or not 0 <= action < ACTION_COUNT
                or not bool(mask[action])
                or type(recorded_active) is not bool
                or not isinstance(recorded_logits, np.ndarray)
                or recorded_logits.shape != (ACTION_COUNT,)
                or recorded_logits.dtype != np.float32
                or not np.isfinite(recorded_logits).all()
                or not _finite_float(recorded_base_probability)
                or not 0.0 <= float(recorded_base_probability) <= 1.0
            ):
                raise ValueError("invalid A25 shield transition field")
            expected_mask, expected_active = effective_action_mask(
                observation, base_mask, shield_enabled=True
            )
            if (
                not np.array_equal(mask, expected_mask)
                or recorded_active != expected_active
                or not np.array_equal(
                    observation, np.asarray(base_batch["observation"])[cursor]
                )
                or not np.array_equal(mask, np.asarray(base_batch["mask"])[cursor])
                or int(np.asarray(base_batch["action"])[cursor]) != action
            ):
                raise ValueError("A25 base/effective mask or batch binding mismatch")
            base_masks.append(base_mask)
            shield_active_values.append(expected_active)
            episode_indices.append(episode_index)
            transition_indices.append(transition_index)
            rollout_logits.append(recorded_logits)
            rollout_base_probabilities.append(float(recorded_base_probability))
            cursor += 1
    result = {key: np.asarray(value).copy() for key, value in base_batch.items()}
    result.update(
        {
            "base_mask": np.stack(base_masks).astype(np.bool_, copy=False),
            "shield_active": np.asarray(shield_active_values, dtype=np.bool_),
            "episode_index": np.asarray(episode_indices, dtype=np.int64),
            "transition_index": np.asarray(transition_indices, dtype=np.int64),
            "rollout_pre_shield_logits": np.stack(rollout_logits).astype(
                np.float32, copy=False
            ),
            "rollout_base_execute_probability": np.asarray(
                rollout_base_probabilities, dtype=np.float32
            ),
        }
    )
    for value in result.values():
        value.setflags(write=False)
    return result


def _validate_shield_batch(
    model: nn.Module,
    batch: Mapping[str, np.ndarray],
    device: torch.device,
    *,
    require_on_policy_snapshot: bool,
) -> dict[str, torch.Tensor]:
    if set(batch) != (_BASE_BATCH_FIELDS | _A25_BATCH_FIELDS):
        raise ValueError("invalid A25 shield-aware batch fields")
    observation_array = np.asarray(batch["observation"])
    mask_array = np.asarray(batch["mask"])
    base_array = np.asarray(batch["base_mask"])
    action_array = np.asarray(batch["action"])
    active_array = np.asarray(batch["shield_active"])
    episode_index_array = np.asarray(batch["episode_index"])
    transition_index_array = np.asarray(batch["transition_index"])
    rollout_logits_array = np.asarray(batch["rollout_pre_shield_logits"])
    rollout_base_probability_array = np.asarray(
        batch["rollout_base_execute_probability"]
    )
    count = len(observation_array)
    if (
        count == 0
        or observation_array.shape != (count, MultiTownLongHorizonEnv.observation_size)
        or observation_array.dtype != np.float32
        or not np.isfinite(observation_array).all()
        or mask_array.shape != (count, ACTION_COUNT)
        or mask_array.dtype != np.bool_
        or base_array.shape != (count, ACTION_COUNT)
        or base_array.dtype != np.bool_
        or not bool(mask_array.any(axis=1).all())
        or not bool(base_array.any(axis=1).all())
        or action_array.shape != (count,)
        or not np.issubdtype(action_array.dtype, np.integer)
        or not bool(((action_array >= 0) & (action_array < ACTION_COUNT)).all())
        or not bool(mask_array[np.arange(count), action_array].all())
        or active_array.shape != (count,)
        or active_array.dtype != np.bool_
        or episode_index_array.shape != (count,)
        or episode_index_array.dtype != np.int64
        or transition_index_array.shape != (count,)
        or transition_index_array.dtype != np.int64
        or rollout_logits_array.shape != (count, ACTION_COUNT)
        or rollout_logits_array.dtype != np.float32
        or not np.isfinite(rollout_logits_array).all()
        or rollout_base_probability_array.shape != (count,)
        or rollout_base_probability_array.dtype != np.float32
        or not np.isfinite(rollout_base_probability_array).all()
        or not bool(
            (
                (rollout_base_probability_array >= 0.0)
                & (rollout_base_probability_array <= 1.0)
            ).all()
        )
    ):
        raise ValueError("invalid A25 shield-aware tensor schema")
    if episode_index_array[0] != 0 or transition_index_array[0] != 0:
        raise ValueError("invalid A25 episode/transition ordering")
    for index in range(count):
        expected_mask, expected_active = effective_action_mask(
            observation_array[index], base_array[index], shield_enabled=True
        )
        if (
            not np.array_equal(mask_array[index], expected_mask)
            or bool(active_array[index]) != expected_active
        ):
            raise ValueError("A25 batch does not bind to hard shield")
        if index:
            same_episode = episode_index_array[index] == episode_index_array[index - 1]
            next_episode = (
                episode_index_array[index] == episode_index_array[index - 1] + 1
            )
            if (
                same_episode
                and transition_index_array[index]
                != transition_index_array[index - 1] + 1
            ) or (
                next_episode and transition_index_array[index] != 0
            ) or not (same_episode or next_episode):
                raise ValueError("invalid A25 episode/transition ordering")
    for key in (
        "old_log_probability",
        "old_value",
        "return",
        "advantage",
        "reward_advantage_raw",
        "unsafe_advantage_raw",
        "wrong_advantage_raw",
    ):
        array = np.asarray(batch[key])
        if array.shape != (count,) or not np.isfinite(array).all():
            raise FloatingPointError(f"invalid A25 batch field: {key}")
        if key in {
            "advantage",
            "reward_advantage_raw",
            "unsafe_advantage_raw",
            "wrong_advantage_raw",
        } and array.dtype != np.float32:
            raise ValueError(f"invalid A25 batch dtype: {key}")
    tensors = {
        "observation": torch.tensor(
            observation_array, dtype=torch.float32, device=device
        ),
        "mask": torch.tensor(mask_array, dtype=torch.bool, device=device),
        "base_mask": torch.tensor(base_array, dtype=torch.bool, device=device),
        "shield_active": torch.tensor(active_array, dtype=torch.bool, device=device),
        "action": torch.tensor(action_array, dtype=torch.long, device=device),
        "old_log_probability": torch.tensor(
            batch["old_log_probability"], dtype=torch.float32, device=device
        ),
        "old_value": torch.tensor(
            batch["old_value"], dtype=torch.float32, device=device
        ),
        "return": torch.tensor(batch["return"], dtype=torch.float32, device=device),
        "advantage": torch.tensor(
            batch["advantage"], dtype=torch.float32, device=device
        ),
        "rollout_pre_shield_logits": torch.tensor(
            rollout_logits_array, dtype=torch.float32, device=device
        ),
        "rollout_base_execute_probability": torch.tensor(
            rollout_base_probability_array, dtype=torch.float32, device=device
        ),
    }
    with torch.no_grad():
        snapshot = _policy_snapshot(
            model,
            tensors["observation"],
            tensors["base_mask"],
            tensors["mask"],
        )
        effective_distribution = _masked_distribution(
            snapshot["logits"], tensors["mask"]
        )
        log_probability = effective_distribution.log_prob(tensors["action"])
    if require_on_policy_snapshot and (
        not torch.allclose(
            log_probability,
            tensors["old_log_probability"],
            rtol=1e-6,
            atol=1e-7,
        )
        or not torch.allclose(
            snapshot["values"], tensors["old_value"], rtol=1e-6, atol=1e-7
        )
        or not torch.allclose(
            snapshot["logits"],
            tensors["rollout_pre_shield_logits"],
            rtol=1e-6,
            atol=1e-7,
        )
        or not torch.allclose(
            snapshot["base_execute_probability"],
            tensors["rollout_base_execute_probability"],
            rtol=1e-6,
            atol=1e-7,
        )
    ):
        raise ValueError("A25 rollout does not bind to current on-policy model")
    return tensors


def shield_dependence_metrics(
    model: nn.Module,
    batch: Mapping[str, np.ndarray],
    device: torch.device,
    *,
    require_on_policy_snapshot: bool = False,
) -> dict[str, float | int]:
    """Measure shield exposure separately from current-policy dependence."""

    tensors = _validate_shield_batch(
        model,
        batch,
        device,
        require_on_policy_snapshot=require_on_policy_snapshot,
    )
    with torch.no_grad():
        snapshot = _policy_snapshot(
            model,
            tensors["observation"],
            tensors["base_mask"],
            tensors["mask"],
        )
    active = tensors["shield_active"]
    base_probability = snapshot["base_execute_probability"]
    would_intervene = active & (
        snapshot["counterfactual_argmax"] == int(RLAction.EXECUTE)
    )
    actions = tensors["action"]
    count = len(actions)
    active_count = int(active.sum().item())
    mass = active.float() * base_probability
    execute_without_review = int(
        ((actions == int(RLAction.EXECUTE)) & active).sum().item()
    )
    if execute_without_review:
        raise RuntimeError("hard shield allowed pre-review execute")
    return {
        "decisions": count,
        "shield_active_decisions": active_count,
        "shield_active_fraction": active_count / count,
        "base_execute_probability_mean_all_decisions": float(
            base_probability.mean().item()
        ),
        "shielded_execute_probability_mass_mean_all_decisions": float(
            mass.mean().item()
        ),
        "base_execute_probability_mean_shield_active": float(
            base_probability[active].mean().item() if active_count else 0.0
        ),
        "counterfactual_argmax_interventions": int(would_intervene.sum().item()),
        "counterfactual_argmax_intervention_fraction_all_decisions": float(
            would_intervene.float().mean().item()
        ),
        "counterfactual_argmax_intervention_fraction_shield_active": float(
            would_intervene.sum().item() / active_count if active_count else 0.0
        ),
        "executed_execute_actions": int(
            (actions == int(RLAction.EXECUTE)).sum().item()
        ),
        "execute_without_review": execute_without_review,
        "stop_actions": int((actions == int(RLAction.STOP)).sum().item()),
        "human_actions": int((actions == int(RLAction.HUMAN)).sum().item()),
    }


def _validate_update_schedule(
    config: PPOConfig, count: int, *, forbid_partial_minibatch: bool
) -> None:
    if (
        type(config.ppo_epochs) is not int
        or config.ppo_epochs <= 0
        or type(config.minibatch_size) is not int
        or config.minibatch_size <= 0
        or not _finite_float(config.clip_ratio)
        or not 0.0 <= float(config.clip_ratio) < 1.0
        or not _finite_float(config.value_coef)
        or float(config.value_coef) < 0.0
        or not _finite_float(config.entropy_coef)
        or float(config.entropy_coef) < 0.0
        or not _finite_float(config.max_grad_norm)
        or float(config.max_grad_norm) <= 0.0
    ):
        raise ValueError("invalid A25 PPO update schedule")
    if (
        forbid_partial_minibatch
        and count > config.minibatch_size
        and count % config.minibatch_size
    ):
        raise ValueError("A25 partial minibatch is forbidden")


def intervention_ppo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: Mapping[str, np.ndarray],
    config: PPOConfig,
    device: torch.device,
    generator: torch.Generator,
    *,
    objective: InterventionObjective,
    observer: NumericalGradientObserver | None = None,
) -> dict[str, Any]:
    """Run PPO plus pre-shield execute-mass regularization under a hard shield.

    ``observer`` is a trusted internal instrumentation seam that receives live
    graph tensors.  Its mutation guard covers enumerated training state as a
    defense in depth; it does not sandbox arbitrary Python callbacks.
    """

    objective = validate_objective(objective)
    beta = float(objective.beta)
    if observer is not None and not isinstance(observer, NumericalGradientObserver):
        raise TypeError("invalid A25 numerical gradient observer")
    if observer is not None and beta == 0.0:
        raise ValueError("A25 numerical observer cannot observe delegated beta-zero")
    if not isinstance(generator, torch.Generator):
        raise TypeError("invalid A25 PPO generator")
    action_candidate = batch.get("action")
    count = (
        int(action_candidate.shape[0])
        if isinstance(action_candidate, np.ndarray) and action_candidate.ndim == 1
        else 0
    )
    _validate_update_schedule(
        config,
        count,
        forbid_partial_minibatch=float(objective.beta) > 0.0,
    )
    _validate_model_optimizer(model, optimizer)
    tensors = _validate_shield_batch(
        model, batch, device, require_on_policy_snapshot=True
    )
    pre = shield_dependence_metrics(
        model, batch, device, require_on_policy_snapshot=True
    )
    count = len(tensors["action"])
    if beta == 0.0:
        writable_base_batch = {
            key: np.asarray(batch[key]).copy() for key in _BASE_BATCH_FIELDS
        }
        ppo_metrics = lagrangian_ppo_update(
            model, optimizer, writable_base_batch, config, device, generator
        )
        post = shield_dependence_metrics(model, batch, device)
        return {
            **ppo_metrics,
            "intervention_beta": 0.0,
            "intervention_loss": 0.0,
            "intervention_penalty": 0.0,
            "pre_update_shield_dependence": pre,
            "post_update_shield_dependence": post,
        }

    metrics: dict[str, list[float]] = {
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "approx_kl": [],
        "clip_fraction": [],
        "intervention_loss": [],
        "intervention_penalty": [],
    }
    metric_weights: list[int] = []
    observer_named_parameters = (
        tuple(model.named_parameters()) if observer is not None else ()
    )
    for epoch_index in range(config.ppo_epochs):
        order = torch.randperm(count, generator=generator, device=device)
        for minibatch_index, start in enumerate(
            range(0, count, config.minibatch_size)
        ):
            indices = order[start : start + config.minibatch_size]
            logits, values = model(tensors["observation"][indices])
            effective_distribution = _masked_distribution(
                logits, tensors["mask"][indices]
            )
            new_log_probability = effective_distribution.log_prob(
                tensors["action"][indices]
            )
            entropy = effective_distribution.entropy().mean()
            auxiliary = unscaled_intervention_auxiliary_loss(
                logits,
                tensors["base_mask"][indices],
                tensors["shield_active"][indices],
            )
            terms = intervention_minibatch_loss_terms(
                new_log_probability=new_log_probability,
                old_log_probability=tensors["old_log_probability"][indices],
                advantage=tensors["advantage"][indices],
                values=values,
                returns=tensors["return"][indices],
                entropy=entropy,
                auxiliary=auxiliary,
                clip_ratio=config.clip_ratio,
                value_coef=config.value_coef,
                entropy_coef=config.entropy_coef,
                beta=beta,
            )
            loss = terms.total_loss
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite A25 PPO loss")
            optimizer.zero_grad(set_to_none=True)
            observer_indices: tuple[int, ...] = ()
            observer_indices_sha256 = ""
            if observer is not None:
                observer_indices, observer_indices_sha256 = (
                    _numerical_observer_indices(indices)
                )
                _notify_numerical_gradient_observer(
                    observer,
                    NumericalGradientEvent(
                        schema_version=NUMERICAL_GRADIENT_EVENT_SCHEMA_VERSION,
                        stage="loss_terms",
                        epoch_index=epoch_index,
                        minibatch_index=minibatch_index,
                        indices=observer_indices,
                        indices_sha256=observer_indices_sha256,
                        named_parameters=observer_named_parameters,
                        loss_terms=terms,
                        gradient_summary=None,
                        clip_returned_norm=None,
                    ),
                    model=model,
                    optimizer=optimizer,
                    generator=generator,
                    named_parameters=observer_named_parameters,
                    device=device,
                )
            loss.backward()
            if observer is not None:
                _notify_numerical_gradient_observer(
                    observer,
                    NumericalGradientEvent(
                        schema_version=NUMERICAL_GRADIENT_EVENT_SCHEMA_VERSION,
                        stage="preclip",
                        epoch_index=epoch_index,
                        minibatch_index=minibatch_index,
                        indices=observer_indices,
                        indices_sha256=observer_indices_sha256,
                        named_parameters=observer_named_parameters,
                        loss_terms=None,
                        gradient_summary=canonical_named_gradient_summary(model),
                        clip_returned_norm=None,
                    ),
                    model=model,
                    optimizer=optimizer,
                    generator=generator,
                    named_parameters=observer_named_parameters,
                    device=device,
                )
            clip_returned_norm = nn.utils.clip_grad_norm_(
                model.parameters(), config.max_grad_norm
            )
            if observer is not None:
                clip_returned_norm_value = float(clip_returned_norm.item())
                if not math.isfinite(clip_returned_norm_value):
                    raise FloatingPointError("non-finite A25 returned gradient norm")
                _notify_numerical_gradient_observer(
                    observer,
                    NumericalGradientEvent(
                        schema_version=NUMERICAL_GRADIENT_EVENT_SCHEMA_VERSION,
                        stage="postclip",
                        epoch_index=epoch_index,
                        minibatch_index=minibatch_index,
                        indices=observer_indices,
                        indices_sha256=observer_indices_sha256,
                        named_parameters=observer_named_parameters,
                        loss_terms=None,
                        gradient_summary=canonical_named_gradient_summary(model),
                        clip_returned_norm=clip_returned_norm_value,
                    ),
                    model=model,
                    optimizer=optimizer,
                    generator=generator,
                    named_parameters=observer_named_parameters,
                    device=device,
                )
            optimizer.step()
            with torch.no_grad():
                metric_weights.append(len(indices))
                metrics["policy_loss"].append(float(terms.policy_loss.item()))
                metrics["value_loss"].append(float(terms.value_loss.item()))
                metrics["entropy"].append(float(terms.entropy.item()))
                metrics["approx_kl"].append(
                    float(((terms.ratio - 1.0) - terms.log_ratio).mean().item())
                )
                metrics["clip_fraction"].append(
                    float(
                        ((terms.ratio - 1.0).abs() > config.clip_ratio)
                        .float()
                        .mean()
                        .item()
                    )
                )
                metrics["intervention_loss"].append(
                    float(terms.auxiliary_loss.item())
                )
                metrics["intervention_penalty"].append(
                    float(terms.intervention_penalty.item())
                )
    _validate_model_optimizer(model, optimizer)
    post = shield_dependence_metrics(model, batch, device)
    return {
        **{
            key: float(np.average(values, weights=metric_weights))
            for key, values in metrics.items()
        },
        "intervention_beta": beta,
        "pre_update_shield_dependence": pre,
        "post_update_shield_dependence": post,
    }


def build_shield_dependence_ledger(
    model: nn.Module,
    episodes: Sequence[Sequence[Mapping[str, Any]]],
    device: torch.device,
    *,
    episode_ids: Sequence[str],
    fold: int,
    training_seed: int,
    update: int,
    checkpoint_sha256: str,
    post_update_model: nn.Module | None = None,
    post_update_checkpoint_sha256: str | None = None,
) -> list[dict[str, Any]]:
    """Create public-state telemetry bound to rollout and optional updated models.

    ``new_log_probability`` is null for a rollout-only ledger.  It is computed
    only when an explicit post-update model and its content hash are supplied;
    the rollout probability is never relabelled as a post-update value.
    """

    if (
        len(episodes) != len(episode_ids)
        or not episodes
        or len(set(episode_ids)) != len(episode_ids)
        or any(not isinstance(item, str) or not item for item in episode_ids)
        or type(fold) is not int
        or not 0 <= fold < 5
        or type(training_seed) is not int
        or training_seed < 0
        or type(update) is not int
        or update < 0
        or _SHA256_RE.fullmatch(checkpoint_sha256) is None
        or (post_update_model is None) != (post_update_checkpoint_sha256 is None)
    ):
        raise ValueError("invalid A25 ledger identity")
    if checkpoint_sha256 != model_parameter_sha256(model):  # type: ignore[arg-type]
        raise ValueError("A25 ledger checkpoint does not bind to rollout model")
    if post_update_model is not None and (
        not isinstance(post_update_checkpoint_sha256, str)
        or _SHA256_RE.fullmatch(post_update_checkpoint_sha256) is None
        or post_update_checkpoint_sha256
        != model_parameter_sha256(post_update_model)  # type: ignore[arg-type]
    ):
        raise ValueError("A25 ledger checkpoint does not bind to updated model")
    rows: list[dict[str, Any]] = []
    for episode_id, episode in zip(episode_ids, episodes, strict=True):
        if not episode:
            raise ValueError("empty A25 ledger episode")
        previous_tokens_after: int | None = None
        previous_latency_after: float | None = None
        previous_incident_index: int | None = None
        for transition_index, transition in enumerate(episode):
            try:
                observation = transition["observation"]
                base_mask = transition["base_mask"]
                mask = transition["mask"]
                action = transition["action"]
                sampled_action = transition["sampled_action"]
                executed_action = transition["executed_action"]
                recorded_logits = transition["pre_shield_logits"]
                recorded_ties = transition["counterfactual_unshielded_ties"]
            except KeyError as exc:
                raise ValueError("missing A25 ledger transition field") from exc
            latency_before = transition.get("latency_before_s")
            latency_after = transition.get("latency_after_s")
            if (
                not isinstance(observation, np.ndarray)
                or transition.get("episode_id") != episode_id
                or not isinstance(transition.get("split"), str)
                or not transition["split"]
                or observation.shape
                != (MultiTownLongHorizonEnv.observation_size,)
                or observation.dtype != np.float32
                or not np.isfinite(observation).all()
                or not isinstance(base_mask, np.ndarray)
                or base_mask.shape != (ACTION_COUNT,)
                or base_mask.dtype != np.bool_
                or not bool(base_mask.any())
                or not isinstance(mask, np.ndarray)
                or mask.shape != (ACTION_COUNT,)
                or mask.dtype != np.bool_
                or not bool(mask.any())
                or type(action) is not int
                or not 0 <= action < ACTION_COUNT
                or not bool(mask[action])
                or type(sampled_action) is not int
                or type(executed_action) is not int
                or sampled_action != action
                or executed_action != action
                or type(transition.get("wrong_execute")) is not bool
                or (
                    bool(transition.get("wrong_execute"))
                    and action != int(RLAction.EXECUTE)
                )
                or type(transition.get("invalid_action")) is not bool
                or type(transition.get("budget_violation")) is not bool
                or type(transition.get("shield_intervened")) is not bool
                or type(transition.get("would_intervene_argmax")) is not bool
                or type(transition.get("tool_failed")) is not bool
                or type(transition.get("prior_review_pass")) is not bool
                or type(transition.get("execute_without_prior_review")) is not bool
                or type(transition.get("terminated")) is not bool
                or type(transition.get("truncated")) is not bool
                or type(transition.get("done")) is not bool
                or type(transition.get("incident_index")) is not int
                or transition["incident_index"] < 0
                or type(transition.get("step_index")) is not int
                or transition["step_index"] != transition_index
                or type(transition.get("tokens_before")) is not int
                or type(transition.get("tokens_after")) is not int
                or transition["tokens_before"] < 0
                or transition["tokens_after"] < transition["tokens_before"]
                or not _finite_float(latency_before)
                or not _finite_float(latency_after)
                or float(latency_before) < 0.0
                or float(latency_after) < float(latency_before)
            ):
                raise ValueError("invalid A25 ledger transition schema")
            if (
                (transition_index == 0 and transition["tokens_before"] != 0)
                or (transition_index == 0 and float(latency_before) != 0.0)
                or (transition_index == 0 and transition["incident_index"] != 0)
                or (
                    transition_index < len(episode) - 1
                    and bool(transition["terminated"])
                )
                or (
                    transition_index == len(episode) - 1
                    and not bool(transition["terminated"])
                )
            ):
                raise ValueError("incomplete A25 physical episode ledger")
            if (
                (previous_tokens_after is not None and transition["tokens_before"] != previous_tokens_after)
                or (
                    previous_latency_after is not None
                    and not math.isclose(
                        float(latency_before), previous_latency_after,
                        rel_tol=0.0, abs_tol=1e-12,
                    )
                )
                or (
                    previous_incident_index is not None
                    and not previous_incident_index
                    <= transition["incident_index"]
                    <= previous_incident_index + 1
                )
            ):
                raise ValueError("invalid A25 ledger resource or incident continuity")
            expected_mask, expected_active = effective_action_mask(
                observation, base_mask, shield_enabled=True
            )
            review_state = public_review_state(observation)
            expected_review_before_execute = (
                review_state == 1 if action == int(RLAction.EXECUTE) else None
            )
            if not bool(transition["terminated"]):
                expected_termination_reason = "not-terminal"
            elif bool(transition["truncated"]):
                expected_termination_reason = "max-steps"
            elif bool(transition["budget_violation"]):
                expected_termination_reason = "budget-exhausted"
            elif action == int(RLAction.STOP):
                expected_termination_reason = "agent-stop"
            else:
                expected_termination_reason = "budget-exhausted"
            if (
                not np.array_equal(mask, expected_mask)
                or bool(transition["done"]) != bool(transition["terminated"])
                or (bool(transition["truncated"]) and not bool(transition["terminated"]))
                or transition.get("termination_reason")
                != expected_termination_reason
                or transition.get("review_before_execute")
                is not expected_review_before_execute
                or bool(transition["prior_review_pass"]) != (review_state == 1)
                or bool(transition["execute_without_prior_review"])
                != bool(action == int(RLAction.EXECUTE) and review_state != 1)
                or bool(transition["invalid_action"])
                or (bool(transition["budget_violation"]) and not bool(transition["terminated"]))
            ):
                raise ValueError("A25 ledger effective mask mismatch")
            obs_tensor = torch.from_numpy(observation).to(device).unsqueeze(0)
            base_tensor = torch.tensor(
                base_mask, dtype=torch.bool, device=device
            ).unsqueeze(0)
            mask_tensor = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(
                0
            )
            with torch.no_grad():
                snapshot = _policy_snapshot(model, obs_tensor, base_tensor, mask_tensor)
                effective_distribution = _masked_distribution(
                    snapshot["logits"], mask_tensor
                )
                action_tensor = torch.tensor([action], dtype=torch.long, device=device)
                log_probability = effective_distribution.log_prob(action_tensor)
                post_log_probability: float | None = None
                if post_update_model is not None:
                    post_snapshot = _policy_snapshot(
                        post_update_model, obs_tensor, base_tensor, mask_tensor
                    )
                    post_distribution = _masked_distribution(
                        post_snapshot["logits"], mask_tensor
                    )
                    post_log_probability = float(
                        post_distribution.log_prob(action_tensor).item()
                    )
            logits = snapshot["logits"].squeeze(0).detach().cpu().numpy()
            ties = (
                snapshot["counterfactual_ties"]
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
            )
            counterfactual_argmax = int(snapshot["counterfactual_argmax"].item())
            base_probability = float(snapshot["base_execute_probability"].item())
            effective_action_probability = float(
                snapshot["effective_probabilities"][0, action].item()
            )
            scalar_keys = (
                "old_log_probability",
                "old_value",
                "reward",
                "unsafe_cost",
                "wrong_cost",
                "base_execute_probability",
                "latency_before_s",
                "latency_after_s",
            )
            if (
                not isinstance(recorded_logits, np.ndarray)
                or recorded_logits.shape != (ACTION_COUNT,)
                or recorded_logits.dtype != np.float32
                or not np.allclose(recorded_logits, logits, rtol=1e-6, atol=1e-7)
                or not isinstance(recorded_ties, np.ndarray)
                or recorded_ties.shape != (ACTION_COUNT,)
                or recorded_ties.dtype != np.bool_
                or not np.array_equal(recorded_ties, ties)
                or any(not _finite_float(transition.get(key)) for key in scalar_keys)
                or float(transition["unsafe_cost"]) < 0.0
                or float(transition["wrong_cost"]) < 0.0
                or bool(transition["wrong_execute"])
                != (float(transition["wrong_cost"]) > 0.0)
                or not math.isclose(
                    float(transition["old_log_probability"]),
                    float(log_probability.item()),
                    rel_tol=1e-6,
                    abs_tol=1e-7,
                )
                or not math.isclose(
                    float(transition["old_value"]),
                    float(snapshot["values"].item()),
                    rel_tol=1e-6,
                    abs_tol=1e-7,
                )
                or not math.isclose(
                    float(transition["base_execute_probability"]),
                    base_probability,
                    rel_tol=1e-6,
                    abs_tol=1e-7,
                )
                or bool(transition["shield_intervened"]) != expected_active
                or type(transition.get("review_state")) is not int
                or int(transition["review_state"]) != review_state
                or type(transition.get("counterfactual_unshielded_argmax"))
                is not int
                or int(transition["counterfactual_unshielded_argmax"])
                != counterfactual_argmax
                or transition.get("counterfactual_argmax_tie_rule")
                != COUNTERFACTUAL_ARGMAX_TIE_RULE
                or bool(transition["would_intervene_argmax"])
                != bool(
                    expected_active and counterfactual_argmax == int(RLAction.EXECUTE)
                )
            ):
                raise ValueError("A25 ledger does not bind to current policy")
            rows.append(
                {
                    "schema_version": A25_LEDGER_SCHEMA_VERSION,
                    "episode_id": episode_id,
                    "split": str(transition["split"]),
                    "transition_index": transition_index,
                    "step_index": int(transition["step_index"]),
                    "fold": fold,
                    "training_seed": training_seed,
                    "update": update,
                    "checkpoint_sha256": checkpoint_sha256,
                    "post_update_checkpoint_sha256": (
                        post_update_checkpoint_sha256
                    ),
                    "observation": [float(item) for item in observation],
                    "review_state": REVIEW_STATE_NAMES[review_state],
                    "base_mask": [bool(item) for item in base_mask],
                    "effective_mask": [bool(item) for item in mask],
                    "pre_shield_logits": [float(item) for item in logits],
                    "base_execute_probability": base_probability,
                    "counterfactual_unshielded_argmax": ACTION_NAMES[
                        counterfactual_argmax
                    ],
                    "counterfactual_unshielded_ties": [
                        ACTION_NAMES[index]
                        for index, tied in enumerate(ties)
                        if bool(tied)
                    ],
                    "counterfactual_argmax_tie_rule": (
                        COUNTERFACTUAL_ARGMAX_TIE_RULE
                    ),
                    "shield_active": expected_active,
                    "would_intervene_argmax": bool(
                        expected_active
                        and counterfactual_argmax == int(RLAction.EXECUTE)
                    ),
                    "sampled_action": ACTION_NAMES[sampled_action],
                    "executed_action": ACTION_NAMES[executed_action],
                    "effective_action_probability": effective_action_probability,
                    "old_log_probability": float(log_probability.item()),
                    "new_log_probability": post_log_probability,
                    "new_log_probability_status": (
                        "computed-post-update"
                        if post_update_model is not None
                        else "not-computed-rollout-only"
                    ),
                    "reward": float(transition["reward"]),
                    "unsafe_cost": float(transition["unsafe_cost"]),
                    "wrong_cost": float(transition["wrong_cost"]),
                    "wrong_execute": bool(transition["wrong_execute"]),
                    "invalid_action": bool(transition["invalid_action"]),
                    "budget_violation": bool(transition["budget_violation"]),
                    "incident_index": int(transition["incident_index"]),
                    "tool_failed": bool(transition["tool_failed"]),
                    "prior_review_pass": bool(transition["prior_review_pass"]),
                    "review_before_execute": transition[
                        "review_before_execute"
                    ],
                    "execute_without_prior_review": bool(
                        transition["execute_without_prior_review"]
                    ),
                    "tokens_before": int(transition["tokens_before"]),
                    "tokens_after": int(transition["tokens_after"]),
                    "latency_before_s": float(transition["latency_before_s"]),
                    "latency_after_s": float(transition["latency_after_s"]),
                    "terminated": bool(transition["terminated"]),
                    "truncated": bool(transition["truncated"]),
                    "termination_reason": str(transition["termination_reason"]),
                    "done": bool(transition["done"]),
                    "stop_action": action == int(RLAction.STOP),
                    "human_action": action == int(RLAction.HUMAN),
                }
            )
            previous_tokens_after = int(transition["tokens_after"])
            previous_latency_after = float(transition["latency_after_s"])
            previous_incident_index = int(transition["incident_index"])
    return rows
