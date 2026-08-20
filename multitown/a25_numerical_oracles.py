"""Deterministic diagnostic-only numerical oracles for A25 Q0 primitives.

The production observations in this module call the A25 loss helpers directly.
The separate float64 implementation exists only as a mathematical reference for
the tiny G1 logit fixture.  These diagnostics do not read benchmark data, create
formal locks, qualify a mechanism, or support performance or safety claims.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from typing import Any

import numpy as np
import torch
from torch import nn

from .a22_constrained_ppo import SafetyThresholds
from .a23_cr_ppo import select_actor_mode
from .a25_shield_dependence import (
    A25_PRIMITIVES_VERSION,
    intervention_minibatch_loss_terms,
    unscaled_intervention_auxiliary_loss,
)
from .long_horizon_env import ACTION_COUNT, MultiTownLongHorizonEnv, RLAction
from .ppo_controller import ActorCritic, _masked_distribution

A25_NUMERICAL_ORACLES_VERSION = "multitown-a25-numerical-oracles-v1"
A25_NUMERICAL_ORACLE_RECEIPT_VERSION = "multitown-a25-numerical-oracle-receipt-v1"
ANALYTIC_FIXTURE_VERSION = "analytic-mixed-mask-v1"
SELECTOR_FIXTURE_VERSION = "selector-boundary-tie-v1"
SHARED_BACKBONE_FIXTURE_VERSION = "shared-backbone-v1"
GLOBAL_CLIP_FIXTURE_VERSION = "global-norm-clip-v1"
FLOAT64_FD_STEP = 1e-5
FLOAT64_FD_ATOL = 1e-8
FLOAT32_ATOL = 2e-6
FLOAT32_RTOL = 2e-5
ZERO_REFERENCE_CUTOFF = 1e-8
ROW_SUM_ATOL = 2e-6
CLIP_BETA = 1024.0
CLIP_MAX_NORM = 1e-4
_MODEL_SEED = 20260825
_CLIP_MODEL_SEED = 20260826


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _finite_float_list(values: np.ndarray | torch.Tensor) -> list[Any]:
    if torch.is_tensor(values):
        array = values.detach().cpu().numpy()
    else:
        array = np.asarray(values)
    if not np.isfinite(array).all():
        raise FloatingPointError("non-finite A25 numerical oracle array")
    return array.tolist()


def _mixed_mask_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    logits = np.asarray(
        [
            [0.10, -0.20, 0.30, -0.10, 0.20, 0.40, -0.30, 0.00],
            [-0.30, 0.25, 0.05, 0.40, -0.15, 0.20, 0.35, -0.05],
            [0.20, 0.15, -0.40, 0.30, -0.10, 0.45, -0.25, 0.05],
            [-0.10, 0.35, 0.15, -0.20, 0.25, 0.50, -0.30, 0.05],
        ],
        dtype=np.float64,
    )
    masks = np.asarray(
        [
            [1, 1, 1, 0, 1, 1, 1, 1],
            [1, 0, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1, 1, 0, 1],
            [1, 1, 1, 1, 1, 0, 1, 1],
        ],
        dtype=np.bool_,
    )
    # The last row deliberately forces active=True while EXECUTE is base-illegal.
    # This isolates the primitive's mask-zero property from the inactive-row case;
    # a physical shield batch would derive active=False for that row.
    active = np.asarray([True, True, False, True], dtype=np.bool_)
    labels = [
        "unknown-active",
        "fail-active",
        "pass-inactive",
        "execute-base-illegal-forced-active-primitive",
    ]
    return logits, masks, active, labels


def _masked_probabilities_f64(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    legal = logits[mask]
    if legal.size == 0 or not np.isfinite(legal).all():
        raise ValueError("invalid float64 masked-softmax row")
    shifted = legal - np.max(legal)
    weights = np.exp(shifted)
    probabilities = np.zeros_like(logits, dtype=np.float64)
    probabilities[mask] = weights / np.sum(weights, dtype=np.float64)
    return probabilities


def _analytic_auxiliary_f64(
    logits: np.ndarray, masks: np.ndarray, active: np.ndarray
) -> tuple[np.float64, np.ndarray]:
    """Independent float64 reference; never used as production observed."""

    count = len(logits)
    execute = int(RLAction.EXECUTE)
    loss = np.float64(0.0)
    gradient = np.zeros_like(logits, dtype=np.float64)
    for index in range(count):
        if not bool(active[index]) or not bool(masks[index, execute]):
            continue
        probability = _masked_probabilities_f64(logits[index], masks[index])
        execute_probability = probability[execute]
        loss += execute_probability / np.float64(count)
        gradient[index, masks[index]] = (
            -execute_probability * probability[masks[index]] / np.float64(count)
        )
        gradient[index, execute] += execute_probability / np.float64(count)
    return loss, gradient


def _auxiliary_value_f64(
    logits: np.ndarray, masks: np.ndarray, active: np.ndarray
) -> np.float64:
    """Independent scalar reference used only by central finite difference."""

    count = len(logits)
    execute = int(RLAction.EXECUTE)
    value = np.float64(0.0)
    for index in range(count):
        if bool(active[index]) and bool(masks[index, execute]):
            probability = _masked_probabilities_f64(logits[index], masks[index])
            value += probability[execute] / np.float64(count)
    return value


def _central_difference_f64(
    logits: np.ndarray, masks: np.ndarray, active: np.ndarray
) -> np.ndarray:
    result = np.zeros_like(logits, dtype=np.float64)
    for index in np.ndindex(logits.shape):
        positive = logits.copy()
        negative = logits.copy()
        positive[index] += FLOAT64_FD_STEP
        negative[index] -= FLOAT64_FD_STEP
        result[index] = (
            _auxiliary_value_f64(positive, masks, active)
            - _auxiliary_value_f64(negative, masks, active)
        ) / (2.0 * FLOAT64_FD_STEP)
    return result


def _float32_gate(observed: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    observed_f64 = np.asarray(observed, dtype=np.float64)
    reference_f64 = np.asarray(reference, dtype=np.float64)
    if (
        observed_f64.shape != reference_f64.shape
        or not np.isfinite(observed_f64).all()
        or not np.isfinite(reference_f64).all()
    ):
        raise ValueError("invalid A25 float32 comparison")
    delta = np.abs(observed_f64 - reference_f64)
    relative = np.where(
        np.abs(reference_f64) >= ZERO_REFERENCE_CUTOFF,
        FLOAT32_RTOL * np.abs(reference_f64),
        0.0,
    )
    allowed = FLOAT32_ATOL + relative
    return {
        "atol": FLOAT32_ATOL,
        "rtol": FLOAT32_RTOL,
        "relative_reference_cutoff": ZERO_REFERENCE_CUTOFF,
        "max_abs_error": float(delta.max(initial=0.0)),
        "max_allowed_error": float(allowed.max(initial=FLOAT32_ATOL)),
        "passed": bool(np.all(delta <= allowed)),
    }


def analytic_mixed_mask_oracle() -> dict[str, Any]:
    """Run G1 against an independent f64 analytic and FD reference."""

    logits_f64, masks, active, labels = _mixed_mask_fixture()
    analytic_loss, analytic_gradient = _analytic_auxiliary_f64(
        logits_f64, masks, active
    )
    finite_difference = _central_difference_f64(logits_f64, masks, active)
    fd_delta = np.abs(finite_difference - analytic_gradient)

    logits_f32 = torch.tensor(logits_f64, dtype=torch.float32, requires_grad=True)
    masks_f32 = torch.tensor(masks, dtype=torch.bool)
    active_f32 = torch.tensor(active, dtype=torch.bool)
    production = unscaled_intervention_auxiliary_loss(logits_f32, masks_f32, active_f32)
    observed_gradient = torch.autograd.grad(production.loss, logits_f32)[0]
    observed_array = observed_gradient.detach().cpu().numpy()
    analytic_f32 = analytic_gradient.astype(np.float32)
    production_gate = _float32_gate(observed_array, analytic_f32)

    inactive_index = 2
    illegal_index = 3
    inactive_exact = bool(
        torch.equal(
            observed_gradient[inactive_index],
            torch.zeros_like(observed_gradient[inactive_index]),
        )
    )
    base_illegal_exact = bool(
        torch.equal(
            observed_gradient[illegal_index],
            torch.zeros_like(observed_gradient[illegal_index]),
        )
        and production.base_execute_probability[illegal_index].item() == 0.0
    )
    active_row_sums = observed_gradient[:2].sum(dim=1).abs()
    row_sum_max = float(active_row_sums.max().item())
    gates = {
        "float64_finite_difference": bool(
            float(fd_delta.max(initial=0.0)) <= FLOAT64_FD_ATOL
        ),
        "float32_production_gradient": production_gate["passed"],
        "inactive_exact_zero": inactive_exact,
        "base_illegal_exact_zero": base_illegal_exact,
        "active_row_gradient_sum": row_sum_max <= ROW_SUM_ATOL,
    }
    return {
        "schema_version": "multitown-a25-g1-analytic-oracle-v1",
        "fixture_version": ANALYTIC_FIXTURE_VERSION,
        "production_helper": "unscaled_intervention_auxiliary_loss",
        "rows": labels,
        "logits_f64": _finite_float_list(logits_f64),
        "base_masks": masks.astype(np.uint8).tolist(),
        "active": active.tolist(),
        "analytic_loss_f64": float(analytic_loss),
        "production_loss_f32": float(production.loss.detach().item()),
        "base_execute_probability_f32": _finite_float_list(
            production.base_execute_probability
        ),
        "analytic_gradient_f64": _finite_float_list(analytic_gradient),
        "finite_difference_gradient_f64": _finite_float_list(finite_difference),
        "observed_gradient_f32": _finite_float_list(observed_gradient),
        "finite_difference": {
            "step": FLOAT64_FD_STEP,
            "atol": FLOAT64_FD_ATOL,
            "max_abs_error": float(fd_delta.max(initial=0.0)),
        },
        "production_gradient_gate": production_gate,
        "active_row_sum_max_abs": row_sum_max,
        "gates": gates,
        "passed": all(gates.values()),
    }


def selector_boundary_tie_oracle() -> dict[str, Any]:
    """Run G4 through the production A23 selector without consuming RNG."""

    common = {
        "unsafe_events": 36,
        "wrong_executions": 48,
        "episodes": 48,
    }
    boundary_thresholds = SafetyThresholds(
        unsafe=0.75,
        wrong_per_incident=0.25,
        mean_incidents=4.0,
    )
    tie_thresholds = SafetyThresholds(
        unsafe=0.375,
        wrong_per_incident=0.125,
        mean_incidents=4.0,
    )
    boundary = select_actor_mode(**common, thresholds=boundary_thresholds)
    tie = select_actor_mode(**common, thresholds=tie_thresholds)
    gates = {
        "boundary_selects_reward": boundary.mode == "reward",
        "boundary_unsafe_not_eligible": boundary.unsafe_eligible is False,
        "boundary_wrong_not_eligible": boundary.wrong_eligible is False,
        "tie_selects_unsafe": tie.mode == "unsafe",
        "tie_flag_exact": tie.unsafe_tie_break_used is True,
        "tie_normalized_violations_exact": (
            tie.unsafe_normalized_violation == 1.0
            and tie.wrong_normalized_violation == 1.0
        ),
    }
    return {
        "schema_version": "multitown-a25-g4-selector-oracle-v1",
        "fixture_version": SELECTOR_FIXTURE_VERSION,
        "production_helper": "select_actor_mode",
        "counts": common,
        "observed_costs": {
            "unsafe": 0.75,
            "wrong_per_incident": 0.25,
        },
        "boundary": {
            "thresholds": asdict(boundary_thresholds),
            "decision": asdict(boundary),
        },
        "tie": {
            "thresholds": asdict(tie_thresholds),
            "decision": asdict(tie),
            "tie_rule": "unsafe-on-exact-equal-normalized-violation",
        },
        "gates": gates,
        "passed": all(gates.values()),
    }


def _diagnostic_observations(count: int) -> torch.Tensor:
    if type(count) is not int or count <= 0:
        raise ValueError("invalid A25 diagnostic observation count")
    observation = torch.zeros(
        (count, MultiTownLongHorizonEnv.observation_size), dtype=torch.float32
    )
    for index in range(count):
        observation[index, index % 8] = 0.25 + 0.05 * index
        observation[index, 8 + index % 8] = -0.20 + 0.03 * index
        observation[index, 20 + index % 10] = 0.10 * ((index % 3) - 1)
        observation[index, 33 + index % 3] = 1.0
        observation[index, 36 + index % 8] = 1.0
    return observation


def _seeded_actor_critic(seed: int) -> ActorCritic:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return ActorCritic(
            MultiTownLongHorizonEnv.observation_size, 8, ACTION_COUNT
        ).cpu()


def _gradient_group_summary(
    named_parameters: tuple[tuple[str, nn.Parameter], ...],
    gradients: tuple[torch.Tensor | None, ...],
) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    parameters: list[dict[str, Any]] = []
    for (name, parameter), gradient in zip(named_parameters, gradients, strict=True):
        group = name.partition(".")[0]
        state = groups.setdefault(
            group,
            {"tensor_count": 0, "none_count": 0, "squared_norm": 0.0},
        )
        if gradient is None:
            state["none_count"] += 1
            parameters.append({"name": name, "state": "none", "l2_norm": None})
            continue
        if gradient.shape != parameter.shape or not bool(
            torch.isfinite(gradient).all()
        ):
            raise FloatingPointError("invalid A25 diagnostic named gradient")
        squared_norm = float(gradient.detach().double().square().sum().item())
        state["tensor_count"] += 1
        state["squared_norm"] += squared_norm
        parameters.append(
            {
                "name": name,
                "state": "tensor",
                "l2_norm": math.sqrt(squared_norm),
            }
        )
    group_rows = {
        name: {
            "tensor_count": int(state["tensor_count"]),
            "none_count": int(state["none_count"]),
            "l2_norm": math.sqrt(float(state["squared_norm"])),
        }
        for name, state in groups.items()
    }
    return {"parameters": parameters, "groups": group_rows}


def shared_backbone_oracle() -> dict[str, Any]:
    """Run G5 on the production ActorCritic and combined loss helper."""

    model = _seeded_actor_critic(_MODEL_SEED)
    observations = _diagnostic_observations(4)
    base_masks = torch.tensor(
        [
            [1, 1, 1, 0, 1, 1, 1, 1],
            [1, 0, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1, 1, 0, 1],
            [1, 1, 1, 1, 1, 0, 1, 1],
        ],
        dtype=torch.bool,
    )
    active = torch.tensor([True, True, False, False], dtype=torch.bool)
    effective_masks = base_masks.clone()
    effective_masks[active, int(RLAction.EXECUTE)] = False
    actions = torch.tensor(
        [
            int(RLAction.REVIEW),
            int(RLAction.HUMAN),
            int(RLAction.EXECUTE),
            int(RLAction.OBSERVE),
        ],
        dtype=torch.long,
    )
    logits, values = model(observations)
    effective_distribution = _masked_distribution(logits, effective_masks)
    new_log_probability = effective_distribution.log_prob(actions)
    auxiliary = unscaled_intervention_auxiliary_loss(logits, base_masks, active)
    terms = intervention_minibatch_loss_terms(
        new_log_probability=new_log_probability,
        old_log_probability=new_log_probability.detach(),
        advantage=torch.zeros(4, dtype=torch.float32),
        values=values,
        returns=values.detach(),
        entropy=effective_distribution.entropy().mean(),
        auxiliary=auxiliary,
        clip_ratio=0.2,
        value_coef=0.0,
        entropy_coef=0.0,
        beta=5.0,
    )
    named_parameters = tuple(model.named_parameters())
    gradients = torch.autograd.grad(
        terms.auxiliary_loss,
        tuple(parameter for _, parameter in named_parameters),
        allow_unused=True,
        materialize_grads=False,
    )
    summary = _gradient_group_summary(named_parameters, gradients)
    groups = summary["groups"]
    critic_names = [
        row["name"]
        for row in summary["parameters"]
        if row["name"].startswith("critic.")
    ]
    critic_none_names = [
        row["name"]
        for row in summary["parameters"]
        if row["name"].startswith("critic.") and row["state"] == "none"
    ]
    gates = {
        "production_combined_helper_called": bool(
            torch.isfinite(terms.total_loss).item()
        ),
        "actor_gradient_nonzero": groups["actor"]["l2_norm"] > 0.0,
        "backbone_gradient_nonzero": groups["backbone"]["l2_norm"] > 0.0,
        "critic_aux_gradient_exact_none": (
            bool(critic_names) and critic_none_names == critic_names
        ),
    }
    return {
        "schema_version": "multitown-a25-g5-shared-backbone-oracle-v1",
        "fixture_version": SHARED_BACKBONE_FIXTURE_VERSION,
        "model_seed": _MODEL_SEED,
        "production_helpers": [
            "unscaled_intervention_auxiliary_loss",
            "intervention_minibatch_loss_terms",
        ],
        "auxiliary_loss_f32": float(terms.auxiliary_loss.detach().item()),
        "total_loss_f32": float(terms.total_loss.detach().item()),
        "gradient_summary": summary,
        "critic_parameter_names": critic_names,
        "critic_none_names": critic_none_names,
        "gates": gates,
        "passed": all(gates.values()),
    }


def _flatten_model_gradients(model: nn.Module) -> tuple[torch.Tensor, int]:
    values: list[torch.Tensor] = []
    none_count = 0
    for parameter in model.parameters():
        if parameter.grad is None:
            none_count += 1
            values.append(torch.zeros_like(parameter).reshape(-1))
        else:
            if not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError("non-finite A25 clipping gradient")
            values.append(parameter.grad.detach().clone().reshape(-1))
    if not values:
        raise ValueError("empty A25 clipping model")
    return torch.cat(values), none_count


def global_norm_clip_oracle() -> dict[str, Any]:
    """Run G6 through the production combined loss and global clip primitive."""

    model = _seeded_actor_critic(_CLIP_MODEL_SEED)
    with torch.no_grad():
        model.critic.weight.zero_()
        model.critic.bias.zero_()
    count = 8
    observations = _diagnostic_observations(count)
    base_masks = torch.ones((count, ACTION_COUNT), dtype=torch.bool)
    active = torch.ones(count, dtype=torch.bool)
    effective_masks = base_masks.clone()
    effective_masks[:, int(RLAction.EXECUTE)] = False
    actions = torch.full((count,), int(RLAction.REVIEW), dtype=torch.long)
    logits, values = model(observations)
    distribution = _masked_distribution(logits, effective_masks)
    new_log_probability = distribution.log_prob(actions)
    auxiliary = unscaled_intervention_auxiliary_loss(logits, base_masks, active)
    terms = intervention_minibatch_loss_terms(
        new_log_probability=new_log_probability,
        old_log_probability=new_log_probability.detach(),
        advantage=torch.zeros(count, dtype=torch.float32),
        values=values,
        returns=torch.zeros(count, dtype=torch.float32),
        entropy=distribution.entropy().mean(),
        auxiliary=auxiliary,
        clip_ratio=0.2,
        value_coef=0.0,
        entropy_coef=0.0,
        beta=CLIP_BETA,
    )
    model.zero_grad(set_to_none=True)
    terms.total_loss.backward()
    preclip, pre_none_count = _flatten_model_gradients(model)
    preclip_norm = float(torch.linalg.vector_norm(preclip.double()).item())
    clip_returned = nn.utils.clip_grad_norm_(model.parameters(), CLIP_MAX_NORM)
    postclip, post_none_count = _flatten_model_gradients(model)
    postclip_norm = float(torch.linalg.vector_norm(postclip.double()).item())
    denominator = preclip_norm * postclip_norm
    cosine = (
        float(torch.dot(preclip.double(), postclip.double()).item() / denominator)
        if denominator > 0.0
        else None
    )
    clip_returned_value = float(clip_returned.detach().cpu().item())
    returned_gate = _float32_gate(
        np.asarray([clip_returned_value], dtype=np.float32),
        np.asarray([preclip_norm], dtype=np.float32),
    )
    gates = {
        "base_loss_exact_zero": float(terms.base_total_loss.detach().item()) == 0.0,
        "preclip_exceeds_100x": preclip_norm > 100.0 * CLIP_MAX_NORM,
        "returned_norm_matches_preclip": returned_gate["passed"],
        "postclip_norm_bounded": (
            postclip_norm <= CLIP_MAX_NORM * (1.0 + FLOAT32_ATOL) + 1e-8
        ),
        "direction_preserved": (cosine is not None and cosine >= 1.0 - FLOAT32_ATOL),
        "gradient_none_pattern_preserved": pre_none_count == post_none_count,
    }
    return {
        "schema_version": "multitown-a25-g6-global-clip-oracle-v1",
        "fixture_version": GLOBAL_CLIP_FIXTURE_VERSION,
        "model_seed": _CLIP_MODEL_SEED,
        "production_helpers": [
            "unscaled_intervention_auxiliary_loss",
            "intervention_minibatch_loss_terms",
            "torch.nn.utils.clip_grad_norm_",
        ],
        "beta": CLIP_BETA,
        "max_grad_norm": CLIP_MAX_NORM,
        "base_total_loss_f32": float(terms.base_total_loss.detach().item()),
        "auxiliary_loss_f32": float(terms.auxiliary_loss.detach().item()),
        "total_loss_f32": float(terms.total_loss.detach().item()),
        "preclip_norm": preclip_norm,
        "clip_returned_total_norm": clip_returned_value,
        "postclip_norm": postclip_norm,
        "preclip_to_max_norm_ratio": preclip_norm / CLIP_MAX_NORM,
        "preclip_postclip_cosine": cosine,
        "preclip_none_count": pre_none_count,
        "postclip_none_count": post_none_count,
        "returned_norm_gate": returned_gate,
        "gates": gates,
        "passed": all(gates.values()),
    }


def build_numerical_oracle_receipt() -> dict[str, Any]:
    """Return a canonical in-memory diagnostic receipt, or fail closed."""

    diagnostics = {
        "G1": analytic_mixed_mask_oracle(),
        "G4": selector_boundary_tie_oracle(),
        "G5": shared_backbone_oracle(),
        "G6": global_norm_clip_oracle(),
    }
    gates = {
        "all_expected_diagnostics_present": set(diagnostics)
        == {"G1", "G4", "G5", "G6"},
        "all_diagnostics_passed": all(
            result["passed"] for result in diagnostics.values()
        ),
        "zero_outer_rows_read": True,
        "no_formal_lock_created": True,
    }
    if not all(gates.values()):
        raise RuntimeError("A25 numerical diagnostic oracle gate failed")
    core = {
        "schema_version": A25_NUMERICAL_ORACLE_RECEIPT_VERSION,
        "implementation_version": A25_NUMERICAL_ORACLES_VERSION,
        "production_primitives_version": A25_PRIMITIVES_VERSION,
        "scope": "deterministic-in-memory-diagnostic-only-no-outer-no-formal",
        "runtime": {
            "device": "cpu",
            "dtype": "torch.float32",
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
        },
        "diagnostics": diagnostics,
        "gates": gates,
        "claim_boundary": {
            "diagnostic_primitives_passed": True,
            "qualification_evidence": False,
            "full_numerical_q0_qualified": False,
            "q1_mechanism_qualified": False,
            "formal_authorized": False,
            "performance_claim_supported": False,
            "safety_claim_supported": False,
            "outer_rows_read": 0,
            "formal_lock_created": False,
        },
    }
    return {
        **core,
        "receipt_id": _canonical_sha256(core),
        "status": "DIAGNOSTIC_PASSED",
    }


def validate_numerical_oracle_receipt(receipt: dict[str, Any]) -> None:
    """Recompute all diagnostics and reject any non-canonical or stale receipt."""

    if type(receipt) is not dict:
        raise TypeError("A25 numerical oracle receipt must be an object")
    expected = build_numerical_oracle_receipt()
    if _canonical_bytes(receipt) != _canonical_bytes(expected):
        raise ValueError("A25 numerical oracle receipt mismatch")
