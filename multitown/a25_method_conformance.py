"""Pure numerical method oracles used before A25 training qualification.

These fixtures test selector and loss/update-direction semantics against pinned
official source profiles.  They are not paper reproductions, training runs, or
performance evidence.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .a22_constrained_ppo import SafetyThresholds
from .a23_cr_ppo import select_actor_mode

A25_METHOD_CONFORMANCE_VERSION = "multitown-a25-method-conformance-v3"
A25_METHOD_RECEIPT_VERSION = "multitown-a25-method-conformance-receipt-v3"
ORACLE_DTYPE = torch.float64
OBSERVED_DTYPE = torch.float32
ORACLE_ATOL = 1e-12
ORACLE_RTOL = 1e-10
OBSERVED_ATOL = 1e-7
OBSERVED_RTOL = 1e-6
RELATIVE_REFERENCE_CUTOFF = 1e-8

OFFICIAL_SOURCE_PROFILES = {
    "oncrpo": {
        "repository": "https://github.com/PKU-Alignment/omnisafe",
        "commit": "15603dd7a654a991d0a4648216b69d60b81a6366",
        "path": "omnisafe/algorithms/on_policy/primal/crpo.py",
        "sha256": "ede3dcde9c8d9bef8ef0691d58adc83ada9975e285fd355a15415a60bb7f9c20",
    },
    "p3o": {
        "repository": "https://github.com/PKU-Alignment/omnisafe",
        "commit": "15603dd7a654a991d0a4648216b69d60b81a6366",
        "path": "omnisafe/algorithms/on_policy/penalty_function/p3o.py",
        "sha256": "9d8ca8c6ccfe67c28d949037dc0d9d3169b2ba7c92703a729db27303a99383fb",
        "ppo_path": "omnisafe/algorithms/on_policy/base/ppo.py",
        "ppo_sha256": "d5d26896b90d685ceca3c835dcdfd6d7c7529a19551c3db2030e8470490698f3",
    },
}

CRPO_CONTRACT = {
    "actions": ["safe", "unsafe", "wrong"],
    "counts": {"unsafe_events": 1, "wrong_executions": 1, "episodes": 3},
    "mean_incidents": 1.0,
    "oncrpo_distance": 0.0,
    "old_logits": [0.0, 0.0, 0.0],
    "advantages": {
        "reward": [-1.0, 1.0, 0.0],
        "unsafe": [-1.0 / 3.0, 2.0 / 3.0, -1.0 / 3.0],
        "wrong": [-1.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0],
    },
    "expected_gradients": {
        "reward": [1.0 / 3.0, -1.0 / 3.0, 0.0],
        "unsafe": [-1.0 / 9.0, 2.0 / 9.0, -1.0 / 9.0],
        "wrong": [-1.0 / 9.0, -1.0 / 9.0, 2.0 / 9.0],
    },
    "profiles": {
        "C-R": {
            "unsafe_limit": 0.5,
            "wrong_limit": 0.5,
            "expected_mode": "reward",
            "expected_tie": False,
        },
        "C-U": {
            "unsafe_limit": 0.25,
            "wrong_limit": 0.5,
            "expected_mode": "unsafe",
            "expected_tie": False,
        },
        "C-W": {
            "unsafe_limit": 0.5,
            "wrong_limit": 0.25,
            "expected_mode": "wrong",
            "expected_tie": False,
        },
        "C-T": {
            "unsafe_limit": 0.25,
            "wrong_limit": 0.25,
            "expected_mode": "unsafe",
            "expected_tie": True,
        },
        "C-B": {
            "unsafe_limit": 1.0 / 3.0,
            "wrong_limit": 1.0 / 3.0,
            "expected_mode": "reward",
            "expected_tie": False,
        },
    },
    "clip_ratio": 0.2,
    "update_rule": "multitown_clipped_ppo_not_official_oncrpo_trpo",
}

P3O_CONTRACT = {
    "actions": ["safe", "risky"],
    "old_logits": [0.0, 0.0],
    "reward_advantage": [-0.5, 0.5],
    "cost_advantage": [-0.5, 0.5],
    "mean_episode_cost": 0.5,
    "clip_ratio": 0.2,
    "entropy_coef": 0.0,
    "single_cost_only": True,
    "profiles": {
        "P-I": {
            "policy": [0.5, 0.5],
            "cost_limit": 0.75,
            "kappa": 2.0,
            "expected_components": {
                "reward_loss": 0.0,
                "cost_surrogate": 0.0,
                "residual": -0.25,
                "cost_penalty": 0.0,
                "total_loss": 0.0,
            },
            "expected_gradient": [0.25, -0.25],
        },
        "P-A": {
            "policy": [0.5, 0.5],
            "cost_limit": 0.25,
            "kappa": 2.0,
            "expected_components": {
                "reward_loss": 0.0,
                "cost_surrogate": 0.0,
                "residual": 0.25,
                "cost_penalty": 0.5,
                "total_loss": 0.5,
            },
            "expected_gradient": [-0.25, 0.25],
        },
        "P-B-": {
            "policy": [0.5, 0.5],
            "cost_limit": 0.5000001,
            "kappa": 2.0,
            "expected_components": {
                "reward_loss": 0.0,
                "cost_surrogate": 0.0,
                "residual": -0.0000001,
                "cost_penalty": 0.0,
                "total_loss": 0.0,
            },
            "expected_gradient": [0.25, -0.25],
        },
        "P-B": {
            "policy": [0.5, 0.5],
            "cost_limit": 0.5,
            "kappa": 2.0,
            "expected_components": {
                "reward_loss": 0.0,
                "cost_surrogate": 0.0,
                "residual": 0.0,
                "cost_penalty": 0.0,
                "total_loss": 0.0,
            },
            "expected_gradient": [0.25, -0.25],
        },
        "P-B+": {
            "policy": [0.5, 0.5],
            "cost_limit": 0.4999999,
            "kappa": 2.0,
            "expected_components": {
                "reward_loss": 0.0,
                "cost_surrogate": 0.0,
                "residual": 0.0000001,
                "cost_penalty": 0.0000002,
                "total_loss": 0.0000002,
            },
            "expected_gradient": [-0.25, 0.25],
        },
        "P-C": {
            "policy": [0.5, 0.5],
            "cost_limit": 0.25,
            "kappa": 1.0,
            "expected_components": {
                "reward_loss": 0.0,
                "cost_surrogate": 0.0,
                "residual": 0.25,
                "cost_penalty": 0.25,
                "total_loss": 0.25,
            },
            "expected_gradient": [0.0, 0.0],
        },
        "P-X": {
            "policy": [0.35, 0.65],
            "cost_limit": 0.5,
            "kappa": 2.0,
            "expected_components": {
                "reward_loss": -0.1,
                "cost_surrogate": 0.15,
                "residual": 0.15,
                "cost_penalty": 0.3,
                "total_loss": 0.2,
            },
            "expected_gradient": [-0.455, 0.455],
        },
    },
}

METHOD_CONTRACT = {
    "schema_version": "multitown-a25-method-diagnostic-contract-v1",
    "crpo": CRPO_CONTRACT,
    "p3o": P3O_CONTRACT,
    "tolerances": {
        "oracle_atol": ORACLE_ATOL,
        "oracle_rtol": ORACLE_RTOL,
        "observed_atol": OBSERVED_ATOL,
        "observed_rtol": OBSERVED_RTOL,
        "relative_reference_cutoff": RELATIVE_REFERENCE_CUTOFF,
    },
}


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


def _finite_vector(
    values: Any, *, length: int, label: str, dtype: torch.dtype
) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=dtype, device="cpu")
    if tensor.shape != (length,) or not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"invalid A25 conformance vector: {label}")
    return tensor


def _ppo_reward_loss(
    logits: torch.Tensor,
    old_logits: torch.Tensor,
    advantage: torch.Tensor,
    *,
    clip_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        logits.ndim != 1
        or logits.shape != old_logits.shape
        or logits.shape != advantage.shape
        or logits.dtype not in {ORACLE_DTYPE, OBSERVED_DTYPE}
        or old_logits.dtype != logits.dtype
        or advantage.dtype != logits.dtype
        or not bool(torch.isfinite(logits).all())
        or not bool(torch.isfinite(old_logits).all())
        or not bool(torch.isfinite(advantage).all())
        or type(clip_ratio) is not float
        or not math.isfinite(clip_ratio)
        or not 0.0 < clip_ratio < 1.0
    ):
        raise ValueError("invalid A25 PPO conformance input")
    probability = torch.softmax(logits, dim=0)
    old_probability = torch.softmax(old_logits, dim=0)
    ratio = probability / old_probability
    unclipped = ratio * advantage
    clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantage
    return -torch.minimum(unclipped, clipped).mean(), ratio


def _gradient(loss: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    gradient = torch.autograd.grad(loss, logits, create_graph=False)[0]
    if gradient.shape != logits.shape or not bool(torch.isfinite(gradient).all()):
        raise FloatingPointError("invalid A25 conformance gradient")
    return gradient.detach()


def _tensor_list(values: torch.Tensor) -> list[float]:
    return [float(item) for item in values.detach().cpu().tolist()]


def _gradient_gate(
    observed: torch.Tensor,
    expected: list[float],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    observed_f64 = observed.detach().to(dtype=torch.float64, device="cpu")
    reference = torch.tensor(expected, dtype=torch.float64, device="cpu")
    delta = (observed_f64 - reference).abs()
    allowed = torch.full_like(reference, float(atol))
    relative = reference.abs() >= RELATIVE_REFERENCE_CUTOFF
    allowed[relative] += float(rtol) * reference[relative].abs()
    max_abs = float(delta.max().item())
    observed_norm = float(torch.linalg.vector_norm(observed_f64).item())
    reference_norm = float(torch.linalg.vector_norm(reference).item())
    if observed_norm == 0.0 or reference_norm == 0.0:
        cosine = None
    else:
        cosine = float(
            torch.nn.functional.cosine_similarity(
                observed_f64.unsqueeze(0), reference.unsqueeze(0)
            ).item()
        )
    return {
        "observed": _tensor_list(observed),
        "expected": expected,
        "max_abs_error": max_abs,
        "cosine": cosine,
        "atol": atol,
        "rtol": rtol,
        "relative_reference_cutoff": RELATIVE_REFERENCE_CUTOFF,
        "passed": bool(torch.all(delta <= allowed)),
    }


def _component_gate(
    observed: float, expected: float, *, atol: float, rtol: float
) -> dict[str, Any]:
    if not math.isfinite(observed) or not math.isfinite(expected):
        raise ValueError("invalid A25 conformance component")
    relative = rtol * abs(expected) if abs(expected) >= RELATIVE_REFERENCE_CUTOFF else 0.0
    absolute_error = abs(observed - expected)
    allowed = atol + relative
    return {
        "observed": observed,
        "expected": expected,
        "absolute_error": absolute_error,
        "allowed_error": allowed,
        "passed": absolute_error <= allowed,
    }


def crpo_selector_gradient_oracle() -> dict[str, Any]:
    """Exercise source-derived selector semantics and project extensions.

    The pinned OnCRPO implementation inherits a TRPO update.  Therefore this
    fixture only attributes the single-cost branch-selection rule to OnCRPO;
    the clipped-PPO gradients and multi-cost tie-break are MultiTown extensions.
    """

    rows: dict[str, Any] = {}
    counts = CRPO_CONTRACT["counts"]
    for profile, profile_contract in CRPO_CONTRACT["profiles"].items():
        unsafe_limit = float(profile_contract["unsafe_limit"])
        wrong_limit = float(profile_contract["wrong_limit"])
        expected_mode = str(profile_contract["expected_mode"])
        expected_tie = bool(profile_contract["expected_tie"])
        decision = select_actor_mode(
            unsafe_events=int(counts["unsafe_events"]),
            wrong_executions=int(counts["wrong_executions"]),
            episodes=int(counts["episodes"]),
            thresholds=SafetyThresholds(
                unsafe=unsafe_limit,
                wrong_per_incident=wrong_limit,
                mean_incidents=float(CRPO_CONTRACT["mean_incidents"]),
            ),
        )
        evaluations: dict[str, Any] = {}
        for label, dtype, atol, rtol in (
            ("float64_oracle", ORACLE_DTYPE, ORACLE_ATOL, ORACLE_RTOL),
            ("float32_observed", OBSERVED_DTYPE, OBSERVED_ATOL, OBSERVED_RTOL),
        ):
            old_logits = _finite_vector(
                CRPO_CONTRACT["old_logits"],
                length=3,
                label="old_logits",
                dtype=dtype,
            )
            advantages = {
                key: _finite_vector(
                    values,
                    length=3,
                    label=key,
                    dtype=dtype,
                )
                for key, values in CRPO_CONTRACT["advantages"].items()
            }
            logits = old_logits.clone().requires_grad_(True)
            selected = (
                advantages["reward"]
                if decision.mode == "reward"
                else -advantages[decision.mode]
            )
            loss, ratio = _ppo_reward_loss(
                logits,
                old_logits,
                selected,
                clip_ratio=float(CRPO_CONTRACT["clip_ratio"]),
            )
            observed_gradient = _gradient(loss, logits)
            gradient = _gradient_gate(
                observed_gradient,
                CRPO_CONTRACT["expected_gradients"][decision.mode],
                atol=atol,
                rtol=rtol,
            )
            evaluations[label] = {
                "selected_advantage": _tensor_list(selected),
                "loss": float(loss.item()),
                "ratio_one": bool(torch.equal(ratio, torch.ones_like(ratio))),
                "gradient": gradient,
                "passed": (
                    float(loss.item()) == 0.0
                    and bool(torch.equal(ratio, torch.ones_like(ratio)))
                    and gradient["passed"]
                ),
            }
        expected_unsafe_cost = 1.0 / 3.0
        expected_wrong_cost = 1.0 / 3.0
        numeric_gates = {
            "unsafe_cost": _component_gate(
                decision.unsafe_cost,
                expected_unsafe_cost,
                atol=ORACLE_ATOL,
                rtol=ORACLE_RTOL,
            ),
            "wrong_cost": _component_gate(
                decision.wrong_cost,
                expected_wrong_cost,
                atol=ORACLE_ATOL,
                rtol=ORACLE_RTOL,
            ),
            "unsafe_threshold": _component_gate(
                decision.unsafe_threshold,
                unsafe_limit,
                atol=ORACLE_ATOL,
                rtol=ORACLE_RTOL,
            ),
            "wrong_threshold": _component_gate(
                decision.wrong_threshold,
                wrong_limit,
                atol=ORACLE_ATOL,
                rtol=ORACLE_RTOL,
            ),
        }
        gates = {
            "distance_zero_bound": CRPO_CONTRACT["oncrpo_distance"] == 0.0,
            "branch_exact": decision.mode == expected_mode,
            "tie_exact": decision.unsafe_tie_break_used is expected_tie,
            "numeric_contract": all(
                gate["passed"] for gate in numeric_gates.values()
            ),
            "float64_oracle": evaluations["float64_oracle"]["passed"],
            "float32_observed": evaluations["float32_observed"]["passed"],
        }
        rows[profile] = {
            "profile": profile,
            "contract": profile_contract,
            "decision": asdict(decision),
            "evaluations": evaluations,
            "scope": (
                "project_multi_cost_selector_extension"
                if profile in {"C-W", "C-T"}
                else "oncrpo_single_cost_selector_semantics"
            ),
            "numeric_gates": numeric_gates,
            "gates": gates,
            "passed": all(gates.values()),
        }
    return {
        "schema_version": "multitown-a25-crpo-conformance-v2",
        "official_source": OFFICIAL_SOURCE_PROFILES["oncrpo"],
        "contract": CRPO_CONTRACT,
        "scope": {
            "oncrpo_attribution": (
                "single-cost selector semantics with distance fixed to zero"
            ),
            "project_extensions": [
                "clipped-PPO gradient fixture",
                "multi-cost wrong branch",
                "normalized-violation tie-break",
            ],
            "official_oncrpo_update_reproduction": False,
        },
        "profiles": rows,
        "passed": all(row["passed"] for row in rows.values()),
    }


def _p3o_loss(
    logits: torch.Tensor,
    old_logits: torch.Tensor,
    reward_advantage: torch.Tensor,
    cost_advantage: torch.Tensor,
    *,
    mean_episode_cost: float,
    cost_limit: float,
    kappa: float,
    clip_ratio: float,
) -> dict[str, torch.Tensor]:
    scalars = (mean_episode_cost, cost_limit, kappa, clip_ratio)
    if (
        any(type(value) is not float or not math.isfinite(value) for value in scalars)
        or kappa < 0.0
    ):
        raise ValueError("invalid A25 P3O conformance scalar")
    reward_loss, ratio = _ppo_reward_loss(
        logits, old_logits, reward_advantage, clip_ratio=clip_ratio
    )
    cost_surrogate = (ratio * cost_advantage).mean()
    residual = cost_surrogate + mean_episode_cost - cost_limit
    cost_penalty = kappa * torch.relu(residual)
    total = reward_loss + cost_penalty
    if not all(
        bool(torch.isfinite(value))
        for value in (reward_loss, cost_surrogate, residual, cost_penalty, total)
    ):
        raise FloatingPointError("non-finite A25 P3O conformance loss")
    return {
        "ratio": ratio,
        "reward_loss": reward_loss,
        "cost_surrogate": cost_surrogate,
        "residual": residual,
        "cost_penalty": cost_penalty,
        "total_loss": total,
    }


def p3o_source_gradient_oracle() -> dict[str, Any]:
    """Exercise pinned P3O source semantics under explicit frozen assumptions."""

    rows: dict[str, Any] = {}
    for profile, profile_contract in P3O_CONTRACT["profiles"].items():
        policy = profile_contract["policy"]
        cost_limit = float(profile_contract["cost_limit"])
        kappa = float(profile_contract["kappa"])
        expected_components = profile_contract["expected_components"]
        expected_gradient = profile_contract["expected_gradient"]
        evaluations: dict[str, Any] = {}
        for label, dtype, atol, rtol in (
            ("float64_oracle", ORACLE_DTYPE, ORACLE_ATOL, ORACLE_RTOL),
            ("float32_observed", OBSERVED_DTYPE, OBSERVED_ATOL, OBSERVED_RTOL),
        ):
            old_logits = _finite_vector(
                P3O_CONTRACT["old_logits"],
                length=2,
                label="old_logits",
                dtype=dtype,
            )
            reward_advantage = _finite_vector(
                P3O_CONTRACT["reward_advantage"],
                length=2,
                label="reward",
                dtype=dtype,
            )
            cost_advantage = _finite_vector(
                P3O_CONTRACT["cost_advantage"],
                length=2,
                label="cost",
                dtype=dtype,
            )
            logits = torch.log(torch.tensor(policy, dtype=dtype)).requires_grad_(
                True
            )
            values = _p3o_loss(
                logits,
                old_logits,
                reward_advantage,
                cost_advantage,
                mean_episode_cost=float(P3O_CONTRACT["mean_episode_cost"]),
                cost_limit=cost_limit,
                kappa=kappa,
                clip_ratio=float(P3O_CONTRACT["clip_ratio"]),
            )
            observed_gradient = _gradient(values["total_loss"], logits)
            gradient = _gradient_gate(
                observed_gradient,
                expected_gradient,
                atol=atol,
                rtol=rtol,
            )
            components = {
                key: (
                    _tensor_list(value)
                    if value.ndim
                    else float(value.detach().item())
                )
                for key, value in values.items()
            }
            component_gates = {
                key: _component_gate(
                    float(components[key]),
                    float(expected),
                    atol=atol,
                    rtol=rtol,
                )
                for key, expected in expected_components.items()
            }
            residual = float(components["residual"])
            penalty = float(components["cost_penalty"])
            boundary_sign_passed = (
                residual < 0.0 and penalty == 0.0
                if profile == "P-B-"
                else residual == 0.0 and penalty == 0.0
                if profile == "P-B"
                else residual > 0.0 and penalty > 0.0
                if profile == "P-B+"
                else True
            )
            evaluations[label] = {
                "components": components,
                "expected_components": expected_components,
                "component_gates": component_gates,
                "boundary_sign_gate": {
                    "profile": profile,
                    "passed": boundary_sign_passed,
                },
                "gradient": gradient,
                "passed": (
                    all(gate["passed"] for gate in component_gates.values())
                    and boundary_sign_passed
                    and gradient["passed"]
                    and bool(torch.isfinite(values["ratio"]).all())
                ),
            }
        gates = {
            "entropy_zero_bound": P3O_CONTRACT["entropy_coef"] == 0.0,
            "single_cost_bound": P3O_CONTRACT["single_cost_only"] is True,
            "float64_oracle": evaluations["float64_oracle"]["passed"],
            "float32_observed": evaluations["float32_observed"]["passed"],
        }
        rows[profile] = {
            "profile": profile,
            "contract": profile_contract,
            "evaluations": evaluations,
            "boundary_semantics": (
                "pytorch_relu_zero_subgradient_at_runtime_boundary"
                if profile == "P-B"
                else (
                    "negative_residual_probe"
                    if profile == "P-B-"
                    else "positive_residual_probe"
                    if profile == "P-B+"
                    else None
                )
            ),
            "gates": gates,
            "passed": all(gates.values()),
        }
    return {
        "schema_version": "multitown-a25-p3o-conformance-v2",
        "official_source": OFFICIAL_SOURCE_PROFILES["p3o"],
        "contract": P3O_CONTRACT,
        "scope": {
            "source_semantics": (
                "P3O clipped reward loss plus rectified single-cost surrogate"
            ),
            "frozen_assumptions": [
                "entropy_coef=0",
                "single cost channel",
                "CPU eager PyTorch ReLU boundary semantics",
            ],
            "official_runtime_reproduction": False,
        },
        "profiles": rows,
        "passed": all(row["passed"] for row in rows.values()),
    }


def _implementation_provenance() -> dict[str, Any]:
    raw_paths = {
        "method_oracle": Path(__file__).absolute(),
        "selector": Path(inspect.getsourcefile(select_actor_mode) or "").absolute(),
        "threshold_contract": Path(
            inspect.getsourcefile(SafetyThresholds) or ""
        ).absolute(),
    }
    if any(path.is_symlink() for path in raw_paths.values()):
        raise RuntimeError("A25 method oracle refuses a symlinked source")
    source_paths = {label: path.resolve(strict=True) for label, path in raw_paths.items()}
    repository = source_paths["method_oracle"].parent.parent
    files: dict[str, Any] = {}
    for label, path in source_paths.items():
        if not path.is_file() or path.is_symlink() or not path.is_relative_to(repository):
            raise RuntimeError(f"invalid A25 method oracle source: {label}")
        payload = path.read_bytes()
        files[label] = {
            "path": path.relative_to(repository).as_posix(),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
    return {"files": files}


def build_method_conformance_receipt() -> dict[str, Any]:
    crpo = crpo_selector_gradient_oracle()
    p3o = p3o_source_gradient_oracle()
    contract_bytes = _canonical_bytes(METHOD_CONTRACT)
    core = {
        "schema_version": A25_METHOD_RECEIPT_VERSION,
        "implementation_version": A25_METHOD_CONFORMANCE_VERSION,
        "system_profile": {
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "device": "cpu",
            "oracle_dtype": str(ORACLE_DTYPE),
            "observed_dtype": str(OBSERVED_DTYPE),
            "oracle_rtol": ORACLE_RTOL,
            "oracle_atol": ORACLE_ATOL,
            "observed_rtol": OBSERVED_RTOL,
            "observed_atol": OBSERVED_ATOL,
        },
        "implementation_provenance": _implementation_provenance(),
        "method_contract": METHOD_CONTRACT,
        "method_contract_sha256": hashlib.sha256(contract_bytes).hexdigest(),
        "crpo": crpo,
        "p3o": p3o,
        "claim_boundary": {
            "diagnostic_oracle_receipt": True,
            "evidence_level": "diagnostic_only",
            "qualification_receipt": False,
            "distance_zero_selector_branch_cases_passed": True,
            "oncrpo_official_update_conformant": False,
            "project_ppo_and_multi_cost_extension_tested": True,
            "zero_entropy_p3o_loss_gradient_cases_passed": True,
            "official_source_executed": False,
            "paper_reproduction": False,
            "official_runtime_reproduction": False,
            "theory_inherited": False,
            "performance_claim_supported": False,
            "formal_authorized": False,
        },
    }
    gates = {
        "crpo_profiles_passed": core["crpo"]["passed"],
        "p3o_profiles_passed": core["p3o"]["passed"],
    }
    if not all(gates.values()):
        raise RuntimeError("A25 method conformance gate failed")
    bound = {**core, "gates": gates}
    return {
        **bound,
        "receipt_id": _canonical_sha256(bound),
        "status": "DIAGNOSTIC_PASSED",
    }


def validate_method_conformance_receipt(receipt: dict[str, Any]) -> None:
    if type(receipt) is not dict:
        raise TypeError("A25 method conformance receipt must be an object")
    expected = build_method_conformance_receipt()
    if _canonical_bytes(receipt) != _canonical_bytes(expected):
        raise ValueError("A25 method conformance receipt mismatch")
