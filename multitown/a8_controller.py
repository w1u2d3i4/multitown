"""Deterministic safety checks and sparse-reorganization decisions for A8."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .scenarios import Scenario


CONTROLLER_VERSION = "multitown-a8-selective-delegation-v1"


@dataclass(frozen=True)
class ValidationResult:
    parse_valid: bool
    hard_constraints_pass: bool
    issue_codes: tuple[str, ...]
    checked_action: str | None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["issue_codes"] = list(self.issue_codes)
        return payload


def _fail(action: str | None, *issues: str) -> ValidationResult:
    return ValidationResult(True, False, tuple(issues), action)


def validate_candidate(scenario: Scenario, action: str | None) -> ValidationResult:
    """Check hard feasibility/safety only; never compare against the oracle action."""
    if action is None or action not in scenario.allowed_actions:
        return ValidationResult(False, False, ("invalid_or_out_of_schema_action",), action)
    metadata = scenario.metadata
    if scenario.family == "resource_allocation":
        if action == "request_mutual_aid":
            feasible = [
                row for row in metadata["facilities"]
                if int(row["required"]) <= int(metadata["available"])
            ]
            return _fail(action, "unnecessary_mutual_aid") if feasible else ValidationResult(True, True, (), action)
        name = action.removeprefix("allocate_")
        row = next(item for item in metadata["facilities"] if item["name"] == name)
        if int(row["required"]) > int(metadata["available"]):
            return _fail(action, "insufficient_repair_crews")
    elif scenario.family == "incident_dispatch":
        feasible = [row for row in metadata["incidents"] if bool(row["feasible"])]
        if action == "request_mutual_aid":
            return _fail(action, "unnecessary_mutual_aid") if feasible else ValidationResult(True, True, (), action)
        name = action.removeprefix("dispatch_")
        row = next(item for item in metadata["incidents"] if item["name"] == name)
        if not bool(row["feasible"]):
            return _fail(action, "incident_deadline_missed")
    elif scenario.family == "evidence_fusion":
        if action.startswith("isolate_"):
            scores = dict(metadata["scores"])
            ordered = sorted(scores, key=lambda key: (scores[key], key), reverse=True)
            top, second = ordered[:2]
            decisive = (
                int(scores[top]) >= int(metadata["threshold"])
                and int(scores[top]) - int(scores[second]) >= int(metadata["min_gap"])
            )
            if not decisive:
                return _fail(action, "unsafe_isolation_under_ambiguous_evidence")
    elif scenario.family == "dependency_recovery":
        if not action.startswith("repair_"):
            return _fail(action, "downstream_restart_before_root_repair")
        name = action.removeprefix("repair_")
        row = next(item for item in metadata["statuses"] if item["node"] == name)
        if row["status"] != "offline":
            return _fail(action, "repair_target_not_offline")
    elif scenario.family == "supply_route":
        feasible = [row for row in metadata["routes"] if bool(row["feasible"])]
        if action == "request_airlift":
            return _fail(action, "unnecessary_airlift") if feasible else ValidationResult(True, True, (), action)
        name = action.removeprefix("use_")
        row = next(item for item in metadata["routes"] if item["name"] == name)
        if not bool(row["feasible"]):
            return _fail(action, "route_violates_capacity_risk_or_time")
    elif scenario.family == "fault_recovery":
        if action == "ignore_failure":
            return _fail(action, "unsafe_ignore_failure")
        if action.startswith("retry_") and (
            int(metadata["retries"]) >= 2 or bool(metadata["irreversible"])
        ):
            return _fail(action, "retry_budget_or_reversibility_violation")
        if action == "switch_safe_fallback" and not bool(metadata["fallback_safe"]):
            return _fail(action, "safe_fallback_unavailable")
    return ValidationResult(True, True, (), action)


def simulate_a8_cell(
    *,
    scenario: Scenario,
    a0: dict[str, Any],
    a1: dict[str, Any],
    a2: dict[str, Any],
    predicted_a0_accuracy: float,
    early_stop_threshold: float,
) -> dict[str, Any]:
    """Approximate A8 from fully observed arms for dev-only threshold selection."""
    initial_action = a0.get("selected_action")
    initial_validation = validate_candidate(scenario, initial_action)
    selected = a0
    route = "initial_weak_early_stop"
    delegated = False
    weak_specialist_called = False
    strong_specialist_called = False
    tokens = float(a0["total_tokens"])
    latency = float(a0["decision_latency_s"])
    if not (
        initial_validation.hard_constraints_pass
        and predicted_a0_accuracy >= early_stop_threshold
    ):
        delegated = True
        if initial_validation.hard_constraints_pass:
            candidates = list(a2.get("candidate_actions") or [])
            second_action = candidates[0] if candidates else a2.get("selected_action")
            second_validation = validate_candidate(scenario, second_action)
            weak_specialist_called = True
            tokens += float(a2["total_tokens"]) / max(1, int(a2.get("weak_calls", 4)))
            latency += float(a2["decision_latency_s"])
            if (
                second_validation.hard_constraints_pass
                and second_action == initial_action
            ):
                route = "two_weak_consensus_stop"
            else:
                strong_specialist_called = True
        else:
            strong_specialist_called = True
        if strong_specialist_called:
            strong_validation = validate_candidate(scenario, a1.get("selected_action"))
            tokens += float(a1["total_tokens"])
            latency += float(a1["decision_latency_s"])
            if bool(a1.get("valid")) and strong_validation.hard_constraints_pass:
                selected = a1
                route = "strong_specialist_resolution"
            else:
                route = "deterministic_safe_fallback"
    return {
        "scenario_id": scenario.scenario_id,
        "family": scenario.family,
        "selected_action": selected.get("selected_action"),
        "correct": bool(selected.get("correct")),
        "valid": bool(selected.get("valid")),
        "total_tokens": tokens,
        "decision_latency_s": latency,
        "route": route,
        "delegated": delegated,
        "early_stop": not delegated,
        "weak_specialist_called": weak_specialist_called,
        "strong_specialist_called": strong_specialist_called,
        "reorganization_count": int(weak_specialist_called) + int(strong_specialist_called),
        "initial_correct": bool(a0.get("correct")),
        "reorganization_gain": int(bool(selected.get("correct"))) - int(bool(a0.get("correct"))),
        "initial_validation": initial_validation.to_dict(),
        "predicted_a0_accuracy": predicted_a0_accuracy,
        "early_stop_threshold": early_stop_threshold,
        "simulation_note": "dev-only approximation: A2 first candidate is the second weak specialist and A1 is the strong specialist",
    }
