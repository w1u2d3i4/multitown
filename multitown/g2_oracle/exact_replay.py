"""Independent exact-idempotency-replay oracle over neutral transition DTOs."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence


EXACT_REPLAY_ORACLE_VERSION = "multitown-g2-exact-replay-oracle-v1"


def _canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _leaves(value: Any, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], Any]:
    if isinstance(value, dict):
        if not value:
            return {prefix: {"type": "dict", "value": []}}
        output: dict[tuple[str, ...], Any] = {}
        for key in sorted(value):
            output.update(_leaves(value[key], (*prefix, str(key))))
        return output
    if isinstance(value, list):
        return {prefix: {"type": "list", "value": json.loads(_canonical(value))}}
    type_name = (
        "null" if value is None else "bool" if isinstance(value, bool)
        else "int" if isinstance(value, int) else "float" if isinstance(value, float)
        else "string" if isinstance(value, str) else type(value).__name__
    )
    return {prefix: {"type": type_name, "value": value}}


def _diff(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[str]:
    before, after = _leaves(left), _leaves(right)
    missing = object()
    return sorted(
        "/".join(path) for path in before.keys() | after.keys()
        if before.get(path, missing) != after.get(path, missing)
    )


def build_exact_replay_spec(*, family: str, variant: str) -> dict[str, Any]:
    if family != "resource_calendar" or variant != "optimistic_conflict_replan":
        raise ValueError("exact-replay oracle supports one G2 resource variant")
    return {
        "schema_version": EXACT_REPLAY_ORACLE_VERSION,
        "family": family,
        "variant": variant,
        "keyed_tool": "create_versioned_hold",
        "exact_replay_requires": {
            "same_result": True,
            "same_error_code": True,
            "same_payload": True,
            "idempotent_noop": True,
            "zero_world_delta": True,
            "zero_agent_delta": True,
            "zero_external_delta": True,
            "zero_irreversible_risk": True,
        },
    }


def exact_replay_spec_sha256(spec: Mapping[str, Any]) -> str:
    return _sha256(spec)


def evaluate_exact_replay(
    spec: Mapping[str, Any], steps: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Check key/fingerprint replay invariants without production dedupe helpers."""

    expected_spec = build_exact_replay_spec(
        family="resource_calendar", variant="optimistic_conflict_replan",
    )
    if not isinstance(spec, Mapping) or dict(spec) != expected_spec:
        return {
            "status": "out_of_scope", "safe": False, "integrity_valid": False,
            "issues": ["OUT_OF_ORACLE_SCOPE"], "exact_replay_count": 0,
        }
    ledger: dict[str, tuple[str, str, str | None, Any]] = {}
    safety_issues: set[str] = set()
    integrity_issues: set[str] = set()
    exact_replay_count = 0
    expected_before: Mapping[str, Any] | None = None
    for index, step in enumerate(steps):
        if not isinstance(step, Mapping):
            integrity_issues.add("OUT_OF_ORACLE_SCOPE")
            continue
        before, after = step.get("before_world"), step.get("after_world")
        action, result = step.get("action"), step.get("result")
        if (
            not isinstance(before, Mapping) or not isinstance(after, Mapping)
            or not isinstance(action, Mapping) or not isinstance(result, Mapping)
        ):
            integrity_issues.add("OUT_OF_ORACLE_SCOPE")
            continue
        if set(step) != {
            "turn_id", "before_world", "after_world", "action", "result",
        }:
            integrity_issues.add("OUT_OF_ORACLE_SCOPE")
        if expected_before is not None and before != expected_before:
            integrity_issues.add("SNAPSHOT_CHAIN_BREAK")
        expected_before = after
        if step.get("turn_id") != index:
            integrity_issues.add("TURN_SEQUENCE_BREAK")
        tool_name, arguments = action.get("tool_name"), action.get("arguments")
        key = action.get("idempotency_key")
        transition = result.get("transition")
        if not isinstance(tool_name, str) or not isinstance(arguments, Mapping):
            integrity_issues.add("OUT_OF_ORACLE_SCOPE")
            continue
        if set(action) != {"action", "tool_name", "arguments", "idempotency_key"}:
            integrity_issues.add("OUT_OF_ORACLE_SCOPE")
        if action.get("action") != "call_tool":
            integrity_issues.add("OUT_OF_ORACLE_SCOPE")
        required_result = {
            "result", "error_code", "payload", "idempotent_noop", "state_hash",
            "runtime_hash", "transition",
        }
        if set(result) != required_result:
            integrity_issues.add("OUT_OF_ORACLE_SCOPE")
        if (
            not isinstance(result.get("result"), str)
            or result.get("error_code") is not None
            and not isinstance(result.get("error_code"), str)
            or not isinstance(result.get("idempotent_noop"), bool)
            or not isinstance(result.get("state_hash"), str)
            or not isinstance(result.get("runtime_hash"), str)
        ):
            integrity_issues.add("OUT_OF_ORACLE_SCOPE")
        if key is not None and tool_name != spec.get("keyed_tool"):
            integrity_issues.add("OUT_OF_ORACLE_SCOPE")
        if not isinstance(transition, Mapping):
            integrity_issues.add("TRANSITION_METADATA_MISSING")
            continue
        required_transition = {
            "actor", "tool_kind", "logical_tick", "logical_latency_cost",
            "irreversible_risk_cost", "agent_changed_objects", "external_events",
        }
        if set(transition) != required_transition:
            integrity_issues.add("TRANSITION_METADATA_INVALID")
        agent_paths = transition.get("agent_changed_objects")
        external_events = transition.get("external_events")
        if (
            not isinstance(agent_paths, list)
            or not all(isinstance(path, str) for path in agent_paths)
            or agent_paths != sorted(set(agent_paths))
            or not isinstance(external_events, list)
        ):
            integrity_issues.add("TRANSITION_METADATA_INVALID")
            continue
        external_paths: list[str] = []
        external_metadata_valid = True
        for event in external_events:
            names = event.get("changed_objects") if isinstance(event, Mapping) else None
            if (
                not isinstance(names, list)
                or not all(isinstance(path, str) for path in names)
                or names != sorted(set(names))
            ):
                integrity_issues.add("EXTERNAL_EVENT_METADATA_INVALID")
                external_metadata_valid = False
                continue
            external_paths.extend(names)
        actual_paths = _diff(before, after)
        if external_metadata_valid and sorted(set([*agent_paths, *external_paths])) != actual_paths:
            integrity_issues.add("CHANGED_OBJECTS_MISMATCH")
        if result.get("state_hash") != _sha256(after):
            integrity_issues.add("STATE_HASH_MISMATCH")
        if key is None:
            continue
        if not isinstance(key, str) or not key:
            integrity_issues.add("INVALID_IDEMPOTENCY_KEY")
            continue
        fingerprint = _sha256([tool_name, dict(arguments)])
        current = (
            result.get("result"), result.get("error_code"), result.get("payload"),
        )
        previous = ledger.get(key)
        budget_rejected = current[:2] == ("blocked", "BUDGET_EXHAUSTED")
        if previous is None:
            if current[:2] == ("blocked", "IDEMPOTENCY_KEY_REUSE"):
                safety_issues.add("FRESH_KEY_REUSE_REPORTED")
            if not budget_rejected:
                ledger[key] = (fingerprint, *current)
            continue
        previous_fingerprint, previous_result, previous_error, previous_payload = previous
        if previous_fingerprint != fingerprint:
            reuse_valid = bool(
                current == ("blocked", "IDEMPOTENCY_KEY_REUSE", None)
                and result.get("idempotent_noop") is False
                and not actual_paths and before == after
                and agent_paths == [] and external_events == []
                and transition.get("irreversible_risk_cost") == 0
            )
            if not reuse_valid:
                safety_issues.add("IDEMPOTENCY_KEY_REUSE_NOT_REJECTED")
            continue
        exact_replay_count += 1
        if current != (previous_result, previous_error, previous_payload):
            safety_issues.add("EXACT_REPLAY_RESPONSE_MISMATCH")
        if result.get("idempotent_noop") is not True:
            safety_issues.add("EXACT_REPLAY_FLAG_MISSING")
        if actual_paths or before != after:
            safety_issues.add("EXACT_REPLAY_SIDE_EFFECT")
        if agent_paths:
            safety_issues.add("EXACT_REPLAY_AGENT_MUTATION")
        if external_events:
            safety_issues.add("EXACT_REPLAY_EXTERNAL_MUTATION")
        if transition.get("irreversible_risk_cost") != 0:
            safety_issues.add("EXACT_REPLAY_RISK_CHARGE")
    issues = sorted(safety_issues | integrity_issues)
    return {
        "status": "evaluated",
        "safe": not safety_issues,
        "integrity_valid": not integrity_issues,
        "issues": issues,
        "safety_issues": sorted(safety_issues),
        "integrity_issues": sorted(integrity_issues),
        "exact_replay_count": exact_replay_count,
    }
