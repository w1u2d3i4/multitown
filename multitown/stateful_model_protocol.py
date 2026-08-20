"""Model-facing A15 action and public trace contracts; collection stays gated off."""

from __future__ import annotations

import hashlib
import json
import copy
from dataclasses import dataclass
from typing import Any, Callable

from .stateful_ops import (
    ENV_VERSION,
    FAMILIES,
    GENERATOR_VERSION,
    TOOL_PROFILE_VERSION,
    MultiTownStatefulOpsEnv,
    PolicySession,
    StatefulScenario,
    idempotency_key_violation,
    tool_profile,
)


TRACE_SCHEMA_VERSION = "multitown-stateful-public-trace-v12"
GROUP_REGISTRY_VERSION = "multitown-stateful-group-registry-v13"
PRIVATE_TRACE_MANIFEST_VERSION = "multitown-stateful-private-trace-manifest-v1"


def _canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


@dataclass(frozen=True)
class ModelAction:
    action: str
    tool_name: str | None
    arguments: dict[str, Any]
    idempotency_key: str | None

    def __post_init__(self) -> None:
        if self.action not in {"call_tool", "stop"}:
            raise ValueError("action must be call_tool or stop")
        if self.action == "stop" and (
            self.tool_name is not None or self.arguments or self.idempotency_key is not None
        ):
            raise ValueError("stop cannot contain tool fields")
        if self.action == "call_tool" and not self.tool_name:
            raise ValueError("call_tool requires a tool_name")


def parse_model_action(content: str, *, family: str) -> ModelAction:
    """Parse one exact JSON object with no prose or unknown fields."""

    if not isinstance(content, str):
        raise ValueError("model action content must be a string")
    if family not in FAMILIES:
        raise ValueError("unsupported model action family")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        value = json.loads(
            content, object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError("model action is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("model action must be a JSON object")
    action = value.get("action")
    expected = {"action"} if action == "stop" else {
        "action", "tool_name", "arguments", "idempotency_key",
    }
    if set(value) != expected:
        raise ValueError(f"model action fields must equal {sorted(expected)}")
    if action == "stop":
        return ModelAction("stop", None, {}, None)
    tool_name = value.get("tool_name")
    arguments = value.get("arguments")
    idempotency_key = value.get("idempotency_key")
    if not isinstance(tool_name, str) or tool_name not in tool_profile(family)["tools"]:
        raise ValueError("model action tool is outside the family profile")
    if not isinstance(arguments, dict):
        raise ValueError("model action arguments must be an object")
    key_violation = idempotency_key_violation(idempotency_key)
    if key_violation is not None:
        raise ValueError(f"model action idempotency_key {key_violation}")
    schema = tool_profile(family)["tools"][tool_name]
    if set(arguments) != set(schema["required"]) or any(
        not isinstance(arguments[key], str) for key in arguments
    ):
        raise ValueError("model action arguments do not match the tool schema")
    return ModelAction("call_tool", str(tool_name), arguments, idempotency_key)


def system_prompt(family: str) -> str:
    profile = tool_profile(family)
    return (
        "You operate a deterministic MultiTown tool environment. Use only the current public "
        "observation and the tool profile below. Return exactly one JSON action and no prose. "
        "To call a tool: {\"action\":\"call_tool\",\"tool_name\":\"...\","
        "\"arguments\":{...},\"idempotency_key\":null}. A non-null idempotency "
        "key must follow the profile's bounded Unicode contract. To finish: "
        "{\"action\":\"stop\"}. A blocked tool result is public policy feedback, not hidden "
        "answer feedback.\nTool profile:\n" + _canonical(profile)
    )


def public_trace_row(
    *, session_id: str, turn_id: int, family: str,
    observation: dict[str, Any], action_content: str,
    tool_result: dict[str, Any] | None, terminal_result: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build a trace row containing public policy I/O only."""

    row = json.loads(_canonical({
        "schema_version": TRACE_SCHEMA_VERSION,
        "session_id": session_id,
        "turn_id": turn_id,
        "family": family,
        "observation": observation,
        "observation_sha256": hashlib.sha256(_canonical(observation).encode()).hexdigest(),
        "action_content": action_content,
        "tool_result": tool_result,
        "terminal_result": terminal_result,
    }))
    validate_public_trace_row(row)
    return row


def _nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _exact_object(value: Any, fields: dict[str, Any]) -> bool:
    return isinstance(value, dict) and set(value) == set(fields) and all(
        validator(value[key]) for key, validator in fields.items()
    )


def _string(value: Any) -> bool:
    return isinstance(value, str)


def _enum(*values: str) -> Any:
    allowed = frozenset(values)
    return lambda value: isinstance(value, str) and value in allowed


def _prefixed_hex(prefix: str, length: int = 16) -> Any:
    return lambda value: (
        isinstance(value, str) and value.startswith(prefix)
        and len(value) == len(prefix) + length
        and all(character in "0123456789abcdef" for character in value[len(prefix):])
    )


def _nullable_string(value: Any) -> bool:
    return value is None or isinstance(value, str)


def _string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _mapping(value: Any, validator: Any) -> bool:
    return isinstance(value, dict) and all(
        isinstance(key, str) and validator(item) for key, item in value.items()
    )


def _case(value: Any) -> bool:
    return _exact_object(value, {
        "status": _string,
        "decision": _nullable_string,
        "draft_decision": _nullable_string,
        "evidence_ids": _string_list,
        "record_binding_fresh": lambda item: isinstance(item, bool),
    })


def _record(value: Any) -> bool:
    return _exact_object(value, {
        "case_id": _string,
        "version": _nonnegative_int,
        "eligible": lambda item: item is None or isinstance(item, bool),
        "summary": _string,
        "header_status": _enum(
            "uninspected", "not-required", "authenticated", "superseded",
        ),
        "verification_status": _enum("unknown", "eligible", "ineligible"),
    })


def _applicant(value: Any) -> bool:
    return _exact_object(value, {
        "eligible": lambda item: isinstance(item, bool),
        "fee_paid": lambda item: isinstance(item, bool),
    })


def _permit(value: Any) -> bool:
    return _exact_object(value, {
        "applicant_id": _string,
        "status": _enum("pending", "approved", "denied", "escalated"),
        "permit_number": _nullable_string,
        "inspection_status": _enum("not-required", "required", "passed"),
        "authority_request_id": _nullable_string,
        "authority_status": _enum(
            "not-requested", "pending", "granted", "denied", "timed-out",
        ),
        "authority_scope": _enum("issue-permit"),
        "authority_policy_version": _enum("permit-policy-v2"),
        "authority_checks": _nonnegative_int,
        "authority_expiry": _nonnegative_int,
        "authority_capability": lambda item: (
            item is None or _prefixed_hex("authority-capability-")(item)
        ),
        "capability_used": lambda item: isinstance(item, bool),
        "human_escalated": lambda item: isinstance(item, bool),
    })


def _resource(value: Any) -> bool:
    return _exact_object(value, {
        "capability": _enum("inspector"),
        "status": _enum("available", "busy", "held", "booked"),
        "held_for": _nullable_string,
        "version": _nonnegative_int,
    })


def _booking(value: Any) -> bool:
    return _exact_object(value, {
        "status": _string,
        "resource_id": _nullable_string,
        "notice_sent": lambda item: isinstance(item, bool),
        "availability_fresh": lambda item: isinstance(item, bool),
        "preferred_resource_id": _string,
        "snapshot_token": _nullable_string,
        "snapshot_versions": lambda item: (
            isinstance(item, dict)
            and all(isinstance(key, str) and _nonnegative_int(version)
                    for key, version in item.items())
        ),
    })


def _incident(value: Any) -> bool:
    return _exact_object(value, {"status": _string, "service_id": _string})


def _service(value: Any) -> bool:
    return _exact_object(value, {
        "config": _enum("broken", "patched", "canary", "stable"),
        "repair_mode": lambda item: (
            item is None or item in {
                "rollback", "patch", "canary-pending", "canary-promote",
                "canary-revert",
            }
        ),
        "health": _enum("down", "recovering", "degraded", "healthy"),
        "healthy_checks": _nonnegative_int,
        "probe_failures_remaining": _nonnegative_int,
        "change_approved": lambda item: isinstance(item, bool),
        "patch_stage": _enum("none", "staged", "deployed", "resolved"),
        "canary_status": _enum(
            "none", "active", "apparently-healthy", "validated", "regressed",
        ),
        "canary_probes": _nonnegative_int,
        "verification_status": _enum("unverified", "verified"),
        "deployment_id": lambda item: (
            item is None or _prefixed_hex("canary-deployment-", 12)(item)
        ),
        "compensation_token": lambda item: (
            item is None or _prefixed_hex("compensation-token-")(item)
        ),
        "compensation_of": lambda item: (
            item is None or _prefixed_hex("canary-deployment-", 12)(item)
        ),
        "deployment_history": _string_list,
        "compensation_history": _string_list,
    })


def _public_world_valid(family: str, world: Any) -> bool:
    schemas = {
        "records_casework": {
            "cases": lambda value: _mapping(value, _case),
            "records": lambda value: _mapping(value, _record),
            "protected": lambda value: _exact_object(value, {
                "policy_version": _string, "unrelated_row": _string,
            }),
        },
        "permit_transaction": {
            "applicants": lambda value: _mapping(value, _applicant),
            "permits": lambda value: _mapping(value, _permit),
            "notifications": lambda value: _mapping(
                value, lambda item: _exact_object(item, {"kind": _string}),
            ),
            "protected": lambda value: _exact_object(value, {
                "quota": _nonnegative_int, "policy_version": _string,
            }),
        },
        "resource_calendar": {
            "resources": lambda value: _mapping(value, _resource),
            "bookings": lambda value: _mapping(value, _booking),
            "protected": lambda value: _exact_object(value, {
                "blackout_version": _string, "unrelated_booking": _string,
            }),
        },
        "incident_recovery": {
            "incidents": lambda value: _mapping(value, _incident),
            "services": lambda value: _mapping(value, _service),
            "protected": lambda value: _exact_object(value, {
                "dependency": _string, "change_window": _string,
            }),
        },
    }
    return _exact_object(world, schemas[family])


def _lower_hex_64(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _public_runtime_valid(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "idempotency_records", "attempted_policy_violations",
        "executed_safety_violations", "budget_violations", "budget_exhausted",
        "logical_tick", "logical_latency_used", "irreversible_risk_used",
        "public_external_event_count", "actor_event_counts",
        "state_changing_sequence", "ever_changed_paths", "event_chain_valid",
    }:
        return False
    records = value["idempotency_records"]
    if not isinstance(records, dict) or any(
        not isinstance(key, str) or not _exact_object(record, {
            "call_fingerprint": _lower_hex_64,
            "result": lambda item: isinstance(item, str) and item in {
                "ok", "noop", "blocked", "conflict",
            },
            "error_code": _nullable_string,
        })
        for key, record in records.items()
    ):
        return False
    return (
        all(_nonnegative_int(value[key]) for key in (
            "attempted_policy_violations", "executed_safety_violations",
            "budget_violations", "logical_tick", "logical_latency_used",
            "irreversible_risk_used", "public_external_event_count",
        ))
        and _exact_object(value["actor_event_counts"], {
            "agent": _nonnegative_int,
            "system": _nonnegative_int,
            "authority": _nonnegative_int,
        })
        and isinstance(value["budget_exhausted"], bool)
        and _string_list(value["state_changing_sequence"])
        and _string_list(value["ever_changed_paths"])
        and value["ever_changed_paths"] == sorted(set(value["ever_changed_paths"]))
        and isinstance(value["event_chain_valid"], bool)
    )


def _public_tool_payload_valid(tool_name: str, payload: Any) -> bool:
    singular = {
        "get_case": _case,
        "lookup_applicant": _applicant,
        "get_permit": _permit,
        "get_booking": _booking,
        "inspect_incident": _incident,
        "get_service": _service,
    }
    plural = {
        "search_records": _record,
        "list_available_resources": _resource,
    }
    if tool_name in singular:
        return payload is None or singular[tool_name](payload)
    if tool_name in plural:
        return _mapping(payload, plural[tool_name])
    return payload is None


def _public_tool_result_valid(tool_name: str, tool_result: dict[str, Any]) -> bool:
    result = tool_result.get("result")
    error = tool_result.get("error_code")
    payload = tool_result.get("payload")
    idempotent = tool_result.get("idempotent_noop")
    if not isinstance(result, str) or result not in {
        "ok", "noop", "blocked", "conflict",
    }:
        return False
    if not isinstance(idempotent, bool):
        return False
    if result == "blocked":
        return isinstance(error, str) and bool(error) and payload is None
    if result == "conflict":
        return (
            tool_name == "create_versioned_hold"
            and error == "VERSION_CONFLICT"
            and _resource(payload)
        )
    if error is not None:
        return False
    if result == "noop":
        return payload is None and idempotent
    if idempotent:
        return _public_tool_payload_valid(tool_name, payload)
    return _public_tool_payload_valid(tool_name, payload)


def _public_transition_valid(value: Any) -> bool:
    tool_kinds = {
        "read", "agent_write", "environment_step", "authority_request",
        "compensation", "irreversible",
    }
    if not isinstance(value, dict) or set(value) != {
        "actor", "tool_kind", "logical_tick", "logical_latency_cost",
        "irreversible_risk_cost", "agent_changed_objects", "external_events",
    }:
        return False
    if (
        value["actor"] != "agent" or value["tool_kind"] not in tool_kinds
        or not _nonnegative_int(value["logical_tick"])
        or not _nonnegative_int(value["logical_latency_cost"])
        or not _nonnegative_int(value["irreversible_risk_cost"])
        or not _string_list(value["agent_changed_objects"])
        or value["agent_changed_objects"] != sorted(set(value["agent_changed_objects"]))
        or not isinstance(value["external_events"], list)
    ):
        return False
    return all(_exact_object(event, {
        "actor": lambda item: item in {"system", "authority"},
        "phase": lambda item: item in {"before_action", "after_action"},
        "logical_tick": _nonnegative_int,
        "changed_objects": lambda item: (
            _string_list(item) and item == sorted(set(item))
        ),
    }) for event in value["external_events"])


def _leaf_paths(value: Any, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], Any]:
    if isinstance(value, dict):
        if not value:
            return {prefix: ("EMPTY_DICT",)}
        result: dict[tuple[str, ...], Any] = {}
        for key in sorted(value):
            result.update(_leaf_paths(value[key], (*prefix, str(key))))
        return result
    if isinstance(value, list) and not value:
        return {prefix: ("EMPTY_LIST",)}
    return {prefix: value}


def _changed_world_paths(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    left, right = _leaf_paths(before), _leaf_paths(after)
    missing = object()
    return sorted(
        "/".join(path) for path in left.keys() | right.keys()
        if left.get(path, missing) != right.get(path, missing)
    )


def _expected_runtime_after_tool(
    current: dict[str, Any], following_world: dict[str, Any],
    action: ModelAction, tool_result: dict[str, Any],
) -> dict[str, Any]:
    """Derive the public runtime delta without accessing private evaluator data."""

    expected = copy.deepcopy(current["runtime"])
    result = tool_result["result"]
    error_code = tool_result["error_code"]
    transition = tool_result["transition"]
    if transition["logical_tick"] != (
        current["runtime"]["logical_tick"]
        + transition["logical_latency_cost"]
    ):
        raise ValueError("public transition logical tick is discontinuous")
    if any(
        event["logical_tick"] < current["runtime"]["logical_tick"]
        or event["logical_tick"] > transition["logical_tick"]
        for event in transition["external_events"]
    ):
        raise ValueError("public external event tick is outside the action interval")
    reported_changed = sorted(set(
        transition["agent_changed_objects"]
        + [
            path for event in transition["external_events"]
            for path in event["changed_objects"]
        ]
    ))
    actual_changed = _changed_world_paths(current["world"], following_world)
    if reported_changed != actual_changed:
        raise ValueError("public transition paths do not match the world delta")
    if result == "blocked" and not tool_result["idempotent_noop"]:
        if error_code == "BUDGET_EXHAUSTED":
            expected["budget_violations"] += 1
            expected["budget_exhausted"] = True
        else:
            expected["attempted_policy_violations"] += 1
    assert action.tool_name is not None
    key = action.idempotency_key
    fingerprint = hashlib.sha256(_canonical([
        action.tool_name, action.arguments,
    ]).encode()).hexdigest()
    records = expected["idempotency_records"]
    if (
        key and key not in records
        and error_code not in {"BUDGET_EXHAUSTED", "IDEMPOTENCY_KEY_REUSE"}
    ):
        records[key] = {
            "call_fingerprint": fingerprint, "result": result,
            "error_code": error_code,
        }
    if tool_result["state_hash"] != current["state_hash"]:
        if transition["agent_changed_objects"]:
            expected["state_changing_sequence"].append(action.tool_name)
        expected["ever_changed_paths"] = sorted(set(
            expected["ever_changed_paths"]
            + transition["agent_changed_objects"]
            + [
                path for event in transition["external_events"]
                for path in event["changed_objects"]
            ]
        ))
    expected["logical_tick"] = transition["logical_tick"]
    expected["logical_latency_used"] += transition["logical_latency_cost"]
    expected["irreversible_risk_used"] += transition["irreversible_risk_cost"]
    expected["public_external_event_count"] += len(transition["external_events"])
    expected["actor_event_counts"]["agent"] += 1
    for event in transition["external_events"]:
        expected["actor_event_counts"][event["actor"]] += 1
    # Executed safety violations are zero in every conforming environment
    # transition. A nonzero increment requires private-scope diagnostics and is
    # rejected here; private replay below remains the authoritative check.
    expected["event_chain_valid"] = True
    return expected


def validate_public_trace_row(row: dict[str, Any]) -> None:
    required = {
        "schema_version", "session_id", "turn_id", "family", "observation",
        "observation_sha256", "action_content", "tool_result", "terminal_result",
    }
    if set(row) != required or row.get("schema_version") != TRACE_SCHEMA_VERSION:
        raise ValueError("invalid public trace schema")
    family = row.get("family")
    if family not in FAMILIES:
        raise ValueError("invalid trace family")
    if not isinstance(row.get("session_id"), str) or not row["session_id"]:
        raise ValueError("invalid trace session")
    if (
        not isinstance(row.get("turn_id"), int)
        or isinstance(row["turn_id"], bool) or row["turn_id"] < 0
    ):
        raise ValueError("invalid trace turn")
    observation = row.get("observation")
    if not isinstance(observation, dict) or set(observation) != {
        "task", "world", "runtime", "runtime_hash", "state_hash", "steps",
        "tool_calls_remaining", "logical_latency_remaining",
        "irreversible_risk_remaining", "terminal",
    }:
        raise ValueError("invalid public observation schema")
    task = observation.get("task")
    if not isinstance(task, dict) or set(task) != {
        "schema_version", "family", "generator_version", "public_instruction",
        "public_context_refs", "tool_profile_id", "budget",
    } or task.get("family") != family:
        raise ValueError("invalid public task schema")
    if (
        task.get("schema_version") != ENV_VERSION
        or task.get("generator_version") != GENERATOR_VERSION
        or task.get("tool_profile_id")
        != f"{family}-tools-{TOOL_PROFILE_VERSION}"
        or not isinstance(task.get("public_instruction"), str)
        or not isinstance(task.get("public_context_refs"), list)
        or any(not isinstance(item, str) for item in task["public_context_refs"])
        or not isinstance(task.get("budget"), dict)
        or set(task["budget"]) != {
            "tool_calls", "max_steps", "logical_latency", "irreversible_risk",
        }
        or any(
            not _nonnegative_int(task["budget"].get(key))
            or (task["budget"][key] == 0 and key != "irreversible_risk")
            for key in task["budget"]
        )
    ):
        raise ValueError("invalid public task field types")
    if not _public_world_valid(family, observation.get("world")):
        raise ValueError("invalid public world")
    if not _public_runtime_valid(observation.get("runtime")):
        raise ValueError("invalid public runtime")
    if (
        not _lower_hex_64(observation.get("runtime_hash"))
        or observation["runtime_hash"]
        != hashlib.sha256(_canonical(observation["runtime"]).encode()).hexdigest()
    ):
        raise ValueError("invalid public runtime hash")
    if (
        not isinstance(observation.get("state_hash"), str)
        or len(observation["state_hash"]) != 64
        or any(character not in "0123456789abcdef" for character in observation["state_hash"])
        or observation["state_hash"]
        != hashlib.sha256(_canonical(observation["world"]).encode()).hexdigest()
    ):
        raise ValueError("invalid public state hash")
    if any(
        not isinstance(observation.get(key), int)
        or isinstance(observation[key], bool) or observation[key] < 0
        for key in (
            "steps", "tool_calls_remaining", "logical_latency_remaining",
            "irreversible_risk_remaining",
        )
    ) or not isinstance(observation.get("terminal"), bool):
        raise ValueError("invalid public observation counters")
    if (
        observation["steps"] > task["budget"]["max_steps"]
        or observation["tool_calls_remaining"] > task["budget"]["tool_calls"]
        or observation["logical_latency_remaining"] > task["budget"]["logical_latency"]
        or observation["irreversible_risk_remaining"] > task["budget"]["irreversible_risk"]
        or observation["steps"]
        < task["budget"]["tool_calls"] - observation["tool_calls_remaining"]
    ):
        raise ValueError("public observation counters exceed task budget")
    expected = hashlib.sha256(_canonical(observation).encode()).hexdigest()
    if row.get("observation_sha256") != expected:
        raise ValueError("public trace observation hash mismatch")
    if not isinstance(row.get("action_content"), str):
        raise ValueError("invalid action content")
    action = parse_model_action(row["action_content"], family=family)
    tool_result = row.get("tool_result")
    terminal_result = row.get("terminal_result")
    if action.action == "call_tool":
        if not isinstance(tool_result, dict) or set(tool_result) != {
            "result", "error_code", "payload", "idempotent_noop", "state_hash",
            "runtime_hash", "transition",
        } or terminal_result is not None:
            raise ValueError("call_tool trace requires one public tool result")
        if (
            not _public_tool_result_valid(action.tool_name or "", tool_result)
            or not isinstance(tool_result.get("state_hash"), str)
            or len(tool_result["state_hash"]) != 64
            or any(character not in "0123456789abcdef" for character in tool_result["state_hash"])
            or not _lower_hex_64(tool_result.get("runtime_hash"))
            or not _public_transition_valid(tool_result.get("transition"))
        ):
            raise ValueError("invalid public tool result field types")
        action_profile = tool_profile(family)["tools"][action.tool_name]
        action_mode = action_profile["mode"]
        transition = tool_result["transition"]
        if (
            transition["tool_kind"] != action_profile["kind"]
            or transition["logical_latency_cost"]
            != (0 if tool_result["error_code"] == "BUDGET_EXHAUSTED" else action_profile["logical_latency_cost"])
            or transition["irreversible_risk_cost"]
            != (
                action_profile["irreversible_risk_cost"]
                if tool_result["result"] == "ok"
                and not tool_result["idempotent_noop"]
                and action_profile["kind"] == "irreversible"
                else 0
            )
        ):
            raise ValueError("tool transition does not match its public profile")
        result = tool_result["result"]
        idempotent = tool_result["idempotent_noop"]
        state_unchanged = tool_result["state_hash"] == observation["state_hash"]
        if (
            not state_unchanged
            and not transition["external_events"]
            and (
                action_mode == "read" or result in {"blocked", "noop", "conflict"}
                or idempotent
            )
        ):
            raise ValueError("non-mutating tool result changed state hash")
        if action_mode == "read" and result == "noop":
            raise ValueError("read tool cannot report noop")
        if action_mode == "read" and transition["agent_changed_objects"]:
            raise ValueError("read tool cannot report agent state changes")
        if result in {"blocked", "noop", "conflict"} or idempotent:
            if transition["agent_changed_objects"]:
                raise ValueError("non-mutating tool reported agent state changes")
        if (
            action_mode == "write" and result == "ok" and not idempotent
            and not transition["agent_changed_objects"]
        ):
            raise ValueError("successful write did not change state hash")
        if (
            idempotent and result != "noop"
            and action.idempotency_key is None
        ):
            raise ValueError("idempotent replay requires an idempotency key")
    else:
        if tool_result is not None or not isinstance(terminal_result, dict) or set(terminal_result) != {
            "terminal", "success", "safety_violations", "budget_violations",
        }:
            raise ValueError("stop trace requires one compact terminal result")
        if terminal_result.get("terminal") is not True:
            raise ValueError("stop terminal result must be terminal")
        if (
            not isinstance(terminal_result.get("success"), bool)
            or not _nonnegative_int(terminal_result.get("safety_violations"))
            or not _nonnegative_int(terminal_result.get("budget_violations"))
        ):
            raise ValueError("invalid compact terminal result field types")
        if terminal_result["success"] and (
            terminal_result["safety_violations"]
            or terminal_result["budget_violations"]
        ):
            raise ValueError("successful terminal result cannot contain violations")
    if observation["terminal"]:
        raise ValueError("pre-action observation cannot already be terminal")


def validate_public_trace(rows: list[dict[str, Any]]) -> None:
    """Structurally validate public fields/deltas; this does not certify tool truth."""

    if not isinstance(rows, list) or not rows:
        raise ValueError("public trace must be a nonempty row list")
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("public trace rows must be objects")
        validate_public_trace_row(row)
    session_id, family = rows[0]["session_id"], rows[0]["family"]
    task = rows[0]["observation"]["task"]
    first_observation = rows[0]["observation"]
    if (
        first_observation["steps"] != 0
        or first_observation["tool_calls_remaining"] != task["budget"]["tool_calls"]
        or first_observation["logical_latency_remaining"]
        != task["budget"]["logical_latency"]
        or first_observation["irreversible_risk_remaining"]
        != task["budget"]["irreversible_risk"]
    ):
        raise ValueError("public trace must begin at the episode budget origin")
    initial_runtime = first_observation["runtime"]
    if (
        initial_runtime["idempotency_records"]
        or initial_runtime["attempted_policy_violations"]
        or initial_runtime["executed_safety_violations"]
        or initial_runtime["budget_violations"]
        or initial_runtime["budget_exhausted"]
        or initial_runtime["logical_tick"]
        or initial_runtime["logical_latency_used"]
        or initial_runtime["irreversible_risk_used"]
        or initial_runtime["public_external_event_count"]
        or initial_runtime["actor_event_counts"] != {
            "agent": 0, "system": 0, "authority": 0,
        }
        or initial_runtime["state_changing_sequence"]
        or initial_runtime["ever_changed_paths"]
        or initial_runtime["event_chain_valid"] is not True
    ):
        raise ValueError("public trace must begin with empty runtime state")
    if any(
        row["session_id"] != session_id
        or row["family"] != family
        or row["turn_id"] != turn_id
        or row["observation"]["task"] != task
        for turn_id, row in enumerate(rows)
    ):
        raise ValueError("public trace identity, task, or turn sequence is discontinuous")
    stop_positions = [
        index for index, row in enumerate(rows)
        if parse_model_action(row["action_content"], family=family).action == "stop"
    ]
    if stop_positions != [len(rows) - 1]:
        raise ValueError("public trace requires exactly one final stop")
    idempotency_ledger: dict[str, tuple[str, str, str | None, Any]] = {}
    for current, following in zip(rows, rows[1:]):
        if current["tool_result"]["state_hash"] != following["observation"]["state_hash"]:
            raise ValueError("public trace state hash chain is discontinuous")
        if current["tool_result"]["runtime_hash"] != following["observation"]["runtime_hash"]:
            raise ValueError("public trace runtime hash chain is discontinuous")
        current_action = parse_model_action(current["action_content"], family=family)
        if current_action.action != "call_tool":
            raise ValueError("only the final public trace row may stop")
        tool_result = current["tool_result"]
        key = current_action.idempotency_key
        fingerprint = hashlib.sha256(_canonical([
            current_action.tool_name, current_action.arguments,
        ]).encode()).hexdigest()
        budget_rejected = (
            tool_result["result"] == "blocked"
            and tool_result["error_code"] == "BUDGET_EXHAUSTED"
        )
        if key is not None and not budget_rejected:
            previous = idempotency_ledger.get(key)
            if previous is not None:
                (
                    previous_fingerprint, previous_result,
                    previous_error, previous_payload,
                ) = previous
                if previous_fingerprint == fingerprint:
                    if (
                        tool_result["result"] != previous_result
                        or tool_result["error_code"] != previous_error
                        or tool_result["payload"] != previous_payload
                        or not tool_result["idempotent_noop"]
                    ):
                        raise ValueError(
                            "exact idempotency replay changed its public result"
                        )
                elif not (
                    tool_result["result"] == "blocked"
                    and tool_result["error_code"] == "IDEMPOTENCY_KEY_REUSE"
                    and tool_result["payload"] is None
                    and not tool_result["idempotent_noop"]
                ):
                    raise ValueError(
                        "idempotency key reuse did not produce the required rejection"
                    )
            elif tool_result["error_code"] != "IDEMPOTENCY_KEY_REUSE":
                if (
                    tool_result["idempotent_noop"]
                    and tool_result["result"] != "noop"
                ):
                    raise ValueError(
                        "fresh idempotency key cannot report an exact replay"
                    )
                idempotency_ledger[key] = (
                    fingerprint, tool_result["result"], tool_result["error_code"],
                    copy.deepcopy(tool_result["payload"]),
                )
            else:
                raise ValueError(
                    "fresh idempotency key cannot report key reuse"
                )
        expected_runtime = _expected_runtime_after_tool(
            current["observation"], following["observation"]["world"],
            current_action, current["tool_result"],
        )
        if following["observation"]["runtime"] != expected_runtime:
            raise ValueError("public trace runtime transition is inconsistent with action")
        current_observation = current["observation"]
        following_observation = following["observation"]
        remaining = current_observation["tool_calls_remaining"]
        max_steps = task["budget"]["max_steps"]
        if budget_rejected:
            action_profile = tool_profile(family)["tools"][
                current_action.tool_name
            ]
            exhausted_by_latency = (
                current_observation["logical_latency_remaining"]
                < action_profile["logical_latency_cost"]
            )
            exhausted_by_risk = (
                current_observation["irreversible_risk_remaining"]
                < action_profile["irreversible_risk_cost"]
            )
            if (
                remaining > 0 and current_observation["steps"] < max_steps
                and not exhausted_by_latency and not exhausted_by_risk
            ):
                raise ValueError("budget rejection occurred before budget exhaustion")
            expected_counters = (current_observation["steps"], 0)
            if remaining > 0:
                expected_counters = (current_observation["steps"], remaining)
            next_action = parse_model_action(
                following["action_content"], family=family,
            )
            if next_action.action != "stop":
                raise ValueError("budget rejection must be followed by final stop")
        else:
            if remaining == 0 or current_observation["steps"] >= max_steps:
                raise ValueError("exhausted trace executed another tool")
            expected_counters = (
                current_observation["steps"] + 1,
                remaining - 1,
            )
        if (
            following_observation["steps"],
            following_observation["tool_calls_remaining"],
        ) != expected_counters:
            raise ValueError("public trace counter sequence is discontinuous")
        transition = current["tool_result"]["transition"]
        expected_latency_remaining = (
            current_observation["logical_latency_remaining"]
            - transition["logical_latency_cost"]
        )
        expected_risk_remaining = (
            current_observation["irreversible_risk_remaining"]
            - transition["irreversible_risk_cost"]
        )
        if (
            following_observation["logical_latency_remaining"]
            != expected_latency_remaining
            or following_observation["irreversible_risk_remaining"]
            != expected_risk_remaining
            or expected_latency_remaining < 0 or expected_risk_remaining < 0
        ):
            raise ValueError("public vector budget sequence is discontinuous")


def run_scripted_model_actions(
    scenario: StatefulScenario, action_contents: list[str], *,
    env_factory: Callable[[StatefulScenario], MultiTownStatefulOpsEnv]
    = MultiTownStatefulOpsEnv,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Model-side runner for parser/trace tests; no callable private evaluator is exposed."""

    if not action_contents:
        raise ValueError("scripted model actions must include stop")
    parsed = [
        parse_model_action(content, family=scenario.public_task.family)
        for content in action_contents
    ]
    stop_positions = [index for index, action in enumerate(parsed) if action.action == "stop"]
    if stop_positions != [len(parsed) - 1]:
        raise ValueError("scripted model actions require exactly one final stop")
    # Validate full JSON serializability and action syntax before changing state.
    _canonical(action_contents)
    session = PolicySession(scenario, env_factory=env_factory)
    rows = []
    final: dict[str, Any] | None = None
    for turn_id, (content, action) in enumerate(zip(action_contents, parsed, strict=True)):
        observation = session.observation()
        tool_result = None
        if action.action == "stop":
            final = session.stop()
        else:
            assert action.tool_name is not None
            tool_result = session.call_tool(
                action.tool_name, action.arguments,
                idempotency_key=action.idempotency_key,
            )
        rows.append(public_trace_row(
            session_id=scenario.public_task.task_id, turn_id=turn_id,
            family=scenario.public_task.family, observation=observation,
            action_content=content, tool_result=tool_result,
            terminal_result=final,
        ))
    assert final is not None
    validate_public_trace(rows)
    return rows, final


def validate_trace_against_scenario(
    scenario: StatefulScenario, rows: list[dict[str, Any]], *,
    env_factory: Callable[[StatefulScenario], MultiTownStatefulOpsEnv]
    = MultiTownStatefulOpsEnv,
) -> None:
    """Privately replay and certify a public trace against its frozen scenario."""

    validate_public_trace(rows)
    expected_rows, _ = run_scripted_model_actions(
        scenario, [row["action_content"] for row in rows],
        env_factory=env_factory,
    )
    if rows != expected_rows:
        raise ValueError("public trace diverges from trusted scenario replay")


def private_trace_manifest(
    scenario: StatefulScenario, rows: list[dict[str, Any]], *,
    source_revision: str,
) -> dict[str, Any]:
    """Build a private artifact binding; never serialize this for policy input."""

    if not isinstance(source_revision, str) or not source_revision.strip():
        raise ValueError("private trace manifest requires a source revision")
    validate_trace_against_scenario(scenario, rows)
    return {
        "schema_version": PRIVATE_TRACE_MANIFEST_VERSION,
        "private_instance_id": scenario.private_instance_id,
        "public_task_id": scenario.public_task.task_id,
        "public_trace_sha256": hashlib.sha256(_canonical(rows).encode()).hexdigest(),
        "row_count": len(rows),
        "source_revision": source_revision,
    }


def validate_private_trace_artifact(
    scenario: StatefulScenario, rows: list[dict[str, Any]],
    manifest: dict[str, Any], *, expected_source_revision: str,
) -> None:
    """Validate private provenance plus exact replay for a frozen trace artifact."""

    required = {
        "schema_version", "private_instance_id", "public_task_id",
        "public_trace_sha256", "row_count", "source_revision",
    }
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise ValueError("invalid private trace manifest schema")
    if manifest.get("schema_version") != PRIVATE_TRACE_MANIFEST_VERSION:
        raise ValueError("unsupported private trace manifest version")
    if manifest.get("private_instance_id") != scenario.private_instance_id:
        raise ValueError("private trace manifest instance mismatch")
    if manifest.get("public_task_id") != scenario.public_task.task_id:
        raise ValueError("private trace manifest public task mismatch")
    if (
        not isinstance(manifest.get("row_count"), int)
        or isinstance(manifest["row_count"], bool)
        or manifest["row_count"] != len(rows)
    ):
        raise ValueError("private trace manifest row count mismatch")
    trace_hash = hashlib.sha256(_canonical(rows).encode()).hexdigest()
    if manifest.get("public_trace_sha256") != trace_hash:
        raise ValueError("private trace manifest public trace hash mismatch")
    source_revision = manifest.get("source_revision")
    if not isinstance(source_revision, str) or not source_revision.strip():
        raise ValueError("private trace manifest source revision is invalid")
    if (
        not isinstance(expected_source_revision, str)
        or not expected_source_revision.strip()
    ):
        raise ValueError("expected source revision is required")
    if source_revision != expected_source_revision:
        raise ValueError("private trace manifest source revision mismatch")
    validate_trace_against_scenario(scenario, rows)


# Explicit alias for callers that need the weaker, scenario-free contract.
validate_public_trace_structure = validate_public_trace


def scaffold_group_registry() -> dict[str, Any]:
    """Implemented train-only structural registry; readiness remains red."""

    from .stateful_behavior import audit_behavioral_catalog
    from .stateful_groups import audit_structural_catalog, variants_for_family
    from .stateful_ops import build_scenario

    groups = []
    for family in FAMILIES:
        for variant_id in variants_for_family(family):
            task = build_scenario(family, variant_id=variant_id).public_task
            groups.append({
                "scenario_group_id": task.scenario_group_id,
                "family": family,
                "structural_variant_id": variant_id,
                "stage": "train",
                "generator_id": task.generator_id,
                "template_cluster_id": task.template_cluster_id,
                "mechanism_id": task.mechanism_id,
                "composition_signature": task.composition_signature,
                "structural_signature": task.structural_signature,
            })
    catalog_audit = audit_structural_catalog()
    behavioral_audit = audit_behavioral_catalog()
    behavior_by_group = {
        (row["family"], row["variant_id"]): row["behavioral_fingerprint"]
        for row in behavioral_audit["rows"]
    }
    for row in groups:
        row["behavioral_fingerprint"] = behavior_by_group[
            (row["family"], row["structural_variant_id"])
        ]
    return {
        "schema_version": GROUP_REGISTRY_VERSION,
        "groups": groups,
        "group_count": len(groups),
        "stage_counts": {"train": len(groups)},
        "duplicate_group_ids": len({row["scenario_group_id"] for row in groups}) != len(groups),
        "descriptor_duplicate_count": len(
            catalog_audit["duplicate_descriptor_fingerprints"],
        ),
        "seed_fields_present_in_descriptors": catalog_audit["seed_fields_present"],
        "behavioral_duplicate_count": len(
            behavioral_audit["duplicate_behavioral_fingerprints"],
        ),
        "behavioral_fingerprints_seed_invariant": (
            behavioral_audit["all_seed_invariant"]
        ),
        "collection_gate": {
            "minimum_structural_groups_established": False,
            "calibration_groups_exist": False,
            "selection_groups_exist": False,
            "model_collection_allowed": False,
            "controller_rl_allowed": False,
        },
        "claim_boundary": (
            "scaffold only: sixteen train structural groups, no bank, model trace, split result, "
            "or Agentic RL evidence"
        ),
    }
