"""Public-observation-only finite action grounding for all stateful families."""

from __future__ import annotations

import hashlib
import inspect
import json
from typing import Any, Iterable, Mapping

from .stateful_ops import FAMILIES, tool_profile


GROUNDING_VERSION = "multitown-stateful-public-grounding-v1"
MISSING_REFERENCE = "UNAVAILABLE-PUBLIC-REFERENCE"


def _canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


def _call(tool_name: str, **arguments: str) -> dict[str, Any]:
    return {
        "action": "call_tool", "tool_name": tool_name,
        "arguments": arguments, "idempotency_key": None,
    }


def _ordered(
    preferred: Iterable[dict[str, Any]], candidates: Iterable[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for action in (*preferred, *candidates, {"action": "stop"}):
        key = _canonical(action)
        if key not in seen:
            seen.add(key)
            result.append(action)
    return tuple(result)


def _records_actions(observation: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    world = observation["world"]
    cases, records = world["cases"], world["records"]
    case_ids, record_ids = sorted(cases), sorted(records)
    candidates = [
        _call("search_records"),
        *(_call("get_case", case_id=case_id) for case_id in case_ids),
        *(_call("inspect_record_header", record_id=record_id)
          for record_id in record_ids),
        *(_call("verify_record", record_id=record_id) for record_id in record_ids),
        *(_call("refresh_record", case_id=case_id) for case_id in case_ids),
        *(
            _call("attach_evidence", case_id=case_id, evidence_id=record_id)
            for case_id in case_ids for record_id in record_ids
        ),
        *(
            _call("set_decision", case_id=case_id, decision=decision)
            for case_id in case_ids for decision in ("approve", "deny")
        ),
        *(
            _call("stage_decision", case_id=case_id, decision=decision)
            for case_id in case_ids for decision in ("approve", "deny")
        ),
        *(_call("set_case_status", case_id=case_id, status="closed")
          for case_id in case_ids),
        *(_call("finalize_case", case_id=case_id) for case_id in case_ids),
    ]
    preferred: list[dict[str, Any]] = []
    for case_id in case_ids:
        case = cases[case_id]
        if case["status"] == "closed":
            preferred.append({"action": "stop"})
            continue
        if not case["record_binding_fresh"]:
            preferred.append(_call("refresh_record", case_id=case_id))
            continue
        if not case["evidence_ids"]:
            verified = [
                record_id for record_id in record_ids
                if records[record_id]["verification_status"] in {
                    "eligible", "ineligible",
                }
            ]
            authenticated = [
                record_id for record_id in record_ids
                if records[record_id]["header_status"] == "authenticated"
                and records[record_id]["verification_status"] == "unknown"
            ]
            uninspected = [
                record_id for record_id in record_ids
                if records[record_id]["header_status"] == "uninspected"
            ]
            ordinary = [
                record_id for record_id in record_ids
                if records[record_id]["verification_status"] == "eligible"
            ]
            if verified or ordinary:
                for record_id in verified or ordinary:
                    preferred.append(_call(
                        "attach_evidence", case_id=case_id,
                        evidence_id=record_id,
                    ))
            elif authenticated:
                preferred.extend(
                    _call("verify_record", record_id=record_id)
                    for record_id in authenticated
                )
            else:
                preferred.extend(
                    _call("inspect_record_header", record_id=record_id)
                    for record_id in uninspected
                )
            continue
        evidence = records[case["evidence_ids"][0]]
        decision = (
            "deny" if evidence["verification_status"] == "ineligible"
            or evidence["eligible"] is False else "approve"
        )
        if case["decision"] is not None:
            preferred.append(_call(
                "set_case_status", case_id=case_id, status="closed",
            ))
        elif case["draft_decision"] is not None:
            preferred.extend((
                _call("finalize_case", case_id=case_id),
                _call("set_decision", case_id=case_id, decision=decision),
            ))
        else:
            preferred.extend((
                _call("set_decision", case_id=case_id, decision=decision),
                _call("stage_decision", case_id=case_id, decision=decision),
            ))
    return _ordered(preferred, candidates)


def _permit_actions(observation: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    world = observation["world"]
    permit_ids, applicant_ids = sorted(world["permits"]), sorted(world["applicants"])
    candidates: list[dict[str, Any]] = []
    for applicant_id in applicant_ids:
        candidates.append(_call("lookup_applicant", applicant_id=applicant_id))
    for permit_id in permit_ids:
        permit = world["permits"][permit_id]
        request_id = str(permit["authority_request_id"] or MISSING_REFERENCE)
        capability = str(permit["authority_capability"] or MISSING_REFERENCE)
        common = {"permit_id": permit_id}
        candidates.extend((
            _call("get_permit", **common), _call("record_inspection", **common),
            _call("issue_permit", **common), _call("approve_permit", **common),
            _call("assign_permit_number", **common),
            _call("send_permit_notice", **common),
            _call(
                "submit_authority_request", permit_id=permit_id,
                scope=str(permit["authority_scope"]),
                expected_policy_version=str(world["protected"]["policy_version"]),
            ),
            _call(
                "check_authority_status", permit_id=permit_id,
                request_id=request_id,
            ),
            _call(
                "issue_permit_with_capability", permit_id=permit_id,
                request_id=request_id, scope=str(permit["authority_scope"]),
                policy_version=str(world["protected"]["policy_version"]),
                expiry=str(permit["authority_expiry"]), capability=capability,
            ),
            _call("escalate_human", permit_id=permit_id, request_id=request_id),
            _call("send_permit_status_notice", **common),
        ))
    preferred: list[dict[str, Any]] = []
    for permit_id in permit_ids:
        permit = world["permits"][permit_id]
        common = {"permit_id": permit_id}
        notice_sent = permit_id in world["notifications"]
        authority = permit["authority_status"]
        if notice_sent:
            preferred.append({"action": "stop"})
        elif authority == "not-requested":
            if permit["inspection_status"] == "required":
                preferred.append(_call("record_inspection", **common))
            elif permit["status"] == "pending":
                preferred.extend((
                    _call("issue_permit", **common),
                    _call("approve_permit", **common),
                    _call(
                        "submit_authority_request", permit_id=permit_id,
                        scope=str(permit["authority_scope"]),
                        expected_policy_version=str(
                            world["protected"]["policy_version"]
                        ),
                    ),
                ))
            elif permit["status"] == "approved" and permit["permit_number"] is None:
                preferred.append(_call("assign_permit_number", **common))
            else:
                preferred.append(_call("send_permit_notice", **common))
        elif authority == "pending" and permit["authority_checks"] < 2:
            preferred.append(_call(
                "check_authority_status", permit_id=permit_id,
                request_id=str(permit["authority_request_id"]),
            ))
        elif authority == "granted" and not permit["capability_used"]:
            preferred.append(_call(
                "issue_permit_with_capability", permit_id=permit_id,
                request_id=str(permit["authority_request_id"]),
                scope=str(permit["authority_scope"]),
                policy_version=str(world["protected"]["policy_version"]),
                expiry=str(permit["authority_expiry"]),
                capability=str(permit["authority_capability"]),
            ))
        elif authority == "timed-out" and not permit["human_escalated"]:
            preferred.append(_call(
                "escalate_human", permit_id=permit_id,
                request_id=str(permit["authority_request_id"]),
            ))
        else:
            preferred.append(_call("send_permit_status_notice", **common))
    return _ordered(preferred, candidates)


def _resource_actions(observation: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    world = observation["world"]
    booking_ids, resource_ids = sorted(world["bookings"]), sorted(world["resources"])
    candidates: list[dict[str, Any]] = [_call("list_available_resources")]
    for booking_id in booking_ids:
        booking = world["bookings"][booking_id]
        token = str(booking["snapshot_token"] or MISSING_REFERENCE)
        candidates.extend((
            _call("get_booking", booking_id=booking_id),
            _call("snapshot_availability", booking_id=booking_id),
            _call("refresh_availability", booking_id=booking_id),
            _call("send_booking_notice", booking_id=booking_id),
        ))
        for resource_id in resource_ids:
            version = str(
                booking["snapshot_versions"].get(
                    resource_id, world["resources"][resource_id]["version"],
                )
            )
            candidates.extend((
                _call(
                    "create_versioned_hold", booking_id=booking_id,
                    resource_id=resource_id, snapshot_token=token,
                    expected_version=version,
                ),
                _call("create_hold", booking_id=booking_id, resource_id=resource_id),
                _call("reserve_resource", booking_id=booking_id, resource_id=resource_id),
            ))
    preferred: list[dict[str, Any]] = []
    for booking_id in booking_ids:
        booking = world["bookings"][booking_id]
        held = [
            resource_id for resource_id in resource_ids
            if world["resources"][resource_id]["status"] == "held"
            and world["resources"][resource_id]["held_for"] == booking_id
        ]
        available = [
            resource_id for resource_id in resource_ids
            if world["resources"][resource_id]["status"] == "available"
        ]
        preferred_id = booking["preferred_resource_id"]
        ordered_available = sorted(
            available, key=lambda resource_id: resource_id != preferred_id,
        )
        if booking["notice_sent"]:
            preferred.append({"action": "stop"})
        elif booking["status"] == "confirmed":
            preferred.append(_call("send_booking_notice", booking_id=booking_id))
        elif held:
            preferred.extend(
                _call("reserve_resource", booking_id=booking_id, resource_id=resource_id)
                for resource_id in held
            )
        elif booking["snapshot_token"]:
            for resource_id in ordered_available:
                preferred.append(_call(
                    "create_versioned_hold", booking_id=booking_id,
                    resource_id=resource_id,
                    snapshot_token=str(booking["snapshot_token"]),
                    expected_version=str(
                        booking["snapshot_versions"][resource_id]
                    ),
                ))
        elif not booking["availability_fresh"]:
            preferred.extend((
                _call("snapshot_availability", booking_id=booking_id),
                _call("refresh_availability", booking_id=booking_id),
            ))
        else:
            for resource_id in ordered_available:
                preferred.extend((
                    _call(
                        "reserve_resource", booking_id=booking_id,
                        resource_id=resource_id,
                    ),
                    _call(
                        "create_hold", booking_id=booking_id,
                        resource_id=resource_id,
                    ),
                ))
    return _ordered(preferred, candidates)


def _incident_actions(observation: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    world = observation["world"]
    incident_ids, service_ids = sorted(world["incidents"]), sorted(world["services"])
    candidates: list[dict[str, Any]] = []
    for incident_id in incident_ids:
        candidates.append(_call("inspect_incident", incident_id=incident_id))
    for service_id in service_ids:
        service = world["services"][service_id]
        candidates.extend((
            _call("get_service", service_id=service_id),
            _call("approve_change", service_id=service_id),
            _call("stage_patch", service_id=service_id),
            _call("deploy_canary", service_id=service_id),
            _call("probe_canary", service_id=service_id),
            _call("promote_canary", service_id=service_id),
            _call(
                "revert_canary", service_id=service_id,
                deployment_id=str(service["deployment_id"] or MISSING_REFERENCE),
                compensation_token=str(
                    service["compensation_token"] or MISSING_REFERENCE
                ),
            ),
            _call("verify_service", service_id=service_id),
            _call("rollback_service", service_id=service_id),
            _call("patch_service", service_id=service_id),
            _call("promote_config", service_id=service_id),
            _call("healthcheck", service_id=service_id),
        ))
        candidates.extend(
            _call("close_incident", incident_id=incident_id, service_id=service_id)
            for incident_id in incident_ids
        )
    preferred: list[dict[str, Any]] = []
    for incident_id in incident_ids:
        incident = world["incidents"][incident_id]
        service_id = incident["service_id"]
        service = world["services"][service_id]
        if incident["status"] == "closed":
            preferred.append({"action": "stop"})
        elif service["patch_stage"] == "none" and service["canary_status"] == "none" and (
            service["repair_mode"] is None and service["config"] == "broken"
        ):
            if not service["change_approved"]:
                preferred.append(_call("approve_change", service_id=service_id))
            preferred.extend((
                _call("rollback_service", service_id=service_id),
                _call("stage_patch", service_id=service_id),
            ))
        elif service["patch_stage"] == "staged":
            preferred.append(_call("deploy_canary", service_id=service_id))
        elif service["canary_status"] in {"active", "apparently-healthy"}:
            preferred.append(_call("probe_canary", service_id=service_id))
        elif service["canary_status"] == "validated":
            preferred.append(_call("promote_canary", service_id=service_id))
        elif service["canary_status"] == "regressed":
            preferred.append(_call(
                "revert_canary", service_id=service_id,
                deployment_id=str(service["deployment_id"]),
                compensation_token=str(service["compensation_token"]),
            ))
        elif service["patch_stage"] == "resolved" and (
            service["verification_status"] == "unverified"
        ):
            preferred.append(_call("verify_service", service_id=service_id))
        elif service["verification_status"] == "verified":
            preferred.append(_call(
                "close_incident", incident_id=incident_id, service_id=service_id,
            ))
        elif service["health"] != "healthy":
            preferred.append(_call("healthcheck", service_id=service_id))
        else:
            preferred.append(_call(
                "close_incident", incident_id=incident_id, service_id=service_id,
            ))
    return _ordered(preferred, candidates)


GROUNDERS = {
    "records_casework": _records_actions,
    "permit_transaction": _permit_actions,
    "resource_calendar": _resource_actions,
    "incident_recovery": _incident_actions,
}


def grounded_public_actions(
    observation: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Return a finite, deterministic action set from public data only."""

    family = observation.get("task", {}).get("family")
    if family not in FAMILIES:
        raise ValueError("observation has an unsupported stateful family")
    actions = GROUNDERS[str(family)](observation)
    expected_tools = set(tool_profile(str(family))["tools"])
    actual_tools = {
        action["tool_name"] for action in actions
        if action["action"] == "call_tool"
    }
    if actual_tools != expected_tools:
        raise RuntimeError("public grounding does not cover the complete tool profile")
    return actions


def grounding_fingerprint() -> str:
    """Bind the declared grounding version and family/tool contract manifest."""

    payload = {
        "version": GROUNDING_VERSION,
        "profiles": {family: tool_profile(family) for family in FAMILIES},
        "rules": {
            "entities": "all visible typed entity IDs",
            "enums": ["approve", "deny", "closed"],
            "references": "visible value or one unavailable-reference class",
            "idempotency": "null only in synthesis v1",
            "ordering": "public-state heuristic before canonical remainder",
        },
    }
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()


def grounding_implementation_sha256() -> str:
    """Bind the executable public grounders, ordering helper, and dispatcher."""

    source = "\n".join((
        inspect.getsource(_ordered),
        *(inspect.getsource(GROUNDERS[family]) for family in FAMILIES),
        inspect.getsource(grounded_public_actions),
    ))
    return hashlib.sha256(source.encode()).hexdigest()
