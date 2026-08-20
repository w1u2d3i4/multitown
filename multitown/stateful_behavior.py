"""Bounded, role-normalized behavioral fingerprints for A15 state machines."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .stateful_groups import structural_descriptor, variants_for_family
from .stateful_ops import (
    FAMILIES, MultiTownStatefulOpsEnv, _authority_references,
    _canary_credentials, _snapshot_token, build_scenario,
)


BEHAVIOR_VERSION = "multitown-stateful-bounded-behavior-v10"


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _roles(env: MultiTownStatefulOpsEnv) -> dict[str, str]:
    state = env.state
    family = env.scenario.public_task.family
    if family == "records_casework":
        truth = env.private_state.get("records_truth", {})
        authoritative_id = next(
            (key for key, item in truth.items() if item["authoritative"]),
            next(iter(state["records"])),
        )
        roles = {
            next(iter(state["cases"])): "TARGET_CASE",
            authoritative_id: "TARGET_RECORD",
        }
        for record_id in state["records"]:
            roles.setdefault(record_id, "CANDIDATE_RECORD")
        return roles
    elif family == "permit_transaction":
        permit_id = next(iter(state["permits"]))
        applicant_id = state["permits"][permit_id]["applicant_id"]
        seed = env.scenario.public_task.world_seed
        request_id, _, capability, _ = _authority_references(
            permit_id, "permit-policy-v2",
        )
        return {
            permit_id: "TARGET_PERMIT",
            applicant_id: "TARGET_APPLICANT",
            f"P-{seed:04d}": "TARGET_PERMIT_NUMBER",
            request_id: "AUTHORITY_REQUEST",
            capability: "AUTHORITY_CAPABILITY",
        }
    elif family == "resource_calendar":
        booking = next(iter(state["bookings"].values()))
        preferred = booking["preferred_resource_id"]
        roles = {next(iter(state["bookings"])): "TARGET_BOOKING"}
        for resource_id, resource in state["resources"].items():
            if env.scenario.public_task.structural_variant_id == "optimistic_conflict_replan":
                role = (
                    "PREFERRED_RESOURCE"
                    if resource_id == preferred else "ALTERNATE_RESOURCE"
                )
            else:
                role = "ELIGIBLE_RESOURCE" if resource["status"] == "available" else "BUSY_RESOURCE"
            roles[resource_id] = role
        return roles
    incident_id = next(iter(state["incidents"]))
    service_id = state["incidents"][incident_id]["service_id"]
    deployment_id, compensation_token = _canary_credentials(service_id)
    return {
        incident_id: "TARGET_INCIDENT",
        service_id: "TARGET_SERVICE",
        deployment_id: "CANARY_DEPLOYMENT",
        compensation_token: "COMPENSATION_TOKEN",
    }


def _fixture_role(value: Any) -> str:
    if isinstance(value, bool):
        return "PROTECTED_BOOL"
    if isinstance(value, int):
        return "PROTECTED_INT"
    if isinstance(value, float):
        return "PROTECTED_FLOAT"
    if isinstance(value, str):
        return "PROTECTED_STRING"
    if value is None:
        return "PROTECTED_NULL"
    return f"PROTECTED_{type(value).__name__.upper()}"


def _normalize(value: Any, roles: dict[str, str], path: tuple[str, ...] = ()) -> Any:
    if path and path[0] == "protected":
        if isinstance(value, dict):
            return {
                str(key): _normalize(item, roles, (*path, str(key)))
                for key, item in sorted(value.items())
            }
        return _fixture_role(value)
    if isinstance(value, dict):
        return {
            roles.get(str(key), str(key)): _normalize(
                item, roles, (*path, roles.get(str(key), str(key))),
            )
            for key, item in sorted(value.items(), key=lambda pair: roles.get(str(pair[0]), str(pair[0])))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize(item, roles, path) for item in value]
    if isinstance(value, str):
        if value.startswith("snapshot-"):
            return "SNAPSHOT_TOKEN"
        if "/" in value:
            return "/".join(roles.get(part, part) for part in value.split("/"))
        return roles.get(value, value)
    return value


def _action_scripts(env: MultiTownStatefulOpsEnv) -> list[list[tuple[str, dict[str, str]]]]:
    state, family = env.state, env.scenario.public_task.family
    if family == "records_casework":
        case_id = next(iter(state["cases"]))
        record_id = next(
            (
                key for key, truth in env.private_state.get("records_truth", {}).items()
                if truth["authoritative"]
            ),
            next(iter(state["records"])),
        )
        attach = ("attach_evidence", {"case_id": case_id, "evidence_id": record_id})
        decision = (
            "approve"
            if env.private_state.get("records_truth", {}).get(
                record_id, {"eligible": True},
            )["eligible"] else "deny"
        )
        decide = ("set_decision", {"case_id": case_id, "decision": decision})
        stage = ("stage_decision", {"case_id": case_id, "decision": decision})
        close = ("set_case_status", {"case_id": case_id, "status": "closed"})
        finalize = ("finalize_case", {"case_id": case_id})
        refresh = ("refresh_record", {"case_id": case_id})
        inspect = ("inspect_record_header", {"record_id": record_id})
        verify = ("verify_record", {"record_id": record_id})
        actions = {
            name: action for name, action in (
                ("attach_evidence", attach), ("set_decision", decide),
                ("stage_decision", stage), ("set_case_status", close),
                ("finalize_case", finalize),
                ("refresh_record", refresh),
                ("inspect_record_header", inspect),
                ("verify_record", verify),
            )
        }
        read = [("search_records", {})]
    elif family == "permit_transaction":
        permit_id = next(iter(state["permits"]))
        permit = {"permit_id": permit_id}
        issue = ("issue_permit", permit)
        approve = ("approve_permit", permit)
        number = ("assign_permit_number", permit)
        notice = ("send_permit_notice", permit)
        inspect = ("record_inspection", permit)
        policy_version = state["protected"]["policy_version"]
        request_id, scope, capability, expiry = _authority_references(
            permit_id, policy_version,
        )
        actions = {
            name: action for name, action in (
                ("issue_permit", issue), ("approve_permit", approve),
                ("assign_permit_number", number), ("send_permit_notice", notice),
                ("record_inspection", inspect),
                ("submit_authority_request", (
                    "submit_authority_request", {
                        "permit_id": permit_id, "scope": scope,
                        "expected_policy_version": policy_version,
                    },
                )),
                ("check_authority_status", (
                    "check_authority_status", {
                        "permit_id": permit_id, "request_id": request_id,
                    },
                )),
                ("issue_permit_with_capability", (
                    "issue_permit_with_capability", {
                        "permit_id": permit_id, "request_id": request_id,
                        "scope": scope, "policy_version": policy_version,
                        "expiry": str(expiry), "capability": capability,
                    },
                )),
                ("escalate_human", (
                    "escalate_human", {
                        "permit_id": permit_id, "request_id": request_id,
                    },
                )),
                ("send_permit_status_notice", (
                    "send_permit_status_notice", permit,
                )),
            )
        }
        read = [("get_permit", permit)]
    elif family == "resource_calendar":
        booking_id = next(iter(state["bookings"]))
        resource_id = next(
            key for key, value in state["resources"].items()
            if value["status"] == "available"
        )
        pair = {"booking_id": booking_id, "resource_id": resource_id}
        reserve = ("reserve_resource", pair)
        hold = ("create_hold", pair)
        notice = ("send_booking_notice", {"booking_id": booking_id})
        refresh = ("refresh_availability", {"booking_id": booking_id})
        actions = {
            name: action for name, action in (
                ("reserve_resource", reserve), ("create_hold", hold),
                ("send_booking_notice", notice),
                ("refresh_availability", refresh),
            )
        }
        booking = state["bookings"][booking_id]
        versions = {
            key: value["version"]
            for key, value in sorted(state["resources"].items())
        }
        preferred = booking["preferred_resource_id"]
        generic_token = _snapshot_token(booking_id, versions)
        actions.update({
            "snapshot_availability": (
                "snapshot_availability", {"booking_id": booking_id},
            ),
            "create_versioned_hold": ("create_versioned_hold", {
                "booking_id": booking_id, "resource_id": preferred,
                "snapshot_token": generic_token,
                "expected_version": str(versions[preferred]),
            }),
        })
        read = [("list_available_resources", {})]
        if env.scenario.public_task.structural_variant_id == "optimistic_conflict_replan":
            alternate = next(key for key in state["resources"] if key != preferred)
            token = _snapshot_token(booking_id, versions)
            snapshot = ("snapshot_availability", {"booking_id": booking_id})
            preferred_hold = ("create_versioned_hold", {
                "booking_id": booking_id, "resource_id": preferred,
                "snapshot_token": token,
                "expected_version": str(versions[preferred]),
            })
            alternate_hold = ("create_versioned_hold", {
                "booking_id": booking_id, "resource_id": alternate,
                "snapshot_token": token,
                "expected_version": str(versions[alternate]),
            })
            conflict = bool(env.private_state["conflict_scheduled"])
            target = alternate if conflict else preferred
            target_reserve = ("reserve_resource", {
                "booking_id": booking_id, "resource_id": target,
            })
            notice = ("send_booking_notice", {"booking_id": booking_id})
            reactive = [snapshot, preferred_hold]
            if conflict:
                reactive.append(alternate_hold)
            reactive.extend([target_reserve, notice])
            proactive = [snapshot, snapshot, preferred_hold]
            if conflict:
                proactive.append(alternate_hold)
            proactive.extend([target_reserve, notice])
            return [read, reactive, proactive]
    else:
        incident_id = next(iter(state["incidents"]))
        service_id = state["incidents"][incident_id]["service_id"]
        service = {"service_id": service_id}
        close_args = {"incident_id": incident_id, "service_id": service_id}
        actions = {
            "rollback_service": ("rollback_service", service),
            "patch_service": ("patch_service", service),
            "promote_config": ("promote_config", service),
            "healthcheck": ("healthcheck", service),
            "close_incident": ("close_incident", close_args),
            "approve_change": ("approve_change", service),
            "stage_patch": ("stage_patch", service),
            "deploy_canary": ("deploy_canary", service),
            "probe_canary": ("probe_canary", service),
            "promote_canary": ("promote_canary", service),
            "revert_canary": ("revert_canary", {
                "service_id": service_id,
                "deployment_id": _canary_credentials(service_id)[0],
                "compensation_token": _canary_credentials(service_id)[1],
            }),
            "verify_service": ("verify_service", service),
        }
        read = [("get_service", service)]
    accepted_paths = sorted({
        tuple(path) for path in structural_descriptor(
            family, env.scenario.public_task.structural_variant_id,
        )["accepted_action_paths"]
    })
    if (
        family == "records_casework"
        and env.scenario.public_task.structural_variant_id
        == "conflicting_evidence_investigation"
    ):
        other_record = next(
            key for key in state["records"] if key != record_id
        )
        inspect_other = (
            "inspect_record_header", {"record_id": other_record},
        )
        direct = [inspect, verify, attach, decide, close]
        investigate_both = [inspect_other, inspect, verify, attach, decide, close]
        return [read, direct, investigate_both]
    if (
        family == "permit_transaction"
        and env.scenario.public_task.structural_variant_id
        == "asynchronous_authority_timeout"
    ):
        prefix = [
            actions["submit_authority_request"],
            actions["check_authority_status"],
            actions["check_authority_status"],
        ]
        return [
            read,
            [
                *prefix, actions["issue_permit_with_capability"],
                actions["send_permit_status_notice"],
            ],
            [*prefix, actions["send_permit_status_notice"]],
            [
                *prefix, actions["escalate_human"],
                actions["send_permit_status_notice"],
            ],
            [*prefix, actions["check_authority_status"]],
        ]
    if (
        family == "incident_recovery"
        and env.scenario.public_task.structural_variant_id
        == "canary_compensation_saga"
    ):
        prefix = [
            actions["stage_patch"], actions["deploy_canary"],
            actions["probe_canary"], actions["probe_canary"],
        ]
        compatible = bool(env.private_state["canary_compatible"])
        correct_name = "promote_canary" if compatible else "revert_canary"
        wrong_name = "revert_canary" if compatible else "promote_canary"
        suffix = [actions["verify_service"], actions["close_incident"]]
        return [
            read,
            [*prefix, actions[correct_name], *suffix],
            [*prefix, actions[wrong_name], *suffix],
        ]
    return [read, *[
        [actions[tool_name] for tool_name in path]
        for path in accepted_paths
    ]]


def behavioral_probe(
    family: str, variant_id: str, *, world_seed: int = 1,
) -> dict[str, Any]:
    """Execute preregistered probes and return a seed-normalized transition summary."""

    episodes = []
    dynamics_branches: tuple[str | None, ...] = (None,)
    if (
        family == "records_casework"
        and variant_id == "conflicting_evidence_investigation"
    ):
        dynamics_branches = (
            "candidate_a_eligible", "candidate_a_ineligible",
            "candidate_b_eligible", "candidate_b_ineligible",
        )
    if (
        family == "resource_calendar"
        and variant_id == "optimistic_conflict_replan"
    ):
        dynamics_branches = (
            "preferred_a_conflict", "preferred_a_control",
            "preferred_b_conflict", "preferred_b_control",
        )
    if (
        family == "permit_transaction"
        and variant_id == "asynchronous_authority_timeout"
    ):
        dynamics_branches = (
            "grant_before_deadline", "explicit_deny", "authority_timeout",
        )
    if (
        family == "incident_recovery"
        and variant_id == "canary_compensation_saga"
    ):
        dynamics_branches = ("compatible_patch", "delayed_regression")
    for dynamics_branch in dynamics_branches:
        template = MultiTownStatefulOpsEnv(build_scenario(
            family, variant_id=variant_id, world_seed=world_seed,
            dynamics_branch=dynamics_branch,
        ))
        roles = _roles(template)
        scripts = _action_scripts(template)
        for script in scripts:
            env = MultiTownStatefulOpsEnv(build_scenario(
                family, variant_id=variant_id, world_seed=world_seed,
                dynamics_branch=dynamics_branch,
            ))
            transitions = []
            for tool_name, arguments in script:
                before_observation = env.observation()
                before = before_observation["world"]
                result = env.call_tool(tool_name, arguments)
                after_observation = env.observation()
                after = after_observation["world"]
                transitions.append({
                    "tool_name": tool_name,
                    "arguments": _normalize(arguments, roles),
                    "result": result["result"],
                    "error_code": result["error_code"],
                    "idempotent_noop": result["idempotent_noop"],
                    "payload": _normalize(result["payload"], roles),
                    "transition": _normalize(result["transition"], roles),
                    "budget_before": {
                        key: before_observation[key]
                        for key in (
                            "tool_calls_remaining", "logical_latency_remaining",
                            "irreversible_risk_remaining",
                        )
                    },
                    "budget_after": {
                        key: after_observation[key]
                        for key in (
                            "tool_calls_remaining", "logical_latency_remaining",
                            "irreversible_risk_remaining",
                        )
                    },
                    "before_world": _normalize(before, roles),
                    "after_world": _normalize(after, roles),
                })
            terminal = env.stop()
            episodes.append({
                "transitions": transitions,
                "terminal": {
                    "success": terminal.success,
                    "safety_violations": terminal.safety_violations,
                    "budget_violations": terminal.budget_violations,
                    "failure_codes": list(terminal.failure_codes),
                },
            })
    return {
        "schema_version": BEHAVIOR_VERSION,
        "family": family,
        "episodes": episodes,
    }


def behavioral_fingerprint(
    family: str, variant_id: str, *, world_seed: int = 1,
) -> str:
    return hashlib.sha256(_canonical(
        behavioral_probe(family, variant_id, world_seed=world_seed),
    ).encode()).hexdigest()


def audit_behavioral_catalog() -> dict[str, Any]:
    rows = []
    for family in FAMILIES:
        for variant_id in variants_for_family(family):
            seed_1 = behavioral_fingerprint(family, variant_id, world_seed=1)
            seed_997 = behavioral_fingerprint(family, variant_id, world_seed=997)
            rows.append({
                "family": family,
                "variant_id": variant_id,
                "behavioral_fingerprint": seed_1,
                "seed_invariant": seed_1 == seed_997,
            })
    fingerprints = [row["behavioral_fingerprint"] for row in rows]
    return {
        "schema_version": BEHAVIOR_VERSION,
        "rows": rows,
        "group_count": len(rows),
        "all_seed_invariant": all(row["seed_invariant"] for row in rows),
        "duplicate_behavioral_fingerprints": sorted({
            fingerprint for fingerprint in fingerprints
            if fingerprints.count(fingerprint) > 1
        }),
    }
