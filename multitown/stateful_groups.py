"""Seed-independent structural descriptors for the train-only A15 generator."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any


DESCRIPTOR_VERSION = "multitown-stateful-structural-descriptor-v1"
GENERATOR_COMPAT_VERSION = "multitown-stateful-ops-generator-v14"
FORBIDDEN_SPLIT_SEED_PATTERN = re.compile(
    r"world_seed|surface_seed|split_stage|(?:^|[^a-z])(?:train|calibration|selection|heldout)(?:[^a-z]|$)",
    re.IGNORECASE,
)
UNORDERED_DESCRIPTOR_FIELDS = {
    "transition_nodes", "dependency_edges", "branch_guards",
    "authority_transitions", "required_terminal_outcomes", "protected_invariants",
}


def _common(
    family: str, mechanism: str, *, nodes: tuple[str, ...],
    edges: tuple[str, ...], guards: tuple[str, ...],
    paths: tuple[tuple[str, ...], ...], outcomes: tuple[str, ...],
    invariants: tuple[str, ...], authority: tuple[str, ...] = ("unchanged",),
    failure_schedule: str = "none", unique_process: bool = False,
    budget_regime: str = (
        "tool_calls=10;max_steps=12;logical_latency=10;"
        "irreversible_risk=0;all declared paths feasible"
    ),
) -> dict[str, Any]:
    return {
        "schema_version": DESCRIPTOR_VERSION,
        "family": family,
        "mechanism": mechanism,
        "transition_nodes": nodes,
        "dependency_edges": edges,
        "branch_guards": guards,
        "authority_transitions": authority,
        "failure_schedule": failure_schedule,
        "required_terminal_outcomes": outcomes,
        "protected_invariants": invariants,
        "accepted_action_paths": paths,
        "unique_process_required": unique_process,
        "budget_regime": budget_regime,
    }


STRUCTURAL_BLUEPRINTS: dict[str, dict[str, dict[str, Any]]] = {
    "records_casework": {
        "direct_or_staged": {
            "instruction_schema": "eligible evidence -> commit decision -> close",
            **_common(
                "records_casework", "evidence-decision-close:direct-or-staged:v1",
                nodes=("open", "evidence_attached", "decision_or_draft", "closed"),
                edges=(
                    "attach_evidence:open->evidence_attached",
                    "set_decision:evidence_attached->decision_committed",
                    "stage_decision:evidence_attached->draft_committed",
                    "finalize_case:draft_committed->closed",
                    "set_case_status:decision_committed->closed",
                ),
                guards=("record.case_id==TARGET_CASE", "record.eligible==true"),
                paths=(
                    ("attach_evidence", "set_decision", "set_case_status"),
                    ("attach_evidence", "stage_decision", "finalize_case"),
                ),
                outcomes=("case.closed", "decision.approve", "evidence.current"),
                invariants=("unrelated_rows_unchanged", "draft_cleared"),
            ),
        },
        "mandatory_stage": {
            "instruction_schema": "eligible evidence -> mandatory stage -> commit -> close",
            **_common(
                "records_casework", "evidence-decision-close:mandatory-stage:v1",
                nodes=(
                    "open", "evidence_attached", "draft_committed",
                    "decision_committed", "closed",
                ),
                edges=(
                    "attach_evidence:open->evidence_attached",
                    "stage_decision:evidence_attached->draft_committed",
                    "set_decision:draft_committed->decision_committed",
                    "finalize_case:draft_committed->closed",
                    "set_case_status:decision_committed->closed",
                ),
                guards=(
                    "record.case_id==TARGET_CASE", "record.eligible==true",
                    "set_decision.requires_matching_draft",
                ),
                authority=("clerk:evidence->draft", "reviewer:draft->final"),
                paths=(
                    ("attach_evidence", "stage_decision", "finalize_case"),
                    (
                        "attach_evidence", "stage_decision", "set_decision",
                        "set_case_status",
                    ),
                ),
                outcomes=("case.closed", "decision.approve", "evidence.current"),
                invariants=("unrelated_rows_unchanged", "draft_cleared"),
            ),
        },
        "refresh_before_evidence": {
            "instruction_schema": "refresh record binding -> evidence -> decision -> close",
            **_common(
                "records_casework", "evidence-decision-close:refresh-before-evidence:v1",
                nodes=("stale", "refreshed", "evidence_attached", "decided", "closed"),
                edges=(
                    "refresh_record:stale->refreshed",
                    "attach_evidence:refreshed->evidence_attached",
                    "set_decision:evidence_attached->decided",
                    "stage_decision:evidence_attached->draft_committed",
                    "set_case_status:decided->closed",
                    "finalize_case:draft_committed->closed",
                ),
                guards=("record_binding.refreshed", "record.eligible==true"),
                failure_schedule="initial record binding is stale until explicit refresh",
                paths=(
                    ("refresh_record", "attach_evidence", "set_decision", "set_case_status"),
                    ("refresh_record", "attach_evidence", "stage_decision", "finalize_case"),
                ),
                outcomes=("record_binding.refreshed", "case.closed", "decision.approve"),
                invariants=("unrelated_rows_unchanged", "draft_cleared"),
            ),
        },
        "conflicting_evidence_investigation": {
            "instruction_schema": (
                "inspect conflicting candidates -> verify authority and eligibility -> "
                "attach -> branch decision -> close"
            ),
            **_common(
                "records_casework",
                "evidence-decision-close:conflicting-evidence-investigation:v1",
                nodes=(
                    "uninspected_candidates", "authority_identified",
                    "eligibility_verified", "evidence_attached", "decided", "closed",
                ),
                edges=(
                    "inspect_record_header:uninspected->authenticated_or_superseded",
                    "verify_record:authenticated->eligibility_verified",
                    "attach_evidence:eligibility_verified->evidence_attached",
                    "set_decision:evidence_attached->decided",
                    "set_case_status:decided->closed",
                ),
                guards=(
                    "header.signature_identifies_current_authority",
                    "verified.eligibility_controls_approve_or_deny",
                    "only_authenticated_record_attachable",
                ),
                failure_schedule=(
                    "paired hidden worlds vary authoritative candidate and eligibility; "
                    "initial public observations are identical"
                ),
                paths=(
                    (
                        "inspect_record_header", "verify_record", "attach_evidence",
                        "set_decision", "set_case_status",
                    ),
                    (
                        "inspect_record_header", "inspect_record_header",
                        "verify_record", "attach_evidence", "set_decision",
                        "set_case_status",
                    ),
                ),
                outcomes=(
                    "case.closed", "evidence.authoritative",
                    "decision.matches_verified_eligibility",
                ),
                invariants=(
                    "non_authoritative_record_not_attached",
                    "unrelated_rows_unchanged", "draft_cleared",
                ),
            ),
        },
    },
    "permit_transaction": {
        "atomic_or_staged": {
            "instruction_schema": "eligibility and payment -> issue -> notify",
            **_common(
                "permit_transaction", "eligibility-issue-notify:atomic-or-staged:v1",
                nodes=("pending", "approved", "numbered", "notified"),
                edges=(
                    "issue_permit:pending->numbered",
                    "approve_permit:pending->approved",
                    "assign_permit_number:approved->numbered",
                    "send_permit_notice:numbered->notified",
                ),
                guards=("applicant.eligible", "applicant.fee_paid"),
                paths=(
                    ("issue_permit", "send_permit_notice"),
                    ("approve_permit", "assign_permit_number", "send_permit_notice"),
                ),
                outcomes=("permit.approved", "permit.numbered", "notice.persisted"),
                invariants=("quota_unchanged", "policy_version_unchanged"),
            ),
        },
        "approval_before_number": {
            "instruction_schema": "mandatory approval -> number allocation -> notify",
            **_common(
                "permit_transaction", "eligibility-issue-notify:approval-before-number:v1",
                nodes=("pending", "approved", "numbered", "notified"),
                edges=(
                    "approve_permit:pending->approved",
                    "assign_permit_number:approved->numbered",
                    "issue_permit:approved->numbered",
                    "send_permit_notice:numbered->notified",
                ),
                guards=(
                    "applicant.eligible", "applicant.fee_paid",
                    "number_allocation.requires_approved",
                ),
                authority=("reviewer:pending->approved", "issuer:approved->numbered"),
                paths=(
                    ("approve_permit", "assign_permit_number", "send_permit_notice"),
                    ("approve_permit", "issue_permit", "send_permit_notice"),
                ),
                outcomes=("permit.approved", "permit.numbered", "notice.persisted"),
                invariants=("quota_unchanged", "policy_version_unchanged"),
            ),
        },
        "inspection_required": {
            "instruction_schema": "inspection -> approval/issue -> number -> notify",
            **_common(
                "permit_transaction", "eligibility-issue-notify:inspection-required:v1",
                nodes=("pending", "inspected", "approved", "numbered", "notified"),
                edges=(
                    "record_inspection:pending->inspected",
                    "issue_permit:inspected->numbered",
                    "approve_permit:inspected->approved",
                    "assign_permit_number:approved->numbered",
                    "send_permit_notice:numbered->notified",
                ),
                guards=("inspection.passed", "applicant.eligible", "applicant.fee_paid"),
                authority=("inspector:pending->inspected", "issuer:inspected->numbered"),
                paths=(
                    ("record_inspection", "issue_permit", "send_permit_notice"),
                    (
                        "record_inspection", "approve_permit", "assign_permit_number",
                        "send_permit_notice",
                    ),
                ),
                outcomes=("inspection.passed", "permit.numbered", "notice.persisted"),
                invariants=("quota_unchanged", "policy_version_unchanged"),
            ),
        },
        "asynchronous_authority_timeout": {
            "instruction_schema": (
                "request authority -> observe pending -> branch on grant, denial, "
                "or timeout -> issue, abstain, or escalate -> notify"
            ),
            **_common(
                "permit_transaction", "permit-authority:asynchronous-timeout:v1",
                nodes=(
                    "not_requested", "pending", "pending_observed",
                    "granted_or_denied_or_timed_out", "issued_or_abstained_or_escalated",
                    "notified",
                ),
                edges=(
                    "submit_authority_request:not_requested->pending",
                    "check_authority_status:first->pending_observed",
                    "authority_event:pending_observed->granted_or_denied_or_timed_out",
                    "issue_permit_with_capability:granted->issued",
                    "send_permit_status_notice:denied->abstained_notified",
                    "escalate_human:timed_out->escalated",
                    "send_permit_status_notice:issued_or_escalated->notified",
                ),
                guards=(
                    "issuance.requires_authority_grant",
                    "capability.binds_request_permit_scope_policy_and_expiry",
                    "denial.requires_no_irreversible_issue",
                    "human_escalation.requires_timeout",
                ),
                authority=(
                    "authority:not_requested->pending",
                    "authority:pending->grant_or_deny_or_timeout",
                    "human:timeout->escalated",
                ),
                failure_schedule=(
                    "paired hidden worlds share request and first pending check; the "
                    "second check reveals grant, explicit denial, or timeout"
                ),
                paths=(
                    (
                        "submit_authority_request", "check_authority_status",
                        "check_authority_status", "issue_permit_with_capability",
                        "send_permit_status_notice",
                    ),
                    (
                        "submit_authority_request", "check_authority_status",
                        "check_authority_status", "send_permit_status_notice",
                    ),
                    (
                        "submit_authority_request", "check_authority_status",
                        "check_authority_status", "escalate_human",
                        "send_permit_status_notice",
                    ),
                ),
                outcomes=(
                    "grant.issued", "denial.abstained", "timeout.human_escalated",
                    "status_notice.persisted",
                ),
                invariants=(
                    "no_issue_without_live_scope_bound_grant",
                    "capability_single_use", "quota_unchanged",
                    "policy_version_unchanged",
                ),
                budget_regime=(
                    "tool_calls=6;max_steps=7;logical_latency=8;"
                    "irreversible_risk=2;one failed replay remains auditable"
                ),
            ),
        },
    },
    "resource_calendar": {
        "direct_or_hold": {
            "instruction_schema": "select eligible capacity -> reserve -> notify",
            **_common(
                "resource_calendar", "capability-reserve-notify:direct-or-hold:v1",
                nodes=("pending", "available", "held", "booked", "notified"),
                edges=(
                    "create_hold:available->held",
                    "reserve_resource:available_or_held->booked",
                    "send_booking_notice:booked->notified",
                ),
                guards=("resource.capability==inspector", "resource.not_busy"),
                paths=(
                    ("reserve_resource", "send_booking_notice"),
                    ("create_hold", "reserve_resource", "send_booking_notice"),
                ),
                outcomes=("booking.confirmed", "resource.booked", "notice.sent"),
                invariants=("no_orphan_hold", "unrelated_booking_unchanged"),
            ),
        },
        "mandatory_hold": {
            "instruction_schema": "select eligible capacity -> required hold -> reserve -> notify",
            **_common(
                "resource_calendar", "capability-reserve-notify:mandatory-hold:v1",
                nodes=("pending", "available", "held", "booked", "notified"),
                edges=(
                    "create_hold:available->held",
                    "reserve_resource:held_for_target->booked",
                    "send_booking_notice:booked->notified",
                ),
                guards=(
                    "resource.capability==inspector", "resource.status==held",
                    "resource.held_for==TARGET_BOOKING",
                ),
                authority=("scheduler:available->held", "committer:held->booked"),
                paths=(("create_hold", "reserve_resource", "send_booking_notice"),),
                outcomes=("booking.confirmed", "resource.booked", "notice.sent"),
                invariants=("no_orphan_hold", "unrelated_booking_unchanged"),
                unique_process=True,
            ),
        },
        "refresh_before_reserve": {
            "instruction_schema": "refresh capacity -> reserve or hold -> notify",
            **_common(
                "resource_calendar", "capability-reserve-notify:refresh-before-reserve:v1",
                nodes=("stale", "refreshed", "held", "booked", "notified"),
                edges=(
                    "refresh_availability:stale->refreshed",
                    "create_hold:refreshed->held",
                    "reserve_resource:refreshed_or_held->booked",
                    "send_booking_notice:booked->notified",
                ),
                guards=("availability.refreshed", "resource.capability==inspector"),
                failure_schedule="initial availability snapshot is stale until refresh",
                paths=(
                    ("refresh_availability", "reserve_resource", "send_booking_notice"),
                    (
                        "refresh_availability", "create_hold", "reserve_resource",
                        "send_booking_notice",
                    ),
                ),
                outcomes=("availability.refreshed", "booking.confirmed", "notice.sent"),
                invariants=("no_orphan_hold", "unrelated_booking_unchanged"),
            ),
        },
        "optimistic_conflict_replan": {
            "instruction_schema": (
                "snapshot capacity -> versioned hold -> recover from competing write -> "
                "reserve -> notify"
            ),
            **_common(
                "resource_calendar", "capability-reserve-notify:optimistic-conflict-replan:v1",
                nodes=(
                    "unsnapshotted", "snapshot_v1", "competing_write_or_control",
                    "version_conflict", "alternate_or_preferred_held", "booked", "notified",
                ),
                edges=(
                    "snapshot_availability:unsnapshotted->snapshot_v1",
                    "system_write:preferred_v1->preferred_busy_v2",
                    "create_versioned_hold:matching_snapshot->held",
                    "create_versioned_hold:stale_snapshot->version_conflict",
                    "create_versioned_hold:conflict->alternate_held",
                    "reserve_resource:held->booked",
                    "send_booking_notice:booked->notified",
                ),
                guards=(
                    "snapshot.required_before_hold",
                    "expected_version.matches_snapshot",
                    "control_world.retains_preferred",
                    "conflict_world.replans_to_alternate",
                ),
                failure_schedule=(
                    "paired hidden worlds either apply or omit a deterministic competing "
                    "write before the post-snapshot action"
                ),
                paths=(
                    (
                        "snapshot_availability", "create_versioned_hold",
                        "reserve_resource", "send_booking_notice",
                    ),
                    (
                        "snapshot_availability", "create_versioned_hold",
                        "create_versioned_hold", "reserve_resource",
                        "send_booking_notice",
                    ),
                    (
                        "snapshot_availability", "snapshot_availability",
                        "create_versioned_hold", "reserve_resource",
                        "send_booking_notice",
                    ),
                ),
                outcomes=(
                    "booking.confirmed", "notice.sent",
                    "selected_resource.matches_post_event_availability",
                ),
                invariants=(
                    "failed_cas_has_no_agent_mutation",
                    "competitor_lease_not_overwritten", "no_orphan_hold",
                    "unrelated_booking_unchanged",
                ),
                budget_regime=(
                    "tool_calls=6;max_steps=7;logical_latency=7;"
                    "irreversible_risk=0;all declared paths feasible"
                ),
            ),
        },
    },
    "incident_recovery": {
        "rollback_or_patch": {
            "instruction_schema": "repair -> two healthy probes -> close",
            **_common(
                "incident_recovery", "repair-healthcheck-close:rollback-or-patch:v1",
                nodes=("broken", "recovering", "healthy", "closed"),
                edges=(
                    "rollback_service:broken->recovering",
                    "patch_service+promote_config:broken->recovering",
                    "healthcheck*2:recovering->healthy",
                    "close_incident:healthy->closed",
                ),
                guards=("incident.open", "healthy_checks>=2"),
                paths=(
                    ("rollback_service", "healthcheck", "healthcheck", "close_incident"),
                    (
                        "patch_service", "promote_config", "healthcheck", "healthcheck",
                        "close_incident",
                    ),
                ),
                outcomes=("service.stable", "service.healthy", "incident.closed"),
                invariants=("dependency_unchanged", "change_window_unchanged"),
            ),
        },
        "transient_health_probe": {
            "instruction_schema": "repair -> absorb transient probe -> two healthy probes -> close",
            **_common(
                "incident_recovery", "repair-healthcheck-close:transient-probe:v1",
                nodes=("broken", "recovering", "transient_probe", "healthy", "closed"),
                edges=(
                    "rollback_service:broken->recovering",
                    "patch_service+promote_config:broken->recovering",
                    "healthcheck:recovering->transient_probe",
                    "healthcheck*2:transient_probe->healthy",
                    "close_incident:healthy->closed",
                ),
                guards=("incident.open", "healthy_checks>=2_after_transient"),
                failure_schedule="first post-repair healthcheck is deterministically transient",
                paths=(
                    (
                        "rollback_service", "healthcheck", "healthcheck", "healthcheck",
                        "close_incident",
                    ),
                    (
                        "patch_service", "promote_config", "healthcheck", "healthcheck",
                        "healthcheck", "close_incident",
                    ),
                ),
                outcomes=("service.stable", "service.healthy", "incident.closed"),
                invariants=("dependency_unchanged", "change_window_unchanged"),
            ),
        },
        "approval_before_repair": {
            "instruction_schema": "approve change -> repair -> verify -> close",
            **_common(
                "incident_recovery", "repair-healthcheck-close:approval-before-repair:v1",
                nodes=("unapproved", "approved", "recovering", "healthy", "closed"),
                edges=(
                    "approve_change:unapproved->approved",
                    "rollback_service:approved->recovering",
                    "patch_service+promote_config:approved->recovering",
                    "healthcheck*2:recovering->healthy",
                    "close_incident:healthy->closed",
                ),
                guards=("change.approved", "incident.open", "healthy_checks>=2"),
                authority=("approver:unapproved->approved", "operator:approved->repair"),
                paths=(
                    (
                        "approve_change", "rollback_service", "healthcheck",
                        "healthcheck", "close_incident",
                    ),
                    (
                        "approve_change", "patch_service", "promote_config",
                        "healthcheck", "healthcheck", "close_incident",
                    ),
                ),
                outcomes=("change.approved", "service.healthy", "incident.closed"),
                invariants=("dependency_unchanged", "change_window_unchanged"),
            ),
        },
        "canary_compensation_saga": {
            "instruction_schema": (
                "stage patch -> deploy canary -> observe delayed outcome -> "
                "promote or compensate -> verify -> close"
            ),
            **_common(
                "incident_recovery", "repair-canary-compensation:saga:v1",
                nodes=(
                    "broken", "patch_staged", "canary_deployed",
                    "first_probe_healthy", "validated_or_regressed",
                    "promoted_or_compensated", "verified", "closed",
                ),
                edges=(
                    "stage_patch:broken->patch_staged",
                    "deploy_canary:patch_staged->canary_deployed",
                    "probe_canary:first->apparently_healthy",
                    "probe_canary:second->validated_or_regressed",
                    "promote_canary:validated->promoted",
                    "revert_canary:regressed->compensated",
                    "verify_service:promoted_or_compensated->verified",
                    "close_incident:verified->closed",
                ),
                guards=(
                    "promotion.requires_two_healthy_canary_probes",
                    "compensation.requires_observed_regression",
                    "compensation.references_successful_deployment",
                    "compensation_token.is_single_use",
                ),
                failure_schedule=(
                    "paired hidden worlds share a healthy first canary probe; the second "
                    "probe either validates compatibility or reveals regression"
                ),
                paths=(
                    (
                        "stage_patch", "deploy_canary", "probe_canary",
                        "probe_canary", "promote_canary", "verify_service",
                        "close_incident",
                    ),
                    (
                        "stage_patch", "deploy_canary", "probe_canary",
                        "probe_canary", "revert_canary", "verify_service",
                        "close_incident",
                    ),
                ),
                outcomes=(
                    "incident.closed", "service.healthy",
                    "compatible.promoted_or_regression.compensated",
                ),
                invariants=(
                    "no_active_canary", "deployment_history_append_only",
                    "compensation_at_most_once", "dependency_unchanged",
                    "change_window_unchanged",
                ),
                budget_regime=(
                    "tool_calls=8;max_steps=9;logical_latency=10;"
                    "irreversible_risk=1;both branch paths feasible"
                ),
            ),
        },
    },
}


def variants_for_family(family: str) -> tuple[str, ...]:
    if family not in STRUCTURAL_BLUEPRINTS:
        raise ValueError(f"unsupported stateful family: {family}")
    return tuple(STRUCTURAL_BLUEPRINTS[family])


def structural_descriptor(family: str, variant_id: str) -> dict[str, Any]:
    try:
        descriptor = STRUCTURAL_BLUEPRINTS[family][variant_id]
    except KeyError as exc:
        raise ValueError(f"unsupported structural variant: {family}/{variant_id}") from exc
    return copy.deepcopy({"variant_id": variant_id, **descriptor})


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _contains_split_or_seed_token(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            FORBIDDEN_SPLIT_SEED_PATTERN.search(str(key))
            or _contains_split_or_seed_token(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_split_or_seed_token(item) for item in value)
    return isinstance(value, str) and bool(FORBIDDEN_SPLIT_SEED_PATTERN.search(value))


def normalize_structural_descriptor(descriptor: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize set-like descriptor fields while preserving path order."""

    normalized = copy.deepcopy(descriptor)
    normalized.pop("variant_id", None)
    normalized.pop("instruction_schema", None)
    normalized.pop("mechanism", None)
    normalized.pop("schema_version", None)
    if _contains_split_or_seed_token(normalized):
        raise ValueError("structural descriptor contains a split or seed token")
    for key in UNORDERED_DESCRIPTOR_FIELDS:
        value = normalized.get(key)
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"structural descriptor {key} must be a sequence")
        normalized[key] = tuple(sorted(value, key=_canonical))
    paths = normalized.get("accepted_action_paths")
    if not isinstance(paths, (list, tuple)) or any(
        not isinstance(path, (list, tuple)) for path in paths
    ):
        raise ValueError("structural descriptor accepted_action_paths must be sequences")
    normalized["accepted_action_paths"] = tuple(
        sorted((tuple(path) for path in paths), key=_canonical)
    )
    return normalized


def descriptor_fingerprint(descriptor: dict[str, Any]) -> str:
    return _sha256(normalize_structural_descriptor(descriptor))


def structural_metadata(family: str, variant_id: str) -> dict[str, str]:
    descriptor = structural_descriptor(family, variant_id)
    instruction_schema = descriptor.pop("instruction_schema")
    mechanism_id = descriptor["mechanism"]
    normalized = normalize_structural_descriptor(descriptor)
    fingerprint = _sha256(normalized)
    graph = {
        key: normalized[key]
        for key in (
            "family", "transition_nodes", "dependency_edges", "branch_guards",
            "authority_transitions", "failure_schedule", "budget_regime",
        )
    }
    composition = {
        key: normalized[key]
        for key in (
            "family", "required_terminal_outcomes", "protected_invariants",
            "accepted_action_paths", "unique_process_required",
        )
    }
    return {
        "generator_id": (
            f"multitown-{family}-{fingerprint[:12]}-train-"
            f"{GENERATOR_COMPAT_VERSION.rsplit('-', 1)[-1]}"
        ),
        "scenario_group_id": fingerprint,
        "template_cluster_id": _sha256([family, instruction_schema]),
        "mechanism_id": str(mechanism_id),
        "structural_signature": _sha256(graph),
        "composition_signature": _sha256(composition),
        "descriptor_fingerprint": fingerprint,
    }


def audit_structural_catalog() -> dict[str, Any]:
    rows = []
    for family in STRUCTURAL_BLUEPRINTS:
        for variant_id in variants_for_family(family):
            metadata = structural_metadata(family, variant_id)
            rows.append({"family": family, "variant_id": variant_id, **metadata})
    fingerprints = [row["descriptor_fingerprint"] for row in rows]
    return {
        "schema_version": DESCRIPTOR_VERSION,
        "rows": rows,
        "group_count": len(rows),
        "family_counts": {
            family: sum(row["family"] == family for row in rows)
            for family in STRUCTURAL_BLUEPRINTS
        },
        "duplicate_descriptor_fingerprints": sorted({
            fingerprint for fingerprint in fingerprints
            if fingerprints.count(fingerprint) > 1
        }),
        "seed_fields_present": any(
            _contains_split_or_seed_token(descriptor)
            for family in STRUCTURAL_BLUEPRINTS.values()
            for descriptor in family.values()
        ),
    }
