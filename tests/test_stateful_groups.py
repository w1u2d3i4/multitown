import pytest
import copy

from multitown.stateful_groups import (
    audit_structural_catalog,
    descriptor_fingerprint,
    normalize_structural_descriptor,
    structural_descriptor,
    structural_metadata,
    variants_for_family,
)
from multitown.stateful_ops import (
    CheckerDecision,
    FAMILIES,
    MultiTownStatefulOpsEnv,
    build_scenario,
    replay_events,
)


def ids(env: MultiTownStatefulOpsEnv) -> dict[str, str]:
    state, family = env.state, env.scenario.public_task.family
    if family == "records_casework":
        return {
            "case_id": next(iter(state["cases"])),
            "evidence_id": next(iter(state["records"])),
        }
    if family == "permit_transaction":
        permit_id = next(iter(state["permits"]))
        return {"permit_id": permit_id}
    if family == "resource_calendar":
        return {
            "booking_id": next(iter(state["bookings"])),
            "resource_id": next(
                key for key, value in state["resources"].items()
                if value["status"] == "available"
            ),
        }
    incident_id = next(iter(state["incidents"]))
    return {
        "incident_id": incident_id,
        "service_id": state["incidents"][incident_id]["service_id"],
    }


def run_variant_path(
    family: str, variant_id: str, *, alternative: bool = False,
) -> MultiTownStatefulOpsEnv:
    env = MultiTownStatefulOpsEnv(build_scenario(
        family, world_seed=101, variant_id=variant_id,
    ))
    entity = ids(env)
    if family == "records_casework":
        decision = "approve"
        if variant_id == "conflicting_evidence_investigation":
            truth = env.private_state["records_truth"]
            evidence_id = next(
                key for key, value in truth.items() if value["authoritative"]
            )
            if alternative:
                other_id = next(key for key in truth if key != evidence_id)
                env.call_tool("inspect_record_header", {"record_id": other_id})
            env.call_tool("inspect_record_header", {"record_id": evidence_id})
            env.call_tool("verify_record", {"record_id": evidence_id})
            entity["evidence_id"] = evidence_id
            decision = "approve" if truth[evidence_id]["eligible"] else "deny"
        if variant_id == "refresh_before_evidence":
            env.call_tool("refresh_record", {"case_id": entity["case_id"]})
        env.call_tool("attach_evidence", entity)
        if variant_id == "conflicting_evidence_investigation":
            env.call_tool("set_decision", {
                "case_id": entity["case_id"], "decision": decision,
            })
            env.call_tool("set_case_status", {
                "case_id": entity["case_id"], "status": "closed",
            })
            return env
        if alternative and variant_id == "refresh_before_evidence":
            env.call_tool("set_decision", {
                "case_id": entity["case_id"], "decision": "approve",
            })
            env.call_tool("set_case_status", {
                "case_id": entity["case_id"], "status": "closed",
            })
        else:
            env.call_tool("stage_decision", {
                "case_id": entity["case_id"], "decision": "approve",
            })
            if alternative:
                env.call_tool("set_decision", {
                    "case_id": entity["case_id"], "decision": "approve",
                })
                env.call_tool("set_case_status", {
                    "case_id": entity["case_id"], "status": "closed",
                })
            else:
                env.call_tool("finalize_case", {"case_id": entity["case_id"]})
    elif family == "permit_transaction":
        if variant_id == "inspection_required":
            env.call_tool("record_inspection", entity)
            if alternative:
                env.call_tool("issue_permit", entity)
                env.call_tool("send_permit_notice", entity)
                return env
        env.call_tool("approve_permit", entity)
        env.call_tool(
            "issue_permit"
            if alternative and variant_id == "approval_before_number"
            else "assign_permit_number",
            entity,
        )
        env.call_tool("send_permit_notice", entity)
    elif family == "resource_calendar":
        if variant_id == "refresh_before_reserve":
            env.call_tool("refresh_availability", {
                "booking_id": entity["booking_id"],
            })
            if alternative:
                env.call_tool("reserve_resource", entity)
                env.call_tool("send_booking_notice", {"booking_id": entity["booking_id"]})
                return env
        env.call_tool("create_hold", entity)
        env.call_tool("reserve_resource", entity)
        env.call_tool("send_booking_notice", {"booking_id": entity["booking_id"]})
    else:
        if variant_id == "approval_before_repair":
            env.call_tool("approve_change", {"service_id": entity["service_id"]})
        env.call_tool(
            "patch_service" if alternative else "rollback_service",
            {"service_id": entity["service_id"]},
        )
        if alternative:
            env.call_tool("promote_config", {"service_id": entity["service_id"]})
        for _ in range(3 if variant_id == "transient_health_probe" else 2):
            env.call_tool("healthcheck", {"service_id": entity["service_id"]})
        env.call_tool("close_incident", entity)
    return env


def test_catalog_has_sixteen_real_train_groups_without_seed_salted_descriptors() -> None:
    audit = audit_structural_catalog()
    assert audit["group_count"] == 16
    assert audit["family_counts"] == {
        "records_casework": 4,
        "permit_transaction": 4,
        "resource_calendar": 4,
        "incident_recovery": 4,
    }
    assert not audit["duplicate_descriptor_fingerprints"]
    assert not audit["seed_fields_present"]
    for family in FAMILIES:
        assert len(variants_for_family(family)) == 4


@pytest.mark.parametrize("family", FAMILIES)
def test_scenario_group_is_seed_invariant_and_variant_sensitive(family: str) -> None:
    first, *others = variants_for_family(family)
    group_a = build_scenario(family, world_seed=1, variant_id=first).public_task
    group_b = build_scenario(family, world_seed=999, variant_id=first).public_task
    assert group_a.scenario_group_id == group_b.scenario_group_id
    assert group_a.structural_signature == group_b.structural_signature
    assert group_a.composition_signature == group_b.composition_signature
    assert group_a.task_id != group_b.task_id
    assert group_a.structural_variant_id == first
    assert "structural_variant_id" not in group_a.to_policy_dict()
    for variant_id in others:
        variant = build_scenario(family, world_seed=1, variant_id=variant_id).public_task
        assert group_a.scenario_group_id != variant.scenario_group_id
        assert variant.structural_variant_id == variant_id


@pytest.mark.parametrize(
    ("family", "variant_id", "supports_alternative"),
    [
        ("records_casework", "mandatory_stage", True),
        ("permit_transaction", "approval_before_number", True),
        ("resource_calendar", "mandatory_hold", False),
        ("incident_recovery", "transient_health_probe", True),
        ("records_casework", "refresh_before_evidence", True),
        ("permit_transaction", "inspection_required", True),
        ("resource_calendar", "refresh_before_reserve", True),
        ("incident_recovery", "approval_before_repair", True),
        ("records_casework", "conflicting_evidence_investigation", True),
    ],
)
def test_new_variant_reference_paths_reach_valid_terminal_state(
    family: str, variant_id: str, supports_alternative: bool,
) -> None:
    choices = (False, True) if supports_alternative else (False,)
    for alternative in choices:
        env = run_variant_path(family, variant_id, alternative=alternative)
        result = env.stop()
        assert result.success, result.failure_codes
        assert not result.unexpected_mutations


def test_transient_probe_is_public_logged_state_not_hidden_counter() -> None:
    scenario = build_scenario(
        "incident_recovery", world_seed=102,
        variant_id="transient_health_probe",
    )
    env = MultiTownStatefulOpsEnv(scenario)
    entity = ids(env)
    service_id = entity["service_id"]
    env.call_tool("rollback_service", {"service_id": service_id})
    before = env.observation()
    transient = env.call_tool("healthcheck", {"service_id": service_id})
    after = env.observation()
    assert transient["result"] == "ok"
    assert not transient["idempotent_noop"]
    assert transient["state_hash"] != before["state_hash"]
    assert before["world"]["services"][service_id]["probe_failures_remaining"] == 1
    assert after["world"]["services"][service_id]["probe_failures_remaining"] == 0
    assert env.events[-1].changed_objects == (
        f"services/{service_id}/probe_failures_remaining",
    )

    control = MultiTownStatefulOpsEnv(scenario)
    control.call_tool("rollback_service", {"service_id": service_id})
    control.call_tool("get_service", {"service_id": service_id})
    assert control.observation()["world"] != after["world"]
    assert control.observation()["state_hash"] != after["state_hash"]


def test_conflicting_evidence_has_paired_hidden_worlds_and_active_information() -> None:
    branch_names = (
        "candidate_a_eligible", "candidate_a_ineligible",
        "candidate_b_eligible", "candidate_b_ineligible",
    )
    variants = {
        branch: build_scenario(
            "records_casework", world_seed=108,
            variant_id="conflicting_evidence_investigation",
            dynamics_branch=branch,
        )
        for branch in branch_names
    }
    left = MultiTownStatefulOpsEnv(variants["candidate_a_eligible"])
    right = MultiTownStatefulOpsEnv(variants["candidate_b_ineligible"])
    assert left.observation() == right.observation()
    assert left.scenario.public_task.task_id == right.scenario.public_task.task_id
    rendered = repr(left.observation())
    assert "records_truth" not in rendered
    assert "candidate_a_eligible" not in rendered
    assert "candidate_b_ineligible" not in rendered

    left_truth = left.private_state["records_truth"]
    right_truth = right.private_state["records_truth"]
    left_id = next(key for key, value in left_truth.items() if value["authoritative"])
    right_id = next(key for key, value in right_truth.items() if value["authoritative"])
    assert left_id != right_id
    assert left_truth[left_id]["eligible"] is True
    assert right_truth[right_id]["eligible"] is False

    left_header = left.call_tool("inspect_record_header", {"record_id": left_id})
    right_header = right.call_tool("inspect_record_header", {"record_id": left_id})
    assert left_header["state_hash"] != right_header["state_hash"]
    assert left.state["records"][left_id]["header_status"] == "authenticated"
    assert right.state["records"][left_id]["header_status"] == "superseded"
    blocked = right.call_tool("verify_record", {"record_id": left_id})
    assert blocked["error_code"] == "RECORD_NOT_AUTHENTICATED"

    for branch, scenario in variants.items():
        env = MultiTownStatefulOpsEnv(scenario)
        truth = env.private_state["records_truth"]
        target = next(key for key, value in truth.items() if value["authoritative"])
        case_id = next(iter(env.state["cases"]))
        env.call_tool("inspect_record_header", {"record_id": target})
        env.call_tool("verify_record", {"record_id": target})
        env.call_tool("attach_evidence", {
            "case_id": case_id, "evidence_id": target,
        })
        expected = "approve" if truth[target]["eligible"] else "deny"
        wrong_decision = "deny" if expected == "approve" else "approve"
        wrong = env.call_tool("set_decision", {
            "case_id": case_id, "decision": wrong_decision,
        })
        assert wrong["error_code"] == "DECISION_MISMATCHES_VERIFIED_EVIDENCE"
        assert not env.stop().success
        env = MultiTownStatefulOpsEnv(scenario)
        env.call_tool("inspect_record_header", {"record_id": target})
        env.call_tool("verify_record", {"record_id": target})
        env.call_tool("attach_evidence", {
            "case_id": case_id, "evidence_id": target,
        })
        env.call_tool("set_decision", {
            "case_id": case_id, "decision": expected,
        })
        env.call_tool("set_case_status", {
            "case_id": case_id, "status": "closed",
        })
        assert env.stop().success

    by_record = {}
    for branch, scenario in variants.items():
        env = MultiTownStatefulOpsEnv(scenario)
        truth = env.private_state["records_truth"]
        target = next(key for key, value in truth.items() if value["authoritative"])
        by_record.setdefault(target, set()).add(truth[target]["eligible"])
    assert all(outcomes == {True, False} for outcomes in by_record.values())


def test_optimistic_conflict_has_counterbalanced_hidden_event_pairs() -> None:
    branches = (
        "preferred_a_conflict", "preferred_a_control",
        "preferred_b_conflict", "preferred_b_control",
    )
    scenarios = {
        branch: build_scenario(
            "resource_calendar", world_seed=109,
            variant_id="optimistic_conflict_replan",
            dynamics_branch=branch,
        )
        for branch in branches
    }
    for conflict, control in (
        ("preferred_a_conflict", "preferred_a_control"),
        ("preferred_b_conflict", "preferred_b_control"),
    ):
        left = MultiTownStatefulOpsEnv(scenarios[conflict])
        right = MultiTownStatefulOpsEnv(scenarios[control])
        assert left.observation() == right.observation()
        assert left.scenario.public_task.task_id == right.scenario.public_task.task_id
        assert left.scenario.private_instance_id != right.scenario.private_instance_id
    assert (
        scenarios["preferred_a_conflict"].public_task.task_id
        != scenarios["preferred_b_conflict"].public_task.task_id
    )
    rendered = repr(MultiTownStatefulOpsEnv(
        scenarios["preferred_a_conflict"],
    ).observation())
    assert "conflict_scheduled" not in rendered
    assert "preferred_a_conflict" not in rendered


@pytest.mark.parametrize(
    "branch",
    (
        "preferred_a_conflict", "preferred_a_control",
        "preferred_b_conflict", "preferred_b_control",
    ),
)
def test_optimistic_conflict_reactive_path_succeeds(branch: str) -> None:
    scenario = build_scenario(
        "resource_calendar", world_seed=110,
        variant_id="optimistic_conflict_replan", dynamics_branch=branch,
    )
    env = MultiTownStatefulOpsEnv(scenario)
    booking_id = next(iter(env.state["bookings"]))
    booking = env.state["bookings"][booking_id]
    preferred = booking["preferred_resource_id"]
    alternate = next(key for key in env.state["resources"] if key != preferred)
    snapshot = env.call_tool("snapshot_availability", {"booking_id": booking_id})
    assert snapshot["result"] == "ok"
    token = env.state["bookings"][booking_id]["snapshot_token"]
    first = env.call_tool("create_versioned_hold", {
        "booking_id": booking_id, "resource_id": preferred,
        "snapshot_token": token, "expected_version": "1",
    })
    target = preferred
    if branch.endswith("conflict"):
        assert first["result"] == "conflict"
        assert first["error_code"] == "VERSION_CONFLICT"
        assert first["transition"]["agent_changed_objects"] == []
        assert first["transition"]["external_events"][0]["actor"] == "system"
        assert env.attempted_policy_violations == 0
        target = alternate
        recovered = env.call_tool("create_versioned_hold", {
            "booking_id": booking_id, "resource_id": alternate,
            "snapshot_token": token, "expected_version": "1",
        })
        assert recovered["result"] == "ok"
    else:
        assert first["result"] == "ok"
    env.call_tool("reserve_resource", {
        "booking_id": booking_id, "resource_id": target,
    })
    env.call_tool("send_booking_notice", {"booking_id": booking_id})
    terminal = env.stop()
    assert terminal.success
    replayed, replay_result = replay_events(scenario, env.export_events())
    assert replay_result.success
    assert replayed.export_events() == env.export_events()


def test_optimistic_conflict_shortcuts_are_rejected() -> None:
    scenario = build_scenario(
        "resource_calendar", world_seed=111,
        variant_id="optimistic_conflict_replan",
        dynamics_branch="preferred_a_control",
    )
    env = MultiTownStatefulOpsEnv(scenario)
    booking_id = next(iter(env.state["bookings"]))
    preferred = env.state["bookings"][booking_id]["preferred_resource_id"]
    alternate = next(key for key in env.state["resources"] if key != preferred)
    no_snapshot = env.call_tool("create_versioned_hold", {
        "booking_id": booking_id, "resource_id": preferred,
        "snapshot_token": "invented", "expected_version": "1",
    })
    assert no_snapshot["error_code"] == "SNAPSHOT_REQUIRED"

    env = MultiTownStatefulOpsEnv(scenario)
    env.call_tool("snapshot_availability", {"booking_id": booking_id})
    token = env.state["bookings"][booking_id]["snapshot_token"]
    shortcut = env.call_tool("create_versioned_hold", {
        "booking_id": booking_id, "resource_id": alternate,
        "snapshot_token": token, "expected_version": "1",
    })
    assert shortcut["error_code"] == "PREFERRED_AVAILABLE"
    assert not env.stop().success


def test_guessed_snapshot_token_cannot_trigger_conflict_before_snapshot() -> None:
    scenario = build_scenario(
        "resource_calendar", world_seed=168,
        variant_id="optimistic_conflict_replan",
        dynamics_branch="preferred_a_conflict",
    )
    env = MultiTownStatefulOpsEnv(scenario)
    booking_id = next(iter(env.state["bookings"]))
    booking = env.state["bookings"][booking_id]
    preferred = booking["preferred_resource_id"]
    versions = {
        key: value["version"] for key, value in env.state["resources"].items()
    }
    from multitown.stateful_ops import _snapshot_token
    guessed = _snapshot_token(booking_id, versions)
    args = {
        "booking_id": booking_id, "resource_id": preferred,
        "snapshot_token": guessed,
        "expected_version": str(versions[preferred]),
    }
    first = env.call_tool(
        "create_versioned_hold", args, idempotency_key="guessed-before-snapshot",
    )
    second = env.call_tool(
        "create_versioned_hold", args, idempotency_key="guessed-before-snapshot",
    )
    assert first["result"] == second["result"] == "blocked"
    assert second["idempotent_noop"]
    assert first["transition"]["external_events"] == []
    assert second["transition"]["external_events"] == []
    assert env._applied_event_ids == set()
    assert env.state["resources"][preferred]["status"] == "available"


def test_idempotency_replay_never_reexecutes_deferred_external_event() -> None:
    scenario = build_scenario(
        "resource_calendar", world_seed=169,
        variant_id="optimistic_conflict_replan",
        dynamics_branch="preferred_a_conflict",
    )
    env = MultiTownStatefulOpsEnv(scenario)
    booking_id = next(iter(env.state["bookings"]))
    booking = env.state["bookings"][booking_id]
    preferred = booking["preferred_resource_id"]
    versions = {
        key: value["version"] for key, value in env.state["resources"].items()
    }
    from multitown.stateful_ops import _snapshot_token
    args = {
        "booking_id": booking_id, "resource_id": preferred,
        "snapshot_token": _snapshot_token(booking_id, versions),
        "expected_version": str(versions[preferred]),
    }
    first = env.call_tool(
        "create_versioned_hold", args, idempotency_key="deferred-event",
    )
    assert first["result"] == "blocked"
    env.call_tool("snapshot_availability", {"booking_id": booking_id})
    replay = env.call_tool(
        "create_versioned_hold", args, idempotency_key="deferred-event",
    )
    assert replay["result"] == first["result"]
    assert replay["error_code"] == first["error_code"]
    assert replay["idempotent_noop"]
    assert replay["transition"]["external_events"] == []
    assert env._applied_event_ids == set()
    assert env.state["resources"][preferred]["status"] == "available"


def _advance_canary_to_second_probe(
    env: MultiTownStatefulOpsEnv,
) -> tuple[str, str, str, dict[str, object], dict[str, object]]:
    entity = ids(env)
    service_id = entity["service_id"]
    service = {"service_id": service_id}
    assert env.call_tool("stage_patch", service)["result"] == "ok"
    deployed = env.call_tool("deploy_canary", service)
    assert deployed["result"] == "ok"
    deployment_id = env.state["services"][service_id]["deployment_id"]
    token = env.state["services"][service_id]["compensation_token"]
    first = env.call_tool("probe_canary", service)
    second = env.call_tool("probe_canary", service)
    return service_id, deployment_id, token, first, second


def test_canary_saga_pairs_share_public_prefix_then_reveal_delayed_outcome() -> None:
    scenarios = {
        branch: build_scenario(
            "incident_recovery", world_seed=116,
            variant_id="canary_compensation_saga",
            dynamics_branch=branch,
        )
        for branch in ("compatible_patch", "delayed_regression")
    }
    compatible = MultiTownStatefulOpsEnv(scenarios["compatible_patch"])
    regression = MultiTownStatefulOpsEnv(scenarios["delayed_regression"])
    assert compatible.observation() == regression.observation()
    assert compatible.scenario.public_task.task_id == regression.scenario.public_task.task_id
    assert compatible.scenario.private_instance_id != regression.scenario.private_instance_id
    assert "canary_compatible" not in repr(compatible.observation())

    left = _advance_canary_to_second_probe(compatible)
    right = _advance_canary_to_second_probe(regression)
    assert left[3] == right[3]
    assert left[4]["state_hash"] != right[4]["state_hash"]
    assert compatible.state["services"][left[0]]["canary_status"] == "validated"
    assert regression.state["services"][right[0]]["canary_status"] == "regressed"


@pytest.mark.parametrize(
    ("branch", "resolution", "expected_mode"),
    [
        ("compatible_patch", "promote_canary", "canary-promote"),
        ("delayed_regression", "revert_canary", "canary-revert"),
    ],
)
def test_canary_saga_correct_conditional_resolution_succeeds_and_replays(
    branch: str, resolution: str, expected_mode: str,
) -> None:
    scenario = build_scenario(
        "incident_recovery", world_seed=117,
        variant_id="canary_compensation_saga", dynamics_branch=branch,
    )
    env = MultiTownStatefulOpsEnv(scenario)
    service_id, deployment_id, token, _, _ = _advance_canary_to_second_probe(env)
    service = {"service_id": service_id}
    if resolution == "revert_canary":
        resolution_args = {
            **service, "deployment_id": deployment_id,
            "compensation_token": token,
        }
    else:
        resolution_args = service
    resolved = env.call_tool(resolution, resolution_args)
    assert resolved["result"] == "ok"
    assert resolved["transition"]["tool_kind"] == (
        "irreversible" if resolution == "promote_canary" else "compensation"
    )
    assert resolved["transition"]["irreversible_risk_cost"] == (
        1 if resolution == "promote_canary" else 0
    )
    env.call_tool("verify_service", service)
    env.call_tool("close_incident", {
        "incident_id": ids(env)["incident_id"], "service_id": service_id,
    })
    terminal = env.stop()
    assert terminal.success, terminal.failure_codes
    final_service = env.state["services"][service_id]
    assert final_service["repair_mode"] == expected_mode
    assert final_service["deployment_history"] == [deployment_id]
    assert final_service["compensation_history"] == (
        [deployment_id] if resolution == "revert_canary" else []
    )
    assert final_service["compensation_token"] is None
    replayed, replay_result = replay_events(scenario, env.export_events())
    assert replay_result.success
    assert replayed.export_events() == env.export_events()


@pytest.mark.parametrize(
    ("branch", "wrong_resolution", "error_code"),
    [
        ("compatible_patch", "revert_canary", "COMPENSATION_NOT_ALLOWED"),
        ("delayed_regression", "promote_canary", "CANARY_NOT_VALIDATED"),
    ],
)
def test_canary_saga_rejects_fixed_wrong_resolution(
    branch: str, wrong_resolution: str, error_code: str,
) -> None:
    scenario = build_scenario(
        "incident_recovery", world_seed=118,
        variant_id="canary_compensation_saga", dynamics_branch=branch,
    )
    env = MultiTownStatefulOpsEnv(scenario)
    service_id, deployment_id, token, _, _ = _advance_canary_to_second_probe(env)
    arguments = {"service_id": service_id}
    if wrong_resolution == "revert_canary":
        arguments.update({
            "deployment_id": deployment_id, "compensation_token": token,
        })
    blocked = env.call_tool(wrong_resolution, arguments)
    assert blocked["result"] == "blocked"
    assert blocked["error_code"] == error_code
    assert not env.stop().success


def test_canary_compensation_requires_issued_reference_and_is_single_use() -> None:
    scenario = build_scenario(
        "incident_recovery", world_seed=119,
        variant_id="canary_compensation_saga",
        dynamics_branch="delayed_regression",
    )
    env = MultiTownStatefulOpsEnv(scenario)
    service_id = ids(env)["service_id"]
    fabricated = env.call_tool("revert_canary", {
        "service_id": service_id, "deployment_id": "invented",
        "compensation_token": "invented",
    })
    assert fabricated["error_code"] == "COMPENSATION_NOT_ALLOWED"

    env = MultiTownStatefulOpsEnv(scenario)
    service_id, deployment_id, token, _, _ = _advance_canary_to_second_probe(env)
    arguments = {
        "service_id": service_id, "deployment_id": deployment_id,
        "compensation_token": token,
    }
    assert env.call_tool("revert_canary", arguments)["result"] == "ok"
    duplicate = env.call_tool("revert_canary", arguments)
    assert duplicate["result"] == "blocked"
    assert duplicate["error_code"] == "COMPENSATION_NOT_ALLOWED"
    assert env.state["services"][service_id]["compensation_history"] == [deployment_id]


def _authority_context(env: MultiTownStatefulOpsEnv) -> dict[str, str]:
    permit_id = next(iter(env.state["permits"]))
    policy_version = env.state["protected"]["policy_version"]
    from multitown.stateful_ops import _authority_references
    request_id, scope, capability, expiry = _authority_references(
        permit_id, policy_version,
    )
    return {
        "permit_id": permit_id, "policy_version": policy_version,
        "request_id": request_id, "scope": scope,
        "capability": capability, "expiry": str(expiry),
    }


def _advance_authority_to_outcome(
    env: MultiTownStatefulOpsEnv, *, insert_read: bool = False,
) -> tuple[dict[str, str], dict[str, object], dict[str, object]]:
    context = _authority_context(env)
    assert env.call_tool("submit_authority_request", {
        "permit_id": context["permit_id"], "scope": context["scope"],
        "expected_policy_version": context["policy_version"],
    })["result"] == "ok"
    if insert_read:
        env.call_tool("get_permit", {"permit_id": context["permit_id"]})
    first = env.call_tool("check_authority_status", {
        "permit_id": context["permit_id"], "request_id": context["request_id"],
    })
    second = env.call_tool("check_authority_status", {
        "permit_id": context["permit_id"], "request_id": context["request_id"],
    })
    return context, first, second


def test_authority_triplet_has_identical_prefix_and_second_check_divergence() -> None:
    branches = ("grant_before_deadline", "explicit_deny", "authority_timeout")
    scenarios = {
        branch: build_scenario(
            "permit_transaction", world_seed=121,
            variant_id="asynchronous_authority_timeout",
            dynamics_branch=branch,
        )
        for branch in branches
    }
    envs = {branch: MultiTownStatefulOpsEnv(scenario) for branch, scenario in scenarios.items()}
    observations = [env.observation() for env in envs.values()]
    assert observations.count(observations[0]) == 3
    assert len({scenario.public_task.task_id for scenario in scenarios.values()}) == 1
    assert len({scenario.private_instance_id for scenario in scenarios.values()}) == 3
    advanced = {
        branch: _advance_authority_to_outcome(env)
        for branch, env in envs.items()
    }
    first_results = [value[1] for value in advanced.values()]
    assert first_results.count(first_results[0]) == 3
    assert {env.state["permits"][advanced[branch][0]["permit_id"]]["authority_status"]
            for branch, env in envs.items()} == {"granted", "denied", "timed-out"}
    assert len({value[2]["state_hash"] for value in advanced.values()}) == 3
    assert all(value[2]["transition"]["external_events"] for value in advanced.values())
    assert "authority_outcome" not in repr(observations)


def test_authority_result_waits_for_second_check_even_after_unrelated_read() -> None:
    env = MultiTownStatefulOpsEnv(build_scenario(
        "permit_transaction", world_seed=122,
        variant_id="asynchronous_authority_timeout",
        dynamics_branch="grant_before_deadline",
    ))
    context = _authority_context(env)
    env.call_tool("submit_authority_request", {
        "permit_id": context["permit_id"], "scope": context["scope"],
        "expected_policy_version": context["policy_version"],
    })
    env.call_tool("get_permit", {"permit_id": context["permit_id"]})
    first = env.call_tool("check_authority_status", {
        "permit_id": context["permit_id"], "request_id": context["request_id"],
    })
    assert not first["transition"]["external_events"]
    assert env.state["permits"][context["permit_id"]]["authority_status"] == "pending"
    second = env.call_tool("check_authority_status", {
        "permit_id": context["permit_id"], "request_id": context["request_id"],
    })
    assert second["transition"]["external_events"]
    assert env.state["permits"][context["permit_id"]]["authority_status"] == "granted"


@pytest.mark.parametrize(
    ("branch", "expected_status", "expected_notice"),
    [
        ("grant_before_deadline", "approved", "permit-authority-granted"),
        ("explicit_deny", "denied", "permit-authority-denied"),
        ("authority_timeout", "escalated", "permit-authority-timeout"),
    ],
)
def test_authority_triplet_correct_policy_succeeds_and_replays(
    branch: str, expected_status: str, expected_notice: str,
) -> None:
    scenario = build_scenario(
        "permit_transaction", world_seed=123,
        variant_id="asynchronous_authority_timeout", dynamics_branch=branch,
    )
    env = MultiTownStatefulOpsEnv(scenario)
    context, _, _ = _advance_authority_to_outcome(env)
    if branch == "grant_before_deadline":
        resolution = env.call_tool("issue_permit_with_capability", {
            "permit_id": context["permit_id"], "request_id": context["request_id"],
            "scope": context["scope"], "policy_version": context["policy_version"],
            "expiry": context["expiry"], "capability": context["capability"],
        })
        assert resolution["transition"]["tool_kind"] == "irreversible"
        assert resolution["transition"]["irreversible_risk_cost"] == 1
    elif branch == "authority_timeout":
        resolution = env.call_tool("escalate_human", {
            "permit_id": context["permit_id"], "request_id": context["request_id"],
        })
        assert resolution["transition"]["tool_kind"] == "authority_request"
    notice = env.call_tool(
        "send_permit_status_notice", {"permit_id": context["permit_id"]},
    )
    assert notice["result"] == "ok"
    terminal = env.stop()
    assert terminal.success, terminal.failure_codes
    permit = env.state["permits"][context["permit_id"]]
    assert permit["status"] == expected_status
    assert env.state["notifications"][context["permit_id"]]["kind"] == expected_notice
    replayed, replay_result = replay_events(scenario, env.export_events())
    assert replay_result.success
    assert replayed.export_events() == env.export_events()


@pytest.mark.parametrize(
    ("branch", "fixed_action", "error"),
    [
        ("explicit_deny", "issue", "AUTHORITY_CAPABILITY_INVALID"),
        ("authority_timeout", "issue", "AUTHORITY_CAPABILITY_INVALID"),
        ("grant_before_deadline", "escalate", "HUMAN_ESCALATION_NOT_ALLOWED"),
        ("explicit_deny", "escalate", "HUMAN_ESCALATION_NOT_ALLOWED"),
    ],
)
def test_authority_triplet_rejects_fixed_wrong_actions(
    branch: str, fixed_action: str, error: str,
) -> None:
    env = MultiTownStatefulOpsEnv(build_scenario(
        "permit_transaction", world_seed=124,
        variant_id="asynchronous_authority_timeout", dynamics_branch=branch,
    ))
    context, _, _ = _advance_authority_to_outcome(env)
    if fixed_action == "issue":
        blocked = env.call_tool("issue_permit_with_capability", {
            "permit_id": context["permit_id"], "request_id": context["request_id"],
            "scope": context["scope"], "policy_version": context["policy_version"],
            "expiry": context["expiry"], "capability": context["capability"],
        })
    else:
        blocked = env.call_tool("escalate_human", {
            "permit_id": context["permit_id"], "request_id": context["request_id"],
        })
    assert blocked["result"] == "blocked"
    assert blocked["error_code"] == error
    assert not env.stop().success


def test_authority_capability_rejects_wrong_binding_expiry_and_replay() -> None:
    scenario = build_scenario(
        "permit_transaction", world_seed=125,
        variant_id="asynchronous_authority_timeout",
        dynamics_branch="grant_before_deadline",
    )
    for override in (
        {"request_id": "wrong"}, {"scope": "wrong"},
        {"policy_version": "wrong"}, {"expiry": "999"},
        {"capability": "wrong"},
    ):
        env = MultiTownStatefulOpsEnv(scenario)
        context, _, _ = _advance_authority_to_outcome(env)
        args = {
            "permit_id": context["permit_id"], "request_id": context["request_id"],
            "scope": context["scope"], "policy_version": context["policy_version"],
            "expiry": context["expiry"], "capability": context["capability"],
            **override,
        }
        blocked = env.call_tool("issue_permit_with_capability", args)
        assert blocked["error_code"] == "AUTHORITY_CAPABILITY_INVALID"

    env = MultiTownStatefulOpsEnv(scenario)
    context, _, _ = _advance_authority_to_outcome(env)
    args = {
        "permit_id": context["permit_id"], "request_id": context["request_id"],
        "scope": context["scope"], "policy_version": context["policy_version"],
        "expiry": context["expiry"], "capability": context["capability"],
    }
    assert env.call_tool("issue_permit_with_capability", args)["result"] == "ok"
    replay = env.call_tool("issue_permit_with_capability", args)
    assert replay["result"] == "blocked"
    assert replay["error_code"] == "AUTHORITY_CAPABILITY_INVALID"

    expired = MultiTownStatefulOpsEnv(scenario)
    context = _authority_context(expired)
    expired.call_tool("submit_authority_request", {
        "permit_id": context["permit_id"], "scope": context["scope"],
        "expected_policy_version": context["policy_version"],
    })
    for _ in range(2):
        expired.call_tool("get_permit", {"permit_id": context["permit_id"]})
    for _ in range(2):
        expired.call_tool("check_authority_status", {
            "permit_id": context["permit_id"],
            "request_id": context["request_id"],
        })
    before = expired.observation()["state_hash"]
    stale = expired.call_tool("issue_permit_with_capability", {
        "permit_id": context["permit_id"], "request_id": context["request_id"],
        "scope": context["scope"], "policy_version": context["policy_version"],
        "expiry": context["expiry"], "capability": context["capability"],
    })
    assert stale["result"] == "blocked"
    assert stale["error_code"] == "AUTHORITY_CAPABILITY_INVALID"
    assert stale["state_hash"] == before


def test_authority_capability_is_not_transferable_across_seeded_worlds() -> None:
    source = MultiTownStatefulOpsEnv(build_scenario(
        "permit_transaction", world_seed=126,
        variant_id="asynchronous_authority_timeout",
        dynamics_branch="grant_before_deadline",
    ))
    source_context, _, _ = _advance_authority_to_outcome(source)

    target = MultiTownStatefulOpsEnv(build_scenario(
        "permit_transaction", world_seed=127,
        variant_id="asynchronous_authority_timeout",
        dynamics_branch="grant_before_deadline",
    ))
    target_context, _, _ = _advance_authority_to_outcome(target)
    before = target.observation()["state_hash"]
    blocked = target.call_tool("issue_permit_with_capability", {
        "permit_id": target_context["permit_id"],
        "request_id": target_context["request_id"],
        "scope": target_context["scope"],
        "policy_version": target_context["policy_version"],
        "expiry": target_context["expiry"],
        "capability": source_context["capability"],
    })
    assert blocked["result"] == "blocked"
    assert blocked["error_code"] == "AUTHORITY_CAPABILITY_INVALID"
    assert blocked["state_hash"] == before
    assert not target.state["permits"][target_context["permit_id"]]["capability_used"]


def test_needs_approval_cannot_execute_irreversible_authority_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_checker = MultiTownStatefulOpsEnv.checker

    def force_needs_approval(
        self: MultiTownStatefulOpsEnv, tool_name: str, args: dict[str, object],
    ) -> CheckerDecision:
        if tool_name == "issue_permit_with_capability":
            return CheckerDecision("NEEDS_APPROVAL", "TEST_APPROVAL_REQUIRED")
        return original_checker(self, tool_name, args)

    monkeypatch.setattr(MultiTownStatefulOpsEnv, "checker", force_needs_approval)
    env = MultiTownStatefulOpsEnv(build_scenario(
        "permit_transaction", world_seed=128,
        variant_id="asynchronous_authority_timeout",
        dynamics_branch="grant_before_deadline",
    ))
    context, _, _ = _advance_authority_to_outcome(env)
    before = env.observation()["state_hash"]
    blocked = env.call_tool("issue_permit_with_capability", {
        "permit_id": context["permit_id"],
        "request_id": context["request_id"],
        "scope": context["scope"],
        "policy_version": context["policy_version"],
        "expiry": context["expiry"],
        "capability": context["capability"],
    })
    assert blocked["result"] == "blocked"
    assert blocked["error_code"] == "TEST_APPROVAL_REQUIRED"
    assert blocked["state_hash"] == before
    permit = env.state["permits"][context["permit_id"]]
    assert permit["status"] == "pending"
    assert not permit["capability_used"]
    assert env.executed_safety_violations == 0


def test_idempotency_and_violation_history_distinguish_public_observations() -> None:
    scenario = build_scenario(
        "incident_recovery", world_seed=105,
        variant_id="transient_health_probe",
    )
    keyed, unkeyed = MultiTownStatefulOpsEnv(scenario), MultiTownStatefulOpsEnv(scenario)
    service_id = ids(keyed)["service_id"]
    repair = {"service_id": service_id}
    keyed.call_tool("rollback_service", repair)
    unkeyed.call_tool("rollback_service", repair)
    keyed.call_tool("healthcheck", repair, idempotency_key="probe")
    unkeyed.call_tool("healthcheck", repair)
    keyed_observation, unkeyed_observation = keyed.observation(), unkeyed.observation()
    assert keyed_observation["world"] == unkeyed_observation["world"]
    assert keyed_observation["state_hash"] == unkeyed_observation["state_hash"]
    assert keyed_observation["steps"] == unkeyed_observation["steps"]
    assert keyed_observation["runtime"] != unkeyed_observation["runtime"]
    assert keyed_observation["runtime_hash"] != unkeyed_observation["runtime_hash"]
    keyed_result = keyed.call_tool("healthcheck", repair, idempotency_key="probe")
    unkeyed_result = unkeyed.call_tool("healthcheck", repair, idempotency_key="probe")
    assert keyed_result["idempotent_noop"]
    assert not unkeyed_result["idempotent_noop"]

    records = build_scenario("records_casework", world_seed=106)
    blocked, read = MultiTownStatefulOpsEnv(records), MultiTownStatefulOpsEnv(records)
    case_id = ids(blocked)["case_id"]
    denied = blocked.call_tool(
        "set_case_status", {"case_id": case_id, "status": "closed"},
    )
    assert denied["result"] == "blocked"
    read.call_tool("get_case", {"case_id": case_id})
    assert blocked.observation()["world"] == read.observation()["world"]
    assert blocked.observation()["state_hash"] == read.observation()["state_hash"]
    assert blocked.observation()["steps"] == read.observation()["steps"]
    assert blocked.observation()["runtime"] != read.observation()["runtime"]
    assert blocked.observation()["runtime_hash"] != read.observation()["runtime_hash"]


def test_unique_process_exception_is_preregistered_and_shortcut_rejected() -> None:
    descriptor = structural_descriptor("resource_calendar", "mandatory_hold")
    assert descriptor["unique_process_required"] is True
    assert descriptor["accepted_action_paths"] == (
        ("create_hold", "reserve_resource", "send_booking_notice"),
    )
    env = MultiTownStatefulOpsEnv(build_scenario(
        "resource_calendar", world_seed=104, variant_id="mandatory_hold",
    ))
    blocked = env.call_tool("reserve_resource", ids(env))
    assert blocked["result"] == "blocked"
    assert not env.stop().success


@pytest.mark.parametrize(
    ("family", "variant_id"),
    [
        ("records_casework", "mandatory_stage"),
        ("permit_transaction", "approval_before_number"),
        ("resource_calendar", "mandatory_hold"),
        ("incident_recovery", "transient_health_probe"),
        ("records_casework", "refresh_before_evidence"),
        ("permit_transaction", "inspection_required"),
        ("resource_calendar", "refresh_before_reserve"),
        ("incident_recovery", "approval_before_repair"),
    ],
)
def test_new_variant_replay_and_collateral_checks(
    family: str, variant_id: str,
) -> None:
    env = run_variant_path(family, variant_id)
    result = env.stop()
    assert result.success
    replayed, replay_result = replay_events(
        env.scenario, env.export_events(),
    )
    assert replay_result.success
    assert replayed.export_events() == env.export_events()

    collateral = run_variant_path(family, variant_id)
    key = next(iter(collateral.state["protected"]))
    collateral.state["protected"][key] = "tampered"
    invalid = collateral.stop()
    assert not invalid.success
    assert "COLLATERAL_MUTATION" in invalid.failure_codes


@pytest.mark.parametrize(
    ("family", "variant_id", "shortcut"),
    [
        ("records_casework", "mandatory_stage", "direct_decision"),
        ("permit_transaction", "approval_before_number", "atomic_issue"),
        ("resource_calendar", "mandatory_hold", "direct_reserve"),
        ("incident_recovery", "transient_health_probe", "two_probes"),
        ("records_casework", "refresh_before_evidence", "attach_without_refresh"),
        ("permit_transaction", "inspection_required", "issue_without_inspection"),
        ("resource_calendar", "refresh_before_reserve", "reserve_without_refresh"),
        ("incident_recovery", "approval_before_repair", "repair_without_approval"),
    ],
)
def test_new_variant_changes_feasible_transition_or_path(
    family: str, variant_id: str, shortcut: str,
) -> None:
    env = MultiTownStatefulOpsEnv(build_scenario(
        family, world_seed=103, variant_id=variant_id,
    ))
    entity = ids(env)
    expected_error = None
    if shortcut == "attach_without_refresh":
        blocked = env.call_tool("attach_evidence", entity)
        expected_error = "RECORD_BINDING_STALE"
        assert blocked["result"] == "blocked"
    elif shortcut == "issue_without_inspection":
        blocked = env.call_tool("issue_permit", entity)
        expected_error = "INSPECTION_REQUIRED"
        assert blocked["result"] == "blocked"
    elif shortcut == "reserve_without_refresh":
        blocked = env.call_tool("reserve_resource", entity)
        expected_error = "AVAILABILITY_STALE"
        assert blocked["result"] == "blocked"
    elif shortcut == "repair_without_approval":
        blocked = env.call_tool("rollback_service", {"service_id": entity["service_id"]})
        expected_error = "CHANGE_APPROVAL_REQUIRED"
        assert blocked["result"] == "blocked"
    elif shortcut == "direct_decision":
        env.call_tool("attach_evidence", entity)
        blocked = env.call_tool("set_decision", {
            "case_id": entity["case_id"], "decision": "approve",
        })
        assert blocked["result"] == "blocked"
    elif shortcut == "atomic_issue":
        blocked = env.call_tool("issue_permit", entity)
        assert blocked["result"] == "blocked"
    elif shortcut == "direct_reserve":
        blocked = env.call_tool("reserve_resource", entity)
        assert blocked["result"] == "blocked"
    else:
        env.call_tool("rollback_service", {"service_id": entity["service_id"]})
        env.call_tool("healthcheck", {"service_id": entity["service_id"]})
        env.call_tool("healthcheck", {"service_id": entity["service_id"]})
        blocked = env.call_tool("close_incident", entity)
        assert blocked["result"] == "blocked"
    if expected_error is not None:
        assert blocked["error_code"] == expected_error
    assert not env.stop().success


def test_invalid_target_errors_precede_missing_prerequisite_errors() -> None:
    records = MultiTownStatefulOpsEnv(build_scenario(
        "records_casework", variant_id="refresh_before_evidence",
    ))
    case_id = ids(records)["case_id"]
    result = records.call_tool("attach_evidence", {
        "case_id": case_id, "evidence_id": "unknown-record",
    })
    assert result["error_code"] == "INVALID_EVIDENCE_OR_CLOSED"

    resource = MultiTownStatefulOpsEnv(build_scenario(
        "resource_calendar", variant_id="refresh_before_reserve",
    ))
    booking_id = ids(resource)["booking_id"]
    result = resource.call_tool("reserve_resource", {
        "booking_id": booking_id, "resource_id": "unknown-resource",
    })
    assert result["error_code"] == "CAPACITY_CONFLICT"


@pytest.mark.parametrize("family", FAMILIES)
def test_structural_metadata_is_descriptor_derived(family: str) -> None:
    for variant_id in variants_for_family(family):
        descriptor = structural_descriptor(family, variant_id)
        metadata = structural_metadata(family, variant_id)
        assert descriptor["schema_version"].endswith("v1")
        assert metadata["scenario_group_id"] == metadata["descriptor_fingerprint"]
        assert metadata["generator_id"].endswith("-train-v14")
        assert descriptor["family"] == family
        assert descriptor["variant_id"] == variant_id


def test_unknown_variant_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported structural variant"):
        build_scenario("records_casework", variant_id="renamed-seed")


def test_descriptor_fingerprint_ignores_set_order_and_free_labels() -> None:
    descriptor = structural_descriptor("records_casework", "mandatory_stage")
    reordered = copy.deepcopy(descriptor)
    for key in (
        "transition_nodes", "dependency_edges", "branch_guards",
        "authority_transitions", "required_terminal_outcomes", "protected_invariants",
        "accepted_action_paths",
    ):
        reordered[key] = tuple(reversed(reordered[key]))
    reordered["variant_id"] = "cosmetic_rename"
    reordered["mechanism"] = "cosmetic-free-label"
    reordered["instruction_schema"] = "paraphrased instruction"
    assert descriptor_fingerprint(descriptor) == descriptor_fingerprint(reordered)


@pytest.mark.parametrize(
    "injected",
    [
        "world_seed==999",
        "surface_seed:7",
        "split_stage=selection",
        "heldout-only-guard",
        "calibration branch",
    ],
)
def test_descriptor_rejects_nested_split_and_seed_tokens(injected: str) -> None:
    descriptor = structural_descriptor("records_casework", "mandatory_stage")
    descriptor["branch_guards"] = (*descriptor["branch_guards"], injected)
    with pytest.raises(ValueError, match="split or seed token"):
        normalize_structural_descriptor(descriptor)
