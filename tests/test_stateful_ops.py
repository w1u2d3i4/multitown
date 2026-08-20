import copy

import pytest

from multitown.stateful_ops import (
    FAMILIES,
    IDEMPOTENCY_KEY_MAX_UTF8_BYTES,
    MultiTownStatefulOpsEnv,
    PolicySession,
    build_scenario,
    replay_events,
    tool_profile,
)


def identifiers(env: MultiTownStatefulOpsEnv) -> dict[str, str]:
    family = env.scenario.public_task.family
    if family == "records_casework":
        return {
            "case_id": next(iter(env.state["cases"])),
            "evidence_id": next(iter(env.state["records"])),
        }
    if family == "permit_transaction":
        permit_id = next(iter(env.state["permits"]))
        return {
            "permit_id": permit_id,
            "applicant_id": env.state["permits"][permit_id]["applicant_id"],
        }
    if family == "resource_calendar":
        resource_id = next(
            key for key, value in env.state["resources"].items()
            if value["status"] == "available"
        )
        return {
            "booking_id": next(iter(env.state["bookings"])),
            "resource_id": resource_id,
        }
    incident_id = next(iter(env.state["incidents"]))
    return {
        "incident_id": incident_id,
        "service_id": env.state["incidents"][incident_id]["service_id"],
    }


def successful_path(env: MultiTownStatefulOpsEnv, *, alternative: bool = False) -> None:
    family, ids = env.scenario.public_task.family, identifiers(env)
    if family == "records_casework":
        env.call_tool("attach_evidence", ids, idempotency_key="evidence")
        if alternative:
            env.call_tool("stage_decision", {"case_id": ids["case_id"], "decision": "approve"})
            env.call_tool("finalize_case", {"case_id": ids["case_id"]})
        else:
            env.call_tool("set_decision", {"case_id": ids["case_id"], "decision": "approve"})
            env.call_tool("set_case_status", {"case_id": ids["case_id"], "status": "closed"})
    elif family == "permit_transaction":
        if alternative:
            env.call_tool("approve_permit", {"permit_id": ids["permit_id"]})
            env.call_tool("assign_permit_number", {"permit_id": ids["permit_id"]})
        else:
            env.call_tool("issue_permit", {"permit_id": ids["permit_id"]}, idempotency_key="issue")
        env.call_tool("send_permit_notice", {"permit_id": ids["permit_id"]})
    elif family == "resource_calendar":
        if alternative:
            env.call_tool("create_hold", ids)
        env.call_tool("reserve_resource", ids, idempotency_key="reserve")
        env.call_tool("send_booking_notice", {"booking_id": ids["booking_id"]})
    else:
        repair = "patch_service" if alternative else "rollback_service"
        env.call_tool(repair, {"service_id": ids["service_id"]})
        if alternative:
            env.call_tool("promote_config", {"service_id": ids["service_id"]})
        env.call_tool("healthcheck", {"service_id": ids["service_id"]})
        env.call_tool("healthcheck", {"service_id": ids["service_id"]})
        env.call_tool("close_incident", ids)


@pytest.mark.parametrize("family", FAMILIES)
def test_gold_and_alternative_paths_pass(family: str) -> None:
    for alternative in (False, True):
        env = MultiTownStatefulOpsEnv(build_scenario(family, world_seed=13))
        successful_path(env, alternative=alternative)
        result = env.stop()
        assert result.success
        assert not result.failure_codes
        assert not result.unexpected_mutations


@pytest.mark.parametrize("family", FAMILIES)
def test_noop_near_miss_and_collateral_mutant_fail(family: str) -> None:
    scenario = build_scenario(family, world_seed=17)
    noop = MultiTownStatefulOpsEnv(scenario).stop()
    assert not noop.success
    assert "REQUIRED_OUTCOME_MISSING" in noop.failure_codes

    near_miss = MultiTownStatefulOpsEnv(scenario)
    successful_path(near_miss)
    # Undo one required outcome while preserving a plausible partial trajectory.
    predicate = scenario.private_evaluator.required_predicates[0]
    target = near_miss.state
    for component in predicate.path[:-1]:
        target = target[component]
        target[predicate.path[-1]] = copy.deepcopy(
        scenario.private_evaluator.initial_state()[predicate.path[0]][predicate.path[1]][predicate.path[2]],
    )
    assert not near_miss.stop().success

    collateral = MultiTownStatefulOpsEnv(scenario)
    successful_path(collateral)
    collateral.state["protected"][next(iter(collateral.state["protected"]))] = "tampered"
    result = collateral.stop()
    assert not result.success
    assert "COLLATERAL_MUTATION" in result.failure_codes


@pytest.mark.parametrize("family", FAMILIES)
def test_reset_and_event_replay_are_deterministic(family: str) -> None:
    scenario = build_scenario(family, world_seed=23)
    env = MultiTownStatefulOpsEnv(scenario)
    initial = env.observation()
    successful_path(env)
    events = env.export_events()
    expected_hash = env.observation()["state_hash"]
    result = env.stop()
    events = env.export_events()
    assert result.success

    replayed, replay_result = replay_events(scenario, events)
    assert replay_result.success
    assert replayed.observation()["state_hash"] == expected_hash
    assert replayed.reset() == initial


def test_private_evaluator_never_enters_policy_payload() -> None:
    scenario = build_scenario("records_casework", world_seed=31)
    env = MultiTownStatefulOpsEnv(scenario)
    rendered = repr(env.observation()).lower()
    assert "required_predicates" not in rendered
    assert "allowed_mutation_scope" not in rendered
    assert "evaluator_hash" not in rendered
    assert scenario.private_evaluator.evaluator_hash not in rendered
    session = PolicySession(scenario)
    assert not hasattr(session, "scenario")
    assert not hasattr(session, "state")
    assert not hasattr(session, "validate_terminal")


def test_runtime_checker_blocks_precondition_violation_without_mutation() -> None:
    env = MultiTownStatefulOpsEnv(build_scenario("incident_recovery", world_seed=37))
    ids = identifiers(env)
    before = env.observation()["state_hash"]
    result = env.call_tool("close_incident", ids)
    assert result["result"] == "blocked"
    assert result["error_code"] == "RECOVERY_UNVERIFIED"
    assert result["state_hash"] == before
    assert env.attempted_policy_violations == 1
    assert env.blocked_unsafe_actions == 1


def test_idempotency_is_explicit_and_conflicting_reuse_is_blocked() -> None:
    env = MultiTownStatefulOpsEnv(build_scenario("permit_transaction", world_seed=41))
    ids = identifiers(env)
    args = {"permit_id": ids["permit_id"]}
    first = env.call_tool("issue_permit", args, idempotency_key="same")
    second = env.call_tool("issue_permit", args, idempotency_key="same")
    assert first["result"] == second["result"] == "ok"
    assert second["idempotent_noop"]
    conflict = env.call_tool(
        "send_permit_notice", args, idempotency_key="same",
    )
    assert conflict["error_code"] == "IDEMPOTENCY_KEY_REUSE"


def test_idempotency_rejects_empty_key_and_replays_read_payload() -> None:
    env = MultiTownStatefulOpsEnv(build_scenario("records_casework", world_seed=42))
    with pytest.raises(TypeError, match="non-empty"):
        env.call_tool("search_records", {}, idempotency_key="")
    before = env.observation()
    first = env.call_tool("search_records", {}, idempotency_key="read-once")
    second = env.call_tool("search_records", {}, idempotency_key="read-once")
    assert first["result"] == second["result"] == "ok"
    assert first["payload"] == second["payload"]
    assert first["payload"] is not second["payload"]
    assert not first["idempotent_noop"]
    assert second["idempotent_noop"]
    assert before["world"] == env.observation()["world"]
    first_record = next(iter(first["payload"]))
    second["payload"][first_record]["summary"] = "caller-mutated-copy"
    third = env.call_tool("search_records", {}, idempotency_key="read-once")
    assert third["payload"][first_record]["summary"] != "caller-mutated-copy"
    conflict = env.call_tool("get_case", {
        "case_id": next(iter(env.state["cases"])),
    }, idempotency_key="read-once")
    assert conflict["result"] == "blocked"
    assert conflict["error_code"] == "IDEMPOTENCY_KEY_REUSE"


@pytest.mark.parametrize(
    ("key", "message"),
    [
        ("e\N{COMBINING ACUTE ACCENT}", "NFC"),
        ("contains\ncontrol", "control"),
        ("x" * (IDEMPOTENCY_KEY_MAX_UTF8_BYTES + 1), "128 UTF-8 bytes"),
    ],
)
def test_idempotency_key_unicode_and_length_contract(
    key: str, message: str,
) -> None:
    env = MultiTownStatefulOpsEnv(
        build_scenario("records_casework", world_seed=142),
    )
    with pytest.raises(TypeError, match=message):
        env.call_tool("search_records", {}, idempotency_key=key)
    assert env.observation()["runtime"]["idempotency_records"] == {}
    accepted = "é" * (IDEMPOTENCY_KEY_MAX_UTF8_BYTES // 2)
    assert env.call_tool(
        "search_records", {}, idempotency_key=accepted,
    )["result"] == "ok"


def test_execution_started_blocked_outcome_is_snapshot_replayed() -> None:
    env = MultiTownStatefulOpsEnv(
        build_scenario("incident_recovery", world_seed=143),
    )
    ids = identifiers(env)
    args = {"incident_id": ids["incident_id"], "service_id": ids["service_id"]}
    first = env.call_tool("close_incident", args, idempotency_key="blocked-once")
    assert first["result"] == "blocked"
    assert first["error_code"] == "RECOVERY_UNVERIFIED"
    record = env.observation()["runtime"]["idempotency_records"]["blocked-once"]
    assert record["result"] == "blocked"
    assert record["error_code"] == "RECOVERY_UNVERIFIED"
    env.call_tool("approve_change", {"service_id": ids["service_id"]})
    replay = env.call_tool("close_incident", args, idempotency_key="blocked-once")
    assert replay["result"] == first["result"]
    assert replay["error_code"] == first["error_code"]
    assert replay["payload"] == first["payload"]
    assert replay["idempotent_noop"]


@pytest.mark.parametrize("stage", ["calibration", "selection", "heldout"])
def test_only_train_stage_is_supported(stage: str) -> None:
    with pytest.raises(ValueError, match="unsupported split stage"):
        build_scenario("records_casework", split_stage=stage)


def test_validator_is_not_callable_before_stop() -> None:
    env = MultiTownStatefulOpsEnv(build_scenario("records_casework", world_seed=43))
    with pytest.raises(RuntimeError, match="irreversible stop"):
        env._validate_terminal()


def test_wrong_intermediate_records_decision_cannot_be_washed_out() -> None:
    env = MultiTownStatefulOpsEnv(build_scenario("records_casework", world_seed=47))
    ids = identifiers(env)
    env.call_tool("attach_evidence", ids)
    env.call_tool("set_decision", {"case_id": ids["case_id"], "decision": "deny"})
    env.call_tool("set_case_status", {"case_id": ids["case_id"], "status": "closed"})
    correction = env.call_tool(
        "set_decision", {"case_id": ids["case_id"], "decision": "approve"},
    )
    assert correction["result"] == "blocked"
    assert not env.stop().success


def test_incident_cannot_mutate_after_close() -> None:
    env = MultiTownStatefulOpsEnv(build_scenario("incident_recovery", world_seed=53))
    successful_path(env)
    ids = identifiers(env)
    result = env.call_tool("patch_service", {"service_id": ids["service_id"]})
    assert result["result"] == "blocked"
    assert not env.stop().success


def test_scope_internal_collateral_leaf_and_orphan_hold_fail() -> None:
    records = MultiTownStatefulOpsEnv(build_scenario("records_casework", world_seed=59))
    successful_path(records)
    case_id = identifiers(records)["case_id"]
    records.state["cases"][case_id]["extra"] = "collateral"
    assert "COLLATERAL_MUTATION" in records.stop().failure_codes

    resource = MultiTownStatefulOpsEnv(build_scenario("resource_calendar", world_seed=61))
    resource_id = identifiers(resource)["resource_id"]
    successful_path(resource)
    resource.state["resources"][resource_id]["held_for"] = "orphan"
    assert not resource.stop().success


@pytest.mark.parametrize("value", [None, {}, []])
def test_forbidden_missing_leaf_additions_are_detected(value: object) -> None:
    env = MultiTownStatefulOpsEnv(build_scenario("records_casework", world_seed=63))
    successful_path(env)
    case_id = identifiers(env)["case_id"]
    env.state["cases"][case_id]["forbidden"] = copy.deepcopy(value)
    result = env.stop()
    assert not result.success
    assert "COLLATERAL_MUTATION" in result.failure_codes


def test_allowed_list_leaf_requires_exact_evidence_content() -> None:
    env = MultiTownStatefulOpsEnv(build_scenario("records_casework", world_seed=65))
    successful_path(env)
    case_id = identifiers(env)["case_id"]
    env.state["cases"][case_id]["evidence_ids"].append("bogus-record")
    result = env.stop()
    assert not result.success
    assert "REQUIRED_OUTCOME_MISSING" in result.failure_codes


@pytest.mark.parametrize(
    ("family", "table", "leaf"),
    [
        ("records_casework", "cases", "draft_decision"),
        ("resource_calendar", "resources", "held_for"),
    ],
)
def test_required_explicit_none_leaf_cannot_be_deleted(
    family: str, table: str, leaf: str,
) -> None:
    env = MultiTownStatefulOpsEnv(build_scenario(family, world_seed=66))
    successful_path(env)
    entity = next(
        value for value in env.state[table].values()
        if leaf in value and value[leaf] is None
    )
    del entity[leaf]
    result = env.stop()
    assert not result.success
    assert set(result.failure_codes) & {
        "REQUIRED_OUTCOME_MISSING", "UNLOGGED_STATE_MUTATION", "COLLATERAL_MUTATION",
    }


@pytest.mark.parametrize(
    ("leaf", "value"),
    [("repair_mode", "patch"), ("healthy_checks", 999)],
)
def test_unlogged_allowed_leaf_mutation_breaks_hash_chain(
    leaf: str, value: object,
) -> None:
    env = MultiTownStatefulOpsEnv(build_scenario("incident_recovery", world_seed=68))
    successful_path(env)
    service_id = identifiers(env)["service_id"]
    env.state["services"][service_id][leaf] = value
    result = env.stop()
    assert not result.success
    assert "UNLOGGED_STATE_MUTATION" in result.failure_codes


def test_cross_family_read_is_blocked_and_logged() -> None:
    env = MultiTownStatefulOpsEnv(build_scenario("records_casework", world_seed=67))
    result = env.call_tool("get_service", {"service_id": "x"})
    assert result["result"] == "blocked"
    assert env.events[-1].error_code == "TOOL_OUT_OF_SCOPE"


def test_cross_type_incident_reference_is_blocked_without_crashing() -> None:
    env = MultiTownStatefulOpsEnv(
        build_scenario("incident_recovery", world_seed=167),
    )
    ids = identifiers(env)
    result = env.call_tool("close_incident", {
        "incident_id": ids["service_id"], "service_id": ids["service_id"],
    })
    assert result["result"] == "blocked"
    assert result["error_code"] == "RECOVERY_UNVERIFIED"
    assert not result["transition"]["agent_changed_objects"]


def test_tool_profile_is_machine_readable_and_arguments_are_exact() -> None:
    profile = tool_profile("records_casework")
    assert profile["tools"]["attach_evidence"]["required"] == [
        "case_id", "evidence_id",
    ]
    env = MultiTownStatefulOpsEnv(build_scenario("records_casework", world_seed=69))
    result = env.call_tool("get_case", {"case_id": "x", "extra": "leak"})
    assert result["error_code"] == "INVALID_ARGUMENT_SCHEMA"
    assert env.events[-1].post_state_hash == env.events[-1].pre_state_hash


def test_budget_rejection_and_idempotent_replay_are_faithful() -> None:
    scenario = build_scenario("permit_transaction", world_seed=71)
    env = MultiTownStatefulOpsEnv(scenario)
    ids = identifiers(env)
    args = {"permit_id": ids["permit_id"]}
    env.call_tool("issue_permit", args, idempotency_key="issue")
    env.call_tool("issue_permit", args, idempotency_key="issue")
    env.call_tool("send_permit_notice", args)
    for _ in range(7):
        env.call_tool("get_permit", args)
    rejected = env.call_tool("get_permit", args)
    assert rejected["error_code"] == "BUDGET_EXHAUSTED"
    with pytest.raises(RuntimeError, match="after budget exhaustion"):
        env.call_tool("get_permit", args)
    original_result = env.stop()
    assert not original_result.success
    replayed, replay_result = replay_events(scenario, env.export_events())
    assert replayed.export_events() == env.export_events()
    assert replay_result == original_result


def test_surface_variants_are_disabled_until_they_change_public_content() -> None:
    with pytest.raises(ValueError, match="surface variants are disabled"):
        build_scenario("records_casework", surface_seed=2)


def test_first_keyed_noop_replays_with_the_same_result() -> None:
    scenario = build_scenario("incident_recovery", world_seed=72)
    env = MultiTownStatefulOpsEnv(scenario)
    ids = identifiers(env)
    first = env.call_tool(
        "healthcheck", {"service_id": ids["service_id"]}, idempotency_key="noop",
    )
    second = env.call_tool(
        "healthcheck", {"service_id": ids["service_id"]}, idempotency_key="noop",
    )
    assert first["result"] == second["result"] == "noop"
    assert second["idempotent_noop"]


@pytest.mark.parametrize("family", FAMILIES)
def test_alternative_paths_have_distinct_state_changing_sequences(family: str) -> None:
    paths = []
    for alternative in (False, True):
        env = MultiTownStatefulOpsEnv(build_scenario(family, world_seed=73))
        successful_path(env, alternative=alternative)
        assert env.stop().success
        paths.append(tuple(
            (event.tool_name, event.post_state_hash) for event in env.events
            if "STATE_CHANGED" in event.audit_codes
        ))
    assert paths[0] != paths[1]
