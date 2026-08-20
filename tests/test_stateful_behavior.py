from multitown.stateful_behavior import (
    _normalize,
    _roles,
    audit_behavioral_catalog,
    behavioral_fingerprint,
    behavioral_probe,
)
from multitown.stateful_ops import FAMILIES, MultiTownStatefulOpsEnv, build_scenario
from multitown.stateful_groups import structural_descriptor, variants_for_family


def test_behavioral_fingerprints_are_seed_invariant_and_variant_distinct() -> None:
    audit = audit_behavioral_catalog()
    assert audit["group_count"] == 16
    assert audit["all_seed_invariant"]
    assert not audit["duplicate_behavioral_fingerprints"]
    for family in FAMILIES:
        rows = [row for row in audit["rows"] if row["family"] == family]
        expected = 4
        assert len(rows) == expected
        assert len({row["behavioral_fingerprint"] for row in rows}) == expected


def test_behavioral_probe_is_role_normalized_and_state_machine_derived() -> None:
    left = behavioral_probe("permit_transaction", "approval_before_number", world_seed=1)
    right = behavioral_probe("permit_transaction", "approval_before_number", world_seed=999)
    assert left == right
    rendered = repr(left)
    assert "permit-0001" not in rendered
    assert "permit-0999" not in rendered
    assert "TARGET_PERMIT" in rendered
    assert "TARGET_PERMIT_NUMBER" in rendered


def test_behavioral_fingerprint_changes_with_real_feasibility() -> None:
    direct = behavioral_fingerprint("resource_calendar", "direct_or_hold")
    mandatory = behavioral_fingerprint("resource_calendar", "mandatory_hold")
    assert direct != mandatory


def test_conflicting_evidence_behavior_covers_both_hidden_branches() -> None:
    probe = behavioral_probe(
        "records_casework", "conflicting_evidence_investigation",
    )
    assert len(probe["episodes"]) == 12
    outcomes = [episode["terminal"]["success"] for episode in probe["episodes"]]
    assert outcomes.count(True) == 8
    assert outcomes.count(False) == 4  # one preregistered read-only near miss per branch
    decisions = {
        transition["arguments"].get("decision")
        for episode in probe["episodes"]
        for transition in episode["transitions"]
        if transition["tool_name"] == "set_decision"
    }
    assert decisions == {"approve", "deny"}


def test_optimistic_conflict_behavior_covers_conflict_and_control_pairs() -> None:
    probe = behavioral_probe(
        "resource_calendar", "optimistic_conflict_replan",
    )
    assert len(probe["episodes"]) == 12
    outcomes = [episode["terminal"]["success"] for episode in probe["episodes"]]
    assert outcomes.count(True) == 8
    assert outcomes.count(False) == 4
    results = {
        transition["result"]
        for episode in probe["episodes"]
        for transition in episode["transitions"]
        if transition["tool_name"] == "create_versioned_hold"
    }
    assert results == {"ok", "conflict"}


def test_canary_saga_behavior_covers_both_delayed_outcomes_and_wrong_actions() -> None:
    probe = behavioral_probe(
        "incident_recovery", "canary_compensation_saga",
    )
    assert len(probe["episodes"]) == 6
    outcomes = [episode["terminal"]["success"] for episode in probe["episodes"]]
    assert outcomes.count(True) == 2
    assert outcomes.count(False) == 4
    resolutions = {
        transition["tool_name"]
        for episode in probe["episodes"]
        for transition in episode["transitions"]
        if transition["tool_name"] in {"promote_canary", "revert_canary"}
    }
    assert resolutions == {"promote_canary", "revert_canary"}
    resolution_transitions = {
        transition["tool_name"]: transition
        for episode in probe["episodes"]
        for transition in episode["transitions"]
        if transition["result"] == "ok"
        and transition["tool_name"] in {"promote_canary", "revert_canary"}
    }
    assert resolution_transitions["promote_canary"]["transition"]["tool_kind"] == (
        "irreversible"
    )
    assert resolution_transitions["promote_canary"]["transition"][
        "irreversible_risk_cost"
    ] == 1
    assert resolution_transitions["revert_canary"]["transition"]["tool_kind"] == (
        "compensation"
    )
    assert resolution_transitions["revert_canary"]["transition"][
        "logical_latency_cost"
    ] == 2
    assert resolution_transitions["revert_canary"]["budget_after"][
        "irreversible_risk_remaining"
    ] == 1
    assert "canary_compatible" not in repr(probe)
    assert "service-0001" not in repr(probe)
    assert "CANARY_DEPLOYMENT" in repr(probe)
    assert "COMPENSATION_TOKEN" in repr(probe)


def test_authority_timeout_behavior_covers_triplet_and_fixed_baselines() -> None:
    probe = behavioral_probe(
        "permit_transaction", "asynchronous_authority_timeout",
    )
    assert len(probe["episodes"]) == 15
    outcomes = [episode["terminal"]["success"] for episode in probe["episodes"]]
    assert outcomes.count(True) == 3
    assert outcomes.count(False) == 12
    authority_events = [
        event
        for episode in probe["episodes"]
        for transition in episode["transitions"]
        for event in transition["transition"]["external_events"]
    ]
    assert authority_events
    assert {event["actor"] for event in authority_events} == {"authority"}
    assert "authority_outcome" not in repr(probe)
    assert "permit-0001" not in repr(probe)
    assert "AUTHORITY_REQUEST" in repr(probe)
    assert "AUTHORITY_CAPABILITY" in repr(probe)


def test_every_declared_accepted_path_is_probed_for_each_family() -> None:
    for family in FAMILIES:
        expected = {
            tuple(path)
            for variant_id in variants_for_family(family)
            for path in structural_descriptor(family, variant_id)["accepted_action_paths"]
        }
        actual = set()
        for variant_id in variants_for_family(family):
            probe = behavioral_probe(family, variant_id)
            actual.update({
                tuple(transition["tool_name"] for transition in episode["transitions"])
                for episode in probe["episodes"]
            })
        assert expected <= actual


def test_nonstructural_protected_fixture_values_are_type_normalized() -> None:
    env = MultiTownStatefulOpsEnv(build_scenario("records_casework", world_seed=11))
    roles = _roles(env)
    left = _normalize(env.state, roles)
    env.state["protected"]["policy_version"] = "cosmetic-policy-v999"
    env.state["protected"]["unrelated_row"] = "different-cosmetic-value"
    right = _normalize(env.state, roles)
    assert left == right
    assert left["protected"] == {
        "policy_version": "PROTECTED_STRING",
        "unrelated_row": "PROTECTED_STRING",
    }
