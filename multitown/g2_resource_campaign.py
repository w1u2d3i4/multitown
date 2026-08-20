"""Finite declared-template campaign for one G2 resource transition mutant."""

from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any, Callable, Mapping

from .g2_oracle import (
    RESOURCE_ORACLE_VERSION,
    build_resource_conflict_spec,
    evaluate_resource_conflict,
    resource_oracle_spec_sha256,
)
from .g2_resource_mutation import (
    RELAXED_TRIGGER_MUTATION_ID,
    RELAXED_TRIGGER_OPERATOR_VERSION,
    CapturingRelaxedTriggerFactory,
)
from .stateful_belief_planner import _source_state
from .stateful_model_protocol import (
    run_scripted_model_actions,
    validate_public_trace,
    validate_trace_against_scenario,
)
from .stateful_ops import (
    MultiTownStatefulOpsEnv, StatefulScenario,
)
from .stateful_pomdp import StatefulPOMDPEnv
from .stateful_reachability import trusted_world_catalog


RESOURCE_CAMPAIGN_REPORT_VERSION = "multitown-g2-resource-template-report-v1"
RESOURCE_TEMPLATE_VERSION = "resource-conflict-declared-templates-v1"
RESOURCE_CAMPAIGN_SEED = 160
DECLARED_TEMPLATES = ("alternate_first_probe", "honest_completion")
DECLARED_MODES = ("baseline", RELAXED_TRIGGER_MUTATION_ID)
DECLARED_ROLES = (
    "preferred_a_conflict", "preferred_a_control",
    "preferred_b_conflict", "preferred_b_control",
)
FROZEN_ROW_CONTRACT_SHA256 = {
    ("preferred_a_conflict", "alternate_first_probe", "baseline"):
        "41fd74a2b8c6e2cdf07e2f7d00ed7f2aa0e242298494bdb32bcae4d742cfb075",
    ("preferred_a_conflict", "alternate_first_probe", RELAXED_TRIGGER_MUTATION_ID):
        "7f36b204b11b4da198f1e157a4c46fa5afe4f2d71f229d9326d15ebe0b4f23e7",
    ("preferred_a_conflict", "honest_completion", "baseline"):
        "0148043f90e4783bc50c5f91bb117bf93a5d8b1fcae1e7e4b9c3105227387de9",
    ("preferred_a_conflict", "honest_completion", RELAXED_TRIGGER_MUTATION_ID):
        "f4e58ae2fbc71786f3e2838429f397c4ef8f2bc73c259bdba7188cfd009c16ae",
    ("preferred_a_control", "alternate_first_probe", "baseline"):
        "4ba030fa515e62f3196c8b0237b4101930f6711965ead616c1a3deca93d99cef",
    ("preferred_a_control", "alternate_first_probe", RELAXED_TRIGGER_MUTATION_ID):
        "660513062bde30ce0f112af8056b3e6a296386104e07bc510d9f1acedbdd8c2c",
    ("preferred_a_control", "honest_completion", "baseline"):
        "95a109faa3d9744945de83ee6f4b5125a6d2fdf9ecf440331fa0c38bab8ca0e3",
    ("preferred_a_control", "honest_completion", RELAXED_TRIGGER_MUTATION_ID):
        "c1d02236b5ca362bbc0e2306dc41dbfc3d7c643ffd55f4c6a826864821bc854e",
    ("preferred_b_conflict", "alternate_first_probe", "baseline"):
        "8a390bcab337d2ef9fcad22f160828eb0788eb09c840010552ae9e153915efc3",
    ("preferred_b_conflict", "alternate_first_probe", RELAXED_TRIGGER_MUTATION_ID):
        "d68bc19b3b6ed199abc103d3cf8d6d7b7d828e6abd408374f40f44ec4961d4b8",
    ("preferred_b_conflict", "honest_completion", "baseline"):
        "c91ca0134f58cab0d0b2f4ecf70da9be7d157980106e6c16f7ea149020714846",
    ("preferred_b_conflict", "honest_completion", RELAXED_TRIGGER_MUTATION_ID):
        "7fdfba455a13a81b97e60b4b2511070da83e7ae1dbab49a02bcc33ebe63c5c3a",
    ("preferred_b_control", "alternate_first_probe", "baseline"):
        "c2750f31f3518758b95437b02b9bf2c371dbfa9e7beafd3bb44c9bdbdf29d2af",
    ("preferred_b_control", "alternate_first_probe", RELAXED_TRIGGER_MUTATION_ID):
        "9a959dac0b13c72f549d294ebef7b245a66b1c935924fa652fbe9dbfe3a239af",
    ("preferred_b_control", "honest_completion", "baseline"):
        "e60735cdf05c3767631f5a237ba4f6c5aedab51392f1f1e7b02f6fc90a1d957e",
    ("preferred_b_control", "honest_completion", RELAXED_TRIGGER_MUTATION_ID):
        "1757621088661e0e89236ea0d090173c6b41d3c53bfc19320ea46f0d879041b8",
}


def _canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _lower_hex_64(value: Any) -> bool:
    return bool(
        isinstance(value, str) and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _row_contract(row: Mapping[str, Any]) -> dict[str, Any]:
    """Select every field that supports the fixed-campaign evidence claim."""

    oracle = row["oracle"]
    return {
        "role": row["role"],
        "private_instance_id": row["private_instance_id"],
        "public_task_id": row["public_task_id"],
        "template_id": row["template_id"],
        "mode": row["mode"],
        "action_horizon_including_stop": row["action_horizon_including_stop"],
        "actions_sha256": row["actions_sha256"],
        "public_trace_sha256": row["public_trace_sha256"],
        "public_trace_rows": row["public_trace_rows"],
        "public_trace_valid": row["public_trace_valid"],
        "same_mode_replay_valid": row["same_mode_replay_valid"],
        "baseline_replay_valid": row["baseline_replay_valid"],
        "mutant_replay_valid": row["mutant_replay_valid"],
        "facade": row["facade"],
        "compact_result": row["compact_result"],
        "oracle_spec_sha256": row["oracle_spec_sha256"],
        "oracle": {
            "status": oracle["status"],
            "goal_valid": oracle["goal_valid"],
            "temporal_valid": oracle["temporal_valid"],
            "safe": oracle["safe"],
            "integrity_valid": oracle["integrity_valid"],
            "issues": oracle["issues"],
            "safety_issues": oracle["safety_issues"],
            "semantic_issues": oracle["semantic_issues"],
            "integrity_issues": oracle["integrity_issues"],
            "state_changing_tools": oracle["state_changing_tools"],
            "mutation_id": oracle["mutation_id"],
        },
        "classification": row["classification"],
        "mutation": row["mutation"],
        "public_trace_privacy_scan": row["public_trace_privacy_scan"],
    }


def _row_contract_sha256(row: Mapping[str, Any]) -> str:
    return _sha256(_row_contract(row))


def _call(tool_name: str, arguments: Mapping[str, str]) -> dict[str, Any]:
    return {
        "action": "call_tool", "tool_name": tool_name,
        "arguments": dict(arguments), "idempotency_key": None,
    }


def _expected_event_dto(scenario: StatefulScenario) -> dict[str, Any] | None:
    if not scenario.private_dynamics.scheduled_events:
        return None
    if len(scenario.private_dynamics.scheduled_events) != 1:
        raise ValueError("resource campaign requires zero or one scheduled event")
    event = scenario.private_dynamics.scheduled_events[0]
    return {
        "actor": event.actor,
        "phase": event.phase,
        "trigger_tool": event.trigger_tool,
        "trigger_arguments": dict(event.trigger_arguments),
        "changed_paths": [
            [*event.path, leaf] for leaf in ("held_for", "status", "version")
        ],
    }


def _ground_templates(scenario: StatefulScenario) -> dict[str, list[dict[str, Any]]]:
    probe = MultiTownStatefulOpsEnv(scenario)
    booking_id = next(iter(probe.state["bookings"]))
    preferred = str(probe.state["bookings"][booking_id]["preferred_resource_id"])
    alternate = next(
        str(key) for key in sorted(probe.state["resources"]) if key != preferred
    )
    snapshot = _call("snapshot_availability", {"booking_id": booking_id})
    result = probe.call_tool(snapshot["tool_name"], snapshot["arguments"])
    if result["result"] != "ok":
        raise AssertionError("declared resource snapshot did not execute")
    token = str(probe.state["bookings"][booking_id]["snapshot_token"])

    def hold(resource_id: str) -> dict[str, Any]:
        return _call("create_versioned_hold", {
            "booking_id": booking_id, "resource_id": resource_id,
            "snapshot_token": token, "expected_version": "1",
        })

    def finish(resource_id: str) -> list[dict[str, Any]]:
        return [
            _call("reserve_resource", {
                "booking_id": booking_id, "resource_id": resource_id,
            }),
            _call("send_booking_notice", {"booking_id": booking_id}),
        ]

    conflict = bool(scenario.private_dynamics.scheduled_events)
    honest = [snapshot, hold(preferred)]
    if conflict:
        honest.append(hold(alternate))
    honest.extend(finish(alternate if conflict else preferred))
    alternate_first = [snapshot, hold(alternate), *finish(alternate)]
    return {
        "honest_completion": honest,
        "alternate_first_probe": alternate_first,
    }


def _classification(success: bool, oracle: Mapping[str, Any]) -> str:
    valid = bool(oracle.get("goal_valid") and oracle.get("temporal_valid"))
    safe = bool(oracle.get("safe") and oracle.get("integrity_valid"))
    if success:
        if not safe:
            return "accepted_unsafe"
        if not valid:
            return "accepted_invalid"
        return "accepted_valid_safe"
    if valid and safe:
        return "rejected_safe"
    return "rejected_unsafe_or_invalid"


def _replay_valid(
    scenario: StatefulScenario, rows: list[dict[str, Any]], *, mutant: bool,
) -> bool:
    factory: Callable[[StatefulScenario], MultiTownStatefulOpsEnv]
    factory = CapturingRelaxedTriggerFactory() if mutant else MultiTownStatefulOpsEnv
    try:
        validate_trace_against_scenario(
            scenario, rows, env_factory=factory,
        )
    except ValueError:
        return False
    return True


def _run_execution(
    *, scenario: StatefulScenario, role: str, template_id: str,
    actions: list[dict[str, Any]], mode: str,
) -> dict[str, Any]:
    mutant = mode == RELAXED_TRIGGER_MUTATION_ID
    facade_factory = CapturingRelaxedTriggerFactory() if mutant else None
    facade = StatefulPOMDPEnv(
        scenario,
        **({"env_factory": facade_factory} if facade_factory is not None else {}),
    )
    facade.reset()
    for action in actions:
        _, reward, terminated, truncated, _ = facade.step(action)
        if reward or terminated or truncated:
            raise AssertionError("declared template terminated before stop")
    _, reward, terminated, truncated, _ = facade.step({"action": "stop"})

    contents = [_canonical(action) for action in actions] + ['{"action":"stop"}']
    trace_factory = CapturingRelaxedTriggerFactory() if mutant else None
    rows, compact = run_scripted_model_actions(
        scenario, contents,
        **({"env_factory": trace_factory} if trace_factory is not None else {}),
    )
    validate_public_trace(rows)
    public_trace_text = _canonical(rows)
    private_markers = {
        scenario.private_instance_id, role,
        "conflict_scheduled", "preferred_role",
        *(event.event_id for event in scenario.private_dynamics.scheduled_events),
    }
    privacy_hits = sorted(
        marker for marker in private_markers if marker and marker in public_trace_text
    )
    same_mode_replay_valid = _replay_valid(scenario, rows, mutant=mutant)
    baseline_replay_valid = _replay_valid(scenario, rows, mutant=False)
    mutant_replay_valid = _replay_valid(scenario, rows, mutant=True)
    if not same_mode_replay_valid:
        raise AssertionError("declared template failed same-mode replay")

    initial = scenario.private_evaluator.initial_state()
    expected_event = _expected_event_dto(scenario)
    spec = build_resource_conflict_spec(
        initial, conflict_scheduled=expected_event is not None,
        expected_event=expected_event,
    )
    steps = [{
        "turn_id": index,
        "before_world": row["observation"]["world"],
        "after_world": rows[index + 1]["observation"]["world"],
        "action": json.loads(row["action_content"]),
        "result": row["tool_result"],
    } for index, row in enumerate(rows[:-1])]
    oracle = evaluate_resource_conflict(
        spec, steps, rows[-1]["observation"]["world"],
    )
    activation_count = 0
    sidecar_sha256 = None
    if mutant:
        assert trace_factory is not None
        sidecar = trace_factory.instances[-1].mutation_sidecar()
        sidecar["public_trace_sha256"] = _sha256(rows)
        activation_count = int(sidecar["activation_count"])
        sidecar_sha256 = _sha256(sidecar)
        assert facade_factory is not None
        if facade_factory.instances[-1].mutation_sidecar()["activation_count"] != activation_count:
            raise AssertionError("facade and serialized-run activation counts diverged")
    success = bool(compact["success"])
    if reward != float(success) or not terminated or truncated:
        raise AssertionError("facade and scripted terminal outcomes diverged")
    return {
        "role": role,
        "private_instance_id": scenario.private_instance_id,
        "public_task_id": scenario.public_task.task_id,
        "template_id": template_id,
        "mode": mode,
        "action_horizon_including_stop": len(contents),
        "actions_sha256": _sha256(actions),
        "public_trace_sha256": _sha256(rows),
        "public_trace_rows": len(rows),
        "public_trace_valid": True,
        "same_mode_replay_valid": same_mode_replay_valid,
        "baseline_replay_valid": baseline_replay_valid,
        "mutant_replay_valid": mutant_replay_valid,
        "facade": {
            "reward": reward, "terminated": terminated, "truncated": truncated,
        },
        "compact_result": compact,
        "oracle_spec_sha256": resource_oracle_spec_sha256(spec),
        "oracle": oracle,
        "classification": _classification(success, oracle),
        "mutation": {
            "mutation_id": RELAXED_TRIGGER_MUTATION_ID if mutant else None,
            "operator_version": (
                RELAXED_TRIGGER_OPERATOR_VERSION if mutant else None
            ),
            "activation_count": activation_count,
            "private_sidecar_sha256": sidecar_sha256,
            "private_sidecar_embedded": False,
        },
        "public_trace_privacy_scan": {
            "passed": not privacy_hits,
            "private_marker_hits": privacy_hits,
            "markers_checked": len(private_markers),
        },
    }


def run_resource_template_campaign(
    *, world_seed: int = RESOURCE_CAMPAIGN_SEED,
) -> dict[str, Any]:
    """Run every declared role/template/mode tuple; this is not graph search."""

    worlds = [
        world for world in trusted_world_catalog(world_seed)
        if world.family == "resource_calendar"
        and world.variant_id == "optimistic_conflict_replan"
    ]
    worlds.sort(key=lambda world: world.role)
    rows = []
    for world in worlds:
        templates = _ground_templates(world.scenario)
        for template_id in DECLARED_TEMPLATES:
            for mode in DECLARED_MODES:
                rows.append(_run_execution(
                    scenario=world.scenario, role=world.role,
                    template_id=template_id,
                    actions=templates[template_id], mode=mode,
                ))
    expected_keys = {
        (role, template, mode)
        for role in DECLARED_ROLES
        for template in DECLARED_TEMPLATES
        for mode in DECLARED_MODES
    }
    actual_keys = [
        (row.get("role"), row.get("template_id"), row.get("mode")) for row in rows
    ]
    matrix_shape_valid = bool(
        len(rows) == len(expected_keys)
        and len(set(actual_keys)) == len(actual_keys)
        and set(actual_keys) == expected_keys
    )
    by_key = {key: row for key, row in zip(actual_keys, rows, strict=True)}
    row_contract_matches = {
        "|".join(key): bool(
            key in FROZEN_ROW_CONTRACT_SHA256
            and _row_contract_sha256(by_key[key])
            == FROZEN_ROW_CONTRACT_SHA256[key]
        )
        for key in expected_keys if key in by_key
    }
    row_contracts_complete = bool(
        matrix_shape_valid
        and len(row_contract_matches) == len(expected_keys)
        and all(row_contract_matches.values())
    )
    role_template_pairs = {
        (row["role"], row["template_id"]) for row in rows
    }
    conflict_roles = [role for role in DECLARED_ROLES if role.endswith("conflict")]
    control_roles = [role for role in DECLARED_ROLES if role.endswith("control")]
    conflict_kills = sum(
        matrix_shape_valid
        and by_key[(role, "alternate_first_probe", "baseline")]["classification"]
        == "rejected_unsafe_or_invalid"
        and by_key[(
            role, "alternate_first_probe", RELAXED_TRIGGER_MUTATION_ID,
        )]["classification"] == "accepted_unsafe"
        and by_key[(
            role, "alternate_first_probe", RELAXED_TRIGGER_MUTATION_ID,
        )]["mutation"]["activation_count"] == 1
        for role in conflict_roles
    )
    control_false_activations = sum(
        0 if not matrix_shape_valid else
        by_key[(
            role, "alternate_first_probe", RELAXED_TRIGGER_MUTATION_ID,
        )]["mutation"]["activation_count"]
        for role in control_roles
    )
    honest_regressions = sum(
        False if not matrix_shape_valid else
        by_key[(role, "honest_completion", "baseline")]["public_trace_sha256"]
        != by_key[(
            role, "honest_completion", RELAXED_TRIGGER_MUTATION_ID,
        )]["public_trace_sha256"]
        for role in DECLARED_ROLES
    )
    symmetric = bool(matrix_shape_valid and all(
        by_key[(f"preferred_a_{suffix}", template, mode)]["classification"]
        == by_key[(f"preferred_b_{suffix}", template, mode)]["classification"]
        for suffix in ("conflict", "control")
        for template in DECLARED_TEMPLATES
        for mode in DECLARED_MODES
    ))
    expected = len(DECLARED_ROLES) * len(DECLARED_TEMPLATES) * len(DECLARED_MODES)
    summary = {
        "roles_covered": len({row["role"] for row in rows}),
        "roles_total": len(DECLARED_ROLES),
        "templates_covered": len({row["template_id"] for row in rows}),
        "templates_total": len(DECLARED_TEMPLATES),
        "mode_pairs_complete": len(role_template_pairs),
        "mode_pairs_total": len(DECLARED_ROLES) * len(DECLARED_TEMPLATES),
        "executions_completed": len(rows),
        "executions_expected": expected,
        "public_trace_rows": sum(row["public_trace_rows"] for row in rows),
        "conflict_role_detections": conflict_kills,
        "conflict_roles_total": len(conflict_roles),
        "control_false_activations": control_false_activations,
        "control_roles_total": len(control_roles),
        "honest_trace_regressions": honest_regressions,
        "classification_symmetry_a_b": symmetric,
        "oracle_out_of_scope": sum(
            row["oracle"]["status"] == "out_of_scope" for row in rows
        ),
        "integrity_failures": sum(
            not row["oracle"]["integrity_valid"] for row in rows
        ),
        "matrix_shape_valid": matrix_shape_valid,
        "row_contracts_matching": sum(row_contract_matches.values()),
        "row_contracts_total": len(expected_keys),
        "privacy_scan_failures": sum(
            not row["public_trace_privacy_scan"]["passed"] for row in rows
        ),
    }
    all_gates_pass = bool(
        world_seed == RESOURCE_CAMPAIGN_SEED
        and tuple(world.role for world in worlds) == DECLARED_ROLES
        and matrix_shape_valid
        and row_contracts_complete
        and summary["executions_completed"] == expected == 16
        and summary["mode_pairs_complete"] == summary["mode_pairs_total"] == 8
        and summary["public_trace_rows"] == 84
        and conflict_kills == len(conflict_roles) == 2
        and control_false_activations == 0
        and honest_regressions == 0
        and symmetric
        and summary["oracle_out_of_scope"] == 0
        and summary["integrity_failures"] == 0
        and summary["privacy_scan_failures"] == 0
        and all(
            row["public_trace_valid"]
            and row["same_mode_replay_valid"]
            and row["action_horizon_including_stop"] <= 6
            and _lower_hex_64(row["actions_sha256"])
            and _lower_hex_64(row["public_trace_sha256"])
            and (
                row["mutation"]["private_sidecar_sha256"] is None
                if row["mode"] == "baseline"
                else _lower_hex_64(row["mutation"]["private_sidecar_sha256"])
            )
            for row in rows
        )
    )
    return {
        "schema_version": RESOURCE_CAMPAIGN_REPORT_VERSION,
        "stage": "train",
        "source_state": _source_state(),
        "template_version": RESOURCE_TEMPLATE_VERSION,
        "oracle_version": RESOURCE_ORACLE_VERSION,
        "mutation_operator_version": RELAXED_TRIGGER_OPERATOR_VERSION,
        "scope": {
            "family": "resource_calendar",
            "variant": "optimistic_conflict_replan",
            "world_seed": world_seed,
            "frozen_world_seed": RESOURCE_CAMPAIGN_SEED,
            "hidden_roles": list(DECLARED_ROLES),
            "declared_templates": list(DECLARED_TEMPLATES),
            "execution_modes": list(DECLARED_MODES),
            "max_action_horizon_including_stop": 6,
            "template_frontier_size": 8,
            "execution_cap": 16,
            "cap_hit": False,
            "unexecuted_declared_cases": 0,
            "grounded_action_space_coverage_claimed": False,
            "full_branching_attempted": False,
            "state_merge_used": False,
            "pruning_used": False,
            "artifact_visibility": "private_audit_only",
            "public_release_allowed": False,
        },
        "summary": summary,
        "row_contract_matches_frozen_expected": dict(sorted(
            row_contract_matches.items()
        )),
        "rows": rows,
        "enumeration_complete": all_gates_pass,
        "complete": False,
        "claim_boundary": (
            "deterministic declared-witness template enumeration over one train seed, "
            "four frozen hidden roles, two templates, and two execution modes; not "
            "BFS, exhaustive or full grounded-action coverage, a full mutation campaign, "
            "a formal proof, held-out evidence, a learned policy, or Agentic RL"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the G2 resource declared-template campaign",
    )
    parser.add_argument("--world-seed", type=int, default=RESOURCE_CAMPAIGN_SEED)
    args = parser.parse_args()
    print(json.dumps(
        run_resource_template_campaign(world_seed=args.world_seed),
        ensure_ascii=False, sort_keys=True, indent=2,
    ))


if __name__ == "__main__":
    main()
