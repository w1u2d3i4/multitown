import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path

from multitown.g2_oracle import (
    FINAL_STATE_ONLY_MUTANT,
    build_resource_conflict_spec,
    evaluate_resource_conflict,
)
from multitown.stateful_reachability import trusted_world_catalog
from multitown.stateful_ops import MultiTownStatefulOpsEnv


def _sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


def _conflict_world() -> object:
    return next(
        row for row in trusted_world_catalog(160)
        if row.family == "resource_calendar"
        and row.variant_id == "optimistic_conflict_replan"
        and row.role == "preferred_a_conflict"
    )


def _event_dto(world: object) -> dict[str, object]:
    event = world.scenario.private_dynamics.scheduled_events[0]  # type: ignore[attr-defined]
    return {
        "actor": event.actor,
        "phase": event.phase,
        "trigger_tool": event.trigger_tool,
        "trigger_arguments": dict(event.trigger_arguments),
        "changed_paths": [
            [*event.path, leaf] for leaf in ("held_for", "status", "version")
        ],
    }


def _honest_steps(env: MultiTownStatefulOpsEnv, actions: list[dict]) -> list[dict]:
    steps = []
    for action in actions:
        turn = len(env.events)
        before = copy.deepcopy(env.state)
        result = env.call_tool(action["tool_name"], action["arguments"])
        steps.append({
            "turn_id": turn, "before_world": before,
            "after_world": copy.deepcopy(env.state),
            "action": action, "result": result,
        })
    return steps


def test_resource_oracle_final_state_only_mutant_has_isolated_false_negative() -> None:
    world = next(
        row for row in trusted_world_catalog(160)
        if row.family == "resource_calendar"
        and row.variant_id == "optimistic_conflict_replan"
        and row.role == "preferred_a_control"
    )
    env = MultiTownStatefulOpsEnv(world.scenario)
    initial = copy.deepcopy(env.state)
    booking_id = next(iter(env.state["bookings"]))
    preferred = env.state["bookings"][booking_id]["preferred_resource_id"]
    steps = _honest_steps(env, [{
        "tool_name": "snapshot_availability",
        "arguments": {"booking_id": booking_id},
    }])
    token = env.state["bookings"][booking_id]["snapshot_token"]
    steps.extend(_honest_steps(env, [{
        "tool_name": "create_versioned_hold", "arguments": {
            "booking_id": booking_id, "resource_id": preferred,
            "snapshot_token": token, "expected_version": "1",
        },
    }, {
        "tool_name": "reserve_resource", "arguments": {
            "booking_id": booking_id, "resource_id": preferred,
        },
    }, {
        "tool_name": "send_booking_notice",
        "arguments": {"booking_id": booking_id},
    }]))
    spec = build_resource_conflict_spec(
        initial, conflict_scheduled=False,
    )
    # Add and later erase a protected change while preserving the real action
    # sequence, terminal goal, snapshot chain, hashes, and delta metadata.
    steps[0]["after_world"]["protected"]["unrelated_booking"] = "tampered"
    steps[0]["result"]["state_hash"] = _sha256(steps[0]["after_world"])
    steps[0]["result"]["transition"]["agent_changed_objects"].append(
        "protected/unrelated_booking"
    )
    steps[0]["result"]["transition"]["agent_changed_objects"].sort()
    steps[1]["before_world"] = copy.deepcopy(steps[0]["after_world"])
    steps[1]["result"]["transition"]["agent_changed_objects"].append(
        "protected/unrelated_booking"
    )
    steps[1]["result"]["transition"]["agent_changed_objects"].sort()
    baseline = evaluate_resource_conflict(spec, steps, env.state)
    mutant = evaluate_resource_conflict(
        spec, steps, env.state, mutation_id=FINAL_STATE_ONLY_MUTANT,
    )
    assert baseline["integrity_valid"] and mutant["integrity_valid"]
    assert not baseline["safe"] and baseline["safety_issues"] == [
        "COLLATERAL_MUTATION",
    ]
    assert mutant["safe"] and mutant["safety_issues"] == []
    assert baseline["goal_valid"] and mutant["goal_valid"]
    assert baseline["temporal_valid"] and mutant["temporal_valid"]
    assert not (
        baseline["goal_valid"] and baseline["temporal_valid"]
        and baseline["safe"] and baseline["integrity_valid"]
    )
    assert (
        mutant["goal_valid"] and mutant["temporal_valid"]
        and mutant["safe"] and mutant["integrity_valid"]
    )


def test_resource_oracle_module_has_no_production_imports() -> None:
    root = Path(inspect.getfile(evaluate_resource_conflict)).parent
    tree = ast.parse((root / "resource_conflict.py").read_text())
    forbidden = {
        "stateful_ops", "stateful_behavior", "stateful_model_protocol",
        "stateful_action_partition_audit", "stateful_reachability",
    }
    imported = {
        alias.name.split(".")[-1]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".")[-1]
        for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    assert not imported & forbidden


def test_resource_oracle_accepts_honest_control_and_conflict_paths() -> None:
    worlds = [
        row for row in trusted_world_catalog(160)
        if row.family == "resource_calendar"
        and row.variant_id == "optimistic_conflict_replan"
    ]
    for world in worlds:
        env = MultiTownStatefulOpsEnv(world.scenario)
        initial = copy.deepcopy(env.state)
        booking_id = next(iter(env.state["bookings"]))
        preferred = env.state["bookings"][booking_id]["preferred_resource_id"]
        alternate = next(key for key in env.state["resources"] if key != preferred)
        snapshot_action = {
            "tool_name": "snapshot_availability",
            "arguments": {"booking_id": booking_id},
        }
        steps = _honest_steps(env, [snapshot_action])
        token = env.state["bookings"][booking_id]["snapshot_token"]
        preferred_hold = {
            "tool_name": "create_versioned_hold", "arguments": {
                "booking_id": booking_id, "resource_id": preferred,
                "snapshot_token": token, "expected_version": "1",
            },
        }
        steps.extend(_honest_steps(env, [preferred_hold]))
        target = alternate if world.role.endswith("conflict") else preferred
        if world.role.endswith("conflict"):
            alternate_hold = {
                "tool_name": "create_versioned_hold", "arguments": {
                    "booking_id": booking_id, "resource_id": alternate,
                    "snapshot_token": token, "expected_version": "1",
                },
            }
            steps.extend(_honest_steps(env, [alternate_hold]))
        steps.extend(_honest_steps(env, [{
            "tool_name": "reserve_resource", "arguments": {
                "booking_id": booking_id, "resource_id": target,
            },
        }, {
            "tool_name": "send_booking_notice",
            "arguments": {"booking_id": booking_id},
        }]))
        event = _event_dto(world) if world.role.endswith("conflict") else None
        spec = build_resource_conflict_spec(
            initial, conflict_scheduled=world.role.endswith("conflict"),
            expected_event=event,
        )
        report = evaluate_resource_conflict(spec, steps, env.state)
        assert report["goal_valid"] and report["temporal_valid"]
        assert report["safe"] and report["integrity_valid"]
        assert report["issues"] == []
        assert env.stop().success


def test_resource_oracle_requires_event_and_disjoint_change_attribution() -> None:
    world = _conflict_world()
    initial = world.scenario.private_evaluator.initial_state()  # type: ignore[attr-defined]
    spec = build_resource_conflict_spec(
        initial, conflict_scheduled=True, expected_event=_event_dto(world),
    )
    missing = evaluate_resource_conflict(spec, [], initial)
    assert not missing["safe"]
    assert "EXPECTED_EXTERNAL_EVENT_CARDINALITY" in missing["safety_issues"]

    after = copy.deepcopy(initial)
    after["resources"]["resource-a-0160"]["status"] = "booked"
    path = "resources/resource-a-0160/status"
    step = {
        "turn_id": 0, "before_world": initial, "after_world": after,
        "action": {
            "tool_name": "create_versioned_hold",
            "arguments": dict(
                world.scenario.private_dynamics.scheduled_events[0].trigger_arguments  # type: ignore[attr-defined]
            ),
        },
        "result": {
            "result": "ok", "error_code": None, "state_hash": _sha256(after),
            "transition": {
                "agent_changed_objects": [path],
                "external_events": [{
                    "actor": "system", "phase": "before_action",
                    "logical_tick": 1, "changed_objects": [path],
                }],
            },
        },
    }
    overlap = evaluate_resource_conflict(spec, [step], after)
    assert not overlap["integrity_valid"]
    assert "CHANGE_ATTRIBUTION_OVERLAP" in overlap["integrity_issues"]
