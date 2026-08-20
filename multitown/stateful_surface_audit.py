"""Reward and preterminal public-surface noninterference audit for G13--G16."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Iterable, Mapping, Sequence

from .stateful_belief_planner import (
    PublicBeliefPlanner,
    PolicyNode,
    _digest,
    _outcome_key,
    _Rollout,
    _source_state,
)
from .stateful_grounding import grounded_public_actions
from .stateful_model_protocol import (
    _changed_world_paths,
    _expected_runtime_after_tool,
    parse_model_action,
    public_trace_row,
    system_prompt,
)
from .stateful_pomdp import StatefulPOMDPEnv
from .stateful_reachability import (
    DEFAULT_AUDIT_SEEDS,
    TrustedWorld,
    _information_sets,
    trusted_world_catalog,
)


REPORT_VERSION = "multitown-stateful-surface-audit-v1"

FORBIDDEN_KEYS = frozenset({
    "private_instance_id", "private_evaluator", "private_state",
    "private_dynamics", "dynamics_branch", "world_seed", "surface_seed",
    "split_stage", "scenario_group_id", "template_cluster_id", "mechanism_id",
    "composition_signature", "structural_signature", "required_predicates",
    "forbidden_predicates", "diagnostic_partial_score", "failure_codes",
    "accepted_audit_sequences", "evaluator_hash", "initial_state_json",
    "records_truth", "conflict_scheduled", "preferred_role",
    "canary_compatible", "authority_outcome",
})
FORBIDDEN_VALUE_PREFIXES = ("a15-private-",)

REVEAL_MANIFEST: dict[tuple[str, str], dict[str, dict[str, Any]]] = {
    ("records_casework", "conflicting_evidence_investigation"): {
        "inspect_record_header": {
            "path_templates": ("records/{record_id}/header_status",),
            "public_guard": "target header_status is uninspected",
        },
        "verify_record": {
            "path_templates": (
                "records/{record_id}/eligible",
                "records/{record_id}/verification_status",
            ),
            "public_guard": (
                "target header_status is authenticated and verification_status is unknown"
            ),
        },
    },
    ("resource_calendar", "optimistic_conflict_replan"): {
        "create_versioned_hold": {
            "path_templates": (
                "resources/{resource_id}/held_for",
                "resources/{resource_id}/status",
            ),
            "public_guard": (
                "booking snapshot is fresh and binds target expected version"
            ),
        },
    },
    ("incident_recovery", "canary_compensation_saga"): {
        "probe_canary": {
            "path_templates": (
                "services/{service_id}/canary_status",
                "services/{service_id}/health",
            ),
            "public_guard": "target canary_probes equals one before the reveal probe",
        },
    },
    ("permit_transaction", "asynchronous_authority_timeout"): {
        "check_authority_status": {
            "path_templates": (
                "permits/{permit_id}/authority_status",
                "permits/{permit_id}/authority_expiry",
                "permits/{permit_id}/authority_capability",
            ),
            "public_guard": (
                "target authority_checks equals one, is pending, and request IDs match"
            ),
        },
    },
}


class SurfaceLeakError(ValueError):
    """A policy-visible JSON surface contains private evaluator metadata."""


def validate_public_surface(value: Any, path: tuple[str, ...] = ()) -> None:
    """Reject private keys/tokens and require a finite JSON round trip."""

    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise SurfaceLeakError("public JSON object key must be a string")
            if key in FORBIDDEN_KEYS:
                raise SurfaceLeakError(f"private key on public surface: {key}")
            validate_public_surface(child, (*path, key))
    elif isinstance(value, list) or isinstance(value, tuple):
        for index, child in enumerate(value):
            validate_public_surface(child, (*path, str(index)))
    elif isinstance(value, str):
        if any(prefix in value for prefix in FORBIDDEN_VALUE_PREFIXES):
            raise SurfaceLeakError("private instance token on public surface")
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, (dict, list)):
            try:
                validate_public_surface(decoded, (*path, "<decoded-json>"))
            except SurfaceLeakError as exc:
                raise SurfaceLeakError(
                    "encoded private metadata on public surface"
                ) from exc
    elif value is not None and not isinstance(value, (bool, int, float)):
        raise SurfaceLeakError("public surface is not finite JSON data")
    try:
        json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise SurfaceLeakError("public surface does not round-trip as finite JSON") from exc


def _leaf_paths(value: Any, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], Any]:
    if isinstance(value, dict):
        result: dict[tuple[str, ...], Any] = {}
        for key in sorted(value):
            result.update(_leaf_paths(value[key], (*prefix, str(key))))
        return result or {prefix: ("EMPTY_OBJECT",)}
    if isinstance(value, (list, tuple)):
        result = {}
        for index, child in enumerate(value):
            result.update(_leaf_paths(child, (*prefix, str(index))))
        return result or {prefix: ("EMPTY_ARRAY",)}
    return {prefix: value}


def _differing_world_paths(observations: Sequence[dict[str, Any]]) -> set[str]:
    flattened = [_leaf_paths(row["world"]) for row in observations]
    paths = set().union(*(row.keys() for row in flattened))
    return {
        "/".join(path) for path in paths
        if len({
            json.dumps(
                ["PRESENT", row[path]] if path in row else ["MISSING"],
                sort_keys=True,
            )
            for row in flattened
        }) > 1
    }


def _declared_reveal_paths(
    family: str, variant_id: str, action: Mapping[str, Any],
    prior_observation: Mapping[str, Any],
) -> frozenset[str]:
    """Resolve exact entity paths only when the public reveal precondition holds."""

    tool_name = action.get("tool_name")
    manifest = REVEAL_MANIFEST[(family, variant_id)]
    if tool_name not in manifest:
        raise AssertionError("public histories diverged outside a declared reveal")
    arguments = action.get("arguments")
    if not isinstance(arguments, dict):
        raise AssertionError("declared reveal action has no public arguments")
    world = prior_observation["world"]
    if tool_name in {"inspect_record_header", "verify_record"}:
        record_id = str(arguments.get("record_id", ""))
        target = world["records"].get(record_id)
        if not isinstance(target, dict):
            raise AssertionError("declared record reveal target is absent")
        if tool_name == "inspect_record_header":
            guard = target["header_status"] == "uninspected"
        else:
            guard = (
                target["header_status"] == "authenticated"
                and target["verification_status"] == "unknown"
            )
    elif tool_name == "create_versioned_hold":
        booking_id = str(arguments.get("booking_id", ""))
        resource_id = str(arguments.get("resource_id", ""))
        booking = world["bookings"].get(booking_id)
        target = world["resources"].get(resource_id)
        guard = (
            isinstance(booking, dict) and isinstance(target, dict)
            and booking["availability_fresh"] is True
            and booking["snapshot_token"] == arguments.get("snapshot_token")
            and str(booking["snapshot_versions"].get(resource_id))
            == str(arguments.get("expected_version"))
        )
    elif tool_name == "probe_canary":
        service_id = str(arguments.get("service_id", ""))
        target = world["services"].get(service_id)
        guard = (
            isinstance(target, dict)
            and target["canary_probes"] == 1
            and target["canary_status"] == "apparently-healthy"
        )
    elif tool_name == "check_authority_status":
        permit_id = str(arguments.get("permit_id", ""))
        target = world["permits"].get(permit_id)
        guard = (
            isinstance(target, dict)
            and target["authority_checks"] == 1
            and target["authority_status"] == "pending"
            and target["authority_request_id"] == arguments.get("request_id")
        )
    else:  # pragma: no cover - manifest/tool mismatch is a developer error
        raise AssertionError("declared reveal has no guard implementation")
    if not guard:
        raise AssertionError("declared reveal occurred before its public guard")
    templates = manifest[str(tool_name)]["path_templates"]
    try:
        return frozenset(template.format(**arguments) for template in templates)
    except KeyError as exc:
        raise AssertionError("declared reveal action is missing its target ID") from exc


@dataclass(frozen=True)
class _Step:
    observation: dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]
    validation_calls: int


def _replay(
    world: TrustedWorld, actions: Sequence[Mapping[str, Any]],
) -> _Step:
    env = StatefulPOMDPEnv(world.scenario)
    observation, reset_info = env.reset()
    validate_public_surface({
        "observation": observation, "reset_info": reset_info,
        "spec": env.spec, "action_contract": env.action_contract,
        "system_prompt": system_prompt(world.family),
        "grounded_actions": grounded_public_actions(observation),
    })
    reward = 0.0
    terminated = truncated = False
    info: dict[str, Any] = reset_info
    for action in actions:
        observation, reward, terminated, truncated, info = env.step(action)
        validate_public_surface({
            "observation": observation, "reward": reward,
            "terminated": terminated, "truncated": truncated, "info": info,
            "grounded_actions": (
                [] if terminated or truncated
                else grounded_public_actions(observation)
            ),
        })
    return _Step(
        observation, reward, terminated, truncated, info,
        validation_calls=1 + len(actions),
    )


def _surface_key(step: _Step) -> str:
    return _digest(_public_step_surface(step))


def _public_step_surface(step: _Step) -> dict[str, Any]:
    return {
        "observation": step.observation,
        "reward": step.reward,
        "terminated": step.terminated,
        "truncated": step.truncated,
        "info": step.info,
        "grounded_actions": (
            [] if step.terminated or step.truncated
            else grounded_public_actions(step.observation)
        ),
    }


def _validate_transition_surface(
    family: str, prior: _Step, action: Mapping[str, Any], advanced: _Step,
) -> None:
    """Apply strict row schema and public delta checks to a replay transition."""

    action_content = json.dumps(
        action, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )
    public_trace_row(
        session_id="surface-audit", turn_id=0, family=family,
        observation=prior.observation, action_content=action_content,
        tool_result=advanced.info.get("tool_result"),
        terminal_result=advanced.info.get("terminal_result"),
    )
    parsed = parse_model_action(action_content, family=family)
    if parsed.action == "stop":
        expected_observation = copy.deepcopy(prior.observation)
        expected_observation["terminal"] = True
        expected_observation["steps"] += 1
        expected_observation["runtime"]["actor_event_counts"]["agent"] += 1
        expected_observation["runtime_hash"] = _digest(
            expected_observation["runtime"]
        )
        if advanced.observation != expected_observation:
            raise AssertionError("stop observation is not the exact terminal projection")
        terminal_result = advanced.info.get("terminal_result")
        if not isinstance(terminal_result, dict):
            raise AssertionError("stop has no compact terminal result")
        zero_cost = {
            "tool_calls": 0, "steps": 1, "logical_latency": 0,
            "irreversible_risk": 0, "attempted_policy_violations": 0,
            "executed_safety_violations": 0, "budget_violations": 0,
        }
        if advanced.info != {
            "schema_version": "multitown-stateful-gym-adapter-v1",
            "cost": zero_cost, "tool_result": None,
            "terminal_result": terminal_result,
            "termination_reason": "agent_stop",
        }:
            raise AssertionError("stop info is not the exact terminal envelope")
        if not (
            advanced.terminated and not advanced.truncated
            and advanced.reward == float(terminal_result["success"])
        ):
            raise AssertionError("stop flags/reward do not match terminal success")
        return
    public_trace_row(
        session_id="surface-audit-post", turn_id=0, family=family,
        observation=advanced.observation, action_content='{"action":"stop"}',
        tool_result=None, terminal_result={
            "terminal": True, "success": False,
            "safety_violations": 0, "budget_violations": 0,
        },
    )
    tool_result = advanced.info["tool_result"]
    if (
        tool_result["state_hash"] != advanced.observation["state_hash"]
        or tool_result["runtime_hash"] != advanced.observation["runtime_hash"]
    ):
        raise AssertionError("tool result hashes do not bind the post-action surface")
    expected_runtime = _expected_runtime_after_tool(
        prior.observation, advanced.observation["world"], parsed, tool_result,
    )
    if expected_runtime != advanced.observation["runtime"]:
        raise AssertionError("post-action runtime is not the public transition delta")
    expected_cost = {
        "tool_calls": (
            prior.observation["tool_calls_remaining"]
            - advanced.observation["tool_calls_remaining"]
        ),
        "steps": advanced.observation["steps"] - prior.observation["steps"],
        **{
            name: (
                advanced.observation["runtime"][runtime_key]
                - prior.observation["runtime"][runtime_key]
            )
            for name, runtime_key in {
                "logical_latency": "logical_latency_used",
                "irreversible_risk": "irreversible_risk_used",
                "attempted_policy_violations": "attempted_policy_violations",
                "executed_safety_violations": "executed_safety_violations",
                "budget_violations": "budget_violations",
            }.items()
        },
    }
    if advanced.info != {
        "schema_version": "multitown-stateful-gym-adapter-v1",
        "cost": expected_cost, "tool_result": tool_result,
        "terminal_result": None, "termination_reason": None,
    }:
        raise AssertionError("step info is not exactly derived from the transition")


def _validate_reveal_semantics(
    family: str, action: Mapping[str, Any], prior: _Step, advanced: _Step,
) -> None:
    """Bind a reveal's values and actor attribution to public transition semantics."""

    tool_name = str(action["tool_name"])
    arguments = action["arguments"]
    result = advanced.info["tool_result"]
    transition = result["transition"]
    actual = set(_changed_world_paths(
        prior.observation["world"], advanced.observation["world"],
    ))
    agent_paths = set(transition["agent_changed_objects"])
    external_paths = {
        path for event in transition["external_events"]
        for path in event["changed_objects"]
    }
    if actual != agent_paths | external_paths or agent_paths & external_paths:
        raise AssertionError("reveal attribution does not partition its world delta")
    if family in {"records_casework", "incident_recovery"}:
        if external_paths or agent_paths != actual:
            raise AssertionError("environment-step reveal must be agent-attributed")
        if transition["external_events"]:
            raise AssertionError("environment-step reveal cannot forge external events")
    if family == "records_casework":
        record = advanced.observation["world"]["records"][arguments["record_id"]]
        if tool_name == "inspect_record_header":
            valid_value = record["header_status"] in {"authenticated", "superseded"}
        else:
            valid_value = (
                isinstance(record["eligible"], bool)
                and record["verification_status"]
                == ("eligible" if record["eligible"] else "ineligible")
            )
        if not valid_value:
            raise AssertionError("record reveal value is outside its public semantics")
    elif family == "incident_recovery":
        service = advanced.observation["world"]["services"][arguments["service_id"]]
        if (service["canary_status"], service["health"]) not in {
            ("validated", "recovering"), ("regressed", "degraded"),
        }:
            raise AssertionError("canary reveal value is outside its public semantics")
    elif family == "resource_calendar":
        resource_id = arguments["resource_id"]
        resource = advanced.observation["world"]["resources"][resource_id]
        changed = {
            f"resources/{resource_id}/{leaf}"
            for leaf in ("held_for", "status", "version")
        }
        if result["result"] == "conflict":
            valid = (
                result["error_code"] == "VERSION_CONFLICT"
                and result["payload"] == resource
                and not agent_paths and external_paths == changed
                and resource == {
                    "capability": "inspector", "status": "booked",
                    "held_for": "external-booking",
                    "version": int(arguments["expected_version"]) + 1,
                }
                and len(transition["external_events"]) == 1
                and transition["external_events"][0]["actor"] == "system"
                and transition["external_events"][0]["phase"] == "before_action"
            )
        else:
            valid = (
                result["result"] == "ok" and result["error_code"] is None
                and result["payload"] is None and agent_paths == changed
                and not external_paths and not transition["external_events"]
                and resource == {
                    "capability": "inspector", "status": "held",
                    "held_for": arguments["booking_id"],
                    "version": int(arguments["expected_version"]) + 1,
                }
            )
        if not valid:
            raise AssertionError("versioned hold reveal is not its exact public outcome")
    elif family == "permit_transaction":
        permit_id = arguments["permit_id"]
        permit = advanced.observation["world"]["permits"][permit_id]
        agent_expected = {f"permits/{permit_id}/authority_checks"}
        status = permit["authority_status"]
        external_expected = {f"permits/{permit_id}/authority_status"}
        if status == "granted":
            external_expected |= {
                f"permits/{permit_id}/authority_capability",
                f"permits/{permit_id}/authority_expiry",
            }
            value_valid = (
                permit["authority_capability"] is not None
                and permit["authority_expiry"] > transition["logical_tick"]
            )
        else:
            value_valid = (
                status in {"denied", "timed-out"}
                and permit["authority_capability"] is None
                and permit["authority_expiry"] == 0
            )
        events = transition["external_events"]
        if not (
            result["result"] == "ok" and result["error_code"] is None
            and result["payload"] is None and agent_paths == agent_expected
            and external_paths == external_expected and value_valid
            and len(events) == len(external_expected)
            and all(
                event["actor"] == "authority"
                and event["phase"] == "after_action"
                and event["logical_tick"] == transition["logical_tick"]
                and len(event["changed_objects"]) == 1
                for event in events
            )
        ):
            raise AssertionError("authority reveal is not its exact public outcome")


def _planner_outcome(step: _Step) -> str:
    return _outcome_key(_Rollout(
        observation=step.observation,
        tool_result=step.info.get("tool_result"),
        terminal_result=step.info.get("terminal_result"),
        terminated=step.terminated,
        truncated=step.truncated,
    ))


def _audit_tree(
    worlds: tuple[TrustedWorld, ...], root: PolicyNode,
) -> dict[str, Any]:
    stack = [(root, tuple((world, ()) for world in worlds))]
    nodes = nonterminal = reveal_nodes = terminal_nodes = 0
    common_surface_checks = zero_reward_checks = strict_terminal_checks = 0
    reveal_tool_checks = reveal_path_checks = transition_surface_checks = 0
    terminal_surface_checks = 0
    public_surfaces = 0
    public_surface_validation_calls = 0
    while stack:
        node, members = stack.pop()
        nodes += 1
        current = [_replay(world, history) for world, history in members]
        public_surfaces += len(current)
        public_surface_validation_calls += sum(
            step.validation_calls for step in current
        )
        if len({_surface_key(step) for step in current}) != 1:
            raise AssertionError("one policy node has non-identical public surfaces")
        common_surface_checks += 1
        if node.observation_sha256 != _digest(current[0].observation):
            raise AssertionError("policy node is not bound to the common observation")
        advanced = [
            (world, (*history, node.action), _replay(world, (*history, node.action)))
            for world, history in members
        ]
        public_surfaces += len(advanced)
        public_surface_validation_calls += sum(
            step.validation_calls for _, _, step in advanced
        )
        for index, (_, _, step) in enumerate(advanced):
            _validate_transition_surface(
                worlds[0].family, current[index], node.action, step,
            )
            transition_surface_checks += 1
        if node.terminal_success:
            terminal_nodes += 1
            if not all(
                step.terminated and not step.truncated and step.reward == 1.0
                and step.info["terminal_result"]["success"]
                for _, _, step in advanced
            ):
                raise AssertionError("terminal reward is not strict success-only")
            if len({_surface_key(step) for _, _, step in advanced}) != 1:
                raise AssertionError(
                    "compatible worlds exposed different terminal policy surfaces"
                )
            strict_terminal_checks += 1
            terminal_surface_checks += 1
            continue
        nonterminal += 1
        if any(
            step.reward != 0.0 or step.terminated or step.truncated
            for _, _, step in advanced
        ):
            raise AssertionError("nonterminal transition leaked terminal reward")
        zero_reward_checks += len(advanced)
        partitions: dict[str, list[tuple[TrustedWorld, tuple[Any, ...]]]] = {}
        observations: dict[str, list[dict[str, Any]]] = {}
        for world, history, step in advanced:
            outcome = _planner_outcome(step)
            partitions.setdefault(outcome, []).append((world, history))
            observations.setdefault(outcome, []).append(step.observation)
        if len(partitions) > 1:
            reveal_nodes += 1
            reveal_tool_checks += 1
            all_observations = [step for rows in observations.values() for step in rows]
            allowed = _declared_reveal_paths(
                worlds[0].family, worlds[0].variant_id, node.action,
                current[0].observation,
            )
            differing = _differing_world_paths(all_observations)
            if not differing or not differing <= allowed:
                raise AssertionError(
                    "reveal changed undeclared business paths: "
                    f"{sorted(differing - allowed)}"
                )
            reveal_path_checks += 1
            for index, (_, _, step) in enumerate(advanced):
                _validate_reveal_semantics(
                    worlds[0].family, node.action, current[index], step,
                )
        if set(partitions) != set(node.outcomes):
            raise AssertionError("policy outcomes do not equal observed public partitions")
        for outcome, partition in partitions.items():
            stack.append((node.outcomes[outcome], tuple(partition)))
    return {
        "policy_nodes": nodes, "nonterminal_nodes": nonterminal,
        "terminal_nodes": terminal_nodes, "declared_reveal_nodes": reveal_nodes,
        "public_surface_instances_checked": public_surfaces,
        "public_surface_validation_calls": public_surface_validation_calls,
        "common_surface_checks": common_surface_checks,
        "zero_reward_transition_checks": zero_reward_checks,
        "strict_terminal_reward_checks": strict_terminal_checks,
        "reveal_tool_checks": reveal_tool_checks,
        "reveal_path_checks": reveal_path_checks,
        "transition_surface_checks": transition_surface_checks,
        "terminal_surface_checks": terminal_surface_checks,
    }


def _actual_surface_carriers(world: TrustedWorld) -> dict[str, Any]:
    """Collect representative objects from each real policy-facing exit."""

    env = StatefulPOMDPEnv(world.scenario)
    observation, reset_info = env.reset()
    grounded = grounded_public_actions(observation)
    first_tool = next(action for action in grounded if action["action"] == "call_tool")
    next_observation, reward, terminated, truncated, step_info = env.step(first_tool)
    return {
        "reset_observation": observation,
        "reset_info": reset_info,
        "spec": env.spec,
        "action_contract": env.action_contract,
        "system_prompt": {"text": system_prompt(world.family)},
        "grounded_action": grounded[0],
        "step_observation": next_observation,
        "step_info": step_info,
        "step_envelope": {
            "reward": reward, "terminated": terminated, "truncated": truncated,
        },
    }


def _replace_first_string(value: Any, replacement: str) -> bool:
    if isinstance(value, dict):
        for key in value:
            if isinstance(value[key], str):
                value[key] = replacement
                return True
            if _replace_first_string(value[key], replacement):
                return True
    elif isinstance(value, list):
        for index, child in enumerate(value):
            if isinstance(child, str):
                value[index] = replacement
                return True
            if _replace_first_string(child, replacement):
                return True
    return False


def _mutation_controls(worlds: Sequence[TrustedWorld]) -> dict[str, Any]:
    """Plant real private values into actual public carriers and require detection."""

    private_values: dict[str, Any] = {}
    for world in worlds:
        private_values.update(
            world.scenario.private_dynamics.initial_private_state()
        )
    representatives = {world.family: world for world in worlds}
    carriers: dict[str, Any] = {}
    for family, world in sorted(representatives.items()):
        for name, value in _actual_surface_carriers(world).items():
            carriers[f"{family}:{name}"] = value
    detected: list[str] = []
    planted = 0
    encoded_planted = encoded_detected = 0
    actual_private_keys = frozenset(private_values)
    control_keys = FORBIDDEN_KEYS | actual_private_keys
    for key in sorted(control_keys):
        private_value = copy.deepcopy(private_values.get(key, "planted-secret"))
        for carrier_name, carrier in carriers.items():
            planted += 1
            attacked = copy.deepcopy(carrier)
            attacked[key] = private_value
            try:
                validate_public_surface(attacked)
            except SurfaceLeakError:
                detected.append(f"{carrier_name}:{key}")
            encoded = copy.deepcopy(carrier)
            if _replace_first_string(
                encoded, json.dumps({key: private_value}, sort_keys=True),
            ):
                planted += 1
                encoded_planted += 1
                try:
                    validate_public_surface(encoded)
                except SurfaceLeakError:
                    encoded_detected += 1
                    detected.append(f"{carrier_name}:{key}:encoded")
    for family, world in sorted(representatives.items()):
        planted += 1
        try:
            validate_public_surface(
                world.scenario.private_instance_id + system_prompt(family)
            )
        except SurfaceLeakError:
            detected.append(f"{family}:system_prompt:private_instance_token")
    return {
        "control_types": len(control_keys) + 1,
        "actual_private_dynamic_keys": sorted(actual_private_keys),
        "uncovered_private_dynamic_keys": sorted(
            actual_private_keys - FORBIDDEN_KEYS
        ),
        "carrier_types": len(carriers) + len(representatives),
        "encoded_value_controls": {
            "planted": encoded_planted, "detected": encoded_detected,
        },
        "planted": planted, "detected": len(detected),
        "all_detected": len(detected) == planted,
        "controls": detected,
    }


def audit_surface_seed(world_seed: int) -> dict[str, Any]:
    catalog = trusted_world_catalog(world_seed)
    information_sets = [
        (set_id, members)
        for set_id, members in _information_sets(catalog)
        if len(members) > 1
    ]
    rows = []
    for set_id, worlds in information_sets:
        planner = PublicBeliefPlanner(
            (world.scenario for world in worlds),
            max_depth=max(
                world.scenario.public_task.budget.max_steps for world in worlds
            ),
            grounder=grounded_public_actions,
        )
        tree = planner.solve()
        if tree is None:
            raise AssertionError("positive planner must solve before surface audit")
        initial = [_replay(world, ()) for world in worlds]
        initial_equal = len({_surface_key(step) for step in initial}) == 1
        pairwise = sum(1 for _ in combinations(worlds, 2))
        rows.append({
            "information_set_id": set_id,
            "family": worlds[0].family, "variant_id": worlds[0].variant_id,
            "world_count": len(worlds), "pairwise_world_comparisons": pairwise,
            "initial_surface_equal": initial_equal,
            **_audit_tree(worlds, tree),
        })
    mutations = _mutation_controls(catalog)
    totals = {
        key: sum(row[key] for row in rows)
        for key in (
            "common_surface_checks", "zero_reward_transition_checks",
            "strict_terminal_reward_checks", "reveal_tool_checks",
            "reveal_path_checks", "transition_surface_checks",
            "terminal_surface_checks", "public_surface_validation_calls",
        )
    }
    gates = {
        "five_multiworld_sets_present": len(rows) == 5,
        "all_initial_public_surfaces_equal": all(
            row["initial_surface_equal"] for row in rows
        ),
        "common_history_surface_checks_executed": totals[
            "common_surface_checks"
        ] > 0,
        "zero_nonterminal_reward_checks_executed": totals[
            "zero_reward_transition_checks"
        ] > 0,
        "strict_terminal_reward_checks_executed": totals[
            "strict_terminal_reward_checks"
        ] > 0,
        "strict_terminal_surface_checks_executed": totals[
            "terminal_surface_checks"
        ] > 0,
        "declared_reveal_tool_checks_executed": totals[
            "reveal_tool_checks"
        ] > 0,
        "declared_reveal_path_checks_executed": totals[
            "reveal_path_checks"
        ] > 0,
        "strict_transition_surface_checks_executed": totals[
            "transition_surface_checks"
        ] > 0,
        "public_surface_validation_executed": sum(
            row["public_surface_validation_calls"] for row in rows
        ) > 0,
        "all_mutation_controls_detected": mutations["all_detected"],
    }
    return {
        "world_seed": world_seed,
        "counts": {
            "multiworld_information_sets": len(rows),
            "hidden_worlds": sum(row["world_count"] for row in rows),
            "pairwise_world_comparisons": sum(
                row["pairwise_world_comparisons"] for row in rows
            ),
            "public_surface_instances_checked": sum(
                row["public_surface_instances_checked"] for row in rows
            ),
        },
        "information_sets": rows,
        "check_counts": totals,
        "mutation_controls": mutations,
        "gates": gates,
    }


def run_surface_audit(
    *, world_seeds: Iterable[int] = DEFAULT_AUDIT_SEEDS,
) -> dict[str, Any]:
    seeds = tuple(world_seeds)
    if not seeds or len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        raise ValueError("world seeds must be a non-empty unique non-negative sequence")
    rows = [audit_surface_seed(seed) for seed in seeds]
    return {
        "schema_version": REPORT_VERSION,
        "stage": "train", "source_state": _source_state(),
        "audit_seeds": list(seeds),
        "aggregate_counts": {
            "multiworld_information_sets": sum(
                row["counts"]["multiworld_information_sets"] for row in rows
            ),
            "hidden_worlds": sum(row["counts"]["hidden_worlds"] for row in rows),
            "pairwise_world_comparisons": sum(
                row["counts"]["pairwise_world_comparisons"] for row in rows
            ),
            "public_surface_instances_checked": sum(
                row["counts"]["public_surface_instances_checked"] for row in rows
            ),
        },
        "all_gates_pass": all(all(row["gates"].values()) for row in rows),
        "rows": rows,
        "threat_model": (
            "trusted runner owns scenario/env objects; policy receives only finite JSON "
            "observations, prompts, actions, results, rewards, and terminal summaries"
        ),
        "remaining_boundary": (
            "audits synthesized positive policy histories and planted metadata leaks; "
            "does not yet exhaust invalid/idempotency partitions, arbitrary histories, "
            "or accepted-unsafe trajectories"
        ),
        "claim_boundary": (
            "train-only bounded reward/public-surface noninterference audit over declared "
            "positive histories; not formal noninterference, model evaluation, held-out "
            "generalization, policy training, or Agentic RL evidence"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit stateful reward and public-surface noninterference",
    )
    parser.add_argument(
        "--world-seeds", type=int, nargs="+", default=list(DEFAULT_AUDIT_SEEDS),
    )
    args = parser.parse_args()
    print(json.dumps(
        run_surface_audit(world_seeds=args.world_seeds),
        ensure_ascii=False, indent=2, sort_keys=True,
    ))


if __name__ == "__main__":
    main()
