"""Finite one-step action, lifecycle, and idempotency partition audit.

This is a trusted, train-only verification runner.  Its completeness claim is
strictly relative to :data:`ACTION_PARTITION_MANIFEST`; it is not exhaustive
over arbitrary JSON strings or arbitrary histories.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import inspect
import json
import textwrap
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Sequence

from .stateful_belief_planner import (
    PublicBeliefPlanner, PolicyNode, _digest, _source_state,
)
from .stateful_grounding import grounded_public_actions
from .stateful_model_protocol import (
    parse_model_action, public_trace_row, validate_public_trace,
)
from .stateful_ops import (
    IDEMPOTENCY_KEY_MAX_UTF8_BYTES,
    READ_TOOLS,
    TOOL_SCHEMAS,
    CheckerDecision,
    MultiTownStatefulOpsEnv,
)
from .stateful_reachability import (
    DEFAULT_AUDIT_SEEDS,
    TrustedWorld,
    _information_sets,
    trusted_world_catalog,
)
from .stateful_surface_audit import (
    _Step,
    _declared_reveal_paths,
    _differing_world_paths,
    _public_step_surface,
    _replay,
    _surface_key,
    _validate_reveal_semantics,
    _validate_transition_surface,
    validate_public_surface,
)


REPORT_VERSION = "multitown-stateful-action-partition-report-v2"
MANIFEST_VERSION = "multitown-stateful-action-partition-manifest-v2"
CHECKER_REASON_WAIVERS = {
    "INVALID_ARGUMENT_SCHEMA": (
        "unreachable through StatefulPOMDPEnv because parse_model_action enforces the "
        "exact tool schema before PolicySession.call_tool; covered by parser classes"
    ),
}

REFERENCE_TABLES = {
    "case_id": "cases", "record_id": "records", "evidence_id": "records",
    "applicant_id": "applicants", "permit_id": "permits",
    "booking_id": "bookings", "resource_id": "resources",
    "incident_id": "incidents", "service_id": "services",
}
OPAQUE_SLOTS = {
    "request_id", "capability", "snapshot_token", "deployment_id",
    "compensation_token",
}
VERSION_SLOTS = {
    "expected_policy_version", "policy_version", "expected_version", "expiry",
}
ENUM_SLOTS = {"decision", "status", "scope"}

ACTION_PARTITION_MANIFEST: dict[str, Any] = {
    "schema_version": MANIFEST_VERSION,
    "history_scope": (
        "initial states and every public belief node on the finite robust positive "
        "policy trees for the frozen train catalog"
    ),
    "combination_rule": (
        "one-factor value-class perturbations at every argument occurrence; all "
        "grounded calls at every scoped lifecycle checkpoint; no full Cartesian product"
    ),
    "parse_classes": [
        "valid_null_key", "valid_fresh_key", "invalid_json", "non_object_root",
        "missing_top_field", "extra_top_field", "duplicate_top_field",
        "unknown_or_cross_family_tool", "arguments_not_object",
        "missing_argument", "extra_argument", "wrong_argument_type",
        "duplicate_argument", "nonfinite_argument", "invalid_idempotency_type",
        "empty_idempotency_key", "non_nfc_idempotency_key",
        "control_idempotency_key", "overlength_idempotency_key",
    ],
    "slot_value_classes": {
        "reference": [
            "visible_current", "other_visible_same_type", "visible_cross_type",
            "well_formed_absent", "malformed_or_empty",
        ],
        "opaque_reference": [
            "visible_current", "stale_or_other_object",
            "malformed_or_empty",
        ],
        "version": ["visible_current", "previous_or_future", "malformed_or_empty"],
        "enum": ["visible_current", "unknown", "malformed_or_empty"],
        "string": ["visible_current", "unknown", "malformed_or_empty"],
    },
    "stable_semantic_outcomes": [
        "MALFORMED_REFERENCE", "NOT_FOUND_OR_OUT_OF_SCOPE",
        "FAILED_PRECONDITION", "ABORTED_OR_CONFLICT", "ALLOW",
    ],
    "idempotency_classes": [
        "null", "fresh", "exact_replay", "exact_replay_after_state_advance",
        "same_key_different_args", "same_key_different_tool", "keyed_read_snapshot",
        "keyed_ok", "keyed_noop", "keyed_blocked", "keyed_conflict",
        "blocked_public_trace_replay",
        "budget_precheck_excluded", "unicode_boundary", "length_boundary",
        "ttl_unsupported",
    ],
    "mutation_controls": [
        "failed_stop_reward_leak", "blocked_hidden_error_divergence",
        "blocked_private_payload", "cross_object_event_target_binding",
    ],
    "execution_boundary": {
        "parse_or_schema_rejection": "not started; no runtime cache record",
        "budget_precheck": "not started; no runtime cache record",
        "checker_and_business_outcomes": (
            "started; first ok/noop/blocked/conflict business result is snapshot-cached"
        ),
        "replay_cost": (
            "exact replay does not repeat agent business mutation or risk, but consumes "
            "one public tool call, one step, and declared latency; it does not reexecute "
            "external events"
        ),
        "ttl": "unsupported in finite episodes",
    },
    "unknown_read_semantics": "ok with payload null; declared, not classified as blocked",
}


def _canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


def manifest_sha256() -> str:
    return hashlib.sha256(_canonical(ACTION_PARTITION_MANIFEST).encode()).hexdigest()


def _slot_kind(name: str) -> str:
    if name in REFERENCE_TABLES:
        return "reference"
    if name in OPAQUE_SLOTS:
        return "opaque_reference"
    if name in VERSION_SLOTS:
        return "version"
    if name in ENUM_SLOTS:
        return "enum"
    return "string"


def _all_grounded_tool_actions(
    observation: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for action in grounded_public_actions(observation):
        if action["action"] == "call_tool":
            key = _canonical(action)
            if key not in seen:
                seen.add(key)
                rows.append(copy.deepcopy(action))
    return tuple(sorted(rows, key=_canonical))


def _all_public_ids(observation: Mapping[str, Any]) -> dict[str, list[str]]:
    world = observation["world"]
    return {
        table: sorted(str(value) for value in world.get(table, {}))
        for table in set(REFERENCE_TABLES.values())
    }


@dataclass(frozen=True)
class ActionRepresentative:
    partition_class: str
    slot: str | None
    concrete_value: str | None
    action: dict[str, Any]


def _runtime_representatives(
    observation: Mapping[str, Any],
) -> tuple[ActionRepresentative, ...]:
    ids = _all_public_ids(observation)
    all_ids = sorted({item for values in ids.values() for item in values})
    rows: list[ActionRepresentative] = []
    seen: set[str] = set()

    def add(
        partition_class: str, action: dict[str, Any],
        *, slot: str | None = None, value: str | None = None,
    ) -> None:
        key = _canonical(action)
        if key not in seen:
            seen.add(key)
            rows.append(ActionRepresentative(partition_class, slot, value, action))

    for baseline in _all_grounded_tool_actions(observation):
        add("visible_current", baseline)
        for slot, current in sorted(baseline["arguments"].items()):
            kind = _slot_kind(slot)
            candidates: list[tuple[str, str]] = []
            if kind == "reference":
                table = REFERENCE_TABLES[slot]
                candidates.extend(
                    ("other_visible_same_type", item)
                    for item in ids.get(table, []) if item != current
                )
                candidates.extend(
                    ("visible_cross_type", item)
                    for item in all_ids
                    if item not in ids.get(table, [])
                )
                candidates.extend((
                    ("well_formed_absent", f"absent-{table}-a"),
                    ("well_formed_absent", f"absent-{table}-b"),
                    ("malformed_or_empty", ""),
                    ("malformed_or_empty", "!"),
                ))
            elif kind == "opaque_reference":
                candidates.extend((
                    ("stale_or_other_object", f"stale-{slot}-a"),
                    ("stale_or_other_object", f"other-{slot}-b"),
                    ("malformed_or_empty", ""),
                    ("malformed_or_empty", "!"),
                ))
            elif kind == "version":
                candidates.extend((
                    ("previous_or_future", "0"),
                    ("previous_or_future", "999999"),
                    ("malformed_or_empty", ""),
                    ("malformed_or_empty", "not-a-version"),
                ))
            else:
                candidates.extend((
                    ("unknown", f"unknown-{slot}-a"),
                    ("unknown", f"unknown-{slot}-b"),
                    ("malformed_or_empty", ""),
                    ("malformed_or_empty", "!"),
                ))
            for partition_class, value in candidates:
                attacked = copy.deepcopy(baseline)
                attacked["arguments"][slot] = value
                add(partition_class, attacked, slot=slot, value=value)
    return tuple(rows)


def _parser_cases() -> tuple[tuple[str, str, str, bool, str | None, str | None], ...]:
    """Return class, family, content, expected accept, tool, slot rows."""

    rows: list[tuple[str, str, str, bool, str | None, str | None]] = []
    catalog = trusted_world_catalog(DEFAULT_AUDIT_SEEDS[0])
    representative = {
        family: next(world for world in catalog if world.family == family)
        for family in TOOL_SCHEMAS
    }
    for family, world in sorted(representative.items()):
        observation = _replay(world, ()).observation
        all_baselines = _all_grounded_tool_actions(observation)
        baselines_by_tool: dict[str, dict[str, Any]] = {}
        for action in all_baselines:
            baselines_by_tool.setdefault(str(action["tool_name"]), action)
        baselines = tuple(
            baselines_by_tool[tool] for tool in sorted(baselines_by_tool)
        )
        for baseline in baselines:
            tool = str(baseline["tool_name"])
            rows.append((
                "valid_null_key", family, _canonical(baseline), True, tool, None,
            ))
            fresh = copy.deepcopy(baseline)
            fresh["idempotency_key"] = f"fresh-{family}-{tool}"
            rows.append((
                "valid_fresh_key", family, _canonical(fresh), True, tool, None,
            ))
            extra = copy.deepcopy(baseline)
            extra["arguments"]["extra"] = "x"
            rows.append(("extra_argument", family, _canonical(extra), False, tool, None))
            for slot, value in sorted(baseline["arguments"].items()):
                missing = copy.deepcopy(baseline)
                del missing["arguments"][slot]
                rows.append((
                    "missing_argument", family, _canonical(missing), False, tool, slot,
                ))
                for wrong in (None, True, [], {}):
                    attacked = copy.deepcopy(baseline)
                    attacked["arguments"][slot] = wrong
                    rows.append((
                        "wrong_argument_type", family, _canonical(attacked),
                        False, tool, slot,
                    ))
                encoded_value = json.dumps(value, ensure_ascii=False)
                arguments = ",".join(
                    f"{json.dumps(key)}:{json.dumps(item, ensure_ascii=False)}"
                    for key, item in baseline["arguments"].items()
                )
                duplicate = (
                    "{" + arguments + ("," if arguments else "")
                    + f"{json.dumps(slot)}:{encoded_value}" + "}"
                )
                content = (
                    '{"action":"call_tool","tool_name":'
                    + json.dumps(tool) + ',"arguments":' + duplicate
                    + ',"idempotency_key":null}'
                )
                rows.append((
                    "duplicate_argument", family, content, False, tool, slot,
                ))
        sample = baselines[0]
        for content in ("not-json", "{"):
            rows.append(("invalid_json", family, content, False, None, None))
        for value in ([], "action"):
            rows.append((
                "non_object_root", family, json.dumps(value), False, None, None,
            ))
        for field in ("action", "tool_name", "arguments", "idempotency_key"):
            attacked = copy.deepcopy(sample)
            del attacked[field]
            rows.append((
                "missing_top_field", family, _canonical(attacked), False,
                str(sample["tool_name"]), None,
            ))
        extra = copy.deepcopy(sample)
        extra["explanation"] = "x"
        rows.append((
            "extra_top_field", family, _canonical(extra), False,
            str(sample["tool_name"]), None,
        ))
        rows.append((
            "duplicate_top_field", family,
            '{"action":"stop","action":"stop"}', False, None, None,
        ))
        for tool in ("unknown_tool", next(
            name for other, tools in TOOL_SCHEMAS.items() if other != family
            for name in tools
        )):
            attacked = copy.deepcopy(sample)
            attacked["tool_name"] = tool
            rows.append((
                "unknown_or_cross_family_tool", family, _canonical(attacked),
                False, tool, None,
            ))
        for value in (None, [], "arguments"):
            attacked = copy.deepcopy(sample)
            attacked["arguments"] = value
            rows.append((
                "arguments_not_object", family, _canonical(attacked), False,
                str(sample["tool_name"]), None,
            ))
        no_args = next((row for row in baselines if not row["arguments"]), None)
        if no_args is not None:
            tool = str(no_args["tool_name"])
            content = (
                '{"action":"call_tool","tool_name":' + json.dumps(tool)
                + ',"arguments":{"x":NaN},"idempotency_key":null}'
            )
            rows.append(("nonfinite_argument", family, content, False, tool, "x"))
        for value in (False, 0, [], {}):
            attacked = copy.deepcopy(sample)
            attacked["idempotency_key"] = value
            rows.append((
                "invalid_idempotency_type", family, _canonical(attacked), False,
                str(sample["tool_name"]), None,
            ))
        key_cases = (
            ("empty_idempotency_key", ""),
            ("non_nfc_idempotency_key", "e\N{COMBINING ACUTE ACCENT}"),
            ("control_idempotency_key", "hidden\ncontrol"),
            ("overlength_idempotency_key", "x" * (
                IDEMPOTENCY_KEY_MAX_UTF8_BYTES + 1
            )),
        )
        for partition_class, value in key_cases:
            attacked = copy.deepcopy(sample)
            attacked["idempotency_key"] = value
            rows.append((
                partition_class, family, _canonical(attacked), False,
                str(sample["tool_name"]), None,
            ))
    return tuple(rows)


def audit_parser_partition() -> dict[str, Any]:
    counts: dict[str, int] = {}
    tools: set[tuple[str, str]] = set()
    slots: set[tuple[str, str, str]] = set()
    failures: list[dict[str, Any]] = []
    for partition_class, family, content, accepts, tool, slot in _parser_cases():
        counts[partition_class] = counts.get(partition_class, 0) + 1
        if tool in TOOL_SCHEMAS[family]:
            tools.add((family, str(tool)))
        if tool in TOOL_SCHEMAS[family] and slot in TOOL_SCHEMAS[family][str(tool)]:
            slots.add((family, str(tool), str(slot)))
        try:
            parse_model_action(content, family=family)
            actual = True
        except (TypeError, ValueError):
            actual = False
        if actual != accepts:
            failures.append({
                "partition_class": partition_class, "family": family,
                "tool": tool, "slot": slot, "expected_accept": accepts,
                "actual_accept": actual,
            })
    expected_tools = {
        (family, tool) for family, tools_by_name in TOOL_SCHEMAS.items()
        for tool in tools_by_name
    }
    expected_slots = {
        (family, tool, slot) for family, tools_by_name in TOOL_SCHEMAS.items()
        for tool, schema in tools_by_name.items() for slot in schema
    }
    return {
        "representatives": sum(counts.values()),
        "class_counts": dict(sorted(counts.items())),
        "tools_covered": len(tools), "tools_total": len(expected_tools),
        "slots_covered": len(slots), "slots_total": len(expected_slots),
        "missing_tools": sorted(expected_tools - tools),
        "missing_slots": sorted(expected_slots - slots),
        "failures": failures,
        "declared_parse_classes": list(
            ACTION_PARTITION_MANIFEST["parse_classes"]
        ),
        "missing_declared_parse_classes": sorted(
            set(ACTION_PARTITION_MANIFEST["parse_classes"]) - set(counts)
        ),
        "undeclared_parse_classes": sorted(
            set(counts) - set(ACTION_PARTITION_MANIFEST["parse_classes"])
        ),
    }


def _checker_reason_manifest() -> tuple[str, ...]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(
        MultiTownStatefulOpsEnv.checker,
    )))
    reasons: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call) and len(node.args) >= 2
            and isinstance(node.func, ast.Name)
            and node.func.id == "CheckerDecision"
        ):
            reasons.update(
                child.value for child in ast.walk(node.args[1])
                if isinstance(child, ast.Constant) and isinstance(child.value, str)
            )
    return tuple(sorted(reasons))


def _ops_replay(
    world: TrustedWorld, history: Sequence[Mapping[str, Any]],
) -> MultiTownStatefulOpsEnv:
    env = MultiTownStatefulOpsEnv(world.scenario)
    for action in history:
        if action["action"] == "stop":
            raise AssertionError("checkpoint histories must be preterminal")
        env.call_tool(
            str(action["tool_name"]), dict(action["arguments"]),
            idempotency_key=action.get("idempotency_key"),
        )
    return env


def _decision_at_execution(
    env: MultiTownStatefulOpsEnv, action: Mapping[str, Any],
) -> CheckerDecision:
    probe = copy.deepcopy(env)
    normalized = json.loads(_canonical(action["arguments"]))
    key = action.get("idempotency_key")
    prior = probe._idempotency.get(key) if key else None
    if prior is not None:
        fingerprint = hashlib.sha256(_canonical([
            action["tool_name"], normalized,
        ]).encode()).hexdigest()
        return CheckerDecision(
            "ALLOW" if prior[0] == fingerprint else "DENY",
            "IDEMPOTENT_REPLAY" if prior[0] == fingerprint
            else "IDEMPOTENCY_KEY_REUSE",
        )
    probe._apply_scheduled_events(
        "before_action", probe.logical_tick, tool_name=str(action["tool_name"]),
        arguments=normalized,
    )
    return probe.checker(str(action["tool_name"]), normalized)


def _fits_budget(observation: Mapping[str, Any], action: Mapping[str, Any]) -> bool:
    task, runtime = observation["task"], observation["runtime"]
    family, tool = str(task["family"]), str(action["tool_name"])
    profile = __import__(
        "multitown.stateful_ops", fromlist=["tool_profile"],
    ).tool_profile(family)["tools"][tool]
    return bool(
        observation["tool_calls_remaining"] > 0
        and observation["steps"] < task["budget"]["max_steps"]
        and observation["logical_latency_remaining"]
        >= profile["logical_latency_cost"]
        and observation["irreversible_risk_remaining"]
        >= profile["irreversible_risk_cost"]
        and not runtime["budget_exhausted"]
    )


def _check_differential(
    worlds: Sequence[TrustedWorld], action: Mapping[str, Any],
    prior: Sequence[_Step], advanced: Sequence[_Step],
) -> bool:
    if len({_surface_key(step) for step in advanced}) == 1:
        return False
    try:
        allowed = _declared_reveal_paths(
            worlds[0].family, worlds[0].variant_id, action,
            prior[0].observation,
        )
    except AssertionError as exc:
        raise AssertionError(
            "one-step surfaces diverged without a declared reveal for "
            f"{_canonical(action)}"
        ) from exc
    differing = _differing_world_paths([step.observation for step in advanced])
    if not differing or not differing <= allowed:
        raise AssertionError(
            "one-step action diverged outside declared reveal paths: "
            f"{sorted(differing - allowed)}"
        )
    for before, after in zip(prior, advanced, strict=True):
        _validate_reveal_semantics(worlds[0].family, action, before, after)
    return True


def _check_stop_partition(
    worlds: Sequence[TrustedWorld], histories: Sequence[Sequence[Mapping[str, Any]]],
) -> tuple[int, int]:
    prior = [_replay(world, history) for world, history in zip(worlds, histories, strict=True)]
    advanced = [
        _replay(world, (*history, {"action": "stop"}))
        for world, history in zip(worlds, histories, strict=True)
    ]
    for world, before, after in zip(worlds, prior, advanced, strict=True):
        _validate_transition_surface(world.family, before, {"action": "stop"}, after)
        terminal = after.info["terminal_result"]
        if after.reward != float(terminal["success"]):
            raise AssertionError("stop reward does not equal strict terminal success")
    partitions: dict[str, set[str]] = {}
    for before, after in zip(prior, advanced, strict=True):
        partitions.setdefault(_surface_key(before), set()).add(_surface_key(after))
    if any(len(values) != 1 for values in partitions.values()):
        raise AssertionError("same public pre-stop surface exposed different terminal surface")
    failed = sum(not step.info["terminal_result"]["success"] for step in advanced)
    return len(advanced), failed


def _policy_checkpoints(
    worlds: tuple[TrustedWorld, ...], root: PolicyNode,
) -> tuple[tuple[tuple[TrustedWorld, tuple[Mapping[str, Any], ...]], ...], ...]:
    result: list[
        tuple[tuple[TrustedWorld, tuple[Mapping[str, Any], ...]], ...]
    ] = []
    stack = [(root, tuple((world, ()) for world in worlds))]
    while stack:
        node, members = stack.pop()
        result.append(members)
        if node.terminal_success:
            continue
        partitions: dict[
            str, list[tuple[TrustedWorld, tuple[Mapping[str, Any], ...]]]
        ] = {}
        for world, history in members:
            next_history = (*history, node.action)
            step = _replay(world, next_history)
            from .stateful_surface_audit import _planner_outcome
            partitions.setdefault(_planner_outcome(step), []).append(
                (world, next_history)
            )
        if set(partitions) != set(node.outcomes):
            raise AssertionError("policy checkpoint outcomes drifted")
        for outcome, partition in partitions.items():
            stack.append((node.outcomes[outcome], tuple(partition)))
    return tuple(result)


def audit_runtime_seed(world_seed: int) -> dict[str, Any]:
    catalog = trusted_world_catalog(world_seed)
    checkpoint_count = checkpoint_worlds = execution_count = 0
    representative_actions_generated = representative_actions_executed = 0
    grounded_actions_generated = grounded_actions_represented = 0
    grounded_actions_budget_feasible = grounded_actions_executed = 0
    differential_checks = reveal_checks = stop_checks = failed_stop_checks = 0
    result_counts: dict[str, int] = {}
    partition_counts: dict[str, int] = {}
    tools: set[tuple[str, str]] = set()
    slots: set[tuple[str, str, str]] = set()
    reasons: set[str] = set()
    for _, members in _information_sets(catalog):
        planner = PublicBeliefPlanner(
            (world.scenario for world in members),
            max_depth=max(
                world.scenario.public_task.budget.max_steps for world in members
            ),
            grounder=grounded_public_actions,
        )
        tree = planner.solve()
        if tree is None:
            raise AssertionError("positive policy tree missing before partition audit")
        for checkpoint in _policy_checkpoints(members, tree):
            checkpoint_count += 1
            worlds = tuple(world for world, _ in checkpoint)
            checkpoint_worlds += len(worlds)
            histories = tuple(history for _, history in checkpoint)
            prior = [_replay(world, history) for world, history in checkpoint]
            if len({_surface_key(step) for step in prior}) != 1:
                raise AssertionError("checkpoint members do not share one public surface")
            checked, failed = _check_stop_partition(worlds, histories)
            stop_checks += checked
            failed_stop_checks += failed
            representatives = _runtime_representatives(prior[0].observation)
            representative_actions_generated += len(representatives)
            grounded_keys = {
                _canonical(action)
                for action in _all_grounded_tool_actions(prior[0].observation)
            }
            representatives_by_key = {
                _canonical(row.action): row.action for row in representatives
            }
            representative_keys = set(representatives_by_key)
            grounded_actions_generated += len(grounded_keys)
            grounded_actions_represented += len(grounded_keys & representative_keys)
            if not grounded_keys <= representative_keys:
                raise AssertionError("runtime partition omitted grounded tool calls")
            feasible_grounded_keys = {
                key for key in grounded_keys
                if _fits_budget(prior[0].observation, representatives_by_key[key])
            }
            grounded_actions_budget_feasible += len(feasible_grounded_keys)
            executed_grounded_keys: set[str] = set()
            for representative in representatives:
                action = representative.action
                if not _fits_budget(prior[0].observation, action):
                    continue
                representative_actions_executed += 1
                action_key = _canonical(action)
                if action_key in grounded_keys:
                    executed_grounded_keys.add(action_key)
                tool = str(action["tool_name"])
                tools.add((worlds[0].family, tool))
                if representative.slot is not None:
                    slots.add((worlds[0].family, tool, representative.slot))
                partition_counts[representative.partition_class] = (
                    partition_counts.get(representative.partition_class, 0) + 1
                )
                advanced = [
                    _replay(world, (*history, action))
                    for world, history in checkpoint
                ]
                for world, before, after in zip(
                    worlds, prior, advanced, strict=True,
                ):
                    execution_count += 1
                    _validate_transition_surface(world.family, before, action, after)
                    if after.reward != 0.0 or after.terminated or after.truncated:
                        raise AssertionError("non-budget tool action must remain nonterminal")
                    result = str(after.info["tool_result"]["result"])
                    result_counts[result] = result_counts.get(result, 0) + 1
                    ops = _ops_replay(world, histories[worlds.index(world)])
                    decision = _decision_at_execution(ops, action)
                    reasons.add(decision.reason_code)
                    before_private = _digest(ops.private_state)
                    tool_result = ops.call_tool(
                        tool, dict(action["arguments"]),
                        idempotency_key=action.get("idempotency_key"),
                    )
                    if tool_result != after.info["tool_result"]:
                        raise AssertionError("trusted and public one-step results diverged")
                    if result == "blocked" and (
                        tool not in READ_TOOLS
                        and after.info["tool_result"]["transition"]["agent_changed_objects"]
                    ):
                        raise AssertionError("blocked reference changed agent business state")
                    if (
                        result == "blocked"
                        and before_private != _digest(ops.private_state)
                        and not after.info["tool_result"]["transition"]["external_events"]
                    ):
                        raise AssertionError("blocked call changed private state without event")
                differential_checks += 1
                reveal_checks += int(_check_differential(
                    worlds, action, prior, advanced,
                ))
                next_histories = tuple(
                    (*history, action) for history in histories
                )
                checked, failed = _check_stop_partition(worlds, next_histories)
                stop_checks += checked
                failed_stop_checks += failed
            grounded_actions_executed += len(executed_grounded_keys)
    expected_tools = {
        (family, tool) for family, tools_by_name in TOOL_SCHEMAS.items()
        for tool in tools_by_name
    }
    expected_slots = {
        (family, tool, slot) for family, tools_by_name in TOOL_SCHEMAS.items()
        for tool, schema in tools_by_name.items() for slot in schema
    }
    declared_reasons = set(_checker_reason_manifest())
    missing_reasons = declared_reasons - reasons
    return {
        "world_seed": world_seed,
        "public_lifecycle_checkpoints": checkpoint_count,
        "checkpoint_worlds": checkpoint_worlds,
        "grounded_actions_generated": grounded_actions_generated,
        "grounded_actions_represented": grounded_actions_represented,
        "grounded_actions_budget_feasible": grounded_actions_budget_feasible,
        "grounded_actions_executed": grounded_actions_executed,
        "representative_actions_generated": representative_actions_generated,
        "representative_actions_budget_feasible": representative_actions_executed,
        "representative_actions_executed": representative_actions_executed,
        "world_executions": execution_count,
        "differential_checks": differential_checks,
        "declared_reveal_checks": reveal_checks,
        "stop_truth_table_checks": stop_checks,
        "failed_stop_reward_checks": failed_stop_checks,
        "result_counts": dict(sorted(result_counts.items())),
        "partition_class_counts": dict(sorted(partition_counts.items())),
        "tools_covered": len(tools), "tools_total": len(expected_tools),
        "slots_covered": len(slots), "slots_total": len(expected_slots),
        "missing_tools": sorted(expected_tools - tools),
        "missing_slots": sorted(expected_slots - slots),
        "checker_reasons_hit": sorted(reasons),
        "checker_reasons_declared": len(declared_reasons),
        "checker_reasons_covered": len(reasons & declared_reasons),
        "checker_reasons_missing": sorted(missing_reasons),
        "checker_reason_waivers": {
            reason: CHECKER_REASON_WAIVERS[reason]
            for reason in sorted(missing_reasons)
            if reason in CHECKER_REASON_WAIVERS
        },
        "checker_reasons_unexplained": sorted(
            missing_reasons - set(CHECKER_REASON_WAIVERS)
        ),
    }


def audit_idempotency_partition() -> dict[str, Any]:
    checks: dict[str, bool] = {}

    records_world = next(
        world for world in trusted_world_catalog(210)
        if world.family == "records_casework"
        and world.variant_id == "direct_or_staged"
    )
    records = MultiTownStatefulOpsEnv(records_world.scenario)
    initial = records.observation()
    read = records.call_tool("search_records", {})
    checks["null"] = read["result"] == "ok" and not records._idempotency
    first = records.call_tool("search_records", {}, idempotency_key="read-snapshot")
    second = records.call_tool("search_records", {}, idempotency_key="read-snapshot")
    checks["fresh"] = "read-snapshot" in records._idempotency
    checks["exact_replay"] = bool(
        second["result"] == first["result"]
        and second["payload"] == first["payload"]
        and second["idempotent_noop"]
    )
    case_id = next(iter(records.state["cases"]))
    record_id = next(iter(records.state["records"]))
    records.call_tool(
        "attach_evidence", {"case_id": case_id, "evidence_id": record_id},
    )
    third = records.call_tool(
        "search_records", {}, idempotency_key="read-snapshot",
    )
    checks["exact_replay_after_state_advance"] = bool(
        third["payload"] == first["payload"] and third["idempotent_noop"]
    )
    mismatch_args = records.call_tool(
        "get_case", {"case_id": "absent-cases-a"},
        idempotency_key="read-snapshot",
    )
    checks["same_key_different_tool"] = (
        mismatch_args["error_code"] == "IDEMPOTENCY_KEY_REUSE"
    )
    checks["keyed_read_snapshot"] = third["payload"] == first["payload"]

    incident_world = next(
        world for world in trusted_world_catalog(211)
        if world.family == "incident_recovery"
        and world.variant_id == "rollback_or_patch"
    )
    incident = MultiTownStatefulOpsEnv(incident_world.scenario)
    incident_id = next(iter(incident.state["incidents"]))
    service_id = incident.state["incidents"][incident_id]["service_id"]
    blocked_args = {"incident_id": incident_id, "service_id": service_id}
    blocked = incident.call_tool(
        "close_incident", blocked_args, idempotency_key="blocked",
    )
    blocked_replay = incident.call_tool(
        "close_incident", dict(reversed(tuple(blocked_args.items()))),
        idempotency_key="blocked",
    )
    checks["keyed_blocked"] = bool(
        blocked["result"] == blocked_replay["result"] == "blocked"
        and blocked["error_code"] == blocked_replay["error_code"]
        and blocked_replay["idempotent_noop"]
        and incident.attempted_policy_violations == 1
        and incident.blocked_unsafe_actions == 1
        and incident.irreversible_risk_used == 0
        and not blocked_replay["transition"]["external_events"]
    )
    blocked_trace_env = MultiTownStatefulOpsEnv(incident_world.scenario)
    before_blocked = blocked_trace_env.observation()
    first_blocked = blocked_trace_env.call_tool(
        "close_incident", blocked_args, idempotency_key="blocked-trace",
    )
    after_first_blocked = blocked_trace_env.observation()
    replayed_blocked = blocked_trace_env.call_tool(
        "close_incident", blocked_args, idempotency_key="blocked-trace",
    )
    after_replayed_blocked = blocked_trace_env.observation()
    terminal_blocked = blocked_trace_env.stop()
    blocked_content = _canonical({
        "action": "call_tool", "tool_name": "close_incident",
        "arguments": blocked_args, "idempotency_key": "blocked-trace",
    })
    blocked_rows = [
        public_trace_row(
            session_id="blocked-replay", turn_id=0,
            family="incident_recovery", observation=before_blocked,
            action_content=blocked_content, tool_result=first_blocked,
            terminal_result=None,
        ),
        public_trace_row(
            session_id="blocked-replay", turn_id=1,
            family="incident_recovery", observation=after_first_blocked,
            action_content=blocked_content, tool_result=replayed_blocked,
            terminal_result=None,
        ),
        public_trace_row(
            session_id="blocked-replay", turn_id=2,
            family="incident_recovery", observation=after_replayed_blocked,
            action_content='{"action":"stop"}', tool_result=None,
            terminal_result={
                "terminal": terminal_blocked.terminal,
                "success": terminal_blocked.success,
                "safety_violations": terminal_blocked.safety_violations,
                "budget_violations": terminal_blocked.budget_violations,
            },
        ),
    ]
    validate_public_trace(blocked_rows)
    checks["blocked_public_trace_replay"] = True
    different_args = incident.call_tool(
        "close_incident", {**blocked_args, "service_id": "absent-services-a"},
        idempotency_key="blocked",
    )
    checks["same_key_different_args"] = (
        different_args["error_code"] == "IDEMPOTENCY_KEY_REUSE"
    )
    noop = incident.call_tool(
        "healthcheck", {"service_id": service_id}, idempotency_key="noop",
    )
    noop_replay = incident.call_tool(
        "healthcheck", {"service_id": service_id}, idempotency_key="noop",
    )
    checks["keyed_noop"] = bool(
        noop["result"] == noop_replay["result"] == "noop"
        and noop_replay["idempotent_noop"]
    )

    permit_world = next(
        world for world in trusted_world_catalog(212)
        if world.family == "permit_transaction"
        and world.variant_id == "atomic_or_staged"
    )
    permit = MultiTownStatefulOpsEnv(permit_world.scenario)
    permit_id = next(iter(permit.state["permits"]))
    ok = permit.call_tool(
        "issue_permit", {"permit_id": permit_id}, idempotency_key="ok",
    )
    ok_replay = permit.call_tool(
        "issue_permit", {"permit_id": permit_id}, idempotency_key="ok",
    )
    checks["keyed_ok"] = bool(
        ok["result"] == ok_replay["result"] == "ok"
        and ok_replay["idempotent_noop"]
    )

    conflict_world = next(
        world for world in trusted_world_catalog(213)
        if world.family == "resource_calendar"
        and world.variant_id == "optimistic_conflict_replan"
        and world.role == "preferred_a_conflict"
    )
    conflict = MultiTownStatefulOpsEnv(conflict_world.scenario)
    booking_id = next(iter(conflict.state["bookings"]))
    conflict.call_tool("snapshot_availability", {"booking_id": booking_id})
    booking = conflict.state["bookings"][booking_id]
    resource_id = booking["preferred_resource_id"]
    conflict_args = {
        "booking_id": booking_id, "resource_id": resource_id,
        "snapshot_token": booking["snapshot_token"],
        "expected_version": str(booking["snapshot_versions"][resource_id]),
    }
    conflict_first = conflict.call_tool(
        "create_versioned_hold", conflict_args, idempotency_key="conflict",
    )
    conflict_replay = conflict.call_tool(
        "create_versioned_hold", conflict_args, idempotency_key="conflict",
    )
    checks["keyed_conflict"] = bool(
        conflict_first["result"] == conflict_replay["result"] == "conflict"
        and conflict_first["payload"] == conflict_replay["payload"]
        and conflict_replay["idempotent_noop"]
    )

    budget = MultiTownStatefulOpsEnv(permit_world.scenario)
    args = {"permit_id": permit_id}
    for _ in range(budget.scenario.public_task.budget.tool_calls):
        budget.call_tool("get_permit", args)
    rejected = budget.call_tool(
        "get_permit", args, idempotency_key="budget-excluded",
    )
    checks["budget_precheck_excluded"] = bool(
        rejected["error_code"] == "BUDGET_EXHAUSTED"
        and "budget-excluded" not in budget._idempotency
    )
    boundary = "é" * (IDEMPOTENCY_KEY_MAX_UTF8_BYTES // 2)
    unicode_env = MultiTownStatefulOpsEnv(records_world.scenario)
    checks["unicode_boundary"] = unicode_env.call_tool(
        "search_records", {}, idempotency_key=boundary,
    )["result"] == "ok"
    try:
        unicode_env.call_tool(
            "search_records", {},
            idempotency_key="x" * (IDEMPOTENCY_KEY_MAX_UTF8_BYTES + 1),
        )
    except TypeError:
        checks["length_boundary"] = True
    else:
        checks["length_boundary"] = False
    checks["ttl_unsupported"] = (
        ACTION_PARTITION_MANIFEST["execution_boundary"]["ttl"]
        == "unsupported in finite episodes"
    )
    validate_public_surface({"initial": initial, "checks": checks})
    return {
        "class_checks": checks,
        "classes_passed": sum(checks.values()),
        "classes_total": len(checks),
        "all_pass": all(checks.values()),
    }


def _mutation_controls() -> dict[str, Any]:
    world = next(
        row for row in trusted_world_catalog(214)
        if row.family == "incident_recovery"
        and row.variant_id == "rollback_or_patch"
    )
    prior = _replay(world, ())
    stopped = _replay(world, ({"action": "stop"},))
    controls: dict[str, bool] = {}
    try:
        _validate_transition_surface(
            world.family, prior, {"action": "stop"},
            replace(stopped, reward=1.0),
        )
    except AssertionError:
        controls["failed_stop_reward_leak"] = True
    else:
        controls["failed_stop_reward_leak"] = False

    action = next(
        row for row in grounded_public_actions(prior.observation)
        if row.get("tool_name") == "close_incident"
    )
    _replay(world, (action,))
    paired = tuple(
        row for row in trusted_world_catalog(214)
        if row.family == "incident_recovery"
        and row.variant_id == "canary_compensation_saga"
    )
    paired_prior = [_replay(row, ()) for row in paired]
    paired_action = next(
        row for row in grounded_public_actions(paired_prior[0].observation)
        if row.get("tool_name") == "close_incident"
    )
    paired_advanced = [_replay(row, (paired_action,)) for row in paired]
    hidden_error = copy.deepcopy(paired_advanced[1].info)
    hidden_error["tool_result"]["error_code"] = "HIDDEN_BRANCH_A"
    error_attacked = [
        paired_advanced[0], replace(paired_advanced[1], info=hidden_error),
    ]
    try:
        _check_differential(
            paired, paired_action, paired_prior, error_attacked,
        )
    except AssertionError:
        controls["blocked_hidden_error_divergence"] = True
    else:
        controls["blocked_hidden_error_divergence"] = False
    private_payload = copy.deepcopy(paired_advanced[1].info)
    private_payload["tool_result"]["payload"] = paired[1].scenario.private_instance_id
    payload_attacked = [
        paired_advanced[0], replace(paired_advanced[1], info=private_payload),
    ]
    try:
        validate_public_surface(private_payload)
        _check_differential(
            paired, paired_action, paired_prior, payload_attacked,
        )
    except (AssertionError, ValueError):
        controls["blocked_private_payload"] = True
    else:
        controls["blocked_private_payload"] = False

    conflict = next(
        row for row in trusted_world_catalog(216)
        if row.family == "resource_calendar"
        and row.variant_id == "optimistic_conflict_replan"
        and row.role == "preferred_a_conflict"
    )
    original = conflict.scenario.private_dynamics.scheduled_events[0]
    trigger_resource = dict(original.trigger_arguments)["resource_id"]
    other_resource = next(
        resource_id
        for resource_id in conflict.scenario.private_evaluator.initial_state()[
            "resources"
        ]
        if resource_id != trigger_resource
    )
    attacked_event = replace(
        original, path=("resources", other_resource),
    )
    attacked_dynamics = replace(
        conflict.scenario.private_dynamics,
        scheduled_events=(attacked_event,),
        allowed_external_public_paths=(("resources", other_resource),),
    )
    try:
        MultiTownStatefulOpsEnv(replace(
            conflict.scenario, private_dynamics=attacked_dynamics,
        ))
    except ValueError as exc:
        controls["cross_object_event_target_binding"] = (
            "target identity" in str(exc)
        )
    else:
        controls["cross_object_event_target_binding"] = False
    return {
        "controls": controls, "killed": sum(controls.values()),
        "planted": len(controls), "all_killed": all(controls.values()),
    }


def _evaluate_gates(
    parser: Mapping[str, Any], runtime: Sequence[Mapping[str, Any]],
    idempotency: Mapping[str, Any], mutations: Mapping[str, Any],
) -> dict[str, bool]:
    declared_idempotency = set(ACTION_PARTITION_MANIFEST["idempotency_classes"])
    declared_mutations = set(ACTION_PARTITION_MANIFEST["mutation_controls"])
    declared_checker_reasons = set(_checker_reason_manifest())
    return {
        "parser_partition_passes": bool(
            not parser["failures"]
            and not parser["missing_declared_parse_classes"]
            and not parser["undeclared_parse_classes"]
            and set(parser["class_counts"])
            == set(ACTION_PARTITION_MANIFEST["parse_classes"])
            and all(parser["class_counts"].values())
            and parser["representatives"] == sum(parser["class_counts"].values())
        ),
        "all_44_tools_parser_covered": (
            parser["tools_covered"] == parser["tools_total"] == 44
            and not parser["missing_tools"]
        ),
        "all_63_argument_slots_parser_covered": (
            parser["slots_covered"] == parser["slots_total"] == 63
            and not parser["missing_slots"]
        ),
        "all_44_tools_runtime_covered_each_seed": bool(runtime) and all(
            row["tools_covered"] == row["tools_total"] == 44
            and not row["missing_tools"] for row in runtime
        ),
        "all_63_argument_slots_runtime_covered_each_seed": bool(runtime) and all(
            row["slots_covered"] == row["slots_total"] == 63
            and not row["missing_slots"] for row in runtime
        ),
        "all_declared_runtime_value_classes_hit_each_seed": bool(runtime) and all(
            set(row["partition_class_counts"]) == set().union(*(
                set(values) for values in ACTION_PARTITION_MANIFEST[
                    "slot_value_classes"
                ].values()
            ))
            and all(row["partition_class_counts"].values())
            for row in runtime
        ),
        "all_grounded_calls_represented_each_seed": bool(runtime) and all(
            row["grounded_actions_represented"]
            == row["grounded_actions_generated"] > 0
            for row in runtime
        ),
        "all_budget_feasible_grounded_calls_executed_each_seed": bool(runtime) and all(
            row["grounded_actions_executed"]
            == row["grounded_actions_budget_feasible"] > 0
            for row in runtime
        ),
        "runtime_action_count_relations_consistent_each_seed": bool(runtime) and all(
            0 < row["grounded_actions_budget_feasible"]
            == row["grounded_actions_executed"]
            <= row["grounded_actions_represented"]
            == row["grounded_actions_generated"]
            <= row["representative_actions_generated"]
            and row["grounded_actions_executed"]
            <= row["representative_actions_executed"]
            == row["representative_actions_budget_feasible"]
            == row["differential_checks"]
            <= row["representative_actions_generated"]
            and row["world_executions"] >= row["representative_actions_executed"]
            for row in runtime
        ),
        "all_checker_reasons_covered_each_seed": bool(runtime) and all(
            declared_checker_reasons
            and set(row["checker_reasons_hit"]) <= declared_checker_reasons
            and set(row["checker_reasons_hit"]) & declared_checker_reasons
            == declared_checker_reasons - set(row["checker_reasons_missing"])
            and row["checker_reasons_declared"] == len(declared_checker_reasons)
            and row["checker_reasons_covered"]
            == len(set(row["checker_reasons_hit"]) & declared_checker_reasons)
            and set(row["checker_reasons_missing"])
            == declared_checker_reasons - set(row["checker_reasons_hit"])
            == set(row["checker_reason_waivers"])
            and row["checker_reason_waivers"] == {
                reason: CHECKER_REASON_WAIVERS[reason]
                for reason in row["checker_reasons_missing"]
                if reason in CHECKER_REASON_WAIVERS
            }
            and not row["checker_reasons_unexplained"]
            for row in runtime
        ),
        "failed_stop_reward_truth_table_exercised_each_seed": bool(runtime) and all(
            row["failed_stop_reward_checks"] > 0 for row in runtime
        ),
        "every_runtime_representative_differential_checked_each_seed": bool(runtime) and all(
            row["representative_actions_executed"]
            == row["representative_actions_budget_feasible"]
            == row["differential_checks"] > 0
            for row in runtime
        ),
        "every_runtime_execution_followed_by_stop_each_seed": bool(runtime) and all(
            row["stop_truth_table_checks"]
            == row["world_executions"] + row["checkpoint_worlds"]
            and row["stop_truth_table_checks"] > 0
            for row in runtime
        ),
        "declared_reveal_checks_exercised_each_seed": bool(runtime) and all(
            row["declared_reveal_checks"] > 0 for row in runtime
        ),
        "idempotency_partition_passes": bool(
            declared_idempotency
            and set(idempotency["class_checks"]) == declared_idempotency
            and all(idempotency["class_checks"].values())
            and idempotency["classes_total"] == len(declared_idempotency)
            and idempotency["classes_passed"] == len(declared_idempotency)
            and idempotency["all_pass"]
        ),
        "declared_mutation_controls_killed": bool(
            declared_mutations
            and set(mutations["controls"]) == declared_mutations
            and all(mutations["controls"].values())
            and mutations["planted"] == len(declared_mutations)
            and mutations["killed"] == len(declared_mutations)
            and mutations["all_killed"]
        ),
    }


def run_action_partition_audit(
    *, world_seeds: Iterable[int] = DEFAULT_AUDIT_SEEDS,
) -> dict[str, Any]:
    seeds = tuple(world_seeds)
    if not seeds or len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        raise ValueError("world seeds must be a non-empty unique non-negative sequence")
    parser = audit_parser_partition()
    runtime = [audit_runtime_seed(seed) for seed in seeds]
    idempotency = audit_idempotency_partition()
    mutations = _mutation_controls()
    gates = _evaluate_gates(parser, runtime, idempotency, mutations)
    return {
        "schema_version": REPORT_VERSION,
        "stage": "train",
        "source_state": _source_state(),
        "manifest": {
            "schema_version": MANIFEST_VERSION,
            "sha256": manifest_sha256(),
            "scope": ACTION_PARTITION_MANIFEST["history_scope"],
        },
        "parser_partition": parser,
        "runtime_partition": runtime,
        "idempotency_partition": idempotency,
        "mutation_controls": mutations,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "claim_boundary": (
            "complete only relative to the versioned finite one-factor action "
            "partition over initial and robust-positive-policy lifecycle checkpoints; "
            "all grounded calls represented and all budget-feasible representatives "
            "executed; "
            "not arbitrary-input noninterference, complete reachable-state coverage, "
            "formal safety proof, held-out evidence, learned policy, or Agentic RL"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the finite stateful action and idempotency partition",
    )
    parser.add_argument(
        "--world-seeds", type=int, nargs="+", default=list(DEFAULT_AUDIT_SEEDS),
    )
    args = parser.parse_args()
    print(json.dumps(
        run_action_partition_audit(world_seeds=args.world_seeds),
        ensure_ascii=False, indent=2, sort_keys=True,
    ))


if __name__ == "__main__":
    main()
