"""Bounded accepted-invalid/unsafe search: initial train-only vertical slice."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from .g2_oracle import (
    ORACLE_VERSION,
    build_records_direct_spec,
    evaluate_records_direct,
    oracle_spec_sha256,
)
from .stateful_action_partition_audit import MANIFEST_VERSION, manifest_sha256
from .stateful_belief_planner import _source_state
from .stateful_grounding import GROUNDING_VERSION, grounded_public_actions
from .stateful_model_protocol import (
    run_scripted_model_actions, validate_trace_against_scenario,
)
from .stateful_ops import MultiTownStatefulOpsEnv
from .stateful_pomdp import StatefulPOMDPEnv
from .stateful_reachability import trusted_world_catalog


SEARCH_VERSION = "multitown-g2-bounded-search-v3"
REPORT_VERSION = "multitown-g2-vertical-slice-report-v3"
MUTATION_REPORT_VERSION = "multitown-g2-mutation-audit-v1"
MUTATION_CAMPAIGN_WORLD_SEED = 160
DECLARED_SEARCH_MUTANTS = {
    "oracle_drop_direct_sequence": "accepted_invalid",
    "oracle_drop_staged_sequence": "accepted_invalid",
    "oracle_flip_expected_decision": "accepted_invalid",
    "production_ignore_attempted_violation": "accepted_unsafe",
}
MUTANT_OPERATOR_CLASSES = {
    "oracle_drop_direct_sequence": "oracle_temporal_branch_drop",
    "oracle_drop_staged_sequence": "oracle_temporal_branch_drop",
    "oracle_flip_expected_decision": "oracle_terminal_goal_flip",
    "production_ignore_attempted_violation": "terminal_acceptor_relaxation",
}
MUTANT_REQUIRED_LABELS = {
    "oracle_drop_direct_sequence": {
        "production_accept": True, "oracle_goal_valid": True,
        "oracle_temporal_valid": False, "oracle_safe": True,
        "oracle_integrity_valid": True,
    },
    "oracle_drop_staged_sequence": {
        "production_accept": True, "oracle_goal_valid": True,
        "oracle_temporal_valid": False, "oracle_safe": True,
        "oracle_integrity_valid": True,
    },
    "oracle_flip_expected_decision": {
        "production_accept": True, "oracle_goal_valid": False,
        "oracle_temporal_valid": True, "oracle_safe": True,
        "oracle_integrity_valid": True,
    },
    "production_ignore_attempted_violation": {
        "production_accept": True, "oracle_goal_valid": True,
        "oracle_temporal_valid": True, "oracle_safe": False,
        "oracle_integrity_valid": True,
    },
}
MUTANT_EXPECTED_SEARCH_COUNTS = {
    "oracle_drop_direct_sequence": {
        "unique_states": 19, "generated_edges": 108, "executed_edges": 108,
        "merged_successors": 21, "merge_congruence_checks": 21,
        "stop_checks": 19,
        "pruned": {"budget": 0, "production_monotonic_violation": 69},
        "terminal_classifications": {
            "accepted_invalid": 1, "accepted_valid_safe": 1,
            "rejected_unsafe_or_invalid": 17,
        },
    },
    "oracle_drop_staged_sequence": {
        "unique_states": 19, "generated_edges": 108, "executed_edges": 108,
        "merged_successors": 21, "merge_congruence_checks": 21,
        "stop_checks": 19,
        "pruned": {"budget": 0, "production_monotonic_violation": 69},
        "terminal_classifications": {
            "accepted_invalid": 1, "accepted_valid_safe": 1,
            "rejected_unsafe_or_invalid": 17,
        },
    },
    "oracle_flip_expected_decision": {
        "unique_states": 19, "generated_edges": 108, "executed_edges": 108,
        "merged_successors": 21, "merge_congruence_checks": 21,
        "stop_checks": 19,
        "pruned": {"budget": 0, "production_monotonic_violation": 69},
        "terminal_classifications": {
            "accepted_invalid": 2, "rejected_unsafe_or_invalid": 17,
        },
    },
    "production_ignore_attempted_violation": {
        "unique_states": 85, "generated_edges": 516, "executed_edges": 516,
        "merged_successors": 432, "merge_congruence_checks": 432,
        "stop_checks": 85,
        "pruned": {"budget": 0, "production_monotonic_violation": 0},
        "terminal_classifications": {
            "accepted_unsafe": 4, "accepted_valid_safe": 4,
            "rejected_unsafe_or_invalid": 77,
        },
    },
}


def _canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


def alpha_normalize_idempotency_keys(
    actions: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Canonicalize key equality patterns without claiming Unicode completeness."""

    names: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    for action in actions:
        row = copy.deepcopy(dict(action))
        key = row.get("idempotency_key")
        if isinstance(key, str):
            names.setdefault(key, f"K{len(names)}")
            row["idempotency_key"] = names[key]
        rows.append(row)
    return tuple(rows)


@dataclass(frozen=True)
class MicroSearchResult:
    found: bool
    witness: tuple[str, ...]
    unique_states: int
    executed_edges: int
    cap_hit: bool


def bounded_bfs(
    initial: Any, actions: Iterable[str],
    transition: Callable[[Any, str], Any | None],
    target: Callable[[Any], bool], signature: Callable[[Any], str],
    *, horizon: int, node_cap: int,
) -> MicroSearchResult:
    """Deterministic explicit-state BFS used by conformance micro-fixtures."""

    if horizon < 0 or node_cap <= 0:
        raise ValueError("horizon and node cap must be positive bounds")
    ordered_actions = tuple(sorted(set(actions)))
    queue = deque([(initial, tuple())])
    seen = {signature(initial)}
    edges = 0
    while queue:
        state, history = queue.popleft()
        if target(state):
            return MicroSearchResult(True, history, len(seen), edges, False)
        if len(history) == horizon:
            continue
        for action in ordered_actions:
            successor = transition(state, action)
            edges += 1
            if successor is None:
                continue
            key = signature(successor)
            if key in seen:
                continue
            if len(seen) >= node_cap:
                return MicroSearchResult(False, (), len(seen), edges, True)
            seen.add(key)
            queue.append((successor, (*history, action)))
    return MicroSearchResult(False, (), len(seen), edges, False)


def merge_congruence_holds(
    left: Any, right: Any, actions: Iterable[str],
    transition: Callable[[Any, str], Any | None],
    signature: Callable[[Any], str], classify: Callable[[Any], str],
) -> bool:
    """Dynamically reject a merge key whose one-step behavior differs."""

    for action in sorted(set(actions)):
        left_next, right_next = transition(left, action), transition(right, action)
        if (left_next is None) != (right_next is None):
            return False
        if left_next is None:
            continue
        if (
            signature(left_next) != signature(right_next)
            or classify(left_next) != classify(right_next)
        ):
            return False
    return classify(left) == classify(right)


def _state_signature(env: MultiTownStatefulOpsEnv, monitor: Mapping[str, Any]) -> str:
    idempotency = {
        key: [fingerprint, result, error, payload]
        for key, (fingerprint, result, error, payload) in sorted(
            env._idempotency.items()
        )
    }
    value = {
        "world": env.state,
        "private": env.private_state,
        "runtime": env.observation()["runtime"],
        "idempotency": idempotency,
        "applied_event_ids": sorted(env._applied_event_ids),
        "ever_agent_changed": sorted("/".join(p) for p in env._ever_agent_changed_paths),
        "ever_external_changed": sorted("/".join(p) for p in env._ever_external_changed_paths),
        "monitor": monitor,
    }
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _classification(production_success: bool, oracle: Mapping[str, Any]) -> str:
    valid = bool(oracle.get("goal_valid") and oracle.get("temporal_valid"))
    safe = bool(oracle.get("safe"))
    if production_success:
        if not safe:
            return "accepted_unsafe"
        if not valid:
            return "accepted_invalid"
        return "accepted_valid_safe"
    if valid and safe:
        return "rejected_safe"
    return "rejected_unsafe_or_invalid"


def _classification_labels(
    production_success: bool, oracle: Mapping[str, Any],
) -> dict[str, bool]:
    return {
        "production_accept": production_success,
        "oracle_goal_valid": bool(oracle.get("goal_valid")),
        "oracle_temporal_valid": bool(oracle.get("temporal_valid")),
        "oracle_safe": bool(oracle.get("safe")),
        "oracle_integrity_valid": bool(oracle.get("integrity_valid")),
    }


def _mutate_oracle_spec(spec: dict[str, Any], mutation_id: str | None) -> None:
    if mutation_id == "oracle_drop_direct_sequence":
        spec["accepted_state_changing_sequences"] = [
            sequence for sequence in spec["accepted_state_changing_sequences"]
            if "stage_decision" in sequence
        ]
    elif mutation_id == "oracle_drop_staged_sequence":
        spec["accepted_state_changing_sequences"] = [
            sequence for sequence in spec["accepted_state_changing_sequences"]
            if "set_decision" in sequence
        ]
    elif mutation_id == "oracle_flip_expected_decision":
        spec["expected_decision"] = (
            "deny" if spec["expected_decision"] == "approve" else "approve"
        )


def _production_success(result: Any, mutation_id: str | None) -> bool:
    if mutation_id == "production_ignore_attempted_violation":
        return not (
            set(result.failure_codes) - {"ATTEMPTED_POLICY_VIOLATION"}
        )
    return bool(result.success)


def _confirm_public_replay(
    scenario: Any, spec: Mapping[str, Any], actions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Confirm one witness through facade, serialized trace, and neutral oracle."""

    facade = StatefulPOMDPEnv(scenario)
    facade.reset()
    facade_reward = 0.0
    for action in (*actions, {"action": "stop"}):
        _, facade_reward, terminated, truncated, _ = facade.step(action)
    if not terminated or truncated or facade_reward != 1.0:
        raise AssertionError("internal success did not reproduce through policy facade")
    contents = [_canonical(action) for action in actions]
    contents.append('{"action":"stop"}')
    rows, result = run_scripted_model_actions(scenario, contents)
    validate_trace_against_scenario(scenario, rows)
    steps = [
        {
            "turn_id": turn_id,
            "before_world": row["observation"]["world"],
            "after_world": rows[turn_id + 1]["observation"]["world"],
            "action": json.loads(row["action_content"]),
            "result": row["tool_result"],
        }
        for turn_id, row in enumerate(rows[:-1])
    ]
    public_oracle = evaluate_records_direct(
        spec, steps, rows[-1]["observation"]["world"],
    )
    policy: dict[str, str] = {}
    conflicts = 0
    observation_history: list[dict[str, Any]] = []
    for row in rows[:-1]:
        history_key = hashlib.sha256(_canonical(
            [*observation_history, row["observation"]]
        ).encode()).hexdigest()
        action_key = row["action_content"]
        prior_action = policy.setdefault(history_key, action_key)
        conflicts += int(prior_action != action_key)
        observation_history.append(row["observation"])
    return {
        "facade_reward": facade_reward,
        "serialized_trace_valid": True,
        "trace_rows": len(rows),
        "public_oracle": public_oracle,
        "production_result": result,
        "public_history_policy": {
            "entries": len(policy), "conflicts": conflicts,
            "single_world_only": True,
        },
    }


def _confirm_counterexample_replay(
    scenario: Any, spec: Mapping[str, Any], actions: Sequence[Mapping[str, Any]],
    mutation_id: str,
) -> dict[str, Any]:
    """Replay one disagreement without pretending the facade itself is mutated."""

    facade = StatefulPOMDPEnv(scenario)
    facade.reset()
    original_facade_reward = 0.0
    for action in (*actions, {"action": "stop"}):
        _, original_facade_reward, terminated, truncated, _ = facade.step(action)
    if not terminated or truncated:
        raise AssertionError("counterexample did not terminate through policy facade")
    contents = [_canonical(action) for action in actions]
    contents.append('{"action":"stop"}')
    rows, compact_result = run_scripted_model_actions(scenario, contents)
    validate_trace_against_scenario(scenario, rows)
    internal = MultiTownStatefulOpsEnv(scenario)
    for action in actions:
        internal.call_tool(
            str(action["tool_name"]), dict(action["arguments"]),
            idempotency_key=action.get("idempotency_key"),
        )
    full_result = internal.stop()
    steps = [{
        "turn_id": turn_id,
        "before_world": row["observation"]["world"],
        "after_world": rows[turn_id + 1]["observation"]["world"],
        "action": json.loads(row["action_content"]),
        "result": row["tool_result"],
    } for turn_id, row in enumerate(rows[:-1])]
    oracle = evaluate_records_direct(
        spec, steps, rows[-1]["observation"]["world"],
    )
    counterfactual_success = _production_success(full_result, mutation_id)
    category = _classification(counterfactual_success, oracle)
    return {
        "serialized_trace_valid": True,
        "trace_rows": len(rows),
        "original_facade_reward": original_facade_reward,
        "original_compact_result": compact_result,
        "original_production_success": full_result.success,
        "original_failure_codes": list(full_result.failure_codes),
        "counterfactual_mutant_success": counterfactual_success,
        "oracle": oracle,
        "category": category,
        "labels": _classification_labels(counterfactual_success, oracle),
        "mutated_facade_executed": False,
    }


@dataclass
class _Node:
    env: MultiTownStatefulOpsEnv
    actions: tuple[dict[str, Any], ...]
    steps: tuple[dict[str, Any], ...]
    monitor: dict[str, Any]


def _advance_node(node: _Node, action: Mapping[str, Any]) -> _Node:
    successor = copy.deepcopy(node.env)
    before = copy.deepcopy(successor.state)
    result = successor.call_tool(
        str(action["tool_name"]), dict(action["arguments"]),
        idempotency_key=action.get("idempotency_key"),
    )
    step = {
        "turn_id": len(node.steps),
        "before_world": before,
        "after_world": copy.deepcopy(successor.state),
        "action": copy.deepcopy(action),
        "result": copy.deepcopy(result),
    }
    steps = (*node.steps, step)
    return _Node(
        successor, (*node.actions, copy.deepcopy(dict(action))), steps, {},
    )


def _refresh_monitor(node: _Node, spec: Mapping[str, Any]) -> _Node:
    result = evaluate_records_direct(spec, node.steps, node.env.state)
    return _Node(node.env, node.actions, node.steps, {
        "state_changing_tools": result["state_changing_tools"],
        "issues": result["issues"],
    })


def _real_merge_congruence(
    left: _Node, right: _Node, spec: Mapping[str, Any],
    mutation_id: str | None,
) -> bool:
    """Check stop and every current public action for a real signature collision."""

    left_oracle = evaluate_records_direct(spec, left.steps, left.env.state)
    right_oracle = evaluate_records_direct(spec, right.steps, right.env.state)
    left_stop, right_stop = copy.deepcopy(left.env), copy.deepcopy(right.env)
    if (
        _classification(_production_success(left_stop.stop(), mutation_id), left_oracle)
        != _classification(_production_success(right_stop.stop(), mutation_id), right_oracle)
    ):
        return False
    left_actions = {
        _canonical(action): action
        for action in grounded_public_actions(left.env.observation())
        if action["action"] == "call_tool"
    }
    right_actions = {
        _canonical(action): action
        for action in grounded_public_actions(right.env.observation())
        if action["action"] == "call_tool"
    }
    if set(left_actions) != set(right_actions):
        return False
    for key in sorted(left_actions):
        left_next = _refresh_monitor(_advance_node(left, left_actions[key]), spec)
        right_next = _refresh_monitor(_advance_node(right, right_actions[key]), spec)
        if (
            _state_signature(left_next.env, left_next.monitor)
            != _state_signature(right_next.env, right_next.monitor)
        ):
            return False
        left_terminal, right_terminal = (
            copy.deepcopy(left_next.env), copy.deepcopy(right_next.env)
        )
        left_next_oracle = evaluate_records_direct(
            spec, left_next.steps, left_next.env.state,
        )
        right_next_oracle = evaluate_records_direct(
            spec, right_next.steps, right_next.env.state,
        )
        if (
            _classification(
                _production_success(left_terminal.stop(), mutation_id),
                left_next_oracle,
            )
            != _classification(
                _production_success(right_terminal.stop(), mutation_id),
                right_next_oracle,
            )
        ):
            return False
    return True


def run_records_direct_vertical_slice(
    *, world_seed: int = 160, horizon: int = 3, node_cap: int = 50_000,
    mutation_id: str | None = None,
) -> dict[str, Any]:
    """Exhaust grounded null-key traces for one train variant up to ``horizon``."""

    if horizon < 0 or node_cap <= 0:
        raise ValueError("horizon must be nonnegative and node_cap positive")
    if mutation_id is not None and mutation_id not in DECLARED_SEARCH_MUTANTS:
        raise ValueError("undeclared G2 search mutant")
    world = next(
        row for row in trusted_world_catalog(world_seed)
        if row.family == "records_casework"
        and row.variant_id == "direct_or_staged"
    )
    initial_env = MultiTownStatefulOpsEnv(world.scenario)
    spec = build_records_direct_spec(initial_env.state)
    _mutate_oracle_spec(spec, mutation_id)
    initial_monitor = {"state_changing_tools": [], "issues": []}
    queue = deque([_Node(initial_env, (), (), initial_monitor)])
    initial_signature = _state_signature(initial_env, initial_monitor)
    seen = {initial_signature}
    representatives = {
        initial_signature: _Node(initial_env, (), (), initial_monitor),
    }
    terminal_counts: dict[str, int] = {}
    counterexamples: list[dict[str, Any]] = []
    accepted_witnesses: list[dict[str, Any]] = []
    oracle_out_of_scope = 0
    generated_edges = executed_edges = merged = merge_congruence_checks = 0
    pruned = {"production_monotonic_violation": 0, "budget": 0}
    stop_checks = 0
    cap_hit = False
    while queue and not cap_hit:
        node = queue.popleft()
        terminal_env = copy.deepcopy(node.env)
        production = terminal_env.stop()
        oracle = evaluate_records_direct(spec, node.steps, node.env.state)
        oracle_out_of_scope += int(
            oracle.get("status") == "out_of_scope"
            or "OUT_OF_ORACLE_SCOPE" in oracle.get("issues", [])
        )
        production_success = _production_success(production, mutation_id)
        category = _classification(production_success, oracle)
        terminal_counts[category] = terminal_counts.get(category, 0) + 1
        stop_checks += 1
        if category in {"accepted_invalid", "accepted_unsafe"}:
            confirmation = _confirm_counterexample_replay(
                world.scenario, spec, node.actions, str(mutation_id),
            ) if mutation_id is not None else None
            if confirmation is not None and (
                confirmation["category"] != category
                or confirmation["labels"]
                != _classification_labels(production_success, oracle)
            ):
                raise AssertionError("counterexample replay classification diverged")
            counterexamples.append({
                "category": category,
                "labels": _classification_labels(production_success, oracle),
                "actions": list(alpha_normalize_idempotency_keys(node.actions)),
                "oracle": oracle,
                "confirmation": confirmation,
            })
        if production.success:
            replay = _confirm_public_replay(
                world.scenario, spec, node.actions,
            )
            replay_category = _classification(
                bool(replay["production_result"]["success"]),
                replay["public_oracle"],
            )
            if replay_category != category:
                raise AssertionError("public replay classification diverged")
            accepted_witnesses.append({
                "category": category,
                "labels": _classification_labels(production.success, oracle),
                "actions": list(alpha_normalize_idempotency_keys(node.actions)),
                "confirmation": replay,
            })
        if len(node.actions) == horizon:
            continue
        actions = tuple(
            action for action in grounded_public_actions(node.env.observation())
            if action["action"] == "call_tool"
        )
        for action in actions:
            generated_edges += 1
            candidate = _advance_node(node, action)
            successor = candidate.env
            result = candidate.steps[-1]["result"]
            executed_edges += 1
            monitor_result = evaluate_records_direct(
                spec, candidate.steps, successor.state,
            )
            oracle_out_of_scope += int(
                monitor_result.get("status") == "out_of_scope"
                or "OUT_OF_ORACLE_SCOPE" in monitor_result.get("issues", [])
            )
            monitor = {
                "state_changing_tools": monitor_result["state_changing_tools"],
                "issues": monitor_result["issues"],
            }
            candidate = _Node(
                successor, candidate.actions, candidate.steps, monitor,
            )
            runtime = successor.observation()["runtime"]
            if result["error_code"] == "BUDGET_EXHAUSTED":
                pruned["budget"] += 1
                continue
            if (
                runtime["attempted_policy_violations"]
                or runtime["executed_safety_violations"]
                or runtime["budget_violations"]
            ) and mutation_id != "production_ignore_attempted_violation":
                pruned["production_monotonic_violation"] += 1
                continue
            key = _state_signature(successor, monitor)
            if key in seen:
                merge_congruence_checks += 1
                if not _real_merge_congruence(
                    representatives[key], candidate, spec, mutation_id,
                ):
                    raise AssertionError("real state merge failed one-step congruence")
                merged += 1
                continue
            if len(seen) >= node_cap:
                cap_hit = True
                break
            seen.add(key)
            representatives[key] = candidate
            queue.append(candidate)
    horizon_frontier_exhausted = not queue and not cap_hit
    return {
        "schema_version": REPORT_VERSION,
        "stage": "train",
        "source_state": _source_state(),
        "search_version": SEARCH_VERSION,
        "oracle_version": ORACLE_VERSION,
        "oracle_spec_sha256": oracle_spec_sha256(spec),
        "action_partition_manifest": {
            "schema_version": MANIFEST_VERSION, "sha256": manifest_sha256(),
        },
        "grounding_version": GROUNDING_VERSION,
        "scope": {
            "world_seed": world_seed,
            "family": "records_casework",
            "variant": "direct_or_staged",
            "private_worlds": 1,
            "horizon": horizon,
            "action_abstraction": "dynamic public grounded calls; null idempotency key only",
            "node_cap": node_cap,
            "full_benchmark_scope_complete": False,
            "natural_budget_frontier_exhausted": False,
            "mutation_id": mutation_id,
        },
        "counts": {
            "unique_states": len(seen), "generated_edges": generated_edges,
            "executed_edges": executed_edges, "merged_successors": merged,
            "merge_congruence_checks": merge_congruence_checks,
            "stop_checks": stop_checks, "pruned": pruned,
            "terminal_classifications": dict(sorted(terminal_counts.items())),
        },
        "horizon_frontier_exhausted": horizon_frontier_exhausted,
        "caps_hit": cap_hit,
        "oracle_out_of_scope": oracle_out_of_scope,
        "counterexamples": counterexamples,
        "accepted_witnesses": accepted_witnesses,
        "open_loop_facade_reproducible_witnesses": sum(
            row["confirmation"]["public_history_policy"]["conflicts"] == 0
            for row in accepted_witnesses
        ),
        "vertical_slice_pass": bool(
            mutation_id is None
            and
            horizon_frontier_exhausted and not cap_hit
            and not counterexamples and not oracle_out_of_scope
            and accepted_witnesses
            and merge_congruence_checks == merged
            and all(
                row["confirmation"]["facade_reward"] == 1.0
                and row["confirmation"]["serialized_trace_valid"]
                and row["confirmation"]["public_oracle"]["safe"]
                and row["confirmation"]["public_oracle"]["goal_valid"]
                and row["confirmation"]["public_oracle"]["temporal_valid"]
                and row["confirmation"]["public_history_policy"]["conflicts"] == 0
                for row in accepted_witnesses
            )
        ),
        "complete": False,
        "claim_boundary": (
            "initial train-only vertical slice for one world, one variant, grounded "
            "null-key calls, and a declared finite horizon; never a full-benchmark "
            "accepted-unsafe absence result, robust multi-world policy, formal proof, "
            "or Agentic RL"
        ),
    }


def _mutation_counterexample_matches(
    row: Mapping[str, Any], *, mutation_id: str,
    expected_category: str, required_labels: Mapping[str, bool],
) -> bool:
    """Bind a mutant kill to a nonempty, independently replayed witness."""

    actions = row.get("actions")
    confirmation = row.get("confirmation")
    oracle = row.get("oracle")
    if (
        not isinstance(actions, list) or not actions
        or not isinstance(confirmation, Mapping)
        or not isinstance(oracle, Mapping)
        or row.get("category") != expected_category
        or row.get("labels") != required_labels
        or confirmation.get("category") != expected_category
        or confirmation.get("labels") != required_labels
        or confirmation.get("oracle") != oracle
        or confirmation.get("serialized_trace_valid") is not True
        or confirmation.get("trace_rows") != len(actions) + 1
        or confirmation.get("counterfactual_mutant_success") is not True
        or confirmation.get("mutated_facade_executed") is not False
    ):
        return False
    compact = confirmation.get("original_compact_result")
    if not isinstance(compact, Mapping):
        return False
    if mutation_id.startswith("oracle_"):
        return bool(
            confirmation.get("original_production_success") is True
            and confirmation.get("original_facade_reward") == 1.0
            and compact.get("success") is True
            and confirmation.get("original_failure_codes") == []
        )
    if mutation_id == "production_ignore_attempted_violation":
        return bool(
            confirmation.get("original_production_success") is False
            and confirmation.get("original_facade_reward") == 0.0
            and compact.get("success") is False
            and confirmation.get("original_failure_codes")
            == ["ATTEMPTED_POLICY_VIOLATION"]
            and oracle.get("goal_valid") is True
            and oracle.get("temporal_valid") is True
            and oracle.get("safe") is False
            and oracle.get("integrity_valid") is True
            and "POLICY_DENIAL_ATTEMPT" in oracle.get("issues", [])
        )
    return False


def _mutation_search_counts_consistent(counts: Mapping[str, Any]) -> bool:
    """Check BFS accounting identities before a subreport can kill a mutant."""

    scalar_names = (
        "unique_states", "generated_edges", "executed_edges",
        "merged_successors", "merge_congruence_checks", "stop_checks",
    )
    if any(
        isinstance(counts.get(name), bool)
        or not isinstance(counts.get(name), int)
        or counts[name] < 0
        for name in scalar_names
    ):
        return False
    pruned = counts.get("pruned")
    terminal = counts.get("terminal_classifications")
    if (
        not isinstance(pruned, Mapping)
        or not isinstance(terminal, Mapping)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (*pruned.values(), *terminal.values())
        )
    ):
        return False
    unique = counts["unique_states"]
    generated = counts["generated_edges"]
    merged = counts["merged_successors"]
    return bool(
        unique > 0
        and counts["stop_checks"] == unique
        and counts["merge_congruence_checks"] == merged
        and generated == counts["executed_edges"]
        and sum(terminal.values()) == counts["stop_checks"]
        and generated == sum(pruned.values()) + merged + unique - 1
    )


def run_g2_mutation_audit(
    *, world_seed: int = MUTATION_CAMPAIGN_WORLD_SEED,
) -> dict[str, Any]:
    """Require deterministic search to expose every declared vertical-slice mutant."""

    rows = []
    for mutation_id, expected_category in sorted(DECLARED_SEARCH_MUTANTS.items()):
        horizon = 4 if mutation_id == "production_ignore_attempted_violation" else 3
        report = run_records_direct_vertical_slice(
            world_seed=world_seed, horizon=horizon, node_cap=100_000,
            mutation_id=mutation_id,
        )
        required_labels = MUTANT_REQUIRED_LABELS[mutation_id]
        world = next(
            world for world in trusted_world_catalog(world_seed)
            if world.family == "records_casework"
            and world.variant_id == "direct_or_staged"
        )
        baseline_spec = build_records_direct_spec(
            world.scenario.private_evaluator.initial_state()
        )
        mutated_spec = copy.deepcopy(baseline_spec)
        _mutate_oracle_spec(mutated_spec, mutation_id)
        matching = []
        for row in report["counterexamples"]:
            if not _mutation_counterexample_matches(
                row, mutation_id=mutation_id,
                expected_category=expected_category,
                required_labels=required_labels,
            ):
                continue
            try:
                independent_confirmation = _confirm_counterexample_replay(
                    world.scenario, mutated_spec, row["actions"], mutation_id,
                )
            except Exception:
                continue
            if independent_confirmation != row["confirmation"]:
                continue
            matching.append(row)
        rows.append({
            "mutation_id": mutation_id,
            "operator_class": MUTANT_OPERATOR_CLASSES[mutation_id],
            "expected_category": expected_category,
            "required_labels": required_labels,
            "killed": bool(matching),
            "counterexample_count": len(report["counterexamples"]),
            "matching_confirmation_count": len(matching),
            "independently_replayed_matching_count": len(matching),
            "shortest_witness_length": min(
                (len(row["actions"]) for row in matching),
                default=None,
            ),
            "horizon": horizon,
            "caps_hit": report["caps_hit"],
            "horizon_frontier_exhausted": report["horizon_frontier_exhausted"],
            "counts": report["counts"],
            "counts_consistent": _mutation_search_counts_consistent(
                report["counts"]
            ),
            "counts_match_frozen_expected": (
                report["counts"]
                == MUTANT_EXPECTED_SEARCH_COUNTS[mutation_id]
            ),
            "expected_counts_sha256": hashlib.sha256(_canonical(
                MUTANT_EXPECTED_SEARCH_COUNTS[mutation_id]
            ).encode()).hexdigest(),
            "counterexample_sha256": hashlib.sha256(_canonical(
                report["counterexamples"]
            ).encode()).hexdigest(),
            "baseline_oracle_spec_sha256": oracle_spec_sha256(baseline_spec),
            "mutated_oracle_spec_sha256": oracle_spec_sha256(mutated_spec),
            "oracle_spec_changed": baseline_spec != mutated_spec,
            "counterfactual_terminal_acceptor_changed": (
                mutation_id == "production_ignore_attempted_violation"
            ),
        })
    return {
        "schema_version": MUTATION_REPORT_VERSION,
        "stage": "train",
        "source_state": _source_state(),
        "scope": {
            "world_seed": world_seed,
            "frozen_campaign_world_seed": MUTATION_CAMPAIGN_WORLD_SEED,
            "frozen_campaign_match": world_seed == MUTATION_CAMPAIGN_WORLD_SEED,
        },
        "declared_mutants": dict(sorted(DECLARED_SEARCH_MUTANTS.items())),
        "operator_classes": dict(sorted(MUTANT_OPERATOR_CLASSES.items())),
        "rows": rows,
        "mutants_killed": sum(row["killed"] for row in rows),
        "mutants_total": len(rows),
        "operator_classes_killed": len({
            row["operator_class"] for row in rows if row["killed"]
        }),
        "operator_classes_total": len(set(MUTANT_OPERATOR_CLASSES.values())),
        "all_killed": (
            world_seed == MUTATION_CAMPAIGN_WORLD_SEED
            and {row["mutation_id"] for row in rows}
            == set(DECLARED_SEARCH_MUTANTS)
            and len(rows) == len(DECLARED_SEARCH_MUTANTS)
            and all(
            row["killed"]
            and row["counterexample_count"] > 0
            and row["shortest_witness_length"] is not None
            and row["shortest_witness_length"] > 0
            and row["shortest_witness_length"] <= row["horizon"]
            and row["matching_confirmation_count"] > 0
            and row["matching_confirmation_count"] <= row["counterexample_count"]
            and row["counts_consistent"]
            and row["counts_match_frozen_expected"]
            and row["oracle_spec_changed"]
            == row["mutation_id"].startswith("oracle_")
            and row["counterfactual_terminal_acceptor_changed"]
            == (row["mutation_id"] == "production_ignore_attempted_violation")
            and (
                (row["baseline_oracle_spec_sha256"]
                 != row["mutated_oracle_spec_sha256"])
                == row["oracle_spec_changed"]
            )
            and not row["caps_hit"]
            and row["horizon_frontier_exhausted"]
            for row in rows
            )
        ),
        "complete": False,
        "claim_boundary": (
            "mutation adequacy for four declared G2 vertical-slice mutants only; "
            "not full oracle, transition, validator, or searcher mutation adequacy"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the G2 bounded-search slice")
    parser.add_argument("--world-seed", type=int, default=160)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--node-cap", type=int, default=50_000)
    parser.add_argument(
        "--mutation-audit", action="store_true",
        help="run the four-instance detector-sensitivity audit instead",
    )
    args = parser.parse_args()
    report = (
        run_g2_mutation_audit(world_seed=args.world_seed)
        if args.mutation_audit
        else run_records_direct_vertical_slice(
            world_seed=args.world_seed, horizon=args.horizon,
            node_cap=args.node_cap,
        )
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
