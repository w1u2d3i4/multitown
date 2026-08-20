"""Finite public-history belief planning for the train-only G16 triplet.

This is a train-free reachability baseline over an explicitly finite grounded
action abstraction.  It is not a learned policy and is not evidence of
Agentic RL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .stateful_authority_bench import BRANCHES, run_authority_baselines
from .stateful_grounding import grounded_public_actions
from .stateful_ops import StatefulScenario, build_scenario, tool_profile
from .stateful_pomdp import StatefulPOMDPEnv, _parse_action


PLANNER_VERSION = "multitown-public-belief-planner-v1"
REPORT_VERSION = "multitown-g16-belief-planner-report-v1"
POLICY_TREE_SCHEMA_VERSION = "multitown-public-policy-tree-v1"
MISSING_REFERENCE = "UNAVAILABLE-PUBLIC-REFERENCE"
MAX_PLANNER_WORLDS = 16
MAX_PLANNER_DEPTH = 32
MAX_POLICY_TREE_NODES = 512


class PolicyTreeMismatch(ValueError):
    """A serialized policy tree does not match the current public history."""


def _source_state() -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project_root,
        capture_output=True, check=True, text=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=project_root, capture_output=True, check=True, text=True,
    ).stdout)
    return {"revision": revision, "dirty": dirty}


def _canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _call(tool_name: str, arguments: Mapping[str, str]) -> dict[str, Any]:
    return {
        "action": "call_tool", "tool_name": tool_name,
        "arguments": dict(arguments), "idempotency_key": None,
    }


def _authority_context(observation: Mapping[str, Any]) -> dict[str, str]:
    world = observation["world"]
    permit_id = next(iter(world["permits"]))
    permit = world["permits"][permit_id]
    return {
        "permit_id": str(permit_id),
        "applicant_id": str(permit["applicant_id"]),
        "scope": str(permit["authority_scope"]),
        "policy_version": str(world["protected"]["policy_version"]),
        "request_id": str(
            permit["authority_request_id"] or MISSING_REFERENCE
        ),
        "expiry": str(permit["authority_expiry"]),
        "capability": str(
            permit["authority_capability"] or MISSING_REFERENCE
        ),
        "authority_status": str(permit["authority_status"]),
        "permit_status": str(permit["status"]),
        "authority_checks": str(permit["authority_checks"]),
        "human_escalated": str(permit["human_escalated"]),
        "notice_sent": str(permit_id in world["notifications"]),
    }


def grounded_authority_actions(
    observation: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Ground one public representative per typed G16 tool plus ``stop``.

    The abstraction covers every tool name.  Identifiers and credentials use
    visible entities; an unavailable reference maps to one explicit missing-
    reference equivalence class.  It does not enumerate the infinite string
    argument space.
    """

    if observation["task"]["family"] != "permit_transaction":
        raise ValueError("authority grounding requires permit_transaction")
    context = _authority_context(observation)
    arguments = {
        "lookup_applicant": {"applicant_id": context["applicant_id"]},
        "get_permit": {"permit_id": context["permit_id"]},
        "record_inspection": {"permit_id": context["permit_id"]},
        "issue_permit": {"permit_id": context["permit_id"]},
        "approve_permit": {"permit_id": context["permit_id"]},
        "assign_permit_number": {"permit_id": context["permit_id"]},
        "send_permit_notice": {"permit_id": context["permit_id"]},
        "submit_authority_request": {
            "permit_id": context["permit_id"], "scope": context["scope"],
            "expected_policy_version": context["policy_version"],
        },
        "check_authority_status": {
            "permit_id": context["permit_id"],
            "request_id": context["request_id"],
        },
        "issue_permit_with_capability": {
            "permit_id": context["permit_id"],
            "request_id": context["request_id"], "scope": context["scope"],
            "policy_version": context["policy_version"],
            "expiry": context["expiry"], "capability": context["capability"],
        },
        "escalate_human": {
            "permit_id": context["permit_id"],
            "request_id": context["request_id"],
        },
        "send_permit_status_notice": {"permit_id": context["permit_id"]},
    }
    expected_tools = set(tool_profile("permit_transaction")["tools"])
    if set(arguments) != expected_tools:
        raise RuntimeError("G16 grounding no longer covers the complete tool profile")

    preferred: list[str] = []
    if context["notice_sent"] == "True":
        preferred.append("stop")
    elif context["authority_status"] == "not-requested":
        preferred.append("submit_authority_request")
    elif (
        context["authority_status"] == "pending"
        and int(context["authority_checks"]) < 2
    ):
        preferred.append("check_authority_status")
    elif context["authority_status"] == "granted":
        preferred.append("issue_permit_with_capability")
    elif context["authority_status"] == "timed-out" and (
        context["human_escalated"] == "False"
    ):
        preferred.append("escalate_human")
    elif context["authority_status"] in {"denied", "timed-out"} or (
        context["permit_status"] in {"approved", "escalated"}
    ):
        preferred.append("send_permit_status_notice")
    order = [
        *preferred,
        "submit_authority_request", "check_authority_status",
        "issue_permit_with_capability", "escalate_human",
        "send_permit_status_notice", "lookup_applicant", "get_permit",
        "record_inspection", "issue_permit", "approve_permit",
        "assign_permit_number", "send_permit_notice", "stop",
    ]
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name in order:
        action = {"action": "stop"} if name == "stop" else _call(name, arguments[name])
        key = _canonical(action)
        if key not in seen:
            seen.add(key)
            unique.append(action)
    return tuple(unique)


@dataclass(frozen=True)
class PolicyNode:
    belief_size: int
    observation_sha256: str
    action: dict[str, Any]
    terminal_success: bool
    outcomes: dict[str, "PolicyNode"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": POLICY_TREE_SCHEMA_VERSION,
            "belief_size": self.belief_size,
            "observation_sha256": self.observation_sha256,
            "action": self.action,
            "terminal_success": self.terminal_success,
            "outcomes": {
                key: value.to_dict() for key, value in sorted(self.outcomes.items())
            },
        }

    @classmethod
    def from_dict(
        cls, value: Any, *, _depth: int = 1, _counter: list[int] | None = None,
    ) -> "PolicyNode":
        """Load one exact versioned tree object and reject type/schema drift."""

        fields = {
            "schema_version", "belief_size", "observation_sha256", "action",
            "terminal_success", "outcomes",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise PolicyTreeMismatch("serialized policy node fields are invalid")
        if value["schema_version"] != POLICY_TREE_SCHEMA_VERSION:
            raise PolicyTreeMismatch("serialized policy tree version is unsupported")
        if _depth > MAX_PLANNER_DEPTH:
            raise PolicyTreeMismatch("serialized policy tree exceeds the depth limit")
        if _counter is None:
            _counter = [0]
        _counter[0] += 1
        if _counter[0] > MAX_POLICY_TREE_NODES:
            raise PolicyTreeMismatch("serialized policy tree exceeds the node limit")
        belief_size = value["belief_size"]
        if (
            not isinstance(belief_size, int) or isinstance(belief_size, bool)
            or not 0 < belief_size <= MAX_PLANNER_WORLDS
        ):
            raise PolicyTreeMismatch("serialized belief size is invalid")
        if not isinstance(value["observation_sha256"], str):
            raise PolicyTreeMismatch("serialized observation digest is invalid")
        if not isinstance(value["action"], dict):
            raise PolicyTreeMismatch("serialized policy action must be an object")
        if not isinstance(value["terminal_success"], bool):
            raise PolicyTreeMismatch("serialized terminal flag must be boolean")
        outcomes = value["outcomes"]
        if not isinstance(outcomes, dict):
            raise PolicyTreeMismatch("serialized policy outcomes must be an object")
        return cls(
            belief_size=belief_size,
            observation_sha256=value["observation_sha256"],
            action=value["action"],
            terminal_success=value["terminal_success"],
            outcomes={
                key: cls.from_dict(
                    child, _depth=_depth + 1, _counter=_counter,
                )
                for key, child in outcomes.items()
            },
        )


def validate_policy_tree(root: PolicyNode, *, family: str) -> dict[str, int]:
    """Reject malformed, cyclic, unbounded, or action-invalid policy trees."""

    if not isinstance(root, PolicyNode):
        raise PolicyTreeMismatch("policy tree root has the wrong type")
    stack = [(root, 1)]
    seen: set[int] = set()
    max_depth = 0
    while stack:
        node, depth = stack.pop()
        identity = id(node)
        if identity in seen:
            raise PolicyTreeMismatch("policy tree contains a cycle or shared node")
        seen.add(identity)
        if len(seen) > MAX_POLICY_TREE_NODES or depth > MAX_PLANNER_DEPTH:
            raise PolicyTreeMismatch("policy tree exceeds the audit size limit")
        max_depth = max(max_depth, depth)
        if (
            not isinstance(node.belief_size, int)
            or isinstance(node.belief_size, bool)
            or not 0 < node.belief_size <= MAX_PLANNER_WORLDS
        ):
            raise PolicyTreeMismatch("policy node belief size must be positive")
        if not isinstance(node.terminal_success, bool):
            raise PolicyTreeMismatch("policy terminal flag must be boolean")
        if not isinstance(node.action, dict) or not isinstance(node.outcomes, dict):
            raise PolicyTreeMismatch("policy node containers have invalid types")
        if len(node.observation_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in node.observation_sha256
        ):
            raise PolicyTreeMismatch("policy observation digest is invalid")
        try:
            parsed = _parse_action(node.action, family)
        except (TypeError, ValueError) as exc:
            raise PolicyTreeMismatch("policy node action is invalid") from exc
        if node.terminal_success != (parsed.action == "stop"):
            raise PolicyTreeMismatch("policy terminal flag and action disagree")
        if node.terminal_success and node.outcomes:
            raise PolicyTreeMismatch("terminal policy node cannot have outcomes")
        if not node.terminal_success and not node.outcomes:
            raise PolicyTreeMismatch("non-terminal policy node needs outcomes")
        if node.outcomes and sum(
            child.belief_size
            for child in node.outcomes.values()
            if isinstance(child, PolicyNode)
        ) != node.belief_size:
            raise PolicyTreeMismatch("policy outcome beliefs do not partition the parent")
        for outcome, child in node.outcomes.items():
            if len(outcome) != 64 or any(
                character not in "0123456789abcdef" for character in outcome
            ):
                raise PolicyTreeMismatch("policy outcome digest is invalid")
            if not isinstance(child, PolicyNode):
                raise PolicyTreeMismatch("policy outcome child has the wrong type")
            stack.append((child, depth + 1))
    return {"nodes": len(seen), "depth": max_depth}


@dataclass(frozen=True)
class _Rollout:
    observation: dict[str, Any]
    tool_result: dict[str, Any] | None
    terminal_result: dict[str, Any] | None
    terminated: bool
    truncated: bool


def _rollout(
    scenario: StatefulScenario, actions: Sequence[Mapping[str, Any]],
) -> _Rollout:
    env = StatefulPOMDPEnv(scenario)
    observation, _ = env.reset()
    tool_result = None
    terminal_result = None
    terminated = truncated = False
    for action in actions:
        observation, _, terminated, truncated, info = env.step(action)
        tool_result = info["tool_result"]
        terminal_result = info["terminal_result"]
    return _Rollout(
        observation, tool_result, terminal_result, terminated, truncated,
    )


def _outcome_key(rollout: _Rollout) -> str:
    return _digest({
        "observation": rollout.observation,
        "tool_result": rollout.tool_result,
        "terminal_result": rollout.terminal_result,
        "terminated": rollout.terminated,
        "truncated": rollout.truncated,
    })


class PublicBeliefPlanner:
    """Max-min reachability search whose decisions use public histories only."""

    def __init__(
        self, scenarios: Iterable[StatefulScenario], *, max_depth: int = 7,
        grounder: Callable[
            [Mapping[str, Any]], tuple[dict[str, Any], ...]
        ] = grounded_authority_actions,
    ):
        self._scenarios = tuple(scenarios)
        if not self._scenarios:
            raise ValueError("belief planner requires at least one world")
        if len(self._scenarios) > MAX_PLANNER_WORLDS:
            raise ValueError("belief planner world count exceeds the safe limit")
        if not 0 < max_depth <= MAX_PLANNER_DEPTH:
            raise ValueError("max_depth is outside the safe range")
        self.max_depth = max_depth
        self._grounder = grounder
        self.nodes_expanded = 0
        self.actions_evaluated = 0
        self._failed: set[str] = set()

    def solve(self) -> PolicyNode | None:
        paths = tuple((index, ()) for index in range(len(self._scenarios)))
        initial = [_rollout(self._scenarios[index], ()) for index, _ in paths]
        if len({_outcome_key(row) for row in initial}) != 1:
            raise ValueError("initial worlds must form one public belief state")
        return self._solve(paths, self.max_depth)

    def _solve(
        self, paths: tuple[tuple[int, tuple[Mapping[str, Any], ...]], ...],
        remaining: int,
    ) -> PolicyNode | None:
        rollouts = [
            _rollout(self._scenarios[index], history) for index, history in paths
        ]
        observations = {_canonical(row.observation) for row in rollouts}
        if len(observations) != 1:
            raise RuntimeError("one belief node must have one public observation")
        observation = rollouts[0].observation
        runtime = observation["runtime"]
        if remaining <= 0 or any((
            runtime["attempted_policy_violations"],
            runtime["executed_safety_violations"],
            runtime["budget_violations"],
        )):
            return None
        memo_key = _digest({
            "world_histories": [
                [index, [_canonical(action) for action in history]]
                for index, history in paths
            ],
            "remaining": remaining,
        })
        if memo_key in self._failed:
            return None
        self.nodes_expanded += 1
        for action in self._grounder(observation):
            self.actions_evaluated += 1
            next_rows: list[tuple[int, tuple[Mapping[str, Any], ...], _Rollout]] = []
            for index, history in paths:
                next_history = (*history, action)
                next_rows.append((
                    index, next_history,
                    _rollout(self._scenarios[index], next_history),
                ))
            if any(row.truncated for _, _, row in next_rows):
                continue
            if any(row.terminated for _, _, row in next_rows):
                if all(
                    row.terminated and row.terminal_result
                    and row.terminal_result["success"]
                    for _, _, row in next_rows
                ):
                    return PolicyNode(
                        belief_size=len(paths),
                        observation_sha256=_digest(observation), action=action,
                        terminal_success=True, outcomes={},
                    )
                continue
            partitions: dict[
                str, list[tuple[int, tuple[Mapping[str, Any], ...]]]
            ] = {}
            for index, history, row in next_rows:
                partitions.setdefault(_outcome_key(row), []).append((index, history))
            children: dict[str, PolicyNode] = {}
            feasible = True
            for outcome, partition in sorted(partitions.items()):
                child = self._solve(tuple(partition), remaining - 1)
                if child is None:
                    feasible = False
                    break
                children[outcome] = child
            if feasible:
                return PolicyNode(
                    belief_size=len(paths), observation_sha256=_digest(observation),
                    action=action, terminal_success=False, outcomes=children,
                )
        self._failed.add(memo_key)
        return None


def evaluate_policy_tree(
    scenario: StatefulScenario, root: PolicyNode,
) -> dict[str, Any]:
    """Fail-closed replay using only public observation and outcome hashes."""

    env = StatefulPOMDPEnv(scenario)
    observation, _ = env.reset()
    validate_policy_tree(root, family=observation["task"]["family"])
    node = root
    calls = 0
    while True:
        if node.observation_sha256 != _digest(observation):
            raise PolicyTreeMismatch(
                "policy node does not match the current public observation"
            )
        observation, reward, terminated, truncated, info = env.step(node.action)
        calls += node.action["action"] == "call_tool"
        if terminated or truncated:
            terminal = info["terminal_result"] or {}
            return {
                "success": bool(reward == 1.0 and terminal.get("success")),
                "terminated": terminated, "truncated": truncated,
                "tool_calls": calls,
                "logical_latency": observation["runtime"]["logical_latency_used"],
                "irreversible_risk": observation["runtime"]["irreversible_risk_used"],
                "attempted_policy_violations": observation["runtime"][
                    "attempted_policy_violations"
                ],
                "safety_violations": terminal.get("safety_violations", 0),
                "budget_violations": terminal.get("budget_violations", 0),
            }
        outcome = _outcome_key(_Rollout(
            observation=observation, tool_result=info["tool_result"],
            terminal_result=None, terminated=False, truncated=False,
        ))
        if outcome not in node.outcomes:
            raise PolicyTreeMismatch(
                "policy tree has no branch for the observed public outcome"
            )
        node = node.outcomes[outcome]


def _tree_stats(root: PolicyNode) -> dict[str, Any]:
    stack = [(root, 1)]
    nodes = leaves = depth = 0
    belief_sizes: list[int] = []
    while stack:
        node, level = stack.pop()
        nodes += 1
        depth = max(depth, level)
        belief_sizes.append(node.belief_size)
        if node.terminal_success:
            leaves += 1
        stack.extend((child, level + 1) for child in node.outcomes.values())
    return {
        "policy_nodes": nodes, "terminal_leaves": leaves,
        "policy_depth": depth, "belief_sizes": belief_sizes,
    }


def run_g16_belief_planner(*, world_seed: int = 1) -> dict[str, Any]:
    if world_seed < 0:
        raise ValueError("world seed must be non-negative")
    scenarios = [
        build_scenario(
            "permit_transaction", world_seed=world_seed,
            variant_id="asynchronous_authority_timeout",
            dynamics_branch=branch,
        )
        for branch in BRANCHES
    ]
    planner = PublicBeliefPlanner(scenarios)
    tree = planner.solve()
    if tree is None:
        public_rows = [
            {"branch": branch, "success": False} for branch in BRANCHES
        ]
    else:
        public_rows = [
            {"branch": branch, **evaluate_policy_tree(scenario, tree)}
            for branch, scenario in zip(BRANCHES, scenarios, strict=True)
        ]

    oracle_rows = []
    for branch, scenario in zip(BRANCHES, scenarios, strict=True):
        oracle = PublicBeliefPlanner([scenario])
        oracle_tree = oracle.solve()
        result = (
            evaluate_policy_tree(scenario, oracle_tree)
            if oracle_tree is not None else {"success": False}
        )
        oracle_rows.append({"branch": branch, **result})
    fixed = run_authority_baselines(world_seed=world_seed)
    fixed_summary = {
        row["policy"]: row["strict_successes"]
        for row in fixed["summaries"]
    }
    serialized_tree = tree.to_dict() if tree is not None else None
    return {
        "schema_version": REPORT_VERSION,
        "planner_version": PLANNER_VERSION,
        "stage": "train",
        "family": "permit_transaction",
        "variant_id": "asynchronous_authority_timeout",
        "world_seed": world_seed,
        "world_count": len(scenarios),
        "source_state": _source_state(),
        "scenario_set_sha256": _digest(sorted(
            scenario.private_instance_id for scenario in scenarios
        )),
        "action_abstraction": {
            "kind": "finite-public-grounding",
            "actions_per_observation": len(grounded_authority_actions(
                StatefulPOMDPEnv(scenarios[0]).reset()[0]
            )),
            "coverage": "every typed permit tool plus stop",
            "argument_scope": (
                "one visible-value grounding per field and one missing-reference "
                "equivalence class; not exhaustive over arbitrary JSON strings"
            ),
        },
        "private_world_oracle": {
            "role": "separate-world reachability upper bound; not deployable",
            "all_worlds_reachable": all(row["success"] for row in oracle_rows),
            "rows": oracle_rows,
        },
        "public_history_belief_planner": {
            "role": "deployable public-observation policy prototype",
            "robust_triplet_success": bool(
                tree is not None and all(row["success"] for row in public_rows)
            ),
            "nodes_expanded": planner.nodes_expanded,
            "actions_evaluated": planner.actions_evaluated,
            "rows": public_rows,
            "tree_stats": _tree_stats(tree) if tree is not None else None,
            "policy_tree_sha256": (
                _digest(serialized_tree) if serialized_tree is not None else None
            ),
            "policy_tree": serialized_tree,
        },
        "fixed_policy_strict_successes": fixed_summary,
        "gate": {
            "oracle_all_worlds": all(row["success"] for row in oracle_rows),
            "public_robust_success": bool(
                tree is not None and all(row["success"] for row in public_rows)
            ),
            "no_executed_safety_violation": all(
                row.get("safety_violations", 0) == 0 for row in public_rows
            ),
            "no_attempted_policy_violation": all(
                row.get("attempted_policy_violations", 0) == 0
                for row in public_rows
            ),
            "no_budget_violation": all(
                row.get("budget_violations", 0) == 0 for row in public_rows
            ),
        },
        "claim_boundary": (
            "train-only finite reachability and policy-synthesis evidence for G16; "
            "not full JSON action-space exhaustiveness, a model result, a held-out "
            "result, policy training, or Agentic RL evidence"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the train-only G16 public belief planner",
    )
    parser.add_argument("--world-seed", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(
        run_g16_belief_planner(world_seed=args.world_seed),
        ensure_ascii=False, indent=2, sort_keys=True,
    ))


if __name__ == "__main__":
    main()
