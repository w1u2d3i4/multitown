"""Train-only reachability and robust public-history synthesis for 16 groups."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any, Iterable

from .stateful_belief_planner import (
    PublicBeliefPlanner,
    _digest,
    _source_state,
    _tree_stats,
    evaluate_policy_tree,
)
from .stateful_grounding import (
    GROUNDING_VERSION,
    grounded_public_actions,
    grounding_fingerprint,
    grounding_implementation_sha256,
)
from .stateful_groups import variants_for_family
from .stateful_ops import FAMILIES, StatefulScenario, build_scenario
from .stateful_pomdp import StatefulPOMDPEnv


REPORT_VERSION = "multitown-stateful-reachability-report-v1"
DEFAULT_AUDIT_SEEDS = (160, 161, 162)


PRIVATE_BRANCHES: dict[tuple[str, str], tuple[str, ...]] = {
    ("records_casework", "conflicting_evidence_investigation"): (
        "candidate_a_eligible", "candidate_a_ineligible",
        "candidate_b_eligible", "candidate_b_ineligible",
    ),
    ("resource_calendar", "optimistic_conflict_replan"): (
        "preferred_a_conflict", "preferred_a_control",
        "preferred_b_conflict", "preferred_b_control",
    ),
    ("incident_recovery", "canary_compensation_saga"): (
        "compatible_patch", "delayed_regression",
    ),
    ("permit_transaction", "asynchronous_authority_timeout"): (
        "grant_before_deadline", "explicit_deny", "authority_timeout",
    ),
}


@dataclass(frozen=True)
class TrustedWorld:
    """Offline catalog row; ``role`` is never supplied to the policy/planner."""

    family: str
    variant_id: str
    role: str
    scenario: StatefulScenario


def trusted_world_catalog(world_seed: int) -> tuple[TrustedWorld, ...]:
    """Build 25 worlds: 12 singleton plus 4/4/2/3 hidden-world cases."""

    if world_seed < 0:
        raise ValueError("world seed must be non-negative")
    rows: list[TrustedWorld] = []
    for family in FAMILIES:
        for variant_id in variants_for_family(family):
            branches = PRIVATE_BRANCHES.get((family, variant_id), ("singleton",))
            for branch in branches:
                scenario = build_scenario(
                    family, world_seed=world_seed, variant_id=variant_id,
                    **({} if branch == "singleton" else {
                        "dynamics_branch": branch,
                    }),
                )
                rows.append(TrustedWorld(family, variant_id, branch, scenario))
    if len(rows) != 25:
        raise RuntimeError("the 16-group trusted catalog must contain 25 worlds")
    return tuple(rows)


def _information_sets(
    worlds: Iterable[TrustedWorld],
) -> tuple[tuple[str, tuple[TrustedWorld, ...]], ...]:
    """Derive information sets from exact initial public-observation hashes."""

    groups: dict[tuple[str, str, str], list[TrustedWorld]] = {}
    for world in worlds:
        env = StatefulPOMDPEnv(world.scenario)
        observation = env.reset()[0]
        groups.setdefault(
            (world.family, world.variant_id, _digest(observation)), [],
        ).append(world)
    result: list[tuple[str, tuple[TrustedWorld, ...]]] = []
    for (family, variant_id, observation_sha256), members in sorted(groups.items()):
        result.append((
            f"{family}/{variant_id}/{observation_sha256[:12]}", tuple(members),
        ))
    if len(result) != 17:
        raise RuntimeError("the trusted catalog must form 17 public information sets")
    return tuple(result)


def _planner(worlds: tuple[TrustedWorld, ...]) -> PublicBeliefPlanner:
    max_depth = max(row.scenario.public_task.budget.max_steps for row in worlds)
    return PublicBeliefPlanner(
        (row.scenario for row in worlds), max_depth=max_depth,
        grounder=grounded_public_actions,
    )


def _metrics(
    worlds: tuple[TrustedWorld, ...], tree: Any,
) -> list[dict[str, Any]]:
    if tree is None:
        return [
            {"world_role": row.role, "success": False} for row in worlds
        ]
    return [
        {
            "world_role": row.role,
            **evaluate_policy_tree(row.scenario, tree),
        }
        for row in worlds
    ]


def _initial_public_alias(worlds: tuple[TrustedWorld, ...]) -> bool:
    observations = []
    for world in worlds:
        env = StatefulPOMDPEnv(world.scenario)
        observations.append(env.reset()[0])
    return len({_digest(observation) for observation in observations}) == 1


def audit_seed(world_seed: int) -> dict[str, Any]:
    worlds = trusted_world_catalog(world_seed)
    conditional_rows = []
    for world in worlds:
        planner = _planner((world,))
        tree = planner.solve()
        metrics = _metrics((world,), tree)[0]
        conditional_rows.append({
            "family": world.family,
            "variant_id": world.variant_id,
            **metrics,
        })

    robust_sets = []
    for set_id, members in _information_sets(worlds):
        planner = _planner(members)
        tree = planner.solve()
        rows = _metrics(members, tree)
        robust_sets.append({
            "information_set_id": set_id,
            "family": members[0].family,
            "variant_id": members[0].variant_id,
            "world_count": len(members),
            "initial_public_alias": _initial_public_alias(members),
            "robust_success": bool(tree is not None and all(
                row["success"] for row in rows
            )),
            "nodes_expanded": planner.nodes_expanded,
            "actions_evaluated": planner.actions_evaluated,
            "tree_stats": _tree_stats(tree) if tree is not None else None,
            "policy_tree_sha256": (
                _digest(tree.to_dict()) if tree is not None else None
            ),
            "rows": rows,
        })

    group_rows = []
    for family in FAMILIES:
        for variant_id in variants_for_family(family):
            sets = [
                row for row in robust_sets
                if row["family"] == family and row["variant_id"] == variant_id
            ]
            group_rows.append({
                "family": family, "variant_id": variant_id,
                "information_set_count": len(sets),
                "robust_group_success": bool(
                    sets and all(row["robust_success"] for row in sets)
                ),
            })

    successful_rows = [
        metric for item in robust_sets for metric in item["rows"]
        if metric["success"]
    ]
    multiworld_sets = [row for row in robust_sets if row["world_count"] > 1]
    singleton_sets = [row for row in robust_sets if row["world_count"] == 1]
    gates = {
        "catalog_has_16_structural_groups": len(group_rows) == 16,
        "catalog_has_25_worlds": len(worlds) == 25,
        "catalog_has_17_information_sets": len(robust_sets) == 17,
        "conditional_reachability_25_of_25": all(
            row["success"] for row in conditional_rows
        ),
        "initial_public_alias_multiworld_5_of_5": bool(
            len(multiworld_sets) == 5
            and all(row["initial_public_alias"] for row in multiworld_sets)
        ),
        "robust_group_success_16_of_16": all(
            row["robust_group_success"] for row in group_rows
        ),
        "zero_attempted_policy_violations": all(
            row["attempted_policy_violations"] == 0 for row in successful_rows
        ),
        "zero_executed_safety_violations": all(
            row["safety_violations"] == 0 for row in successful_rows
        ),
        "zero_budget_violations": all(
            row["budget_violations"] == 0 for row in successful_rows
        ),
    }
    return {
        "world_seed": world_seed,
        "scenario_set_sha256": _digest(sorted(
            row.scenario.private_instance_id for row in worlds
        )),
        "counts": {
            "structural_groups": len(group_rows), "worlds": len(worlds),
            "information_sets": len(robust_sets),
            "multiworld_information_sets": len(multiworld_sets),
            "singleton_information_sets": len(singleton_sets),
        },
        "conditional_reachability": {
            "role": (
                "per-world conditional reachability upper bound through the public "
                "facade; not a deployable policy and not full-private-state search"
            ),
            "successes": sum(row["success"] for row in conditional_rows),
            "rows": conditional_rows,
        },
        "public_history_robust_synthesis": {
            "role": (
                "finite max-min contingent synthesis over each compatible initial "
                "public information set"
            ),
            "successful_groups": sum(
                row["robust_group_success"] for row in group_rows
            ),
            "groups": group_rows,
            "information_sets": robust_sets,
        },
        "gates": gates,
    }


def run_reachability_audit(
    *, world_seeds: Iterable[int] = DEFAULT_AUDIT_SEEDS,
) -> dict[str, Any]:
    seeds = tuple(world_seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("world seeds must be a non-empty unique sequence")
    if any(seed < 0 for seed in seeds):
        raise ValueError("world seeds must be non-negative")
    rows = [audit_seed(seed) for seed in seeds]
    passed = all(all(row["gates"].values()) for row in rows)
    aggregate = {
        "group_seed_instances": sum(
            row["counts"]["structural_groups"] for row in rows
        ),
        "worlds": sum(row["counts"]["worlds"] for row in rows),
        "information_sets": sum(
            row["counts"]["information_sets"] for row in rows
        ),
        "multiworld_information_sets": sum(
            row["counts"]["multiworld_information_sets"] for row in rows
        ),
        "singleton_information_sets": sum(
            row["counts"]["singleton_information_sets"] for row in rows
        ),
    }
    return {
        "schema_version": REPORT_VERSION,
        "stage": "train",
        "source_state": _source_state(),
        "grounding": {
            "version": GROUNDING_VERSION,
            "contract_fingerprint": grounding_fingerprint(),
            "implementation_sha256": grounding_implementation_sha256(),
            "coverage": (
                "all public visible entity IDs, declared public enum values, current "
                "dynamic references, and one unavailable-reference class"
            ),
            "ordering": (
                "public-state heuristic first, deterministic remainder second; search "
                "counts are implementation traces, not task difficulty"
            ),
            "known_gaps": (
                "wrong/stale/cross-object/consumed reference classes, idempotency-key "
                "classes, and arbitrary JSON strings are not exhaustively enumerated"
            ),
        },
        "audit_seeds": list(seeds),
        "aggregate_counts": aggregate,
        "seed_count_and_gate_stability": {
            "all_seed_gates_pass": passed,
            "structural_group_count_stable": len({
                row["counts"]["structural_groups"] for row in rows
            }) == 1,
            "world_count_stable": len({
                row["counts"]["worlds"] for row in rows
            }) == 1,
            "information_set_count_stable": len({
                row["counts"]["information_sets"] for row in rows
            }) == 1,
        },
        "rows": rows,
        "remaining_gates": {
            "all_declared_legal_paths_reachable": False,
            "bounded_adversarial_acceptance_search": False,
            "invalid_argument_equivalence_classes_complete": False,
            "idempotency_equivalence_classes_complete": False,
            "reward_leak_audit_complete": False,
            "preterminal_information_leak_audit_complete": False,
            "collection_ready": False,
            "rl_ready": False,
        },
        "claim_boundary": (
            "train-only bounded positive reachability and robust contingent synthesis "
            "for 16 structural groups across three role-changing seeds; not exhaustive "
            "action-space verification, formal safety proof, model evaluation, held-out "
            "generalization, learned policy, or Agentic RL evidence"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit 16 train-only stateful groups with public-history synthesis",
    )
    parser.add_argument(
        "--world-seeds", type=int, nargs="+", default=list(DEFAULT_AUDIT_SEEDS),
    )
    args = parser.parse_args()
    print(json.dumps(
        run_reachability_audit(world_seeds=args.world_seeds),
        ensure_ascii=False, indent=2, sort_keys=True,
    ))


if __name__ == "__main__":
    main()
