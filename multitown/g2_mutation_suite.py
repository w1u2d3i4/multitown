"""Read-only aggregation for the frozen train-only G2 mutation evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .stateful_belief_planner import _source_state


SUITE_REPORT_VERSION = "multitown-g2-cross-fault-mutation-suite-v2"
FROZEN_WORLD_SEED = 160
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\n?\Z")


@dataclass(frozen=True)
class FrozenInput:
    key: str
    schema_version: str
    sha256: str
    source_revision: str


FROZEN_INPUTS = {
    "bounded_search": FrozenInput(
        key="bounded_search",
        schema_version="multitown-g2-mutation-audit-v1",
        sha256="a1d31e47c196e06b5ffd0165252433f3c3e51013b2c8e7ef9daec8cf0a88f000",
        source_revision="b42c663785dde3b4aa220c2d5fb11aa6c19c17c9",
    ),
    "relaxed_trigger": FrozenInput(
        key="relaxed_trigger",
        schema_version="multitown-g2-resource-template-report-v1",
        sha256="09144f39833ef81069908efbfce6b1bfa5e92a15e2e9a0b057e33f4032535058",
        source_revision="10923a07f2dd4e96b43906e24fd3b44b82d9e8f1",
    ),
    "failed_cas": FrozenInput(
        key="failed_cas",
        schema_version="multitown-g2-failed-cas-private-audit-v1",
        sha256="111527810ea4966774515bc003baa91bc7f9c48168161bdd73f258718207a75b",
        source_revision="3aaf99d73f167f643735b6ebf8def3402fbcb772",
    ),
    "exact_replay": FrozenInput(
        key="exact_replay",
        schema_version="multitown-g2-exact-replay-private-audit-v1",
        sha256="1a98b4e1818e427023b0dbc08303c6204d714ea5839b17d7c541824936a6934b",
        source_revision="884deb46d12de59a4ccf898ff36899e49c1d8b72",
    ),
}


def _canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_source_state(source: Mapping[str, Any]) -> None:
    if set(source) != {"revision", "dirty"}:
        raise ValueError("invalid suite source-state schema")
    revision = source.get("revision")
    if not (
        isinstance(revision, str) and len(revision) == 40
        and all(character in "0123456789abcdef" for character in revision)
    ):
        raise ValueError("invalid suite source revision")
    if not isinstance(source.get("dirty"), bool):
        raise ValueError("invalid suite dirty flag")


def _strict_json(payload: bytes, *, label: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant in {label}: {value}")

    try:
        return json.loads(
            payload.decode("utf-8"), object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical UTF-8 JSON") from exc


def _load_frozen(path: Path, spec: FrozenInput) -> tuple[dict[str, Any], dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"missing frozen input for {spec.key}: {path}")
    payload = path.read_bytes()
    digest = _sha256_bytes(payload)
    if digest != spec.sha256:
        raise ValueError(
            f"frozen input hash mismatch for {spec.key}: expected {spec.sha256}, got {digest}"
        )
    value = _strict_json(payload, label=spec.key)
    if not isinstance(value, dict):
        raise ValueError(f"{spec.key} report must be a JSON object")
    if value.get("schema_version") != spec.schema_version:
        raise ValueError(f"unexpected schema version for {spec.key}")
    source = value.get("source_state")
    if not isinstance(source, Mapping):
        raise ValueError(f"missing source state for {spec.key}")
    if set(source) not in (
        {"revision", "dirty"},
        {
            "revision", "dirty", "tracked_diff_sha256", "tracked_diff_size_bytes",
            "tracked_head_files", "tracked_head_manifest_sha256",
            "tracked_matches_head", "tracked_worktree_files",
            "tracked_worktree_manifest_sha256", "untracked_manifest",
            "untracked_manifest_sha256", "worktree_binding_sha256",
            "binding_complete",
        },
    ):
        raise ValueError(f"unexpected source-state schema for {spec.key}")
    if source.get("revision") != spec.source_revision or source.get("dirty") is not False:
        raise ValueError(f"untrusted source state for {spec.key}")
    if value.get("stage") != "train" or value.get("complete") is not False:
        raise ValueError(f"{spec.key} must remain incomplete train-only evidence")
    scope = value.get("scope")
    if not isinstance(scope, Mapping):
        raise ValueError(f"missing scope for {spec.key}")
    if scope.get("world_seed") != FROZEN_WORLD_SEED:
        raise ValueError(f"unexpected world seed for {spec.key}")
    return value, {
        "campaign": spec.key,
        "report_schema_version": spec.schema_version,
        "report_sha256": digest,
        "source_revision": spec.source_revision,
        "source_dirty": False,
    }


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    if denominator <= 0 or numerator < 0 or numerator > denominator:
        raise ValueError("invalid rate")
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator,
    }


def _bounded_search(report: Mapping[str, Any]) -> dict[str, Any]:
    rows = report.get("rows")
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError("bounded-search report must contain four rows")
    expected = {
        "oracle_drop_direct_sequence": ("oracle_temporal_branch_drop", 3),
        "oracle_drop_staged_sequence": ("oracle_temporal_branch_drop", 3),
        "oracle_flip_expected_decision": ("oracle_terminal_goal_flip", 3),
        "production_ignore_attempted_violation": (
            "terminal_acceptor_relaxation", 4,
        ),
    }
    seen: set[str] = set()
    stage_counts = {"bounded_search_counterexample": 0}
    cost = {
        "executed_edges": 0, "generated_edges": 0,
        "unique_states_summed_across_mutants": 0,
        "stop_checks": 0, "independent_counterexample_replays": 0,
    }
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("invalid bounded-search row")
        mutation_id = row.get("mutation_id")
        if mutation_id not in expected or mutation_id in seen:
            raise ValueError("unexpected or duplicate bounded-search mutant")
        seen.add(str(mutation_id))
        operator_class, witness = expected[str(mutation_id)]
        if (
            row.get("operator_class") != operator_class
            or row.get("shortest_witness_length") != witness
            or row.get("killed") is not True
            or row.get("caps_hit") is not False
            or row.get("counts_consistent") is not True
            or row.get("counts_match_frozen_expected") is not True
            or row.get("horizon_frontier_exhausted") is not True
        ):
            raise ValueError(f"invalid bounded-search evidence for {mutation_id}")
        matches = row.get("matching_confirmation_count")
        replays = row.get("independently_replayed_matching_count")
        counterexamples = row.get("counterexample_count")
        if not isinstance(matches, int) or matches <= 0 or matches != replays:
            raise ValueError(f"invalid replay confirmation for {mutation_id}")
        if counterexamples != matches:
            raise ValueError(f"counterexample count mismatch for {mutation_id}")
        counts = row.get("counts")
        if not isinstance(counts, Mapping):
            raise ValueError(f"missing search counts for {mutation_id}")
        for output_key, input_key in (
            ("executed_edges", "executed_edges"),
            ("generated_edges", "generated_edges"),
            ("unique_states_summed_across_mutants", "unique_states"),
            ("stop_checks", "stop_checks"),
        ):
            value = counts.get(input_key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"invalid search cost {input_key}")
            cost[output_key] += value
        cost["independent_counterexample_replays"] += replays
        stage_counts["bounded_search_counterexample"] += 1
    if seen != set(expected):
        raise ValueError("bounded-search mutant matrix is incomplete")
    if (
        report.get("mutants_total") != 4
        or report.get("mutants_killed") != 4
        or report.get("operator_classes_total") != 3
        or report.get("operator_classes_killed") != 3
        or report.get("all_killed") is not True
    ):
        raise ValueError("bounded-search summary is inconsistent")
    return {
        "campaign": "bounded_search",
        "fault_surface": "oracle_and_terminal_acceptor",
        "mutation_definitions": _rate(4, 4),
        "operator_classes": _rate(3, 3),
        "activated_role_opportunities": None,
        "declared_negative_controls": None,
        "detection_stage_counts": stage_counts,
        "public_oracle_joint_detection": None,
        "cost": cost,
        "scope_note": (
            "Definition-level bounded-search evidence; it is not pooled into "
            "the activated fault-role opportunity denominator."
        ),
    }


def _relaxed_trigger(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = report.get("summary")
    rows = report.get("rows")
    if not isinstance(summary, Mapping) or not isinstance(rows, list) or len(rows) != 16:
        raise ValueError("invalid relaxed-trigger campaign shape")
    expected_summary = {
        "conflict_role_detections": 2, "conflict_roles_total": 2,
        "control_false_activations": 0, "control_roles_total": 2,
        "executions_completed": 16, "executions_expected": 16,
        "honest_trace_regressions": 0, "integrity_failures": 0,
        "matrix_shape_valid": True, "mode_pairs_complete": 8,
        "mode_pairs_total": 8, "oracle_out_of_scope": 0,
        "privacy_scan_failures": 0, "public_trace_rows": 84,
        "roles_covered": 4, "roles_total": 4,
        "row_contracts_matching": 16, "row_contracts_total": 16,
        "templates_covered": 2, "templates_total": 2,
        "classification_symmetry_a_b": True,
    }
    if dict(summary) != expected_summary:
        raise ValueError("relaxed-trigger frozen summary changed")
    contracts = report.get("row_contract_matches_frozen_expected")
    if not isinstance(contracts, Mapping) or len(contracts) != 16 or not all(
        value is True for value in contracts.values()
    ):
        raise ValueError("relaxed-trigger row contracts are incomplete")
    activated = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("invalid relaxed-trigger row")
        mutation = row.get("mutation")
        oracle = row.get("oracle")
        if not isinstance(mutation, Mapping) or not isinstance(oracle, Mapping):
            raise ValueError("invalid relaxed-trigger mutation/oracle row")
        activation_count = mutation.get("activation_count")
        if activation_count not in {0, 1} or isinstance(activation_count, bool):
            raise ValueError("invalid relaxed-trigger activation count")
        if activation_count:
            activated.append(row)
    if len(activated) != 2:
        raise ValueError("relaxed-trigger activation count changed")
    for row in activated:
        oracle = row["oracle"]
        if (
            row.get("classification") != "accepted_unsafe"
            or row.get("public_trace_valid") is not True
            or row.get("baseline_replay_valid") is not False
            or row.get("mutant_replay_valid") is not True
            or oracle.get("status") != "evaluated"
            or oracle.get("integrity_valid") is not True
            or oracle.get("integrity_issues") != []
            or oracle.get("goal_valid") is not True
            or oracle.get("temporal_valid") is not True
            or oracle.get("safe") is not False
            or oracle.get("issues") != ["EVENT_TRIGGER_MISMATCH"]
            or oracle.get("safety_issues") != ["EVENT_TRIGGER_MISMATCH"]
        ):
            raise ValueError("invalid relaxed-trigger detection evidence")
    return {
        "campaign": "relaxed_trigger",
        "fault_surface": "external_event_trigger_matching",
        "mutation_definitions": _rate(1, 1),
        "operator_classes": _rate(1, 1),
        "activated_role_opportunities": {
            "detected": _rate(2, 2),
            "public_contract": _rate(0, 2),
            "independent_oracle": _rate(2, 2),
            "accepted_unsafe_before_oracle": 2,
        },
        "declared_negative_controls": {
            "mutation_specific_false_detections": _rate(0, 2),
            "mutation_false_activations": _rate(0, 2),
        },
        "detection_stage_counts": {"independent_oracle_post_admission": 2},
        "public_oracle_joint_detection": {
            "agree_detect": 0, "public_only": 0,
            "oracle_only": 2, "agree_miss": 0,
        },
        "cost": {
            "declared_executions": 16,
            "public_trace_rows": 84,
            "max_action_horizon_including_stop": 6,
        },
        "scope_note": (
            "Two fault-by-role activation opportunities; accepted_unsafe denotes "
            "public admission followed by independent-oracle rejection."
        ),
    }


def _transition_campaign(
    report: Mapping[str, Any], *, campaign: str,
) -> dict[str, Any]:
    if campaign not in {"failed_cas", "exact_replay"}:
        raise ValueError("unknown transition campaign")
    summary = report.get("summary")
    rows = report.get("rows")
    if not isinstance(summary, Mapping) or not isinstance(rows, list) or len(rows) != 4:
        raise ValueError(f"invalid {campaign} campaign shape")
    if report.get("audit_complete") is not True:
        raise ValueError(f"{campaign} semantic audit is incomplete")
    contracts = report.get("row_contract_matches_frozen_expected")
    if not isinstance(contracts, Mapping) or len(contracts) != 8 or not all(
        value is True for value in contracts.values()
    ):
        raise ValueError(f"{campaign} row contracts are incomplete")
    if summary.get("raw_facade_runs_completed") != 16 or summary.get(
        "raw_facade_runs_total"
    ) != 16:
        raise ValueError(f"{campaign} raw-run matrix is incomplete")

    activated = [row for row in rows if row.get("state_infected") is True]
    controls = [row for row in rows if row.get("state_infected") is False]
    if len(activated) != 2 or len(controls) != 2:
        raise ValueError(f"{campaign} activation/control matrix changed")
    expected_stage = (
        "cas_row_admission" if campaign == "failed_cas"
        else "replay_row_admission"
    )
    expected_issues = (
        ["FAILED_CAS_AGENT_MUTATION", "FAILED_OPERATION_AGENT_MUTATION"]
        if campaign == "failed_cas"
        else ["EXACT_REPLAY_AGENT_MUTATION", "EXACT_REPLAY_SIDE_EFFECT"]
    )
    for row in activated:
        mutant = row.get("mutant")
        if not isinstance(mutant, Mapping):
            raise ValueError(f"missing {campaign} mutant evidence")
        public = mutant.get("public_admission")
        oracle_container = mutant.get("oracle")
        oracle = (
            oracle_container.get("report")
            if isinstance(oracle_container, Mapping) else None
        )
        if (
            row.get("mutation_killed") is not True
            or row.get("kill_stage") != expected_stage
            or row.get("valid_complete_public_trace_emitted") is not False
            or mutant.get("activation_count") != 1
            or mutant.get("infection_count") != 1
            or not isinstance(public, Mapping)
            or public.get("valid_public_trace_emitted") is not False
            or public.get("same_mode_replay_valid") is not False
            or public.get("rejection") is None
            or not isinstance(mutant.get("scripted_runner_admission"), Mapping)
            or mutant["scripted_runner_admission"].get("admitted") is not False
            or not isinstance(oracle, Mapping)
            or oracle.get("status") != "evaluated"
            or oracle.get("integrity_valid") is not True
            or oracle.get("integrity_issues") != []
            or oracle.get("safe") is not False
            or oracle.get("issues") != expected_issues
            or oracle.get("safety_issues") != expected_issues
        ):
            raise ValueError(f"invalid activated {campaign} evidence")
    for row in controls:
        mutant = row.get("mutant")
        public = mutant.get("public_admission") if isinstance(mutant, Mapping) else None
        runner = (
            mutant.get("scripted_runner_admission")
            if isinstance(mutant, Mapping) else None
        )
        oracle_container = mutant.get("oracle") if isinstance(mutant, Mapping) else None
        oracle = (
            oracle_container.get("report")
            if isinstance(oracle_container, Mapping) else None
        )
        if (
            row.get("mutation_killed") is not False
            or row.get("kill_stage") is not None
            or row.get("valid_complete_public_trace_emitted") is not True
            or not isinstance(mutant, Mapping)
            or mutant.get("activation_count") != 0
            or mutant.get("infection_count") != 0
            or not isinstance(public, Mapping)
            or public.get("valid_public_trace_emitted") is not True
            or public.get("same_mode_replay_valid") is not True
            or public.get("rejection") is not None
            or not isinstance(runner, Mapping)
            or runner.get("admitted") is not True
            or not isinstance(oracle, Mapping)
            or oracle.get("status") != "evaluated"
            or oracle.get("integrity_valid") is not True
            or oracle.get("integrity_issues") != []
            or oracle.get("safe") is not True
            or oracle.get("issues") != []
            or oracle.get("safety_issues") != []
        ):
            raise ValueError(f"false detection in {campaign} control")
    detection_key = (
        "conflict_role_detections" if campaign == "failed_cas"
        else "activation_role_detections"
    )
    total_key = (
        "conflict_roles_total" if campaign == "failed_cas"
        else "activation_roles_total"
    )
    control_key = (
        "control_activations" if campaign == "failed_cas"
        else "negative_control_activations"
    )
    control_total_key = (
        "control_roles_total" if campaign == "failed_cas"
        else "negative_control_roles_total"
    )
    if (
        summary.get(detection_key) != 2 or summary.get(total_key) != 2
        or summary.get(control_key) != 0
        or summary.get(control_total_key) != 2
        or summary.get("matrix_valid") is not True
        or summary.get("row_contracts_matching") != 8
        or summary.get("row_contracts_total") != 8
    ):
        raise ValueError(f"{campaign} frozen summary changed")
    fault = (
        "failed_compare_and_swap_atomicity"
        if campaign == "failed_cas" else "successful_idempotency_replay"
    )
    return {
        "campaign": campaign,
        "fault_surface": fault,
        "mutation_definitions": _rate(1, 1),
        "operator_classes": _rate(1, 1),
        "activated_role_opportunities": {
            "detected": _rate(2, 2),
            "public_contract": _rate(2, 2),
            "independent_oracle": _rate(2, 2),
            "accepted_unsafe_before_oracle": 0,
        },
        "declared_negative_controls": {
            "mutation_specific_false_detections": _rate(0, 2),
            "mutation_false_activations": _rate(0, 2),
        },
        "detection_stage_counts": {"public_row_admission": 2},
        "public_oracle_joint_detection": {
            "agree_detect": 2, "public_only": 0,
            "oracle_only": 0, "agree_miss": 0,
        },
        "cost": {
            "declared_mode_cases": 8,
            "raw_facade_runs": 16,
            "max_action_horizon_including_stop": report["scope"][
                "max_horizon_including_stop"
            ],
        },
        "scope_note": (
            "The public row validator and independent DTO oracle both detect "
            "each activated role before a complete trace is admitted."
        ),
    }


def _sum_rate(campaigns: list[Mapping[str, Any]], key: str) -> dict[str, Any]:
    numerator = sum(int(row[key]["numerator"]) for row in campaigns)
    denominator = sum(int(row[key]["denominator"]) for row in campaigns)
    return _rate(numerator, denominator)


def _joint_detection_proportion(
    joint: Mapping[str, int], *, total: int,
) -> dict[str, Any]:
    """Count only cases detected by both layers, never joint misses."""

    expected = {"agree_detect", "public_only", "oracle_only", "agree_miss"}
    if set(joint) != expected or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in joint.values()
    ) or sum(joint.values()) != total:
        raise ValueError("invalid public/oracle joint-detection table")
    return _rate(joint["agree_detect"], total)


def _assert_sanitized(report: Mapping[str, Any]) -> None:
    encoded = _canonical(report)
    forbidden = (
        "preferred_a_", "preferred_b_", "booking-0160",
        "resource-a-0160", "resource-b-0160", "private_instance_id",
        "public_task_id",
    )
    leaks = [marker for marker in forbidden if marker in encoded]
    if leaks:
        raise ValueError(f"aggregate report leaked identifiers: {leaks}")


def _build_aggregate_report(
    reports: Mapping[str, Mapping[str, Any]],
    input_metadata: Mapping[str, Mapping[str, Any]],
    *, source_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Build an unverified aggregate for tests; path verification upgrades it."""

    if set(reports) != set(FROZEN_INPUTS) or set(input_metadata) != set(FROZEN_INPUTS):
        raise ValueError("all four frozen campaigns are required exactly once")
    _validate_source_state(source_state)
    for key, spec in FROZEN_INPUTS.items():
        expected_metadata = {
            "campaign": key,
            "report_schema_version": spec.schema_version,
            "report_sha256": spec.sha256,
            "source_revision": spec.source_revision,
            "source_dirty": False,
        }
        if dict(input_metadata[key]) != expected_metadata:
            raise ValueError(f"untrusted input metadata for {key}")
    campaigns = [
        _bounded_search(reports["bounded_search"]),
        _relaxed_trigger(reports["relaxed_trigger"]),
        _transition_campaign(reports["failed_cas"], campaign="failed_cas"),
        _transition_campaign(reports["exact_replay"], campaign="exact_replay"),
    ]
    transition = campaigns[1:]
    activated_detected = sum(
        row["activated_role_opportunities"]["detected"]["numerator"]
        for row in transition
    )
    activated_total = sum(
        row["activated_role_opportunities"]["detected"]["denominator"]
        for row in transition
    )
    public_detected = sum(
        row["activated_role_opportunities"]["public_contract"]["numerator"]
        for row in transition
    )
    oracle_detected = sum(
        row["activated_role_opportunities"]["independent_oracle"]["numerator"]
        for row in transition
    )
    false_detections = sum(
        row["declared_negative_controls"][
            "mutation_specific_false_detections"
        ]["numerator"] for row in transition
    )
    negative_total = sum(
        row["declared_negative_controls"][
            "mutation_specific_false_detections"
        ]["denominator"] for row in transition
    )
    joint = {
        key: sum(row["public_oracle_joint_detection"][key] for row in transition)
        for key in ("agree_detect", "public_only", "oracle_only", "agree_miss")
    }
    detection_stages: dict[str, int] = {}
    for campaign in campaigns:
        for stage, count in campaign["detection_stage_counts"].items():
            detection_stages[stage] = detection_stages.get(stage, 0) + count
    accepted_unsafe = sum(
        row["activated_role_opportunities"]["accepted_unsafe_before_oracle"]
        for row in transition
    )
    report = {
        "schema_version": SUITE_REPORT_VERSION,
        "stage": "train",
        "held_out_accessed": False,
        "complete": False,
        "audit_complete": False,
        "generation_verification": {
            "frozen_input_bytes_hash_verified": False,
            "frozen_input_metadata_derived_by_loader": False,
            "current_source_read_from_worktree": False,
            "source_clean": source_state.get("dirty") is False,
        },
        "source_state": dict(source_state),
        "input_evidence": [dict(input_metadata[key]) for key in FROZEN_INPUTS],
        "campaigns": campaigns,
        "aggregate": {
            "declared_mutation_definition_detection_proportion": _sum_rate(
                campaigns, "mutation_definitions",
            ),
            "declared_operator_class_all_instances_detected_proportion": _sum_rate(
                campaigns, "operator_classes",
            ),
            "conditional_detection_proportion_on_declared_activated_and_infected_fault_role_opportunities": _rate(
                activated_detected, activated_total,
            ),
            "observed_mutation_specific_false_detection_proportion_on_declared_non_activation_fault_role_controls": _rate(
                false_detections, negative_total,
            ),
            "public_contract_detection_proportion_on_declared_activated_and_infected_fault_role_opportunities": _rate(
                public_detected, activated_total,
            ),
            "independent_oracle_detection_proportion_on_declared_activated_and_infected_fault_role_opportunities": _rate(
                oracle_detected, activated_total,
            ),
            "public_oracle_joint_detection_proportion_on_declared_activated_and_infected_fault_role_opportunities": _joint_detection_proportion(
                joint, total=activated_total,
            ),
            "public_oracle_joint_detection": joint,
            "publicly_admitted_then_oracle_rejected_fault_role_opportunities": (
                accepted_unsafe
            ),
            "heterogeneous_evidence_unit_stage_counts_do_not_normalize": (
                detection_stages
            ),
        },
        "metric_semantics": {
            "mutation_definition": (
                "One declared mutant in bounded search, or one declared "
                "transition mutation whose complete activation-role matrix passed."
            ),
            "operator_class": (
                "One class declared within a campaign; a class counts only when "
                "all its declared executable instances were detected. Class "
                "granularity is not standardized across campaigns."
            ),
            "activated_fault_role_opportunity": (
                "One fault-by-role opportunity in which a real transition mutant "
                "both activated and infected state. Roles repeat across faults and "
                "are not independent samples; bounded-search rows are excluded."
            ),
            "negative_control_false_detection": (
                "A mutation-specific detector firing in a declared non-activation "
                "role; unrelated policy rejection is not counted as a false positive."
            ),
            "cost": (
                "Raw structural counters only. Wall time and token cost were not "
                "present in every frozen input and are not imputed or pooled."
            ),
            "selection_conditioning": (
                "Included campaigns were already required to pass their declared "
                "train audit. These proportions summarize the curated frozen suite "
                "and are not estimates of population sensitivity or false-positive rate."
            ),
            "heterogeneous_stage_counts": (
                "Four counts use mutation-definition units and six use activated "
                "fault-by-role units; they must not be normalized into one distribution."
            ),
        },
        "evidence_boundary": {
            "world_seeds": [FROZEN_WORLD_SEED],
            "held_out_or_formal_test_claim": False,
            "learned_policy_claim": False,
            "agentic_rl_claim": False,
            "remote_or_deployed_system_claim": False,
            "exhaustive_fault_coverage_claim": False,
            "statistical_generalization_claim": False,
            "release_status": "unverified_test_fixture",
            "note": (
                "Descriptive aggregation of four immutable train-only reports. "
                "Definition and activated-role denominators are intentionally separate."
            ),
        },
        "privacy": {
            "private_input_rows_copied": False,
            "role_or_task_identifiers_included": False,
        },
    }
    _assert_sanitized(report)
    return report


def aggregate_frozen_reports(
    paths: Mapping[str, Path],
) -> dict[str, Any]:
    if set(paths) != set(FROZEN_INPUTS):
        raise ValueError("all four frozen input paths are required")
    resolved = [paths[key].resolve() for key in FROZEN_INPUTS]
    if len(set(resolved)) != len(resolved):
        raise ValueError("frozen input paths must be distinct")
    source_before = _source_state()
    _validate_source_state(source_before)
    reports: dict[str, Mapping[str, Any]] = {}
    metadata: dict[str, Mapping[str, Any]] = {}
    for key, spec in FROZEN_INPUTS.items():
        reports[key], metadata[key] = _load_frozen(paths[key], spec)
    source_after = _source_state()
    if source_after != source_before:
        raise ValueError("suite source changed while frozen inputs were read")
    report = _build_aggregate_report(
        reports, metadata,
        source_state=source_before,
    )
    report["audit_complete"] = source_before["dirty"] is False
    report["generation_verification"] = {
        "frozen_input_bytes_hash_verified": True,
        "frozen_input_metadata_derived_by_loader": True,
        "current_source_read_from_worktree": True,
        "source_clean": source_before["dirty"] is False,
    }
    report["evidence_boundary"]["release_status"] = "private_review_only"
    _assert_sanitized(report)
    return report


def _sidecar_path(path: Path) -> Path:
    return path.with_name(path.name + ".sha256")


def _atomic_new_file(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_verified_report(
    report: Mapping[str, Any], *, expected_source: Mapping[str, Any],
) -> None:
    required = {
        "schema_version", "stage", "held_out_accessed", "complete",
        "audit_complete", "generation_verification", "source_state",
        "input_evidence", "campaigns", "aggregate", "metric_semantics",
        "evidence_boundary", "privacy",
    }
    gates = report.get("generation_verification")
    if (
        set(report) != required
        or report.get("schema_version") != SUITE_REPORT_VERSION
        or report.get("stage") != "train"
        or report.get("held_out_accessed") is not False
        or report.get("complete") is not False
        or report.get("audit_complete") is not True
        or report.get("source_state") != expected_source
        or not isinstance(gates, Mapping)
        or dict(gates) != {
            "frozen_input_bytes_hash_verified": True,
            "frozen_input_metadata_derived_by_loader": True,
            "current_source_read_from_worktree": True,
            "source_clean": True,
        }
    ):
        raise ValueError("suite report is not verified clean evidence")
    _assert_sanitized(report)


def _write_and_validate_verified_report(
    path: Path, report: Mapping[str, Any], *, expected_source: Mapping[str, Any],
) -> dict[str, Any]:
    """Exclusively persist an externally located report and SHA sidecar."""

    path = Path(path)
    project_root = Path(__file__).resolve().parents[1]
    try:
        path.resolve().relative_to(project_root)
    except ValueError:
        pass
    else:
        raise ValueError("suite output path must be outside the source checkout")
    sidecar = _sidecar_path(path)
    if (
        path.exists() or path.is_symlink()
        or sidecar.exists() or sidecar.is_symlink()
    ):
        raise FileExistsError("suite report or SHA-256 sidecar already exists")
    _validate_verified_report(report, expected_source=expected_source)
    payload = (_canonical(report) + "\n").encode("utf-8")
    payload_sha256 = _sha256_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    report_written = False
    sidecar_written = False
    try:
        _atomic_new_file(path, payload)
        report_written = True
        _atomic_new_file(sidecar, (payload_sha256 + "\n").encode("ascii"))
        sidecar_written = True
        reread = path.read_bytes()
        sidecar_value = sidecar.read_text(encoding="ascii")
        if not _SHA256_PATTERN.fullmatch(sidecar_value):
            raise ValueError("invalid suite SHA-256 sidecar")
        if _sha256_bytes(reread) != sidecar_value.strip():
            raise ValueError("suite report payload SHA-256 mismatch")
        loaded = _strict_json(reread, label="persisted mutation suite")
        if not isinstance(loaded, Mapping) or _canonical(loaded) != _canonical(report):
            raise ValueError("suite report canonical reread mismatch")
        _validate_verified_report(loaded, expected_source=expected_source)
        if _source_state() != expected_source:
            raise ValueError("suite source changed during report persistence")
    except Exception:
        if sidecar_written:
            sidecar.unlink(missing_ok=True)
        if report_written:
            path.unlink(missing_ok=True)
        raise
    return {
        "artifact_written": True,
        "payload_sha256": payload_sha256,
        "size_bytes": len(payload),
        "sha256_sidecar_verified": True,
        "canonical_reread_verified": True,
        "source_stable_after_persistence": True,
    }


def generate_and_write_frozen_report(
    paths: Mapping[str, Path], *, output: Path,
) -> dict[str, Any]:
    """Formal path: verify inputs/source, aggregate, then persist fail-closed."""

    report = aggregate_frozen_reports(paths)
    source = report["source_state"]
    persistence = _write_and_validate_verified_report(
        output, report, expected_source=source,
    )
    return {"report": report, "persistence": persistence}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate four immutable train-only G2 mutation reports",
    )
    parser.add_argument("--bounded-search-report", type=Path, required=True)
    parser.add_argument("--relaxed-trigger-report", type=Path, required=True)
    parser.add_argument("--failed-cas-report", type=Path, required=True)
    parser.add_argument("--exact-replay-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "bounded_search": args.bounded_search_report,
        "relaxed_trigger": args.relaxed_trigger_report,
        "failed_cas": args.failed_cas_report,
        "exact_replay": args.exact_replay_report,
    }
    result = generate_and_write_frozen_report(paths, output=args.output)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "sidecar": str(_sidecar_path(args.output).resolve()),
        **result["persistence"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
