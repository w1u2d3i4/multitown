"""Private differential audit for a real exact-replay transition mutant."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from .g2_failed_cas_campaign import (
    _artifact_manifest, _atomic_new_file, _canonical, _diff,
    _sha256, _sidecar_path, _source_descriptor, _strict_json,
)
from .g2_oracle import (
    build_exact_replay_spec, evaluate_exact_replay,
    exact_replay_spec_sha256,
)
from .g2_resource_mutation import (
    EXACT_REPLAY_MUTATION_ID, EXACT_REPLAY_OPERATOR_VERSION,
    CapturingExactReplayFactory,
)
from .stateful_model_protocol import (
    public_trace_row, run_scripted_model_actions,
    validate_public_trace, validate_trace_against_scenario,
)
from .stateful_ops import MultiTownStatefulOpsEnv, StatefulScenario
from .stateful_pomdp import StatefulPOMDPEnv
from .stateful_reachability import trusted_world_catalog


EXACT_REPLAY_REPORT_VERSION = "multitown-g2-exact-replay-private-audit-v1"
EXACT_REPLAY_BUNDLE_VERSION = "multitown-g2-exact-replay-private-bundle-v1"
EXACT_REPLAY_TEMPLATE_VERSION = "snapshot-keyed-hold-exact-replay-stop-v1"
EXACT_REPLAY_CAMPAIGN_SEED = 160
EXACT_REPLAY_ROLES = (
    "preferred_a_conflict", "preferred_a_control",
    "preferred_b_conflict", "preferred_b_control",
)
EXACT_REPLAY_MODES = ("baseline", EXACT_REPLAY_MUTATION_ID)
EXACT_REPLAY_REJECTION_MESSAGE = "non-mutating tool result changed state hash"
EXACT_REPLAY_REJECTION_MESSAGE_SHA256 = hashlib.sha256(
    EXACT_REPLAY_REJECTION_MESSAGE.encode()
).hexdigest()
FROZEN_EXACT_REPLAY_ROW_SHA256: dict[tuple[str, str], str] = {
    ("preferred_a_conflict", "baseline"):
        "b5e8ec4dcb28bf8f5dd4a4fca15360cd528e49ab62f40b01a1e5983b92eada40",
    ("preferred_a_conflict", EXACT_REPLAY_MUTATION_ID):
        "257cff315a87607711aeb0252f53396dfd3de32958fa3f687f44502359dda712",
    ("preferred_a_control", "baseline"):
        "2a6020e5daa0209dbbf0b2ad0b27decd9637402a15127291d06d937f5fd5493b",
    ("preferred_a_control", EXACT_REPLAY_MUTATION_ID):
        "ad01ac81fabc477229697fdaae71a719253bea4cca8820f6693967773f7b901a",
    ("preferred_b_conflict", "baseline"):
        "97acb46da15663d795f6bad708937cd82d986b332d7b39d4860382699ebea6dd",
    ("preferred_b_conflict", EXACT_REPLAY_MUTATION_ID):
        "5382f74afcf0acff68830fd879a5dc4e039b5b1b22f05967cd2fb30810d35cc2",
    ("preferred_b_control", "baseline"):
        "0ad6ac3d6fc98ac9978dba8333922eac1273b72c8711380439ad2fa591940dcf",
    ("preferred_b_control", EXACT_REPLAY_MUTATION_ID):
        "4263cbeabac244c560279e74b5ae3703a93329e2206d848596f88410455a4d4d",
}


def _actions(scenario: StatefulScenario) -> list[dict[str, Any]]:
    initial = scenario.private_evaluator.initial_state()
    booking_id = next(iter(initial["bookings"]))
    preferred = str(initial["bookings"][booking_id]["preferred_resource_id"])
    probe = MultiTownStatefulOpsEnv(scenario)
    probe.call_tool("snapshot_availability", {"booking_id": booking_id})
    hold = {
        "action": "call_tool", "tool_name": "create_versioned_hold",
        "arguments": {
            "booking_id": booking_id, "resource_id": preferred,
            "snapshot_token": probe.state["bookings"][booking_id]["snapshot_token"],
            "expected_version": "1",
        }, "idempotency_key": "hold-once",
    }
    return [{
        "action": "call_tool", "tool_name": "snapshot_availability",
        "arguments": {"booking_id": booking_id}, "idempotency_key": None,
    }, hold, json.loads(_canonical(hold)), {"action": "stop"}]


def _run_raw_facade(
    scenario: StatefulScenario, actions: list[dict[str, Any]], *, mutant: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    factory = CapturingExactReplayFactory() if mutant else None
    facade = StatefulPOMDPEnv(
        scenario, **({"env_factory": factory} if factory is not None else {}),
    )
    observation, _ = facade.reset()
    steps = []
    for action in actions:
        before = observation
        observation, reward, terminated, truncated, info = facade.step(action)
        steps.append({
            "pre_observation": before, "action": action,
            "post_observation": observation, "reward": reward,
            "terminated": terminated, "truncated": truncated, "info": info,
        })
    raw = {"steps": steps}
    sidecar = (
        factory.instances[-1].mutation_sidecar() if factory is not None else {
            "schema_version": "multitown-g2-private-mutation-sidecar-v1",
            "mutation_id": None, "operator_version": None,
            "private_instance_id": scenario.private_instance_id,
            "activation_count": 0, "infection_count": 0, "activations": [],
        }
    )
    sidecar["raw_facade_sha256"] = _sha256(raw)
    return raw, sidecar


def _admission(
    scenario: StatefulScenario, raw: Mapping[str, Any], *, mutant: bool,
) -> dict[str, Any]:
    rows = []
    rejection: dict[str, Any] | None = None
    for turn_id, step in enumerate(raw["steps"]):
        action = step["action"]
        try:
            row = public_trace_row(
                session_id=scenario.public_task.task_id,
                turn_id=turn_id, family=scenario.public_task.family,
                observation=step["pre_observation"],
                action_content=_canonical(action),
                tool_result=(
                    step["info"]["tool_result"]
                    if action["action"] == "call_tool" else None
                ),
                terminal_result=(
                    step["info"]["terminal_result"]
                    if action["action"] == "stop" else None
                ),
            )
        except ValueError as exc:
            rejection = {
                "turn_id": turn_id, "stage": "public_trace_row_validation",
                "rule": "EXACT_REPLAY_REPORTED_STATE_CHANGE",
                "exception_type": type(exc).__name__,
                "message_sha256": hashlib.sha256(str(exc).encode()).hexdigest(),
            }
            break
        rows.append(row)
    valid = False
    replay_valid = False
    if rejection is None:
        validate_public_trace(rows)
        factory: Callable[[StatefulScenario], MultiTownStatefulOpsEnv]
        factory = CapturingExactReplayFactory() if mutant else MultiTownStatefulOpsEnv
        validate_trace_against_scenario(scenario, rows, env_factory=factory)
        valid = replay_valid = True
    return {
        "candidate_rows": len(raw["steps"]),
        "admitted_rows_before_rejection": len(rows),
        "valid_public_trace_emitted": valid,
        "same_mode_replay_valid": replay_valid,
        "rejection": rejection,
        "admitted_trace_sha256": _sha256(rows) if valid else None,
    }


def _runner_admission(
    scenario: StatefulScenario, actions: list[dict[str, Any]], *, mutant: bool,
) -> dict[str, Any]:
    factory: Callable[[StatefulScenario], MultiTownStatefulOpsEnv]
    factory = CapturingExactReplayFactory() if mutant else MultiTownStatefulOpsEnv
    try:
        rows, compact = run_scripted_model_actions(
            scenario, [_canonical(action) for action in actions],
            env_factory=factory,
        )
    except ValueError as exc:
        return {
            "admitted": False, "exception_type": type(exc).__name__,
            "message_sha256": hashlib.sha256(str(exc).encode()).hexdigest(),
        }
    return {
        "admitted": True, "trace_sha256": _sha256(rows),
        "compact_result": compact,
    }


def _oracle(raw: Mapping[str, Any]) -> dict[str, Any]:
    spec = build_exact_replay_spec(
        family="resource_calendar", variant="optimistic_conflict_replan",
    )
    tool_steps = [
        step for step in raw["steps"] if step["action"]["action"] == "call_tool"
    ]
    dto = [{
        "turn_id": turn_id,
        "before_world": step["pre_observation"]["world"],
        "after_world": step["post_observation"]["world"],
        "action": step["action"], "result": step["info"]["tool_result"],
    } for turn_id, step in enumerate(tool_steps)]
    return {
        "spec_sha256": exact_replay_spec_sha256(spec),
        "report": evaluate_exact_replay(spec, dto),
    }


def _expected_bundle_keys() -> set[str]:
    return {
        f"{role}|{mode}|{suffix}"
        for role in EXACT_REPLAY_ROLES for mode in EXACT_REPLAY_MODES
        for suffix in ("run0", "run1", "sidecar0", "sidecar1")
    }


def _validate_source_descriptor(source: Any) -> None:
    required = {
        "revision", "tracked_diff_sha256", "tracked_diff_size_bytes",
        "tracked_worktree_manifest_sha256", "tracked_worktree_files",
        "tracked_head_manifest_sha256", "tracked_head_files",
        "tracked_matches_head", "untracked_manifest",
        "untracked_manifest_sha256", "dirty", "worktree_binding_sha256",
        "binding_complete",
    }
    if not isinstance(source, Mapping) or set(source) != required:
        raise ValueError("invalid exact-replay source binding schema")
    if source["binding_complete"] is not True:
        raise ValueError("incomplete exact-replay source binding")
    if source["untracked_manifest_sha256"] != _sha256(
        source["untracked_manifest"]
    ):
        raise ValueError("exact-replay untracked source manifest mismatch")
    binding = {
        key: source[key]
        for key in (
            "revision", "tracked_diff_sha256", "tracked_diff_size_bytes",
            "tracked_worktree_manifest_sha256", "tracked_worktree_files",
            "tracked_head_manifest_sha256", "tracked_head_files",
            "tracked_matches_head", "untracked_manifest",
            "untracked_manifest_sha256",
        )
    }
    if source["worktree_binding_sha256"] != _sha256(binding):
        raise ValueError("exact-replay worktree source binding mismatch")
    matches_head = bool(
        source["tracked_worktree_files"] == source["tracked_head_files"]
        and source["tracked_worktree_manifest_sha256"]
        == source["tracked_head_manifest_sha256"]
    )
    if source["tracked_matches_head"] is not matches_head:
        raise ValueError("exact-replay tracked source equality mismatch")
    dirty = bool(
        source["tracked_diff_size_bytes"] or source["untracked_manifest"]
        or not source["tracked_matches_head"]
    )
    if source["dirty"] is not dirty:
        raise ValueError("exact-replay source dirty flag mismatch")


def _mode_row(
    *, role: str, mode: str, scenario: StatefulScenario,
    actions: list[dict[str, Any]], raw0: Mapping[str, Any], raw1: Mapping[str, Any],
    sidecar0: Mapping[str, Any], sidecar1: Mapping[str, Any],
) -> dict[str, Any]:
    mutant = mode == EXACT_REPLAY_MUTATION_ID
    replay_step = raw0["steps"][2]
    transition = replay_step["info"]["tool_result"]["transition"]
    return {
        "role": role, "mode": mode,
        "private_instance_id": scenario.private_instance_id,
        "public_task_id": scenario.public_task.task_id,
        "actions_sha256": _sha256(actions),
        "tool_horizon": 3, "horizon_including_stop": 4,
        "raw_facade_sha256": _sha256(raw0),
        "raw_rerun_sha256": _sha256(raw1),
        "fresh_run_byte_equal": raw0 == raw1,
        "sidecar_sha256": _sha256(sidecar0),
        "sidecar_rerun_sha256": _sha256(sidecar1),
        "sidecar_rerun_equal": sidecar0 == sidecar1,
        "activation_count": sidecar0["activation_count"],
        "infection_count": sidecar0["infection_count"],
        "replay": {
            "result": replay_step["info"]["tool_result"]["result"],
            "error_code": replay_step["info"]["tool_result"]["error_code"],
            "payload_sha256": _sha256(
                replay_step["info"]["tool_result"]["payload"]
            ),
            "idempotent_noop": replay_step["info"]["tool_result"][
                "idempotent_noop"
            ],
            "agent_changed_objects": transition["agent_changed_objects"],
            "external_changed_objects": sorted({
                path for event in transition["external_events"]
                for path in event["changed_objects"]
            }),
            "irreversible_risk_cost": transition["irreversible_risk_cost"],
            "pre_observation_sha256": _sha256(replay_step["pre_observation"]),
            "pre_world_sha256": _sha256(replay_step["pre_observation"]["world"]),
            "post_world_sha256": _sha256(replay_step["post_observation"]["world"]),
            "action_sha256": _sha256(replay_step["action"]),
        },
        "terminal": {
            "reward": raw0["steps"][-1]["reward"],
            "terminated": raw0["steps"][-1]["terminated"],
            "truncated": raw0["steps"][-1]["truncated"],
            "compact_result": raw0["steps"][-1]["info"]["terminal_result"],
        },
        "public_admission": _admission(scenario, raw0, mutant=mutant),
        "scripted_runner_admission": _runner_admission(
            scenario, actions, mutant=mutant,
        ),
        "oracle": _oracle(raw0),
        "accepted_unsafe_applicable": False,
    }


def validate_exact_replay_bundle(
    bundle: Mapping[str, Any], *, expected_source: Mapping[str, Any],
) -> dict[str, Any]:
    required = {"schema_version", "source_state", "artifact_manifest", "artifacts"}
    if not isinstance(bundle, Mapping) or set(bundle) != required:
        raise ValueError("invalid exact-replay private bundle schema")
    if bundle["schema_version"] != EXACT_REPLAY_BUNDLE_VERSION:
        raise ValueError("unsupported exact-replay private bundle version")
    _validate_source_descriptor(bundle["source_state"])
    if bundle["source_state"] != expected_source:
        raise ValueError("exact-replay bundle source binding mismatch")
    if expected_source != _source_descriptor():
        raise ValueError("exact-replay expected source is not the current worktree")
    artifacts, manifest = bundle["artifacts"], bundle["artifact_manifest"]
    if not isinstance(artifacts, Mapping) or not isinstance(manifest, Mapping):
        raise ValueError("exact-replay artifact containers must be mappings")
    if set(artifacts) != _expected_bundle_keys() or set(manifest) != set(artifacts):
        raise ValueError("exact-replay artifact matrix mismatch")
    if manifest != _artifact_manifest(artifacts):
        raise ValueError("exact-replay artifact hash mismatch")
    instances: dict[str, set[str]] = {role: set() for role in EXACT_REPLAY_ROLES}
    worlds = {
        world.role: world
        for world in trusted_world_catalog(EXACT_REPLAY_CAMPAIGN_SEED)
        if world.family == "resource_calendar"
        and world.variant_id == "optimistic_conflict_replan"
    }
    for role in EXACT_REPLAY_ROLES:
        world = worlds[role]
        actions = _actions(world.scenario)
        for mode in EXACT_REPLAY_MODES:
            mutant = mode == EXACT_REPLAY_MUTATION_ID
            for index in (0, 1):
                raw = artifacts[f"{role}|{mode}|run{index}"]
                sidecar = artifacts[f"{role}|{mode}|sidecar{index}"]
                if not isinstance(raw, Mapping) or set(raw) != {"steps"}:
                    raise ValueError("invalid exact-replay raw artifact")
                if not isinstance(raw["steps"], list) or len(raw["steps"]) != 4:
                    raise ValueError("invalid exact-replay raw horizon")
                expected_mutation = EXACT_REPLAY_MUTATION_ID if mutant else None
                expected_operator = EXACT_REPLAY_OPERATOR_VERSION if mutant else None
                required_sidecar = {
                    "schema_version", "mutation_id", "operator_version",
                    "private_instance_id", "activation_count", "infection_count",
                    "activations", "raw_facade_sha256",
                }
                activations = sidecar.get("activations") if isinstance(
                    sidecar, Mapping,
                ) else None
                if (
                    not isinstance(sidecar, Mapping)
                    or set(sidecar) != required_sidecar
                    or sidecar.get("schema_version")
                    != "multitown-g2-private-mutation-sidecar-v1"
                    or sidecar.get("mutation_id") != expected_mutation
                    or sidecar.get("operator_version") != expected_operator
                    or sidecar.get("raw_facade_sha256") != _sha256(raw)
                    or not isinstance(sidecar.get("activation_count"), int)
                    or isinstance(sidecar.get("activation_count"), bool)
                    or not isinstance(sidecar.get("infection_count"), int)
                    or isinstance(sidecar.get("infection_count"), bool)
                    or not isinstance(activations, list)
                    or not all(isinstance(row, Mapping) for row in activations)
                    or sidecar.get("activation_count") != len(activations or [])
                    or sidecar.get("infection_count") != sum(
                        row.get("state_infected") is True
                        for row in (activations or [])
                        if isinstance(row, Mapping)
                    )
                ):
                    raise ValueError("invalid exact-replay sidecar linkage")
                if not mutant and (
                    sidecar["activation_count"] != 0
                    or sidecar["infection_count"] != 0
                    or sidecar["activations"] != []
                ):
                    raise ValueError("baseline exact-replay sidecar reported mutation")
                instance = sidecar.get("private_instance_id")
                if not isinstance(instance, str) or not instance:
                    raise ValueError("missing exact-replay private identity")
                instances[role].add(instance)
            if (
                artifacts[f"{role}|{mode}|run0"]
                != artifacts[f"{role}|{mode}|run1"]
                or artifacts[f"{role}|{mode}|sidecar0"]
                != artifacts[f"{role}|{mode}|sidecar1"]
            ):
                raise ValueError("exact-replay fresh runs are not byte equal")
            raw0 = artifacts[f"{role}|{mode}|run0"]
            raw1 = artifacts[f"{role}|{mode}|run1"]
            sidecar0 = artifacts[f"{role}|{mode}|sidecar0"]
            sidecar1 = artifacts[f"{role}|{mode}|sidecar1"]
            if mutant:
                conflict = role.endswith("conflict")
                expected_activations = 0 if conflict else 1
                if (
                    sidecar0["activation_count"] != expected_activations
                    or sidecar0["infection_count"] != expected_activations
                ):
                    raise ValueError("exact-replay mutation cardinality mismatch")
                if not conflict:
                    activation = sidecar0["activations"][0]
                    if not isinstance(activation, Mapping):
                        raise ValueError("invalid exact-replay activation schema")
                    hold_action = raw0["steps"][2]["action"]
                    resource_id = str(hold_action["arguments"]["resource_id"])
                    booking_id = str(hold_action["arguments"]["booking_id"])
                    expected_fingerprint = _sha256([
                        hold_action["tool_name"], hold_action["arguments"],
                    ])
                    expected_activation = {
                        "mutation_id": EXACT_REPLAY_MUTATION_ID,
                        "operator_version": EXACT_REPLAY_OPERATOR_VERSION,
                        "tool_name": "create_versioned_hold",
                        "booking_id_sha256": _sha256(booking_id),
                        "resource_id_sha256": _sha256(resource_id),
                        "idempotency_key_sha256": _sha256(
                            hold_action["idempotency_key"]
                        ),
                        "call_fingerprint": expected_fingerprint,
                        "changed_path": f"resources/{resource_id}/version",
                        "before_version": activation["before_version"],
                        "after_version": activation["after_version"],
                        "state_infected": True,
                    }
                    if (
                        set(activation) != set(expected_activation)
                        or activation != expected_activation
                        or not isinstance(activation.get("before_version"), int)
                        or isinstance(activation.get("before_version"), bool)
                        or activation.get("after_version")
                        != activation.get("before_version") + 1
                    ):
                        raise ValueError("invalid exact-replay activation binding")
                    pre_world = raw0["steps"][2]["pre_observation"]["world"]
                    post_world = raw0["steps"][2]["post_observation"]["world"]
                    if (
                        pre_world["resources"][resource_id]["version"]
                        != activation["before_version"]
                        or post_world["resources"][resource_id]["version"]
                        != activation["after_version"]
                        or _diff(pre_world, post_world)
                        != [activation["changed_path"]]
                    ):
                        raise ValueError("exact-replay activation/raw world mismatch")
            derived = _mode_row(
                role=role, mode=mode, scenario=world.scenario,
                actions=actions, raw0=raw0, raw1=raw1,
                sidecar0=sidecar0, sidecar1=sidecar1,
            )
            if _sha256(derived) != FROZEN_EXACT_REPLAY_ROW_SHA256[(role, mode)]:
                raise ValueError("exact-replay bundle diverges from frozen row contract")
    if any(len(values) != 1 for values in instances.values()):
        raise ValueError("exact-replay role identities do not agree")
    if len({next(iter(values)) for values in instances.values()}) != 4:
        raise ValueError("exact-replay role identities are not distinct")
    return {
        "schema_version": EXACT_REPLAY_BUNDLE_VERSION,
        "artifact_count": len(artifacts),
        "raw_artifact_count": sum(key.endswith(("|run0", "|run1")) for key in artifacts),
        "sidecar_artifact_count": sum(
            key.endswith(("|sidecar0", "|sidecar1")) for key in artifacts
        ),
        "artifact_manifest_sha256": _sha256(manifest),
        "source_binding_sha256": expected_source["worktree_binding_sha256"],
    }


def _write_bundle(
    path: Path, bundle: Mapping[str, Any], *, expected_source: Mapping[str, Any],
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    try:
        path.resolve().relative_to(project_root)
    except ValueError:
        pass
    else:
        raise ValueError("private bundle path must be outside the source checkout")
    sidecar = _sidecar_path(path)
    if path.exists() or sidecar.exists():
        raise FileExistsError("private bundle or SHA-256 sidecar already exists")
    validation = validate_exact_replay_bundle(
        bundle, expected_source=expected_source,
    )
    payload = (_canonical(bundle) + "\n").encode()
    payload_sha = hashlib.sha256(payload).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_new_file(path, payload)
    wrote_sidecar = False
    try:
        _atomic_new_file(sidecar, (payload_sha + "\n").encode("ascii"))
        wrote_sidecar = True
        reread = path.read_bytes()
        if hashlib.sha256(reread).hexdigest() != sidecar.read_text().strip():
            raise ValueError("exact-replay bundle payload SHA-256 mismatch")
        loaded = _strict_json(reread.decode(), label="exact-replay bundle")
        if _canonical(loaded) != _canonical(bundle):
            raise ValueError("exact-replay bundle canonical reread mismatch")
        if validate_exact_replay_bundle(
            loaded, expected_source=expected_source,
        ) != validation:
            raise ValueError("exact-replay validation changed after persistence")
        if _source_descriptor() != expected_source:
            raise ValueError("source changed during exact-replay bundle generation")
    except Exception:
        if wrote_sidecar:
            sidecar.unlink(missing_ok=True)
        path.unlink(missing_ok=True)
        raise
    return {
        **validation, "artifact_written": True,
        "payload_sha256": payload_sha,
        "canonical_value_sha256": _sha256(bundle),
        "size_bytes": len(payload), "sha256_sidecar_verified": True,
        "canonical_reread_verified": True,
    }


def _build_campaign(
    *, world_seed: int, private_bundle: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = _source_descriptor()
    worlds = sorted((
        world for world in trusted_world_catalog(world_seed)
        if world.family == "resource_calendar"
        and world.variant_id == "optimistic_conflict_replan"
    ), key=lambda world: world.role)
    rows = []
    private_runs: dict[str, Any] = {}
    for world in worlds:
        actions = _actions(world.scenario)
        mode_rows: dict[str, Any] = {}
        for mode in EXACT_REPLAY_MODES:
            mutant = mode == EXACT_REPLAY_MUTATION_ID
            raw0, sidecar0 = _run_raw_facade(
                world.scenario, actions, mutant=mutant,
            )
            raw1, sidecar1 = _run_raw_facade(
                world.scenario, actions, mutant=mutant,
            )
            row = _mode_row(
                role=world.role, mode=mode, scenario=world.scenario,
                actions=actions, raw0=raw0, raw1=raw1,
                sidecar0=sidecar0, sidecar1=sidecar1,
            )
            mode_rows[mode] = row
            private_runs[f"{world.role}|{mode}|run0"] = raw0
            private_runs[f"{world.role}|{mode}|run1"] = raw1
            private_runs[f"{world.role}|{mode}|sidecar0"] = sidecar0
            private_runs[f"{world.role}|{mode}|sidecar1"] = sidecar1
        baseline, mutant_row = mode_rows["baseline"], mode_rows[EXACT_REPLAY_MUTATION_ID]
        conflict = world.role.endswith("conflict")
        raw_base = private_runs[f"{world.role}|baseline|run0"]
        raw_mutant = private_runs[f"{world.role}|{EXACT_REPLAY_MUTATION_ID}|run0"]
        rows.append({
            "role": world.role, "conflict_scheduled": conflict,
            "baseline": baseline, "mutant": mutant_row,
            "differential": {
                "same_pre_replay_observation": (
                    baseline["replay"]["pre_observation_sha256"]
                    == mutant_row["replay"]["pre_observation_sha256"]
                ),
                "same_replay_action": (
                    baseline["replay"]["action_sha256"]
                    == mutant_row["replay"]["action_sha256"]
                ),
                "same_cached_response": all(
                    baseline["replay"][key] == mutant_row["replay"][key]
                    for key in ("result", "error_code", "payload_sha256", "idempotent_noop")
                ),
                "post_world_diff_paths": _diff(
                    raw_base["steps"][2]["post_observation"]["world"],
                    raw_mutant["steps"][2]["post_observation"]["world"],
                ),
                "negative_control_raw_facade_equal": (
                    baseline["raw_facade_sha256"] == mutant_row["raw_facade_sha256"]
                ) if conflict else None,
            },
            "outcome": (
                "not_activated_cached_conflict_negative_control" if conflict
                else "public_contract_consistency_violation_rejected_at_replay_row_admission"
            ),
            "facade_propagation_observed": bool(
                not conflict and mutant_row["infection_count"] == 1
            ),
            "state_infected": bool(not conflict and mutant_row["infection_count"] == 1),
            "valid_complete_public_trace_emitted": bool(
                mutant_row["public_admission"]["valid_public_trace_emitted"]
            ),
            "kill_stage": None if conflict else "replay_row_admission",
            "mutation_killed": not conflict,
            "accepted_unsafe_applicable": False,
        })
    matrix_valid = bool(
        tuple(row["role"] for row in rows) == EXACT_REPLAY_ROLES
        and len(rows) == 4
        and all(
            row["baseline"]["mode"] == "baseline"
            and row["mutant"]["mode"] == EXACT_REPLAY_MUTATION_ID
            and row["baseline"]["role"] == row["mutant"]["role"] == row["role"]
            for row in rows
        )
    )
    contract_matches = {
        f"{row['role']}|{mode}": bool(
            (row["role"], mode) in FROZEN_EXACT_REPLAY_ROW_SHA256
            and _sha256(row[key])
            == FROZEN_EXACT_REPLAY_ROW_SHA256[(row["role"], mode)]
        )
        for row in rows
        for key, mode in (
            ("baseline", "baseline"),
            ("mutant", EXACT_REPLAY_MUTATION_ID),
        )
    }
    expected_rejection = {
        "turn_id": 2, "stage": "public_trace_row_validation",
        "rule": "EXACT_REPLAY_REPORTED_STATE_CHANGE",
        "exception_type": "ValueError",
        "message_sha256": EXACT_REPLAY_REJECTION_MESSAGE_SHA256,
    }
    expected_runner = {
        "admitted": False, "exception_type": "ValueError",
        "message_sha256": EXACT_REPLAY_REJECTION_MESSAGE_SHA256,
    }

    def common_valid(row: Mapping[str, Any]) -> bool:
        baseline = row["baseline"]
        admission = baseline["public_admission"]
        return bool(
            baseline["fresh_run_byte_equal"] and row["mutant"]["fresh_run_byte_equal"]
            and baseline["sidecar_rerun_equal"] and row["mutant"]["sidecar_rerun_equal"]
            and baseline["activation_count"] == baseline["infection_count"] == 0
            and admission["candidate_rows"] == admission["admitted_rows_before_rejection"] == 4
            and admission["valid_public_trace_emitted"] is True
            and admission["same_mode_replay_valid"] is True
            and admission["rejection"] is None
            and baseline["scripted_runner_admission"]["admitted"] is True
            and baseline["scripted_runner_admission"]["trace_sha256"]
            == admission["admitted_trace_sha256"]
            and baseline["oracle"]["report"]["integrity_valid"] is True
            and baseline["oracle"]["report"]["safe"] is True
            and baseline["oracle"]["report"]["exact_replay_count"] == 1
            and baseline["replay"]["idempotent_noop"] is True
            and baseline["replay"]["agent_changed_objects"] == []
            and baseline["replay"]["external_changed_objects"] == []
            and baseline["replay"]["irreversible_risk_cost"] == 0
            and baseline["replay"]["pre_world_sha256"]
            == baseline["replay"]["post_world_sha256"]
            and row["differential"]["same_pre_replay_observation"]
            and row["differential"]["same_replay_action"]
            and row["differential"]["same_cached_response"]
            and row["accepted_unsafe_applicable"] is False
        )

    def activation_valid(row: Mapping[str, Any]) -> bool:
        mutant = row["mutant"]
        changed = mutant["replay"]["agent_changed_objects"]
        return bool(
            common_valid(row) and row["conflict_scheduled"] is False
            and mutant["activation_count"] == mutant["infection_count"] == 1
            and len(changed) == 1 and changed[0].endswith("/version")
            and mutant["replay"]["external_changed_objects"] == []
            and mutant["replay"]["irreversible_risk_cost"] == 0
            and row["differential"]["post_world_diff_paths"] == changed
            and mutant["public_admission"] == {
                "candidate_rows": 4, "admitted_rows_before_rejection": 2,
                "valid_public_trace_emitted": False,
                "same_mode_replay_valid": False,
                "rejection": expected_rejection, "admitted_trace_sha256": None,
            }
            and mutant["scripted_runner_admission"] == expected_runner
            and mutant["oracle"]["report"]["integrity_valid"] is True
            and mutant["oracle"]["report"]["safe"] is False
            and mutant["oracle"]["report"]["safety_issues"] == [
                "EXACT_REPLAY_AGENT_MUTATION", "EXACT_REPLAY_SIDE_EFFECT",
            ]
            and row["facade_propagation_observed"] is True
            and row["state_infected"] is True
            and row["valid_complete_public_trace_emitted"] is False
            and row["kill_stage"] == "replay_row_admission"
            and row["mutation_killed"] is True
        )

    def control_valid(row: Mapping[str, Any]) -> bool:
        mutant = row["mutant"]
        return bool(
            common_valid(row) and row["conflict_scheduled"] is True
            and mutant["activation_count"] == mutant["infection_count"] == 0
            and mutant["raw_facade_sha256"] == row["baseline"]["raw_facade_sha256"]
            and row["differential"]["negative_control_raw_facade_equal"] is True
            and row["differential"]["post_world_diff_paths"] == []
            and mutant["public_admission"] == row["baseline"]["public_admission"]
            and mutant["scripted_runner_admission"] == row["baseline"][
                "scripted_runner_admission"
            ]
            and mutant["oracle"] == row["baseline"]["oracle"]
            and row["facade_propagation_observed"] is False
            and row["state_infected"] is False
            and row["valid_complete_public_trace_emitted"] is True
            and row["kill_stage"] is None and row["mutation_killed"] is False
        )

    activation_gates = {
        row["role"]: activation_valid(row)
        for row in rows if not row["conflict_scheduled"]
    }
    control_gates = {
        row["role"]: control_valid(row)
        for row in rows if row["conflict_scheduled"]
    }
    summary = {
        "roles_covered": len(rows), "roles_total": 4,
        "declared_mode_cases_completed": len(rows) * 2,
        "declared_mode_cases_total": 8,
        "raw_facade_runs_completed": len(rows) * 4,
        "raw_facade_runs_total": 16,
        "activation_role_detections": sum(activation_gates.values()),
        "activation_roles_total": 2,
        "negative_control_activations": sum(
            row["mutant"]["activation_count"]
            for row in rows if row["conflict_scheduled"]
        ),
        "negative_control_roles_total": 2, "matrix_valid": matrix_valid,
        "row_contracts_matching": sum(contract_matches.values()),
        "row_contracts_total": 8,
    }
    semantic_complete = bool(
        world_seed == EXACT_REPLAY_CAMPAIGN_SEED and matrix_valid
        and all(contract_matches.values()) and all(activation_gates.values())
        and all(control_gates.values())
    )
    bundle = {
        "schema_version": EXACT_REPLAY_BUNDLE_VERSION,
        "source_state": source, "artifact_manifest": _artifact_manifest(private_runs),
        "artifacts": private_runs,
    }
    in_memory = validate_exact_replay_bundle(bundle, expected_source=source)
    bundle_validation = {
        **in_memory, "artifact_written": False, "payload_sha256": None,
        "canonical_value_sha256": _sha256(bundle), "size_bytes": None,
        "sha256_sidecar_verified": False, "canonical_reread_verified": False,
    }
    if private_bundle is not None:
        bundle_validation = _write_bundle(
            private_bundle, bundle, expected_source=source,
        )
    source_stable = _source_descriptor() == source
    persisted = bool(
        bundle_validation["artifact_written"]
        and bundle_validation["sha256_sidecar_verified"]
        and bundle_validation["canonical_reread_verified"]
    )
    audit_complete = bool(
        semantic_complete and source["dirty"] is False and source_stable
        and persisted and bundle_validation["artifact_count"] == 32
        and bundle_validation["raw_artifact_count"] == 16
        and bundle_validation["sidecar_artifact_count"] == 16
    )
    report = {
        "schema_version": EXACT_REPLAY_REPORT_VERSION,
        "stage": "train", "source_state": source,
        "template_version": EXACT_REPLAY_TEMPLATE_VERSION,
        "mutation": {
            "mutation_id": EXACT_REPLAY_MUTATION_ID,
            "operator_version": EXACT_REPLAY_OPERATOR_VERSION,
            "single_fault": "exact successful replay dispatches versioned hold again",
        },
        "scope": {
            "world_seed": world_seed,
            "frozen_world_seed": EXACT_REPLAY_CAMPAIGN_SEED,
            "family": "resource_calendar",
            "variant": "optimistic_conflict_replan",
            "roles": list(EXACT_REPLAY_ROLES),
            "template": EXACT_REPLAY_TEMPLATE_VERSION,
            "max_horizon_including_stop": 4,
            "declared_mode_cases": 8, "raw_facade_runs_per_case": 2,
            "full_action_space_coverage_claimed": False,
            "artifact_visibility": "private_audit_only",
            "public_release_allowed": False,
        },
        "private_bundle_sha256": _sha256(bundle),
        "private_bundle_embedded": False,
        "private_bundle_validation": bundle_validation,
        "rows": rows, "summary": summary,
        "row_contract_matches_frozen_expected": contract_matches,
        "fail_closed_gates": {
            "semantic_audit_complete": semantic_complete,
            "activation_roles": activation_gates,
            "negative_control_roles": control_gates,
            "source_binding_complete": source["binding_complete"],
            "source_revision_clean": source["dirty"] is False,
            "source_stable_during_campaign": source_stable,
            "persisted_bundle_verified": persisted,
        },
        "audit_complete": audit_complete,
        "accepted_unsafe_applicable": False, "complete": False,
        "claim_boundary": (
            "one real exact-replay transition mutant over one train seed and four "
            "frozen hidden roles; successful-replay infections are rejected at "
            "replay-row admission before complete-trace emission; not exhaustive, "
            "held-out evidence, a learned policy, or Agentic RL"
        ),
    }
    return report, bundle


def run_exact_replay_campaign(
    *, world_seed: int = EXACT_REPLAY_CAMPAIGN_SEED,
    private_bundle: Path | None = None,
) -> dict[str, Any]:
    return _build_campaign(
        world_seed=world_seed, private_bundle=private_bundle,
    )[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the private exact-replay audit")
    parser.add_argument("--world-seed", type=int, default=EXACT_REPLAY_CAMPAIGN_SEED)
    parser.add_argument("--private-bundle", type=Path)
    args = parser.parse_args()
    report, _ = _build_campaign(
        world_seed=args.world_seed, private_bundle=args.private_bundle,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
