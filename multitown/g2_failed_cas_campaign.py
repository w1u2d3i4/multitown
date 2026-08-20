"""Private differential audit for a real failed-CAS transition mutant."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Callable, Mapping

from .g2_oracle import (
    build_resource_conflict_spec,
    evaluate_resource_conflict,
    resource_oracle_spec_sha256,
)
from .g2_resource_campaign import _expected_event_dto
from .g2_resource_mutation import (
    FAILED_CAS_MUTATION_ID,
    FAILED_CAS_OPERATOR_VERSION,
    CapturingFailedCASFactory,
)
from .stateful_model_protocol import (
    public_trace_row, run_scripted_model_actions,
    validate_public_trace, validate_trace_against_scenario,
)
from .stateful_ops import MultiTownStatefulOpsEnv, StatefulScenario
from .stateful_pomdp import StatefulPOMDPEnv
from .stateful_reachability import trusted_world_catalog


FAILED_CAS_REPORT_VERSION = "multitown-g2-failed-cas-private-audit-v1"
FAILED_CAS_TEMPLATE_VERSION = "snapshot-exact-preferred-cas-stop-v1"
FAILED_CAS_CAMPAIGN_SEED = 160
FAILED_CAS_ROLES = (
    "preferred_a_conflict", "preferred_a_control",
    "preferred_b_conflict", "preferred_b_control",
)
FAILED_CAS_MODES = ("baseline", FAILED_CAS_MUTATION_ID)
FAILED_CAS_BUNDLE_VERSION = "multitown-g2-failed-cas-private-bundle-v2"
FAILED_CAS_REJECTION_RULE = "CONFLICT_REPORTED_AGENT_MUTATION"
FAILED_CAS_REJECTION_MESSAGE = "non-mutating tool reported agent state changes"
FAILED_CAS_REJECTION_MESSAGE_SHA256 = hashlib.sha256(
    FAILED_CAS_REJECTION_MESSAGE.encode()
).hexdigest()
FAILED_CAS_ORACLE_ISSUES = (
    "FAILED_CAS_AGENT_MUTATION", "FAILED_OPERATION_AGENT_MUTATION",
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\n?\Z")
FROZEN_FAILED_CAS_ROW_SHA256 = {
    ("preferred_a_conflict", "baseline"):
        "5c326577ba2e356fc10a4ab8db1c418f9dc89928c97d2b13b497cb5c5c140498",
    ("preferred_a_conflict", FAILED_CAS_MUTATION_ID):
        "14b87b51a1a5aabcfbf7c3e63e5af874e27fe0bd8ce81ace38807a6922839c44",
    ("preferred_a_control", "baseline"):
        "4d58945d3bc13311289423b9152a57bf543acead164482ad60a49feaf0b99997",
    ("preferred_a_control", FAILED_CAS_MUTATION_ID):
        "0717b028c2da207339537429bc715272c2027f87ba234cf9e048ba88a31a19d8",
    ("preferred_b_conflict", "baseline"):
        "438ebd177f087e24f31f405b6ac3ebd3b4c9a35858d5b44adb322cf03af2b65e",
    ("preferred_b_conflict", FAILED_CAS_MUTATION_ID):
        "aa1bcaa803ac5a4436c4e56c017160f8a6aca2fdc04e74d06253de41ce2331e7",
    ("preferred_b_control", "baseline"):
        "5b630080f4ec614e8f0069aaaf07fe04fd19361bfd9642a6526f551af86a6271",
    ("preferred_b_control", FAILED_CAS_MUTATION_ID):
        "46e4218372932cc92d1d7d380c8a789274ad9b1614857f980956fb7b8b2c56ab",
}


def _canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strict_json(payload: str, *, label: str) -> Any:
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
            payload, object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON") from exc


def _git_output(project_root: Path, *args: str, binary: bool = False) -> bytes | str:
    result = subprocess.run(
        ["git", *args], cwd=project_root, capture_output=True, check=False,
        **({} if binary else {"text": True}),
    )
    if result.returncode:
        raise RuntimeError(f"cannot read source state: git {' '.join(args)}")
    return result.stdout


def _source_descriptor() -> dict[str, Any]:
    """Bind a report to HEAD plus every tracked and untracked worktree byte."""

    project_root = Path(__file__).resolve().parents[1]
    revision = str(_git_output(project_root, "rev-parse", "HEAD")).strip()
    tracked_diff = bytes(_git_output(
        project_root, "diff", "--binary", "HEAD", "--", binary=True,
    ))
    tracked_raw = bytes(_git_output(
        project_root, "ls-files", "-z", binary=True,
    ))
    tracked_paths = sorted(
        path.decode("utf-8", errors="strict")
        for path in tracked_raw.split(b"\0") if path
    )
    tracked_manifest = []
    head_manifest = []
    for relative in tracked_paths:
        target = project_root / relative
        if target.is_file():
            payload = target.read_bytes()
            tracked_manifest.append({
                "path": relative, "size_bytes": len(payload),
                "content_sha256": _bytes_sha256(payload),
            })
        head_payload = bytes(_git_output(
            project_root, "show", f"HEAD:{relative}", binary=True,
        ))
        head_manifest.append({
            "path": relative, "size_bytes": len(head_payload),
            "content_sha256": _bytes_sha256(head_payload),
        })
    untracked_raw = bytes(_git_output(
        project_root, "ls-files", "--others", "--exclude-standard", "-z",
        binary=True,
    ))
    untracked_paths = sorted(
        path.decode("utf-8", errors="strict")
        for path in untracked_raw.split(b"\0") if path
    )
    untracked_manifest = []
    for relative in untracked_paths:
        payload = (project_root / relative).read_bytes()
        untracked_manifest.append({
            "path": relative, "size_bytes": len(payload),
            "content_sha256": _bytes_sha256(payload),
        })
    binding = {
        "revision": revision,
        "tracked_diff_sha256": _bytes_sha256(tracked_diff),
        "tracked_diff_size_bytes": len(tracked_diff),
        "tracked_worktree_manifest_sha256": _sha256(tracked_manifest),
        "tracked_worktree_files": len(tracked_manifest),
        "tracked_head_manifest_sha256": _sha256(head_manifest),
        "tracked_head_files": len(head_manifest),
        "tracked_matches_head": tracked_manifest == head_manifest,
        "untracked_manifest": untracked_manifest,
        "untracked_manifest_sha256": _sha256(untracked_manifest),
    }
    dirty = bool(
        tracked_diff or untracked_manifest or tracked_manifest != head_manifest
    )
    return {
        **binding,
        "dirty": dirty,
        "worktree_binding_sha256": _sha256(binding),
        "binding_complete": True,
    }


def _artifact_manifest(artifacts: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: {
            "kind": "raw_facade" if key.endswith(("|run0", "|run1")) else "sidecar",
            "value_sha256": _sha256(value),
        }
        for key, value in sorted(artifacts.items())
    }


def validate_private_bundle(
    bundle: Mapping[str, Any], *, expected_source: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate structure/hash links against a trusted source descriptor."""

    required = {"schema_version", "source_state", "artifact_manifest", "artifacts"}
    if not isinstance(bundle, Mapping) or set(bundle) != required:
        raise ValueError("invalid failed-CAS private bundle schema")
    if bundle["schema_version"] != FAILED_CAS_BUNDLE_VERSION:
        raise ValueError("unsupported failed-CAS private bundle version")
    source = bundle["source_state"]
    if not isinstance(source, Mapping) or source.get("binding_complete") is not True:
        raise ValueError("incomplete failed-CAS source binding")
    required_source = {
        "revision", "tracked_diff_sha256", "tracked_diff_size_bytes",
        "tracked_worktree_manifest_sha256", "tracked_worktree_files",
        "tracked_head_manifest_sha256", "tracked_head_files",
        "tracked_matches_head",
        "untracked_manifest", "untracked_manifest_sha256", "dirty",
        "worktree_binding_sha256", "binding_complete",
    }
    if set(source) != required_source:
        raise ValueError("invalid failed-CAS source binding schema")
    if source["untracked_manifest_sha256"] != _sha256(
        source["untracked_manifest"]
    ):
        raise ValueError("failed-CAS untracked source manifest mismatch")
    binding = {
        key: source[key]
        for key in (
            "revision", "tracked_diff_sha256", "tracked_diff_size_bytes",
            "tracked_worktree_manifest_sha256", "tracked_worktree_files",
            "tracked_head_manifest_sha256", "tracked_head_files",
            "tracked_matches_head",
            "untracked_manifest", "untracked_manifest_sha256",
        )
    }
    if source["worktree_binding_sha256"] != _sha256(binding):
        raise ValueError("failed-CAS worktree source binding mismatch")
    if source["tracked_matches_head"] is not bool(
        source["tracked_worktree_files"] == source["tracked_head_files"]
        and source["tracked_worktree_manifest_sha256"]
        == source["tracked_head_manifest_sha256"]
    ):
        raise ValueError("failed-CAS tracked source equality mismatch")
    if source["dirty"] is not bool(
        source["tracked_diff_size_bytes"] or source["untracked_manifest"]
        or not source["tracked_matches_head"]
    ):
        raise ValueError("failed-CAS source dirty flag mismatch")
    if source != expected_source:
        raise ValueError("failed-CAS bundle source binding mismatch")
    artifacts = bundle["artifacts"]
    manifest = bundle["artifact_manifest"]
    if not isinstance(artifacts, Mapping) or not isinstance(manifest, Mapping):
        raise ValueError("failed-CAS artifact containers must be mappings")
    expected_keys = {
        f"{role}|{mode}|{suffix}"
        for role in FAILED_CAS_ROLES for mode in FAILED_CAS_MODES
        for suffix in ("run0", "run1", "sidecar0", "sidecar1")
    }
    if set(artifacts) != expected_keys or set(manifest) != expected_keys:
        raise ValueError("failed-CAS private bundle artifact matrix mismatch")
    if manifest != _artifact_manifest(artifacts):
        raise ValueError("failed-CAS private bundle artifact hash mismatch")
    role_instance_ids: dict[str, set[str]] = {
        role: set() for role in FAILED_CAS_ROLES
    }
    for role in FAILED_CAS_ROLES:
        for mode in FAILED_CAS_MODES:
            mutant = mode == FAILED_CAS_MUTATION_ID
            for run_index in (0, 1):
                raw = artifacts[f"{role}|{mode}|run{run_index}"]
                sidecar = artifacts[f"{role}|{mode}|sidecar{run_index}"]
                if not isinstance(raw, Mapping) or set(raw) != {"steps"}:
                    raise ValueError("invalid raw facade artifact")
                if not isinstance(raw["steps"], list) or len(raw["steps"]) != 3:
                    raise ValueError("invalid raw facade horizon")
                if not isinstance(sidecar, Mapping):
                    raise ValueError("invalid failed-CAS sidecar artifact")
                required_sidecar = {
                    "schema_version", "mutation_id", "operator_version",
                    "private_instance_id", "activation_count", "infection_count",
                    "activations", "raw_facade_sha256",
                }
                if set(sidecar) != required_sidecar:
                    raise ValueError("invalid failed-CAS sidecar schema")
                if sidecar.get("raw_facade_sha256") != _sha256(raw):
                    raise ValueError("failed-CAS raw/sidecar hash mismatch")
                expected_mutation = FAILED_CAS_MUTATION_ID if mutant else None
                expected_operator = FAILED_CAS_OPERATOR_VERSION if mutant else None
                if (
                    sidecar.get("mutation_id") != expected_mutation
                    or sidecar.get("operator_version") != expected_operator
                ):
                    raise ValueError("failed-CAS sidecar identity mismatch")
                private_instance_id = sidecar.get("private_instance_id")
                if not isinstance(private_instance_id, str) or not private_instance_id:
                    raise ValueError("missing failed-CAS private instance identity")
                activations = sidecar.get("activations")
                if not isinstance(activations, list):
                    raise ValueError("invalid failed-CAS activation log")
                if sidecar.get("activation_count") != len(activations):
                    raise ValueError("failed-CAS activation count mismatch")
                if sidecar.get("infection_count") != sum(
                    activation.get("state_infected") is True
                    for activation in activations if isinstance(activation, Mapping)
                ):
                    raise ValueError("failed-CAS infection count mismatch")
                if not mutant and (
                    sidecar["activation_count"] != 0
                    or sidecar["infection_count"] != 0
                ):
                    raise ValueError("baseline sidecar reported a mutation")
                role_instance_ids[role].add(private_instance_id)
            if (
                artifacts[f"{role}|{mode}|run0"]
                != artifacts[f"{role}|{mode}|run1"]
                or artifacts[f"{role}|{mode}|sidecar0"]
                != artifacts[f"{role}|{mode}|sidecar1"]
            ):
                raise ValueError("failed-CAS fresh runs are not byte equal")
    if any(len(instance_ids) != 1 for instance_ids in role_instance_ids.values()):
        raise ValueError("failed-CAS role instance identities do not agree")
    if len({next(iter(ids)) for ids in role_instance_ids.values()}) != 4:
        raise ValueError("failed-CAS role instance identities are not distinct")
    return {
        "schema_version": FAILED_CAS_BUNDLE_VERSION,
        "artifact_count": len(artifacts),
        "raw_artifact_count": sum(key.endswith(("|run0", "|run1")) for key in artifacts),
        "sidecar_artifact_count": sum(
            key.endswith(("|sidecar0", "|sidecar1")) for key in artifacts
        ),
        "artifact_manifest_sha256": _sha256(manifest),
        "source_binding_sha256": source["worktree_binding_sha256"],
    }


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


def write_and_validate_private_bundle(
    path: Path, bundle: Mapping[str, Any], *, expected_source: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically persist, hash, reread and validate a private bundle."""

    path = Path(path)
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
    value_validation = validate_private_bundle(bundle, expected_source=expected_source)
    payload = (_canonical(bundle) + "\n").encode()
    payload_hash = _bytes_sha256(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact_written = False
    sidecar_written = False
    _atomic_new_file(path, payload)
    artifact_written = True
    try:
        _atomic_new_file(sidecar, (payload_hash + "\n").encode("ascii"))
        sidecar_written = True
        reread = path.read_bytes()
        expected_hash = sidecar.read_text(encoding="ascii")
        if not _SHA256_PATTERN.fullmatch(expected_hash):
            raise ValueError("invalid failed-CAS bundle SHA-256 sidecar")
        if _bytes_sha256(reread) != expected_hash.strip():
            raise ValueError("failed-CAS bundle payload SHA-256 mismatch")
        loaded = _strict_json(reread.decode("utf-8"), label="failed-CAS bundle")
        if _canonical(loaded) != _canonical(bundle):
            raise ValueError("failed-CAS bundle canonical reread mismatch")
        reread_validation = validate_private_bundle(
            loaded, expected_source=expected_source,
        )
        if reread_validation != value_validation:
            raise ValueError("failed-CAS bundle validation changed after persistence")
        if _source_descriptor() != expected_source:
            raise ValueError("failed-CAS source changed during bundle generation")
    except Exception:
        if sidecar_written:
            sidecar.unlink(missing_ok=True)
        if artifact_written:
            path.unlink(missing_ok=True)
        raise
    return {
        **value_validation,
        "artifact_written": True,
        "payload_sha256": payload_hash,
        "canonical_value_sha256": _sha256(bundle),
        "size_bytes": len(payload),
        "sha256_sidecar_verified": True,
        "canonical_reread_verified": True,
    }


def _leaves(value: Any, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], Any]:
    if isinstance(value, dict):
        if not value:
            return {prefix: {"type": "dict", "value": []}}
        result: dict[tuple[str, ...], Any] = {}
        for key in sorted(value):
            result.update(_leaves(value[key], (*prefix, str(key))))
        return result
    if isinstance(value, list):
        return {prefix: {"type": "list", "value": json.loads(_canonical(value))}}
    type_name = (
        "null" if value is None else "bool" if isinstance(value, bool)
        else "int" if isinstance(value, int) else "float" if isinstance(value, float)
        else "string" if isinstance(value, str) else type(value).__name__
    )
    return {prefix: {"type": type_name, "value": value}}


def _diff(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[str]:
    before, after = _leaves(left), _leaves(right)
    missing = object()
    return sorted(
        "/".join(path) for path in before.keys() | after.keys()
        if before.get(path, missing) != after.get(path, missing)
    )


def _actions(scenario: StatefulScenario) -> list[dict[str, Any]]:
    initial = scenario.private_evaluator.initial_state()
    booking_id = next(iter(initial["bookings"]))
    preferred = str(initial["bookings"][booking_id]["preferred_resource_id"])
    probe = MultiTownStatefulOpsEnv(scenario)
    probe.call_tool("snapshot_availability", {"booking_id": booking_id})
    token = str(probe.state["bookings"][booking_id]["snapshot_token"])
    return [{
        "action": "call_tool", "tool_name": "snapshot_availability",
        "arguments": {"booking_id": booking_id}, "idempotency_key": None,
    }, {
        "action": "call_tool", "tool_name": "create_versioned_hold",
        "arguments": {
            "booking_id": booking_id, "resource_id": preferred,
            "snapshot_token": token, "expected_version": "1",
        }, "idempotency_key": None,
    }, {"action": "stop"}]


def _run_raw_facade(
    scenario: StatefulScenario, actions: list[dict[str, Any]], *, mutant: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    factory = CapturingFailedCASFactory() if mutant else None
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
            message = str(exc)
            rejection = {
                "turn_id": turn_id,
                "stage": "public_trace_row_validation",
                "rule": FAILED_CAS_REJECTION_RULE,
                "exception_type": type(exc).__name__,
                "message_sha256": hashlib.sha256(message.encode()).hexdigest(),
            }
            break
        rows.append(row)
    full_trace_valid = False
    same_mode_replay_valid = False
    if rejection is None:
        validate_public_trace(rows)
        env_factory: Callable[[StatefulScenario], MultiTownStatefulOpsEnv]
        env_factory = CapturingFailedCASFactory() if mutant else MultiTownStatefulOpsEnv
        validate_trace_against_scenario(
            scenario, rows, env_factory=env_factory,
        )
        full_trace_valid = same_mode_replay_valid = True
    return {
        "candidate_rows": len(raw["steps"]),
        "admitted_rows_before_rejection": len(rows),
        "valid_public_trace_emitted": full_trace_valid,
        "same_mode_replay_valid": same_mode_replay_valid,
        "rejection": rejection,
        "admitted_trace_sha256": _sha256(rows) if full_trace_valid else None,
    }


def _runner_admission(
    scenario: StatefulScenario, actions: list[dict[str, Any]], *, mutant: bool,
) -> dict[str, Any]:
    factory: Callable[[StatefulScenario], MultiTownStatefulOpsEnv]
    factory = CapturingFailedCASFactory() if mutant else MultiTownStatefulOpsEnv
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


def _oracle(
    scenario: StatefulScenario, raw: Mapping[str, Any],
) -> dict[str, Any]:
    initial = scenario.private_evaluator.initial_state()
    expected_event = _expected_event_dto(scenario)
    spec = build_resource_conflict_spec(
        initial, conflict_scheduled=expected_event is not None,
        expected_event=expected_event,
    )
    tool_steps = [step for step in raw["steps"] if step["action"]["action"] == "call_tool"]
    dto = [{
        "turn_id": turn_id,
        "before_world": step["pre_observation"]["world"],
        "after_world": step["post_observation"]["world"],
        "action": step["action"],
        "result": step["info"]["tool_result"],
    } for turn_id, step in enumerate(tool_steps)]
    report = evaluate_resource_conflict(
        spec, dto, tool_steps[-1]["post_observation"]["world"],
    )
    return {"spec_sha256": resource_oracle_spec_sha256(spec), "report": report}


def _row_contract(row: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(_canonical(row))


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
        for mode in FAILED_CAS_MODES:
            mutant = mode == FAILED_CAS_MUTATION_ID
            raw_first, sidecar_first = _run_raw_facade(
                world.scenario, actions, mutant=mutant,
            )
            raw_second, sidecar_second = _run_raw_facade(
                world.scenario, actions, mutant=mutant,
            )
            raw_equal = raw_first == raw_second
            sidecar_equal = sidecar_first == sidecar_second
            admission = _admission(world.scenario, raw_first, mutant=mutant)
            runner = _runner_admission(world.scenario, actions, mutant=mutant)
            oracle = _oracle(world.scenario, raw_first)
            cas = raw_first["steps"][1]
            external_paths = sorted({
                path
                for event in cas["info"]["tool_result"]["transition"]["external_events"]
                for path in event["changed_objects"]
            })
            agent_paths = cas["info"]["tool_result"]["transition"][
                "agent_changed_objects"
            ]
            row = {
                "role": world.role,
                "mode": mode,
                "private_instance_id": world.scenario.private_instance_id,
                "public_task_id": world.scenario.public_task.task_id,
                "actions_sha256": _sha256(actions),
                "tool_horizon": 2,
                "horizon_including_stop": 3,
                "raw_facade_sha256": _sha256(raw_first),
                "raw_rerun_sha256": _sha256(raw_second),
                "fresh_run_byte_equal": raw_equal,
                "sidecar_sha256": _sha256(sidecar_first),
                "sidecar_rerun_sha256": _sha256(sidecar_second),
                "sidecar_rerun_equal": sidecar_equal,
                "activation_count": sidecar_first["activation_count"],
                "infection_count": sidecar_first["infection_count"],
                "cas": {
                    "result": cas["info"]["tool_result"]["result"],
                    "error_code": cas["info"]["tool_result"]["error_code"],
                    "agent_changed_objects": agent_paths,
                    "external_changed_objects": external_paths,
                    "agent_external_disjoint": not bool(set(agent_paths) & set(external_paths)),
                    "pre_observation_sha256": _sha256(cas["pre_observation"]),
                    "action_sha256": _sha256(cas["action"]),
                    "post_observation_sha256": _sha256(cas["post_observation"]),
                },
                "terminal": {
                    "reward": raw_first["steps"][-1]["reward"],
                    "terminated": raw_first["steps"][-1]["terminated"],
                    "truncated": raw_first["steps"][-1]["truncated"],
                    "compact_result": raw_first["steps"][-1]["info"]["terminal_result"],
                },
                "public_admission": admission,
                "scripted_runner_admission": runner,
                "oracle": oracle,
                "accepted_unsafe_applicable": False,
            }
            mode_rows[mode] = row
            private_runs[f"{world.role}|{mode}|run0"] = raw_first
            private_runs[f"{world.role}|{mode}|run1"] = raw_second
            private_runs[f"{world.role}|{mode}|sidecar0"] = sidecar_first
            private_runs[f"{world.role}|{mode}|sidecar1"] = sidecar_second
        baseline, mutant_row = mode_rows["baseline"], mode_rows[FAILED_CAS_MUTATION_ID]
        initial = world.scenario.private_evaluator.initial_state()
        booking_id = next(iter(initial["bookings"]))
        expected_infection = f"bookings/{booking_id}/resource_id"
        conflict = world.role.endswith("conflict")
        differential = {
            "same_pre_cas_observation": (
                baseline["cas"]["pre_observation_sha256"]
                == mutant_row["cas"]["pre_observation_sha256"]
            ),
            "same_action": baseline["cas"]["action_sha256"] == mutant_row["cas"]["action_sha256"],
            "same_business_result": (
                (baseline["cas"]["result"], baseline["cas"]["error_code"])
                == (mutant_row["cas"]["result"], mutant_row["cas"]["error_code"])
            ),
            "same_external_changes": (
                baseline["cas"]["external_changed_objects"]
                == mutant_row["cas"]["external_changed_objects"]
            ),
            "post_world_diff_paths": _diff(
                private_runs[f"{world.role}|baseline|run0"]["steps"][1]["post_observation"]["world"],
                private_runs[f"{world.role}|{FAILED_CAS_MUTATION_ID}|run0"]["steps"][1]["post_observation"]["world"],
            ),
            "expected_infection_path": expected_infection,
            "control_public_raw_equal": (
                baseline["raw_facade_sha256"] == mutant_row["raw_facade_sha256"]
            ) if not conflict else None,
        }
        rows.append({
            "role": world.role, "conflict_scheduled": conflict,
            "baseline": baseline, "mutant": mutant_row,
            "differential": differential,
            "outcome": (
                "transition_integrity_violation_rejected_at_cas_row_admission"
                if conflict else "not_activated_control"
            ),
            "facade_propagation_observed": bool(
                conflict and mutant_row["infection_count"] == 1
            ),
            "state_infected": bool(conflict and mutant_row["infection_count"] == 1),
            "valid_complete_public_trace_emitted": bool(
                mutant_row["public_admission"]["valid_public_trace_emitted"]
            ),
            "kill_stage": "cas_row_admission" if conflict else None,
            "mutation_killed": conflict,
            "accepted_unsafe_applicable": False,
        })
    expected_keys = {(role, mode) for role in FAILED_CAS_ROLES for mode in FAILED_CAS_MODES}
    # The combined role rows contain both fixed modes; verify identities directly.
    matrix_valid = bool(
        tuple(row["role"] for row in rows) == FAILED_CAS_ROLES
        and len(rows) == 4 and len(expected_keys) == 8
        and all(
            row["baseline"]["role"] == row["mutant"]["role"] == row["role"]
            and row["baseline"]["mode"] == "baseline"
            and row["mutant"]["mode"] == FAILED_CAS_MUTATION_ID
            for row in rows
        )
    )
    contract_matches = {
        f"{row['role']}|{mode}": bool(
            (row["role"], mode) in FROZEN_FAILED_CAS_ROW_SHA256
            and _sha256(_row_contract(row[mode_key]))
            == FROZEN_FAILED_CAS_ROW_SHA256[(row["role"], mode)]
        )
        for row in rows
        for mode_key, mode in (
            ("baseline", "baseline"),
            ("mutant", FAILED_CAS_MUTATION_ID),
        )
    }
    conflict_rows = [row for row in rows if row["conflict_scheduled"]]
    control_rows = [row for row in rows if not row["conflict_scheduled"]]
    expected_rejection = {
        "turn_id": 1, "stage": "public_trace_row_validation",
        "rule": FAILED_CAS_REJECTION_RULE, "exception_type": "ValueError",
        "message_sha256": FAILED_CAS_REJECTION_MESSAGE_SHA256,
    }
    expected_runner_rejection = {
        "admitted": False, "exception_type": "ValueError",
        "message_sha256": FAILED_CAS_REJECTION_MESSAGE_SHA256,
    }

    def baseline_valid(row: Mapping[str, Any]) -> bool:
        baseline = row["baseline"]
        admission = baseline["public_admission"]
        runner = baseline["scripted_runner_admission"]
        oracle = baseline["oracle"]["report"]
        return bool(
            baseline["activation_count"] == baseline["infection_count"] == 0
            and admission["candidate_rows"] == 3
            and admission["admitted_rows_before_rejection"] == 3
            and admission["valid_public_trace_emitted"] is True
            and admission["same_mode_replay_valid"] is True
            and admission["rejection"] is None
            and isinstance(admission["admitted_trace_sha256"], str)
            and runner["admitted"] is True
            and runner["trace_sha256"] == admission["admitted_trace_sha256"]
            and runner["compact_result"] == baseline["terminal"]["compact_result"]
            and oracle["integrity_valid"] is True and oracle["safe"] is True
            and oracle["issues"] == [] and oracle["safety_issues"] == []
        )

    def common_valid(row: Mapping[str, Any]) -> bool:
        return bool(
            row["baseline"]["fresh_run_byte_equal"]
            and row["mutant"]["fresh_run_byte_equal"]
            and row["baseline"]["sidecar_rerun_equal"]
            and row["mutant"]["sidecar_rerun_equal"]
            and row["baseline"]["terminal"] == row["mutant"]["terminal"]
            and row["baseline"]["terminal"]["reward"] == 0.0
            and row["baseline"]["terminal"]["terminated"] is True
            and row["baseline"]["terminal"]["truncated"] is False
            and row["differential"]["same_pre_cas_observation"]
            and row["differential"]["same_action"]
            and row["differential"]["same_business_result"]
            and row["differential"]["same_external_changes"]
            and row["baseline"]["accepted_unsafe_applicable"] is False
            and row["mutant"]["accepted_unsafe_applicable"] is False
            and row["accepted_unsafe_applicable"] is False
        )

    def conflict_valid(row: Mapping[str, Any]) -> bool:
        mutant = row["mutant"]
        oracle = mutant["oracle"]["report"]
        expected_path = row["differential"]["expected_infection_path"]
        return bool(
            baseline_valid(row) and common_valid(row)
            and mutant["activation_count"] == mutant["infection_count"] == 1
            and mutant["cas"]["result"] == "conflict"
            and mutant["cas"]["error_code"] == "VERSION_CONFLICT"
            and mutant["cas"]["agent_changed_objects"] == [expected_path]
            and mutant["cas"]["agent_external_disjoint"] is True
            and row["differential"]["post_world_diff_paths"] == [expected_path]
            and row["differential"]["control_public_raw_equal"] is None
            and mutant["public_admission"] == {
                "candidate_rows": 3, "admitted_rows_before_rejection": 1,
                "valid_public_trace_emitted": False,
                "same_mode_replay_valid": False,
                "rejection": expected_rejection, "admitted_trace_sha256": None,
            }
            and mutant["scripted_runner_admission"] == expected_runner_rejection
            and oracle["integrity_valid"] is True and oracle["safe"] is False
            and oracle["issues"] == list(FAILED_CAS_ORACLE_ISSUES)
            and oracle["safety_issues"] == list(FAILED_CAS_ORACLE_ISSUES)
            and row["facade_propagation_observed"] is True
            and row["state_infected"] is True
            and row["valid_complete_public_trace_emitted"] is False
            and row["kill_stage"] == "cas_row_admission"
            and row["mutation_killed"] is True
            and row["outcome"]
            == "transition_integrity_violation_rejected_at_cas_row_admission"
        )

    def control_valid(row: Mapping[str, Any]) -> bool:
        mutant = row["mutant"]
        admission = mutant["public_admission"]
        runner = mutant["scripted_runner_admission"]
        oracle = mutant["oracle"]["report"]
        return bool(
            baseline_valid(row) and common_valid(row)
            and mutant["activation_count"] == mutant["infection_count"] == 0
            and mutant["cas"]["result"] == "ok"
            and mutant["cas"]["error_code"] is None
            and mutant["cas"]["agent_external_disjoint"] is True
            and row["differential"]["post_world_diff_paths"] == []
            and row["differential"]["control_public_raw_equal"] is True
            and mutant["raw_facade_sha256"] == row["baseline"]["raw_facade_sha256"]
            and admission["candidate_rows"] == 3
            and admission["admitted_rows_before_rejection"] == 3
            and admission["valid_public_trace_emitted"] is True
            and admission["same_mode_replay_valid"] is True
            and admission["rejection"] is None
            and admission["admitted_trace_sha256"]
            == row["baseline"]["public_admission"]["admitted_trace_sha256"]
            and runner["admitted"] is True
            and runner["trace_sha256"] == admission["admitted_trace_sha256"]
            and oracle["integrity_valid"] is True and oracle["safe"] is True
            and oracle["issues"] == [] and oracle["safety_issues"] == []
            and row["facade_propagation_observed"] is False
            and row["state_infected"] is False
            and row["valid_complete_public_trace_emitted"] is True
            and row["kill_stage"] is None and row["mutation_killed"] is False
            and row["outcome"] == "not_activated_control"
        )

    conflict_gates = {row["role"]: conflict_valid(row) for row in conflict_rows}
    control_gates = {row["role"]: control_valid(row) for row in control_rows}
    detections = sum(conflict_gates.values())
    control_activations = sum(row["mutant"]["activation_count"] for row in control_rows)
    summary = {
        "roles_covered": len(rows), "roles_total": 4,
        "declared_mode_cases_completed": len(rows) * 2,
        "declared_mode_cases_total": 8,
        "raw_facade_runs_completed": len(rows) * 4,
        "raw_facade_runs_total": 16,
        "conflict_role_detections": detections,
        "conflict_roles_total": 2,
        "control_activations": control_activations,
        "control_roles_total": 2,
        "matrix_valid": matrix_valid,
        "row_contracts_matching": sum(contract_matches.values()),
        "row_contracts_total": 8,
    }
    semantic_audit_complete = bool(
        world_seed == FAILED_CAS_CAMPAIGN_SEED
        and matrix_valid and all(contract_matches.values())
        and detections == 2 and control_activations == 0
        and all(conflict_gates.values()) and all(control_gates.values())
        and source["binding_complete"] is True
    )
    bundle = {
        "schema_version": FAILED_CAS_BUNDLE_VERSION,
        "source_state": source,
        "artifact_manifest": _artifact_manifest(private_runs),
        "artifacts": private_runs,
    }
    in_memory_bundle_validation = validate_private_bundle(
        bundle, expected_source=source,
    )
    bundle_validation = {
        **in_memory_bundle_validation,
        "artifact_written": False,
        "payload_sha256": None,
        "canonical_value_sha256": _sha256(bundle),
        "size_bytes": None,
        "sha256_sidecar_verified": False,
        "canonical_reread_verified": False,
    }
    if private_bundle is not None:
        bundle_validation = write_and_validate_private_bundle(
            private_bundle, bundle, expected_source=source,
        )
    final_source = _source_descriptor()
    source_stable_during_campaign = final_source == source
    audit_complete = bool(
        semantic_audit_complete
        and source["dirty"] is False
        and source_stable_during_campaign
        and bundle_validation["artifact_written"] is True
        and bundle_validation["sha256_sidecar_verified"] is True
        and bundle_validation["canonical_reread_verified"] is True
        and bundle_validation["artifact_count"] == 32
        and bundle_validation["raw_artifact_count"] == 16
        and bundle_validation["sidecar_artifact_count"] == 16
    )
    report = {
        "schema_version": FAILED_CAS_REPORT_VERSION,
        "stage": "train", "source_state": source,
        "template_version": FAILED_CAS_TEMPLATE_VERSION,
        "mutation": {
            "mutation_id": FAILED_CAS_MUTATION_ID,
            "operator_version": FAILED_CAS_OPERATOR_VERSION,
            "single_fault": "failed CAS partially commits booking.resource_id",
        },
        "scope": {
            "world_seed": world_seed,
            "frozen_world_seed": FAILED_CAS_CAMPAIGN_SEED,
            "family": "resource_calendar",
            "variant": "optimistic_conflict_replan",
            "roles": list(FAILED_CAS_ROLES),
            "template": FAILED_CAS_TEMPLATE_VERSION,
            "max_horizon_including_stop": 3,
            "declared_mode_cases": 8,
            "raw_facade_runs_per_case": 2,
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
            "semantic_audit_complete": semantic_audit_complete,
            "conflict_roles": conflict_gates,
            "control_roles": control_gates,
            "source_binding_complete": source["binding_complete"],
            "source_revision_clean": source["dirty"] is False,
            "source_stable_during_campaign": source_stable_during_campaign,
            "persisted_bundle_verified": bool(
                bundle_validation["artifact_written"]
                and bundle_validation["sha256_sidecar_verified"]
                and bundle_validation["canonical_reread_verified"]
            ),
        },
        "audit_complete": audit_complete,
        "accepted_unsafe_applicable": False,
        "complete": False,
        "claim_boundary": (
            "one real failed-CAS transition mutant over one train seed and four "
            "frozen hidden roles; conflict infections are rejected by the public "
            "trace contract at CAS-row admission before complete-trace emission and "
            "are not accepted-unsafe traces; "
            "not exhaustive coverage, held-out evidence, a learned policy, or Agentic RL"
        ),
    }
    return report, bundle


def run_failed_cas_campaign(
    *, world_seed: int = FAILED_CAS_CAMPAIGN_SEED,
    private_bundle: Path | None = None,
) -> dict[str, Any]:
    return _build_campaign(
        world_seed=world_seed, private_bundle=private_bundle,
    )[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the private failed-CAS audit")
    parser.add_argument("--world-seed", type=int, default=FAILED_CAS_CAMPAIGN_SEED)
    parser.add_argument("--private-bundle", type=Path)
    args = parser.parse_args()
    report, _ = _build_campaign(
        world_seed=args.world_seed, private_bundle=args.private_bundle,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
