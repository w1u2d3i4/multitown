"""Fail-closed semantic verifier for sealed A24 smoke artifacts.

The verifier is deliberately downstream of the experiment runner: it never
imports the runner, never trains a policy, never creates a formal lock, and
never writes inside the artifact being inspected.  The first public profile is
restricted to gate-negative non-evidentiary smoke artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import math
import os
import platform
import secrets
import stat
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from .a24_artifact_state import (
    canonical_value_sha256,
    sha256_file,
    validate_manifest,
)
from .a24_contract import (
    ALL_FITS_VERSION,
    CALIBRATION_GATE_VERSION,
    FIT_COMPLETE_VERSION,
    MECHANISM,
    RESULT_VERSION,
    RUN_CONTRACT_VERSION,
    TRAINING_CONTRACT_VERSION,
    UPDATE_LOG_VERSION,
    canonical_json_bytes,
    same_typed_json,
    strict_json_loads,
    strict_read_json,
    strict_read_jsonl,
)
from .a24_monitor import (
    ConcurrentObservationError,
    MonitorObservationError,
    _tree_snapshot,
)
from .a24_statistics import (
    build_claim_boundary,
    calibration_comparator_diagnostic,
    evaluate_calibration_gate,
)

VERIFIER_VERSION = "multitown-a24-semantic-verifier-v1"
RECEIPT_VERSION = "multitown-a24-semantic-verifier-receipt-v1"
POLICY_VERSION = "multitown-a24-trusted-input-policy-v1"
PROFILE = "host-private-gate-negative-v1"
EVIDENCE_LEVEL = "ARTIFACT_SEMANTICS_VERIFIED"
PQ1_PRIMITIVES_VERSION = "multitown-pq1-rowwise-on-policy-primitives-v1"
MAX_PROBABILITY_RATIO_DRIFT = 2e-5

_HEX40 = frozenset("0123456789abcdef")
_FIT_KEYS = {
    "schema_version",
    "pq1_primitives_version",
    "outer_fold",
    "training_seed",
    "mechanism",
    "shield_enabled",
    "final_update",
    "training_episode_draws",
    "sample_sequence_sha256",
    "sampled_unique_episodes",
    "initial_model_sha256",
    "initial_optimizer_sha256",
    "final_model_sha256",
    "final_optimizer_sha256",
    "mode_sequence_sha256",
    "mode_counts",
    "training_thresholds",
    "training_log_sha256",
    "checkpoint_sha256",
    "inner_train_ids_sha256",
    "calibration_ids_sha256",
    "outer_ids_sha256",
    "environment_steps",
    "optimizer_minibatches",
    "run_contract_sha256",
    "calibration_evaluations_during_training",
    "outer_evaluations_during_training",
    "training_seconds",
    "selected_checkpoint",
}
_LOG_KEYS = {
    "schema_version",
    "pq1_primitives_version",
    "outer_fold",
    "training_seed",
    "mechanism",
    "update",
    "episodes_per_update",
    "sampled_episode_ids",
    "sampled_episode_ids_sha256",
    "transition_episode_sha256",
    "pre_rollout_model_sha256",
    "post_rollout_model_sha256",
    "pre_rollout_optimizer_sha256",
    "post_rollout_optimizer_sha256",
    "post_update_model_sha256",
    "post_update_optimizer_sha256",
    "selected_actor_mode",
    "mode_counts",
    "environment_steps",
    "optimizer_minibatches",
    "rollout_summary",
    "actor_mode_decision",
    "snapshot_diagnostics",
    "ppo_metrics",
    "advantage_diagnostics",
}


class VerificationRejected(RuntimeError):
    """The artifact does not satisfy the selected trusted policy."""


class UnsupportedArtifact(VerificationRejected):
    """The artifact is outside this verifier profile."""


class VerificationConcurrent(RuntimeError):
    """The artifact or a trusted input changed during verification."""


@dataclass(frozen=True)
class VerificationRequest:
    root: Path
    output: Path
    expect: Literal["smoke"]
    trusted_policy: Path
    receipt_out: Path | None = None


def _reject(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationRejected(message)


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in _HEX40 for character in value)
    )


def _canonical_digest(value: Any) -> str:
    return _sha_bytes(canonical_json_bytes(value)[:-1])


def _read_bounded_json(path: Path, *, limit: int, label: str) -> dict[str, Any]:
    raw = _read_private_bytes(path, max_bytes=limit, label=label)
    try:
        value = strict_json_loads(raw.decode("utf-8"), label=label)
    except UnicodeDecodeError as exc:
        raise VerificationRejected(f"{label} is not UTF-8") from exc
    _reject(type(value) is dict, f"{label} is not an object")
    return value


def _read_private_bytes(path: Path, *, max_bytes: int, label: str) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise VerificationRejected(f"cannot stat {label}: {path}") from exc
    _reject(
        not path.is_symlink()
        and stat.S_ISREG(before.st_mode)
        and before.st_mode & 0o777 == 0o600
        and before.st_uid == os.getuid()
        and before.st_nlink == 1
        and before.st_size <= max_bytes,
        f"{label} is not a bounded private owned single-link file",
    )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise VerificationRejected(f"cannot open {label}: {path}") from exc
    try:
        opened = os.fstat(descriptor)
        identity = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_uid",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(getattr(before, name) != getattr(opened, name) for name in identity):
            raise VerificationConcurrent(f"{label} changed before open completed")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > max_bytes:
                raise VerificationRejected(f"{label} exceeds the byte limit")
        after = os.fstat(descriptor)
        current = path.lstat()
        if any(
            getattr(before, name) != getattr(after, name)
            or getattr(before, name) != getattr(current, name)
            for name in identity
        ):
            raise VerificationConcurrent(f"{label} changed during read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _validate_policy(policy: Mapping[str, Any]) -> None:
    _reject(
        set(policy)
        == {
            "schema_version",
            "policy_id",
            "scope",
            "producer",
            "resource_profile",
            "trusted_files",
        }
        and policy.get("schema_version") == POLICY_VERSION,
        "trusted policy schema changed",
    )
    scope = policy["scope"]
    _reject(
        type(scope) is dict
        and scope
        == {
            "evidence_level": EVIDENCE_LEVEL,
            "expect": "smoke",
            "gate": "negative-only",
            "independent_implementation": False,
            "inference_reexecution": False,
            "optimizer_update_reexecution": False,
            "profile": PROFILE,
        },
        "trusted policy scope changed",
    )
    profile = policy["resource_profile"]
    required_resources = {
        "checkpoint_max_address_space_bytes",
        "checkpoint_cpu_seconds",
        "checkpoint_max_bytes",
        "checkpoint_max_tensor_bytes",
        "checkpoint_max_tensors",
        "checkpoint_timeout_seconds",
        "json_max_bytes",
        "jsonl_max_lines",
        "trusted_jsonl_max_bytes",
        "tree_max_bytes",
        "tree_max_depth",
        "tree_max_entries",
    }
    _reject(
        type(profile) is dict
        and set(profile) == required_resources
        and all(type(profile[name]) is int and profile[name] > 0 for name in profile),
        "trusted policy resource profile changed",
    )
    trusted = policy["trusted_files"]
    _reject(
        type(trusted) is dict
        and bool(trusted)
        and all(
            type(path) is str and _is_sha256(digest) for path, digest in trusted.items()
        ),
        "trusted file mapping is malformed",
    )
    producer = policy["producer"]
    required_producer = {
        "allowed_protocol_revisions",
        "allowed_protocol_sha256",
        "allowed_source_revisions",
        "allowed_source_set_sha256",
        "expected_bindings",
        "expected_a8",
        "expected_nested_sha256",
        "expected_partitions_sha256",
        "required_source_paths",
        "smoke_schedule",
    }
    _reject(
        type(producer) is dict and set(producer) == required_producer,
        "trusted producer policy changed",
    )


def _verify_trusted_files(
    root: Path,
    policy: Mapping[str, Any],
    *,
    staging: Path | None = None,
) -> dict[str, str]:
    profile = policy["resource_profile"]
    observed: dict[str, str] = {}
    for relative, expected in sorted(policy["trusted_files"].items()):
        relative_path = Path(relative)
        _reject(
            not relative_path.is_absolute() and ".." not in relative_path.parts,
            f"trusted path is not repository-relative: {relative}",
        )
        candidate = root / relative_path
        path = candidate.resolve()
        _reject(path.is_relative_to(root), f"trusted path escapes the root: {relative}")
        _reject(
            path == Path(os.path.abspath(candidate)),
            f"trusted path traverses a symlink: {relative}",
        )
        limit = (
            profile["trusted_jsonl_max_bytes"]
            if path.suffix == ".jsonl"
            else profile["json_max_bytes"]
            if path.suffix == ".json"
            else profile["checkpoint_max_bytes"]
        )
        raw = _read_private_bytes(
            path,
            max_bytes=limit,
            label=f"trusted file {relative}",
        )
        actual = _sha_bytes(raw)
        _reject(actual == expected, f"trusted file digest changed: {relative}")
        observed[relative] = actual
        if staging is not None:
            destination = staging / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            current = destination.parent
            while current != staging:
                os.chmod(current, 0o700)
                current = current.parent
            descriptor = os.open(
                destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
    return observed


def _portable_snapshot(
    snapshot: tuple[tuple[Any, ...], ...],
) -> tuple[list[dict[str, Any]], str]:
    entries: list[dict[str, Any]] = []
    for item in snapshot:
        if item[0] == "directory":
            entries.append({"type": "directory", "path": item[1], "mode": item[2]})
        else:
            entries.append(
                {
                    "type": "file",
                    "path": item[1],
                    "mode": item[2],
                    "bytes": item[9],
                    "sha256": item[10],
                }
            )
    return entries, _canonical_digest(entries)


def _stage_artifact(
    output: Path, snapshot: tuple[tuple[Any, ...], ...], staging: Path
) -> None:
    directories = sorted(
        (item for item in snapshot if item[0] == "directory"),
        key=lambda item: (item[1].count("/"), item[1]),
    )
    for item in directories:
        destination = staging if item[1] == "." else staging / item[1]
        if destination == staging:
            os.chmod(destination, 0o700)
        else:
            destination.mkdir(mode=0o700)
    for item in sorted(
        (entry for entry in snapshot if entry[0] == "file"), key=lambda entry: entry[1]
    ):
        source = output / item[1]
        raw = _read_private_bytes(
            source, max_bytes=item[9], label=f"artifact file {item[1]}"
        )
        _reject(
            len(raw) == item[9] and _sha_bytes(raw) == item[10],
            f"artifact file changed while staging: {item[1]}",
        )
        destination = staging / item[1]
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise


def _enforce_staged_resource_limits(
    staged: Path | None,
    snapshot: tuple[tuple[Any, ...], ...],
    policy: Mapping[str, Any],
) -> None:
    limits = policy["resource_profile"]
    files = [item for item in snapshot if item[0] == "file"]
    _reject(
        len(snapshot) <= limits["tree_max_entries"],
        "artifact exceeds the policy entry limit",
    )
    _reject(
        sum(item[9] for item in files) <= limits["tree_max_bytes"],
        "artifact exceeds the policy tree-byte limit",
    )
    _reject(
        all(len(Path(item[1]).parts) <= limits["tree_max_depth"] for item in snapshot),
        "artifact exceeds the policy depth limit",
    )
    for item in files:
        path = Path(item[1]) if staged is None else staged / item[1]
        if path.suffix in {".json", ".jsonl"}:
            _reject(
                item[9] <= limits["json_max_bytes"],
                f"JSON artifact exceeds the byte limit: {item[1]}",
            )
        if path.suffix == ".jsonl" and staged is not None:
            with path.open("rb") as handle:
                lines = sum(
                    chunk.count(b"\n")
                    for chunk in iter(lambda: handle.read(1024 * 1024), b"")
                )
            _reject(
                lines <= limits["jsonl_max_lines"],
                f"JSONL artifact exceeds the row limit: {item[1]}",
            )
        if path.suffix == ".pt":
            _reject(
                item[9] <= limits["checkpoint_max_bytes"],
                f"checkpoint exceeds the byte limit: {item[1]}",
            )


def _verify_source(
    root: Path, source: Mapping[str, Any], policy: Mapping[str, Any]
) -> None:
    producer = policy["producer"]
    required = producer["required_source_paths"]
    _reject(
        type(source) is dict
        and set(source) == {"revision", "dirty", "files", "source_set_sha256"}
        and type(source["revision"]) is str
        and len(source["revision"]) == 40
        and source["dirty"] is False
        and type(source["files"]) is dict
        and set(source["files"]) == set(required),
        "producer source binding changed",
    )
    _reject(
        source["revision"] in producer["allowed_source_revisions"]
        and source["source_set_sha256"] in producer["allowed_source_set_sha256"],
        "producer source is not allowed by the trusted policy",
    )
    files: dict[str, str] = {}
    for relative in required:
        completed = subprocess.run(
            ["git", "show", f"{source['revision']}:{relative}"],
            cwd=root,
            capture_output=True,
            check=False,
            timeout=15,
        )
        _reject(
            completed.returncode == 0, f"cannot resolve producer source: {relative}"
        )
        files[relative] = _sha_bytes(completed.stdout)
    _reject(files == source["files"], "producer Git blobs differ from source receipt")
    _reject(
        _canonical_digest(files) == source["source_set_sha256"],
        "producer source-set digest changed",
    )


def _verify_contracts(
    root: Path, staged: Path, policy: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], str]:
    training = strict_read_json(staged / "training-contract.json")
    contract = strict_read_json(staged / "run-contract.json")
    producer = policy["producer"]
    _reject(
        training.get("schema_version") == TRAINING_CONTRACT_VERSION,
        "training contract version changed",
    )
    protocol = canonical_value_sha256(training)
    _reject(
        protocol in producer["allowed_protocol_sha256"],
        "training protocol is not allowed",
    )
    _reject(
        training.get("protocol_revision") in producer["allowed_protocol_revisions"],
        "protocol revision is not allowed",
    )
    _reject(
        training.get("schedule") == producer["smoke_schedule"], "smoke schedule changed"
    )
    _reject(
        training.get("mechanism") == MECHANISM
        and training.get("shield_enabled") is False,
        "mechanism or shield setting changed",
    )
    _reject(
        training.get("formal_execution_authorized") is False
        and training.get("non_evidentiary_smoke") is True,
        "smoke/formal classification changed",
    )
    _reject(
        canonical_value_sha256(training.get("partitions"))
        == producer["expected_partitions_sha256"],
        "partition contract changed",
    )
    _reject(
        type(contract) is dict
        and set(contract)
        == {
            "schema_version",
            "mode",
            "source",
            "runtime",
            "protocol_sha256",
            "preflight_signature",
            "pq1",
            "a23_failure",
            "bindings",
            "a22_comparator_ledger",
            "training_contract_sha256",
            "formal_lock",
        }
        and contract.get("schema_version") == RUN_CONTRACT_VERSION
        and contract.get("mode") == "non-evidentiary-smoke"
        and contract.get("formal_lock") is None
        and contract.get("protocol_sha256") == protocol
        and contract.get("training_contract_sha256") == protocol,
        "run contract changed or claims formal authority",
    )
    _verify_source(root, contract["source"], policy)
    nested = producer["expected_nested_sha256"]
    for name in ("pq1", "a23_failure", "bindings", "a22_comparator_ledger", "runtime"):
        _reject(
            canonical_value_sha256(contract[name]) == nested[name],
            f"run-contract {name} binding changed",
        )
    runtime = contract["runtime"]
    _reject(
        type(runtime) is dict
        and set(runtime)
        == {
            "python",
            "numpy",
            "torch",
            "platform",
            "requested_threads",
            "torch_num_threads",
            "torch_num_interop_threads",
            "torch_deterministic_algorithms",
            "torch_deterministic_warn_only",
        }
        and runtime["requested_threads"]
        == runtime["torch_num_threads"]
        == training["schedule"]["threads"]
        and type(runtime["torch_num_interop_threads"]) is int
        and runtime["torch_num_interop_threads"] > 0
        and runtime["torch_deterministic_algorithms"] is True
        and runtime["torch_deterministic_warn_only"] is False,
        "producer runtime contract changed",
    )
    _reject(
        contract["bindings"] == producer["expected_bindings"],
        "upstream input bindings changed",
    )
    ledger = contract["a22_comparator_ledger"]
    preflight = {
        "source": contract["source"],
        "runtime": contract["runtime"],
        "pq1": contract["pq1"],
        "a23_failure": contract["a23_failure"],
        "a22": {
            "raw": ledger["source_artifacts"],
            "canonical_run_contract": ledger["source_run_contract_sha256"],
            "comparator_ledger_sha256": ledger["ledger_sha256"],
        },
        "a8": producer["expected_a8"],
        "bindings": contract["bindings"],
    }
    _reject(
        _canonical_digest(preflight) == contract["preflight_signature"],
        "preflight signature cannot be recomputed",
    )
    return training, contract, canonical_value_sha256(contract)


def _validate_actor_decision(
    row: Mapping[str, Any], thresholds: Mapping[str, Any]
) -> None:
    decision = row.get("actor_mode_decision")
    diagnostics = row.get("advantage_diagnostics")
    _reject(
        type(decision) is dict and type(diagnostics) is dict,
        "actor diagnostics are malformed",
    )
    expected_keys = {
        "mode",
        "unsafe_cost",
        "wrong_cost",
        "unsafe_threshold",
        "wrong_threshold",
        "unsafe_violation",
        "wrong_violation",
        "unsafe_normalized_violation",
        "wrong_normalized_violation",
        "unsafe_eligible",
        "wrong_eligible",
        "unsafe_tie_break_used",
    }
    _reject(
        set(decision) == expected_keys
        and decision["mode"] in {"reward", "unsafe", "wrong"},
        "actor decision schema changed",
    )
    numeric = expected_keys - {
        "mode",
        "unsafe_eligible",
        "wrong_eligible",
        "unsafe_tie_break_used",
    }
    _reject(
        all(
            type(decision[name]) is float and math.isfinite(decision[name])
            for name in numeric
        ),
        "actor decision contains invalid numbers",
    )
    rollout = row["rollout_summary"]
    unsafe_cost = rollout["unsafe_events"] / rollout["episodes"]
    wrong_cost = (
        rollout["wrong_executions"] / rollout["episodes"] / thresholds["mean_incidents"]
    )
    unsafe_violation = unsafe_cost - thresholds["unsafe"]
    wrong_violation = wrong_cost - thresholds["wrong_per_incident"]
    unsafe_normalized = unsafe_violation / max(thresholds["unsafe"], 1e-8)
    wrong_normalized = wrong_violation / max(thresholds["wrong_per_incident"], 1e-8)
    unsafe_eligible = unsafe_cost > thresholds["unsafe"]
    wrong_eligible = wrong_cost > thresholds["wrong_per_incident"]
    mode = (
        "reward"
        if not unsafe_eligible and not wrong_eligible
        else (
            "unsafe"
            if unsafe_eligible
            and (not wrong_eligible or unsafe_normalized >= wrong_normalized)
            else "wrong"
        )
    )
    expected = {
        "mode": mode,
        "unsafe_cost": unsafe_cost,
        "wrong_cost": wrong_cost,
        "unsafe_threshold": thresholds["unsafe"],
        "wrong_threshold": thresholds["wrong_per_incident"],
        "unsafe_violation": unsafe_violation,
        "wrong_violation": wrong_violation,
        "unsafe_normalized_violation": unsafe_normalized,
        "wrong_normalized_violation": wrong_normalized,
        "unsafe_eligible": unsafe_eligible,
        "wrong_eligible": wrong_eligible,
        "unsafe_tie_break_used": bool(
            unsafe_eligible and wrong_eligible and unsafe_normalized == wrong_normalized
        ),
    }
    _reject(
        same_typed_json(decision, expected)
        and row["selected_actor_mode"] == mode
        and diagnostics.get("selected_actor_mode") == mode,
        "actor mode arithmetic changed",
    )
    summaries = {
        name: diagnostics.get(f"{name}_advantage_raw")
        for name in ("reward", "unsafe", "wrong")
    }
    selected = diagnostics.get("selected_advantage_raw")
    _reject(
        set(diagnostics)
        == {
            "selected_actor_mode",
            "normalized_advantage_sha256",
            "selected_advantage_constant",
            "reward_advantage_raw",
            "unsafe_advantage_raw",
            "wrong_advantage_raw",
            "selected_advantage_raw",
        }
        and _is_sha256(diagnostics["normalized_advantage_sha256"])
        and type(diagnostics["selected_advantage_constant"]) is bool
        and all(
            type(value) is dict
            and set(value) == {"mean", "std", "max_abs"}
            and all(
                type(value[field]) is float and math.isfinite(value[field])
                for field in value
            )
            and value["std"] >= 0.0
            and value["max_abs"] >= 0.0
            and value["max_abs"] >= abs(value["mean"])
            for value in (*summaries.values(), selected)
        )
        and diagnostics["selected_advantage_constant"] is (selected["std"] == 0.0),
        "advantage diagnostics changed",
    )
    source = summaries[mode]
    sign = 1.0 if mode == "reward" else -1.0
    _reject(
        selected
        == {
            "mean": sign * source["mean"],
            "std": source["std"],
            "max_abs": source["max_abs"],
        },
        "selected advantage source/sign changed",
    )


def _validate_snapshot_diagnostics(
    snapshot: Mapping[str, Any],
    *,
    transitions: Sequence[str],
    environment_steps: int,
) -> None:
    expected_keys = {
        "schema_version",
        "rowwise_log_probability_exact",
        "rowwise_value_exact",
        "rowwise_forward_calls",
        "rowwise_max_batch_size",
        "transition_count",
        "observation_sha256",
        "mask_sha256",
        "action_sha256",
        "full_batch_log_probability",
        "full_batch_value",
        "max_probability_ratio_drift",
        "diagnostic_gates",
        "first_legacy_exceed_transition_sha256",
        "ordered_transition_episode_sha256",
        "ordered_transition_batch_sha256",
    }
    expected_batch = _sha_bytes("".join(transitions).encode("ascii"))
    _reject(
        type(snapshot) is dict
        and set(snapshot) == expected_keys
        and snapshot["schema_version"] == PQ1_PRIMITIVES_VERSION
        and snapshot["rowwise_log_probability_exact"] is True
        and snapshot["rowwise_value_exact"] is True
        and type(snapshot["rowwise_forward_calls"]) is int
        and snapshot["rowwise_forward_calls"] == environment_steps
        and type(snapshot["rowwise_max_batch_size"]) is int
        and snapshot["rowwise_max_batch_size"] == 1
        and type(snapshot["transition_count"]) is int
        and snapshot["transition_count"] == environment_steps
        and all(
            _is_sha256(snapshot[name])
            for name in ("observation_sha256", "mask_sha256", "action_sha256")
        )
        and type(snapshot["max_probability_ratio_drift"]) is float
        and math.isfinite(snapshot["max_probability_ratio_drift"])
        and 0.0
        <= snapshot["max_probability_ratio_drift"]
        <= MAX_PROBABILITY_RATIO_DRIFT
        and (
            snapshot["first_legacy_exceed_transition_sha256"] is None
            or _is_sha256(snapshot["first_legacy_exceed_transition_sha256"])
        )
        and snapshot["ordered_transition_episode_sha256"] == list(transitions)
        and snapshot["ordered_transition_batch_sha256"] == expected_batch
        and same_typed_json(
            snapshot["diagnostic_gates"],
            {
                "full_batch_log_within_frozen_tolerance": True,
                "full_batch_value_within_frozen_tolerance": True,
                "probability_ratio_drift_within_2e_5": True,
            },
        ),
        "PQ snapshot schema or digest binding changed",
    )
    summary_keys = {
        "legacy_exceedances",
        "legacy_max_tolerance_ratio",
        "max_abs",
        "max_relative",
        "max_ulp",
        "p50_abs",
        "p95_abs",
        "p99_abs",
    }
    for name in ("full_batch_log_probability", "full_batch_value"):
        summary = snapshot[name]
        _reject(
            type(summary) is dict
            and set(summary) == summary_keys
            and type(summary["legacy_exceedances"]) is int
            and 0 <= summary["legacy_exceedances"] <= environment_steps
            and type(summary["max_ulp"]) is int
            and summary["max_ulp"] >= 0
            and all(
                type(summary[field]) is float and math.isfinite(summary[field])
                for field in summary_keys - {"legacy_exceedances", "max_ulp"}
            )
            and all(
                summary[field] >= 0.0
                for field in summary_keys - {"legacy_exceedances", "max_ulp"}
            )
            and summary["p50_abs"]
            <= summary["p95_abs"]
            <= summary["p99_abs"]
            <= summary["max_abs"],
            "PQ full-batch summary changed",
        )
        _reject(
            (summary["legacy_exceedances"] == 0)
            == (summary["legacy_max_tolerance_ratio"] <= 1.0),
            "PQ legacy tolerance/exceedance binding changed",
        )
        zero_drift = summary["max_abs"] == 0.0
        _reject(
            zero_drift == (summary["max_relative"] == 0.0)
            and zero_drift == (summary["max_ulp"] == 0)
            and zero_drift == (summary["legacy_max_tolerance_ratio"] == 0.0),
            "PQ zero/nonzero drift equivalence changed",
        )
        if zero_drift:
            _reject(
                summary["p50_abs"] == 0.0
                and summary["p95_abs"] == 0.0
                and summary["p99_abs"] == 0.0
                and summary["legacy_exceedances"] == 0,
                "PQ zero-drift summary changed",
            )
    log_exceedances = snapshot["full_batch_log_probability"]["legacy_exceedances"]
    _reject(
        (log_exceedances == 0)
        == (snapshot["first_legacy_exceed_transition_sha256"] is None),
        "PQ first legacy exceedance binding changed",
    )


def _validate_pq1_version(value: Any) -> None:
    _reject(value == PQ1_PRIMITIVES_VERSION, "PQ-1 primitives version changed")


def _validate_mode_counts(
    value: Any, expected: Mapping[str, int], *, expected_total: int
) -> None:
    _reject(
        type(value) is dict
        and set(value) == {"reward", "unsafe", "wrong"}
        and all(type(item) is int and item >= 0 for item in value.values())
        and same_typed_json(value, dict(expected))
        and sum(value.values()) == expected_total,
        "actor mode counts changed",
    )


def _validate_ppo_metrics(value: Any, rollout: Mapping[str, Any]) -> None:
    scalar_fields = {
        "approx_kl",
        "clip_fraction",
        "entropy",
        "policy_loss",
        "value_loss",
    }
    _reject(
        type(value) is dict
        and set(value)
        == scalar_fields | {"rollout_unsafe_events", "rollout_wrong_executions"}
        and all(
            type(value[name]) is float and math.isfinite(value[name])
            for name in scalar_fields
        )
        and type(value["rollout_unsafe_events"]) is int
        and type(value["rollout_wrong_executions"]) is int
        and value["rollout_unsafe_events"] == rollout["unsafe_events"]
        and value["rollout_wrong_executions"] == rollout["wrong_executions"]
        and value["clip_fraction"] >= 0.0
        and value["clip_fraction"] <= 1.0
        and value["entropy"] >= 0.0
        and value["value_loss"] >= 0.0,
        "PPO metric/rollout binding changed",
    )


def _run_checkpoint_worker(
    worker_script: Path,
    worker_cwd: Path,
    checkpoint: Path,
    fit: Mapping[str, Any],
    ppo: Mapping[str, Any],
    runtime: Mapping[str, Any],
    contract_sha: str,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    limits = policy["resource_profile"]
    _reject(
        checkpoint.stat().st_size <= limits["checkpoint_max_bytes"],
        "checkpoint exceeds the byte limit",
    )
    request = {
        "fit": dict(fit),
        "ppo_config": dict(ppo),
        "runtime": dict(runtime),
        "run_contract_sha256": contract_sha,
        "max_tensors": limits["checkpoint_max_tensors"],
        "max_tensor_bytes": limits["checkpoint_max_tensor_bytes"],
    }

    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            str(worker_script),
            "--checkpoint",
            str(checkpoint),
        ],
        cwd=worker_cwd,
        input=canonical_json_bytes(request),
        capture_output=True,
        check=False,
        timeout=limits["checkpoint_timeout_seconds"],
        env={
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "MULTITOWN_WORKER_MAX_AS": str(limits["checkpoint_max_address_space_bytes"]),
            "MULTITOWN_WORKER_MAX_CPU": str(limits["checkpoint_cpu_seconds"]),
            "MULTITOWN_WORKER_MAX_FSIZE": str(4 * 1024 * 1024),
            "MULTITOWN_WORKER_MAX_NOFILE": "32",
        },
    )
    _reject(
        completed.returncode == 0, f"safe checkpoint worker rejected {checkpoint.name}"
    )
    try:
        result = strict_json_loads(
            completed.stdout.decode("utf-8"), label="checkpoint worker output"
        )
    except UnicodeDecodeError as exc:
        raise VerificationRejected("checkpoint worker output is not UTF-8") from exc
    expected_keys = {
        "schema_version",
        "checkpoint_payload_schema_valid",
        "weights_only_load",
        "runtime_sha256",
        "torch_num_threads",
        "process_limits",
        "isolation",
        "initial_model_sha256",
        "initial_optimizer_sha256",
        "final_model_sha256",
        "final_optimizer_sha256",
        "tensor_count",
        "tensor_bytes",
    }
    _reject(
        type(result) is dict
        and set(result) == expected_keys
        and result["schema_version"] == "multitown-a24-weights-only-checkpoint-worker-v2"
        and result["checkpoint_payload_schema_valid"] is True
        and result["weights_only_load"] is True
        and result["runtime_sha256"] == _canonical_digest(runtime)
        and result["torch_num_threads"] == runtime["torch_num_threads"]
        and same_typed_json(
            result["process_limits"],
            {
                "address_space_bytes": [
                    limits["checkpoint_max_address_space_bytes"],
                    limits["checkpoint_max_address_space_bytes"],
                ],
                "cpu_seconds": [
                    limits["checkpoint_cpu_seconds"],
                    limits["checkpoint_cpu_seconds"],
                ],
                "file_size_bytes": [4 * 1024 * 1024, 4 * 1024 * 1024],
                "open_files": [32, 32],
            },
        )
        and result["isolation"]
        == {
            "dont_write_bytecode": True,
            "ignore_environment": True,
            "isolated_mode": True,
            "no_user_site": True,
            "safe_path": True,
        }
        and all(
            result[name] == fit[name]
            for name in (
                "initial_model_sha256",
                "initial_optimizer_sha256",
                "final_model_sha256",
                "final_optimizer_sha256",
            )
        )
        and type(result["tensor_count"]) is int
        and 0 < result["tensor_count"] <= limits["checkpoint_max_tensors"]
        and type(result["tensor_bytes"]) is int
        and 0 < result["tensor_bytes"] <= limits["checkpoint_max_tensor_bytes"],
        "checkpoint worker result changed",
    )
    return result


def _fit_provenance_expectations(
    trusted_root: Path,
    training: Mapping[str, Any],
) -> tuple[
    dict[int, Mapping[str, Any]],
    dict[int, dict[str, float]],
    dict[tuple[int, int], list[str]],
]:
    schedule = training["schedule"]
    partitions = training["partitions"]
    _reject(
        type(partitions) is list and len(partitions) == len(schedule["folds"]),
        "partition product changed",
    )
    partition_index: dict[int, Mapping[str, Any]] = {}
    partition_keys = {
        "outer_fold",
        "calibration_fold",
        "inner_train_ids",
        "inner_calibration_ids",
        "outer_ids",
        "inner_train_ids_sha256",
        "inner_calibration_ids_sha256",
        "outer_ids_sha256",
    }
    for partition in partitions:
        _reject(
            type(partition) is dict
            and set(partition) == partition_keys
            and type(partition["outer_fold"]) is int
            and type(partition["calibration_fold"]) is int
            and all(
                type(partition[name]) is list
                and bool(partition[name])
                and all(type(item) is str for item in partition[name])
                for name in (
                    "inner_train_ids",
                    "inner_calibration_ids",
                    "outer_ids",
                )
            )
            and partition["inner_train_ids_sha256"]
            == _canonical_digest(partition["inner_train_ids"])
            and partition["inner_calibration_ids_sha256"]
            == _canonical_digest(partition["inner_calibration_ids"])
            and partition["outer_ids_sha256"]
            == _canonical_digest(partition["outer_ids"]),
            "partition ID receipt changed",
        )
        partition_index[partition["outer_fold"]] = partition
    _reject(
        len(partition_index) == len(partitions)
        and set(partition_index) == set(schedule["folds"]),
        "partition fold product changed",
    )
    a8_rows = strict_read_jsonl(
        trusted_root / "artifacts/a9-v2-ppo-oof-20260813-r2/a8-oof-decisions.jsonl"
    )
    a8_index = {row.get("episode_id"): row for row in a8_rows}
    _reject(
        len(a8_index) == len(a8_rows),
        "trusted A8 episode index contains duplicates",
    )
    thresholds: dict[int, dict[str, float]] = {}
    sample_sequences: dict[tuple[int, int], list[str]] = {}
    draw_count = schedule["updates"] * schedule["episodes_per_update"]
    for fold, partition in partition_index.items():
        try:
            inner = [
                a8_index[episode_id] for episode_id in partition["inner_train_ids"]
            ]
        except KeyError as exc:
            raise VerificationRejected(
                "partition inner-train ID is absent from trusted A8"
            ) from exc
        mean_incidents = sum(int(row["incidents"]) for row in inner) / len(inner)
        unsafe = sum(bool(row["had_wrong_execution"]) for row in inner) / len(inner)
        wrong = (
            sum(int(row["wrong_executions"]) for row in inner)
            / len(inner)
            / mean_incidents
        )
        thresholds[fold] = {
            "unsafe": min(1.0, unsafe + 0.02),
            "wrong_per_incident": min(1.0, wrong + 0.01),
            "mean_incidents": mean_incidents,
        }
        for seed in schedule["seeds"]:
            log_path = (
                trusted_root
                / "artifacts/a22-adaptive-formal-20260814/fits"
                / f"outer-fold-{fold}/seed-{seed}/lagrangian/training-metrics.jsonl"
            )
            rows = strict_read_jsonl(log_path)
            flattened = [
                episode_id
                for row in rows
                for episode_id in row.get("sampled_episode_ids", [])
            ]
            _reject(
                len(flattened) >= draw_count
                and all(type(episode_id) is str for episode_id in flattened),
                "trusted A22 sample sequence changed",
            )
            sample_sequences[(fold, seed)] = flattened[:draw_count]
    return partition_index, thresholds, sample_sequences


def _fit_receipt_index(fits: Any) -> dict[tuple[int, int], dict[str, Any]]:
    _reject(
        type(fits) is list
        and all(
            type(fit) is dict
            and set(fit) == _FIT_KEYS
            and fit.get("schema_version") == FIT_COMPLETE_VERSION
            and type(fit.get("outer_fold")) is int
            and type(fit.get("training_seed")) is int
            for fit in fits
        ),
        "fit receipt index schema changed",
    )
    return {(fit["outer_fold"], fit["training_seed"]): fit for fit in fits}


def _validate_fit_logs(
    trusted_root: Path,
    staged: Path,
    training: Mapping[str, Any],
    training_runtime: Mapping[str, Any],
    contract_sha: str,
    policy: Mapping[str, Any],
    worker_script: Path,
    worker_cwd: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    schedule = training["schedule"]
    partitions, expected_thresholds, expected_samples = _fit_provenance_expectations(
        trusted_root, training
    )
    all_fits = strict_read_json(staged / "all-fits-complete.json")
    expected_keys = {
        "schema_version",
        "fits",
        "expected_fits",
        "expected_optimizer_updates",
        "expected_training_episode_draws",
        "all_reached_final_update",
        "exact_fit_key_product_verified",
        "pq1_provenance_and_checkpoint_chain_verified",
        "calibration_started_at_receipt",
        "outer_evaluation_started_at_receipt",
    }
    expected_cells = {
        (fold, seed) for fold in schedule["folds"] for seed in schedule["seeds"]
    }
    _reject(
        type(all_fits) is dict
        and set(all_fits) == expected_keys
        and all_fits["schema_version"] == ALL_FITS_VERSION
        and type(all_fits["fits"]) is list,
        "all-fits receipt changed",
    )
    fits = all_fits["fits"]
    index = _fit_receipt_index(fits)
    draws = len(expected_cells) * schedule["updates"] * schedule["episodes_per_update"]
    _reject(
        len(fits) == len(index) == len(expected_cells)
        and set(index) == expected_cells
        and all_fits["expected_fits"] == len(expected_cells)
        and all_fits["expected_optimizer_updates"]
        == len(expected_cells) * schedule["updates"]
        and all_fits["expected_training_episode_draws"] == draws
        and all_fits["all_reached_final_update"] is True
        and all_fits["exact_fit_key_product_verified"] is True
        and all_fits["pq1_provenance_and_checkpoint_chain_verified"] is True
        and all_fits["calibration_started_at_receipt"] is False
        and all_fits["outer_evaluation_started_at_receipt"] is False,
        "fit product/count receipt changed",
    )
    worker_results: list[dict[str, Any]] = []
    for fold, seed in sorted(expected_cells):
        fit = index[(fold, seed)]
        _reject(
            set(fit) == _FIT_KEYS and fit["schema_version"] == FIT_COMPLETE_VERSION,
            "fit receipt schema changed",
        )
        _validate_pq1_version(fit["pq1_primitives_version"])
        _reject(
            type(fit["outer_fold"]) is int
            and fit["outer_fold"] == fold
            and type(fit["training_seed"]) is int
            and fit["training_seed"] == seed
            and type(fit["final_update"]) is int
            and type(fit["training_episode_draws"]) is int
            and type(fit["sampled_unique_episodes"]) is int
            and type(fit["environment_steps"]) is int
            and type(fit["optimizer_minibatches"]) is int
            and type(fit["calibration_evaluations_during_training"]) is int
            and type(fit["outer_evaluations_during_training"]) is int
            and type(fit["training_seconds"]) is float
            and math.isfinite(fit["training_seconds"])
            and fit["training_seconds"] >= 0.0,
            "fit receipt scalar types changed",
        )
        fit_dir = staged / f"fits/outer-fold-{fold}/seed-{seed}/{MECHANISM}"
        _reject(
            same_typed_json(strict_read_json(fit_dir / "fit-complete.json"), fit),
            "fit receipt and aggregate differ",
        )
        logs = strict_read_jsonl(fit_dir / "training-metrics.jsonl")
        _reject(len(logs) == schedule["updates"], "fit update count changed")
        thresholds = fit["training_thresholds"]
        _reject(
            type(thresholds) is dict
            and set(thresholds) == {"unsafe", "wrong_per_incident", "mean_incidents"}
            and all(
                type(value) is float and math.isfinite(value)
                for value in thresholds.values()
            )
            and 0 <= thresholds["unsafe"] <= 1
            and 0 <= thresholds["wrong_per_incident"] <= 1
            and thresholds["mean_incidents"] > 0,
            "fit thresholds changed",
        )
        partition = partitions[fold]
        _reject(
            same_typed_json(thresholds, expected_thresholds[fold])
            and fit["inner_train_ids_sha256"] == partition["inner_train_ids_sha256"]
            and fit["calibration_ids_sha256"]
            == partition["inner_calibration_ids_sha256"]
            and fit["outer_ids_sha256"] == partition["outer_ids_sha256"],
            "fit partition or A8-derived threshold provenance changed",
        )
        modes: list[str] = []
        samples: list[str] = []
        counts: Counter[str] = Counter()
        previous_model = fit["initial_model_sha256"]
        previous_optimizer = fit["initial_optimizer_sha256"]
        total_steps = 0
        total_minibatches = 0
        for update, row in enumerate(logs, start=1):
            _reject(
                type(row) is dict
                and set(row) == _LOG_KEYS
                and row["schema_version"] == UPDATE_LOG_VERSION,
                "update-log schema changed",
            )
            _validate_pq1_version(row["pq1_primitives_version"])
            mode = row["selected_actor_mode"]
            _reject(
                type(mode) is str and mode in {"reward", "unsafe", "wrong"},
                "actor mode changed",
            )
            counts[mode] += 1
            modes.append(mode)
            _reject(
                type(row["sampled_episode_ids"]) is list
                and type(row["transition_episode_sha256"]) is list
                and type(row["snapshot_diagnostics"]) is dict
                and type(row["rollout_summary"]) is dict
                and type(row["ppo_metrics"]) is dict
                and type(row["environment_steps"]) is int
                and row["environment_steps"] > 0,
                "update-log value types changed",
            )
            samples.extend(row["sampled_episode_ids"])
            transitions = row["transition_episode_sha256"]
            snapshot = row["snapshot_diagnostics"]
            rollout = row["rollout_summary"]
            expected_batch = (
                _sha_bytes("".join(transitions).encode("ascii"))
                if type(transitions) is list
                and all(_is_sha256(value) for value in transitions)
                else None
            )
            expected_minibatches = training["ppo"]["ppo_epochs"] * math.ceil(
                row["environment_steps"] / training["ppo"]["minibatch_size"]
            )
            _reject(
                type(row["update"]) is int
                and row["update"] == update
                and type(row["outer_fold"]) is int
                and row["outer_fold"] == fold
                and type(row["training_seed"]) is int
                and row["training_seed"] == seed
                and row["mechanism"] == MECHANISM
                and type(row["episodes_per_update"]) is int
                and row["episodes_per_update"] == schedule["episodes_per_update"]
                and type(row["sampled_episode_ids"]) is list
                and len(row["sampled_episode_ids"]) == schedule["episodes_per_update"]
                and all(type(value) is str for value in row["sampled_episode_ids"])
                and _canonical_digest(row["sampled_episode_ids"])
                == row["sampled_episode_ids_sha256"]
                and type(transitions) is list
                and len(transitions) == schedule["episodes_per_update"]
                and all(_is_sha256(value) for value in transitions)
                and expected_batch is not None
                and all(
                    _is_sha256(row[name])
                    for name in (
                        "pre_rollout_model_sha256",
                        "post_rollout_model_sha256",
                        "pre_rollout_optimizer_sha256",
                        "post_rollout_optimizer_sha256",
                        "post_update_model_sha256",
                        "post_update_optimizer_sha256",
                    )
                )
                and row["pre_rollout_model_sha256"]
                == row["post_rollout_model_sha256"]
                == previous_model
                and row["pre_rollout_optimizer_sha256"]
                == row["post_rollout_optimizer_sha256"]
                == previous_optimizer
                and type(row["environment_steps"]) is int
                and row["environment_steps"] > 0
                and type(row["optimizer_minibatches"]) is int
                and row["optimizer_minibatches"] == expected_minibatches
                and type(rollout) is dict
                and set(rollout)
                == {
                    "episodes",
                    "incidents",
                    "unsafe_events",
                    "wrong_executions",
                    "shield_interventions",
                }
                and all(
                    type(rollout[name]) is int and rollout[name] >= 0
                    for name in rollout
                )
                and rollout["episodes"] == schedule["episodes_per_update"]
                and rollout["unsafe_events"] <= rollout["episodes"]
                and rollout["wrong_executions"] >= rollout["unsafe_events"]
                and rollout["wrong_executions"] <= rollout["incidents"]
                and rollout["incidents"] <= row["environment_steps"]
                and rollout["shield_interventions"] == 0
                and type(snapshot) is dict,
                "update-log provenance or arithmetic changed",
            )
            _validate_mode_counts(
                row["mode_counts"],
                {name: counts[name] for name in ("reward", "unsafe", "wrong")},
                expected_total=update,
            )
            _validate_ppo_metrics(row["ppo_metrics"], rollout)
            _validate_snapshot_diagnostics(
                snapshot,
                transitions=transitions,
                environment_steps=row["environment_steps"],
            )
            _validate_actor_decision(row, thresholds)
            previous_model = row["post_update_model_sha256"]
            previous_optimizer = row["post_update_optimizer_sha256"]
            total_steps += row["environment_steps"]
            total_minibatches += row["optimizer_minibatches"]
        progress = strict_read_json(fit_dir / "progress.json")
        expected_progress = {
            "schema_version": "multitown-a24-fit-progress-v1",
            "outer_fold": fold,
            "training_seed": seed,
            "mechanism": MECHANISM,
            "current_update": schedule["updates"],
            "scheduled_updates": schedule["updates"],
            "mode_counts": fit["mode_counts"],
            "calibration_started": False,
            "outer_evaluation_started": False,
        }
        _validate_mode_counts(
            fit["mode_counts"],
            {name: counts[name] for name in ("reward", "unsafe", "wrong")},
            expected_total=schedule["updates"],
        )
        _reject(
            fit["mechanism"] == MECHANISM
            and fit["shield_enabled"] is False
            and fit["selected_checkpoint"] == "final"
            and fit["calibration_evaluations_during_training"] == 0
            and fit["outer_evaluations_during_training"] == 0
            and fit["final_update"] == schedule["updates"]
            and fit["training_episode_draws"]
            == len(samples)
            == schedule["updates"] * schedule["episodes_per_update"]
            and fit["sampled_unique_episodes"] == len(set(samples))
            and fit["sample_sequence_sha256"] == _canonical_digest(samples)
            and samples == expected_samples[(fold, seed)]
            and fit["mode_sequence_sha256"] == _canonical_digest(modes)
            and fit["training_log_sha256"]
            == sha256_file(fit_dir / "training-metrics.jsonl")
            and fit["checkpoint_sha256"] == sha256_file(fit_dir / "final.pt")
            and fit["final_model_sha256"] == previous_model
            and fit["final_optimizer_sha256"] == previous_optimizer
            and fit["environment_steps"] == total_steps
            and fit["optimizer_minibatches"] == total_minibatches
            and fit["run_contract_sha256"] == contract_sha
            and same_typed_json(progress, expected_progress),
            "fit log/checkpoint/progress chain changed",
        )
        worker_results.append(
            _run_checkpoint_worker(
                worker_script=worker_script,
                worker_cwd=worker_cwd,
                checkpoint=fit_dir / "final.pt",
                fit=fit,
                ppo=training["ppo"],
                runtime=training_runtime,
                contract_sha=contract_sha,
                policy=policy,
            )
        )
    return fits, worker_results


def _select_calibration_inputs(
    trusted_root: Path,
    training: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    max_lines = policy["resource_profile"]["jsonl_max_lines"]
    a8_all = strict_read_jsonl(
        trusted_root / "artifacts/a9-v2-ppo-oof-20260813-r2/a8-oof-decisions.jsonl"
    )
    a22_all = strict_read_jsonl(
        trusted_root
        / "artifacts/a22-adaptive-formal-20260814/calibration-decisions.jsonl"
    )
    _reject(
        len(a8_all) <= max_lines and len(a22_all) <= max_lines,
        "trusted calibration source exceeds the row limit",
    )
    a8_index = {row.get("episode_id"): row for row in a8_all}
    schedule = training["schedule"]
    partitions = {
        partition["outer_fold"]: partition for partition in training["partitions"]
    }
    selected_ids = {
        fold: partitions[fold]["inner_calibration_ids"][
            : schedule["calibration_episodes_per_fold"]
        ]
        for fold in schedule["folds"]
    }
    a8 = [
        a8_index[episode_id]
        for fold in schedule["folds"]
        for episode_id in selected_ids[fold]
    ]
    expected = {
        (fold, seed, episode_id)
        for fold in schedule["folds"]
        for seed in schedule["seeds"]
        for episode_id in selected_ids[fold]
    }
    a22 = [
        row
        for row in a22_all
        if row.get("mechanism") == "lagrangian"
        and (
            row.get("design_outer_fold"),
            row.get("training_seed"),
            row.get("episode_id"),
        )
        in expected
    ]
    _reject(
        len(a8) == len({row["episode_id"] for row in a8}) and len(a22) == len(expected),
        "trusted calibration selection changed",
    )
    return a8, a22


def _validate_calibration(
    trusted_root: Path,
    staged: Path,
    training: Mapping[str, Any],
    contract: Mapping[str, Any],
    contract_sha: str,
    fits: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = strict_read_jsonl(staged / "calibration-decisions.jsonl")
    schedule = training["schedule"]
    expected_rows = (
        len(schedule["folds"])
        * len(schedule["seeds"])
        * schedule["calibration_episodes_per_fold"]
    )
    _reject(len(rows) == expected_rows, "A24 calibration row count changed")
    a8, a22 = _select_calibration_inputs(trusted_root, training, policy)
    checkpoints = {
        (fit["outer_fold"], fit["training_seed"]): fit["checkpoint_sha256"]
        for fit in fits
    }
    expected_a8 = policy["producer"]["expected_a8"]["run_digest"]
    decision = evaluate_calibration_gate(
        a8,
        rows,
        folds=schedule["folds"],
        seeds=schedule["seeds"],
        episodes_per_fold=schedule["calibration_episodes_per_fold"],
        smoke=True,
        expected_a8_run_contract_sha256=expected_a8,
        expected_run_contract_sha256=contract_sha,
        expected_checkpoint_sha256=checkpoints,
    )
    ledger = contract["a22_comparator_ledger"]
    diagnostic = calibration_comparator_diagnostic(
        a8,
        rows,
        a22,
        folds=schedule["folds"],
        seeds=schedule["seeds"],
        episodes_per_fold=schedule["calibration_episodes_per_fold"],
        expected_a8_run_contract_sha256=expected_a8,
        expected_a24_run_contract_sha256=contract_sha,
        expected_a22_run_contract_sha256=ledger["source_run_contract_sha256"],
        expected_a22_source_artifacts=ledger["source_artifacts"],
        expected_a24_checkpoint_sha256=checkpoints,
        comparator_ledger=ledger,
    )
    expected_gate = {
        "schema_version": CALIBRATION_GATE_VERSION,
        "run_contract_sha256": contract_sha,
        "comparator_ledger_sha256": ledger["ledger_sha256"],
        "calibration_sha256": sha256_file(staged / "calibration-decisions.jsonl"),
        "a24_calibration_rows": len(rows),
        "a22_lagrangian_descriptive_rows": len(a22),
        "a22_lagrangian_enters_gate": False,
        "a22_lagrangian_calibration_diagnostic": diagnostic,
        "decision": decision,
        "smoke_forced_outer": False,
    }
    gate = strict_read_json(staged / "calibration-gate.json")
    _reject(
        same_typed_json(gate, expected_gate),
        "calibration gate cannot be independently recomputed",
    )
    _reject(
        decision["raw_conjunction"] is False
        and decision["outer_gate_permitted"] is False,
        "gate-open artifacts are unsupported by this profile",
    )
    return rows, gate


def _negative_products(schedule: Mapping[str, Any]) -> dict[str, int]:
    cells = len(schedule["folds"]) * len(schedule["seeds"])
    return {
        "fits": cells,
        "optimizer_updates": cells * schedule["updates"],
        "training_episode_draws": cells
        * schedule["updates"]
        * schedule["episodes_per_update"],
        "a24_calibration_rows": cells * schedule["calibration_episodes_per_fold"],
        "a22_lagrangian_calibration_source_rows": cells
        * schedule["calibration_episodes_per_fold"],
        "a8_outer_source_rows": 0,
        "a24_outer_rows": 0,
        "a22_lagrangian_outer_rows": 0,
        "manifest_entries": 6 + cells * 4,
    }


def _validate_result(
    staged: Path,
    training: Mapping[str, Any],
    contract: Mapping[str, Any],
    contract_sha: str,
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    result = strict_read_json(staged / "result.json")
    expected = {
        "schema_version": RESULT_VERSION,
        "mode": "non-evidentiary-smoke",
        "non_evidentiary_smoke": True,
        "formal_execution_authorized": False,
        "formal_lock_acquired": False,
        "source_revision": contract["source"]["revision"],
        "run_contract_sha256": contract_sha,
        "protocol_sha256": contract["protocol_sha256"],
        "comparator_ledger_sha256": contract["a22_comparator_ledger"]["ledger_sha256"],
        "calibration_raw_conjunction": False,
        "formal_calibration_gate_evaluable": False,
        "formal_outer_gate_open": False,
        "smoke_outer_path_exercised": False,
        "products": _negative_products(training["schedule"]),
        "statistics": None,
        "claim_boundary": build_claim_boundary(
            terminal_state="NON_EVIDENTIARY_SMOKE",
            smoke=True,
            outer_performance_evaluable=False,
        ),
        "validation": {
            "pq1_functions_executed_every_update": True,
            "exact_fit_log_checkpoint_chain": True,
            "a22_calibration_enters_gate": False,
            "a8_rows_logically_paired_not_replicated": True,
            "formal_lock_untouched": True,
            "formal_lock_verified": False,
        },
    }
    _reject(
        gate["decision"]["raw_conjunction"] is False
        and same_typed_json(result, expected),
        "smoke result cannot be recomputed",
    )
    return result


def _verifier_identity(
    root: Path,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    package_root = Path(__file__).resolve().parents[1]
    _reject(
        package_root == root,
        "host-private verifier profile requires code and data in one clean checkout",
    )
    relatives = (
        "multitown/__init__.py",
        "multitown/a24_verifier.py",
        "multitown/a24_checkpoint_worker.py",
        "multitown/a24_artifact_state.py",
        "multitown/a24_contract.py",
        "multitown/a24_monitor.py",
        "multitown/a24_statistics.py",
    )
    source_bytes = {
        relative: _read_private_bytes(
            root / relative,
            max_bytes=4 * 1024 * 1024,
            label=f"verifier source {relative}",
        )
        for relative in relatives
    }
    files = {relative: _sha_bytes(raw) for relative, raw in source_bytes.items()}
    git_environment = {
        **os.environ,
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_CONFIG_NOSYSTEM": "1",
    }
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
        timeout=15,
        env=git_environment,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
        env=git_environment,
    ).stdout.strip()
    identity = {
        "version": VERIFIER_VERSION,
        "profile": PROFILE,
        "source_revision": revision,
        "source_dirty": bool(status),
        "source_files": files,
        "source_set_sha256": _canonical_digest(files),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": importlib.metadata.version("numpy"),
            "torch": importlib.metadata.version("torch"),
        },
    }
    return identity, source_bytes


def _receipt(core: Mapping[str, Any]) -> dict[str, Any]:
    digest = _canonical_digest(core)
    return {
        "schema_version": RECEIPT_VERSION,
        "verification_id": digest,
        "core": dict(core),
        "envelope": {
            "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "signature": None,
            "signature_scheme": None,
            "self_reported_not_an_external_attestation": True,
        },
    }


def _validate_receipt_core(core: Mapping[str, Any]) -> None:
    _reject(
        set(core)
        == {
            "target",
            "trusted_policy",
            "verifier",
            "recomputed",
            "counts",
            "decision",
            "limitations",
        },
        "verifier receipt core schema changed",
    )
    target = core["target"]
    _reject(
        type(target) is dict
        and set(target)
        == {
            "expect",
            "profile",
            "evidence_level",
            "artifact_snapshot_sha256",
            "manifest_sha256",
            "manifest_terminal_state",
            "source_revision",
            "source_set_sha256",
            "protocol_sha256",
            "run_contract_sha256",
            "formal_lock_observed_absent",
        }
        and target["expect"] == "smoke"
        and target["profile"] == PROFILE
        and target["evidence_level"] == EVIDENCE_LEVEL
        and target["manifest_terminal_state"] == "NON_EVIDENTIARY_SMOKE"
        and target["formal_lock_observed_absent"] is True
        and type(target["source_revision"]) is str
        and len(target["source_revision"]) == 40
        and all(character in _HEX40 for character in target["source_revision"])
        and all(
            _is_sha256(target[name])
            for name in (
                "artifact_snapshot_sha256",
                "manifest_sha256",
                "source_set_sha256",
                "protocol_sha256",
                "run_contract_sha256",
            )
        ),
        "verifier receipt target changed",
    )
    trusted = core["trusted_policy"]
    _reject(
        type(trusted) is dict
        and set(trusted) == {"policy_id", "policy_sha256", "trusted_files_sha256"}
        and type(trusted["policy_id"]) is str
        and bool(trusted["policy_id"])
        and _is_sha256(trusted["policy_sha256"])
        and _is_sha256(trusted["trusted_files_sha256"]),
        "verifier receipt trusted-policy binding changed",
    )
    verifier = core["verifier"]
    _reject(
        type(verifier) is dict
        and set(verifier)
        == {
            "version",
            "profile",
            "source_revision",
            "source_dirty",
            "source_files",
            "source_set_sha256",
            "runtime",
        }
        and verifier["version"] == VERIFIER_VERSION
        and verifier["profile"] == PROFILE
        and type(verifier["source_revision"]) is str
        and len(verifier["source_revision"]) == 40
        and all(character in _HEX40 for character in verifier["source_revision"])
        and type(verifier["source_dirty"]) is bool
        and type(verifier["source_files"]) is dict
        and len(verifier["source_files"]) >= 3
        and all(
            type(name) is str and _is_sha256(digest)
            for name, digest in verifier["source_files"].items()
        )
        and _is_sha256(verifier["source_set_sha256"])
        and verifier["source_set_sha256"] == _canonical_digest(verifier["source_files"])
        and type(verifier["runtime"]) is dict
        and set(verifier["runtime"]) == {"python", "platform", "numpy", "torch"}
        and all(
            type(value) is str and bool(value) for value in verifier["runtime"].values()
        ),
        "verifier receipt verifier identity changed",
    )
    recomputed = core["recomputed"]
    _reject(
        type(recomputed) is dict
        and set(recomputed)
        == {
            "preflight_signature",
            "fit_receipts_sha256",
            "checkpoint_summaries_sha256",
            "calibration_rows_sha256",
            "calibration_gate_sha256",
            "outer_rows_sha256",
            "statistics_sha256",
            "result_sha256",
        }
        and all(
            _is_sha256(recomputed[name])
            for name in (
                "preflight_signature",
                "fit_receipts_sha256",
                "checkpoint_summaries_sha256",
                "calibration_rows_sha256",
                "calibration_gate_sha256",
                "result_sha256",
            )
        )
        and recomputed["outer_rows_sha256"] is None
        and recomputed["statistics_sha256"] is None,
        "verifier receipt recomputation summary changed",
    )
    counts = core["counts"]
    _reject(
        type(counts) is dict
        and set(counts)
        == {
            "files_and_directories",
            "fits",
            "optimizer_updates",
            "training_episode_draws",
            "calibration_rows",
            "outer_rows",
        }
        and all(type(value) is int and value >= 0 for value in counts.values())
        and counts["files_and_directories"] > 0
        and counts["fits"] > 0
        and counts["optimizer_updates"] > 0
        and counts["training_episode_draws"] > 0
        and counts["outer_rows"] == 0,
        "verifier receipt product counts changed",
    )
    expected_decision = {
        "inventory_valid": True,
        "artifact_semantic_recomputation_valid": True,
        "checkpoint_weights_only_reconstruction": True,
        "fit_log_chain_valid": True,
        "calibration_gate_recomputed": True,
        "result_recomputed": True,
        "execution_transition_derivation_replayed": False,
        "inference_reexecution": False,
        "optimizer_update_reexecution": False,
        "results_reproduced": False,
        "formal_terminal_valid": False,
        "calibration_evidence_evaluable": False,
        "outer_performance_evaluable": False,
        "formal_evidence_accepted": False,
    }
    _reject(
        same_typed_json(core["decision"], expected_decision),
        "verifier receipt evidence decision changed",
    )
    _reject(
        type(core["limitations"]) is list
        and bool(core["limitations"])
        and all(type(item) is str and bool(item) for item in core["limitations"]),
        "verifier receipt limitations changed",
    )


def validate_receipt(receipt: Mapping[str, Any]) -> None:
    _reject(
        type(receipt) is dict
        and set(receipt) == {"schema_version", "verification_id", "core", "envelope"}
        and receipt["schema_version"] == RECEIPT_VERSION
        and _is_sha256(receipt["verification_id"])
        and type(receipt["core"]) is dict
        and type(receipt["envelope"]) is dict
        and set(receipt["envelope"])
        == {
            "generated_at_utc",
            "signature",
            "signature_scheme",
            "self_reported_not_an_external_attestation",
        }
        and receipt["envelope"]["signature"] is None
        and receipt["envelope"]["signature_scheme"] is None
        and type(receipt["envelope"]["generated_at_utc"]) is str
        and receipt["envelope"]["self_reported_not_an_external_attestation"] is True
        and _canonical_digest(receipt["core"]) == receipt["verification_id"],
        "verifier receipt is malformed or was tampered with",
    )
    try:
        generated_at = datetime.fromisoformat(receipt["envelope"]["generated_at_utc"])
    except ValueError as exc:
        raise VerificationRejected(
            "receipt timestamp is not RFC 3339 compatible"
        ) from exc
    _reject(
        generated_at.tzinfo is not None
        and generated_at.utcoffset() == datetime.now(UTC).utcoffset()
        and receipt["envelope"]["generated_at_utc"].endswith("Z"),
        "receipt timestamp is not canonical UTC",
    )
    _validate_receipt_core(receipt["core"])


def _publish_receipt(
    path: Path,
    receipt: Mapping[str, Any],
    *,
    root: Path,
    output: Path,
    policy_path: Path,
) -> None:
    _reject(path.name not in {"", ".", ".."}, "receipt filename is invalid")
    lexical_parent = Path(os.path.abspath(path.parent))
    try:
        resolved_parent = path.parent.resolve(strict=True)
    except OSError as exc:
        raise VerificationRejected("receipt directory must already exist") from exc
    _reject(
        lexical_parent == resolved_parent,
        "receipt directory must not traverse a symlink",
    )
    resolved = resolved_parent / path.name
    _reject(
        not resolved.is_relative_to(root)
        and not resolved.is_relative_to(output)
        and resolved != policy_path,
        "receipt destination must be outside the repository, artifact, and policy",
    )
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        directory = os.open(resolved_parent, directory_flags)
    except OSError as exc:
        raise VerificationRejected("receipt directory cannot be opened safely") from exc
    metadata = os.fstat(directory)
    directory_valid = (
        stat.S_ISDIR(metadata.st_mode)
        and metadata.st_mode & 0o777 == 0o700
        and metadata.st_uid == os.getuid()
    )
    if not directory_valid:
        os.close(directory)
        raise VerificationRejected(
            "receipt directory must be a private owned 0700 directory"
        )
    payload = canonical_json_bytes(dict(receipt))
    temporary_name = f".multitown-receipt-{os.getpid()}-{secrets.token_hex(16)}.partial"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=directory,
        )
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("receipt publication made no write progress")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.link(
            temporary_name,
            path.name,
            src_dir_fd=directory,
            dst_dir_fd=directory,
            follow_symlinks=False,
        )
        os.unlink(temporary_name, dir_fd=directory)
        os.fsync(directory)
        published = os.open(
            path.name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=directory,
        )
        try:
            published_metadata = os.fstat(published)
            chunks = []
            while True:
                chunk = os.read(published, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
        finally:
            os.close(published)
        _reject(
            stat.S_ISREG(published_metadata.st_mode)
            and published_metadata.st_mode & 0o777 == 0o600
            and published_metadata.st_uid == os.getuid()
            and published_metadata.st_nlink == 1
            and b"".join(chunks) == payload,
            "published receipt changed",
        )
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary_name, dir_fd=directory)
        except FileNotFoundError:
            pass
        raise
    finally:
        os.close(directory)


def verify_a24(request: VerificationRequest) -> dict[str, Any]:
    root = request.root.resolve()
    output = request.output.resolve()
    policy_candidate = request.trusted_policy
    try:
        policy_path = policy_candidate.resolve(strict=True)
    except OSError as exc:
        raise VerificationRejected("trusted policy does not exist") from exc
    _reject(request.expect == "smoke", "only explicit smoke verification is supported")
    _reject(
        output.parent == (root / "artifacts").resolve(),
        "artifact must be a direct child of the repository artifacts directory",
    )
    _reject(
        policy_path == Path(os.path.abspath(policy_candidate))
        and not policy_path.is_relative_to(output),
        "trusted policy must be a real file outside the artifact",
    )
    policy_raw = _read_private_bytes(
        policy_path, max_bytes=16 * 1024 * 1024, label="trusted policy"
    )
    try:
        policy = strict_json_loads(policy_raw.decode("utf-8"), label="trusted policy")
    except UnicodeDecodeError as exc:
        raise VerificationRejected("trusted policy is not UTF-8") from exc
    _reject(type(policy) is dict, "trusted policy is not an object")
    _validate_policy(policy)
    policy_digest = _sha_bytes(policy_raw)
    verifier_before, verifier_sources = _verifier_identity(root)
    formal_lock = root / "artifacts/a24-cr-ppo-no-shield-attempt-v1.lock"
    _reject(
        not os.path.lexists(formal_lock),
        "smoke verification requires the global A24 formal lock to be absent",
    )
    try:
        initial = _tree_snapshot(output)
    except ConcurrentObservationError as exc:
        raise VerificationConcurrent(str(exc)) from exc
    except MonitorObservationError as exc:
        raise VerificationRejected(str(exc)) from exc
    portable_entries, portable_digest = _portable_snapshot(initial)
    portable_files = {
        entry["path"]: entry for entry in portable_entries if entry["type"] == "file"
    }
    _reject(
        "artifact-manifest.json" in portable_files,
        "artifact snapshot does not contain a manifest",
    )
    _enforce_staged_resource_limits(None, initial, policy)
    with tempfile.TemporaryDirectory(prefix="multitown-a24-verifier-") as temporary:
        temporary_root = Path(temporary)
        os.chmod(temporary_root, 0o700)
        trusted_staged = temporary_root / "trusted-inputs"
        trusted_staged.mkdir(mode=0o700)
        trusted_before = _verify_trusted_files(root, policy, staging=trusted_staged)
        executable_staged = temporary_root / "trusted-executable"
        executable_staged.mkdir(mode=0o700)
        worker_script = executable_staged / "a24_checkpoint_worker.py"
        worker_descriptor = os.open(
            worker_script, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        with os.fdopen(worker_descriptor, "wb") as handle:
            handle.write(verifier_sources["multitown/a24_checkpoint_worker.py"])
            handle.flush()
            os.fsync(handle.fileno())
        staged = temporary_root / "artifact"
        staged.mkdir(mode=0o700)
        _stage_artifact(output, initial, staged)
        _enforce_staged_resource_limits(staged, initial, policy)
        training, contract, contract_sha = _verify_contracts(root, staged, policy)
        _reject(
            contract_sha
            == canonical_value_sha256(strict_read_json(staged / "run-contract.json")),
            "run-contract digest is unstable",
        )
        manifest = validate_manifest(
            staged,
            folds=training["schedule"]["folds"],
            seeds=training["schedule"]["seeds"],
            gate_open=False,
            expected_source_revision=contract["source"]["revision"],
            expected_run_contract_sha256=contract_sha,
            expected_lock_descriptor=None,
            smoke=True,
        )
        fits, workers = _validate_fit_logs(
            trusted_staged,
            staged,
            training,
            contract["runtime"],
            contract_sha,
            policy,
            worker_script,
            executable_staged,
        )
        calibration_rows, gate = _validate_calibration(
            trusted_staged,
            staged,
            training,
            contract,
            contract_sha,
            fits,
            policy,
        )
        _validate_result(staged, training, contract, contract_sha, gate)
        recomputed = {
            "preflight_signature": contract["preflight_signature"],
            "fit_receipts_sha256": _canonical_digest(fits),
            "checkpoint_summaries_sha256": _canonical_digest(workers),
            "calibration_rows_sha256": sha256_file(
                staged / "calibration-decisions.jsonl"
            ),
            "calibration_gate_sha256": sha256_file(staged / "calibration-gate.json"),
            "outer_rows_sha256": None,
            "statistics_sha256": None,
            "result_sha256": sha256_file(staged / "result.json"),
        }
        try:
            final = _tree_snapshot(output)
        except ConcurrentObservationError as exc:
            raise VerificationConcurrent(str(exc)) from exc
        except MonitorObservationError as exc:
            raise VerificationRejected(str(exc)) from exc
        if final != initial:
            raise VerificationConcurrent("artifact changed during verification")
        trusted_after = _verify_trusted_files(root, policy)
        if trusted_after != trusted_before:
            raise VerificationConcurrent("trusted inputs changed during verification")
        policy_after = _read_private_bytes(
            policy_path, max_bytes=16 * 1024 * 1024, label="trusted policy"
        )
        if policy_after != policy_raw:
            raise VerificationConcurrent("trusted policy changed during verification")
        verifier_after, _ = _verifier_identity(root)
        if not same_typed_json(verifier_after, verifier_before):
            raise VerificationConcurrent(
                "verifier identity changed during verification"
            )
        _reject(
            not os.path.lexists(formal_lock),
            "A24 formal lock appeared during smoke verification",
        )
        schedule = training["schedule"]
        core = {
            "target": {
                "expect": "smoke",
                "profile": PROFILE,
                "evidence_level": EVIDENCE_LEVEL,
                "artifact_snapshot_sha256": portable_digest,
                "manifest_sha256": portable_files["artifact-manifest.json"]["sha256"],
                "manifest_terminal_state": manifest["terminal_state"],
                "source_revision": contract["source"]["revision"],
                "source_set_sha256": contract["source"]["source_set_sha256"],
                "protocol_sha256": contract["protocol_sha256"],
                "run_contract_sha256": contract_sha,
                "formal_lock_observed_absent": True,
            },
            "trusted_policy": {
                "policy_id": policy["policy_id"],
                "policy_sha256": policy_digest,
                "trusted_files_sha256": _canonical_digest(trusted_before),
            },
            "verifier": verifier_before,
            "recomputed": recomputed,
            "counts": {
                "files_and_directories": len(portable_entries),
                "fits": len(fits),
                "optimizer_updates": len(fits) * schedule["updates"],
                "training_episode_draws": len(fits)
                * schedule["updates"]
                * schedule["episodes_per_update"],
                "calibration_rows": len(calibration_rows),
                "outer_rows": 0,
            },
            "decision": {
                "inventory_valid": True,
                "artifact_semantic_recomputation_valid": True,
                "checkpoint_weights_only_reconstruction": True,
                "fit_log_chain_valid": True,
                "calibration_gate_recomputed": True,
                "result_recomputed": True,
                "execution_transition_derivation_replayed": False,
                "inference_reexecution": False,
                "optimizer_update_reexecution": False,
                "results_reproduced": False,
                "formal_terminal_valid": False,
                "calibration_evidence_evaluable": False,
                "outer_performance_evaluable": False,
                "formal_evidence_accepted": False,
            },
            "limitations": [
                "non-evidentiary smoke",
                "artifact semantics verification, not independent implementation",
                "not an independent experimental reproduction or replication",
                "no inference, environment, or optimizer-update reexecution",
                "not hidden-test or OOD evidence",
                "not an LLM-weight RL result",
                "not a formal safety or state-of-the-art claim",
            ],
        }
        receipt = _receipt(core)
        validate_receipt(receipt)
    if request.receipt_out is not None:
        _publish_receipt(
            request.receipt_out,
            receipt,
            root=root,
            output=output,
            policy_path=policy_path,
        )
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify", help="verify one sealed smoke artifact")
    verify.add_argument("--root", required=True, type=Path)
    verify.add_argument("--artifact", required=True, type=Path)
    verify.add_argument("--expect", required=True, choices=("smoke",))
    verify.add_argument("--trusted-input-snapshot", required=True, type=Path)
    verify.add_argument("--receipt-out", type=Path)
    receipt = subparsers.add_parser(
        "validate-receipt", help="validate a published receipt"
    )
    receipt.add_argument("--receipt", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate-receipt":
            value = _read_bounded_json(
                args.receipt,
                limit=4 * 1024 * 1024,
                label="verifier receipt",
            )
            validate_receipt(value)
            sys.stdout.buffer.write(
                canonical_json_bytes(
                    {
                        "receipt_integrity_valid": True,
                        "unsigned_self_reported_receipt": True,
                        "verification_id": value["verification_id"],
                    }
                )
            )
            return 0
        receipt = verify_a24(
            VerificationRequest(
                root=args.root,
                output=args.artifact,
                expect=args.expect,
                trusted_policy=args.trusted_input_snapshot,
                receipt_out=args.receipt_out,
            )
        )
        sys.stdout.buffer.write(canonical_json_bytes(receipt))
        return 0
    except VerificationConcurrent as exc:
        sys.stderr.write(f"CONCURRENT: {exc}\n")
        return 3
    except (
        VerificationRejected,
        UnsupportedArtifact,
        RuntimeError,
        ValueError,
        OSError,
        subprocess.SubprocessError,
    ) as exc:
        sys.stderr.write(f"REJECTED: {exc}\n")
        return 2
    except Exception as exc:  # noqa: BLE001 - CLI maps unexpected failures to 4.
        sys.stderr.write(f"INTERNAL: {type(exc).__name__}: {exc}\n")
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
