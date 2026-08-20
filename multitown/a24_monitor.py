"""Read-only, standard-library-only state monitor for A24 artifacts.

The monitor deliberately has no recovery path.  In particular, it never
creates the formal attempt lock, mutates an output, or infers process liveness
from the presence or age of ``RUNNING.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .a24_artifact_state import (
    SUCCESS_TO_INVALID,
    canonical_value_sha256,
)
from .a24_contract import (
    ALL_FITS_VERSION,
    CALIBRATION_GATE_VERSION,
    FIT_COMPLETE_VERSION,
    FORMAL_FOLDS,
    FORMAL_LOCK,
    FORMAL_SEEDS,
    INVALIDATED_VERSION,
    LOCK_VERSION,
    MANIFEST_VERSION,
    MECHANISM,
    RESULT_VERSION,
    RUN_CONTRACT_VERSION,
    RUNNER_VERSION,
    TRAINING_CONTRACT_VERSION,
    OutcomeState,
    canonical_json_bytes,
    expected_directories,
    expected_managed_paths,
    fit_prefix,
    same_typed_json,
    strict_json_loads,
)

MONITOR_VERSION = "multitown-a24-read-only-monitor-v1"
PROGRESS_VERSION = "multitown-a24-fit-progress-v1"
FORMAL_MODE = "adaptive-same-bank-development"
SMOKE_MODE = "non-evidentiary-smoke"
FORMAL_UPDATES = 120
FORMAL_EPISODES_PER_UPDATE = 48
FORMAL_CALIBRATION_EPISODES_PER_FOLD = 600
FORMAL_OUTER_EPISODES_PER_FOLD = 600
FORMAL_BOOTSTRAP_ITERATIONS = 20_000
FORMAL_THREADS = 8
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_MAX_JSON_BYTES = 16 * 1024 * 1024
_MAX_ARTIFACT_BYTES = 512 * 1024 * 1024
_MAX_TREE_BYTES = 1024 * 1024 * 1024
_MAX_TREE_ENTRIES = 256
_MAX_TREE_DEPTH = 8
_MAX_PENDING_LOCKS = 32
_MAX_ARTIFACT_ROOT_ENTRIES = 4096


class MonitorObservationError(RuntimeError):
    """The observed namespace cannot be classified safely."""


class ConcurrentObservationError(MonitorObservationError):
    """A replaceable artifact changed during the read-only observation."""


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _require_directory(path: Path, *, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise MonitorObservationError(f"{label} is unreadable: {path}") from exc
    if (
        path.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_mode & 0o777 != 0o700
        or metadata.st_uid != os.getuid()
    ):
        raise MonitorObservationError(
            f"{label} is not a private owned 0700 real directory: {path}"
        )
    return metadata


def _open_private_file(
    path: Path, *, label: str, max_bytes: int
) -> tuple[int, os.stat_result]:
    try:
        initial = path.lstat()
    except OSError as exc:
        raise MonitorObservationError(f"{label} is unreadable: {path}") from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(initial.st_mode)
        or initial.st_mode & 0o777 != 0o600
        or initial.st_uid != os.getuid()
        or initial.st_nlink != 1
        or initial.st_size > max_bytes
    ):
        raise MonitorObservationError(
            f"{label} is not a bounded private owned single-link regular file: {path}"
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
        raise MonitorObservationError(f"{label} is unreadable: {path}") from exc
    try:
        before = os.fstat(descriptor)
    except BaseException:
        os.close(descriptor)
        raise
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_uid,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    initial_identity = (
        initial.st_dev,
        initial.st_ino,
        initial.st_mode,
        initial.st_uid,
        initial.st_nlink,
        initial.st_size,
        initial.st_mtime_ns,
        initial.st_ctime_ns,
    )
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_mode & 0o777 != 0o600
        or before.st_uid != os.getuid()
        or before.st_nlink != 1
        or before.st_size > max_bytes
        or identity != initial_identity
    ):
        os.close(descriptor)
        raise ConcurrentObservationError(
            f"{label} changed before open completed: {path}"
        )
    return descriptor, before


def _finish_private_read(
    descriptor: int,
    path: Path,
    before: os.stat_result,
    *,
    label: str,
) -> None:
    after = os.fstat(descriptor)
    try:
        current = path.lstat()
    except OSError as exc:
        raise ConcurrentObservationError(f"{label} changed while read: {path}") from exc
    fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_uid",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(
        getattr(before, field) != getattr(after, field)
        or getattr(before, field) != getattr(current, field)
        for field in fields
    ):
        raise ConcurrentObservationError(f"{label} changed while read: {path}")


def _read_private_file(path: Path, *, label: str) -> tuple[bytes, os.stat_result]:
    descriptor, before = _open_private_file(
        path, label=label, max_bytes=_MAX_JSON_BYTES
    )
    try:
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        _finish_private_read(descriptor, path, before, label=label)
    finally:
        os.close(descriptor)
    return b"".join(chunks), before


def _observe_private_file(
    path: Path, *, label: str, max_bytes: int = _MAX_ARTIFACT_BYTES
) -> tuple[os.stat_result, int, str, int, bytes]:
    descriptor, before = _open_private_file(path, label=label, max_bytes=max_bytes)
    digest = hashlib.sha256()
    size = 0
    lines = 0
    last = b""
    try:
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
            lines += chunk.count(b"\n")
            last = chunk[-1:]
        _finish_private_read(descriptor, path, before, label=label)
    finally:
        os.close(descriptor)
    return before, size, digest.hexdigest(), lines, last


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    raw, _ = _read_private_file(path, label=label)
    try:
        value = strict_json_loads(raw.decode("utf-8"), label=str(path))
    except UnicodeDecodeError as exc:
        raise MonitorObservationError(f"{label} is not UTF-8 JSON: {path}") from exc
    if type(value) is not dict:
        raise MonitorObservationError(f"{label} is not a JSON object: {path}")
    return value


def _tree_snapshot(output: Path) -> tuple[tuple[Any, ...], ...]:
    """Return a stable, byte-bound read-only snapshot of an output tree."""

    _require_directory(output, label="A24 output")
    entries: list[tuple[Any, ...]] = []
    pending = [output]
    total_bytes = 0
    discovered_entries = 1
    while pending:
        directory_path = pending.pop()
        depth = len(directory_path.relative_to(output).parts)
        if depth > _MAX_TREE_DEPTH:
            raise MonitorObservationError("A24 tree exceeds the depth limit")
        directory_meta = _require_directory(directory_path, label="A24 directory")
        relative_directory = directory_path.relative_to(output).as_posix()
        entries.append(
            (
                "directory",
                "." if relative_directory == "." else relative_directory,
                directory_meta.st_mode & 0o777,
                directory_meta.st_uid,
                directory_meta.st_dev,
                directory_meta.st_ino,
                directory_meta.st_mtime_ns,
                directory_meta.st_ctime_ns,
            )
        )
        if len(entries) > _MAX_TREE_ENTRIES:
            raise MonitorObservationError("A24 tree exceeds the entry limit")
        try:
            children = os.scandir(directory_path)
        except OSError as exc:
            raise ConcurrentObservationError(
                f"A24 directory changed during traversal: {directory_path}"
            ) from exc
        with children:
            for child in children:
                discovered_entries += 1
                if discovered_entries > _MAX_TREE_ENTRIES:
                    raise MonitorObservationError("A24 tree exceeds the entry limit")
                path = directory_path / child.name
                try:
                    metadata = child.stat(follow_symlinks=False)
                except OSError as exc:
                    raise ConcurrentObservationError(
                        f"A24 path changed during traversal: {path}"
                    ) from exc
                if child.is_symlink():
                    raise MonitorObservationError(
                        f"A24 tree contains a symlink: {path}"
                    )
                if stat.S_ISDIR(metadata.st_mode):
                    pending.append(path)
                    continue
                if not stat.S_ISREG(metadata.st_mode):
                    raise MonitorObservationError(
                        f"A24 tree contains a special file: {path}"
                    )
                remaining = _MAX_TREE_BYTES - total_bytes
                if metadata.st_size > min(_MAX_ARTIFACT_BYTES, remaining):
                    raise MonitorObservationError(
                        "A24 tree exceeds a file or aggregate byte limit"
                    )
                metadata, size, digest, _, _ = _observe_private_file(
                    path,
                    label="A24 artifact",
                    max_bytes=min(_MAX_ARTIFACT_BYTES, remaining),
                )
                total_bytes += size
                if total_bytes > _MAX_TREE_BYTES:
                    raise MonitorObservationError(
                        "A24 tree exceeds the aggregate byte limit"
                    )
                entries.append(
                    (
                        "file",
                        path.relative_to(output).as_posix(),
                        metadata.st_mode & 0o777,
                        metadata.st_uid,
                        metadata.st_nlink,
                        metadata.st_dev,
                        metadata.st_ino,
                        metadata.st_mtime_ns,
                        metadata.st_ctime_ns,
                        size,
                        digest,
                    )
                )
                if len(entries) > _MAX_TREE_ENTRIES:
                    raise MonitorObservationError("A24 tree exceeds the entry limit")
    return tuple(sorted(entries))


def _snapshot_digest(snapshot: tuple[tuple[Any, ...], ...]) -> str:
    return hashlib.sha256(canonical_json_bytes(snapshot)).hexdigest()


def _pending_locks(artifacts: Path, lock: Path) -> list[str]:
    prefix = f".{lock.name}.pending-"
    matches: list[str] = []
    observed = 0
    with os.scandir(artifacts) as children:
        for child in children:
            observed += 1
            if observed > _MAX_ARTIFACT_ROOT_ENTRIES:
                raise MonitorObservationError(
                    "A24 artifacts root exceeds the monitor entry limit"
                )
            if child.name.startswith(prefix):
                matches.append(child.name)
                if len(matches) > _MAX_PENDING_LOCKS:
                    raise MonitorObservationError(
                        "A24 pending-lock namespace exceeds the monitor limit"
                    )
    return sorted(matches)


def _directory_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _formal_schedule() -> dict[str, Any]:
    return {
        "mode": FORMAL_MODE,
        "seeds": list(FORMAL_SEEDS),
        "folds": list(FORMAL_FOLDS),
        "updates": FORMAL_UPDATES,
        "episodes_per_update": FORMAL_EPISODES_PER_UPDATE,
        "calibration_episodes_per_fold": FORMAL_CALIBRATION_EPISODES_PER_FOLD,
        "outer_episodes_per_fold": FORMAL_OUTER_EPISODES_PER_FOLD,
        "bootstrap_iterations": FORMAL_BOOTSTRAP_ITERATIONS,
        "threads": FORMAL_THREADS,
    }


def _smoke_schedule() -> dict[str, Any]:
    schedule = _formal_schedule()
    schedule.update(
        {
            "mode": SMOKE_MODE,
            "seeds": [FORMAL_SEEDS[0]],
            "updates": 1,
            "episodes_per_update": 4,
            "calibration_episodes_per_fold": 8,
            "outer_episodes_per_fold": 8,
            "bootstrap_iterations": 200,
            "threads": 2,
        }
    )
    return schedule


def _schedule_from_training_contract(
    output: Path,
    *,
    expect: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = _read_json(
        output / "training-contract.json", label="A24 training contract"
    )
    expected_keys = {
        "schema_version",
        "protocol_revision",
        "runner_version",
        "policy_version",
        "pq1_primitives_version",
        "mechanism",
        "shield_enabled",
        "schedule",
        "formal_schedule",
        "ppo",
        "partitions",
        "formal_products_if_gate_open",
        "current_schedule_products_if_gate_open",
        "comparator_ledger_sha256",
        "training_contract_is_canonical_protocol",
        "formal_execution_authorized",
        "non_evidentiary_smoke",
    }
    if (
        set(contract) != expected_keys
        or contract.get("schema_version") != TRAINING_CONTRACT_VERSION
    ):
        raise MonitorObservationError("A24 training-contract schema changed")
    schedule = contract.get("schedule")
    if type(schedule) is not dict or set(schedule) != set(_formal_schedule()):
        raise MonitorObservationError("A24 training schedule schema changed")
    expected_schedule = _formal_schedule() if expect == "formal" else _smoke_schedule()
    formal = expect == "formal"
    if (
        not same_typed_json(schedule, expected_schedule)
        or not same_typed_json(contract.get("formal_schedule"), _formal_schedule())
        or not same_typed_json(
            contract.get("formal_products_if_gate_open"),
            _expected_products(_formal_schedule(), gate_open=True),
        )
        or not same_typed_json(
            contract.get("current_schedule_products_if_gate_open"),
            _expected_products(expected_schedule, gate_open=True),
        )
        or contract.get("formal_execution_authorized") is not formal
        or contract.get("non_evidentiary_smoke") is formal
        or contract.get("runner_version") != RUNNER_VERSION
        or contract.get("training_contract_is_canonical_protocol") is not True
        or type(contract.get("comparator_ledger_sha256")) is not str
        or not _HEX64.fullmatch(contract["comparator_ledger_sha256"])
    ):
        raise MonitorObservationError(f"A24 {expect} training schedule changed")
    if (
        contract.get("mechanism") != MECHANISM
        or contract.get("shield_enabled") is not False
    ):
        raise MonitorObservationError("A24 mechanism/shield contract changed")
    return schedule, contract


def _validate_run_contract(
    output: Path,
    *,
    expect: str,
    descriptor: Mapping[str, Any] | None,
    training_contract: Mapping[str, Any],
) -> dict[str, Any]:
    contract = _read_json(output / "run-contract.json", label="A24 run contract")
    expected_keys = {
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
    if (
        set(contract) != expected_keys
        or contract.get("schema_version") != RUN_CONTRACT_VERSION
        or contract.get("mode") != (FORMAL_MODE if expect == "formal" else SMOKE_MODE)
    ):
        raise MonitorObservationError("A24 run-contract version changed")
    source = contract.get("source")
    if (
        type(source) is not dict
        or type(source.get("revision")) is not str
        or not _HEX40.fullmatch(source["revision"])
        or type(source.get("source_set_sha256")) is not str
        or not _HEX64.fullmatch(source["source_set_sha256"])
    ):
        raise MonitorObservationError("A24 run-contract source binding is malformed")
    protocol_sha = canonical_value_sha256(training_contract)
    if (
        contract.get("training_contract_sha256") != protocol_sha
        or contract.get("protocol_sha256") != protocol_sha
    ):
        raise MonitorObservationError("A24 training/run contract digest changed")
    if expect == "smoke":
        if contract.get("formal_lock") is not None:
            raise MonitorObservationError(
                "A24 smoke run contract carries a formal lock"
            )
        return contract
    if descriptor is None:
        raise MonitorObservationError("A24 formal run contract lacks a lock descriptor")
    expected_projection = {
        "schema_version": descriptor["schema_version"],
        "attempt": descriptor["attempt"],
        "path": str((output.parents[1] / FORMAL_LOCK).resolve()),
        "output": descriptor["output"],
        "source_revision": descriptor["source_revision"],
        "protocol_sha256": descriptor["protocol_sha256"],
        "source_set_sha256": descriptor["source_set_sha256"],
    }
    if (
        descriptor["run_contract_sha256"] != canonical_value_sha256(contract)
        or descriptor["protocol_sha256"] != protocol_sha
        or source.get("revision") != descriptor["source_revision"]
        or source.get("source_set_sha256") != descriptor["source_set_sha256"]
        or not same_typed_json(contract.get("formal_lock"), expected_projection)
    ):
        raise MonitorObservationError("A24 formal run-contract lock binding changed")
    return contract


def _validate_lock(
    lock: Path, output: Path
) -> tuple[dict[str, Any], tuple[os.stat_result, int, str, int, bytes]]:
    raw, metadata = _read_private_file(lock, label="A24 permanent lock")
    try:
        descriptor = strict_json_loads(raw.decode("utf-8"), label=str(lock))
    except UnicodeDecodeError as exc:
        raise MonitorObservationError("A24 permanent lock is not UTF-8") from exc
    expected_keys = {
        "schema_version",
        "attempt",
        "output",
        "source_revision",
        "run_contract_sha256",
        "protocol_sha256",
        "source_set_sha256",
    }
    if (
        type(descriptor) is not dict
        or set(descriptor) != expected_keys
        or descriptor.get("schema_version") != LOCK_VERSION
        or type(descriptor.get("attempt")) is not int
        or descriptor["attempt"] != 1
        or descriptor.get("output") != str(output)
        or type(descriptor.get("source_revision")) is not str
        or not _HEX40.fullmatch(descriptor["source_revision"])
        or any(
            type(descriptor.get(field)) is not str
            or not _HEX64.fullmatch(descriptor[field])
            for field in (
                "run_contract_sha256",
                "protocol_sha256",
                "source_set_sha256",
            )
        )
        or raw != canonical_json_bytes(descriptor)
    ):
        raise MonitorObservationError("A24 permanent lock descriptor is malformed")
    observation = (
        metadata,
        len(raw),
        hashlib.sha256(raw).hexdigest(),
        raw.count(b"\n"),
        raw[-1:] if raw else b"",
    )
    return descriptor, observation


def _validate_running(output: Path, *, expect: str) -> dict[str, Any]:
    running = _read_json(output / "RUNNING.json", label="A24 running receipt")
    expected_keys = {
        "schema_version",
        "mode",
        "formal_lock_acquired",
        "calibration_started",
        "outer_evaluation_started",
    }
    if (
        set(running) != expected_keys
        or running.get("schema_version") != RUNNER_VERSION
        or running.get("mode") != (FORMAL_MODE if expect == "formal" else SMOKE_MODE)
        or running.get("formal_lock_acquired") is not (expect == "formal")
        or type(running.get("calibration_started")) is not bool
        or type(running.get("outer_evaluation_started")) is not bool
        or (running["outer_evaluation_started"] and not running["calibration_started"])
    ):
        raise MonitorObservationError("A24 RUNNING receipt is malformed")
    return running


def _expected_products(
    schedule: Mapping[str, Any], *, gate_open: bool
) -> dict[str, int]:
    fits = len(schedule["folds"]) * len(schedule["seeds"])
    return {
        "fits": fits,
        "optimizer_updates": fits * schedule["updates"],
        "training_episode_draws": fits
        * schedule["updates"]
        * schedule["episodes_per_update"],
        "a24_calibration_rows": fits * schedule["calibration_episodes_per_fold"],
        "a22_lagrangian_calibration_source_rows": fits
        * schedule["calibration_episodes_per_fold"],
        "a24_outer_rows": fits * schedule["outer_episodes_per_fold"]
        if gate_open
        else 0,
        "a22_lagrangian_outer_rows": fits * schedule["outer_episodes_per_fold"]
        if gate_open
        else 0,
        "a8_outer_source_rows": len(schedule["folds"])
        * schedule["outer_episodes_per_fold"]
        if gate_open
        else 0,
        "manifest_entries": 9 + fits * 4 if gate_open else 6 + fits * 4,
    }


def _progress_report(output: Path, schedule: Mapping[str, Any]) -> dict[str, Any]:
    updates = 0
    completion_markers = 0
    cells_seen = 0
    mode_counts = {"reward": 0, "unsafe": 0, "wrong": 0}
    issues: list[str] = []
    for fold in schedule["folds"]:
        for seed in schedule["seeds"]:
            fit_dir = output / fit_prefix(fold, seed)
            progress_path = fit_dir / "progress.json"
            complete_path = fit_dir / "fit-complete.json"
            if not _lexists(progress_path) and not _lexists(complete_path):
                continue
            cells_seen += 1
            try:
                progress = _read_json(progress_path, label="A24 fit progress")
                expected_progress_keys = {
                    "schema_version",
                    "outer_fold",
                    "training_seed",
                    "mechanism",
                    "current_update",
                    "scheduled_updates",
                    "mode_counts",
                    "calibration_started",
                    "outer_evaluation_started",
                }
                counts = progress.get("mode_counts")
                if (
                    set(progress) != expected_progress_keys
                    or progress.get("schema_version") != PROGRESS_VERSION
                    or type(progress.get("outer_fold")) is not int
                    or progress["outer_fold"] != fold
                    or type(progress.get("training_seed")) is not int
                    or progress["training_seed"] != seed
                    or progress.get("mechanism") != MECHANISM
                    or type(progress.get("current_update")) is not int
                    or not 1 <= progress["current_update"] <= schedule["updates"]
                    or type(progress.get("scheduled_updates")) is not int
                    or progress["scheduled_updates"] != schedule["updates"]
                    or type(counts) is not dict
                    or set(counts) != set(mode_counts)
                    or any(
                        type(value) is not int or value < 0 for value in counts.values()
                    )
                    or sum(counts.values()) != progress["current_update"]
                    or progress.get("calibration_started") is not False
                    or progress.get("outer_evaluation_started") is not False
                ):
                    raise MonitorObservationError("A24 fit progress schema changed")
                updates += progress["current_update"]
                for name in mode_counts:
                    mode_counts[name] += counts[name]
                if _lexists(complete_path):
                    complete = _read_json(
                        complete_path, label="A24 fit-complete receipt"
                    )
                    if (
                        complete.get("schema_version") != FIT_COMPLETE_VERSION
                        or type(complete.get("outer_fold")) is not int
                        or complete["outer_fold"] != fold
                        or type(complete.get("training_seed")) is not int
                        or complete["training_seed"] != seed
                        or complete.get("mechanism") != MECHANISM
                        or complete.get("shield_enabled") is not False
                        or type(complete.get("final_update")) is not int
                        or complete["final_update"] != schedule["updates"]
                        or type(complete.get("training_episode_draws")) is not int
                        or complete["training_episode_draws"]
                        != schedule["updates"] * schedule["episodes_per_update"]
                        or progress["current_update"] != schedule["updates"]
                    ):
                        raise MonitorObservationError(
                            "A24 fit-complete receipt changed"
                        )
                    completion_markers += 1
            except ConcurrentObservationError:
                raise
            except (MonitorObservationError, RuntimeError, OSError) as exc:
                issues.append(f"fold={fold},seed={seed}: {exc}")
    expected_fits = len(schedule["folds"]) * len(schedule["seeds"])
    scheduled_updates = expected_fits * schedule["updates"]
    return {
        "expected_fits": expected_fits,
        "fit_cells_seen": cells_seen,
        "completion_markers_seen": completion_markers,
        "persisted_optimizer_updates_lower_bound": updates,
        "scheduled_optimizer_updates": scheduled_updates,
        "persisted_episode_draws_lower_bound": updates
        * schedule["episodes_per_update"],
        "scheduled_episode_draws": scheduled_updates * schedule["episodes_per_update"],
        "persisted_actor_mode_counts": mode_counts,
        "calibration_rows_published": _line_count_if_present(
            output / "calibration-decisions.jsonl"
        ),
        "a24_outer_rows_published": _line_count_if_present(
            output / "a24-outer-decisions.jsonl"
        ),
        "a22_outer_rows_published": _line_count_if_present(
            output / "a22-lagrangian-outer-decisions.jsonl"
        ),
        "issues": issues,
    }


def _line_count_if_present(path: Path) -> int | None:
    if not _lexists(path):
        return None
    _, size, _, lines, last = _observe_private_file(path, label="A24 row artifact")
    if size and last != b"\n":
        raise MonitorObservationError(f"A24 JSONL lacks final newline: {path}")
    return lines


def _validate_all_fits(output: Path, expected: Mapping[str, int]) -> None:
    receipt = _read_json(
        output / "all-fits-complete.json", label="A24 all-fits receipt"
    )
    if (
        receipt.get("schema_version") != ALL_FITS_VERSION
        or type(receipt.get("fits")) is not list
        or len(receipt["fits"]) != expected["fits"]
        or receipt.get("expected_fits") != expected["fits"]
        or receipt.get("expected_optimizer_updates") != expected["optimizer_updates"]
        or receipt.get("expected_training_episode_draws")
        != expected["training_episode_draws"]
        or receipt.get("all_reached_final_update") is not True
        or receipt.get("exact_fit_key_product_verified") is not True
        or receipt.get("pq1_provenance_and_checkpoint_chain_verified") is not True
        or receipt.get("calibration_started_at_receipt") is not False
        or receipt.get("outer_evaluation_started_at_receipt") is not False
    ):
        raise MonitorObservationError("A24 all-fits receipt changed")


def _validate_critical_receipt_summary(
    output: Path,
    *,
    expect: str,
    gate_open: bool,
    schedule: Mapping[str, Any],
    descriptor: Mapping[str, Any] | None,
) -> dict[str, Any]:
    expected = _expected_products(schedule, gate_open=gate_open)
    progress = _progress_report(output, schedule)
    if (
        progress["issues"]
        or progress["completion_markers_seen"] != expected["fits"]
        or progress["persisted_optimizer_updates_lower_bound"]
        != expected["optimizer_updates"]
        or progress["calibration_rows_published"] != expected["a24_calibration_rows"]
        or progress["a24_outer_rows_published"]
        != (expected["a24_outer_rows"] if gate_open else None)
        or progress["a22_outer_rows_published"]
        != (expected["a22_lagrangian_outer_rows"] if gate_open else None)
    ):
        raise MonitorObservationError("A24 terminal fit receipts are incomplete")
    _validate_all_fits(output, expected)
    gate = _read_json(output / "calibration-gate.json", label="A24 calibration gate")
    decision = gate.get("decision")
    run_contract = _read_json(output / "run-contract.json", label="A24 run contract")
    run_digest = canonical_value_sha256(run_contract)
    protocol_sha = run_contract.get("protocol_sha256")
    _, _, calibration_sha, _, _ = _observe_private_file(
        output / "calibration-decisions.jsonl",
        label="A24 calibration decisions",
    )
    expected_gate_keys = {
        "schema_version",
        "run_contract_sha256",
        "comparator_ledger_sha256",
        "calibration_sha256",
        "a24_calibration_rows",
        "a22_lagrangian_descriptive_rows",
        "a22_lagrangian_enters_gate",
        "a22_lagrangian_calibration_diagnostic",
        "decision",
        "smoke_forced_outer",
    }
    if (
        set(gate) != expected_gate_keys
        or gate.get("schema_version") != CALIBRATION_GATE_VERSION
        or gate.get("run_contract_sha256") != run_digest
        or type(gate.get("comparator_ledger_sha256")) is not str
        or not _HEX64.fullmatch(gate["comparator_ledger_sha256"])
        or gate.get("calibration_sha256") != calibration_sha
        or type(decision) is not dict
        or type(decision.get("raw_conjunction")) is not bool
        or type(decision.get("outer_gate_permitted")) is not bool
        or type(gate.get("a22_lagrangian_calibration_diagnostic")) is not dict
        or type(gate.get("smoke_forced_outer")) is not bool
        or gate.get("a24_calibration_rows") != expected["a24_calibration_rows"]
        or gate.get("a22_lagrangian_descriptive_rows")
        != expected["a22_lagrangian_calibration_source_rows"]
        or gate.get("a22_lagrangian_enters_gate") is not False
        or (expect == "formal" and decision["outer_gate_permitted"] is not gate_open)
    ):
        raise MonitorObservationError("A24 calibration-gate semantics changed")
    result = _read_json(output / "result.json", label="A24 result")
    formal = expect == "formal"
    source = run_contract.get("source")
    source_revision = source.get("revision") if type(source) is dict else None
    expected_result_keys = {
        "schema_version",
        "mode",
        "non_evidentiary_smoke",
        "formal_execution_authorized",
        "formal_lock_acquired",
        "source_revision",
        "run_contract_sha256",
        "protocol_sha256",
        "comparator_ledger_sha256",
        "calibration_raw_conjunction",
        "formal_calibration_gate_evaluable",
        "formal_outer_gate_open",
        "smoke_outer_path_exercised",
        "products",
        "statistics",
        "claim_boundary",
        "validation",
    }
    expected_claims = {
        "adaptive_same_bank_development": formal,
        "controller_level_agentic_rl_experiment": formal,
        "outer_performance_evaluable": formal and gate_open,
        "independent_confirmation": False,
        "hidden_test_or_ood": False,
        "crpo_reproduction": False,
        "crpo_guarantees": False,
        "formal_safety": False,
        "llm_weight_rl": False,
        "state_of_the_art": False,
    }
    expected_validation = {
        "pq1_functions_executed_every_update": True,
        "exact_fit_log_checkpoint_chain": True,
        "a22_calibration_enters_gate": False,
        "a8_rows_logically_paired_not_replicated": True,
        "formal_lock_untouched": not formal,
        "formal_lock_verified": formal,
    }
    if (
        set(result) != expected_result_keys
        or result.get("schema_version") != RESULT_VERSION
        or result.get("mode") != schedule["mode"]
        or result.get("non_evidentiary_smoke") is formal
        or result.get("formal_execution_authorized") is not formal
        or result.get("formal_lock_acquired") is not formal
        or result.get("source_revision") != source_revision
        or result.get("run_contract_sha256") != run_digest
        or result.get("protocol_sha256") != protocol_sha
        or result.get("comparator_ledger_sha256") != gate["comparator_ledger_sha256"]
        or result.get("calibration_raw_conjunction") is not decision["raw_conjunction"]
        or result.get("formal_calibration_gate_evaluable") is not formal
        or result.get("formal_outer_gate_open") is not (formal and gate_open)
        or result.get("smoke_outer_path_exercised") is not ((not formal) and gate_open)
        or not same_typed_json(result.get("products"), expected)
        or (gate_open and result.get("statistics") is None)
        or (not gate_open and result.get("statistics") is not None)
        or not same_typed_json(result.get("claim_boundary"), expected_claims)
        or not same_typed_json(result.get("validation"), expected_validation)
    ):
        raise MonitorObservationError("A24 terminal result semantics changed")
    return progress


def _validate_manifest_snapshot(
    output: Path,
    *,
    snapshot: tuple[tuple[Any, ...], ...],
    schedule: Mapping[str, Any],
    gate_open: bool,
    expect: str,
    descriptor: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate manifest bytes against the already streamed tree snapshot."""

    manifest = _read_json(
        output / "artifact-manifest.json", label="A24 artifact manifest"
    )
    expected_manifest_keys = {
        "schema_version",
        "source_revision",
        "run_contract_sha256",
        "terminal_state",
        "files",
    }
    expected_files = expected_managed_paths(
        tuple(schedule["folds"]),
        tuple(schedule["seeds"]),
        gate_open=gate_open,
    )
    expected_dirs = expected_directories(
        tuple(schedule["folds"]),
        tuple(schedule["seeds"]),
        gate_open=gate_open,
    )
    file_entries = {entry[1]: entry for entry in snapshot if entry[0] == "file"}
    directories = {entry[1] for entry in snapshot if entry[0] == "directory"}
    run_contract = _read_json(output / "run-contract.json", label="A24 run contract")
    run_digest = canonical_value_sha256(run_contract)
    source = run_contract.get("source")
    source_revision = source.get("revision") if type(source) is dict else None
    expected_terminal = (
        OutcomeState.NON_EVIDENTIARY_SMOKE.value
        if expect == "smoke"
        else (
            OutcomeState.VALID_GATE_OPEN_SUCCESS.value
            if gate_open
            else OutcomeState.VALID_CALIBRATION_NEGATIVE.value
        )
    )
    if (
        set(manifest) != expected_manifest_keys
        or manifest.get("schema_version") != MANIFEST_VERSION
        or manifest.get("source_revision") != source_revision
        or manifest.get("run_contract_sha256") != run_digest
        or manifest.get("terminal_state") != expected_terminal
        or type(manifest.get("files")) is not dict
        or set(manifest["files"]) != expected_files
        or set(file_entries) != expected_files | {"artifact-manifest.json"}
        or directories != expected_dirs
        or (
            descriptor is not None
            and (
                descriptor["source_revision"] != source_revision
                or descriptor["run_contract_sha256"] != run_digest
            )
        )
    ):
        raise MonitorObservationError("A24 exact manifest inventory changed")
    for name, receipt in manifest["files"].items():
        entry = file_entries[name]
        if (
            type(receipt) is not dict
            or set(receipt) != {"bytes", "sha256"}
            or type(receipt.get("bytes")) is not int
            or receipt["bytes"] < 0
            or type(receipt.get("sha256")) is not str
            or not _HEX64.fullmatch(receipt["sha256"])
            or receipt["bytes"] != entry[9]
            or receipt["sha256"] != entry[10]
        ):
            raise MonitorObservationError(f"A24 manifest receipt changed: {name}")
    return manifest


def _validate_invalidation_snapshot(
    output: Path,
    *,
    snapshot: tuple[tuple[Any, ...], ...],
    descriptor: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _read_json(output / "INVALIDATED.json", label="A24 invalidation")
    expected_keys = {
        "schema_version",
        "invalidated",
        "reason",
        "error_type",
        "error",
        "formal_lock_acquired",
        "selective_retry_forbidden",
        "performance_evaluable",
        "orphan_inventory",
        "failed_at_utc",
    }
    if (
        set(payload) != expected_keys
        or payload.get("schema_version") != INVALIDATED_VERSION
        or payload.get("invalidated") is not True
        or type(payload.get("reason")) is not str
        or type(payload.get("error_type")) is not str
        or type(payload.get("error")) is not str
        or payload.get("formal_lock_acquired") is not True
        or payload.get("selective_retry_forbidden") is not True
        or payload.get("performance_evaluable") is not False
        or type(payload.get("failed_at_utc")) is not str
        or descriptor["output"] != str(output)
    ):
        raise MonitorObservationError("A24 invalidation schema changed")
    inventory = payload.get("orphan_inventory")
    if type(inventory) is not dict or set(inventory) != {
        "files",
        "directories",
        "quarantined",
    }:
        raise MonitorObservationError("A24 invalidation inventory schema changed")
    files = inventory["files"]
    directories = inventory["directories"]
    quarantined = inventory["quarantined"]
    if (
        type(files) is not dict
        or type(directories) is not list
        or any(type(name) is not str for name in directories)
        or directories != sorted(set(directories))
        or type(quarantined) is not dict
    ):
        raise MonitorObservationError("A24 invalidation inventory types changed")
    expected_quarantined = {
        valid_name: invalid_name
        for valid_name, invalid_name in SUCCESS_TO_INVALID.items()
        if valid_name in files
    }
    if "INVALIDATED.json.partial" in files:
        target = quarantined.get("INVALIDATED.json.partial")
        prefix = "INVALID_INVALIDATED_PARTIAL-"
        suffix = ".json.partial"
        number = (
            target[len(prefix) : -len(suffix)]
            if type(target) is str
            and target.startswith(prefix)
            and target.endswith(suffix)
            else ""
        )
        if not number.isdigit() or int(number) <= 0:
            raise MonitorObservationError("A24 invalidation partial quarantine changed")
        expected_quarantined["INVALIDATED.json.partial"] = target
    if quarantined != expected_quarantined:
        raise MonitorObservationError("A24 invalidation quarantine set changed")
    file_entries = {entry[1]: entry for entry in snapshot if entry[0] == "file"}
    actual_directories = {entry[1] for entry in snapshot if entry[0] == "directory"}
    expected_physical = (
        (set(files) - set(quarantined))
        | set(quarantined.values())
        | {"INVALIDATED.json"}
    )
    if set(file_entries) != expected_physical or actual_directories != set(directories):
        raise MonitorObservationError("A24 invalidation physical inventory changed")
    if any(name in file_entries for name in SUCCESS_TO_INVALID):
        raise MonitorObservationError(
            "A24 invalidation retains a success-shaped artifact"
        )
    for source_name, receipt in files.items():
        physical_name = quarantined.get(source_name, source_name)
        entry = file_entries.get(physical_name)
        if (
            type(source_name) is not str
            or type(receipt) is not dict
            or set(receipt) != {"bytes", "sha256"}
            or type(receipt.get("bytes")) is not int
            or receipt["bytes"] < 0
            or type(receipt.get("sha256")) is not str
            or not _HEX64.fullmatch(receipt["sha256"])
            or entry is None
            or receipt["bytes"] != entry[9]
            or receipt["sha256"] != entry[10]
        ):
            raise MonitorObservationError(
                f"A24 invalidation file receipt changed: {source_name}"
            )
    return payload


def _terminal_state(
    output: Path,
    *,
    lock: Path,
    expect: str,
    schedule: Mapping[str, Any],
    descriptor: Mapping[str, Any] | None,
    expected_lock_observation: tuple[os.stat_result, int, str, int, bytes] | None,
) -> tuple[str, tuple[tuple[Any, ...], ...], dict[str, Any]]:
    if expect == "formal" and expected_lock_observation != _observe_private_file(
        lock, label="A24 permanent lock", max_bytes=_MAX_JSON_BYTES
    ):
        raise ConcurrentObservationError(
            "A24 permanent lock changed before terminal scan"
        )
    first = _tree_snapshot(output)
    accepted: list[tuple[bool, dict[str, Any]]] = []
    for gate_open in (False, True):
        try:
            manifest = _validate_manifest_snapshot(
                output,
                snapshot=first,
                schedule=schedule,
                gate_open=gate_open,
                expect=expect,
                descriptor=descriptor,
            )
            accepted.append((gate_open, manifest))
        except ConcurrentObservationError:
            raise
        except (RuntimeError, OSError):
            continue
    if len(accepted) != 1:
        raise MonitorObservationError(
            "A24 terminal manifest has no unique valid inventory"
        )
    gate_open, manifest = accepted[0]
    progress = _validate_critical_receipt_summary(
        output,
        expect=expect,
        gate_open=gate_open,
        schedule=schedule,
        descriptor=descriptor,
    )
    _validate_manifest_snapshot(
        output,
        snapshot=first,
        schedule=schedule,
        gate_open=gate_open,
        expect=expect,
        descriptor=descriptor,
    )
    second = _tree_snapshot(output)
    if first != second:
        raise ConcurrentObservationError("A24 terminal changed during observation")
    if expect == "formal" and expected_lock_observation != _observe_private_file(
        lock, label="A24 permanent lock", max_bytes=_MAX_JSON_BYTES
    ):
        raise ConcurrentObservationError(
            "A24 permanent lock changed during observation"
        )
    return str(manifest["terminal_state"]), second, progress


def _guidance(state: str) -> str:
    if state == "NOT_STARTED":
        return "No attempt is present. Formal execution still requires separate explicit authorization."
    if state in {"LOCK_PUBLISHING", "PARTIAL_PUBLICATION_AMBIGUOUS"}:
        return "Publication is in flight or ambiguous; rescan later and do not delete partial files."
    if state in {
        "CORRUPT_AMBIGUOUS",
        "POSTLOCK_INCOMPLETE_LIVENESS_UNKNOWN",
        "INVALIDATION_PARTIAL_AMBIGUOUS",
        "CONCURRENT_OBSERVATION_AMBIGUOUS",
    }:
        return "Fail closed. Do not repair, rerun, or finalize without a separate audited operator action."
    if state == "LOCKED_PRE_OUTPUT":
        return "The one-shot attempt is consumed; the output may be in a short publication window. Liveness is unknown."
    if state == "POSTLOCK_INVALIDATED":
        return "The consumed attempt has a validated invalidation receipt and is not performance-evaluable."
    if state == "MANIFEST_INVENTORY_VALID_OUTCOME_UNVERIFIED":
        return "The stable manifest/inventory is only a terminal candidate. Independent deep typed replay is required before accepting the declared formal outcome or performance."
    return "Artifact phase observed; process liveness remains unknown because A24 has no PID/heartbeat receipt."


def inspect_a24(
    root: Path,
    output: Path,
    *,
    expect: str = "formal",
) -> dict[str, Any]:
    """Inspect one A24 namespace without writing to it."""

    if expect not in {"formal", "smoke"}:
        raise ValueError("A24 monitor expect must be formal or smoke")
    root = root.resolve()
    artifacts = root / "artifacts"
    output = Path(os.path.abspath(output))
    lock = root / FORMAL_LOCK
    initial_lock_present = _lexists(lock)
    lock_observation: tuple[os.stat_result, int, str, int, bytes] | None = None
    output_observation: tuple[tuple[Any, ...], ...] | None = None
    namespace_observation: tuple[bool, bool, tuple[str, ...]] | None = None
    report: dict[str, Any] = {
        "schema_version": MONITOR_VERSION,
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "root": str(root),
        "output": str(output),
        "expect": expect,
        "state": "CORRUPT_AMBIGUOUS",
        "classification": "ambiguous",
        "observed_phase_hint": None,
        "declared_terminal_state": None,
        "terminal": False,
        "terminal_candidate": False,
        "attempt_consumed": expect == "formal" and initial_lock_present,
        "formal_terminal_valid": False,
        "calibration_evidence_evaluable": False,
        "outer_performance_evaluable": False,
        "deep_typed_replay_valid": False,
        "manifest_inventory_valid": False,
        "verification_level": {
            "inventory_hashes": False,
            "critical_receipt_summary_consistency": False,
            "independent_semantic_replay": False,
        },
        "read_only": True,
        "automatic_repair": False,
        "liveness": "unknown",
        "liveness_inferred": False,
        "formal_lock": {
            "path": str(lock),
            "present": initial_lock_present,
            "valid": False,
        },
        "pending_locks": [],
        "namespace_stable": False,
        "progress": None,
        "snapshot_sha256": None,
        "issues": [],
        "exit_code": 2,
    }

    def finish(
        state: str, classification: str, *, exit_code: int = 0
    ) -> dict[str, Any]:
        nonlocal lock_observation, output_observation, namespace_observation
        concurrent_issue: str | None = None
        artifacts_before: os.stat_result | None = None
        if namespace_observation is not None:
            try:
                artifacts_before = _require_directory(
                    artifacts, label="A24 artifacts root"
                )
            except (MonitorObservationError, OSError) as exc:
                concurrent_issue = str(exc)
        pre_lock_present = _lexists(lock)
        pre_output_present = _lexists(output)
        if expect == "formal" and pre_lock_present:
            report["attempt_consumed"] = True
        if namespace_observation is not None:
            try:
                current_pending = tuple(_pending_locks(artifacts, lock))
                current_lock_present = _lexists(lock)
                current_output_present = _lexists(output)
                current_output = (
                    _tree_snapshot(output) if output_observation is not None else None
                )
                current_lock = (
                    _observe_private_file(
                        lock,
                        label="A24 permanent lock",
                        max_bytes=_MAX_JSON_BYTES,
                    )
                    if lock_observation is not None
                    else None
                )
                artifacts_after = _require_directory(
                    artifacts, label="A24 artifacts root"
                )
            except (MonitorObservationError, OSError) as exc:
                concurrent_issue = str(exc)
            else:
                report["formal_lock"]["present"] = current_lock_present
                if expect == "formal" and current_lock_present:
                    report["attempt_consumed"] = True
                if (
                    pre_lock_present != current_lock_present
                    or pre_output_present != current_output_present
                ):
                    concurrent_issue = (
                        "A24 lock/output namespace changed during final scan"
                    )
                if artifacts_before is None or _directory_identity(
                    artifacts_before
                ) != _directory_identity(artifacts_after):
                    concurrent_issue = (
                        "A24 artifacts root changed during final observation"
                    )
                current_namespace = (
                    current_lock_present,
                    current_output_present,
                    current_pending,
                )
                if current_namespace != namespace_observation:
                    concurrent_issue = (
                        "A24 lock/output/pending namespace changed during observation"
                    )
                if (
                    output_observation is not None
                    and current_output != output_observation
                ):
                    concurrent_issue = "A24 output changed during observation"
                if lock_observation is not None and current_lock != lock_observation:
                    concurrent_issue = "A24 permanent lock changed during observation"
        if concurrent_issue is not None:
            report["issues"].append(concurrent_issue)
            report["terminal"] = False
            report["terminal_candidate"] = False
            report["manifest_inventory_valid"] = False
            report["deep_typed_replay_valid"] = False
            report["formal_terminal_valid"] = False
            report["calibration_evidence_evaluable"] = False
            report["outer_performance_evaluable"] = False
            report["verification_level"] = {
                "inventory_hashes": False,
                "critical_receipt_summary_consistency": False,
                "independent_semantic_replay": False,
            }
            report["formal_lock"]["valid"] = False
            report["declared_terminal_state"] = None
            report["observed_phase_hint"] = None
            report["progress"] = None
            report["snapshot_sha256"] = None
            report.pop("invalidation_reason", None)
            state = "CONCURRENT_OBSERVATION_AMBIGUOUS"
            classification = "ambiguous"
            exit_code = 1
        elif namespace_observation is not None:
            report["namespace_stable"] = True
        report["state"] = state
        report["classification"] = classification
        report["exit_code"] = exit_code
        report["guidance"] = _guidance(state)
        return report

    try:
        _require_directory(artifacts, label="A24 artifacts root")
        if output.parent != artifacts:
            raise MonitorObservationError(
                "A24 output must be a direct child of artifacts"
            )
        pending = _pending_locks(artifacts, lock)
        report["pending_locks"] = pending
        lock_present = _lexists(lock)
        output_present = _lexists(output)
        namespace_observation = (
            lock_present,
            output_present,
            tuple(pending),
        )
        report["formal_lock"]["present"] = lock_present
        if expect == "formal" and lock_present:
            report["attempt_consumed"] = True
        if expect == "formal" and pending:
            return finish("LOCK_PUBLISHING", "ambiguous", exit_code=1)
        if not lock_present and not output_present:
            return finish("NOT_STARTED", "not_started")
        if expect == "formal" and not lock_present:
            raise MonitorObservationError(
                "A24 formal-like output exists without its permanent lock"
            )

        descriptor: dict[str, Any] | None = None
        if expect == "formal":
            descriptor, lock_observation = _validate_lock(lock, output)
            report["formal_lock"]["valid"] = True
            report["formal_lock"]["descriptor_sha256"] = hashlib.sha256(
                canonical_json_bytes(descriptor)
            ).hexdigest()
            if not output_present:
                return finish("LOCKED_PRE_OUTPUT", "incomplete", exit_code=1)

        _require_directory(output, label="A24 output")
        initial_snapshot = _tree_snapshot(output)
        output_observation = initial_snapshot
        report["snapshot_sha256"] = _snapshot_digest(initial_snapshot)
        relative_files = {entry[1] for entry in initial_snapshot if entry[0] == "file"}
        partials = sorted(name for name in relative_files if name.endswith(".partial"))
        invalid_files = sorted(
            name for name in relative_files if Path(name).name.startswith("INVALID_")
        )

        if "INVALIDATED.json" in relative_files:
            if expect != "formal" or descriptor is None:
                raise MonitorObservationError(
                    "A24 smoke output carries a formal invalidation"
                )
            before = _tree_snapshot(output)
            invalidation = _validate_invalidation_snapshot(
                output,
                snapshot=before,
                descriptor=descriptor,
            )
            after = _tree_snapshot(output)
            if before != after:
                raise ConcurrentObservationError(
                    "A24 invalidation changed during observation"
                )
            output_observation = after
            report["terminal"] = True
            report["invalidation_reason"] = invalidation["reason"]
            report["snapshot_sha256"] = _snapshot_digest(after)
            return finish("POSTLOCK_INVALIDATED", "invalidated")

        if "artifact-manifest.json" in relative_files:
            if "RUNNING.json" in relative_files or partials or invalid_files:
                raise MonitorObservationError(
                    "A24 manifest coexists with non-terminal markers"
                )
            schedule, training_contract = _schedule_from_training_contract(
                output, expect=expect
            )
            _validate_run_contract(
                output,
                expect=expect,
                descriptor=descriptor,
                training_contract=training_contract,
            )
            declared_state, terminal_snapshot, terminal_progress = _terminal_state(
                output,
                lock=lock,
                expect=expect,
                schedule=schedule,
                descriptor=descriptor,
                expected_lock_observation=lock_observation,
            )
            output_observation = terminal_snapshot
            report["terminal_candidate"] = True
            report["snapshot_sha256"] = _snapshot_digest(terminal_snapshot)
            report["progress"] = terminal_progress
            report["manifest_inventory_valid"] = True
            report["declared_terminal_state"] = declared_state
            report["observed_phase_hint"] = "TERMINAL_CANDIDATE"
            report["verification_level"]["inventory_hashes"] = True
            report["verification_level"]["critical_receipt_summary_consistency"] = True
            report["issues"].append(
                "deep typed replay was not performed by the stdlib-only monitor"
            )
            return finish(
                "MANIFEST_INVENTORY_VALID_OUTCOME_UNVERIFIED",
                "terminal_candidate",
                exit_code=1,
            )

        if invalid_files or "INVALIDATED.json.partial" in partials:
            report["issues"].append(
                "invalidation publication is incomplete: "
                + ",".join(sorted(set(invalid_files + partials)))
            )
            return finish("INVALIDATION_PARTIAL_AMBIGUOUS", "ambiguous", exit_code=1)
        if partials:
            report["issues"].append("partial publications: " + ",".join(partials))
            return finish("PARTIAL_PUBLICATION_AMBIGUOUS", "ambiguous", exit_code=1)

        def finish_phase(phase: str) -> dict[str, Any]:
            nonlocal output_observation
            report["observed_phase_hint"] = phase
            final_snapshot = _tree_snapshot(output)
            if initial_snapshot != final_snapshot:
                raise ConcurrentObservationError(
                    "A24 output changed during phase observation"
                )
            output_observation = final_snapshot
            report["snapshot_sha256"] = _snapshot_digest(final_snapshot)
            state = (
                "POSTLOCK_INCOMPLETE_LIVENESS_UNKNOWN"
                if expect == "formal"
                else "SMOKE_INCOMPLETE_LIVENESS_UNKNOWN"
            )
            return finish(state, "incomplete", exit_code=1)

        contracts_present = {
            name: name in relative_files
            for name in ("training-contract.json", "run-contract.json")
        }
        schedule = _formal_schedule() if expect == "formal" else None
        if all(contracts_present.values()):
            schedule, training_contract = _schedule_from_training_contract(
                output, expect=expect
            )
            _validate_run_contract(
                output,
                expect=expect,
                descriptor=descriptor,
                training_contract=training_contract,
            )
            report["progress"] = _progress_report(output, schedule)
            if report["progress"]["issues"]:
                raise MonitorObservationError("; ".join(report["progress"]["issues"]))

        if "result.json" in relative_files:
            result_candidate = _read_json(
                output / "result.json", label="A24 result candidate"
            )
            if result_candidate.get("schema_version") != RESULT_VERSION:
                raise MonitorObservationError("A24 result candidate is malformed")
            return finish_phase("FINALIZING")
        if "RUNNING.json" not in relative_files:
            return finish_phase("UNKNOWN")
        running = _validate_running(output, expect=expect)
        if schedule is None or not all(contracts_present.values()):
            return finish_phase("INITIALIZING")
        if running["outer_evaluation_started"]:
            return finish_phase("OUTER_EVALUATING")
        if running["calibration_started"]:
            if "calibration-gate.json" in relative_files:
                gate = _read_json(
                    output / "calibration-gate.json", label="A24 calibration gate"
                )
                decision = gate.get("decision")
                if (
                    type(decision) is not dict
                    or type(decision.get("outer_gate_permitted")) is not bool
                ):
                    raise MonitorObservationError("A24 calibration gate is malformed")
                if decision["outer_gate_permitted"] is False:
                    return finish_phase("GATE_CLOSED_FINALIZING")
            return finish_phase("CALIBRATING")
        if "all-fits-complete.json" in relative_files:
            return finish_phase("CALIBRATING")
        return finish_phase("TRAINING")
    except ConcurrentObservationError as exc:
        report["issues"].append(str(exc))
        return finish("CONCURRENT_OBSERVATION_AMBIGUOUS", "ambiguous", exit_code=1)
    except (MonitorObservationError, RuntimeError, OSError, ValueError) as exc:
        report["issues"].append(str(exc))
        return finish("CORRUPT_AMBIGUOUS", "ambiguous", exit_code=2)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only A24 lock/progress/terminal monitor"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (default: installed source checkout)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect", choices=("formal", "smoke"), default="formal")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output if args.output.is_absolute() else args.root / args.output
    report = inspect_a24(args.root, output, expect=args.expect)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
