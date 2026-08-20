"""Standard-library-only A24 lock, artifact, and terminal-state machinery."""

from __future__ import annotations

import hashlib
import os
import secrets
import signal
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from .a24_contract import (
    FORMAL_FOLDS,
    FORMAL_LOCK,
    FORMAL_SEEDS,
    INVALIDATED_VERSION,
    LOCK_VERSION,
    MANIFEST_VERSION,
    RUN_CONTRACT_VERSION,
    OutcomeState,
    atomic_write_json,
    canonical_json_bytes,
    expected_directories,
    expected_managed_paths,
    fsync_directory,
    require_exact_types,
    same_typed_json,
    strict_read_json,
)


SUCCESS_TO_INVALID = {
    "result.json": "INVALID_RESULT.json",
    "artifact-manifest.json": "INVALID_ARTIFACT_MANIFEST.json",
    "OUTER_GATE_OPEN.json": "INVALID_OUTER_GATE_OPEN.json",
    "a24-outer-decisions.jsonl": "INVALID_A24_OUTER_DECISIONS.jsonl",
    "a22-lagrangian-outer-decisions.jsonl": (
        "INVALID_A22_LAGRANGIAN_OUTER_DECISIONS.jsonl"
    ),
}


class FormalLockCreatedError(RuntimeError):
    """The permanent attempt lock exists even though descriptor write failed."""


class FormalTerminationRequested(RuntimeError):
    """A catchable termination request received after formal lock acquisition."""

    def __init__(self, signum: int) -> None:
        self.signum = signum
        super().__init__(f"received supervised post-lock signal {signum}")


class UncatchableTermination(RuntimeError):
    """Represent an orphaned attempt caused by an uncatchable host failure."""


@dataclass(frozen=True)
class RawSnapshot:
    directories: tuple[tuple[str, int], ...]
    files: tuple[tuple[str, int, str, bytes], ...]
    lock_bytes: bytes
    lock_mode: int


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_value_sha256(value: Any) -> str:
    raw = canonical_json_bytes(value)
    if not raw.endswith(b"\n"):
        raise RuntimeError("A24 canonical JSON encoding lost its terminator")
    return sha256_bytes(raw[:-1])


def ensure_private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=False, mode=0o700)
    os.chmod(path, 0o700)
    metadata = path.lstat()
    if path.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError("A24 output is not a real directory")
    if metadata.st_mode & 0o777 != 0o700:
        raise RuntimeError("A24 output directory mode is not 0700")
    fsync_directory(path.parent)


def lock_descriptor(
    *,
    output: Path,
    source_revision: str,
    run_contract_sha256: str,
    protocol_sha256: str,
    source_set_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": LOCK_VERSION,
        "attempt": 1,
        "output": str(output.resolve()),
        "source_revision": source_revision,
        "run_contract_sha256": run_contract_sha256,
        "protocol_sha256": protocol_sha256,
        "source_set_sha256": source_set_sha256,
    }


def lock_binding(lock: Path, descriptor: Mapping[str, Any]) -> dict[str, Any]:
    raw = canonical_json_bytes(dict(descriptor))
    return {
        "path": str(lock.resolve()),
        "descriptor": dict(descriptor),
        "descriptor_sha256": sha256_bytes(raw),
        "bytes": len(raw),
        "file_sha256": sha256_bytes(raw),
        "mode": "0600",
    }


def acquire_formal_lock(lock: Path, descriptor: Mapping[str, Any]) -> dict[str, Any]:
    raw = canonical_json_bytes(dict(descriptor))
    lock.parent.mkdir(parents=True, exist_ok=True)
    pending = lock.with_name(
        f".{lock.name}.pending-{os.getpid()}-{secrets.token_hex(8)}"
    )
    file_descriptor = os.open(
        pending,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    linked = False
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(pending, 0o600)
        os.link(pending, lock)
        linked = True
        fsync_directory(lock.parent)
        pending.unlink()
        fsync_directory(lock.parent)
    except BaseException as exc:
        try:
            os.close(file_descriptor)
        except OSError:
            pass
        try:
            pending.unlink()
            fsync_directory(lock.parent)
        except OSError:
            pass
        if linked:
            raise FormalLockCreatedError(
                "A24 permanent lock was created but descriptor publication failed"
            ) from exc
        raise
    binding = lock_binding(lock, descriptor)
    try:
        verify_formal_lock(lock, binding)
    except BaseException as exc:
        raise FormalLockCreatedError(
            "A24 permanent lock was created but final verification failed"
        ) from exc
    return binding


def verify_formal_lock(lock: Path, binding: Mapping[str, Any]) -> None:
    require_exact_types(
        binding,
        strings=("path", "descriptor_sha256", "file_sha256", "mode"),
        integers=("bytes",),
        objects=("descriptor",),
    )
    descriptor = binding["descriptor"]
    require_exact_types(
        descriptor,
        strings=(
            "schema_version",
            "output",
            "source_revision",
            "run_contract_sha256",
            "protocol_sha256",
            "source_set_sha256",
        ),
        integers=("attempt",),
    )
    expected_raw = canonical_json_bytes(descriptor)
    try:
        metadata = lock.lstat()
        raw = lock.read_bytes()
    except OSError as exc:
        raise RuntimeError("A24 formal lock is unreadable") from exc
    if (
        descriptor["schema_version"] != LOCK_VERSION
        or descriptor["attempt"] != 1
        or binding["path"] != str(lock.resolve())
        or binding["descriptor_sha256"] != sha256_bytes(expected_raw)
        or binding["file_sha256"] != sha256_bytes(expected_raw)
        or binding["bytes"] != len(expected_raw)
        or binding["mode"] != "0600"
        or lock.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_mode & 0o777 != 0o600
        or raw != expected_raw
    ):
        raise RuntimeError("A24 permanent lock binding changed")


@contextmanager
def supervised_postlock_signals():
    """Convert catchable termination signals into the invalidation path."""

    signals = (signal.SIGTERM, signal.SIGINT)
    previous = {item: signal.getsignal(item) for item in signals}

    def terminate(signum: int, _frame: Any) -> None:
        raise FormalTerminationRequested(signum)

    try:
        for item in signals:
            signal.signal(item, terminate)
        yield
    finally:
        for item, handler in previous.items():
            signal.signal(item, handler)


def manifest_payload(
    output: Path,
    *,
    source_revision: str,
    run_contract_sha256: str,
    folds: Sequence[int],
    seeds: Sequence[int],
    gate_open: bool,
    smoke: bool,
) -> dict[str, Any]:
    if type(smoke) is not bool:
        raise ValueError("A24 manifest smoke flag is not boolean")
    run_contract = strict_read_json(output / "run-contract.json")
    if (
        type(run_contract) is not dict
        or run_contract.get("schema_version") != RUN_CONTRACT_VERSION
    ):
        raise RuntimeError("A24 manifest run contract is malformed")
    if smoke:
        if run_contract.get("formal_lock") is not None:
            raise RuntimeError("A24 smoke run contract carries a formal lock")
    else:
        lock = (output.parents[1] / FORMAL_LOCK).resolve()
        descriptor = strict_read_json(lock)
        binding = lock_binding(lock, descriptor)
        verify_formal_lock(lock, binding)
        if (
            descriptor["output"] != str(output.resolve())
            or descriptor["source_revision"]
            != run_contract.get("source", {}).get("revision")
            or descriptor["run_contract_sha256"]
            != canonical_value_sha256(run_contract)
            or not same_typed_json(
                run_contract.get("formal_lock"),
                {
                    "schema_version": descriptor["schema_version"],
                    "attempt": descriptor["attempt"],
                    "path": str(lock),
                    "output": descriptor["output"],
                    "source_revision": descriptor["source_revision"],
                    "protocol_sha256": descriptor["protocol_sha256"],
                    "source_set_sha256": descriptor["source_set_sha256"],
                },
            )
        ):
            raise RuntimeError("A24 formal manifest lock binding changed")
    expected = expected_managed_paths(folds, seeds, gate_open=gate_open)
    return {
        "schema_version": MANIFEST_VERSION,
        "source_revision": source_revision,
        "run_contract_sha256": run_contract_sha256,
        "terminal_state": (
            OutcomeState.NON_EVIDENTIARY_SMOKE
            if smoke
            else (
                OutcomeState.VALID_GATE_OPEN_SUCCESS
                if gate_open
                else OutcomeState.VALID_CALIBRATION_NEGATIVE
            )
        ),
        "files": {
            name: {
                "bytes": (output / name).stat().st_size,
                "sha256": sha256_file(output / name),
            }
            for name in sorted(expected)
        },
    }


def _actual_inventory(output: Path) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories = {"."}
    for path in output.rglob("*"):
        relative = path.relative_to(output).as_posix()
        metadata = path.lstat()
        if stat.S_ISDIR(metadata.st_mode) and not path.is_symlink():
            directories.add(relative)
        elif stat.S_ISREG(metadata.st_mode) and not path.is_symlink():
            files.add(relative)
        else:
            raise RuntimeError("A24 inventory contains a symlink or special file")
    return files, directories


def validate_manifest(
    output: Path,
    *,
    folds: Sequence[int],
    seeds: Sequence[int],
    gate_open: bool,
    expected_source_revision: str | None = None,
    expected_run_contract_sha256: str | None = None,
    expected_lock_descriptor: Mapping[str, Any] | None = None,
    smoke: bool,
) -> dict[str, Any]:
    if type(smoke) is not bool:
        raise ValueError("A24 manifest smoke flag is not boolean")
    if (not smoke and expected_lock_descriptor is None) or (
        smoke and expected_lock_descriptor is not None
    ):
        raise RuntimeError("A24 manifest formal/smoke lock classification changed")
    manifest_path = output / "artifact-manifest.json"
    manifest = strict_read_json(manifest_path)
    require_exact_types(
        manifest,
        strings=(
            "schema_version",
            "source_revision",
            "run_contract_sha256",
            "terminal_state",
        ),
        objects=("files",),
    )
    expected = expected_managed_paths(folds, seeds, gate_open=gate_open)
    expected_dirs = expected_directories(folds, seeds, gate_open=gate_open)
    actual_files, actual_dirs = _actual_inventory(output)
    run_contract = strict_read_json(output / "run-contract.json")
    if (
        type(run_contract) is not dict
        or run_contract.get("schema_version") != RUN_CONTRACT_VERSION
    ):
        raise RuntimeError("A24 run contract is not an object")
    source = run_contract.get("source")
    if type(source) is not dict or type(source.get("revision")) is not str:
        raise RuntimeError("A24 run contract source binding is malformed")
    run_contract_sha256 = canonical_value_sha256(run_contract)
    formal_lock = run_contract.get("formal_lock")
    if smoke and formal_lock is not None:
        raise RuntimeError("A24 smoke manifest carries a formal lock")
    if expected_lock_descriptor is not None:
        formal_lock_path = (output.parents[1] / FORMAL_LOCK).resolve()
        verify_formal_lock(
            formal_lock_path,
            lock_binding(formal_lock_path, expected_lock_descriptor),
        )
        require_exact_types(
            expected_lock_descriptor,
            strings=(
                "schema_version",
                "output",
                "source_revision",
                "run_contract_sha256",
                "protocol_sha256",
                "source_set_sha256",
            ),
            integers=("attempt",),
        )
        expected_contract_lock = {
            "schema_version": expected_lock_descriptor["schema_version"],
            "attempt": expected_lock_descriptor["attempt"],
            "path": str((output.parents[1] / FORMAL_LOCK).resolve()),
            "output": expected_lock_descriptor["output"],
            "source_revision": expected_lock_descriptor["source_revision"],
            "protocol_sha256": expected_lock_descriptor["protocol_sha256"],
            "source_set_sha256": expected_lock_descriptor["source_set_sha256"],
        }
        if not same_typed_json(formal_lock, expected_contract_lock):
            raise RuntimeError("A24 run contract formal-lock binding changed")
    if (
        manifest["schema_version"] != MANIFEST_VERSION
        or manifest["terminal_state"]
        != (
            OutcomeState.NON_EVIDENTIARY_SMOKE
            if smoke
            else (
                OutcomeState.VALID_GATE_OPEN_SUCCESS
                if gate_open
                else OutcomeState.VALID_CALIBRATION_NEGATIVE
            )
        )
        or (expected_source_revision is not None and manifest["source_revision"] != expected_source_revision)
        or (
            expected_run_contract_sha256 is not None
            and manifest["run_contract_sha256"] != expected_run_contract_sha256
        )
        or manifest["run_contract_sha256"] != run_contract_sha256
        or source["revision"] != manifest["source_revision"]
        or set(manifest["files"]) != expected
        or actual_files != expected | {"artifact-manifest.json"}
        or actual_dirs != expected_dirs
    ):
        raise RuntimeError("A24 exact manifest inventory mismatch")
    for name, metadata in manifest["files"].items():
        if type(metadata) is not dict or set(metadata) != {"bytes", "sha256"}:
            raise RuntimeError("A24 manifest metadata schema mismatch")
        if (
            type(metadata["bytes"]) is not int
            or metadata["bytes"] < 0
            or type(metadata["sha256"]) is not str
            or len(metadata["sha256"]) != 64
        ):
            raise RuntimeError("A24 manifest metadata type mismatch")
        path = output / name
        file_metadata = path.lstat()
        if (
            path.is_symlink()
            or not stat.S_ISREG(file_metadata.st_mode)
            or file_metadata.st_mode & 0o777 != 0o600
            or path.stat().st_size != metadata["bytes"]
            or sha256_file(path) != metadata["sha256"]
        ):
            raise RuntimeError("A24 manifest byte binding changed")
    manifest_metadata = manifest_path.lstat()
    if (
        manifest_path.is_symlink()
        or not stat.S_ISREG(manifest_metadata.st_mode)
        or manifest_metadata.st_mode & 0o777 != 0o600
    ):
        raise RuntimeError("A24 artifact manifest mode/type changed")
    for name in expected_dirs:
        directory = output if name == "." else output / name
        metadata = directory.lstat()
        if (
            directory.is_symlink()
            or not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_mode & 0o777 != 0o700
        ):
            raise RuntimeError("A24 terminal directory mode/type changed")
    return manifest


def raw_snapshot(
    output: Path,
    lock: Path,
    *,
    folds: Sequence[int],
    seeds: Sequence[int],
    gate_open: bool,
) -> RawSnapshot:
    expected = expected_managed_paths(folds, seeds, gate_open=gate_open) | {
        "artifact-manifest.json"
    }
    actual_files, actual_dirs = _actual_inventory(output)
    if (
        actual_files != expected
        or actual_dirs != expected_directories(folds, seeds, gate_open=gate_open)
    ):
        raise RuntimeError("A24 final snapshot inventory mismatch")
    entries = []
    for name in sorted(expected):
        path = output / name
        metadata = path.lstat()
        raw = path.read_bytes()
        if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError("A24 final snapshot contains non-regular file")
        entries.append((name, metadata.st_mode & 0o777, sha256_bytes(raw), raw))
    lock_metadata = lock.lstat()
    lock_raw = lock.read_bytes()
    if lock.is_symlink() or not stat.S_ISREG(lock_metadata.st_mode):
        raise RuntimeError("A24 final snapshot lock is not regular")
    directory_entries = []
    for name in sorted(actual_dirs):
        directory = output if name == "." else output / name
        metadata = directory.lstat()
        if directory.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError("A24 final snapshot contains non-directory path")
        directory_entries.append((name, metadata.st_mode & 0o777))
    return RawSnapshot(
        directories=tuple(directory_entries),
        files=tuple(entries),
        lock_bytes=lock_raw,
        lock_mode=lock_metadata.st_mode & 0o777,
    )


def verify_raw_snapshot(
    expected: RawSnapshot,
    output: Path,
    lock: Path,
    *,
    folds: Sequence[int],
    seeds: Sequence[int],
    gate_open: bool,
) -> None:
    actual = raw_snapshot(
        output,
        lock,
        folds=folds,
        seeds=seeds,
        gate_open=gate_open,
    )
    if actual != expected:
        raise RuntimeError("A24 final direct raw snapshot changed")


def isolate_success_shaped(output: Path) -> dict[str, str]:
    isolated: dict[str, str] = {}
    collisions = [
        (valid_name, invalid_name)
        for valid_name, invalid_name in SUCCESS_TO_INVALID.items()
        if (output / valid_name).exists() and (output / invalid_name).exists()
    ]
    if collisions:
        raise RuntimeError("A24 invalid-artifact quarantine collision")
    for valid_name, invalid_name in SUCCESS_TO_INVALID.items():
        valid = output / valid_name
        invalid = output / invalid_name
        if valid.exists():
            os.rename(valid, invalid)
            fsync_directory(output)
            isolated[valid_name] = invalid_name
    return isolated


def publish_invalidation(
    output: Path,
    *,
    reason: str,
    error_type: str,
    error: str,
    lock_acquired: bool,
    orphan_inventory: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": INVALIDATED_VERSION,
        "invalidated": True,
        "reason": reason,
        "error_type": error_type,
        "error": error,
        "formal_lock_acquired": lock_acquired,
        "selective_retry_forbidden": True,
        "performance_evaluable": False,
        "orphan_inventory": dict(orphan_inventory or {}),
        "failed_at_utc": datetime.now(UTC).isoformat(),
    }
    atomic_write_json(output / "INVALIDATED.json", payload)
    return payload


def validate_invalidation(
    output: Path,
    *,
    lock: Path,
    expected_binding: Mapping[str, Any],
) -> dict[str, Any]:
    verify_formal_lock(lock, expected_binding)
    path = output / "INVALIDATED.json"
    payload = strict_read_json(path)
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
    if type(payload) is not dict or set(payload) != expected_keys:
        raise RuntimeError("A24 invalidation schema mismatch")
    require_exact_types(
        payload,
        strings=("schema_version", "reason", "error_type", "error", "failed_at_utc"),
        booleans=(
            "invalidated",
            "formal_lock_acquired",
            "selective_retry_forbidden",
            "performance_evaluable",
        ),
        objects=("orphan_inventory",),
    )
    if (
        payload["schema_version"] != INVALIDATED_VERSION
        or payload["invalidated"] is not True
        or payload["formal_lock_acquired"] is not True
        or payload["selective_retry_forbidden"] is not True
        or payload["performance_evaluable"] is not False
        or expected_binding["descriptor"]["output"] != str(output.resolve())
    ):
        raise RuntimeError("A24 invalidation semantic mismatch")
    metadata = path.lstat()
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_mode & 0o777 != 0o600
    ):
        raise RuntimeError("A24 invalidation mode/type changed")
    inventory = payload["orphan_inventory"]
    if set(inventory) != {"files", "directories", "quarantined"}:
        raise RuntimeError("A24 invalidation inventory schema mismatch")
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
        raise RuntimeError("A24 invalidation inventory types changed")
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
            raise RuntimeError("A24 invalidation partial quarantine changed")
        expected_quarantined["INVALIDATED.json.partial"] = target
    if quarantined != expected_quarantined:
        raise RuntimeError("A24 invalidation quarantine set is incomplete")
    if any((output / valid_name).exists() for valid_name in SUCCESS_TO_INVALID):
        raise RuntimeError("A24 invalidation retains a success-shaped artifact")
    for name, receipt in files.items():
        if type(name) is not str or type(receipt) is not dict or set(receipt) != {
            "bytes",
            "sha256",
        }:
            raise RuntimeError("A24 invalidation file receipt is malformed")
        if (
            type(receipt["bytes"]) is not int
            or receipt["bytes"] < 0
            or type(receipt["sha256"]) is not str
            or len(receipt["sha256"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in receipt["sha256"]
            )
        ):
            raise RuntimeError("A24 invalidation file receipt type changed")
    for valid_name, invalid_name in quarantined.items():
        if valid_name == "INVALIDATED.json.partial":
            allowed = expected_quarantined.get(valid_name) == invalid_name
        else:
            allowed = SUCCESS_TO_INVALID.get(valid_name) == invalid_name
        if not allowed:
            raise RuntimeError("A24 invalidation quarantine mapping changed")
        source = files.get(valid_name)
        target = output / invalid_name
        if type(source) is not dict or set(source) != {"bytes", "sha256"}:
            raise RuntimeError("A24 quarantine source receipt is malformed")
        target_metadata = target.lstat()
        if (
            target.is_symlink()
            or not stat.S_ISREG(target_metadata.st_mode)
            or target_metadata.st_mode & 0o777 != 0o600
            or target.stat().st_size != source["bytes"]
            or sha256_file(target) != source["sha256"]
        ):
            raise RuntimeError("A24 quarantined artifact binding changed")
    actual_files, actual_directories = _actual_inventory(output)
    expected_files = (
        (set(files) - set(quarantined))
        | set(quarantined.values())
        | {"INVALIDATED.json"}
    )
    if actual_files != expected_files or actual_directories != set(directories):
        raise RuntimeError("A24 invalidation physical inventory changed")
    for name in set(files) - set(quarantined):
        receipt = files[name]
        retained = output / name
        metadata = retained.lstat()
        if (
            retained.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_mode & 0o777 != 0o600
            or retained.stat().st_size != receipt["bytes"]
            or sha256_file(retained) != receipt["sha256"]
        ):
            raise RuntimeError("A24 retained orphan artifact binding changed")
    for name in directories:
        directory = output if name == "." else output / name
        metadata = directory.lstat()
        if (
            directory.is_symlink()
            or not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_mode & 0o777 != 0o700
        ):
            raise RuntimeError("A24 invalidation directory mode/type changed")
    return payload


def _orphan_inventory(output: Path) -> dict[str, Any]:
    files, directories = _actual_inventory(output)
    return {
        "files": {
            name: {
                "bytes": (output / name).stat().st_size,
                "sha256": sha256_file(output / name),
            }
            for name in sorted(files)
        },
        "directories": sorted(directories),
    }


def invalidate_postlock_failure(
    output: Path,
    *,
    lock: Path,
    expected_binding: Mapping[str, Any],
    reason: str,
    error: BaseException,
) -> dict[str, Any]:
    """Fail closed after a valid permanent lock has consumed the attempt."""

    expected_lock = (output.parents[1] / FORMAL_LOCK).resolve()
    if (
        output.parent != output.parents[1] / "artifacts"
        or lock.resolve() != expected_lock
    ):
        raise RuntimeError("A24 post-lock invalidation target is outside namespace")
    verify_formal_lock(lock, expected_binding)
    if not output.is_dir() or output.is_symlink():
        raise RuntimeError("A24 post-lock output is not a real directory")
    if (output / "INVALIDATED.json").exists():
        return validate_invalidation(
            output,
            lock=lock,
            expected_binding=expected_binding,
        )
    inventory = _orphan_inventory(output)
    isolated = isolate_success_shaped(output)
    invalidation_partial = output / "INVALIDATED.json.partial"
    if invalidation_partial.exists():
        index = 1
        while True:
            target_name = f"INVALID_INVALIDATED_PARTIAL-{index}.json.partial"
            target = output / target_name
            if not target.exists():
                break
            index += 1
        os.rename(invalidation_partial, target)
        fsync_directory(output)
        isolated["INVALIDATED.json.partial"] = target_name
    inventory["quarantined"] = isolated
    publish_invalidation(
        output,
        reason=reason,
        error_type=type(error).__name__,
        error=str(error),
        lock_acquired=True,
        orphan_inventory=inventory,
    )
    return validate_invalidation(
        output,
        lock=lock,
        expected_binding=expected_binding,
    )


def finalize_orphaned_attempt(
    root: Path,
    output: Path,
    *,
    lock_relative: Path = FORMAL_LOCK,
) -> dict[str, Any]:
    """One-way artifact-only ABANDONED -> INVALIDATED finalization."""

    root = root.resolve()
    output = output.resolve()
    lock = (root / lock_relative).resolve()
    artifacts = (root / "artifacts").resolve()
    if output.parent != artifacts or lock != (root / FORMAL_LOCK).resolve():
        raise RuntimeError("A24 orphan finalizer target is outside fixed namespace")
    descriptor = strict_read_json(lock)
    expected_binding = lock_binding(lock, descriptor)
    verify_formal_lock(lock, expected_binding)
    if descriptor["output"] != str(output):
        raise RuntimeError("A24 orphan lock does not bind requested output")
    if not output.exists():
        ensure_private_directory(output)
    elif not output.is_dir() or output.is_symlink():
        raise RuntimeError("A24 orphan output is not a real directory")
    if (output / "INVALIDATED.json").exists():
        return validate_invalidation(
            output,
            lock=lock,
            expected_binding=expected_binding,
        )
    # A valid manifest is never downgraded by the finalizer. Try both exact
    # terminal inventories; only an invalid/incomplete product may proceed.
    for gate_open in (False, True):
        try:
            validate_manifest(
                output,
                folds=FORMAL_FOLDS,
                seeds=FORMAL_SEEDS,
                gate_open=gate_open,
                expected_source_revision=descriptor["source_revision"],
                expected_run_contract_sha256=descriptor[
                    "run_contract_sha256"
                ],
                expected_lock_descriptor=descriptor,
                smoke=False,
            )
        except RuntimeError:
            continue
        raise RuntimeError("A24 valid terminal output cannot be orphan-finalized")
    return invalidate_postlock_failure(
        output,
        lock=lock,
        expected_binding=expected_binding,
        reason="orphaned_postlock_attempt",
        error=UncatchableTermination(
            "permanent lock exists without a valid terminal inventory"
        ),
    )
