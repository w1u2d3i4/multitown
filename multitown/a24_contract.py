"""Typed contracts and exact path inventory for A24."""

from __future__ import annotations

import json
import math
import os
import stat
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping, Sequence


RUNNER_VERSION = "multitown-a24-cr-ppo-no-shield-runner-v2"
POLICY_VERSION = "multitown-a24-cr-ppo-no-shield-policy-v1"
TRAINING_CONTRACT_VERSION = "multitown-a24-training-contract-v2"
RUN_CONTRACT_VERSION = "multitown-a24-run-contract-v2"
UPDATE_LOG_VERSION = "multitown-a24-update-log-v1"
FIT_COMPLETE_VERSION = "multitown-a24-fit-complete-v2"
ALL_FITS_VERSION = "multitown-a24-all-fits-complete-v2"
CALIBRATION_GATE_VERSION = "multitown-a24-calibration-gate-v2"
CALIBRATION_DIAGNOSTIC_VERSION = (
    "multitown-a24-a22-calibration-diagnostic-v1"
)
OUTER_GATE_VERSION = "multitown-a24-outer-gate-v1"
RESULT_VERSION = "multitown-a24-adaptive-development-result-v2"
MANIFEST_VERSION = "multitown-a24-artifact-manifest-v2"
LOCK_VERSION = "multitown-a24-formal-lock-v1"
INVALIDATED_VERSION = "multitown-a24-invalidated-attempt-v1"

MECHANISM = "cr-ppo-no-shield"
FORMAL_LOCK = Path("artifacts/a24-cr-ppo-no-shield-attempt-v1.lock")
FORMAL_SEEDS = (20260812, 20260813, 20260814)
FORMAL_FOLDS = (0, 1, 2, 3, 4)
A24_FORMAL_THREADS = 8

ALWAYS_ROOT_PATHS = frozenset(
    {
        "training-contract.json",
        "run-contract.json",
        "all-fits-complete.json",
        "calibration-decisions.jsonl",
        "calibration-gate.json",
        "result.json",
    }
)
CONDITIONAL_ROOT_PATHS = frozenset(
    {
        "OUTER_GATE_OPEN.json",
        "a24-outer-decisions.jsonl",
        "a22-lagrangian-outer-decisions.jsonl",
    }
)
FIT_FILENAMES = frozenset(
    {"training-metrics.jsonl", "final.pt", "fit-complete.json", "progress.json"}
)
SUCCESS_EXCLUDED_NAMES = frozenset(
    {
        "RUNNING.json",
        "INVALIDATED.json",
        "INVALID_RESULT.json",
        "INVALID_ARTIFACT_MANIFEST.json",
        "INVALID_OUTER_GATE_OPEN.json",
        "INVALID_A24_OUTER_DECISIONS.jsonl",
        "INVALID_A22_LAGRANGIAN_OUTER_DECISIONS.jsonl",
    }
)


class OutcomeState(StrEnum):
    NON_EVIDENTIARY_SMOKE = "NON_EVIDENTIARY_SMOKE"
    PRELOCK_FAILURE = "PRELOCK_FAILURE"
    POSTLOCK_INVALIDATED = "POSTLOCK_INVALIDATED"
    POSTLOCK_ABANDONED = "POSTLOCK_ABANDONED"
    VALID_CALIBRATION_NEGATIVE = "VALID_CALIBRATION_NEGATIVE"
    VALID_GATE_OPEN_SUCCESS = "VALID_GATE_OPEN_SUCCESS"


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) for row in rows)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json_loads(payload: str, *, label: str) -> Any:
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_strict_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value: {item}")
            ),
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"invalid strict A24 JSON: {label}") from exc
    if not all_finite(value):
        raise RuntimeError(f"non-finite A24 JSON value: {label}")
    return value


def strict_read_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = strict_json_loads(raw.decode("utf-8"), label=str(path))
    except (OSError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"cannot read A24 JSON artifact: {path}") from exc
    if type(value) is not dict:
        raise RuntimeError(f"A24 JSON artifact is not an object: {path}")
    return value


def strict_read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_bytes().decode("utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"cannot read A24 JSONL artifact: {path}") from exc
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines, start=1):
        if not line:
            raise RuntimeError(f"blank A24 JSONL row: {path}:{index}")
        value = strict_json_loads(line, label=f"{path}:{index}")
        if type(value) is not dict:
            raise RuntimeError(f"A24 JSONL row is not an object: {path}:{index}")
        rows.append(value)
    return rows


def all_finite(value: Any) -> bool:
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is dict:
        return all(type(key) is str and all_finite(item) for key, item in value.items())
    if type(value) in {list, tuple}:
        return all(all_finite(item) for item in value)
    return value is None or type(value) in {str, int, bool}


def same_typed_json(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if type(actual) is dict:
        return set(actual) == set(expected) and all(
            same_typed_json(actual[key], expected[key]) for key in actual
        )
    if type(actual) is list:
        return len(actual) == len(expected) and all(
            same_typed_json(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    return actual == expected


def require_exact_types(
    value: Mapping[str, Any],
    *,
    strings: Sequence[str] = (),
    integers: Sequence[str] = (),
    floats: Sequence[str] = (),
    booleans: Sequence[str] = (),
    lists: Sequence[str] = (),
    objects: Sequence[str] = (),
) -> None:
    expected = set(strings) | set(integers) | set(floats) | set(booleans) | set(
        lists
    ) | set(objects)
    if type(value) is not dict or set(value) != expected:
        raise RuntimeError("A24 exact typed-object key mismatch")
    checks = (
        (strings, str),
        (integers, int),
        (floats, float),
        (booleans, bool),
        (lists, list),
        (objects, dict),
    )
    for fields, expected_type in checks:
        if any(type(value[field]) is not expected_type for field in fields):
            raise RuntimeError("A24 exact typed-object value mismatch")


def fit_prefix(fold: int, seed: int) -> str:
    if type(fold) is not int or fold not in FORMAL_FOLDS:
        raise ValueError("invalid A24 fold")
    if type(seed) is not int or seed not in FORMAL_SEEDS:
        raise ValueError("invalid A24 seed")
    return f"fits/outer-fold-{fold}/seed-{seed}/{MECHANISM}"


def expected_managed_paths(
    folds: Sequence[int], seeds: Sequence[int], *, gate_open: bool,
) -> set[str]:
    paths = set(ALWAYS_ROOT_PATHS)
    for fold in folds:
        for seed in seeds:
            prefix = fit_prefix(int(fold), int(seed))
            paths.update(f"{prefix}/{name}" for name in FIT_FILENAMES)
    if gate_open:
        paths.update(CONDITIONAL_ROOT_PATHS)
    return paths


def expected_directories(
    folds: Sequence[int], seeds: Sequence[int], *, gate_open: bool,
) -> set[str]:
    del gate_open
    directories = {".", "fits"}
    for fold in folds:
        directories.add(f"fits/outer-fold-{int(fold)}")
        for seed in seeds:
            directories.add(f"fits/outer-fold-{int(fold)}/seed-{int(seed)}")
            directories.add(fit_prefix(int(fold), int(seed)))
    return directories


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    partial = path.with_name(path.name + ".partial")
    if path.exists() or partial.exists():
        raise FileExistsError(path if path.exists() else partial)
    descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
        fsync_directory(path.parent)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    metadata = path.lstat()
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_mode & 0o777 != 0o600
        or path.read_bytes() != payload
    ):
        raise RuntimeError("A24 atomic byte publication changed")


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(dict(value))
    trusted = strict_json_loads(payload.decode("utf-8"), label=f"trusted:{path}")
    if type(trusted) is not dict:
        raise RuntimeError("A24 trusted JSON payload is not an object")
    atomic_write_bytes(path, payload)
    if not same_typed_json(strict_read_json(path), trusted):
        raise RuntimeError("A24 JSON readback changed typed payload")


def atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    source = [dict(row) for row in rows]
    payload = canonical_jsonl_bytes(source)
    trusted = [
        strict_json_loads(line, label=f"trusted:{path}:{index}")
        for index, line in enumerate(payload.decode("utf-8").splitlines(), start=1)
    ]
    if any(type(row) is not dict for row in trusted):
        raise RuntimeError("A24 trusted JSONL payload contains non-object")
    atomic_write_bytes(path, payload)
    persisted = strict_read_jsonl(path)
    if not same_typed_json(persisted, trusted):
        raise RuntimeError("A24 JSONL readback changed typed payload")


def atomic_replace_json(path: Path, value: Mapping[str, Any]) -> None:
    if not path.exists() or path.is_symlink():
        raise FileNotFoundError(path)
    payload = canonical_json_bytes(dict(value))
    trusted = strict_json_loads(payload.decode("utf-8"), label=f"trusted:{path}")
    if type(trusted) is not dict:
        raise RuntimeError("A24 trusted replacement payload is not an object")
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        raise FileExistsError(partial)
    descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
        fsync_directory(path.parent)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    if path.read_bytes() != payload or not same_typed_json(
        strict_read_json(path), trusted
    ):
        raise RuntimeError("A24 atomic JSON replacement changed typed payload")
