"""Portable synthetic fixture for A24 numerical-semantic verification.

This module is deliberately separate from the host-private A24 verifier. It
uses a frozen external policy, strict JSON, and Safetensors. Its receipts are
self-reported engineering evidence: they do not reproduce A24 training,
inference, transitions, optimizer updates, or scientific results.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.resources
import math
import os
import platform
import re
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import rfc8785
import safetensors
from safetensors import SafetensorError, safe_open

from . import a24_contract
from .a24_contract import canonical_json_bytes, strict_json_loads

PORTABLE_VERIFIER_VERSION = "multitown-a24-portable-verifier-v1"
FIXTURE_TEMPLATE_VERSION = "multitown-a24-portable-fixture-template-v1"
SUBJECT_VERSION = "multitown-a24-portable-subject-v1"
DIAGNOSTIC_VERSION = "multitown-a24-portable-diagnostic-v1"
MANIFEST_VERSION = "multitown-a24-portable-manifest-v1"
POLICY_VERSION = "multitown-a24-portable-policy-v1"
RECEIPT_VERSION = "multitown-a24-portable-receipt-v1"
RECEIPT_SCHEMA_ID = "urn:multitown-bench:schema:a24-portable-receipt:v1"
PROFILE = "portable-a24-numerical-semantics-v1"
FIXTURE_ID = "multitown-a24-portable-synthetic-v1"
RECEIPT_ID_DOMAIN = "multitown-a24-portable-receipt-core-v1"
CANONICALIZATION_ID = "rfc8785-jcs-v1"
VARIANTS = frozenset({"passed", "failed", "unevaluated"})
SEMANTIC_OUTCOMES = frozenset({"PASSED", "FAILED", "UNEVALUATED"})
OUTCOMES = frozenset({*SEMANTIC_OUTCOMES, "ERROR"})
EVALUATION_STATUSES = frozenset({"COMPLETED", "UNEVALUATED", "ERROR"})
_POLICY_RESOURCE = "a24_portable_policy_v1.json"
_TEMPLATE_RESOURCE = "a24_portable_v1.json"
_RECEIPT_SCHEMA_RESOURCE = "a24_portable_receipt_v1.schema.json"
_MAX_POLICY_BYTES = 1024 * 1024
_MAX_RECEIPT_BYTES = 2 * 1024 * 1024
EXIT_PASSED = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_CONCURRENT = 3
EXIT_ERROR = 4
EXIT_UNEVALUATED = 5
EXIT_REJECTED = 6
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_HEX_F32 = re.compile(r"[0-9a-f]{8}\Z")
_FILE_CONTRACT = {
    "numerical-diagnostic.json": ("numerical-diagnostic", "application/json"),
    "numerics.safetensors": ("numerical-tensors", "application/vnd.safetensors"),
    "subject.json": ("subject", "application/json"),
}
_FILES = frozenset({*_FILE_CONTRACT, "manifest.json"})
_LIMITATIONS = [
    "synthetic non-evidentiary fixture",
    "not A24 training, safety, or performance evidence",
    "not independent experimental reproduction or replication",
    "no transition, inference, or optimizer-update replay",
    "unsigned self-reported receipt",
]
_CAPABILITY_KEYS = frozenset(
    {
        "portable_tensor_structure_verified",
        "raw_numerical_recomputation",
        "producer_exact_initialization_reconstruction",
        "transition_derivation_replayed",
        "inference_reexecution",
        "optimizer_update_reexecution",
        "results_reproduced",
        "formal_evidence_accepted",
    }
)
_SUBJECT_BOUNDARY = {
    "origin": "synthetic",
    "evidence_level": "NON_EVIDENTIARY_FIXTURE",
    "contains_real_multitown_episode": False,
    "derived_from_private_artifact": False,
    "experiment_run": False,
    "inference_reexecution": False,
    "optimizer_update_reexecution": False,
    "results_reproduced": False,
    "formal_execution_authorized": False,
    "scientific_claims_permitted": False,
}


class PortableVerificationRejected(RuntimeError):
    """The fixture, policy, receipt, or requested outcome failed closed."""


class PortableConcurrentObservation(RuntimeError):
    """The fixture or policy changed while it was being observed."""


def _reject(condition: bool, message: str) -> None:
    if not condition:
        raise PortableVerificationRejected(message)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_digest(value: Any, *, domain: str) -> str:
    payload = domain.encode("ascii") + b"\0" + rfc8785.dumps(value)
    return _sha256(payload)


def _strict_object(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = strict_json_loads(raw.decode("utf-8"), label=label)
    except (UnicodeDecodeError, RuntimeError) as exc:
        raise PortableVerificationRejected(f"invalid strict JSON: {label}") from exc
    _reject(type(value) is dict, f"JSON value is not an object: {label}")
    return value


def _resource_bytes(name: str, *, limit: int) -> bytes:
    resource = (
        importlib.resources.files("multitown").joinpath("fixtures").joinpath(name)
    )
    payload = resource.read_bytes()
    _reject(0 < len(payload) <= limit, f"packaged resource is not bounded: {name}")
    return payload


def _safe_directory_open(path: Path, *, label: str) -> int:
    _reject(
        all(hasattr(os, name) for name in ("O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")),
        f"platform lacks hardened directory flags: {label}",
    )
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
    try:
        return os.open(path, flags)
    except OSError as exc:
        raise PortableVerificationRejected(
            f"cannot safely open directory: {label}"
        ) from exc


def _bounded_regular_read_at(
    directory_fd: int, name: str, *, limit: int, label: str
) -> bytes:
    _reject(
        type(name) is str
        and bool(name)
        and "/" not in name
        and name not in {".", ".."},
        f"unsafe member name: {label}",
    )
    try:
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as exc:
        raise PortableVerificationRejected(f"cannot stat {label}") from exc
    _reject(
        stat.S_ISREG(before.st_mode)
        and before.st_nlink == 1
        and 0 <= before.st_size <= limit,
        f"{label} is not a bounded single-link regular file",
    )
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise PortableVerificationRejected(f"cannot safely open {label}") from exc
    try:
        opened = os.fstat(descriptor)
        identity = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size", "st_mtime_ns")
        if any(getattr(before, name) != getattr(opened, name) for name in identity):
            raise PortableConcurrentObservation(f"{label} changed before open")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, limit + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > limit:
                raise PortableVerificationRejected(f"{label} exceeds its byte limit")
        after = os.fstat(descriptor)
        try:
            current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError as exc:
            raise PortableConcurrentObservation(
                f"{label} disappeared during read"
            ) from exc
        if any(
            getattr(before, name) != getattr(after, name)
            or getattr(before, name) != getattr(current, name)
            for name in identity
        ):
            raise PortableConcurrentObservation(f"{label} changed during read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _bounded_regular_read(path: Path, *, limit: int, label: str) -> bytes:
    parent = path.parent if path.parent != Path("") else Path(".")
    directory_fd = _safe_directory_open(parent, label=f"parent of {label}")
    try:
        return _bounded_regular_read_at(
            directory_fd, path.name, limit=limit, label=label
        )
    finally:
        os.close(directory_fd)


def _exclusive_bytes_at(directory_fd: int, name: str, payload: bytes) -> None:
    _reject(
        type(name) is str
        and bool(name)
        and "/" not in name
        and name not in {".", ".."},
        "unsafe output member name",
    )
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | os.O_NOFOLLOW
        | os.O_CLOEXEC
        | os.O_NONBLOCK
    )
    try:
        descriptor = os.open(name, flags, 0o600, dir_fd=directory_fd)
    except FileExistsError as exc:
        raise PortableVerificationRejected(f"output already exists: {name}") from exc
    primary: BaseException | None = None
    try:
        os.fstat(descriptor)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("short write while materializing portable fixture")
            offset += written
        os.fsync(descriptor)
    except BaseException as exc:
        primary = exc
        raise
    finally:
        try:
            os.close(descriptor)
        except OSError as exc:
            if primary is not None:
                primary.add_note(f"also failed to close exclusive output fd: {exc}")
            else:
                raise


def _tensor_raw_sha(array: np.ndarray) -> str:
    dtype = np.dtype(array.dtype).newbyteorder("<")
    little = np.ascontiguousarray(array, dtype=dtype)
    return _sha256(little.tobytes(order="C"))


def _deterministic_safetensors_bytes(tensors: Mapping[str, np.ndarray]) -> bytes:
    """Serialize MultiTown's restricted deterministic Safetensors encoding v1."""
    metadata = {
        "format": "multitown-a24-portable-fixture-v1",
        "synthetic": "true",
    }
    header: dict[str, Any] = {"__metadata__": metadata}
    regions: list[bytes] = []
    offset = 0
    for name, array in sorted(tensors.items()):
        _reject(
            type(name) is str and bool(name) and name != "__metadata__",
            "unsafe generated Safetensors tensor name",
        )
        dtype = np.dtype(array.dtype)
        contract = {
            ("b", 1): ("BOOL", np.dtype(np.bool_)),
            ("f", 4): ("F32", np.dtype("<f4")),
            ("i", 8): ("I64", np.dtype("<i8")),
        }.get((dtype.kind, dtype.itemsize))
        _reject(contract is not None, f"unsupported generated tensor dtype: {name}")
        _reject(
            array.ndim > 0 and array.size > 0 and all(size > 0 for size in array.shape),
            f"empty or scalar generated tensor: {name}",
        )
        dtype_name, little_dtype = contract
        little = np.ascontiguousarray(array, dtype=little_dtype)
        raw = little.tobytes(order="C")
        end = offset + len(raw)
        header[name] = {
            "dtype": dtype_name,
            "shape": list(little.shape),
            "data_offsets": [offset, end],
        }
        regions.append(raw)
        offset = end
    header_json = canonical_json_bytes(header)[:-1]
    header_raw = header_json + (b" " * (-len(header_json) % 8))
    return (
        len(header_raw).to_bytes(8, "little", signed=False)
        + header_raw
        + b"".join(regions)
    )


def _decode_tensor(definition: Mapping[str, Any], *, label: str) -> np.ndarray:
    _reject(type(definition) is dict, f"tensor definition changed: {label}")
    dtype_name = definition.get("dtype")
    shape = definition.get("shape")
    _reject(
        type(shape) is list
        and bool(shape)
        and all(type(item) is int and 0 < item <= 1024 for item in shape),
        f"tensor shape changed: {label}",
    )
    size = math.prod(shape)
    if dtype_name == "float32":
        _reject(
            set(definition) == {"dtype", "shape", "f32_hex"}
            and type(definition.get("f32_hex")) is list
            and len(definition["f32_hex"]) == size
            and all(
                type(item) is str and _HEX_F32.fullmatch(item)
                for item in definition["f32_hex"]
            ),
            f"float32 tensor encoding changed: {label}",
        )
        bits = np.asarray(
            [int(item, 16) for item in definition["f32_hex"]], dtype="<u4"
        )
        array = bits.view("<f4").reshape(shape)
    elif dtype_name == "bool":
        _reject(
            set(definition) == {"dtype", "shape", "values"}
            and type(definition.get("values")) is list
            and len(definition["values"]) == size
            and all(type(item) is bool for item in definition["values"]),
            f"bool tensor encoding changed: {label}",
        )
        array = np.asarray(definition["values"], dtype=np.bool_).reshape(shape)
    elif dtype_name == "int64":
        _reject(
            set(definition) == {"dtype", "shape", "values"}
            and type(definition.get("values")) is list
            and len(definition["values"]) == size
            and all(type(item) is int for item in definition["values"]),
            f"int64 tensor encoding changed: {label}",
        )
        array = np.asarray(definition["values"], dtype="<i8").reshape(shape)
    else:
        raise PortableVerificationRejected(f"unsupported tensor dtype: {label}")
    _reject(
        not np.issubdtype(array.dtype, np.floating) or bool(np.isfinite(array).all()),
        f"tensor is non-finite: {label}",
    )
    return array


def _template(*, expected_sha256: str) -> dict[str, Any]:
    raw = _resource_bytes(_TEMPLATE_RESOURCE, limit=1024 * 1024)
    _reject(
        _sha256(raw) == expected_sha256,
        "packaged fixture template does not match frozen policy",
    )
    value = _strict_object(raw, label="packaged portable fixture template")
    _reject(
        set(value) == {"schema_version", "fixture_id", "shared_tensors", "variants"}
        and value["schema_version"] == FIXTURE_TEMPLATE_VERSION
        and value["fixture_id"] == FIXTURE_ID
        and type(value["shared_tensors"]) is dict
        and type(value["variants"]) is dict
        and set(value["variants"]) == VARIANTS,
        "packaged fixture template changed",
    )
    return value


def _validate_policy_object(policy: Mapping[str, Any]) -> None:
    _reject(
        type(policy) is dict
        and set(policy)
        == {
            "schema_version",
            "policy_id",
            "profile",
            "allowed_fixture_id",
            "allowed_variants",
            "limits",
            "tensor_contracts",
            "numerical_policy",
            "template_sha256",
            "fixture_cases",
        }
        and policy["schema_version"] == POLICY_VERSION
        and policy["policy_id"] == "multitown-a24-portable-synthetic-policy-v1"
        and policy["profile"] == PROFILE
        and policy["allowed_fixture_id"] == FIXTURE_ID
        and policy["allowed_variants"] == ["failed", "passed", "unevaluated"]
        and type(policy["template_sha256"]) is str
        and _HEX_SHA256.fullmatch(policy["template_sha256"])
        and policy["template_sha256"]
        == _sha256(_resource_bytes(_TEMPLATE_RESOURCE, limit=1024 * 1024)),
        "portable policy schema changed",
    )
    limits = policy["limits"]
    _reject(
        type(limits) is dict
        and set(limits)
        == {
            "artifact_max_bytes",
            "file_max_bytes",
            "safetensors_header_max_bytes",
            "tensor_max_count",
            "tensor_max_bytes",
            "numerical_max_points",
        }
        and all(
            type(value) is int and 0 < value <= 64 * 1024 * 1024
            for value in limits.values()
        )
        and limits["tensor_max_count"] <= 128,
        "portable policy limits changed",
    )
    contracts = policy["tensor_contracts"]
    _reject(
        type(contracts) is dict
        and set(contracts) == {"base", "evaluated"}
        and type(contracts["base"]) is dict
        and type(contracts["evaluated"]) is dict
        and not (set(contracts["base"]) & set(contracts["evaluated"])),
        "portable tensor contracts changed",
    )
    for name, contract in {**contracts["base"], **contracts["evaluated"]}.items():
        _reject(
            type(name) is str
            and bool(name)
            and type(contract) is dict
            and set(contract) == {"dtype", "shape"}
            and contract["dtype"] in {"bool", "float32", "int64"}
            and type(contract["shape"]) is list
            and bool(contract["shape"])
            and all(type(item) is int and item > 0 for item in contract["shape"]),
            f"portable tensor contract changed: {name}",
        )
    cases = policy["fixture_cases"]
    _reject(
        type(cases) is dict and set(cases) == VARIANTS,
        "portable fixture cases changed",
    )
    expected_case_outcomes = {
        "failed": ("raw-tensors", "FAILED"),
        "passed": ("raw-tensors", "PASSED"),
        "unevaluated": ("actual-tensors-withheld", "UNEVALUATED"),
    }
    for variant, (availability, outcome) in expected_case_outcomes.items():
        case = cases[variant]
        required_tensors = set(contracts["base"])
        if availability == "raw-tensors":
            required_tensors |= set(contracts["evaluated"])
        _reject(
            type(case) is dict
            and set(case)
            == {
                "availability",
                "expected_outcome",
                "subject_sha256",
                "diagnostic_sha256",
                "tensor_raw_sha256",
            }
            and case["availability"] == availability
            and case["expected_outcome"] == outcome
            and all(
                type(case[name]) is str and _HEX_SHA256.fullmatch(case[name])
                for name in ("subject_sha256", "diagnostic_sha256")
            )
            and type(case["tensor_raw_sha256"]) is dict
            and set(case["tensor_raw_sha256"]) == required_tensors
            and all(
                type(value) is str and _HEX_SHA256.fullmatch(value)
                for value in case["tensor_raw_sha256"].values()
            ),
            f"portable fixture case changed: {variant}",
        )
    numerical = policy["numerical_policy"]
    _reject(
        type(numerical) is dict
        and set(numerical)
        == {
            "transition_count",
            "full_batch",
            "probability_ratio",
            "legacy_diagnostic",
            "summary",
            "nonfinite_policy",
            "equal_nan",
        }
        and numerical["transition_count"] == 4
        and numerical["nonfinite_policy"] == "reject"
        and numerical["equal_nan"] is False,
        "portable numerical policy changed",
    )
    full_batch = numerical["full_batch"]
    ratio = numerical["probability_ratio"]
    legacy = numerical["legacy_diagnostic"]
    summary = numerical["summary"]
    _reject(
        type(full_batch) is dict
        and set(full_batch)
        == {
            "comparison",
            "reference_operand",
            "rtol_decimal",
            "atol_decimal",
            "evaluation_dtype",
        }
        and full_batch["comparison"]
        == "abs_actual_expected_le_atol_plus_rtol_abs_expected"
        and full_batch["reference_operand"] == "expected"
        and full_batch["evaluation_dtype"] == "float32"
        and type(ratio) is dict
        and set(ratio) == {"formula", "threshold_decimal", "evaluation_dtype"}
        and ratio["formula"] == "max_abs_exp_actual_log_minus_expected_log_minus_one"
        and ratio["evaluation_dtype"] == "float32"
        and type(legacy) is dict
        and set(legacy)
        == {"rtol_decimal", "atol_decimal", "acceptance_gate", "tolerance_dtype"}
        and legacy["acceptance_gate"] is False
        and legacy["tolerance_dtype"] == "float32"
        and type(summary) is dict
        and set(summary)
        == {
            "input_dtype",
            "accumulator_dtype",
            "relative_denominator",
            "quantiles",
            "quantile_method",
            "quantile_algorithm",
            "axis",
            "weights",
            "ulp_algorithm",
        }
        and summary["input_dtype"] == "float32"
        and summary["accumulator_dtype"] == "float64"
        and summary["relative_denominator"] == "max_abs_expected_float32_tiny"
        and summary["quantiles"] == [0.5, 0.95, 0.99]
        and summary["quantile_method"] == "linear"
        and summary["quantile_algorithm"] == "Hyndman-Fan-type-7"
        and summary["axis"] == "flattened"
        and summary["weights"] is None
        and summary["ulp_algorithm"] == "ordered-ieee754-binary32-distance-v1",
        "portable numerical algorithms changed",
    )
    decimals = (
        full_batch["rtol_decimal"],
        full_batch["atol_decimal"],
        ratio["threshold_decimal"],
        legacy["rtol_decimal"],
        legacy["atol_decimal"],
    )
    try:
        numeric = [float(value) for value in decimals]
    except (TypeError, ValueError) as exc:
        raise PortableVerificationRejected(
            "portable policy decimals are invalid"
        ) from exc
    _reject(
        all(type(value) is str for value in decimals)
        and all(math.isfinite(value) and value >= 0 for value in numeric),
        "portable policy decimals are invalid",
    )


def _packaged_policy() -> tuple[bytes, dict[str, Any]]:
    raw = _resource_bytes(_POLICY_RESOURCE, limit=_MAX_POLICY_BYTES)
    policy = _strict_object(raw, label="packaged portable policy")
    _validate_policy_object(policy)
    return raw, policy


def _receipt_schema_identity() -> dict[str, str]:
    raw = _resource_bytes(_RECEIPT_SCHEMA_RESOURCE, limit=1024 * 1024)
    schema = _strict_object(raw, label="packaged portable receipt schema")
    _reject(
        schema.get("$id") == RECEIPT_SCHEMA_ID
        and schema.get("title") == "MultiTown A24 portable verification receipt v1",
        "packaged portable receipt schema changed",
    )
    return {"schema_id": RECEIPT_SCHEMA_ID, "schema_sha256": _sha256(raw)}


def materialize_fixture(
    output: Path,
    policy_out: Path,
    *,
    variant: Literal["passed", "failed", "unevaluated"],
) -> dict[str, Any]:
    _reject(variant in VARIANTS, "unknown portable fixture variant")
    output_absolute = output.absolute()
    policy_absolute = policy_out.absolute()
    _reject(
        not policy_absolute.is_relative_to(output_absolute),
        "portable policy must be outside the artifact",
    )
    policy_raw, policy = _packaged_policy()
    template = _template(expected_sha256=policy["template_sha256"])
    case = policy["fixture_cases"][variant]
    variant_definition = template["variants"][variant]
    _reject(type(variant_definition) is dict, "portable variant definition changed")
    availability = variant_definition.get("availability")
    _reject(
        availability in {"raw-tensors", "actual-tensors-withheld"},
        "portable variant availability changed",
    )
    _reject(
        availability == case["availability"],
        "portable variant availability does not match frozen policy",
    )
    expected_variant_keys = (
        {"availability", "tensors"}
        if availability == "raw-tensors"
        else {"availability"}
    )
    _reject(
        set(variant_definition) == expected_variant_keys,
        "portable variant definition changed",
    )
    tensors: dict[str, np.ndarray] = {}
    for name, definition in sorted(template["shared_tensors"].items()):
        tensors[name] = _decode_tensor(definition, label=f"shared {name}")
    if availability == "raw-tensors":
        _reject(
            type(variant_definition["tensors"]) is dict,
            "portable evaluated tensors changed",
        )
        for name, definition in sorted(variant_definition["tensors"].items()):
            _reject(name not in tensors, f"duplicate portable tensor: {name}")
            tensors[name] = _decode_tensor(definition, label=f"{variant} {name}")
    required = dict(policy["tensor_contracts"]["base"])
    if availability == "raw-tensors":
        required.update(policy["tensor_contracts"]["evaluated"])
    _reject(set(tensors) == set(required), "portable tensor template contract changed")
    specs: dict[str, Any] = {}
    for name, array in sorted(tensors.items()):
        contract = required[name]
        _reject(
            str(array.dtype) == contract["dtype"]
            and list(array.shape) == contract["shape"],
            f"portable tensor template violates policy: {name}",
        )
        specs[name] = {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "raw_little_endian_sha256": _tensor_raw_sha(array),
            "bytes": int(array.nbytes),
        }
    _reject(
        {name: spec["raw_little_endian_sha256"] for name, spec in specs.items()}
        == case["tensor_raw_sha256"],
        "portable tensor template does not match frozen case",
    )
    subject = {
        "schema_version": SUBJECT_VERSION,
        "fixture_id": FIXTURE_ID,
        "profile": PROFILE,
        "variant": variant,
        **_SUBJECT_BOUNDARY,
        "tensors_file": "numerics.safetensors",
        "diagnostic_file": "numerical-diagnostic.json",
        "tensor_specs": specs,
    }
    diagnostic = {
        "schema_version": DIAGNOSTIC_VERSION,
        "metric_profile": "a24-pq1-numerical-conformance-subset-v1",
        "availability": availability,
        "transition_count": policy["numerical_policy"]["transition_count"],
    }
    subject_raw = canonical_json_bytes(subject)
    diagnostic_raw = canonical_json_bytes(diagnostic)
    _reject(
        _sha256(subject_raw) == case["subject_sha256"]
        and _sha256(diagnostic_raw) == case["diagnostic_sha256"],
        "portable generated metadata does not match frozen case",
    )
    tensor_raw = _deterministic_safetensors_bytes(tensors)
    generated = {
        "numerical-diagnostic.json": diagnostic_raw,
        "numerics.safetensors": tensor_raw,
        "subject.json": subject_raw,
    }
    entries = []
    for name, raw in sorted(generated.items()):
        role, media_type = _FILE_CONTRACT[name]
        entries.append(
            {
                "path": name,
                "role": role,
                "media_type": media_type,
                "bytes": len(raw),
                "sha256": _sha256(raw),
            }
        )
    manifest_raw = canonical_json_bytes(
        {
            "schema_version": MANIFEST_VERSION,
            "fixture_id": FIXTURE_ID,
            "variant": variant,
            "files": entries,
        }
    )
    generated["manifest.json"] = manifest_raw
    output_parent_fd = _safe_directory_open(
        output.parent, label="portable fixture output parent"
    )
    try:
        policy_parent_fd = _safe_directory_open(
            policy_out.parent, label="portable policy output parent"
        )
    except BaseException as exc:
        try:
            os.close(output_parent_fd)
        except OSError as close_exc:
            exc.add_note(f"also failed to close output parent fd: {close_exc}")
        raise
    output_fd: int | None = None
    primary: BaseException | None = None
    try:
        try:
            os.mkdir(output.name, mode=0o700, dir_fd=output_parent_fd)
        except FileExistsError as exc:
            raise PortableVerificationRejected(
                "portable fixture output already exists"
            ) from exc
        output_fd = os.open(
            output.name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
            dir_fd=output_parent_fd,
        )
        for name, payload in sorted(generated.items()):
            _exclusive_bytes_at(output_fd, name, payload)
        os.fsync(output_fd)
        _exclusive_bytes_at(policy_parent_fd, policy_out.name, policy_raw)
        os.fsync(policy_parent_fd)
        os.fsync(output_parent_fd)
        return {
            "artifact": str(output_absolute),
            "policy": str(policy_absolute),
            "variant": variant,
            "manifest_sha256": _sha256(manifest_raw),
            "policy_sha256": _sha256(policy_raw),
        }
    except BaseException as exc:
        primary = exc
        raise
    finally:
        close_error: OSError | None = None
        for label, descriptor in (
            ("portable output", output_fd),
            ("portable policy parent", policy_parent_fd),
            ("portable output parent", output_parent_fd),
        ):
            if descriptor is None:
                continue
            try:
                os.close(descriptor)
            except OSError as exc:
                if primary is not None:
                    primary.add_note(f"also failed to close {label} fd: {exc}")
                elif close_error is None:
                    close_error = exc
        if primary is None and close_error is not None:
            raise close_error


def _snapshot(path: Path, *, limits: Mapping[str, int]) -> tuple[dict[str, bytes], str]:
    directory_fd = _safe_directory_open(path, label="portable artifact")
    try:
        before = os.fstat(directory_fd)
        _reject(stat.S_ISDIR(before.st_mode), "artifact is not a real directory")
        try:
            names = set(os.listdir(directory_fd))
        except OSError as exc:
            raise PortableVerificationRejected(
                "cannot enumerate portable artifact"
            ) from exc
        _reject(names == _FILES, "portable artifact inventory changed")
        raw: dict[str, bytes] = {}
        total = 0
        for name in sorted(_FILES):
            payload = _bounded_regular_read_at(
                directory_fd,
                name,
                limit=limits["file_max_bytes"],
                label=f"artifact {name}",
            )
            raw[name] = payload
            total += len(payload)
        _reject(
            total <= limits["artifact_max_bytes"],
            "portable artifact is too large",
        )
        after = os.fstat(directory_fd)
        identity_fields = ("st_dev", "st_ino", "st_mode", "st_mtime_ns")
        if any(
            getattr(before, name) != getattr(after, name) for name in identity_fields
        ):
            raise PortableConcurrentObservation("portable artifact directory changed")
        identity = {
            name: {"bytes": len(payload), "sha256": _sha256(payload)}
            for name, payload in sorted(raw.items())
        }
        return raw, _canonical_digest(
            identity, domain="multitown-a24-portable-snapshot-v1"
        )
    finally:
        os.close(directory_fd)


def _validate_manifest(manifest: Mapping[str, Any], raw: Mapping[str, bytes]) -> None:
    _reject(
        type(manifest) is dict
        and set(manifest) == {"schema_version", "fixture_id", "variant", "files"}
        and manifest["schema_version"] == MANIFEST_VERSION
        and manifest["fixture_id"] == FIXTURE_ID
        and manifest["variant"] in VARIANTS
        and type(manifest["files"]) is list
        and len(manifest["files"]) == len(_FILE_CONTRACT),
        "portable manifest changed",
    )
    observed: set[str] = set()
    for entry in manifest["files"]:
        _reject(
            type(entry) is dict
            and set(entry) == {"path", "role", "media_type", "bytes", "sha256"}
            and type(entry["path"]) is str
            and entry["path"] in _FILE_CONTRACT
            and entry["path"] not in observed,
            "portable manifest entry changed",
        )
        name = entry["path"]
        observed.add(name)
        role, media_type = _FILE_CONTRACT[name]
        _reject(
            entry["role"] == role
            and entry["media_type"] == media_type
            and type(entry["bytes"]) is int
            and entry["bytes"] == len(raw[name])
            and type(entry["sha256"]) is str
            and _HEX_SHA256.fullmatch(entry["sha256"])
            and entry["sha256"] == _sha256(raw[name]),
            f"portable manifest binding changed: {name}",
        )
    _reject(observed == set(_FILE_CONTRACT), "portable manifest inventory changed")


def _validate_subject(subject: Mapping[str, Any], *, variant: str) -> None:
    expected_keys = {
        "schema_version",
        "fixture_id",
        "profile",
        "variant",
        *_SUBJECT_BOUNDARY,
        "tensors_file",
        "diagnostic_file",
        "tensor_specs",
    }
    _reject(
        type(subject) is dict
        and set(subject) == expected_keys
        and subject["schema_version"] == SUBJECT_VERSION
        and subject["fixture_id"] == FIXTURE_ID
        and subject["profile"] == PROFILE
        and subject["variant"] == variant
        and all(subject[key] == value for key, value in _SUBJECT_BOUNDARY.items())
        and subject["tensors_file"] == "numerics.safetensors"
        and subject["diagnostic_file"] == "numerical-diagnostic.json"
        and type(subject["tensor_specs"]) is dict,
        "portable subject changed",
    )


def _validate_diagnostic(
    diagnostic: Mapping[str, Any], policy: Mapping[str, Any]
) -> str:
    _reject(
        type(diagnostic) is dict
        and set(diagnostic)
        == {"schema_version", "metric_profile", "availability", "transition_count"}
        and diagnostic["schema_version"] == DIAGNOSTIC_VERSION
        and diagnostic["metric_profile"] == "a24-pq1-numerical-conformance-subset-v1"
        and diagnostic["availability"] in {"raw-tensors", "actual-tensors-withheld"}
        and type(diagnostic["transition_count"]) is int
        and diagnostic["transition_count"]
        == policy["numerical_policy"]["transition_count"],
        "portable numerical diagnostic changed",
    )
    return diagnostic["availability"]


def _parse_safetensors_header(
    payload: bytes, *, policy: Mapping[str, Any]
) -> dict[str, Any]:
    limits = policy["limits"]
    _reject(len(payload) >= 10, "Safetensors file is truncated")
    header_size = int.from_bytes(payload[:8], "little", signed=False)
    _reject(
        2 <= header_size <= limits["safetensors_header_max_bytes"]
        and 8 + header_size <= len(payload),
        "Safetensors header length is invalid",
    )
    header_raw = payload[8 : 8 + header_size]
    header = _strict_object(header_raw, label="Safetensors header")
    metadata = header.pop("__metadata__", None)
    _reject(
        metadata == {"format": "multitown-a24-portable-fixture-v1", "synthetic": "true"},
        "Safetensors metadata changed",
    )
    _reject(
        0 < len(header) <= limits["tensor_max_count"],
        "Safetensors tensor count exceeds policy",
    )
    data_size = len(payload) - 8 - header_size
    _reject(
        data_size <= limits["tensor_max_bytes"],
        "Safetensors aggregate data exceeds policy",
    )
    regions: list[tuple[int, int, str]] = []
    for name, descriptor in header.items():
        _reject(
            type(name) is str
            and bool(name)
            and type(descriptor) is dict
            and set(descriptor) == {"dtype", "shape", "data_offsets"}
            and descriptor["dtype"] in {"BOOL", "F32", "I64"}
            and type(descriptor["shape"]) is list
            and bool(descriptor["shape"])
            and all(type(item) is int and item > 0 for item in descriptor["shape"])
            and type(descriptor["data_offsets"]) is list
            and len(descriptor["data_offsets"]) == 2
            and all(type(item) is int for item in descriptor["data_offsets"]),
            f"Safetensors descriptor changed: {name}",
        )
        start, end = descriptor["data_offsets"]
        item_size = {"BOOL": 1, "F32": 4, "I64": 8}[descriptor["dtype"]]
        expected_bytes = math.prod(descriptor["shape"]) * item_size
        _reject(
            0 <= start <= end <= data_size
            and end - start == expected_bytes
            and expected_bytes <= limits["tensor_max_bytes"],
            f"Safetensors offsets changed: {name}",
        )
        regions.append((start, end, name))
    cursor = 0
    for start, end, name in sorted(regions):
        _reject(start == cursor, f"Safetensors data region is not contiguous: {name}")
        cursor = end
    _reject(cursor == data_size, "Safetensors data has trailing or missing bytes")
    return {
        "header_bytes": header_size,
        "header_sha256": _sha256(header_raw),
        "data_bytes": data_size,
        "tensor_count": len(header),
        "tensor_names": sorted(header),
    }


def _tensor_contract_for(
    policy: Mapping[str, Any], *, availability: str
) -> dict[str, Any]:
    required = dict(policy["tensor_contracts"]["base"])
    if availability == "raw-tensors":
        required.update(policy["tensor_contracts"]["evaluated"])
    return required


def _load_tensors(
    staged: Path,
    tensor_payload: bytes,
    subject: Mapping[str, Any],
    policy: Mapping[str, Any],
    *,
    availability: str,
    case: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    header = _parse_safetensors_header(tensor_payload, policy=policy)
    required = _tensor_contract_for(policy, availability=availability)
    specs = subject["tensor_specs"]
    _reject(
        set(required) == set(specs) == set(header["tensor_names"]),
        "portable tensor key set changed",
    )
    observed_specs: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    total = 0
    try:
        with safe_open(staged / subject["tensors_file"], framework="numpy") as handle:
            _reject(
                handle.metadata()
                == {"format": "multitown-a24-portable-fixture-v1", "synthetic": "true"},
                "Safetensors metadata changed",
            )
            _reject(set(handle.keys()) == set(required), "Safetensors key set changed")
            for name in sorted(handle.keys()):
                array = np.asarray(handle.get_tensor(name))
                contract = required[name]
                expected = specs[name]
                _reject(
                    type(expected) is dict
                    and set(expected)
                    == {"dtype", "shape", "raw_little_endian_sha256", "bytes"}
                    and expected["dtype"] == contract["dtype"]
                    and expected["shape"] == contract["shape"]
                    and type(expected["bytes"]) is int
                    and expected["bytes"] > 0
                    and type(expected["raw_little_endian_sha256"]) is str
                    and _HEX_SHA256.fullmatch(expected["raw_little_endian_sha256"])
                    and expected["raw_little_endian_sha256"]
                    == case["tensor_raw_sha256"][name]
                    and str(array.dtype) == contract["dtype"]
                    and list(array.shape) == contract["shape"]
                    and (
                        not np.issubdtype(array.dtype, np.floating)
                        or bool(np.isfinite(array).all())
                    )
                    and int(array.nbytes) == expected["bytes"]
                    and _tensor_raw_sha(array) == expected["raw_little_endian_sha256"],
                    f"portable tensor changed: {name}",
                )
                total += int(array.nbytes)
                observed_specs[name] = dict(expected)
                arrays[name] = np.array(array, copy=True)
    except PortableVerificationRejected:
        raise
    except (OSError, SafetensorError, ValueError) as exc:
        raise PortableVerificationRejected("cannot safely load Safetensors") from exc
    _reject(total <= policy["limits"]["tensor_max_bytes"], "tensor bytes exceed policy")
    return (
        {
            "tensor_count": len(observed_specs),
            "tensor_bytes": total,
            "tensor_specs_sha256": _canonical_digest(
                observed_specs, domain="multitown-a24-portable-tensor-specs-v1"
            ),
            "safetensors_file_sha256": _sha256(tensor_payload),
            "safetensors_header_bytes": header["header_bytes"],
            "safetensors_header_sha256": header["header_sha256"],
            "safetensors_structure_valid": True,
        },
        arrays,
    )


def _ordered_f32(bits: np.ndarray) -> np.ndarray:
    unsigned = np.asarray(bits, dtype=np.uint32)
    negative = (unsigned & np.uint32(0x80000000)) != 0
    return np.where(
        negative,
        np.bitwise_not(unsigned),
        unsigned | np.uint32(0x80000000),
    ).astype(np.uint64)


def _ulp_max(actual: np.ndarray, expected: np.ndarray) -> int:
    actual_bits = np.asarray(actual, dtype="<f4").view("<u4")
    expected_bits = np.asarray(expected, dtype="<f4").view("<u4")
    actual_ordered = _ordered_f32(actual_bits)
    expected_ordered = _ordered_f32(expected_bits)
    distance = np.where(
        actual_ordered >= expected_ordered,
        actual_ordered - expected_ordered,
        expected_ordered - actual_ordered,
    )
    return int(distance.max(initial=np.uint64(0)))


def _array_exact_equal(left: np.ndarray, right: np.ndarray) -> bool:
    """Match PQ-1 ``torch.equal`` semantics for finite float32 arrays.

    Artifact loading rejects non-finite tensors before this comparison.  A
    numerical comparison is therefore exact while still treating ``+0.0``
    and ``-0.0`` as equal, as the frozen PQ-1 row-wise gate does.
    """

    return bool(
        np.array_equal(
            np.asarray(left, dtype="<f4"),
            np.asarray(right, dtype="<f4"),
            equal_nan=False,
        )
    )


def _difference_summary(
    actual32: np.ndarray,
    expected32: np.ndarray,
    *,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    actual_f32 = np.asarray(actual32, dtype=np.float32)
    expected_f32 = np.asarray(expected32, dtype=np.float32)
    actual = actual_f32.astype(np.float64)
    expected = expected_f32.astype(np.float64)
    absolute = np.abs(actual - expected)
    full_batch = policy["numerical_policy"]["full_batch"]
    absolute_f32 = np.abs(np.subtract(actual_f32, expected_f32, dtype=np.float32))
    threshold_f32 = np.add(
        np.float32(full_batch["atol_decimal"]),
        np.multiply(
            np.float32(full_batch["rtol_decimal"]),
            np.abs(expected_f32),
            dtype=np.float32,
        ),
        dtype=np.float32,
    )
    quantiles = np.quantile(
        absolute,
        policy["numerical_policy"]["summary"]["quantiles"],
        method=policy["numerical_policy"]["summary"]["quantile_method"],
    )
    denominator = np.maximum(np.abs(expected), float(np.finfo(np.float32).tiny))
    tolerance_ratio = absolute / threshold_f32.astype(np.float64)
    return {
        "within_frozen_tolerance": bool(
            np.less_equal(absolute_f32, threshold_f32).all()
        ),
        "max_abs": float(absolute.max()),
        "p50_abs": float(quantiles[0]),
        "p95_abs": float(quantiles[1]),
        "p99_abs": float(quantiles[2]),
        "max_relative": float((absolute / denominator).max()),
        "max_tolerance_ratio": float(tolerance_ratio.max()),
        "max_ulp": _ulp_max(actual32, expected32),
    }


def _reason_codes(gates: Mapping[str, bool]) -> list[str]:
    mapping = (
        ("rowwise_log_probability_exact", "ROWWISE_LOG_PROBABILITY_NOT_EXACT"),
        ("rowwise_value_exact", "ROWWISE_VALUE_NOT_EXACT"),
        ("full_batch_log_within_frozen_tolerance", "FULL_BATCH_LOG_TOLERANCE_EXCEEDED"),
        (
            "full_batch_value_within_frozen_tolerance",
            "FULL_BATCH_VALUE_TOLERANCE_EXCEEDED",
        ),
        (
            "probability_ratio_drift_within_threshold",
            "PROBABILITY_RATIO_DRIFT_EXCEEDED",
        ),
    )
    failures = [reason for gate, reason in mapping if not gates[gate]]
    return failures or ["ALL_NUMERICAL_GATES_PASSED"]


def _numerical_evaluation(
    arrays: Mapping[str, np.ndarray],
    policy: Mapping[str, Any],
    *,
    availability: str,
) -> tuple[dict[str, Any], str, list[str]]:
    if availability == "actual-tensors-withheld":
        return (
            {
                "evaluation_status": "UNEVALUATED",
                "outcome": "UNEVALUATED",
                "reason_codes": ["FULL_BATCH_ACTUAL_TENSORS_WITHHELD"],
                "raw_numerical_recomputation": False,
            },
            "UNEVALUATED",
            ["FULL_BATCH_ACTUAL_TENSORS_WITHHELD"],
        )
    expected_log = arrays["expected_log_probability"]
    expected_value = arrays["expected_value"]
    actual_log = arrays["full_batch_log_probability"]
    actual_value = arrays["full_batch_value"]
    rowwise_log = arrays["rowwise_log_probability"]
    rowwise_value = arrays["rowwise_value"]
    log_summary = _difference_summary(actual_log, expected_log, policy=policy)
    value_summary = _difference_summary(actual_value, expected_value, policy=policy)
    log_delta_f32 = np.subtract(
        np.asarray(actual_log, dtype=np.float32),
        np.asarray(expected_log, dtype=np.float32),
        dtype=np.float32,
    )
    ratio_drift = np.abs(
        np.subtract(
            np.exp(log_delta_f32, dtype=np.float32),
            np.float32(1.0),
            dtype=np.float32,
        )
    )
    ratio_threshold = float(
        policy["numerical_policy"]["probability_ratio"]["threshold_decimal"]
    )
    legacy = policy["numerical_policy"]["legacy_diagnostic"]
    legacy_rtol = float(legacy["rtol_decimal"])
    legacy_atol = float(legacy["atol_decimal"])
    log_actual64 = np.asarray(actual_log, dtype=np.float32).astype(np.float64)
    log_expected64 = np.asarray(expected_log, dtype=np.float32).astype(np.float64)
    log_absolute = np.abs(log_actual64 - log_expected64)
    legacy_threshold = np.add(
        np.float32(legacy_atol),
        np.multiply(
            np.float32(legacy_rtol),
            np.abs(np.asarray(expected_log, dtype=np.float32)),
            dtype=np.float32,
        ),
        dtype=np.float32,
    )
    legacy_ratio = log_absolute / legacy_threshold
    gates = {
        "rowwise_log_probability_exact": _array_exact_equal(rowwise_log, expected_log),
        "rowwise_value_exact": _array_exact_equal(rowwise_value, expected_value),
        "full_batch_log_within_frozen_tolerance": log_summary[
            "within_frozen_tolerance"
        ],
        "full_batch_value_within_frozen_tolerance": value_summary[
            "within_frozen_tolerance"
        ],
        "probability_ratio_drift_within_threshold": bool(
            np.less_equal(ratio_drift, ratio_threshold).all()
        ),
    }
    reasons = _reason_codes(gates)
    passed = all(gates.values())
    outcome = "PASSED" if passed else "FAILED"
    population_hasher = hashlib.sha256()
    for name in (
        "expected_log_probability",
        "expected_value",
        "full_batch_log_probability",
        "full_batch_value",
        "rowwise_log_probability",
        "rowwise_value",
    ):
        population_hasher.update(name.encode("ascii") + b"\0")
        population_hasher.update(
            np.ascontiguousarray(arrays[name], dtype="<f4").tobytes(order="C")
        )
    numerical_policy = policy["numerical_policy"]
    return (
        {
            "evaluation_status": "COMPLETED",
            "outcome": outcome,
            "reason_codes": reasons,
            "raw_numerical_recomputation": True,
            "sample_count": int(np.asarray(actual_log).size),
            "population_sha256": population_hasher.hexdigest(),
            "algorithms": {
                "comparison": numerical_policy["full_batch"]["comparison"],
                "reference_operand": numerical_policy["full_batch"][
                    "reference_operand"
                ],
                "input_dtype": numerical_policy["summary"]["input_dtype"],
                "accumulator_dtype": numerical_policy["summary"]["accumulator_dtype"],
                "gate_evaluation_dtype": numerical_policy["full_batch"][
                    "evaluation_dtype"
                ],
                "quantile_method": numerical_policy["summary"]["quantile_method"],
                "quantile_algorithm": numerical_policy["summary"]["quantile_algorithm"],
                "ulp_algorithm": numerical_policy["summary"]["ulp_algorithm"],
                "nonfinite_policy": numerical_policy["nonfinite_policy"],
                "equal_nan": numerical_policy["equal_nan"],
            },
            "gates": gates,
            "log_probability": log_summary,
            "value": value_summary,
            "probability_ratio": {
                "formula": numerical_policy["probability_ratio"]["formula"],
                "max_drift": float(ratio_drift.max()),
                "within_threshold": gates["probability_ratio_drift_within_threshold"],
            },
            "legacy_log_probability_diagnostic": {
                "acceptance_gate": False,
                "exceedances": int(np.count_nonzero(log_absolute > legacy_threshold)),
                "max_tolerance_ratio": float(legacy_ratio.max()),
            },
        },
        outcome,
        reasons,
    )


def _verifier_identity() -> dict[str, Any]:
    template_raw = _resource_bytes(_TEMPLATE_RESOURCE, limit=1024 * 1024)
    return {
        "verifier_id": "multitown-a24-portable-verifier",
        "verifier_version": PORTABLE_VERIFIER_VERSION,
        "source_files": {
            "multitown/a24_contract.py": _sha256(
                Path(a24_contract.__file__).read_bytes()
            ),
            "multitown/a24_portable_fixture.py": _sha256(
                Path(__file__).read_bytes()
            ),
        },
        "template_sha256": _sha256(template_raw),
        "canonicalization_id": CANONICALIZATION_ID,
        "receipt_id_domain": RECEIPT_ID_DOMAIN,
        "receipt_schema": _receipt_schema_identity(),
    }


def _runtime_identity() -> dict[str, str]:
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "numpy_version": np.__version__,
        "rfc8785_version": rfc8785.__version__,
        "safetensors_version": safetensors.__version__,
    }


def _build_receipt(core: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": RECEIPT_VERSION,
        "receipt_id": _canonical_digest(core, domain=RECEIPT_ID_DOMAIN),
        "core": dict(core),
        "metadata": {
            "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "receipt_id_scope": (
                "domain-separated RFC 8785 JCS core; metadata excluded"
            ),
            "signature": None,
            "signature_scheme": None,
            "self_reported_not_an_external_attestation": True,
            "not_in_toto_or_dsse_envelope": True,
        },
    }


def _error_receipt(reason_code: str) -> dict[str, Any]:
    _reject(
        reason_code in {"CONCURRENT_OBSERVATION", "OPERATIONAL_ERROR"},
        "portable error reason code changed",
    )
    core = {
        "profile": PROFILE,
        "subject": None,
        "policy": None,
        "verifier": _verifier_identity(),
        "runtime": _runtime_identity(),
        "evaluation": {
            "evaluation_status": "ERROR",
            "outcome": "ERROR",
            "reason_codes": [reason_code],
        },
        "tensor_summary": None,
        "numerical": {
            "evaluation_status": "ERROR",
            "outcome": "ERROR",
            "reason_codes": [reason_code],
            "raw_numerical_recomputation": False,
        },
        "capabilities": {name: False for name in sorted(_CAPABILITY_KEYS)},
        "limitations": list(_LIMITATIONS),
    }
    receipt = _build_receipt(core)
    validate_receipt(receipt)
    return receipt


def _stage(first: Mapping[str, bytes], staged: Path) -> None:
    os.chmod(staged, 0o700)
    for name, payload in first.items():
        destination = staged / name
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())


def verify_fixture(artifact: Path, policy_path: Path) -> dict[str, Any]:
    packaged_policy_raw, _ = _packaged_policy()
    policy_raw = _bounded_regular_read(
        policy_path, limit=_MAX_POLICY_BYTES, label="portable policy"
    )
    _reject(
        policy_raw == packaged_policy_raw, "portable policy is not the frozen policy"
    )
    policy = _strict_object(policy_raw, label="portable policy")
    _validate_policy_object(policy)
    first, snapshot_sha = _snapshot(artifact, limits=policy["limits"])
    manifest = _strict_object(first["manifest.json"], label="portable manifest")
    _validate_manifest(manifest, first)
    case = policy["fixture_cases"][manifest["variant"]]
    _reject(
        _sha256(first["subject.json"]) == case["subject_sha256"]
        and _sha256(first["numerical-diagnostic.json"]) == case["diagnostic_sha256"],
        "portable artifact metadata does not match frozen case",
    )
    subject = _strict_object(first["subject.json"], label="portable subject")
    _validate_subject(subject, variant=manifest["variant"])
    diagnostic = _strict_object(
        first["numerical-diagnostic.json"], label="portable numerical diagnostic"
    )
    availability = _validate_diagnostic(diagnostic, policy)
    _reject(
        (subject["variant"] == "unevaluated")
        == (availability == "actual-tensors-withheld"),
        "portable variant and availability disagree",
    )
    _reject(
        availability == case["availability"],
        "portable availability does not match frozen case",
    )
    with tempfile.TemporaryDirectory(prefix="multitown-a24-portable-") as temporary:
        staged = Path(temporary)
        _stage(first, staged)
        tensor_summary, arrays = _load_tensors(
            staged,
            first["numerics.safetensors"],
            subject,
            policy,
            availability=availability,
            case=case,
        )
        numerical, outcome, reasons = _numerical_evaluation(
            arrays, policy, availability=availability
        )
        _reject(
            outcome == case["expected_outcome"],
            "portable recomputed outcome does not match frozen case",
        )
    second, second_sha = _snapshot(artifact, limits=policy["limits"])
    policy_after = _bounded_regular_read(
        policy_path, limit=_MAX_POLICY_BYTES, label="portable policy"
    )
    if first != second or snapshot_sha != second_sha or policy_raw != policy_after:
        raise PortableConcurrentObservation("portable subject or policy changed")
    core = {
        "profile": PROFILE,
        "subject": {
            "fixture_id": subject["fixture_id"],
            "variant": subject["variant"],
            **_SUBJECT_BOUNDARY,
            "artifact_snapshot_sha256": snapshot_sha,
            "manifest_sha256": _sha256(first["manifest.json"]),
        },
        "policy": {
            "policy_id": policy["policy_id"],
            "policy_sha256": _sha256(policy_raw),
        },
        "verifier": _verifier_identity(),
        "runtime": _runtime_identity(),
        "evaluation": {
            "evaluation_status": numerical["evaluation_status"],
            "outcome": outcome,
            "reason_codes": reasons,
        },
        "tensor_summary": tensor_summary,
        "numerical": numerical,
        "capabilities": {
            "portable_tensor_structure_verified": True,
            "raw_numerical_recomputation": numerical["raw_numerical_recomputation"],
            "producer_exact_initialization_reconstruction": False,
            "transition_derivation_replayed": False,
            "inference_reexecution": False,
            "optimizer_update_reexecution": False,
            "results_reproduced": False,
            "formal_evidence_accepted": False,
        },
        "limitations": list(_LIMITATIONS),
    }
    receipt = _build_receipt(core)
    validate_receipt(receipt)
    return receipt


def _is_finite_number(value: Any, *, nonnegative: bool = False) -> bool:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        return False
    return not nonnegative or float(value) >= 0


def _validate_difference_summary(value: Any, *, label: str) -> None:
    _reject(
        type(value) is dict
        and set(value)
        == {
            "within_frozen_tolerance",
            "max_abs",
            "p50_abs",
            "p95_abs",
            "p99_abs",
            "max_relative",
            "max_tolerance_ratio",
            "max_ulp",
        }
        and type(value["within_frozen_tolerance"]) is bool
        and all(
            _is_finite_number(value[name], nonnegative=True)
            for name in (
                "max_abs",
                "p50_abs",
                "p95_abs",
                "p99_abs",
                "max_relative",
                "max_tolerance_ratio",
            )
        )
        and type(value["max_ulp"]) is int
        and value["max_ulp"] >= 0
        and value["p50_abs"] <= value["p95_abs"] <= value["p99_abs"] <= value["max_abs"]
        and value["within_frozen_tolerance"] == (value["max_tolerance_ratio"] <= 1.0),
        f"portable receipt difference summary changed: {label}",
    )


def _validate_numerical_receipt(numerical: Any) -> None:
    _reject(type(numerical) is dict, "portable receipt numerical result changed")
    if numerical.get("evaluation_status") == "ERROR":
        _reject(
            type(numerical.get("reason_codes")) is list
            and len(numerical["reason_codes"]) == 1
            and numerical["reason_codes"][0]
            in {"CONCURRENT_OBSERVATION", "OPERATIONAL_ERROR"}
            and numerical
            == {
                "evaluation_status": "ERROR",
                "outcome": "ERROR",
                "reason_codes": numerical["reason_codes"],
                "raw_numerical_recomputation": False,
            },
            "portable error receipt changed",
        )
        return
    if numerical.get("evaluation_status") == "UNEVALUATED":
        _reject(
            numerical
            == {
                "evaluation_status": "UNEVALUATED",
                "outcome": "UNEVALUATED",
                "reason_codes": ["FULL_BATCH_ACTUAL_TENSORS_WITHHELD"],
                "raw_numerical_recomputation": False,
            },
            "portable unevaluated receipt changed",
        )
        return
    _reject(
        set(numerical)
        == {
            "evaluation_status",
            "outcome",
            "reason_codes",
            "raw_numerical_recomputation",
            "sample_count",
            "population_sha256",
            "algorithms",
            "gates",
            "log_probability",
            "value",
            "probability_ratio",
            "legacy_log_probability_diagnostic",
        }
        and numerical["evaluation_status"] == "COMPLETED"
        and numerical["outcome"] in {"PASSED", "FAILED"}
        and type(numerical["reason_codes"]) is list
        and all(type(reason) is str for reason in numerical["reason_codes"])
        and len(numerical["reason_codes"]) == len(set(numerical["reason_codes"]))
        and numerical["raw_numerical_recomputation"] is True
        and type(numerical["sample_count"]) is int
        and numerical["sample_count"] == 4
        and type(numerical["population_sha256"]) is str
        and _HEX_SHA256.fullmatch(numerical["population_sha256"]),
        "portable completed numerical receipt changed",
    )
    algorithms = numerical["algorithms"]
    _reject(
        algorithms
        == {
            "comparison": "abs_actual_expected_le_atol_plus_rtol_abs_expected",
            "reference_operand": "expected",
            "input_dtype": "float32",
            "accumulator_dtype": "float64",
            "gate_evaluation_dtype": "float32",
            "quantile_method": "linear",
            "quantile_algorithm": "Hyndman-Fan-type-7",
            "ulp_algorithm": "ordered-ieee754-binary32-distance-v1",
            "nonfinite_policy": "reject",
            "equal_nan": False,
        },
        "portable receipt algorithm identity changed",
    )
    gates = numerical["gates"]
    _reject(
        type(gates) is dict
        and set(gates)
        == {
            "rowwise_log_probability_exact",
            "rowwise_value_exact",
            "full_batch_log_within_frozen_tolerance",
            "full_batch_value_within_frozen_tolerance",
            "probability_ratio_drift_within_threshold",
        }
        and all(type(value) is bool for value in gates.values()),
        "portable receipt gates changed",
    )
    expected_reasons = _reason_codes(gates)
    expected_outcome = "PASSED" if all(gates.values()) else "FAILED"
    _reject(
        len(numerical["reason_codes"]) == len(expected_reasons)
        and set(numerical["reason_codes"]) == set(expected_reasons)
        and numerical["outcome"] == expected_outcome,
        "portable receipt gate aggregation changed",
    )
    _validate_difference_summary(numerical["log_probability"], label="log_probability")
    _validate_difference_summary(numerical["value"], label="value")
    _reject(
        numerical["log_probability"]["within_frozen_tolerance"]
        == gates["full_batch_log_within_frozen_tolerance"]
        and numerical["value"]["within_frozen_tolerance"]
        == gates["full_batch_value_within_frozen_tolerance"],
        "portable receipt summary and gate changed",
    )
    ratio = numerical["probability_ratio"]
    _, frozen_policy = _packaged_policy()
    ratio_threshold = float(
        frozen_policy["numerical_policy"]["probability_ratio"]["threshold_decimal"]
    )
    _reject(
        type(ratio) is dict
        and set(ratio) == {"formula", "max_drift", "within_threshold"}
        and ratio["formula"] == "max_abs_exp_actual_log_minus_expected_log_minus_one"
        and _is_finite_number(ratio["max_drift"], nonnegative=True)
        and type(ratio["within_threshold"]) is bool
        and ratio["within_threshold"]
        == gates["probability_ratio_drift_within_threshold"]
        and ratio["within_threshold"] == (ratio["max_drift"] <= ratio_threshold),
        "portable receipt probability-ratio result changed",
    )
    legacy = numerical["legacy_log_probability_diagnostic"]
    _reject(
        type(legacy) is dict
        and set(legacy)
        == {
            "acceptance_gate",
            "exceedances",
            "max_tolerance_ratio",
        }
        and legacy["acceptance_gate"] is False
        and type(legacy["exceedances"]) is int
        and legacy["exceedances"] >= 0
        and _is_finite_number(legacy["max_tolerance_ratio"], nonnegative=True)
        and (legacy["exceedances"] == 0) == (legacy["max_tolerance_ratio"] <= 1.0),
        "portable receipt legacy diagnostic changed",
    )


def _validate_runtime(runtime: Any) -> None:
    _reject(
        type(runtime) is dict
        and set(runtime)
        == {
            "python_implementation",
            "python_version",
            "platform_system",
            "platform_machine",
            "numpy_version",
            "rfc8785_version",
            "safetensors_version",
        }
        and all(type(value) is str and bool(value) for value in runtime.values()),
        "portable receipt runtime identity changed",
    )


def _validate_metadata(metadata: Any) -> None:
    _reject(
        type(metadata) is dict
        and set(metadata)
        == {
            "generated_at_utc",
            "receipt_id_scope",
            "signature",
            "signature_scheme",
            "self_reported_not_an_external_attestation",
            "not_in_toto_or_dsse_envelope",
        }
        and type(metadata["generated_at_utc"]) is str
        and metadata["receipt_id_scope"]
        == "domain-separated RFC 8785 JCS core; metadata excluded"
        and metadata["signature"] is None
        and metadata["signature_scheme"] is None
        and metadata["self_reported_not_an_external_attestation"] is True
        and metadata["not_in_toto_or_dsse_envelope"] is True,
        "portable receipt unsigned metadata changed",
    )
    try:
        generated = datetime.fromisoformat(metadata["generated_at_utc"])
    except ValueError as exc:
        raise PortableVerificationRejected(
            "portable receipt timestamp changed"
        ) from exc
    _reject(
        generated.tzinfo == UTC and metadata["generated_at_utc"].endswith("Z"),
        "portable receipt timestamp is not canonical UTC",
    )


def validate_receipt(receipt: Mapping[str, Any]) -> None:
    _reject(
        type(receipt) is dict
        and set(receipt) == {"schema_version", "receipt_id", "core", "metadata"}
        and receipt["schema_version"] == RECEIPT_VERSION
        and type(receipt["receipt_id"]) is str
        and _HEX_SHA256.fullmatch(receipt["receipt_id"]),
        "portable receipt record changed",
    )
    core = receipt["core"]
    _reject(
        type(core) is dict
        and set(core)
        == {
            "profile",
            "subject",
            "policy",
            "verifier",
            "runtime",
            "evaluation",
            "tensor_summary",
            "numerical",
            "capabilities",
            "limitations",
        }
        and core["profile"] == PROFILE
        and receipt["receipt_id"] == _canonical_digest(core, domain=RECEIPT_ID_DOMAIN),
        "portable receipt core changed",
    )
    verifier = core["verifier"]
    expected_verifier = _verifier_identity()
    _reject(
        type(verifier) is dict and verifier == expected_verifier,
        "portable receipt verifier identity changed",
    )
    _validate_runtime(core["runtime"])
    _validate_numerical_receipt(core["numerical"])
    evaluation = core["evaluation"]
    numerical = core["numerical"]
    _reject(
        type(evaluation) is dict
        and set(evaluation) == {"evaluation_status", "outcome", "reason_codes"}
        and evaluation["evaluation_status"] == numerical["evaluation_status"]
        and evaluation["outcome"] == numerical["outcome"]
        and type(evaluation["reason_codes"]) is list
        and len(evaluation["reason_codes"]) == len(numerical["reason_codes"])
        and set(evaluation["reason_codes"]) == set(numerical["reason_codes"])
        and evaluation["evaluation_status"] in EVALUATION_STATUSES
        and evaluation["outcome"] in OUTCOMES,
        "portable receipt evaluation changed",
    )
    capabilities = core["capabilities"]
    _reject(
        type(capabilities) is dict
        and set(capabilities) == _CAPABILITY_KEYS
        and all(type(value) is bool for value in capabilities.values())
        and core["limitations"] == _LIMITATIONS,
        "portable receipt capabilities changed",
    )
    if evaluation["outcome"] == "ERROR":
        _reject(
            core["subject"] is None
            and core["policy"] is None
            and core["tensor_summary"] is None
            and all(value is False for value in capabilities.values()),
            "portable error receipt conclusion boundary changed",
        )
        _validate_metadata(receipt["metadata"])
        return
    subject = core["subject"]
    _reject(
        type(subject) is dict
        and set(subject)
        == {
            "fixture_id",
            "variant",
            *_SUBJECT_BOUNDARY,
            "artifact_snapshot_sha256",
            "manifest_sha256",
        }
        and subject["fixture_id"] == FIXTURE_ID
        and subject["variant"] in VARIANTS
        and all(subject[key] == value for key, value in _SUBJECT_BOUNDARY.items())
        and all(
            type(subject[name]) is str and _HEX_SHA256.fullmatch(subject[name])
            for name in ("artifact_snapshot_sha256", "manifest_sha256")
        ),
        "portable receipt subject changed",
    )
    policy = core["policy"]
    packaged_policy_raw, packaged_policy = _packaged_policy()
    _reject(
        policy
        == {
            "policy_id": packaged_policy["policy_id"],
            "policy_sha256": _sha256(packaged_policy_raw),
        },
        "portable receipt policy changed",
    )
    _reject(
        evaluation["outcome"]
        == packaged_policy["fixture_cases"][subject["variant"]]["expected_outcome"],
        "portable receipt outcome does not match frozen case",
    )
    tensor_summary = core["tensor_summary"]
    _reject(
        type(tensor_summary) is dict
        and set(tensor_summary)
        == {
            "tensor_count",
            "tensor_bytes",
            "tensor_specs_sha256",
            "safetensors_file_sha256",
            "safetensors_header_bytes",
            "safetensors_header_sha256",
            "safetensors_structure_valid",
        }
        and type(tensor_summary["tensor_count"]) is int
        and tensor_summary["tensor_count"] > 0
        and type(tensor_summary["tensor_bytes"]) is int
        and tensor_summary["tensor_bytes"] > 0
        and type(tensor_summary["safetensors_header_bytes"]) is int
        and tensor_summary["safetensors_header_bytes"] > 0
        and all(
            type(tensor_summary[name]) is str
            and _HEX_SHA256.fullmatch(tensor_summary[name])
            for name in (
                "tensor_specs_sha256",
                "safetensors_file_sha256",
                "safetensors_header_sha256",
            )
        )
        and tensor_summary["safetensors_structure_valid"] is True,
        "portable receipt tensor summary changed",
    )
    _reject(
        capabilities["portable_tensor_structure_verified"] is True
        and capabilities["raw_numerical_recomputation"]
        is numerical["raw_numerical_recomputation"]
        and all(
            capabilities[name] is False
            for name in _CAPABILITY_KEYS
            - {"portable_tensor_structure_verified", "raw_numerical_recomputation"}
        ),
        "portable receipt capabilities changed",
    )
    _validate_metadata(receipt["metadata"])


def validate_receipt_file(path: Path) -> dict[str, Any]:
    raw = _bounded_regular_read(
        path, limit=_MAX_RECEIPT_BYTES, label="portable receipt"
    )
    receipt = _strict_object(raw, label="portable receipt")
    validate_receipt(receipt)
    return receipt


def reverify_receipt(
    artifact: Path, policy_path: Path, receipt_path: Path
) -> dict[str, Any]:
    stored = validate_receipt_file(receipt_path)
    _reject(
        stored["core"]["evaluation"]["outcome"] != "ERROR",
        "an operational-error receipt has no evaluated subject to reverify",
    )
    recomputed = verify_fixture(artifact, policy_path)
    _reject(
        stored["receipt_id"] == recomputed["receipt_id"]
        and stored["core"] == recomputed["core"],
        "receipt core does not match artifact reverification",
    )
    return stored


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    materialize = subparsers.add_parser(
        "materialize", help="create a synthetic fixture"
    )
    materialize.add_argument("--output", required=True, type=Path)
    materialize.add_argument("--policy-out", required=True, type=Path)
    materialize.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    verify = subparsers.add_parser("verify", help="verify a synthetic fixture")
    verify.add_argument("--artifact", required=True, type=Path)
    verify.add_argument("--policy", required=True, type=Path)
    verify.add_argument("--expect-outcome", choices=sorted(SEMANTIC_OUTCOMES))
    validate = subparsers.add_parser(
        "validate-receipt",
        help="check receipt structure, self-consistency, and installed profile",
    )
    validate.add_argument("--receipt", required=True, type=Path)
    reverify = subparsers.add_parser(
        "reverify-receipt",
        help="recompute an artifact and compare its deterministic receipt core",
    )
    reverify.add_argument("--artifact", required=True, type=Path)
    reverify.add_argument("--policy", required=True, type=Path)
    reverify.add_argument("--receipt", required=True, type=Path)
    return parser


def _write_stderr_best_effort(message: str) -> None:
    try:
        sys.stderr.write(message + "\n")
    except Exception:  # noqa: BLE001, S110 - stderr failure cannot change CLI exit.
        pass


def _emit_error_receipt_best_effort(command: str, reason_code: str) -> None:
    if command not in {"verify", "reverify-receipt"}:
        return
    try:
        sys.stdout.buffer.write(canonical_json_bytes(_error_receipt(reason_code)))
    except Exception as exc:  # noqa: BLE001 - secondary reporting must be contained.
        _write_stderr_best_effort(
            f"ERROR_RECEIPT_UNAVAILABLE: {type(exc).__name__}: {exc}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "materialize":
            result = materialize_fixture(
                args.output, args.policy_out, variant=args.variant
            )
            sys.stdout.buffer.write(canonical_json_bytes(result))
            return 0
        if args.command == "validate-receipt":
            receipt = validate_receipt_file(args.receipt)
            result = {
                "receipt_structurally_valid": True,
                "receipt_self_consistent": True,
                "matches_installed_profile": True,
                "artifact_recomputed": False,
                "authenticated": False,
                "receipt_id": receipt["receipt_id"],
                "outcome": receipt["core"]["evaluation"]["outcome"],
            }
            sys.stdout.buffer.write(canonical_json_bytes(result))
            return EXIT_PASSED
        if args.command == "reverify-receipt":
            receipt = reverify_receipt(args.artifact, args.policy, args.receipt)
            result = {
                "artifact_recomputed": True,
                "receipt_core_matched": True,
                "authenticated": False,
                "receipt_id": receipt["receipt_id"],
                "outcome": receipt["core"]["evaluation"]["outcome"],
            }
            sys.stdout.buffer.write(canonical_json_bytes(result))
            return EXIT_PASSED
        receipt = verify_fixture(args.artifact, args.policy)
        actual = receipt["core"]["evaluation"]["outcome"]
        if args.expect_outcome is not None:
            _reject(
                actual == args.expect_outcome,
                "portable outcome differs from expectation",
            )
        sys.stdout.buffer.write(canonical_json_bytes(receipt))
        if args.expect_outcome is not None:
            return EXIT_PASSED
        return {
            "PASSED": EXIT_PASSED,
            "FAILED": EXIT_FAILED,
            "UNEVALUATED": EXIT_UNEVALUATED,
        }[actual]
    except PortableVerificationRejected as exc:
        _write_stderr_best_effort(f"REJECTED: {exc}")
        return EXIT_REJECTED
    except PortableConcurrentObservation as exc:
        _emit_error_receipt_best_effort(args.command, "CONCURRENT_OBSERVATION")
        _write_stderr_best_effort(f"CONCURRENT: {exc}")
        return EXIT_CONCURRENT
    except (OSError, SafetensorError, TypeError, ValueError) as exc:
        _emit_error_receipt_best_effort(args.command, "OPERATIONAL_ERROR")
        _write_stderr_best_effort(f"ERROR: {type(exc).__name__}: {exc}")
        return EXIT_ERROR
    except Exception as exc:  # noqa: BLE001 - CLI must not alias crashes to FAILED.
        _emit_error_receipt_best_effort(args.command, "OPERATIONAL_ERROR")
        _write_stderr_best_effort(f"INTERNAL: {type(exc).__name__}: {exc}")
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
