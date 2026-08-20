"""Artifact-only A25 portable conformance checker.

This reference checker intentionally imports only the Python standard library.
It treats a four-file directory as hostile input, parses a bounded Safetensors
subset without producer code, and derives all common-prestate and gradient
gates from raw typed bytes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import stat
import struct
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, Self

CHECKER_NAME = "multitown-a25-portable-checker"
CHECKER_VERSION = "1.0.0"
MANIFEST_VERSION = "multitown-a25-portable-manifest-v1"
CONTRACT_VERSION = "multitown-a25-portable-contract-v1"
REPORT_VERSION = "multitown-a25-portable-report-v1"
PROFILE = "multitown-a25-common-state-gradient-micro-v1"
BUNDLE_DOMAIN = b"multitown:a25:portable-bundle:v1\0"

CORE_FILES = frozenset(
    {
        "manifest.json",
        "contract.json",
        "common-state.safetensors",
        "observations.safetensors",
    }
)
MAX_JSON_BYTES = 1024 * 1024
MAX_CORE_FILE_BYTES = 16 * 1024 * 1024
MAX_TOTAL_TENSOR_PAYLOAD_BYTES = 16 * 1024 * 1024
MAX_SAFETENSORS_HEADER_BYTES = 1024 * 1024
MAX_TENSORS = 4096
MAX_RANK = 8
MAX_DIMENSION = 1024 * 1024
MAX_JSON_DEPTH = 64
MAX_JSON_ITEMS = 32768
MAX_JSON_STRING_BYTES = 1024 * 1024
MAX_SAFE_INTEGER = (1 << 53) - 1

SCHEMA_DESCRIPTORS = (
    {
        "name": "manifest",
        "id": "https://multitown.dev/schemas/a25-portable-manifest-v1.json",
        "sha256": "3c5379a159121ccc417b69ba78197014c970eeaaae17d06537cb75f4cbd5a49d",
    },
    {
        "name": "contract",
        "id": "https://multitown.dev/schemas/a25-portable-contract-v1.json",
        "sha256": "8de2809ffc1256adcd0e9c6d98e8f5e98b774530ebb35e5a095514df93500f28",
    },
    {
        "name": "report",
        "id": "https://multitown.dev/schemas/a25-portable-report-v1.json",
        "sha256": "68774f2f4b3a0ae53ec6904dd31cea1deec5aeb199839c7e9aa9ee57a7e717de",
    },
)

EXPECTED_CLAIMS = {
    "typed_payload_rehashed": True,
    "per_cell_prestate_recomputed": True,
    "gradient_decomposition_recomputed": True,
    "torch_rng_sequence_reproduced": False,
    "autograd_reexecuted": False,
    "optimizer_step_reexecuted": False,
    "authenticated_provenance": False,
    "freshness_attested": False,
}
EXPECTED_DIGEST_SPEC = {
    "algorithm": "sha256",
    "record_encoding": "multitown-a25-semantic-tensor-record-v1",
    "name_order": "bytewise-ascii",
    "dtype_codes": {"F32": 1, "I64": 2, "U8": 3},
    "domains": {
        "model": "multitown:a25:model-state:v1\0",
        "optimizer": "multitown:a25:optimizer-state:v1\0",
        "rng": "multitown:a25:explicit-update-rng-state:v1\0",
    },
}
EXPECTED_NUMERIC_SPEC = {
    "beta_encoding": "ieee754-binary32-big-endian-hex",
    "tolerance_encoding": "ieee754-binary64-big-endian-hex",
    "atol_f64_hex": "3eb0000000000000",
    "rtol_f64_hex": "3ef0000000000000",
    "evaluation": (
        "round_f32(round_f32(base)+round_f32(round_f32(beta)*round_f32(aux)))"
    ),
    "fused_multiply_add": False,
    "subnormal_mode": "preserve",
    "negative_zero": "allowed",
    "finite_only": True,
    "beta_zero_aux": "forbidden",
    "beta_positive_aux": "required",
    "beta_zero_total": "bitwise-equal-base",
}
EXPECTED_LIMITS = {
    "max_core_file_bytes": MAX_CORE_FILE_BYTES,
    "max_total_tensor_payload_bytes": MAX_TOTAL_TENSOR_PAYLOAD_BYTES,
    "max_safetensors_header_bytes": MAX_SAFETENSORS_HEADER_BYTES,
    "max_tensors": MAX_TENSORS,
    "max_rank": MAX_RANK,
    "max_dimension": MAX_DIMENSION,
    "max_cells": 64,
    "max_state_components": 1024,
    "max_gradient_parameters": 1024,
}
EXPECTED_INVARIANTS = [
    "common-prestate-component-digests-equal",
    "gradient-decomposition-f32-v1",
    "beta-zero-aux-absent-v1",
]

DESCRIPTOR_IDENTITIES = (
    (
        "contract",
        "contract.json",
        "application/vnd.multitown.a25.contract.v1+json",
    ),
    (
        "common-state",
        "common-state.safetensors",
        "application/vnd.multitown.a25.state.v1+safetensors",
    ),
    (
        "observations",
        "observations.safetensors",
        "application/vnd.multitown.a25.observations.v1+safetensors",
    ),
)

GOLDEN_BUNDLE_ID = (
    "sha256:c6341d39419410813d668e7501b78a96517a424b59093774c055cef0047a749c"
)
GOLDEN_DESCRIPTORS: tuple[tuple[str, str, int, str], ...] = (
    (
        "contract",
        "contract.json",
        4044,
        "0cb2ceb504f142bf194fa85c950eace30b3cdbc40f08ab24df78a673d0a16e27",
    ),
    (
        "common-state",
        "common-state.safetensors",
        784,
        "18fe1466c2dc440447a5972a4c4bb313a6dc146083afe3841b483f7b420e1d25",
    ),
    (
        "observations",
        "observations.safetensors",
        5768,
        "07818144dd8da31564ef451db31da7b9192303ba8ad31906ab283ac376b72028",
    ),
)

_ASCII_NAME = re.compile(r"^[A-Za-z0-9._/-]+$")
_HEX32 = re.compile(r"^[0-9a-f]{8}$")
_HEX64 = re.compile(r"^[0-9a-f]{16}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_BUNDLE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_DTYPE_WIDTH = {"F32": 4, "I64": 8, "U8": 1}
_DTYPE_CODE = {"F32": 1, "I64": 2, "U8": 3}


class CheckerFailure(Exception):
    """Base class for stable CLI failures."""

    exit_code = 4
    status = "OPERATIONAL_ERROR"

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


class Rejected(CheckerFailure):
    exit_code = 2
    status = "REJECTED"


class Unsupported(CheckerFailure):
    exit_code = 3
    status = "UNSUPPORTED"


class OperationalError(CheckerFailure):
    exit_code = 4
    status = "OPERATIONAL_ERROR"


class _StrictJsonFailure(Exception):
    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


@dataclass(frozen=True, slots=True)
class Tensor:
    dtype: str
    shape: tuple[int, ...]
    data: bytes


@dataclass(frozen=True, slots=True)
class ParsedBundle:
    manifest: dict[str, Any]
    contract: dict[str, Any]
    common: dict[str, Tensor]
    observations: dict[str, Tensor]


@dataclass(frozen=True, slots=True)
class _PinnedCoreFile:
    file_descriptor: int
    identity: tuple[int, ...]


def _fail_json(reason_code: str) -> NoReturn:
    raise _StrictJsonFailure(reason_code)


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            _fail_json("JSON_DUPLICATE_KEY")
        output[key] = value
    return output


def _parse_integer(text: str) -> int:
    try:
        value = int(text, 10)
    except ValueError:
        _fail_json("JSON_INTEGER_RANGE")
    if not 0 <= value <= MAX_SAFE_INTEGER:
        _fail_json("JSON_INTEGER_RANGE")
    return value


def _reject_float(_text: str) -> NoReturn:
    _fail_json("JSON_FLOAT_FORBIDDEN")


def _reject_constant(_text: str) -> NoReturn:
    _fail_json("JSON_NONFINITE_FORBIDDEN")


def _validate_json_tree(value: Any) -> None:
    remaining = MAX_JSON_ITEMS
    stack: list[tuple[Any, int]] = [(value, 0)]
    while stack:
        current, depth = stack.pop()
        remaining -= 1
        if remaining < 0 or depth > MAX_JSON_DEPTH:
            _fail_json("JSON_RESOURCE_LIMIT")
        if isinstance(current, dict):
            for key, child in current.items():
                if not isinstance(key, str) or not key.isascii():
                    _fail_json("JSON_NON_ASCII_KEY")
                if len(key.encode("ascii")) > 256:
                    _fail_json("JSON_RESOURCE_LIMIT")
                stack.append((child, depth + 1))
        elif isinstance(current, list):
            stack.extend((child, depth + 1) for child in current)
        elif isinstance(current, str):
            if not current.isascii():
                _fail_json("JSON_NON_ASCII_STRING")
            if len(current.encode("ascii")) > MAX_JSON_STRING_BYTES:
                _fail_json("JSON_RESOURCE_LIMIT")
        elif current is None or isinstance(current, (bool, int)):
            continue
        else:
            _fail_json("JSON_VALUE_UNSUPPORTED")


def _strict_json_loads(payload: bytes, *, max_bytes: int) -> Any:
    if len(payload) > max_bytes:
        raise Rejected("JSON_SIZE_LIMIT")
    if payload.startswith(b"\xef\xbb\xbf"):
        raise Rejected("JSON_BOM_FORBIDDEN")
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise Rejected("JSON_INVALID_UTF8") from error
    try:
        value = json.loads(
            text,
            object_pairs_hook=_pairs_no_duplicates,
            parse_int=_parse_integer,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
        _validate_json_tree(value)
    except _StrictJsonFailure as error:
        raise Rejected(error.reason_code) from error
    except (json.JSONDecodeError, RecursionError) as error:
        raise Rejected("JSON_MALFORMED") from error
    return value


def _jcs_bytes(value: Any) -> bytes:
    """Return JCS bytes for the profile's ASCII/no-float JSON subset."""

    try:
        _validate_json_tree(value)
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (_StrictJsonFailure, UnicodeEncodeError, ValueError) as error:
        raise Rejected("JCS_PROFILE_VIOLATION") from error


def _exact_object(value: Any, keys: set[str], reason_code: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise Rejected(reason_code)
    return value


def _plain_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _semantic_name(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) <= 256
        and _ASCII_NAME.fullmatch(value) is not None
        and not value.startswith("/")
        and not value.endswith("/")
        and "//" not in value
        and ".." not in value.split("/")
        and "." not in value.split("/")
        and "\\" not in value
    )


def _shape(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) > MAX_RANK:
        raise Rejected("SHAPE_INVALID")
    shape: list[int] = []
    elements = 1
    for dimension in value:
        if not _plain_integer(dimension) or dimension < 1 or dimension > MAX_DIMENSION:
            raise Rejected("SHAPE_INVALID")
        if elements > MAX_CORE_FILE_BYTES // dimension:
            raise Rejected("SHAPE_OVERFLOW")
        elements *= dimension
        shape.append(dimension)
    return tuple(shape)


def _validate_manifest(manifest: Any) -> dict[str, Any]:
    manifest = _exact_object(
        manifest,
        {
            "schema_version",
            "profile",
            "bundle_id",
            "schemas",
            "materials",
            "subjects",
            "claims",
        },
        "MANIFEST_SCHEMA_VIOLATION",
    )
    if manifest["schema_version"] != MANIFEST_VERSION:
        raise Unsupported("MANIFEST_VERSION_UNSUPPORTED")
    if manifest["profile"] != PROFILE:
        raise Unsupported("PROFILE_UNSUPPORTED")
    if not isinstance(manifest["bundle_id"], str) or not _BUNDLE_ID.fullmatch(
        manifest["bundle_id"]
    ):
        raise Rejected("BUNDLE_ID_INVALID")
    if manifest["schemas"] != list(SCHEMA_DESCRIPTORS):
        raise Unsupported("SCHEMA_SET_UNSUPPORTED")
    if manifest["claims"] != EXPECTED_CLAIMS:
        raise Rejected("CLAIM_MATRIX_VIOLATION")
    materials = manifest["materials"]
    subjects = manifest["subjects"]
    if not isinstance(materials, list) or not isinstance(subjects, list):
        raise Rejected("DESCRIPTOR_SET_VIOLATION")
    descriptors = materials + subjects
    if len(materials) != 2 or len(subjects) != 1 or len(descriptors) != 3:
        raise Rejected("DESCRIPTOR_SET_VIOLATION")
    seen_names: set[str] = set()
    seen_paths: set[str] = set()
    for descriptor, identity in zip(descriptors, DESCRIPTOR_IDENTITIES, strict=True):
        descriptor = _exact_object(
            descriptor,
            {"name", "path", "media_type", "size", "digest"},
            "DESCRIPTOR_SCHEMA_VIOLATION",
        )
        name, path, media_type = identity
        if (
            descriptor["name"] != name
            or descriptor["path"] != path
            or descriptor["media_type"] != media_type
        ):
            raise Rejected("DESCRIPTOR_IDENTITY_VIOLATION")
        if name in seen_names or path in seen_paths:
            raise Rejected("DESCRIPTOR_DUPLICATE")
        seen_names.add(name)
        seen_paths.add(path)
        size = descriptor["size"]
        if not _plain_integer(size) or size <= 0 or size > MAX_CORE_FILE_BYTES:
            raise Rejected("DESCRIPTOR_SIZE_INVALID")
        digest = _exact_object(
            descriptor["digest"], {"sha256"}, "DESCRIPTOR_DIGEST_INVALID"
        )
        if not isinstance(digest["sha256"], str) or not _SHA256.fullmatch(
            digest["sha256"]
        ):
            raise Rejected("DESCRIPTOR_DIGEST_INVALID")
    core = dict(manifest)
    del core["bundle_id"]
    expected = "sha256:" + hashlib.sha256(BUNDLE_DOMAIN + _jcs_bytes(core)).hexdigest()
    if manifest["bundle_id"] != expected:
        raise Rejected("BUNDLE_ID_MISMATCH")
    return manifest


def _decode_f32_hex(value: Any) -> float:
    if not isinstance(value, str) or not _HEX32.fullmatch(value):
        raise Rejected("BETA_ENCODING_INVALID")
    result = struct.unpack(">f", bytes.fromhex(value))[0]
    if not math.isfinite(result) or result < 0.0 or value == "80000000":
        raise Rejected("BETA_VALUE_UNSUPPORTED")
    return result


def _decode_f64_hex(value: Any) -> float:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise Rejected("TOLERANCE_ENCODING_INVALID")
    result = struct.unpack(">d", bytes.fromhex(value))[0]
    if not math.isfinite(result) or result < 0.0:
        raise Rejected("TOLERANCE_VALUE_UNSUPPORTED")
    return result


def _validate_contract(contract: Any) -> dict[str, Any]:
    contract = _exact_object(
        contract,
        {
            "schema_version",
            "profile",
            "digest_spec",
            "numeric_spec",
            "limits",
            "state_components",
            "gradient_parameters",
            "cells",
            "invariants",
        },
        "CONTRACT_SCHEMA_VIOLATION",
    )
    if contract["schema_version"] != CONTRACT_VERSION:
        raise Unsupported("CONTRACT_VERSION_UNSUPPORTED")
    if contract["profile"] != PROFILE:
        raise Unsupported("PROFILE_UNSUPPORTED")
    if contract["digest_spec"] != EXPECTED_DIGEST_SPEC:
        raise Unsupported("DIGEST_SPEC_UNSUPPORTED")
    if contract["numeric_spec"] != EXPECTED_NUMERIC_SPEC:
        raise Unsupported("NUMERIC_SPEC_UNSUPPORTED")
    if contract["limits"] != EXPECTED_LIMITS:
        raise Unsupported("LIMITS_PROFILE_UNSUPPORTED")
    if contract["invariants"] != EXPECTED_INVARIANTS:
        raise Unsupported("INVARIANT_SET_UNSUPPORTED")
    _decode_f64_hex(contract["numeric_spec"]["atol_f64_hex"])
    _decode_f64_hex(contract["numeric_spec"]["rtol_f64_hex"])

    state_components = contract["state_components"]
    if not isinstance(state_components, list) or not 1 <= len(state_components) <= 1024:
        raise Rejected("STATE_COMPONENT_SET_INVALID")
    state_names: set[tuple[str, str]] = set()
    component_counts = {"model": 0, "optimizer": 0, "rng": 0}
    for item in state_components:
        item = _exact_object(
            item, {"component", "name", "dtype", "shape"}, "STATE_COMPONENT_INVALID"
        )
        component = item["component"]
        name = item["name"]
        dtype = item["dtype"]
        shape = _shape(item["shape"])
        if component not in component_counts or not _semantic_name(name):
            raise Rejected("STATE_COMPONENT_INVALID")
        if (component, name) in state_names:
            raise Rejected("STATE_COMPONENT_DUPLICATE")
        state_names.add((component, name))
        component_counts[component] += 1
        if component == "model" and dtype != "F32":
            raise Rejected("STATE_COMPONENT_DTYPE_INVALID")
        if component == "rng" and (dtype != "U8" or name != "update"):
            raise Rejected("STATE_COMPONENT_DTYPE_INVALID")
        if component == "optimizer":
            if name.startswith("state/") and dtype != "F32":
                raise Rejected("STATE_COMPONENT_DTYPE_INVALID")
            if name.startswith("scalar/") and (dtype != "I64" or shape):
                raise Rejected("STATE_COMPONENT_DTYPE_INVALID")
            if not name.startswith(("state/", "scalar/")):
                raise Rejected("STATE_COMPONENT_INVALID")
        if dtype not in _DTYPE_WIDTH:
            raise Unsupported("DTYPE_UNSUPPORTED")
    if any(count == 0 for count in component_counts.values()):
        raise Rejected("STATE_COMPONENT_SET_INVALID")

    parameters = contract["gradient_parameters"]
    if not isinstance(parameters, list) or not 1 <= len(parameters) <= 1024:
        raise Rejected("GRADIENT_PARAMETER_SET_INVALID")
    parameter_names: set[str] = set()
    for item in parameters:
        item = _exact_object(
            item, {"name", "dtype", "shape"}, "GRADIENT_PARAMETER_INVALID"
        )
        if (
            not _semantic_name(item["name"])
            or item["name"] in parameter_names
            or item["dtype"] != "F32"
        ):
            raise Rejected("GRADIENT_PARAMETER_INVALID")
        _shape(item["shape"])
        parameter_names.add(item["name"])

    cells = contract["cells"]
    if not isinstance(cells, list) or not 1 <= len(cells) <= 64:
        raise Rejected("CELL_SET_INVALID")
    expected_cell_ids = ["F00", "F01", "F10", "F11"]
    expected_beta = {
        "F00": "00000000",
        "F01": "3e800000",
        "F10": "00000000",
        "F11": "3f000000",
    }
    if [item.get("cell_id") if isinstance(item, dict) else None for item in cells] != (
        expected_cell_ids
    ):
        raise Rejected("CELL_SET_INVALID")
    zero_count = 0
    positive_count = 0
    for item in cells:
        item = _exact_object(
            item,
            {
                "cell_id",
                "arm_id",
                "stress_id",
                "beta_f32_hex",
                "prestate_namespace",
                "gradient_step_namespace",
                "aux_semantics",
            },
            "CELL_INVALID",
        )
        cell_id = item["cell_id"]
        if not _semantic_name(item["arm_id"]) or not _semantic_name(item["stress_id"]):
            raise Rejected("CELL_INVALID")
        if item["prestate_namespace"] != f"cells/{cell_id}/pre":
            raise Rejected("CELL_NAMESPACE_INVALID")
        if item["gradient_step_namespace"] != f"cells/{cell_id}/steps/000":
            raise Rejected("CELL_NAMESPACE_INVALID")
        if item["beta_f32_hex"] != expected_beta[cell_id]:
            raise Rejected("CELL_PROFILE_INVALID")
        beta = _decode_f32_hex(item["beta_f32_hex"])
        expected_aux = "forbidden" if beta == 0.0 else "required"
        if item["aux_semantics"] != expected_aux:
            raise Rejected("AUX_POLICY_INCONSISTENT")
        if beta == 0.0:
            zero_count += 1
        else:
            positive_count += 1
    if zero_count == 0 or positive_count == 0:
        raise Rejected("CELL_BETA_COVERAGE_INVALID")
    return contract


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


class _BundleDirectory:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.fd = -1
        self.directory_identity: tuple[int, ...] | None = None
        self.files: dict[str, _PinnedCoreFile] = {}

    def __enter__(self) -> Self:
        try:
            before = os.lstat(self.path)
        except FileNotFoundError as error:
            raise Rejected("INPUT_DIRECTORY_MISSING") from error
        except OSError as error:
            raise OperationalError("INPUT_DIRECTORY_IO_ERROR") from error
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
            raise Rejected("INPUT_DIRECTORY_UNSAFE")
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            self.fd = os.open(self.path, flags)
            after = os.fstat(self.fd)
            if before.st_dev != after.st_dev or before.st_ino != after.st_ino:
                raise OperationalError("INPUT_DIRECTORY_RACE")
            self.directory_identity = _directory_entry_identity(after)
            self._pin_inventory()
            self.verify_snapshot()
            return self
        except CheckerFailure:
            self.close()
            raise
        except OSError as error:
            self.close()
            raise OperationalError("INPUT_DIRECTORY_IO_ERROR") from error

    def close(self) -> None:
        for pinned in self.files.values():
            os.close(pinned.file_descriptor)
        self.files.clear()
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _entry_names(self) -> list[str]:
        try:
            names = os.listdir(self.fd)
        except OSError as error:
            raise OperationalError("DIRECTORY_ENUMERATION_ERROR") from error
        if len(names) != len(set(names)) or set(names) != CORE_FILES:
            raise Rejected("CORE_INVENTORY_MISMATCH")
        return names

    @staticmethod
    def _validate_file_stat(info: os.stat_result) -> None:
        if not stat.S_ISREG(info.st_mode):
            raise Rejected("CORE_FILE_NOT_REGULAR")
        if info.st_nlink != 1:
            raise Rejected("CORE_FILE_HARDLINKED")
        if info.st_size > MAX_CORE_FILE_BYTES:
            raise Rejected("CORE_FILE_SIZE_LIMIT")

    def _pin_inventory(self) -> None:
        flags = os.O_RDONLY | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        for name in sorted(self._entry_names()):
            file_descriptor = -1
            try:
                entry = os.stat(name, dir_fd=self.fd, follow_symlinks=False)
                self._validate_file_stat(entry)
                file_descriptor = os.open(name, flags, dir_fd=self.fd)
                opened = os.fstat(file_descriptor)
                self._validate_file_stat(opened)
            except CheckerFailure:
                if file_descriptor >= 0:
                    os.close(file_descriptor)
                raise
            except OSError as error:
                if file_descriptor >= 0:
                    os.close(file_descriptor)
                raise OperationalError("CORE_FILE_OPEN_ERROR") from error
            if _stat_identity(entry) != _stat_identity(opened):
                os.close(file_descriptor)
                raise OperationalError("CORE_FILE_ENTRY_RACE")
            self.files[name] = _PinnedCoreFile(
                file_descriptor=file_descriptor,
                identity=_stat_identity(opened),
            )

    def verify_snapshot(self) -> None:
        self._entry_names()
        if self.directory_identity is None:
            raise OperationalError("INPUT_DIRECTORY_RACE")
        try:
            directory_fd_stat = os.fstat(self.fd)
            directory_entry_stat = os.lstat(self.path)
        except OSError as error:
            raise OperationalError("INPUT_DIRECTORY_RACE") from error
        if (
            _directory_entry_identity(directory_fd_stat) != self.directory_identity
            or _directory_entry_identity(directory_entry_stat)
            != self.directory_identity
        ):
            raise OperationalError("INPUT_DIRECTORY_RACE")
        for name, pinned in self.files.items():
            try:
                opened = os.fstat(pinned.file_descriptor)
                entry = os.stat(name, dir_fd=self.fd, follow_symlinks=False)
            except OSError as error:
                raise OperationalError("CORE_FILE_ENTRY_RACE") from error
            if (
                _stat_identity(opened) != pinned.identity
                or _stat_identity(entry) != pinned.identity
            ):
                raise OperationalError("CORE_FILE_ENTRY_RACE")

    def read(self, name: str, *, limit: int) -> bytes:
        if name not in self.files:
            raise Rejected("CORE_FILE_NAME_INVALID")
        pinned = self.files[name]
        file_descriptor = pinned.file_descriptor
        try:
            before = os.fstat(file_descriptor)
            if _stat_identity(before) != pinned.identity:
                raise OperationalError("CORE_FILE_READ_RACE")
            if before.st_size > limit:
                raise Rejected("CORE_FILE_SIZE_LIMIT")
            os.lseek(file_descriptor, 0, os.SEEK_SET)
            chunks: list[bytes] = []
            remaining = before.st_size
            while remaining:
                chunk = os.read(file_descriptor, min(remaining, 1024 * 1024))
                if not chunk:
                    raise OperationalError("CORE_FILE_READ_RACE")
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(file_descriptor, 1):
                raise OperationalError("CORE_FILE_READ_RACE")
            after = os.fstat(file_descriptor)
            if _stat_identity(after) != pinned.identity:
                raise OperationalError("CORE_FILE_READ_RACE")
            return b"".join(chunks)
        except CheckerFailure:
            raise
        except OSError as error:
            raise OperationalError("CORE_FILE_IO_ERROR") from error


def _parse_safetensors(payload: bytes) -> dict[str, Tensor]:
    if len(payload) < 10:
        raise Rejected("SAFETENSORS_TRUNCATED")
    header_length = struct.unpack("<Q", payload[:8])[0]
    if header_length < 2 or header_length > MAX_SAFETENSORS_HEADER_BYTES:
        raise Rejected("SAFETENSORS_HEADER_SIZE_INVALID")
    data_start = 8 + header_length
    if data_start > len(payload):
        raise Rejected("SAFETENSORS_HEADER_TRUNCATED")
    header_bytes = payload[8:data_start]
    if not header_bytes.startswith(b"{"):
        raise Rejected("SAFETENSORS_HEADER_INVALID")
    header = _strict_json_loads(header_bytes, max_bytes=MAX_SAFETENSORS_HEADER_BYTES)
    if not isinstance(header, dict) or not header or "__metadata__" in header:
        raise Rejected("SAFETENSORS_HEADER_INVALID")
    if len(header) > MAX_TENSORS:
        raise Rejected("SAFETENSORS_TENSOR_LIMIT")
    data = payload[data_start:]
    descriptors: list[tuple[int, int, str, str, tuple[int, ...]]] = []
    for name, descriptor in header.items():
        if (
            not _semantic_name(name)
            or len(name.encode("ascii")) > 512
            or name == "__metadata__"
        ):
            raise Rejected("SAFETENSORS_TENSOR_NAME_INVALID")
        descriptor = _exact_object(
            descriptor,
            {"dtype", "shape", "data_offsets"},
            "SAFETENSORS_DESCRIPTOR_INVALID",
        )
        dtype = descriptor["dtype"]
        if dtype not in _DTYPE_WIDTH:
            raise Unsupported("DTYPE_UNSUPPORTED")
        shape = _shape(descriptor["shape"])
        offsets = descriptor["data_offsets"]
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(_plain_integer(item) for item in offsets)
        ):
            raise Rejected("SAFETENSORS_OFFSET_INVALID")
        begin, end = offsets
        if begin >= end or end > len(data):
            raise Rejected("SAFETENSORS_OFFSET_INVALID")
        elements = math.prod(shape) if shape else 1
        expected_bytes = elements * _DTYPE_WIDTH[dtype]
        if expected_bytes != end - begin:
            raise Rejected("SAFETENSORS_LENGTH_MISMATCH")
        descriptors.append((begin, end, name, dtype, shape))
    descriptors.sort(key=lambda item: (item[0], item[1], item[2].encode("ascii")))
    cursor = 0
    tensors: dict[str, Tensor] = {}
    for begin, end, name, dtype, shape in descriptors:
        if begin != cursor:
            raise Rejected("SAFETENSORS_NOT_PACKED")
        raw = data[begin:end]
        if dtype == "F32" and any(
            not math.isfinite(item[0]) for item in struct.iter_unpack("<f", raw)
        ):
            raise Rejected("SAFETENSORS_NONFINITE_F32")
        tensors[name] = Tensor(dtype=dtype, shape=shape, data=raw)
        cursor = end
    if cursor != len(data):
        raise Rejected("SAFETENSORS_NOT_PACKED")
    return tensors


def _descriptor_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [*manifest["materials"], *manifest["subjects"]]


def _read_bundle(path: Path, *, verify_sample: bool) -> ParsedBundle:
    with _BundleDirectory(path) as directory:
        manifest_bytes = directory.read("manifest.json", limit=MAX_JSON_BYTES)
        manifest = _validate_manifest(
            _strict_json_loads(manifest_bytes, max_bytes=MAX_JSON_BYTES)
        )
        if verify_sample and manifest["bundle_id"] != GOLDEN_BUNDLE_ID:
            raise Rejected("GOLDEN_ID_MISMATCH")
        blobs: dict[str, bytes] = {}
        observed_descriptors: list[tuple[str, str, int, str]] = []
        for descriptor in _descriptor_rows(manifest):
            path_name = descriptor["path"]
            blob = directory.read(path_name, limit=descriptor["size"])
            digest = hashlib.sha256(blob).hexdigest()
            if len(blob) != descriptor["size"]:
                raise Rejected("DESCRIPTOR_SIZE_MISMATCH")
            if digest != descriptor["digest"]["sha256"]:
                raise Rejected("DESCRIPTOR_DIGEST_MISMATCH")
            blobs[path_name] = blob
            observed_descriptors.append(
                (descriptor["name"], path_name, len(blob), digest)
            )
        if verify_sample and tuple(observed_descriptors) != GOLDEN_DESCRIPTORS:
            raise Rejected("GOLDEN_DESCRIPTOR_MISMATCH")
        contract = _validate_contract(
            _strict_json_loads(blobs["contract.json"], max_bytes=MAX_JSON_BYTES)
        )
        common = _parse_safetensors(blobs["common-state.safetensors"])
        observations = _parse_safetensors(blobs["observations.safetensors"])
        total_payload = sum(len(tensor.data) for tensor in common.values()) + sum(
            len(tensor.data) for tensor in observations.values()
        )
        if total_payload > MAX_TOTAL_TENSOR_PAYLOAD_BYTES:
            raise Rejected("TENSOR_PAYLOAD_LIMIT")
        directory.verify_snapshot()
    return ParsedBundle(
        manifest=manifest,
        contract=contract,
        common=common,
        observations=observations,
    )


def _component_tensor_name(component: str, semantic_name: str) -> str:
    return f"{component}/{semantic_name}"


def _expected_state_tensors(
    contract: dict[str, Any],
) -> dict[str, tuple[str, tuple[int, ...]]]:
    expected: dict[str, tuple[str, tuple[int, ...]]] = {}
    for item in contract["state_components"]:
        name = _component_tensor_name(item["component"], item["name"])
        expected[name] = (item["dtype"], tuple(item["shape"]))
    return expected


def _validate_tensor_contract(
    tensors: dict[str, Tensor],
    expected: dict[str, tuple[str, tuple[int, ...]]],
    *,
    exact: bool,
) -> None:
    if exact and set(tensors) != set(expected):
        raise Rejected("TENSOR_INVENTORY_MISMATCH")
    if not set(expected) <= set(tensors):
        raise Rejected("TENSOR_INVENTORY_MISMATCH")
    for name, (dtype, shape) in expected.items():
        tensor = tensors[name]
        if tensor.dtype != dtype or tensor.shape != shape:
            raise Rejected("TENSOR_SEMANTIC_MISMATCH")


def _semantic_digest(
    domain: str,
    records: list[tuple[str, Tensor]],
) -> str:
    digest = hashlib.sha256()
    digest.update(domain.encode("ascii"))
    ordered = sorted(records, key=lambda item: item[0].encode("ascii"))
    digest.update(struct.pack("<I", len(ordered)))
    for name, tensor in ordered:
        name_bytes = name.encode("ascii")
        digest.update(struct.pack("<I", len(name_bytes)))
        digest.update(name_bytes)
        digest.update(bytes([_DTYPE_CODE[tensor.dtype]]))
        digest.update(bytes([len(tensor.shape)]))
        for dimension in tensor.shape:
            digest.update(struct.pack("<Q", dimension))
        digest.update(struct.pack("<Q", len(tensor.data)))
        digest.update(tensor.data)
    return "sha256:" + digest.hexdigest()


def _component_digests(
    contract: dict[str, Any], tensors: dict[str, Tensor], *, prefix: str
) -> dict[str, str]:
    output: dict[str, str] = {}
    for component in ("model", "optimizer", "rng"):
        records: list[tuple[str, Tensor]] = []
        for item in contract["state_components"]:
            if item["component"] != component:
                continue
            semantic_name = item["name"]
            tensor_name = prefix + _component_tensor_name(component, semantic_name)
            records.append((semantic_name, tensors[tensor_name]))
        output[component] = _semantic_digest(
            contract["digest_spec"]["domains"][component], records
        )
    return output


def _round_f32(value: float) -> float:
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def _tensor_f32_values(tensor: Tensor) -> tuple[float, ...]:
    return tuple(item[0] for item in struct.iter_unpack("<f", tensor.data))


def _derive_gates(bundle: ParsedBundle) -> tuple[dict[str, Any], str | None]:
    contract = bundle.contract
    expected_common = _expected_state_tensors(contract)
    _validate_tensor_contract(bundle.common, expected_common, exact=True)

    expected_observations: dict[str, tuple[str, tuple[int, ...]]] = {}
    mandatory_observations: set[str] = set()
    optional_aux: set[str] = set()
    for cell in contract["cells"]:
        pre_prefix = cell["prestate_namespace"] + "/"
        for name, spec in expected_common.items():
            full = pre_prefix + name
            expected_observations[full] = spec
            mandatory_observations.add(full)
        step = cell["gradient_step_namespace"]
        for parameter in contract["gradient_parameters"]:
            spec = ("F32", tuple(parameter["shape"]))
            for kind in ("base", "total"):
                full = f"{step}/{kind}/{parameter['name']}"
                expected_observations[full] = spec
                mandatory_observations.add(full)
            aux_name = f"{step}/aux/{parameter['name']}"
            expected_observations[aux_name] = spec
            optional_aux.add(aux_name)
    observed = set(bundle.observations)
    if not mandatory_observations <= observed:
        raise Rejected("TENSOR_INVENTORY_MISMATCH")
    if not observed <= mandatory_observations | optional_aux:
        raise Rejected("TENSOR_INVENTORY_MISMATCH")
    _validate_tensor_contract(
        bundle.observations,
        {name: expected_observations[name] for name in observed},
        exact=True,
    )

    common_digests = _component_digests(contract, bundle.common, prefix="")
    cell_digest_rows: list[dict[str, Any]] = []
    common_passed = True
    for cell in contract["cells"]:
        digests = _component_digests(
            contract,
            bundle.observations,
            prefix=cell["prestate_namespace"] + "/",
        )
        matches = digests == common_digests
        common_passed = common_passed and matches
        cell_digest_rows.append(
            {"cell_id": cell["cell_id"], "digests": digests, "matches_common": matches}
        )

    atol = _decode_f64_hex(contract["numeric_spec"]["atol_f64_hex"])
    rtol = _decode_f64_hex(contract["numeric_spec"]["rtol_f64_hex"])
    gradient_rows: list[dict[str, Any]] = []
    gradient_passed = True
    aux_semantics_passed = True
    for cell in contract["cells"]:
        beta = _decode_f32_hex(cell["beta_f32_hex"])
        step = cell["gradient_step_namespace"]
        cell_passed = True
        aux_names = [
            f"{step}/aux/{parameter['name']}"
            for parameter in contract["gradient_parameters"]
        ]
        aux_present = [name in bundle.observations for name in aux_names]
        expected_aux = cell["aux_semantics"] == "required"
        aux_ok = all(aux_present) if expected_aux else not any(aux_present)
        if any(aux_present) and not all(aux_present):
            aux_ok = False
        aux_semantics_passed = aux_semantics_passed and aux_ok
        cell_passed = cell_passed and aux_ok
        for parameter in contract["gradient_parameters"]:
            base = bundle.observations[f"{step}/base/{parameter['name']}"]
            total = bundle.observations[f"{step}/total/{parameter['name']}"]
            if beta == 0.0:
                if total.data != base.data:
                    cell_passed = False
                continue
            aux_key = f"{step}/aux/{parameter['name']}"
            if aux_key not in bundle.observations:
                cell_passed = False
                continue
            aux = bundle.observations[aux_key]
            for base_value, aux_value, total_value in zip(
                _tensor_f32_values(base),
                _tensor_f32_values(aux),
                _tensor_f32_values(total),
                strict=True,
            ):
                product = _round_f32(_round_f32(beta) * _round_f32(aux_value))
                expected = _round_f32(_round_f32(base_value) + product)
                if not math.isfinite(expected) or abs(total_value - expected) > (
                    atol + rtol * abs(expected)
                ):
                    cell_passed = False
        gradient_passed = gradient_passed and cell_passed
        gradient_rows.append(
            {
                "cell_id": cell["cell_id"],
                "beta_f32_hex": cell["beta_f32_hex"],
                "aux_semantics": cell["aux_semantics"],
                "aux_semantics_passed": aux_ok,
                "parameter_count": len(contract["gradient_parameters"]),
                "passed": cell_passed,
            }
        )

    reason: str | None = None
    if not common_passed:
        reason = "COMMON_PRESTATE_MISMATCH"
    elif not aux_semantics_passed:
        reason = "AUX_SEMANTICS_FAILED"
    elif not gradient_passed:
        reason = "GRADIENT_DECOMPOSITION_FAILED"
    gates = {
        "typed_payload": {
            "passed": True,
            "common_tensor_count": len(bundle.common),
            "observation_tensor_count": len(bundle.observations),
            "payload_bytes": sum(len(item.data) for item in bundle.common.values())
            + sum(len(item.data) for item in bundle.observations.values()),
        },
        "common_prestate": {
            "passed": common_passed,
            "digest_spec": contract["digest_spec"]["record_encoding"],
            "common_digests": common_digests,
            "cells": cell_digest_rows,
        },
        "gradient_decomposition": {
            "passed": gradient_passed,
            "aux_semantics_passed": aux_semantics_passed,
            "cells": gradient_rows,
        },
    }
    return gates, reason


def _verified_descriptor(descriptor: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": descriptor["name"],
        "path": descriptor["path"],
        "size": descriptor["size"],
        "sha256": descriptor["digest"]["sha256"],
    }


def check_bundle(
    command: str, bundle_path: str | os.PathLike[str]
) -> tuple[dict[str, Any], int]:
    """Check one bundle and return its deterministic report and exit code."""

    if command not in {"verify-sample", "inspect-bundle"}:
        raise Rejected("CLI_COMMAND_INVALID")
    bundle = _read_bundle(Path(bundle_path), verify_sample=command == "verify-sample")
    gates, failure_reason = _derive_gates(bundle)
    if failure_reason is None:
        status = (
            "VERIFIED_SAMPLE" if command == "verify-sample" else "CONSISTENT_BUNDLE"
        )
        reason_code = "OK"
        exit_code = 0
    else:
        status = "CONTRACT_FAILED"
        reason_code = failure_reason
        exit_code = 1
    report = {
        "schema_version": REPORT_VERSION,
        "status": status,
        "reason_code": reason_code,
        "bundle_id": bundle.manifest["bundle_id"],
        "checker": {
            "name": CHECKER_NAME,
            "version": CHECKER_VERSION,
            "policy_id": PROFILE,
        },
        "verified_materials": [
            _verified_descriptor(item) for item in bundle.manifest["materials"]
        ],
        "verified_subjects": [
            _verified_descriptor(item) for item in bundle.manifest["subjects"]
        ],
        "derived_gates": gates,
        "claims": {
            "artifact_only": True,
            "golden_identity_pinned": status == "VERIFIED_SAMPLE",
            "typed_payload_rehashed": True,
            "explicit_update_rng_state_bytes_rehashed": True,
            "torch_rng_sequence_reproduced": False,
            "autograd_reexecuted": False,
            "optimizer_step_reexecuted": False,
            "training_reexecuted": False,
            "authenticated_provenance": False,
            "freshness_attested": False,
            "full_numerical_q0_qualified": False,
            "q1_mechanism_qualified": False,
            "performance_claim_supported": False,
            "safety_claim_supported": False,
        },
    }
    return report, exit_code


def _json_line(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        + b"\n"
    )


def _write_all(file_descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(file_descriptor, payload[offset:])
        if written <= 0:
            raise OperationalError("REPORT_WRITE_ERROR")
        offset += written


def _unlink_if_opened_identity_matches(
    directory_fd: int, name: str, file_descriptor: int
) -> None:
    try:
        opened = os.fstat(file_descriptor)
        entry = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError as error:
        raise OperationalError("REPORT_CLEANUP_ERROR") from error
    if (
        stat.S_ISREG(entry.st_mode)
        and entry.st_dev == opened.st_dev
        and entry.st_ino == opened.st_ino
    ):
        try:
            os.unlink(name, dir_fd=directory_fd)
        except FileNotFoundError:
            return
        except OSError as error:
            raise OperationalError("REPORT_CLEANUP_ERROR") from error


def _same_directory_identity(
    directory_path: Path, directory_fd: int, expected: tuple[int, ...]
) -> bool:
    try:
        return (
            _directory_entry_identity(os.fstat(directory_fd)) == expected
            and _directory_entry_identity(os.lstat(directory_path)) == expected
        )
    except OSError:
        return False


def _directory_entry_identity(value: os.stat_result) -> tuple[int, ...]:
    return (value.st_dev, value.st_ino, stat.S_IFMT(value.st_mode))


def _write_report(output: Path, input_directory: Path, payload: bytes) -> None:
    try:
        input_resolved = input_directory.resolve(strict=True)
        output_parent = output.parent.resolve(strict=True)
    except OSError as error:
        raise OperationalError("REPORT_PATH_IO_ERROR") from error
    output_resolved = output_parent / output.name
    try:
        if os.path.commonpath((input_resolved, output_resolved)) == str(input_resolved):
            raise Rejected("REPORT_PATH_OVERLAPS_INPUT")
    except ValueError as error:
        raise Rejected("REPORT_PATH_INVALID") from error
    final_name = output.name
    if not final_name or final_name in {".", ".."}:
        raise Rejected("REPORT_PATH_INVALID")
    directory_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    try:
        directory_before = os.lstat(output_parent)
        if stat.S_ISLNK(directory_before.st_mode) or not stat.S_ISDIR(
            directory_before.st_mode
        ):
            raise Rejected("REPORT_PATH_INVALID")
        directory_fd = os.open(output_parent, directory_flags)
        directory_opened = os.fstat(directory_fd)
    except CheckerFailure:
        raise
    except OSError as error:
        raise OperationalError("REPORT_PATH_IO_ERROR") from error
    directory_identity = _directory_entry_identity(directory_opened)
    if _directory_entry_identity(directory_before) != directory_identity:
        os.close(directory_fd)
        raise OperationalError("REPORT_PATH_RACE")

    temporary_fd = -1
    temporary_name: str | None = None
    published = False
    failure: CheckerFailure | None = None
    try:
        temporary_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            temporary_flags |= os.O_NOFOLLOW
        for _attempt in range(32):
            candidate = (
                f".{final_name}.multitown-a25-{os.getpid()}-{secrets.token_hex(16)}.tmp"
            )
            try:
                temporary_fd = os.open(
                    candidate, temporary_flags, 0o600, dir_fd=directory_fd
                )
                temporary_name = candidate
                break
            except FileExistsError:
                continue
        if temporary_fd < 0 or temporary_name is None:
            raise OperationalError("REPORT_TEMP_CREATE_ERROR")
        os.fchmod(temporary_fd, 0o600)
        _write_all(temporary_fd, payload)
        os.fsync(temporary_fd)
        if not _same_directory_identity(
            output_parent, directory_fd, directory_identity
        ):
            raise OperationalError("REPORT_PATH_RACE")
        try:
            os.link(
                temporary_name,
                final_name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise Rejected("REPORT_OUTPUT_EXISTS") from error
        published = True
        final_entry = os.stat(final_name, dir_fd=directory_fd, follow_symlinks=False)
        temporary_opened = os.fstat(temporary_fd)
        if (
            not stat.S_ISREG(final_entry.st_mode)
            or final_entry.st_dev != temporary_opened.st_dev
            or final_entry.st_ino != temporary_opened.st_ino
        ):
            raise OperationalError("REPORT_PUBLISH_RACE")
        _unlink_if_opened_identity_matches(directory_fd, temporary_name, temporary_fd)
        temporary_name = None
        os.fsync(directory_fd)
        if not _same_directory_identity(
            output_parent, directory_fd, directory_identity
        ):
            raise OperationalError("REPORT_PATH_RACE")
    except CheckerFailure as error:
        failure = error
    except OSError as error:
        failure = OperationalError("REPORT_WRITE_ERROR")
        failure.__cause__ = error
    finally:
        if failure is not None and temporary_fd >= 0:
            try:
                if published:
                    _unlink_if_opened_identity_matches(
                        directory_fd, final_name, temporary_fd
                    )
                if temporary_name is not None:
                    _unlink_if_opened_identity_matches(
                        directory_fd, temporary_name, temporary_fd
                    )
                os.fsync(directory_fd)
            except CheckerFailure as cleanup_error:
                failure = cleanup_error
        if temporary_fd >= 0:
            os.close(temporary_fd)
        os.close(directory_fd)
    if failure is not None:
        raise failure


def _usage() -> str:
    return (
        "usage: multitown-a25-portable-checker "
        "({verify-sample,inspect-bundle} BUNDLE | verify-installed-sample) "
        "[--write-report OUT]"
    )


def _installed_sample_path() -> Path:
    return (
        Path(__file__).resolve().parent
        / "fixtures"
        / "a25_portable_sample_v1"
        / "bundle"
    )


def _parse_cli(argv: Sequence[str]) -> tuple[str, Path, Path | None]:
    if len(argv) == 1 and argv[0] in {"-h", "--help"}:
        raise SystemExit(0)
    if not argv:
        raise Rejected("CLI_USAGE_INVALID")
    requested_command = argv[0]
    if requested_command == "verify-installed-sample":
        if len(argv) not in {1, 3}:
            raise Rejected("CLI_USAGE_INVALID")
        command = "verify-sample"
        bundle = _installed_sample_path()
        option_offset = 1
    elif requested_command in {"verify-sample", "inspect-bundle"}:
        if len(argv) not in {2, 4}:
            raise Rejected("CLI_USAGE_INVALID")
        command = requested_command
        bundle = Path(argv[1])
        option_offset = 2
    else:
        raise Rejected("CLI_COMMAND_INVALID")
    output: Path | None = None
    if len(argv) == option_offset + 2:
        if argv[option_offset] != "--write-report" or not argv[option_offset + 1]:
            raise Rejected("CLI_USAGE_INVALID")
        output = Path(argv[option_offset + 1])
    return command, bundle, output


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        try:
            command, bundle, output = _parse_cli(arguments)
        except SystemExit:
            sys.stdout.write(_usage() + "\n")
            return 0
        report, exit_code = check_bundle(command, bundle)
        payload = _json_line(report)
        if output is not None:
            _write_report(output, bundle, payload)
        sys.stdout.buffer.write(payload)
        return exit_code
    except CheckerFailure as error:
        payload = _json_line({"status": error.status, "reason_code": error.reason_code})
        sys.stderr.buffer.write(payload)
        return error.exit_code
    except Exception:  # noqa: BLE001 - the CLI maps all internal faults to stable exit 4
        payload = _json_line(
            {"status": "OPERATIONAL_ERROR", "reason_code": "CHECKER_INTERNAL_ERROR"}
        )
        sys.stderr.buffer.write(payload)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
