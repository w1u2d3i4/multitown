"""Publish and verify the clean-source A25 partial numerical-Q0 bundle.

This runner wraps the synthetic twelve-cell diagnostic and its raw gradient
sidecar.  It never reads an outer split, creates a formal lock, or upgrades the
diagnostic to full numerical-Q0 qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import shutil
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import a25_full_numerical_q0 as full_q0
from . import a25_independent_verifier as independent_verifier
from . import a25_numerical_oracles as numerical_oracles

A25_FULL_NUMERICAL_RUNNER_VERSION = "multitown-a25-full-numerical-q0-runner-v2"
A25_FULL_NUMERICAL_WRAPPER_VERSION = "multitown-a25-full-numerical-q0-wrapper-receipt-v2"
A25_FULL_NUMERICAL_RUNTIME_VERSION = "multitown-a25-full-numerical-q0-runtime-v1"
A25_FULL_NUMERICAL_SOURCE_VERSION = "multitown-a25-full-numerical-q0-source-v2"
A25_FULL_NUMERICAL_RNG_GUARD_SEED = 2026081504
RECEIPT_BASENAME = "receipt.json"
GRADIENT_BASENAME = "gradients.f32le.bin"
FORMAL_LOCK = "artifacts/a24-cr-ppo-no-shield-attempt-v1.lock"
MAX_RECEIPT_BYTES = 64 * 1024 * 1024
MAX_GRADIENT_BYTES = 64 * 1024 * 1024

_HEX_SHA256 = set("0123456789abcdef")
_FORBIDDEN_ENVIRONMENT = frozenset(
    {
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_CEILING_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_CONFIG",
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_NOSYSTEM",
        "GIT_DIR",
        "GIT_DISCOVERY_ACROSS_FILESYSTEM",
        "GIT_EXEC_PATH",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_WORK_TREE",
        "PYTHONBREAKPOINT",
        "PYTHONHOME",
        "PYTHONINSPECT",
        "PYTHONPATH",
        "PYTHONSAFEPATH",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
    }
)
_RECORDED_ENVIRONMENT = (
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "PYTHONHASHSEED",
)


def _resolve_git_executable() -> Path:
    candidate = shutil.which("git")
    if candidate is None:
        raise RuntimeError("A25 full numerical Q0 requires Git")
    path = Path(candidate)
    if path.is_symlink():
        raise RuntimeError("A25 full numerical Q0 Git executable is a symlink")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise RuntimeError("A25 full numerical Q0 Git executable is invalid")
    return resolved


_GIT_EXECUTABLE = _resolve_git_executable()
_REQUIRED_MODULES = (
    "multitown",
    "multitown.a22_constrained_ppo",
    "multitown.a23_cr_ppo",
    "multitown.a25_full_numerical_q0",
    "multitown.a25_full_numerical_q0_runner",
    "multitown.a25_independent_verifier",
    "multitown.a25_numerical_oracles",
    "multitown.a25_qualification",
    "multitown.a25_shield_dependence",
    "multitown.a9_long_horizon_env",
    "multitown.a9_oof_protocol",
    "multitown.a9_ppo_oof",
    "multitown.long_horizon_env",
    "multitown.ppo_controller",
    "multitown.pq1_numerical_conformance",
)
_WRAPPER_CORE_KEYS = {
    "schema_version",
    "runner_version",
    "scope",
    "source",
    "runtime",
    "global_rng_guard",
    "formal_lock",
    "diagnostic_receipt",
    "inner_receipt_id",
    "isolation_diagnostic",
    "isolation_receipt_id",
    "numerical_oracle_receipt",
    "numerical_oracle_receipt_id",
    "gradient_sidecar",
    "gates",
    "claim_boundary",
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in _HEX_SHA256 for character in value)
    )


def _validate_environment() -> None:
    present = sorted(_FORBIDDEN_ENVIRONMENT.intersection(os.environ))
    if present:
        raise RuntimeError(
            "A25 full numerical Q0 refuses source/import override environment: "
            + ",".join(present)
        )


def _git_output(root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        [str(_GIT_EXECUTABLE), *arguments],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if completed.returncode:
        raise RuntimeError(
            "A25 full numerical Q0 git command failed: " + " ".join(arguments)
        )
    return completed.stdout


def _logical_loaded_module(name: str) -> Any:
    module = sys.modules.get(name)
    if module is not None:
        return module
    main_module = sys.modules.get("__main__")
    spec = getattr(main_module, "__spec__", None)
    if getattr(spec, "name", None) == name:
        return main_module
    raise RuntimeError(f"A25 full numerical Q0 module is not loaded: {name}")


def _loaded_module_origins(root: Path) -> dict[str, dict[str, Any]]:
    origins: dict[str, dict[str, Any]] = {}
    for name in _REQUIRED_MODULES:
        module = _logical_loaded_module(name)
        raw_origin = getattr(module, "__file__", None)
        if type(raw_origin) is not str or not raw_origin:
            raise RuntimeError(f"A25 full numerical Q0 module has no origin: {name}")
        raw_path = Path(raw_origin)
        if raw_path.is_symlink():
            raise RuntimeError(
                f"A25 full numerical Q0 module origin is a symlink: {name}"
            )
        try:
            origin = raw_path.resolve(strict=True)
            relative = origin.relative_to(root).as_posix()
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"A25 full numerical Q0 module is outside the bound root: {name}"
            ) from exc
        metadata = origin.stat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or origin.suffix != ".py"
        ):
            raise RuntimeError(f"A25 full numerical Q0 module origin is unsafe: {name}")
        payload = origin.read_bytes()
        head_payload = _git_output(root, "show", f"HEAD:{relative}")
        if payload != head_payload:
            raise RuntimeError(
                f"A25 full numerical Q0 module differs from HEAD: {name}"
            )
        origins[name] = {
            "path": relative,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
    return origins


def _source_state(root: Path) -> dict[str, Any]:
    _validate_environment()
    repository = root.resolve(strict=True)
    if not repository.is_dir() or root.is_symlink():
        raise RuntimeError("A25 full numerical Q0 root is unsafe")
    top = Path(
        _git_output(repository, "rev-parse", "--show-toplevel").decode("utf-8").strip()
    ).resolve(strict=True)
    if top != repository:
        raise RuntimeError("A25 full numerical Q0 root is not the Git top level")
    if _git_output(repository, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("A25 full numerical Q0 requires exact clean source")
    revision = _git_output(repository, "rev-parse", "HEAD").decode().strip()
    tree = _git_output(repository, "rev-parse", "HEAD^{tree}").decode().strip()
    origins = _loaded_module_origins(repository)
    source_sha256 = {row["path"]: row["sha256"] for row in origins.values()}
    git_version = _git_output(repository, "--version").decode().strip()
    return {
        "schema_version": A25_FULL_NUMERICAL_SOURCE_VERSION,
        "revision": revision,
        "tree": tree,
        "source_sha256": dict(sorted(source_sha256.items())),
        "source_bundle_sha256": _canonical_sha256(source_sha256),
        "module_origins": origins,
        "git": {
            "executable": str(_GIT_EXECUTABLE),
            "executable_sha256": hashlib.sha256(
                _GIT_EXECUTABLE.read_bytes()
            ).hexdigest(),
            "version": git_version,
        },
    }


def _module_identity(module: Any, *, label: str) -> dict[str, str]:
    raw_origin = getattr(module, "__file__", None)
    if type(raw_origin) is not str or not raw_origin:
        raise RuntimeError(f"A25 full numerical Q0 {label} has no origin")
    origin = Path(raw_origin)
    if origin.is_symlink():
        raise RuntimeError(f"A25 full numerical Q0 {label} origin is unsafe")
    resolved = origin.resolve(strict=True)
    if not resolved.is_file():
        raise RuntimeError(f"A25 full numerical Q0 {label} origin is not a file")
    return {
        "version": str(module.__version__),
        "origin_sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
    }


def _runtime_profile() -> dict[str, Any]:
    _validate_environment()
    executable = Path(sys.executable).resolve(strict=True)
    if not executable.is_file():
        raise RuntimeError("A25 full numerical Q0 Python executable is invalid")
    return {
        "schema_version": A25_FULL_NUMERICAL_RUNTIME_VERSION,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "compiler": platform.python_compiler(),
            "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        },
        "numpy": _module_identity(np, label="NumPy"),
        "torch": {
            **_module_identity(torch, label="Torch"),
            "git_version": str(torch.version.git_version),
            "cuda_build": torch.version.cuda,
            "default_dtype": str(torch.get_default_dtype()),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "byteorder": sys.byteorder,
        },
        "execution": {
            "device": "cpu",
            "torch_intraop_threads": int(torch.get_num_threads()),
            "torch_interop_threads": int(torch.get_num_interop_threads()),
        },
        "environment": {key: os.environ.get(key) for key in _RECORDED_ENVIRONMENT},
    }


def _capture_global_rng() -> tuple[Any, tuple[Any, ...], torch.Tensor]:
    numpy_state = np.random.get_state()
    copied_numpy = (
        numpy_state[0],
        numpy_state[1].copy(),
        numpy_state[2],
        numpy_state[3],
        numpy_state[4],
    )
    return random.getstate(), copied_numpy, torch.random.get_rng_state().clone()


def _restore_global_rng(
    state: tuple[Any, tuple[Any, ...], torch.Tensor],
) -> None:
    python_state, numpy_state, torch_state = state
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.random.set_rng_state(torch_state)


def _global_rng_sha256() -> dict[str, str]:
    numpy_state = np.random.get_state()
    numpy_values = np.ascontiguousarray(numpy_state[1])
    numpy_digest = hashlib.sha256()
    for value in (
        str(numpy_state[0]).encode("ascii"),
        numpy_values.dtype.str.encode("ascii"),
        _canonical_bytes(list(numpy_values.shape)),
        numpy_values.tobytes(order="C"),
        _canonical_bytes(
            [int(numpy_state[2]), int(numpy_state[3]), float(numpy_state[4])]
        ),
    ):
        numpy_digest.update(value + b"\0")
    torch_state = torch.random.get_rng_state().detach().cpu().contiguous().numpy()
    torch_digest = hashlib.sha256()
    for value in (
        torch_state.dtype.str.encode("ascii"),
        _canonical_bytes(list(torch_state.shape)),
        torch_state.tobytes(order="C"),
    ):
        torch_digest.update(value + b"\0")
    return {
        "python_sha256": hashlib.sha256(
            _canonical_bytes(random.getstate())
        ).hexdigest(),
        "numpy_sha256": numpy_digest.hexdigest(),
        "torch_cpu_sha256": torch_digest.hexdigest(),
    }


def _assert_formal_lock_absent(root: Path) -> None:
    try:
        os.lstat(root / FORMAL_LOCK)
    except FileNotFoundError:
        return
    raise RuntimeError("A25 full numerical Q0 refuses an A24 formal lock")


def _inner_core(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in receipt.items()
        if key not in {"receipt_id", "status"}
    }


def _validate_inner_receipt(receipt: Any, payload: bytes) -> None:
    if type(receipt) is not dict:
        raise TypeError("invalid A25 inner diagnostic receipt")
    if (
        receipt.get("status") != "DIAGNOSTIC_PASSED"
        or not _valid_sha256(receipt.get("receipt_id"))
        or receipt["receipt_id"] != _canonical_sha256(_inner_core(receipt))
    ):
        raise ValueError("invalid A25 inner diagnostic receipt identity")
    claim = receipt.get("claim_boundary")
    if (
        type(claim) is not dict
        or claim.get("twelve_cell_gradient_diagnostic_passed") is not True
        or claim.get("beta_zero_base_gradient_capture_passed") is not True
        or claim.get("clean_source_bound") is not False
        or claim.get("gradient_sidecar_bound") is not False
        or claim.get("full_numerical_q0_qualified") is not False
        or claim.get("formal_authorized") is not False
        or claim.get("outer_rows_read") != 0
    ):
        raise ValueError("A25 inner diagnostic claim boundary changed")
    full_q0.validate_gradient_payload(receipt, payload)


def _validate_isolation_receipt(receipt: Any, payload: bytes) -> None:
    if type(receipt) is not dict:
        raise TypeError("invalid A25 G3 isolation diagnostic receipt")
    if (
        receipt.get("status") != "DIAGNOSTIC_PASSED"
        or not _valid_sha256(receipt.get("receipt_id"))
        or receipt["receipt_id"] != _canonical_sha256(_inner_core(receipt))
    ):
        raise ValueError("invalid A25 G3 isolation diagnostic identity")
    claim = receipt.get("claim_boundary")
    gates = receipt.get("gates")
    if (
        type(claim) is not dict
        or claim.get("fresh_adam_isolation_diagnostic_passed") is not True
        or claim.get("warm_adam_equivalence_inferred") is not False
        or claim.get("full_numerical_q0_qualified") is not False
        or claim.get("formal_authorized") is not False
        or type(gates) is not dict
        or not gates
        or not all(type(value) is bool and value for value in gates.values())
    ):
        raise ValueError("invalid A25 G3 isolation diagnostic boundary")
    step = receipt.get("gradient_step")
    if type(step) is not dict:
        raise TypeError("invalid A25 G3 isolation gradient step")
    full_q0.validate_gradient_payload(
        {
            "gradient_artifact": receipt.get("gradient_artifact"),
            "cells": {"isolation/G3": {"steps": [step]}},
        },
        payload,
    )


def _validate_numerical_oracle_receipt(receipt: Any) -> None:
    if type(receipt) is not dict:
        raise TypeError("invalid A25 numerical oracle receipt")
    if (
        receipt.get("status") != "DIAGNOSTIC_PASSED"
        or not _valid_sha256(receipt.get("receipt_id"))
        or receipt["receipt_id"] != _canonical_sha256(_inner_core(receipt))
    ):
        raise ValueError("invalid A25 numerical oracle identity")
    diagnostics = receipt.get("diagnostics")
    claim = receipt.get("claim_boundary")
    if (
        type(diagnostics) is not dict
        or set(diagnostics) != {"G1", "G4", "G5", "G6"}
        or not all(
            type(result) is dict and result.get("passed") is True
            for result in diagnostics.values()
        )
        or type(claim) is not dict
        or claim.get("diagnostic_primitives_passed") is not True
        or claim.get("qualification_evidence") is not False
        or claim.get("full_numerical_q0_qualified") is not False
        or claim.get("formal_authorized") is not False
        or claim.get("outer_rows_read") != 0
        or claim.get("formal_lock_created") is not False
    ):
        raise ValueError("invalid A25 numerical oracle boundary")


def _claim_boundary() -> dict[str, Any]:
    return {
        "partial_numerical_q0_bundle_passed": True,
        "clean_source_bound": True,
        "gradient_sidecar_bound": True,
        "fresh_adam_g3_isolation_bound": True,
        "numerical_oracles_g1_g4_g5_g6_bound": True,
        "beta_zero_base_gradient_capture_qualified": True,
        "dependency_light_checker_available": True,
        "independent_persisted_contract_verification_executed": False,
        "full_numerical_q0_qualified": False,
        "q1_mechanism_qualified": False,
        "qualification_receipt": False,
        "formal_authorized": False,
        "performance_claim_supported": False,
        "safety_claim_supported": False,
        "outer_rows_read": 0,
        "formal_lock_created": False,
        "remaining_blockers": [
            "independent-persisted-contract-verification-not-yet-executed",
        ],
    }


def _build_wrapper(
    *,
    source: dict[str, Any],
    runtime: dict[str, Any],
    rng_before: dict[str, str],
    rng_after: dict[str, str],
    diagnostic_receipt: dict[str, Any],
    isolation_diagnostic: dict[str, Any],
    numerical_oracle_receipt: dict[str, Any],
    twelve_payload: bytes,
    isolation_payload: bytes,
) -> dict[str, Any]:
    artifact = diagnostic_receipt["gradient_artifact"]
    isolation_artifact = isolation_diagnostic["gradient_artifact"]
    payload = twelve_payload + isolation_payload
    core = {
        "schema_version": A25_FULL_NUMERICAL_WRAPPER_VERSION,
        "runner_version": A25_FULL_NUMERICAL_RUNNER_VERSION,
        "scope": ("clean-source-synthetic-partial-numerical-q0-no-outer-no-formal"),
        "source": source,
        "runtime": runtime,
        "global_rng_guard": {
            "seed": A25_FULL_NUMERICAL_RNG_GUARD_SEED,
            "before": rng_before,
            "after": rng_after,
        },
        "formal_lock": {
            "path": FORMAL_LOCK,
            "observed_absent_before": True,
            "observed_absent_after": True,
            "created": False,
        },
        "diagnostic_receipt": diagnostic_receipt,
        "inner_receipt_id": diagnostic_receipt["receipt_id"],
        "isolation_diagnostic": isolation_diagnostic,
        "isolation_receipt_id": isolation_diagnostic["receipt_id"],
        "numerical_oracle_receipt": numerical_oracle_receipt,
        "numerical_oracle_receipt_id": numerical_oracle_receipt["receipt_id"],
        "gradient_sidecar": {
            "schema_version": "multitown-a25-combined-gradient-sidecar-v1",
            "basename": GRADIENT_BASENAME,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "nbytes": len(payload),
            "encoding": "ordered-byte-concatenation-v1",
            "segments": [
                {
                    "name": "twelve-cell",
                    "offset": 0,
                    "nbytes": len(twelve_payload),
                    "sha256": hashlib.sha256(twelve_payload).hexdigest(),
                    "receipt_id": diagnostic_receipt["receipt_id"],
                    "artifact_sha256": artifact["sha256"],
                    "artifact_nbytes": artifact["nbytes"],
                },
                {
                    "name": "isolation-g3",
                    "offset": len(twelve_payload),
                    "nbytes": len(isolation_payload),
                    "sha256": hashlib.sha256(isolation_payload).hexdigest(),
                    "receipt_id": isolation_diagnostic["receipt_id"],
                    "artifact_sha256": isolation_artifact["sha256"],
                    "artifact_nbytes": isolation_artifact["nbytes"],
                },
            ],
        },
        "gates": {
            "exact_clean_source_bound": True,
            "declared_execution_module_origins_bound": True,
            "global_rng_preserved": rng_before == rng_after,
            "formal_lock_absent_and_unmodified": True,
            "inner_diagnostic_passed": True,
            "fresh_adam_g3_isolation_passed": True,
            "numerical_oracles_g1_g4_g5_g6_passed": True,
            "gradient_sidecar_validated": True,
            "beta_zero_base_gradient_capture_passed": diagnostic_receipt["gates"][
                "beta_zero_base_gradient_capture_complete"
            ],
            "dependency_light_checker_source_bound": (
                "multitown.a25_independent_verifier"
                in source["module_origins"]
                and independent_verifier.INDEPENDENT_VERIFIER_VERSION
                == "multitown-a25-independent-persisted-verifier-v2"
            ),
            "zero_outer_rows_read": True,
            "no_formal_authorization": True,
        },
        "claim_boundary": _claim_boundary(),
    }
    if not all(core["gates"].values()):
        raise RuntimeError("A25 partial numerical Q0 wrapper gate failed")
    return {
        **core,
        "receipt_id": _canonical_sha256(core),
        "status": "PARTIAL_Q0_PASSED",
    }


def recompute_bundle(root: Path) -> tuple[dict[str, Any], bytes]:
    """Recompute the exact clean-source wrapper while preserving caller RNG."""

    repository = root.resolve(strict=True)
    external_rng = _capture_global_rng()
    try:
        random.seed(A25_FULL_NUMERICAL_RNG_GUARD_SEED)
        np.random.seed(A25_FULL_NUMERICAL_RNG_GUARD_SEED % (2**32))
        torch.random.default_generator.manual_seed(A25_FULL_NUMERICAL_RNG_GUARD_SEED)
        rng_before = _global_rng_sha256()
        _assert_formal_lock_absent(repository)
        source_before = _source_state(repository)
        runtime_before = _runtime_profile()
        diagnostic_receipt, twelve_payload = full_q0._build_twelve_cell_artifacts()
        _validate_inner_receipt(diagnostic_receipt, twelve_payload)
        isolation_diagnostic, isolation_payload = full_q0._build_isolation_artifacts()
        _validate_isolation_receipt(isolation_diagnostic, isolation_payload)
        numerical_oracle_receipt = numerical_oracles.build_numerical_oracle_receipt()
        _validate_numerical_oracle_receipt(numerical_oracle_receipt)
        _assert_formal_lock_absent(repository)
        source_after = _source_state(repository)
        runtime_after = _runtime_profile()
        rng_after = _global_rng_sha256()
        if _canonical_bytes(source_before) != _canonical_bytes(source_after):
            raise RuntimeError("A25 full numerical Q0 source changed during run")
        if _canonical_bytes(runtime_before) != _canonical_bytes(runtime_after):
            raise RuntimeError("A25 full numerical Q0 runtime changed during run")
        if rng_before != rng_after:
            raise RuntimeError("A25 full numerical Q0 mutated global RNG")
        wrapper = _build_wrapper(
            source=source_before,
            runtime=runtime_before,
            rng_before=rng_before,
            rng_after=rng_after,
            diagnostic_receipt=diagnostic_receipt,
            isolation_diagnostic=isolation_diagnostic,
            numerical_oracle_receipt=numerical_oracle_receipt,
            twelve_payload=twelve_payload,
            isolation_payload=isolation_payload,
        )
        payload = twelve_payload + isolation_payload
        _validate_wrapper_record(wrapper, payload)
        return wrapper, payload
    finally:
        _restore_global_rng(external_rng)


def _strict_json_bytes(payload: bytes) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate A25 partial Q0 JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite A25 partial Q0 JSON constant: {token}")
            ),
        )
    except UnicodeDecodeError as exc:
        raise ValueError("invalid A25 partial Q0 JSON encoding") from exc

    def reject_nonfinite(value: Any) -> None:
        if type(value) is float and not math.isfinite(value):
            raise ValueError("non-finite A25 partial Q0 JSON number")
        if type(value) is dict:
            for nested in value.values():
                reject_nonfinite(nested)
        elif type(value) is list:
            for nested in value:
                reject_nonfinite(nested)

    reject_nonfinite(value)
    if type(value) is not dict:
        raise TypeError("A25 partial Q0 receipt must be an object")
    return value


def _validated_sidecar_segments(
    sidecar: Any,
    payload: bytes,
    diagnostic: Mapping[str, Any],
    isolation: Mapping[str, Any],
) -> tuple[bytes, bytes]:
    if (
        type(sidecar) is not dict
        or set(sidecar)
        != {
            "schema_version",
            "basename",
            "sha256",
            "nbytes",
            "encoding",
            "segments",
        }
        or sidecar["schema_version"] != "multitown-a25-combined-gradient-sidecar-v1"
        or sidecar["basename"] != GRADIENT_BASENAME
        or sidecar["encoding"] != "ordered-byte-concatenation-v1"
        or sidecar["sha256"] != hashlib.sha256(payload).hexdigest()
        or sidecar["nbytes"] != len(payload)
        or type(sidecar["segments"]) is not list
        or len(sidecar["segments"]) != 2
    ):
        raise ValueError("invalid A25 partial Q0 sidecar binding")
    expected = (
        ("twelve-cell", diagnostic),
        ("isolation-g3", isolation),
    )
    offset = 0
    slices: list[bytes] = []
    for segment, (expected_name, bound_receipt) in zip(
        sidecar["segments"], expected, strict=True
    ):
        artifact = bound_receipt.get("gradient_artifact")
        if type(artifact) is not dict:
            raise TypeError("invalid A25 partial Q0 segment artifact")
        if (
            type(segment) is not dict
            or set(segment)
            != {
                "name",
                "offset",
                "nbytes",
                "sha256",
                "receipt_id",
                "artifact_sha256",
                "artifact_nbytes",
            }
            or segment["name"] != expected_name
            or type(segment["offset"]) is not int
            or segment["offset"] != offset
            or type(segment["nbytes"]) is not int
            or segment["nbytes"] <= 0
            or offset + segment["nbytes"] > len(payload)
        ):
            raise ValueError("invalid A25 partial Q0 sidecar segment layout")
        segment_payload = payload[offset : offset + segment["nbytes"]]
        if (
            segment["sha256"] != hashlib.sha256(segment_payload).hexdigest()
            or segment["receipt_id"] != bound_receipt.get("receipt_id")
            or segment["artifact_sha256"] != artifact.get("sha256")
            or segment["artifact_nbytes"] != artifact.get("nbytes")
            or segment["sha256"] != segment["artifact_sha256"]
            or segment["nbytes"] != segment["artifact_nbytes"]
        ):
            raise ValueError("invalid A25 partial Q0 sidecar segment binding")
        slices.append(segment_payload)
        offset += segment["nbytes"]
    if offset != len(payload):
        raise ValueError("incomplete A25 partial Q0 sidecar segment coverage")
    return slices[0], slices[1]


def _validate_wrapper_record(receipt: Any, payload: bytes) -> None:
    if type(receipt) is not dict or type(payload) is not bytes:
        raise TypeError("invalid A25 partial Q0 bundle")
    if set(receipt) != _WRAPPER_CORE_KEYS | {"receipt_id", "status"}:
        raise ValueError("invalid A25 partial Q0 wrapper fields")
    core = {key: receipt[key] for key in _WRAPPER_CORE_KEYS}
    if (
        receipt["schema_version"] != A25_FULL_NUMERICAL_WRAPPER_VERSION
        or receipt["runner_version"] != A25_FULL_NUMERICAL_RUNNER_VERSION
        or receipt["status"] != "PARTIAL_Q0_PASSED"
        or not _valid_sha256(receipt["receipt_id"])
        or receipt["receipt_id"] != _canonical_sha256(core)
    ):
        raise ValueError("invalid A25 partial Q0 wrapper identity")
    diagnostic = receipt["diagnostic_receipt"]
    if type(diagnostic) is not dict or receipt["inner_receipt_id"] != diagnostic.get(
        "receipt_id"
    ):
        raise ValueError("invalid A25 partial Q0 inner receipt binding")
    isolation = receipt["isolation_diagnostic"]
    if type(isolation) is not dict or receipt["isolation_receipt_id"] != isolation.get(
        "receipt_id"
    ):
        raise ValueError("invalid A25 partial Q0 G3 receipt binding")
    numerical_oracle = receipt["numerical_oracle_receipt"]
    if type(numerical_oracle) is not dict or receipt[
        "numerical_oracle_receipt_id"
    ] != numerical_oracle.get("receipt_id"):
        raise ValueError("invalid A25 partial Q0 numerical oracle binding")
    _validate_numerical_oracle_receipt(numerical_oracle)
    sidecar = receipt["gradient_sidecar"]
    twelve_payload, isolation_payload = _validated_sidecar_segments(
        sidecar, payload, diagnostic, isolation
    )
    _validate_inner_receipt(diagnostic, twelve_payload)
    _validate_isolation_receipt(isolation, isolation_payload)
    rng = receipt["global_rng_guard"]
    if (
        type(rng) is not dict
        or rng.get("seed") != A25_FULL_NUMERICAL_RNG_GUARD_SEED
        or rng.get("before") != rng.get("after")
    ):
        raise ValueError("invalid A25 partial Q0 RNG boundary")
    formal = receipt["formal_lock"]
    if formal != {
        "path": FORMAL_LOCK,
        "observed_absent_before": True,
        "observed_absent_after": True,
        "created": False,
    }:
        raise ValueError("invalid A25 partial Q0 formal boundary")
    gates = receipt["gates"]
    if (
        type(gates) is not dict
        or not gates
        or not all(type(value) is bool and value for value in gates.values())
    ):
        raise ValueError("invalid A25 partial Q0 gates")
    if receipt["claim_boundary"] != _claim_boundary():
        raise ValueError("invalid A25 partial Q0 claim boundary")
    if not isinstance(receipt["source"], Mapping) or not isinstance(
        receipt["runtime"], Mapping
    ):
        raise TypeError("invalid A25 partial Q0 provenance")


def _output_directory(root: Path, path: Path) -> tuple[Path, int]:
    raw = path
    if raw.is_symlink():
        raise ValueError("A25 partial Q0 output directory is a symlink")
    directory = raw.resolve(strict=True)
    repository = root.resolve(strict=True)
    if directory.is_relative_to(repository):
        raise ValueError("A25 partial Q0 output must be outside the repository")
    metadata = directory.stat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise ValueError("A25 partial Q0 output directory must be owner-only mode 0700")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open(directory, flags)
    opened = os.fstat(descriptor)
    if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
        os.close(descriptor)
        raise RuntimeError("A25 partial Q0 output directory changed")
    return directory, descriptor


def _exclusive_write_at(directory_fd: int, name: str, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open(name, flags, 0o600, dir_fd=directory_fd)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("short A25 partial Q0 bundle write")
            offset += written
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_size != len(payload)
        ):
            raise RuntimeError("unsafe A25 partial Q0 output file")
    finally:
        os.close(descriptor)


def _bounded_read_at(directory_fd: int, name: str, *, limit: int, label: str) -> bytes:
    before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) != 0o600
        or not 0 < before.st_size <= limit
    ):
        raise ValueError(f"unsafe A25 partial Q0 {label}")
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
        dir_fd=directory_fd,
    )
    try:
        opened = os.fstat(descriptor)
        identity = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size")
        if any(getattr(before, key) != getattr(opened, key) for key in identity):
            raise RuntimeError(f"A25 partial Q0 {label} changed before open")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, limit + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > limit:
                raise ValueError(f"A25 partial Q0 {label} is oversized")
        after = os.fstat(descriptor)
        current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if any(
            getattr(before, key) != getattr(after, key)
            or getattr(before, key) != getattr(current, key)
            for key in identity
        ):
            raise RuntimeError(f"A25 partial Q0 {label} changed during read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _read_bundle(directory_fd: int) -> tuple[bytes, bytes]:
    receipt = _bounded_read_at(
        directory_fd,
        RECEIPT_BASENAME,
        limit=MAX_RECEIPT_BYTES,
        label="receipt",
    )
    payload = _bounded_read_at(
        directory_fd,
        GRADIENT_BASENAME,
        limit=MAX_GRADIENT_BYTES,
        label="gradient sidecar",
    )
    return receipt, payload


def write_bundle(root: Path, output_dir: Path) -> dict[str, Any]:
    """Publish sidecar first and receipt last, without overwriting any path."""

    receipt, payload = recompute_bundle(root)
    receipt_bytes = _canonical_bytes(receipt) + b"\n"
    _, directory_fd = _output_directory(root, output_dir)
    try:
        for name in (GRADIENT_BASENAME, RECEIPT_BASENAME):
            try:
                os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            raise FileExistsError(f"A25 partial Q0 output exists: {name}")
        _exclusive_write_at(directory_fd, GRADIENT_BASENAME, payload)
        _exclusive_write_at(directory_fd, RECEIPT_BASENAME, receipt_bytes)
        os.fsync(directory_fd)
        reread_receipt, reread_payload = _read_bundle(directory_fd)
        recorded = _strict_json_bytes(reread_receipt)
        _validate_wrapper_record(recorded, reread_payload)
        if (
            _canonical_bytes(recorded) != _canonical_bytes(receipt)
            or reread_payload != payload
        ):
            raise RuntimeError("A25 partial Q0 published bundle changed")
    finally:
        os.close(directory_fd)
    if _canonical_bytes(_source_state(root)) != _canonical_bytes(receipt["source"]):
        raise RuntimeError("A25 partial Q0 source changed after publication")
    _assert_formal_lock_absent(root.resolve(strict=True))
    return receipt


def verify_bundle(root: Path, verify_dir: Path) -> dict[str, Any]:
    """Recompute in this process and require exact recorded bytes and payload."""

    _, directory_fd = _output_directory(root, verify_dir)
    try:
        first_receipt, first_payload = _read_bundle(directory_fd)
        recorded = _strict_json_bytes(first_receipt)
        _validate_wrapper_record(recorded, first_payload)
        expected, expected_payload = recompute_bundle(root)
        second_receipt, second_payload = _read_bundle(directory_fd)
        if first_receipt != second_receipt or first_payload != second_payload:
            raise RuntimeError("A25 partial Q0 bundle changed during verification")
        if (
            _canonical_bytes(recorded) != _canonical_bytes(expected)
            or first_payload != expected_payload
        ):
            raise ValueError(
                "A25 partial Q0 bundle differs from canonical recomputation"
            )
        return expected
    finally:
        os.close(directory_fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output-dir", type=Path)
    group.add_argument("--verify-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.output_dir is not None:
            receipt = write_bundle(args.root, args.output_dir)
        else:
            receipt = verify_bundle(args.root, args.verify_dir)
    except (
        FileExistsError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        print(
            json.dumps(
                {
                    "status": "ERROR",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "receipt_id": receipt["receipt_id"],
                "inner_receipt_id": receipt["inner_receipt_id"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
