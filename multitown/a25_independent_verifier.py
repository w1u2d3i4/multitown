"""Dependency-light verifier for persisted A25 partial numerical-Q0 bundles.

This module deliberately does not import the A25 producer, PPO code, shield
primitive, NumPy, or Torch.  It checks the persisted receipt through a
dependency-light code path, including source/runtime claims, binary manifests,
and numerical relations recoverable from the recorded float32 payload.  It
does not rerun autograd, training, or an outer/formal evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import stat
import struct
import subprocess
import sys
from collections.abc import Mapping, Sequence
from importlib import metadata as importlib_metadata
from pathlib import Path, PurePosixPath
from typing import Any

INDEPENDENT_VERIFIER_VERSION = "multitown-a25-independent-persisted-verifier-v2"
WRAPPER_VERSION = "multitown-a25-full-numerical-q0-wrapper-receipt-v2"
RUNNER_VERSION = "multitown-a25-full-numerical-q0-runner-v2"
SOURCE_VERSION = "multitown-a25-full-numerical-q0-source-v2"
RUNTIME_VERSION = "multitown-a25-full-numerical-q0-runtime-v1"
SIDECAR_VERSION = "multitown-a25-combined-gradient-sidecar-v1"
GRADIENT_ARTIFACT_VERSION = "multitown-a25-gradient-payload-v1"
RECEIPT_BASENAME = "receipt.json"
GRADIENT_BASENAME = "gradients.f32le.bin"
REPORT_BASENAME = "persisted-contract-verification.json"
FORMAL_LOCK = "artifacts/a24-cr-ppo-no-shield-attempt-v1.lock"
MAX_RECEIPT_BYTES = 64 * 1024 * 1024
MAX_GRADIENT_BYTES = 64 * 1024 * 1024
RNG_GUARD_SEED = 2026081504
FLOAT32_ATOL = 2e-6
FLOAT32_RTOL = 2e-5
ZERO_REFERENCE_CUTOFF = 1e-8
EXECUTE_ACTION_INDEX = 5
AUX_NOT_OBSERVED_REASON = "not-defined-in-frozen-a22-beta-zero-path"
PERSISTED_CONTRACT_STATUS = "PERSISTED_CONTRACT_CHECK_PASSED"

_HEX = frozenset("0123456789abcdef")
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
        raise RuntimeError("persisted-contract checker requires Git")
    path = Path(candidate)
    if path.is_symlink():
        raise RuntimeError("persisted-contract checker Git executable is a symlink")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise RuntimeError("persisted-contract checker Git executable is invalid")
    return resolved


_GIT_EXECUTABLE = _resolve_git_executable()
_MODULE_PATHS = {
    "multitown": "multitown/__init__.py",
    "multitown.a22_constrained_ppo": "multitown/a22_constrained_ppo.py",
    "multitown.a23_cr_ppo": "multitown/a23_cr_ppo.py",
    "multitown.a25_full_numerical_q0": ("multitown/a25_full_numerical_q0.py"),
    "multitown.a25_full_numerical_q0_runner": (
        "multitown/a25_full_numerical_q0_runner.py"
    ),
    "multitown.a25_independent_verifier": (
        "multitown/a25_independent_verifier.py"
    ),
    "multitown.a25_numerical_oracles": ("multitown/a25_numerical_oracles.py"),
    "multitown.a25_qualification": "multitown/a25_qualification.py",
    "multitown.a25_shield_dependence": ("multitown/a25_shield_dependence.py"),
    "multitown.a9_long_horizon_env": "multitown/a9_long_horizon_env.py",
    "multitown.a9_oof_protocol": "multitown/a9_oof_protocol.py",
    "multitown.a9_ppo_oof": "multitown/a9_ppo_oof.py",
    "multitown.long_horizon_env": "multitown/long_horizon_env.py",
    "multitown.ppo_controller": "multitown/ppo_controller.py",
    "multitown.pq1_numerical_conformance": (
        "multitown/pq1_numerical_conformance.py"
    ),
}
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
_EXPECTED_GATES = {
    "exact_clean_source_bound": True,
    "declared_execution_module_origins_bound": True,
    "global_rng_preserved": True,
    "formal_lock_absent_and_unmodified": True,
    "inner_diagnostic_passed": True,
    "fresh_adam_g3_isolation_passed": True,
    "numerical_oracles_g1_g4_g5_g6_passed": True,
    "gradient_sidecar_validated": True,
    "beta_zero_base_gradient_capture_passed": True,
    "dependency_light_checker_source_bound": True,
    "zero_outer_rows_read": True,
    "no_formal_authorization": True,
}
_EXPECTED_CLAIM = {
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
_EXPECTED_CONFIG = {
    "updates": 1,
    "episodes_per_update": 48,
    "dev_interval": 0,
    "learning_rate": 0.001,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_ratio": 0.2,
    "value_coef": 0.5,
    "entropy_coef": 0.02,
    "ppo_epochs": 4,
    "minibatch_size": 4096,
    "max_grad_norm": 0.5,
    "hidden_size": 8,
}
_EXPECTED_PARAMETER_SPECS = (
    ("backbone.0.weight", [8, 47]),
    ("backbone.0.bias", [8]),
    ("backbone.2.weight", [8, 8]),
    ("backbone.2.bias", [8]),
    ("actor.weight", [8, 8]),
    ("actor.bias", [8]),
    ("critic.weight", [1, 8]),
    ("critic.bias", [1]),
)
_TWELVE_CLAIM = {
    "twelve_cell_gradient_diagnostic_passed": True,
    "beta_zero_base_gradient_capture_passed": True,
    "clean_source_bound": False,
    "gradient_sidecar_bound": False,
    "full_numerical_q0_qualified": False,
    "q1_mechanism_qualified": False,
    "formal_authorized": False,
    "performance_claim_supported": False,
    "safety_claim_supported": False,
    "outer_rows_read": 0,
}
_TWELVE_GATES = {
    "twelve_cells_complete": True,
    "all_cells_passed": True,
    "beta_zero_base_gradient_capture_complete": True,
    "all_cells_share_common_prestate": True,
    "common_source_unmodified": True,
    "zero_outer_rows_read": True,
    "no_formal_lock_created": True,
}
_DEPENDENCE_KEYS = {
    "base_execute_probability_mean_all_decisions",
    "base_execute_probability_mean_shield_active",
    "counterfactual_argmax_intervention_fraction_all_decisions",
    "counterfactual_argmax_intervention_fraction_shield_active",
    "counterfactual_argmax_interventions",
    "decisions",
    "execute_without_review",
    "executed_execute_actions",
    "human_actions",
    "shield_active_decisions",
    "shield_active_fraction",
    "shielded_execute_probability_mass_mean_all_decisions",
    "stop_actions",
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


def _exact_equal(observed: Any, expected: Any) -> bool:
    """Compare JSON values without Python bool/int/float coercion."""

    return _canonical_bytes(observed) == _canonical_bytes(expected)


def _valid_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in _HEX for character in value)
    )


def _strict_json_bytes(payload: bytes) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate independent-verifier JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {token}")
            ),
        )
    except UnicodeDecodeError as exc:
        raise ValueError("invalid independent-verifier JSON encoding") from exc

    def reject_nonfinite(nested: Any) -> None:
        if type(nested) is float and not math.isfinite(nested):
            raise ValueError("non-finite independent-verifier JSON number")
        if type(nested) is dict:
            for item in nested.values():
                reject_nonfinite(item)
        elif type(nested) is list:
            for item in nested:
                reject_nonfinite(item)

    reject_nonfinite(value)
    if type(value) is not dict:
        raise TypeError("independent-verifier receipt must be an object")
    return value


def _fields(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"invalid {label} fields")
    return value


def _number(value: Any, label: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        raise TypeError(f"invalid finite number for {label}")
    return float(value)


def _positive_integer(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise TypeError(f"invalid positive integer for {label}")
    return value


def _close(
    observed: Any,
    expected: float,
    *,
    label: str,
    rtol: float = 1e-12,
    atol: float = 1e-12,
) -> None:
    value = _number(observed, label)
    if not math.isclose(value, expected, rel_tol=rtol, abs_tol=atol):
        raise ValueError(f"independent numerical mismatch for {label}")


def _float32_allowed(reference: float) -> float:
    return FLOAT32_ATOL + (
        FLOAT32_RTOL * abs(reference)
        if abs(reference) >= ZERO_REFERENCE_CUTOFF
        else 0.0
    )


def _validate_environment() -> None:
    present = sorted(_FORBIDDEN_ENVIRONMENT.intersection(os.environ))
    if present:
        raise RuntimeError(
            "independent verifier refuses source/import override environment: "
            + ",".join(present)
        )


def _git(root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        [str(_GIT_EXECUTABLE), *arguments],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if completed.returncode:
        raise RuntimeError(
            "independent verifier git command failed: " + " ".join(arguments)
        )
    return completed.stdout


def _assert_formal_lock_absent(root: Path) -> None:
    try:
        os.lstat(root / FORMAL_LOCK)
    except FileNotFoundError:
        return
    raise RuntimeError("independent verifier refuses an A24 formal lock")


def _safe_relative_path(raw: Any, label: str) -> PurePosixPath:
    if type(raw) is not str or not raw:
        raise TypeError(f"invalid path for {label}")
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe path for {label}")
    if path.as_posix() != raw:
        raise ValueError(f"non-canonical path for {label}")
    return path


def _validate_clean_source(root: Path, source: Any) -> dict[str, Any]:
    repository = root.resolve(strict=True)
    if root.is_symlink() or not repository.is_dir():
        raise ValueError("unsafe independent-verifier repository root")
    top = Path(
        _git(repository, "rev-parse", "--show-toplevel").decode().strip()
    ).resolve(strict=True)
    if top != repository:
        raise ValueError("independent-verifier root is not Git top level")
    if _git(repository, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("independent verifier requires exact clean source")
    source = _fields(
        source,
        {
            "schema_version",
            "revision",
            "tree",
            "source_sha256",
            "source_bundle_sha256",
            "module_origins",
            "git",
        },
        "source receipt",
    )
    if source["schema_version"] != SOURCE_VERSION:
        raise ValueError("unsupported independent-verifier source schema")
    revision = _git(repository, "rev-parse", "HEAD").decode().strip()
    tree = _git(repository, "rev-parse", "HEAD^{tree}").decode().strip()
    if source["revision"] != revision or source["tree"] != tree:
        raise ValueError("source revision/tree differs from clean HEAD")
    origins = _fields(
        source["module_origins"], set(_MODULE_PATHS), "module-origin mapping"
    )
    observed_sha: dict[str, str] = {}
    for module_name, expected_path in _MODULE_PATHS.items():
        row = _fields(
            origins[module_name], {"path", "sha256", "bytes"}, "module origin"
        )
        relative = _safe_relative_path(row["path"], module_name)
        if relative.as_posix() != expected_path:
            raise ValueError("module name/path binding changed")
        if not _valid_sha256(row["sha256"]):
            raise ValueError("invalid module source SHA-256")
        path = repository.joinpath(*relative.parts)
        if path.is_symlink():
            raise ValueError("module source is a symlink")
        resolved = path.resolve(strict=True)
        try:
            resolved.relative_to(repository)
        except ValueError as exc:
            raise ValueError("module source escapes repository") from exc
        file_stat = resolved.stat()
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
            raise ValueError("unsafe module source file")
        payload = resolved.read_bytes()
        if (
            type(row["bytes"]) is not int
            or row["bytes"] != len(payload)
            or row["sha256"] != hashlib.sha256(payload).hexdigest()
            or payload != _git(repository, "show", f"HEAD:{relative.as_posix()}")
        ):
            raise ValueError("module source bytes differ from receipt/HEAD")
        observed_sha[relative.as_posix()] = row["sha256"]
    if source["source_sha256"] != dict(sorted(observed_sha.items())) or source[
        "source_bundle_sha256"
    ] != _canonical_sha256(observed_sha):
        raise ValueError("invalid independent source bundle digest")
    git = _fields(
        source["git"],
        {"executable", "executable_sha256", "version"},
        "Git executable identity",
    )
    expected_git = {
        "executable": str(_GIT_EXECUTABLE),
        "executable_sha256": hashlib.sha256(_GIT_EXECUTABLE.read_bytes()).hexdigest(),
        "version": _git(repository, "--version").decode().strip(),
    }
    if not _exact_equal(git, expected_git):
        raise ValueError("Git executable identity differs from receipt")

    verifier = Path(__file__)
    if verifier.is_symlink():
        raise ValueError("independent verifier source is a symlink")
    verifier = verifier.resolve(strict=True)
    try:
        verifier_relative = verifier.relative_to(repository).as_posix()
    except ValueError as exc:
        raise ValueError("independent verifier is outside bound source") from exc
    verifier_payload = verifier.read_bytes()
    if verifier_payload != _git(repository, "show", f"HEAD:{verifier_relative}"):
        raise ValueError("independent verifier differs from clean HEAD")
    return {
        "revision": revision,
        "tree": tree,
        "source_bundle_sha256": source["source_bundle_sha256"],
        "verifier_path": verifier_relative,
        "verifier_sha256": hashlib.sha256(verifier_payload).hexdigest(),
        "git_executable_sha256": expected_git["executable_sha256"],
        "git_version": expected_git["version"],
    }


def _distribution_identity(name: str) -> tuple[str, str]:
    try:
        distribution = importlib_metadata.distribution(name)
    except importlib_metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"required runtime distribution missing: {name}") from exc
    origin = Path(distribution.locate_file(f"{name}/__init__.py"))
    if origin.is_symlink():
        raise ValueError(f"unsafe runtime distribution origin: {name}")
    origin = origin.resolve(strict=True)
    return distribution.version, hashlib.sha256(origin.read_bytes()).hexdigest()


def _validate_runtime(runtime: Any) -> dict[str, Any]:
    runtime = _fields(
        runtime,
        {
            "schema_version",
            "python",
            "numpy",
            "torch",
            "platform",
            "execution",
            "environment",
        },
        "runtime",
    )
    if runtime["schema_version"] != RUNTIME_VERSION:
        raise ValueError("unsupported independent-verifier runtime schema")
    python = _fields(
        runtime["python"],
        {"implementation", "version", "compiler", "executable_sha256"},
        "Python runtime",
    )
    executable = Path(sys.executable).resolve(strict=True)
    if python != {
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
        "compiler": platform.python_compiler(),
        "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
    }:
        raise ValueError("Python runtime differs from receipt")
    system = _fields(
        runtime["platform"],
        {"system", "release", "machine", "byteorder"},
        "platform runtime",
    )
    if system != {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "byteorder": sys.byteorder,
    }:
        raise ValueError("platform runtime differs from receipt")
    numpy = _fields(runtime["numpy"], {"version", "origin_sha256"}, "NumPy runtime")
    torch = _fields(
        runtime["torch"],
        {"version", "origin_sha256", "git_version", "cuda_build", "default_dtype"},
        "Torch runtime",
    )
    numpy_version, numpy_sha = _distribution_identity("numpy")
    torch_version, torch_sha = _distribution_identity("torch")
    if numpy != {"version": numpy_version, "origin_sha256": numpy_sha}:
        raise ValueError("NumPy distribution differs from receipt")
    if (
        type(torch["version"]) is not str
        or torch["version"].split("+", 1)[0] != torch_version
        or torch["origin_sha256"] != torch_sha
        or type(torch["git_version"]) is not str
        or len(torch["git_version"]) != 40
        or not all(character in _HEX for character in torch["git_version"])
        or (torch["cuda_build"] is not None and type(torch["cuda_build"]) is not str)
        or torch["default_dtype"] != "torch.float32"
    ):
        raise ValueError("Torch distribution metadata differs from receipt")
    execution = _fields(
        runtime["execution"],
        {"device", "torch_intraop_threads", "torch_interop_threads"},
        "execution runtime",
    )
    if (
        execution["device"] != "cpu"
        or type(execution["torch_intraop_threads"]) is not int
        or execution["torch_intraop_threads"] <= 0
        or type(execution["torch_interop_threads"]) is not int
        or execution["torch_interop_threads"] <= 0
    ):
        raise ValueError("invalid recorded execution runtime")
    environment = _fields(
        runtime["environment"], set(_RECORDED_ENVIRONMENT), "runtime environment"
    )
    expected_environment = {key: os.environ.get(key) for key in _RECORDED_ENVIRONMENT}
    if environment != expected_environment:
        raise ValueError("runtime environment differs from receipt")
    return {
        "python_version": python["version"],
        "numpy_version": numpy["version"],
        "torch_version": torch["version"],
        "device": execution["device"],
        "torch_thread_counts_reexecuted": False,
    }


def _validate_receipt_identity(receipt: Any, status: str, label: str) -> dict[str, Any]:
    if type(receipt) is not dict:
        raise TypeError(f"invalid {label} receipt")
    identity = receipt.get("receipt_id")
    core = {
        key: value
        for key, value in receipt.items()
        if key not in {"receipt_id", "status"}
    }
    if (
        receipt.get("status") != status
        or not _valid_sha256(identity)
        or identity != _canonical_sha256(core)
    ):
        raise ValueError(f"invalid {label} receipt identity")
    return receipt


def _float_vector(payload: bytes, label: str) -> tuple[float, ...]:
    if len(payload) % 4:
        raise ValueError(f"unaligned float32 payload for {label}")
    values = tuple(value[0] for value in struct.iter_unpack("<f", payload))
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"invalid float32 values for {label}")
    return values


def _validate_manifest(
    artifact: Any, payload: bytes
) -> dict[tuple[str, str], tuple[dict[str, Any], bytes, tuple[float, ...]]]:
    artifact = _fields(
        artifact,
        {"schema_version", "encoding", "sha256", "nbytes", "entry_count", "manifest"},
        "gradient artifact",
    )
    manifest = artifact["manifest"]
    if (
        artifact["schema_version"] != GRADIENT_ARTIFACT_VERSION
        or artifact["encoding"] != "contiguous-little-endian-float32"
        or artifact["sha256"] != hashlib.sha256(payload).hexdigest()
        or type(artifact["nbytes"]) is not int
        or artifact["nbytes"] != len(payload)
        or type(manifest) is not list
        or type(artifact["entry_count"]) is not int
        or artifact["entry_count"] != len(manifest)
    ):
        raise ValueError("invalid independent gradient artifact identity")
    indexed: dict[tuple[str, str], tuple[dict[str, Any], bytes, tuple[float, ...]]] = {}
    offset = 0
    for item in manifest:
        row = _fields(
            item,
            {"context", "name", "dtype", "shape", "offset", "nbytes", "sha256"},
            "gradient manifest row",
        )
        shape = row["shape"]
        if (
            type(row["context"]) is not str
            or not row["context"]
            or type(row["name"]) is not str
            or not row["name"]
            or row["dtype"] != "<f4"
            or type(shape) is not list
            or not shape
            or any(type(dimension) is not int or dimension <= 0 for dimension in shape)
            or type(row["offset"]) is not int
            or row["offset"] != offset
            or type(row["nbytes"]) is not int
            or row["nbytes"] != math.prod(shape) * 4
            or offset + row["nbytes"] > len(payload)
            or not _valid_sha256(row["sha256"])
        ):
            raise ValueError("invalid typed gradient manifest boundary")
        raw = payload[offset : offset + row["nbytes"]]
        key = (row["context"], row["name"])
        if row["sha256"] != hashlib.sha256(raw).hexdigest() or key in indexed:
            raise ValueError("invalid gradient manifest byte binding")
        indexed[key] = (row, raw, _float_vector(raw, row["name"]))
        offset += row["nbytes"]
    if offset != len(payload) or len(indexed) != len(manifest):
        raise ValueError("incomplete independent gradient payload coverage")
    return indexed


def _validate_gradient_summary(
    summary: Any,
    *,
    context: str,
    indexed: Mapping[tuple[str, str], tuple[dict[str, Any], bytes, tuple[float, ...]]],
    referenced: set[tuple[str, str]],
) -> tuple[
    tuple[str, ...], dict[str, tuple[list[int] | None, tuple[float, ...] | None]]
]:
    summary = _fields(
        summary, {"digest_sha256", "l2_norm", "parameters"}, "gradient summary"
    )
    if not _valid_sha256(summary["digest_sha256"]):
        raise ValueError("invalid gradient summary digest")
    parameters = summary["parameters"]
    if type(parameters) is not list or not parameters:
        raise TypeError("invalid gradient parameter sequence")
    digest = hashlib.sha256()
    squared_norm = 0.0
    names: list[str] = []
    values: dict[str, tuple[list[int] | None, tuple[float, ...] | None]] = {}
    for item in parameters:
        if type(item) is not dict:
            raise TypeError("invalid gradient parameter row")
        name = item.get("name")
        state = item.get("state")
        if type(name) is not str or not name or name in values:
            raise ValueError("invalid/duplicate gradient parameter name")
        names.append(name)
        digest.update(name.encode("utf-8") + b"\0")
        if state == "none":
            if set(item) != {"name", "state", "sha256"} or item["sha256"] is not None:
                raise ValueError("invalid None-gradient row")
            digest.update(b"none\0")
            values[name] = (None, None)
            continue
        if state != "tensor" or set(item) != {
            "name",
            "state",
            "shape",
            "sha256",
            "nbytes",
            "payload_offset",
        }:
            raise ValueError("invalid tensor-gradient row")
        key = (context, name)
        bound = indexed.get(key)
        if bound is None:
            raise ValueError("gradient parameter is absent from manifest")
        manifest, raw, vector = bound
        if (
            item["shape"] != manifest["shape"]
            or item["sha256"] != manifest["sha256"]
            or item["nbytes"] != manifest["nbytes"]
            or item["payload_offset"] != manifest["offset"]
        ):
            raise ValueError("gradient parameter differs from manifest")
        digest.update(b"tensor\0" + raw)
        squared_norm += math.fsum(value * value for value in vector)
        values[name] = (manifest["shape"], vector)
        referenced.add(key)
    if digest.hexdigest() != summary["digest_sha256"]:
        raise ValueError("gradient digest differs from independent bytes")
    _close(
        summary["l2_norm"],
        math.sqrt(squared_norm),
        label="gradient L2 norm",
        rtol=1e-11,
        atol=1e-12,
    )
    return tuple(names), values


def _validate_step(
    step: Any,
    *,
    cell_id: str,
    beta: float,
    indexed: Mapping[tuple[str, str], tuple[dict[str, Any], bytes, tuple[float, ...]]],
    referenced: set[tuple[str, str]],
) -> None:
    step = _fields(
        step,
        {
            "epoch",
            "minibatch",
            "indices_sha256",
            "gradients",
            "decomposition_max_abs_error",
            "decomposition_max_allowed_error",
            "preclip_total_max_abs_error",
            "clip",
            "gates",
            "passed",
        },
        "gradient step",
    )
    if (
        type(step["epoch"]) is not int
        or step["epoch"] < 0
        or type(step["minibatch"]) is not int
        or step["minibatch"] < 0
        or not _valid_sha256(step["indices_sha256"])
    ):
        raise ValueError("invalid typed gradient step boundary")
    gradients = _fields(
        step["gradients"],
        {
            "g_base",
            "g_aux_actual",
            "g_aux_actual_reason",
            "g_total",
            "preclip",
            "postclip",
        },
        "gradient kinds",
    )
    by_kind: dict[
        str, dict[str, tuple[list[int] | None, tuple[float, ...] | None]]
    ] = {}
    parameter_order: tuple[str, ...] | None = None
    for kind in ("g_base", "g_total", "preclip", "postclip"):
        context = (
            f"{cell_id}/epoch-{step['epoch']}/minibatch-{step['minibatch']}/{kind}"
        )
        names, values = _validate_gradient_summary(
            gradients[kind], context=context, indexed=indexed, referenced=referenced
        )
        if parameter_order is None:
            parameter_order = names
        elif names != parameter_order:
            raise ValueError("gradient parameter order differs across kinds")
        by_kind[kind] = values
    auxiliary = gradients["g_aux_actual"]
    auxiliary_reason = gradients["g_aux_actual_reason"]
    if beta == 0.0:
        if auxiliary is not None or auxiliary_reason != AUX_NOT_OBSERVED_REASON:
            raise ValueError("beta-zero auxiliary must be explicitly unobserved")
    else:
        if auxiliary is None or auxiliary_reason is not None:
            raise ValueError("positive-beta auxiliary gradient must be observed")
        context = (
            f"{cell_id}/epoch-{step['epoch']}/minibatch-{step['minibatch']}"
            "/g_aux_actual"
        )
        names, values = _validate_gradient_summary(
            auxiliary, context=context, indexed=indexed, referenced=referenced
        )
        if parameter_order is None or names != parameter_order:
            raise ValueError("auxiliary gradient parameter order differs")
        by_kind["g_aux_actual"] = values
    if parameter_order is None:
        raise ValueError("empty gradient step")
    shapes: dict[str, list[int]] = {}
    for name in parameter_order:
        observed_shapes = {
            tuple(shape)
            for kind in by_kind.values()
            for shape, vector in [kind[name]]
            if vector is not None and shape is not None
        }
        if len(observed_shapes) != 1:
            raise ValueError("gradient shape differs across kinds")
        shapes[name] = list(next(iter(observed_shapes)))

    def vector(kind: str, name: str) -> tuple[float, ...]:
        shape, values = by_kind[kind][name]
        if values is None:
            return (0.0,) * math.prod(shapes[name])
        if shape != shapes[name]:
            raise ValueError("gradient tensor shape mismatch")
        return values

    decomposition_error = 0.0
    decomposition_allowed = 0.0
    preclip_error = 0.0
    preclip_flat: list[float] = []
    postclip_flat: list[float] = []
    for name in parameter_order:
        base = vector("g_base", name)
        total = vector("g_total", name)
        preclip = vector("preclip", name)
        postclip = vector("postclip", name)
        auxiliary_values = (
            None if beta == 0.0 else vector("g_aux_actual", name)
        )
        for index, (base_value, total_value, pre_value) in enumerate(
            zip(base, total, preclip, strict=True)
        ):
            expected = (
                base_value
                if auxiliary_values is None
                else base_value + beta * auxiliary_values[index]
            )
            allowed = _float32_allowed(expected)
            error = abs(total_value - expected)
            if error > allowed:
                raise ValueError("base + beta * auxiliary differs from total")
            decomposition_error = max(decomposition_error, error)
            decomposition_allowed = max(decomposition_allowed, allowed)
            error = abs(pre_value - total_value)
            if error > _float32_allowed(total_value):
                raise ValueError("total differs from actual preclip gradient")
            preclip_error = max(preclip_error, error)
        preclip_flat.extend(preclip)
        postclip_flat.extend(postclip)
    preclip_norm = math.sqrt(math.fsum(value * value for value in preclip_flat))
    postclip_norm = math.sqrt(math.fsum(value * value for value in postclip_flat))
    clip = _fields(
        step["clip"],
        {
            "max_grad_norm",
            "returned_preclip_norm",
            "observed_preclip_norm",
            "observed_postclip_norm",
            "preclip_postclip_cosine",
        },
        "clip record",
    )
    max_norm = _number(clip["max_grad_norm"], "maximum gradient norm")
    returned = _number(clip["returned_preclip_norm"], "returned preclip norm")
    if max_norm <= 0.0:
        raise ValueError("non-positive maximum gradient norm")
    _close(
        clip["observed_preclip_norm"],
        preclip_norm,
        label="observed preclip norm",
        rtol=1e-11,
    )
    _close(
        clip["observed_postclip_norm"],
        postclip_norm,
        label="observed postclip norm",
        rtol=1e-11,
    )
    returned_matches = abs(returned - preclip_norm) <= _float32_allowed(preclip_norm)
    post_bounded = postclip_norm <= max_norm * (1.0 + FLOAT32_ATOL) + 1e-8
    denominator = preclip_norm * postclip_norm
    cosine = (
        math.fsum(
            before * after
            for before, after in zip(preclip_flat, postclip_flat, strict=True)
        )
        / denominator
        if denominator > 0.0
        else None
    )
    if cosine is None:
        if clip["preclip_postclip_cosine"] is not None:
            raise ValueError("invalid zero-norm clip cosine")
    else:
        _close(
            clip["preclip_postclip_cosine"],
            cosine,
            label="preclip/postclip cosine",
            rtol=1e-11,
        )
    direction = (
        cosine is not None and cosine >= 1.0 - FLOAT32_ATOL
        if preclip_norm > max_norm
        else True
    )
    gates = {
        "decomposition": True,
        "total_matches_actual_preclip": True,
        "returned_norm_matches_preclip": returned_matches,
        "postclip_norm_bounded": post_bounded,
        "clip_direction": direction,
    }
    if not _exact_equal(step["gates"], gates) or step["passed"] is not all(
        gates.values()
    ):
        raise ValueError("recorded gradient gates differ from independent checks")
    _close(
        step["decomposition_max_abs_error"],
        decomposition_error,
        label="decomposition maximum error",
        rtol=1e-9,
        atol=1e-7,
    )
    _close(
        step["decomposition_max_allowed_error"],
        decomposition_allowed,
        label="decomposition maximum allowance",
        rtol=1e-9,
        atol=1e-10,
    )
    _close(
        step["preclip_total_max_abs_error"],
        preclip_error,
        label="preclip maximum error",
        rtol=1e-9,
        atol=1e-7,
    )


def _validate_reference_metrics(value: Any, label: str) -> dict[str, Any]:
    metrics = _fields(
        value,
        {"policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"},
        label,
    )
    for key, observed in metrics.items():
        if type(observed) is not float:
            raise TypeError(f"invalid float metric for {label} {key}")
        _number(observed, f"{label} {key}")
    return metrics


def _validate_dependence_metrics(
    value: Any, *, decisions: int, shield_active: int, label: str
) -> dict[str, Any]:
    metrics = _fields(value, _DEPENDENCE_KEYS, label)
    integer_fields = {
        "counterfactual_argmax_interventions",
        "decisions",
        "execute_without_review",
        "executed_execute_actions",
        "human_actions",
        "shield_active_decisions",
        "stop_actions",
    }
    for key in integer_fields:
        if type(metrics[key]) is not int or metrics[key] < 0:
            raise TypeError(f"invalid integer dependence metric: {label} {key}")
    if (
        metrics["decisions"] != decisions
        or metrics["shield_active_decisions"] != shield_active
        or metrics["execute_without_review"] != 0
    ):
        raise ValueError(f"invalid frozen dependence counts: {label}")
    for key in _DEPENDENCE_KEYS - integer_fields:
        if type(metrics[key]) is not float:
            raise TypeError(f"invalid float dependence metric: {label} {key}")
        observed = _number(metrics[key], f"{label} {key}")
        if (
            "fraction" in key or "probability" in key or "mass" in key
        ) and not 0.0 <= observed <= 1.0:
            raise ValueError(f"invalid bounded dependence metric: {label} {key}")
    _close(
        metrics["shield_active_fraction"],
        shield_active / decisions,
        label=f"{label} shield-active fraction",
        rtol=1e-15,
        atol=1e-15,
    )
    return metrics


def _validate_policy_metrics(
    value: Any,
    *,
    beta: float,
    decisions: int,
    shield_active: int,
    label: str,
) -> dict[str, Any]:
    metrics = _fields(
        value,
        {
            "policy_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "clip_fraction",
            "intervention_beta",
            "intervention_loss",
            "intervention_penalty",
            "pre_update_shield_dependence",
            "post_update_shield_dependence",
        },
        label,
    )
    _validate_reference_metrics(
        {key: metrics[key] for key in (
            "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"
        )},
        label,
    )
    if any(
        type(metrics[key]) is not float
        for key in ("intervention_beta", "intervention_loss", "intervention_penalty")
    ):
        raise TypeError(f"invalid intervention metric type: {label}")
    observed_beta = _number(metrics["intervention_beta"], f"{label} beta")
    loss = _number(metrics["intervention_loss"], f"{label} intervention loss")
    penalty = _number(metrics["intervention_penalty"], f"{label} penalty")
    if observed_beta != beta or loss < 0.0:
        raise ValueError(f"invalid intervention objective metrics: {label}")
    _close(
        penalty,
        beta * loss,
        label=f"{label} intervention penalty",
        rtol=2e-6,
        atol=2e-7,
    )
    if beta == 0.0 and (loss != 0.0 or penalty != 0.0):
        raise ValueError(f"nonzero beta-zero intervention metrics: {label}")
    _validate_dependence_metrics(
        metrics["pre_update_shield_dependence"],
        decisions=decisions,
        shield_active=shield_active,
        label=f"{label} pre-update dependence",
    )
    _validate_dependence_metrics(
        metrics["post_update_shield_dependence"],
        decisions=decisions,
        shield_active=shield_active,
        label=f"{label} post-update dependence",
    )
    return metrics


def _validate_warm_common_state(value: Any) -> dict[str, str]:
    warm = _fields(
        value,
        {
            "schema_version",
            "seed",
            "warmup_seed",
            "warmup_batch_sha256",
            "warmup_generator_before_sha256",
            "warmup_generator_after_sha256",
            "model_sha256",
            "optimizer_sha256",
            "optimizer",
            "metrics",
        },
        "warm common state",
    )
    if (
        warm["schema_version"] != "multitown-a25-warm-adam-common-state-v1"
        or type(warm["seed"]) is not int
        or warm["seed"] != 2026081501
        or type(warm["warmup_seed"]) is not int
        or warm["warmup_seed"] != 2026081502
        or not all(
            _valid_sha256(warm[key])
            for key in (
                "warmup_batch_sha256",
                "warmup_generator_before_sha256",
                "warmup_generator_after_sha256",
                "model_sha256",
                "optimizer_sha256",
            )
        )
        or warm["warmup_generator_before_sha256"]
        == warm["warmup_generator_after_sha256"]
    ):
        raise ValueError("invalid warm common-state identity")
    _validate_reference_metrics(warm["metrics"], "warmup metrics")
    optimizer = _fields(
        warm["optimizer"],
        {"parameter_states", "all_parameter_moments_nonzero"},
        "warm Adam optimizer",
    )
    states = optimizer["parameter_states"]
    if (
        optimizer["all_parameter_moments_nonzero"] is not True
        or type(states) is not list
        or len(states) != len(_EXPECTED_PARAMETER_SPECS)
    ):
        raise ValueError("invalid warm Adam state coverage")
    for row, (expected_name, expected_shape) in zip(
        states, _EXPECTED_PARAMETER_SPECS, strict=True
    ):
        row = _fields(
            row,
            {
                "name",
                "step",
                "shape",
                "dtype",
                "exp_avg_nonzero",
                "exp_avg_sq_nonzero",
                "exp_avg_l2_norm",
                "exp_avg_sq_l2_norm",
                "exp_avg_sha256",
                "exp_avg_sq_sha256",
            },
            "warm Adam parameter state",
        )
        if (
            row["name"] != expected_name
            or row["shape"] != expected_shape
            or row["dtype"] != "torch.float32"
            or type(row["step"]) is not float
            or _number(row["step"], "warm Adam step") != 1.0
            or row["exp_avg_nonzero"] is not True
            or row["exp_avg_sq_nonzero"] is not True
            or _number(row["exp_avg_l2_norm"], "warm first-moment norm") <= 0.0
            or _number(row["exp_avg_sq_l2_norm"], "warm second-moment norm") <= 0.0
            or not _valid_sha256(row["exp_avg_sha256"])
            or not _valid_sha256(row["exp_avg_sq_sha256"])
        ):
            raise ValueError("invalid warm Adam parameter semantics")
    return {
        "model_sha256": warm["model_sha256"],
        "optimizer_sha256": warm["optimizer_sha256"],
    }


def _validate_transition_fixture(value: Any) -> None:
    fixture = _fields(
        value,
        {
            "schema_version",
            "episodes",
            "decisions",
            "unsafe_events",
            "wrong_executions",
            "shield_active_decisions",
            "unsafe_cost_per_episode",
            "wrong_cost_per_fixed_mean_incident",
            "mean_incidents",
            "transition_sha256",
        },
        "transition fixture",
    )
    expected = {
        "schema_version": "multitown-a25-full-numerical-fixture-v1",
        "episodes": 48,
        "decisions": 108,
        "unsafe_events": 36,
        "wrong_executions": 48,
        "shield_active_decisions": 36,
        "unsafe_cost_per_episode": 0.75,
        "wrong_cost_per_fixed_mean_incident": 0.25,
        "mean_incidents": 4.0,
    }
    observed_fixed = {key: fixture[key] for key in expected}
    if not _exact_equal(observed_fixed, expected):
        raise ValueError("frozen transition fixture semantics changed")
    if not _valid_sha256(fixture["transition_sha256"]):
        raise ValueError("invalid transition fixture digest")


def _validate_twelve_cell_semantics(receipt: dict[str, Any]) -> None:
    expected_fields = {
        "schema_version",
        "scope",
        "config",
        "warm_common_state",
        "transition_fixture",
        "common_state",
        "gradient_artifact",
        "cells",
        "gates",
        "claim_boundary",
        "receipt_id",
        "status",
    }
    if set(receipt) != expected_fields:
        raise ValueError("invalid twelve-cell receipt fields")
    if (
        receipt["schema_version"] != "multitown-a25-twelve-cell-diagnostic-receipt-v2"
        or receipt["scope"]
        != "synthetic-warm-adam-twelve-cell-diagnostic-no-outer-no-formal"
        or _canonical_bytes(receipt["config"]) != _canonical_bytes(_EXPECTED_CONFIG)
        or not _exact_equal(receipt["claim_boundary"], _TWELVE_CLAIM)
        or not _exact_equal(receipt["gates"], _TWELVE_GATES)
    ):
        raise ValueError("invalid twelve-cell frozen contract")
    warm = _validate_warm_common_state(receipt["warm_common_state"])
    _validate_transition_fixture(receipt["transition_fixture"])
    common = _fields(
        receipt["common_state"],
        {"model_sha256", "optimizer_sha256", "update_generator_prestate_sha256"},
        "common update state",
    )
    if (
        common["model_sha256"] != warm["model_sha256"]
        or common["optimizer_sha256"] != warm["optimizer_sha256"]
        or not _valid_sha256(common["update_generator_prestate_sha256"])
    ):
        raise ValueError("warm/common state binding failed")


def _expected_cell_contract(stress: str, arm: str) -> dict[str, Any]:
    if stress not in {"reward", "unsafe", "wrong"} or arm not in {
        "F00", "F01", "F10", "F11"
    }:
        raise ValueError("invalid cell identity")
    cr_thresholds = {
        "reward": {"unsafe": 1.0, "wrong_per_incident": 0.25, "mean_incidents": 4.0},
        "unsafe": {"unsafe": 0.5, "wrong_per_incident": 0.30, "mean_incidents": 4.0},
        "wrong": {"unsafe": 1.0, "wrong_per_incident": 0.20, "mean_incidents": 4.0},
    }
    duals = {
        "reward": {"unsafe": 0.0, "wrong_per_incident": 0.0},
        "unsafe": {"unsafe": 1.0, "wrong_per_incident": 0.0},
        "wrong": {"unsafe": 0.0, "wrong_per_incident": 1.0},
    }
    lagrangian = arm in {"F00", "F01"}
    beta_zero = arm in {"F00", "F10"}
    return {
        "update_rule": "ppo-lagrangian" if lagrangian else "cr-ppo",
        "actor_mode": None if lagrangian else stress,
        "dual": duals[stress] if lagrangian else None,
        "cr_thresholds": None if lagrangian else cr_thresholds[stress],
        "beta": 0.0 if beta_zero else 5.0,
    }


def _validate_cell_semantics(
    cell: Any, *, cell_id: str, common: Mapping[str, str]
) -> tuple[float, list[dict[str, Any]], dict[str, Any]]:
    cell = _fields(
        cell,
        {
            "cell_id",
            "stress",
            "arm",
            "update_rule",
            "actor_mode",
            "dual",
            "cr_thresholds",
            "beta",
            "batch_sha256",
            "pre",
            "steps",
            "beta_zero_reference",
            "post",
            "metrics",
            "gates",
            "passed",
        },
        "twelve-cell row",
    )
    stress, arm = cell_id.split("/", 1)
    expected = _expected_cell_contract(stress, arm)
    if (
        cell["cell_id"] != cell_id
        or cell["stress"] != stress
        or cell["arm"] != arm
        or not _exact_equal(
            {key: cell[key] for key in expected}, expected
        )
        or not _valid_sha256(cell["batch_sha256"])
    ):
        raise ValueError("invalid twelve-cell method contract")
    pre = _fields(
        cell["pre"],
        {"model_sha256", "optimizer_sha256", "rng_sha256"},
        "cell prestate",
    )
    expected_pre = {
        "model_sha256": common["model_sha256"],
        "optimizer_sha256": common["optimizer_sha256"],
        "rng_sha256": common["update_generator_prestate_sha256"],
    }
    if pre != expected_pre:
        raise ValueError("cell does not share common prestate")
    post = _fields(
        cell["post"],
        {"model_sha256", "optimizer_sha256", "rng_sha256"},
        "cell poststate",
    )
    if not all(_valid_sha256(value) for value in post.values()):
        raise ValueError("invalid cell poststate identity")
    beta = float(expected["beta"])
    metrics = _validate_policy_metrics(
        cell["metrics"],
        beta=beta,
        decisions=108,
        shield_active=36,
        label=f"{cell_id} metrics",
    )
    beta_zero = beta == 0.0
    expected_gates = {
        "common_pre_model": True,
        "common_pre_optimizer": True,
        "common_pre_rng": True,
        "four_observed_optimizer_steps": True,
        "all_gradient_steps_passed": True,
    }
    reference = cell["beta_zero_reference"]
    if beta_zero:
        expected_gates["beta_zero_reference_exact"] = True
        reference = _fields(
            reference,
            {
                "post_model_sha256",
                "post_optimizer_sha256",
                "post_rng_sha256",
                "metrics",
                "exact",
            },
            "beta-zero reference",
        )
        if (
            reference["exact"] is not True
            or reference["post_model_sha256"] != post["model_sha256"]
            or reference["post_optimizer_sha256"] != post["optimizer_sha256"]
            or reference["post_rng_sha256"] != post["rng_sha256"]
            or _validate_reference_metrics(
                reference["metrics"], f"{cell_id} reference metrics"
            )
            != {
                key: metrics[key]
                for key in (
                    "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"
                )
            }
        ):
            raise ValueError("beta-zero exact reference binding failed")
    elif reference is not None:
        raise ValueError("positive-beta cell has beta-zero reference")
    if (
        not _exact_equal(cell["gates"], expected_gates)
        or cell["passed"] is not all(expected_gates.values())
        or type(cell["steps"]) is not list
        or len(cell["steps"]) != 4
    ):
        raise ValueError("invalid twelve-cell gates/step count")
    return beta, cell["steps"], metrics


def _validate_isolation_semantics(receipt: dict[str, Any]) -> float:
    expected_fields = {
        "schema_version",
        "fixture",
        "beta_zero_metrics",
        "regularized_metrics",
        "blocked_execute_mass_delta",
        "gradient_step",
        "gradient_artifact",
        "gates",
        "claim_boundary",
        "receipt_id",
        "status",
    }
    if set(receipt) != expected_fields:
        raise ValueError("invalid G3 isolation receipt fields")
    expected_fixture = {
        "episodes": 48,
        "decisions": 96,
        "beta": 5.0,
        "entropy_coef": 0.0,
        "value_coef": 0.0,
        "optimizer": "fresh-adam",
        "direction_threshold": -1e-6,
    }
    expected_gates = {
        "base_advantage_exact_zero": True,
        "base_return_exact_zero": True,
        "g_base_exact_zero": True,
        "g_aux_nonzero": True,
        "critic_direct_aux_exact_none": True,
        "gradient_step_passed": True,
        "payload_valid": True,
        "beta_zero_parameters_unchanged": True,
        "regularized_mass_delta_below_threshold": True,
    }
    expected_claim = {
        "fresh_adam_isolation_diagnostic_passed": True,
        "warm_adam_equivalence_inferred": False,
        "full_numerical_q0_qualified": False,
        "formal_authorized": False,
        "performance_claim_supported": False,
    }
    if (
        receipt["schema_version"] != "multitown-a25-g3-isolation-diagnostic-v1"
        or _canonical_bytes(receipt["fixture"]) != _canonical_bytes(expected_fixture)
        or not _exact_equal(receipt["gates"], expected_gates)
        or not _exact_equal(receipt["claim_boundary"], expected_claim)
    ):
        raise ValueError("invalid frozen G3 isolation contract")
    beta_zero = _validate_policy_metrics(
        receipt["beta_zero_metrics"],
        beta=0.0,
        decisions=96,
        shield_active=48,
        label="G3 beta-zero metrics",
    )
    regularized = _validate_policy_metrics(
        receipt["regularized_metrics"],
        beta=5.0,
        decisions=96,
        shield_active=48,
        label="G3 regularized metrics",
    )
    if (
        beta_zero["pre_update_shield_dependence"]
        != regularized["pre_update_shield_dependence"]
    ):
        raise ValueError("G3 arms do not share pre-update dependence")
    expected_delta = (
        regularized["post_update_shield_dependence"]
        ["shielded_execute_probability_mass_mean_all_decisions"]
        - beta_zero["post_update_shield_dependence"]
        ["shielded_execute_probability_mass_mean_all_decisions"]
    )
    _close(
        receipt["blocked_execute_mass_delta"],
        expected_delta,
        label="G3 blocked-execute mass delta",
        rtol=1e-12,
        atol=1e-12,
    )
    return float(expected_fixture["beta"])


def _validate_gradient_receipt(
    receipt: Any,
    payload: bytes,
    *,
    isolation: bool,
) -> dict[str, int]:
    receipt = _validate_receipt_identity(receipt, "DIAGNOSTIC_PASSED", "gradient")
    indexed = _validate_manifest(receipt.get("gradient_artifact"), payload)
    referenced: set[tuple[str, str]] = set()
    step_count = 0
    if isolation:
        beta = _validate_isolation_semantics(receipt)
        direction = _number(
            receipt["fixture"]["direction_threshold"], "G3 direction"
        )
        if (
            _number(receipt.get("blocked_execute_mass_delta"), "G3 mass delta")
            >= direction
        ):
            raise ValueError("G3 mass direction threshold failed")
        _validate_step(
            receipt.get("gradient_step"),
            cell_id="isolation/G3",
            beta=beta,
            indexed=indexed,
            referenced=referenced,
        )
        step_count = 1
    else:
        _validate_twelve_cell_semantics(receipt)
        cells = receipt.get("cells")
        expected_cells = {
            f"{stress}/{arm}"
            for stress in ("reward", "unsafe", "wrong")
            for arm in ("F00", "F01", "F10", "F11")
        }
        cells = _fields(cells, expected_cells, "twelve-cell mapping")
        common = receipt["common_state"]
        pre_dependence: dict[str, Any] | None = None
        batch_by_stress_arm: dict[tuple[str, str], str] = {}
        for cell_id in sorted(cells):
            cell = cells[cell_id]
            beta, steps, metrics = _validate_cell_semantics(
                cell, cell_id=cell_id, common=common
            )
            observed_pre = metrics["pre_update_shield_dependence"]
            if pre_dependence is None:
                pre_dependence = observed_pre
            elif observed_pre != pre_dependence:
                raise ValueError("twelve cells do not share pre-update dependence")
            stress, arm = cell_id.split("/", 1)
            batch_by_stress_arm[(stress, arm)] = cell["batch_sha256"]
            observed_step_ids: set[tuple[int, int]] = set()
            for step in steps:
                _validate_step(
                    step,
                    cell_id=cell_id,
                    beta=beta,
                    indexed=indexed,
                    referenced=referenced,
                )
                step_id = (step["epoch"], step["minibatch"])
                if step_id in observed_step_ids:
                    raise ValueError("duplicate optimizer step identity")
                observed_step_ids.add(step_id)
                step_count += 1
            if observed_step_ids != {(epoch, 0) for epoch in range(4)}:
                raise ValueError("invalid epoch/minibatch coverage")
        for stress in ("reward", "unsafe", "wrong"):
            if (
                batch_by_stress_arm[(stress, "F00")]
                != batch_by_stress_arm[(stress, "F01")]
                or batch_by_stress_arm[(stress, "F10")]
                != batch_by_stress_arm[(stress, "F11")]
            ):
                raise ValueError("beta twins do not share the same batch")
    if referenced != set(indexed):
        raise ValueError("gradient sidecar contains unreferenced manifest entries")
    return {"steps": step_count, "manifest_entries": len(indexed)}


def _matrix(value: Any, label: str) -> list[list[float]]:
    if type(value) is not list or not value:
        raise TypeError(f"invalid matrix for {label}")
    result: list[list[float]] = []
    width: int | None = None
    for row in value:
        if type(row) is not list or not row:
            raise TypeError(f"invalid matrix row for {label}")
        values = [_number(item, label) for item in row]
        if width is None:
            width = len(values)
        elif len(values) != width:
            raise ValueError(f"ragged matrix for {label}")
        result.append(values)
    return result


def _masked_probabilities(logits: list[float], mask: list[int]) -> list[float]:
    legal = [value for value, allowed in zip(logits, mask, strict=True) if allowed]
    if not legal:
        raise ValueError("empty independent masked-softmax row")
    maximum = max(legal)
    weights = [
        math.exp(value - maximum) if allowed else 0.0
        for value, allowed in zip(logits, mask, strict=True)
    ]
    denominator = math.fsum(weights)
    return [weight / denominator for weight in weights]


def _g1_scalar_loss(
    logits: list[list[float]], masks: list[list[int]], active: list[bool]
) -> float:
    count = len(logits)
    return math.fsum(
        _masked_probabilities(row, mask)[EXECUTE_ACTION_INDEX] / count
        for row, mask, enabled in zip(logits, masks, active, strict=True)
        if enabled and mask[EXECUTE_ACTION_INDEX]
    )


def _validate_g1(result: Any) -> None:
    if (
        type(result) is not dict
        or result.get("schema_version") != "multitown-a25-g1-analytic-oracle-v1"
    ):
        raise ValueError("invalid independent G1 oracle")
    logits = _matrix(result.get("logits_f64"), "G1 logits")
    analytic = _matrix(result.get("analytic_gradient_f64"), "G1 analytic gradient")
    finite_difference = _matrix(
        result.get("finite_difference_gradient_f64"), "G1 finite difference"
    )
    observed = _matrix(result.get("observed_gradient_f32"), "G1 observed gradient")
    raw_masks = result.get("base_masks")
    active = result.get("active")
    if (
        type(raw_masks) is not list
        or len(raw_masks) != len(logits)
        or type(active) is not list
        or len(active) != len(logits)
        or any(type(value) is not bool for value in active)
    ):
        raise TypeError("invalid G1 mask/active fixture")
    masks: list[list[int]] = []
    for row, logit_row in zip(raw_masks, logits, strict=True):
        if (
            type(row) is not list
            or len(row) != len(logit_row)
            or any(type(value) is not int or value not in {0, 1} for value in row)
        ):
            raise ValueError("invalid typed G1 mask")
        masks.append(row)
    if any(
        len(matrix) != len(logits) or any(len(row) != len(logits[0]) for row in matrix)
        for matrix in (analytic, finite_difference, observed)
    ):
        raise ValueError("invalid G1 gradient shape")
    expected = [[0.0] * len(logits[0]) for _ in logits]
    probabilities = [
        _masked_probabilities(row, mask)
        for row, mask in zip(logits, masks, strict=True)
    ]
    count = len(logits)
    for index, (probability, enabled, mask) in enumerate(
        zip(probabilities, active, masks, strict=True)
    ):
        if not enabled or not mask[EXECUTE_ACTION_INDEX]:
            continue
        execute = probability[EXECUTE_ACTION_INDEX]
        for action, allowed in enumerate(mask):
            if allowed:
                expected[index][action] = -execute * probability[action] / count
        expected[index][EXECUTE_ACTION_INDEX] += execute / count
    loss = _g1_scalar_loss(logits, masks, active)
    _close(result.get("analytic_loss_f64"), loss, label="G1 analytic loss")
    fd = result.get("finite_difference")
    if type(fd) is not dict:
        raise TypeError("invalid G1 finite-difference gate")
    step = _number(fd.get("step"), "G1 finite-difference step")
    fd_atol = _number(fd.get("atol"), "G1 finite-difference tolerance")
    max_fd_error = 0.0
    max_observed_error = 0.0
    for row_index in range(len(logits)):
        for column_index in range(len(logits[0])):
            _close(
                analytic[row_index][column_index],
                expected[row_index][column_index],
                label="G1 analytic gradient",
                rtol=1e-10,
            )
            positive = [row.copy() for row in logits]
            negative = [row.copy() for row in logits]
            positive[row_index][column_index] += step
            negative[row_index][column_index] -= step
            expected_fd = (
                _g1_scalar_loss(positive, masks, active)
                - _g1_scalar_loss(negative, masks, active)
            ) / (2.0 * step)
            _close(
                finite_difference[row_index][column_index],
                expected_fd,
                label="G1 finite-difference gradient",
                rtol=1e-8,
            )
            max_fd_error = max(
                max_fd_error,
                abs(
                    finite_difference[row_index][column_index]
                    - expected[row_index][column_index]
                ),
            )
            observed_error = abs(
                observed[row_index][column_index] - expected[row_index][column_index]
            )
            if observed_error > _float32_allowed(expected[row_index][column_index]):
                raise ValueError("G1 production gradient differs from analytic oracle")
            max_observed_error = max(max_observed_error, observed_error)
    if max_fd_error > fd_atol:
        raise ValueError("G1 finite-difference tolerance failed")
    _close(
        fd.get("max_abs_error"), max_fd_error, label="G1 maximum FD error", rtol=1e-7
    )
    recorded_probabilities = result.get("base_execute_probability_f32")
    if type(recorded_probabilities) is not list or len(recorded_probabilities) != len(
        logits
    ):
        raise TypeError("invalid G1 execute probabilities")
    for recorded, probability in zip(
        recorded_probabilities, probabilities, strict=True
    ):
        if (
            abs(
                _number(recorded, "G1 execute probability")
                - probability[EXECUTE_ACTION_INDEX]
            )
            > FLOAT32_ATOL
        ):
            raise ValueError("G1 execute probability mismatch")
    production_gate = result.get("production_gradient_gate")
    if type(production_gate) is not dict or production_gate.get("passed") is not True:
        raise ValueError("invalid G1 production gate")
    if (
        _number(production_gate.get("max_abs_error"), "G1 observed error") + 1e-8
        < max_observed_error
    ):
        raise ValueError("G1 production error under-reported")
    expected_gates = {
        "float64_finite_difference": True,
        "float32_production_gradient": True,
        "inactive_exact_zero": all(value == 0.0 for value in observed[2]),
        "base_illegal_exact_zero": all(value == 0.0 for value in observed[3]),
        "active_row_gradient_sum": max(
            abs(math.fsum(observed[index])) for index in (0, 1)
        )
        <= FLOAT32_ATOL,
    }
    if not _exact_equal(result.get("gates"), expected_gates) or result.get(
        "passed"
    ) is not True:
        raise ValueError("G1 gates differ from independent equations")


def _validate_g4(result: Any) -> None:
    if (
        type(result) is not dict
        or result.get("schema_version") != "multitown-a25-g4-selector-oracle-v1"
    ):
        raise ValueError("invalid independent G4 oracle")
    counts = result.get("counts")
    if type(counts) is not dict:
        raise TypeError("invalid G4 counts")
    episodes = _positive_integer(counts.get("episodes"), "G4 episodes")
    unsafe = _positive_integer(counts.get("unsafe_events"), "G4 unsafe count")
    wrong = _positive_integer(counts.get("wrong_executions"), "G4 wrong count")
    observed_unsafe = unsafe / episodes
    boundary = result.get("boundary")
    tie = result.get("tie")
    if type(boundary) is not dict or type(tie) is not dict:
        raise TypeError("invalid G4 cases")
    mean_incidents = _number(
        boundary.get("thresholds", {}).get("mean_incidents"), "G4 incidents"
    )
    observed_wrong = wrong / (episodes * mean_incidents)
    observed = result.get("observed_costs")
    if type(observed) is not dict:
        raise TypeError("invalid G4 observed costs")
    _close(observed.get("unsafe"), observed_unsafe, label="G4 unsafe cost")
    _close(observed.get("wrong_per_incident"), observed_wrong, label="G4 wrong cost")
    boundary_decision = boundary.get("decision")
    tie_decision = tie.get("decision")
    if (
        type(boundary_decision) is not dict
        or boundary_decision.get("mode") != "reward"
        or boundary_decision.get("unsafe_eligible") is not False
        or boundary_decision.get("wrong_eligible") is not False
        or type(tie_decision) is not dict
        or tie_decision.get("mode") != "unsafe"
        or tie_decision.get("unsafe_eligible") is not True
        or tie_decision.get("wrong_eligible") is not True
        or tie_decision.get("unsafe_tie_break_used") is not True
    ):
        raise ValueError("G4 selector boundary/tie semantics failed")
    _close(
        tie_decision.get("unsafe_normalized_violation"),
        1.0,
        label="G4 unsafe normalized violation",
    )
    _close(
        tie_decision.get("wrong_normalized_violation"),
        1.0,
        label="G4 wrong normalized violation",
    )
    if not all(result.get("gates", {}).values()) or result.get("passed") is not True:
        raise ValueError("G4 gates failed")


def _validate_g5(result: Any) -> None:
    if (
        type(result) is not dict
        or result.get("schema_version") != "multitown-a25-g5-shared-backbone-oracle-v1"
    ):
        raise ValueError("invalid independent G5 oracle")
    summary = result.get("gradient_summary")
    if type(summary) is not dict or type(summary.get("parameters")) is not list:
        raise TypeError("invalid G5 gradient summary")
    calculated: dict[str, dict[str, float | int]] = {
        name: {"squared": 0.0, "none_count": 0, "tensor_count": 0}
        for name in ("backbone", "actor", "critic")
    }
    critic_none: list[str] = []
    for parameter in summary["parameters"]:
        if type(parameter) is not dict or type(parameter.get("name")) is not str:
            raise TypeError("invalid G5 parameter")
        group = parameter["name"].split(".", 1)[0]
        if group not in calculated:
            raise ValueError("unknown G5 parameter group")
        if parameter.get("state") == "none":
            if parameter.get("l2_norm") is not None:
                raise ValueError("invalid G5 None-gradient norm")
            calculated[group]["none_count"] += 1
            if group == "critic":
                critic_none.append(parameter["name"])
        elif parameter.get("state") == "tensor":
            norm = _number(parameter.get("l2_norm"), "G5 parameter norm")
            calculated[group]["squared"] += norm * norm
            calculated[group]["tensor_count"] += 1
        else:
            raise ValueError("invalid G5 gradient state")
    groups = summary.get("groups")
    if type(groups) is not dict or set(groups) != set(calculated):
        raise ValueError("invalid G5 groups")
    for name, values in calculated.items():
        row = groups[name]
        if (
            type(row) is not dict
            or row.get("none_count") != values["none_count"]
            or row.get("tensor_count") != values["tensor_count"]
        ):
            raise ValueError("G5 group counts mismatch")
        _close(
            row.get("l2_norm"),
            math.sqrt(float(values["squared"])),
            label=f"G5 {name} norm",
            rtol=1e-10,
        )
    if (
        critic_none != result.get("critic_none_names")
        or critic_none != result.get("critic_parameter_names")
        or groups["backbone"]["l2_norm"] <= 0.0
        or groups["actor"]["l2_norm"] <= 0.0
        or groups["critic"]["tensor_count"] != 0
        or not all(result.get("gates", {}).values())
        or result.get("passed") is not True
    ):
        raise ValueError("G5 shared-backbone boundary failed")


def _validate_g6(result: Any) -> None:
    if (
        type(result) is not dict
        or result.get("schema_version") != "multitown-a25-g6-global-clip-oracle-v1"
    ):
        raise ValueError("invalid independent G6 oracle")
    base = _number(result.get("base_total_loss_f32"), "G6 base loss")
    auxiliary = _number(result.get("auxiliary_loss_f32"), "G6 auxiliary loss")
    beta = _number(result.get("beta"), "G6 beta")
    total = _number(result.get("total_loss_f32"), "G6 total loss")
    if abs(total - (base + beta * auxiliary)) > _float32_allowed(total):
        raise ValueError("G6 total loss decomposition failed")
    preclip = _number(result.get("preclip_norm"), "G6 preclip norm")
    postclip = _number(result.get("postclip_norm"), "G6 postclip norm")
    maximum = _number(result.get("max_grad_norm"), "G6 max norm")
    returned = _number(
        result.get("clip_returned_total_norm"), "G6 returned preclip norm"
    )
    cosine = _number(result.get("preclip_postclip_cosine"), "G6 cosine")
    _close(
        result.get("preclip_to_max_norm_ratio"),
        preclip / maximum,
        label="G6 preclip/max ratio",
        rtol=1e-12,
    )
    gates = {
        "base_loss_exact_zero": base == 0.0,
        "preclip_exceeds_100x": preclip > 100.0 * maximum,
        "returned_norm_matches_preclip": abs(returned - preclip)
        <= _float32_allowed(preclip),
        "postclip_norm_bounded": postclip <= maximum * (1.0 + FLOAT32_ATOL) + 1e-8,
        "direction_preserved": cosine >= 1.0 - FLOAT32_ATOL,
        "gradient_none_pattern_preserved": result.get("preclip_none_count")
        == result.get("postclip_none_count"),
    }
    if not _exact_equal(result.get("gates"), gates) or result.get(
        "passed"
    ) is not all(gates.values()):
        raise ValueError("G6 clipping gates differ from independent equations")


def _validate_numerical_oracles(receipt: Any) -> None:
    receipt = _validate_receipt_identity(
        receipt, "DIAGNOSTIC_PASSED", "numerical oracle"
    )
    diagnostics = _fields(
        receipt.get("diagnostics"), {"G1", "G4", "G5", "G6"}, "oracle diagnostics"
    )
    _validate_g1(diagnostics["G1"])
    _validate_g4(diagnostics["G4"])
    _validate_g5(diagnostics["G5"])
    _validate_g6(diagnostics["G6"])
    if not _exact_equal(receipt.get("gates"), {
        "all_expected_diagnostics_present": True,
        "all_diagnostics_passed": True,
        "zero_outer_rows_read": True,
        "no_formal_lock_created": True,
    }):
        raise ValueError("invalid numerical-oracle top-level gates")
    claim = receipt.get("claim_boundary")
    if (
        type(claim) is not dict
        or claim.get("diagnostic_primitives_passed") is not True
        or claim.get("qualification_evidence") is not False
        or claim.get("full_numerical_q0_qualified") is not False
        or claim.get("formal_authorized") is not False
        or claim.get("performance_claim_supported") is not False
        or claim.get("safety_claim_supported") is not False
        or claim.get("outer_rows_read") != 0
        or claim.get("formal_lock_created") is not False
    ):
        raise ValueError("invalid numerical-oracle claim boundary")


def _validate_segments(
    sidecar: Any,
    payload: bytes,
    diagnostic: Mapping[str, Any],
    isolation: Mapping[str, Any],
) -> tuple[bytes, bytes]:
    sidecar = _fields(
        sidecar,
        {"schema_version", "basename", "sha256", "nbytes", "encoding", "segments"},
        "combined sidecar",
    )
    segments = sidecar["segments"]
    if (
        sidecar["schema_version"] != SIDECAR_VERSION
        or sidecar["basename"] != GRADIENT_BASENAME
        or sidecar["encoding"] != "ordered-byte-concatenation-v1"
        or sidecar["sha256"] != hashlib.sha256(payload).hexdigest()
        or type(sidecar["nbytes"]) is not int
        or sidecar["nbytes"] != len(payload)
        or type(segments) is not list
        or len(segments) != 2
    ):
        raise ValueError("invalid independent combined-sidecar identity")
    expected = (("twelve-cell", diagnostic), ("isolation-g3", isolation))
    offset = 0
    result: list[bytes] = []
    for segment, (name, receipt) in zip(segments, expected, strict=True):
        segment = _fields(
            segment,
            {
                "name",
                "offset",
                "nbytes",
                "sha256",
                "receipt_id",
                "artifact_sha256",
                "artifact_nbytes",
            },
            "sidecar segment",
        )
        artifact = receipt.get("gradient_artifact")
        if type(artifact) is not dict:
            raise TypeError("missing segment gradient artifact")
        length = _positive_integer(segment["nbytes"], "sidecar segment length")
        if (
            segment["name"] != name
            or type(segment["offset"]) is not int
            or segment["offset"] != offset
            or offset + length > len(payload)
        ):
            raise ValueError("invalid sidecar segment layout")
        raw = payload[offset : offset + length]
        if (
            segment["sha256"] != hashlib.sha256(raw).hexdigest()
            or segment["receipt_id"] != receipt.get("receipt_id")
            or segment["artifact_sha256"] != artifact.get("sha256")
            or segment["artifact_nbytes"] != artifact.get("nbytes")
            or segment["artifact_sha256"] != segment["sha256"]
            or segment["artifact_nbytes"] != length
        ):
            raise ValueError("invalid sidecar segment binding")
        result.append(raw)
        offset += length
    if offset != len(payload):
        raise ValueError("incomplete sidecar segment coverage")
    return result[0], result[1]


def _output_directory(root: Path, path: Path) -> int:
    if path.is_symlink():
        raise ValueError("independent-verifier bundle directory is a symlink")
    directory = path.resolve(strict=True)
    repository = root.resolve(strict=True)
    if directory.is_relative_to(repository):
        raise ValueError("independent-verifier bundle must be outside repository")
    metadata = directory.stat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise ValueError("bundle directory must be owner-only mode 0700")
    descriptor = os.open(
        directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    )
    opened = os.fstat(descriptor)
    if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
        os.close(descriptor)
        raise RuntimeError("bundle directory changed during open")
    return descriptor


def _read_at(directory_fd: int, name: str, limit: int) -> bytes:
    before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) != 0o600
        or not 0 < before.st_size <= limit
    ):
        raise ValueError(f"unsafe independent-verifier bundle file: {name}")
    descriptor = os.open(
        name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=directory_fd
    )
    try:
        opened = os.fstat(descriptor)
        identity = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(getattr(before, key) != getattr(opened, key) for key in identity):
            raise RuntimeError(f"bundle file changed before read: {name}")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, limit + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > limit:
                raise ValueError(f"oversized bundle file: {name}")
        after = os.fstat(descriptor)
        current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if any(
            getattr(before, key) != getattr(after, key)
            or getattr(before, key) != getattr(current, key)
            for key in identity
        ):
            raise RuntimeError(f"bundle file changed during read: {name}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def write_verification_report(
    root: Path, bundle_dir: Path
) -> tuple[Path, dict[str, Any]]:
    """Verify the subject internally, then persist its exact checker report."""

    report = verify_bundle(root, bundle_dir)
    payload = _canonical_bytes(report) + b"\n"
    directory_fd = _output_directory(root.resolve(strict=True), bundle_dir)
    descriptor: int | None = None
    created = False
    committed = False
    try:
        descriptor = os.open(
            REPORT_BASENAME,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_CLOEXEC
            | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        created = True
        if not _exact_equal(verify_bundle(root, bundle_dir), report):
            raise RuntimeError("bundle subject changed before report write")
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count <= 0:
                raise OSError("short independent verification report write")
            written += count
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.fsync(directory_fd)
        observed = _read_at(directory_fd, REPORT_BASENAME, MAX_RECEIPT_BYTES)
        if observed != payload:
            raise RuntimeError("independent verification report reread mismatch")
        if not _exact_equal(verify_bundle(root, bundle_dir), report):
            raise RuntimeError("bundle subject changed after report write")
        committed = True
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if created and not committed:
            os.unlink(REPORT_BASENAME, dir_fd=directory_fd)
            os.fsync(directory_fd)
        os.close(directory_fd)
    return bundle_dir.resolve(strict=True) / REPORT_BASENAME, report


def verify_bundle(root: Path, bundle_dir: Path) -> dict[str, Any]:
    """Check persisted contracts without authenticating provenance or training."""

    _validate_environment()
    repository = root.resolve(strict=True)
    _assert_formal_lock_absent(repository)
    directory_fd = _output_directory(repository, bundle_dir)
    try:
        receipt_bytes = _read_at(directory_fd, RECEIPT_BASENAME, MAX_RECEIPT_BYTES)
        payload = _read_at(directory_fd, GRADIENT_BASENAME, MAX_GRADIENT_BYTES)
        receipt = _strict_json_bytes(receipt_bytes)
        if receipt_bytes != _canonical_bytes(receipt) + b"\n":
            raise ValueError("receipt file is not exact canonical JSON plus newline")
        if set(receipt) != _WRAPPER_CORE_KEYS | {"receipt_id", "status"}:
            raise ValueError("invalid independent wrapper fields")
        core = {key: receipt[key] for key in _WRAPPER_CORE_KEYS}
        if (
            receipt["schema_version"] != WRAPPER_VERSION
            or receipt["runner_version"] != RUNNER_VERSION
            or receipt["scope"]
            != "clean-source-synthetic-partial-numerical-q0-no-outer-no-formal"
            or receipt["status"] != "PARTIAL_Q0_PASSED"
            or not _valid_sha256(receipt["receipt_id"])
            or receipt["receipt_id"] != _canonical_sha256(core)
        ):
            raise ValueError("invalid independent wrapper identity")
        source = _validate_clean_source(repository, receipt["source"])
        runtime = _validate_runtime(receipt["runtime"])
        if not _exact_equal(receipt["formal_lock"], {
            "path": FORMAL_LOCK,
            "observed_absent_before": True,
            "observed_absent_after": True,
            "created": False,
        }):
            raise ValueError("invalid formal-lock claim")
        rng = receipt["global_rng_guard"]
        if (
            type(rng) is not dict
            or set(rng) != {"seed", "before", "after"}
            or type(rng["seed"]) is not int
            or rng["seed"] != RNG_GUARD_SEED
            or rng["before"] != rng["after"]
            or type(rng["before"]) is not dict
            or set(rng["before"])
            != {"python_sha256", "numpy_sha256", "torch_cpu_sha256"}
            or not all(_valid_sha256(value) for value in rng["before"].values())
        ):
            raise ValueError("invalid global-RNG preservation claim")
        if (
            not _exact_equal(receipt["gates"], _EXPECTED_GATES)
            or not _exact_equal(receipt["claim_boundary"], _EXPECTED_CLAIM)
        ):
            raise ValueError("wrapper gates/claim boundary changed")
        diagnostic = _validate_receipt_identity(
            receipt["diagnostic_receipt"], "DIAGNOSTIC_PASSED", "twelve-cell"
        )
        isolation = _validate_receipt_identity(
            receipt["isolation_diagnostic"], "DIAGNOSTIC_PASSED", "G3"
        )
        numerical = _validate_receipt_identity(
            receipt["numerical_oracle_receipt"],
            "DIAGNOSTIC_PASSED",
            "numerical oracle",
        )
        if (
            receipt["inner_receipt_id"] != diagnostic["receipt_id"]
            or receipt["isolation_receipt_id"] != isolation["receipt_id"]
            or receipt["numerical_oracle_receipt_id"] != numerical["receipt_id"]
        ):
            raise ValueError("nested receipt ID binding failed")
        twelve_payload, isolation_payload = _validate_segments(
            receipt["gradient_sidecar"], payload, diagnostic, isolation
        )
        twelve = _validate_gradient_receipt(diagnostic, twelve_payload, isolation=False)
        g3 = _validate_gradient_receipt(isolation, isolation_payload, isolation=True)
        _validate_numerical_oracles(numerical)
        _assert_formal_lock_absent(repository)
        if (
            _read_at(directory_fd, RECEIPT_BASENAME, MAX_RECEIPT_BYTES)
            != receipt_bytes
            or _read_at(directory_fd, GRADIENT_BASENAME, MAX_GRADIENT_BYTES)
            != payload
        ):
            raise RuntimeError("bundle subject changed during verification")
        if _git(repository, "rev-parse", "HEAD").decode().strip() != source[
            "revision"
        ] or _git(repository, "status", "--porcelain=v1", "--untracked-files=all"):
            raise RuntimeError("source changed during independent verification")
        report_core = {
            "schema_version": INDEPENDENT_VERIFIER_VERSION,
            "status": PERSISTED_CONTRACT_STATUS,
            "receipt_id": receipt["receipt_id"],
            "source": source,
            "runtime": runtime,
            "gradient_contract": {
                "twelve_cell_steps": twelve["steps"],
                "twelve_cell_manifest_entries": twelve["manifest_entries"],
                "isolation_steps": g3["steps"],
                "isolation_manifest_entries": g3["manifest_entries"],
                "combined_sidecar_sha256": hashlib.sha256(payload).hexdigest(),
                "combined_sidecar_nbytes": len(payload),
            },
            "claim_boundary": {
                "dependency_light_checker_path": True,
                "same_project_trust_domain": True,
                "content_identity_verified": True,
                "persisted_receipt_schema_and_relations_checked": True,
                "opaque_fixture_state_digests_recomputed": False,
                "persisted_gradient_algebra_verified": True,
                "source_claims_checked_against_clean_head": True,
                "python_platform_distribution_metadata_checked": True,
                "authenticated_provenance": False,
                "freshness_attested": False,
                "torch_execution_runtime_reexecuted": False,
                "autograd_reproduced": False,
                "optimizer_training_reproduced": False,
                "full_numerical_q0_qualified": False,
                "q1_mechanism_qualified": False,
                "formal_authorized": False,
                "outer_rows_read": 0,
                "performance_claim_supported": False,
                "safety_claim_supported": False,
            },
        }
        return {
            **report_core,
            "verification_id": _canonical_sha256(report_core),
        }
    finally:
        os.close(directory_fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--verify-dir", type=Path, required=True)
    parser.add_argument("--write-report", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.write_report:
            _, report = write_verification_report(args.root, args.verify_dir)
        else:
            report = verify_bundle(args.root, args.verify_dir)
    except (
        FileNotFoundError,
        KeyError,
        OSError,
        RecursionError,
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
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
