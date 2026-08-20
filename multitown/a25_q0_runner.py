"""Run the replay-bound A25 common-state Q0 qualification fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import stat
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .a9_oof_protocol import (
    EXPECTED_TRAIN_SHA256,
    FROZEN_TRAIN_PATH,
    assign_stratified_group_folds,
    fold_manifest_sha256,
    load_frozen_train_bank,
)
from .a22_constrained_ppo import SafetyThresholds
from .a23_cr_ppo import model_parameter_sha256
from .a25_qualification import (
    A25_FORMAL_MINIBATCH_SIZE,
    A25_QUALIFICATION_VERSION,
    QualificationBindings,
    canonical_sha256,
    common_state_update_panel,
    generator_state_sha256,
)
from .a25_shield_dependence import shield_aware_rollout
from .long_horizon_env import ACTION_COUNT, MultiTownLongHorizonEnv, RLAction
from .ppo_controller import ActorCritic, PPOConfig
from .pq1_numerical_conformance import optimizer_state_sha256

A25_Q0_RUNNER_VERSION = "multitown-a25-common-state-q0-runner-v2"
A25_Q0_RECEIPT_VERSION = "multitown-a25-common-state-q0-receipt-v2"
A25_Q0_TRAINING_SEED = 20260815
A25_Q0_ROLLOUT_SEED = 20260816
A25_Q0_TENSOR_SEED = 20260817
A25_Q0_GLOBAL_RNG_GUARD_SEED = 20260818
A25_Q0_BETA = 5.0
A25_Q0_OUTER_FOLD = 0
A25_Q0_EPISODE_IDS = (
    "a9-lh-train-40000051",
    "a9-lh-train-40000052",
    "a9-lh-train-40000193",
    "a9-lh-train-40000196",
)
A25_Q0_SOURCE_PATHS = (
    "multitown/__init__.py",
    "multitown/a25_shield_dependence.py",
    "multitown/a25_qualification.py",
    "multitown/a25_q0_runner.py",
    "multitown/a9_oof_protocol.py",
    "multitown/a9_ppo_oof.py",
    "multitown/a22_constrained_ppo.py",
    "multitown/a23_cr_ppo.py",
    "multitown/long_horizon_env.py",
    "multitown/a9_long_horizon_env.py",
    "multitown/ppo_controller.py",
    "multitown/pq1_numerical_conformance.py",
    "docs/A25_COMMON_STATE_Q0_PROTOCOL.md",
    "docs/A25_Q0_STATUS_AND_NEXT_PLAN_20260815_ZH.md",
    "tests/test_a25_shield_dependence.py",
    "tests/test_a25_qualification.py",
    "tests/test_a25_q0_runner.py",
    "pyproject.toml",
)

_A25_Q0_FORBIDDEN_ENVIRONMENT = (
    "GIT_DIR",
    "GIT_INDEX_FILE",
    "GIT_WORK_TREE",
    "PYTHONHOME",
    "PYTHONPATH",
)
_A25_Q0_RECORDED_ENVIRONMENT = (
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDA_VISIBLE_DEVICES",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "PYTHONHASHSEED",
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _global_rng_state_sha256() -> dict[str, str]:
    numpy_state = np.random.get_state()
    numpy_values = np.ascontiguousarray(numpy_state[1])
    numpy_digest = hashlib.sha256()
    for payload in (
        str(numpy_state[0]).encode("ascii"),
        numpy_values.dtype.str.encode("ascii"),
        _canonical_json_bytes(list(numpy_values.shape)),
        numpy_values.tobytes(order="C"),
        _canonical_json_bytes(
            [int(numpy_state[2]), int(numpy_state[3]), float(numpy_state[4])]
        ),
    ):
        numpy_digest.update(payload)
        numpy_digest.update(b"\0")
    torch_state = torch.random.get_rng_state().detach().cpu().contiguous().numpy()
    torch_digest = hashlib.sha256()
    for payload in (
        torch_state.dtype.str.encode("ascii"),
        _canonical_json_bytes(list(torch_state.shape)),
        torch_state.tobytes(order="C"),
    ):
        torch_digest.update(payload)
        torch_digest.update(b"\0")
    return {
        "python_sha256": _sha256_bytes(
            _canonical_json_bytes(random.getstate())
        ),
        "numpy_sha256": numpy_digest.hexdigest(),
        "torch_cpu_sha256": torch_digest.hexdigest(),
    }


def _shared_panel_trace_identity(
    panels: dict[str, dict[str, Any]],
) -> dict[str, str]:
    if set(panels) != {"reward", "unsafe", "wrong"}:
        raise RuntimeError("A25 Q0 common trace requires all three panels")
    identities: list[dict[str, str]] = []
    for mode in ("reward", "unsafe", "wrong"):
        identity = panels[mode].get("identity")
        if not isinstance(identity, dict):
            raise TypeError("A25 Q0 panel trace identity is missing")
        trace: dict[str, str] = {}
        for key in (
            "replay_sha256",
            "replay_policy_rng_before_sha256",
            "replay_policy_rng_after_sha256",
        ):
            value = identity.get(key)
            if type(value) is not str or len(value) != 64:
                raise RuntimeError("A25 Q0 panel trace identity is invalid")
            trace[key] = value
        identities.append(trace)
    first = identities[0]
    if any(
        _canonical_json_bytes(identity) != _canonical_json_bytes(first)
        for identity in identities[1:]
    ):
        raise RuntimeError("A25 Q0 panels do not share one common trace")
    return first


def _git_output(root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if completed.returncode:
        raise RuntimeError(f"A25 Q0 git command failed: {' '.join(arguments)}")
    return completed.stdout


def _validate_runtime_environment() -> None:
    present = [key for key in _A25_Q0_FORBIDDEN_ENVIRONMENT if key in os.environ]
    if present:
        raise RuntimeError(
            "A25 Q0 refuses source/import override environment: "
            + ",".join(present)
        )


def _loaded_multitown_module_origins(root: Path) -> dict[str, dict[str, str]]:
    repository = root.resolve(strict=True)
    origins: dict[str, dict[str, str]] = {}
    for name, module in sorted(sys.modules.items()):
        logical_name = name
        if name == "__main__":
            spec = getattr(module, "__spec__", None)
            spec_name = getattr(spec, "name", None)
            if isinstance(spec_name, str):
                logical_name = spec_name
        if logical_name != "multitown" and not logical_name.startswith(
            "multitown."
        ):
            continue
        raw_origin = getattr(module, "__file__", None)
        if not isinstance(raw_origin, str) or not raw_origin:
            raise RuntimeError(
                f"A25 Q0 loaded module has no physical origin: {logical_name}"
            )
        raw_path = Path(raw_origin)
        try:
            origin = raw_path.resolve(strict=True)
        except OSError as exc:
            raise RuntimeError(
                f"A25 Q0 loaded module origin cannot be resolved: {logical_name}"
            ) from exc
        try:
            relative = origin.relative_to(repository).as_posix()
        except ValueError as exc:
            raise RuntimeError(
                f"A25 Q0 loaded module is outside the bound root: {logical_name}"
            ) from exc
        if raw_path.is_symlink() or origin.suffix != ".py" or not origin.is_file():
            raise RuntimeError(
                f"A25 Q0 loaded module origin is unsafe: {logical_name}"
            )
        head_payload = _git_output(repository, "show", f"HEAD:{relative}")
        payload = origin.read_bytes()
        if payload != head_payload:
            raise RuntimeError(
                f"A25 Q0 loaded module differs from HEAD: {logical_name}"
            )
        candidate = {
            "path": relative,
            "sha256": _sha256_bytes(payload),
        }
        previous = origins.get(logical_name)
        if previous is not None and previous != candidate:
            raise RuntimeError(
                f"A25 Q0 loaded module has conflicting origins: {logical_name}"
            )
        origins[logical_name] = candidate
    if "multitown.a25_q0_runner" not in origins:
        raise RuntimeError("A25 Q0 runner module origin is missing")
    return origins


def _external_module_identity(module: Any, *, label: str) -> dict[str, str]:
    raw_origin = getattr(module, "__file__", None)
    if not isinstance(raw_origin, str) or not raw_origin:
        raise RuntimeError(f"A25 Q0 {label} module has no physical origin")
    origin = Path(raw_origin).resolve(strict=True)
    if not origin.is_file():
        raise RuntimeError(f"A25 Q0 {label} module origin is not a file")
    return {
        "origin": str(origin),
        "origin_sha256": _sha256_bytes(origin.read_bytes()),
        "version": str(module.__version__),
    }


def _runtime_profile(root: Path) -> dict[str, Any]:
    _validate_runtime_environment()
    repository = root.resolve(strict=True)
    executable = Path(sys.executable).resolve(strict=True)
    if not executable.is_file():
        raise RuntimeError("A25 Q0 Python executable is not a file")
    sys_path = [
        str(Path(entry or os.getcwd()).resolve(strict=False)) for entry in sys.path
    ]
    if str(repository) not in sys_path:
        raise RuntimeError("A25 Q0 bound root is absent from sys.path")
    libc_name, libc_version = platform.libc_ver()
    return {
        "schema_version": "multitown-a25-q0-runtime-profile-v1",
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "compiler": platform.python_compiler(),
            "reported_executable": sys.executable,
            "resolved_executable": str(executable),
            "executable_sha256": _sha256_bytes(executable.read_bytes()),
            "sys_path": sys_path,
        },
        "numpy": _external_module_identity(np, label="NumPy"),
        "torch": {
            **_external_module_identity(torch, label="Torch"),
            "git_version": str(torch.version.git_version),
            "cuda_build": torch.version.cuda,
            "default_dtype": str(torch.get_default_dtype()),
            "default_device": str(torch.get_default_device()),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "libc": {"name": libc_name, "version": libc_version},
            "cpu_count": os.cpu_count(),
            "byteorder": sys.byteorder,
        },
        "device": {
            "execution": "cpu",
            "rollout_generator": "cpu",
            "update_generator": "cpu",
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
        },
        "threads": {
            "torch_intraop": int(torch.get_num_threads()),
            "torch_interop": int(torch.get_num_interop_threads()),
        },
        "determinism": {
            "algorithms_enabled": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
            "algorithms_warn_only": bool(
                torch.is_deterministic_algorithms_warn_only_enabled()
            ),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "cuda_matmul_allow_tf32": bool(
                torch.backends.cuda.matmul.allow_tf32
            ),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "mkldnn_enabled": bool(torch.backends.mkldnn.enabled),
        },
        "environment": {
            key: os.environ.get(key) for key in _A25_Q0_RECORDED_ENVIRONMENT
        },
    }


def _source_state(root: Path) -> dict[str, Any]:
    _validate_runtime_environment()
    if _git_output(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("A25 Q0 requires an exact clean source revision")
    revision = _git_output(root, "rev-parse", "HEAD").decode("ascii").strip()
    tree = _git_output(root, "rev-parse", "HEAD^{tree}").decode("ascii").strip()
    source_sha256: dict[str, str] = {}
    for relative in A25_Q0_SOURCE_PATHS:
        path = root / relative
        head_payload = _git_output(root, "show", f"HEAD:{relative}")
        if (
            not path.is_file()
            or path.is_symlink()
            or path.read_bytes() != head_payload
        ):
            raise RuntimeError(f"A25 Q0 source differs from HEAD: {relative}")
        source_sha256[relative] = _sha256_bytes(head_payload)
    return {
        "revision": revision,
        "tree": tree,
        "source_sha256": dict(sorted(source_sha256.items())),
        "module_origins": _loaded_multitown_module_origins(root),
        "runtime": _runtime_profile(root),
    }


def _fixture_model() -> ActorCritic:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(A25_Q0_TRAINING_SEED)
        model = ActorCritic(
            MultiTownLongHorizonEnv.observation_size, 8, ACTION_COUNT
        ).cpu()
        with torch.no_grad():
            model.actor.weight.zero_()
            model.actor.bias.fill_(0.0)
            model.actor.bias[RLAction.OBSERVE] = 8.0
            model.actor.bias[RLAction.REVIEW] = 6.0
            model.actor.bias[RLAction.EXECUTE] = 4.0
            model.actor.bias[RLAction.HUMAN] = 2.0
            model.actor.bias[RLAction.STOP] = -8.0
    return model


def _config() -> PPOConfig:
    return PPOConfig(
        updates=1,
        episodes_per_update=len(A25_Q0_EPISODE_IDS),
        hidden_size=8,
        learning_rate=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        ppo_epochs=1,
        minibatch_size=A25_FORMAL_MINIBATCH_SIZE,
        value_coef=0.0,
        entropy_coef=0.0,
        max_grad_norm=100.0,
        dev_interval=0,
    )


def _run_contract(
    *,
    source: dict[str, Any],
    train_bank_sha256: str,
    fold_manifest_sha: str,
    episode_sha256: dict[str, str],
    environment_source_sha256: str,
    mean_incidents: float,
) -> dict[str, Any]:
    return {
        "schema_version": "multitown-a25-common-state-q0-contract-v2",
        "runner_version": A25_Q0_RUNNER_VERSION,
        "qualification_version": A25_QUALIFICATION_VERSION,
        "scope": "development-only-common-state-q0-no-outer",
        "source": source,
        "train_bank_sha256": train_bank_sha256,
        "fold_manifest_sha256": fold_manifest_sha,
        "environment_source_sha256": environment_source_sha256,
        "outer_fold": A25_Q0_OUTER_FOLD,
        "episode_ids": list(A25_Q0_EPISODE_IDS),
        "episode_sha256": dict(sorted(episode_sha256.items())),
        "training_seed": A25_Q0_TRAINING_SEED,
        "rollout_seed": A25_Q0_ROLLOUT_SEED,
        "tensor_seed": A25_Q0_TENSOR_SEED,
        "global_rng_guard_seed": A25_Q0_GLOBAL_RNG_GUARD_SEED,
        "intervention_beta": A25_Q0_BETA,
        "mean_incidents": mean_incidents,
        "config": asdict(_config()),
        "whole_rollout_invariant": {
            "minibatch_size": A25_FORMAL_MINIBATCH_SIZE,
            "qualification_episode_count": len(A25_Q0_EPISODE_IDS),
            "max_steps_per_episode": 50,
            "formal_episode_count": 48,
            "formal_max_decisions": 2400,
        },
        "cr_mode_profiles": {
            "reward": {"unsafe": 1.0, "wrong_per_incident": 1.0},
            "unsafe": {"unsafe": 0.01, "wrong_per_incident": 1.0},
            "wrong": {"unsafe": 1.0, "wrong_per_incident": 0.001},
        },
        "outer_rows_read": 0,
        "formal_lock_created": False,
    }


def recompute_receipt(root: Path) -> dict[str, Any]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    try:
        random.seed(A25_Q0_GLOBAL_RNG_GUARD_SEED)
        np.random.seed(A25_Q0_GLOBAL_RNG_GUARD_SEED)
        torch.random.default_generator.manual_seed(A25_Q0_GLOBAL_RNG_GUARD_SEED)
        global_rng_before = _global_rng_state_sha256()
        return _recompute_receipt_guarded(root, global_rng_before)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)


def _recompute_receipt_guarded(
    root: Path, global_rng_before: dict[str, str]
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    source = _source_state(root)
    if (root / "artifacts/a24-cr-ppo-no-shield-attempt-v1.lock").exists():
        raise RuntimeError("A25 Q0 refuses an active A24 formal lock")
    bank = load_frozen_train_bank(FROZEN_TRAIN_PATH)
    if bank.payload_sha256 != EXPECTED_TRAIN_SHA256:
        raise RuntimeError("A25 Q0 frozen train bank identity changed")
    assignments = assign_stratified_group_folds(bank)
    assignment_index = {row.episode_id: row for row in assignments}
    bank_index = {episode.episode_id: episode for episode in bank.episodes}
    if (
        any(episode_id not in bank_index for episode_id in A25_Q0_EPISODE_IDS)
        or any(
            assignment_index[episode_id].fold != A25_Q0_OUTER_FOLD
            for episode_id in A25_Q0_EPISODE_IDS
        )
    ):
        raise RuntimeError("A25 Q0 fixed episode/fold binding changed")
    episodes = [bank_index[episode_id] for episode_id in A25_Q0_EPISODE_IDS]
    episode_sha256 = {
        episode_id: bank.episode_sha256[episode_id]
        for episode_id in A25_Q0_EPISODE_IDS
    }
    mean_incidents = sum(len(item.incidents) for item in episodes) / len(episodes)
    environment_source_sha256 = canonical_sha256(
        {
            relative: source["source_sha256"][relative]
            for relative in (
                "multitown/long_horizon_env.py",
                "multitown/a9_long_horizon_env.py",
            )
        }
    )
    contract = _run_contract(
        source=source,
        train_bank_sha256=bank.payload_sha256,
        fold_manifest_sha=fold_manifest_sha256(assignments),
        episode_sha256=episode_sha256,
        environment_source_sha256=environment_source_sha256,
        mean_incidents=mean_incidents,
    )
    run_id = canonical_sha256(contract)
    bindings = QualificationBindings(
        run_id=run_id,
        source_revision=source["revision"],
        source_tree=source["tree"],
        train_bank_sha256=bank.payload_sha256,
        environment_source_sha256=environment_source_sha256,
        fold_manifest_sha256=contract["fold_manifest_sha256"],
        episode_sha256=episode_sha256,
        source_sha256=source["source_sha256"],
        outer_fold=A25_Q0_OUTER_FOLD,
        training_seed=A25_Q0_TRAINING_SEED,
        rollout_seed=A25_Q0_ROLLOUT_SEED,
    )
    model = _fixture_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-5)
    rollout_generator = torch.Generator(device="cpu").manual_seed(
        A25_Q0_ROLLOUT_SEED
    )
    pre_rollout_rng = generator_state_sha256(rollout_generator)
    transition_episodes = [
        shield_aware_rollout(
            model,
            episode,
            torch.device("cpu"),
            mean_incidents=mean_incidents,
            generator=rollout_generator,
        )[0]
        for episode in episodes
    ]
    post_rollout_rng = generator_state_sha256(rollout_generator)
    pre_panel_model = model_parameter_sha256(model)
    pre_panel_optimizer = optimizer_state_sha256(optimizer, model)
    profiles = contract["cr_mode_profiles"]
    panels = {
        mode: common_state_update_panel(
            model,
            optimizer,
            episodes,
            transition_episodes,
            _config(),
            beta=A25_Q0_BETA,
            cr_thresholds=SafetyThresholds(
                unsafe=float(profile["unsafe"]),
                wrong_per_incident=float(profile["wrong_per_incident"]),
                mean_incidents=mean_incidents,
            ),
            expected_cr_mode=mode,  # type: ignore[arg-type]
            tensor_seed=A25_Q0_TENSOR_SEED,
            bindings=bindings,
        )
        for mode, profile in profiles.items()
    }
    if (
        model_parameter_sha256(model) != pre_panel_model
        or optimizer_state_sha256(optimizer, model) != pre_panel_optimizer
    ):
        raise RuntimeError("A25 Q0 panel mutated common source state")
    if _canonical_json_bytes(_source_state(root)) != _canonical_json_bytes(source):
        raise RuntimeError("A25 Q0 source or runtime changed during recomputation")
    common_trace = _shared_panel_trace_identity(panels)
    global_rng_after = _global_rng_state_sha256()
    core = {
        "schema_version": A25_Q0_RECEIPT_VERSION,
        "run_id": run_id,
        "contract": contract,
        "common_state": {
            "model_sha256": pre_panel_model,
            "optimizer_sha256": pre_panel_optimizer,
            "pre_rollout_tensor_rng_sha256": pre_rollout_rng,
            "post_rollout_tensor_rng_sha256": post_rollout_rng,
            "global_rng_before": global_rng_before,
            "global_rng_after": global_rng_after,
            "common_trace": common_trace,
        },
        "panels": panels,
        "gates": {
            "all_three_cr_modes_covered": set(panels) == {"reward", "unsafe", "wrong"},
            "all_common_state_panels_passed": all(
                panel["passed"] for panel in panels.values()
            ),
            "zero_outer_rows_read": contract["outer_rows_read"] == 0,
            "no_formal_lock_created": contract["formal_lock_created"] is False,
            "frozen_bank_bound": contract["train_bank_sha256"]
            == EXPECTED_TRAIN_SHA256,
            "all_panels_share_common_trace": True,
            "global_rng_preserved": global_rng_before == global_rng_after,
        },
        "claim_boundary": {
            "q0_primitive_qualified": True,
            "q1_mechanism_qualified": False,
            "formal_authorized": False,
            "performance_claim_supported": False,
            "sailr_reproduction": False,
            "crpo_reproduction": False,
        },
    }
    if not all(core["gates"].values()):
        raise RuntimeError("A25 Q0 ordered qualification gate failed")
    receipt_id = canonical_sha256(core)
    return {**core, "receipt_id": receipt_id, "status": "PASSED"}


def _read_strict_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate A25 Q0 receipt key")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite A25 Q0 JSON constant: {value}")
        ),
    )
    if type(value) is not dict:
        raise ValueError("A25 Q0 receipt must be an object")
    return value


def verify_receipt(root: Path, receipt_path: Path) -> dict[str, Any]:
    path = receipt_path.resolve(strict=True)
    metadata = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise ValueError("A25 Q0 receipt must be a regular non-symlink file")
    recorded = _read_strict_json(path)
    recorded_core = {
        key: value
        for key, value in recorded.items()
        if key not in {"receipt_id", "status"}
    }
    if (
        type(recorded.get("receipt_id")) is not str
        or canonical_sha256(recorded_core) != recorded["receipt_id"]
    ):
        raise ValueError("A25 Q0 receipt self-identity mismatch")
    expected = recompute_receipt(root)
    if _canonical_json_bytes(recorded) != _canonical_json_bytes(expected):
        raise ValueError("A25 Q0 receipt differs from deterministic recomputation")
    return expected


def _write_private_json(path: Path, value: dict[str, Any], *, root: Path) -> None:
    target = path.resolve(strict=False)
    repository = root.resolve(strict=True)
    if target.is_relative_to(repository):
        raise ValueError("A25 Q0 receipt output must be outside the repository")
    parent = target.parent.resolve(strict=True)
    parent_metadata = parent.stat()
    if (
        not stat.S_ISDIR(parent_metadata.st_mode)
        or parent_metadata.st_uid != os.geteuid()
        or parent_metadata.st_mode & 0o022
        or target.exists()
        or target.is_symlink()
    ):
        raise ValueError("unsafe A25 Q0 receipt output path")
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ).encode("utf-8") + b"\n"
    descriptor = os.open(
        target,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--verify", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.output is not None:
            receipt = recompute_receipt(args.root)
            _write_private_json(args.output, receipt, root=args.root)
        else:
            receipt = verify_receipt(args.root, args.verify)
    except (OSError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        print(
            json.dumps(
                {"status": "ERROR", "error_type": type(exc).__name__, "error": str(exc)},
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
                "run_id": receipt["run_id"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
