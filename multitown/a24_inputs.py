"""Frozen input verification and A22 comparator ledger for A24."""

from __future__ import annotations

import json
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .a22_constrained_ppo import MECHANISMS as A22_MECHANISMS
from .a22_runner import (
    _formal_config as _a22_formal_config,
    _validate_checkpoint as _validate_a22_checkpoint,
    partition_ids,
    schedule as a22_schedule,
)
from .a23_cr_ppo import model_parameter_sha256
from .a23_runner import (
    A22_RAW_PATH,
    EXPECTED_A22,
    EXPECTED_A22_RUN_DIGEST,
    EXPECTED_BINDINGS,
    EXPECTED_R2,
    _preflight as _a23_preflight,
)
from .a24_artifact_state import sha256_file
from .a24_contract import FORMAL_FOLDS, FORMAL_SEEDS
from .a9_ppo_oof import _digest, _set_seed
from .long_horizon_env import ACTION_COUNT, MultiTownLongHorizonEnv
from .ppo_controller import ActorCritic
from .pq1_numerical_conformance import optimizer_state_sha256
from .pq1_runner import (
    _verify_a23_failure,
    _verify_qualification_lock,
    validate_manifest as validate_pq1_manifest,
)


PQ1_ROOT = Path("artifacts/pq1-on-policy-qualification-formal-20260814")
PQ1_LOCK = Path("artifacts/pq1-on-policy-qualification-attempt-v1.lock")
EXPECTED_PQ1 = {
    "lock": "ea7ff8ed65d0a404dfffa933a4e651b2623d6ea3e78ee4aa712c04d94889182e",
    "result.json": "2c608d58089bf11397656499c61b5bcba242997b3f2ae28ca942554c222814af",
    "run-contract.json": "29f8f6aa7661dc215fbc62a056994b1db40231fbe664210040a70a2c8ebb96ad",
    "protocol.json": "081f71c4ef21a2e4303c0470fa0597390216b7a108e325f761f93dd5d62df256",
    "artifact-manifest.json": (
        "96096ed69378c541aa20db330a778184a8ed4bdac7d03fe1dfcaeb7be10f70b6"
    ),
}
EXPECTED_PQ1_CANONICAL = {
    "run_contract": "ad433856d374284e46d2d5389f0ad41e7a4deddab11af6d5da54934dcc9adb70",
    "protocol": "3c41699481e8dbbc2fa22d09a5ea4b9d5676b06152b1875d8f75bc4fda8cf86e",
    "source_set": "4f72bf649f2af6f8a787e6baa8608efa17197110e40d242f8fe7c4793759df86",
}
A22_MECHANISM_INDEX = {item.name: item for item in A22_MECHANISMS}

A24_SOURCE_PATHS = (
    "multitown/a24_contract.py",
    "multitown/a24_artifact_state.py",
    "multitown/a24_inputs.py",
    "multitown/a24_statistics.py",
    "multitown/a24_runner.py",
    "multitown/a24_monitor.py",
    "multitown/a24_orphan_finalizer.py",
    "multitown/pq1_numerical_conformance.py",
    "multitown/pq1_runner.py",
    "multitown/a23_cr_ppo.py",
    "multitown/a23_runner.py",
    "multitown/a23_statistics.py",
    "multitown/a22_constrained_ppo.py",
    "multitown/a22_runner.py",
    "multitown/a22_report.py",
    "multitown/a9_oof_protocol.py",
    "multitown/a9_ppo_oof.py",
    "multitown/a9_safety_development.py",
    "multitown/long_horizon_env.py",
    "multitown/ppo_controller.py",
    "tests/test_a24_artifact_state.py",
    "tests/test_a24_statistics.py",
    "tests/test_a24_runner.py",
    "tests/test_a24_monitor.py",
    "docs/A24_PQ_QUALIFIED_NO_SHIELD_CR_PPO.md",
    "docs/A23_FORMAL_INVALIDATION_20260814.md",
    "docs/PQ1_ON_POLICY_NUMERICAL_CONFORMANCE.md",
    "pyproject.toml",
)


def source_state(root: Path, *, require_clean: bool) -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if require_clean and status:
        raise RuntimeError("formal A24 requires a clean source checkout")
    files: dict[str, str] = {}
    for relative in A24_SOURCE_PATHS:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"missing or invalid A24 source: {relative}")
        raw = path.read_bytes()
        if require_clean:
            committed = subprocess.run(
                ["git", "show", f"HEAD:{relative}"],
                cwd=root,
                capture_output=True,
                check=True,
            ).stdout
            if raw != committed:
                raise RuntimeError(f"executed A24 source differs from HEAD: {relative}")
        files[relative] = sha256_file(path)
    return {
        "revision": revision,
        "dirty": bool(status),
        "files": files,
        "source_set_sha256": _digest(files),
    }


def runtime_fingerprint(*, threads: int) -> dict[str, Any]:
    if type(threads) is not int or threads <= 0:
        raise ValueError("A24 requested thread count is invalid")
    actual_threads = torch.get_num_threads()
    if actual_threads != threads:
        raise RuntimeError(
            "A24 requested torch threads differ from the active runtime"
        )
    return {
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "requested_threads": threads,
        "torch_num_threads": actual_threads,
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "torch_deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "torch_deterministic_warn_only": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
    }


def verify_pq1(root: Path) -> dict[str, Any]:
    output = (root / PQ1_ROOT).resolve()
    lock = (root / PQ1_LOCK).resolve()
    if sha256_file(lock) != EXPECTED_PQ1["lock"]:
        raise RuntimeError("A24 pinned PQ-1 lock changed")
    for name, digest in EXPECTED_PQ1.items():
        if name == "lock":
            continue
        if sha256_file(output / name) != digest:
            raise RuntimeError(f"A24 pinned PQ-1 artifact changed: {name}")
    validate_pq1_manifest(output)
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    contract = json.loads((output / "run-contract.json").read_text(encoding="utf-8"))
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    if (
        _digest(contract) != EXPECTED_PQ1_CANONICAL["run_contract"]
        or _digest(protocol) != EXPECTED_PQ1_CANONICAL["protocol"]
        or result.get("qualification_passed") is not True
        or result.get("deterministic_replica_match") is not True
        or result.get("performance_evaluable") is not False
        or result.get("permits_a23_retry") is not False
        or result.get("permits_a24_design") is not True
        or result.get("products", {}).get("replicas") != 2
        or result.get("products", {}).get("updates") != 240
        or result.get("products", {}).get("training_episode_draws") != 11_520
        or result.get("products", {}).get("calibration_rows") != 0
        or result.get("products", {}).get("outer_rows") != 0
    ):
        raise RuntimeError("A24 pinned PQ-1 semantic result changed")
    binding = result.get("qualification_lock")
    if type(binding) is not dict:
        raise RuntimeError("A24 pinned PQ-1 lock binding is missing")
    _verify_qualification_lock(lock, binding)
    if binding["descriptor"].get("pq1_source_set_sha256") != EXPECTED_PQ1_CANONICAL[
        "source_set"
    ]:
        raise RuntimeError("A24 pinned PQ-1 source-set binding changed")
    return {
        "raw_sha256": EXPECTED_PQ1,
        "canonical_sha256": EXPECTED_PQ1_CANONICAL,
        "result_sha256": EXPECTED_PQ1["result.json"],
        "source_revision": result["source_revision"],
        "qualification_lock": binding,
    }


def _derived_initialization(seed: int) -> dict[str, str]:
    config = _a22_formal_config(a22_schedule(False))
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        _set_seed(seed)
        model = ActorCritic(
            MultiTownLongHorizonEnv.observation_size,
            config.hidden_size,
            ACTION_COUNT,
        ).cpu()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            eps=1e-5,
        )
        return {
            "algorithm": "pinned-a22-source-runtime-seed-derived-expectation-v1",
            "model_sha256": model_parameter_sha256(model),
            "named_optimizer_sha256": optimizer_state_sha256(optimizer, model),
            "historical_initial_tensor_receipt": False,
        }
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def build_comparator_ledger(
    context: Mapping[str, Any],
    *,
    folds: Sequence[int] = FORMAL_FOLDS,
    seeds: Sequence[int] = FORMAL_SEEDS,
) -> dict[str, Any]:
    a22_root = Path(context["a22_root"])
    all_fits = json.loads(
        (a22_root / "all-fits-complete.json").read_text(encoding="utf-8")
    )
    raw_fits = all_fits.get("fits")
    if type(raw_fits) is not list or len(raw_fits) != 60:
        raise RuntimeError("A24 A22 all-fits product changed")
    fit_index = {
        (int(row["outer_fold"]), int(row["training_seed"]), str(row["mechanism"])): row
        for row in raw_fits
    }
    expected_keys = {(fold, seed, "lagrangian") for fold in folds for seed in seeds}
    if len(fit_index) != 60 or not expected_keys <= set(fit_index):
        raise RuntimeError("A24 A22 Lagrangian fit keys are incomplete")
    calibration_rows = [
        json.loads(line)
        for line in (a22_root / "calibration-decisions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    partitions = {
        fold: partition_ids(context["assignments"], fold) for fold in folds
    }
    mechanism = A22_MECHANISM_INDEX["lagrangian"]
    formal_a22_schedule = a22_schedule(False)
    entries = []
    for fold, seed, name in sorted(expected_keys):
        fit = fit_index[(fold, seed, name)]
        prefix = (
            a22_root
            / "fits"
            / f"outer-fold-{fold}"
            / f"seed-{seed}"
            / name
        )
        checkpoint = prefix / "final.pt"
        fit_complete = prefix / "fit-complete.json"
        training_log = prefix / "training-metrics.jsonl"
        if any(not path.is_file() or path.is_symlink() for path in (checkpoint, fit_complete, training_log)):
            raise RuntimeError("A24 A22 comparator artifact is missing or symlinked")
        persisted_fit = json.loads(fit_complete.read_text(encoding="utf-8"))
        if persisted_fit != fit:
            raise RuntimeError("A24 A22 comparator fit-complete changed")
        logs = [json.loads(line) for line in training_log.read_text(encoding="utf-8").splitlines()]
        sampled_ids = [
            str(episode_id)
            for row in logs
            for episode_id in row["sampled_episode_ids"]
        ]
        expected_samples = list(context["a22_sample_sequences"][(fold, seed)])
        if (
            len(logs) != 120
            or sampled_ids != expected_samples
            or len(sampled_ids) != 5760
            or _digest(sampled_ids) != fit["sample_sequence_sha256"]
            or sha256_file(checkpoint) != fit["checkpoint_sha256"]
        ):
            raise RuntimeError("A24 A22 comparator training ledger changed")
        _validate_a22_checkpoint(
            checkpoint,
            fit=fit,
            mechanism=mechanism,
            outer_fold=fold,
            training_seed=seed,
            run_schedule=formal_a22_schedule,
            run_contract_sha256=EXPECTED_A22_RUN_DIGEST,
        )
        expected_ids = set(partitions[fold]["inner_calibration_ids"])
        subset = [
            row
            for row in calibration_rows
            if int(row["design_outer_fold"]) == fold
            and int(row["training_seed"]) == seed
            and row["mechanism"] == "lagrangian"
        ]
        subset.sort(key=lambda row: str(row["episode_id"]))
        if (
            len(subset) != 600
            or {str(row["episode_id"]) for row in subset} != expected_ids
            or any(
                row["final_checkpoint_sha256"] != fit["checkpoint_sha256"]
                or row["run_contract_sha256"] != EXPECTED_A22_RUN_DIGEST
                for row in subset
            )
        ):
            raise RuntimeError("A24 A22 comparator calibration subset changed")
        entries.append(
            {
                "outer_fold": fold,
                "training_seed": seed,
                "mechanism": name,
                "paths": {
                    "final.pt": str(checkpoint.relative_to(a22_root)),
                    "fit-complete.json": str(fit_complete.relative_to(a22_root)),
                    "training-metrics.jsonl": str(training_log.relative_to(a22_root)),
                },
                "raw_sha256": {
                    "final.pt": sha256_file(checkpoint),
                    "fit-complete.json": sha256_file(fit_complete),
                    "training-metrics.jsonl": sha256_file(training_log),
                },
                "sample_sequence_sha256": fit["sample_sequence_sha256"],
                "calibration_subset_rows": len(subset),
                "calibration_subset_sha256": _digest(subset),
                "derived_expected_initialization": _derived_initialization(seed),
            }
        )
    if len(entries) != 15:
        raise RuntimeError("A24 comparator ledger is not the exact 15-cell product")
    return {
        "schema_version": "multitown-a24-a22-lagrangian-comparator-ledger-v1",
        "source_artifacts": EXPECTED_A22,
        "source_run_contract_sha256": EXPECTED_A22_RUN_DIGEST,
        "source_calibration_rows": 36_000,
        "selected_calibration_rows": 9_000,
        "entries": entries,
        "ledger_sha256": _digest(entries),
        "historical_initial_tensor_receipt_available": False,
    }


def verify_inputs(
    root: Path,
    *,
    smoke: bool,
    threads: int,
) -> dict[str, Any]:
    source = source_state(root, require_clean=not smoke)
    a23_context = _a23_preflight(root, smoke=smoke)
    pq1 = verify_pq1(root)
    a23_failure = _verify_a23_failure(root)
    comparator = build_comparator_ledger(a23_context)
    runtime = runtime_fingerprint(threads=threads)
    bindings = a23_context["bindings"]
    if (
        bindings != EXPECTED_BINDINGS
        or sha256_file(root / A22_RAW_PATH / "artifact-manifest.json")
        != EXPECTED_A22["artifact-manifest.json"]
        or _digest(a23_context["a22_contract"]) != EXPECTED_A22_RUN_DIGEST
        or a23_context["a22_contract"]["frozen_r2_run_digest"]
        != EXPECTED_R2["run_digest"]
    ):
        raise RuntimeError("A24 inherited A22/A8 binding changed")
    signature_payload = {
        "source": source,
        "runtime": runtime,
        "pq1": pq1,
        "a23_failure": a23_failure,
        "a22": {
            "raw": EXPECTED_A22,
            "canonical_run_contract": EXPECTED_A22_RUN_DIGEST,
            "comparator_ledger_sha256": comparator["ledger_sha256"],
        },
        "a8": EXPECTED_R2,
        "bindings": bindings,
    }
    return {
        **a23_context,
        "a24_source": source,
        "a24_runtime": runtime,
        "pq1": pq1,
        "a23_failure": a23_failure,
        "a22_comparator_ledger": comparator,
        "a24_preflight_signature": _digest(signature_payload),
    }


def revalidate_inputs(
    root: Path,
    *,
    smoke: bool,
    threads: int,
    expected_signature: str,
) -> None:
    current = verify_inputs(root, smoke=smoke, threads=threads)
    if current["a24_preflight_signature"] != expected_signature:
        raise RuntimeError("A24 frozen input/source/runtime signature changed")
