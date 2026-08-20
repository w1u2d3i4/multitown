"""Run the train-only PQ-1 on-policy numerical-conformance qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import stat
import subprocess
import sys
import tempfile
import time
import traceback
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .a9_ppo_oof import FORMAL_SEEDS, _set_seed
from .a22_constrained_ppo import constrained_rollout, thresholds_from_inner_train
from .a22_runner import _expected_sample_sequence, partition_ids
from .a23_cr_ppo import mode_sequence_sha256, model_parameter_sha256
from .a23_runner import (
    _atomic_json,
    _atomic_jsonl,
    _digest,
    _formal_config,
    _isolate_valid_outputs,
    _json_payload,
    _preflight,
    _read_json,
    _sha256,
)
from .long_horizon_env import ACTION_COUNT, MultiTownLongHorizonEnv
from .ppo_controller import ActorCritic, _save_checkpoint
from .pq1_numerical_conformance import (
    FULL_BATCH_ATOL,
    FULL_BATCH_RTOL,
    MAX_PROBABILITY_RATIO_DRIFT,
    PQ1_PRIMITIVES_VERSION,
    optimizer_state_sha256,
    pq1_cr_ppo_update,
    transition_episode_sha256,
)

PQ1_RUNNER_VERSION = "multitown-pq1-numerical-conformance-runner-v1"
PQ1_POLICY_VERSION = "multitown-pq1-rowwise-cr-ppo-policy-v1"
PQ1_RESULT_VERSION = "multitown-pq1-numerical-conformance-result-v1"
PQ1_LOCK = Path("artifacts/pq1-on-policy-qualification-attempt-v1.lock")
PQ1_REPLICAS = ("replica-a", "replica-b")
PQ1_SOURCE_PATHS = (
    "multitown/pq1_numerical_conformance.py",
    "multitown/pq1_runner.py",
    "docs/PQ1_ON_POLICY_NUMERICAL_CONFORMANCE.md",
    "docs/A23_FORMAL_INVALIDATION_20260814.md",
    "tests/test_pq1_numerical_conformance.py",
    "tests/test_pq1_runner.py",
    "pyproject.toml",
)
A23_FAILURE_ANCHORS = {
    "lock": "7f6894a67693eb84df7ffd92afd11ed23ae7f213b14c18a8083522257da5027b",
    "RUNNING.json": "30fb13b58bab6ea12d58e15d1329cfe6f026d988b3b2208a7866669576de8458",
    "INVALIDATED.json": "658d5125290671e6d3d4455722551efc6a32406817fb0fefcc6967e39253f071",
    "run-contract.json": "4d7c5d2c833f53f091db59d4af3adba13f00ffeae6d02e2506a9fe602a245556",
    "training-contract.json": "43bd2e35547210b8a83009cb8d8a99684b1ae94c62d66d41d05549ec82c7e6bb",
    "progress.json": "5ad3d8a17819a473e91fd44ae71af79953b10a31c878b9059d5335f678d33f7e",
}
A23_FAILURE_FILE_METADATA = {
    "INVALIDATED.json": {
        "bytes": 1121,
        "sha256": A23_FAILURE_ANCHORS["INVALIDATED.json"],
    },
    "RUNNING.json": {
        "bytes": 157,
        "sha256": A23_FAILURE_ANCHORS["RUNNING.json"],
    },
    "fits/outer-fold-0/seed-20260812/cr-ppo/progress.json": {
        "bytes": 294,
        "sha256": A23_FAILURE_ANCHORS["progress.json"],
    },
    "run-contract.json": {
        "bytes": 3341,
        "sha256": A23_FAILURE_ANCHORS["run-contract.json"],
    },
    "training-contract.json": {
        "bytes": 484040,
        "sha256": A23_FAILURE_ANCHORS["training-contract.json"],
    },
}
A23_FAILURE_DIRECTORY_DIGEST = (
    "a2f1befc5ed838426267808797e58c395d0f7eea28e449427bfedf7f70833b3e"
)
A23_FAILURE_COMBINED_DIGEST = (
    "c5287a589aaeb51cfd2b238aac45e303b3546f8398f7910f7060eef8c5519ef3"
)
A23_FAILURE_RUN_CONTRACT_DIGEST = (
    "adb7b306b70f4296b950740f10ea02895bfcce9120af042badf54c6d9eb59cac"
)
A23_FAILURE_LOCK_METADATA = {
    "bytes": 254,
    "sha256": A23_FAILURE_ANCHORS["lock"],
}
A23_FAILURE_COMMIT = "c58c08af93cb73641822a00ca2be29d2c2a7c47c"
EXPECTED_LEGACY_BOUNDARY = {
    "update": 41,
    "transition_count": 125,
    "legacy_exceedances": 1,
    "max_abs": 1.0728836059570312e-6,
    "legacy_max_tolerance_ratio": 1.0026292874660714,
    "max_probability_ratio_drift": 1.0728836059570312e-6,
}
PQ1_REQUIRED_TEST_COUNT = 23
PQ1_REQUIRED_SUBTEST_COUNT = 20
PQ1_REQUIRED_JUNIT_CASE_COUNT = 43
PQ1_REQUIRED_JUNIT_TESTCASE_COUNT = 23


@dataclass(frozen=True)
class PQ1Schedule:
    mode: str
    replicas: tuple[str, ...]
    outer_fold: int
    seed: int
    mechanism: str
    updates: int
    episodes_per_update: int
    threads: int


def schedule(smoke: bool) -> PQ1Schedule:
    return PQ1Schedule(
        mode="smoke" if smoke else "qualification",
        replicas=PQ1_REPLICAS,
        outer_fold=0,
        seed=FORMAL_SEEDS[0],
        mechanism="cr-ppo",
        updates=2 if smoke else 120,
        episodes_per_update=4 if smoke else 48,
        threads=2 if smoke else 8,
    )


def _verify_a23_failure(root: Path) -> dict[str, Any]:
    output = root / "artifacts/a23-cr-ppo-formal-20260814"
    paths = {
        "lock": root / "artifacts/a23-cr-ppo-attempt-v1.lock",
        "RUNNING.json": output / "RUNNING.json",
        "INVALIDATED.json": output / "INVALIDATED.json",
        "run-contract.json": output / "run-contract.json",
        "training-contract.json": output / "training-contract.json",
        "progress.json": (
            output / "fits/outer-fold-0/seed-20260812/cr-ppo/progress.json"
        ),
    }
    actual_files = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    }
    if actual_files != set(A23_FAILURE_FILE_METADATA):
        raise RuntimeError("PQ-1 pinned A23 invalidation inventory changed")
    if any(
        not path.is_file()
        or path.is_symlink()
        or not stat.S_ISREG(path.lstat().st_mode)
        for path in paths.values()
    ) or any(
        _sha256(paths[name]) != digest for name, digest in A23_FAILURE_ANCHORS.items()
    ):
        raise RuntimeError("PQ-1 pinned A23 invalidation anchors changed")
    actual_metadata = {
        name: {
            "bytes": (output / name).stat().st_size,
            "sha256": _sha256(output / name),
        }
        for name in sorted(actual_files)
    }
    if (
        actual_metadata != A23_FAILURE_FILE_METADATA
        or _digest(actual_metadata) != A23_FAILURE_DIRECTORY_DIGEST
        or (paths["lock"].stat().st_mode & 0o777) != 0o600
    ):
        raise RuntimeError("PQ-1 pinned A23 invalidation ledger changed")
    combined = {
        f"a23-cr-ppo-formal-20260814/{name}": metadata
        for name, metadata in actual_metadata.items()
    }
    combined["a23-cr-ppo-attempt-v1.lock"] = {
        "bytes": paths["lock"].stat().st_size,
        "sha256": _sha256(paths["lock"]),
    }
    if (
        combined["a23-cr-ppo-attempt-v1.lock"] != A23_FAILURE_LOCK_METADATA
        or _digest(combined) != A23_FAILURE_COMBINED_DIGEST
    ):
        raise RuntimeError("PQ-1 pinned A23 lock/directory ledger changed")
    running = _read_json(paths["RUNNING.json"])
    invalidated = _read_json(paths["INVALIDATED.json"])
    progress = _read_json(paths["progress.json"])
    failed_contract = _read_json(paths["run-contract.json"])
    if (
        set(running)
        != {
            "schema_version",
            "started_at_utc",
            "outer_evaluation_started",
        }
        or running.get("schema_version") != "multitown-a23-cr-ppo-adaptive-runner-v1"
        or running.get("outer_evaluation_started") is not False
        or set(invalidated)
        != {
            "schema_version",
            "invalidated",
            "error_type",
            "error",
            "traceback",
            "selective_retry_forbidden",
            "formal_lock_acquired",
            "failed_at_utc",
        }
        or invalidated.get("invalidated") is not True
        or invalidated.get("formal_lock_acquired") is not True
        or invalidated.get("selective_retry_forbidden") is not True
        or invalidated.get("error_type") != "ValueError"
        or invalidated.get("error")
        != "A23 rollout does not bind to current on-policy model"
        or set(progress)
        != {
            "schema_version",
            "outer_fold",
            "training_seed",
            "mechanism",
            "current_update",
            "scheduled_updates",
            "reward_mode_count",
            "unsafe_mode_count",
            "wrong_mode_count",
            "outer_evaluation_started",
        }
        or progress.get("schema_version") != "multitown-a23-fit-progress-v1"
        or progress.get("outer_fold") != 0
        or progress.get("training_seed") != FORMAL_SEEDS[0]
        or progress.get("mechanism") != "cr-ppo"
        or progress.get("current_update") != 40
        or progress.get("scheduled_updates") != 120
        or (
            progress.get("reward_mode_count"),
            progress.get("unsafe_mode_count"),
            progress.get("wrong_mode_count"),
        )
        != (29, 9, 2)
        or progress.get("outer_evaluation_started") is not False
        or _digest(failed_contract) != A23_FAILURE_RUN_CONTRACT_DIGEST
        or failed_contract.get("source", {}).get("revision") != A23_FAILURE_COMMIT
    ):
        raise RuntimeError("PQ-1 A23 invalidation state is not fail-closed")
    return {
        "source_commit": A23_FAILURE_COMMIT,
        "anchors": A23_FAILURE_ANCHORS,
        "directory_digest": A23_FAILURE_DIRECTORY_DIGEST,
        "lock_and_directory_digest": A23_FAILURE_COMBINED_DIGEST,
        "run_contract_digest": A23_FAILURE_RUN_CONTRACT_DIGEST,
        "runtime": failed_contract["source"]["runtime"],
        "failed_before_optimizer_update": 41,
        "completed_optimizer_updates": 40,
        "performance_evaluable": False,
        "selective_retry_forbidden": True,
    }


def _runtime_fingerprint(run_schedule: PQ1Schedule) -> dict[str, Any]:
    """Capture the pinned A23 runtime plus qualification execution details."""

    base = {
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "platform": platform.platform(),
    }
    return {
        "a23_comparable": base,
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "requested_torch_threads": run_schedule.threads,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "torch_deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "torch_deterministic_warn_only": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "mkldnn_available": torch.backends.mkldnn.is_available(),
        "openmp_available": torch.backends.openmp.is_available(),
        "torch_parallel_info_sha256": hashlib.sha256(
            torch.__config__.parallel_info().encode("utf-8")
        ).hexdigest(),
        "torch_build_config_sha256": hashlib.sha256(
            torch.__config__.show().encode("utf-8")
        ).hexdigest(),
    }


def _source_state(root: Path, *, require_clean: bool) -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if require_clean and status:
        raise RuntimeError("formal PQ-1 requires a clean source checkout")
    hashes = {}
    for relative in PQ1_SOURCE_PATHS:
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"missing PQ-1 source: {relative}")
        if require_clean:
            committed = subprocess.run(
                ["git", "show", f"HEAD:{relative}"],
                cwd=root,
                capture_output=True,
                check=True,
            ).stdout
            if path.read_bytes() != committed:
                raise RuntimeError(
                    f"executed PQ-1 source differs from HEAD: {relative}"
                )
        hashes[relative] = _sha256(path)
    return {"revision": revision, "dirty": bool(status), "sha256": hashes}


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _strict_json_loads(payload: str, *, label: str) -> Any:
    try:
        return json.loads(
            payload,
            object_pairs_hook=_strict_json_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid strict PQ-1 JSON: {label}") from exc


def _strict_read_json(path: Path) -> dict[str, Any]:
    try:
        value = _strict_json_loads(path.read_text(encoding="utf-8"), label=str(path))
    except OSError as exc:
        raise RuntimeError(f"unreadable PQ-1 JSON: {path}") from exc
    if type(value) is not dict:
        raise RuntimeError(f"PQ-1 JSON is not an object: {path}")
    return value


def _strict_read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RuntimeError(f"unreadable PQ-1 JSONL: {path}") from exc
    if not lines or any(not line.strip() for line in lines):
        raise RuntimeError(f"PQ-1 JSONL has an empty row: {path}")
    rows = [
        _strict_json_loads(line, label=f"{path}:{index}")
        for index, line in enumerate(lines, start=1)
    ]
    if any(type(row) is not dict for row in rows):
        raise RuntimeError(f"PQ-1 JSONL row is not an object: {path}")
    return rows


def _all_finite(value: Any) -> bool:
    if torch.is_tensor(value):
        return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value, np.ndarray):
        return not np.issubdtype(value.dtype, np.floating) or bool(
            np.isfinite(value).all()
        )
    if isinstance(value, Mapping):
        return all(
            _all_finite(key) and _all_finite(item) for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    return type(value) is not float or math.isfinite(value)


def _same_typed_json(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if type(actual) is dict:
        return set(actual) == set(expected) and all(
            _same_typed_json(actual[key], expected[key]) for key in expected
        )
    if type(actual) is list:
        return len(actual) == len(expected) and all(
            _same_typed_json(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    return actual == expected and (type(actual) is not float or math.isfinite(actual))


def _has_exact_types(
    value: Any,
    *,
    strings: Sequence[str] = (),
    integers: Sequence[str] = (),
    floats: Sequence[str] = (),
    booleans: Sequence[str] = (),
) -> bool:
    if type(value) is not dict:
        return False
    return (
        all(type(value.get(key)) is str for key in strings)
        and all(type(value.get(key)) is int for key in integers)
        and all(
            type(value.get(key)) is float and math.isfinite(value[key])
            for key in floats
        )
        and all(type(value.get(key)) is bool for key in booleans)
    )


def _hex_digest(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _checkpoint_atomic(
    path: Path,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    config: Any,
    *,
    seed: int,
    update: int,
    metadata: Mapping[str, Any],
) -> None:
    partial = path.with_name(path.name + ".partial")
    if path.exists() or partial.exists():
        raise FileExistsError(path if path.exists() else partial)
    _save_checkpoint(
        partial,
        model,
        config,
        seed=seed,
        update=update,
        policy_version=PQ1_POLICY_VERSION,
    )
    payload = torch.load(partial, map_location="cpu", weights_only=False)
    payload["optimizer_state"] = optimizer.state_dict()
    payload.update(dict(metadata))
    if not _all_finite(payload):
        raise FloatingPointError("PQ-1 checkpoint has non-finite tensors")
    torch.save(payload, partial)
    verified = torch.load(partial, map_location="cpu", weights_only=False)
    if any(
        verified.get(key) != value for key, value in metadata.items()
    ) or not _all_finite(verified):
        raise RuntimeError("PQ-1 checkpoint metadata changed")
    os.replace(partial, path)


def _fit_replica(
    *,
    replica: str,
    output: Path,
    episodes: Sequence[Any],
    a8_rows: Sequence[Mapping[str, Any]],
    run_schedule: PQ1Schedule,
    run_contract_sha256: str,
    expected_sample_ids: Sequence[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = _formal_config(run_schedule)
    thresholds = thresholds_from_inner_train(a8_rows)
    _set_seed(run_schedule.seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
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
    initial_model = model_parameter_sha256(model)
    initial_optimizer = optimizer_state_sha256(optimizer, model)
    sample_rng = random.Random(run_schedule.seed)
    tensor_generator = torch.Generator(device="cpu").manual_seed(run_schedule.seed)
    mode_counts: Counter[str] = Counter()
    logs = []
    sampled_ids = []
    modes = []
    started = time.perf_counter()
    for update in range(1, run_schedule.updates + 1):
        pre_model = model_parameter_sha256(model)
        pre_optimizer = optimizer_state_sha256(optimizer, model)
        transition_episodes = []
        transition_hashes = []
        update_ids = []
        for _ in range(run_schedule.episodes_per_update):
            episode = episodes[sample_rng.randrange(len(episodes))]
            update_ids.append(episode.episode_id)
            transitions, _ = constrained_rollout(
                model,
                episode,
                torch.device("cpu"),
                mean_incidents=thresholds.mean_incidents,
                shield_enabled=False,
            )
            transition_episodes.append(transitions)
            transition_hashes.append(transition_episode_sha256(transitions))
        post_rollout_model = model_parameter_sha256(model)
        post_rollout_optimizer = optimizer_state_sha256(optimizer, model)
        if post_rollout_model != pre_model or post_rollout_optimizer != pre_optimizer:
            raise RuntimeError("PQ-1 rollout mutated model or optimizer")
        decision, metrics = pq1_cr_ppo_update(
            model,
            optimizer,
            transition_episodes,
            config,
            torch.device("cpu"),
            tensor_generator,
            thresholds=thresholds,
            rollout_model_sha256=pre_model,
            rollout_optimizer_sha256=pre_optimizer,
            rollout_transition_sha256=transition_hashes,
        )
        post_model = model_parameter_sha256(model)
        post_optimizer = optimizer_state_sha256(optimizer, model)
        mode_counts[decision.mode] += 1
        modes.append(decision.mode)
        sampled_ids.extend(update_ids)
        snapshot = metrics.pop("snapshot_diagnostics")
        row = {
            "schema_version": "multitown-pq1-update-log-v1",
            "primitives_version": PQ1_PRIMITIVES_VERSION,
            "replica": replica,
            "update": update,
            "outer_fold": run_schedule.outer_fold,
            "training_seed": run_schedule.seed,
            "mechanism": run_schedule.mechanism,
            "episodes_per_update": run_schedule.episodes_per_update,
            "sampled_episode_ids": update_ids,
            "sampled_episode_ids_sha256": _digest(update_ids),
            "pre_rollout_model_sha256": pre_model,
            "post_rollout_model_sha256": post_rollout_model,
            "pre_rollout_optimizer_sha256": pre_optimizer,
            "post_rollout_optimizer_sha256": post_rollout_optimizer,
            "post_update_model_sha256": post_model,
            "post_update_optimizer_sha256": post_optimizer,
            "rollout_digest_unchanged": True,
            "selected_actor_mode": decision.mode,
            "reward_mode_count": mode_counts["reward"],
            "unsafe_mode_count": mode_counts["unsafe"],
            "wrong_mode_count": mode_counts["wrong"],
            "snapshot_diagnostics": snapshot,
            "actor_mode_decision": asdict(decision),
            "ppo_metrics": {
                key: value
                for key, value in metrics.items()
                if key
                not in {
                    "selected_actor_mode",
                    "normalized_advantage_sha256",
                    "selected_advantage_constant",
                    "reward_advantage_raw",
                    "unsafe_advantage_raw",
                    "wrong_advantage_raw",
                    "selected_advantage_raw",
                }
            },
            "advantage_diagnostics": {
                key: metrics[key]
                for key in (
                    "selected_actor_mode",
                    "normalized_advantage_sha256",
                    "selected_advantage_constant",
                    "reward_advantage_raw",
                    "unsafe_advantage_raw",
                    "wrong_advantage_raw",
                    "selected_advantage_raw",
                )
            },
        }
        if (
            row["reward_mode_count"]
            + row["unsafe_mode_count"]
            + row["wrong_mode_count"]
            != update
            or not all(snapshot["diagnostic_gates"].values())
            or snapshot["rowwise_max_batch_size"] != 1
            or snapshot["rowwise_forward_calls"] != snapshot["transition_count"]
        ):
            raise RuntimeError("PQ-1 update qualification invariant failed")
        logs.append(row)
        _atomic_json(
            output / "progress.json",
            {
                "schema_version": "multitown-pq1-progress-v1",
                "replica": replica,
                "current_update": update,
                "scheduled_updates": run_schedule.updates,
                "reward_mode_count": mode_counts["reward"],
                "unsafe_mode_count": mode_counts["unsafe"],
                "wrong_mode_count": mode_counts["wrong"],
                "calibration_started": False,
                "outer_evaluation_started": False,
            },
            replace=update > 1,
        )
    if sampled_ids != list(expected_sample_ids):
        raise RuntimeError("PQ-1 sample sequence differs from pinned A22")
    _atomic_jsonl(output / "qualification-metrics.jsonl", logs)
    log_sha = _sha256(output / "qualification-metrics.jsonl")
    checkpoint = output / "final.pt"
    metadata = {
        "pq1_primitives_version": PQ1_PRIMITIVES_VERSION,
        "run_contract_sha256": run_contract_sha256,
        "replica": replica,
        "outer_fold": run_schedule.outer_fold,
        "mechanism": run_schedule.mechanism,
        "sample_sequence_sha256": _digest(sampled_ids),
        "mode_sequence_sha256": mode_sequence_sha256(modes),
        "qualification_log_sha256": log_sha,
        "final_model_sha256": model_parameter_sha256(model),
        "final_optimizer_sha256": optimizer_state_sha256(optimizer, model),
    }
    _checkpoint_atomic(
        checkpoint,
        model,
        optimizer,
        config,
        seed=run_schedule.seed,
        update=run_schedule.updates,
        metadata=metadata,
    )
    legacy_updates = [
        row["update"]
        for row in logs
        if row["snapshot_diagnostics"]["full_batch_log_probability"][
            "legacy_exceedances"
        ]
        > 0
    ]
    complete = {
        "schema_version": "multitown-pq1-replica-complete-v1",
        "replica": replica,
        "updates": run_schedule.updates,
        "training_episode_draws": len(sampled_ids),
        "sample_sequence_sha256": _digest(sampled_ids),
        "mode_sequence_sha256": mode_sequence_sha256(modes),
        "mode_counts": dict(sorted(mode_counts.items())),
        "initial_model_sha256": initial_model,
        "initial_optimizer_sha256": initial_optimizer,
        "final_model_sha256": metadata["final_model_sha256"],
        "final_optimizer_sha256": metadata["final_optimizer_sha256"],
        "qualification_log_sha256": log_sha,
        "checkpoint_sha256": _sha256(checkpoint),
        "legacy_tolerance_exceeded_updates": legacy_updates,
        "legacy_boundary_fingerprint": _legacy_boundary_fingerprint(logs),
        "all_rowwise_checks_exact": all(
            row["snapshot_diagnostics"]["rowwise_log_probability_exact"]
            and row["snapshot_diagnostics"]["rowwise_value_exact"]
            for row in logs
        ),
        "all_rollout_digests_unchanged": all(
            row["rollout_digest_unchanged"] for row in logs
        ),
        "all_diagnostic_gates_passed": all(
            all(row["snapshot_diagnostics"]["diagnostic_gates"].values())
            for row in logs
        ),
        "calibration_evaluations": 0,
        "outer_evaluations": 0,
        "training_seconds": time.perf_counter() - started,
        "run_contract_sha256": run_contract_sha256,
    }
    _atomic_json(output / "complete.json", complete)
    return complete, logs


def _legacy_boundary_fingerprint(
    logs: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    selected = [row for row in logs if int(row.get("update", -1)) == 41]
    if not selected:
        return None
    if len(selected) != 1:
        raise RuntimeError("PQ-1 update-41 row is not unique")
    snapshot = selected[0]["snapshot_diagnostics"]
    log_drift = snapshot["full_batch_log_probability"]
    return {
        "update": 41,
        "transition_count": snapshot["transition_count"],
        "legacy_exceedances": log_drift["legacy_exceedances"],
        "max_abs": log_drift["max_abs"],
        "legacy_max_tolerance_ratio": log_drift["legacy_max_tolerance_ratio"],
        "max_probability_ratio_drift": snapshot["max_probability_ratio_drift"],
    }


def _checkpoint_digests(
    path: Path,
    *,
    expected_replica: str,
    run_schedule: PQ1Schedule,
    run_contract_sha256: str,
) -> tuple[str, str, dict[str, Any]]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise RuntimeError(f"invalid PQ-1 checkpoint: {path}") from exc
    expected_keys = {
        "policy_version",
        "observation_size",
        "action_count",
        "hidden_size",
        "seed",
        "update",
        "ppo_config",
        "model_state",
        "optimizer_state",
        "pq1_primitives_version",
        "run_contract_sha256",
        "replica",
        "outer_fold",
        "mechanism",
        "sample_sequence_sha256",
        "mode_sequence_sha256",
        "qualification_log_sha256",
        "final_model_sha256",
        "final_optimizer_sha256",
    }
    if (
        type(payload) is not dict
        or set(payload) != expected_keys
        or not _all_finite(payload)
        or not _has_exact_types(
            payload,
            strings=(
                "policy_version",
                "pq1_primitives_version",
                "run_contract_sha256",
                "replica",
                "mechanism",
                "sample_sequence_sha256",
                "mode_sequence_sha256",
                "qualification_log_sha256",
                "final_model_sha256",
                "final_optimizer_sha256",
            ),
            integers=(
                "observation_size",
                "action_count",
                "hidden_size",
                "seed",
                "update",
                "outer_fold",
            ),
        )
        or type(payload.get("ppo_config")) is not dict
        or not _same_typed_json(
            payload["ppo_config"], asdict(_formal_config(run_schedule))
        )
        or type(payload.get("model_state")) is not dict
        or type(payload.get("optimizer_state")) is not dict
        or not all(
            _hex_digest(payload.get(key))
            for key in (
                "run_contract_sha256",
                "sample_sequence_sha256",
                "mode_sequence_sha256",
                "qualification_log_sha256",
                "final_model_sha256",
                "final_optimizer_sha256",
            )
        )
        or payload.get("policy_version") != PQ1_POLICY_VERSION
        or payload.get("pq1_primitives_version") != PQ1_PRIMITIVES_VERSION
        or payload.get("replica") != expected_replica
        or payload.get("outer_fold") != run_schedule.outer_fold
        or payload.get("mechanism") != run_schedule.mechanism
        or payload.get("seed") != run_schedule.seed
        or payload.get("update") != run_schedule.updates
        or payload.get("run_contract_sha256") != run_contract_sha256
        or payload.get("observation_size") != MultiTownLongHorizonEnv.observation_size
        or payload.get("action_count") != ACTION_COUNT
        or payload.get("ppo_config") != asdict(_formal_config(run_schedule))
    ):
        raise RuntimeError("PQ-1 checkpoint schema/metadata mismatch")
    config = _formal_config(run_schedule)
    if payload.get("hidden_size") != config.hidden_size:
        raise RuntimeError("PQ-1 checkpoint model shape mismatch")
    model = ActorCritic(
        MultiTownLongHorizonEnv.observation_size,
        config.hidden_size,
        ACTION_COUNT,
    ).cpu()
    try:
        model.load_state_dict(payload["model_state"], strict=True)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            eps=1e-5,
        )
        optimizer.load_state_dict(payload["optimizer_state"])
    except Exception as exc:
        raise RuntimeError("PQ-1 checkpoint state cannot be reconstructed") from exc
    model_digest = model_parameter_sha256(model)
    optimizer_digest = optimizer_state_sha256(optimizer, model)
    if (
        model_digest != payload["final_model_sha256"]
        or optimizer_digest != payload["final_optimizer_sha256"]
    ):
        raise RuntimeError("PQ-1 checkpoint state digest mismatch")
    return model_digest, optimizer_digest, payload


def _validate_replica_artifacts(
    output: Path,
    *,
    replica: str,
    run_schedule: PQ1Schedule,
    run_contract_sha256: str,
    expected_sample_ids: Sequence[str],
    expected_complete: Mapping[str, Any],
    expected_logs: Sequence[Mapping[str, Any]],
    require_legacy_boundary: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    expected_names = {
        "progress.json",
        "qualification-metrics.jsonl",
        "final.pt",
        "complete.json",
    }
    actual_names = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    }
    if actual_names != expected_names or any(
        path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode)
        for path in output.rglob("*")
        if path.is_file()
    ):
        raise RuntimeError(f"PQ-1 {replica} artifact inventory mismatch")
    complete = _strict_read_json(output / "complete.json")
    logs = _strict_read_jsonl(output / "qualification-metrics.jsonl")
    progress = _strict_read_json(output / "progress.json")
    if not _same_typed_json(complete, dict(expected_complete)) or not _same_typed_json(
        logs, list(expected_logs)
    ):
        raise RuntimeError(f"PQ-1 {replica} disk payload differs from trusted memory")
    complete_keys = {
        "schema_version",
        "replica",
        "updates",
        "training_episode_draws",
        "sample_sequence_sha256",
        "mode_sequence_sha256",
        "mode_counts",
        "initial_model_sha256",
        "initial_optimizer_sha256",
        "final_model_sha256",
        "final_optimizer_sha256",
        "qualification_log_sha256",
        "checkpoint_sha256",
        "legacy_tolerance_exceeded_updates",
        "legacy_boundary_fingerprint",
        "all_rowwise_checks_exact",
        "all_rollout_digests_unchanged",
        "all_diagnostic_gates_passed",
        "calibration_evaluations",
        "outer_evaluations",
        "training_seconds",
        "run_contract_sha256",
    }
    fingerprint = complete.get("legacy_boundary_fingerprint")
    if (
        set(complete) != complete_keys
        or not _has_exact_types(
            complete,
            strings=(
                "schema_version",
                "replica",
                "sample_sequence_sha256",
                "mode_sequence_sha256",
                "initial_model_sha256",
                "initial_optimizer_sha256",
                "final_model_sha256",
                "final_optimizer_sha256",
                "qualification_log_sha256",
                "checkpoint_sha256",
                "run_contract_sha256",
            ),
            integers=(
                "updates",
                "training_episode_draws",
                "calibration_evaluations",
                "outer_evaluations",
            ),
            floats=("training_seconds",),
            booleans=(
                "all_rowwise_checks_exact",
                "all_rollout_digests_unchanged",
                "all_diagnostic_gates_passed",
            ),
        )
        or not all(
            _hex_digest(complete.get(key))
            for key in (
                "sample_sequence_sha256",
                "mode_sequence_sha256",
                "initial_model_sha256",
                "initial_optimizer_sha256",
                "final_model_sha256",
                "final_optimizer_sha256",
                "qualification_log_sha256",
                "checkpoint_sha256",
                "run_contract_sha256",
            )
        )
        or type(complete.get("mode_counts")) is not dict
        or not set(complete["mode_counts"]).issubset({"reward", "unsafe", "wrong"})
        or not complete["mode_counts"]
        or any(type(value) is not int for value in complete["mode_counts"].values())
        or type(complete.get("legacy_tolerance_exceeded_updates")) is not list
        or any(
            type(value) is not int
            for value in complete["legacy_tolerance_exceeded_updates"]
        )
        or (
            fingerprint is not None
            and (
                type(fingerprint) is not dict
                or set(fingerprint) != set(EXPECTED_LEGACY_BOUNDARY)
                or not _has_exact_types(
                    fingerprint,
                    integers=("update", "transition_count", "legacy_exceedances"),
                    floats=(
                        "max_abs",
                        "legacy_max_tolerance_ratio",
                        "max_probability_ratio_drift",
                    ),
                )
            )
        )
        or len(logs) != run_schedule.updates
        or not _all_finite(logs)
        or not _all_finite(complete)
        or complete.get("schema_version") != "multitown-pq1-replica-complete-v1"
        or complete.get("replica") != replica
        or complete.get("updates") != run_schedule.updates
        or complete.get("training_episode_draws") != len(expected_sample_ids)
        or complete.get("run_contract_sha256") != run_contract_sha256
        or complete.get("calibration_evaluations") != 0
        or complete.get("outer_evaluations") != 0
        or complete.get("training_seconds", -1) < 0
    ):
        raise RuntimeError(f"PQ-1 {replica} complete schema mismatch")
    row_keys = {
        "schema_version",
        "primitives_version",
        "replica",
        "update",
        "outer_fold",
        "training_seed",
        "mechanism",
        "episodes_per_update",
        "sampled_episode_ids",
        "sampled_episode_ids_sha256",
        "pre_rollout_model_sha256",
        "post_rollout_model_sha256",
        "pre_rollout_optimizer_sha256",
        "post_rollout_optimizer_sha256",
        "post_update_model_sha256",
        "post_update_optimizer_sha256",
        "rollout_digest_unchanged",
        "selected_actor_mode",
        "reward_mode_count",
        "unsafe_mode_count",
        "wrong_mode_count",
        "snapshot_diagnostics",
        "actor_mode_decision",
        "ppo_metrics",
        "advantage_diagnostics",
    }
    snapshot_keys = {
        "schema_version",
        "transition_count",
        "rowwise_forward_calls",
        "rowwise_max_batch_size",
        "rowwise_log_probability_exact",
        "rowwise_value_exact",
        "observation_sha256",
        "mask_sha256",
        "action_sha256",
        "full_batch_log_probability",
        "full_batch_value",
        "max_probability_ratio_drift",
        "first_legacy_exceed_transition_sha256",
        "diagnostic_gates",
        "ordered_transition_episode_sha256",
        "ordered_transition_batch_sha256",
    }
    gate_keys = {
        "full_batch_log_within_frozen_tolerance",
        "full_batch_value_within_frozen_tolerance",
        "probability_ratio_drift_within_2e_5",
    }
    drift_keys = {
        "max_abs",
        "p50_abs",
        "p95_abs",
        "p99_abs",
        "max_relative",
        "max_ulp",
        "legacy_exceedances",
        "legacy_max_tolerance_ratio",
    }
    actor_keys = {
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
    ppo_keys = {
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "rollout_unsafe_events",
        "rollout_wrong_executions",
    }
    advantage_keys = {
        "selected_actor_mode",
        "normalized_advantage_sha256",
        "selected_advantage_constant",
        "reward_advantage_raw",
        "unsafe_advantage_raw",
        "wrong_advantage_raw",
        "selected_advantage_raw",
    }
    sampled: list[str] = []
    modes: list[str] = []
    mode_counts: Counter[str] = Counter()
    previous_model: str | None = None
    previous_optimizer: str | None = None
    for update, row in enumerate(logs, start=1):
        expected_ids = list(expected_sample_ids)[
            (update - 1) * run_schedule.episodes_per_update : update
            * run_schedule.episodes_per_update
        ]
        snapshot = row.get("snapshot_diagnostics")
        actor = row.get("actor_mode_decision")
        ppo = row.get("ppo_metrics")
        advantage = row.get("advantage_diagnostics")
        if (
            set(row) != row_keys
            or not _has_exact_types(
                row,
                strings=(
                    "schema_version",
                    "primitives_version",
                    "replica",
                    "mechanism",
                    "sampled_episode_ids_sha256",
                    "pre_rollout_model_sha256",
                    "post_rollout_model_sha256",
                    "pre_rollout_optimizer_sha256",
                    "post_rollout_optimizer_sha256",
                    "post_update_model_sha256",
                    "post_update_optimizer_sha256",
                    "selected_actor_mode",
                ),
                integers=(
                    "update",
                    "outer_fold",
                    "training_seed",
                    "episodes_per_update",
                    "reward_mode_count",
                    "unsafe_mode_count",
                    "wrong_mode_count",
                ),
                booleans=("rollout_digest_unchanged",),
            )
            or type(row.get("sampled_episode_ids")) is not list
            or any(type(value) is not str for value in row["sampled_episode_ids"])
            or row.get("schema_version") != "multitown-pq1-update-log-v1"
            or row.get("primitives_version") != PQ1_PRIMITIVES_VERSION
            or row.get("replica") != replica
            or row.get("update") != update
            or row.get("outer_fold") != run_schedule.outer_fold
            or row.get("training_seed") != run_schedule.seed
            or row.get("mechanism") != run_schedule.mechanism
            or row.get("episodes_per_update") != run_schedule.episodes_per_update
            or row.get("sampled_episode_ids") != expected_ids
            or row.get("sampled_episode_ids_sha256") != _digest(expected_ids)
            or row.get("rollout_digest_unchanged") is not True
            or row.get("pre_rollout_model_sha256")
            != row.get("post_rollout_model_sha256")
            or row.get("pre_rollout_optimizer_sha256")
            != row.get("post_rollout_optimizer_sha256")
            or not all(
                _hex_digest(row.get(key))
                for key in (
                    "pre_rollout_model_sha256",
                    "post_rollout_model_sha256",
                    "pre_rollout_optimizer_sha256",
                    "post_rollout_optimizer_sha256",
                    "post_update_model_sha256",
                    "post_update_optimizer_sha256",
                )
            )
            or type(snapshot) is not dict
            or set(snapshot) != snapshot_keys
            or not _has_exact_types(
                snapshot,
                strings=(
                    "schema_version",
                    "observation_sha256",
                    "mask_sha256",
                    "action_sha256",
                    "ordered_transition_batch_sha256",
                ),
                integers=(
                    "transition_count",
                    "rowwise_forward_calls",
                    "rowwise_max_batch_size",
                ),
                floats=("max_probability_ratio_drift",),
                booleans=(
                    "rowwise_log_probability_exact",
                    "rowwise_value_exact",
                ),
            )
            or snapshot.get("schema_version") != PQ1_PRIMITIVES_VERSION
            or snapshot.get("rowwise_log_probability_exact") is not True
            or snapshot.get("rowwise_value_exact") is not True
            or snapshot.get("rowwise_max_batch_size") != 1
            or snapshot.get("rowwise_forward_calls") != snapshot.get("transition_count")
            or type(snapshot.get("transition_count")) is not int
            or snapshot.get("transition_count", 0) <= 0
            or type(snapshot.get("diagnostic_gates")) is not dict
            or set(snapshot["diagnostic_gates"]) != gate_keys
            or not all(
                snapshot["diagnostic_gates"].get(key) is True for key in gate_keys
            )
            or any(
                type(snapshot.get(key)) is not dict
                or set(snapshot[key]) != drift_keys
                or not _has_exact_types(
                    snapshot[key],
                    integers=("max_ulp", "legacy_exceedances"),
                    floats=(
                        "max_abs",
                        "p50_abs",
                        "p95_abs",
                        "p99_abs",
                        "max_relative",
                        "legacy_max_tolerance_ratio",
                    ),
                )
                for key in ("full_batch_log_probability", "full_batch_value")
            )
            or type(actor) is not dict
            or set(actor) != actor_keys
            or not _has_exact_types(
                actor,
                strings=("mode",),
                floats=(
                    "unsafe_cost",
                    "wrong_cost",
                    "unsafe_threshold",
                    "wrong_threshold",
                    "unsafe_violation",
                    "wrong_violation",
                    "unsafe_normalized_violation",
                    "wrong_normalized_violation",
                ),
                booleans=(
                    "unsafe_eligible",
                    "wrong_eligible",
                    "unsafe_tie_break_used",
                ),
            )
            or type(ppo) is not dict
            or set(ppo) != ppo_keys
            or not _has_exact_types(
                ppo,
                integers=("rollout_unsafe_events", "rollout_wrong_executions"),
                floats=(
                    "policy_loss",
                    "value_loss",
                    "entropy",
                    "approx_kl",
                    "clip_fraction",
                ),
            )
            or type(advantage) is not dict
            or set(advantage) != advantage_keys
            or not _has_exact_types(
                advantage,
                strings=("selected_actor_mode", "normalized_advantage_sha256"),
                booleans=("selected_advantage_constant",),
            )
            or not _hex_digest(advantage.get("normalized_advantage_sha256"))
            or any(
                type(advantage.get(key)) is not dict
                or set(advantage[key]) != {"mean", "std", "max_abs"}
                or not _has_exact_types(
                    advantage[key], floats=("mean", "std", "max_abs")
                )
                for key in (
                    "reward_advantage_raw",
                    "unsafe_advantage_raw",
                    "wrong_advantage_raw",
                    "selected_advantage_raw",
                )
            )
        ):
            raise RuntimeError(f"PQ-1 {replica} update row invariant mismatch")
        transition_hashes = snapshot["ordered_transition_episode_sha256"]
        if (
            type(transition_hashes) is not list
            or len(transition_hashes) != run_schedule.episodes_per_update
            or not all(_hex_digest(value) for value in transition_hashes)
            or snapshot["ordered_transition_batch_sha256"]
            != hashlib.sha256("".join(transition_hashes).encode("ascii")).hexdigest()
            or not all(
                _hex_digest(snapshot[key])
                for key in (
                    "observation_sha256",
                    "mask_sha256",
                    "action_sha256",
                )
            )
            or (
                snapshot["first_legacy_exceed_transition_sha256"] is not None
                and not _hex_digest(snapshot["first_legacy_exceed_transition_sha256"])
            )
        ):
            raise RuntimeError(f"PQ-1 {replica} transition binding mismatch")
        mode = row["selected_actor_mode"]
        mode_counts[mode] += 1
        modes.append(mode)
        sampled.extend(expected_ids)
        if (
            mode not in {"reward", "unsafe", "wrong"}
            or row.get("actor_mode_decision", {}).get("mode") != mode
            or row.get("advantage_diagnostics", {}).get("selected_actor_mode") != mode
            or (
                row["reward_mode_count"],
                row["unsafe_mode_count"],
                row["wrong_mode_count"],
            )
            != (
                mode_counts["reward"],
                mode_counts["unsafe"],
                mode_counts["wrong"],
            )
            or (
                previous_model is not None
                and row["pre_rollout_model_sha256"] != previous_model
            )
            or (
                previous_optimizer is not None
                and row["pre_rollout_optimizer_sha256"] != previous_optimizer
            )
        ):
            raise RuntimeError(f"PQ-1 {replica} update sequence mismatch")
        previous_model = row["post_update_model_sha256"]
        previous_optimizer = row["post_update_optimizer_sha256"]
    if sampled != list(expected_sample_ids):
        raise RuntimeError(f"PQ-1 {replica} full sample sequence mismatch")
    log_path = output / "qualification-metrics.jsonl"
    checkpoint_path = output / "final.pt"
    if (
        complete["sample_sequence_sha256"] != _digest(sampled)
        or complete["mode_sequence_sha256"] != mode_sequence_sha256(modes)
        or complete["mode_counts"] != dict(sorted(mode_counts.items()))
        or complete["initial_model_sha256"] != logs[0]["pre_rollout_model_sha256"]
        or complete["initial_optimizer_sha256"]
        != logs[0]["pre_rollout_optimizer_sha256"]
        or complete["final_model_sha256"] != previous_model
        or complete["final_optimizer_sha256"] != previous_optimizer
        or complete["qualification_log_sha256"] != _sha256(log_path)
        or complete["checkpoint_sha256"] != _sha256(checkpoint_path)
        or complete["legacy_tolerance_exceeded_updates"]
        != [
            row["update"]
            for row in logs
            if row["snapshot_diagnostics"]["full_batch_log_probability"][
                "legacy_exceedances"
            ]
            > 0
        ]
        or complete["legacy_boundary_fingerprint"] != _legacy_boundary_fingerprint(logs)
        or complete["all_rowwise_checks_exact"] is not True
        or complete["all_rollout_digests_unchanged"] is not True
        or complete["all_diagnostic_gates_passed"] is not True
    ):
        raise RuntimeError(f"PQ-1 {replica} cross-file binding mismatch")
    expected_progress = {
        "schema_version": "multitown-pq1-progress-v1",
        "replica": replica,
        "current_update": run_schedule.updates,
        "scheduled_updates": run_schedule.updates,
        "reward_mode_count": mode_counts["reward"],
        "unsafe_mode_count": mode_counts["unsafe"],
        "wrong_mode_count": mode_counts["wrong"],
        "calibration_started": False,
        "outer_evaluation_started": False,
    }
    if not _same_typed_json(progress, expected_progress):
        raise RuntimeError(f"PQ-1 {replica} progress mismatch")
    model_digest, optimizer_digest, checkpoint = _checkpoint_digests(
        checkpoint_path,
        expected_replica=replica,
        run_schedule=run_schedule,
        run_contract_sha256=run_contract_sha256,
    )
    if (
        model_digest != complete["final_model_sha256"]
        or optimizer_digest != complete["final_optimizer_sha256"]
        or checkpoint["sample_sequence_sha256"] != complete["sample_sequence_sha256"]
        or checkpoint["mode_sequence_sha256"] != complete["mode_sequence_sha256"]
        or checkpoint["qualification_log_sha256"]
        != complete["qualification_log_sha256"]
    ):
        raise RuntimeError(f"PQ-1 {replica} checkpoint binding mismatch")
    if require_legacy_boundary and (
        41 not in complete["legacy_tolerance_exceeded_updates"]
        or complete["legacy_boundary_fingerprint"] != EXPECTED_LEGACY_BOUNDARY
    ):
        raise RuntimeError(f"PQ-1 {replica} legacy update-41 fingerprint mismatch")
    return complete, logs


def _revalidate_inputs(
    root: Path,
    *,
    smoke: bool,
    expected_preflight_signature: str,
    expected_source: Mapping[str, Any],
    expected_a23_failure: Mapping[str, Any],
    expected_runtime: Mapping[str, Any],
    run_schedule: PQ1Schedule,
    qualification_lock: tuple[Path, Mapping[str, Any]] | None = None,
) -> None:
    if _verify_a23_failure(root) != dict(expected_a23_failure):
        raise RuntimeError("PQ-1 A23 invalidation anchors changed during execution")
    current_context = _preflight(root, smoke=smoke)
    if current_context["preflight_signature"] != expected_preflight_signature:
        raise RuntimeError("PQ-1 pinned inputs changed during execution")
    if _source_state(root, require_clean=not smoke) != dict(expected_source):
        raise RuntimeError("PQ-1 source state changed during execution")
    if _runtime_fingerprint(run_schedule) != dict(expected_runtime):
        raise RuntimeError("PQ-1 runtime changed during execution")
    if qualification_lock is not None:
        _verify_qualification_lock(*qualification_lock)


def _expected_paths() -> set[str]:
    return {
        "protocol.json",
        "run-contract.json",
        "result.json",
        *{
            f"{replica}/{name}"
            for replica in PQ1_REPLICAS
            for name in (
                "progress.json",
                "qualification-metrics.jsonl",
                "final.pt",
                "complete.json",
            )
        },
    }


def _manifest(output: Path, *, source_revision: str) -> dict[str, Any]:
    expected = _expected_paths()
    actual = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    }
    if actual != expected or any(
        path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode)
        for path in output.rglob("*")
        if path.is_file()
    ):
        raise RuntimeError("PQ-1 final path inventory mismatch")
    return {
        "schema_version": "multitown-pq1-artifact-manifest-v1",
        "source_revision": source_revision,
        "files": {
            name: {
                "bytes": (output / name).stat().st_size,
                "sha256": _sha256(output / name),
            }
            for name in sorted(expected)
        },
    }


def validate_manifest(output: Path) -> dict[str, Any]:
    manifest = _strict_read_json(output / "artifact-manifest.json")
    files = manifest.get("files")
    if (
        set(manifest) != {"schema_version", "source_revision", "files"}
        or manifest.get("schema_version") != "multitown-pq1-artifact-manifest-v1"
        or type(manifest.get("source_revision")) is not str
        or len(manifest["source_revision"]) != 40
        or any(
            character not in "0123456789abcdef"
            for character in manifest["source_revision"]
        )
        or not isinstance(files, dict)
        or set(files) != _expected_paths()
    ):
        raise RuntimeError("PQ-1 manifest path inventory mismatch")
    actual = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    }
    if actual != _expected_paths() | {"artifact-manifest.json"} or any(
        path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode)
        for path in output.rglob("*")
        if path.is_file()
    ):
        raise RuntimeError("PQ-1 physical path inventory mismatch")
    for name, metadata in files.items():
        path = output / name
        if (
            type(metadata) is not dict
            or set(metadata) != {"bytes", "sha256"}
            or type(metadata.get("bytes")) is not int
            or metadata["bytes"] < 0
            or type(metadata.get("sha256")) is not str
            or len(metadata["sha256"]) != 64
            or any(
                character not in "0123456789abcdef" for character in metadata["sha256"]
            )
            or path.stat().st_size != metadata["bytes"]
            or _sha256(path) != metadata["sha256"]
        ):
            raise RuntimeError("PQ-1 manifest payload mismatch")
    return manifest


class _QualificationLockCreatedError(RuntimeError):
    pass


def _acquire_lock(lock: Path, descriptor: Mapping[str, Any]) -> None:
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(_json_payload(descriptor))
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException as exc:
        try:
            os.close(fd)
        except OSError:
            pass
        raise _QualificationLockCreatedError(
            "PQ-1 lock was created but descriptor write failed"
        ) from exc


def _qualification_lock_binding(
    lock: Path, descriptor: Mapping[str, Any]
) -> dict[str, Any]:
    payload = _json_payload(descriptor).encode("utf-8")
    return {
        "path": lock.as_posix(),
        "descriptor": dict(descriptor),
        "descriptor_sha256": _digest(descriptor),
        "bytes": len(payload),
        "file_sha256": hashlib.sha256(payload).hexdigest(),
        "mode": "0600",
    }


def _qualification_test_receipt(root: Path, *, execute: bool) -> dict[str, Any]:
    command_prefix = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-o",
        "addopts=",
        "-p",
        "no:cacheprovider",
        "tests/test_pq1_numerical_conformance.py",
        "tests/test_pq1_runner.py",
    ]
    test_sources = {
        name: _sha256(root / name)
        for name in (
            "tests/test_pq1_numerical_conformance.py",
            "tests/test_pq1_runner.py",
        )
    }
    if not execute:
        return {
            "schema_version": "multitown-pq1-required-tests-receipt-v1",
            "executed": False,
            "passed": False,
            "reason": "non-evidentiary smoke",
            "command": command_prefix,
            "test_source_sha256": test_sources,
        }
    environment = os.environ.copy()
    environment.pop("PYTEST_ADDOPTS", None)
    environment.pop("PYTEST_PLUGINS", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    with tempfile.TemporaryDirectory(prefix="multitown-pq1-tests-") as temporary:
        junit_path = Path(temporary) / "required-tests.xml"
        command = [*command_prefix, f"--junitxml={junit_path}"]
        completed = subprocess.run(
            command,
            cwd=root,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        try:
            junit_raw = junit_path.read_bytes()
            junit_root = ET.fromstring(junit_raw)
            suites = (
                [junit_root]
                if junit_root.tag == "testsuite"
                else list(junit_root.findall("./testsuite"))
            )
            if not suites:
                raise ValueError("JUnit XML has no test suite")
            junit_stats = {
                key: sum(int(suite.attrib[key]) for suite in suites)
                for key in ("tests", "failures", "errors", "skipped")
            }
            junit_stats["testcases"] = len(junit_root.findall(".//testcase"))
        except (OSError, ET.ParseError, KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "PQ-1 required tests did not produce valid JUnit XML"
            ) from exc
    stdout = completed.stdout.encode("utf-8")
    stderr = completed.stderr.encode("utf-8")
    summary = next(
        (line for line in reversed(completed.stdout.splitlines()) if line.strip()),
        "",
    )
    summary_valid = bool(
        re.fullmatch(
            (
                rf"{PQ1_REQUIRED_TEST_COUNT} passed, "
                rf"{PQ1_REQUIRED_SUBTEST_COUNT} subtests passed in .+"
            ),
            summary,
        )
    )
    structured_valid = junit_stats == {
        "tests": PQ1_REQUIRED_JUNIT_CASE_COUNT,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
        "testcases": PQ1_REQUIRED_JUNIT_TESTCASE_COUNT,
    }
    receipt = {
        "schema_version": "multitown-pq1-required-tests-receipt-v1",
        "executed": True,
        "passed": completed.returncode == 0 and summary_valid and structured_valid,
        "command": command,
        "exit_code": completed.returncode,
        "stdout_bytes": len(stdout),
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_bytes": len(stderr),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "summary": summary,
        "expected_test_count": PQ1_REQUIRED_TEST_COUNT,
        "expected_subtest_count": PQ1_REQUIRED_SUBTEST_COUNT,
        "expected_junit_case_count": PQ1_REQUIRED_JUNIT_CASE_COUNT,
        "expected_junit_testcase_count": PQ1_REQUIRED_JUNIT_TESTCASE_COUNT,
        "junit": {
            **junit_stats,
            "bytes": len(junit_raw),
            "sha256": hashlib.sha256(junit_raw).hexdigest(),
        },
        "test_source_sha256": test_sources,
    }
    if not receipt["passed"]:
        raise RuntimeError(
            "PQ-1 required tests failed before formal lock: "
            f"{summary or completed.stderr[-500:]}; junit={junit_stats}"
        )
    return receipt


def _verify_qualification_lock(lock: Path, binding: Mapping[str, Any]) -> None:
    try:
        metadata = lock.lstat()
        raw = lock.read_bytes()
    except OSError as exc:
        raise RuntimeError("PQ-1 qualification lock is missing") from exc
    expected = _qualification_lock_binding(lock, binding["descriptor"])
    if (
        type(binding) is not dict
        or set(binding)
        != {"path", "descriptor", "descriptor_sha256", "bytes", "file_sha256", "mode"}
        or not _same_typed_json(dict(binding), expected)
        or not stat.S_ISREG(metadata.st_mode)
        or lock.is_symlink()
        or metadata.st_mode & 0o777 != 0o600
        or len(raw) != binding["bytes"]
        or hashlib.sha256(raw).hexdigest() != binding["file_sha256"]
        or raw != _json_payload(binding["descriptor"]).encode("utf-8")
        or not _same_typed_json(
            _strict_json_loads(raw.decode("utf-8"), label=str(lock)),
            binding["descriptor"],
        )
    ):
        raise RuntimeError("PQ-1 qualification lock binding changed")


def _validate_final_artifacts(
    output: Path,
    *,
    root: Path,
    smoke: bool,
    protocol: Mapping[str, Any],
    contract: Mapping[str, Any],
    result: Mapping[str, Any],
    expected_manifest: Mapping[str, Any],
    trusted_products: Mapping[
        str, tuple[Mapping[str, Any], Sequence[Mapping[str, Any]]]
    ],
    run_schedule: PQ1Schedule,
    run_contract_sha256: str,
    expected_sample_ids: Sequence[str],
    expected_preflight_signature: str,
    expected_source: Mapping[str, Any],
    expected_a23_failure: Mapping[str, Any],
    expected_runtime: Mapping[str, Any],
    qualification_lock: tuple[Path, Mapping[str, Any]] | None,
) -> dict[str, Any]:
    expected_protocol = _strict_json_loads(
        _json_payload(protocol), label="trusted PQ-1 protocol"
    )
    expected_contract = _strict_json_loads(
        _json_payload(contract), label="trusted PQ-1 contract"
    )
    expected_result = _strict_json_loads(
        _json_payload(result), label="trusted PQ-1 result"
    )
    manifest: dict[str, Any] = {}
    # Two ordered passes close mutations triggered by either input revalidation
    # or manifest construction/readback before returning success.
    for _ in range(2):
        _revalidate_inputs(
            root,
            smoke=smoke,
            expected_preflight_signature=expected_preflight_signature,
            expected_source=expected_source,
            expected_a23_failure=expected_a23_failure,
            expected_runtime=expected_runtime,
            run_schedule=run_schedule,
            qualification_lock=qualification_lock,
        )
        for replica in run_schedule.replicas:
            _validate_replica_artifacts(
                output / replica,
                replica=replica,
                run_schedule=run_schedule,
                run_contract_sha256=run_contract_sha256,
                expected_sample_ids=expected_sample_ids,
                expected_complete=trusted_products[replica][0],
                expected_logs=trusted_products[replica][1],
                require_legacy_boundary=not smoke,
            )
        disk_protocol = _strict_read_json(output / "protocol.json")
        disk_contract = _strict_read_json(output / "run-contract.json")
        disk_result = _strict_read_json(output / "result.json")
        if (
            not _same_typed_json(disk_protocol, expected_protocol)
            or not _same_typed_json(disk_contract, expected_contract)
            or not _same_typed_json(disk_result, expected_result)
            or disk_contract.get("protocol_sha256") != _digest(disk_protocol)
            or disk_result.get("run_contract_sha256") != run_contract_sha256
            or disk_result.get("source_revision") != expected_source.get("revision")
            or disk_result.get("replicas")
            != [trusted_products[item][0] for item in run_schedule.replicas]
        ):
            raise RuntimeError("PQ-1 trusted root publication binding changed")
        manifest = validate_manifest(output)
        if (
            manifest["source_revision"] != expected_source["revision"]
            or manifest["files"]["result.json"]["sha256"]
            != _sha256(output / "result.json")
            or manifest["files"]["run-contract.json"]["sha256"]
            != _sha256(output / "run-contract.json")
        ):
            raise RuntimeError("PQ-1 final manifest binding changed")
    manifest_path = output / "artifact-manifest.json"
    manifest_raw = manifest_path.read_bytes()
    if not _same_typed_json(
        _strict_json_loads(manifest_raw.decode("utf-8"), label=str(manifest_path)),
        dict(expected_manifest),
    ) or manifest_raw != _json_payload(expected_manifest).encode("utf-8"):
        raise RuntimeError("PQ-1 final manifest bytes changed")
    expected_files = expected_manifest["files"]
    actual_files = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual_files != set(expected_files):
        raise RuntimeError("PQ-1 final direct path inventory changed")
    for name, metadata in expected_files.items():
        path = output / name
        file_metadata = path.lstat()
        raw = path.read_bytes()
        if (
            path.is_symlink()
            or not stat.S_ISREG(file_metadata.st_mode)
            or len(raw) != metadata["bytes"]
            or hashlib.sha256(raw).hexdigest() != metadata["sha256"]
        ):
            raise RuntimeError("PQ-1 final direct file snapshot changed")
    return manifest


def run(output: Path, *, smoke: bool) -> int:
    root = Path(__file__).resolve().parents[1]
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if not smoke and output.parent != (root / "artifacts").resolve():
        raise ValueError("formal PQ-1 output must be a direct child of artifacts")
    run_schedule = schedule(smoke)
    torch.set_num_threads(run_schedule.threads)
    torch.use_deterministic_algorithms(True, warn_only=False)
    a23_failure = _verify_a23_failure(root)
    context = _preflight(root, smoke=smoke)
    source = _source_state(root, require_clean=not smoke)
    runtime = _runtime_fingerprint(run_schedule)
    if source["revision"] != context["source"]["revision"]:
        raise RuntimeError("PQ-1 source revisions differ")
    if (
        runtime["a23_comparable"] != a23_failure["runtime"]
        or context["source"]["runtime"] != a23_failure["runtime"]
        or runtime["torch_num_threads"] != run_schedule.threads
        or runtime["torch_deterministic_algorithms"] is not True
        or runtime["torch_deterministic_warn_only"] is not False
    ):
        raise RuntimeError("PQ-1 runtime differs from the pinned A23 runtime")
    partition = partition_ids(context["assignments"], run_schedule.outer_fold)
    train_ids = partition["inner_train_ids"]
    episodes = [context["episode_index"][item] for item in train_ids]
    a8_rows = [context["a8_index"][item] for item in train_ids]
    expected_samples = context["a22_sample_sequences"][
        (
            run_schedule.outer_fold,
            run_schedule.seed,
        )
    ][: run_schedule.updates * run_schedule.episodes_per_update]
    if list(expected_samples) != _expected_sample_sequence(
        train_ids,
        seed=run_schedule.seed,
        draws=len(expected_samples),
    ):
        raise RuntimeError("PQ-1 deterministic sample oracle differs from A22")
    protocol = {
        "schema_version": "multitown-pq1-frozen-protocol-v1",
        "purpose": "train-only on-policy numerical conformance",
        "performance_evaluable": False,
        "calibration_forbidden": True,
        "outer_evaluation_forbidden": True,
        "schedule": asdict(run_schedule),
        "ppo": asdict(_formal_config(run_schedule)),
        "train_partition": {
            "outer_fold": run_schedule.outer_fold,
            "inner_train_ids": list(train_ids),
            "inner_train_ids_sha256": _digest(list(train_ids)),
        },
        "rowwise_binding": "batch-size-1 exact float32 equality",
        "full_batch_secondary_acceptance_gate": {
            "rtol": FULL_BATCH_RTOL,
            "atol": FULL_BATCH_ATOL,
            "max_probability_ratio_drift": MAX_PROBABILITY_RATIO_DRIFT,
        },
        "formal_legacy_boundary_required": not smoke,
        "formal_legacy_boundary": EXPECTED_LEGACY_BOUNDARY,
        "subsequent_legacy_exceedances": "recorded but not rejection conditions",
        "replica_match_fields": [
            "sample_sequence_sha256",
            "mode_sequence_sha256",
            "initial_model_sha256",
            "initial_optimizer_sha256",
            "final_model_sha256",
            "final_optimizer_sha256",
            "qualification_log_canonical_digest",
        ],
    }
    test_receipt = _qualification_test_receipt(root, execute=not smoke)
    if not smoke and _source_state(root, require_clean=True) != source:
        raise RuntimeError("PQ-1 source changed while running required tests")
    lock_path = (root / PQ1_LOCK).resolve()
    lock_descriptor = (
        None
        if smoke
        else {
            "schema_version": "multitown-pq1-qualification-lock-v1",
            "attempt": 1,
            "output": str(output),
            "source_revision": source["revision"],
            "protocol_sha256": _digest(protocol),
            "pq1_source_set_sha256": _digest(source["sha256"]),
        }
    )
    lock_binding = (
        None
        if lock_descriptor is None
        else _qualification_lock_binding(lock_path, lock_descriptor)
    )
    qualification_lock = None if lock_binding is None else (lock_path, lock_binding)
    contract = {
        "schema_version": "multitown-pq1-run-contract-v1",
        "source": source,
        "a23_failure": a23_failure,
        "a23_reused_source": context["source"],
        "runtime": runtime,
        "bindings": context["bindings"],
        "protocol_sha256": _digest(protocol),
        "qualification_lock": lock_binding,
        "required_tests": test_receipt,
        "non_evidentiary_smoke": smoke,
    }
    contract_sha = _digest(contract)
    running = {
        "schema_version": PQ1_RUNNER_VERSION,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "calibration_started": False,
        "outer_evaluation_started": False,
    }
    lock_acquired = False
    try:
        if not smoke:
            if lock_descriptor is None or lock_binding is None:
                raise RuntimeError("formal PQ-1 lock binding was not constructed")
            _acquire_lock(lock_path, lock_descriptor)
            lock_acquired = True
            _verify_qualification_lock(lock_path, lock_binding)
        output.mkdir(parents=True)
        _atomic_json(output / "RUNNING.json", running)
        _atomic_json(output / "protocol.json", protocol)
        _atomic_json(output / "run-contract.json", contract)
        trusted_products: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
        for replica in run_schedule.replicas:
            replica_output = output / replica
            replica_output.mkdir()
            trusted_complete, trusted_logs = _fit_replica(
                replica=replica,
                output=replica_output,
                episodes=episodes,
                a8_rows=a8_rows,
                run_schedule=run_schedule,
                run_contract_sha256=contract_sha,
                expected_sample_ids=expected_samples,
            )
            trusted_products[replica] = (trusted_complete, trusted_logs)
            _revalidate_inputs(
                root,
                smoke=smoke,
                expected_preflight_signature=context["preflight_signature"],
                expected_source=source,
                expected_a23_failure=a23_failure,
                expected_runtime=runtime,
                run_schedule=run_schedule,
                qualification_lock=qualification_lock,
            )
            _validate_replica_artifacts(
                replica_output,
                replica=replica,
                run_schedule=run_schedule,
                run_contract_sha256=contract_sha,
                expected_sample_ids=expected_samples,
                expected_complete=trusted_complete,
                expected_logs=trusted_logs,
                require_legacy_boundary=not smoke,
            )
        _revalidate_inputs(
            root,
            smoke=smoke,
            expected_preflight_signature=context["preflight_signature"],
            expected_source=source,
            expected_a23_failure=a23_failure,
            expected_runtime=runtime,
            run_schedule=run_schedule,
            qualification_lock=qualification_lock,
        )
        validated = {
            replica: _validate_replica_artifacts(
                output / replica,
                replica=replica,
                run_schedule=run_schedule,
                run_contract_sha256=contract_sha,
                expected_sample_ids=expected_samples,
                expected_complete=trusted_products[replica][0],
                expected_logs=trusted_products[replica][1],
                require_legacy_boundary=not smoke,
            )
            for replica in run_schedule.replicas
        }
        complete = [validated[replica][0] for replica in run_schedule.replicas]
        logs = {replica: validated[replica][1] for replica in run_schedule.replicas}
        canonical_log_digests = {
            replica: _digest(
                [
                    {key: value for key, value in row.items() if key != "replica"}
                    for row in logs[replica]
                ]
            )
            for replica in run_schedule.replicas
        }
        deterministic_fields = (
            "sample_sequence_sha256",
            "mode_sequence_sha256",
            "initial_model_sha256",
            "initial_optimizer_sha256",
            "final_model_sha256",
            "final_optimizer_sha256",
        )
        deterministic = bool(
            len(set(canonical_log_digests.values())) == 1
            and all(
                len({row[field] for row in complete}) == 1
                for field in deterministic_fields
            )
        )
        passed = bool(
            deterministic
            and all(row["updates"] == run_schedule.updates for row in complete)
            and all(row["all_rowwise_checks_exact"] for row in complete)
            and all(row["all_rollout_digests_unchanged"] for row in complete)
            and all(row["all_diagnostic_gates_passed"] for row in complete)
            and (smoke or test_receipt.get("passed") is True)
            and (
                smoke
                or all(
                    41 in row["legacy_tolerance_exceeded_updates"]
                    and row["legacy_boundary_fingerprint"] == EXPECTED_LEGACY_BOUNDARY
                    for row in complete
                )
            )
            and all(row["calibration_evaluations"] == 0 for row in complete)
            and all(row["outer_evaluations"] == 0 for row in complete)
        )
        if not passed:
            raise RuntimeError("PQ-1 qualification acceptance gate failed")
        result = {
            "schema_version": PQ1_RESULT_VERSION,
            "qualification_passed": True,
            "performance_evaluable": False,
            "permits_a23_retry": False,
            "permits_a24_design": not smoke,
            "source_revision": source["revision"],
            "run_contract_sha256": contract_sha,
            "qualification_lock": lock_binding,
            "required_tests": test_receipt,
            "replicas": complete,
            "canonical_log_digests": canonical_log_digests,
            "deterministic_replica_match": deterministic,
            "legacy_boundary_expected": (None if smoke else EXPECTED_LEGACY_BOUNDARY),
            "products": {
                "replicas": len(run_schedule.replicas),
                "updates": len(run_schedule.replicas) * run_schedule.updates,
                "training_episode_draws": (
                    len(run_schedule.replicas)
                    * run_schedule.updates
                    * run_schedule.episodes_per_update
                ),
                "calibration_rows": 0,
                "outer_rows": 0,
                "manifest_entries": len(_expected_paths()),
            },
            "claim_boundary": {
                "performance_result": False,
                "a23_recovery": False,
                "a24_experiment": False,
                "crpo_reproduction": False,
                "formal_safety": False,
                "llm_weight_rl": False,
            },
        }
        if _digest(_strict_read_json(output / "protocol.json")) != _digest(
            protocol
        ) or _digest(_strict_read_json(output / "run-contract.json")) != _digest(
            contract
        ):
            raise RuntimeError("PQ-1 root publications changed")
        _revalidate_inputs(
            root,
            smoke=smoke,
            expected_preflight_signature=context["preflight_signature"],
            expected_source=source,
            expected_a23_failure=a23_failure,
            expected_runtime=runtime,
            run_schedule=run_schedule,
            qualification_lock=qualification_lock,
        )
        _atomic_json(output / "result.json", result)
        if not _same_typed_json(
            _strict_read_json(output / "result.json"),
            _strict_json_loads(_json_payload(result), label="trusted PQ-1 result"),
        ):
            raise RuntimeError("PQ-1 result publication changed")
        (output / "RUNNING.json").unlink()
        trusted_manifest = _manifest(output, source_revision=source["revision"])
        _atomic_json(output / "artifact-manifest.json", trusted_manifest)
        _validate_final_artifacts(
            output,
            root=root,
            smoke=smoke,
            protocol=protocol,
            contract=contract,
            result=result,
            expected_manifest=trusted_manifest,
            trusted_products=trusted_products,
            run_schedule=run_schedule,
            run_contract_sha256=contract_sha,
            expected_sample_ids=expected_samples,
            expected_preflight_signature=context["preflight_signature"],
            expected_source=source,
            expected_a23_failure=a23_failure,
            expected_runtime=runtime,
            qualification_lock=qualification_lock,
        )
        print(
            json.dumps(
                {
                    "output": str(output),
                    "qualification_passed": True,
                    **result["products"],
                    "legacy_tolerance_exceeded_updates": {
                        row["replica"]: row["legacy_tolerance_exceeded_updates"]
                        for row in complete
                    },
                },
                indent=2,
            )
        )
        return 0
    except BaseException as exc:
        if isinstance(exc, _QualificationLockCreatedError):
            lock_acquired = True
        elif not smoke and not lock_acquired:
            raise
        if not output.exists():
            output.mkdir(parents=True, exist_ok=True)
        _isolate_valid_outputs(output)
        if not (output / "RUNNING.json").exists():
            try:
                _atomic_json(output / "RUNNING.json", running)
            except Exception as publication_error:  # noqa: BLE001
                print(
                    f"PQ-1 could not restore RUNNING marker: {publication_error}",
                    file=sys.stderr,
                )
        try:
            _atomic_json(
                output / "INVALIDATED.json",
                {
                    "schema_version": "multitown-pq1-invalidated-attempt-v1",
                    "invalidated": True,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "qualification_lock_acquired": lock_acquired,
                    "a23_retry_forbidden": True,
                    "selective_retry_forbidden": True,
                    "performance_evaluable": False,
                    "failed_at_utc": datetime.now(UTC).isoformat(),
                },
            )
        except Exception as publication_error:  # noqa: BLE001
            print(
                f"PQ-1 could not publish INVALIDATED marker: {publication_error}",
                file=sys.stderr,
            )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    raise SystemExit(run(args.output_dir, smoke=args.smoke))


if __name__ == "__main__":
    main()
