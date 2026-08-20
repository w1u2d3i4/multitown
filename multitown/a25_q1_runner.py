"""Non-evidentiary A25-Q1 Stage W multi-update wiring smoke.

This module exercises real controller-level, hard-shield-on PPO primitives on
the frozen train-only bank.  It never authorizes Q1, performance evaluation, an
outer evaluation, or an experiment that updates language-model weights.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import random
import shutil
import stat
import struct
import subprocess
import sys
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


def _capture_import_trust_root() -> dict[str, Any]:
    """Capture the nominal source chain before importing numerical dependencies."""

    source_path = Path(os.path.abspath(__file__))
    repository_path = source_path.parent.parent
    components = tuple(repository_path.parts[1:])
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    read_flags = (
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptors: list[int] = []
    identities: list[tuple[int, int, int]] = []
    file_descriptor: int | None = None
    try:
        descriptor = os.open("/", directory_flags)
        descriptors.append(descriptor)
        metadata = os.fstat(descriptor)
        identities.append(
            (metadata.st_dev, metadata.st_ino, stat.S_IFMT(metadata.st_mode))
        )
        for component in components:
            visible = os.stat(
                component,
                dir_fd=descriptor,
                follow_symlinks=False,
            )
            child = os.open(component, directory_flags, dir_fd=descriptor)
            opened = os.fstat(child)
            visible_identity = (
                visible.st_dev,
                visible.st_ino,
                stat.S_IFMT(visible.st_mode),
            )
            opened_identity = (
                opened.st_dev,
                opened.st_ino,
                stat.S_IFMT(opened.st_mode),
            )
            if not stat.S_ISDIR(opened.st_mode) or opened_identity != visible_identity:
                os.close(child)
                raise RuntimeError("Stage W import directory chain changed")
            descriptors.append(child)
            identities.append(opened_identity)
            descriptor = child
        source_parent = os.open(
            source_path.parent.name,
            directory_flags,
            dir_fd=descriptors[-1],
        )
        source_parent_metadata = os.fstat(source_parent)
        source_parent_identity = (
            source_parent_metadata.st_dev,
            source_parent_metadata.st_ino,
            stat.S_IFMT(source_parent_metadata.st_mode),
        )
        descriptors.append(source_parent)
        identities.append(source_parent_identity)
        before = os.stat(
            source_path.name,
            dir_fd=source_parent,
            follow_symlinks=False,
        )
        file_descriptor = os.open(source_path.name, read_flags, dir_fd=source_parent)
        opened_file = os.fstat(file_descriptor)
        file_identity = (
            opened_file.st_dev,
            opened_file.st_ino,
            stat.S_IFMT(opened_file.st_mode),
            opened_file.st_size,
        )
        before_identity = (
            before.st_dev,
            before.st_ino,
            stat.S_IFMT(before.st_mode),
            before.st_size,
        )
        if not stat.S_ISREG(opened_file.st_mode) or file_identity != before_identity:
            raise RuntimeError("Stage W import source file changed")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(file_descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        os.lseek(file_descriptor, 0, os.SEEK_SET)
        return {
            "repository_path": repository_path,
            "repository_descriptor_index": len(components),
            "directory_components": (*components, source_path.parent.name),
            "directory_descriptors": tuple(descriptors),
            "directory_identities": tuple(identities),
            "source_path": source_path,
            "source_descriptor": file_descriptor,
            "source_identity": file_identity,
            "source_sha256": digest.hexdigest(),
        }
    except BaseException:
        if file_descriptor is not None:
            os.close(file_descriptor)
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise


_EARLY_IMPORT_TRUST_ROOT = _capture_import_trust_root()

import numpy as np
import torch

from . import a9_oof_protocol
from .a9_oof_protocol import (
    EXPECTED_TRAIN_EPISODES,
    EXPECTED_TRAIN_SHA256,
    FROZEN_TRAIN_PATH,
    LoadedTrainBank,
    assign_stratified_group_folds,
    fold_manifest_sha256,
    load_frozen_train_bank,
)
from .a22_constrained_ppo import (
    DualState,
    SafetyThresholds,
    lagrangian_batch,
    thresholds_from_inner_train,
)
from .a23_cr_ppo import (
    model_parameter_sha256,
    select_actor_mode,
    validate_optimizer_model_binding,
)
from .a25_qualification import (
    array_mapping_sha256,
    canonical_sha256,
    generator_state_sha256,
)
from .a25_shield_dependence import (
    InterventionObjective,
    NumericalGradientEvent,
    intervention_ppo_update,
    shield_aware_batch,
    shield_aware_rollout,
)
from .long_horizon_env import (
    ACTION_COUNT,
    LongHorizonEpisode,
    MultiTownLongHorizonEnv,
    a8_heuristic_policy,
    run_policy,
)
from .ppo_controller import ActorCritic, PPOConfig
from .pq1_numerical_conformance import optimizer_state_sha256

STAGE_W_PROTOCOL_VERSION = "multitown-a25-q1-stage-w-wiring-smoke-v1"
STAGE_W_RESULT_VERSION = "multitown-a25-q1-stage-w-result-v1"
STAGE_W_STATUS = "NON_EVIDENTIARY_WIRING_SMOKE"
STAGE_W_LOCK_VERSION = "multitown-a25-q1-smoke-lock-v1"
STAGE_W_LOCK_PROTOCOL_VERSION = "multitown-a25-q1-smoke-v1"
STAGE_W_LOCK_STATUS = "LOCKED_NON_EVIDENTIARY_WIRING_SMOKE"
BOOTSTRAP_DESCRIPTOR_VERSION = "multitown-a25-q1-input-descriptors-v1"
STAGE_W_SOURCE_VERSION = "multitown-a25-q1-source-v1"
STAGE_W_RUNTIME_VERSION = "multitown-a25-q1-runtime-v1"
TRAIN_BANK_ROLE = "train-only-bank"
TRAIN_BANK_MEDIA_TYPE = "application/x-ndjson"
PROTOCOL_LOCK_BASENAME = "protocol.lock.json"
FORMAL_LOCK = "artifacts/a24-cr-ppo-no-shield-attempt-v1.lock"
MAX_BOOTSTRAP_DESCRIPTOR_BYTES = 1024 * 1024
MAX_PROTOCOL_LOCK_BYTES = 16 * 1024 * 1024
STAGE_W_SEEDS = (20260812, 20260813)
STAGE_W_ARMS = ("F00", "F01", "F10", "F11")
STAGE_W_UPDATES = 2
STAGE_W_EPISODES_PER_UPDATE = 4
STAGE_W_HIDDEN_SIZE = 8
STAGE_W_MINIBATCH_SIZE = 4096
STAGE_W_MAX_EPISODE_STEPS = 50
STAGE_W_DESIGN_FOLD = 0
STAGE_W_INNER_CALIBRATION_FOLD = 1
STAGE_W_INNER_TRAIN_FOLDS = (2, 3, 4)
STAGE_W_POSITIVE_BETA = 5.0
_SEED_DOMAIN = b"multitown-a25-q1-rng-v1\0"
_FORBIDDEN_OUTPUT_COMPONENTS = frozenset(
    {"outer", "formal", "confirmation", "hidden", "test"}
)
_FORBIDDEN_INPUT_ROLES = (
    "dev-outcome",
    "validation-outcome",
    "holdout-outcome",
    "outer",
    "test",
    "formal",
    "confirmation",
    "hidden",
    "adaptive-outer",
)
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
_REQUIRED_PRIMITIVE_MODULES = {
    "multitown": "multitown/__init__.py",
    "multitown.a22_constrained_ppo": "multitown/a22_constrained_ppo.py",
    "multitown.a23_cr_ppo": "multitown/a23_cr_ppo.py",
    "multitown.a25_q1_runner": "multitown/a25_q1_runner.py",
    "multitown.a25_qualification": "multitown/a25_qualification.py",
    "multitown.a25_shield_dependence": "multitown/a25_shield_dependence.py",
    "multitown.a9_long_horizon_env": "multitown/a9_long_horizon_env.py",
    "multitown.a9_oof_protocol": "multitown/a9_oof_protocol.py",
    "multitown.a9_ppo_oof": "multitown/a9_ppo_oof.py",
    "multitown.long_horizon_env": "multitown/long_horizon_env.py",
    "multitown.ppo_controller": "multitown/ppo_controller.py",
    "multitown.pq1_numerical_conformance": (
        "multitown/pq1_numerical_conformance.py"
    ),
}
_ORDERED_STAGE_W_GATES = (
    "exact_schedule",
    "common_initial_state",
    "arm_local_objects",
    "update_zero_common_rollout",
    "persistent_model_optimizer_rng_chains",
    "exact_continuation_restore",
    "on_policy_snapshot_binding",
    "whole_rollout_single_minibatch",
    "hard_invariants_zero",
    "positive_beta_objective_activated",
    "cr_selector_bound",
    "arm_identity_bound",
    "global_rng_unchanged",
    "zero_outer_or_formal_access",
)
_STATE_IDENTITY_FIELDS = (
    "model_sha256",
    "optimizer_sha256",
    "rollout_rng_sha256",
    "tensor_rng_sha256",
)
UpdateRule = Literal["frozen-zero-dual", "cr-ppo-inspired"]


@dataclass(frozen=True, slots=True)
class StageWSchedule:
    stage: str = "W"
    design_fold: int = STAGE_W_DESIGN_FOLD
    replica_seeds: tuple[int, ...] = STAGE_W_SEEDS
    arms: tuple[str, ...] = STAGE_W_ARMS
    updates: int = STAGE_W_UPDATES
    episodes_per_update: int = STAGE_W_EPISODES_PER_UPDATE
    q1_qualified: bool = False
    formal_authorized: bool = False
    outer_rows: int = 0

    @property
    def fits(self) -> int:
        return len(self.replica_seeds) * len(self.arms)

    @property
    def arm_updates(self) -> int:
        return self.fits * self.updates

    @property
    def training_rollouts(self) -> int:
        return self.arm_updates * self.episodes_per_update

    @property
    def shared_external_schedule_draws(self) -> int:
        return len(self.replica_seeds) * self.updates * self.episodes_per_update


@dataclass(frozen=True, slots=True)
class StateIdentity:
    model_sha256: str
    optimizer_sha256: str
    rollout_rng_sha256: str
    tensor_rng_sha256: str


@dataclass(slots=True)
class ArmState:
    arm: str
    update_rule: UpdateRule
    beta: float
    model: ActorCritic
    optimizer: torch.optim.Optimizer
    rollout_generator: torch.Generator
    tensor_generator: torch.Generator

    @property
    def identity(self) -> StateIdentity:
        validate_optimizer_model_binding(self.optimizer, self.model)
        return StateIdentity(
            model_sha256=model_parameter_sha256(self.model),
            optimizer_sha256=optimizer_state_sha256(self.optimizer, self.model),
            rollout_rng_sha256=generator_state_sha256(self.rollout_generator),
            tensor_rng_sha256=generator_state_sha256(self.tensor_generator),
        )


@dataclass(frozen=True, slots=True)
class ArmSnapshot:
    arm: str
    update_rule: UpdateRule
    beta: float
    model_state: Mapping[str, torch.Tensor]
    optimizer_state: Mapping[str, Any]
    rollout_generator_state: torch.Tensor
    tensor_generator_state: torch.Tensor
    identity: StateIdentity


@dataclass(frozen=True, slots=True)
class StageWPopulation:
    episodes: tuple[LongHorizonEpisode, ...]
    episode_sha256: Mapping[str, str]
    bank_sha256: str
    fold_manifest_sha256: str
    thresholds: SafetyThresholds
    threshold_episode_count: int


@dataclass(frozen=True, slots=True)
class AuthorizedTrainTarget:
    descriptor: Mapping[str, Any]
    path: Path
    before: os.stat_result
    parent_descriptor: int
    leaf_name: str
    parent_chain: PinnedPathChain


@dataclass(slots=True)
class PinnedPathChain:
    path: Path
    components: tuple[str, ...]
    descriptors: tuple[int, ...]
    identities: tuple[tuple[int, int, int], ...]
    closed: bool = False

    @property
    def leaf_descriptor(self) -> int:
        if self.closed:
            raise RuntimeError("pinned path chain is closed")
        return self.descriptors[-1]

    def revalidate(self, *, label: str) -> None:
        if self.closed or len(self.descriptors) != len(self.identities):
            raise RuntimeError(f"{label} pinned path chain is unavailable")
        for index, (descriptor, expected) in enumerate(
            zip(self.descriptors, self.identities, strict=True)
        ):
            if _path_entry_identity(os.fstat(descriptor)) != expected:
                raise RuntimeError(f"{label} pinned descriptor identity changed")
            if index:
                visible = os.stat(
                    self.components[index - 1],
                    dir_fd=self.descriptors[index - 1],
                    follow_symlinks=False,
                )
                if _path_entry_identity(visible) != expected:
                    raise RuntimeError(f"{label} nominal path chain changed")

    def close(self) -> None:
        if self.closed:
            return
        for descriptor in reversed(self.descriptors):
            os.close(descriptor)
        self.closed = True


@dataclass(frozen=True, slots=True)
class PinnedDirectory:
    path: Path
    parent_descriptor: int
    descriptor: int
    leaf_name: str
    identity: tuple[int, ...]
    chain: PinnedPathChain


def _resolve_git_executable() -> Path:
    candidate = shutil.which("git")
    if candidate is None:
        raise RuntimeError("Stage W prepare requires Git")
    path = Path(candidate)
    if path.is_symlink():
        raise RuntimeError("Stage W prepare Git executable is a symlink")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise RuntimeError("Stage W prepare Git executable is invalid")
    return resolved


_GIT_EXECUTABLE = _resolve_git_executable()


class _AuxiliaryGradientObserver:
    """Measure the live auxiliary gradient without mutating guarded state."""

    def __init__(self) -> None:
        self.l2_values: list[float] = []

    def observe(self, event: NumericalGradientEvent) -> None:
        if event.stage != "loss_terms":
            return
        if event.loss_terms is None:
            raise RuntimeError("Stage W auxiliary gradient event is incomplete")
        gradients = torch.autograd.grad(
            event.loss_terms.auxiliary_loss,
            tuple(parameter for _, parameter in event.named_parameters),
            retain_graph=True,
            allow_unused=True,
        )
        squared = 0.0
        for gradient in gradients:
            if gradient is None:
                continue
            if not bool(torch.isfinite(gradient).all()):
                raise FloatingPointError("non-finite Stage W auxiliary gradient")
            squared += float(gradient.detach().double().square().sum().item())
        norm = math.sqrt(squared)
        if not math.isfinite(norm):
            raise FloatingPointError("non-finite Stage W auxiliary gradient norm")
        self.l2_values.append(norm)


def stage_w_schedule() -> StageWSchedule:
    schedule = StageWSchedule()
    if (
        schedule.arm_updates != 16
        or schedule.training_rollouts != 64
        or schedule.shared_external_schedule_draws != 16
    ):
        raise RuntimeError("Stage W static schedule identity changed")
    return schedule


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _valid_sha256(value: Any) -> bool:
    return bool(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _exact_json_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return bool(
            set(actual) == set(expected)
            and all(_exact_json_equal(actual[key], expected[key]) for key in expected)
        )
    if type(expected) in {list, tuple}:
        return bool(
            len(actual) == len(expected)
            and all(
                _exact_json_equal(actual_value, expected_value)
                for actual_value, expected_value in zip(actual, expected, strict=True)
            )
        )
    return bool(actual == expected)


def _require_exact_json(actual: Any, expected: Any, *, label: str) -> None:
    if not _exact_json_equal(actual, expected):
        raise ValueError(f"{label} differs in value or exact JSON type")


def _strict_json_bytes(payload: bytes, *, label: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant in {label}: {value}")

    try:
        text_payload = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"non-UTF-8 JSON in {label}") from error
    try:
        return json.loads(
            text_payload,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {label}") from error


def _stat_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _directory_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
    )


def _path_entry_identity(metadata: os.stat_result) -> tuple[int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
    )


def _early_import_source_binding(repository: Path) -> dict[str, Any]:
    capture = _EARLY_IMPORT_TRUST_ROOT
    if repository != capture["repository_path"]:
        raise RuntimeError("Stage W repository differs from the import trust root")
    descriptors = capture["directory_descriptors"]
    identities = capture["directory_identities"]
    components = capture["directory_components"]
    for index, (descriptor, expected) in enumerate(
        zip(descriptors, identities, strict=True)
    ):
        if _path_entry_identity(os.fstat(descriptor)) != expected:
            raise RuntimeError("Stage W import directory descriptor changed")
        if index:
            visible = os.stat(
                components[index - 1],
                dir_fd=descriptors[index - 1],
                follow_symlinks=False,
            )
            if _path_entry_identity(visible) != expected:
                raise RuntimeError("Stage W import directory chain changed")
    repository_index = capture["repository_descriptor_index"]
    repository_identity = identities[repository_index]
    current_repository = _pin_directory_chain(
        repository,
        label="Stage W nominal repository import binding",
    )
    try:
        if current_repository.identities[-1] != repository_identity:
            raise RuntimeError("Stage W nominal repository root was replaced")
        current_repository.revalidate(label="Stage W nominal repository import binding")
    finally:
        current_repository.close()

    retained_file = os.fstat(capture["source_descriptor"])
    retained_file_identity = (
        retained_file.st_dev,
        retained_file.st_ino,
        stat.S_IFMT(retained_file.st_mode),
        retained_file.st_size,
    )
    visible_file = os.stat(
        capture["source_path"].name,
        dir_fd=descriptors[-1],
        follow_symlinks=False,
    )
    visible_file_identity = (
        visible_file.st_dev,
        visible_file.st_ino,
        stat.S_IFMT(visible_file.st_mode),
        visible_file.st_size,
    )
    if (
        retained_file_identity != capture["source_identity"]
        or visible_file_identity != capture["source_identity"]
    ):
        raise RuntimeError("Stage W imported Q1 source identity changed")
    digest = hashlib.sha256()
    offset = 0
    while offset < retained_file.st_size:
        chunk = os.pread(
            capture["source_descriptor"],
            min(1024 * 1024, retained_file.st_size - offset),
            offset,
        )
        if not chunk:
            raise RuntimeError("Stage W imported Q1 source was short-read")
        digest.update(chunk)
        offset += len(chunk)
    if digest.hexdigest() != capture["source_sha256"]:
        raise RuntimeError("Stage W imported Q1 source bytes changed")
    return {
        "status": "EARLY_IMPORT_PATH_AND_BYTES_BOUND",
        "repository": {
            "absolute_path": str(repository),
            "device": repository_identity[0],
            "inode": repository_identity[1],
            "file_type": repository_identity[2],
        },
        "q1_source": {
            "absolute_path": str(capture["source_path"]),
            "device": capture["source_identity"][0],
            "inode": capture["source_identity"][1],
            "file_type": capture["source_identity"][2],
            "bytes": capture["source_identity"][3],
            "sha256": capture["source_sha256"],
        },
        "full_directory_chain_bound": True,
        "trusted_launcher": False,
        "memory_bytecode_attested": False,
    }


def _absolute_canonical_path(raw: Any, *, label: str) -> Path:
    if type(raw) is not str or not raw:
        raise TypeError(f"{label} must be a non-empty absolute path string")
    if (
        not raw.startswith("/")
        or raw.startswith("//")
        or "//" in raw
        or (raw != "/" and raw.endswith("/"))
        or "\x00" in raw
    ):
        raise ValueError(f"{label} is not a single-anchor canonical path")
    components = raw.split("/")[1:]
    if any(component in {"", ".", ".."} for component in components):
        raise ValueError(f"{label} contains a non-canonical path component")
    path = Path(raw)
    if (
        path.anchor != "/"
        or not path.is_absolute()
        or str(path) != raw
        or os.path.normpath(raw) != raw
    ):
        raise ValueError(f"{label} is not an absolute canonical path")
    return path


def _caller_absolute_path(path: Path, *, label: str) -> Path:
    raw = os.fspath(path)
    if not isinstance(raw, str):
        raise TypeError(f"{label} must be a filesystem path")
    return _absolute_canonical_path(raw, label=label)


_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_READ_FLAGS = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)


def _pin_directory_chain(path: Path, *, label: str) -> PinnedPathChain:
    """Pin every nominal directory entry from ``/`` through ``path``."""

    supplied = _caller_absolute_path(path, label=label)
    components = tuple(supplied.parts[1:])
    descriptors: list[int] = []
    identities: list[tuple[int, int, int]] = []
    try:
        root_descriptor = os.open("/", _DIRECTORY_FLAGS)
        descriptors.append(root_descriptor)
        root_metadata = os.fstat(root_descriptor)
        if not stat.S_ISDIR(root_metadata.st_mode):
            raise RuntimeError("filesystem anchor is not a directory")
        identities.append(_path_entry_identity(root_metadata))
        for component in components:
            parent_descriptor = descriptors[-1]
            try:
                visible = os.stat(
                    component,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
                next_descriptor = os.open(
                    component,
                    _DIRECTORY_FLAGS,
                    dir_fd=parent_descriptor,
                )
            except OSError as error:
                raise ValueError(
                    f"unsafe or symlink directory component in {label}"
                ) from error
            try:
                opened = os.fstat(next_descriptor)
                if (
                    not stat.S_ISDIR(visible.st_mode)
                    or not stat.S_ISDIR(opened.st_mode)
                    or _path_entry_identity(opened) != _path_entry_identity(visible)
                ):
                    raise RuntimeError(f"{label} changed while pinning")
            except BaseException:
                os.close(next_descriptor)
                raise
            descriptors.append(next_descriptor)
            identities.append(_path_entry_identity(opened))
        chain = PinnedPathChain(
            path=supplied,
            components=components,
            descriptors=tuple(descriptors),
            identities=tuple(identities),
        )
        chain.revalidate(label=label)
        return chain
    except BaseException:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise


def _open_directory_path(path: Path, *, label: str) -> int:
    """Compatibility helper returning a duplicate of a fully pinned leaf fd."""

    chain = _pin_directory_chain(path, label=label)
    try:
        return os.dup(chain.leaf_descriptor)
    finally:
        chain.close()


@contextmanager
def _pinned_parent(path: Path, *, label: str):
    supplied = _caller_absolute_path(path, label=label)
    if supplied == Path("/") or supplied.name in {"", ".", ".."}:
        raise ValueError(f"{label} has no safe leaf name")
    parent_chain = _pin_directory_chain(
        supplied.parent,
        label=f"{label} parent",
    )
    try:
        parent_chain.revalidate(label=f"{label} parent before operation")
        yield parent_chain, supplied.name
        parent_chain.revalidate(label=f"{label} parent after operation")
    finally:
        parent_chain.close()


def _pin_existing_directory(path: Path, *, label: str) -> PinnedDirectory:
    supplied = _caller_absolute_path(path, label=label)
    if supplied == Path("/"):
        raise ValueError(f"{label} may not be the filesystem anchor")
    chain = _pin_directory_chain(supplied, label=label)
    try:
        parent_descriptor = chain.descriptors[-2]
        descriptor = chain.leaf_descriptor
        before = os.stat(
            supplied.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISDIR(before.st_mode):
            raise ValueError(f"{label} is not a no-follow directory")
        opened = os.fstat(descriptor)
        if _path_entry_identity(opened) != _path_entry_identity(before):
            raise RuntimeError(f"{label} changed during pin")
        return PinnedDirectory(
            path=supplied,
            parent_descriptor=parent_descriptor,
            descriptor=descriptor,
            leaf_name=supplied.name,
            identity=_directory_identity(opened),
            chain=chain,
        )
    except BaseException:
        chain.close()
        raise


def _close_pinned_directory(pinned: PinnedDirectory) -> None:
    pinned.chain.close()


def _pinned_directory_entry_unchanged(pinned: PinnedDirectory) -> bool:
    try:
        pinned.chain.revalidate(label=f"{pinned.path} directory chain")
        current = os.stat(
            pinned.leaf_name,
            dir_fd=pinned.parent_descriptor,
            follow_symlinks=False,
        )
    except (FileNotFoundError, RuntimeError, ValueError):
        return False
    return _directory_identity(current) == pinned.identity


def _directory_has_ancestor(
    descriptor: int,
    *,
    ancestor_device: int,
    ancestor_inode: int,
) -> bool:
    current = os.dup(descriptor)
    try:
        for _ in range(1024):
            metadata = os.fstat(current)
            if (metadata.st_dev, metadata.st_ino) == (
                ancestor_device,
                ancestor_inode,
            ):
                return True
            parent = os.open("..", _DIRECTORY_FLAGS, dir_fd=current)
            parent_metadata = os.fstat(parent)
            if (parent_metadata.st_dev, parent_metadata.st_ino) == (
                metadata.st_dev,
                metadata.st_ino,
            ):
                os.close(parent)
                return False
            os.close(current)
            current = parent
        raise RuntimeError("directory ancestry exceeds safety limit")
    finally:
        os.close(current)


def _exact_integer(
    value: Any,
    *,
    label: str,
    minimum: int = 0,
) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an exact integer")
    if value < minimum:
        raise ValueError(f"{label} is below its minimum")
    return value


def _validate_target_numeric_fields(descriptor: Mapping[str, Any]) -> None:
    _exact_integer(descriptor.get("device"), label="target device", minimum=0)
    _exact_integer(descriptor.get("inode"), label="target inode", minimum=1)
    _exact_integer(descriptor.get("size"), label="target size", minimum=1)


def _secure_read_file(
    path: Path,
    *,
    label: str,
    maximum_bytes: int,
    required_mode: int,
    required_parent_mode: int | None = None,
) -> bytes:
    supplied = _caller_absolute_path(path, label=label)
    with _pinned_parent(supplied, label=label) as (parent_chain, leaf_name):
        parent_descriptor = parent_chain.leaf_descriptor
        parent_chain.revalidate(label=f"{label} parent before read")
        parent_metadata = os.fstat(parent_descriptor)
        if (
            required_parent_mode is not None
            and stat.S_IMODE(parent_metadata.st_mode) != required_parent_mode
        ):
            raise ValueError(
                f"{label} parent directory must be mode {required_parent_mode:04o}"
            )
        before = os.stat(
            leaf_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != required_mode
            or not 0 < before.st_size <= maximum_bytes
        ):
            raise ValueError(
                f"{label} must be a regular non-symlink mode "
                f"{required_mode:04o} file within its size limit"
            )
        descriptor = os.open(leaf_name, _READ_FLAGS, dir_fd=parent_descriptor)
        try:
            opened = os.fstat(descriptor)
            if _stat_identity(opened) != _stat_identity(before):
                raise RuntimeError(f"{label} changed before read")
            chunks: list[bytes] = []
            observed_size = 0
            while True:
                chunk = os.read(
                    descriptor,
                    min(1024 * 1024, maximum_bytes + 1 - observed_size),
                )
                if not chunk:
                    break
                observed_size += len(chunk)
                if observed_size > maximum_bytes:
                    raise ValueError(f"{label} exceeds its size limit")
                chunks.append(chunk)
            after = os.fstat(descriptor)
            current = os.stat(
                leaf_name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if _stat_identity(after) != _stat_identity(before) or _stat_identity(
                current
            ) != _stat_identity(before):
                raise RuntimeError(f"{label} changed during read")
            parent_chain.revalidate(label=f"{label} parent after read")
            return b"".join(chunks)
        finally:
            os.close(descriptor)


def _validate_environment() -> None:
    present = sorted(_FORBIDDEN_ENVIRONMENT.intersection(os.environ))
    if present:
        raise RuntimeError(
            "Stage W prepare refuses source/import override environment: "
            + ",".join(present)
        )


def _git_output(root: Path, *arguments: str) -> bytes:
    chain = _pin_directory_chain(root, label="Stage W Git repository root")
    try:
        chain.revalidate(label="Stage W Git repository before command")
        completed = subprocess.run(
            [str(_GIT_EXECUTABLE), *arguments],
            cwd=f"/proc/self/fd/{chain.leaf_descriptor}",
            check=False,
            capture_output=True,
        )
        chain.revalidate(label="Stage W Git repository after command")
    finally:
        chain.close()
    if completed.returncode:
        raise RuntimeError("Stage W prepare Git command failed: " + " ".join(arguments))
    return completed.stdout


def _repository_root(root: Path) -> Path:
    supplied = _caller_absolute_path(root, label="repository root")
    descriptor = _open_directory_path(supplied, label="repository root")
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ValueError("Stage W repository root is unsafe")
    finally:
        os.close(descriptor)
    return supplied


def _logical_loaded_module(name: str) -> Any:
    module = sys.modules.get(name)
    if module is not None:
        return module
    main_module = sys.modules.get("__main__")
    spec = getattr(main_module, "__spec__", None)
    if getattr(spec, "name", None) == name:
        return main_module
    raise RuntimeError(f"Stage W required module is not loaded: {name}")


def _loaded_multitown_source_modules(root: Path) -> dict[str, str]:
    loaded: dict[str, str] = {}
    for module_name, module in tuple(sys.modules.items()):
        logical_name = module_name
        if module_name == "__main__":
            spec = getattr(module, "__spec__", None)
            logical_name = getattr(spec, "name", module_name)
        if not (
            logical_name == "multitown" or logical_name.startswith("multitown.")
        ):
            continue
        raw_origin = getattr(module, "__file__", None)
        if type(raw_origin) is not str or not raw_origin:
            raise RuntimeError(
                f"Stage W loaded MultiTown module has no file origin: {logical_name}"
            )
        try:
            origin = Path(raw_origin).resolve(strict=True)
            relative = origin.relative_to(root).as_posix()
        except (OSError, ValueError) as error:
            raise RuntimeError(
                f"Stage W loaded MultiTown module is outside repository: {logical_name}"
            ) from error
        if origin.suffix != ".py":
            raise RuntimeError(
                f"Stage W loaded MultiTown module is not Python source: {logical_name}"
            )
        loaded[logical_name] = relative
    return dict(sorted(loaded.items()))


def _loaded_module_origins(
    root: Path,
    *,
    repository_chain: PinnedPathChain,
) -> dict[str, dict[str, Any]]:
    repository_chain.revalidate(label="Stage W source root before module closure")
    loaded_closure = _loaded_multitown_source_modules(root)
    if loaded_closure != dict(sorted(_REQUIRED_PRIMITIVE_MODULES.items())):
        raise RuntimeError("Stage W loaded MultiTown source closure changed")
    origins: dict[str, dict[str, Any]] = {}
    for module_name, expected_relative in _REQUIRED_PRIMITIVE_MODULES.items():
        module = _logical_loaded_module(module_name)
        raw_origin = getattr(module, "__file__", None)
        if type(raw_origin) is not str or not raw_origin:
            raise RuntimeError(f"Stage W module has no origin: {module_name}")
        origin = Path(raw_origin)
        if origin.is_symlink():
            raise RuntimeError(f"Stage W module origin is a symlink: {module_name}")
        try:
            resolved = origin.resolve(strict=True)
            relative = resolved.relative_to(root).as_posix()
        except (OSError, ValueError) as error:
            raise RuntimeError(
                f"Stage W module is outside the bound repository: {module_name}"
            ) from error
        if relative != expected_relative:
            raise RuntimeError(f"Stage W module origin changed: {module_name}")
        metadata = resolved.stat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or resolved.suffix != ".py"
        ):
            raise RuntimeError(f"Stage W module origin is unsafe: {module_name}")
        payload = _secure_read_file(
            resolved,
            label=f"Stage W source module {module_name}",
            maximum_bytes=16 * 1024 * 1024,
            required_mode=stat.S_IMODE(metadata.st_mode),
        )
        if payload != _git_output(root, "show", f"HEAD:{relative}"):
            raise RuntimeError(f"Stage W module differs from clean HEAD: {module_name}")
        origins[module_name] = {
            "path": relative,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
    repository_chain.revalidate(label="Stage W source root after module closure")
    return origins


def _source_state(root: Path) -> dict[str, Any]:
    _validate_environment()
    repository = _repository_root(root)
    import_binding = _early_import_source_binding(repository)
    repository_chain = _pin_directory_chain(
        repository,
        label="Stage W source repository",
    )
    try:
        repository_chain.revalidate(label="Stage W source repository before Git")
        top_raw = (
            _git_output(repository, "rev-parse", "--show-toplevel")
            .decode("utf-8")
            .strip()
        )
        top = Path(top_raw).resolve(strict=True)
        top_metadata = os.stat(top, follow_symlinks=False)
        expected_repository = import_binding["repository"]
        if (
            top != repository
            or top_metadata.st_dev != expected_repository["device"]
            or top_metadata.st_ino != expected_repository["inode"]
        ):
            raise RuntimeError("Stage W root is not the imported Git top level")
        if _git_output(repository, "status", "--porcelain=v1", "--untracked-files=all"):
            raise RuntimeError("Stage W prepare requires exact clean source")
        revision = _git_output(repository, "rev-parse", "HEAD").decode().strip()
        tree = _git_output(repository, "rev-parse", "HEAD^{tree}").decode().strip()
        if (
            len(revision) != 40
            or len(tree) != 40
            or not all(value in "0123456789abcdef" for value in revision + tree)
        ):
            raise RuntimeError("Stage W prepare requires 40-hex Git revision and tree")
        origins = _loaded_module_origins(
            repository,
            repository_chain=repository_chain,
        )
        source_files = {row["path"]: row["sha256"] for row in origins.values()}
        source = {
            "schema_version": STAGE_W_SOURCE_VERSION,
            "revision": revision,
            "tree": tree,
            "clean": True,
            "import_trust_root": import_binding,
            "source_files_sha256": dict(sorted(source_files.items())),
            "source_bundle_sha256": canonical_sha256(source_files),
            "module_origins": origins,
            "git": {
                "executable": str(_GIT_EXECUTABLE),
                "executable_sha256": hashlib.sha256(
                    _GIT_EXECUTABLE.read_bytes()
                ).hexdigest(),
                "version": _git_output(repository, "--version").decode().strip(),
            },
        }
        repository_chain.revalidate(label="Stage W source repository after Git")
        if _early_import_source_binding(repository) != import_binding:
            raise RuntimeError(
                "Stage W import trust root changed during source binding"
            )
        return source
    finally:
        repository_chain.close()


def _runtime_module_identity(module: Any, *, label: str) -> dict[str, Any]:
    raw_origin = getattr(module, "__file__", None)
    if type(raw_origin) is not str or not raw_origin:
        raise RuntimeError(f"Stage W {label} has no import origin")
    origin = Path(raw_origin)
    if origin.is_symlink():
        raise RuntimeError(f"Stage W {label} import origin is a symlink")
    resolved = origin.resolve(strict=True)
    metadata = resolved.stat()
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"Stage W {label} import origin is not regular")
    return {
        "version": str(module.__version__),
        "origin": str(resolved),
        "origin_sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
    }


def _runtime_profile() -> dict[str, Any]:
    _validate_environment()
    executable = Path(sys.executable)
    if executable.is_symlink():
        executable = executable.resolve(strict=True)
    else:
        executable = executable.resolve(strict=True)
    if not executable.is_file():
        raise RuntimeError("Stage W Python executable is invalid")
    return {
        "schema_version": STAGE_W_RUNTIME_VERSION,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "compiler": platform.python_compiler(),
            "executable": str(executable),
            "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        },
        "torch": {
            **_runtime_module_identity(torch, label="Torch"),
            "git_version": str(torch.version.git_version),
            "cuda_build": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "default_dtype": str(torch.get_default_dtype()),
        },
        "numpy": _runtime_module_identity(np, label="NumPy"),
        "os": {
            "name": os.name,
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "byteorder": sys.byteorder,
        },
        "cpu": {
            "processor": platform.processor(),
            "machine": platform.machine(),
            "logical_count": os.cpu_count(),
        },
        "execution": {
            "device": "cpu",
            "torch_intraop_threads": int(torch.get_num_threads()),
            "torch_interop_threads": int(torch.get_num_interop_threads()),
            "environment": {key: os.environ.get(key) for key in _RECORDED_ENVIRONMENT},
        },
    }


def _validate_runtime_binding(runtime: Any) -> None:
    if type(runtime) is not dict or set(runtime) != {
        "schema_version",
        "python",
        "torch",
        "numpy",
        "os",
        "cpu",
        "execution",
    }:
        raise ValueError("invalid Stage W runtime schema")
    if runtime["schema_version"] != STAGE_W_RUNTIME_VERSION:
        raise ValueError("unsupported Stage W runtime schema")
    python = runtime["python"]
    if (
        type(python) is not dict
        or set(python)
        != {"implementation", "version", "compiler", "executable", "executable_sha256"}
        or any(
            type(python[key]) is not str or not python[key]
            for key in ("implementation", "version", "compiler")
        )
        or not _valid_sha256(python["executable_sha256"])
    ):
        raise ValueError("invalid Stage W Python runtime binding")
    _absolute_canonical_path(python["executable"], label="Python executable")
    for label in ("torch", "numpy"):
        module = runtime[label]
        expected_keys = {"version", "origin", "origin_sha256"}
        if label == "torch":
            expected_keys |= {
                "git_version",
                "cuda_build",
                "cuda_available",
                "default_dtype",
            }
        if (
            type(module) is not dict
            or set(module) != expected_keys
            or type(module["version"]) is not str
            or not module["version"]
            or not _valid_sha256(module["origin_sha256"])
        ):
            raise ValueError(f"invalid Stage W {label} runtime binding")
        _absolute_canonical_path(module["origin"], label=f"{label} import origin")
        if label == "torch" and (
            type(module["git_version"]) is not str
            or not module["git_version"]
            or type(module["cuda_available"]) is not bool
            or type(module["default_dtype"]) is not str
            or module["default_dtype"]
            not in {
                "torch.float16",
                "torch.float32",
                "torch.float64",
                "torch.bfloat16",
            }
            or type(module["cuda_build"]) not in {str, type(None)}
            or (type(module["cuda_build"]) is str and not module["cuda_build"])
        ):
            raise ValueError("invalid Stage W Torch runtime details")
    operating_system = runtime["os"]
    if (
        type(operating_system) is not dict
        or set(operating_system)
        != {"name", "system", "release", "version", "machine", "byteorder"}
        or any(
            type(value) is not str or not value for value in operating_system.values()
        )
        or operating_system["byteorder"] not in {"little", "big"}
    ):
        raise ValueError("invalid Stage W OS runtime binding")
    cpu = runtime["cpu"]
    if (
        type(cpu) is not dict
        or set(cpu) != {"processor", "machine", "logical_count"}
        or type(cpu["processor"]) is not str
        or type(cpu["machine"]) is not str
        or not cpu["machine"]
    ):
        raise ValueError("invalid Stage W CPU runtime binding")
    _exact_integer(cpu["logical_count"], label="CPU logical count", minimum=1)
    execution = runtime["execution"]
    if (
        type(execution) is not dict
        or set(execution)
        != {"device", "torch_intraop_threads", "torch_interop_threads", "environment"}
        or execution["device"] != "cpu"
    ):
        raise ValueError("invalid Stage W execution runtime binding")
    _exact_integer(
        execution["torch_intraop_threads"],
        label="Torch intraop threads",
        minimum=1,
    )
    _exact_integer(
        execution["torch_interop_threads"],
        label="Torch interop threads",
        minimum=1,
    )
    environment = execution["environment"]
    if (
        type(environment) is not dict
        or tuple(environment) != _RECORDED_ENVIRONMENT
        or any(type(value) not in {str, type(None)} for value in environment.values())
    ):
        raise ValueError("invalid Stage W recorded runtime environment")


def _read_bootstrap_descriptors(path: Path) -> dict[str, Any]:
    payload = _secure_read_file(
        path,
        label="bootstrap descriptor",
        maximum_bytes=MAX_BOOTSTRAP_DESCRIPTOR_BYTES,
        required_mode=0o600,
    )
    value = _strict_json_bytes(payload, label="bootstrap descriptor")
    if type(value) is not dict or set(value) != {"schema_version", "targets"}:
        raise ValueError("invalid bootstrap descriptor schema")
    if value["schema_version"] != BOOTSTRAP_DESCRIPTOR_VERSION:
        raise ValueError("unsupported bootstrap descriptor schema")
    targets = value["targets"]
    if type(targets) is not list or len(targets) != 1:
        raise ValueError("bootstrap descriptor requires exactly one target")
    target = targets[0]
    required = {
        "role",
        "absolute_path",
        "media_type",
        "device",
        "inode",
        "size",
        "sha256",
        "allowed_root",
    }
    if type(target) is not dict or set(target) != required:
        raise ValueError("invalid bootstrap target schema")
    return {
        "value": value,
        "file_sha256": hashlib.sha256(payload).hexdigest(),
        "target": target,
    }


def _authorize_train_bank_descriptor(
    descriptor: Mapping[str, Any],
) -> AuthorizedTrainTarget:
    if type(descriptor) is not dict:
        raise TypeError("train-bank descriptor must be an object")
    required = {
        "role",
        "absolute_path",
        "media_type",
        "device",
        "inode",
        "size",
        "sha256",
        "allowed_root",
    }
    if set(descriptor) != required:
        raise ValueError("invalid train-bank descriptor schema")
    role = descriptor["role"]
    if type(role) is not str or role != TRAIN_BANK_ROLE:
        raise PermissionError("train-bank descriptor role is not authorized")
    if role in _FORBIDDEN_INPUT_ROLES:
        raise PermissionError("forbidden train-bank descriptor role")

    target = _absolute_canonical_path(
        descriptor["absolute_path"], label="train-bank target path"
    )
    expected_target = _caller_absolute_path(
        Path(FROZEN_TRAIN_PATH),
        label="frozen train-bank contract path",
    )
    if target != expected_target:
        raise PermissionError("train-bank target path is not authorized")
    if any(part.casefold() in _FORBIDDEN_OUTPUT_COMPONENTS for part in target.parts):
        raise PermissionError("train-bank target path contains a forbidden role")

    allowed_root = _absolute_canonical_path(
        descriptor["allowed_root"], label="train-bank allowed_root"
    )
    expected_root = expected_target.parent
    if allowed_root != expected_root or target.parent != allowed_root:
        raise PermissionError("train-bank allowed_root is not authorized")
    if descriptor["media_type"] != TRAIN_BANK_MEDIA_TYPE:
        raise PermissionError("train-bank media type is not authorized")
    if not _valid_sha256(descriptor["sha256"]):
        raise ValueError("train-bank descriptor has invalid SHA-256")
    if descriptor["sha256"] != EXPECTED_TRAIN_SHA256:
        raise PermissionError("train-bank descriptor SHA-256 is not authorized")
    _validate_target_numeric_fields(descriptor)

    parent_chain = _pin_directory_chain(
        allowed_root,
        label="train-bank allowed_root",
    )
    parent_descriptor = parent_chain.leaf_descriptor
    try:
        parent_chain.revalidate(label="train-bank allowed_root before authorization")
        root_metadata = os.fstat(parent_descriptor)
        if not stat.S_ISDIR(root_metadata.st_mode):
            raise ValueError("train-bank allowed_root is not a directory")
        before = os.stat(
            target.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError("train-bank target must be a regular non-symlink file")
        if (
            before.st_dev != descriptor["device"]
            or before.st_ino != descriptor["inode"]
            or before.st_size != descriptor["size"]
        ):
            raise ValueError("train-bank descriptor metadata identity mismatch")
        return AuthorizedTrainTarget(
            descriptor=dict(descriptor),
            path=target,
            before=before,
            parent_descriptor=parent_descriptor,
            leaf_name=target.name,
            parent_chain=parent_chain,
        )
    except BaseException:
        parent_chain.close()
        raise


def _read_authorized_train_target(target: AuthorizedTrainTarget) -> bytes:
    descriptor: int | None = None
    try:
        target.parent_chain.revalidate(label="train-bank parent before target read")
        descriptor = os.open(
            target.leaf_name,
            _READ_FLAGS,
            dir_fd=target.parent_descriptor,
        )
        opened = os.fstat(descriptor)
        if _stat_identity(opened) != _stat_identity(target.before):
            raise RuntimeError("train-bank target changed before open")
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        observed_size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            observed_size += len(chunk)
            if observed_size > target.descriptor["size"]:
                raise ValueError("train-bank target exceeds authorized size")
            digest.update(chunk)
            chunks.append(chunk)
        after = os.fstat(descriptor)
        current = os.stat(
            target.leaf_name,
            dir_fd=target.parent_descriptor,
            follow_symlinks=False,
        )
        if (
            observed_size != target.descriptor["size"]
            or _stat_identity(after) != _stat_identity(target.before)
            or _stat_identity(current) != _stat_identity(target.before)
        ):
            raise RuntimeError("train-bank target changed during read")
        if digest.hexdigest() != target.descriptor["sha256"]:
            raise ValueError("train-bank target SHA-256 mismatch")
        target.parent_chain.revalidate(label="train-bank parent after target read")
        return b"".join(chunks)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        target.parent_chain.close()


def _parse_authorized_train_bank(payload: bytes, *, path: Path) -> LoadedTrainBank:
    if hashlib.sha256(payload).hexdigest() != EXPECTED_TRAIN_SHA256:
        raise ValueError("authorized train-bank bytes changed before parser")
    try:
        text_payload = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("train bank is not UTF-8") from error
    raw_lines = text_payload.splitlines()
    if len(raw_lines) != EXPECTED_TRAIN_EPISODES or any(not row for row in raw_lines):
        raise ValueError("train bank row count or blank-line contract changed")
    episodes: list[LongHorizonEpisode] = []
    episode_sha256: dict[str, str] = {}
    seeds: set[int] = set()
    for line_number, line in enumerate(raw_lines, start=1):
        value = a9_oof_protocol._strict_json(line, label=f"train line {line_number}")
        episode = a9_oof_protocol._validate_episode_row(value, line_number=line_number)
        if episode.episode_id in episode_sha256 or episode.seed in seeds:
            raise ValueError("duplicate train episode ID or seed")
        episode_sha256[episode.episode_id] = hashlib.sha256(
            a9_oof_protocol._canonical(value).encode("utf-8")
        ).hexdigest()
        seeds.add(episode.seed)
        episodes.append(episode)
    return LoadedTrainBank(
        path=path,
        payload_sha256=EXPECTED_TRAIN_SHA256,
        episodes=tuple(episodes),
        episode_sha256=dict(sorted(episode_sha256.items())),
    )


def _binary64(value: float) -> dict[str, Any]:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError("threshold is not a finite binary64 value")
    return {
        "value": value,
        "float_hex": value.hex(),
        "ieee754_binary64_be_hex": struct.pack(">d", value).hex(),
    }


def _lock_claim_boundary() -> dict[str, Any]:
    return {
        "q1_qualified": False,
        "formal_authorized": False,
        "outer_outcomes_authorized": False,
        "performance_evaluable": False,
        "safety_evaluable": False,
        "generalization_evaluable": False,
        "agentic_rl_success": False,
        "llm_weights_updated": False,
        "formal_absence_race_free": False,
        "formal_absence_linearizable": False,
        "outer_rows": 0,
    }


def _formal_absence_boundary() -> dict[str, Any]:
    return {
        "status": "DISCRETE_BARRIER_SNAPSHOT_ONLY",
        "race_free": False,
        "linearizable": False,
        "shared_a24_arbitration": False,
        "checkpoints": [
            "prepare-entry",
            "writer-pre-final-link",
            "writer-post-commit-fsync",
        ],
    }


def _expected_method_blockers() -> dict[str, Any]:
    return {
        "diagnostic_status": "CR_AXIS_INERT_UNDER_POST_SHIELD_COST",
        "cr_axis_natural_nonreward_covered": False,
        "observed_stage_w_cr_updates": "8/8 reward mode",
        "f00_equals_f10_observed": True,
        "f01_equals_f11_observed": True,
        "four_arm_2x2_behavioral_separation_established": False,
        "q1_method_gate_blocked": True,
    }


def _environment_contracts() -> dict[str, Any]:
    return {
        "environment": "multitown-long-horizon-pomdp-v1",
        "state": "multitown-long-horizon-public-observation-47-v1",
        "action": "multitown-long-horizon-eight-actions-v1",
        "reward": "multitown-long-horizon-reward-v1",
        "cost": "multitown-a22-unsafe-wrong-cost-v1",
        "mask": "multitown-a25-base-effective-mask-v1",
        "hard_shield": "review-pass-required-for-execute",
    }


def _stop_and_resume_policy() -> dict[str, Any]:
    return {
        "prepare_only": True,
        "training_started": False,
        "external_checkpoint_allowed": False,
        "formal_lock_created": False,
        "outer_read_allowed": False,
    }


def _locked_stage_w_schedule() -> dict[str, Any]:
    schedule = stage_w_schedule()
    return {
        "stage": schedule.stage,
        "design_fold": schedule.design_fold,
        "replica_seeds": list(schedule.replica_seeds),
        "arms": list(schedule.arms),
        "updates": schedule.updates,
        "episodes_per_update": schedule.episodes_per_update,
        "q1_qualified": False,
        "formal_authorized": False,
        "outer_rows": 0,
        "fits": schedule.fits,
        "arm_updates": schedule.arm_updates,
        "training_rollouts": schedule.training_rollouts,
        "shared_external_schedule_draws": schedule.shared_external_schedule_draws,
    }


def _freeze_stage_w_contract(
    *,
    bank: LoadedTrainBank,
    source: Mapping[str, Any],
    runtime: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
) -> dict[str, Any]:
    assignments = assign_stratified_group_folds(bank)
    assignment_by_id = {row.episode_id: row for row in assignments}
    episode_by_id = {episode.episode_id: episode for episode in bank.episodes}
    inner_train_ids = sorted(
        episode_id
        for episode_id, assignment in assignment_by_id.items()
        if assignment.fold in STAGE_W_INNER_TRAIN_FOLDS
    )
    if len(inner_train_ids) != 1800:
        raise RuntimeError("Stage W threshold-row population changed")
    selected_ids = inner_train_ids[:STAGE_W_EPISODES_PER_UPDATE]
    threshold_rows = [
        _threshold_row(episode_by_id[episode_id]) for episode_id in inner_train_ids
    ]
    thresholds = thresholds_from_inner_train(threshold_rows)
    rows_by_fold = {
        str(fold): [
            row
            for row in threshold_rows
            if assignment_by_id[row["episode_id"]].fold == fold
        ]
        for fold in STAGE_W_INNER_TRAIN_FOLDS
    }
    config = _stage_w_config()
    source_files = source["source_files_sha256"]
    arms: dict[str, dict[str, Any]] = {}
    for arm in STAGE_W_ARMS:
        identity = {
            **_arm_identity(arm, config),
            "source_files_sha256": source_files,
        }
        arms[arm] = {
            "identity": identity,
            "arm_identity_sha256": canonical_sha256(identity),
        }
    target_descriptor = bootstrap["target"]
    core = {
        "schema_version": STAGE_W_LOCK_VERSION,
        "protocol_version": STAGE_W_LOCK_PROTOCOL_VERSION,
        "status": STAGE_W_LOCK_STATUS,
        "stage": "W",
        "evidence_class": "development-smoke-only",
        "developer_debug": False,
        "q1_qualified": False,
        "formal_authorized": False,
        "outer_outcomes_authorized": False,
        "outer_rows": 0,
        "trust_boundary_status": "DISCRETE_BARRIER_SNAPSHOT_ONLY",
        "formal_absence_boundary": _formal_absence_boundary(),
        "source": source,
        "runtime": runtime,
        "runtime_sha256": canonical_sha256(runtime),
        "bootstrap_descriptor": {
            "schema_version": BOOTSTRAP_DESCRIPTOR_VERSION,
            "file_sha256": bootstrap["file_sha256"],
            "target": target_descriptor,
        },
        "allowed_input_roles": [TRAIN_BANK_ROLE],
        "forbidden_input_roles": list(_FORBIDDEN_INPUT_ROLES),
        "input_manifest_sha256": target_descriptor["sha256"],
        "fold_manifest_sha256": fold_manifest_sha256(assignments),
        "population": {
            "role": "train-only-development-smoke",
            "full_train_bank_parsed_for_fold_assignment": True,
            "design_fold": STAGE_W_DESIGN_FOLD,
            "inner_calibration_fold": STAGE_W_INNER_CALIBRATION_FOLD,
            "inner_calibration_used_for_thresholds": False,
            "inner_calibration_used_for_learned_rollout": False,
            "inner_train_folds": list(STAGE_W_INNER_TRAIN_FOLDS),
            "episode_ids": selected_ids,
            "episode_sha256": {
                episode_id: bank.episode_sha256[episode_id]
                for episode_id in selected_ids
            },
            "episode_schedule_sha256": canonical_sha256(selected_ids),
        },
        "threshold_rows": {
            "role": "train-only-a8-public-heuristic-preprocessing",
            "row_count": len(threshold_rows),
            "row_ids": inner_train_ids,
            "rows_sha256": canonical_sha256(threshold_rows),
            "fold_rows_sha256": {
                fold: canonical_sha256(rows) for fold, rows in rows_by_fold.items()
            },
        },
        "thresholds": {
            "unsafe": _binary64(thresholds.unsafe),
            "wrong_per_incident": _binary64(thresholds.wrong_per_incident),
            "mean_incidents": _binary64(thresholds.mean_incidents),
        },
        "contracts": _environment_contracts(),
        "config": asdict(config),
        "config_sha256": canonical_sha256(asdict(config)),
        "arms": arms,
        "schedule": _locked_stage_w_schedule(),
        "seed_derivation": {
            "version": "multitown-a25-q1-rng-v1",
            "domain_hex": _SEED_DOMAIN.hex(),
            "digest": "sha256",
            "byte_order": "little",
            "truncation_bytes": 8,
            "sign_mask_bits": 63,
            "replica_seeds": list(STAGE_W_SEEDS),
            "roles": ["model_init", "episode_schedule", "rollout", "tensor_update"],
            "design_fold": STAGE_W_DESIGN_FOLD,
        },
        "ordered_gates": list(_ORDERED_STAGE_W_GATES),
        "stop_and_resume_policy": _stop_and_resume_policy(),
        "method_blockers": _expected_method_blockers(),
        "claim_boundary": _lock_claim_boundary(),
    }
    return {
        **core,
        "protocol_lock_sha256": canonical_sha256(core),
    }


def _assert_formal_lock_absent(
    root: Path,
    *,
    repository_chain: PinnedPathChain | None = None,
) -> None:
    formal_path = Path(FORMAL_LOCK)
    if formal_path.is_absolute() or any(
        component in {"", ".", ".."} for component in formal_path.parts
    ):
        raise RuntimeError("Stage W formal-lock contract path is unsafe")
    repository = _caller_absolute_path(root, label="formal-lock repository root")
    owns_repository_chain = repository_chain is None
    chain = repository_chain or _pin_directory_chain(
        repository,
        label="formal-lock repository root",
    )
    if chain.path != repository:
        if owns_repository_chain:
            chain.close()
        raise RuntimeError("formal-lock repository chain path mismatch")
    relative_descriptors: list[int] = []
    relative_identities: list[tuple[int, int, int]] = []

    def revalidate_formal_parent() -> None:
        chain.revalidate(label="formal-lock repository chain")
        parent_descriptor = chain.leaf_descriptor
        for component, descriptor, expected in zip(
            formal_path.parts[: len(relative_descriptors)],
            relative_descriptors,
            relative_identities,
            strict=True,
        ):
            visible = os.stat(
                component,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if (
                _path_entry_identity(os.fstat(descriptor)) != expected
                or _path_entry_identity(visible) != expected
            ):
                raise RuntimeError("Stage W formal-lock parent chain changed")
            parent_descriptor = descriptor

    try:
        chain.revalidate(label="formal-lock repository before check")
        descriptor = chain.leaf_descriptor
        for component in formal_path.parts[:-1]:
            try:
                visible = os.stat(
                    component,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
                next_descriptor = os.open(
                    component,
                    _DIRECTORY_FLAGS,
                    dir_fd=descriptor,
                )
            except FileNotFoundError:
                revalidate_formal_parent()
                return
            except OSError as error:
                raise RuntimeError("Stage W formal-lock parent is unsafe") from error
            opened = os.fstat(next_descriptor)
            if not stat.S_ISDIR(visible.st_mode) or _path_entry_identity(
                opened
            ) != _path_entry_identity(visible):
                os.close(next_descriptor)
                raise RuntimeError("Stage W formal-lock parent changed during open")
            relative_descriptors.append(next_descriptor)
            relative_identities.append(_path_entry_identity(opened))
            descriptor = next_descriptor
        try:
            os.stat(
                formal_path.name,
                dir_fd=descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            revalidate_formal_parent()
            return
        revalidate_formal_parent()
        raise RuntimeError("Stage W prepare refuses an existing A24 formal lock")
    finally:
        for descriptor in reversed(relative_descriptors):
            os.close(descriptor)
        if owns_repository_chain:
            chain.close()


def _validate_stage_w_lock(lock: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "protocol_version",
        "status",
        "stage",
        "evidence_class",
        "developer_debug",
        "q1_qualified",
        "formal_authorized",
        "outer_outcomes_authorized",
        "outer_rows",
        "trust_boundary_status",
        "formal_absence_boundary",
        "source",
        "runtime",
        "runtime_sha256",
        "bootstrap_descriptor",
        "allowed_input_roles",
        "forbidden_input_roles",
        "input_manifest_sha256",
        "fold_manifest_sha256",
        "population",
        "threshold_rows",
        "thresholds",
        "contracts",
        "config",
        "config_sha256",
        "arms",
        "schedule",
        "seed_derivation",
        "ordered_gates",
        "stop_and_resume_policy",
        "method_blockers",
        "claim_boundary",
        "protocol_lock_sha256",
    }
    if type(lock) is not dict or set(lock) != required:
        raise ValueError("invalid Stage W protocol lock schema")
    if (
        lock["schema_version"] != STAGE_W_LOCK_VERSION
        or lock["protocol_version"] != STAGE_W_LOCK_PROTOCOL_VERSION
        or lock["status"] != STAGE_W_LOCK_STATUS
        or lock["stage"] != "W"
        or lock["evidence_class"] != "development-smoke-only"
        or lock["developer_debug"] is not False
        or lock["q1_qualified"] is not False
        or lock["formal_authorized"] is not False
        or lock["outer_outcomes_authorized"] is not False
        or type(lock["outer_rows"]) is not int
        or lock["outer_rows"] != 0
        or lock["trust_boundary_status"] != "DISCRETE_BARRIER_SNAPSHOT_ONLY"
        or not _exact_json_equal(
            lock["formal_absence_boundary"], _formal_absence_boundary()
        )
        or not _exact_json_equal(lock["claim_boundary"], _lock_claim_boundary())
        or type(lock["claim_boundary"].get("outer_rows")) is not int
    ):
        raise ValueError("Stage W protocol lock claim boundary violation")
    core = {key: value for key, value in lock.items() if key != "protocol_lock_sha256"}
    if lock["protocol_lock_sha256"] != canonical_sha256(core):
        raise ValueError("Stage W protocol lock SHA-256 mismatch")

    source = lock["source"]
    source_keys = {
        "schema_version",
        "revision",
        "tree",
        "clean",
        "import_trust_root",
        "source_files_sha256",
        "source_bundle_sha256",
        "module_origins",
        "git",
    }
    if (
        type(source) is not dict
        or set(source) != source_keys
        or source["schema_version"] != STAGE_W_SOURCE_VERSION
        or source["clean"] is not True
        or type(source["revision"]) is not str
        or type(source["tree"]) is not str
        or len(source["revision"]) != 40
        or len(source["tree"]) != 40
        or not all(
            character in "0123456789abcdef"
            for character in source["revision"] + source["tree"]
        )
        or type(source["source_files_sha256"]) is not dict
        or not source["source_files_sha256"]
        or not all(
            type(path) is str and path and _valid_sha256(digest)
            for path, digest in source["source_files_sha256"].items()
        )
        or source["source_bundle_sha256"]
        != canonical_sha256(source["source_files_sha256"])
        or type(source["module_origins"]) is not dict
        or type(source["git"]) is not dict
    ):
        raise ValueError("invalid Stage W source binding")
    import_root = source["import_trust_root"]
    if (
        type(import_root) is not dict
        or set(import_root)
        != {
            "status",
            "repository",
            "q1_source",
            "full_directory_chain_bound",
            "trusted_launcher",
            "memory_bytecode_attested",
        }
        or import_root["status"] != "EARLY_IMPORT_PATH_AND_BYTES_BOUND"
        or import_root["full_directory_chain_bound"] is not True
        or import_root["trusted_launcher"] is not False
        or import_root["memory_bytecode_attested"] is not False
    ):
        raise ValueError("invalid Stage W import trust-root binding")
    import_repository = import_root["repository"]
    import_source = import_root["q1_source"]
    if (
        type(import_repository) is not dict
        or set(import_repository) != {"absolute_path", "device", "inode", "file_type"}
        or type(import_source) is not dict
        or set(import_source)
        != {"absolute_path", "device", "inode", "file_type", "bytes", "sha256"}
    ):
        raise ValueError("invalid Stage W import trust-root identity")
    _absolute_canonical_path(
        import_repository["absolute_path"], label="import repository root"
    )
    _absolute_canonical_path(import_source["absolute_path"], label="import Q1 source")
    if (
        import_repository["file_type"] != stat.S_IFDIR
        or import_source["file_type"] != stat.S_IFREG
        or Path(import_source["absolute_path"])
        != Path(import_repository["absolute_path"])
        / _REQUIRED_PRIMITIVE_MODULES["multitown.a25_q1_runner"]
    ):
        raise ValueError("invalid Stage W import trust-root path binding")
    for label, value, minimum in (
        ("import repository device", import_repository["device"], 0),
        ("import repository inode", import_repository["inode"], 1),
        ("import repository type", import_repository["file_type"], 1),
        ("import source device", import_source["device"], 0),
        ("import source inode", import_source["inode"], 1),
        ("import source type", import_source["file_type"], 1),
        ("import source bytes", import_source["bytes"], 1),
    ):
        _exact_integer(value, label=label, minimum=minimum)
    if not _valid_sha256(import_source["sha256"]):
        raise ValueError("invalid Stage W import source SHA-256")
    origins = source["module_origins"]
    if set(origins) != set(_REQUIRED_PRIMITIVE_MODULES):
        raise ValueError("incomplete Stage W primitive origin binding")
    observed_source_files: dict[str, str] = {}
    for module_name, expected_path in _REQUIRED_PRIMITIVE_MODULES.items():
        row = origins[module_name]
        if (
            type(row) is not dict
            or set(row) != {"path", "sha256", "bytes"}
            or row["path"] != expected_path
            or not _valid_sha256(row["sha256"])
            or type(row["bytes"]) is not int
            or row["bytes"] <= 0
        ):
            raise ValueError("invalid Stage W primitive origin binding")
        observed_source_files[row["path"]] = row["sha256"]
    if source["source_files_sha256"] != dict(sorted(observed_source_files.items())):
        raise ValueError("Stage W source-file digest mapping mismatch")
    git = source["git"]
    if (
        set(git) != {"executable", "executable_sha256", "version"}
        or type(git["executable"]) is not str
        or type(git["version"]) is not str
        or not git["version"]
        or not _valid_sha256(git["executable_sha256"])
    ):
        raise ValueError("invalid Stage W Git runtime binding")

    runtime = lock["runtime"]
    _validate_runtime_binding(runtime)
    if lock["runtime_sha256"] != canonical_sha256(runtime):
        raise ValueError("invalid Stage W runtime binding")

    bootstrap = lock["bootstrap_descriptor"]
    target = bootstrap.get("target") if type(bootstrap) is dict else None
    expected_locked_target = _caller_absolute_path(
        Path(FROZEN_TRAIN_PATH),
        label="locked frozen train-bank contract path",
    )
    target_keys = {
        "role",
        "absolute_path",
        "media_type",
        "device",
        "inode",
        "size",
        "sha256",
        "allowed_root",
    }
    if (
        type(bootstrap) is not dict
        or set(bootstrap) != {"schema_version", "file_sha256", "target"}
        or bootstrap["schema_version"] != BOOTSTRAP_DESCRIPTOR_VERSION
        or not _valid_sha256(bootstrap["file_sha256"])
        or type(target) is not dict
        or set(target) != target_keys
        or target["role"] != TRAIN_BANK_ROLE
        or target["media_type"] != TRAIN_BANK_MEDIA_TYPE
        or target["absolute_path"] != str(expected_locked_target)
        or target["allowed_root"] != str(expected_locked_target.parent)
        or not _valid_sha256(target["sha256"])
        or target["sha256"] != EXPECTED_TRAIN_SHA256
        or lock["input_manifest_sha256"] != target["sha256"]
        or lock["allowed_input_roles"] != [TRAIN_BANK_ROLE]
        or lock["forbidden_input_roles"] != list(_FORBIDDEN_INPUT_ROLES)
        or not _valid_sha256(lock["fold_manifest_sha256"])
    ):
        raise ValueError("invalid Stage W locked input descriptor")
    _validate_target_numeric_fields(target)

    population = lock["population"]
    episode_ids = population.get("episode_ids") if type(population) is dict else None
    episode_sha256 = (
        population.get("episode_sha256") if type(population) is dict else None
    )
    if (
        type(population) is not dict
        or set(population)
        != {
            "role",
            "full_train_bank_parsed_for_fold_assignment",
            "design_fold",
            "inner_calibration_fold",
            "inner_calibration_used_for_thresholds",
            "inner_calibration_used_for_learned_rollout",
            "inner_train_folds",
            "episode_ids",
            "episode_sha256",
            "episode_schedule_sha256",
        }
        or population["role"] != "train-only-development-smoke"
        or population["full_train_bank_parsed_for_fold_assignment"] is not True
        or type(population["design_fold"]) is not int
        or population["design_fold"] != STAGE_W_DESIGN_FOLD
        or type(population["inner_calibration_fold"]) is not int
        or population["inner_calibration_fold"] != STAGE_W_INNER_CALIBRATION_FOLD
        or population["inner_calibration_used_for_thresholds"] is not False
        or population["inner_calibration_used_for_learned_rollout"] is not False
        or not _exact_json_equal(
            population["inner_train_folds"], list(STAGE_W_INNER_TRAIN_FOLDS)
        )
        or type(episode_ids) is not list
        or len(episode_ids) != STAGE_W_EPISODES_PER_UPDATE
        or episode_ids != sorted(set(episode_ids))
        or type(episode_sha256) is not dict
        or set(episode_sha256) != set(episode_ids)
        or not all(_valid_sha256(value) for value in episode_sha256.values())
        or population["episode_schedule_sha256"] != canonical_sha256(episode_ids)
    ):
        raise ValueError("invalid Stage W locked population")

    threshold_rows = lock["threshold_rows"]
    if (
        type(threshold_rows) is not dict
        or set(threshold_rows)
        != {"role", "row_count", "row_ids", "rows_sha256", "fold_rows_sha256"}
        or threshold_rows["role"] != "train-only-a8-public-heuristic-preprocessing"
        or type(threshold_rows["row_count"]) is not int
        or threshold_rows["row_count"] != 1800
        or type(threshold_rows["row_ids"]) is not list
        or len(threshold_rows["row_ids"]) != 1800
        or threshold_rows["row_ids"] != sorted(set(threshold_rows["row_ids"]))
        or not all(type(value) is str and value for value in threshold_rows["row_ids"])
        or not _valid_sha256(threshold_rows["rows_sha256"])
        or type(threshold_rows["fold_rows_sha256"]) is not dict
        or set(threshold_rows["fold_rows_sha256"])
        != {str(value) for value in STAGE_W_INNER_TRAIN_FOLDS}
        or not all(
            _valid_sha256(value)
            for value in threshold_rows["fold_rows_sha256"].values()
        )
    ):
        raise ValueError("invalid Stage W threshold-row binding")
    thresholds = lock["thresholds"]
    if type(thresholds) is not dict or set(thresholds) != {
        "unsafe",
        "wrong_per_incident",
        "mean_incidents",
    }:
        raise ValueError("invalid Stage W threshold binding")
    for name, value in thresholds.items():
        if (
            type(value) is not dict
            or set(value) != {"value", "float_hex", "ieee754_binary64_be_hex"}
            or type(value["value"]) is not float
            or not math.isfinite(value["value"])
            or value != _binary64(value["value"])
            or (name != "mean_incidents" and not 0.0 <= value["value"] <= 1.0)
            or (name == "mean_incidents" and value["value"] <= 0.0)
        ):
            raise ValueError("invalid Stage W binary64 threshold binding")

    config = asdict(_stage_w_config())
    if not _exact_json_equal(lock["config"], config) or lock[
        "config_sha256"
    ] != canonical_sha256(config):
        raise ValueError("invalid Stage W locked PPO config")
    if (
        not _exact_json_equal(lock["schedule"], _locked_stage_w_schedule())
        or type(lock["schedule"].get("outer_rows")) is not int
    ):
        raise ValueError("invalid Stage W locked schedule")
    if type(lock["arms"]) is not dict or tuple(lock["arms"]) != STAGE_W_ARMS:
        raise ValueError("invalid Stage W locked arm product")
    for arm in STAGE_W_ARMS:
        expected_identity = {
            **_arm_identity(arm, _stage_w_config()),
            "source_files_sha256": source["source_files_sha256"],
        }
        if not _exact_json_equal(
            lock["arms"][arm],
            {
                "identity": expected_identity,
                "arm_identity_sha256": canonical_sha256(expected_identity),
            },
        ):
            raise ValueError("invalid Stage W locked arm identity")
    if not _exact_json_equal(lock["contracts"], _environment_contracts()):
        raise ValueError("invalid Stage W locked environment contracts")
    if not _exact_json_equal(lock["ordered_gates"], list(_ORDERED_STAGE_W_GATES)):
        raise ValueError("invalid Stage W ordered gates")
    if not _exact_json_equal(lock["stop_and_resume_policy"], _stop_and_resume_policy()):
        raise ValueError("invalid Stage W stop/resume policy")
    if not _exact_json_equal(lock["method_blockers"], _expected_method_blockers()):
        raise ValueError("invalid Stage W method-blocker boundary")
    seed = lock["seed_derivation"]
    if not _exact_json_equal(
        seed,
        {
            "version": "multitown-a25-q1-rng-v1",
            "domain_hex": _SEED_DOMAIN.hex(),
            "digest": "sha256",
            "byte_order": "little",
            "truncation_bytes": 8,
            "sign_mask_bits": 63,
            "replica_seeds": list(STAGE_W_SEEDS),
            "roles": ["model_init", "episode_schedule", "rollout", "tensor_update"],
            "design_fold": STAGE_W_DESIGN_FOLD,
        },
    ):
        raise ValueError("invalid Stage W seed namespace")


def _write_protocol_lock(
    root: Path,
    lock_out: Path,
    lock: Mapping[str, Any],
    *,
    repository_chain: PinnedPathChain | None = None,
) -> None:
    repository = _caller_absolute_path(root, label="protocol lock repository")
    path = _caller_absolute_path(lock_out, label="protocol lock output")
    if path.name != PROTOCOL_LOCK_BASENAME:
        raise ValueError("protocol lock output basename is invalid")
    if any(part.casefold() in _FORBIDDEN_OUTPUT_COMPONENTS for part in path.parts):
        raise PermissionError("protocol lock output contains a forbidden role")
    output_directory = path.parent
    if output_directory == repository or output_directory.is_relative_to(repository):
        raise ValueError("protocol lock output directory must be outside repository")
    external_parent = _pin_existing_directory(
        output_directory.parent,
        label="protocol lock external parent",
    )
    owns_repository_chain = repository_chain is None
    fixed_repository_chain: PinnedPathChain | None = None
    try:
        fixed_repository_chain = repository_chain or _pin_directory_chain(
            repository,
            label="protocol lock repository",
        )
        if fixed_repository_chain.path != repository:
            raise RuntimeError("protocol lock repository chain path mismatch")
        fixed_repository_chain.revalidate(label="protocol lock repository before write")
        repository_descriptor = fixed_repository_chain.leaf_descriptor
        repository_metadata = os.fstat(repository_descriptor)
    except BaseException:
        if owns_repository_chain and fixed_repository_chain is not None:
            fixed_repository_chain.close()
        _close_pinned_directory(external_parent)
        raise
    output_name = output_directory.name
    created_directory = False
    output_descriptor: int | None = None
    output_identity: tuple[int, int] | None = None
    file_descriptor: int | None = None
    file_identity: tuple[int, int] | None = None
    temporary_created = False
    final_created = False
    temporary_name = ".protocol.lock.json.tmp"
    payload = _canonical_bytes(lock)

    def output_entry_matches() -> bool:
        if output_identity is None:
            return False
        try:
            current = os.stat(
                output_name,
                dir_fd=external_parent.descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return False
        return (current.st_dev, current.st_ino) == output_identity

    def unlink_owned(name: str) -> bool:
        if file_identity is None:
            return False
        try:
            current = os.stat(
                name,
                dir_fd=output_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return False
        if (
            not stat.S_ISREG(current.st_mode)
            or (current.st_dev, current.st_ino) != file_identity
        ):
            return False
        os.unlink(name, dir_fd=output_descriptor)
        return True

    try:
        if not _pinned_directory_entry_unchanged(external_parent):
            raise RuntimeError("protocol lock external parent changed before mkdir")
        if _directory_has_ancestor(
            external_parent.descriptor,
            ancestor_device=repository_metadata.st_dev,
            ancestor_inode=repository_metadata.st_ino,
        ):
            raise ValueError("protocol lock external parent is inside repository")
        try:
            os.stat(
                output_name,
                dir_fd=external_parent.descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError("protocol lock output directory must be new")
        os.mkdir(output_name, 0o700, dir_fd=external_parent.descriptor)
        created_directory = True
        metadata = os.stat(
            output_name,
            dir_fd=external_parent.descriptor,
            follow_symlinks=False,
        )
        output_identity = (metadata.st_dev, metadata.st_ino)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o700
            or metadata.st_uid != os.geteuid()
        ):
            raise RuntimeError("protocol lock directory identity is unsafe")
        os.fsync(external_parent.descriptor)
        output_descriptor = os.open(
            output_name,
            _DIRECTORY_FLAGS,
            dir_fd=external_parent.descriptor,
        )
        opened_directory = os.fstat(output_descriptor)
        if (opened_directory.st_dev, opened_directory.st_ino) != (
            metadata.st_dev,
            metadata.st_ino,
        ):
            raise RuntimeError("protocol lock directory changed during open")
        if _directory_has_ancestor(
            output_descriptor,
            ancestor_device=repository_metadata.st_dev,
            ancestor_inode=repository_metadata.st_ino,
        ):
            raise RuntimeError("protocol lock output was redirected into repository")
        file_descriptor = os.open(
            temporary_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=output_descriptor,
        )
        temporary_created = True
        written = 0
        while written < len(payload):
            count = os.write(file_descriptor, payload[written:])
            if count <= 0:
                raise OSError("short protocol lock write")
            written += count
        opened_file = os.fstat(file_descriptor)
        if (
            not stat.S_ISREG(opened_file.st_mode)
            or stat.S_IMODE(opened_file.st_mode) != 0o600
            or opened_file.st_size != len(payload)
            or opened_file.st_nlink != 1
        ):
            raise RuntimeError("protocol lock temporary file is unsafe")
        file_identity = (opened_file.st_dev, opened_file.st_ino)
        os.fsync(file_descriptor)
        if (
            not _pinned_directory_entry_unchanged(external_parent)
            or not output_entry_matches()
            or _directory_has_ancestor(
                output_descriptor,
                ancestor_device=repository_metadata.st_dev,
                ancestor_inode=repository_metadata.st_ino,
            )
        ):
            raise RuntimeError("protocol lock output changed before commit")
        try:
            os.stat(
                PROTOCOL_LOCK_BASENAME,
                dir_fd=output_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError("protocol lock final file already exists")
        _assert_formal_lock_absent(
            repository,
            repository_chain=fixed_repository_chain,
        )
        os.link(
            temporary_name,
            PROTOCOL_LOCK_BASENAME,
            src_dir_fd=output_descriptor,
            dst_dir_fd=output_descriptor,
            follow_symlinks=False,
        )
        final_created = True
        linked = os.stat(
            PROTOCOL_LOCK_BASENAME,
            dir_fd=output_descriptor,
            follow_symlinks=False,
        )
        if (
            (linked.st_dev, linked.st_ino) != file_identity
            or linked.st_nlink != 2
            or linked.st_size != len(payload)
        ):
            raise RuntimeError("protocol lock no-clobber link identity mismatch")
        os.fsync(output_descriptor)
        if not unlink_owned(temporary_name):
            raise RuntimeError("protocol lock temporary cleanup identity mismatch")
        temporary_created = False
        final = os.stat(
            PROTOCOL_LOCK_BASENAME,
            dir_fd=output_descriptor,
            follow_symlinks=False,
        )
        if (
            (final.st_dev, final.st_ino) != file_identity
            or final.st_nlink != 1
            or stat.S_IMODE(final.st_mode) != 0o600
        ):
            raise RuntimeError("protocol lock final identity mismatch")
        os.fsync(output_descriptor)
        os.fsync(external_parent.descriptor)
        if (
            not _pinned_directory_entry_unchanged(external_parent)
            or not output_entry_matches()
            or _directory_has_ancestor(
                output_descriptor,
                ancestor_device=repository_metadata.st_dev,
                ancestor_inode=repository_metadata.st_ino,
            )
        ):
            raise RuntimeError("protocol lock output changed after commit")
        _assert_formal_lock_absent(
            repository,
            repository_chain=fixed_repository_chain,
        )
        nominal_output = _pin_existing_directory(
            output_directory,
            label="protocol lock nominal output directory",
        )
        try:
            if (
                os.fstat(nominal_output.descriptor).st_dev,
                os.fstat(nominal_output.descriptor).st_ino,
            ) != output_identity:
                raise RuntimeError("protocol lock nominal output directory changed")
            nominal_final_descriptor = os.open(
                PROTOCOL_LOCK_BASENAME,
                _READ_FLAGS,
                dir_fd=nominal_output.descriptor,
            )
            try:
                nominal_final = os.fstat(nominal_final_descriptor)
                if (
                    not stat.S_ISREG(nominal_final.st_mode)
                    or (nominal_final.st_dev, nominal_final.st_ino) != file_identity
                    or nominal_final.st_size != len(payload)
                    or stat.S_IMODE(nominal_final.st_mode) != 0o600
                ):
                    raise RuntimeError("protocol lock nominal final identity mismatch")
                nominal_output.chain.revalidate(
                    label="protocol lock nominal output after reopen"
                )
            finally:
                os.close(nominal_final_descriptor)
        finally:
            _close_pinned_directory(nominal_output)
    except BaseException:
        if file_descriptor is not None:
            os.close(file_descriptor)
            file_descriptor = None
        if output_descriptor is not None:
            if temporary_created:
                unlink_owned(temporary_name)
            if final_created:
                unlink_owned(PROTOCOL_LOCK_BASENAME)
            os.fsync(output_descriptor)
        if created_directory and output_entry_matches():
            try:
                os.rmdir(output_name, dir_fd=external_parent.descriptor)
            except OSError:
                # A concurrently inserted, non-owned entry must never be deleted.
                pass
            os.fsync(external_parent.descriptor)
        raise
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        if output_descriptor is not None:
            os.close(output_descriptor)
        if owns_repository_chain:
            fixed_repository_chain.close()
        _close_pinned_directory(external_parent)


def prepare_stage_w(
    *, root: Path, input_descriptors: Path, lock_out: Path
) -> dict[str, Any]:
    """Freeze the Stage W train-only trust root without starting PPO."""

    repository = _repository_root(root)
    repository_chain = _pin_directory_chain(
        repository,
        label="Stage W fixed repository root",
    )
    try:
        _assert_formal_lock_absent(
            repository,
            repository_chain=repository_chain,
        )
        source_before = _source_state(repository)
        runtime_before = _runtime_profile()
        bootstrap = _read_bootstrap_descriptors(input_descriptors)
        target = _authorize_train_bank_descriptor(bootstrap["target"])
        payload = _read_authorized_train_target(target)
        bank = _parse_authorized_train_bank(payload, path=target.path)
        lock = _freeze_stage_w_contract(
            bank=bank,
            source=source_before,
            runtime=runtime_before,
            bootstrap=bootstrap,
        )
        _validate_stage_w_lock(lock)
        source_after = _source_state(repository)
        runtime_after = _runtime_profile()
        if source_after != source_before:
            raise RuntimeError("Stage W source changed during prepare")
        if runtime_after != runtime_before:
            raise RuntimeError("Stage W runtime changed during prepare")
        _assert_formal_lock_absent(
            repository,
            repository_chain=repository_chain,
        )
        _write_protocol_lock(
            repository,
            lock_out,
            lock,
            repository_chain=repository_chain,
        )
        repository_chain.revalidate(label="Stage W repository after prepare")
        return lock
    finally:
        repository_chain.close()


def _lock_metadata(lock: Mapping[str, Any], *, status: str) -> dict[str, Any]:
    return {
        "status": status,
        "stage": "W",
        "trust_boundary_status": lock["trust_boundary_status"],
        "formal_absence_boundary": lock["formal_absence_boundary"],
        "expected_counts": lock["schedule"],
        "digests": {
            "protocol_lock_sha256": lock["protocol_lock_sha256"],
            "source_bundle_sha256": lock["source"]["source_bundle_sha256"],
            "runtime_sha256": lock["runtime_sha256"],
            "input_manifest_sha256": lock["input_manifest_sha256"],
            "fold_manifest_sha256": lock["fold_manifest_sha256"],
            "episode_schedule_sha256": lock["population"]["episode_schedule_sha256"],
            "threshold_rows_sha256": lock["threshold_rows"]["rows_sha256"],
        },
        "method_blockers": lock["method_blockers"],
        "claim_boundary": lock["claim_boundary"],
    }


def inspect_stage_w_lock(lock_path: Path) -> dict[str, Any]:
    """Inspect lock metadata without statting or opening its episode target."""

    path = _caller_absolute_path(lock_path, label="protocol lock")
    payload = _secure_read_file(
        path,
        label="protocol lock",
        maximum_bytes=MAX_PROTOCOL_LOCK_BYTES,
        required_mode=0o600,
        required_parent_mode=0o700,
    )
    lock = _strict_json_bytes(payload, label="protocol lock")
    if payload != _canonical_bytes(lock):
        raise ValueError("protocol lock is not exact canonical JSON")
    _validate_stage_w_lock(lock)
    return _lock_metadata(lock, status="STAGE_W_LOCK_INSPECTION")


def _derive_seed(*, replica_index: int, replica_seed: int, role: str) -> int:
    if (
        type(replica_index) is not int
        or replica_index not in range(len(STAGE_W_SEEDS))
        or type(replica_seed) is not int
        or replica_seed != STAGE_W_SEEDS[replica_index]
        or role not in {"model_init", "episode_schedule", "rollout", "tensor_update"}
    ):
        raise ValueError("invalid Stage W seed namespace")
    payload = {
        "stage": "W",
        "protocol_version": STAGE_W_PROTOCOL_VERSION,
        "design_fold": STAGE_W_DESIGN_FOLD,
        "replica_index": replica_index,
        "replica_seed": replica_seed,
        "role": role,
    }
    return int.from_bytes(
        hashlib.sha256(_SEED_DOMAIN + _canonical_bytes(payload)).digest()[:8],
        "little",
    ) & ((1 << 63) - 1)


def _stage_w_config() -> PPOConfig:
    return PPOConfig(
        updates=STAGE_W_UPDATES,
        episodes_per_update=STAGE_W_EPISODES_PER_UPDATE,
        hidden_size=STAGE_W_HIDDEN_SIZE,
        learning_rate=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        ppo_epochs=1,
        minibatch_size=STAGE_W_MINIBATCH_SIZE,
        value_coef=0.5,
        entropy_coef=0.02,
        max_grad_norm=0.5,
        dev_interval=0,
    )


def _new_model(seed: int) -> ActorCritic:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return ActorCritic(
            MultiTownLongHorizonEnv.observation_size,
            STAGE_W_HIDDEN_SIZE,
            ACTION_COUNT,
        )


def _arm_definition(arm: str) -> tuple[UpdateRule, float]:
    definitions: dict[str, tuple[UpdateRule, float]] = {
        "F00": ("frozen-zero-dual", 0.0),
        "F01": ("frozen-zero-dual", STAGE_W_POSITIVE_BETA),
        "F10": ("cr-ppo-inspired", 0.0),
        "F11": ("cr-ppo-inspired", STAGE_W_POSITIVE_BETA),
    }
    try:
        return definitions[arm]
    except KeyError as error:
        raise ValueError("invalid Stage W arm") from error


def _arm_identity(arm: str, config: PPOConfig) -> dict[str, Any]:
    update_rule, beta = _arm_definition(arm)
    return {
        "arm_id": arm,
        "update_rule": update_rule,
        "update_rule_version": (
            "multitown-a22-frozen-zero-dual-batch-v1"
            if update_rule == "frozen-zero-dual"
            else "multitown-a23-cr-ppo-selected-batch-v1"
        ),
        "intervention_beta": beta,
        "dual_policy": (
            "fixed-zero-no-update"
            if update_rule == "frozen-zero-dual"
            else "not-applicable"
        ),
        "shield_policy": "hard-review-shield-on",
        "state_version": "multitown-long-horizon-public-observation-47-v1",
        "action_version": "multitown-long-horizon-eight-actions-v1",
        "reward_version": "multitown-long-horizon-reward-v1",
        "cost_version": "multitown-a22-unsafe-wrong-cost-v1",
        "mask_version": "multitown-a25-base-effective-mask-v1",
        "ppo_config_sha256": canonical_sha256(asdict(config)),
        "protocol_version": STAGE_W_PROTOCOL_VERSION,
    }


def initialize_replica(*, replica_index: int, replica_seed: int) -> dict[str, ArmState]:
    model_seed = _derive_seed(
        replica_index=replica_index,
        replica_seed=replica_seed,
        role="model_init",
    )
    rollout_seed = _derive_seed(
        replica_index=replica_index,
        replica_seed=replica_seed,
        role="rollout",
    )
    tensor_seed = _derive_seed(
        replica_index=replica_index,
        replica_seed=replica_seed,
        role="tensor_update",
    )
    source_model = _new_model(model_seed)
    config = _stage_w_config()
    source_optimizer = torch.optim.Adam(
        source_model.parameters(), lr=config.learning_rate, eps=1e-5
    )
    states: dict[str, ArmState] = {}
    for arm in STAGE_W_ARMS:
        model, optimizer = copy.deepcopy((source_model, source_optimizer))
        update_rule, beta = _arm_definition(arm)
        states[arm] = ArmState(
            arm=arm,
            update_rule=update_rule,
            beta=beta,
            model=model,
            optimizer=optimizer,
            rollout_generator=torch.Generator(device="cpu").manual_seed(rollout_seed),
            tensor_generator=torch.Generator(device="cpu").manual_seed(tensor_seed),
        )
    identities = [state.identity for state in states.values()]
    if len(set(identities)) != 1:
        raise RuntimeError("Stage W arms do not share a common initial state")
    return states


def capture_arm_snapshot(state: ArmState) -> ArmSnapshot:
    if not isinstance(state, ArmState):
        raise TypeError("invalid Stage W arm state")
    state.optimizer.zero_grad(set_to_none=True)
    return ArmSnapshot(
        arm=state.arm,
        update_rule=state.update_rule,
        beta=state.beta,
        model_state=copy.deepcopy(state.model.state_dict()),
        optimizer_state=copy.deepcopy(state.optimizer.state_dict()),
        rollout_generator_state=state.rollout_generator.get_state().clone(),
        tensor_generator_state=state.tensor_generator.get_state().clone(),
        identity=state.identity,
    )


def restore_arm_snapshot(snapshot: ArmSnapshot) -> ArmState:
    if not isinstance(snapshot, ArmSnapshot):
        raise TypeError("invalid Stage W arm snapshot")
    model = _new_model(0)
    model.load_state_dict(snapshot.model_state, strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-5)
    optimizer.load_state_dict(snapshot.optimizer_state)
    rollout_generator = torch.Generator(device="cpu")
    rollout_generator.set_state(snapshot.rollout_generator_state.clone())
    tensor_generator = torch.Generator(device="cpu")
    tensor_generator.set_state(snapshot.tensor_generator_state.clone())
    restored = ArmState(
        arm=snapshot.arm,
        update_rule=snapshot.update_rule,
        beta=snapshot.beta,
        model=model,
        optimizer=optimizer,
        rollout_generator=rollout_generator,
        tensor_generator=tensor_generator,
    )
    if restored.identity != snapshot.identity:
        raise ValueError("Stage W exact continuation restore failed")
    return restored


def _threshold_row(episode: LongHorizonEpisode) -> dict[str, Any]:
    row = run_policy(episode, a8_heuristic_policy)
    wrong_executions = sum(
        int(float(transition["reward"]["safety_penalty"]) < 0.0)
        for transition in row["trajectory"]
    )
    return {
        "episode_id": episode.episode_id,
        "incidents": int(row["incidents"]),
        "had_wrong_execution": bool(wrong_executions),
        "wrong_executions": wrong_executions,
    }


def load_stage_w_population() -> StageWPopulation:
    bank = load_frozen_train_bank(FROZEN_TRAIN_PATH.resolve())
    assignments = assign_stratified_group_folds(bank)
    assignment_by_id = {row.episode_id: row for row in assignments}
    episode_by_id = {episode.episode_id: episode for episode in bank.episodes}
    inner_train_ids = sorted(
        episode_id
        for episode_id, assignment in assignment_by_id.items()
        if assignment.fold in STAGE_W_INNER_TRAIN_FOLDS
    )
    if len(inner_train_ids) != 1800:
        raise RuntimeError("Stage W inner-train population changed")
    selected_ids = tuple(inner_train_ids[:STAGE_W_EPISODES_PER_UPDATE])
    episodes = tuple(episode_by_id[episode_id] for episode_id in selected_ids)
    if any(episode.split != "train" for episode in episodes):
        raise RuntimeError("Stage W selected a non-train episode")
    threshold_rows = [
        _threshold_row(episode_by_id[episode_id]) for episode_id in inner_train_ids
    ]
    thresholds = thresholds_from_inner_train(threshold_rows)
    return StageWPopulation(
        episodes=episodes,
        episode_sha256={
            episode_id: bank.episode_sha256[episode_id] for episode_id in selected_ids
        },
        bank_sha256=bank.payload_sha256,
        fold_manifest_sha256=fold_manifest_sha256(assignments),
        thresholds=thresholds,
        threshold_episode_count=len(threshold_rows),
    )


def _rollout_trace_sha256(
    transition_episodes: Sequence[Sequence[Mapping[str, Any]]],
) -> str:
    projection: list[dict[str, Any]] = []
    for transitions in transition_episodes:
        for row in transitions:
            projection.append(
                {
                    "episode_id": row["episode_id"],
                    "step_index": row["step_index"],
                    "action": row["action"],
                    "old_log_probability_hex": float(row["old_log_probability"]).hex(),
                    "old_value_hex": float(row["old_value"]).hex(),
                    "policy_rng_before_sha256": row["policy_rng_before_sha256"],
                    "policy_rng_after_sha256": row["policy_rng_after_sha256"],
                }
            )
    return canonical_sha256(projection)


def _finite_metrics(metrics: Mapping[str, Any]) -> bool:
    for key in (
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "intervention_loss",
        "intervention_penalty",
    ):
        value = metrics.get(key)
        if type(value) is not float or not math.isfinite(value):
            return False
    return True


def _collect_and_update(
    state: ArmState,
    episodes: Sequence[LongHorizonEpisode],
    *,
    config: PPOConfig,
    thresholds: SafetyThresholds,
    update: int,
) -> tuple[dict[str, Any], ArmSnapshot]:
    if (
        state.arm not in STAGE_W_ARMS
        or update not in range(STAGE_W_UPDATES)
        or len(episodes) != STAGE_W_EPISODES_PER_UPDATE
        or config != _stage_w_config()
    ):
        raise ValueError("invalid Stage W arm update schedule")
    pre_state = state.identity
    rollout_rng_before = generator_state_sha256(state.rollout_generator)
    transition_episodes: list[list[dict[str, Any]]] = []
    episode_summaries: list[dict[str, Any]] = []
    for episode in episodes:
        transitions, summary = shield_aware_rollout(
            state.model,
            episode,
            torch.device("cpu"),
            mean_incidents=thresholds.mean_incidents,
            generator=state.rollout_generator,
        )
        transition_episodes.append(transitions)
        episode_summaries.append(summary)
    rollout_rng_after = generator_state_sha256(state.rollout_generator)
    decision_count = sum(len(rows) for rows in transition_episodes)
    if (
        not 1
        <= decision_count
        <= (STAGE_W_EPISODES_PER_UPDATE * STAGE_W_MAX_EPISODE_STEPS)
        or decision_count > config.minibatch_size
    ):
        raise ValueError("Stage W whole-rollout minibatch invariant failed")
    hard_invariants_zero = bool(
        all(
            int(summary["invalid_actions"]) == 0
            and int(summary["budget_violations"]) == 0
            for summary in episode_summaries
        )
        and all(
            not bool(row["execute_without_prior_review"])
            for rows in transition_episodes
            for row in rows
        )
    )
    if not hard_invariants_zero:
        raise RuntimeError("Stage W hard invariant failed")

    unsafe_events = sum(
        any(bool(row["wrong_execute"]) for row in rows) for rows in transition_episodes
    )
    wrong_executions = sum(
        int(row["wrong_execute"]) for rows in transition_episodes for row in rows
    )
    cr_decision = select_actor_mode(
        unsafe_events=unsafe_events,
        wrong_executions=wrong_executions,
        episodes=len(episodes),
        thresholds=thresholds,
    )
    if state.update_rule == "frozen-zero-dual":
        base_batch = lagrangian_batch(
            transition_episodes,
            config,
            dual=DualState(),
        )
        actor_mode = "reward-zero-dual"
        actor_mode_details: dict[str, Any] | None = None
        cr_selector_sha256: str | None = None
        selected_advantage_sha256: str | None = None
    else:
        from .a23_cr_ppo import cr_ppo_batch

        base_batch, diagnostics = cr_ppo_batch(
            transition_episodes,
            config,
            decision=cr_decision,
        )
        actor_mode = cr_decision.mode
        actor_mode_details = {
            "decision": asdict(cr_decision),
            "batch": diagnostics,
        }
        cr_selector_sha256 = canonical_sha256(asdict(cr_decision))
        selected_advantage_sha256 = str(diagnostics["normalized_advantage_sha256"])
    batch = shield_aware_batch(transition_episodes, base_batch)
    batch_sha256 = array_mapping_sha256(batch)
    observer = _AuxiliaryGradientObserver() if state.beta > 0.0 else None
    metrics = intervention_ppo_update(
        state.model,
        state.optimizer,
        batch,
        config,
        torch.device("cpu"),
        state.tensor_generator,
        objective=InterventionObjective(beta=state.beta),
        observer=observer,
    )
    if not _finite_metrics(metrics):
        raise FloatingPointError("Stage W update produced non-finite metrics")
    state.optimizer.zero_grad(set_to_none=True)
    post_state = state.identity
    snapshot = capture_arm_snapshot(state)
    restored = restore_arm_snapshot(snapshot)
    if restored.identity != post_state:
        raise RuntimeError("Stage W exact continuation identity mismatch")
    episode_ids = [episode.episode_id for episode in episodes]
    record = {
        "update": update,
        "policy_checkpoint_id": pre_state.model_sha256,
        "pre_state": asdict(pre_state),
        "post_state": asdict(post_state),
        "episode_ids": episode_ids,
        "episode_schedule_sha256": canonical_sha256(episode_ids),
        "environment_episode_count": len(transition_episodes),
        "decision_count": decision_count,
        "optimizer_steps": config.ppo_epochs,
        "rollout_rng_before_sha256": rollout_rng_before,
        "rollout_rng_after_sha256": rollout_rng_after,
        "rollout_trace_sha256": _rollout_trace_sha256(transition_episodes),
        "batch_sha256": batch_sha256,
        "on_policy_snapshot_bound": True,
        "whole_rollout_single_minibatch": True,
        "hard_invariants_zero": hard_invariants_zero,
        "unsafe_events": unsafe_events,
        "wrong_executions": wrong_executions,
        "actor_mode": actor_mode,
        "actor_mode_details": actor_mode_details,
        "cr_selector_sha256": cr_selector_sha256,
        "selected_advantage_sha256": selected_advantage_sha256,
        "intervention_beta": state.beta,
        "intervention_loss": float(metrics["intervention_loss"]),
        "intervention_penalty": float(metrics["intervention_penalty"]),
        "shield_active_decisions": int(
            metrics["pre_update_shield_dependence"]["shield_active_decisions"]
        ),
        "auxiliary_gradient_l2": (
            max(observer.l2_values)
            if observer is not None and observer.l2_values
            else 0.0
        ),
        "ppo_metrics": {
            key: float(metrics[key])
            for key in (
                "policy_loss",
                "value_loss",
                "entropy",
                "approx_kl",
                "clip_fraction",
            )
        },
        "exact_continuation_restore": True,
    }
    if not math.isclose(
        record["intervention_penalty"],
        state.beta * record["intervention_loss"],
        rel_tol=1e-6,
        abs_tol=1e-7,
    ):
        raise RuntimeError("Stage W intervention penalty decomposition failed")
    if state.beta > 0.0 and (
        record["shield_active_decisions"] <= 0
        or record["intervention_loss"] <= 0.0
        or record["intervention_penalty"] <= 0.0
        or record["auxiliary_gradient_l2"] <= 0.0
        or not math.isfinite(record["auxiliary_gradient_l2"])
    ):
        raise RuntimeError("Stage W positive-beta objective was not activated")
    return record, snapshot


def _rng_state_equal(left: tuple[Any, ...], right: tuple[Any, ...]) -> bool:
    return bool(
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def _run_replica(
    *,
    replica_index: int,
    replica_seed: int,
    population: StageWPopulation,
    config: PPOConfig,
) -> dict[str, Any]:
    states = initialize_replica(
        replica_index=replica_index,
        replica_seed=replica_seed,
    )
    initial_identities = {arm: state.identity for arm, state in states.items()}
    initial_object_ids = {
        arm: (
            id(state.model),
            id(state.optimizer),
            id(state.rollout_generator),
            id(state.tensor_generator),
        )
        for arm, state in states.items()
    }
    arm_results: dict[str, dict[str, Any]] = {
        arm: {
            "arm": arm,
            "update_rule": state.update_rule,
            "intervention_beta": state.beta,
            "arm_identity": _arm_identity(arm, config),
            "arm_identity_sha256": canonical_sha256(_arm_identity(arm, config)),
            "initial_state": asdict(state.identity),
            "updates": [],
        }
        for arm, state in states.items()
    }
    last_snapshots: dict[str, ArmSnapshot] = {}
    for update in range(STAGE_W_UPDATES):
        if update:
            for arm, snapshot in last_snapshots.items():
                if states[arm].identity != snapshot.identity:
                    raise RuntimeError("Stage W state changed outside update barrier")
                restored = restore_arm_snapshot(snapshot)
                if restored.identity != states[arm].identity:
                    raise RuntimeError(
                        "Stage W continuation restore verification failed"
                    )
        for arm in STAGE_W_ARMS:
            record, snapshot = _collect_and_update(
                states[arm],
                population.episodes,
                config=config,
                thresholds=population.thresholds,
                update=update,
            )
            arm_results[arm]["updates"].append(record)
            last_snapshots[arm] = snapshot
    return {
        "replica_index": replica_index,
        "replica_seed": replica_seed,
        "initial_state_sha256": canonical_sha256(
            asdict(next(iter(initial_identities.values())))
        ),
        "common_initial_state": len(set(initial_identities.values())) == 1,
        "arm_local_objects": len(
            {
                object_id
                for values in initial_object_ids.values()
                for object_id in values
            }
        )
        == 16,
        "arms": arm_results,
    }


def _claim_boundary() -> dict[str, bool]:
    return {
        "q1_qualified": False,
        "formal_authorized": False,
        "outer_outcomes_authorized": False,
        "performance_evaluable": False,
        "safety_evaluable": False,
        "generalization_evaluable": False,
        "llm_weights_updated": False,
        "formal_absence_race_free": False,
        "formal_absence_linearizable": False,
    }


def run_stage_w() -> dict[str, Any]:
    """Execute exactly 64 train-only controller rollout episodes."""

    schedule = stage_w_schedule()
    config = _stage_w_config()
    python_rng_before = random.getstate()
    numpy_rng_before = np.random.get_state()
    torch_rng_before = torch.random.get_rng_state().clone()
    population = load_stage_w_population()
    replicas = [
        _run_replica(
            replica_index=index,
            replica_seed=seed,
            population=population,
            config=config,
        )
        for index, seed in enumerate(schedule.replica_seeds)
    ]
    global_rng_unchanged = bool(
        random.getstate() == python_rng_before
        and _rng_state_equal(np.random.get_state(), numpy_rng_before)
        and torch.equal(torch.random.get_rng_state(), torch_rng_before)
    )
    update_rows = [
        update
        for replica in replicas
        for fit in replica["arms"].values()
        for update in fit["updates"]
    ]
    update_zero_common = all(
        len(
            {
                replica["arms"][arm]["updates"][0]["rollout_trace_sha256"]
                for arm in STAGE_W_ARMS
            }
        )
        == 1
        for replica in replicas
    )
    chains_closed = all(
        all(
            fit["updates"][0]["post_state"][field]
            == fit["updates"][1]["pre_state"][field]
            for field in _STATE_IDENTITY_FIELDS
        )
        for replica in replicas
        for fit in replica["arms"].values()
    )
    cr_modes = [
        row["actor_mode"]
        for replica in replicas
        for arm in ("F10", "F11")
        for row in replica["arms"][arm]["updates"]
    ]
    f00_equals_f10 = all(
        replica["arms"]["F00"]["updates"][update]["post_state"]
        == replica["arms"]["F10"]["updates"][update]["post_state"]
        for replica in replicas
        for update in range(STAGE_W_UPDATES)
    )
    f01_equals_f11 = all(
        replica["arms"]["F01"]["updates"][update]["post_state"]
        == replica["arms"]["F11"]["updates"][update]["post_state"]
        for replica in replicas
        for update in range(STAGE_W_UPDATES)
    )
    if (
        len(cr_modes) != 8
        or not all(mode == "reward" for mode in cr_modes)
        or not f00_equals_f10
        or not f01_equals_f11
    ):
        raise RuntimeError("Stage W v1 frozen negative diagnostic changed")
    method_blockers = _expected_method_blockers()
    gates = {
        "exact_schedule": len(update_rows) == schedule.arm_updates
        and sum(row["environment_episode_count"] for row in update_rows)
        == schedule.training_rollouts,
        "common_initial_state": all(
            replica["common_initial_state"] for replica in replicas
        ),
        "arm_local_objects": all(replica["arm_local_objects"] for replica in replicas),
        "update_zero_common_rollout": update_zero_common,
        "persistent_model_optimizer_rng_chains": chains_closed,
        "exact_continuation_restore": all(
            row["exact_continuation_restore"] for row in update_rows
        ),
        "on_policy_snapshot_binding": all(
            row["on_policy_snapshot_bound"] for row in update_rows
        ),
        "whole_rollout_single_minibatch": all(
            row["whole_rollout_single_minibatch"] for row in update_rows
        ),
        "hard_invariants_zero": all(row["hard_invariants_zero"] for row in update_rows),
        "positive_beta_objective_activated": all(
            row["shield_active_decisions"] > 0
            and row["intervention_loss"] > 0.0
            and row["intervention_penalty"] > 0.0
            and row["auxiliary_gradient_l2"] > 0.0
            and math.isfinite(row["auxiliary_gradient_l2"])
            for replica in replicas
            for arm in ("F01", "F11")
            for row in replica["arms"][arm]["updates"]
        ),
        "cr_selector_bound": all(
            isinstance(row["cr_selector_sha256"], str)
            and len(row["cr_selector_sha256"]) == 64
            and isinstance(row["selected_advantage_sha256"], str)
            and len(row["selected_advantage_sha256"]) == 64
            for replica in replicas
            for arm in ("F10", "F11")
            for row in replica["arms"][arm]["updates"]
        ),
        "arm_identity_bound": all(
            fit["arm_identity_sha256"] == canonical_sha256(fit["arm_identity"])
            for replica in replicas
            for fit in replica["arms"].values()
        ),
        "global_rng_unchanged": global_rng_unchanged,
        "zero_outer_or_formal_access": True,
    }
    result = {
        "schema_version": STAGE_W_RESULT_VERSION,
        "protocol_version": STAGE_W_PROTOCOL_VERSION,
        "status": STAGE_W_STATUS,
        "stage": "W",
        "evidence_class": "development-wiring-smoke-only",
        "developer_debug": True,
        "source_clean_bound": False,
        "schedule": {
            **asdict(schedule),
            "fits": schedule.fits,
            "arm_updates": schedule.arm_updates,
            "training_rollouts": schedule.training_rollouts,
            "shared_external_schedule_draws": schedule.shared_external_schedule_draws,
        },
        "counts": {
            "replicas": len(schedule.replica_seeds),
            "arms": len(schedule.arms),
            "updates_per_fit": schedule.updates,
            "episodes_per_arm_update": schedule.episodes_per_update,
            "fits": schedule.fits,
            "arm_updates": schedule.arm_updates,
            "training_rollout_episodes": schedule.training_rollouts,
            "shared_external_schedule_draws": schedule.shared_external_schedule_draws,
            "outer_rows": 0,
            "formal_reads": 0,
        },
        "population": {
            "role": "train-only-development-smoke",
            "full_train_bank_parsed_for_fold_assignment": True,
            "design_fold": STAGE_W_DESIGN_FOLD,
            "inner_calibration_fold": STAGE_W_INNER_CALIBRATION_FOLD,
            "inner_calibration_used_for_thresholds": False,
            "inner_calibration_used_for_learned_rollout": False,
            "inner_train_folds": list(STAGE_W_INNER_TRAIN_FOLDS),
            "bank_sha256": population.bank_sha256,
            "fold_manifest_sha256": population.fold_manifest_sha256,
            "episode_ids": [episode.episode_id for episode in population.episodes],
            "episode_sha256": dict(sorted(population.episode_sha256.items())),
            "threshold_episode_count": population.threshold_episode_count,
            "thresholds": asdict(population.thresholds),
        },
        "config": asdict(config),
        "config_sha256": canonical_sha256(asdict(config)),
        "replicas": replicas,
        "gates": gates,
        "method_blockers": method_blockers,
        "wiring_checks_passed": all(gates.values()),
        "q1_qualified": False,
        "formal_authorized": False,
        "outer_rows": 0,
        "trust_boundary_status": "DISCRETE_BARRIER_SNAPSHOT_ONLY",
        "formal_absence_boundary": _formal_absence_boundary(),
        "claim_boundary": _claim_boundary(),
    }
    receipt_core = {
        "schema_version": "multitown-a25-q1-stage-w-receipt-v1",
        "status": STAGE_W_STATUS,
        "stage": "W",
        "wiring_checks_passed": result["wiring_checks_passed"],
        "q1_qualified": False,
        "formal_authorized": False,
        "outer_rows": 0,
        "performance_evaluable": False,
        "safety_evaluable": False,
        "trust_boundary_status": "DISCRETE_BARRIER_SNAPSHOT_ONLY",
        "formal_absence_race_free": False,
        "formal_absence_linearizable": False,
        "arm_identity_sha256": {
            arm: replicas[0]["arms"][arm]["arm_identity_sha256"] for arm in STAGE_W_ARMS
        },
        "counts_sha256": canonical_sha256(result["counts"]),
        "gates_sha256": canonical_sha256(result["gates"]),
        "method_blockers_sha256": canonical_sha256(method_blockers),
    }
    result["receipt"] = {
        **receipt_core,
        "receipt_id": canonical_sha256(receipt_core),
    }
    validate_stage_w_result(result)
    return result


def validate_stage_w_result(result: Mapping[str, Any]) -> None:
    """Validate one in-memory Stage W result; this is not an artifact verifier."""

    required_result_keys = {
        "schema_version",
        "protocol_version",
        "status",
        "stage",
        "evidence_class",
        "developer_debug",
        "source_clean_bound",
        "schedule",
        "counts",
        "population",
        "config",
        "config_sha256",
        "replicas",
        "gates",
        "method_blockers",
        "wiring_checks_passed",
        "q1_qualified",
        "formal_authorized",
        "outer_rows",
        "trust_boundary_status",
        "formal_absence_boundary",
        "claim_boundary",
        "receipt",
    }
    if (
        type(result) is not dict
        or set(result) != required_result_keys
        or result.get("schema_version") != STAGE_W_RESULT_VERSION
        or result.get("protocol_version") != STAGE_W_PROTOCOL_VERSION
        or result.get("status") != STAGE_W_STATUS
        or result.get("stage") != "W"
        or result.get("evidence_class") != "development-wiring-smoke-only"
        or result.get("developer_debug") is not True
        or result.get("source_clean_bound") is not False
        or result.get("q1_qualified") is not False
        or result.get("formal_authorized") is not False
        or type(result.get("outer_rows")) is not int
        or result.get("outer_rows") != 0
        or result.get("trust_boundary_status") != "DISCRETE_BARRIER_SNAPSHOT_ONLY"
        or not _exact_json_equal(
            result.get("formal_absence_boundary"), _formal_absence_boundary()
        )
        or not _exact_json_equal(result.get("claim_boundary"), _claim_boundary())
    ):
        raise ValueError("Stage W claim boundary violation")
    schedule = result.get("schedule")
    expected_schedule = stage_w_schedule()
    if (
        type(schedule) is not dict
        or not _exact_json_equal(
            schedule,
            {
                **asdict(expected_schedule),
                "fits": expected_schedule.fits,
                "arm_updates": expected_schedule.arm_updates,
                "training_rollouts": expected_schedule.training_rollouts,
                "shared_external_schedule_draws": (
                    expected_schedule.shared_external_schedule_draws
                ),
            },
        )
        or type(schedule.get("outer_rows")) is not int
    ):
        raise ValueError("Stage W schedule identity violation")
    counts = result.get("counts")
    expected_counts = {
        "replicas": 2,
        "arms": 4,
        "updates_per_fit": 2,
        "episodes_per_arm_update": 4,
        "fits": 8,
        "arm_updates": 16,
        "training_rollout_episodes": 64,
        "shared_external_schedule_draws": 16,
        "outer_rows": 0,
        "formal_reads": 0,
    }
    if (
        type(counts) is not dict
        or not _exact_json_equal(counts, expected_counts)
        or any(type(value) is not int for value in counts.values())
    ):
        raise ValueError("Stage W count identity violation")
    config = result.get("config")
    expected_config = asdict(_stage_w_config())
    if (
        type(config) is not dict
        or not _exact_json_equal(config, expected_config)
        or result.get("config_sha256") != canonical_sha256(expected_config)
    ):
        raise ValueError("Stage W config identity violation")
    population = result.get("population")
    if (
        type(population) is not dict
        or set(population)
        != {
            "role",
            "full_train_bank_parsed_for_fold_assignment",
            "design_fold",
            "inner_calibration_fold",
            "inner_calibration_used_for_thresholds",
            "inner_calibration_used_for_learned_rollout",
            "inner_train_folds",
            "bank_sha256",
            "fold_manifest_sha256",
            "episode_ids",
            "episode_sha256",
            "threshold_episode_count",
            "thresholds",
        }
        or population.get("role") != "train-only-development-smoke"
        or population.get("full_train_bank_parsed_for_fold_assignment") is not True
        or type(population.get("design_fold")) is not int
        or population.get("design_fold") != STAGE_W_DESIGN_FOLD
        or type(population.get("inner_calibration_fold")) is not int
        or population.get("inner_calibration_fold") != STAGE_W_INNER_CALIBRATION_FOLD
        or population.get("inner_calibration_used_for_thresholds") is not False
        or population.get("inner_calibration_used_for_learned_rollout") is not False
        or not _exact_json_equal(
            population.get("inner_train_folds"), list(STAGE_W_INNER_TRAIN_FOLDS)
        )
        or not _valid_sha256(population.get("bank_sha256"))
        or not _valid_sha256(population.get("fold_manifest_sha256"))
        or type(population.get("threshold_episode_count")) is not int
        or population.get("threshold_episode_count") != 1800
    ):
        raise ValueError("Stage W train-only population violation")
    population_episode_ids = population.get("episode_ids")
    episode_sha256 = population.get("episode_sha256")
    if (
        not isinstance(population_episode_ids, list)
        or len(population_episode_ids) != STAGE_W_EPISODES_PER_UPDATE
        or len(set(population_episode_ids)) != STAGE_W_EPISODES_PER_UPDATE
        or not all(isinstance(value, str) and value for value in population_episode_ids)
        or type(episode_sha256) is not dict
        or set(episode_sha256) != set(population_episode_ids)
        or not all(_valid_sha256(value) for value in episode_sha256.values())
        or type(population.get("thresholds")) is not dict
        or set(population["thresholds"])
        != {"unsafe", "wrong_per_incident", "mean_incidents"}
        or not all(
            type(value) is float and math.isfinite(value)
            for value in population["thresholds"].values()
        )
    ):
        raise ValueError("Stage W episode schedule violation")

    replicas = result.get("replicas")
    if not isinstance(replicas, list) or len(replicas) != 2:
        raise ValueError("Stage W key product violation")
    update_rows: list[Mapping[str, Any]] = []
    persistent_chains_closed = True
    update_zero_common = True
    arm_identities_bound = True
    positive_beta_activated = True
    cr_selectors_bound = True
    all_initial_states: list[Mapping[str, Any]] = []
    for replica_index, replica in enumerate(replicas):
        if (
            not isinstance(replica, Mapping)
            or replica.get("replica_index") != replica_index
            or replica.get("replica_seed") != STAGE_W_SEEDS[replica_index]
            or replica.get("common_initial_state") is not True
            or replica.get("arm_local_objects") is not True
            or tuple(replica.get("arms", ())) != STAGE_W_ARMS
        ):
            raise ValueError("Stage W key product violation")
        replica_initial_states: list[Mapping[str, Any]] = []
        update_zero_trace_hashes: set[str] = set()
        for arm in STAGE_W_ARMS:
            fit = replica["arms"][arm]
            expected_update_rule, expected_beta = _arm_definition(arm)
            expected_arm_identity = _arm_identity(arm, _stage_w_config())
            if (
                not isinstance(fit, Mapping)
                or fit.get("arm") != arm
                or fit.get("update_rule") != expected_update_rule
                or fit.get("intervention_beta") != expected_beta
                or fit.get("arm_identity") != expected_arm_identity
                or fit.get("arm_identity_sha256")
                != canonical_sha256(expected_arm_identity)
            ):
                raise ValueError("Stage W arm identity violation")
            arm_identities_bound = arm_identities_bound and bool(
                fit["arm_identity_sha256"] == canonical_sha256(fit["arm_identity"])
            )
            initial_state = fit.get("initial_state")
            if (
                not isinstance(initial_state, Mapping)
                or set(initial_state) != set(_STATE_IDENTITY_FIELDS)
                or not all(
                    isinstance(initial_state[field], str)
                    and len(initial_state[field]) == 64
                    for field in _STATE_IDENTITY_FIELDS
                )
            ):
                raise ValueError("Stage W common initial state violation")
            replica_initial_states.append(initial_state)
            all_initial_states.append(initial_state)
            updates = fit.get("updates") if isinstance(fit, Mapping) else None
            if not isinstance(updates, list) or len(updates) != 2:
                raise ValueError("Stage W key product violation")
            for update_index, row in enumerate(updates):
                if not isinstance(row, Mapping) or row.get("update") != update_index:
                    raise ValueError("Stage W key product violation")
                pre_state = row.get("pre_state")
                post_state = row.get("post_state")
                if (
                    not isinstance(pre_state, Mapping)
                    or not isinstance(post_state, Mapping)
                    or set(pre_state) != set(_STATE_IDENTITY_FIELDS)
                    or set(post_state) != set(_STATE_IDENTITY_FIELDS)
                    or not all(
                        isinstance(state[field], str) and len(state[field]) == 64
                        for state in (pre_state, post_state)
                        for field in _STATE_IDENTITY_FIELDS
                    )
                    or row.get("policy_checkpoint_id") != pre_state.get("model_sha256")
                    or row.get("rollout_rng_before_sha256")
                    != pre_state.get("rollout_rng_sha256")
                    or row.get("rollout_rng_after_sha256")
                    != post_state.get("rollout_rng_sha256")
                ):
                    raise ValueError("Stage W state chain violation")
                episode_ids = row.get("episode_ids")
                if (
                    episode_ids != population_episode_ids
                    or row.get("episode_schedule_sha256")
                    != canonical_sha256(episode_ids)
                    or row.get("environment_episode_count")
                    != STAGE_W_EPISODES_PER_UPDATE
                    or type(row.get("decision_count")) is not int
                    or not 1
                    <= row["decision_count"]
                    <= STAGE_W_EPISODES_PER_UPDATE * STAGE_W_MAX_EPISODE_STEPS
                    or row.get("optimizer_steps") != _stage_w_config().ppo_epochs
                ):
                    raise ValueError("Stage W episode schedule violation")
                if (
                    row.get("on_policy_snapshot_bound") is not True
                    or row.get("whole_rollout_single_minibatch") is not True
                    or row.get("hard_invariants_zero") is not True
                    or row.get("exact_continuation_restore") is not True
                ):
                    raise ValueError("Stage W wiring invariant violation")
                beta = row.get("intervention_beta")
                loss = row.get("intervention_loss")
                penalty = row.get("intervention_penalty")
                auxiliary_gradient = row.get("auxiliary_gradient_l2")
                if (
                    beta != expected_beta
                    or not all(
                        type(value) is float and math.isfinite(value)
                        for value in (loss, penalty, auxiliary_gradient)
                    )
                    or not math.isclose(
                        penalty,
                        expected_beta * loss,
                        rel_tol=1e-6,
                        abs_tol=1e-7,
                    )
                ):
                    raise ValueError("Stage W intervention objective violation")
                if expected_beta > 0.0:
                    active = bool(
                        type(row.get("shield_active_decisions")) is int
                        and row["shield_active_decisions"] > 0
                        and loss > 0.0
                        and penalty > 0.0
                        and auxiliary_gradient > 0.0
                    )
                    positive_beta_activated = positive_beta_activated and active
                    if not active:
                        raise ValueError("Stage W positive-beta activation violation")
                elif penalty != 0.0 or auxiliary_gradient != 0.0:
                    raise ValueError("Stage W zero-beta objective violation")
                if expected_update_rule == "frozen-zero-dual":
                    if (
                        row.get("actor_mode") != "reward-zero-dual"
                        or row.get("actor_mode_details") is not None
                        or row.get("cr_selector_sha256") is not None
                        or row.get("selected_advantage_sha256") is not None
                    ):
                        raise ValueError("Stage W frozen-zero-dual identity violation")
                else:
                    details = row.get("actor_mode_details")
                    decision = (
                        details.get("decision")
                        if isinstance(details, Mapping)
                        else None
                    )
                    batch = (
                        details.get("batch") if isinstance(details, Mapping) else None
                    )
                    selector_hash = row.get("cr_selector_sha256")
                    selected_hash = row.get("selected_advantage_sha256")
                    bound = bool(
                        isinstance(decision, Mapping)
                        and isinstance(batch, Mapping)
                        and isinstance(selector_hash, str)
                        and len(selector_hash) == 64
                        and selector_hash == canonical_sha256(decision)
                        and isinstance(selected_hash, str)
                        and len(selected_hash) == 64
                        and selected_hash == batch.get("normalized_advantage_sha256")
                        and row.get("actor_mode") == decision.get("mode")
                    )
                    cr_selectors_bound = cr_selectors_bound and bound
                    if not bound:
                        raise ValueError("Stage W CR selector binding violation")
                ppo_metrics = row.get("ppo_metrics")
                if not isinstance(ppo_metrics, Mapping) or not all(
                    type(ppo_metrics.get(key)) is float
                    and math.isfinite(ppo_metrics[key])
                    for key in (
                        "policy_loss",
                        "value_loss",
                        "entropy",
                        "approx_kl",
                        "clip_fraction",
                    )
                ):
                    raise ValueError("Stage W finite PPO metrics violation")
                update_rows.append(row)
            first, second = updates
            chain_closed = bool(
                all(
                    first["post_state"][field] == second["pre_state"][field]
                    for field in _STATE_IDENTITY_FIELDS
                )
                and second["policy_checkpoint_id"]
                == first["post_state"]["model_sha256"]
            )
            persistent_chains_closed = persistent_chains_closed and chain_closed
            if not chain_closed:
                raise ValueError("Stage W state chain violation")
            if first["pre_state"] != initial_state:
                raise ValueError("Stage W common initial state violation")
            update_zero_trace_hashes.add(str(first.get("rollout_trace_sha256")))
        if len({canonical_sha256(value) for value in replica_initial_states}) != 1:
            raise ValueError("Stage W common initial state violation")
        if replica.get("initial_state_sha256") != canonical_sha256(
            replica_initial_states[0]
        ):
            raise ValueError("Stage W common initial state violation")
        update_zero_common = update_zero_common and len(update_zero_trace_hashes) == 1

    observed_arm_updates = len(update_rows)
    observed_training_rollouts = sum(
        int(row["environment_episode_count"]) for row in update_rows
    )
    observed_shared_schedule_draws = sum(
        int(
            replicas[replica_index]["arms"][STAGE_W_ARMS[0]]["updates"][update][
                "environment_episode_count"
            ]
        )
        for replica_index in range(len(STAGE_W_SEEDS))
        for update in range(STAGE_W_UPDATES)
    )
    recomputed_gates = {
        "exact_schedule": observed_arm_updates == expected_counts["arm_updates"]
        and observed_training_rollouts == expected_counts["training_rollout_episodes"],
        "common_initial_state": all(
            replica["common_initial_state"] is True for replica in replicas
        ),
        "arm_local_objects": all(
            replica["arm_local_objects"] is True for replica in replicas
        ),
        "update_zero_common_rollout": update_zero_common,
        "persistent_model_optimizer_rng_chains": persistent_chains_closed,
        "exact_continuation_restore": all(
            row["exact_continuation_restore"] is True for row in update_rows
        ),
        "on_policy_snapshot_binding": all(
            row["on_policy_snapshot_bound"] is True for row in update_rows
        ),
        "whole_rollout_single_minibatch": all(
            row["whole_rollout_single_minibatch"] is True for row in update_rows
        ),
        "hard_invariants_zero": all(
            row["hard_invariants_zero"] is True for row in update_rows
        ),
        "positive_beta_objective_activated": positive_beta_activated,
        "cr_selector_bound": cr_selectors_bound,
        "arm_identity_bound": arm_identities_bound,
        # These two flags are runtime attestations.  An in-memory result validator
        # can enforce but cannot independently reproduce process-global or I/O state.
        "global_rng_unchanged": result.get("gates", {}).get("global_rng_unchanged")
        is True,
        "zero_outer_or_formal_access": type(counts["outer_rows"]) is int
        and counts["outer_rows"] == 0
        and type(counts["formal_reads"]) is int
        and counts["formal_reads"] == 0
        and type(result["outer_rows"]) is int
        and result["outer_rows"] == 0
        and result["formal_authorized"] is False,
    }
    if (
        observed_shared_schedule_draws
        != expected_counts["shared_external_schedule_draws"]
    ):
        raise ValueError("Stage W schedule identity violation")
    gates = result.get("gates")
    if (
        not isinstance(gates, Mapping)
        or gates != recomputed_gates
        or not all(value is True for value in recomputed_gates.values())
        or result.get("wiring_checks_passed") is not True
    ):
        raise ValueError("Stage W wiring gate violation")

    cr_modes = [
        row["actor_mode"]
        for replica in replicas
        for arm in ("F10", "F11")
        for row in replica["arms"][arm]["updates"]
    ]
    f00_equals_f10 = all(
        replica["arms"]["F00"]["updates"][update]["post_state"]
        == replica["arms"]["F10"]["updates"][update]["post_state"]
        for replica in replicas
        for update in range(STAGE_W_UPDATES)
    )
    f01_equals_f11 = all(
        replica["arms"]["F01"]["updates"][update]["post_state"]
        == replica["arms"]["F11"]["updates"][update]["post_state"]
        for replica in replicas
        for update in range(STAGE_W_UPDATES)
    )
    if (
        len(cr_modes) != 8
        or not all(mode == "reward" for mode in cr_modes)
        or not f00_equals_f10
        or not f01_equals_f11
    ):
        raise ValueError("Stage W frozen negative diagnostic violation")
    recomputed_method_blockers = _expected_method_blockers()
    if not _exact_json_equal(result.get("method_blockers"), recomputed_method_blockers):
        raise ValueError("Stage W method-blocker boundary violation")

    receipt = result.get("receipt")
    receipt_core = {
        "schema_version": "multitown-a25-q1-stage-w-receipt-v1",
        "status": STAGE_W_STATUS,
        "stage": "W",
        "wiring_checks_passed": True,
        "q1_qualified": False,
        "formal_authorized": False,
        "outer_rows": 0,
        "performance_evaluable": False,
        "safety_evaluable": False,
        "trust_boundary_status": "DISCRETE_BARRIER_SNAPSHOT_ONLY",
        "formal_absence_race_free": False,
        "formal_absence_linearizable": False,
        "arm_identity_sha256": {
            arm: replicas[0]["arms"][arm]["arm_identity_sha256"] for arm in STAGE_W_ARMS
        },
        "counts_sha256": canonical_sha256(counts),
        "gates_sha256": canonical_sha256(gates),
        "method_blockers_sha256": canonical_sha256(recomputed_method_blockers),
    }
    if (
        type(receipt) is not dict
        or not _exact_json_equal(
            receipt,
            {
                **receipt_core,
                "receipt_id": canonical_sha256(receipt_core),
            },
        )
        or type(receipt.get("outer_rows")) is not int
    ):
        raise ValueError("Stage W receipt firewall violation")


def _authorize_output_path(path: Path) -> None:
    if not path.is_absolute() or not path.name:
        raise ValueError("OUTPUT_PATH_INVALID")
    for component in path.parts:
        lowered = component.casefold()
        if lowered in _FORBIDDEN_OUTPUT_COMPONENTS:
            raise PermissionError("FORBIDDEN_EVIDENCE_PATH")


def _write_result(path: Path, result: Mapping[str, Any]) -> None:
    _authorize_output_path(path)
    payload = (
        json.dumps(
            result,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        + b"\n"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    file_descriptor = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(file_descriptor, payload[offset:])
            if written <= 0:
                raise OSError("short Stage W result write")
            offset += written
        os.fsync(file_descriptor)
    finally:
        os.close(file_descriptor)


def _json_line(value: Mapping[str, Any]) -> str:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )


def _exact_cli_options(
    arguments: Sequence[str],
    *,
    command: str,
    option_names: tuple[str, ...],
) -> dict[str, str]:
    if not arguments or arguments[0] != command:
        raise ValueError("CLI_USAGE_INVALID")
    tokens = list(arguments[1:])
    if len(tokens) != 2 * len(option_names):
        raise ValueError("CLI_USAGE_INVALID")
    options: dict[str, str] = {}
    for index in range(0, len(tokens), 2):
        name, value = tokens[index : index + 2]
        if name not in option_names or name in options or not value:
            raise ValueError("CLI_USAGE_INVALID")
        options[name] = value
    if set(options) != set(option_names):
        raise ValueError("CLI_USAGE_INVALID")
    return options


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        if not arguments:
            raise ValueError("CLI_USAGE_INVALID")
        command = arguments[0]
        if command == "run-stage-w":
            output: Path | None = None
            if len(arguments) == 3 and arguments[1] == "--write-result":
                output = _absolute_canonical_path(
                    arguments[2], label="Stage W debug result output"
                )
                _authorize_output_path(output)
            elif len(arguments) != 1:
                raise ValueError("CLI_USAGE_INVALID")
            result = run_stage_w()
            if output is not None:
                _write_result(output, result)
            sys.stdout.write(_json_line(result))
        elif command == "prepare":
            options = _exact_cli_options(
                arguments,
                command="prepare",
                option_names=(
                    "--root",
                    "--stage",
                    "--input-descriptors",
                    "--lock-out",
                ),
            )
            if options["--stage"] != "W":
                raise ValueError("CLI_USAGE_INVALID")
            repository = _absolute_canonical_path(
                options["--root"], label="repository root"
            )
            input_descriptors = _absolute_canonical_path(
                options["--input-descriptors"], label="bootstrap descriptor"
            )
            lock_out = _absolute_canonical_path(
                options["--lock-out"], label="protocol lock output"
            )
            lock = prepare_stage_w(
                root=repository,
                input_descriptors=input_descriptors,
                lock_out=lock_out,
            )
            sys.stdout.write(
                _json_line(_lock_metadata(lock, status="STAGE_W_LOCK_PREPARED"))
            )
        elif command == "inspect":
            options = _exact_cli_options(
                arguments,
                command="inspect",
                option_names=("--lock",),
            )
            lock_path = _absolute_canonical_path(
                options["--lock"], label="protocol lock"
            )
            sys.stdout.write(_json_line(inspect_stage_w_lock(lock_path)))
        else:
            raise ValueError("CLI_USAGE_INVALID")
        return 0
    except PermissionError as error:
        sys.stderr.write(_json_line({"status": "REJECTED", "reason_code": str(error)}))
        return 2
    except ValueError as error:
        reason = str(error)
        if reason not in {"CLI_USAGE_INVALID", "OUTPUT_PATH_INVALID"}:
            reason = "STAGE_W_VALIDATION_ERROR"
        sys.stderr.write(_json_line({"status": "REJECTED", "reason_code": reason}))
        return 2
    except (FileExistsError, OSError, RuntimeError, TypeError):
        sys.stderr.write(
            _json_line(
                {"status": "REJECTED", "reason_code": "STAGE_W_PREPARE_REJECTED"}
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
