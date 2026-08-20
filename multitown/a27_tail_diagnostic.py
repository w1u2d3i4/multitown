"""Frozen A27-E same-bank exploratory tail-risk diagnostic.

This module is intentionally independent of Torch and of every training runner.
It validates byte-pinned A8/A9/A22 rows, recomputes their episode losses, and
performs one deterministic, paired, fold-stratified CVaR bootstrap.  It does
not train a policy, open an outer split, or authorize an A24/A25 formal run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROTOCOL_NAME = "multitown-a27-e-same-bank-tail-diagnostic-v1"
RESULT_SCHEMA = "multitown-a27-e-same-bank-tail-diagnostic-result-v1"
ARTIFACT_SCHEMA = "multitown-a27-e-tail-artifact-manifest-v1"
EVIDENCE_BASE_REVISION = "e19f8c9f4ffefcf0c941c65201851cae1de946ee"

ALPHA = 0.90
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20_260_813
FOLDS = tuple(range(5))
TRAINING_SEEDS = (20_260_812, 20_260_813, 20_260_814)
EPISODES_PER_FOLD = 600
TOTAL_EPISODES = 3_000
CI_PROBABILITIES = (0.025, 0.975)
LOSS_METRICS = (
    "wrong_executions",
    "safety_penalty_burden",
)
RESOURCE_TAIL_DIAGNOSTICS = (
    "tokens_used",
    "latency_used_s",
)
METRICS = LOSS_METRICS + RESOURCE_TAIL_DIAGNOSTICS
MECHANISMS = (
    "reference",
    "lagrangian",
    "shield",
    "lagrangian-plus-shield",
)
SELECTED_MECHANISM = {
    0: "lagrangian-plus-shield",
    1: "shield",
    2: "shield",
    3: "shield",
    4: "lagrangian-plus-shield",
}

_A9_ROOT = Path("artifacts/a9-v2-ppo-oof-20260813-r2")
_A22_ROOT = Path("artifacts/a22-adaptive-formal-20260814")
_INPUT_SHA256 = {
    Path(
        "benchmarks/multitown-a9-long-horizon-v0.2/train.jsonl"
    ): "ee0c3d9677ea44694373405c2337b3e61cc5562f302976cab8af9fb143b7b777",
    Path(
        "benchmarks/multitown-a9-long-horizon-v0.2/manifest.json"
    ): "30eb7edb6758064e17c9ee6422f080f26923a34e9f103888c52a46cc5ab015aa",
    _A9_ROOT
    / "artifact-manifest.json": "62e0b5dc34219bf1816509deaf036f824ece96808eaa58a50a559dfa53497e3a",
    _A9_ROOT
    / "folds.jsonl": "7b4697b2a60aa28e86fae34ceb5ec57a7e274d7ae8588c8329c506cd1342609b",
    _A9_ROOT
    / "a8-oof-decisions.jsonl": "9bf3ed9148d58c1c49f85807f063c19946de8e12d982d6d04b92865153c0ed40",
    _A9_ROOT
    / "a9-oof-decisions.jsonl": "bb473042361a2515b2eab257a8993859301d42c5303622f67afb8de0fb8df717",
    _A9_ROOT
    / "result.json": "2d863fabd6a7ec69331e1cb831a91e28c04ed93311b772f9b429075e2ec8cbfe",
    _A22_ROOT
    / "artifact-manifest.json": "fd1898483012c4595b8536b2f7af146a03f347be083543f0b21a39c706ae0015",
    _A22_ROOT
    / "outer-decisions.jsonl": "b49b6e9e696b584681ee8db208207c470707c086e6dd417586141220f19e9b23",
    _A22_ROOT
    / "calibration-decisions.jsonl": "60988e4ad8920defe604965d83163bf1c22263c518f6308c27951616cfe656d4",
    _A22_ROOT
    / "all-selections-frozen.json": "e39aff85db69ad446ad7b60f3cb745d5ed02a7461368d72ea86022cb61da9ead",
    _A22_ROOT
    / "result.json": "d8193ee28c2c81c87858408870c8af5d91b2f47ae49ca683d05512439154a719",
    Path(
        "records/a9-v2-ppo-oof-20260813-r2/record.json"
    ): "9cd062b3432c488c2c98de80eae4962126b58ea546a50066415774857ea82b0a",
    Path(
        "records/a22-adaptive-formal-20260814/record.json"
    ): "1697fd966df3acb5bc4f8505e3ef69d9e9cbe1e4c9a17003cd02f2fe569d2c37",
}
_EXPECTED_ARTIFACT_FILES = {
    "protocol.json",
    "input-validation.json",
    "bootstrap-summary.json",
    "result.json",
    "artifact-manifest.json",
}


class A27DiagnosticError(RuntimeError):
    """Fail-closed A27-E validation or replay error."""


@dataclass(frozen=True)
class EpisodeBinding:
    episode_id: str
    episode_sha256: str
    outer_fold: int
    row_sha256: str
    train_bank_sha256: str
    resource_contract_sha256: str
    environment_source_sha256: str
    token_cap: float
    latency_cap_s: float
    metrics: tuple[float, float, float, float]


@dataclass(frozen=True)
class Panel:
    episode_ids: tuple[str, ...]
    folds: np.ndarray
    a8: np.ndarray
    a9_by_seed: np.ndarray
    a22_by_seed: np.ndarray
    validation: Mapping[str, Any]
    a22_joint_development_gate_passed: bool


def _reject_constant(value: str) -> None:
    raise A27DiagnosticError(f"non-finite JSON constant is forbidden: {value}")


def _no_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise A27DiagnosticError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _strict_json_bytes(raw: bytes, *, label: str) -> Any:
    try:
        text = raw.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise A27DiagnosticError(f"invalid strict JSON for {label}: {exc}") from exc


def _strict_json_file(path: Path) -> Any:
    _require_regular_file(path)
    return _strict_json_bytes(path.read_bytes(), label=str(path))


def _jsonl_rows(path: Path) -> Iterable[dict[str, Any]]:
    _require_regular_file(path)
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                raise A27DiagnosticError(f"blank JSONL row: {path}:{line_number}")
            value = _strict_json_bytes(raw, label=f"{path}:{line_number}")
            if type(value) is not dict:
                raise A27DiagnosticError(
                    f"JSONL row must be an object: {path}:{line_number}"
                )
            yield value


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_regular_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise A27DiagnosticError(f"required regular file is missing or unsafe: {path}")


def _as_int(value: Any, *, label: str) -> int:
    if type(value) is not int:
        raise A27DiagnosticError(f"{label} must be an integer")
    return value


def _as_float(value: Any, *, label: str) -> float:
    if type(value) not in (int, float):
        raise A27DiagnosticError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise A27DiagnosticError(f"{label} must be finite")
    return result


def _as_string(value: Any, *, label: str) -> str:
    if type(value) is not str or not value:
        raise A27DiagnosticError(f"{label} must be a non-empty string")
    return value


def _validate_manifest(root: Path, *, expected_files: int) -> int:
    manifest_path = root / "artifact-manifest.json"
    manifest = _strict_json_file(manifest_path)
    if type(manifest) is not dict or type(manifest.get("files")) is not dict:
        raise A27DiagnosticError(f"invalid raw artifact manifest: {manifest_path}")
    files = manifest["files"]
    if len(files) != expected_files:
        raise A27DiagnosticError(
            f"raw manifest file count mismatch: {len(files)} != {expected_files}"
        )
    expected: set[str] = set()
    for name, metadata in files.items():
        relative = Path(name)
        if (
            type(name) is not str
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != name
            or type(metadata) is not dict
        ):
            raise A27DiagnosticError(f"unsafe raw manifest entry: {name!r}")
        expected.add(name)
        subject = root / relative
        _require_regular_file(subject)
        if _as_int(
            metadata.get("bytes"), label=f"{name}.bytes"
        ) != subject.stat().st_size or _as_string(
            metadata.get("sha256"), label=f"{name}.sha256"
        ) != _sha256(subject):
            raise A27DiagnosticError(f"raw manifest binding mismatch: {name}")
    actual: set[str] = set()
    for subject in root.rglob("*"):
        if subject.is_symlink():
            raise A27DiagnosticError(f"symlink in raw artifact: {subject}")
        if subject.is_file():
            actual.add(subject.relative_to(root).as_posix())
    if actual != expected | {"artifact-manifest.json"}:
        raise A27DiagnosticError("raw artifact path set differs from manifest")
    if {"RUNNING.json", "INVALIDATED.json"} & actual:
        raise A27DiagnosticError("raw artifact has a non-complete state marker")
    return len(files)


def _row_metrics(row: Mapping[str, Any], *, label: str) -> tuple[float, ...]:
    wrong = _as_int(row.get("wrong_executions"), label=f"{label}.wrong")
    if wrong < 0:
        raise A27DiagnosticError(f"negative wrong-execution count: {label}")
    if type(row.get("had_wrong_execution")) is not bool:
        raise A27DiagnosticError(f"wrong-execution indicator is not boolean: {label}")
    if row["had_wrong_execution"] != (wrong > 0):
        raise A27DiagnosticError(f"wrong-execution indicator mismatch: {label}")
    trajectory = row.get("trajectory")
    if type(trajectory) is not list or not trajectory:
        raise A27DiagnosticError(f"trajectory must be a non-empty list: {label}")
    recomputed_wrong = 0
    recomputed_burden = 0.0
    previous_tokens = -math.inf
    previous_latency = -math.inf
    for index, transition in enumerate(trajectory):
        if type(transition) is not dict or type(transition.get("reward")) is not dict:
            raise A27DiagnosticError(f"invalid trajectory transition: {label}:{index}")
        action = _as_string(
            transition.get("action"), label=f"{label}.trajectory[{index}].action"
        )
        penalty = _as_float(
            transition["reward"].get("safety_penalty"),
            label=f"{label}.trajectory[{index}].safety_penalty",
        )
        if penalty > 0.0:
            raise A27DiagnosticError(f"positive safety penalty: {label}:{index}")
        if penalty < 0.0:
            if action == "execute":
                recomputed_wrong += 1
            recomputed_burden -= penalty
        transition_tokens = _as_float(
            transition.get("tokens_used"),
            label=f"{label}.trajectory[{index}].tokens_used",
        )
        transition_latency = _as_float(
            transition.get("latency_used_s"),
            label=f"{label}.trajectory[{index}].latency_used_s",
        )
        if (
            transition_tokens < 0.0
            or transition_latency < 0.0
            or transition_tokens < previous_tokens
            or transition_latency < previous_latency
        ):
            raise A27DiagnosticError(f"non-monotone cumulative resource: {label}")
        previous_tokens = transition_tokens
        previous_latency = transition_latency
    burden = _as_float(row.get("safety_penalty_burden"), label=f"{label}.safety burden")
    if burden < 0.0 and burden != 0.0:
        raise A27DiagnosticError(f"negative safety burden: {label}")
    if recomputed_wrong != wrong or not math.isclose(
        recomputed_burden, burden, rel_tol=0.0, abs_tol=1e-12
    ):
        raise A27DiagnosticError(f"trajectory safety recomputation mismatch: {label}")
    tokens = _as_float(row.get("tokens_used"), label=f"{label}.tokens")
    latency = _as_float(row.get("latency_used_s"), label=f"{label}.latency")
    token_cap = _as_float(row.get("token_cap"), label=f"{label}.token cap")
    latency_cap = _as_float(row.get("latency_cap_s"), label=f"{label}.latency cap")
    if (
        token_cap <= 0.0
        or latency_cap <= 0.0
        or tokens != previous_tokens
        or not math.isclose(latency, previous_latency, rel_tol=0.0, abs_tol=1e-12)
        or tokens > token_cap
        or latency > latency_cap + 1e-12
        or _as_int(row.get("invalid_actions"), label=f"{label}.invalid actions") != 0
        or _as_int(row.get("budget_violations"), label=f"{label}.budget violations")
        != 0
    ):
        raise A27DiagnosticError(f"trajectory resource or validity mismatch: {label}")
    return float(wrong), burden, tokens, latency


def _a8_binding(row: Mapping[str, Any]) -> EpisodeBinding:
    episode_id = _as_string(row.get("episode_id"), label="A8 episode_id")
    if (
        row.get("training_seed") is not None
        or row.get("system") != "A8-long-public-view"
        or row.get("split") != "train"
    ):
        raise A27DiagnosticError(f"invalid A8 identity: {episode_id}")
    metrics = _row_metrics(row, label=f"A8:{episode_id}")
    return EpisodeBinding(
        episode_id=episode_id,
        episode_sha256=_as_string(row.get("episode_sha256"), label="A8 episode SHA"),
        outer_fold=_as_int(row.get("outer_fold"), label="A8 fold"),
        row_sha256=hashlib.sha256(_canonical_json_bytes(row)).hexdigest(),
        train_bank_sha256=_as_string(row.get("train_bank_sha256"), label="A8 bank SHA"),
        resource_contract_sha256=_as_string(
            row.get("resource_contract_sha256"), label="A8 resource SHA"
        ),
        environment_source_sha256=_as_string(
            row.get("environment_source_sha256"), label="A8 environment SHA"
        ),
        token_cap=_as_float(row.get("token_cap"), label="A8 token cap"),
        latency_cap_s=_as_float(row.get("latency_cap_s"), label="A8 latency cap"),
        metrics=metrics,
    )


def _validate_candidate_binding(
    row: Mapping[str, Any], binding: EpisodeBinding, *, label: str
) -> tuple[int, tuple[float, ...]]:
    if (
        _as_string(row.get("episode_sha256"), label=f"{label}.episode SHA")
        != binding.episode_sha256
        or _as_int(row.get("outer_fold"), label=f"{label}.fold") != binding.outer_fold
        or _as_string(row.get("a8_row_sha256"), label=f"{label}.A8 binding")
        != binding.row_sha256
        or row.get("train_bank_sha256") != binding.train_bank_sha256
        or row.get("resource_contract_sha256") != binding.resource_contract_sha256
        or row.get("environment_source_sha256") != binding.environment_source_sha256
        or _as_float(row.get("token_cap"), label=f"{label}.token cap")
        != binding.token_cap
        or not math.isclose(
            _as_float(row.get("latency_cap_s"), label=f"{label}.latency cap"),
            binding.latency_cap_s,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise A27DiagnosticError(f"candidate/A8 provenance mismatch: {label}")
    seed = _as_int(row.get("training_seed"), label=f"{label}.training seed")
    checkpoint = row.get("final_checkpoint_sha256")
    if (
        seed not in TRAINING_SEEDS
        or row.get("split") != "train"
        or type(checkpoint) is not str
        or len(checkpoint) != 64
        or any(character not in "0123456789abcdef" for character in checkpoint)
    ):
        raise A27DiagnosticError(f"unexpected training seed: {label}")
    return seed, _row_metrics(row, label=label)


def _validate_fixed_hashes(root: Path) -> list[dict[str, Any]]:
    result = []
    for relative, expected in _INPUT_SHA256.items():
        path = root / relative
        _require_regular_file(path)
        observed = _sha256(path)
        if observed != expected:
            raise A27DiagnosticError(f"fixed input SHA mismatch: {relative}")
        result.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": observed,
            }
        )
    return result


def load_validated_panel(root: Path) -> Panel:
    """Load only compact metrics after validating every frozen input row."""

    root = root.resolve()
    inputs = _validate_fixed_hashes(root)
    a9_manifest_files = _validate_manifest(root / _A9_ROOT, expected_files=69)
    a22_manifest_files = _validate_manifest(root / _A22_ROOT, expected_files=248)
    a9_artifact_manifest = _strict_json_file(root / _A9_ROOT / "artifact-manifest.json")
    a22_artifact_manifest = _strict_json_file(
        root / _A22_ROOT / "artifact-manifest.json"
    )
    a9_artifact_files = a9_artifact_manifest["files"]
    a22_artifact_files = a22_artifact_manifest["files"]
    a9_run_contract_sha256 = hashlib.sha256(
        _canonical_json_bytes(_strict_json_file(root / _A9_ROOT / "run-contract.json"))
    ).hexdigest()
    a22_run_contract_sha256 = hashlib.sha256(
        _canonical_json_bytes(_strict_json_file(root / _A22_ROOT / "run-contract.json"))
    ).hexdigest()
    selection = _strict_json_file(root / _A22_ROOT / "all-selections-frozen.json")
    if (
        type(selection) is not dict
        or type(selection.get("selections")) is not list
        or len(selection["selections"]) != len(FOLDS)
        or any(type(row) is not dict for row in selection["selections"])
        or selection.get("schema_version") != "multitown-a22-all-selections-frozen-v1"
        or selection.get("all_fits_and_calibrations_complete") is not True
        or selection.get("exact_fit_and_calibration_key_products_verified") is not True
        or selection.get("outer_evaluation_started") is not False
        or selection.get("run_contract_sha256") != a22_run_contract_sha256
        or type(selection.get("checkpoint_sha256")) is not dict
        or len(selection["checkpoint_sha256"])
        != len(FOLDS) * len(TRAINING_SEEDS) * len(MECHANISMS)
    ):
        raise A27DiagnosticError("A22 frozen selection structure mismatch")
    a22_checkpoint_sha256 = selection["checkpoint_sha256"]

    def manifest_file_sha256(files: Mapping[str, Any], name: str, *, label: str) -> str:
        metadata = files.get(name)
        if type(metadata) is not dict:
            raise A27DiagnosticError(f"missing checkpoint manifest entry: {label}")
        return _as_string(metadata.get("sha256"), label=f"{label}.sha256")

    source_episodes: dict[str, tuple[str, float, float, int]] = {}
    for row in _jsonl_rows(
        root / "benchmarks/multitown-a9-long-horizon-v0.2/train.jsonl"
    ):
        episode_id = _as_string(row.get("episode_id"), label="source episode_id")
        incidents = row.get("incidents")
        if (
            episode_id in source_episodes
            or row.get("schema_version") != "multitown-a9-long-horizon-pomdp-v2"
            or row.get("split") != "train"
            or type(incidents) is not list
            or not incidents
        ):
            raise A27DiagnosticError(f"invalid source episode: {episode_id}")
        source_episodes[episode_id] = (
            hashlib.sha256(_canonical_json_bytes(row)).hexdigest(),
            _as_float(row.get("token_budget"), label="source token budget"),
            _as_float(row.get("latency_budget_s"), label="source latency budget"),
            len(incidents),
        )
    if len(source_episodes) != TOTAL_EPISODES:
        raise A27DiagnosticError("source-bank episode count mismatch")

    fold_index: dict[str, int] = {}
    fold_counts: Counter[int] = Counter()
    for row in _jsonl_rows(root / _A9_ROOT / "folds.jsonl"):
        episode_id = _as_string(row.get("episode_id"), label="fold episode_id")
        source = source_episodes.get(episode_id)
        fold = _as_int(row.get("fold"), label=f"fold:{episode_id}")
        if (
            source is None
            or episode_id in fold_index
            or fold not in FOLDS
            or row.get("group_id") != episode_id
            or _as_string(row.get("episode_sha256"), label="fold episode SHA")
            != source[0]
            or type(row.get("stratum")) is not str
            or not row["stratum"]
        ):
            raise A27DiagnosticError(f"invalid frozen fold row: {episode_id}")
        fold_index[episode_id] = fold
        fold_counts[fold] += 1
    if set(fold_index) != set(source_episodes) or dict(fold_counts) != {
        fold: EPISODES_PER_FOLD for fold in FOLDS
    }:
        raise A27DiagnosticError("source-bank/fold key product mismatch")

    a8_index: dict[str, EpisodeBinding] = {}
    by_fold: dict[int, list[str]] = defaultdict(list)
    for row in _jsonl_rows(root / _A9_ROOT / "a8-oof-decisions.jsonl"):
        binding = _a8_binding(row)
        source = source_episodes.get(binding.episode_id)
        if binding.episode_id in a8_index:
            raise A27DiagnosticError(f"duplicate A8 episode: {binding.episode_id}")
        if (
            source is None
            or binding.episode_sha256 != source[0]
            or binding.outer_fold != fold_index[binding.episode_id]
            or binding.token_cap != source[1]
            or not math.isclose(
                binding.latency_cap_s, source[2], rel_tol=0.0, abs_tol=1e-12
            )
            or _as_int(row.get("incidents"), label="A8 incidents") != source[3]
        ):
            raise A27DiagnosticError(
                f"A8/source-bank provenance mismatch: {binding.episode_id}"
            )
        a8_index[binding.episode_id] = binding
        by_fold[binding.outer_fold].append(binding.episode_id)
    if len(a8_index) != TOTAL_EPISODES or {
        fold: len(ids) for fold, ids in by_fold.items()
    } != {fold: EPISODES_PER_FOLD for fold in FOLDS}:
        raise A27DiagnosticError("A8 physical episode/fold product mismatch")
    ordered_ids = tuple(
        episode_id for fold in FOLDS for episode_id in sorted(by_fold[fold])
    )
    positions = {episode_id: index for index, episode_id in enumerate(ordered_ids)}
    a8 = np.asarray(
        [a8_index[episode_id].metrics for episode_id in ordered_ids],
        dtype=np.float64,
    ).T
    folds = np.asarray(
        [a8_index[episode_id].outer_fold for episode_id in ordered_ids], dtype=np.int8
    )

    def candidate_matrix(path: Path, *, system: str, selected: bool) -> np.ndarray:
        values = np.empty(
            (len(TRAINING_SEEDS), len(METRICS), TOTAL_EPISODES),
            dtype=np.float64,
        )
        seen: set[tuple[str, int]] = set()
        seed_position = {seed: index for index, seed in enumerate(TRAINING_SEEDS)}
        for row in _jsonl_rows(path):
            episode_id = _as_string(row.get("episode_id"), label=f"{system} episode")
            binding = a8_index.get(episode_id)
            if binding is None:
                raise A27DiagnosticError(f"candidate has unknown episode: {episode_id}")
            seed, metrics = _validate_candidate_binding(
                row, binding, label=f"{system}:{episode_id}"
            )
            key = episode_id, seed
            if key in seen:
                raise A27DiagnosticError(f"duplicate candidate key: {system}:{key}")
            seen.add(key)
            if selected:
                fold = binding.outer_fold
                mechanism = SELECTED_MECHANISM[fold]
                checkpoint_key = f"{fold}:{seed}:{mechanism}"
                checkpoint_path = (
                    f"fits/outer-fold-{fold}/seed-{seed}/{mechanism}/final.pt"
                )
                expected_checkpoint = manifest_file_sha256(
                    a22_artifact_files,
                    checkpoint_path,
                    label=f"A22:{checkpoint_key}",
                )
                if (
                    row.get("evaluation_phase") != "selected-outer"
                    or _as_int(row.get("design_outer_fold"), label="A22 design fold")
                    != fold
                    or row.get("mechanism") != mechanism
                    or row.get("system") != f"A22-{mechanism}"
                    or row.get("selection_manifest_sha256")
                    != _INPUT_SHA256[_A22_ROOT / "all-selections-frozen.json"]
                    or row.get("run_contract_sha256") != a22_run_contract_sha256
                    or row.get("final_checkpoint_sha256") != expected_checkpoint
                    or a22_checkpoint_sha256.get(checkpoint_key) != expected_checkpoint
                ):
                    raise A27DiagnosticError(f"A22 selected identity mismatch: {key}")
            else:
                checkpoint_path = f"fits/fold-{binding.outer_fold}/seed-{seed}/final.pt"
                expected_checkpoint = manifest_file_sha256(
                    a9_artifact_files,
                    checkpoint_path,
                    label=f"A9:{binding.outer_fold}:{seed}",
                )
                if (
                    row.get("system") != "A9-v2-PPO"
                    or row.get("run_contract_sha256") != a9_run_contract_sha256
                    or row.get("final_checkpoint_sha256") != expected_checkpoint
                ):
                    raise A27DiagnosticError(f"A9 system binding mismatch: {key}")
            values[seed_position[seed], :, positions[episode_id]] = metrics
        expected = {
            (episode_id, seed) for episode_id in ordered_ids for seed in TRAINING_SEEDS
        }
        if seen != expected:
            raise A27DiagnosticError(f"{system} episode x seed product mismatch")
        return values

    a9 = candidate_matrix(
        root / _A9_ROOT / "a9-oof-decisions.jsonl", system="A9", selected=False
    )
    a22 = candidate_matrix(
        root / _A22_ROOT / "outer-decisions.jsonl", system="A22", selected=True
    )

    calibration_counts: Counter[tuple[int, int, str]] = Counter()
    calibration_keys: set[tuple[int, int, str, str]] = set()
    for row in _jsonl_rows(root / _A22_ROOT / "calibration-decisions.jsonl"):
        episode_id = _as_string(row.get("episode_id"), label="calibration episode")
        binding = a8_index.get(episode_id)
        if binding is None:
            raise A27DiagnosticError(f"calibration has unknown episode: {episode_id}")
        seed, _ = _validate_candidate_binding(
            row, binding, label=f"A22-calibration:{episode_id}"
        )
        design_fold = _as_int(
            row.get("design_outer_fold"), label="calibration design fold"
        )
        mechanism = _as_string(row.get("mechanism"), label="calibration mechanism")
        key = design_fold, seed, mechanism, episode_id
        checkpoint_key = f"{design_fold}:{seed}:{mechanism}"
        checkpoint_path = (
            f"fits/outer-fold-{design_fold}/seed-{seed}/{mechanism}/final.pt"
        )
        expected_checkpoint = manifest_file_sha256(
            a22_artifact_files,
            checkpoint_path,
            label=f"A22-calibration:{checkpoint_key}",
        )
        if (
            design_fold not in FOLDS
            or mechanism not in MECHANISMS
            or row.get("evaluation_phase") != "inner-calibration"
            or row.get("system") != f"A22-{mechanism}"
            or row.get("selection_manifest_sha256") is not None
            or row.get("run_contract_sha256") != a22_run_contract_sha256
            or row.get("final_checkpoint_sha256") != expected_checkpoint
            or a22_checkpoint_sha256.get(checkpoint_key) != expected_checkpoint
            or binding.outer_fold != (design_fold + 1) % len(FOLDS)
            or key in calibration_keys
        ):
            raise A27DiagnosticError(f"invalid calibration key: {key}")
        calibration_keys.add(key)
        calibration_counts[(design_fold, seed, mechanism)] += 1
    expected_calibration_counts = {
        (fold, seed, mechanism): EPISODES_PER_FOLD
        for fold in FOLDS
        for seed in TRAINING_SEEDS
        for mechanism in MECHANISMS
    }
    if (
        len(calibration_keys) != 36_000
        or dict(calibration_counts) != expected_calibration_counts
    ):
        raise A27DiagnosticError("A22 calibration key product mismatch")

    observed_selection = {
        _as_int(row.get("outer_fold"), label="selection fold"): _as_string(
            row.get("selected_mechanism"), label="selection mechanism"
        )
        for row in selection["selections"]
        if type(row) is dict
    }
    if observed_selection != SELECTED_MECHANISM:
        raise A27DiagnosticError("A22 frozen selection values mismatch")
    if any(
        row.get("status") != "selected"
        or row.get("tie_order") != list(MECHANISMS)
        or row.get("selection_key")
        != "all-seed joint feasibility, max mean autonomous success, "
        "min mean tokens, fixed mechanism order"
        for row in selection["selections"]
    ):
        raise A27DiagnosticError("A22 frozen selection rule mismatch")
    a22_result = _strict_json_file(root / _A22_ROOT / "result.json")
    if (
        type(a22_result) is not dict
        or type(a22_result.get("adaptive_development_gate_passed")) is not bool
        or a22_result.get("selection_manifest_sha256")
        != _INPUT_SHA256[_A22_ROOT / "all-selections-frozen.json"]
        or a22_result.get("selections") != selection["selections"]
    ):
        raise A27DiagnosticError("A22 joint development-gate result is missing")

    a9_record = _strict_json_file(
        root / "records/a9-v2-ppo-oof-20260813-r2/record.json"
    )
    a22_record = _strict_json_file(
        root / "records/a22-adaptive-formal-20260814/record.json"
    )
    a9_bindings = a9_record.get("bindings") if type(a9_record) is dict else None
    a22_bindings = a22_record.get("bindings") if type(a22_record) is dict else None
    train_bank_sha256 = _INPUT_SHA256[
        Path("benchmarks/multitown-a9-long-horizon-v0.2/train.jsonl")
    ]
    if (
        type(a9_bindings) is not dict
        or a9_bindings.get("train_bank_sha256") != train_bank_sha256
        or a9_bindings.get("run_contract_sha256") != a9_run_contract_sha256
        or a9_bindings.get("result_sha256") != _INPUT_SHA256[_A9_ROOT / "result.json"]
        or a9_bindings.get("raw_artifact_manifest_sha256")
        != _INPUT_SHA256[_A9_ROOT / "artifact-manifest.json"]
    ):
        raise A27DiagnosticError("A9 compact-record bindings mismatch")
    if (
        type(a22_bindings) is not dict
        or a22_bindings.get("train_bank_sha256") != train_bank_sha256
        or a22_bindings.get("run_contract_canonical_sha256") != a22_run_contract_sha256
        or a22_bindings.get("raw_result_sha256")
        != _INPUT_SHA256[_A22_ROOT / "result.json"]
        or a22_bindings.get("selection_sha256")
        != _INPUT_SHA256[_A22_ROOT / "all-selections-frozen.json"]
        or a22_bindings.get("raw_artifact_manifest_sha256")
        != _INPUT_SHA256[_A22_ROOT / "artifact-manifest.json"]
        or a22_record.get("selection")
        != {str(fold): mechanism for fold, mechanism in SELECTED_MECHANISM.items()}
    ):
        raise A27DiagnosticError("A22 compact-record bindings mismatch")

    return Panel(
        episode_ids=ordered_ids,
        folds=folds,
        a8=a8,
        a9_by_seed=a9,
        a22_by_seed=a22,
        validation={
            "status": "PASSED",
            "inputs": inputs,
            "a9_manifest_files_verified": a9_manifest_files,
            "a22_manifest_files_verified": a22_manifest_files,
            "a8_rows": len(a8_index),
            "a9_rows": len(TRAINING_SEEDS) * TOTAL_EPISODES,
            "a22_outer_rows": len(TRAINING_SEEDS) * TOTAL_EPISODES,
            "a22_calibration_rows": len(calibration_keys),
            "source_bank_rows": len(source_episodes),
            "fold_assignment_rows": len(fold_index),
            "physical_episodes": TOTAL_EPISODES,
            "fold_counts": {str(fold): int(np.sum(folds == fold)) for fold in FOLDS},
            "training_seeds": list(TRAINING_SEEDS),
            "pairing_valid": True,
            "trajectory_recomputation_valid": True,
            "invalid_actions": 0,
            "budget_violations": 0,
            "calibration_selection_contaminated": True,
        },
        a22_joint_development_gate_passed=a22_result[
            "adaptive_development_gate_passed"
        ],
    )


def empirical_upper_cvar(
    values: Sequence[float] | np.ndarray, alpha: float
) -> dict[str, Any]:
    """Return an inverse-ECDF upper-tail CVaR with fractional boundary ties."""

    sample = np.asarray(values, dtype=np.float64)
    if sample.ndim != 1 or not sample.size or not np.isfinite(sample).all():
        raise A27DiagnosticError("CVaR sample must be a finite non-empty vector")
    if not (0.0 < alpha < 1.0):
        raise A27DiagnosticError("CVaR alpha must be in (0, 1)")
    tail_mass = (1.0 - alpha) * sample.size
    if not math.isfinite(tail_mass) or tail_mass <= 0.0:
        raise A27DiagnosticError("CVaR tail mass is invalid")
    sorted_values = np.sort(sample)
    quantile_index = max(0, math.ceil(alpha * sample.size) - 1)
    value_at_risk = float(sorted_values[quantile_index])
    greater = sample > value_at_risk
    equal = sample == value_at_risk
    greater_count = int(np.count_nonzero(greater))
    equal_count = int(np.count_nonzero(equal))
    remaining = tail_mass - greater_count
    if equal_count <= 0 or remaining < -1e-12 or remaining > equal_count + 1e-12:
        raise A27DiagnosticError("CVaR fractional tie allocation is inconsistent")
    boundary_weight = min(1.0, max(0.0, remaining / equal_count))
    numerator = float(sample[greater].sum()) + boundary_weight * float(
        sample[equal].sum()
    )
    return {
        "alpha": alpha,
        "observations": int(sample.size),
        "tail_mass": float(tail_mass),
        "value_at_risk": value_at_risk,
        "strict_tail_count": greater_count,
        "boundary_tie_count": equal_count,
        "boundary_weight": boundary_weight,
        "cvar": numerator / tail_mass,
    }


def _batch_integer_tail_cvar(matrix: np.ndarray, *, alpha: float) -> np.ndarray:
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise A27DiagnosticError("bootstrap CVaR matrix is invalid")
    tail_mass = (1.0 - alpha) * matrix.shape[1]
    tail_count = round(tail_mass)
    if tail_count <= 0 or not math.isclose(
        tail_mass, tail_count, rel_tol=0.0, abs_tol=1e-12
    ):
        raise A27DiagnosticError("bootstrap requires an integral empirical tail mass")
    boundary = matrix.shape[1] - tail_count
    partitioned = np.partition(matrix, boundary, axis=1)
    return partitioned[:, boundary:].mean(axis=1)


def _draw_fold_stratified_indices(rng: np.random.Generator, width: int) -> np.ndarray:
    if type(width) is not int or width <= 0:
        raise A27DiagnosticError("bootstrap batch width must be a positive integer")
    draws = rng.integers(
        0,
        EPISODES_PER_FOLD,
        size=(width, len(FOLDS), EPISODES_PER_FOLD),
        dtype=np.int64,
    )
    fold_offsets = np.arange(len(FOLDS), dtype=np.int64)[None, :, None]
    fold_offsets *= EPISODES_PER_FOLD
    return (draws + fold_offsets).reshape(width, TOTAL_EPISODES)


def _point_estimates(panel: Panel) -> dict[str, Any]:
    systems = {
        "a8": panel.a8,
        "a9_fixed_seed_mean": panel.a9_by_seed.mean(axis=0),
        "a22_fixed_seed_mean": panel.a22_by_seed.mean(axis=0),
    }
    overall = {
        system: {
            metric: empirical_upper_cvar(matrix[index], ALPHA)
            for index, metric in enumerate(METRICS)
        }
        for system, matrix in systems.items()
    }
    per_seed = {
        "a22": {
            str(seed): {
                metric: empirical_upper_cvar(
                    panel.a22_by_seed[seed_index, metric_index], ALPHA
                )
                for metric_index, metric in enumerate(METRICS)
            }
            for seed_index, seed in enumerate(TRAINING_SEEDS)
        },
        "a9_motivation_only": {
            str(seed): {
                metric: empirical_upper_cvar(
                    panel.a9_by_seed[seed_index, metric_index], ALPHA
                )
                for metric_index, metric in enumerate(METRICS)
            }
            for seed_index, seed in enumerate(TRAINING_SEEDS)
        },
    }
    a22_mean = panel.a22_by_seed.mean(axis=0)
    a9_mean = panel.a9_by_seed.mean(axis=0)
    per_fold = {
        str(fold): {
            "a8": {
                metric: empirical_upper_cvar(
                    panel.a8[index, panel.folds == fold], ALPHA
                )
                for index, metric in enumerate(METRICS)
            },
            "a22_fixed_seed_mean": {
                metric: empirical_upper_cvar(
                    a22_mean[index, panel.folds == fold], ALPHA
                )
                for index, metric in enumerate(METRICS)
            },
            "a9_fixed_seed_mean_motivation_only": {
                metric: empirical_upper_cvar(a9_mean[index, panel.folds == fold], ALPHA)
                for index, metric in enumerate(METRICS)
            },
            "a22_per_seed": {
                str(seed): {
                    metric: empirical_upper_cvar(
                        panel.a22_by_seed[
                            seed_index, metric_index, panel.folds == fold
                        ],
                        ALPHA,
                    )
                    for metric_index, metric in enumerate(METRICS)
                }
                for seed_index, seed in enumerate(TRAINING_SEEDS)
            },
            "a9_per_seed_motivation_only": {
                str(seed): {
                    metric: empirical_upper_cvar(
                        panel.a9_by_seed[seed_index, metric_index, panel.folds == fold],
                        ALPHA,
                    )
                    for metric_index, metric in enumerate(METRICS)
                }
                for seed_index, seed in enumerate(TRAINING_SEEDS)
            },
        }
        for fold in FOLDS
    }
    return {"overall": overall, "per_seed": per_seed, "per_fold": per_fold}


def paired_fold_cluster_bootstrap(panel: Panel) -> dict[str, Any]:
    """Run the frozen PCG64 stream and return compact replayable summaries."""

    a8 = panel.a8
    candidates = {
        "a9_fixed_seed_mean": panel.a9_by_seed.mean(axis=0),
        "a22_fixed_seed_mean": panel.a22_by_seed.mean(axis=0),
        **{
            f"a9_seed_{seed}": panel.a9_by_seed[index]
            for index, seed in enumerate(TRAINING_SEEDS)
        },
        **{
            f"a22_seed_{seed}": panel.a22_by_seed[index]
            for index, seed in enumerate(TRAINING_SEEDS)
        },
    }
    deltas = {
        name: np.empty((len(METRICS), BOOTSTRAP_REPLICATES), dtype=np.float64)
        for name in candidates
    }
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    batch_size = 64
    for start in range(0, BOOTSTRAP_REPLICATES, batch_size):
        width = min(batch_size, BOOTSTRAP_REPLICATES - start)
        indices = _draw_fold_stratified_indices(rng, width)
        for metric_index in range(len(METRICS)):
            base = _batch_integer_tail_cvar(a8[metric_index][indices], alpha=ALPHA)
            for name, candidate in candidates.items():
                selected = _batch_integer_tail_cvar(
                    candidate[metric_index][indices], alpha=ALPHA
                )
                deltas[name][metric_index, start : start + width] = selected - base
    comparisons: dict[str, Any] = {}
    point_estimates = _point_estimates(panel)["overall"]
    for name, matrix in deltas.items():
        comparisons[name] = {}
        for metric_index, metric in enumerate(METRICS):
            values = matrix[metric_index]
            little_endian = np.ascontiguousarray(values.astype("<f8", copy=False))
            point = (
                point_estimates[name][metric]["cvar"]
                - point_estimates["a8"][metric]["cvar"]
                if name in point_estimates
                else empirical_upper_cvar(candidates[name][metric_index], ALPHA)["cvar"]
                - point_estimates["a8"][metric]["cvar"]
            )
            comparisons[name][metric] = {
                "estimand": f"{name}_minus_a8_cvar",
                "point": float(point),
                "ci95_low": float(
                    np.quantile(values, CI_PROBABILITIES[0], method="linear")
                ),
                "ci95_high": float(
                    np.quantile(values, CI_PROBABILITIES[1], method="linear")
                ),
                "replicates": BOOTSTRAP_REPLICATES,
                "replicate_vector_dtype": "little-endian-float64",
                "replicate_vector_sha256": hashlib.sha256(
                    little_endian.tobytes()
                ).hexdigest(),
            }
    return {
        "schema_version": "multitown-a27-e-paired-bootstrap-summary-v1",
        "rng": "numpy.Generator(PCG64)",
        "rng_seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_REPLICATES,
        "fold_order": list(FOLDS),
        "episodes_resampled_per_fold": EPISODES_PER_FOLD,
        "training_seeds_fixed_not_resampled": list(TRAINING_SEEDS),
        "metric_roles": {
            "ordered_decision_losses": list(LOSS_METRICS),
            "decision_excluded_resource_tail_diagnostics": list(
                RESOURCE_TAIL_DIAGNOSTICS
            ),
        },
        "motivation_only_comparisons": [
            "a9_fixed_seed_mean",
            *[f"a9_seed_{seed}" for seed in TRAINING_SEEDS],
        ],
        "ci_probabilities": list(CI_PROBABILITIES),
        "ci_quantile_method": "linear",
        "comparisons": comparisons,
    }


def _protocol() -> dict[str, Any]:
    implementation = Path(__file__).resolve()
    return {
        "schema_version": PROTOCOL_NAME,
        "evidence_base_revision": EVIDENCE_BASE_REVISION,
        "implementation_sha256": _sha256(implementation),
        "inputs": {path.as_posix(): digest for path, digest in _INPUT_SHA256.items()},
        "population": {
            "folds": list(FOLDS),
            "episodes_per_fold": EPISODES_PER_FOLD,
            "physical_episodes": TOTAL_EPISODES,
            "training_seeds": list(TRAINING_SEEDS),
            "a8_rows": TOTAL_EPISODES,
            "a9_rows": TOTAL_EPISODES * len(TRAINING_SEEDS),
            "a22_outer_rows": TOTAL_EPISODES * len(TRAINING_SEEDS),
            "a22_calibration_rows": 36_000,
        },
        "estimator": {
            "alpha": ALPHA,
            "loss_direction": "larger_is_worse",
            "ordered_decision_losses": list(LOSS_METRICS),
            "decision_excluded_resource_tail_diagnostics": list(
                RESOURCE_TAIL_DIAGNOSTICS
            ),
            "seed_aggregation": "physical-episode mean before pooled CVaR",
            "tail_mass_rule": "inverse-ecdf-fractional-boundary-tie",
            "bootstrap": "paired-fold-stratified-physical-episode-cluster",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "rng": "numpy.Generator(PCG64)",
            "ci_probabilities": list(CI_PROBABILITIES),
            "ci_quantile_method": "linear",
        },
        "ordered_stop_rule": [
            "INVALID_INPUT",
            "A22_JOINT_GATE_INELIGIBLE",
            "TAIL_DETERIORATION_SUPPORTED",
            "TAIL_BRANCH_NOT_SUPPORTED",
            "TAIL_DIAGNOSTIC_INCONCLUSIVE",
        ],
        "evidence_scope": "same-bank post-hoc exploratory diagnostic",
    }


def _stop_decision(panel: Panel, bootstrap: Mapping[str, Any]) -> dict[str, Any]:
    comparison = bootstrap["comparisons"]["a22_fixed_seed_mean"]
    lower_w = float(comparison["wrong_executions"]["ci95_low"])
    lower_burden = float(comparison["safety_penalty_burden"]["ci95_low"])
    if not panel.a22_joint_development_gate_passed:
        code = "A22_JOINT_GATE_INELIGIBLE"
    elif lower_w > 0.0 and lower_burden > 0.0:
        code = "TAIL_DETERIORATION_SUPPORTED"
    elif lower_w <= 0.0 and lower_burden <= 0.0:
        code = "TAIL_BRANCH_NOT_SUPPORTED"
    else:
        code = "TAIL_DIAGNOSTIC_INCONCLUSIVE"
    return {
        "code": code,
        "ordered_joint_development_gate_passed": (
            panel.a22_joint_development_gate_passed
        ),
        "a22_minus_a8_safety_ci95_lower": {
            "wrong_executions": lower_w,
            "safety_penalty_burden": lower_burden,
        },
        "permits_training": False,
        "interpretation": (
            "stop CVaR/SRCPO training branch; existing A22 system gate is negative"
            if code == "A22_JOINT_GATE_INELIGIBLE"
            else "development-only method-design signal; no training is authorized"
        ),
    }


def compute_diagnostic(root: Path) -> dict[str, Any]:
    panel = load_validated_panel(root)
    protocol = _protocol()
    protocol_sha256 = hashlib.sha256(_canonical_json_bytes(protocol)).hexdigest()
    points = _point_estimates(panel)
    bootstrap = paired_fold_cluster_bootstrap(panel)
    result = {
        "schema_version": RESULT_SCHEMA,
        "protocol_sha256": protocol_sha256,
        "evidence_scope": "same-bank post-hoc exploratory diagnostic",
        "validation": panel.validation,
        "estimator": protocol["estimator"],
        "point_estimates": points,
        "paired_deltas": bootstrap["comparisons"],
        "stop_decision": _stop_decision(panel, bootstrap),
        "motivation_only": {
            "a9_v2": True,
            "a22_calibration_mechanisms": True,
        },
        "claim_boundary": {
            "confirmatory": False,
            "hidden_test_or_ood": False,
            "independent_replication": False,
            "seed_population_inference": False,
            "policy_retraining": False,
            "llm_weight_rl": False,
            "a24_formal": False,
            "permits_cvar_or_srcpo_training": False,
            "resource_tail_diagnostics_decision_eligible": False,
            "bootstrap_population_coverage_guarantee": False,
            "calibration_selection_adaptivity_removed": False,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    return {
        "protocol.json": protocol,
        "input-validation.json": panel.validation,
        "bootstrap-summary.json": bootstrap,
        "result.json": result,
    }


def _clean_git_source_identity(root: Path) -> dict[str, Any]:
    root = root.resolve()

    def git_output(*arguments: str) -> str:
        try:
            completed = subprocess.run(
                ["git", *arguments],
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            raise A27DiagnosticError(
                f"cannot inspect Git source identity: {exc}"
            ) from exc
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise A27DiagnosticError(f"Git source identity failed: {detail}")
        return completed.stdout.strip()

    status = git_output("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        paths = " | ".join(status.splitlines()[:10])
        raise A27DiagnosticError(f"A27-E source worktree is not clean: {paths}")
    revision = git_output("rev-parse", "--verify", "HEAD")
    tree = git_output("rev-parse", "--verify", "HEAD^{tree}")
    if (
        len(revision) != 40
        or len(tree) != 40
        or any(character not in "0123456789abcdef" for character in revision + tree)
    ):
        raise A27DiagnosticError("invalid Git source revision or tree")
    return {"revision": revision, "tree": tree, "clean": True}


def _verify_git_source_identity(root: Path, manifest: Mapping[str, Any]) -> None:
    root = root.resolve()
    revision = manifest["source_revision"]
    try:
        observed_tree = subprocess.run(
            ["git", "rev-parse", "--verify", f"{revision}^{{tree}}"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        object_type = subprocess.run(
            ["git", "cat-file", "-t", revision],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        source_file = subprocess.run(
            [
                "git",
                "show",
                f"{revision}:multitown/a27_tail_diagnostic.py",
            ],
            cwd=root,
            check=False,
            capture_output=True,
        )
    except OSError as exc:
        raise A27DiagnosticError(f"cannot verify Git source identity: {exc}") from exc
    if (
        observed_tree.returncode != 0
        or observed_tree.stdout.strip() != manifest["source_tree"]
        or object_type.returncode != 0
        or object_type.stdout.strip() != "commit"
        or source_file.returncode != 0
        or hashlib.sha256(source_file.stdout).hexdigest()
        != manifest["implementation_sha256"]
    ):
        raise A27DiagnosticError("A27-E producer Git source identity mismatch")


def _manifest_for(
    directory: Path,
    names: Sequence[str],
    *,
    source_identity: Mapping[str, Any],
    protocol_sha256: str,
    implementation_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": ARTIFACT_SCHEMA,
        "protocol": PROTOCOL_NAME,
        "evidence_base_revision": EVIDENCE_BASE_REVISION,
        "source_revision": source_identity["revision"],
        "source_tree": source_identity["tree"],
        "source_clean": source_identity["clean"],
        "implementation_sha256": implementation_sha256,
        "protocol_sha256": protocol_sha256,
        "files": {
            name: {
                "bytes": (directory / name).stat().st_size,
                "sha256": _sha256(directory / name),
            }
            for name in sorted(names)
        },
    }


def run_diagnostic(root: Path, output: Path) -> dict[str, Any]:
    source_identity = _clean_git_source_identity(root)
    output = output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        output.mkdir()
    except FileExistsError as exc:
        raise A27DiagnosticError(f"A27-E output already exists: {output}") from exc
    incomplete = output / "INCOMPLETE.json"
    incomplete.write_bytes(
        _pretty_json_bytes(
            {
                "schema_version": "multitown-a27-e-incomplete-artifact-v1",
                "status": "INCOMPLETE",
            }
        )
    )
    # If any later operation raises, the exclusive target and marker are
    # deliberately preserved. A failed attempt can neither be mistaken for a
    # complete artifact nor silently overwritten at the same evidence path.
    objects = compute_diagnostic(root)
    for name, value in objects.items():
        (output / name).write_bytes(_pretty_json_bytes(value))
    protocol = objects["protocol.json"]
    protocol_sha256 = objects["result.json"]["protocol_sha256"]
    manifest = _manifest_for(
        output,
        tuple(objects),
        source_identity=source_identity,
        protocol_sha256=protocol_sha256,
        implementation_sha256=protocol["implementation_sha256"],
    )
    _verify_git_source_identity(root, manifest)
    (output / "artifact-manifest.json").write_bytes(_pretty_json_bytes(manifest))
    incomplete.unlink()
    return {
        "status": "PASSED",
        "artifact": str(output),
        "artifact_manifest_sha256": _sha256(output / "artifact-manifest.json"),
        "protocol_sha256": protocol_sha256,
        "source_revision": source_identity["revision"],
        "source_tree": source_identity["tree"],
        "stop_decision": objects["result.json"]["stop_decision"],
    }


def verify_artifact(root: Path, artifact: Path) -> dict[str, Any]:
    artifact = artifact.absolute()
    if artifact.is_symlink() or not artifact.is_dir():
        raise A27DiagnosticError("A27-E artifact must be a real directory")
    actual = {path.name for path in artifact.iterdir() if path.is_file()}
    if any(path.is_symlink() or not path.is_file() for path in artifact.iterdir()):
        raise A27DiagnosticError("A27-E artifact contains an unsafe member")
    if actual != _EXPECTED_ARTIFACT_FILES:
        raise A27DiagnosticError("A27-E artifact path set mismatch")
    manifest = _strict_json_file(artifact / "artifact-manifest.json")
    manifest_keys = {
        "schema_version",
        "protocol",
        "evidence_base_revision",
        "source_revision",
        "source_tree",
        "source_clean",
        "implementation_sha256",
        "protocol_sha256",
        "files",
    }
    if (
        type(manifest) is not dict
        or set(manifest) != manifest_keys
        or manifest.get("schema_version") != ARTIFACT_SCHEMA
        or manifest.get("protocol") != PROTOCOL_NAME
        or manifest.get("evidence_base_revision") != EVIDENCE_BASE_REVISION
        or manifest.get("source_clean") is not True
        or type(manifest.get("source_revision")) is not str
        or len(manifest["source_revision"]) != 40
        or type(manifest.get("source_tree")) is not str
        or len(manifest["source_tree"]) != 40
        or any(
            character not in "0123456789abcdef"
            for character in manifest["source_revision"] + manifest["source_tree"]
        )
    ):
        raise A27DiagnosticError("A27-E artifact manifest schema mismatch")
    _verify_git_source_identity(root, manifest)
    expected_files = manifest.get("files")
    if type(expected_files) is not dict or set(expected_files) != (
        _EXPECTED_ARTIFACT_FILES - {"artifact-manifest.json"}
    ):
        raise A27DiagnosticError("A27-E artifact manifest inventory mismatch")
    for name, metadata in expected_files.items():
        if (
            type(metadata) is not dict
            or set(metadata) != {"bytes", "sha256"}
            or (
                _as_int(metadata.get("bytes"), label=f"artifact {name} bytes")
                != (artifact / name).stat().st_size
                or _as_string(metadata.get("sha256"), label=f"artifact {name} SHA")
                != _sha256(artifact / name)
            )
        ):
            raise A27DiagnosticError(f"A27-E artifact binding mismatch: {name}")
    observed_protocol = _strict_json_file(artifact / "protocol.json")
    observed_protocol_sha256 = hashlib.sha256(
        _canonical_json_bytes(observed_protocol)
    ).hexdigest()
    if (
        manifest.get("implementation_sha256")
        != observed_protocol.get("implementation_sha256")
        or manifest.get("implementation_sha256") != _sha256(Path(__file__).resolve())
        or manifest.get("protocol_sha256") != observed_protocol_sha256
    ):
        raise A27DiagnosticError("A27-E source/protocol identity mismatch")
    replay = compute_diagnostic(root)
    producer_runtime: Mapping[str, Any] | None = None
    for name, expected in replay.items():
        observed = _strict_json_file(artifact / name)
        if name == "result.json" and type(observed) is dict and type(expected) is dict:
            observed = dict(observed)
            expected = dict(expected)
            producer_runtime = observed.pop("runtime", None)
            expected.pop("runtime", None)
            if (
                type(producer_runtime) is not dict
                or set(producer_runtime) != {"python", "numpy", "platform"}
                or any(type(value) is not str for value in producer_runtime.values())
            ):
                raise A27DiagnosticError("A27-E producer runtime is invalid")
        if observed != expected:
            raise A27DiagnosticError(f"A27-E artifact replay mismatch: {name}")
    return {
        "status": "PASSED",
        "artifact_recomputed": True,
        "artifact_manifest_sha256": _sha256(artifact / "artifact-manifest.json"),
        "protocol_sha256": replay["result.json"]["protocol_sha256"],
        "source_revision": manifest["source_revision"],
        "source_tree": manifest["source_tree"],
        "producer_runtime": producer_runtime,
        "verifier_runtime": replay["result.json"]["runtime"],
        "stop_decision": replay["result.json"]["stop_decision"],
        "claim_scope": "same-bank post-hoc exploratory diagnostic",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    run = subcommands.add_parser("run", help="run the frozen A27-E diagnostic once")
    run.add_argument("--root", type=Path, default=Path.cwd())
    run.add_argument("--output", type=Path, required=True)
    verify = subcommands.add_parser(
        "verify", help="recompute and verify an A27-E artifact"
    )
    verify.add_argument("--root", type=Path, default=Path.cwd())
    verify.add_argument("--artifact", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "run":
            result = run_diagnostic(arguments.root, arguments.output)
        else:
            result = verify_artifact(arguments.root, arguments.artifact)
    except A27DiagnosticError as exc:
        print(f"INVALID_INPUT: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
