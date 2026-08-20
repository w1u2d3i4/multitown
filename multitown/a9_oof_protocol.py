"""Fail-closed train-only protocol for cross-fitted PPO development evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import stat
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .a9_long_horizon_env import A9_ENV_VERSION
from .long_horizon_env import (
    ACTION_COSTS, ACTION_NAMES, LongHorizonEpisode, RLAction,
    MultiTownLongHorizonEnv,
)


PROTOCOL_VERSION = "multitown-a9-train-only-ppo-oof-protocol-v2-recovery"
RESOURCE_TABLE_VERSION = "multitown-a9-shared-episode-resource-table-v1"
FOLD_VERSION = "multitown-a9-train-only-multilabel-stratified-group-folds-v2"
EXPECTED_TRAIN_SHA256 = (
    "ee0c3d9677ea44694373405c2337b3e61cc5562f302976cab8af9fb143b7b777"
)
EXPECTED_TRAIN_EPISODES = 3000
DEFAULT_FOLDS = 5
DEFAULT_FOLD_SEED = 20260813
FROZEN_TRAIN_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks" / "multitown-a9-long-horizon-v0.2" / "train.jsonl"
)
MAX_FOLD_EPISODE_SPREAD = 0
MAX_STRATUM_EPISODE_SPREAD = 1
MAX_LENGTH_EPISODE_SPREAD = 1
MAX_FAMILY_INCIDENT_SPREAD = 10
MAX_FAMILY_FAILURE_INCIDENT_SPREAD = 6


@dataclass(frozen=True)
class LoadedTrainBank:
    path: Path
    payload_sha256: str
    episodes: tuple[LongHorizonEpisode, ...]
    episode_sha256: Mapping[str, str]


@dataclass(frozen=True)
class FoldAssignment:
    episode_id: str
    episode_sha256: str
    stratum: str
    group_id: str
    fold: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


def _episode_sha256(episode: LongHorizonEpisode) -> str:
    return hashlib.sha256(
        _canonical(episode.to_dict()).encode("utf-8")
    ).hexdigest()


def _serialized_bank_sha256(episodes: Sequence[LongHorizonEpisode]) -> str:
    digest = hashlib.sha256()
    for episode in episodes:
        digest.update((json.dumps(
            episode.to_dict(), sort_keys=True, allow_nan=False,
        ) + "\n").encode("utf-8"))
    return digest.hexdigest()


def _strict_json(payload: str, *, label: str) -> Any:
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
        return json.loads(
            payload, object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {label}") from exc


def _is_int(value: Any) -> bool:
    return type(value) is int


def _is_number(value: Any) -> bool:
    return type(value) in {int, float} and math.isfinite(float(value))


def _validate_episode_row(value: Any, *, line_number: int) -> LongHorizonEpisode:
    label = f"train line {line_number}"
    required = {
        "schema_version", "episode_id", "split", "seed", "token_budget",
        "latency_budget_s", "max_steps", "incidents",
    }
    incident_required = {
        "family", "failure_mode", "severity", "correct_action",
        "sensor_candidate", "weak_candidate", "strong_candidate",
        "reviewer_pass", "fail_first_actions",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"invalid episode schema in {label}")
    if value.get("schema_version") != A9_ENV_VERSION or value.get("split") != "train":
        raise ValueError(f"non-train or unsupported episode in {label}")
    if (
        not isinstance(value.get("episode_id"), str)
        or not _is_int(value.get("seed"))
        or not _is_int(value.get("token_budget"))
        or not _is_number(value.get("latency_budget_s"))
        or not _is_int(value.get("max_steps"))
    ):
        raise ValueError(f"invalid episode scalar types in {label}")
    incidents = value.get("incidents")
    if not isinstance(incidents, list) or not 4 <= len(incidents) <= 7:
        raise ValueError(f"invalid incident list in {label}")
    for incident in incidents:
        if not isinstance(incident, dict) or set(incident) != incident_required:
            raise ValueError(f"invalid incident schema in {label}")
        integer_fields = (
            "family", "failure_mode", "correct_action", "sensor_candidate",
            "weak_candidate", "strong_candidate",
        )
        if any(not _is_int(incident.get(field)) for field in integer_fields):
            raise ValueError(f"invalid incident integer types in {label}")
        if (
            incident["family"] not in range(4)
            or incident["failure_mode"] not in range(4)
            or any(incident[field] not in range(4) for field in integer_fields[2:])
            or not _is_number(incident.get("severity"))
            or not 0.0 <= float(incident["severity"]) <= 1.0
        ):
            raise ValueError(f"invalid incident value bounds in {label}")
        reviewers = incident.get("reviewer_pass")
        failures = incident.get("fail_first_actions")
        if (
            not isinstance(reviewers, list)
            or len(reviewers) != 4
            or any(type(item) is not bool for item in reviewers)
            or not isinstance(failures, list)
            or any(not _is_int(item) or item not in range(5) for item in failures)
            or failures != sorted(set(failures))
        ):
            raise ValueError(f"invalid reviewer or tool-failure contract in {label}")
    episode = LongHorizonEpisode.from_dict(
        value, expected_schema_version=A9_ENV_VERSION,
    )
    if (
        episode.schema_version != A9_ENV_VERSION
        or episode.split != "train"
        or
        episode.episode_id != f"a9-lh-train-{episode.seed:08d}"
        or episode.token_budget != len(episode.incidents) * 650
        or not math.isclose(
            episode.latency_budget_s, len(episode.incidents) * 1.65,
            rel_tol=0.0, abs_tol=1e-12,
        )
        or episode.max_steps != min(50, max(20, len(episode.incidents) * 6 + 6))
    ):
        raise ValueError(f"invalid train episode contract in {label}")
    return episode


def load_frozen_train_bank(path: Path) -> LoadedTrainBank:
    """Load only the one byte-pinned physical train file."""

    requested = Path(path)
    if not requested.is_absolute():
        raise ValueError("frozen train bank path must be absolute")
    supplied = Path(os.path.abspath(os.fspath(requested)))
    allowed = Path(os.path.abspath(os.fspath(FROZEN_TRAIN_PATH)))
    if supplied != allowed:
        raise ValueError(
            "refusing to open anything except the frozen physical train bank"
        )
    directory_flags = (
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = (
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    directory_descriptor = os.open(supplied.anchor, directory_flags)
    try:
        for component in supplied.parts[1:-1]:
            next_descriptor = os.open(
                component, directory_flags, dir_fd=directory_descriptor,
            )
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor
        descriptor = os.open(
            supplied.name, file_flags, dir_fd=directory_descriptor,
        )
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError("train bank must be a regular non-symlink file")
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                payload = handle.read()
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise ValueError("train bank must be a readable non-symlink file") from exc
    finally:
        os.close(directory_descriptor)
    path = supplied
    digest = hashlib.sha256(payload).hexdigest()
    if digest != EXPECTED_TRAIN_SHA256:
        raise ValueError(
            f"train bank SHA-256 mismatch: expected {EXPECTED_TRAIN_SHA256}, got {digest}"
        )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("train bank is not UTF-8") from exc
    raw_lines = text.splitlines()
    if len(raw_lines) != EXPECTED_TRAIN_EPISODES or any(not row for row in raw_lines):
        raise ValueError("train bank row count or blank-line contract changed")
    episodes: list[LongHorizonEpisode] = []
    hashes: dict[str, str] = {}
    seeds: set[int] = set()
    for line_number, line in enumerate(raw_lines, start=1):
        value = _strict_json(line, label=f"train line {line_number}")
        episode = _validate_episode_row(value, line_number=line_number)
        if episode.episode_id in hashes or episode.seed in seeds:
            raise ValueError("duplicate train episode ID or seed")
        hashes[episode.episode_id] = hashlib.sha256(
            _canonical(value).encode("utf-8")
        ).hexdigest()
        seeds.add(episode.seed)
        episodes.append(episode)
    if len(episodes) != EXPECTED_TRAIN_EPISODES:
        raise ValueError("incomplete frozen train bank")
    return LoadedTrainBank(
        path=path, payload_sha256=digest, episodes=tuple(episodes),
        episode_sha256=dict(sorted(hashes.items())),
    )


def episode_stratum(episode: LongHorizonEpisode) -> str:
    """Observable family-profile stratum; it contains no correctness label."""

    counts = Counter(incident.family for incident in episode.incidents)
    if set(counts) - set(range(4)):
        raise ValueError("unexpected incident family")
    dominant = min(
        range(4), key=lambda family: (-counts.get(family, 0), family),
    )
    return f"dominant_family={dominant}|incidents={len(episode.incidents)}"


def assign_stratified_group_folds(
    bank: LoadedTrainBank, *, folds: int = DEFAULT_FOLDS,
    seed: int = DEFAULT_FOLD_SEED,
) -> tuple[FoldAssignment, ...]:
    """Balance multi-label incident composition while isolating episode groups."""

    if type(folds) is not int or folds != DEFAULT_FOLDS:
        raise ValueError(f"formal protocol requires exactly {DEFAULT_FOLDS} folds")
    if type(seed) is not int or seed != DEFAULT_FOLD_SEED:
        raise ValueError(f"formal protocol requires fold seed {DEFAULT_FOLD_SEED}")
    if folds > len(bank.episodes):
        raise ValueError("invalid fold count")
    strata = sorted({episode_stratum(episode) for episode in bank.episodes})
    family_modes = sorted({
        (incident.family, incident.failure_mode)
        for episode in bank.episodes for incident in episode.incidents
    })
    feature_names = [
        f"family={family}|failure_mode={failure_mode}"
        for family, failure_mode in family_modes
    ] + [f"stratum:{stratum}" for stratum in strata] + ["episodes"]
    vectors: dict[str, tuple[int, ...]] = {}
    for episode in bank.episodes:
        pairs = Counter(
            (incident.family, incident.failure_mode) for incident in episode.incidents
        )
        stratum = episode_stratum(episode)
        vectors[episode.episode_id] = tuple(
            [pairs[pair] for pair in family_modes]
            + [int(stratum == candidate) for candidate in strata]
            + [1]
        )
    totals = [
        sum(vector[index] for vector in vectors.values())
        for index in range(len(feature_names))
    ]
    targets = [total / folds for total in totals]
    fold_features = [[0] * len(feature_names) for _ in range(folds)]
    fold_counts = [0] * folds
    fold_strata = {stratum: [0] * folds for stratum in strata}
    stratum_quotas: dict[str, list[int]] = {}
    lengths = sorted({len(episode.incidents) for episode in bank.episodes})
    length_quotas: dict[int, list[int]] = {}
    quota_totals = [0] * folds
    for length in lengths:
        count = sum(len(episode.incidents) == length for episode in bank.episodes)
        base, remainder = divmod(count, folds)
        quotas = [base] * folds
        offset = int(hashlib.sha256(
            f"{FOLD_VERSION}:{seed}:length-quota:{length}".encode()
        ).hexdigest(), 16) % folds
        tie_order = [*range(offset, folds), *range(0, offset)]
        selected = sorted(
            range(folds),
            key=lambda fold: (quota_totals[fold], tie_order.index(fold)),
        )[:remainder]
        for fold in selected:
            quotas[fold] += 1
        length_quotas[length] = quotas
        for fold, quota in enumerate(quotas):
            quota_totals[fold] += quota
    if quota_totals != [len(bank.episodes) // folds] * folds:
        raise ValueError("balanced length quotas could not fill equal outer folds")
    for length in lengths:
        length_strata = [
            stratum for stratum in strata
            if stratum.endswith(f"incidents={length}")
        ]
        base_totals = [0] * folds
        remainders: dict[str, int] = {}
        for stratum in length_strata:
            count = sum(
                episode_stratum(episode) == stratum for episode in bank.episodes
            )
            base, remainder = divmod(count, folds)
            stratum_quotas[stratum] = [base] * folds
            remainders[stratum] = remainder
            for fold in range(folds):
                base_totals[fold] += base
        capacity = [
            length_quotas[length][fold] - base_totals[fold]
            for fold in range(folds)
        ]
        for stratum in sorted(
            length_strata, key=lambda item: (-remainders[item], item),
        ):
            remainder = remainders[stratum]
            offset = int(hashlib.sha256(
                f"{FOLD_VERSION}:{seed}:stratum-quota:{stratum}".encode()
            ).hexdigest(), 16) % folds
            tie_order = [*range(offset, folds), *range(0, offset)]
            selected = sorted(
                [fold for fold in range(folds) if capacity[fold] > 0],
                key=lambda fold: (-capacity[fold], tie_order.index(fold)),
            )[:remainder]
            if len(selected) != remainder:
                raise ValueError("could not satisfy nested stratum quotas")
            for fold in selected:
                stratum_quotas[stratum][fold] += 1
                capacity[fold] -= 1
        if any(capacity):
            raise ValueError("incomplete nested stratum quotas")
    assignments: list[FoldAssignment] = []
    for stratum in strata:
        members = [
            episode for episode in bank.episodes
            if episode_stratum(episode) == stratum
        ]
        members.sort(key=lambda episode: episode.episode_id)
        rng = random.Random(f"{FOLD_VERSION}:{seed}:{stratum}")
        rng.shuffle(members)
        members.sort(
            key=lambda episode: -sum(
                value * value for value in vectors[episode.episode_id][:-1]
            )
        )
        for episode in members:
            vector = vectors[episode.episode_id]

            def imbalance_delta(fold: int) -> float:
                return sum(
                    (
                        (fold_features[fold][index] + value - targets[index]) ** 2
                        - (fold_features[fold][index] - targets[index]) ** 2
                    ) / max(targets[index], 1e-12)
                    for index, value in enumerate(vector)
                )

            candidate_folds = [
                fold for fold in range(folds)
                if fold_strata[stratum][fold] < stratum_quotas[stratum][fold]
            ]
            fold = min(
                candidate_folds,
                key=lambda index: (
                    imbalance_delta(index), fold_counts[index], index,
                ),
            )
            fold_counts[fold] += 1
            fold_strata[stratum][fold] += 1
            for index, value in enumerate(vector):
                fold_features[fold][index] += value
            assignments.append(FoldAssignment(
                episode_id=episode.episode_id,
                episode_sha256=bank.episode_sha256[episode.episode_id],
                stratum=stratum,
                group_id=episode.episode_id,
                fold=fold,
            ))
    ordered = tuple(sorted(assignments, key=lambda item: item.episode_id))
    validate_fold_assignments(bank, ordered, folds=folds)
    return ordered


def validate_fold_assignments(
    bank: LoadedTrainBank, assignments: Sequence[FoldAssignment], *, folds: int,
) -> dict[str, Any]:
    if type(folds) is not int or folds != DEFAULT_FOLDS:
        raise ValueError(f"formal protocol requires exactly {DEFAULT_FOLDS} folds")
    expected = {episode.episode_id for episode in bank.episodes}
    ids = [row.episode_id for row in assignments]
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError("each train episode must occur in exactly one outer fold")
    groups = [row.group_id for row in assignments]
    if len(groups) != len(set(groups)) or groups != ids:
        raise ValueError("episode group isolation failed")
    episode_index = {episode.episode_id: episode for episode in bank.episodes}
    for row in assignments:
        if (
            type(row.fold) is not int
            or row.fold not in range(folds)
            or row.episode_sha256 != bank.episode_sha256[row.episode_id]
            or row.stratum != episode_stratum(episode_index[row.episode_id])
        ):
            raise ValueError("fold assignment binding mismatch")
    fold_counts = Counter(row.fold for row in assignments)
    if set(fold_counts) != set(range(folds)):
        raise ValueError("an outer fold is empty")
    by_stratum: dict[str, Counter[int]] = defaultdict(Counter)
    for row in assignments:
        by_stratum[row.stratum][row.fold] += 1
    if any(
        max(counts.get(fold, 0) for fold in range(folds))
        - min(counts.get(fold, 0) for fold in range(folds))
        > MAX_STRATUM_EPISODE_SPREAD
        for counts in by_stratum.values()
    ):
        raise ValueError("stratum imbalance exceeds one episode")
    stratum_differences = [
        max(counts.get(fold, 0) for fold in range(folds))
        - min(counts.get(fold, 0) for fold in range(folds))
        for counts in by_stratum.values()
    ]
    family_counts: dict[int, Counter[int]] = defaultdict(Counter)
    pair_counts: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
    length_counts: dict[int, Counter[int]] = defaultdict(Counter)
    for row in assignments:
        episode = episode_index[row.episode_id]
        length_counts[len(episode.incidents)][row.fold] += 1
        for incident in episode.incidents:
            family_counts[incident.family][row.fold] += 1
            pair_counts[(incident.family, incident.failure_mode)][row.fold] += 1

    def maximum_spread(values: Mapping[Any, Counter[int]]) -> int:
        return max(
            max(counts.get(fold, 0) for fold in range(folds))
            - min(counts.get(fold, 0) for fold in range(folds))
            for counts in values.values()
        )

    fold_spread = max(fold_counts.values()) - min(fold_counts.values())
    stratum_spread = max(stratum_differences)
    family_spread = maximum_spread(family_counts)
    family_failure_spread = maximum_spread(pair_counts)
    length_spread = maximum_spread(length_counts)
    if (
        fold_spread > MAX_FOLD_EPISODE_SPREAD
        or stratum_spread > MAX_STRATUM_EPISODE_SPREAD
        or length_spread > MAX_LENGTH_EPISODE_SPREAD
        or family_spread > MAX_FAMILY_INCIDENT_SPREAD
        or family_failure_spread > MAX_FAMILY_FAILURE_INCIDENT_SPREAD
    ):
        raise ValueError("pre-registered fold balance threshold exceeded")
    return {
        "folds": folds,
        "episodes": len(assignments),
        "unique_groups": len(set(groups)),
        "fold_counts": {
            str(fold): fold_counts[fold] for fold in range(folds)
        },
        "stratum_count": len(by_stratum),
        "max_fold_episode_count_difference": fold_spread,
        "max_within_stratum_fold_count_difference": stratum_spread,
        "max_family_incident_count_difference": family_spread,
        "max_family_failure_incident_count_difference": family_failure_spread,
        "episode_length_fold_counts": {
            str(length): {
                str(fold): counts.get(fold, 0) for fold in range(folds)
            }
            for length, counts in sorted(length_counts.items())
        },
        "max_episode_length_count_difference": length_spread,
        "pre_registered_maximums": {
            "fold_episode_spread": MAX_FOLD_EPISODE_SPREAD,
            "stratum_episode_spread": MAX_STRATUM_EPISODE_SPREAD,
            "length_episode_spread": MAX_LENGTH_EPISODE_SPREAD,
            "family_incident_spread": MAX_FAMILY_INCIDENT_SPREAD,
            "family_failure_incident_spread": MAX_FAMILY_FAILURE_INCIDENT_SPREAD,
        },
        "each_episode_exactly_once": True,
        "group_overlap": 0,
    }


def fold_manifest_sha256(assignments: Sequence[FoldAssignment]) -> str:
    return hashlib.sha256(_canonical([
        row.to_dict() for row in assignments
    ]).encode("utf-8")).hexdigest()


def shared_resource_contract(
    bank: LoadedTrainBank,
) -> dict[str, Any]:
    """One immutable resource/cost contract consumed by A8 and PPO."""

    if (
        bank.path != FROZEN_TRAIN_PATH
        or bank.payload_sha256 != EXPECTED_TRAIN_SHA256
        or len(bank.episodes) != EXPECTED_TRAIN_EPISODES
        or _serialized_bank_sha256(bank.episodes) != EXPECTED_TRAIN_SHA256
        or {
            episode.episode_id: _episode_sha256(episode)
            for episode in bank.episodes
        } != bank.episode_sha256
    ):
        raise ValueError("resource contract requires the frozen train bank")
    return _resource_contract_payload()


def _resource_contract_payload() -> dict[str, Any]:
    environment_path = Path(__file__).with_name("long_horizon_env.py")
    environment_source_sha256 = hashlib.sha256(
        environment_path.read_bytes()
    ).hexdigest()
    contract = {
        "schema_version": RESOURCE_TABLE_VERSION,
        "environment_version": A9_ENV_VERSION,
        "environment_source_sha256": environment_source_sha256,
        "train_bank_sha256": EXPECTED_TRAIN_SHA256,
        "episodes": EXPECTED_TRAIN_EPISODES,
        "observation": {
            "shape": [MultiTownLongHorizonEnv.observation_size],
            "policy_visible": (
                "normalized public runtime, candidate, review, prior-action, "
                "failure and budget state only"
            ),
            "oracle_or_correct_action_visible": False,
        },
        "actions": list(ACTION_NAMES),
        "required_sequential_actions": [
            "stop", "delegate", "escalate", "review", "human",
        ],
        "incremental_costs": {
            action.name.lower(): {
                "tokens": ACTION_COSTS[action][0],
                "latency_s": ACTION_COSTS[action][1],
            }
            for action in RLAction
        },
        "constraints": {
            "token_cap": "per-episode frozen bank field",
            "latency_cap_s": "per-episode frozen bank field",
            "action_legality": "one shared environment action_mask",
            "over_budget_calls_masked": True,
            "human_subject_to_shared_budget_mask": True,
        },
        "reward_components": [
            "final_success", "subgoal_progress", "action_cost",
            "invalid_action", "budget_violation", "safety_penalty",
            "tool_failure_recovery", "unnecessary_delegation",
            "human_penalty",
        ],
        "reward_source": "deterministic environment; no LLM-as-judge",
        "consumers": ["A8-long-heuristic", "A9-train-only-PPO"],
        "labels": {
            "correct_action": "private environment transition/reward only",
            "policy_feature": False,
        },
    }
    return contract


def resource_contract_sha256(contract: Mapping[str, Any]) -> str:
    expected = _resource_contract_payload()
    if dict(contract) != expected:
        raise ValueError("resource contract does not exactly match frozen resources")
    return hashlib.sha256(_canonical(contract).encode("utf-8")).hexdigest()
