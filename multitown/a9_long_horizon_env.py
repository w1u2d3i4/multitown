"""Leakage-resistant train-only episode generator for A9 PPO OOF evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .long_horizon_env import (
    CANDIDATE_COUNT, HELD_OUT_COMBINATIONS, IncidentSpec, LongHorizonEpisode,
    RLAction,
)


A9_ENV_VERSION = "multitown-a9-long-horizon-pomdp-v2"
A9_BANK_VERSION = "multitown-a9-train-only-bank-v2"
A9_LEAKAGE_AUDIT_VERSION = "multitown-a9-precall-label-leakage-audit-v1"
DEFAULT_EPISODES = 3000
DEFAULT_SEED_OFFSET = 40_000_000


def _wrong_candidate(rng: random.Random, correct: int) -> int:
    return rng.choice([item for item in range(CANDIDATE_COUNT) if item != correct])


def _noisy_candidate(
    rng: random.Random, correct: int, probability: float,
) -> int:
    return correct if rng.random() < probability else _wrong_candidate(rng, correct)


def generate_train_episode(seed: int) -> LongHorizonEpisode:
    """Generate one episode without a direct public-state-to-label formula.

    A dedicated private RNG produces the correct action.  The public/runtime RNG
    produces observable covariates and tool failures.  Worker candidates and
    reviewer outcomes are deliberately label-dependent because paying for those
    observations is the organization decision the controller must learn.
    """

    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    public_rng = random.Random(f"{A9_ENV_VERSION}:public:train:{seed}")
    private_rng = random.Random(f"{A9_ENV_VERSION}:private-label:train:{seed}")
    outcome_rng = random.Random(f"{A9_ENV_VERSION}:worker-outcome:train:{seed}")
    combinations = [
        (family, failure_mode)
        for family in range(4) for failure_mode in range(4)
        if (family, failure_mode) not in HELD_OUT_COMBINATIONS
    ]
    incident_count = public_rng.randint(4, 7)
    incidents: list[IncidentSpec] = []
    for _ in range(incident_count):
        family, failure_mode = public_rng.choice(combinations)
        severity = public_rng.uniform(0.15, 1.0)
        correct = private_rng.randrange(CANDIDATE_COUNT)
        sensor_accuracy = 0.50 + 0.18 * float(severity < 0.55)
        weak_accuracy = (
            0.90 if family in {0, 2} else 0.58
        ) - 0.06 * float(severity >= 0.8)
        strong_accuracy = 0.72 if family in {0, 2} else 0.90
        reviewer_pass = tuple(
            (outcome_rng.random() < 0.91)
            if candidate == correct else (outcome_rng.random() < 0.08)
            for candidate in range(CANDIDATE_COUNT)
        )
        failure_candidates = (
            RLAction.OBSERVE, RLAction.DELEGATE, RLAction.ESCALATE,
            RLAction.CONNECT, RLAction.REVIEW,
        )
        fail_first = tuple(
            int(action) for action in failure_candidates
            if public_rng.random() < 0.10
        )
        incidents.append(IncidentSpec(
            family=family,
            failure_mode=failure_mode,
            severity=severity,
            correct_action=correct,
            sensor_candidate=_noisy_candidate(
                outcome_rng, correct, sensor_accuracy,
            ),
            weak_candidate=_noisy_candidate(
                outcome_rng, correct, weak_accuracy,
            ),
            strong_candidate=_noisy_candidate(
                outcome_rng, correct, strong_accuracy,
            ),
            reviewer_pass=reviewer_pass,
            fail_first_actions=fail_first,
        ))
    max_steps = min(50, max(20, incident_count * 6 + 6))
    return LongHorizonEpisode(
        episode_id=f"a9-lh-train-{seed:08d}",
        split="train",
        seed=seed,
        token_budget=incident_count * 650,
        latency_budget_s=incident_count * 1.65,
        max_steps=max_steps,
        incidents=tuple(incidents),
        schema_version=A9_ENV_VERSION,
    )


def generate_train_bank(
    count: int = DEFAULT_EPISODES, *, seed_offset: int = DEFAULT_SEED_OFFSET,
) -> list[LongHorizonEpisode]:
    if type(count) is not int or count <= 0:
        raise ValueError("episode count must be a positive integer")
    return [generate_train_episode(seed_offset + index) for index in range(count)]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tracked_source_state(root: Path) -> dict[str, str]:
    sources = (
        "multitown/a9_long_horizon_env.py",
        "multitown/long_horizon_env.py",
    )
    result: dict[str, str] = {}
    for relative in sources:
        path = root / relative
        head = subprocess.run(
            ["git", "show", f"HEAD:{relative}"], cwd=root,
            capture_output=True, check=False,
        )
        if head.returncode or not path.is_file() or path.read_bytes() != head.stdout:
            raise RuntimeError(f"executed source differs from HEAD: {relative}")
        result[relative] = hashlib.sha256(head.stdout).hexdigest()
    return result


def _precall_features(
    episodes: Sequence[LongHorizonEpisode],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: list[list[float]] = []
    labels: list[int] = []
    groups: list[int] = []
    for episode_index, episode in enumerate(episodes):
        for incident_index, incident in enumerate(episode.incidents):
            features.append([
                *[float(incident.family == family) for family in range(4)],
                *[
                    float(incident.failure_mode == mode)
                    for mode in range(4)
                ],
                incident.severity,
                incident_index / len(episode.incidents),
                float(len(episode.incidents)),
                float(episode.token_budget),
                episode.latency_budget_s,
                float(episode.max_steps),
            ])
            labels.append(incident.correct_action)
            groups.append(episode_index)
    return (
        np.asarray(features, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
        np.asarray(groups, dtype=np.int64),
    )


def precall_label_leakage_audit(
    episodes: Sequence[LongHorizonEpisode], *, folds: int = 5,
) -> dict[str, Any]:
    """Probe only pre-call policy-visible state for a label shortcut."""

    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.model_selection import GroupKFold

    if folds < 2 or len(episodes) < folds:
        raise ValueError("insufficient episodes for grouped leakage audit")
    features, labels, groups = _precall_features(episodes)
    predictions = np.full(labels.shape, -1, dtype=np.int64)
    for train_indices, held_indices in GroupKFold(n_splits=folds).split(
        features, labels, groups,
    ):
        model = ExtraTreesClassifier(
            n_estimators=128,
            min_samples_leaf=16,
            max_features="sqrt",
            n_jobs=1,
            random_state=20260813,
        )
        model.fit(features[train_indices], labels[train_indices])
        predictions[held_indices] = model.predict(features[held_indices])
    if np.any(predictions < 0):
        raise RuntimeError("incomplete grouped leakage predictions")
    legacy_formula = np.asarray([
        (incident.family + 2 * incident.failure_mode
         + int(incident.severity >= 0.7)) % CANDIDATE_COUNT
        for episode in episodes for incident in episode.incidents
    ])
    correct_counts = Counter(labels.tolist())
    cell_counts: dict[str, Counter[int]] = defaultdict(Counter)
    for episode in episodes:
        for incident in episode.incidents:
            severity_bin = min(4, int((incident.severity - 0.15) / 0.17))
            cell = (
                f"family={incident.family}|failure_mode={incident.failure_mode}"
                f"|severity_bin={severity_bin}"
            )
            cell_counts[cell][incident.correct_action] += 1
    eligible_cells = [counts for counts in cell_counts.values() if sum(counts.values()) >= 80]
    worst_cell_majority = max(
        (
            max(counts.values()) / sum(counts.values())
            for counts in eligible_cells
        ),
        default=0.0,
    )
    grouped_probe_accuracy = float(np.mean(predictions == labels))
    legacy_formula_accuracy = float(np.mean(legacy_formula == labels))
    gates = {
        "grouped_extra_trees_accuracy_at_most_0_28": grouped_probe_accuracy <= 0.28,
        "legacy_formula_accuracy_at_most_0_28": legacy_formula_accuracy <= 0.28,
        "minimum_correct_action_count_at_least_0_23_fraction": (
            min(correct_counts.values()) / len(labels) >= 0.23
        ),
        "eligible_cell_majority_at_most_0_40": worst_cell_majority <= 0.40,
    }
    return {
        "schema_version": A9_LEAKAGE_AUDIT_VERSION,
        "scope": "train-only pre-call policy-visible covariates; no worker/reviewer outcomes",
        "episodes": len(episodes),
        "incidents": len(labels),
        "grouped_folds": folds,
        "correct_action_counts": {
            str(key): correct_counts[key] for key in range(CANDIDATE_COUNT)
        },
        "grouped_extra_trees_accuracy": grouped_probe_accuracy,
        "legacy_public_formula_accuracy": legacy_formula_accuracy,
        "eligible_public_cells": len(eligible_cells),
        "minimum_public_cell_size": 80,
        "worst_eligible_public_cell_majority_fraction": worst_cell_majority,
        "gates": gates,
        "passed": all(gates.values()),
        "limitations": [
            "Empirical probes do not prove absence of every possible shortcut.",
            "Paid worker and reviewer outcomes intentionally carry label information.",
            "This audit accesses train episodes only.",
        ],
    }


def freeze_train_bank(
    output: Path, *, count: int = DEFAULT_EPISODES,
    seed_offset: int = DEFAULT_SEED_OFFSET,
) -> dict[str, Any]:
    if count != DEFAULT_EPISODES or seed_offset != DEFAULT_SEED_OFFSET:
        raise ValueError(
            "formal A9 freeze requires exactly 3000 episodes at seed offset 40000000"
        )
    output = Path(output)
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, text=True,
        capture_output=True, check=True,
    ).stdout.strip())
    if dirty:
        raise RuntimeError("A9 train bank must be frozen from a clean source checkout")
    source_hashes = _tracked_source_state(root)
    episodes = generate_train_bank(count, seed_offset=seed_offset)
    leakage = precall_label_leakage_audit(episodes)
    if not leakage["passed"]:
        raise RuntimeError("pre-call label leakage gate failed")
    if (
        subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            capture_output=True, check=True,
        ).stdout.strip() != revision
        or _tracked_source_state(root) != source_hashes
    ):
        raise RuntimeError("source changed during A9 train-bank generation")
    output.mkdir(parents=True, exist_ok=False)
    path = output / "train.jsonl"
    with path.open("x", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(
                episode.to_dict(), sort_keys=True, allow_nan=False,
            ) + "\n")
    leakage_path = output / "precall-leakage-audit.json"
    leakage_path.write_text(
        json.dumps(leakage, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": A9_BANK_VERSION,
        "environment_version": A9_ENV_VERSION,
        "split": "train",
        "evaluation_splits_present": False,
        "source_revision": revision,
        "source_dirty_at_start": False,
        "source_sha256": source_hashes,
        "source_stable_during_generation": True,
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
        "seed_offset": seed_offset,
        "files": {
            path.name: {
                "episodes": len(episodes),
                "incidents": sum(len(item.incidents) for item in episodes),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            },
            leakage_path.name: {
                "bytes": leakage_path.stat().st_size,
                "sha256": _sha256(leakage_path),
            },
        },
        "label_generation": (
            "namespace-separated deterministic private RNG stream; no direct "
            "formula from pre-call family, failure-mode, severity, progress "
            "or budget covariates"
        ),
        "shared_runtime": (
            "MultiTownLongHorizonEnv transition, action mask, action costs and rewards"
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path("benchmarks/multitown-a9-long-horizon-v0.2"),
    )
    parser.add_argument("--count", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed-offset", type=int, default=DEFAULT_SEED_OFFSET)
    args = parser.parse_args()
    manifest = freeze_train_bank(
        args.output, count=args.count, seed_offset=args.seed_offset,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
