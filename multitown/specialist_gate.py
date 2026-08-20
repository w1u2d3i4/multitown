"""Audit whether a real-model pool is ready for learned specialist routing."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

from .live_model_env import MeasuredIncidentOutcome, incident_sha256, read_outcomes, sha256_file
from .long_horizon_env import LongHorizonEpisode, read_episode_bank


SCHEMA_VERSION = "multitown-specialist-readiness-v1"


def _source_state(project_root: Path) -> tuple[str | None, bool | None]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project_root, text=True,
        capture_output=True, check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=project_root, text=True,
        capture_output=True, check=False,
    )
    return (
        revision.stdout.strip() if revision.returncode == 0 else None,
        bool(status.stdout.strip()) if status.returncode == 0 else None,
    )


def _correct(outcome: MeasuredIncidentOutcome, role: str) -> bool:
    call = getattr(outcome, role)
    if not call.valid:
        return False
    if role == "reviewer":
        expected = [index == outcome.correct_action for index in range(4)]
        return call.parsed == expected
    return call.parsed == outcome.correct_action


def _fraction(count: int, total: int) -> float:
    return count / total if total else 0.0


def _model_locks(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    locks = config.get("model_files")
    if not isinstance(locks, dict) or set(locks) != {"weak", "strong", "reviewer"}:
        raise ValueError("collection config must lock weak, strong, and reviewer model files")
    normalized = {}
    for role, value in locks.items():
        if not isinstance(value, dict) or not isinstance(value.get("sha256"), str):
            raise ValueError(f"missing model SHA-256 for {role}")
        normalized[role] = {
            "sha256": value["sha256"],
            "bytes": int(value["bytes"]),
        }
    return normalized


def build_readiness_report(
    episodes: list[LongHorizonEpisode],
    outcomes: list[MeasuredIncidentOutcome],
    collection_config: dict[str, Any],
    *,
    min_incidents: int = 500,
    min_valid_rate: float = 0.99,
    min_unique_rate: float = 0.05,
    min_union_gain: float = 0.05,
    family_win_margin: float = 0.02,
) -> dict[str, Any]:
    """Build a train-only, pre-RL readiness gate for specialist routing."""

    if min_incidents <= 0:
        raise ValueError("min_incidents must be positive")
    for name, value in {
        "min_valid_rate": min_valid_rate,
        "min_unique_rate": min_unique_rate,
        "min_union_gain": min_union_gain,
        "family_win_margin": family_win_margin,
    }.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")

    episode_index = {episode.episode_id: episode for episode in episodes}
    if len(episode_index) != len(episodes):
        raise ValueError("duplicate episode ids in bank")
    if not outcomes:
        raise ValueError("no outcomes")

    family_rows: dict[int, list[tuple[bool, bool, bool]]] = defaultdict(list)
    role_valid = {role: 0 for role in ("weak", "strong", "reviewer")}
    weak_only = strong_only = both = neither = 0
    for outcome in outcomes:
        episode = episode_index.get(outcome.episode_id)
        if episode is None or not 0 <= outcome.incident_index < len(episode.incidents):
            raise ValueError(f"outcome is outside bank: {outcome.key}")
        incident = episode.incidents[outcome.incident_index]
        if outcome.incident_sha256 != incident_sha256(incident):
            raise ValueError(f"incident hash mismatch: {outcome.key}")
        if outcome.correct_action != incident.correct_action:
            raise ValueError(f"evaluator mismatch: {outcome.key}")
        values = tuple(_correct(outcome, role) for role in ("weak", "strong", "reviewer"))
        for role in role_valid:
            role_valid[role] += int(getattr(outcome, role).valid)
        weak_ok, strong_ok, reviewer_ok = values
        both += int(weak_ok and strong_ok)
        weak_only += int(weak_ok and not strong_ok)
        strong_only += int(strong_ok and not weak_ok)
        neither += int(not weak_ok and not strong_ok)
        family_rows[incident.family].append((weak_ok, strong_ok, reviewer_ok))

    total = len(outcomes)
    weak_correct = both + weak_only
    strong_correct = both + strong_only
    best_single = max(weak_correct, strong_correct) / total
    union_accuracy = (both + weak_only + strong_only) / total
    models = _model_locks(collection_config)

    families: dict[str, Any] = {}
    weak_family_wins = strong_family_wins = 0
    for family, rows in sorted(family_rows.items()):
        count = len(rows)
        weak_accuracy = sum(row[0] for row in rows) / count
        strong_accuracy = sum(row[1] for row in rows) / count
        reviewer_accuracy = sum(row[2] for row in rows) / count
        delta = weak_accuracy - strong_accuracy
        winner = "tie"
        if delta >= family_win_margin:
            winner = "weak"
            weak_family_wins += 1
        elif delta <= -family_win_margin:
            winner = "strong"
            strong_family_wins += 1
        families[str(family)] = {
            "incidents": count,
            "weak_accuracy": weak_accuracy,
            "strong_accuracy": strong_accuracy,
            "reviewer_accuracy": reviewer_accuracy,
            "weak_minus_strong": delta,
            "winner_at_margin": winner,
        }

    valid_rates = {role: count / total for role, count in role_valid.items()}
    reviewer_accuracy = sum(_correct(row, "reviewer") for row in outcomes) / total
    checks = {
        "enough_incidents": total >= min_incidents,
        "all_roles_valid": min(valid_rates.values()) >= min_valid_rate,
        "weak_has_unique_mass": weak_only / total >= min_unique_rate,
        "strong_has_unique_mass": strong_only / total >= min_unique_rate,
        "union_beats_best_single": union_accuracy - best_single >= min_union_gain,
        "both_specialists_win_a_family": weak_family_wins >= 1 and strong_family_wins >= 1,
        "reviewer_model_is_independent": models["reviewer"]["sha256"] not in {
            models["weak"]["sha256"], models["strong"]["sha256"],
        },
        "reviewer_matches_best_worker": reviewer_accuracy >= best_single,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": "train-only readiness audit; no dev or test selection",
        "incidents": total,
        "thresholds": {
            "min_incidents": min_incidents,
            "min_valid_rate": min_valid_rate,
            "min_unique_rate": min_unique_rate,
            "min_union_gain": min_union_gain,
            "family_win_margin": family_win_margin,
        },
        "model_locks": models,
        "valid_rate": valid_rates,
        "overall": {
            "weak_accuracy": weak_correct / total,
            "strong_accuracy": strong_correct / total,
            "reviewer_accuracy": reviewer_accuracy,
            "best_single_accuracy": best_single,
            "union_oracle_accuracy": union_accuracy,
            "union_gain_over_best_single": union_accuracy - best_single,
            "both_correct_rate": both / total,
            "weak_only_correct_rate": weak_only / total,
            "strong_only_correct_rate": strong_only / total,
            "neither_correct_rate": neither / total,
        },
        "families": families,
        "family_wins": {"weak": weak_family_wins, "strong": strong_family_wins},
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--outcomes", required=True)
    parser.add_argument("--collection-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-incidents", type=int, default=500)
    parser.add_argument("--min-valid-rate", type=float, default=0.99)
    parser.add_argument("--min-unique-rate", type=float, default=0.05)
    parser.add_argument("--min-union-gain", type=float, default=0.05)
    parser.add_argument("--family-win-margin", type=float, default=0.02)
    args = parser.parse_args()

    bank = Path(args.bank)
    outcomes_path = Path(args.outcomes)
    config_path = Path(args.collection_config)
    project_root = Path(__file__).resolve().parents[1]
    source_revision, source_dirty = _source_state(project_root)
    report = build_readiness_report(
        read_episode_bank(bank), read_outcomes(outcomes_path),
        json.loads(config_path.read_text(encoding="utf-8")),
        min_incidents=args.min_incidents,
        min_valid_rate=args.min_valid_rate,
        min_unique_rate=args.min_unique_rate,
        min_union_gain=args.min_union_gain,
        family_win_margin=args.family_win_margin,
    )
    report["input_sha256"] = {
        "bank": sha256_file(bank),
        "outcomes": sha256_file(outcomes_path),
        "collection_config": sha256_file(config_path),
    }
    report["source_revision"] = source_revision
    report["source_dirty_at_start"] = source_dirty
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output.resolve()),
        "incidents": report["incidents"],
        "passed": report["passed"],
        "failed_checks": [name for name, passed in report["checks"].items() if not passed],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
