"""Risk-calibrated policy improvement for the MultiTown controller.

A26 is deliberately narrower than full sequential Agentic RL.  It learns the
worker-routing map from train episodes and selects one review-skipping threshold
on a calibration split.  The selected controller is frozen before an OOD test
bank is generated.  Low-confidence states retain the review/escalation path.

The module also defines a v2 synthetic environment generator.  Unlike the
historical long-horizon fixture, its correct action is random conditional on
the public task fields, so family/failure/severity cannot reveal the label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .long_horizon_env import (
    ACTION_NAMES,
    CANDIDATE_COUNT,
    HELD_OUT_COMBINATIONS,
    IncidentSpec,
    LongHorizonEpisode,
    MultiTownLongHorizonEnv,
    RLAction,
    a8_heuristic_policy,
    run_policy,
    summarize_results,
)

A26_ENV_VERSION = "multitown-a26-nonleaking-long-horizon-v1"
A26_PROTOCOL_VERSION = "multitown-a26-risk-calibrated-router-v1"
A26_RESULT_VERSION = "multitown-a26-result-v1"
TRAIN_EPISODES = 3000
CALIBRATION_EPISODES = 500
TEST_EPISODES = 1000
TRAIN_OFFSET = 41_000_000
CALIBRATION_OFFSET = 42_000_000
TEST_OFFSET = 43_000_000
CALIBRATION_THRESHOLDS = (0.0, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 1.01)
UNSAFE_MARGIN = 0.02
WRONG_PER_INCIDENT_MARGIN = 0.01
BOOTSTRAP_SEED = 20260824


@dataclass(frozen=True)
class RouterConfig:
    """Frozen public-state policy parameters."""

    specialist_by_family: tuple[int, ...]
    unreviewed_agreement_severity: float

    def __post_init__(self) -> None:
        if (
            len(self.specialist_by_family) != 4
            or any(
                action not in (int(RLAction.DELEGATE), int(RLAction.ESCALATE))
                for action in self.specialist_by_family
            )
            or not math.isfinite(self.unreviewed_agreement_severity)
            or not 0.0 <= self.unreviewed_agreement_severity <= 1.01
        ):
            raise ValueError("invalid A26 router configuration")

    def to_dict(self) -> dict[str, Any]:
        return {
            "specialist_by_family": [
                ACTION_NAMES[action] for action in self.specialist_by_family
            ],
            "unreviewed_agreement_severity": self.unreviewed_agreement_severity,
        }


def _wrong_candidate(rng: random.Random, correct: int) -> int:
    return rng.choice([item for item in range(CANDIDATE_COUNT) if item != correct])


def _candidate(rng: random.Random, correct: int, probability: float) -> int:
    return correct if rng.random() < probability else _wrong_candidate(rng, correct)


def generate_a26_episode(seed: int, split: str) -> LongHorizonEpisode:
    """Generate one non-leaking episode for the A26 protocol."""

    if split not in {"train", "dev", "test"}:
        raise ValueError(f"invalid A26 split: {split}")
    rng = random.Random(f"{A26_ENV_VERSION}:{split}:{seed}")
    all_combinations = [(family, mode) for family in range(4) for mode in range(4)]
    if split == "test":
        combinations = list(HELD_OUT_COMBINATIONS)
        tool_failure_probability = 0.24
    else:
        combinations = [
            item for item in all_combinations if item not in HELD_OUT_COMBINATIONS
        ]
        tool_failure_probability = 0.10 if split == "train" else 0.14
    incidents: list[IncidentSpec] = []
    incident_count = rng.randint(4, 7)
    for _ in range(incident_count):
        family, failure_mode = rng.choice(combinations)
        severity = rng.uniform(0.15, 1.0)
        # This draw is intentionally independent of the public task fields.
        correct = rng.randrange(CANDIDATE_COUNT)
        sensor_accuracy = 0.50 + 0.18 * float(severity < 0.55)
        weak_accuracy = (0.90 if family in {0, 2} else 0.58) - 0.06 * float(
            severity >= 0.8
        )
        strong_accuracy = (0.72 if family in {0, 2} else 0.90) - 0.04 * float(
            split == "test"
        )
        reviewer_pass = tuple(
            (rng.random() < 0.91) if candidate == correct else (rng.random() < 0.08)
            for candidate in range(CANDIDATE_COUNT)
        )
        failure_candidates = (
            RLAction.OBSERVE,
            RLAction.DELEGATE,
            RLAction.ESCALATE,
            RLAction.CONNECT,
            RLAction.REVIEW,
        )
        fail_first = tuple(
            int(action)
            for action in failure_candidates
            if rng.random() < tool_failure_probability
        )
        incidents.append(
            IncidentSpec(
                family=family,
                failure_mode=failure_mode,
                severity=severity,
                correct_action=correct,
                sensor_candidate=_candidate(rng, correct, sensor_accuracy),
                weak_candidate=_candidate(rng, correct, weak_accuracy),
                strong_candidate=_candidate(rng, correct, strong_accuracy),
                reviewer_pass=reviewer_pass,
                fail_first_actions=fail_first,
            )
        )
    max_steps = min(50, max(20, incident_count * 6 + 6))
    return LongHorizonEpisode(
        episode_id=f"a26-{split}-{seed:08d}",
        split=split,
        seed=seed,
        token_budget=incident_count * 650,
        latency_budget_s=incident_count * 1.65,
        max_steps=max_steps,
        incidents=tuple(incidents),
        schema_version=A26_ENV_VERSION,
    )


def a26_episode_bank(
    split: str, count: int, *, seed_offset: int
) -> list[LongHorizonEpisode]:
    if type(count) is not int or count <= 0:
        raise ValueError("A26 episode count must be positive")
    return [generate_a26_episode(seed_offset + index, split) for index in range(count)]


def bank_sha256(episodes: Sequence[LongHorizonEpisode]) -> str:
    digest = hashlib.sha256()
    for episode in episodes:
        payload = json.dumps(
            episode.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        digest.update(payload)
        digest.update(b"\n")
    return digest.hexdigest()


def learn_specialist_map(episodes: Sequence[LongHorizonEpisode]) -> tuple[int, ...]:
    """Fit the family-level weak/strong router using train labels only."""

    counts: dict[int, dict[int, list[int]]] = defaultdict(
        lambda: {
            int(RLAction.DELEGATE): [0, 0],
            int(RLAction.ESCALATE): [0, 0],
        }
    )
    for episode in episodes:
        if episode.split != "train":
            raise ValueError("specialist routing may only fit on train episodes")
        for incident in episode.incidents:
            for action, candidate in (
                (int(RLAction.DELEGATE), incident.weak_candidate),
                (int(RLAction.ESCALATE), incident.strong_candidate),
            ):
                counts[incident.family][action][0] += int(
                    candidate == incident.correct_action
                )
                counts[incident.family][action][1] += 1
    if set(counts) != set(range(4)):
        raise RuntimeError("A26 training bank does not cover every family")
    result = []
    for family in range(4):
        # Deterministic tie-break prefers the cheaper weak specialist.
        result.append(
            max(
                (int(RLAction.DELEGATE), int(RLAction.ESCALATE)),
                key=lambda action: (
                    counts[family][action][0] / counts[family][action][1],
                    action == int(RLAction.DELEGATE),
                ),
            )
        )
    return tuple(result)


def _candidate_for(env: MultiTownLongHorizonEnv, action: RLAction) -> int | None:
    if action == RLAction.DELEGATE:
        return env.weak_candidate
    if action == RLAction.ESCALATE:
        return env.strong_candidate
    raise ValueError("A26 specialist action must be delegate or escalate")


def make_router_policy(config: RouterConfig) -> Callable[..., int]:
    """Create the deterministic risk-calibrated sequential controller."""

    def policy(
        env: MultiTownLongHorizonEnv, observation: np.ndarray, mask: np.ndarray
    ) -> int:
        if env.incident is None:
            return int(RLAction.STOP)
        if (
            env.last_tool_failed
            and env.last_action is not None
            and mask[env.last_action]
        ):
            return env.last_action

        family = int(np.argmax(observation[:4]))
        severity = float(observation[8])
        preferred = RLAction(config.specialist_by_family[family])
        alternate = (
            RLAction.ESCALATE if preferred == RLAction.DELEGATE else RLAction.DELEGATE
        )
        preferred_candidate = _candidate_for(env, preferred)
        alternate_candidate = _candidate_for(env, alternate)

        if not env.observed and mask[RLAction.OBSERVE]:
            return int(RLAction.OBSERVE)
        if preferred_candidate is None and mask[preferred]:
            return int(preferred)

        # Before the first CONNECT, sensor/specialist agreement is observable.
        # Only calibrated low-severity agreements may skip REVIEW.
        if not env.connected and alternate_candidate is None:
            may_skip_review = (
                env.review_state == 0
                and env.current_candidate == preferred_candidate
                and severity < config.unreviewed_agreement_severity
            )
            if may_skip_review and mask[RLAction.EXECUTE]:
                return int(RLAction.EXECUTE)
            if env.review_state == 0 and mask[RLAction.REVIEW]:
                return int(RLAction.REVIEW)
            if env.review_state == 1 and mask[RLAction.EXECUTE]:
                return int(RLAction.EXECUTE)
            if env.review_state == 2 and mask[RLAction.CONNECT]:
                return int(RLAction.CONNECT)

        # A rejected sensor is replaced by the learned preferred specialist.
        if env.connected:
            if env.review_state == 0 and mask[RLAction.REVIEW]:
                return int(RLAction.REVIEW)
            if env.review_state == 1 and mask[RLAction.EXECUTE]:
                return int(RLAction.EXECUTE)
            if (
                env.review_state == 2
                and alternate_candidate is None
                and mask[alternate]
            ):
                return int(alternate)

        # A second worker is only called after both cheaper evidence paths fail.
        if (
            alternate_candidate is not None
            and not env.connected
            and mask[RLAction.CONNECT]
        ):
            return int(RLAction.CONNECT)
        if env.review_state == 0 and mask[RLAction.REVIEW]:
            return int(RLAction.REVIEW)
        if env.review_state == 1 and mask[RLAction.EXECUTE]:
            return int(RLAction.EXECUTE)
        if mask[RLAction.HUMAN]:
            return int(RLAction.HUMAN)
        return int(RLAction.STOP)

    return policy


def _evaluate(
    episodes: Sequence[LongHorizonEpisode], policy: Callable[..., int]
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows = [run_policy(episode, policy) for episode in episodes]
    summary = summarize_results(rows)
    wrong = [
        sum(
            int(float(step["reward"]["safety_penalty"]) < 0.0)
            for step in row["trajectory"]
        )
        for row in rows
    ]
    summary.update(
        {
            "unsafe_episode_rate": sum(value > 0 for value in wrong) / len(rows),
            "wrong_executions_per_incident": sum(wrong)
            / sum(int(row["incidents"]) for row in rows),
        }
    )
    return rows, {
        key: float(value)
        for key, value in summary.items()
        if isinstance(value, (int, float))
    }


def _paired_bootstrap(
    left: Sequence[float], right: Sequence[float], *, iterations: int
) -> dict[str, float]:
    if len(left) != len(right) or not left or iterations <= 0:
        raise ValueError("invalid A26 paired bootstrap input")
    differences = np.asarray(left, dtype=np.float64) - np.asarray(
        right, dtype=np.float64
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(differences), size=(iterations, len(differences)))
    estimates = differences[draws].mean(axis=1)
    return {
        "point": float(differences.mean()),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
        "iterations": float(iterations),
    }


def _metric_vector(rows: Sequence[Mapping[str, Any]], metric: str) -> list[float]:
    if metric == "unsafe_episode":
        return [
            float(
                any(
                    float(step["reward"]["safety_penalty"]) < 0.0
                    for step in row["trajectory"]
                )
            )
            for row in rows
        ]
    return [float(row[metric]) for row in rows]


def _source_revision() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "commit": revision.stdout.strip() if revision.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _plot_frontier(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    x = [float(row["unsafe_episode_rate"]) for row in rows]
    y = [float(row["episode_success_rate"]) for row in rows]
    labels = [str(row["threshold"]) for row in rows]
    axis.plot(x, y, marker="o", color="#2563eb")
    for x_value, y_value, label in zip(x, y, labels):
        axis.annotate(
            label, (x_value, y_value), xytext=(4, 4), textcoords="offset points"
        )
    axis.set_xlabel("unsafe episode rate")
    axis.set_ylabel("autonomous episode success")
    axis.set_title("A26 calibration frontier (threshold labels)")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run_a26(output: Path, *, bootstrap_iterations: int = 5000) -> dict[str, Any]:
    """Run train/calibration/frozen-OOD evaluation exactly once."""

    if output.exists():
        raise FileExistsError(f"A26 output already exists: {output}")
    output.mkdir(parents=True, mode=0o700)

    train = a26_episode_bank("train", TRAIN_EPISODES, seed_offset=TRAIN_OFFSET)
    calibration = a26_episode_bank(
        "dev", CALIBRATION_EPISODES, seed_offset=CALIBRATION_OFFSET
    )
    specialist_map = learn_specialist_map(train)
    calibration_a8_rows, calibration_a8 = _evaluate(calibration, a8_heuristic_policy)
    del calibration_a8_rows

    frontier = []
    for threshold in CALIBRATION_THRESHOLDS:
        config = RouterConfig(specialist_map, threshold)
        _, summary = _evaluate(calibration, make_router_policy(config))
        feasible = bool(
            summary["unsafe_episode_rate"]
            <= calibration_a8["unsafe_episode_rate"] + UNSAFE_MARGIN
            and summary["wrong_executions_per_incident"]
            <= calibration_a8["wrong_executions_per_incident"]
            + WRONG_PER_INCIDENT_MARGIN
            and summary["episode_success_rate"]
            >= calibration_a8["episode_success_rate"]
        )
        frontier.append({"threshold": threshold, "feasible": feasible, **summary})
    feasible_rows = [row for row in frontier if row["feasible"]]
    if feasible_rows:
        selected = max(
            feasible_rows,
            key=lambda row: (
                row["episode_success_rate"],
                -row["tokens_per_episode"],
                row["mean_return"],
            ),
        )
    else:
        selected = frontier[0]
    selected_config = RouterConfig(specialist_map, float(selected["threshold"]))
    lock = {
        "schema_version": "multitown-a26-selection-lock-v1",
        "protocol_version": A26_PROTOCOL_VERSION,
        "source": _source_revision(),
        "train_bank_sha256": bank_sha256(train),
        "calibration_bank_sha256": bank_sha256(calibration),
        "test_bank_generated": False,
        "selection_rule": (
            "maximize calibration autonomous success among A8-relative safety-"
            "feasible thresholds; tie-break fewer tokens then higher return"
        ),
        "safety_margins": {
            "unsafe_episode": UNSAFE_MARGIN,
            "wrong_execution_per_incident": WRONG_PER_INCIDENT_MARGIN,
        },
        "baseline": calibration_a8,
        "selected_config": selected_config.to_dict(),
    }
    _write_json(output / "selection-lock.json", lock)
    _plot_frontier(frontier, output / "calibration-frontier.png")

    # The OOD test bank is not constructed until the selection lock exists.
    test = a26_episode_bank("test", TEST_EPISODES, seed_offset=TEST_OFFSET)
    a8_rows, a8_summary = _evaluate(test, a8_heuristic_policy)
    a26_rows, a26_summary = _evaluate(test, make_router_policy(selected_config))
    paired = {
        metric: _paired_bootstrap(
            _metric_vector(a26_rows, metric),
            _metric_vector(a8_rows, metric),
            iterations=bootstrap_iterations,
        )
        for metric in ("episode_success", "unsafe_episode", "tokens_used", "return")
    }
    gates = {
        "success_difference_ci95_low_positive": paired["episode_success"]["ci95_low"]
        > 0.0,
        "unsafe_within_a8_plus_margin": a26_summary["unsafe_episode_rate"]
        <= a8_summary["unsafe_episode_rate"] + UNSAFE_MARGIN,
        "wrong_per_incident_within_a8_plus_margin": a26_summary[
            "wrong_executions_per_incident"
        ]
        <= a8_summary["wrong_executions_per_incident"] + WRONG_PER_INCIDENT_MARGIN,
        "zero_invalid_actions": a26_summary["invalid_actions"] == 0,
        "zero_budget_violations": a26_summary["budget_violations"] == 0,
    }
    result = {
        "schema_version": A26_RESULT_VERSION,
        "protocol_version": A26_PROTOCOL_VERSION,
        "method": "risk-calibrated specialist router with conservative review fallback",
        "method_class": "constrained contextual policy improvement; not full Agentic RL",
        "agentic_rl_claim": False,
        "environment": {
            "schema_version": A26_ENV_VERSION,
            "correct_action_independent_of_public_task_fields": True,
            "test_is_family_failure_combination_ood": True,
        },
        "counts": {
            "train": len(train),
            "calibration": len(calibration),
            "test": len(test),
        },
        "bank_sha256": {
            "train": lock["train_bank_sha256"],
            "calibration": lock["calibration_bank_sha256"],
            "test": bank_sha256(test),
        },
        "selected_config": selected_config.to_dict(),
        "calibration_baseline": calibration_a8,
        "calibration_frontier": frontier,
        "test": {"A8": a8_summary, "A26": a26_summary},
        "paired_A26_minus_A8": paired,
        "gates": gates,
        "validated_improvement": all(gates.values()),
        "claim_boundary": {
            "synthetic_fixture_only": True,
            "general_multi_agent_superiority": False,
            "language_model_weights_trained": False,
            "raw_trajectories_for_publication": False,
        },
    }
    _write_json(output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    args = parser.parse_args()
    result = run_a26(args.output, bootstrap_iterations=args.bootstrap_iterations)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "selected_config": result["selected_config"],
                "A8": result["test"]["A8"],
                "A26": result["test"]["A26"],
                "gates": result["gates"],
                "validated_improvement": result["validated_improvement"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
