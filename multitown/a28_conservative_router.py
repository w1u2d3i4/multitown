"""A28 agreement-gated specialist-first conservative controller.

A28 preserves the A26 non-leaking environment and split discipline, but fixes
its unsafe disagreement path: when sensor and preferred specialist disagree,
the controller connects to the higher-confidence specialist before asking the
reviewer.  It never executes the lower-accuracy sensor solely because of one
positive review.  The specialist map is fitted on train and the agreement gate
is selected on calibration before an independent confirmation bank is built.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .a26_safe_router import (
    CALIBRATION_THRESHOLDS,
    UNSAFE_MARGIN,
    WRONG_PER_INCIDENT_MARGIN,
    RouterConfig,
    _evaluate,
    _metric_vector,
    _paired_bootstrap,
    _plot_frontier,
    _source_revision,
    _write_json,
    a26_episode_bank,
    bank_sha256,
    learn_specialist_map,
)
from .long_horizon_env import (
    MultiTownLongHorizonEnv,
    RLAction,
    a8_heuristic_policy,
)

A28_PROTOCOL_VERSION = "multitown-a28-conservative-router-v1"
A28_RESULT_VERSION = "multitown-a28-result-v1"
TRAIN_EPISODES = 3000
CALIBRATION_EPISODES = 500
CONFIRMATION_EPISODES = 1000
TRAIN_OFFSET = 51_000_000
CALIBRATION_OFFSET = 52_000_000
CONFIRMATION_OFFSET = 53_000_000


def _candidate_for(env: MultiTownLongHorizonEnv, action: RLAction) -> int | None:
    if action == RLAction.DELEGATE:
        return env.weak_candidate
    if action == RLAction.ESCALATE:
        return env.strong_candidate
    raise ValueError("A28 specialist action must be delegate or escalate")


def make_conservative_router_policy(config: RouterConfig) -> Callable[..., int]:
    """Build the frozen A28 controller without reading private correctness."""

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

        if not env.connected and alternate_candidate is None:
            agreement = env.current_candidate == preferred_candidate
            may_skip_review = (
                agreement
                and env.review_state == 0
                and severity < config.unreviewed_agreement_severity
            )
            if may_skip_review and mask[RLAction.EXECUTE]:
                return int(RLAction.EXECUTE)
            # The important A28 change: on disagreement, move to the learned
            # specialist before review instead of reviewing the noisy sensor.
            if mask[RLAction.CONNECT]:
                return int(RLAction.CONNECT)

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


def run_a28(output: Path, *, bootstrap_iterations: int = 5000) -> dict[str, Any]:
    """Run the pre-registered A28 train/calibration/confirmation protocol."""

    if output.exists():
        raise FileExistsError(f"A28 output already exists: {output}")
    output.mkdir(parents=True, mode=0o700)

    train = a26_episode_bank("train", TRAIN_EPISODES, seed_offset=TRAIN_OFFSET)
    calibration = a26_episode_bank(
        "dev", CALIBRATION_EPISODES, seed_offset=CALIBRATION_OFFSET
    )
    specialist_map = learn_specialist_map(train)
    _, calibration_a8 = _evaluate(calibration, a8_heuristic_policy)
    frontier = []
    for threshold in CALIBRATION_THRESHOLDS:
        config = RouterConfig(specialist_map, threshold)
        _, summary = _evaluate(calibration, make_conservative_router_policy(config))
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
    selected = (
        max(
            feasible_rows,
            key=lambda row: (
                row["episode_success_rate"],
                -row["tokens_per_episode"],
                row["mean_return"],
            ),
        )
        if feasible_rows
        else frontier[0]
    )
    selected_config = RouterConfig(specialist_map, float(selected["threshold"]))
    lock = {
        "schema_version": "multitown-a28-selection-lock-v1",
        "protocol_version": A28_PROTOCOL_VERSION,
        "source": _source_revision(),
        "predecessor": {
            "A26_status": "NEGATIVE_RESULT",
            "A26_test_used_for_A28_threshold_selection": False,
            "change": "connect-to-preferred-specialist-before-review-on-disagreement",
        },
        "train_bank_sha256": bank_sha256(train),
        "calibration_bank_sha256": bank_sha256(calibration),
        "confirmation_bank_generated": False,
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

    # The independent confirmation bank is constructed only after the lock.
    confirmation = a26_episode_bank(
        "test", CONFIRMATION_EPISODES, seed_offset=CONFIRMATION_OFFSET
    )
    a8_rows, a8_summary = _evaluate(confirmation, a8_heuristic_policy)
    a28_rows, a28_summary = _evaluate(
        confirmation, make_conservative_router_policy(selected_config)
    )
    paired = {
        metric: _paired_bootstrap(
            _metric_vector(a28_rows, metric),
            _metric_vector(a8_rows, metric),
            iterations=bootstrap_iterations,
        )
        for metric in ("episode_success", "unsafe_episode", "tokens_used", "return")
    }
    gates = {
        "success_difference_ci95_low_positive": paired["episode_success"]["ci95_low"]
        > 0.0,
        "unsafe_within_a8_plus_margin": a28_summary["unsafe_episode_rate"]
        <= a8_summary["unsafe_episode_rate"] + UNSAFE_MARGIN,
        "wrong_per_incident_within_a8_plus_margin": a28_summary[
            "wrong_executions_per_incident"
        ]
        <= a8_summary["wrong_executions_per_incident"] + WRONG_PER_INCIDENT_MARGIN,
        "tokens_no_greater_than_a8": a28_summary["tokens_per_episode"]
        <= a8_summary["tokens_per_episode"],
        "zero_invalid_actions": a28_summary["invalid_actions"] == 0,
        "zero_budget_violations": a28_summary["budget_violations"] == 0,
    }
    result = {
        "schema_version": A28_RESULT_VERSION,
        "protocol_version": A28_PROTOCOL_VERSION,
        "method": "agreement-gated specialist-first conservative router",
        "method_class": "constrained contextual policy improvement; not full Agentic RL",
        "agentic_rl_claim": False,
        "counts": {
            "train": len(train),
            "calibration": len(calibration),
            "confirmation": len(confirmation),
        },
        "bank_sha256": {
            "train": lock["train_bank_sha256"],
            "calibration": lock["calibration_bank_sha256"],
            "confirmation": bank_sha256(confirmation),
        },
        "selected_config": selected_config.to_dict(),
        "calibration_baseline": calibration_a8,
        "calibration_frontier": frontier,
        "confirmation": {"A8": a8_summary, "A28": a28_summary},
        "paired_A28_minus_A8": paired,
        "gates": gates,
        "validated_improvement": all(gates.values()),
        "claim_boundary": {
            "synthetic_fixture_only": True,
            "general_multi_agent_superiority": False,
            "full_agentic_rl": False,
            "language_model_weights_trained": False,
        },
    }
    _write_json(output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    args = parser.parse_args()
    result = run_a28(args.output, bootstrap_iterations=args.bootstrap_iterations)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "selected_config": result["selected_config"],
                "A8": result["confirmation"]["A8"],
                "A28": result["confirmation"]["A28"],
                "gates": result["gates"],
                "validated_improvement": result["validated_improvement"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
