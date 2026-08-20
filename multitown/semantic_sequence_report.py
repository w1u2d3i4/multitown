"""Evaluate fixed A14 semantic sequence policies without claiming independent episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .masbench_routing import git_state, utc_now, write_json
from .semantic_sequence_env import (
    SemanticSequenceEpisode,
    always_35b_policy,
    always_4b_policy,
    family_review_cascade_policy,
    family_model_policy,
    hindsight_oracle_policy,
    read_episode_bank,
    review_cascade_policy,
    run_policy,
    run_privileged_policy,
)


POLICIES = {
    "always_4b_live": always_4b_policy,
    "budgeted_35b_first_with_4b_fallback": always_35b_policy,
}

POSTHOC_POLICIES = {
    "posthoc_family_model_rule": family_model_policy,
}

PERFECT_VERIFIER_POLICIES = {
    "perfect_verifier_review_cascade": review_cascade_policy,
    "perfect_verifier_family_review_cascade": family_review_cascade_policy,
}


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def evaluate(episodes: list[SemanticSequenceEpisode]) -> dict[str, Any]:
    if not episodes:
        raise ValueError("A14 report requires episodes")
    results = {
        name: [run_policy(episode, policy) for episode in episodes]
        for name, policy in POLICIES.items()
    }
    perfect_verifier_results = {
        name: [run_policy(episode, policy) for episode in episodes]
        for name, policy in PERFECT_VERIFIER_POLICIES.items()
    }
    posthoc_results = {
        name: [run_policy(episode, policy) for episode in episodes]
        for name, policy in POSTHOC_POLICIES.items()
    }
    privileged_results = {
        "privileged_hindsight_output_oracle": [
            run_privileged_policy(episode, hindsight_oracle_policy) for episode in episodes
        ],
    }
    summaries = {}
    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        task_count = len(rows) * len(episodes[0].tasks)
        model_task_ids = {
            (row["episode_id"], event["task_id"])
            for row in rows
            for event in row["trajectory"]
            if event["action"] in {"delegate", "escalate"}
        }
        null_stops = sum(
            event["action"] == "stop" and event.get("committed_candidate") is None
            for row in rows for event in row["trajectory"]
        )
        return {
            "episodes": len(rows),
            "mean_autonomous_accuracy": _mean(rows, "autonomous_accuracy"),
            "perfect_autonomous_episode_rate": sum(
                row["autonomous_accuracy"] == 1.0 for row in rows
            ) / len(rows),
            "mean_assisted_completion_rate": _mean(rows, "assisted_completion_rate"),
            "mean_model_calls": _mean(rows, "model_calls"),
            "mean_tokens": _mean(rows, "tokens_used"),
            "mean_latency_s": _mean(rows, "latency_used_s"),
            "mean_reviews": _mean(rows, "reviews_used"),
            "mean_humans": _mean(rows, "humans_used"),
            "mean_reward": _mean(rows, "total_reward"),
            "invalid_actions": sum(row["invalid_actions"] for row in rows),
            "budget_violations": sum(row["budget_violations"] for row in rows),
            "horizon_exhaustions": sum(row["horizon_exhaustions"] for row in rows),
            "unauthorized_source_calls": sum(
                row["unauthorized_source_calls"] for row in rows
            ),
            "safety_violations": sum(row["safety_violations"] for row in rows),
            "post_step_budget_invariant_violations": sum(
                row["post_step_budget_invariant_violations"] for row in rows
            ),
            "model_task_coverage": len(model_task_ids) / task_count,
            "null_stop_tasks": null_stops,
        }
    for name, rows in results.items():
        summaries[name] = summarize(rows)
    perfect_verifier_summaries = {
        name: summarize(rows) for name, rows in perfect_verifier_results.items()
    }
    posthoc_summaries = {
        name: summarize(rows) for name, rows in posthoc_results.items()
    }
    privileged_summaries = {
        name: summarize(rows) for name, rows in privileged_results.items()
    }

    review_rows = perfect_verifier_results["perfect_verifier_review_cascade"]
    review_changed_episodes = sum(
        any(
            event["action"] == "review" and event.get("review_passed") is False
            and index + 1 < len(row["trajectory"])
            and row["trajectory"][index + 1]["task_id"] == event["task_id"]
            and row["trajectory"][index + 1]["action"] == "escalate"
            for index, event in enumerate(row["trajectory"])
        )
        for row in review_rows
    )
    best_static_name = max(
        summaries,
        key=lambda name: summaries[name]["mean_autonomous_accuracy"],
    )
    best_observable_name, best_observable = max(
        {**summaries, **posthoc_summaries}.items(),
        key=lambda item: item[1]["mean_autonomous_accuracy"],
    )
    oracle = privileged_summaries["privileged_hindsight_output_oracle"][
        "mean_autonomous_accuracy"
    ]
    best_static = summaries[best_static_name]["mean_autonomous_accuracy"]
    atomic_frequency = Counter(
        task.task_id for episode in episodes for task in episode.tasks
    )
    atomic_tasks = {
        task.task_id: task for episode in episodes for task in episode.tasks
    }
    equal_weight_atomic = {
        "always_4b_live_accuracy": sum(
            task.qwen4b.valid and not task.qwen4b.abstained
            and task.qwen4b.candidate == task.correct_option
            for task in atomic_tasks.values()
        ) / len(atomic_tasks),
        "always_35b_live_accuracy": sum(
            task.qwen35b.valid and not task.qwen35b.abstained
            and task.qwen35b.candidate == task.correct_option
            for task in atomic_tasks.values()
        ) / len(atomic_tasks),
        "same_live_hindsight_oracle_accuracy": sum(
            (
                task.qwen4b.valid and not task.qwen4b.abstained
                and task.qwen4b.candidate == task.correct_option
            ) or (
                task.qwen35b.valid and not task.qwen35b.abstained
                and task.qwen35b.candidate == task.correct_option
            )
            for task in atomic_tasks.values()
        ) / len(atomic_tasks),
    }
    equal_weight_atomic["oracle_gain_over_always_4b_live"] = (
        equal_weight_atomic["same_live_hindsight_oracle_accuracy"]
        - equal_weight_atomic["always_4b_live_accuracy"]
    )
    return {
        "schema_version": "multitown-semantic-sequence-fixed-policy-report-v2",
        "evaluation_status": "train-only composition replay; no policy training",
        "episodes": len(episodes),
        "tasks_per_episode": len(episodes[0].tasks),
        "unique_atomic_task_outcome_pairs": len(atomic_frequency),
        "unique_model_call_outcomes": 2 * len(atomic_frequency),
        "atomic_trace_reuse": {
            "total_task_placements": sum(atomic_frequency.values()),
            "minimum_uses": min(atomic_frequency.values()),
            "maximum_uses": max(atomic_frequency.values()),
            "counts": dict(sorted(atomic_frequency.items())),
            "unique_compositions": len({episode.composition_id for episode in episodes}),
            "independence_warning": (
                f"episodes reuse the same atomic model traces; {len(episodes)} compositions are "
                f"not {len(episodes)} independent model-sampled episodes"
            ),
        },
        "equal_weight_atomic_diagnostic": {
            **equal_weight_atomic,
            "clusters": len(atomic_tasks),
            "selection_warning": (
                "descriptive train-only macro average; no untouched cluster interval"
            ),
        },
        "preregistered_observable_policies": summaries,
        "posthoc_observable_diagnostics": {
            "policies": posthoc_summaries,
            "selection_warning": (
                "family rule was chosen after seeing this train cohort and is not an "
                "independently validated deployable baseline"
            ),
        },
        "perfect_verifier_upper_bounds": {
            "policies": perfect_verifier_summaries,
            "review_contract": (
                "candidate-bound equality to hidden correct option; idealized and non-deployable"
            ),
        },
        "privileged_upper_bounds": privileged_summaries,
        "sequential_diagnostics": {
            "episodes_where_failed_review_changes_next_action": review_changed_episodes,
            "rate": review_changed_episodes / len(episodes),
            "contract": "idealized perfect-verifier upper bound",
            "deployable_observation_dependent_transition_established": False,
        },
        "headroom": {
            "best_preregistered_no_verifier_policy": best_static_name,
            "best_preregistered_autonomous_accuracy": best_static,
            "best_observed_no_verifier_policy": best_observable_name,
            "best_observed_autonomous_accuracy": best_observable[
                "mean_autonomous_accuracy"
            ],
            "hindsight_output_oracle_accuracy": oracle,
            "oracle_gain_over_best_preregistered": oracle - best_static,
            "oracle_gain_over_best_observed": (
                oracle - best_observable["mean_autonomous_accuracy"]
            ),
            "paired_interval_computed": False,
            "reason": (
                "episode-level pairs are cluster-dependent because atomic traces are reused"
            ),
        },
        "gate": {
            "idealized_verifier_observation_rate_at_least_20_percent": (
                review_changed_episodes / len(episodes) >= 0.20
            ),
            "deployable_sequential_observation_established": False,
            "oracle_vs_preregistered_headroom_at_least_5pp": (
                oracle - best_static >= 0.05
            ),
            "oracle_vs_best_observed_headroom_at_least_5pp": (
                oracle - best_observable["mean_autonomous_accuracy"] >= 0.05
            ),
            "positive_paired_interval_established": False,
            "five_pp_weighting_diagnostic": {
                "composition_weighted_gain": oracle - best_static,
                "equal_weight_atomic_gain": equal_weight_atomic[
                    "oracle_gain_over_always_4b_live"
                ],
                "composition_weighted_margin_over_5pp": oracle - best_static - 0.05,
                "equal_weight_atomic_margin_over_5pp": equal_weight_atomic[
                    "oracle_gain_over_always_4b_live"
                ] - 0.05,
                "near_5pp_decision_boundary": min(
                    abs(oracle - best_static - 0.05),
                    abs(
                        equal_weight_atomic["oracle_gain_over_always_4b_live"]
                        - 0.05
                    ),
                ) <= 0.005,
                "gate_warning": (
                    "the observed gain lies on or near the threshold under alternate "
                    "weighting of the same 40 atomic traces; it is not a robust RL gate"
                ),
            },
            "all_observable_policies_zero_hard_violations": all(
                value["invalid_actions"] == 0
                and value["budget_violations"] == 0
                and value["horizon_exhaustions"] == 0
                and value["unauthorized_source_calls"] == 0
                and value["safety_violations"] == 0
                and value["post_step_budget_invariant_violations"] == 0
                for value in {**summaries, **posthoc_summaries}.values()
            ),
            "allow_controller_rl": False,
        },
        "source_safety_contract": {
            "unauthorized_source_actions_exposed": False,
            "zero_unauthorized_calls_is_structural_not_learned": True,
            "safety_violation_count_scope": (
                "placement-weighted wrong safety-policy commits over reused task "
                "placements; not independent safety events or a general safety rate"
            ),
        },
    }


def plot_report(report: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    policies = {
        **report["preregistered_observable_policies"],
        **report["posthoc_observable_diagnostics"]["policies"],
        **report["perfect_verifier_upper_bounds"]["policies"],
        **report["privileged_upper_bounds"],
    }
    names = list(policies)
    accuracies = [policies[name]["mean_autonomous_accuracy"] for name in names]
    tokens = [policies[name]["mean_tokens"] for name in names]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].barh(names, accuracies)
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("Mean autonomous accuracy")
    axes[1].scatter(tokens, accuracies)
    for name, token, accuracy in zip(names, tokens, accuracies, strict=True):
        axes[1].annotate(name, (token, accuracy), fontsize=7)
    axes[1].set_xlabel("Mean replay tokens / episode")
    axes[1].set_ylabel("Mean autonomous accuracy")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    revision, dirty = git_state(project_root)
    if dirty:
        raise RuntimeError("A14 fixed-policy report requires a clean source revision")
    bank = Path(args.bank).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    report = evaluate(read_episode_bank(bank))
    report.update({
        "created_at_utc": utc_now(), "source_revision": revision,
        "source_dirty_at_start": dirty, "bank": str(bank),
        "bank_sha256": sha256_file(bank),
    })
    write_json(output / "fixed-policy-report.json", report)
    plot_report(report, output / "policy-comparison.png")
    manifest = {
        "schema_version": "multitown-semantic-sequence-report-manifest-v2",
        "source_revision": revision,
        "files": {
            name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for name, path in {
                "fixed-policy-report.json": output / "fixed-policy-report.json",
                "policy-comparison.png": output / "policy-comparison.png",
            }.items()
        },
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
