"""Build deployable baseline comparisons and curves from an A13 semantic probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .masbench_routing import write_json
from .semantic_model_probe import (
    SemanticMeasuredCall,
    SemanticProbeOutcome,
    _mcnemar_exact,
    balanced_subset,
    read_outcomes,
)
from .semantic_tasks import SemanticTask, read_bank


SCHEMA_VERSION = "multitown-semantic-baseline-comparison-v1"


def _call_metrics(calls: list[SemanticMeasuredCall]) -> dict[str, Any]:
    count = len(calls)
    return {
        "accuracy": sum(call.correct for call in calls) / count,
        "valid_rate": sum(call.valid for call in calls) / count,
        "answer_coverage": sum(
            call.valid and not call.abstained for call in calls
        ) / count,
        "mean_tokens": sum(call.total_tokens for call in calls) / count,
        "mean_latency_s": sum(call.latency_s for call in calls) / count,
    }


def _paired(left: list[bool], right: list[bool]) -> dict[str, Any]:
    left_only = sum(a and not b for a, b in zip(left, right, strict=True))
    right_only = sum(b and not a for a, b in zip(left, right, strict=True))
    return _mcnemar_exact(left_only, right_only)


def analyze_baselines(
    tasks: list[SemanticTask], outcomes: list[SemanticProbeOutcome],
) -> dict[str, Any]:
    if not tasks:
        raise ValueError("baseline analysis requires at least one task")
    rows = {row.task_id: row for row in outcomes}
    if len(rows) != len(outcomes):
        raise ValueError("baseline analysis rejects duplicate outcome task ids")
    if set(rows) != {task.task_id for task in tasks}:
        raise ValueError("baseline analysis task coverage mismatch")

    policies: dict[str, list[SemanticMeasuredCall]] = {
        "always_4b_live": [],
        "always_35b_live": [],
        "authority_model_bundle": [],
        "qwen4b_union_context": [],
        "qwen35b_union_context": [],
    }
    ordered_ids = []
    for task in tasks:
        row = rows[task.task_id]
        live_role = "weak" if task.world_state["authority"] == "local" else "strong"
        q4_live = row.calls[f"qwen4b_{live_role}_context"]
        q35_live = row.calls[f"qwen35b_{live_role}_context"]
        policies["always_4b_live"].append(q4_live)
        policies["always_35b_live"].append(q35_live)
        policies["authority_model_bundle"].append(
            q4_live if live_role == "weak" else q35_live
        )
        policies["qwen4b_union_context"].append(row.calls["qwen4b_union_context"])
        policies["qwen35b_union_context"].append(row.calls["qwen35b_union_context"])
        ordered_ids.append(task.task_id)

    baseline_correct = [call.correct for call in policies["always_4b_live"]]
    rendered = {}
    cumulative = {}
    for name, calls in policies.items():
        correct = [call.correct for call in calls]
        running = 0
        curve = []
        for index, value in enumerate(correct, start=1):
            running += int(value)
            curve.append(running / index)
        rendered[name] = {
            **_call_metrics(calls),
            "model_calls_per_task": 1.0,
            "paired_vs_always_4b_live": _paired(correct, baseline_correct),
        }
        cumulative[name] = curve

    q4_calls = policies["always_4b_live"]
    q35_calls = policies["always_35b_live"]
    cascade_success = [
        q4.correct or (not q4.correct and q35.correct)
        for q4, q35 in zip(q4_calls, q35_calls, strict=True)
    ]
    escalated = [not q4.correct for q4 in q4_calls]
    cascade_tokens = [
        q4.total_tokens + (q35.total_tokens if escalate else 0)
        for q4, q35, escalate in zip(q4_calls, q35_calls, escalated, strict=True)
    ]
    cascade_latency = [
        q4.latency_s + (q35.latency_s if escalate else 0.0)
        for q4, q35, escalate in zip(q4_calls, q35_calls, escalated, strict=True)
    ]
    rendered["review_then_escalate_35b"] = {
        "accuracy": sum(cascade_success) / len(tasks),
        "escalation_rate": sum(escalated) / len(tasks),
        "mean_tokens": sum(cascade_tokens) / len(tasks),
        "model_calls_per_task": 1.0 + sum(escalated) / len(tasks),
        "validator_checks_per_task": 1.0,
        "mean_model_latency_s": sum(cascade_latency) / len(tasks),
        "review_cost_included": False,
        "paired_vs_always_4b_live": _paired(cascade_success, baseline_correct),
    }
    running = 0
    cascade_curve = []
    for index, value in enumerate(cascade_success, start=1):
        running += int(value)
        cascade_curve.append(running / index)
    cumulative["review_then_escalate_35b"] = cascade_curve

    q4_only = sum(
        q4.correct and not q35.correct
        for q4, q35 in zip(q4_calls, q35_calls, strict=True)
    )
    q35_only = sum(
        q35.correct and not q4.correct
        for q4, q35 in zip(q4_calls, q35_calls, strict=True)
    )
    both = sum(
        q4.correct and q35.correct
        for q4, q35 in zip(q4_calls, q35_calls, strict=True)
    )
    same_live_oracle = [
        a or b for a, b in zip(
            baseline_correct, [call.correct for call in q35_calls], strict=True,
        )
    ]
    by_family = {}
    for family in sorted({task.family for task in tasks}):
        indices = [index for index, task in enumerate(tasks) if task.family == family]
        by_family[family] = {
            name: {
                "requests": len(indices),
                "accuracy": sum(calls[index].correct for index in indices) / len(indices),
                "answer_coverage": sum(
                    calls[index].valid and not calls[index].abstained for index in indices
                ) / len(indices),
            }
            for name, calls in policies.items()
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "train-only trace replay; no development or held-out evaluation",
        "tasks": len(tasks),
        "ordered_task_ids": ordered_ids,
        "policies": rendered,
        "same_live_model_output_oracle": {
            "accuracy": sum(same_live_oracle) / len(tasks),
            "gain_over_always_4b_live": (
                sum(same_live_oracle) - sum(baseline_correct)
            ) / len(tasks),
            "qwen4b_only_correct_rate": q4_only / len(tasks),
            "qwen35b_only_correct_rate": q35_only / len(tasks),
            "both_correct_rate": both / len(tasks),
            "label_access": "hindsight only; not deployable",
            "paired_vs_always_4b_live": _paired(same_live_oracle, baseline_correct),
        },
        "family_diagnostics": by_family,
        "cumulative_accuracy": cumulative,
        "caveats": [
            "authority is explicit, so selecting the live feed is deterministic preprocessing",
            "review_then_escalate uses the benchmark's candidate-bound deterministic validator",
            "review CPU latency/cost is not measured and is excluded from the reported cascade cost",
            "serving latency is an observed serial trace, not a hardware-normalized model comparison",
        ],
    }


def plot_comparison(comparison: dict[str, Any], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    curves = comparison["cumulative_accuracy"]
    figure, axis = plt.subplots(figsize=(10, 6))
    for name, values in curves.items():
        axis.plot(range(1, len(values) + 1), values, label=name)
    axis.set_xlabel("Training tasks processed")
    axis.set_ylabel("Cumulative strict accuracy")
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "cumulative-accuracy.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--outcomes", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-tasks", type=int, required=True)
    parser.add_argument("--cohort-index", type=int, required=True)
    args = parser.parse_args()

    tasks = balanced_subset(
        read_bank(Path(args.bank)), args.max_tasks, cohort_index=args.cohort_index,
    )
    outcomes = read_outcomes(Path(args.outcomes))
    comparison = analyze_baselines(tasks, outcomes)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "baseline-comparison.json", comparison)
    plot_comparison(comparison, output_dir)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
