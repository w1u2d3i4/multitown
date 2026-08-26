from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .baseline_report import _mcnemar
from .common import read_json, sha256_file, write_json
from .report import _latest_results, _method_summary, _monitor_summary, _paired_bootstrap


MATCHED_CONFIG_FIELDS = ("split", "sampling_seed", "temperature", "max_tokens", "tasks")


def _assert_matched_configs(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> None:
    mismatches = {
        field: {"baseline": baseline.get(field), "candidate": candidate.get(field)}
        for field in MATCHED_CONFIG_FIELDS
        if baseline.get(field) != candidate.get(field)
    }
    if mismatches:
        raise ValueError(f"unmatched experiment configuration: {mismatches}")


def _paired_statistics(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    differences = {
        "partial_score": np.array(
            [
                float(candidate_row.get("partial_score", 0))
                - float(baseline_row.get("partial_score", 0))
                for baseline_row, candidate_row in zip(
                    baseline, candidate, strict=True
                )
            ],
            dtype=float,
        ),
        "tokens": np.array(
            [
                float(candidate_row.get("total_tokens", 0))
                - float(baseline_row.get("total_tokens", 0))
                for baseline_row, candidate_row in zip(
                    baseline, candidate, strict=True
                )
            ],
            dtype=float,
        ),
        "latency_s": np.array(
            [
                float(candidate_row.get("latency_s", 0))
                - float(baseline_row.get("latency_s", 0))
                for baseline_row, candidate_row in zip(
                    baseline, candidate, strict=True
                )
            ],
            dtype=float,
        ),
    }
    baseline_tokens = np.array(
        [float(row.get("total_tokens", 0)) for row in baseline], dtype=float
    )
    candidate_tokens = np.array(
        [float(row.get("total_tokens", 0)) for row in candidate], dtype=float
    )
    baseline_latency = np.array(
        [float(row.get("latency_s", 0)) for row in baseline], dtype=float
    )
    candidate_latency = np.array(
        [float(row.get("latency_s", 0)) for row in candidate], dtype=float
    )
    quality = differences["partial_score"]
    return {
        "candidate_minus_baseline": {
            name: _paired_bootstrap(values, samples=bootstrap_samples, seed=seed + index)
            for index, (name, values) in enumerate(differences.items())
        },
        "partial_score_outcomes": {
            "candidate_better": int((quality > 1e-12).sum()),
            "tied": int((np.abs(quality) <= 1e-12).sum()),
            "baseline_better": int((quality < -1e-12).sum()),
        },
        "mcnemar": _mcnemar(baseline, candidate),
        "mean_token_reduction": (
            float(1.0 - candidate_tokens.mean() / baseline_tokens.mean())
            if baseline_tokens.mean()
            else None
        ),
        "mean_latency_reduction": (
            float(1.0 - candidate_latency.mean() / baseline_latency.mean())
            if baseline_latency.mean()
            else None
        ),
    }


def _write_paired_csv(
    task_ids: list[str],
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    output: Path,
) -> None:
    fields = [
        "task_id",
        "category",
        "difficulty",
        "baseline_passed",
        "candidate_passed",
        "baseline_partial",
        "candidate_partial",
        "partial_difference",
        "baseline_tokens",
        "candidate_tokens",
        "token_difference",
        "baseline_latency_s",
        "candidate_latency_s",
        "latency_difference_s",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for task_id, left, right in zip(task_ids, baseline, candidate, strict=True):
            left_partial = float(left.get("partial_score", 0))
            right_partial = float(right.get("partial_score", 0))
            left_tokens = int(left.get("total_tokens", 0))
            right_tokens = int(right.get("total_tokens", 0))
            left_latency = float(left.get("latency_s", 0))
            right_latency = float(right.get("latency_s", 0))
            writer.writerow(
                {
                    "task_id": task_id,
                    "category": left.get("category") or right.get("category"),
                    "difficulty": left.get("difficulty") or right.get("difficulty"),
                    "baseline_passed": bool(left.get("passed")),
                    "candidate_passed": bool(right.get("passed")),
                    "baseline_partial": left_partial,
                    "candidate_partial": right_partial,
                    "partial_difference": right_partial - left_partial,
                    "baseline_tokens": left_tokens,
                    "candidate_tokens": right_tokens,
                    "token_difference": right_tokens - left_tokens,
                    "baseline_latency_s": left_latency,
                    "candidate_latency_s": right_latency,
                    "latency_difference_s": right_latency - left_latency,
                }
            )


def _plot_quality_cost(
    summaries: dict[str, dict[str, Any]],
    baseline_label: str,
    candidate_label: str,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for label, color in (
        (baseline_label, "#5B6CFF"),
        (candidate_label, "#FF7A45"),
    ):
        row = summaries[label]
        ax.scatter(
            row["mean_tokens"], row["mean_partial_score"], s=140, color=color
        )
        ax.annotate(
            label,
            (row["mean_tokens"], row["mean_partial_score"]),
            xytext=(7, 7),
            textcoords="offset points",
        )
    ax.set_xlabel("Mean tokens per task")
    ax.set_ylabel("Mean deterministic partial score")
    ax.set_title("TeamBench matched candidate comparison")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def build_paired_report(
    *,
    baseline_dir: Path,
    candidate_dir: Path,
    baseline_label: str,
    candidate_label: str,
    output: Path,
    expected_count: int | None,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    directories = {baseline_label: baseline_dir, candidate_label: candidate_dir}
    maps = {
        label: _latest_results(path / "results.jsonl")
        for label, path in directories.items()
    }
    if set(maps[baseline_label]) != set(maps[candidate_label]):
        raise ValueError("baseline and candidate task sets are not paired")

    configs = {
        label: read_json(path / "config.json") for label, path in directories.items()
    }
    _assert_matched_configs(configs[baseline_label], configs[candidate_label])
    configured_order = [str(value) for value in configs[baseline_label]["tasks"]]
    if set(configured_order) != set(maps[baseline_label]):
        raise ValueError("configured tasks do not match completed results")
    if expected_count is not None and len(configured_order) != expected_count:
        raise ValueError(
            f"expected {expected_count} paired tasks, found {len(configured_order)}"
        )

    rows = {
        label: [maps[label][task_id] for task_id in configured_order]
        for label in directories
    }
    errors = [
        f"{label}:{row['task_id']}"
        for label, values in rows.items()
        for row in values
        if int(row.get("request_errors", 0)) or row.get("error")
    ]
    if errors:
        raise ValueError(f"invocation errors present: {errors}")

    summaries = {label: _method_summary(values) for label, values in rows.items()}
    monitoring = {
        label: _monitor_summary(path / "system_metrics.jsonl")
        for label, path in directories.items()
    }
    paired = _paired_statistics(
        rows[baseline_label],
        rows[candidate_label],
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    report = {
        "schema_version": "general-mas-teambench-paired-comparison-v1",
        "paired_tasks": len(configured_order),
        "task_ids_sha256": hashlib.sha256(
            "\n".join(configured_order).encode()
        ).hexdigest(),
        "baseline_label": baseline_label,
        "candidate_label": candidate_label,
        "methods": summaries,
        "paired": paired,
        "system_monitoring": monitoring,
        "invocation_error_rows": errors,
        "inputs": {
            label: {
                "results_sha256": sha256_file(path / "results.jsonl"),
                "config": configs[label],
            }
            for label, path in directories.items()
        },
    }

    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", report)
    _write_paired_csv(
        configured_order,
        rows[baseline_label],
        rows[candidate_label],
        output / "paired_tasks.csv",
    )
    _plot_quality_cost(
        summaries, baseline_label, candidate_label, output / "quality_cost.png"
    )

    difference = paired["candidate_minus_baseline"]["partial_score"]
    lines = [
        "# TeamBench matched candidate comparison",
        "",
        f"Paired tasks: **{len(configured_order)}**",
        "",
        "| Method | Passes | Mean partial | Mean tokens | P95 latency (s) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label in (baseline_label, candidate_label):
        value = summaries[label]
        lines.append(
            f"| {label} | {value['passes']}/{value['n']} | "
            f"{value['mean_partial_score']:.5f} | {value['mean_tokens']:.1f} | "
            f"{value['p95_latency_s']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Paired result",
            "",
            f"Candidate minus baseline partial score: {difference['mean']:+.5f} "
            f"(95% paired bootstrap CI [{difference['ci95_lower']:+.5f}, "
            f"{difference['ci95_upper']:+.5f}]).",
            "",
            f"Mean token reduction: {paired['mean_token_reduction']:+.2%}. "
            f"Mean latency reduction: {paired['mean_latency_reduction']:+.2%}.",
            "",
            "Positive reduction means the candidate used fewer resources.",
            "",
            "![Matched quality/cost comparison](quality_cost.png)",
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a paired TeamBench report")
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--baseline-label", default="Baseline")
    parser.add_argument("--candidate-label", default="Candidate")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-task-count", type=int, default=30)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()
    report = build_paired_report(
        baseline_dir=args.baseline_dir.resolve(),
        candidate_dir=args.candidate_dir.resolve(),
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
        output=args.output_dir.resolve(),
        expected_count=args.expected_task_count,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(json.dumps(report["paired"], indent=2))


if __name__ == "__main__":
    main()
