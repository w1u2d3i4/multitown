from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import read_json, sha256_file, write_json
from .report import _latest_results, _method_summary, _monitor_summary, _paired_bootstrap


METHODS = ("Solo", "A4", "A8")


def _mcnemar(
    baseline: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> dict[str, Any]:
    baseline_only = sum(
        bool(x.get("passed")) and not bool(y.get("passed"))
        for x, y in zip(baseline, candidate, strict=True)
    )
    candidate_only = sum(
        not bool(x.get("passed")) and bool(y.get("passed"))
        for x, y in zip(baseline, candidate, strict=True)
    )
    discordant = baseline_only + candidate_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, k)
            for k in range(min(baseline_only, candidate_only) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "baseline_pass_candidate_fail": baseline_only,
        "baseline_fail_candidate_pass": candidate_only,
        "discordant": discordant,
        "two_sided_exact_p": p_value,
    }


def _pairwise(
    *,
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    quality = np.array(
        [
            float(y.get("partial_score", 0)) - float(x.get("partial_score", 0))
            for x, y in zip(baseline, candidate, strict=True)
        ],
        dtype=float,
    )
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
    return {
        "partial_score_difference": _paired_bootstrap(
            quality, samples=bootstrap_samples, seed=seed
        ),
        "partial_score_outcomes": {
            "candidate_better": int((quality > 1e-12).sum()),
            "tied": int((np.abs(quality) <= 1e-12).sum()),
            "baseline_better": int((quality < -1e-12).sum()),
        },
        "mcnemar": _mcnemar(baseline, candidate),
        "mean_token_reduction": (
            float(1.0 - candidate_tokens.mean() / baseline_tokens.mean())
            if baseline_tokens.mean() else None
        ),
        "mean_latency_reduction": (
            float(1.0 - candidate_latency.mean() / baseline_latency.mean())
            if baseline_latency.mean() else None
        ),
    }


def _pareto_frontier(summaries: dict[str, dict[str, Any]]) -> list[str]:
    frontier = []
    for method, row in summaries.items():
        dominated = False
        for other, rival in summaries.items():
            if other == method:
                continue
            quality_no_worse = rival["mean_partial_score"] >= row["mean_partial_score"]
            tokens_no_worse = rival["mean_tokens"] <= row["mean_tokens"]
            strictly_better = (
                rival["mean_partial_score"] > row["mean_partial_score"]
                or rival["mean_tokens"] < row["mean_tokens"]
            )
            if quality_no_worse and tokens_no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(method)
    return frontier


def _plot_quality_cost(summaries: dict[str, dict[str, Any]], output: Path) -> None:
    colors = {"Solo": "#2CA02C", "A4": "#5B6CFF", "A8": "#FF7A45"}
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for method in METHODS:
        row = summaries[method]
        ax.scatter(
            row["mean_tokens"], row["mean_partial_score"],
            s=140, color=colors[method], label=method,
        )
        ax.annotate(
            method, (row["mean_tokens"], row["mean_partial_score"]),
            xytext=(7, 7), textcoords="offset points",
        )
    ax.set_xlabel("Mean tokens per task")
    ax.set_ylabel("Mean deterministic partial score")
    ax.set_title("TeamBench: single agent, fixed team, and selective team")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _write_paired_csv(
    task_ids: list[str], rows: dict[str, list[dict[str, Any]]], output: Path
) -> None:
    fields = ["task_id", "category", "difficulty"]
    for prefix in ("solo", "a4", "a8"):
        fields.extend(
            [f"{prefix}_passed", f"{prefix}_partial", f"{prefix}_tokens", f"{prefix}_latency_s"]
        )
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, task_id in enumerate(task_ids):
            solo = rows["Solo"][index]
            a4 = rows["A4"][index]
            a8 = rows["A8"][index]
            row: dict[str, Any] = {
                "task_id": task_id,
                "category": solo.get("category") or a4.get("category") or a8.get("category"),
                "difficulty": solo.get("difficulty") or a4.get("difficulty") or a8.get("difficulty"),
            }
            for prefix, value in (("solo", solo), ("a4", a4), ("a8", a8)):
                row.update({
                    f"{prefix}_passed": bool(value.get("passed")),
                    f"{prefix}_partial": float(value.get("partial_score", 0)),
                    f"{prefix}_tokens": int(value.get("total_tokens", 0)),
                    f"{prefix}_latency_s": float(value.get("latency_s", 0)),
                })
            writer.writerow(row)


def build_baseline_report(
    *,
    solo_dir: Path,
    a4_dir: Path,
    a8_dir: Path,
    output: Path,
    expected_count: int | None,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    directories = {"Solo": solo_dir, "A4": a4_dir, "A8": a8_dir}
    maps = {
        method: _latest_results(path / "results.jsonl")
        for method, path in directories.items()
    }
    task_sets = {method: set(values) for method, values in maps.items()}
    if len({frozenset(value) for value in task_sets.values()}) != 1:
        raise ValueError(f"unpaired task sets: {task_sets}")
    configs = {
        method: read_json(path / "config.json") for method, path in directories.items()
    }
    configured_order = [str(task_id) for task_id in configs["A4"].get("tasks", [])]
    shared = task_sets["A4"]
    task_ids = configured_order if set(configured_order) == shared else sorted(shared)
    if expected_count is not None and len(task_ids) != expected_count:
        raise ValueError(f"expected {expected_count} paired tasks, found {len(task_ids)}")
    rows = {
        method: [maps[method][task_id] for task_id in task_ids] for method in METHODS
    }
    errors = [
        f"{method}:{row['task_id']}"
        for method in METHODS
        for row in rows[method]
        if int(row.get("request_errors", 0)) or row.get("error")
    ]
    if errors:
        raise ValueError(f"invocation errors present: {errors}")

    summaries = {method: _method_summary(rows[method]) for method in METHODS}
    monitoring = {
        method: _monitor_summary(directories[method] / "system_metrics.jsonl")
        for method in METHODS
    }
    pairwise = {
        "A4_minus_Solo": _pairwise(
            baseline=rows["Solo"], candidate=rows["A4"],
            bootstrap_samples=bootstrap_samples, seed=seed,
        ),
        "A8_minus_Solo": _pairwise(
            baseline=rows["Solo"], candidate=rows["A8"],
            bootstrap_samples=bootstrap_samples, seed=seed + 1,
        ),
        "A8_minus_A4": _pairwise(
            baseline=rows["A4"], candidate=rows["A8"],
            bootstrap_samples=bootstrap_samples, seed=seed + 2,
        ),
    }
    report = {
        "schema_version": "general-mas-teambench-baseline-comparison-v1",
        "paired_tasks": len(task_ids),
        "task_ids_sha256": hashlib.sha256("\n".join(task_ids).encode()).hexdigest(),
        "methods": summaries,
        "pairwise": pairwise,
        "quality_token_pareto_frontier": _pareto_frontier(summaries),
        "system_monitoring": monitoring,
        "invocation_error_rows": errors,
        "inputs": {
            method: {
                "results_sha256": sha256_file(directories[method] / "results.jsonl"),
                "config": configs[method],
            }
            for method in METHODS
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", report)
    _write_paired_csv(task_ids, rows, output / "paired_tasks.csv")
    _plot_quality_cost(summaries, output / "quality_cost_three_way.png")
    lines = [
        "# TeamBench Solo-TB / A4-TB / A8-TB comparison",
        "", f"Paired tasks: **{len(task_ids)}**", "",
        "| Method | Pass rate | Mean partial | Mean tokens | Median latency (s) | P95 latency (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        value = summaries[method]
        lines.append(
            f"| {method}-TB | {value['pass_rate']:.3f} | "
            f"{value['mean_partial_score']:.3f} | {value['mean_tokens']:.1f} | "
            f"{value['median_latency_s']:.1f} | {value['p95_latency_s']:.1f} |"
        )
    lines.extend(["", "## Paired comparisons", ""])
    for name, value in pairwise.items():
        quality = value["partial_score_difference"]
        lines.append(
            f"- **{name.replace('_', ' ')}:** partial-score difference "
            f"{quality['mean']:+.4f} (95% paired bootstrap CI "
            f"[{quality['ci95_lower']:+.4f}, {quality['ci95_upper']:+.4f}]); "
            f"mean token reduction {value['mean_token_reduction']:+.1%}; "
            f"mean latency reduction {value['mean_latency_reduction']:+.1%}."
        )
    lines.extend([
        "", "Positive reduction means the candidate used fewer resources than its baseline.",
        "Unlike-model token counts are not converted into monetary cost.", "",
        "![Three-way quality/cost comparison](quality_cost_three_way.png)", "",
    ])
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a three-way TeamBench Solo/A4/A8 baseline report"
    )
    parser.add_argument("--solo-dir", type=Path, required=True)
    parser.add_argument("--a4-dir", type=Path, required=True)
    parser.add_argument("--a8-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-task-count", type=int, default=89)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()
    report = build_baseline_report(
        solo_dir=args.solo_dir.resolve(), a4_dir=args.a4_dir.resolve(),
        a8_dir=args.a8_dir.resolve(), output=args.output_dir.resolve(),
        expected_count=args.expected_task_count,
        bootstrap_samples=args.bootstrap_samples, seed=args.seed,
    )
    print(json.dumps(report["pairwise"], indent=2))


if __name__ == "__main__":
    main()
