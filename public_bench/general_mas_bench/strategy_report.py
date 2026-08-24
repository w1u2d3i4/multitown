from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib

from .baseline_report import _pairwise, _pareto_frontier
from .common import read_json, sha256_file, write_json
from .report import _latest_results, _method_summary, _monitor_summary

matplotlib.use("Agg")
plt = importlib.import_module("matplotlib.pyplot")

METHODS = ("Solo", "PlanExecute", "ExecuteReview", "A4", "A8")
PUBLIC_LABELS = {
    "Solo": "Solo-TB",
    "PlanExecute": "PlanExecute-TB",
    "ExecuteReview": "ExecuteReview-TB",
    "A4": "FixedTeam-TB",
    "A8": "MultiTown-TB",
}
COLORS = {
    "Solo": "#2CA02C",
    "PlanExecute": "#9467BD",
    "ExecuteReview": "#17BECF",
    "A4": "#5B6CFF",
    "A8": "#FF7A45",
}


def _load_paired(
    directories: dict[str, Path], expected_count: int | None
) -> tuple[list[str], dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    maps = {
        method: _latest_results(directories[method] / "results.jsonl")
        for method in METHODS
    }
    task_sets = {method: set(values) for method, values in maps.items()}
    if len({frozenset(value) for value in task_sets.values()}) != 1:
        raise ValueError(f"unpaired task sets: {task_sets}")
    configs = {
        method: read_json(directories[method] / "config.json")
        for method in METHODS
    }
    for method in METHODS:
        configured_method = configs[method].get("method")
        if configured_method not in {None, method}:
            raise ValueError(
                f"{method} directory contains method={configured_method!r}"
            )
    shared = task_sets["A4"]
    configured_order = [str(task_id) for task_id in configs["A4"].get("tasks", [])]
    task_ids = configured_order if set(configured_order) == shared else sorted(shared)
    if expected_count is not None and len(task_ids) != expected_count:
        raise ValueError(f"expected {expected_count} paired tasks, found {len(task_ids)}")
    rows = {
        method: [maps[method][task_id] for task_id in task_ids]
        for method in METHODS
    }
    errors = [
        f"{method}:{row['task_id']}"
        for method in METHODS
        for row in rows[method]
        if int(row.get("request_errors", 0)) or row.get("error")
    ]
    if errors:
        raise ValueError(f"invocation errors present: {errors}")
    return task_ids, rows, configs


def _all_pairwise(
    rows: dict[str, list[dict[str, Any]]],
    bootstrap_samples: int,
    seed: int,
    *,
    include_runtime: bool,
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for offset, (baseline, candidate) in enumerate(combinations(METHODS, 2)):
        comparison = _pairwise(
            baseline=rows[baseline],
            candidate=rows[candidate],
            bootstrap_samples=bootstrap_samples,
            seed=seed + offset,
        )
        if not include_runtime:
            comparison.pop("mean_latency_reduction", None)
        comparisons[f"{candidate}_minus_{baseline}"] = comparison
    return comparisons


def _provenance_value(config: dict[str, Any], key: str) -> Any:
    if key == "source_revision":
        return config.get("source", {}).get("revision")
    if key == "source_dirty":
        return config.get("source", {}).get("dirty")
    if key == "strong_model":
        return config.get("strong", {}).get("model")
    if key == "weak_model":
        return config.get("weak", {}).get("model")
    return config.get(key)


def _check_provenance(
    configs: dict[str, dict[str, Any]], *, quality_only: bool
) -> dict[str, Any]:
    quality_keys = (
        "split_sha256",
        "docker_image_id",
        "temperature",
        "max_tokens",
        "strong_model",
        "weak_model",
    )
    runtime_keys = ("source_revision", "source_dirty")
    values = {
        key: {method: _provenance_value(config, key) for method, config in configs.items()}
        for key in quality_keys + runtime_keys
    }
    quality_mismatches = {
        key: value for key, value in values.items() if key in quality_keys and len(set(value.values())) > 1
    }
    if quality_mismatches:
        raise ValueError(f"incompatible quality/token provenance: {quality_mismatches}")
    runtime_mismatches = {
        key: value for key, value in values.items() if key in runtime_keys and len(set(value.values())) > 1
    }
    if runtime_mismatches and not quality_only:
        raise ValueError(
            "runtime provenance differs; rerun on one source revision or pass "
            f"quality_only=True: {runtime_mismatches}"
        )
    return {
        "quality_token_compatible": True,
        "runtime_compatible": not runtime_mismatches,
        "values": values,
        "runtime_mismatches": runtime_mismatches,
    }


def _write_paired_csv(
    task_ids: list[str],
    rows: dict[str, list[dict[str, Any]]],
    output: Path,
    *,
    include_runtime: bool,
) -> None:
    fields = ["task_id", "category", "difficulty"]
    slugs = {
        "Solo": "solo",
        "PlanExecute": "plan_execute",
        "ExecuteReview": "execute_review",
        "A4": "fixed_team",
        "A8": "multitown",
    }
    for method in METHODS:
        prefix = slugs[method]
        fields.extend([f"{prefix}_passed", f"{prefix}_partial", f"{prefix}_tokens"])
        if include_runtime:
            fields.append(f"{prefix}_latency_s")
        fields.append(f"{prefix}_route")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, task_id in enumerate(task_ids):
            first = rows["Solo"][index]
            value: dict[str, Any] = {
                "task_id": task_id,
                "category": first.get("category"),
                "difficulty": first.get("difficulty"),
            }
            for method in METHODS:
                prefix = slugs[method]
                result = rows[method][index]
                value.update(
                    {
                        f"{prefix}_passed": bool(result.get("passed")),
                        f"{prefix}_partial": float(result.get("partial_score", 0)),
                        f"{prefix}_tokens": int(result.get("total_tokens", 0)),
                        f"{prefix}_route": str(result.get("route", "")),
                    }
                )
                if include_runtime:
                    value[f"{prefix}_latency_s"] = float(result.get("latency_s", 0))
            writer.writerow(value)


def _plot_quality_cost(summaries: dict[str, dict[str, Any]], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for method in METHODS:
        row = summaries[method]
        ax.scatter(
            row["mean_tokens"],
            row["mean_partial_score"],
            s=145,
            color=COLORS[method],
            label=PUBLIC_LABELS[method],
        )
        ax.annotate(
            PUBLIC_LABELS[method],
            (row["mean_tokens"], row["mean_partial_score"]),
            xytext=(7, 7),
            textcoords="offset points",
        )
    ax.set_xlabel("Mean tokens per task")
    ax.set_ylabel("Mean deterministic partial score")
    ax.set_title("TeamBench strategy matrix: quality vs inference cost")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _write_markdown(report: dict[str, Any], output: Path) -> None:
    include_runtime = report["comparison_scope"] == "same_harness_all_metrics"
    lines = [
        "# TeamBench five-strategy comparison",
        "",
        f"Paired public tasks: **{report['paired_tasks']}**",
        "",
    ]
    if include_runtime:
        lines.extend(
            [
                "| Strategy | Pass | Mean partial | Mean tokens | Median latency (s) | p95 latency (s) | Energy (Wh) |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
    else:
        lines.extend(
            [
                "> **Quality/token-only comparison.** The inputs use different runner source revisions. Latency, energy, and other system metrics are intentionally excluded.",
                "",
                "| Strategy | Pass | Mean partial | Mean tokens |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
    for method in METHODS:
        summary = report["methods"][method]
        row = (
            f"| {PUBLIC_LABELS[method]} | {summary['passes']}/{summary['n']} | "
            f"{summary['mean_partial_score']:.5f} | {summary['mean_tokens']:.1f}"
        )
        if include_runtime:
            monitor = report["system_monitoring"][method] or {}
            energy = monitor.get("energy_wh")
            energy_text = f"{energy:.2f}" if energy is not None else "n/a"
            row += (
                f" | {summary['median_latency_s']:.2f} | "
                f"{summary['p95_latency_s']:.2f} | {energy_text}"
            )
        lines.append(row + " |")
    lines.extend(
        [
            "",
            "## MultiTown against each direct baseline",
            "",
        ]
    )
    for baseline in METHODS[:-1]:
        comparison = report["pairwise"][f"A8_minus_{baseline}"]
        quality = comparison["partial_score_difference"]
        resource_text = f"tokens {comparison['mean_token_reduction']:+.2%} reduction"
        if include_runtime:
            resource_text += f"; latency {comparison['mean_latency_reduction']:+.2%} reduction"
        lines.append(
            f"- **MultiTown-TB minus {PUBLIC_LABELS[baseline]}:** partial-score "
            f"difference {quality['mean']:+.5f} (95% paired bootstrap CI "
            f"[{quality['ci95_lower']:+.5f}, {quality['ci95_upper']:+.5f}]); "
            f"{resource_text}."
        )
    frontier = ", ".join(
        PUBLIC_LABELS[method] for method in report["quality_token_pareto_frontier"]
    )
    lines.extend(
        [
            "",
            f"Quality/token Pareto frontier: **{frontier}**.",
            "",
            (
                "Positive resource reduction means MultiTown-TB used less than the named "
                "baseline. Unlike-model token counts are not converted into monetary cost. "
                "External-paper results are not mixed into this local same-harness table."
            ),
            "",
            "![Five-strategy quality/cost comparison](quality_cost_five_way.png)",
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def build_strategy_report(
    *,
    solo_dir: Path,
    plan_execute_dir: Path,
    execute_review_dir: Path,
    a4_dir: Path,
    a8_dir: Path,
    output: Path,
    expected_count: int | None,
    bootstrap_samples: int,
    seed: int,
    quality_only: bool = False,
) -> dict[str, Any]:
    directories = {
        "Solo": solo_dir,
        "PlanExecute": plan_execute_dir,
        "ExecuteReview": execute_review_dir,
        "A4": a4_dir,
        "A8": a8_dir,
    }
    task_ids, rows, configs = _load_paired(directories, expected_count)
    provenance = _check_provenance(configs, quality_only=quality_only)
    include_runtime = not quality_only
    summaries = {method: _method_summary(rows[method]) for method in METHODS}
    if not include_runtime:
        for summary in summaries.values():
            for key in (
                "mean_latency_s",
                "median_latency_s",
                "p95_latency_s",
                "total_latency_s",
            ):
                summary.pop(key, None)
    monitoring = (
        {
            method: _monitor_summary(directories[method] / "system_metrics.jsonl")
            for method in METHODS
        }
        if include_runtime
        else {method: None for method in METHODS}
    )
    report = {
        "schema_version": "general-mas-teambench-strategy-comparison-v2",
        "comparison_scope": (
            "same_harness_all_metrics" if include_runtime else "quality_and_tokens_only"
        ),
        "provenance_compatibility": provenance,
        "paired_tasks": len(task_ids),
        "task_ids_sha256": hashlib.sha256("\n".join(task_ids).encode()).hexdigest(),
        "public_labels": PUBLIC_LABELS,
        "methods": summaries,
        "pairwise": _all_pairwise(
            rows, bootstrap_samples, seed, include_runtime=include_runtime
        ),
        "quality_token_pareto_frontier": _pareto_frontier(summaries),
        "system_monitoring": monitoring,
        "invocation_error_rows": [],
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
    _write_paired_csv(
        task_ids,
        rows,
        output / "paired_tasks.csv",
        include_runtime=include_runtime,
    )
    _plot_quality_cost(summaries, output / "quality_cost_five_way.png")
    _write_markdown(report, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a five-strategy same-harness TeamBench report"
    )
    parser.add_argument("--solo-dir", type=Path, required=True)
    parser.add_argument("--plan-execute-dir", type=Path, required=True)
    parser.add_argument("--execute-review-dir", type=Path, required=True)
    parser.add_argument("--a4-dir", type=Path, required=True)
    parser.add_argument("--a8-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-task-count", type=int, default=89)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument(
        "--quality-only",
        action="store_true",
        help="allow differing runner revisions and exclude latency/energy metrics",
    )
    args = parser.parse_args()
    report = build_strategy_report(
        solo_dir=args.solo_dir.resolve(),
        plan_execute_dir=args.plan_execute_dir.resolve(),
        execute_review_dir=args.execute_review_dir.resolve(),
        a4_dir=args.a4_dir.resolve(),
        a8_dir=args.a8_dir.resolve(),
        output=args.output_dir.resolve(),
        expected_count=args.expected_task_count,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        quality_only=args.quality_only,
    )
    print(json.dumps(report["pairwise"], indent=2))


if __name__ == "__main__":
    main()
