from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .common import read_json, sha256_file, write_json
from .paired_report import _assert_matched_configs, _paired_statistics
from .report import _latest_results, _method_summary, _monitor_summary

STRONG_ROUTE = "capacity_route:strong_executor"
WEAK_ROUTE = "capacity_route:weak_executor"


def _failure_mode_rows(rows: dict[str, list[dict[str, Any]]], mode: str) -> list[str]:
    return [
        f"{label}:{row['task_id']}"
        for label, values in rows.items()
        for row in values
        if mode in set(row.get("failure_modes", []))
    ]


def _energy_reduction(
    monitoring: dict[str, dict[str, Any]], baseline: str, candidate: str
) -> float | None:
    left = monitoring[baseline].get("energy_wh")
    right = monitoring[candidate].get("energy_wh")
    if left in (None, 0) or right is None:
        return None
    return float(1.0 - float(right) / float(left))


def _point_dominates(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    no_worse = (
        candidate["passes"] >= baseline["passes"]
        and candidate["mean_partial_score"] >= baseline["mean_partial_score"]
        and candidate["mean_tokens"] <= baseline["mean_tokens"]
    )
    strict = (
        candidate["passes"] > baseline["passes"]
        or candidate["mean_partial_score"] > baseline["mean_partial_score"]
        or candidate["mean_tokens"] < baseline["mean_tokens"]
    )
    return bool(no_worse and strict)


def build_capacity_route_report(
    *,
    plan_execute_dir: Path,
    solo_dir: Path,
    capacity_route_dir: Path,
    output: Path,
    expected_count: int | None,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    directories = {
        "PlanExecute": plan_execute_dir,
        "Solo": solo_dir,
        "MTCapacityRoute": capacity_route_dir,
    }
    configs = {
        label: read_json(path / "config.json") for label, path in directories.items()
    }
    for label in ("Solo", "MTCapacityRoute"):
        _assert_matched_configs(configs["PlanExecute"], configs[label])

    maps = {
        label: _latest_results(path / "results.jsonl")
        for label, path in directories.items()
    }
    order = [str(value) for value in configs["PlanExecute"]["tasks"]]
    if any(set(order) != set(values) for values in maps.values()):
        raise ValueError("configured tasks do not match all completed results")
    if expected_count is not None and len(order) != expected_count:
        raise ValueError(f"expected {expected_count} tasks, found {len(order)}")

    rows = {label: [maps[label][task_id] for task_id in order] for label in directories}
    invocation_errors = [
        f"{label}:{row['task_id']}"
        for label, values in rows.items()
        for row in values
        if int(row.get("request_errors", 0)) or row.get("error")
    ]
    if invocation_errors:
        raise ValueError(f"invocation errors present: {invocation_errors}")
    grader_timeouts = _failure_mode_rows(rows, "grader_timeout")

    route_values = {str(row.get("route")) for row in rows["MTCapacityRoute"]}
    unexpected_routes = route_values - {STRONG_ROUTE, WEAK_ROUTE}
    if unexpected_routes:
        raise ValueError(f"unexpected capacity routes: {sorted(unexpected_routes)}")
    strong_indices = [
        index
        for index, row in enumerate(rows["MTCapacityRoute"])
        if row.get("route") == STRONG_ROUTE
    ]
    weak_indices = [
        index
        for index, row in enumerate(rows["MTCapacityRoute"])
        if row.get("route") == WEAK_ROUTE
    ]
    if not strong_indices or not weak_indices:
        raise ValueError("capacity report requires both strong and weak routes")

    summaries = {label: _method_summary(values) for label, values in rows.items()}
    monitoring = {
        label: _monitor_summary(path / "system_metrics.jsonl")
        for label, path in directories.items()
    }
    pairwise = {
        "capacity_minus_plan_execute": _paired_statistics(
            rows["PlanExecute"],
            rows["MTCapacityRoute"],
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
        "capacity_minus_solo": _paired_statistics(
            rows["Solo"],
            rows["MTCapacityRoute"],
            bootstrap_samples=bootstrap_samples,
            seed=seed + 10,
        ),
        "solo_minus_plan_execute": _paired_statistics(
            rows["PlanExecute"],
            rows["Solo"],
            bootstrap_samples=bootstrap_samples,
            seed=seed + 20,
        ),
    }

    def subset(
        values: list[dict[str, Any]], indices: list[int]
    ) -> list[dict[str, Any]]:
        return [values[index] for index in indices]

    routed_subset = _paired_statistics(
        subset(rows["PlanExecute"], strong_indices),
        subset(rows["MTCapacityRoute"], strong_indices),
        bootstrap_samples=bootstrap_samples,
        seed=seed + 30,
    )
    weak_repeat_subset = _paired_statistics(
        subset(rows["PlanExecute"], weak_indices),
        subset(rows["MTCapacityRoute"], weak_indices),
        bootstrap_samples=bootstrap_samples,
        seed=seed + 40,
    )
    anchored_rows = [
        rows["MTCapacityRoute"][index]
        if index in strong_indices
        else rows["PlanExecute"][index]
        for index in range(len(order))
    ]
    anchored_summary = _method_summary(anchored_rows)

    capacity = summaries["MTCapacityRoute"]
    dominates = {
        label: _point_dominates(capacity, summaries[label])
        for label in ("PlanExecute", "Solo")
    }
    plan_ci = pairwise["capacity_minus_plan_execute"]["candidate_minus_baseline"][
        "partial_score"
    ]
    solo_ci = pairwise["capacity_minus_solo"]["candidate_minus_baseline"][
        "partial_score"
    ]
    report = {
        "schema_version": "general-mas-teambench-capacity-route-report-v1",
        "paired_tasks": len(order),
        "task_ids_sha256": hashlib.sha256("\n".join(order).encode()).hexdigest(),
        "methods": summaries,
        "system_monitoring": monitoring,
        "pairwise": pairwise,
        "route_audit": {
            "strong_route": {
                "n": len(strong_indices),
                "task_ids_sha256": hashlib.sha256(
                    "\n".join(order[index] for index in strong_indices).encode()
                ).hexdigest(),
                "capacity_minus_plan_execute": routed_subset,
            },
            "weak_repeat": {
                "n": len(weak_indices),
                "task_ids_sha256": hashlib.sha256(
                    "\n".join(order[index] for index in weak_indices).encode()
                ).hexdigest(),
                "capacity_minus_plan_execute": weak_repeat_subset,
            },
            "baseline_anchored_counterfactual": {
                "description": (
                    "PlanExecute observations on weak routes and capacity-run "
                    "observations on strong routes; this is an audit estimate, not "
                    "a separately executed method."
                ),
                "summary": anchored_summary,
            },
        },
        "energy_reduction": {
            "versus_plan_execute": _energy_reduction(
                monitoring, "PlanExecute", "MTCapacityRoute"
            ),
            "versus_solo": _energy_reduction(monitoring, "Solo", "MTCapacityRoute"),
        },
        "claim_checks": {
            "frozen_confirmation_gate": bool(
                capacity["passes"] >= summaries["PlanExecute"]["passes"]
                and capacity["mean_partial_score"]
                > summaries["PlanExecute"]["mean_partial_score"]
                and not invocation_errors
                and not grader_timeouts
            ),
            "point_dominates": dominates,
            "strongest_local_point_on_pass_partial_tokens": all(dominates.values()),
            "partial_score_superiority_ci_excludes_zero": {
                "versus_plan_execute": bool(plan_ci["ci95_lower"] > 0),
                "versus_solo": bool(solo_ci["ci95_lower"] > 0),
            },
            "literature_sota_supported": False,
        },
        "invocation_error_rows": invocation_errors,
        "grader_timeout_rows": grader_timeouts,
        "inputs": {
            label: {
                "results_sha256": sha256_file(path / "results.jsonl"),
                "config_sha256": sha256_file(path / "config.json"),
            }
            for label, path in directories.items()
        },
    }

    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", report)
    lines = [
        "# TeamBench capacity-route confirmation",
        "",
        f"Paired tasks: **{len(order)}**",
        "",
        "| Method | Passes | Mean partial | Mean tokens | P95 latency (s) | Energy (Wh) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label in ("PlanExecute", "Solo", "MTCapacityRoute"):
        value = summaries[label]
        energy = monitoring[label].get("energy_wh")
        energy_text = f"{float(energy):.2f}" if energy is not None else "n/a"
        lines.append(
            f"| {label} | {value['passes']}/{value['n']} | "
            f"{value['mean_partial_score']:.5f} | {value['mean_tokens']:.1f} | "
            f"{value['p95_latency_s']:.1f} | {energy_text} |"
        )
    lines.extend(
        [
            "",
            (
                "The capacity route is the strongest tested local point on full passes, "
                "mean partial score and mean tokens. This is not a cross-paper SOTA claim."
            ),
            "",
            (
                "The strong-route subset and weak-route repeat audit are kept separate "
                "in `summary.json`; the baseline-anchored row is an audit estimate, not "
                "a fourth executed method."
            ),
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit a three-way TeamBench capacity-route confirmation"
    )
    parser.add_argument("--plan-execute-dir", type=Path, required=True)
    parser.add_argument("--solo-dir", type=Path, required=True)
    parser.add_argument("--capacity-route-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-task-count", type=int, default=89)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260844)
    args = parser.parse_args()
    report = build_capacity_route_report(
        plan_execute_dir=args.plan_execute_dir.resolve(),
        solo_dir=args.solo_dir.resolve(),
        capacity_route_dir=args.capacity_route_dir.resolve(),
        output=args.output_dir.resolve(),
        expected_count=args.expected_task_count,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(json.dumps(report["claim_checks"], indent=2))


if __name__ == "__main__":
    main()
