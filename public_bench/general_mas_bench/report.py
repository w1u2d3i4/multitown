from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import read_json, read_jsonl, sha256_file, write_json


def _latest_results(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        rows[str(row["task_id"])] = row
    return rows


def _method_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    tokens = np.array([float(row.get("total_tokens", 0)) for row in rows], dtype=float)
    latency = np.array([float(row.get("latency_s", 0)) for row in rows], dtype=float)
    partial = np.array([float(row.get("partial_score", 0)) for row in rows], dtype=float)
    passed = np.array([bool(row.get("passed", False)) for row in rows], dtype=bool)
    routes = Counter(str(row.get("route", "missing")) for row in rows)
    roles: Counter[str] = Counter()
    for row in rows:
        roles.update({key: int(value) for key, value in row.get("role_activations", {}).items()})
    return {
        "n": n,
        "passes": int(passed.sum()),
        "pass_rate": float(passed.mean()) if n else None,
        "mean_partial_score": float(partial.mean()) if n else None,
        "mean_tokens": float(tokens.mean()) if n else None,
        "median_tokens": float(np.median(tokens)) if n else None,
        "p95_tokens": float(np.quantile(tokens, 0.95)) if n else None,
        "total_tokens": int(tokens.sum()),
        "mean_latency_s": float(latency.mean()) if n else None,
        "median_latency_s": float(np.median(latency)) if n else None,
        "p95_latency_s": float(np.quantile(latency, 0.95)) if n else None,
        "total_latency_s": float(latency.sum()),
        "routes": dict(sorted(routes.items())),
        "role_activations": dict(sorted(roles.items())),
        "invocation_errors": sum(int(row.get("request_errors", 0)) for row in rows),
    }


def _paired_bootstrap(values: np.ndarray, *, samples: int, seed: int) -> dict[str, float]:
    if not len(values):
        raise ValueError("cannot bootstrap an empty paired sample")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=float)
    chunk = 1000
    for start in range(0, samples, chunk):
        size = min(chunk, samples - start)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        means[start:start + size] = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return {"mean": float(values.mean()), "ci95_lower": float(lower), "ci95_upper": float(upper)}


def _mcnemar_exact(a4: list[dict[str, Any]], a8: list[dict[str, Any]]) -> dict[str, Any]:
    a4_only = sum(bool(x.get("passed")) and not bool(y.get("passed")) for x, y in zip(a4, a8, strict=True))
    a8_only = sum(not bool(x.get("passed")) and bool(y.get("passed")) for x, y in zip(a4, a8, strict=True))
    discordant = a4_only + a8_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(a4_only, a8_only) + 1)) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "a4_pass_a8_fail": a4_only,
        "a4_fail_a8_pass": a8_only,
        "discordant": discordant,
        "two_sided_exact_p": p_value,
    }


def _energy_wh(path: Path) -> float | None:
    if not path.is_file():
        return None
    rows = read_jsonl(path)
    samples = [
        (float(row["elapsed_s"]), float(row["gpu_power_w"]))
        for row in rows if row.get("gpu_power_w") is not None
    ]
    if len(samples) < 2:
        return None
    elapsed = np.array([item[0] for item in samples])
    power = np.array([item[1] for item in samples])
    return float(np.trapezoid(power, elapsed) / 3600.0)


def _monitor_summary(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    rows = read_jsonl(path)
    if not rows:
        return None
    result: dict[str, Any] = {
        "samples": len(rows),
        "duration_s": float(rows[-1]["elapsed_s"]) - float(rows[0]["elapsed_s"]),
    }
    for key, output_key, scale in (
        ("cpu_percent", "cpu_percent", 1.0),
        ("ram_used_bytes", "ram_used_gib", 1024.0**3),
        ("gpu_util_percent", "gpu_util_percent", 1.0),
        ("gpu_power_w", "gpu_power_w", 1.0),
        ("gpu_temperature_c", "gpu_temperature_c", 1.0),
    ):
        values = np.array(
            [float(row[key]) / scale for row in rows if row.get(key) is not None], dtype=float
        )
        if len(values):
            result[output_key] = {
                "mean": float(values.mean()),
                "p95": float(np.quantile(values, 0.95)),
                "max": float(values.max()),
            }
    result["energy_wh"] = _energy_wh(path)
    return result


def _plot_tradeoff(summary: dict[str, Any], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    for method, color in (("A4", "#5B6CFF"), ("A8", "#FF7A45")):
        row = summary[method]
        ax.scatter(row["mean_tokens"], row["mean_partial_score"], s=130, color=color, label=method)
        ax.annotate(method, (row["mean_tokens"], row["mean_partial_score"]), xytext=(7, 7), textcoords="offset points")
    ax.set_xlabel("Mean tokens per task")
    ax.set_ylabel("Mean deterministic partial score")
    ax.set_title("TeamBench A4 vs A8")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_categories(a4: list[dict[str, Any]], a8: list[dict[str, Any]], output: Path) -> None:
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"A4": [], "A8": []})
    for method, rows in (("A4", a4), ("A8", a8)):
        for row in rows:
            values[str(row.get("category") or "Unknown")][method].append(float(row.get("partial_score", 0)))
    categories = sorted(values)
    y = np.arange(len(categories))
    height = 0.38
    fig, ax = plt.subplots(figsize=(8.5, max(4.5, len(categories) * 0.38)))
    ax.barh(y - height / 2, [np.mean(values[c]["A4"]) for c in categories], height, label="A4", color="#5B6CFF")
    ax.barh(y + height / 2, [np.mean(values[c]["A8"]) for c in categories], height, label="A8", color="#FF7A45")
    ax.set_yticks(y, labels=categories)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Mean deterministic partial score")
    ax.set_title("Category-level performance")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_cumulative_tokens(a4: list[dict[str, Any]], a8: list[dict[str, Any]], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for method, rows, color in (("A4", a4, "#5B6CFF"), ("A8", a8, "#FF7A45")):
        cumulative = np.cumsum([int(row.get("total_tokens", 0)) for row in rows])
        ax.plot(np.arange(1, len(rows) + 1), cumulative, label=method, color=color, linewidth=2)
    ax.set_xlabel("Paired tasks completed")
    ax.set_ylabel("Cumulative tokens")
    ax.set_title("Cumulative inference cost")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_per_task_metrics(
    a4: list[dict[str, Any]], a8: list[dict[str, Any]], output: Path
) -> None:
    x = np.arange(1, len(a4) + 1)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for method, rows, color in (("A4", a4, "#5B6CFF"), ("A8", a8, "#FF7A45")):
        axes[0].plot(x, [float(row.get("partial_score", 0)) for row in rows], color=color, alpha=0.8, label=method)
        axes[1].plot(x, [float(row.get("total_tokens", 0)) for row in rows], color=color, alpha=0.8, label=method)
        axes[2].plot(x, [float(row.get("latency_s", 0)) for row in rows], color=color, alpha=0.8, label=method)
    axes[0].set_ylabel("Partial score")
    axes[0].set_ylim(-0.03, 1.03)
    axes[1].set_ylabel("Tokens")
    axes[2].set_ylabel("Latency (s)")
    axes[2].set_xlabel("Paired task index (frozen split order)")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right")
    fig.suptitle("Per-task quality, token cost, and latency")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_paired_differences(
    a4: list[dict[str, Any]], a8: list[dict[str, Any]], output: Path
) -> None:
    diff = np.array([
        float(y.get("partial_score", 0)) - float(x.get("partial_score", 0))
        for x, y in zip(a4, a8, strict=True)
    ])
    order = np.argsort(diff)
    sorted_diff = diff[order]
    colors = np.where(sorted_diff > 1e-12, "#2CA02C", np.where(sorted_diff < -1e-12, "#D62728", "#9A9A9A"))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(np.arange(1, len(diff) + 1), sorted_diff, color=colors, width=0.9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(-0.02, color="#D62728", linewidth=1, linestyle="--", label="preregistered margin −0.02")
    ax.set_xlabel("Tasks sorted by A8−A4 partial-score difference")
    ax.set_ylabel("A8−A4 partial score")
    ax.set_title("Paired quality differences")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_system_monitoring(a4_path: Path, a8_path: Path, output: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=False)
    specs = (
        ("cpu_percent", 1.0, "CPU (%)"),
        ("ram_used_bytes", 1024.0**3, "RAM used (GiB)"),
        ("gpu_power_w", 1.0, "GPU/SoC power (W)"),
    )
    for method, path, color in (("A4", a4_path, "#5B6CFF"), ("A8", a8_path, "#FF7A45")):
        rows = read_jsonl(path) if path.is_file() else []
        for ax, (key, scale, label) in zip(axes, specs, strict=True):
            points = [
                (float(row["elapsed_s"]) / 3600.0, float(row[key]) / scale)
                for row in rows if row.get(key) is not None
            ]
            if points:
                ax.plot([p[0] for p in points], [p[1] for p in points], color=color, alpha=0.72, linewidth=0.8, label=method)
            ax.set_ylabel(label)
            ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Elapsed experiment time (hours)")
    for ax in axes:
        ax.legend(loc="upper right")
    fig.suptitle("System monitoring during formal runs (5 s sampling)")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _write_paired_csv(
    task_ids: list[str], a4: list[dict[str, Any]], a8: list[dict[str, Any]], output: Path
) -> None:
    fields = [
        "task_id", "category", "difficulty", "a4_passed", "a8_passed",
        "a4_partial", "a8_partial", "partial_diff_a8_minus_a4",
        "a4_tokens", "a8_tokens", "token_diff_a8_minus_a4",
        "a4_latency_s", "a8_latency_s", "latency_diff_s_a8_minus_a4",
        "a4_route", "a8_route",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for task_id, x, y in zip(task_ids, a4, a8, strict=True):
            writer.writerow({
                "task_id": task_id,
                "category": x.get("category") or y.get("category"),
                "difficulty": x.get("difficulty") or y.get("difficulty"),
                "a4_passed": bool(x.get("passed")),
                "a8_passed": bool(y.get("passed")),
                "a4_partial": float(x.get("partial_score", 0)),
                "a8_partial": float(y.get("partial_score", 0)),
                "partial_diff_a8_minus_a4": float(y.get("partial_score", 0)) - float(x.get("partial_score", 0)),
                "a4_tokens": int(x.get("total_tokens", 0)),
                "a8_tokens": int(y.get("total_tokens", 0)),
                "token_diff_a8_minus_a4": int(y.get("total_tokens", 0)) - int(x.get("total_tokens", 0)),
                "a4_latency_s": float(x.get("latency_s", 0)),
                "a8_latency_s": float(y.get("latency_s", 0)),
                "latency_diff_s_a8_minus_a4": float(y.get("latency_s", 0)) - float(x.get("latency_s", 0)),
                "a4_route": x.get("route", ""),
                "a8_route": y.get("route", ""),
            })


def build_report(
    *, a4_dir: Path, a8_dir: Path, output: Path, expected_count: int | None,
    allow_incomplete: bool, bootstrap_samples: int, seed: int,
) -> dict[str, Any]:
    a4_map = _latest_results(a4_dir / "results.jsonl")
    a8_map = _latest_results(a8_dir / "results.jsonl")
    a4_config = read_json(a4_dir / "config.json")
    a8_config = read_json(a8_dir / "config.json")
    if set(a4_map) != set(a8_map):
        missing_a4 = sorted(set(a8_map) - set(a4_map))
        missing_a8 = sorted(set(a4_map) - set(a8_map))
        raise ValueError(f"unpaired task sets; missing A4={missing_a4}, missing A8={missing_a8}")
    configured_order = [str(task_id) for task_id in a4_config.get("tasks", [])]
    task_ids = configured_order if set(configured_order) == set(a4_map) else sorted(a4_map)
    if expected_count is not None and len(task_ids) != expected_count and not allow_incomplete:
        raise ValueError(f"expected {expected_count} paired tasks, found {len(task_ids)}")
    a4 = [a4_map[task_id] for task_id in task_ids]
    a8 = [a8_map[task_id] for task_id in task_ids]
    errors = [
        f"{method}:{row['task_id']}" for method, rows in (("A4", a4), ("A8", a8))
        for row in rows if int(row.get("request_errors", 0)) or row.get("error")
    ]
    if errors and not allow_incomplete:
        raise ValueError(f"invocation errors present: {errors}")
    partial_diff = np.array([
        float(y.get("partial_score", 0)) - float(x.get("partial_score", 0))
        for x, y in zip(a4, a8, strict=True)
    ])
    a4_summary = _method_summary(a4)
    a8_summary = _method_summary(a8)
    token_reduction = (
        1.0 - a8_summary["mean_tokens"] / a4_summary["mean_tokens"]
        if a4_summary["mean_tokens"] else None
    )
    paired = _paired_bootstrap(partial_diff, samples=bootstrap_samples, seed=seed)
    wins = {
        "a8_better": int((partial_diff > 1e-12).sum()),
        "tied": int((np.abs(partial_diff) <= 1e-12).sum()),
        "a4_better": int((partial_diff < -1e-12).sum()),
    }
    monitoring = {
        "A4": _monitor_summary(a4_dir / "system_metrics.jsonl"),
        "A8": _monitor_summary(a8_dir / "system_metrics.jsonl"),
    }
    energy = {
        method: values.get("energy_wh") if values else None
        for method, values in monitoring.items()
    }
    gate = {
        "partial_ci_lower_at_least_minus_0_02": paired["ci95_lower"] >= -0.02,
        "token_reduction_at_least_0_30": token_reduction is not None and token_reduction >= 0.30,
    }
    gate["passed"] = all(gate.values())
    report = {
        "schema_version": "general-mas-teambench-comparison-v2",
        "paired_tasks": len(task_ids),
        "task_ids_sha256": hashlib.sha256("\n".join(task_ids).encode()).hexdigest(),
        "A4": a4_summary,
        "A8": a8_summary,
        "paired_partial_difference_a8_minus_a4": paired,
        "paired_partial_outcomes": wins,
        "mcnemar": _mcnemar_exact(a4, a8),
        "mean_token_reduction": token_reduction,
        "gpu_energy_wh": energy,
        "system_monitoring": monitoring,
        "preregistered_gate": gate,
        "invocation_error_rows": errors,
        "inputs": {
            "a4_results_sha256": sha256_file(a4_dir / "results.jsonl"),
            "a8_results_sha256": sha256_file(a8_dir / "results.jsonl"),
            "a4_config": a4_config,
            "a8_config": a8_config,
            "a8_replacement_manifest": (
                read_json(a8_dir / "replacement_manifest.json")
                if (a8_dir / "replacement_manifest.json").is_file() else None
            ),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", report)
    _plot_tradeoff({"A4": a4_summary, "A8": a8_summary}, output / "quality_cost.png")
    _plot_categories(a4, a8, output / "category_partial.png")
    _plot_cumulative_tokens(a4, a8, output / "cumulative_tokens.png")
    _plot_per_task_metrics(a4, a8, output / "per_task_metrics.png")
    _plot_paired_differences(a4, a8, output / "paired_quality_difference.png")
    _plot_system_monitoring(
        a4_dir / "system_metrics.jsonl", a8_dir / "system_metrics.jsonl",
        output / "system_monitoring.png",
    )
    _write_paired_csv(task_ids, a4, a8, output / "paired_tasks.csv")
    markdown = (
        "# TeamBench A4-TB / A8-TB comparison\n\n"
        f"Paired tasks: **{len(task_ids)}**\n\n"
        "| Method | Pass rate | Mean partial | Mean tokens | Mean latency (s) |\n"
        "|---|---:|---:|---:|---:|\n"
        f"| A4-TB | {a4_summary['pass_rate']:.3f} | {a4_summary['mean_partial_score']:.3f} | {a4_summary['mean_tokens']:.1f} | {a4_summary['mean_latency_s']:.1f} |\n"
        f"| A8-TB | {a8_summary['pass_rate']:.3f} | {a8_summary['mean_partial_score']:.3f} | {a8_summary['mean_tokens']:.1f} | {a8_summary['mean_latency_s']:.1f} |\n\n"
        f"A8−A4 partial-score difference: {paired['mean']:.4f} "
        f"(paired bootstrap 95% CI [{paired['ci95_lower']:.4f}, {paired['ci95_upper']:.4f}]).\n\n"
        f"Partial-score outcomes (A8 better / tied / A4 better): "
        f"{wins['a8_better']} / {wins['tied']} / {wins['a4_better']}.\n\n"
        f"Mean token reduction: {token_reduction:.1%}. Gate passed: **{gate['passed']}**.\n\n"
        f"Latency median / P95 (s): A4 {a4_summary['median_latency_s']:.1f} / {a4_summary['p95_latency_s']:.1f}; "
        f"A8 {a8_summary['median_latency_s']:.1f} / {a8_summary['p95_latency_s']:.1f}.\n\n"
        "![Quality/cost](quality_cost.png)\n\n"
        "![Category partial score](category_partial.png)\n\n"
        "![Cumulative tokens](cumulative_tokens.png)\n\n"
        "![Per-task metrics](per_task_metrics.png)\n\n"
        "![Paired quality difference](paired_quality_difference.png)\n\n"
        "![System monitoring](system_monitoring.png)\n"
    )
    (output / "REPORT.md").write_text(markdown, encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a paired TeamBench A4/A8 report")
    parser.add_argument("--a4-dir", type=Path, required=True)
    parser.add_argument("--a8-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-task-count", type=int, default=89)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()
    report = build_report(
        a4_dir=args.a4_dir.resolve(), a8_dir=args.a8_dir.resolve(),
        output=args.output_dir.resolve(), expected_count=args.expected_task_count,
        allow_incomplete=args.allow_incomplete, bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(json.dumps(report["preregistered_gate"], indent=2))


if __name__ == "__main__":
    main()
