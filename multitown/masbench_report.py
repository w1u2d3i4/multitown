"""Create paired statistics and plots for the frozen MASBench comparison."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .masbench_routing import ARM_ORDER, ARM_RUN_NAMES, AXES, load_jsonl, write_json
from .provenance import build_manifest


DISPLAY_NAMES = {"A0": "A0 weak", "A1": "A1 strong", "A2": "A2 vote", "A4": "A4 heavy", "A6": "A6 routed"}
COLORS = {"A0": "#8da0cb", "A1": "#4daf4a", "A2": "#984ea3", "A4": "#e41a1c", "A6": "#ff7f00"}


def normalized_rows(masbench_root: Path, routing_root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in ARM_ORDER:
        rows = load_jsonl(masbench_root / f"{ARM_RUN_NAMES[arm]}-test" / "decisions.jsonl")
        result[arm] = {row["sample_id"]: row for row in rows}
    routed = [row for row in load_jsonl(routing_root / "routing_decisions.jsonl") if row["selected_by_a6"]]
    result["A6"] = {row["sample_id"]: row for row in routed}
    expected = set(result["A0"])
    for name, rows in result.items():
        if set(rows) != expected:
            raise ValueError(f"{name} sample IDs do not match A0")
    return result


def row_latency(row: dict[str, Any]) -> float:
    return float(row["decision_latency_s"])


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    tokens = sum(int(row["total_tokens"]) for row in rows)
    latencies = [row_latency(row) for row in rows]
    correct = sum(bool(row["correct"]) for row in rows)
    return {
        "decisions": count,
        "correct": correct,
        "accuracy": correct / count,
        "total_tokens": tokens,
        "tokens_per_decision": tokens / count,
        "latency_mean_s": float(np.mean(latencies)),
        "latency_p95_s": float(np.percentile(latencies, 95)),
    }


def exact_mcnemar_p(left: list[bool], right: list[bool]) -> dict[str, Any]:
    left_only = sum(a and not b for a, b in zip(left, right, strict=True))
    right_only = sum(b and not a for a, b in zip(left, right, strict=True))
    discordant = left_only + right_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(left_only, right_only) + 1)) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {"left_only_correct": left_only, "right_only_correct": right_only, "discordant": discordant, "p_value_two_sided": p_value}


def bootstrap_difference(
    left: np.ndarray, right: np.ndarray, *, seed: int, iterations: int = 10000
) -> dict[str, float]:
    if left.shape != right.shape:
        raise ValueError("paired arrays must have identical shapes")
    rng = np.random.default_rng(seed)
    observed = float(np.mean(left - right))
    estimates = np.empty(iterations, dtype=np.float64)
    for start in range(0, iterations, 1000):
        size = min(1000, iterations - start)
        indices = rng.integers(0, len(left), size=(size, len(left)))
        estimates[start : start + size] = np.mean(left[indices] - right[indices], axis=1)
    low, high = np.percentile(estimates, [2.5, 97.5])
    return {"difference": observed, "ci95_low": float(low), "ci95_high": float(high), "iterations": iterations}


def paired_comparison(
    left_name: str,
    right_name: str,
    rows_by_arm: dict[str, dict[str, dict[str, Any]]],
    sample_ids: list[str],
    *,
    seed: int,
) -> dict[str, Any]:
    left = [rows_by_arm[left_name][sample_id] for sample_id in sample_ids]
    right = [rows_by_arm[right_name][sample_id] for sample_id in sample_ids]
    left_correct = np.array([bool(row["correct"]) for row in left], dtype=np.float64)
    right_correct = np.array([bool(row["correct"]) for row in right], dtype=np.float64)
    left_tokens = np.array([int(row["total_tokens"]) for row in left], dtype=np.float64)
    right_tokens = np.array([int(row["total_tokens"]) for row in right], dtype=np.float64)
    left_latency = np.array([row_latency(row) for row in left], dtype=np.float64)
    right_latency = np.array([row_latency(row) for row in right], dtype=np.float64)
    return {
        "left": left_name,
        "right": right_name,
        "accuracy": bootstrap_difference(left_correct, right_correct, seed=seed),
        "tokens_per_decision": bootstrap_difference(left_tokens, right_tokens, seed=seed + 1),
        "latency_mean_s": bootstrap_difference(left_latency, right_latency, seed=seed + 2),
        "mcnemar_exact": exact_mcnemar_p(
            [bool(value) for value in left_correct], [bool(value) for value in right_correct]
        ),
    }


def build_comparison(masbench_root: Path, routing_root: Path, *, seed: int) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, Any]]]]:
    rows_by_arm = normalized_rows(masbench_root, routing_root)
    sample_ids = sorted(rows_by_arm["A0"])
    overall = {
        arm: summarize_rows([rows[sample_id] for sample_id in sample_ids])
        for arm, rows in rows_by_arm.items()
    }
    by_axis: dict[str, dict[str, Any]] = {}
    for arm, rows in rows_by_arm.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for sample_id in sample_ids:
            grouped[str(rows[sample_id]["axis"])].append(rows[sample_id])
        by_axis[arm] = {axis: summarize_rows(grouped[axis]) for axis in AXES}
    return {
        "schema_version": "multitown-masbench-comparison-v1",
        "seed": seed,
        "sample_count": len(sample_ids),
        "overall": overall,
        "by_axis": by_axis,
        "paired": {
            "A6_vs_A1": paired_comparison("A6", "A1", rows_by_arm, sample_ids, seed=seed),
            "A6_vs_A4": paired_comparison("A6", "A4", rows_by_arm, sample_ids, seed=seed + 100),
            "A1_vs_A4": paired_comparison("A1", "A4", rows_by_arm, sample_ids, seed=seed + 200),
        },
    }, rows_by_arm


def plot_tradeoff(comparison: dict[str, Any], routing_summary: dict[str, Any], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for arm, metrics in comparison["overall"].items():
        ax.scatter(metrics["tokens_per_decision"], 100 * metrics["accuracy"], s=90, color=COLORS[arm], label=DISPLAY_NAMES[arm])
        ax.annotate(arm, (metrics["tokens_per_decision"], 100 * metrics["accuracy"]), xytext=(5, 5), textcoords="offset points")
    for name in ("knn", "svm", "mlp"):
        metrics = routing_summary["test_metrics"][name]
        ax.scatter(metrics["tokens_per_decision"], 100 * metrics["accuracy"], marker="x", s=70, color="#555555")
        ax.annotate(name.upper(), (metrics["tokens_per_decision"], 100 * metrics["accuracy"]), xytext=(5, -11), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Tokens per decision")
    ax.set_ylabel("Exact-match accuracy (%)")
    ax.set_title("MASBench test cost-quality trade-off")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_axis_accuracy(comparison: dict[str, Any], output: Path) -> None:
    arms = list((*ARM_ORDER, "A6"))
    x = np.arange(len(AXES))
    width = 0.16
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for index, arm in enumerate(arms):
        values = [100 * comparison["by_axis"][arm][axis]["accuracy"] for axis in AXES]
        ax.bar(x + (index - 2) * width, values, width, label=arm, color=COLORS[arm])
    ax.set_xticks(x, AXES)
    ax.set_ylabel("Exact-match accuracy (%)")
    ax.set_title("MASBench test accuracy by structural axis")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_cumulative(rows_by_arm: dict[str, dict[str, dict[str, Any]]], output: Path) -> None:
    sample_ids = sorted(rows_by_arm["A0"])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for arm in (*ARM_ORDER, "A6"):
        cumulative = np.cumsum([int(rows_by_arm[arm][sample_id]["total_tokens"]) for sample_id in sample_ids])
        ax.plot(np.arange(1, len(sample_ids) + 1), cumulative / 1_000_000, label=arm, color=COLORS[arm])
    ax.set_xlabel("Test decisions (stable sample order)")
    ax.set_ylabel("Cumulative tokens (millions)")
    ax.set_title("MASBench cumulative inference cost")
    ax.grid(alpha=0.25)
    ax.legend(ncol=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_latency_ecdf(rows_by_arm: dict[str, dict[str, dict[str, Any]]], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for arm in (*ARM_ORDER, "A6"):
        values = np.sort([row_latency(row) for row in rows_by_arm[arm].values()])
        ax.plot(values, np.arange(1, len(values) + 1) / len(values), label=arm, color=COLORS[arm])
    ax.set_xlabel("Decision latency (s)")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("MASBench latency distribution")
    ax.grid(alpha=0.25)
    ax.legend(ncol=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def markdown_report(comparison: dict[str, Any], routing_summary: dict[str, Any]) -> str:
    lines = [
        "# MASBench local organization comparison",
        "",
        "Evidence level: **subset reproduced**. A0/A1/A2/A4 are MultiTown adapters; A6 uses",
        "LLMRouter-inspired classical routing families and is not an upstream end-to-end reproduction.",
        "",
        "| System | Accuracy | Correct | Tokens/decision | Mean latency |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in (*ARM_ORDER, "A6"):
        value = comparison["overall"][arm]
        lines.append(
            f"| {arm} | {100 * value['accuracy']:.1f}% | {value['correct']}/{value['decisions']} | "
            f"{value['tokens_per_decision']:.1f} | {value['latency_mean_s']:.3f}s |"
        )
    paired = comparison["paired"]["A6_vs_A1"]
    lines.extend([
        "",
        f"Dev selected `{routing_summary['selected_policy']}` for A6. On test, A6 minus A1 accuracy was "
        f"{100 * paired['accuracy']['difference']:.1f} percentage points "
        f"(paired bootstrap 95% CI {100 * paired['accuracy']['ci95_low']:.1f} to "
        f"{100 * paired['accuracy']['ci95_high']:.1f}); exact McNemar p="
        f"{paired['mcnemar_exact']['p_value_two_sided']:.4f}.",
        "",
        "The simple routed policy did not beat fixed A1: it lost one correct answer and used more tokens.",
        "It did substantially reduce cost relative to A4 while improving exact-match accuracy. This",
        "negative result motivates A7's larger split and calibrated contextual policy, then A8's",
        "execution-time early stopping for long horizon/robustness traces.",
        "",
        "Plots: `tradeoff.png`, `axis-accuracy.png`, `cumulative-tokens.png`, and `latency-ecdf.png`.",
        "",
    ])
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    masbench_root = args.masbench_root.resolve()
    routing_root = args.routing_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    comparison, rows_by_arm = build_comparison(masbench_root, routing_root, seed=args.seed)
    routing_summary = json.loads((routing_root / "summary.json").read_text(encoding="utf-8"))
    comparison["source_revisions"] = {
        "fixed_arm_runner": json.loads((masbench_root / "A4-heavy-test" / "summary.json").read_text())["config"]["source_revision"],
        "routing": routing_summary["source_revision"],
    }
    write_json(output / "comparison.json", comparison)
    plot_tradeoff(comparison, routing_summary, output / "tradeoff.png")
    plot_axis_accuracy(comparison, output / "axis-accuracy.png")
    plot_cumulative(rows_by_arm, output / "cumulative-tokens.png")
    plot_latency_ecdf(rows_by_arm, output / "latency-ecdf.png")
    (output / "RESULTS.md").write_text(markdown_report(comparison, routing_summary), encoding="utf-8")
    manifest = build_manifest(project_root, [masbench_root, routing_root, output])
    write_json(output / "artifact-manifest.json", manifest)
    if args.record_dir:
        record = args.record_dir.resolve()
        record.mkdir(parents=True, exist_ok=True)
        copies = {
            output / "comparison.json": record / "comparison.json",
            output / "RESULTS.md": record / "RESULTS.md",
            output / "artifact-manifest.json": record / "artifact-manifest.json",
            routing_root / "summary.json": record / "routing-summary.json",
            routing_root / "input_manifest.json": record / "routing-input-manifest.json",
            output / "tradeoff.png": record / "tradeoff.png",
            output / "axis-accuracy.png": record / "axis-accuracy.png",
            output / "cumulative-tokens.png": record / "cumulative-tokens.png",
            output / "latency-ecdf.png": record / "latency-ecdf.png",
        }
        for source, target in copies.items():
            shutil.copy2(source, target)
    print(json.dumps({
        "output": str(output),
        "record_dir": str(args.record_dir.resolve()) if args.record_dir else None,
        "overall": comparison["overall"],
        "source_dirty_at_manifest": manifest["source_dirty"],
    }, ensure_ascii=False, indent=2))
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--masbench-root", type=Path, required=True)
    parser.add_argument("--routing-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--record-dir", type=Path)
    parser.add_argument("--seed", type=int, default=20260810)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
