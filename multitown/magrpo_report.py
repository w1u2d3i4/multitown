"""Build a paired equal-output-budget report for TLDR MAGRPO/GRPO runs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPORT_SCHEMA = "multitown-magrpo-comparison-v1"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("schema") != "multitown-magrpo-evaluation-v1":
        raise ValueError(f"not a MAGRPO evaluation artifact: {path}")
    if value.get("status") != "complete":
        raise ValueError(f"evaluation is not complete: {path}")
    return value


def _reward_matrix(run: dict[str, Any]) -> np.ndarray:
    records = run["seed_records"]
    matrix = np.asarray([record["rewards"] for record in records], dtype=float)
    expected = (len(run["protocol"]["seeds"]), run["protocol"]["eval_samples"])
    if matrix.shape != expected:
        raise ValueError(f"reward matrix {matrix.shape} does not match {expected}")
    return matrix


def _validate_matched(runs: Sequence[dict[str, Any]]) -> None:
    first = runs[0]
    exact_fields = (
        "dataset",
        "data",
        "upstream",
    )
    for run in runs[1:]:
        for field in exact_fields:
            if run[field] != first[field]:
                raise ValueError(f"comparison mismatch: {field}")
        for field in (
            "eval_start",
            "eval_samples",
            "seeds",
            "maximum_total_output_tokens_per_task",
            "temperature",
            "top_p",
        ):
            if run["protocol"][field] != first["protocol"][field]:
                raise ValueError(f"protocol mismatch: {field}")
        if (
            run["generation"]["completion_tokens"]
            != first["generation"]["completion_tokens"]
        ):
            raise ValueError("actual completion-token totals do not match")
    if first["method"] != "magrpo" or runs[1]["method"] != "magrpo":
        raise ValueError("first two inputs must be base and trained MAGRPO")
    if runs[2]["method"] != "grpo":
        raise ValueError("third input must be trained GRPO")


def _bootstrap_interval(
    task_values: np.ndarray,
    *,
    rng: np.random.Generator,
    iterations: int,
) -> tuple[float, float]:
    task_count = task_values.shape[0]
    indices = rng.integers(0, task_count, size=(iterations, task_count))
    estimates = task_values[indices].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def _method_summary(
    name: str,
    run: dict[str, Any],
    matrix: np.ndarray,
    *,
    rng: np.random.Generator,
    iterations: int,
) -> dict[str, Any]:
    task_means = matrix.mean(axis=0)
    low, high = _bootstrap_interval(task_means, rng=rng, iterations=iterations)
    generation = run["generation"]
    resources = run["resources"]
    text_count = generation.get("completion_text_count")
    delimiter_count = generation.get("paragraph_split_delimiter_count")
    return {
        "name": name,
        "method": run["method"],
        "reward_mean": float(matrix.mean()),
        "task_clustered_95_ci": [low, high],
        "reward_std": float(matrix.std(ddof=1)),
        "prompt_tokens": generation["prompt_tokens"],
        "completion_tokens": generation["completion_tokens"],
        "wall_seconds": run["wall_seconds"],
        "estimated_gpu_energy_wh": resources.get("estimated_gpu_energy_wh"),
        "prompt_profile": run["protocol"].get("prompt_profile", "official"),
        "grpo_split_policy": run["protocol"].get(
            "grpo_split_policy", "official-fallback"
        ),
        "paragraph_split_delimiter_count": delimiter_count,
        "paragraph_split_delimiter_rate": (
            delimiter_count / text_count
            if text_count not in {None, 0} and delimiter_count is not None
            else None
        ),
        "model_fingerprints": [model["tree_sha256"] for model in run["models"]],
    }


def _paired_summary(
    candidate_name: str,
    reference_name: str,
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    rng: np.random.Generator,
    iterations: int,
) -> dict[str, Any]:
    differences = candidate - reference
    task_differences = differences.mean(axis=0)
    low, high = _bootstrap_interval(task_differences, rng=rng, iterations=iterations)
    epsilon = 1e-12
    return {
        "candidate": candidate_name,
        "reference": reference_name,
        "mean_reward_difference": float(differences.mean()),
        "task_clustered_95_ci": [low, high],
        "observation_wins": int((differences > epsilon).sum()),
        "observation_ties": int((np.abs(differences) <= epsilon).sum()),
        "observation_losses": int((differences < -epsilon).sum()),
        "task_wins": int((task_differences > epsilon).sum()),
        "task_ties": int((np.abs(task_differences) <= epsilon).sum()),
        "task_losses": int((task_differences < -epsilon).sum()),
        "cluster_unit": "evaluation task averaged across seeds",
    }


def build_report(
    base: dict[str, Any],
    magrpo: dict[str, Any],
    grpo: dict[str, Any],
    *,
    bootstrap_seed: int = 20260833,
    bootstrap_iterations: int = 20_000,
) -> dict[str, Any]:
    runs = (base, magrpo, grpo)
    _validate_matched(runs)
    matrices = tuple(_reward_matrix(run) for run in runs)
    rng = np.random.default_rng(bootstrap_seed)
    names = ("untrained_two_agent", "magrpo_trained_8", "grpo_trained_8")
    methods = [
        _method_summary(
            name,
            run,
            matrix,
            rng=rng,
            iterations=bootstrap_iterations,
        )
        for name, run, matrix in zip(names, runs, matrices)
    ]
    comparisons = [
        _paired_summary(
            names[1],
            names[0],
            matrices[1],
            matrices[0],
            rng=rng,
            iterations=bootstrap_iterations,
        ),
        _paired_summary(
            names[2],
            names[0],
            matrices[2],
            matrices[0],
            rng=rng,
            iterations=bootstrap_iterations,
        ),
        _paired_summary(
            names[1],
            names[2],
            matrices[1],
            matrices[2],
            rng=rng,
            iterations=bootstrap_iterations,
        ),
    ]
    return {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "dataset": base["dataset"],
        "protocol": base["protocol"],
        "bootstrap": {
            "seed": bootstrap_seed,
            "iterations": bootstrap_iterations,
            "cluster_unit": "evaluation task averaged across seeds",
        },
        "methods": methods,
        "comparisons": comparisons,
        "claim": {
            "magrpo_improved_over_untrained": comparisons[0]["task_clustered_95_ci"][0]
            > 0,
            "magrpo_outperformed_grpo": comparisons[2]["task_clustered_95_ci"][0] > 0,
            "scope": "32 TLDR tasks x 3 seeds after an 8-example pilot update",
            "not_a_formal_1000_example_result": True,
        },
    }


def _plot(report: dict[str, Any], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = report["methods"]
    names = [method["name"].replace("_", "\n") for method in methods]
    means = [method["reward_mean"] for method in methods]
    lows = [
        mean - method["task_clustered_95_ci"][0] for mean, method in zip(means, methods)
    ]
    highs = [
        method["task_clustered_95_ci"][1] - mean for mean, method in zip(means, methods)
    ]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(names, means, color=("#9aa0a6", "#4c78a8", "#f58518"))
    axes[0].errorbar(
        range(len(means)),
        means,
        yerr=[lows, highs],
        fmt="none",
        color="black",
        capsize=4,
    )
    axes[0].set_ylabel("Official TLDR reward")
    axes[0].set_title("Equal-output-budget evaluation")

    comparisons = report["comparisons"]
    labels = [f"{item['candidate']}\n− {item['reference']}" for item in comparisons]
    deltas = [item["mean_reward_difference"] for item in comparisons]
    delta_lows = [
        delta - item["task_clustered_95_ci"][0]
        for delta, item in zip(deltas, comparisons)
    ]
    delta_highs = [
        item["task_clustered_95_ci"][1] - delta
        for delta, item in zip(deltas, comparisons)
    ]
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].errorbar(
        range(len(deltas)),
        deltas,
        yerr=[delta_lows, delta_highs],
        fmt="o",
        color="#4c78a8",
        capsize=4,
    )
    axes[1].set_xticks(range(len(labels)), labels, rotation=15, ha="right")
    axes[1].set_ylabel("Paired reward difference")
    axes[1].set_title("Task-clustered bootstrap 95% CI")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--magrpo", type=Path, required=True)
    parser.add_argument("--grpo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260833)
    parser.add_argument("--bootstrap-iterations", type=int, default=20_000)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.bootstrap_iterations < 1_000:
        raise ValueError("at least 1,000 bootstrap iterations are required")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    report = build_report(
        _load(args.base),
        _load(args.magrpo),
        _load(args.grpo),
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    (args.output / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    _plot(report, args.output / "comparison.png")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
