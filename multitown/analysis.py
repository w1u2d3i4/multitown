"""Post-run statistical analysis for completed MultiTown architectures."""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .report import read_jsonl


ARCHITECTURES = ("A0", "A1", "A2", "A3", "A4", "A5", "A6")
FAMILY_ORDER = (
    "resource_allocation",
    "incident_dispatch",
    "evidence_fusion",
    "dependency_recovery",
    "supply_route",
    "fault_recovery",
)


def wilson_interval(successes: int, count: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    """Return a 95% Wilson score interval for a binomial proportion."""
    if count <= 0:
        return None, None
    proportion = successes / count
    denominator = 1 + z * z / count
    centre = (proportion + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / count + z * z / (4 * count * count)) / denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)


def bootstrap_mean_interval(
    values: pd.Series | np.ndarray, *, seed: int = 20_260_807, repetitions: int = 20_000,
) -> tuple[float | None, float | None]:
    """Deterministic percentile bootstrap interval over scenario-level means."""
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if not len(values):
        return None, None
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    bootstrap_means = values[indices].mean(axis=1)
    low, high = np.percentile(bootstrap_means, [2.5, 97.5])
    return float(low), float(high)


def load_completed(root: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for architecture in ARCHITECTURES:
        status_path = root / architecture / "status.json"
        decisions_path = root / architecture / "decisions.jsonl"
        if not status_path.exists() or not decisions_path.exists():
            continue
        with status_path.open("r", encoding="utf-8") as handle:
            status = json.load(handle)
        if status.get("state") != "complete":
            continue
        frame = read_jsonl(decisions_path)
        if not frame.empty:
            frames[architecture] = frame
    return frames


def overall_table(root: Path, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine aggregate run metrics with scenario-cluster uncertainty."""
    rows: list[dict[str, Any]] = []
    for architecture in ARCHITECTURES:
        if architecture not in frames:
            continue
        summary_path = root / architecture / "summary.json"
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        frame = frames[architecture]
        by_scenario = frame.assign(_correct=frame["correct"].astype(float)).groupby("scenario_id")["_correct"].mean()
        low, high = bootstrap_mean_interval(by_scenario)
        summary["scenario_cluster_ci95_low"] = low
        summary["scenario_cluster_ci95_high"] = high
        rows.append(summary)
    return pd.DataFrame(rows)


def family_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for architecture, frame in frames.items():
        for family, group in frame.groupby("family", sort=False):
            correct = int(group["correct"].astype(bool).sum())
            count = int(len(group))
            by_scenario = group.assign(_correct=group["correct"].astype(float)).groupby("scenario_id")["_correct"].mean()
            ci_low, ci_high = bootstrap_mean_interval(by_scenario)
            rows.append({
                "architecture": architecture,
                "family": family,
                "decisions": count,
                "correct": correct,
                "accuracy": correct / count,
                "scenario_cluster_ci95_low": ci_low,
                "scenario_cluster_ci95_high": ci_high,
                "invalid_rate": float((~group["valid"].astype(bool)).mean()),
                "latency_mean_s": float(pd.to_numeric(group["decision_latency_s"], errors="coerce").mean()),
                "tokens_per_decision": float(pd.to_numeric(group["total_tokens"], errors="coerce").mean()),
            })
    return pd.DataFrame(rows)


def pairwise_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    key = ["trial_index", "scenario_id"]
    for left, right in combinations(ARCHITECTURES, 2):
        if left not in frames or right not in frames:
            continue
        left_frame = frames[left][key + ["correct"]].rename(columns={"correct": "left_correct"})
        right_frame = frames[right][key + ["correct"]].rename(columns={"correct": "right_correct"})
        paired = left_frame.merge(right_frame, on=key, how="inner", validate="one_to_one")
        left_correct = paired["left_correct"].astype(bool)
        right_correct = paired["right_correct"].astype(bool)
        difference = right_correct.astype(float) - left_correct.astype(float)
        scenario_difference = paired.assign(_difference=difference).groupby("scenario_id")["_difference"].mean()
        ci_low, ci_high = bootstrap_mean_interval(scenario_difference)
        rows.append({
            "comparison": f"{right}-{left}",
            "left_architecture": left,
            "right_architecture": right,
            "common_decisions": int(len(paired)),
            "left_accuracy": float(left_correct.mean()),
            "right_accuracy": float(right_correct.mean()),
            "accuracy_difference": float(difference.mean()),
            "scenario_cluster_difference_ci95_low": ci_low,
            "scenario_cluster_difference_ci95_high": ci_high,
            "right_wins": int((right_correct & ~left_correct).sum()),
            "left_wins": int((left_correct & ~right_correct).sum()),
            "both_correct": int((right_correct & left_correct).sum()),
            "both_wrong": int((~right_correct & ~left_correct).sum()),
        })
    return pd.DataFrame(rows)


def ensemble_details(root: Path, frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    if "A2" not in frames:
        return {}
    decisions = frames["A2"]
    requests = read_jsonl(root / "A2" / "requests.jsonl")
    correct = decisions["correct"].astype(bool)
    agents: list[dict[str, Any]] = []
    if not requests.empty:
        for agent_index, group in requests.groupby("agent_index"):
            individual = group["correct_individual"].astype(bool)
            by_scenario = group.assign(_correct=individual.astype(float)).groupby("scenario_id")["_correct"].mean()
            low, high = bootstrap_mean_interval(by_scenario, seed=20_260_807 + int(agent_index))
            agents.append({
                "agent_index": int(agent_index),
                "requests": int(len(group)),
                "accuracy": float(individual.mean()),
                "scenario_cluster_ci95_low": low,
                "scenario_cluster_ci95_high": high,
                "invalid_rate": float((~group["valid"].astype(bool)).mean()),
            })
    agreement = pd.to_numeric(decisions["agreement"], errors="coerce")
    diversity = pd.to_numeric(decisions["action_diversity"], errors="coerce")
    by_scenario = decisions.assign(_correct=correct.astype(float)).groupby("scenario_id")["_correct"].mean()
    ensemble_low, ensemble_high = bootstrap_mean_interval(by_scenario)
    mean_agent_accuracy = float(np.mean([row["accuracy"] for row in agents])) if agents else None
    individual_fraction = decisions["agent_correct"].apply(
        lambda values: float(np.mean(values)) if isinstance(values, list) and values else math.nan
    )
    lift_by_scenario = decisions.assign(
        _lift=correct.astype(float) - individual_fraction
    ).groupby("scenario_id")["_lift"].mean()
    lift_low, lift_high = bootstrap_mean_interval(lift_by_scenario)
    return {
        "decisions": int(len(decisions)),
        "ensemble_accuracy": float(correct.mean()),
        "ensemble_scenario_cluster_ci95_low": ensemble_low,
        "ensemble_scenario_cluster_ci95_high": ensemble_high,
        "individual_agents": agents,
        "mean_individual_accuracy": mean_agent_accuracy,
        "ensemble_lift_over_mean_individual": (
            float(correct.mean()) - mean_agent_accuracy if mean_agent_accuracy is not None else None
        ),
        "ensemble_lift_scenario_cluster_ci95_low": lift_low,
        "ensemble_lift_scenario_cluster_ci95_high": lift_high,
        "mean_vote_agreement": float(agreement.mean()),
        "unanimous_rate": float((agreement == 1.0).mean()),
        "split_vote_rate": float((agreement <= 0.5).mean()),
        "mean_action_diversity": float(diversity.mean()),
    }


def organization_tables(
    frames: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize organization overhead and dynamic routing choices."""
    overview_rows: list[dict[str, Any]] = []
    route_rows: list[dict[str, Any]] = []
    for architecture, frame in frames.items():
        if "route" not in frame:
            continue
        overview: dict[str, Any] = {
            "architecture": architecture,
            "decisions": int(len(frame)),
        }
        for column, output in [
            ("weak_calls", "mean_weak_calls"),
            ("strong_calls", "mean_strong_calls"),
            ("organization_switches", "mean_organization_switches"),
            ("request_count", "mean_request_count"),
            ("communication_messages", "mean_communication_messages"),
            ("weak_tokens", "mean_weak_tokens"),
            ("strong_tokens", "mean_strong_tokens"),
        ]:
            if column in frame:
                overview[output] = float(pd.to_numeric(frame[column], errors="coerce").mean())
        if "verifier_called" in frame:
            overview["verifier_rate"] = float(frame["verifier_called"].astype(bool).mean())
        if "strict_json_rate" in frame:
            overview["mean_episode_strict_json_rate"] = float(
                pd.to_numeric(frame["strict_json_rate"], errors="coerce").mean()
            )
        if "strict_json_calls" in frame and "candidate_actions" in frame:
            strict_denominator = int(frame["candidate_actions"].apply(
                lambda value: len(value) if isinstance(value, list) else 0
            ).sum())
            overview["strict_json_request_rate"] = (
                float(pd.to_numeric(frame["strict_json_calls"], errors="coerce").fillna(0).sum())
                / strict_denominator if strict_denominator else math.nan
            )
        overview_rows.append(overview)

        for route, group in frame.groupby("route", dropna=False):
            strict_denominator = int(group.get("candidate_actions", pd.Series([], dtype=object)).apply(
                lambda value: len(value) if isinstance(value, list) else 0
            ).sum())
            route_rows.append({
                "architecture": architecture,
                "route": str(route),
                "decisions": int(len(group)),
                "route_share": float(len(group) / len(frame)),
                "accuracy": float(group["correct"].astype(bool).mean()),
                "invalid_rate": float((~group["valid"].astype(bool)).mean()),
                "tokens_per_decision": float(pd.to_numeric(group["total_tokens"], errors="coerce").mean()),
                "latency_mean_s": float(pd.to_numeric(group["decision_latency_s"], errors="coerce").mean()),
                "mean_weak_calls": float(pd.to_numeric(group.get("weak_calls", 0), errors="coerce").mean()),
                "mean_strong_calls": float(pd.to_numeric(group.get("strong_calls", 0), errors="coerce").mean()),
                "verifier_rate": float(group.get("verifier_called", pd.Series(False, index=group.index)).astype(bool).mean()),
                "strict_json_rate": (
                    float(pd.to_numeric(group.get("strict_json_calls", 0), errors="coerce").fillna(0).sum())
                    / strict_denominator if strict_denominator else math.nan
                ),
            })
    return pd.DataFrame(overview_rows), pd.DataFrame(route_rows)


def write_markdown_summary(
    root: Path, overall: pd.DataFrame, families: pd.DataFrame,
    pairs: pd.DataFrame, ensemble: dict[str, Any],
    organizations: pd.DataFrame, routes: pd.DataFrame,
) -> None:
    lines = [
        "# MultiTown Bench formal results",
        "",
        "Only completed timed runs are included. Model load and warm-up are excluded. "
        "Accuracy intervals resample the 120 scenario IDs, not repeated calls.",
        "",
        "## Overall",
        "",
        "| Architecture | Decisions | Accuracy | Scenario-cluster 95% CI | Decisions/hour | P95 latency (s) | Tokens/correct | GPU energy (kWh) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall.to_dict(orient="records"):
        energy = row.get("gpu_energy_kwh")
        energy_text = "n/a" if energy is None or pd.isna(energy) else f"{energy:.3f}"
        token_cost = row.get("tokens_per_correct")
        token_text = "n/a" if token_cost is None or pd.isna(token_cost) else f"{token_cost:,.1f}"
        lines.append(
            f"| {row['architecture']} | {int(row['decisions']):,} | {row['accuracy']:.3%} | "
            f"[{row['scenario_cluster_ci95_low']:.3%}, {row['scenario_cluster_ci95_high']:.3%}] | "
            f"{row['decisions_per_hour']:,.0f} | {row['latency_p95_s']:.3f} | "
            f"{token_text} | {energy_text} |"
        )
    lines.extend([
        "",
        "## Paired common-trial differences",
        "",
        "| Comparison | Common decisions | Accuracy difference | Scenario-cluster 95% CI | Right wins | Left wins |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in pairs.to_dict(orient="records"):
        lines.append(
            f"| {row['comparison']} | {int(row['common_decisions']):,} | {row['accuracy_difference']:+.3%} | "
            f"[{row['scenario_cluster_difference_ci95_low']:+.3%}, "
            f"{row['scenario_cluster_difference_ci95_high']:+.3%}] | "
            f"{int(row['right_wins']):,} | {int(row['left_wins']):,} |"
        )
    present = [architecture for architecture in ARCHITECTURES if architecture in set(families["architecture"])]
    lines.extend([
        "",
        "## Accuracy by family",
        "",
        "| Family | " + " | ".join(present) + " |",
        "|---|" + "---:|" * len(present),
    ])
    family_pivot = families.pivot(index="family", columns="architecture", values="accuracy")
    for family in FAMILY_ORDER:
        if family not in family_pivot.index:
            continue
        values = family_pivot.loc[family]
        cells = [f"{float(values[architecture]):.3%}" if architecture in values and pd.notna(values[architecture]) else "n/a" for architecture in present]
        lines.append(f"| {family} | " + " | ".join(cells) + " |")

    if not organizations.empty:
        lines.extend([
            "",
            "## Organization overhead",
            "",
            "| Architecture | Weak calls | Strong calls | Switches | Messages | Verifier rate | Strict JSON |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for row in organizations.to_dict(orient="records"):
            lines.append(
                f"| {row['architecture']} | {row.get('mean_weak_calls', math.nan):.2f} | "
                f"{row.get('mean_strong_calls', math.nan):.2f} | "
                f"{row.get('mean_organization_switches', math.nan):.2f} | "
                f"{row.get('mean_communication_messages', math.nan):.2f} | "
                f"{row.get('verifier_rate', math.nan):.3%} | "
                f"{row.get('strict_json_request_rate', row.get('mean_episode_strict_json_rate', math.nan)):.3%} |"
            )
    if not routes.empty:
        lines.extend([
            "",
            "## Route distribution",
            "",
            "| Architecture | Route | Share | Accuracy | Tokens/decision | Mean latency (s) |",
            "|---|---|---:|---:|---:|---:|",
        ])
        for row in routes.to_dict(orient="records"):
            lines.append(
                f"| {row['architecture']} | {row['route']} | {row['route_share']:.3%} | "
                f"{row['accuracy']:.3%} | {row['tokens_per_decision']:,.1f} | {row['latency_mean_s']:.3f} |"
            )

    if ensemble:
        lines.extend([
            "",
            "## A2 ensemble",
            "",
            f"- Ensemble accuracy: {ensemble['ensemble_accuracy']:.3%}",
            f"- Mean individual accuracy: {ensemble['mean_individual_accuracy']:.3%}",
            f"- Ensemble lift: {ensemble['ensemble_lift_over_mean_individual']:+.3%} "
            f"(scenario-cluster 95% CI [{ensemble['ensemble_lift_scenario_cluster_ci95_low']:+.3%}, "
            f"{ensemble['ensemble_lift_scenario_cluster_ci95_high']:+.3%}])",
            f"- Mean vote agreement: {ensemble['mean_vote_agreement']:.3%}",
            f"- Unanimous rate: {ensemble['unanimous_rate']:.3%}",
            f"- Split-vote rate: {ensemble['split_vote_rate']:.3%}",
        ])
    errors = int(pd.to_numeric(overall.get("request_errors", 0), errors="coerce").fillna(0).sum())
    lines.extend(["", f"Total recorded request errors across included runs: {errors}.", ""])
    with (root / "RESULTS.md").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def plot_family_accuracy(frame: pd.DataFrame, output: Path) -> None:
    if frame.empty:
        return
    pivot = frame.pivot(index="family", columns="architecture", values="accuracy").reindex(FAMILY_ORDER)
    axes = pivot.plot.bar(figsize=(14, 7), ylim=(0, 1.02), width=0.8)
    axes.set(title="Accuracy by scenario family", xlabel="Scenario family", ylabel="Accuracy")
    axes.grid(axis="y", alpha=0.25)
    axes.legend(title="Architecture")
    axes.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def plot_accuracy_ci(frames: dict[str, pd.DataFrame], output: Path) -> None:
    labels: list[str] = []
    means: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    for architecture in ARCHITECTURES:
        if architecture not in frames:
            continue
        values = frames[architecture]["correct"].astype(bool)
        mean = float(values.mean())
        by_scenario = frames[architecture].assign(_correct=values.astype(float)).groupby("scenario_id")["_correct"].mean()
        low, high = bootstrap_mean_interval(by_scenario)
        labels.append(architecture)
        means.append(mean)
        lower.append(mean - float(low))
        upper.append(float(high) - mean)
    if not labels:
        return
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    x = np.arange(len(labels))
    axis.errorbar(x, means, yerr=np.array([lower, upper]), fmt="o", capsize=7, markersize=8)
    axis.set(xticks=x, xticklabels=labels, ylim=(0, 1.02), ylabel="Accuracy", title="Overall accuracy with scenario-cluster bootstrap 95% CI")
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def analyze(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    frames = load_completed(root)
    if not frames:
        raise RuntimeError(f"No completed architecture found under {root}")
    overall = overall_table(root, frames)
    families = family_table(frames)
    pairs = pairwise_table(frames)
    ensemble = ensemble_details(root, frames)
    organizations, routes = organization_tables(frames)
    overall.to_csv(root / "overall_accuracy.csv", index=False)
    families.to_csv(root / "family_comparison.csv", index=False)
    pairs.to_csv(root / "pairwise_comparison.csv", index=False)
    organizations.to_csv(root / "organization_comparison.csv", index=False)
    routes.to_csv(root / "route_comparison.csv", index=False)
    plot_family_accuracy(families, root / "family_accuracy.png")
    plot_accuracy_ci(frames, root / "accuracy_confidence.png")
    write_markdown_summary(root, overall, families, pairs, ensemble, organizations, routes)
    payload = {
        "completed_architectures": list(frames),
        "overall_rows": overall.to_dict(orient="records"),
        "family_rows": families.to_dict(orient="records"),
        "pairwise_rows": pairs.where(pd.notnull(pairs), None).to_dict(orient="records"),
        "a2_ensemble": ensemble,
        "organization_rows": organizations.where(pd.notnull(organizations), None).to_dict(orient="records"),
        "route_rows": routes.where(pd.notnull(routes), None).to_dict(orient="records"),
    }
    with (root / "detailed_analysis.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Run root containing A0-A6 architecture directories")
    args = parser.parse_args()
    result = analyze(args.path)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
