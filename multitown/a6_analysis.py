"""A6-specific counterfactual routing and organization-regret analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .a6_policy import ARM_ORDER
from .analysis import bootstrap_mean_interval
from .report import read_jsonl
from .runner import atomic_json, utc_now


def _scenario_metrics(frame: pd.DataFrame, token_penalty: float, latency_penalty: float) -> pd.DataFrame:
    work = frame.copy()
    work["correct"] = work["correct"].astype(float)
    work["total_tokens"] = pd.to_numeric(work["total_tokens"], errors="coerce")
    work["decision_latency_s"] = pd.to_numeric(work["decision_latency_s"], errors="coerce")
    grouped = work.groupby(["scenario_id", "family"], as_index=False).agg(
        accuracy=("correct", "mean"),
        tokens=("total_tokens", "mean"),
        latency_s=("decision_latency_s", "mean"),
        decisions=("correct", "size"),
    )
    grouped["utility"] = (
        grouped["accuracy"]
        - token_penalty * grouped["tokens"] / 1000.0
        - latency_penalty * grouped["latency_s"]
    )
    return grouped


def _interval(values: pd.Series) -> dict[str, float | None]:
    low, high = bootstrap_mean_interval(values)
    return {"mean": float(values.mean()), "ci95_low": low, "ci95_high": high}


def analyze_a6(run_root: str | Path) -> dict[str, Any]:
    root = Path(run_root).resolve()
    a6_dir = root / "A6" if (root / "A6").is_dir() else root
    policy = json.loads((a6_dir / "policy.json").read_text(encoding="utf-8"))
    status = json.loads((a6_dir / "status.json").read_text(encoding="utf-8"))
    if status.get("state") != "complete":
        raise RuntimeError(f"A6 is not complete: {a6_dir}")
    decisions = read_jsonl(a6_dir / "decisions.jsonl")
    utility_spec = policy["utility"]
    token_penalty = float(utility_spec["token_penalty_per_1k"])
    latency_penalty = float(utility_spec["latency_penalty_per_s"])
    a6 = _scenario_metrics(decisions, token_penalty, latency_penalty).set_index("scenario_id")

    baselines: dict[str, pd.DataFrame] = {}
    for arm in ARM_ORDER:
        baseline_dir = Path(policy["source_manifest"][arm]["run_dir"])
        frame = read_jsonl(baseline_dir / "decisions.jsonl")
        baselines[arm] = _scenario_metrics(
            frame, token_penalty, latency_penalty,
        ).set_index("scenario_id")

    counterfactual = a6[["family", "accuracy", "tokens", "latency_s", "utility"]].copy()
    counterfactual = counterfactual.rename(columns={
        "accuracy": "a6_accuracy", "tokens": "a6_tokens",
        "latency_s": "a6_latency_s", "utility": "a6_utility",
    })
    selection_rows = pd.DataFrame(policy["selections"].values()).set_index("scenario_id")
    counterfactual["selected_arm"] = selection_rows["selected_arm"]
    counterfactual["fold"] = selection_rows["fold"]
    counterfactual["predicted_utility"] = selection_rows["predicted_utility"]
    for arm, frame in baselines.items():
        counterfactual[f"{arm.lower()}_accuracy"] = frame["accuracy"]
        counterfactual[f"{arm.lower()}_tokens"] = frame["tokens"]
        counterfactual[f"{arm.lower()}_latency_s"] = frame["latency_s"]
        counterfactual[f"{arm.lower()}_utility"] = frame["utility"]
    utility_columns = [f"{arm.lower()}_utility" for arm in ARM_ORDER]
    counterfactual["evaluation_oracle_utility"] = counterfactual[utility_columns].max(axis=1)
    counterfactual["evaluation_oracle_arm"] = counterfactual[utility_columns].idxmax(axis=1).str[:2].str.upper()
    counterfactual["organization_regret"] = (
        counterfactual["evaluation_oracle_utility"] - counterfactual["a6_utility"]
    )
    counterfactual["reorganization_gain_vs_a4"] = (
        counterfactual["a6_utility"] - counterfactual["a4_utility"]
    )
    counterfactual["accuracy_gain_vs_a4"] = (
        counterfactual["a6_accuracy"] - counterfactual["a4_accuracy"]
    )
    counterfactual["token_delta_vs_a4"] = (
        counterfactual["a6_tokens"] - counterfactual["a4_tokens"]
    )
    counterfactual["latency_delta_vs_a4"] = (
        counterfactual["a6_latency_s"] - counterfactual["a4_latency_s"]
    )
    counterfactual["prediction_error"] = (
        counterfactual["a6_utility"] - counterfactual["predicted_utility"]
    )
    counterfactual.reset_index().to_csv(a6_dir / "counterfactual_by_scenario.csv", index=False)

    baseline_overall = {
        arm: {
            "accuracy": float(frame["accuracy"].mean()),
            "tokens_per_decision": float(frame["tokens"].mean()),
            "latency_mean_s": float(frame["latency_s"].mean()),
            "utility": float(frame["utility"].mean()),
        }
        for arm, frame in baselines.items()
    }
    best_fixed_arm = max(ARM_ORDER, key=lambda arm: baseline_overall[arm]["utility"])
    actual = {
        "accuracy": float(a6["accuracy"].mean()),
        "tokens_per_decision": float(a6["tokens"].mean()),
        "latency_mean_s": float(a6["latency_s"].mean()),
        "utility": float(a6["utility"].mean()),
    }
    arm_table = decisions.groupby("selected_arm", as_index=False).agg(
        decisions=("correct", "size"),
        accuracy=("correct", "mean"),
        tokens_per_decision=("total_tokens", "mean"),
        latency_mean_s=("decision_latency_s", "mean"),
    )
    arm_table["share"] = arm_table["decisions"] / len(decisions)
    arm_table.to_csv(a6_dir / "selected_arm_results.csv", index=False)
    fold_table = decisions.groupby("policy_fold", as_index=False).agg(
        decisions=("correct", "size"),
        accuracy=("correct", "mean"),
        tokens_per_decision=("total_tokens", "mean"),
        latency_mean_s=("decision_latency_s", "mean"),
    )
    fold_table.to_csv(a6_dir / "fold_results.csv", index=False)

    payload: dict[str, Any] = {
        "analyzed_at_utc": utc_now(),
        "run_dir": str(a6_dir),
        "policy_version": policy["policy_version"],
        "utility": utility_spec,
        "actual_a6": actual,
        "crossfit_offline_estimate": policy["crossfit_offline_estimate"],
        "baseline_overall": baseline_overall,
        "best_fixed_arm_by_utility": best_fixed_arm,
        "reorganization_gain_vs_a4": _interval(counterfactual["reorganization_gain_vs_a4"]),
        "accuracy_gain_vs_a4": _interval(counterfactual["accuracy_gain_vs_a4"]),
        "token_delta_vs_a4": _interval(counterfactual["token_delta_vs_a4"]),
        "latency_delta_vs_a4": _interval(counterfactual["latency_delta_vs_a4"]),
        "token_reduction_vs_a4_percent": float(
            (1.0 - actual["tokens_per_decision"] / baseline_overall["A4"]["tokens_per_decision"]) * 100
        ),
        "latency_reduction_vs_a4_percent": float(
            (1.0 - actual["latency_mean_s"] / baseline_overall["A4"]["latency_mean_s"]) * 100
        ),
        "organization_regret": _interval(counterfactual["organization_regret"]),
        "policy_prediction_error": _interval(counterfactual["prediction_error"]),
        "selected_arm_matches_evaluation_oracle_rate": float(
            (counterfactual["selected_arm"] == counterfactual["evaluation_oracle_arm"]).mean()
        ),
        "fold_accuracy_range": {
            "minimum": float(fold_table["accuracy"].min()),
            "maximum": float(fold_table["accuracy"].max()),
            "spread": float(fold_table["accuracy"].max() - fold_table["accuracy"].min()),
        },
        "selected_arm_results": arm_table.to_dict(orient="records"),
        "fold_results": fold_table.to_dict(orient="records"),
        "evaluation_oracle_note": (
            "The per-scenario oracle is used only after A6 completes and is an optimistic diagnostic, "
            "never an input to routing."
        ),
    }
    atomic_json(a6_dir / "a6_analysis.json", payload)

    offline = policy["crossfit_offline_estimate"]
    a4 = baseline_overall["A4"]
    gain = payload["reorganization_gain_vs_a4"]
    accuracy_gain = payload["accuracy_gain_vs_a4"]
    regret = payload["organization_regret"]
    token_delta = payload["token_delta_vs_a4"]
    latency_delta = payload["latency_delta_vs_a4"]
    lines = [
        "# A6 budget-aware routing analysis",
        "",
        "A6 uses five-fold scenario-level cross-fitting. Each scenario is routed using only "
        "A0-A5 outcomes from the other four folds; current-scenario oracle labels, outcomes, "
        "and self-reported confidence are excluded.",
        "",
        "## Actual versus pre-run estimate",
        "",
        "| Metric | Cross-fit estimate | Formal A6 | Fixed A4 |",
        "|---|---:|---:|---:|",
        f"| Accuracy | {offline['accuracy']:.3%} | {actual['accuracy']:.3%} | {a4['accuracy']:.3%} |",
        f"| Tokens/decision | {offline['tokens_per_decision']:,.1f} | {actual['tokens_per_decision']:,.1f} | {a4['tokens_per_decision']:,.1f} |",
        f"| Mean latency (s) | {offline['latency_mean_s']:.3f} | {actual['latency_mean_s']:.3f} | {a4['latency_mean_s']:.3f} |",
        f"| Utility | {offline['utility']:.4f} | {actual['utility']:.4f} | {a4['utility']:.4f} |",
        "",
        "## Routing diagnostics",
        "",
        f"- Reorganization Gain vs A4: {gain['mean']:+.4f} "
        f"(scenario-bootstrap 95% CI [{gain['ci95_low']:+.4f}, {gain['ci95_high']:+.4f}]).",
        f"- Accuracy gain vs A4: {accuracy_gain['mean']:+.3%} "
        f"(95% CI [{accuracy_gain['ci95_low']:+.3%}, {accuracy_gain['ci95_high']:+.3%}]).",
        f"- Token delta vs A4: {token_delta['mean']:+,.1f} per decision "
        f"(95% CI [{token_delta['ci95_low']:+,.1f}, {token_delta['ci95_high']:+,.1f}]); "
        f"aggregate reduction {payload['token_reduction_vs_a4_percent']:.2f}%.",
        f"- Mean-latency delta vs A4: {latency_delta['mean']:+.3f}s "
        f"(95% CI [{latency_delta['ci95_low']:+.3f}, {latency_delta['ci95_high']:+.3f}]); "
        f"aggregate reduction {payload['latency_reduction_vs_a4_percent']:.2f}%.",
        f"- Organization Regret against the evaluation-only per-scenario oracle: "
        f"{regret['mean']:.4f} (95% CI [{regret['ci95_low']:.4f}, {regret['ci95_high']:.4f}]).",
        f"- Selected arm matches the evaluation-only per-scenario oracle on "
        f"{payload['selected_arm_matches_evaluation_oracle_rate']:.3%} of scenarios.",
        f"- Fold accuracy range: {payload['fold_accuracy_range']['minimum']:.3%} to "
        f"{payload['fold_accuracy_range']['maximum']:.3%} "
        f"(spread {payload['fold_accuracy_range']['spread']:.3%}).",
        f"- Best fixed arm by the declared utility: {best_fixed_arm}.",
        "",
        "The evaluation oracle is an optimistic post-hoc diagnostic and was never visible to the router.",
        "",
        "## Selected-arm results",
        "",
        "| Arm | Share | Decisions | Accuracy | Tokens/decision | Mean latency (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in arm_table.to_dict(orient="records"):
        lines.append(
            f"| {row['selected_arm']} | {row['share']:.3%} | {int(row['decisions']):,} | "
            f"{row['accuracy']:.3%} | {row['tokens_per_decision']:,.1f} | "
            f"{row['latency_mean_s']:.3f} |"
        )
    lines.append("")
    (a6_dir / "A6_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="A6 run root or the A6 directory")
    args = parser.parse_args()
    print(json.dumps(analyze_a6(args.path), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
