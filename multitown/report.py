"""Generate live and final MultiTown benchmark curves and comparison tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def discover_architectures(root: Path) -> list[str]:
    """Return data-bearing architecture directories in numeric order."""
    found = [
        child.name for child in root.iterdir()
        if child.is_dir() and child.name.startswith("A")
        and child.name[1:].isdigit() and (child / "decisions.jsonl").exists()
    ]
    return sorted(found, key=lambda value: int(value[1:]))


def read_jsonl(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return pd.DataFrame()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)


def percentile(series: pd.Series, value: float) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return None if clean.empty else float(np.percentile(clean, value))


def expected_calibration_error(confidence: pd.Series, correct: pd.Series, bins: int = 10) -> float | None:
    valid = confidence.notna() & correct.notna()
    if not valid.any():
        return None
    conf = confidence[valid].clip(0, 1)
    outcome = correct[valid].astype(float)
    bucket = pd.cut(conf, bins=np.linspace(0, 1, bins + 1), include_lowest=True, labels=False)
    total = len(conf)
    error = 0.0
    for index in range(bins):
        selected = bucket == index
        if selected.any():
            error += float(selected.sum() / total) * abs(float(conf[selected].mean()) - float(outcome[selected].mean()))
    return error


def generate_run_report(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    decisions = read_jsonl(run_dir / "decisions.jsonl")
    requests = read_jsonl(run_dir / "requests.jsonl")
    system = read_jsonl(run_dir / "system_metrics.jsonl")
    if decisions.empty:
        return {}
    architecture = str(decisions.iloc[0]["architecture"])
    elapsed_hours = float(decisions["elapsed_s"].max()) / 3600
    correct = decisions["correct"].astype(bool)
    confidence = pd.to_numeric(decisions["confidence"], errors="coerce")
    total_tokens = pd.to_numeric(decisions["total_tokens"], errors="coerce").fillna(0)
    latency = pd.to_numeric(decisions["decision_latency_s"], errors="coerce")
    summary: dict[str, Any] = {
        "architecture": architecture,
        "decisions": int(len(decisions)),
        "correct": int(correct.sum()),
        "accuracy": float(correct.mean()),
        "invalid_rate": float((~decisions["valid"].astype(bool)).mean()),
        "request_errors": int(pd.to_numeric(decisions["request_errors"], errors="coerce").fillna(0).sum()),
        "wall_hours": elapsed_hours,
        "decisions_per_hour": float(len(decisions) / elapsed_hours) if elapsed_hours else None,
        "total_tokens": int(total_tokens.sum()),
        "tokens_per_decision": float(total_tokens.mean()),
        "tokens_per_correct": float(total_tokens.sum() / correct.sum()) if correct.sum() else None,
        "latency_mean_s": float(latency.mean()),
        "latency_p50_s": percentile(latency, 50),
        "latency_p95_s": percentile(latency, 95),
        "latency_p99_s": percentile(latency, 99),
        "ttft_p50_s": percentile(requests.get("ttft_s", pd.Series(dtype=float)), 50),
        "ttft_p95_s": percentile(requests.get("ttft_s", pd.Series(dtype=float)), 95),
        "mean_agreement": float(pd.to_numeric(decisions["agreement"], errors="coerce").mean()),
        "mean_confidence": float(confidence.mean()),
        "confidence_accuracy_gap": float(confidence.mean() - correct.mean()),
        "confidence_brier_score": float(((confidence - correct.astype(float)) ** 2).mean()),
        "confidence_ece_10bin": expected_calibration_error(confidence, correct.astype(float)),
    }
    if "strict_json_rate" in decisions:
        summary["mean_episode_strict_json_rate"] = float(
            pd.to_numeric(decisions["strict_json_rate"], errors="coerce").mean()
        )
    if "strict_json_compliant" in requests:
        decision_requests = requests[requests.get("expects_decision", True).fillna(False).astype(bool)] if "expects_decision" in requests else requests
        if not decision_requests.empty:
            summary["strict_json_request_rate"] = float(
                decision_requests["strict_json_compliant"].fillna(False).astype(bool).mean()
            )
    for column, output_name in [
        ("weak_calls", "mean_weak_calls"),
        ("strong_calls", "mean_strong_calls"),
        ("organization_switches", "mean_organization_switches"),
        ("request_count", "mean_request_count"),
        ("communication_messages", "mean_communication_messages"),
        ("weak_tokens", "mean_weak_tokens"),
        ("strong_tokens", "mean_strong_tokens"),
    ]:
        if column in decisions:
            summary[output_name] = float(pd.to_numeric(decisions[column], errors="coerce").mean())
    if "verifier_called" in decisions:
        summary["verifier_rate"] = float(decisions["verifier_called"].astype(bool).mean())
    if "route" in decisions:
        summary["route_counts"] = {
            str(name): int(count) for name, count in decisions["route"].value_counts().items()
        }
    if not system.empty and "gpu_power_w" in system:
        power = pd.to_numeric(system["gpu_power_w"], errors="coerce")
        times = pd.to_numeric(system["elapsed_s"], errors="coerce")
        valid = power.notna() & times.notna()
        if valid.sum() >= 2:
            summary["gpu_energy_kwh"] = float(np.trapezoid(power[valid], times[valid]) / 3_600_000)
            summary["gpu_power_mean_w"] = float(power[valid].mean())
        else:
            summary["gpu_energy_kwh"] = None
            summary["gpu_power_mean_w"] = None
    else:
        summary["gpu_energy_kwh"] = None
        summary["gpu_power_mean_w"] = None
    for column, output_name, aggregate in [
        ("gpu_util_percent", "gpu_util_mean_percent", "mean"),
        ("gpu_temp_c", "gpu_temp_mean_c", "mean"),
        ("gpu_temp_c", "gpu_temp_max_c", "max"),
        ("ram_used_gb", "ram_used_mean_gb", "mean"),
        ("ram_used_gb", "ram_used_max_gb", "max"),
        ("cpu_percent", "cpu_mean_percent", "mean"),
        ("cpu_percent", "cpu_max_percent", "max"),
    ]:
        if not system.empty and column in system:
            values = pd.to_numeric(system[column], errors="coerce").dropna()
            summary[output_name] = float(getattr(values, aggregate)()) if not values.empty else None
        else:
            summary[output_name] = None
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    pd.DataFrame([summary]).to_csv(run_dir / "summary.csv", index=False)

    x = pd.to_numeric(decisions["elapsed_s"], errors="coerce") / 3600
    window = min(50, max(5, len(decisions) // 20))
    rolling_accuracy = correct.astype(float).rolling(window, min_periods=1).mean()
    rolling_latency = latency.rolling(window, min_periods=1).median()
    rolling_rate = pd.Series(range(1, len(decisions) + 1)) / x.clip(lower=1 / 3600)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    axes[0, 0].plot(x, rolling_accuracy, linewidth=1.4)
    axes[0, 0].axhline(float(correct.mean()), linestyle="--", linewidth=1)
    axes[0, 0].set(title=f"{architecture} rolling accuracy", xlabel="Elapsed hours", ylabel="Accuracy", ylim=(0, 1.02))
    axes[0, 1].plot(x, total_tokens.cumsum() / 1_000_000, linewidth=1.4)
    axes[0, 1].set(title="Cumulative model tokens", xlabel="Elapsed hours", ylabel="Million tokens")
    axes[1, 0].plot(x, rolling_latency, linewidth=1.4)
    axes[1, 0].set(title="Rolling median decision latency", xlabel="Elapsed hours", ylabel="Seconds")
    axes[1, 1].plot(x, rolling_rate, linewidth=1.4)
    axes[1, 1].set(title="Cumulative decision throughput", xlabel="Elapsed hours", ylabel="Decisions/hour")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    fig.savefig(run_dir / "decision_curves.png", dpi=150)
    plt.close(fig)

    valid_calibration = confidence.notna()
    if valid_calibration.any():
        calibration = pd.DataFrame({
            "confidence": confidence[valid_calibration].clip(0, 1),
            "correct": correct[valid_calibration].astype(float),
        })
        calibration["bin"] = pd.cut(
            calibration["confidence"], bins=np.linspace(0, 1, 11),
            include_lowest=True, labels=False,
        )
        grouped = calibration.groupby("bin", observed=False).agg(
            mean_confidence=("confidence", "mean"), accuracy=("correct", "mean"), count=("correct", "size")
        ).dropna()
        grouped.to_csv(run_dir / "confidence_calibration.csv", index_label="bin")
        fig, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
        axis.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfect calibration")
        if not grouped.empty:
            axis.plot(grouped["mean_confidence"], grouped["accuracy"], marker="o", label=architecture)
        axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean reported confidence", ylabel="Observed accuracy", title=f"{architecture} confidence calibration")
        axis.grid(alpha=0.25)
        axis.legend()
        fig.savefig(run_dir / "confidence_calibration.png", dpi=150)
        plt.close(fig)

    if not system.empty:
        sx = pd.to_numeric(system["elapsed_s"], errors="coerce") / 3600
        fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True, constrained_layout=True)
        for axis, column, label in [
            (axes[0, 0], "gpu_util_percent", "GPU utilization (%)"),
            (axes[0, 1], "gpu_power_w", "GPU power (W)"),
            (axes[1, 0], "gpu_temp_c", "GPU temperature (C)"),
            (axes[1, 1], "gpu_clock_mhz", "GPU SM clock (MHz)"),
            (axes[2, 0], "ram_used_gb", "System RAM used (GiB)"),
            (axes[2, 1], "cpu_percent", "CPU utilization (%)"),
        ]:
            if column in system:
                axis.plot(sx, pd.to_numeric(system[column], errors="coerce"), linewidth=1)
            axis.set_ylabel(label)
            axis.grid(alpha=0.25)
        for axis in axes[-1, :]:
            axis.set_xlabel("Elapsed hours")
        fig.suptitle(f"{architecture} system monitoring")
        fig.savefig(run_dir / "system_curves.png", dpi=150)
        plt.close(fig)

        server_columns = [
            column for column in system.columns
            if column.startswith(("strong_", "weak_"))
            and not column.endswith("metrics_error")
        ]
        if server_columns:
            fig, axes = plt.subplots(
                len(server_columns), 1,
                figsize=(13, max(4, 2.3 * len(server_columns))),
                sharex=True, constrained_layout=True,
            )
            axes_array = np.atleast_1d(axes)
            for axis, column in zip(axes_array, server_columns, strict=True):
                axis.plot(sx, pd.to_numeric(system[column], errors="coerce"), linewidth=1)
                axis.set_ylabel(column.replace("_llamacpp_", "\n"), fontsize=8)
                axis.grid(alpha=0.25)
            axes_array[-1].set_xlabel("Elapsed hours")
            fig.suptitle(f"{architecture} per-model llama.cpp metrics")
            fig.savefig(run_dir / "model_server_curves.png", dpi=150)
            plt.close(fig)

    if "route" in decisions:
        routes = sorted(str(value) for value in decisions["route"].dropna().unique())
        if routes:
            fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, constrained_layout=True)
            for route in routes:
                indicator = (decisions["route"].astype(str) == route).astype(float)
                axes[0].plot(x, indicator.rolling(window, min_periods=1).mean(), label=route)
            axes[0].set(ylabel="Rolling route share", ylim=(0, 1.02), title=f"{architecture} dynamic route mix")
            axes[0].legend(fontsize=8)
            axes[1].plot(
                x,
                pd.to_numeric(decisions.get("weak_calls", 0), errors="coerce").rolling(window, min_periods=1).mean(),
                label="weak calls",
            )
            axes[1].plot(
                x,
                pd.to_numeric(decisions.get("strong_calls", 0), errors="coerce").rolling(window, min_periods=1).mean(),
                label="strong calls",
            )
            axes[1].plot(
                x,
                pd.to_numeric(decisions.get("organization_switches", 0), errors="coerce").rolling(window, min_periods=1).mean(),
                label="organization switches",
            )
            axes[1].set(xlabel="Elapsed hours", ylabel="Rolling mean per decision", title="Organization cost")
            axes[1].legend()
            for axis in axes:
                axis.grid(alpha=0.25)
            fig.savefig(run_dir / "organization_curves.png", dpi=150)
            plt.close(fig)
    return summary


def generate_comparison(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    rows = []
    for architecture in discover_architectures(root):
        run_dir = root / architecture
        if (run_dir / "decisions.jsonl").exists():
            summary = generate_run_report(run_dir)
            if summary:
                rows.append(summary)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame.to_csv(root / "comparison.csv", index=False)
    with (root / "comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)

    labels = frame["architecture"].tolist()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    series = [
        ("accuracy", "Accuracy"),
        ("tokens_per_correct", "Tokens per correct decision"),
        ("latency_p95_s", "P95 decision latency (s)"),
        ("decisions_per_hour", "Decisions per hour"),
    ]
    for axis, (column, title) in zip(axes.flat, series, strict=True):
        values = pd.to_numeric(frame[column], errors="coerce")
        bars = axis.bar(labels, values)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.bar_label(bars, fmt="%.3g", padding=3)
    fig.savefig(root / "comparison.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    x = pd.to_numeric(frame["tokens_per_correct"], errors="coerce")
    y = pd.to_numeric(frame["accuracy"], errors="coerce")
    axis.scatter(x, y, s=90)
    for label, xv, yv in zip(labels, x, y, strict=True):
        axis.annotate(label, (xv, yv), xytext=(6, 5), textcoords="offset points")
    axis.set(xlabel="Tokens per correct decision (lower is better)", ylabel="Accuracy (higher is better)", title="Effect-cost comparison")
    axis.grid(alpha=0.25)
    fig.savefig(root / "pareto_tokens.png", dpi=160)
    plt.close(fig)

    resource_columns = [
        ("gpu_energy_kwh", "GPU energy (kWh)"),
        ("gpu_power_mean_w", "Mean GPU power (W)"),
        ("gpu_util_mean_percent", "Mean GPU utilization (%)"),
        ("gpu_temp_max_c", "Peak GPU temperature (C)"),
        ("ram_used_mean_gb", "Mean system RAM (GiB)"),
        ("cpu_mean_percent", "Mean CPU utilization (%)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for axis, (column, title) in zip(axes.flat, resource_columns, strict=True):
        values = pd.to_numeric(frame[column], errors="coerce")
        bars = axis.bar(labels, values)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.bar_label(bars, fmt="%.3g", padding=3)
    fig.savefig(root / "resource_comparison.png", dpi=160)
    plt.close(fig)

    if {"mean_confidence", "confidence_brier_score"}.issubset(frame.columns):
        fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
        mean_confidence = pd.to_numeric(frame["mean_confidence"], errors="coerce")
        accuracy = pd.to_numeric(frame["accuracy"], errors="coerce")
        x_positions = np.arange(len(labels))
        width = 0.36
        axes[0].bar(x_positions - width / 2, accuracy, width, label="accuracy")
        axes[0].bar(x_positions + width / 2, mean_confidence, width, label="reported confidence")
        axes[0].set(xticks=x_positions, xticklabels=labels, ylim=(0, 1.02), title="Confidence versus accuracy")
        axes[0].legend()
        brier = pd.to_numeric(frame["confidence_brier_score"], errors="coerce")
        bars = axes[1].bar(labels, brier)
        axes[1].bar_label(bars, fmt="%.3f", padding=3)
        axes[1].set(title="Brier score (lower is better)")
        for axis in axes:
            axis.grid(axis="y", alpha=0.25)
        fig.savefig(root / "confidence_comparison.png", dpi=160)
        plt.close(fig)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--comparison", action="store_true")
    args = parser.parse_args()
    if args.comparison:
        print(generate_comparison(args.path).to_string(index=False))
    else:
        print(json.dumps(generate_run_report(args.path), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
