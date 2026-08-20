"""Generate formal paired statistics, curves and a compact A7/A8 report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .counterfactual_runner import read_jsonl
from .masbench_report import bootstrap_difference, exact_mcnemar_p
from .masbench_routing import write_json
from .provenance import build_manifest


DISPLAY_ORDER = ("A0", "A1", "A2", "A3", "A4", "A5", "A7-offline", "A7-online", "A8-online")
COLORS = {
    "A0": "#8da0cb", "A1": "#4daf4a", "A2": "#984ea3", "A3": "#a65628",
    "A4": "#e41a1c", "A5": "#999999", "A7-offline": "#ff7f00",
    "A7-online": "#fdbf6f", "A8-online": "#377eb8",
}

TRADEOFF_LABEL_OFFSETS = {
    "A7-offline": (6, 8),
    "A7-online": (6, -12),
}


def index_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed = {str(row["scenario_id"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("duplicate scenario rows")
    return indexed


def load_systems(
    *, matrix_dir: Path, a7_policy_dir: Path, a7_online_dir: Path, a8_online_dir: Path,
) -> dict[str, dict[str, dict[str, Any]]]:
    matrix_test: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(matrix_dir / "decisions.jsonl"):
        if row["split"] == "test":
            matrix_test[str(row["arm"])].append(row)
    systems = {arm: index_rows(matrix_test[arm]) for arm in ("A0", "A1", "A2", "A3", "A4", "A5")}
    systems["A7-offline"] = index_rows(read_jsonl(a7_policy_dir / "test-selections.jsonl"))
    systems["A7-online"] = index_rows(read_jsonl(a7_online_dir / "decisions.jsonl"))
    systems["A8-online"] = index_rows(read_jsonl(a8_online_dir / "decisions.jsonl"))
    expected = set(systems["A0"])
    for name, rows in systems.items():
        if set(rows) != expected:
            raise ValueError(f"{name} test IDs differ: {len(rows)} != {len(expected)}")
    return systems


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    latencies = np.asarray([float(row["decision_latency_s"]) for row in rows])
    return {
        "decisions": count,
        "correct": sum(bool(row["correct"]) for row in rows),
        "accuracy": sum(bool(row["correct"]) for row in rows) / count,
        "valid_rate": sum(bool(row["valid"]) for row in rows) / count,
        "total_tokens": sum(int(row["total_tokens"]) for row in rows),
        "tokens_per_decision": sum(int(row["total_tokens"]) for row in rows) / count,
        "latency_mean_s": float(latencies.mean()),
        "latency_p95_s": float(np.percentile(latencies, 95)),
        "request_errors": sum(int(row.get("request_errors", 0)) for row in rows),
    }


def paired(
    left_name: str,
    right_name: str,
    systems: dict[str, dict[str, dict[str, Any]]],
    *, seed: int,
) -> dict[str, Any]:
    scenario_ids = sorted(systems[left_name])
    left = [systems[left_name][key] for key in scenario_ids]
    right = [systems[right_name][key] for key in scenario_ids]
    left_correct = np.asarray([bool(row["correct"]) for row in left], dtype=float)
    right_correct = np.asarray([bool(row["correct"]) for row in right], dtype=float)
    left_tokens = np.asarray([float(row["total_tokens"]) for row in left])
    right_tokens = np.asarray([float(row["total_tokens"]) for row in right])
    left_latency = np.asarray([float(row["decision_latency_s"]) for row in left])
    right_latency = np.asarray([float(row["decision_latency_s"]) for row in right])
    accuracy = bootstrap_difference(left_correct, right_correct, seed=seed)
    tokens = bootstrap_difference(left_tokens, right_tokens, seed=seed + 1)
    latency = bootstrap_difference(left_latency, right_latency, seed=seed + 2)
    return {
        "left": left_name,
        "right": right_name,
        "accuracy": accuracy,
        "tokens_per_decision": tokens,
        "latency_mean_s": latency,
        "mcnemar_exact": exact_mcnemar_p(
            [bool(value) for value in left_correct],
            [bool(value) for value in right_correct],
        ),
        "accuracy_noninferior_at_minus_1pp": accuracy["ci95_low"] >= -0.01,
        "token_reduction_fraction": 1.0 - float(left_tokens.mean() / right_tokens.mean()),
        "latency_reduction_fraction": 1.0 - float(left_latency.mean() / right_latency.mean()),
    }


def by_family(
    systems: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for name, indexed in systems.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in indexed.values():
            grouped[str(row["family"])].append(row)
        result[name] = {family: summarize(rows) for family, rows in sorted(grouped.items())}
    return result


def energy_summary(path: Path, decisions: int) -> dict[str, Any] | None:
    rows = read_jsonl(path)
    samples = [
        (float(row["elapsed_s"]), float(row["gpu_power_w"]))
        for row in rows if row.get("gpu_power_w") is not None
    ]
    if len(samples) < 2:
        return None
    watt_seconds = sum(
        (left[1] + right[1]) * 0.5 * max(0.0, right[0] - left[0])
        for left, right in zip(samples, samples[1:])
    )
    return {
        "samples": len(samples),
        "gpu_energy_wh": watt_seconds / 3600.0,
        "gpu_energy_wh_per_decision": watt_seconds / 3600.0 / decisions,
        "mean_gpu_power_w": float(np.mean([value for _, value in samples])),
    }


def build_report_manifest(
    project_root: Path,
    artifact_roots: list[Path],
    output: Path,
) -> dict[str, Any]:
    """Build a rerunnable manifest without hashing its previous incarnation."""
    (output / "artifact-manifest.json").unlink(missing_ok=True)
    return build_manifest(project_root, artifact_roots)


def plot_tradeoff(overall: dict[str, dict[str, Any]], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.8))
    for name in DISPLAY_ORDER:
        row = overall[name]
        marker = "D" if name.startswith("A7") or name.startswith("A8") else "o"
        ax.scatter(row["tokens_per_decision"], 100 * row["accuracy"], s=80, marker=marker, color=COLORS[name])
        ax.annotate(
            name,
            (row["tokens_per_decision"], 100 * row["accuracy"]),
            xytext=TRADEOFF_LABEL_OFFSETS.get(name, (5, 5)),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Tokens per decision")
    ax.set_ylabel("Exact-match accuracy (%)")
    ax.set_title("MultiTown v0.2 held-out cost-quality trade-off")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_family_accuracy(family: dict[str, dict[str, dict[str, Any]]], output: Path) -> None:
    names = ("A0", "A1", "A4", "A7-online", "A8-online")
    families = list(family["A0"])
    x = np.arange(len(families))
    width = 0.16
    fig, ax = plt.subplots(figsize=(12, 5.8))
    for index, name in enumerate(names):
        values = [100 * family[name][item]["accuracy"] for item in families]
        ax.bar(x + (index - 2) * width, values, width, label=name, color=COLORS[name])
    ax.set_xticks(x, [item.replace("_", "\n") for item in families], fontsize=8)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Held-out accuracy by scenario family")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_cumulative(systems: dict[str, dict[str, dict[str, Any]]], output: Path) -> None:
    names = ("A0", "A1", "A4", "A7-online", "A8-online")
    scenario_ids = sorted(systems["A0"])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name in names:
        values = np.cumsum([int(systems[name][key]["total_tokens"]) for key in scenario_ids])
        ax.plot(np.arange(1, len(values) + 1), values / 1_000, label=name, color=COLORS[name])
    ax.set_xlabel("Held-out scenarios (stable ID order)")
    ax.set_ylabel("Cumulative tokens (thousands)")
    ax.set_title("A7/A8 cumulative inference cost")
    ax.grid(alpha=0.25)
    ax.legend(ncol=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_a8_routes(rows: list[dict[str, Any]], output: Path) -> None:
    counts = Counter(str(row["route"]) for row in rows)
    labels, values = zip(*sorted(counts.items(), key=lambda item: -item[1]), strict=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(range(len(labels)), values, color="#377eb8")
    ax.set_yticks(range(len(labels)), [label.replace("_", " ") for label in labels], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Held-out scenarios")
    ax.set_title("A8 execution-time routes")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_system_curves(paths: dict[str, Path], output: Path) -> None:
    fig, axes = plt.subplots(len(paths), 2, figsize=(11, 3.2 * len(paths)), squeeze=False)
    for row_index, (name, path) in enumerate(paths.items()):
        rows = read_jsonl(path)
        elapsed = np.asarray([float(row["elapsed_s"]) / 3600.0 for row in rows])
        utilization = np.asarray([
            float(row["gpu_util_percent"]) if row.get("gpu_util_percent") is not None else np.nan
            for row in rows
        ])
        power = np.asarray([
            float(row["gpu_power_w"]) if row.get("gpu_power_w") is not None else np.nan
            for row in rows
        ])
        axes[row_index, 0].plot(elapsed, utilization, color="#984ea3", linewidth=0.8)
        axes[row_index, 0].set_ylabel(f"{name}\nGPU util (%)")
        axes[row_index, 1].plot(elapsed, power, color="#e41a1c", linewidth=0.8)
        axes[row_index, 1].set_ylabel(f"{name}\nPower (W)")
        for axis in axes[row_index]:
            axis.set_xlabel("Elapsed hours")
            axis.grid(alpha=0.2)
    fig.suptitle("Formal-run system telemetry")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def markdown_report(comparison: dict[str, Any]) -> str:
    lines = [
        "# MultiTown A7/A8 formal held-out comparison",
        "",
        "The 1,200-scenario bank uses a fixed 840/180/180 train/dev/test split. A7 fits",
        "only train data, chooses model/budget/penalties only on dev, and reports test once.",
        "A8 chooses its early-stop threshold only on dev and is then executed with a fresh",
        "test inference seed. Its hard validator checks safety/feasibility, not oracle equality.",
        "",
        "| System | Accuracy | Correct | Tokens/decision | Mean latency | P95 latency | Errors |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in DISPLAY_ORDER:
        row = comparison["overall"][name]
        lines.append(
            f"| {name} | {100*row['accuracy']:.2f}% | {row['correct']}/{row['decisions']} | "
            f"{row['tokens_per_decision']:.1f} | {row['latency_mean_s']:.3f}s | "
            f"{row['latency_p95_s']:.3f}s | {row['request_errors']} |"
        )
    lines.extend(["", "## Preregistered gates", ""])
    for label in ("A7_online_vs_A4", "A8_online_vs_A4", "A8_online_vs_A7_online"):
        row = comparison["paired"][label]
        lines.append(
            f"- {row['left']} vs {row['right']}: accuracy difference "
            f"{100*row['accuracy']['difference']:.2f}pp (paired bootstrap 95% CI "
            f"{100*row['accuracy']['ci95_low']:.2f} to {100*row['accuracy']['ci95_high']:.2f}pp); "
            f"token reduction {100*row['token_reduction_fraction']:.2f}%; "
            f"-1pp non-inferiority={'passed' if row['accuracy_noninferior_at_minus_1pp'] else 'not passed'}."
        )
    a8 = comparison["a8_mechanisms"]
    lines.extend([
        "",
        "## A8 mechanism metrics",
        "",
        f"- Delegation rate: {100*a8['delegation_rate']:.2f}%; early-stop rate: {100*a8['early_stop_rate']:.2f}%.",
        f"- Mean reorganizations: {a8['mean_reorganizations']:.3f}; mean reorganization gain: {a8['mean_reorganization_gain']:.4f}.",
        f"- Mean communication density: {a8['mean_communication_density']:.4f}; message tokens/decision: {a8['message_tokens_per_decision']:.1f}.",
        f"- Unnecessary delegation rate: {100*a8['unnecessary_delegation_rate']:.2f}%; hard-failure recovery: {a8['hard_failure_recovery_rate']}.",
        "",
        "These are dynamic-controller results, not reinforcement learning. A9 is the first stage",
        "that would train a controller with RL.",
        "",
    ])
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    systems = load_systems(
        matrix_dir=args.matrix_dir.resolve(),
        a7_policy_dir=args.a7_policy_dir.resolve(),
        a7_online_dir=args.a7_online_dir.resolve(),
        a8_online_dir=args.a8_online_dir.resolve(),
    )
    scenario_ids = sorted(systems["A0"])
    overall = {
        name: summarize([systems[name][key] for key in scenario_ids])
        for name in DISPLAY_ORDER
    }
    comparison = {
        "schema_version": "multitown-a78-formal-comparison-v1",
        "test_scenarios": len(scenario_ids),
        "overall": overall,
        "by_family": by_family(systems),
        "paired": {
            "A7_offline_vs_A4": paired("A7-offline", "A4", systems, seed=args.seed),
            "A7_online_vs_A4": paired("A7-online", "A4", systems, seed=args.seed + 100),
            "A8_online_vs_A4": paired("A8-online", "A4", systems, seed=args.seed + 200),
            "A8_online_vs_A7_online": paired("A8-online", "A7-online", systems, seed=args.seed + 300),
        },
        "a7_policy": json.loads((args.a7_policy_dir / "policy.json").read_text(encoding="utf-8")),
        "a8_tuning": json.loads((args.a8_tuning_dir / "a8-config.json").read_text(encoding="utf-8")),
        "a8_mechanisms": json.loads((args.a8_online_dir / "summary.json").read_text(encoding="utf-8")),
        "energy": {
            "A7-online": energy_summary(args.a7_online_dir / "system_metrics.jsonl", len(scenario_ids)),
            "A8-online": energy_summary(args.a8_online_dir / "system_metrics.jsonl", len(scenario_ids)),
        },
        "environment_lock": {
            "path": str(args.environment_lock.resolve()),
            "sha256": hashlib.sha256(args.environment_lock.read_bytes()).hexdigest(),
        },
    }
    write_json(output / "comparison.json", comparison)
    with (output / "overall.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("system", *next(iter(overall.values())).keys()))
        writer.writeheader()
        for name in DISPLAY_ORDER:
            writer.writerow({"system": name, **overall[name]})
    plot_tradeoff(overall, output / "tradeoff.png")
    plot_family_accuracy(comparison["by_family"], output / "family-accuracy.png")
    plot_cumulative(systems, output / "cumulative-tokens.png")
    plot_a8_routes(list(systems["A8-online"].values()), output / "a8-routes.png")
    plot_system_curves({
        "A0-A5 matrix": args.matrix_dir / "system_metrics.jsonl",
        "A7 online": args.a7_online_dir / "system_metrics.jsonl",
        "A8 online": args.a8_online_dir / "system_metrics.jsonl",
    }, output / "system-curves.png")
    (output / "RESULTS.md").write_text(markdown_report(comparison), encoding="utf-8")
    manifest = build_report_manifest(project_root, [
        args.matrix_dir.resolve(), args.a7_policy_dir.resolve(), args.a7_online_dir.resolve(),
        args.a8_tuning_dir.resolve(), args.a8_online_dir.resolve(), output,
    ], output)
    write_json(output / "artifact-manifest.json", manifest)
    if args.record_dir:
        record = args.record_dir.resolve()
        record.mkdir(parents=True, exist_ok=True)
        for name in (
            "comparison.json", "overall.csv", "RESULTS.md", "artifact-manifest.json",
            "tradeoff.png", "family-accuracy.png", "cumulative-tokens.png", "a8-routes.png",
            "system-curves.png",
        ):
            shutil.copy2(output / name, record / name)
        shutil.copy2(args.a7_policy_dir / "policy.json", record / "a7-policy.json")
        shutil.copy2(args.a8_tuning_dir / "a8-config.json", record / "a8-config.json")
        shutil.copy2(args.a7_online_dir / "summary.json", record / "a7-online-summary.json")
        shutil.copy2(args.a8_online_dir / "summary.json", record / "a8-online-summary.json")
        shutil.copy2(args.environment_lock, record / "environment-lock.json")
    print(json.dumps({"output": str(output), "overall": overall}, ensure_ascii=False, indent=2))
    return comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--a7-policy-dir", type=Path, required=True)
    parser.add_argument("--a7-online-dir", type=Path, required=True)
    parser.add_argument("--a8-tuning-dir", type=Path, required=True)
    parser.add_argument("--a8-online-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--record-dir", type=Path)
    parser.add_argument(
        "--environment-lock", type=Path,
        default=Path("records/reproductions/20260810/environment-lock.json"),
    )
    parser.add_argument("--seed", type=int, default=20260810)
    return parser


def main() -> None:
    raise SystemExit(0 if run(build_parser().parse_args()) else 1)


if __name__ == "__main__":
    main()
