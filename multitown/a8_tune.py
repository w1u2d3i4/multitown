"""Tune the deterministic A8 early-stop threshold on dev, then freeze test simulation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .a7_policy import load_bundle, predict_arms
from .a7_train import scenario_maps, validate_matrix, write_jsonl
from .a8_controller import CONTROLLER_VERSION, simulate_a8_cell
from .counterfactual_runner import read_jsonl
from .masbench_routing import git_state, utc_now, write_json


THRESHOLDS = (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.01)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize_simulation(
    rows: list[dict[str, Any]],
    *,
    objective_token_penalty_per_1k: float,
    objective_latency_penalty_per_s: float,
) -> dict[str, Any]:
    count = len(rows)
    accuracy = sum(bool(row["correct"]) for row in rows) / count
    tokens = sum(float(row["total_tokens"]) for row in rows) / count
    latency = sum(float(row["decision_latency_s"]) for row in rows) / count
    return {
        "scenario_count": count,
        "accuracy": accuracy,
        "tokens_per_decision": tokens,
        "latency_mean_s": latency,
        "objective_utility": (
            accuracy
            - objective_token_penalty_per_1k * tokens / 1000.0
            - objective_latency_penalty_per_s * latency
        ),
        "delegation_rate": sum(bool(row["delegated"]) for row in rows) / count,
        "early_stop_rate": sum(bool(row["early_stop"]) for row in rows) / count,
        "weak_specialist_rate": sum(bool(row["weak_specialist_called"]) for row in rows) / count,
        "strong_specialist_rate": sum(bool(row["strong_specialist_called"]) for row in rows) / count,
        "mean_reorganizations": sum(int(row["reorganization_count"]) for row in rows) / count,
        "mean_reorganization_gain": sum(int(row["reorganization_gain"]) for row in rows) / count,
        "unnecessary_delegation_rate": sum(
            bool(row["delegated"]) and bool(row["initial_correct"]) and bool(row["correct"])
            for row in rows
        ) / count,
        "harmful_reorganization_rate": sum(int(row["reorganization_gain"]) < 0 for row in rows) / count,
        "route_counts": dict(sorted(Counter(row["route"] for row in rows).items())),
    }


def simulate_split(
    *,
    scenario_ids: list[str],
    scenarios: dict[str, Any],
    matrix: dict[tuple[str, str], dict[str, Any]],
    bundle: dict[str, Any],
    threshold: float,
) -> list[dict[str, Any]]:
    model_name = str(bundle["selected_config"]["model_name"])
    rows: list[dict[str, Any]] = []
    for scenario_id in scenario_ids:
        scenario = scenarios[scenario_id]
        predictions = predict_arms(bundle, scenario, model_name=model_name)
        row = simulate_a8_cell(
            scenario=scenario,
            a0=matrix[(scenario_id, "A0")],
            a1=matrix[(scenario_id, "A1")],
            a2=matrix[(scenario_id, "A2")],
            predicted_a0_accuracy=float(predictions["A0"]["predicted_accuracy"]),
            early_stop_threshold=threshold,
        )
        row["split"] = matrix[(scenario_id, "A0")]["split"]
        rows.append(row)
    return rows


def run(args: argparse.Namespace) -> int:
    project_root = Path(__file__).resolve().parents[1]
    revision, dirty = git_state(project_root)
    bank_path = Path(args.bank).resolve()
    matrix_path = Path(args.matrix_dir).resolve() / "decisions.jsonl"
    bundle_path = Path(args.policy_bundle).resolve()
    output = Path(args.output_dir).resolve()
    scenarios, splits, hashes = scenario_maps(bank_path)
    matrix = validate_matrix(
        read_jsonl(matrix_path), splits=splits, hashes=hashes,
        allow_request_errors=args.allow_request_errors,
    )
    bundle = load_bundle(bundle_path)
    dev_ids = sorted(key for key, value in splits.items() if value == "dev")
    test_ids = sorted(key for key, value in splits.items() if value == "test")
    leaderboard: list[dict[str, Any]] = []
    dev_by_threshold: dict[float, list[dict[str, Any]]] = {}
    for threshold in THRESHOLDS:
        rows = simulate_split(
            scenario_ids=dev_ids,
            scenarios=scenarios,
            matrix=matrix,
            bundle=bundle,
            threshold=threshold,
        )
        dev_by_threshold[threshold] = rows
        summary = summarize_simulation(
            rows,
            objective_token_penalty_per_1k=args.objective_token_penalty_per_1k,
            objective_latency_penalty_per_s=args.objective_latency_penalty_per_s,
        )
        summary["early_stop_threshold"] = threshold
        leaderboard.append(summary)
    eligible = [row for row in leaderboard if row["tokens_per_decision"] <= args.max_average_tokens]
    if not eligible:
        eligible = leaderboard
    selected_dev = max(
        eligible,
        key=lambda row: (
            row["objective_utility"], row["accuracy"],
            -row["tokens_per_decision"], -row["latency_mean_s"],
            -row["early_stop_threshold"],
        ),
    )
    selected_threshold = float(selected_dev["early_stop_threshold"])
    dev_rows = dev_by_threshold[selected_threshold]
    # Test simulations are first constructed after dev has frozen the threshold.
    test_rows = simulate_split(
        scenario_ids=test_ids,
        scenarios=scenarios,
        matrix=matrix,
        bundle=bundle,
        threshold=selected_threshold,
    )
    test_summary = summarize_simulation(
        test_rows,
        objective_token_penalty_per_1k=args.objective_token_penalty_per_1k,
        objective_latency_penalty_per_s=args.objective_latency_penalty_per_s,
    )
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "dev-simulation.jsonl", dev_rows)
    write_jsonl(output / "test-simulation.jsonl", test_rows)
    config = {
        "controller_version": CONTROLLER_VERSION,
        "created_at_utc": utc_now(),
        "source_revision": revision,
        "source_dirty_at_start": dirty,
        "a7_policy_bundle": str(bundle_path),
        "a7_policy_bundle_sha256": _sha256(bundle_path),
        "matrix_decisions": str(matrix_path),
        "matrix_decisions_sha256": _sha256(matrix_path),
        "scenario_bank": str(bank_path),
        "scenario_bank_sha256": _sha256(bank_path),
        "selected": {
            "early_stop_threshold": selected_threshold,
            "a7_model_name": bundle["selected_config"]["model_name"],
            "direct_strong_on_constraint_failure": True,
            "second_weak_on_safe_uncertainty": True,
            "strong_on_weak_disagreement": True,
        },
        "dev_selection": {
            "protocol": "threshold selected on dev only under average token cap",
            "max_average_tokens": args.max_average_tokens,
            "objective_token_penalty_per_1k": args.objective_token_penalty_per_1k,
            "objective_latency_penalty_per_s": args.objective_latency_penalty_per_s,
            "selected_result": selected_dev,
            "leaderboard": sorted(leaderboard, key=lambda row: -row["objective_utility"]),
        },
        "test_simulation": test_summary,
        "limitations": [
            "The dev/test simulation reuses A0, A1 and the first A2 candidate rather than targeted online prompts.",
            "Simulation latency is a conservative sum; the formal result is the fresh-seed online test run.",
            "The hard validator checks safety/feasibility and is intentionally not an answer oracle.",
        ],
    }
    write_json(output / "a8-config.json", config)
    manifest = {
        "schema_version": "multitown-a8-tuning-manifest-v1",
        "created_at_utc": utc_now(),
        "source_revision": revision,
        "source_dirty_at_start": dirty,
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in sorted(output.iterdir()) if path.is_file()
        },
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"selected_dev": selected_dev, "test_simulation": test_summary}, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", required=True)
    parser.add_argument("--policy-bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bank", default="benchmarks/multitown-v0.2-1200/scenario-bank.jsonl")
    parser.add_argument("--max-average-tokens", type=float, default=1500.0)
    parser.add_argument("--objective-token-penalty-per-1k", type=float, default=0.005)
    parser.add_argument("--objective-latency-penalty-per-s", type=float, default=0.0025)
    parser.add_argument("--allow-request-errors", action="store_true")
    return parser


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
