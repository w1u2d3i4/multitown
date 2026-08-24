from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .common import read_json, write_json

METHODS = ("Solo", "PlanExecute", "ExecuteReview", "A4", "A8")
SCHEMA_VERSION = "multitown-teambench-selector-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_paired_runs(
    directories: dict[str, Path],
    *,
    required_split: str = "dev",
) -> tuple[list[str], dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    if set(directories) != set(METHODS):
        raise ValueError(f"expected directories for {METHODS}, got {sorted(directories)}")

    configs: dict[str, dict[str, Any]] = {}
    maps: dict[str, dict[str, dict[str, Any]]] = {}
    hashes: dict[str, str] = {}
    for method in METHODS:
        directory = directories[method]
        config_path = directory / "config.json"
        results_path = directory / "results.jsonl"
        config = read_json(config_path)
        if config.get("method") != method:
            raise ValueError(f"{method} config declares {config.get('method')!r}")
        if config.get("split") != required_split:
            raise ValueError(f"{method} is not a {required_split} run")
        rows = _load_jsonl(results_path)
        by_id = {str(row["task_id"]): row for row in rows}
        if len(rows) != len(by_id):
            raise ValueError(f"{method} contains duplicate task IDs")
        bad = [
            task_id for task_id, row in by_id.items()
            if int(row.get("request_errors", 0)) != 0 or "error" in row
        ]
        if bad:
            raise ValueError(f"{method} contains invocation errors: {bad[:5]}")
        configs[method] = config
        maps[method] = by_id
        hashes[method] = _sha256(results_path)

    shared = set.intersection(*(set(maps[method]) for method in METHODS))
    if any(set(maps[method]) != shared for method in METHODS):
        raise ValueError("strategy result files do not contain the same task IDs")
    configured = [str(value) for value in configs["A4"].get("tasks", [])]
    task_ids = configured if set(configured) == shared else sorted(shared)

    comparable = ("split_sha256", "docker_image_id", "max_tokens", "temperature")
    for field in comparable:
        values = {json.dumps(configs[method].get(field), sort_keys=True) for method in METHODS}
        if len(values) != 1:
            raise ValueError(f"run configs disagree on {field}")
    for tier in ("strong", "weak"):
        models = {configs[method].get(tier, {}).get("model") for method in METHODS}
        if len(models) != 1:
            raise ValueError(f"run configs disagree on {tier} model")

    provenance = {
        "split": required_split,
        "split_sha256": configs["A4"].get("split_sha256"),
        "docker_image_id": configs["A4"].get("docker_image_id"),
        "temperature": configs["A4"].get("temperature"),
        "max_tokens": configs["A4"].get("max_tokens"),
        "strong_model": configs["A4"].get("strong", {}).get("model"),
        "weak_model": configs["A4"].get("weak", {}).get("model"),
        "result_sha256": hashes,
        "task_ids_sha256": hashlib.sha256("\n".join(task_ids).encode()).hexdigest(),
    }
    return task_ids, maps, provenance


def outcome_reward(row: dict[str, Any], reference_median_tokens: float) -> float:
    if reference_median_tokens <= 0:
        raise ValueError("reference median tokens must be positive")
    return (
        float(row["partial_score"])
        + 0.25 * int(bool(row["passed"]))
        - 0.05 * math.log1p(float(row["total_tokens"]) / reference_median_tokens)
        - 0.25 * int(int(row.get("request_errors", 0)) > 0)
    )


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized)


def fit_policy(
    task_ids: list[str],
    rows: dict[str, dict[str, dict[str, Any]]],
    *,
    alpha: float = 5.0,
) -> dict[str, Any]:
    if not task_ids:
        raise ValueError("cannot fit an empty policy")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    reference = statistics.median(
        float(rows["PlanExecute"][task_id]["total_tokens"]) for task_id in task_ids
    )
    rewards = {
        method: {
            task_id: outcome_reward(rows[method][task_id], reference)
            for task_id in task_ids
        }
        for method in METHODS
    }
    global_reward = {
        method: _mean(rewards[method][task_id] for task_id in task_ids)
        for method in METHODS
    }
    mean_tokens = {
        method: _mean(float(rows[method][task_id]["total_tokens"]) for task_id in task_ids)
        for method in METHODS
    }

    def best(scores: dict[str, float]) -> str:
        return max(METHODS, key=lambda method: (scores[method], -mean_tokens[method], method))

    categories = sorted(
        {str(rows["Solo"][task_id].get("category") or "Other") for task_id in task_ids}
    )
    category_method: dict[str, str] = {}
    category_scores: dict[str, dict[str, float]] = {}
    category_counts: dict[str, int] = {}
    for category in categories:
        selected = [
            task_id for task_id in task_ids
            if str(rows["Solo"][task_id].get("category") or "Other") == category
        ]
        category_counts[category] = len(selected)
        scores = {
            method: (
                sum(rewards[method][task_id] for task_id in selected)
                + alpha * global_reward[method]
            ) / (len(selected) + alpha)
            for method in METHODS
        }
        category_scores[category] = scores
        category_method[category] = best(scores)

    return {
        "schema_version": SCHEMA_VERSION,
        "policy_type": "empirical_bayes_contextual_router",
        "claim_class": "contextual_bandit_not_agentic_rl",
        "observable_features": ["category"],
        "methods": list(METHODS),
        "default_method": best(global_reward),
        "category_method": category_method,
        "training": {
            "task_count": len(task_ids),
            "alpha": alpha,
            "reference_median_tokens": reference,
            "reward": (
                "partial_score + 0.25*passed - "
                "0.05*log1p(total_tokens/reference_median_tokens) - "
                "0.25*invocation_error"
            ),
            "global_mean_reward": global_reward,
            "mean_tokens": mean_tokens,
            "category_count": category_counts,
            "category_posterior_reward": category_scores,
        },
    }


def select_method(policy: dict[str, Any], row: dict[str, Any]) -> str:
    if policy.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported selector policy schema")
    category = str(row.get("category") or "Other")
    method = policy.get("category_method", {}).get(category, policy.get("default_method"))
    if method not in METHODS:
        raise ValueError(f"selector chose unsupported method {method!r}")
    return str(method)


def summarize_policy(
    task_ids: list[str],
    rows: dict[str, dict[str, dict[str, Any]]],
    policy: dict[str, Any],
) -> dict[str, Any]:
    chosen = {task_id: select_method(policy, rows["Solo"][task_id]) for task_id in task_ids}
    selected_rows = [rows[chosen[task_id]][task_id] for task_id in task_ids]
    return {
        "task_count": len(task_ids),
        "passes": sum(int(bool(row["passed"])) for row in selected_rows),
        "mean_partial_score": _mean(float(row["partial_score"]) for row in selected_rows),
        "mean_total_tokens": _mean(float(row["total_tokens"]) for row in selected_rows),
        "action_counts": {
            method: sum(value == method for value in chosen.values()) for method in METHODS
        },
    }


def leave_one_out_summary(
    task_ids: list[str],
    rows: dict[str, dict[str, dict[str, Any]]],
    *,
    alpha: float,
) -> dict[str, Any]:
    selected_rows: list[dict[str, Any]] = []
    actions: list[str] = []
    for held_out in task_ids:
        training = [task_id for task_id in task_ids if task_id != held_out]
        policy = fit_policy(training, rows, alpha=alpha)
        action = select_method(policy, rows["Solo"][held_out])
        actions.append(action)
        selected_rows.append(rows[action][held_out])
    return {
        "protocol": "leave_one-task-out selection-set estimate",
        "task_count": len(task_ids),
        "passes": sum(int(bool(row["passed"])) for row in selected_rows),
        "mean_partial_score": _mean(float(row["partial_score"]) for row in selected_rows),
        "mean_total_tokens": _mean(float(row["total_tokens"]) for row in selected_rows),
        "action_counts": {method: actions.count(method) for method in METHODS},
    }


def train_selector(
    directories: dict[str, Path],
    *,
    output: Path,
    alpha: float,
) -> dict[str, Any]:
    task_ids, rows, provenance = load_paired_runs(directories)
    policy = fit_policy(task_ids, rows, alpha=alpha)
    policy["provenance"] = provenance
    policy["selection_set_fit"] = summarize_policy(task_ids, rows, policy)
    policy["selection_set_loo"] = leave_one_out_summary(task_ids, rows, alpha=alpha)
    write_json(output, policy)
    return policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solo-dir", type=Path)
    parser.add_argument("--plan-execute-dir", type=Path)
    parser.add_argument("--execute-review-dir", type=Path)
    parser.add_argument("--a4-dir", type=Path)
    parser.add_argument("--a8-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=5.0)
    args = parser.parse_args()
    directories = {
        "Solo": args.solo_dir,
        "PlanExecute": args.plan_execute_dir,
        "ExecuteReview": args.execute_review_dir,
        "A4": args.a4_dir,
        "A8": args.a8_dir,
    }
    if any(path is None for path in directories.values()):
        parser.error("all five strategy directories are required")
    policy = train_selector(
        {method: path.resolve() for method, path in directories.items()},
        output=args.output.resolve(),
        alpha=args.alpha,
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "default_method": policy["default_method"],
        "category_method": policy["category_method"],
        "selection_set_loo": policy["selection_set_loo"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
