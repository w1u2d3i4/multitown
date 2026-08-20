"""Fit leakage-safe routing baselines on MASBench counterfactual arm logs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import statistics
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


ARM_ORDER = ("A0", "A1", "A2", "A4")
ARM_RUN_NAMES = {
    "A0": "A0-weak",
    "A1": "A1-strong",
    "A2": "A2-vote",
    "A4": "A4-heavy",
}
POLICY_ORDER = ("rule", "knn", "svm", "mlp")
AXES = ("breadth", "depth", "horizon", "parallel", "robustness")
SAFE_METADATA_FIELDS = (
    "actual_breadth",
    "actual_depth",
    "breadth",
    "depth",
    "num_problems",
    "longest_path",
    "num_igsm_problems",
    "num_niah_samples",
    "num_total_problems",
)
STRUCTURAL_LIST_FIELDS = ("graph_nodes", "graph_edges")
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(project_root: Path) -> tuple[str | None, bool]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project_root, capture_output=True, text=True, check=False
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=project_root, capture_output=True, text=True, check=False
    )
    return (revision.stdout.strip() if revision.returncode == 0 else None, bool(status.stdout.strip()))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _finite_number(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return 0.0


def safe_features(record: dict[str, Any]) -> dict[str, Any]:
    """Return pre-execution features without answer, solution or oracle fields."""

    messages = record.get("messages", [])
    text = "\n".join(
        str(message.get("content", ""))
        for message in messages
        if isinstance(message, dict)
    )
    numbers = [float(value) for value in NUMBER_RE.findall(text)]
    axis_values = [float(value) for value in NUMBER_RE.findall(str(record.get("axis_value", "")))]
    metadata = record.get("extra_info", {}) if isinstance(record.get("extra_info"), dict) else {}
    numeric = [
        float(len(text)),
        float(len(WORD_RE.findall(text))),
        float(len(numbers)),
        float(len(re.findall(r"[?？]", text))),
        float(len(re.findall(r"[.!。；;]", text))),
        float(len(re.findall(r"[+\-*/=]", text))),
        float(len(messages)),
        float(len(axis_values)),
        statistics.mean(axis_values) if axis_values else 0.0,
        max(axis_values) if axis_values else 0.0,
        min(axis_values) if axis_values else 0.0,
    ]
    numeric.extend(_finite_number(metadata.get(field)) for field in SAFE_METADATA_FIELDS)
    numeric.extend(
        float(len(metadata.get(field, []))) if isinstance(metadata.get(field), list) else 0.0
        for field in STRUCTURAL_LIST_FIELDS
    )
    return {
        "sample_id": str(record["sample_id"]),
        "axis": str(record["axis"]),
        "axis_value": str(record.get("axis_value", "")),
        "text": text,
        "numeric": numeric,
    }


def stable_fit_dev_split(
    records: Iterable[dict[str, Any]], *, dev_per_axis: int = 10, seed: int = 20260810
) -> tuple[list[str], list[str]]:
    """Hash-rank each axis so the 250-row train subset becomes 200 fit/50 dev."""

    grouped: dict[str, list[str]] = defaultdict(list)
    for record in records:
        grouped[str(record["axis"])].append(str(record["sample_id"]))
    fit: list[str] = []
    dev: list[str] = []
    for axis in sorted(grouped):
        ranked = sorted(
            grouped[axis],
            key=lambda sample_id: hashlib.sha256(
                f"{seed}:{axis}:{sample_id}".encode("utf-8")
            ).hexdigest(),
        )
        if len(ranked) <= dev_per_axis:
            raise ValueError(f"axis {axis!r} has only {len(ranked)} records")
        dev.extend(ranked[:dev_per_axis])
        fit.extend(ranked[dev_per_axis:])
    return sorted(fit), sorted(dev)


def load_counterfactuals(
    subset_path: Path, results_root: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    records = {row["sample_id"]: row for row in load_jsonl(subset_path)}
    decisions: dict[str, dict[str, dict[str, Any]]] = {split: {} for split in ("train", "test")}
    files = [{"path": str(subset_path.resolve()), "sha256": sha256_file(subset_path)}]
    for split in ("train", "test"):
        expected = {sample_id for sample_id, row in records.items() if row["split"] == split}
        for arm in ARM_ORDER:
            path = results_root / f"{ARM_RUN_NAMES[arm]}-{split}" / "decisions.jsonl"
            rows = {row["sample_id"]: row for row in load_jsonl(path)}
            if set(rows) != expected:
                missing = sorted(expected - set(rows))[:3]
                extra = sorted(set(rows) - expected)[:3]
                raise ValueError(f"{arm}/{split} sample mismatch; missing={missing}, extra={extra}")
            decisions[split][arm] = rows
            files.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
    return records, decisions, {"schema_version": "multitown-counterfactual-input-v1", "files": files}


def best_arm_label(sample_id: str, arm_rows: dict[str, dict[str, Any]]) -> str:
    correct = [arm for arm in ARM_ORDER if bool(arm_rows[arm][sample_id]["correct"])]
    if not correct:
        return "A0"
    return min(correct, key=lambda arm: (arm_rows[arm][sample_id]["total_tokens"], ARM_ORDER.index(arm)))


class FeatureEncoder:
    """Small deterministic text/structure encoder fitted on training rows only."""

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=768,
            sublinear_tf=True,
        )
        self.axis_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.numeric_scaler = StandardScaler()

    def fit_transform(self, rows: list[dict[str, Any]]) -> np.ndarray:
        text = self.vectorizer.fit_transform([row["text"] for row in rows]).toarray()
        axis = self.axis_encoder.fit_transform([[row["axis"]] for row in rows])
        numeric = self.numeric_scaler.fit_transform([row["numeric"] for row in rows])
        return np.concatenate((text, axis, numeric), axis=1).astype(np.float32)

    def transform(self, rows: list[dict[str, Any]]) -> np.ndarray:
        text = self.vectorizer.transform([row["text"] for row in rows]).toarray()
        axis = self.axis_encoder.transform([[row["axis"]] for row in rows])
        numeric = self.numeric_scaler.transform([row["numeric"] for row in rows])
        return np.concatenate((text, axis, numeric), axis=1).astype(np.float32)


def make_classifier(name: str, *, sample_count: int, seed: int):
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=min(7, sample_count), weights="distance", metric="cosine")
    if name == "svm":
        return SVC(C=2.0, kernel="rbf", gamma="scale", class_weight="balanced", random_state=seed)
    if name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(64, 32),
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=600,
            random_state=seed,
        )
    raise ValueError(name)


def fit_classifier_policy(
    name: str,
    train_rows: list[dict[str, Any]],
    labels: list[str],
    target_rows: list[dict[str, Any]],
    *,
    seed: int,
) -> list[str]:
    encoder = FeatureEncoder()
    train_x = encoder.fit_transform(train_rows)
    target_x = encoder.transform(target_rows)
    if len(set(labels)) == 1:
        classifier = DummyClassifier(strategy="most_frequent")
    else:
        classifier = make_classifier(name, sample_count=len(train_rows), seed=seed)
    classifier.fit(train_x, labels)
    return [str(value) for value in classifier.predict(target_x)]


def utility_metrics(
    sample_ids: list[str],
    predictions: list[str],
    arm_rows: dict[str, dict[str, Any]],
    *,
    token_penalty_per_1k: float,
    latency_penalty_per_s: float,
) -> dict[str, Any]:
    selected = [arm_rows[arm][sample_id] for sample_id, arm in zip(sample_ids, predictions, strict=True)]
    count = len(selected)
    correct = sum(bool(row["correct"]) for row in selected)
    total_tokens = sum(int(row["total_tokens"]) for row in selected)
    latency = sum(float(row["decision_latency_s"]) for row in selected)
    accuracy = correct / count if count else 0.0
    tokens_per_decision = total_tokens / count if count else 0.0
    latency_mean_s = latency / count if count else 0.0
    return {
        "decisions": count,
        "correct": correct,
        "accuracy": accuracy,
        "total_tokens": total_tokens,
        "tokens_per_decision": tokens_per_decision,
        "latency_mean_s": latency_mean_s,
        "request_errors": sum(int(row.get("request_errors", 0)) for row in selected),
        "selected_arms": dict(sorted(Counter(predictions).items())),
        "objective": (
            accuracy
            - token_penalty_per_1k * tokens_per_decision / 1000.0
            - latency_penalty_per_s * latency_mean_s
        ),
    }


def fit_rule_policy(
    fit_ids: list[str],
    target_rows: list[dict[str, Any]],
    features: dict[str, dict[str, Any]],
    arm_rows: dict[str, dict[str, Any]],
    *,
    token_penalty_per_1k: float,
    latency_penalty_per_s: float,
) -> tuple[list[str], dict[str, str]]:
    choices: dict[str, str] = {}
    global_choice = "A0"
    for axis in (*AXES, "__global__"):
        ids = fit_ids if axis == "__global__" else [sample_id for sample_id in fit_ids if features[sample_id]["axis"] == axis]
        scores = {}
        for arm in ARM_ORDER:
            predictions = [arm] * len(ids)
            scores[arm] = utility_metrics(
                ids,
                predictions,
                arm_rows,
                token_penalty_per_1k=token_penalty_per_1k,
                latency_penalty_per_s=latency_penalty_per_s,
            )["objective"]
        selected = max(ARM_ORDER, key=lambda arm: (scores[arm], -ARM_ORDER.index(arm)))
        if axis == "__global__":
            global_choice = selected
        else:
            choices[axis] = selected
    predictions = [choices.get(row["axis"], global_choice) for row in target_rows]
    return predictions, {**choices, "__global__": global_choice}


def fixed_arm_metrics(
    sample_ids: list[str], arm_rows: dict[str, dict[str, Any]], *, token_penalty_per_1k: float, latency_penalty_per_s: float
) -> dict[str, Any]:
    return {
        arm: utility_metrics(
            sample_ids,
            [arm] * len(sample_ids),
            arm_rows,
            token_penalty_per_1k=token_penalty_per_1k,
            latency_penalty_per_s=latency_penalty_per_s,
        )
        for arm in ARM_ORDER
    }


def cost_table(
    fit_ids: list[str], features: dict[str, dict[str, Any]], arm_rows: dict[str, dict[str, Any]]
) -> dict[str, dict[str, float]]:
    table: dict[str, dict[str, float]] = {}
    for arm in ARM_ORDER:
        global_mean = statistics.mean(float(arm_rows[arm][sample_id]["total_tokens"]) for sample_id in fit_ids)
        table[arm] = {"__global__": global_mean}
        for axis in AXES:
            values = [
                float(arm_rows[arm][sample_id]["total_tokens"])
                for sample_id in fit_ids
                if features[sample_id]["axis"] == axis
            ]
            table[arm][axis] = statistics.mean(values) if values else global_mean
    return table


def enforce_estimated_budget(
    predictions: list[str],
    target_rows: list[dict[str, Any]],
    table: dict[str, dict[str, float]],
    max_predicted_tokens: float,
) -> list[str]:
    bounded: list[str] = []
    for selected, row in zip(predictions, target_rows, strict=True):
        axis = row["axis"]
        estimate = table[selected].get(axis, table[selected]["__global__"])
        if estimate <= max_predicted_tokens:
            bounded.append(selected)
            continue
        eligible = [
            arm
            for arm in ARM_ORDER
            if table[arm].get(axis, table[arm]["__global__"]) <= max_predicted_tokens
        ]
        bounded.append(
            min(
                eligible or list(ARM_ORDER),
                key=lambda arm: (table[arm].get(axis, table[arm]["__global__"]), ARM_ORDER.index(arm)),
            )
        )
    return bounded


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    subset_path = args.subset.resolve()
    results_root = args.results_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parents[1]
    source_revision, source_dirty = git_state(project_root)
    records, decisions, manifest = load_counterfactuals(subset_path, results_root)
    features = {sample_id: safe_features(record) for sample_id, record in records.items()}
    train_records = [row for row in records.values() if row["split"] == "train"]
    fit_ids, dev_ids = stable_fit_dev_split(train_records, dev_per_axis=args.dev_per_axis, seed=args.seed)
    train_ids = sorted(fit_ids + dev_ids)
    test_ids = sorted(sample_id for sample_id, row in records.items() if row["split"] == "test")

    def rows(ids: list[str]) -> list[dict[str, Any]]:
        return [features[sample_id] for sample_id in ids]

    fit_labels = [best_arm_label(sample_id, decisions["train"]) for sample_id in fit_ids]
    fit_costs = cost_table(fit_ids, features, decisions["train"])
    dev_predictions: dict[str, list[str]] = {}
    dev_predictions["rule"], rule_map = fit_rule_policy(
        fit_ids,
        rows(dev_ids),
        features,
        decisions["train"],
        token_penalty_per_1k=args.token_penalty_per_1k,
        latency_penalty_per_s=args.latency_penalty_per_s,
    )
    for name in ("knn", "svm", "mlp"):
        dev_predictions[name] = fit_classifier_policy(
            name, rows(fit_ids), fit_labels, rows(dev_ids), seed=args.seed
        )
    dev_predictions = {
        name: enforce_estimated_budget(
            predictions, rows(dev_ids), fit_costs, args.max_predicted_tokens
        )
        for name, predictions in dev_predictions.items()
    }
    dev_metrics = {
        name: utility_metrics(
            dev_ids,
            predictions,
            decisions["train"],
            token_penalty_per_1k=args.token_penalty_per_1k,
            latency_penalty_per_s=args.latency_penalty_per_s,
        )
        for name, predictions in dev_predictions.items()
    }
    selected_policy = max(
        POLICY_ORDER,
        key=lambda name: (dev_metrics[name]["objective"], -POLICY_ORDER.index(name)),
    )

    full_labels = [best_arm_label(sample_id, decisions["train"]) for sample_id in train_ids]
    full_costs = cost_table(train_ids, features, decisions["train"])
    test_predictions: dict[str, list[str]] = {}
    test_predictions["rule"], full_rule_map = fit_rule_policy(
        train_ids,
        rows(test_ids),
        features,
        decisions["train"],
        token_penalty_per_1k=args.token_penalty_per_1k,
        latency_penalty_per_s=args.latency_penalty_per_s,
    )
    for name in ("knn", "svm", "mlp"):
        test_predictions[name] = fit_classifier_policy(
            name, rows(train_ids), full_labels, rows(test_ids), seed=args.seed
        )
    test_predictions = {
        name: enforce_estimated_budget(
            predictions, rows(test_ids), full_costs, args.max_predicted_tokens
        )
        for name, predictions in test_predictions.items()
    }
    test_metrics = {
        name: utility_metrics(
            test_ids,
            predictions,
            decisions["test"],
            token_penalty_per_1k=args.token_penalty_per_1k,
            latency_penalty_per_s=args.latency_penalty_per_s,
        )
        for name, predictions in test_predictions.items()
    }
    fixed = fixed_arm_metrics(
        test_ids,
        decisions["test"],
        token_penalty_per_1k=args.token_penalty_per_1k,
        latency_penalty_per_s=args.latency_penalty_per_s,
    )
    summary = {
        "schema_version": "multitown-masbench-routing-v1",
        "created_at_utc": utc_now(),
        "source_revision": source_revision,
        "source_dirty": source_dirty,
        "method": "LLMRouter-inspired classical policies over frozen MultiTown counterfactual arms",
        "evidence_level": "subset_reproduced",
        "adaptation_scope": "classical router families adapted to MultiTown arms; not upstream end-to-end LLMRouter reproduction",
        "selected_policy": selected_policy,
        "selection_rule": "maximum dev objective; ties follow rule, knn, svm, mlp order",
        "a6_test": test_metrics[selected_policy],
        "dev_metrics": dev_metrics,
        "test_metrics": test_metrics,
        "fixed_arm_test_metrics": fixed,
        "fit_label_distribution": dict(sorted(Counter(fit_labels).items())),
        "full_train_label_distribution": dict(sorted(Counter(full_labels).items())),
        "fit_count": len(fit_ids),
        "dev_count": len(dev_ids),
        "test_count": len(test_ids),
        "rule_map_fit": rule_map,
        "rule_map_full_train": full_rule_map,
        "token_penalty_per_1k": args.token_penalty_per_1k,
        "latency_penalty_per_s": args.latency_penalty_per_s,
        "max_predicted_tokens_per_decision": args.max_predicted_tokens,
        "estimated_cost_table_fit": fit_costs,
        "estimated_cost_table_full_train": full_costs,
        "seed": args.seed,
    }
    config = {
        "schema_version": "multitown-masbench-routing-config-v1",
        "source_revision": source_revision,
        "source_dirty": source_dirty,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "subset": str(subset_path),
        "results_root": str(results_root),
        "seed": args.seed,
        "dev_per_axis": args.dev_per_axis,
        "token_penalty_per_1k": args.token_penalty_per_1k,
        "latency_penalty_per_s": args.latency_penalty_per_s,
        "max_predicted_tokens_per_decision": args.max_predicted_tokens,
        "feature_policy": "prompt text + structural whitelist only; answers/solutions/oracles excluded",
        "models": {
            "rule": "axis-wise empirical utility",
            "knn": "TF-IDF + structural features, cosine distance, k=7",
            "svm": "TF-IDF + structural features, balanced RBF SVC",
            "mlp": "TF-IDF + structural features, 64x32 MLP",
        },
    }
    assignment = {"fit": fit_ids, "dev": dev_ids, "test": test_ids}
    write_json(output / "config.json", config)
    write_json(output / "input_manifest.json", manifest)
    write_json(output / "split_assignment.json", assignment)
    write_json(output / "summary.json", summary)
    with (output / "routing_decisions.jsonl").open("w", encoding="utf-8") as handle:
        for policy in POLICY_ORDER:
            for sample_id, arm in zip(test_ids, test_predictions[policy], strict=True):
                source = decisions["test"][arm][sample_id]
                handle.write(json.dumps({
                    "policy": policy,
                    "selected_by_a6": policy == selected_policy,
                    "sample_id": sample_id,
                    "axis": features[sample_id]["axis"],
                    "selected_arm": arm,
                    "correct": bool(source["correct"]),
                    "total_tokens": int(source["total_tokens"]),
                    "decision_latency_s": float(source["decision_latency_s"]),
                    "counterfactual_source_architecture": source["architecture"],
                }, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, default=Path("benchmarks/external/masbench-v1/subset.jsonl"))
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--dev-per-axis", type=int, default=10)
    parser.add_argument("--token-penalty-per-1k", type=float, default=0.002)
    parser.add_argument("--latency-penalty-per-s", type=float, default=0.0)
    parser.add_argument("--max-predicted-tokens", type=float, default=5000.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if (
        args.dev_per_axis <= 0
        or args.token_penalty_per_1k < 0
        or args.latency_penalty_per_s < 0
        or args.max_predicted_tokens <= 0
    ):
        raise SystemExit("dev-per-axis/budget must be positive and penalties must be non-negative")
    run(args)


if __name__ == "__main__":
    main()
