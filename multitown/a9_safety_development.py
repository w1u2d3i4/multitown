"""Post-hoc, same-bank A9 review-shield development diagnostic."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from .a9_oof_protocol import (
    DEFAULT_FOLDS, EXPECTED_TRAIN_SHA256, FROZEN_TRAIN_PATH,
    assign_stratified_group_folds, fold_manifest_sha256, load_frozen_train_bank,
    resource_contract_sha256, shared_resource_contract,
)
from .a9_ppo_oof import (
    FORMAL_SEEDS, POLICY_VERSION, RunSchedule, _digest, _evaluate_episode,
    _model_action, _paired_effects, _sha256, _write_json, _write_jsonl,
)
from .long_horizon_env import ACTION_COUNT, MultiTownLongHorizonEnv, RLAction
from .ppo_controller import load_checkpoint


DIAGNOSTIC_VERSION = "multitown-a9-v3-posthoc-review-shield-diagnostic-v1"
FROZEN_R2_PATH = Path("artifacts/a9-v2-ppo-oof-20260813-r2")
EXPECTED_R2_MANIFEST_SHA256 = (
    "62e0b5dc34219bf1816509deaf036f824ece96808eaa58a50a559dfa53497e3a"
)
EXPECTED_R2_RESULT_SHA256 = (
    "2d863fabd6a7ec69331e1cb831a91e28c04ed93311b772f9b429075e2ec8cbfe"
)
REVIEW_STATE_SLICE = slice(33, 36)
FROZEN_SHARED_SOURCE_PATHS = (
    "multitown/a9_long_horizon_env.py",
    "multitown/a9_oof_protocol.py",
    "multitown/a9_ppo_oof.py",
    "multitown/long_horizon_env.py",
    "multitown/ppo_controller.py",
)


def public_review_state(observation: np.ndarray) -> int:
    """Read only the documented public review one-hot from the observation."""

    if (
        observation.shape != (MultiTownLongHorizonEnv.observation_size,)
        or observation.dtype != np.float32
    ):
        raise ValueError("invalid public observation for review shield")
    values = observation[REVIEW_STATE_SLICE]
    index = int(np.argmax(values))
    if not np.array_equal(values, np.eye(3, dtype=np.float32)[index]):
        raise ValueError("review-state observation is not strict one-hot")
    return index


def review_shielded_action(
    selector: Callable[[np.ndarray, np.ndarray], int],
    observation: np.ndarray, mask: np.ndarray,
) -> tuple[int, int, bool]:
    """Apply a public pre-decision mask and expose the blocked base proposal."""

    proposed = int(selector(observation, mask))
    if proposed not in range(ACTION_COUNT) or not bool(mask[proposed]):
        raise ValueError("base policy proposed an invalid action")
    should_block = (
        proposed == int(RLAction.EXECUTE) and public_review_state(observation) != 1
    )
    if not should_block:
        return proposed, proposed, False
    shielded = mask.copy()
    shielded[int(RLAction.EXECUTE)] = False
    shielded.flags.writeable = False
    selected = int(selector(observation, shielded))
    if selected not in range(ACTION_COUNT) or not bool(shielded[selected]):
        raise ValueError("shielded policy proposed an invalid action")
    return selected, proposed, True


def _verify_r2_artifacts(root: Path) -> dict[str, Any]:
    raw = (root / FROZEN_R2_PATH).resolve()
    manifest_path = raw / "artifact-manifest.json"
    result_path = raw / "result.json"
    if (
        not manifest_path.is_file()
        or _sha256(manifest_path) != EXPECTED_R2_MANIFEST_SHA256
        or not result_path.is_file()
        or _sha256(result_path) != EXPECTED_R2_RESULT_SHA256
        or (raw / "INVALIDATED.json").exists()
    ):
        raise RuntimeError("frozen A9-v2 r2 artifacts do not match the accepted record")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_paths = set(manifest["files"])
    actual_paths = {
        str(path.relative_to(raw)) for path in raw.rglob("*")
        if path.is_file() and path.name not in {"artifact-manifest.json", "RUNNING.json"}
    }
    if expected_paths != actual_paths:
        raise RuntimeError("frozen A9-v2 artifact path set changed")
    for relative, metadata in manifest["files"].items():
        path = raw / relative
        if path.stat().st_size != metadata["bytes"] or _sha256(path) != metadata["sha256"]:
            raise RuntimeError(f"frozen A9-v2 artifact changed: {relative}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not result["claim_gates"]["validity_gates_passed"]:
        raise RuntimeError("frozen A9-v2 result validity gate is not accepted")
    run_contract = json.loads(
        (raw / "run-contract.json").read_text(encoding="utf-8")
    )
    if _digest(run_contract) != result["run_contract_sha256"]:
        raise RuntimeError("frozen A9-v2 run contract no longer binds to result")
    return {
        "path": raw, "manifest": manifest, "result": result,
        "run_contract": run_contract,
    }


def _validate_shared_stack_bindings(
    root: Path, run_contract: Mapping[str, Any], *, bank: Any,
    assignments: Sequence[Any], resource: Mapping[str, Any],
) -> None:
    """Fail closed if any shared data, fold, resource, or runtime source drifts."""

    tracked = run_contract.get("source", {}).get("tracked_source_sha256", {})
    for relative in FROZEN_SHARED_SOURCE_PATHS:
        if tracked.get(relative) != _sha256(root / relative):
            raise RuntimeError(f"shared frozen source drifted: {relative}")
    if (
        bank.payload_sha256 != EXPECTED_TRAIN_SHA256
        or bank.payload_sha256 != run_contract.get("train_bank_sha256")
        or fold_manifest_sha256(assignments)
        != run_contract.get("fold_manifest_sha256")
        or resource_contract_sha256(resource)
        != run_contract.get("resource_contract_sha256")
        or resource.get("environment_source_sha256")
        != run_contract.get("environment_source_sha256")
    ):
        raise RuntimeError("shared frozen data/fold/resource binding drifted")


def _paired_seed_arrays(
    source_rows: Sequence[Mapping[str, Any]],
    shield_rows: Sequence[Mapping[str, Any]], *, metric: str,
    folds: int, seeds: Sequence[int],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    seed_tuple = tuple(seeds)
    source_index = {
        (str(row["episode_id"]), int(row["training_seed"])): row
        for row in source_rows
    }
    shield_index = {
        (str(row["episode_id"]), int(row["training_seed"])): row
        for row in shield_rows
    }
    if (
        len(source_index) != len(source_rows)
        or len(shield_index) != len(shield_rows)
        or set(source_index) != set(shield_index)
    ):
        raise ValueError("source/shield pairing is duplicate, missing, or mismatched")
    episode_folds: dict[str, int] = {}
    for (episode_id, training_seed), source in source_index.items():
        if training_seed not in seed_tuple:
            raise ValueError("unexpected fixed training seed")
        fold = int(source["outer_fold"])
        shield = shield_index[(episode_id, training_seed)]
        if int(shield["outer_fold"]) != fold:
            raise ValueError("source/shield fold mismatch")
        if episode_id in episode_folds and episode_folds[episode_id] != fold:
            raise ValueError("episode appears in multiple folds")
        episode_folds[episode_id] = fold
    expected = {
        (episode_id, training_seed)
        for episode_id in episode_folds for training_seed in seed_tuple
    }
    if set(source_index) != expected:
        raise ValueError("source/shield rows are not a full episode x seed product")
    by_fold: dict[int, list[str]] = defaultdict(list)
    for episode_id, fold in episode_folds.items():
        by_fold[fold].append(episode_id)
    if set(by_fold) != set(range(folds)):
        raise ValueError("source/shield fold coverage mismatch")

    def value(row: Mapping[str, Any]) -> float:
        if metric in {"episode_success", "assisted_episode_success"}:
            return float(bool(row[metric]))
        if metric == "wrong_execution":
            return float(bool(row["had_wrong_execution"]))
        return float(row[metric])

    arrays: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for fold in range(folds):
        episode_ids = sorted(by_fold[fold])
        shield_values = np.asarray([
            np.mean([
                value(shield_index[(episode_id, training_seed)])
                for training_seed in seed_tuple
            ]) for episode_id in episode_ids
        ], dtype=np.float64)
        source_values = np.asarray([
            np.mean([
                value(source_index[(episode_id, training_seed)])
                for training_seed in seed_tuple
            ]) for episode_id in episode_ids
        ], dtype=np.float64)
        arrays[fold] = shield_values, source_values
    return arrays


def paired_seed_cluster_bootstrap(
    source_rows: Sequence[Mapping[str, Any]],
    shield_rows: Sequence[Mapping[str, Any]], *, metric: str,
    folds: int = DEFAULT_FOLDS, seeds: Sequence[int] = FORMAL_SEEDS,
    iterations: int = 20_000, seed: int = 20260813,
    ratio_reduction: bool = False,
) -> dict[str, Any]:
    """Pair shield and frozen A9 by episode/seed; resample episodes in folds."""

    if type(iterations) is not int or iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    arrays = _paired_seed_arrays(
        source_rows, shield_rows, metric=metric, folds=folds, seeds=seeds,
    )
    shield_point = float(np.mean([left.mean() for left, _ in arrays.values()]))
    source_point = float(np.mean([right.mean() for _, right in arrays.values()]))
    if ratio_reduction and source_point == 0.0:
        raise ValueError("undefined paired ratio denominator")
    point = (
        1.0 - shield_point / source_point
        if ratio_reduction else shield_point - source_point
    )
    rng = np.random.default_rng(seed)
    values = np.empty(iterations, dtype=np.float64)
    for start in range(0, iterations, 256):
        width = min(256, iterations - start)
        shield_replicates = np.zeros(width, dtype=np.float64)
        source_replicates = np.zeros(width, dtype=np.float64)
        for fold in range(folds):
            shield_values, source_values = arrays[fold]
            indices = rng.integers(
                0, len(shield_values), size=(width, len(shield_values)),
            )
            shield_replicates += shield_values[indices].mean(axis=1) / folds
            source_replicates += source_values[indices].mean(axis=1) / folds
        if ratio_reduction:
            if np.any(source_replicates == 0.0):
                raise ValueError("undefined paired bootstrap ratio denominator")
            values[start:start + width] = 1.0 - shield_replicates / source_replicates
        else:
            values[start:start + width] = shield_replicates - source_replicates
    return {
        "schema_version": "multitown-a9-v3-paired-seed-fold-bootstrap-v1",
        "metric": metric,
        "estimand": (
            "shield_ratio_reduction_vs_frozen_A9" if ratio_reduction
            else "shield_minus_frozen_A9"
        ),
        "point": float(point),
        "shield_mean": shield_point,
        "frozen_a9_mean": source_point,
        "ci95_low": float(np.quantile(values, 0.025, method="linear")),
        "ci95_high": float(np.quantile(values, 0.975, method="linear")),
        "iterations": iterations,
        "rng_seed": seed,
        "fold_weighting": "equal",
        "fold_resampling": {
            str(fold): len(arrays[fold][0]) for fold in range(folds)
        },
        "training_seeds_fixed_not_resampled": list(seeds),
    }


def paired_seed_cluster_ratio_bootstrap(
    source_rows: Sequence[Mapping[str, Any]],
    shield_rows: Sequence[Mapping[str, Any]], *, numerator: str,
    denominator: str, folds: int = DEFAULT_FOLDS,
    seeds: Sequence[int] = FORMAL_SEEDS, iterations: int = 20_000,
    seed: int = 20260813,
) -> dict[str, Any]:
    """Compare equal-fold aggregate ratios with episode/seed pairing."""

    if type(iterations) is not int or iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    source_num = _paired_seed_arrays(
        source_rows, shield_rows, metric=numerator, folds=folds, seeds=seeds,
    )
    source_den = _paired_seed_arrays(
        source_rows, shield_rows, metric=denominator, folds=folds, seeds=seeds,
    )
    shield_point = float(np.mean([
        source_num[fold][0].sum() / source_den[fold][0].sum()
        for fold in range(folds)
    ]))
    source_point = float(np.mean([
        source_num[fold][1].sum() / source_den[fold][1].sum()
        for fold in range(folds)
    ]))
    rng = np.random.default_rng(seed)
    values = np.empty(iterations, dtype=np.float64)
    for start in range(0, iterations, 256):
        width = min(256, iterations - start)
        differences = np.zeros(width, dtype=np.float64)
        for fold in range(folds):
            shield_num, source_num_values = source_num[fold]
            shield_den, source_den_values = source_den[fold]
            indices = rng.integers(0, len(shield_num), size=(width, len(shield_num)))
            sampled_shield_den = shield_den[indices].sum(axis=1)
            sampled_source_den = source_den_values[indices].sum(axis=1)
            if np.any(sampled_shield_den == 0.0) or np.any(sampled_source_den == 0.0):
                raise ValueError("paired ratio bootstrap denominator is zero")
            differences += (
                shield_num[indices].sum(axis=1) / sampled_shield_den
                - source_num_values[indices].sum(axis=1) / sampled_source_den
            ) / folds
        values[start:start + width] = differences
    return {
        "schema_version": "multitown-a9-v3-paired-seed-fold-ratio-bootstrap-v1",
        "metric": f"sum({numerator})/sum({denominator})",
        "estimand": "shield_minus_frozen_A9",
        "point": float(shield_point - source_point),
        "shield_ratio": shield_point,
        "frozen_a9_ratio": source_point,
        "ci95_low": float(np.quantile(values, 0.025, method="linear")),
        "ci95_high": float(np.quantile(values, 0.975, method="linear")),
        "iterations": iterations,
        "rng_seed": seed,
        "fold_weighting": "equal; ratio recomputed within every fold replicate",
        "training_seeds_fixed_not_resampled": list(seeds),
    }


def _artifact_manifest(output: Path, *, source_revision: str) -> dict[str, Any]:
    files = {
        str(path.relative_to(output)): {
            "bytes": path.stat().st_size, "sha256": _sha256(path),
        }
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact-manifest.json"
    }
    return {
        "schema_version": "multitown-a9-v3-development-artifact-manifest-v1",
        "source_revision": source_revision,
        "files": files,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }


def run(output: Path) -> int:
    root = Path(__file__).resolve().parents[1]
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    frozen = _verify_r2_artifacts(root)
    raw = Path(frozen["path"])
    r2_result = frozen["result"]
    frozen_run = frozen["run_contract"]
    r2_run_contract = str(r2_result["run_contract_sha256"])
    bank = load_frozen_train_bank(FROZEN_TRAIN_PATH)
    assignments = assign_stratified_group_folds(bank)
    resource = shared_resource_contract(bank)
    _validate_shared_stack_bindings(
        root, frozen_run, bank=bank, assignments=assignments, resource=resource,
    )
    resource_sha = resource_contract_sha256(resource)
    environment_sha = str(resource["environment_source_sha256"])
    assignment_index = {row.episode_id: row for row in assignments}
    episode_index = {row.episode_id: row for row in bank.episodes}
    a8_rows = [
        json.loads(line) for line in
        (raw / "a8-oof-decisions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    a8_index = {str(row["episode_id"]): row for row in a8_rows}
    if len(a8_rows) != 3000 or set(a8_index) != set(episode_index):
        raise RuntimeError("frozen A8 OOF baseline is incomplete")
    source_a9_rows = [
        json.loads(line) for line in
        (raw / "a9-oof-decisions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    source_a9_index = {
        (str(row["episode_id"]), int(row["training_seed"])): row
        for row in source_a9_rows
    }
    expected_source_keys = {
        (episode_id, training_seed) for episode_id in episode_index
        for training_seed in FORMAL_SEEDS
    }
    if (
        len(source_a9_rows) != 9000
        or len(source_a9_index) != len(source_a9_rows)
        or set(source_a9_index) != expected_source_keys
    ):
        raise RuntimeError("frozen A9 OOF source is not a full episode x seed product")
    for episode_id, baseline in a8_index.items():
        assignment = assignment_index[episode_id]
        if any((
            int(baseline["outer_fold"]) != assignment.fold,
            baseline["episode_sha256"] != assignment.episode_sha256,
            baseline["train_bank_sha256"] != bank.payload_sha256,
            baseline["resource_contract_sha256"] != resource_sha,
            baseline["environment_source_sha256"] != environment_sha,
            baseline["run_contract_sha256"] != r2_run_contract,
            int(baseline["invalid_actions"]) != 0,
            int(baseline["budget_violations"]) != 0,
        )):
            raise RuntimeError("frozen A8 row binding changed")
        for training_seed in FORMAL_SEEDS:
            source = source_a9_index[(episode_id, training_seed)]
            if any((
                int(source["outer_fold"]) != assignment.fold,
                source["episode_sha256"] != assignment.episode_sha256,
                source["train_bank_sha256"] != bank.payload_sha256,
                source["resource_contract_sha256"] != resource_sha,
                source["environment_source_sha256"] != environment_sha,
                source["run_contract_sha256"] != r2_run_contract,
                source["a8_row_sha256"] != _digest(baseline),
                int(source["invalid_actions"]) != 0,
                int(source["budget_violations"]) != 0,
            )):
                raise RuntimeError("frozen A9 row binding changed")
    schedule = RunSchedule(
        mode="adaptive-development", seeds=FORMAL_SEEDS,
        folds=tuple(range(DEFAULT_FOLDS)), updates=120,
        episodes_per_update=48, evaluation_episodes_per_fold=600,
        bootstrap_iterations=20_000, bootstrap_seed=20260813, threads=8,
    )
    contract = {
        "schema_version": DIAGNOSTIC_VERSION,
        "evidence_scope": "post-hoc same-bank adaptive mechanism diagnostic",
        "source_revision": revision,
        "source_dirty": bool(status),
        "executed_source_sha256": {
            relative: _sha256(root / relative)
            for relative in (
                "multitown/a9_safety_development.py",
                "multitown/a9_long_horizon_env.py",
                "multitown/a9_ppo_oof.py",
                "multitown/a9_oof_protocol.py",
                "multitown/ppo_controller.py",
                "multitown/long_horizon_env.py",
            )
        },
        "source_a9_v2_run_contract_sha256": r2_run_contract,
        "source_a9_v2_manifest_sha256": EXPECTED_R2_MANIFEST_SHA256,
        "source_a9_v2_decisions_sha256": _sha256(raw / "a9-oof-decisions.jsonl"),
        "train_bank_sha256": bank.payload_sha256,
        "fold_manifest_sha256": fold_manifest_sha256(assignments),
        "resource_contract_sha256": resource_sha,
        "environment_source_sha256": environment_sha,
        "checkpoint_reuse": True,
        "policy_retraining": False,
        "shield": "mask execute unless public review_state == pass",
        "formal_or_independent_claim_allowed": False,
        "schedule": {
            "folds": list(schedule.folds), "seeds": list(schedule.seeds),
            "episodes_per_fold": schedule.evaluation_episodes_per_fold,
            "bootstrap_iterations": schedule.bootstrap_iterations,
            "bootstrap_seed": schedule.bootstrap_seed,
        },
    }
    contract_sha = _digest(contract)
    output.mkdir(parents=True)
    _write_json(output / "development-contract.json", contract)
    rows: list[dict[str, Any]] = []
    for fold in schedule.folds:
        episode_ids = sorted(
            row.episode_id for row in assignments if row.fold == fold
        )
        for training_seed in schedule.seeds:
            checkpoint = (
                raw / "fits" / f"fold-{fold}"
                / f"seed-{training_seed}" / "final.pt"
            )
            model, metadata = load_checkpoint(
                checkpoint, torch.device("cpu"), expected_policy_version=POLICY_VERSION,
            )
            if (
                int(metadata["update"]) != 120
                or int(metadata["seed"]) != training_seed
                or metadata.get("run_contract_sha256") != r2_run_contract
            ):
                raise RuntimeError("source checkpoint metadata changed")
            checkpoint_sha = _sha256(checkpoint)
            for episode_id in episode_ids:
                counters = Counter()

                def action(observation: np.ndarray, mask: np.ndarray) -> int:
                    selected, proposed, intervened = review_shielded_action(
                        lambda obs, valid: _model_action(model, obs, valid),
                        observation, mask,
                    )
                    counters["base_execute_proposals"] += int(
                        proposed == int(RLAction.EXECUTE)
                    )
                    counters["shield_interventions"] += int(intervened)
                    return selected

                assignment = assignment_index[episode_id]
                baseline = a8_index[episode_id]
                source_a9 = source_a9_index[(episode_id, training_seed)]
                if source_a9["final_checkpoint_sha256"] != checkpoint_sha:
                    raise RuntimeError("frozen A9 row/checkpoint binding changed")
                row = _evaluate_episode(
                    episode_index[episode_id], action, fold=fold,
                    system="A9-v3-posthoc-review-shield", training_seed=training_seed,
                    episode_sha256=assignment.episode_sha256,
                    bank_sha256=bank.payload_sha256,
                    resource_sha256=str(baseline["resource_contract_sha256"]),
                    environment_sha256=str(baseline["environment_source_sha256"]),
                    checkpoint_sha256=checkpoint_sha,
                    a8_row_sha256=_digest(baseline),
                    run_contract_sha256=contract_sha,
                )
                row.update({
                    "source_policy_run_contract_sha256": r2_run_contract,
                    "source_a9_row_sha256": _digest(source_a9),
                    "base_execute_proposals": counters["base_execute_proposals"],
                    "shield_interventions": counters["shield_interventions"],
                })
                rows.append(row)
    if len(rows) != 9000 or any(
        int(row["invalid_actions"]) or int(row["budget_violations"]) for row in rows
    ):
        raise RuntimeError("incomplete or invalid shield diagnostic")
    _write_jsonl(output / "shielded-oof-replay.jsonl", rows)
    comparisons_a8 = _paired_effects(a8_rows, rows, schedule)
    comparisons_a9 = {
        metric: paired_seed_cluster_bootstrap(
            source_a9_rows, rows, metric=metric, seeds=schedule.seeds,
            iterations=schedule.bootstrap_iterations,
            seed=schedule.bootstrap_seed + index,
        )
        for index, metric in enumerate((
            "episode_success", "wrong_execution", "tokens_used",
            "latency_used_s", "safety_penalty_burden", "return",
        ))
    }
    comparisons_a9["token_reduction_fraction"] = paired_seed_cluster_bootstrap(
        source_a9_rows, rows, metric="tokens_used", seeds=schedule.seeds,
        iterations=schedule.bootstrap_iterations,
        seed=schedule.bootstrap_seed + 100, ratio_reduction=True,
    )
    comparisons_a9["wrong_executions_per_incident"] = (
        paired_seed_cluster_ratio_bootstrap(
            source_a9_rows, rows, numerator="wrong_executions",
            denominator="incidents", seeds=schedule.seeds,
            iterations=schedule.bootstrap_iterations,
            seed=schedule.bootstrap_seed + 101,
        )
    )
    result = {
        "schema_version": DIAGNOSTIC_VERSION,
        "evidence_scope": "post-hoc same-bank adaptive mechanism diagnostic",
        "development_contract_sha256": contract_sha,
        "formal_claim_evaluable": False,
        "independent_replication": False,
        "checkpoint_reuse": True,
        "policy_retraining": False,
        "rows": len(rows),
        "unique_episodes": len({row["episode_id"] for row in rows}),
        "shield_interventions": sum(int(row["shield_interventions"]) for row in rows),
        "episodes_with_shield_intervention": sum(
            int(row["shield_interventions"] > 0) for row in rows
        ),
        "comparisons_vs_frozen_a8": comparisons_a8,
        "paired_comparisons_vs_frozen_a9": comparisons_a9,
        "summary": {
            "episode_success_rate": float(np.mean([
                bool(row["episode_success"]) for row in rows
            ])),
            "unsafe_episode_rate": float(np.mean([
                bool(row["had_wrong_execution"]) for row in rows
            ])),
            "tokens_per_episode": float(np.mean([
                int(row["tokens_used"]) for row in rows
            ])),
            "wrong_executions_per_incident": sum(
                int(row["wrong_executions"]) for row in rows
            ) / sum(int(row["incidents"]) for row in rows),
        },
        "limitations": [
            "The shield was designed after observing A9-v2 outcomes.",
            "The same train-only OOF episodes and frozen checkpoints are reused.",
            "This is a mechanism diagnostic, not new confirmatory evidence.",
        ],
    }
    _write_json(output / "result.json", result)
    _write_json(
        output / "artifact-manifest.json",
        _artifact_manifest(output, source_revision=revision),
    )
    print(json.dumps({
        "output": str(output), "summary": result["summary"],
        "shield_interventions": result["shield_interventions"],
        "formal_claim_evaluable": False,
    }, ensure_ascii=False, indent=2))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    raise SystemExit(run(args.output_dir))


if __name__ == "__main__":
    main()
