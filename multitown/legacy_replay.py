"""Normalize A0--A6 one-step decisions and verify lossless replay invariants."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator

from .contracts import (
    Budget,
    ControllerAction,
    ControllerActionKind,
    RewardComponents,
    StateFact,
    StateSnapshot,
    TaskContract,
    TrajectoryStep,
)


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSON at {path}:{number}: {exc}") from exc
            if not isinstance(value, dict):
                raise RuntimeError(f"expected a JSON object at {path}:{number}")
            yield value


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    os.replace(temporary, path)
    return count


def _load_scenarios(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("scenarios", "scenario_bank", "items"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise RuntimeError(f"unsupported scenario bank structure: {path}")
    scenarios = {str(item["scenario_id"]): item for item in payload}
    if len(scenarios) != len(payload):
        raise RuntimeError(f"duplicate scenario ids in {path}")
    return scenarios


def _task_contract(scenario: dict[str, Any]) -> TaskContract:
    return TaskContract(
        task_id=str(scenario["scenario_id"]),
        family=str(scenario["family"]),
        instruction=str(scenario["prompt"]),
        allowed_actions=tuple(str(item) for item in scenario["allowed_actions"]),
        validator_id="legacy_deterministic_oracle",
        budget=Budget(),
        metadata={
            "world_seed": scenario.get("seed"),
            "oracle_action": scenario.get("oracle_action"),
            "scenario_metadata": scenario.get("metadata", {}),
        },
    )


def _activated_agents(row: dict[str, Any]) -> tuple[str, ...]:
    roles = tuple(str(item) for item in row.get("candidate_roles", []) if item)
    if roles:
        return roles
    architecture = str(row["architecture"])
    count = max(1, len(row.get("agent_actions", [])))
    tier = "strong" if architecture == "A1" else "weak"
    return tuple(f"legacy_{tier}_agent_{index + 1}" for index in range(count))


def normalize_decision(row: dict[str, Any], scenario: dict[str, Any], source_file: Path) -> dict[str, Any]:
    architecture = str(row["architecture"])
    episode_index = int(row["episode_index"])
    trial_index = int(row["trial_index"])
    scenario_id = str(row["scenario_id"])
    episode_id = f"legacy-{architecture}-{episode_index:08d}"
    request_count = int(row.get("request_count", max(1, len(row.get("agent_actions", [])))))
    route = row.get("route") or f"{architecture.lower()}_legacy_route"
    observation = StateSnapshot(
        episode_id=episode_id,
        step_index=0,
        state_version=0,
        observer_id="legacy_global_observer",
        facts=(
            StateFact(
                key="scenario_reference",
                value={"scenario_id": scenario_id, "world_seed": row.get("world_seed")},
                source="scenario_bank.json",
                observed_by="legacy_global_observer",
                state_version=0,
                updated_step=0,
            ),
        ),
    )
    action = ControllerAction(
        kind=ControllerActionKind.SUBMIT,
        controller_id=f"legacy_{architecture.lower()}_controller",
        task_id=scenario_id,
        selected_action=str(row["selected_action"]),
        activated_agents=_activated_agents(row),
        assigned_role=f"legacy_{architecture.lower()}_organization",
        model_tier="strong" if architecture == "A1" else ("mixed" if architecture in {"A3", "A4", "A5", "A6"} else "weak"),
        reason_codes=("legacy_completed_decision",),
        metadata={
            "route": route,
            "selected_arm": row.get("selected_arm"),
            "policy_version": row.get("policy_version"),
            "final_source": row.get("final_source"),
            "organization_switches": int(row.get("organization_switches", 0)),
        },
    )
    reward = RewardComponents(
        final_success=1.0 if bool(row["correct"]) else 0.0,
        invalid_action=-1.0 if not bool(row["valid"]) else 0.0,
    )
    step = TrajectoryStep(
        trajectory_id=f"{episode_id}:{trial_index}:{scenario_id}",
        episode_id=episode_id,
        architecture=architecture,
        step_index=0,
        timestamp_utc=str(row["timestamp_utc"]),
        task_id=scenario_id,
        observation=observation,
        controller_action=action,
        messages=(),
        tool_result={
            "selected_action": row["selected_action"],
            "oracle_action": row["oracle_action"],
            "valid": bool(row["valid"]),
            "correct": bool(row["correct"]),
        },
        reward=reward,
        metrics={
            "episode_index": episode_index,
            "trial_index": trial_index,
            "family": row["family"],
            "prompt_tokens": int(row.get("prompt_tokens", 0)),
            "completion_tokens": int(row.get("completion_tokens", 0)),
            "total_tokens": int(row.get("total_tokens", 0)),
            "request_count": request_count,
            "request_errors": int(row.get("request_errors", 0)),
            "decision_latency_s": float(row.get("decision_latency_s", 0.0)),
            "communication_messages": int(row.get("communication_messages", 0)),
            "verifier_called": bool(row.get("verifier_called", False)),
            "strict_json_rate": row.get("strict_json_rate"),
        },
        terminated=True,
        legacy_source={"file": _portable_path(source_file), "line_identity": [episode_index, trial_index, scenario_id]},
    )
    return step.to_dict()


def _replay_compare(decision: dict[str, Any], step: dict[str, Any]) -> list[str]:
    checks = {
        "architecture": (decision["architecture"], step["architecture"]),
        "scenario_id": (decision["scenario_id"], step["task_id"]),
        "episode_index": (decision["episode_index"], step["metrics"]["episode_index"]),
        "trial_index": (decision["trial_index"], step["metrics"]["trial_index"]),
        "family": (decision["family"], step["metrics"]["family"]),
        "selected_action": (decision["selected_action"], step["tool_result"]["selected_action"]),
        "oracle_action": (decision["oracle_action"], step["tool_result"]["oracle_action"]),
        "valid": (bool(decision["valid"]), step["tool_result"]["valid"]),
        "correct": (bool(decision["correct"]), step["tool_result"]["correct"]),
        "total_tokens": (int(decision.get("total_tokens", 0)), step["metrics"]["total_tokens"]),
        "request_errors": (int(decision.get("request_errors", 0)), step["metrics"]["request_errors"]),
        "decision_latency_s": (float(decision.get("decision_latency_s", 0.0)), step["metrics"]["decision_latency_s"]),
    }
    return [name for name, (left, right) in checks.items() if left != right]


def validate_replay(architecture_dir: Path, normalized_path: Path) -> dict[str, Any]:
    decisions_path = architecture_dir / "decisions.jsonl"
    mismatch_count = 0
    mismatch_examples: list[dict[str, Any]] = []
    decisions = 0
    correct = 0
    tokens = 0
    requests = 0
    for line, (decision, step) in enumerate(zip(iter_jsonl(decisions_path), iter_jsonl(normalized_path), strict=True), 1):
        mismatches = _replay_compare(decision, step)
        if mismatches:
            mismatch_count += 1
            if len(mismatch_examples) < 10:
                mismatch_examples.append({"line": line, "fields": mismatches})
        decisions += 1
        correct += int(bool(step["tool_result"]["correct"]))
        tokens += int(step["metrics"]["total_tokens"])
        requests += int(step["metrics"]["request_count"])
    summary = json.loads((architecture_dir / "summary.json").read_text(encoding="utf-8"))
    aggregate_checks = {
        "decision_count": decisions == int(summary["decisions"]),
        "correct_count": correct == int(summary["correct"]),
        "total_tokens": tokens == int(summary["total_tokens"]),
        "row_equality": mismatch_count == 0,
    }
    return {
        "architecture": architecture_dir.name,
        "passed": all(aggregate_checks.values()),
        "checks": aggregate_checks,
        "decisions": decisions,
        "correct": correct,
        "total_tokens": tokens,
        "request_count_reconstructed": requests,
        "mismatch_count": mismatch_count,
        "mismatch_examples": mismatch_examples,
    }


def convert_architecture(architecture_dir: Path, output_dir: Path) -> dict[str, Any]:
    architecture_dir = architecture_dir.resolve()
    output_dir = output_dir.resolve()
    decisions_path = architecture_dir / "decisions.jsonl"
    scenario_path = architecture_dir / "scenario_bank.json"
    scenarios = _load_scenarios(scenario_path)
    contracts_path = output_dir / "task_contracts.jsonl"
    trajectories_path = output_dir / "trajectory_steps.jsonl"
    _atomic_jsonl(contracts_path, (_task_contract(item).to_dict() for item in scenarios.values()))

    def normalized_rows() -> Iterator[dict[str, Any]]:
        for decision in iter_jsonl(decisions_path):
            scenario_id = str(decision["scenario_id"])
            try:
                scenario = scenarios[scenario_id]
            except KeyError as exc:
                raise RuntimeError(f"decision references unknown scenario {scenario_id}") from exc
            yield normalize_decision(decision, scenario, decisions_path)

    converted_count = _atomic_jsonl(trajectories_path, normalized_rows())
    replay = validate_replay(architecture_dir, trajectories_path)
    payload = {
        "schema_version": "multitown-legacy-conversion-v1",
        "architecture": architecture_dir.name,
        "source": {
            "architecture_dir": _portable_path(architecture_dir),
            "decisions_sha256": sha256_file(decisions_path),
            "scenario_bank_sha256": sha256_file(scenario_path),
            "summary_sha256": sha256_file(architecture_dir / "summary.json"),
        },
        "output": {
            "task_contracts": _portable_path(contracts_path),
            "task_contracts_sha256": sha256_file(contracts_path),
            "trajectory_steps": _portable_path(trajectories_path),
            "trajectory_steps_sha256": sha256_file(trajectories_path),
            "converted_count": converted_count,
        },
        "replay": replay,
    }
    _atomic_json(output_dir / "conversion.json", payload)
    if not replay["passed"]:
        raise RuntimeError(f"legacy replay validation failed for {architecture_dir.name}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture-dir", action="append", required=True, help="A0--A6 artifact directory; repeatable")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--report", help="optional combined replay report path")
    args = parser.parse_args()
    output_root = Path(args.output_root)
    results = []
    for value in args.architecture_dir:
        architecture_dir = Path(value)
        results.append(convert_architecture(architecture_dir, output_root / architecture_dir.name))
    report = {
        "schema_version": "multitown-legacy-replay-report-v1",
        "passed": all(item["replay"]["passed"] for item in results),
        "architectures": [item["replay"] for item in results],
        "source_hashes": {item["architecture"]: item["source"] for item in results},
        "normalized_hashes": {item["architecture"]: item["output"] for item in results},
    }
    if args.report:
        _atomic_json(Path(args.report), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
