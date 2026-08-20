"""Collect a resumable full-information arm matrix on the frozen MultiTown bank."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import signal
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import httpx

from .a6_policy import ARM_ORDER
from .a6_runner import _execute_selected_arm
from .advanced_runner import AgentCall, monitor_dual_system
from .masbench_routing import git_state
from .parsing import ParsedDecision
from .runner import atomic_json, utc_now
from .scenarios import Scenario


SCHEMA_VERSION = "multitown-counterfactual-matrix-v1"
DEFAULT_BANK = Path("benchmarks/multitown-v0.2-1200/scenario-bank.jsonl")
DEFAULT_SPLITS = ("train", "dev", "test")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_csv(raw: str, allowed: Iterable[str], label: str) -> tuple[str, ...]:
    allowed_tuple = tuple(allowed)
    values = tuple(dict.fromkeys(part.strip() for part in raw.split(",") if part.strip()))
    unknown = sorted(set(values) - set(allowed_tuple))
    if not values or unknown:
        raise ValueError(f"invalid {label}: values={values}, unknown={unknown}")
    return values


def scenario_from_frozen_row(row: dict[str, Any]) -> Scenario:
    return Scenario(
        scenario_id=str(row["scenario_id"]),
        family=str(row["family"]),
        seed=int(row["seed"]),
        prompt=str(row["prompt"]),
        allowed_actions=tuple(str(value) for value in row["allowed_actions"]),
        oracle_action=str(row["oracle_action"]),
        metadata=dict(row["metadata"]),
    )


def load_frozen_bank(path: Path) -> list[tuple[Scenario, str, str]]:
    rows: list[tuple[Scenario, str, str]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            scenario = scenario_from_frozen_row(raw)
            if scenario.scenario_id in seen:
                raise ValueError(f"duplicate scenario ID in frozen bank: {scenario.scenario_id}")
            seen.add(scenario.scenario_id)
            rows.append((scenario, str(raw["split"]), str(raw["scenario_sha256"])))
    return rows


def completed_cells(path: Path) -> set[tuple[str, str]]:
    completed: set[tuple[str, str]] = set()
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["scenario_id"]), str(row["arm"]))
            if key in completed:
                raise ValueError(f"duplicate completed cell in {path}: {key}")
            completed.add(key)
    return completed


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize_rows(
    rows: list[dict[str, Any]], *, expected_cells: int | None = None,
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["split"]), str(row["arm"]))].append(row)
    cells: list[dict[str, Any]] = []
    for (split, arm), members in sorted(grouped.items()):
        count = len(members)
        cells.append({
            "split": split,
            "arm": arm,
            "count": count,
            "correct": sum(bool(row["correct"]) for row in members),
            "accuracy": sum(bool(row["correct"]) for row in members) / count,
            "valid_rate": sum(bool(row["valid"]) for row in members) / count,
            "request_errors": sum(int(row["request_errors"]) for row in members),
            "total_tokens": sum(int(row["total_tokens"]) for row in members),
            "tokens_per_decision": sum(int(row["total_tokens"]) for row in members) / count,
            "mean_decision_latency_s": sum(float(row["decision_latency_s"]) for row in members) / count,
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "updated_at_utc": utc_now(),
        "completed_cells": len(rows),
        "expected_cells": expected_cells,
        "completion_rate": (
            len(rows) / expected_cells if expected_cells else None
        ),
        "request_errors": sum(int(row["request_errors"]) for row in rows),
        "total_tokens": sum(int(row["total_tokens"]) for row in rows),
        "cells": cells,
    }


def _decision_row(
    *, scenario: Scenario, split: str, scenario_hash: str, arm: str,
    selected: ParsedDecision, final_calls: list[AgentCall], all_calls: list[AgentCall],
    organization: dict[str, Any], decision_latency: float, attempt_count: int,
) -> dict[str, Any]:
    candidate_calls = [call for call in final_calls if call.row["expects_decision"]]
    prompt_tokens = sum(call.response.prompt_tokens for call in all_calls)
    completion_tokens = sum(call.response.completion_tokens for call in all_calls)
    total_tokens = sum(call.response.total_tokens for call in all_calls)
    weak_tokens = sum(
        call.response.total_tokens for call in all_calls if call.row["model_tier"] == "weak"
    )
    request_errors = sum(call.response.error is not None for call in all_calls)
    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": utc_now(),
        "scenario_id": scenario.scenario_id,
        "scenario_sha256": scenario_hash,
        "split": split,
        "family": scenario.family,
        "world_seed": scenario.seed,
        "arm": arm,
        "oracle_action": scenario.oracle_action,
        "selected_action": selected.action,
        "correct": selected.action == scenario.oracle_action,
        "valid": selected.valid,
        "confidence": selected.confidence,
        "agreement": organization["agreement"],
        "action_diversity": organization["action_diversity"],
        "route": organization["route"],
        "final_source": organization["final_source"],
        "organization_switches": organization["organization_switches"],
        "weak_calls": organization["weak_calls"],
        "strong_calls": organization["strong_calls"],
        "verifier_called": organization["verifier_called"],
        "worker_aggregate_action": organization["worker_aggregate_action"],
        "candidate_actions": [call.parsed.action for call in candidate_calls],
        "candidate_correct": [
            call.parsed.action == scenario.oracle_action for call in candidate_calls
        ],
        "candidate_roles": [call.row["role"] for call in candidate_calls],
        "request_count": len(all_calls),
        "cell_attempts": attempt_count,
        "decision_latency_s": decision_latency,
        "mean_request_latency_s": (
            sum(call.response.latency_s for call in all_calls) / len(all_calls)
        ),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "weak_tokens": weak_tokens,
        "strong_tokens": total_tokens - weak_tokens,
        "request_errors": request_errors,
    }


async def run(args: argparse.Namespace) -> int:
    project_root = Path(__file__).resolve().parents[1]
    run_dir = Path(args.output_dir).resolve()
    bank_path = Path(args.bank).resolve()
    arms = _parse_csv(args.arms, ARM_ORDER, "arms")
    splits = _parse_csv(args.splits, DEFAULT_SPLITS, "splits")
    frozen = [row for row in load_frozen_bank(bank_path) if row[1] in splits]
    expected_cells = len(frozen) * len(arms)
    run_dir.mkdir(parents=True, exist_ok=True)
    decision_path = run_dir / "decisions.jsonl"
    request_path = run_dir / "requests.jsonl"
    done = completed_cells(decision_path) if args.resume else set()
    if not args.resume and (decision_path.exists() or request_path.exists()):
        raise FileExistsError("output contains prior data; choose a new directory or use --resume")

    revision, dirty = git_state(project_root)
    config_path = run_dir / "config.json"
    config = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "source_revision": revision,
        "source_dirty_at_start": dirty,
        "scenario_bank": str(bank_path),
        "scenario_bank_sha256": _sha256(bank_path),
        "arms": list(arms),
        "splits": list(splits),
        "scenario_count": len(frozen),
        "expected_cells": expected_cells,
        "inference_seed": args.inference_seed,
        "weak_endpoint": args.weak_endpoint,
        "weak_model": args.weak_model,
        "strong_endpoint": args.strong_endpoint,
        "strong_model": args.strong_model,
        "max_tokens": args.max_tokens,
        "planner_max_tokens": args.planner_max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_cell_attempts": args.max_cell_attempts,
        "oracle_visible_to_models_or_controller": False,
        "cost_includes_failed_attempts": True,
    }
    if config_path.exists():
        prior = json.loads(config_path.read_text(encoding="utf-8"))
        for key in (
            "scenario_bank_sha256", "arms", "splits", "inference_seed",
            "weak_model", "strong_model", "max_tokens", "planner_max_tokens",
            "temperature", "top_p",
        ):
            if prior.get(key) != config.get(key):
                raise ValueError(f"resume config mismatch for {key}: {prior.get(key)!r} != {config.get(key)!r}")
        config = prior
    else:
        atomic_json(config_path, config)

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, stop_event.set)
        except NotImplementedError:
            pass
    started = time.perf_counter()
    monitor_task = asyncio.create_task(monitor_dual_system(
        run_dir=run_dir,
        endpoints={"strong": args.strong_endpoint, "weak": args.weak_endpoint},
        api_key=args.api_key,
        api_keys={"strong": args.strong_api_key, "weak": args.weak_api_key},
        started=started,
        interval=args.monitor_interval,
        stop_event=stop_event,
    ))
    limits = httpx.Limits(max_connections=16, max_keepalive_connections=16)
    timeout = httpx.Timeout(connect=30, read=args.request_timeout, write=60, pool=60)
    newly_completed = 0
    last_progress = started
    interrupted = False
    try:
        with (
            request_path.open("a", encoding="utf-8") as request_handle,
            decision_path.open("a", encoding="utf-8") as decision_handle,
        ):
            async with httpx.AsyncClient(
                limits=limits, timeout=timeout, trust_env=False,
            ) as client:
                for scenario_index, (scenario, split, scenario_hash) in enumerate(frozen):
                    for arm_index, arm in enumerate(arms):
                        if stop_event.is_set():
                            interrupted = True
                            break
                        if (scenario.scenario_id, arm) in done:
                            continue
                        if args.max_cells is not None and newly_completed >= args.max_cells:
                            interrupted = True
                            stop_event.set()
                            break
                        base_seed = (
                            args.inference_seed
                            + scenario_index * 100_003
                            + arm_index * 10_000_019
                        )
                        cell_started = time.perf_counter()
                        all_calls: list[AgentCall] = []
                        selected: ParsedDecision | None = None
                        final_calls: list[AgentCall] = []
                        organization: dict[str, Any] | None = None
                        attempt_count = 0
                        for attempt_count in range(1, args.max_cell_attempts + 1):
                            selected, final_calls, organization = await _execute_selected_arm(
                                client,
                                args=args,
                                arm=arm,
                                scenario=scenario,
                                episode_index=len(done) + newly_completed,
                                trial_index=0,
                                seed=base_seed,
                            )
                            for call_index, call in enumerate(final_calls):
                                call.row.update({
                                    "architecture": "counterfactual_matrix",
                                    "target_arm": arm,
                                    "split": split,
                                    "scenario_sha256": scenario_hash,
                                    "cell_attempt": attempt_count,
                                    "request_index_in_attempt": call_index,
                                })
                                request_handle.write(json.dumps(call.row, ensure_ascii=False) + "\n")
                            request_handle.flush()
                            all_calls.extend(final_calls)
                            if not any(call.response.error for call in final_calls):
                                break
                        assert selected is not None and organization is not None
                        row = _decision_row(
                            scenario=scenario,
                            split=split,
                            scenario_hash=scenario_hash,
                            arm=arm,
                            selected=selected,
                            final_calls=final_calls,
                            all_calls=all_calls,
                            organization=organization,
                            decision_latency=time.perf_counter() - cell_started,
                            attempt_count=attempt_count,
                        )
                        decision_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        decision_handle.flush()
                        newly_completed += 1
                        done.add((scenario.scenario_id, arm))
                        now = time.perf_counter()
                        current_rows = read_jsonl(decision_path)
                        status = summarize_rows(current_rows, expected_cells=expected_cells)
                        status.update({
                            "state": "running",
                            "current_scenario": scenario.scenario_id,
                            "current_split": split,
                            "current_arm": arm,
                            "elapsed_s": now - started,
                        })
                        atomic_json(run_dir / "status.json", status)
                        if now - last_progress >= args.progress_interval:
                            print(
                                f"[matrix] {len(done)}/{expected_cells} "
                                f"({len(done)/expected_cells:.1%}) split={split} arm={arm} "
                                f"accuracy={int(row['correct'])} tokens={row['total_tokens']} "
                                f"latency={row['decision_latency_s']:.2f}s "
                                f"errors={row['request_errors']}",
                                flush=True,
                            )
                            last_progress = now
                    if interrupted:
                        break
    finally:
        stop_event.set()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass

    rows = read_jsonl(decision_path)
    summary = summarize_rows(rows, expected_cells=expected_cells)
    gpu_failed = (run_dir / "gpu-health-error.json").exists()
    complete = len(rows) == expected_cells and not gpu_failed
    summary.update({
        "state": "complete" if complete else "failed_gpu_health" if gpu_failed else "stopped",
        "finished_at_utc": utc_now(),
        "elapsed_s_this_invocation": time.perf_counter() - started,
    })
    atomic_json(run_dir / "summary.json", summary)
    atomic_json(run_dir / "status.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if complete or args.max_cells is not None else 130


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bank", default=str(DEFAULT_BANK))
    parser.add_argument("--arms", default=",".join(ARM_ORDER))
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS))
    parser.add_argument("--strong-endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--strong-model", default="qwen-game")
    parser.add_argument("--weak-endpoint", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--weak-model", default="qwen-mm-backup")
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--strong-api-key", default="local")
    parser.add_argument("--weak-api-key", default="EMPTY")
    parser.add_argument("--inference-seed", type=int, default=170_801)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--planner-max-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--request-timeout", type=float, default=900)
    parser.add_argument("--monitor-interval", type=float, default=5)
    parser.add_argument("--progress-interval", type=float, default=30)
    parser.add_argument("--a5-vote-agreement", type=float, default=1.0)
    parser.add_argument("--max-cell-attempts", type=int, default=3)
    parser.add_argument("--max-cells", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    raise SystemExit(asyncio.run(run(build_parser().parse_args())))


if __name__ == "__main__":
    main()
