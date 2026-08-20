"""Execute a frozen A7 policy online on a held-out MultiTown split."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import signal
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx

from .a6_runner import _execute_selected_arm
from .a7_policy import POLICY_VERSION, load_bundle, select_for_scenario
from .advanced_runner import AgentCall, monitor_dual_system
from .counterfactual_runner import _decision_row, load_frozen_bank, read_jsonl
from .masbench_routing import git_state
from .parsing import ParsedDecision
from .runner import atomic_json, utc_now
from .scenarios import Scenario


SCHEMA_VERSION = "multitown-a7-online-run-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def completed_scenarios(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    for row in read_jsonl(path):
        scenario_id = str(row["scenario_id"])
        if scenario_id in completed:
            raise ValueError(f"duplicate A7 scenario result: {scenario_id}")
        completed.add(scenario_id)
    return completed


def summarize(rows: list[dict[str, Any]], *, expected: int) -> dict[str, Any]:
    count = len(rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "updated_at_utc": utc_now(),
        "completed_scenarios": count,
        "expected_scenarios": expected,
        "completion_rate": count / expected if expected else 0.0,
        "accuracy": sum(bool(row["correct"]) for row in rows) / count if count else 0.0,
        "valid_rate": sum(bool(row["valid"]) for row in rows) / count if count else 0.0,
        "request_errors": sum(int(row["request_errors"]) for row in rows),
        "total_tokens": sum(int(row["total_tokens"]) for row in rows),
        "tokens_per_decision": (
            sum(int(row["total_tokens"]) for row in rows) / count if count else 0.0
        ),
        "mean_decision_latency_s": (
            sum(float(row["decision_latency_s"]) for row in rows) / count if count else 0.0
        ),
        "selected_arm_counts": dict(sorted(Counter(row["selected_arm"] for row in rows).items())),
        "family_counts": dict(sorted(Counter(row["family"] for row in rows).items())),
    }


async def run(args: argparse.Namespace) -> int:
    project_root = Path(__file__).resolve().parents[1]
    revision, dirty = git_state(project_root)
    run_dir = Path(args.output_dir).resolve()
    bank_path = Path(args.bank).resolve()
    policy_path = Path(args.policy_bundle).resolve()
    bundle = load_bundle(policy_path)
    frozen = [row for row in load_frozen_bank(bank_path) if row[1] == args.split]
    if not frozen:
        raise ValueError(f"no scenarios in split {args.split!r}")
    run_dir.mkdir(parents=True, exist_ok=True)
    decision_path = run_dir / "decisions.jsonl"
    request_path = run_dir / "requests.jsonl"
    done = completed_scenarios(decision_path) if args.resume else set()
    if not args.resume and (decision_path.exists() or request_path.exists()):
        raise FileExistsError("output contains prior data; choose a new directory or use --resume")
    config = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "source_revision": revision,
        "source_dirty_at_start": dirty,
        "policy_version": POLICY_VERSION,
        "policy_bundle": str(policy_path),
        "policy_bundle_sha256": _sha256(policy_path),
        "selected_config": bundle["selected_config"],
        "scenario_bank": str(bank_path),
        "scenario_bank_sha256": _sha256(bank_path),
        "split": args.split,
        "scenario_count": len(frozen),
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
        "policy_inputs": "pre-execution context only",
        "oracle_visible_to_policy_or_models": False,
    }
    config_path = run_dir / "config.json"
    if config_path.exists():
        prior = json.loads(config_path.read_text(encoding="utf-8"))
        for key in (
            "policy_bundle_sha256", "scenario_bank_sha256", "split", "inference_seed",
            "weak_model", "strong_model", "max_tokens", "planner_max_tokens",
            "temperature", "top_p",
        ):
            if prior.get(key) != config.get(key):
                raise ValueError(f"resume config mismatch for {key}")
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
                    if stop_event.is_set():
                        interrupted = True
                        break
                    if scenario.scenario_id in done:
                        continue
                    if args.max_scenarios is not None and newly_completed >= args.max_scenarios:
                        interrupted = True
                        stop_event.set()
                        break
                    policy_choice = select_for_scenario(bundle, scenario)
                    selected_arm = str(policy_choice["selected_arm"])
                    base_seed = args.inference_seed + scenario_index * 100_003
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
                            arm=selected_arm,
                            scenario=scenario,
                            episode_index=len(done),
                            trial_index=0,
                            seed=base_seed,
                        )
                        for call_index, call in enumerate(final_calls):
                            call.row.update({
                                "architecture": "A7",
                                "selected_arm": selected_arm,
                                "split": split,
                                "scenario_sha256": scenario_hash,
                                "policy_version": POLICY_VERSION,
                                "policy_model_name": policy_choice["model_name"],
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
                        arm=selected_arm,
                        selected=selected,
                        final_calls=final_calls,
                        all_calls=all_calls,
                        organization=organization,
                        decision_latency=time.perf_counter() - cell_started,
                        attempt_count=attempt_count,
                    )
                    row.update({
                        "schema_version": SCHEMA_VERSION,
                        "architecture": "A7",
                        "selected_arm": selected_arm,
                        "policy_version": POLICY_VERSION,
                        "policy_model_name": policy_choice["model_name"],
                        "policy_predicted_accuracy": policy_choice["predicted_accuracy"],
                        "policy_predicted_tokens": policy_choice["predicted_tokens"],
                        "policy_predicted_latency_s": policy_choice["predicted_latency_s"],
                        "policy_predicted_utility": policy_choice["predicted_utility"],
                        "policy_budget_fallback": policy_choice["budget_fallback"],
                        "policy_eligible_arms": policy_choice["eligible_arms"],
                        "policy_arm_predictions": policy_choice["predictions"],
                    })
                    decision_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    decision_handle.flush()
                    done.add(scenario.scenario_id)
                    newly_completed += 1
                    now = time.perf_counter()
                    status = summarize(read_jsonl(decision_path), expected=len(frozen))
                    status.update({
                        "state": "running",
                        "elapsed_s": now - started,
                        "current_scenario": scenario.scenario_id,
                        "current_arm": selected_arm,
                    })
                    atomic_json(run_dir / "status.json", status)
                    if now - last_progress >= args.progress_interval:
                        print(
                            f"[A7] {len(done)}/{len(frozen)} ({len(done)/len(frozen):.1%}) "
                            f"arm={selected_arm} correct={int(row['correct'])} "
                            f"tokens={row['total_tokens']} latency={row['decision_latency_s']:.2f}s "
                            f"errors={row['request_errors']}",
                            flush=True,
                        )
                        last_progress = now
    finally:
        stop_event.set()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass

    rows = read_jsonl(decision_path)
    summary = summarize(rows, expected=len(frozen))
    gpu_failed = (run_dir / "gpu-health-error.json").exists()
    complete = len(rows) == len(frozen) and not gpu_failed
    summary.update({
        "state": "complete" if complete else "failed_gpu_health" if gpu_failed else "stopped",
        "finished_at_utc": utc_now(),
        "elapsed_s_this_invocation": time.perf_counter() - started,
        "interrupted": interrupted,
    })
    atomic_json(run_dir / "summary.json", summary)
    atomic_json(run_dir / "status.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if complete or args.max_scenarios is not None else 130


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--policy-bundle", required=True)
    parser.add_argument("--bank", default="benchmarks/multitown-v0.2-1200/scenario-bank.jsonl")
    parser.add_argument("--split", default="test", choices=("train", "dev", "test"))
    parser.add_argument("--strong-endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--strong-model", default="qwen-game")
    parser.add_argument("--weak-endpoint", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--weak-model", default="qwen-mm-backup")
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--strong-api-key", default="local")
    parser.add_argument("--weak-api-key", default="EMPTY")
    parser.add_argument("--inference-seed", type=int, default=270_801)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--planner-max-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--request-timeout", type=float, default=900)
    parser.add_argument("--monitor-interval", type=float, default=5)
    parser.add_argument("--progress-interval", type=float, default=30)
    parser.add_argument("--a5-vote-agreement", type=float, default=1.0)
    parser.add_argument("--max-cell-attempts", type=int, default=3)
    parser.add_argument("--max-scenarios", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    raise SystemExit(asyncio.run(run(build_parser().parse_args())))


if __name__ == "__main__":
    main()
