"""Run A6: a cross-fitted, budget-aware router over A0-A5 organizations."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import signal
import time
from pathlib import Path
from typing import Any

import httpx

from .a6_policy import ARM_ORDER, POLICY_VERSION, build_crossfit_policy, choice_for
from .advanced_runner import (
    AgentCall,
    DECISION_SYSTEM_PROMPT,
    _call_agent,
    _run_a3,
    _run_a4,
    _run_a5,
    monitor_dual_system,
)
from .parsing import ParsedDecision, aggregate_decisions
from .report import generate_run_report
from .runner import atomic_json, utc_now
from .scenarios import Scenario, build_scenario_bank


async def _run_base_arm(
    client: httpx.AsyncClient, *, args: argparse.Namespace, arm: str,
    scenario: Scenario, episode_index: int, trial_index: int, seed: int,
) -> tuple[ParsedDecision, list[AgentCall], dict[str, Any]]:
    if arm not in {"A0", "A1", "A2"}:
        raise ValueError(f"Unsupported base arm: {arm}")
    agent_count = 4 if arm == "A2" else 1
    model_tier = "strong" if arm == "A1" else "weak"
    endpoint = args.strong_endpoint if model_tier == "strong" else args.weak_endpoint
    model = args.strong_model if model_tier == "strong" else args.weak_model
    calls = await asyncio.gather(*[
        _call_agent(
            client,
            args=args,
            architecture="A6",
            episode_index=episode_index,
            trial_index=trial_index,
            scenario=scenario,
            phase=f"routed_{arm.lower()}_execution",
            role=(
                f"{model_tier}_single_solver"
                if agent_count == 1 else f"weak_vote_member_{index + 1}"
            ),
            model_tier=model_tier,
            endpoint=endpoint,
            model=model,
            seed=seed + 1001 + index * 1_000_003,
            messages=[
                {"role": "system", "content": DECISION_SYSTEM_PROMPT},
                {"role": "user", "content": scenario.prompt},
            ],
            expects_decision=True,
        )
        for index in range(agent_count)
    ])
    if arm == "A2":
        selected, agreement, diversity = aggregate_decisions(
            [call.parsed for call in calls], scenario.allowed_actions
        )
        final_source = "deterministic_weak_vote"
        route = "a2_four_weak_vote"
    else:
        selected = calls[0].parsed
        agreement = 1.0 if selected.valid else 0.0
        diversity = int(selected.valid)
        final_source = str(calls[0].row["role"])
        route = f"{arm.lower()}_{model_tier}_solo"
    return selected, list(calls), {
        "route": route,
        "final_source": final_source,
        "agreement": agreement,
        "action_diversity": diversity,
        "organization_switches": 0,
        "weak_calls": agent_count if model_tier == "weak" else 0,
        "strong_calls": agent_count if model_tier == "strong" else 0,
        "verifier_called": False,
        "planner_valid": None,
        "worker_aggregate_action": selected.action,
    }


async def _execute_selected_arm(
    client: httpx.AsyncClient, *, args: argparse.Namespace, arm: str,
    scenario: Scenario, episode_index: int, trial_index: int, seed: int,
) -> tuple[ParsedDecision, list[AgentCall], dict[str, Any]]:
    if arm in {"A0", "A1", "A2"}:
        selected, calls, organization = await _run_base_arm(
            client, args=args, arm=arm, scenario=scenario,
            episode_index=episode_index, trial_index=trial_index, seed=seed,
        )
    else:
        runner = {"A3": _run_a3, "A4": _run_a4, "A5": _run_a5}[arm]
        selected, calls, organization = await runner(
            client,
            args=args,
            scenario=scenario,
            episode_index=episode_index,
            trial_index=trial_index,
            seed=seed,
            architecture="A6",
        )
    organization = dict(organization)
    organization["route"] = f"{arm.lower()}:{organization['route']}"
    organization["selected_arm"] = arm
    return selected, calls, organization


def _baseline_dirs(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "A0": Path(args.baseline_a0_dir),
        "A1": Path(args.baseline_a1_dir),
        "A2": Path(args.baseline_a2_dir),
        "A3": Path(args.baseline_a3_dir),
        "A4": Path(args.baseline_a4_dir),
        "A5": Path(args.baseline_a5_dir),
    }


async def run(args: argparse.Namespace) -> int:
    run_dir = Path(args.output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    scenarios = build_scenario_bank(args.world_seed, args.case_count)
    baseline_dirs = _baseline_dirs(args)
    policy = build_crossfit_policy(
        scenarios,
        baseline_dirs=baseline_dirs,
        folds=args.policy_folds,
        token_penalty_per_1k=args.token_penalty_per_1k,
        latency_penalty_per_s=args.latency_penalty_per_s,
    )
    atomic_json(run_dir / "policy.json", policy)
    config = vars(args).copy()
    config.update({
        "started_at_utc": utc_now(),
        "architecture": "A6",
        "method": "cross_fitted_budget_aware_router_over_A0_A5_organization_arms",
        "timing_scope": "after_both_models_loaded_and_warmed_until_duration_deadline",
        "deterministic_oracle": True,
        "oracle_visible_to_router_or_models": False,
        "self_reported_confidence_used_for_routing": False,
        "policy_version": POLICY_VERSION,
        "policy_crossfit_offline_estimate": policy["crossfit_offline_estimate"],
        "policy_arms": list(ARM_ORDER),
    })
    atomic_json(run_dir / "config.json", config)
    atomic_json(run_dir / "scenario_bank.json", {
        "scenarios": [item.to_dict() for item in scenarios]
    })

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, stop_event.set)
        except NotImplementedError:
            pass

    started = time.perf_counter()
    deadline = started + args.duration_seconds
    monitor_task = asyncio.create_task(monitor_dual_system(
        run_dir=run_dir,
        endpoints={"strong": args.strong_endpoint, "weak": args.weak_endpoint},
        api_key=args.api_key,
        started=started,
        interval=args.monitor_interval,
        stop_event=stop_event,
    ))
    decisions = correct = invalid = request_errors = total_tokens = 0
    total_latency = 0.0
    last_progress = last_report = started
    interrupted_before_deadline = False
    request_path = run_dir / "requests.jsonl"
    decision_path = run_dir / "decisions.jsonl"
    limits = httpx.Limits(max_connections=16, max_keepalive_connections=16)
    timeout = httpx.Timeout(connect=30, read=args.request_timeout, write=60, pool=60)

    try:
        with (
            request_path.open("a", encoding="utf-8") as request_handle,
            decision_path.open("a", encoding="utf-8") as decision_handle,
        ):
            async with httpx.AsyncClient(
                limits=limits, timeout=timeout, trust_env=False,
            ) as client:
                while time.perf_counter() < deadline and not stop_event.is_set():
                    scenario_index = decisions % len(scenarios)
                    scenario = scenarios[scenario_index]
                    trial_index = decisions // len(scenarios)
                    base_seed = (
                        args.inference_seed
                        + trial_index * 10_000_019
                        + scenario_index * 100_003
                    )
                    policy_choice = choice_for(policy, scenario.scenario_id)
                    selected_arm = str(policy_choice["selected_arm"])
                    decision_started = time.perf_counter()
                    selected, calls, organization = await _execute_selected_arm(
                        client,
                        args=args,
                        arm=selected_arm,
                        scenario=scenario,
                        episode_index=decisions,
                        trial_index=trial_index,
                        seed=base_seed,
                    )
                    decision_latency = time.perf_counter() - decision_started
                    for call_index, call in enumerate(calls):
                        call.row.update({
                            "request_index_in_episode": call_index,
                            "selected_arm": selected_arm,
                            "policy_fold": policy_choice["fold"],
                            "policy_version": POLICY_VERSION,
                            "policy_predicted_utility": policy_choice["predicted_utility"],
                        })
                        request_handle.write(json.dumps(call.row, ensure_ascii=False) + "\n")
                    request_handle.flush()

                    is_correct = selected.action == scenario.oracle_action
                    prompt_tokens = sum(call.response.prompt_tokens for call in calls)
                    completion_tokens = sum(call.response.completion_tokens for call in calls)
                    decision_tokens = sum(call.response.total_tokens for call in calls)
                    weak_request_tokens = sum(
                        call.response.total_tokens
                        for call in calls if call.row["model_tier"] == "weak"
                    )
                    strong_request_tokens = decision_tokens - weak_request_tokens
                    decision_errors = sum(call.response.error is not None for call in calls)
                    candidate_calls = [call for call in calls if call.row["expects_decision"]]
                    row: dict[str, Any] = {
                        "timestamp_utc": utc_now(),
                        "elapsed_s": time.perf_counter() - started,
                        "architecture": "A6",
                        "episode_index": decisions,
                        "trial_index": trial_index,
                        "scenario_id": scenario.scenario_id,
                        "family": scenario.family,
                        "world_seed": scenario.seed,
                        "oracle_action": scenario.oracle_action,
                        "selected_action": selected.action,
                        "correct": is_correct,
                        "valid": selected.valid,
                        "confidence": selected.confidence,
                        "agreement": organization["agreement"],
                        "action_diversity": organization["action_diversity"],
                        "route": organization["route"],
                        "selected_arm": selected_arm,
                        "policy_fold": policy_choice["fold"],
                        "policy_version": POLICY_VERSION,
                        "policy_predicted_accuracy": policy_choice["predicted_accuracy"],
                        "policy_predicted_tokens": policy_choice["predicted_tokens"],
                        "policy_predicted_latency_s": policy_choice["predicted_latency_s"],
                        "policy_predicted_utility": policy_choice["predicted_utility"],
                        "policy_arm_scores": policy_choice["arm_scores"],
                        "final_source": organization["final_source"],
                        "organization_switches": organization["organization_switches"],
                        "router_decisions": 1,
                        "weak_calls": organization["weak_calls"],
                        "strong_calls": organization["strong_calls"],
                        "verifier_called": organization["verifier_called"],
                        "planner_valid": organization["planner_valid"],
                        "worker_aggregate_action": organization["worker_aggregate_action"],
                        "candidate_actions": [call.parsed.action for call in candidate_calls],
                        "candidate_correct": [
                            call.parsed.action == scenario.oracle_action for call in candidate_calls
                        ],
                        "candidate_roles": [call.row["role"] for call in candidate_calls],
                        "strict_json_calls": sum(
                            bool(call.row["strict_json_compliant"]) for call in candidate_calls
                        ),
                        "strict_json_rate": (
                            sum(bool(call.row["strict_json_compliant"]) for call in candidate_calls)
                            / len(candidate_calls) if candidate_calls else None
                        ),
                        "request_count": len(calls),
                        "communication_messages": len(calls) * 2,
                        "decision_latency_s": decision_latency,
                        "mean_request_latency_s": (
                            sum(call.response.latency_s for call in calls) / len(calls)
                        ),
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": decision_tokens,
                        "weak_tokens": weak_request_tokens,
                        "strong_tokens": strong_request_tokens,
                        "request_errors": decision_errors,
                    }
                    decision_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    decision_handle.flush()

                    decisions += 1
                    correct += int(is_correct)
                    invalid += int(not selected.valid)
                    request_errors += decision_errors
                    total_tokens += decision_tokens
                    total_latency += decision_latency
                    now = time.perf_counter()
                    status = {
                        "state": "running",
                        "architecture": "A6",
                        "policy_version": POLICY_VERSION,
                        "updated_at_utc": utc_now(),
                        "elapsed_s": now - started,
                        "remaining_s": max(0.0, deadline - now),
                        "duration_s": args.duration_seconds,
                        "decisions": decisions,
                        "correct": correct,
                        "accuracy": correct / decisions,
                        "invalid_decisions": invalid,
                        "request_errors": request_errors,
                        "total_tokens": total_tokens,
                        "tokens_per_decision": total_tokens / decisions,
                        "mean_decision_latency_s": total_latency / decisions,
                        "current_scenario": scenario.scenario_id,
                        "current_selected_arm": selected_arm,
                        "current_route": organization["route"],
                    }
                    atomic_json(run_dir / "status.json", status)
                    if now - last_progress >= args.progress_interval:
                        print(
                            f"[A6] elapsed={(now-started)/3600:.3f}h "
                            f"remaining={max(0.0, deadline-now)/3600:.3f}h "
                            f"decisions={decisions} accuracy={correct/decisions:.4f} "
                            f"arm={selected_arm} route={organization['route']} "
                            f"tokens={total_tokens} mean_latency={total_latency/decisions:.2f}s "
                            f"errors={request_errors}",
                            flush=True,
                        )
                        last_progress = now
                    if now - last_report >= args.report_interval:
                        try:
                            await asyncio.to_thread(generate_run_report, run_dir)
                        except Exception as exc:
                            print(f"[A6] interim report failed: {exc}", flush=True)
                        last_report = now
        interrupted_before_deadline = stop_event.is_set() and time.perf_counter() < deadline
    finally:
        stop_event.set()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass

    ended = time.perf_counter()
    gpu_failure_path = run_dir / "gpu-health-error.json"
    final_state = (
        "failed_gpu_health" if gpu_failure_path.exists()
        else "stopped" if interrupted_before_deadline
        else "complete"
    )
    final_status = {
        "state": final_state,
        "architecture": "A6",
        "policy_version": POLICY_VERSION,
        "updated_at_utc": utc_now(),
        "elapsed_s": ended - started,
        "remaining_s": max(0.0, deadline - ended),
        "duration_s": args.duration_seconds,
        "decisions": decisions,
        "correct": correct,
        "accuracy": correct / decisions if decisions else 0.0,
        "invalid_decisions": invalid,
        "request_errors": request_errors,
        "total_tokens": total_tokens,
        "tokens_per_decision": total_tokens / decisions if decisions else 0.0,
        "mean_decision_latency_s": total_latency / decisions if decisions else math.nan,
        "finished_at_utc": utc_now(),
        "stop_reason": "gpu_health_watchdog" if gpu_failure_path.exists() else None,
    }
    atomic_json(run_dir / "status.json", final_status)
    generate_run_report(run_dir)
    print(json.dumps(final_status, ensure_ascii=False), flush=True)
    return 0 if final_state == "complete" else 130


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strong-endpoint", required=True)
    parser.add_argument("--strong-model", required=True)
    parser.add_argument("--weak-endpoint", required=True)
    parser.add_argument("--weak-model", required=True)
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--duration-seconds", type=float, default=14_400)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--case-count", type=int, default=120)
    parser.add_argument("--world-seed", type=int, default=20_260_807)
    parser.add_argument("--inference-seed", type=int, default=90_801)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--planner-max-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--request-timeout", type=float, default=900)
    parser.add_argument("--monitor-interval", type=float, default=5)
    parser.add_argument("--report-interval", type=float, default=300)
    parser.add_argument("--progress-interval", type=float, default=60)
    parser.add_argument("--reasoning-mode", default="off")
    parser.add_argument("--a5-vote-agreement", type=float, default=1.0)
    parser.add_argument("--policy-folds", type=int, default=5)
    parser.add_argument("--token-penalty-per-1k", type=float, default=0.005)
    parser.add_argument("--latency-penalty-per-s", type=float, default=0.0025)
    for arm in ARM_ORDER:
        parser.add_argument(f"--baseline-{arm.lower()}-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
