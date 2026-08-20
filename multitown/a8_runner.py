"""Run A8 selective delegation and sparse execution-time reorganization."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import signal
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx

from .a7_policy import load_bundle, predict_arms
from .a8_controller import CONTROLLER_VERSION, ValidationResult, validate_candidate
from .advanced_runner import (
    AgentCall,
    DECISION_SYSTEM_PROMPT,
    VERIFIER_SYSTEM_PROMPT,
    _call_agent,
    _decision_instruction,
    monitor_dual_system,
)
from .counterfactual_runner import _decision_row, load_frozen_bank, read_jsonl
from .masbench_routing import git_state
from .parsing import ParsedDecision
from .runner import atomic_json, utc_now
from .scenarios import Scenario


SCHEMA_VERSION = "multitown-a8-online-run-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validation_text(result: ValidationResult) -> str:
    return json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True)


def _edge(source: str, target: str, purpose: str) -> dict[str, str]:
    return {"source": source, "target": target, "purpose": purpose}


async def execute_a8_cell(
    client: httpx.AsyncClient,
    *,
    args: argparse.Namespace,
    scenario: Scenario,
    bundle: dict[str, Any],
    threshold: float,
    episode_index: int,
    seed: int,
) -> tuple[ParsedDecision, list[AgentCall], dict[str, Any]]:
    model_name = str(bundle["selected_config"]["model_name"])
    arm_predictions = predict_arms(bundle, scenario, model_name=model_name)
    predicted_a0_accuracy = float(arm_predictions["A0"]["predicted_accuracy"])
    calls: list[AgentCall] = []
    edges = [
        _edge("controller", "weak_initial", "initial_task"),
        _edge("weak_initial", "controller", "initial_candidate"),
    ]
    initial = await _call_agent(
        client,
        args=args,
        architecture="A8",
        episode_index=episode_index,
        trial_index=0,
        scenario=scenario,
        phase="initial_attempt",
        role="weak_initial_solver",
        model_tier="weak",
        endpoint=args.weak_endpoint,
        model=args.weak_model,
        seed=seed + 1001,
        messages=[
            {"role": "system", "content": DECISION_SYSTEM_PROMPT},
            {"role": "user", "content": scenario.prompt + "\n" + _decision_instruction(scenario)},
        ],
        expects_decision=True,
    )
    calls.append(initial)
    initial_validation = validate_candidate(scenario, initial.parsed.action)
    trace: list[dict[str, Any]] = [{
        "phase": "initial_attempt",
        "role": initial.row["role"],
        "action": initial.parsed.action,
        "validation": initial_validation.to_dict(),
    }]
    selected = initial.parsed
    final_source = "weak_initial_solver"
    route = "initial_weak_early_stop"
    stop_reason = "high_predicted_reliability_and_hard_constraints_pass"
    weak_specialist: AgentCall | None = None
    weak_validation: ValidationResult | None = None
    strong_specialist: AgentCall | None = None
    strong_validation: ValidationResult | None = None
    reviewer: AgentCall | None = None
    reviewer_validation: ValidationResult | None = None
    delegated = not (
        initial_validation.hard_constraints_pass
        and predicted_a0_accuracy >= threshold
    )
    if delegated and initial_validation.hard_constraints_pass:
        edges.extend([
            _edge("controller", "weak_specialist", "independent_second_opinion"),
            _edge("weak_specialist", "controller", "specialist_candidate"),
        ])
        weak_specialist = await _call_agent(
            client,
            args=args,
            architecture="A8",
            episode_index=episode_index,
            trial_index=0,
            scenario=scenario,
            phase="selective_weak_delegation",
            role=f"weak_{scenario.family}_specialist",
            model_tier="weak",
            endpoint=args.weak_endpoint,
            model=args.weak_model,
            seed=seed + 2001,
            messages=[
                {"role": "system", "content": DECISION_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    scenario.prompt
                    + "\n你是被选择性唤醒的领域专家。请独立复算，不采纳其他 Agent 的候选。\n"
                    + _decision_instruction(scenario)
                )},
            ],
            expects_decision=True,
        )
        calls.append(weak_specialist)
        weak_validation = validate_candidate(scenario, weak_specialist.parsed.action)
        trace.append({
            "phase": "selective_weak_delegation",
            "role": weak_specialist.row["role"],
            "action": weak_specialist.parsed.action,
            "validation": weak_validation.to_dict(),
        })
        if (
            weak_validation.hard_constraints_pass
            and weak_specialist.parsed.action == initial.parsed.action
        ):
            route = "two_weak_consensus_stop"
            stop_reason = "independent_weak_agreement_and_hard_constraints_pass"
        else:
            strong_specialist = None  # set below through the common escalation path
    if delegated and (
        not initial_validation.hard_constraints_pass
        or weak_specialist is None
        or weak_validation is None
        or not weak_validation.hard_constraints_pass
        or weak_specialist.parsed.action != initial.parsed.action
    ):
        edges.extend([
            _edge("controller", "strong_specialist", "constraint_or_disagreement_resolution"),
            _edge("strong_specialist", "controller", "strong_candidate"),
        ])
        candidates = {
            "initial_weak": initial.parsed.action,
            "initial_validation": initial_validation.to_dict(),
            "weak_specialist": weak_specialist.parsed.action if weak_specialist else None,
            "weak_specialist_validation": weak_validation.to_dict() if weak_validation else None,
        }
        strong_specialist = await _call_agent(
            client,
            args=args,
            architecture="A8",
            episode_index=episode_index,
            trial_index=0,
            scenario=scenario,
            phase="strong_specialist_escalation",
            role=f"strong_{scenario.family}_specialist",
            model_tier="strong",
            endpoint=args.strong_endpoint,
            model=args.strong_model,
            seed=seed + 3001,
            messages=[
                {"role": "system", "content": DECISION_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    scenario.prompt
                    + "\n\n控制器检测到硬约束失败或独立候选分歧。候选与安全检查：\n"
                    + json.dumps(candidates, ensure_ascii=False, sort_keys=True)
                    + "\n请独立复算并解决冲突。\n"
                    + _decision_instruction(scenario)
                )},
            ],
            expects_decision=True,
        )
        calls.append(strong_specialist)
        strong_validation = validate_candidate(scenario, strong_specialist.parsed.action)
        trace.append({
            "phase": "strong_specialist_escalation",
            "role": strong_specialist.row["role"],
            "action": strong_specialist.parsed.action,
            "validation": strong_validation.to_dict(),
        })
        if strong_specialist.parsed.valid and strong_validation.hard_constraints_pass:
            selected = strong_specialist.parsed
            final_source = str(strong_specialist.row["role"])
            route = "strong_specialist_resolution"
            stop_reason = "strong_candidate_passed_hard_constraints"
        else:
            edges.extend([
                _edge("controller", "independent_reviewer", "unsafe_strong_candidate_review"),
                _edge("independent_reviewer", "controller", "reviewer_candidate"),
            ])
            reviewer = await _call_agent(
                client,
                args=args,
                architecture="A8",
                episode_index=episode_index,
                trial_index=0,
                scenario=scenario,
                phase="independent_review",
                role="independent_strong_reviewer",
                model_tier="strong",
                endpoint=args.strong_endpoint,
                model=args.strong_model,
                seed=seed + 4001,
                messages=[
                    {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
                    {"role": "user", "content": (
                        scenario.prompt
                        + "\n\n上一强候选未通过安全检查："
                        + _validation_text(strong_validation)
                        + "\n请隔离上下文独立复算。\n"
                        + _decision_instruction(scenario)
                    )},
                ],
                expects_decision=True,
            )
            calls.append(reviewer)
            reviewer_validation = validate_candidate(scenario, reviewer.parsed.action)
            trace.append({
                "phase": "independent_review",
                "role": reviewer.row["role"],
                "action": reviewer.parsed.action,
                "validation": reviewer_validation.to_dict(),
            })
            if reviewer.parsed.valid and reviewer_validation.hard_constraints_pass:
                selected = reviewer.parsed
                final_source = "independent_strong_reviewer"
                route = "independent_reviewer_resolution"
                stop_reason = "reviewer_candidate_passed_hard_constraints"
            else:
                safe_candidates = [
                    (call, validation)
                    for call, validation in (
                        (strong_specialist, strong_validation),
                        (weak_specialist, weak_validation),
                        (initial, initial_validation),
                    )
                    if call is not None and validation is not None
                    and call.parsed.valid and validation.hard_constraints_pass
                ]
                if safe_candidates:
                    selected = safe_candidates[0][0].parsed
                    final_source = str(safe_candidates[0][0].row["role"])
                    route = "deterministic_safe_fallback"
                    stop_reason = "review_failed_use_latest_hard_safe_candidate"
                else:
                    selected = reviewer.parsed
                    final_source = "human_escalation_required"
                    route = "human_escalation_required"
                    stop_reason = "no_candidate_passed_hard_constraints"
    action_candidates = [call.parsed.action for call in calls if call.row["expects_decision"]]
    active_agents = {"controller", *(edge["source"] for edge in edges), *(edge["target"] for edge in edges)}
    possible_edges = len(active_agents) * max(1, len(active_agents) - 1)
    organization = {
        "route": route,
        "final_source": final_source,
        "agreement": (
            max(Counter(action_candidates).values()) / len(action_candidates)
            if action_candidates else 0.0
        ),
        "action_diversity": len(set(action_candidates)),
        "organization_switches": len(calls) - 1,
        "weak_calls": sum(call.row["model_tier"] == "weak" for call in calls),
        "strong_calls": sum(call.row["model_tier"] == "strong" for call in calls),
        "verifier_called": reviewer is not None,
        "planner_valid": None,
        "worker_aggregate_action": initial.parsed.action,
        "controller_version": CONTROLLER_VERSION,
        "policy_model_name": model_name,
        "predicted_a0_accuracy": predicted_a0_accuracy,
        "early_stop_threshold": threshold,
        "delegated": delegated,
        "early_stop": not delegated,
        "stop_reason": stop_reason,
        "initial_action": initial.parsed.action,
        "initial_validation": initial_validation.to_dict(),
        "final_validation": validate_candidate(scenario, selected.action).to_dict(),
        "phase_trace": trace,
        "communication_edges": edges,
        "active_agents": sorted(active_agents),
        "communication_density": len(edges) / possible_edges,
        "message_tokens": sum(call.response.prompt_tokens for call in calls),
        "reorganization_count": len(calls) - 1,
        "constraint_failure_triggered": not initial_validation.hard_constraints_pass,
        "weak_specialist_called": weak_specialist is not None,
        "strong_specialist_called": strong_specialist is not None,
        "human_escalation_required": final_source == "human_escalation_required",
        "a7_arm_predictions": arm_predictions,
    }
    return selected, calls, organization


def summarize(rows: list[dict[str, Any]], *, expected: int) -> dict[str, Any]:
    count = len(rows)
    sorted_latency = sorted(float(row["decision_latency_s"]) for row in rows)
    p95_index = max(0, math.ceil(0.95 * count) - 1) if count else 0
    family_accuracy = {}
    for family in sorted({str(row["family"]) for row in rows}):
        members = [row for row in rows if row["family"] == family]
        family_accuracy[family] = sum(bool(row["correct"]) for row in members) / len(members)
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
        "tokens_per_decision": sum(int(row["total_tokens"]) for row in rows) / count if count else 0.0,
        "mean_decision_latency_s": sum(float(row["decision_latency_s"]) for row in rows) / count if count else 0.0,
        "p95_decision_latency_s": sorted_latency[p95_index] if count else 0.0,
        "delegation_rate": sum(bool(row["delegated"]) for row in rows) / count if count else 0.0,
        "early_stop_rate": sum(bool(row["early_stop"]) for row in rows) / count if count else 0.0,
        "mean_communication_density": sum(float(row["communication_density"]) for row in rows) / count if count else 0.0,
        "message_tokens_per_decision": sum(int(row["message_tokens"]) for row in rows) / count if count else 0.0,
        "mean_reorganizations": sum(int(row["reorganization_count"]) for row in rows) / count if count else 0.0,
        "mean_reorganization_gain": sum(int(row["reorganization_gain"]) for row in rows) / count if count else 0.0,
        "unnecessary_delegation_rate": sum(bool(row["unnecessary_delegation"]) for row in rows) / count if count else 0.0,
        "hard_failure_recovery_rate": (
            sum(bool(row["final_validation"]["hard_constraints_pass"]) for row in rows if row["constraint_failure_triggered"])
            / sum(bool(row["constraint_failure_triggered"]) for row in rows)
            if any(bool(row["constraint_failure_triggered"]) for row in rows) else None
        ),
        "route_counts": dict(sorted(Counter(row["route"] for row in rows).items())),
        "family_accuracy": family_accuracy,
        "worst_family_accuracy": min(family_accuracy.values()) if family_accuracy else 0.0,
    }


async def run(args: argparse.Namespace) -> int:
    project_root = Path(__file__).resolve().parents[1]
    revision, dirty = git_state(project_root)
    run_dir = Path(args.output_dir).resolve()
    bank_path = Path(args.bank).resolve()
    bundle_path = Path(args.policy_bundle).resolve()
    controller_config_path = Path(args.controller_config).resolve()
    bundle = load_bundle(bundle_path)
    controller_config = json.loads(controller_config_path.read_text(encoding="utf-8"))
    if controller_config.get("controller_version") != CONTROLLER_VERSION:
        raise ValueError("unsupported A8 controller config")
    threshold = float(controller_config["selected"]["early_stop_threshold"])
    frozen = [row for row in load_frozen_bank(bank_path) if row[1] == args.split]
    run_dir.mkdir(parents=True, exist_ok=True)
    decision_path = run_dir / "decisions.jsonl"
    request_path = run_dir / "requests.jsonl"
    prior_rows = read_jsonl(decision_path) if args.resume else []
    done = {str(row["scenario_id"]) for row in prior_rows}
    if len(done) != len(prior_rows):
        raise ValueError("duplicate A8 decision rows")
    if not args.resume and (decision_path.exists() or request_path.exists()):
        raise FileExistsError("output contains prior data; choose a new directory or use --resume")
    config = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "source_revision": revision,
        "source_dirty_at_start": dirty,
        "controller_version": CONTROLLER_VERSION,
        "controller_config": str(controller_config_path),
        "controller_config_sha256": _sha256(controller_config_path),
        "policy_bundle": str(bundle_path),
        "policy_bundle_sha256": _sha256(bundle_path),
        "scenario_bank": str(bank_path),
        "scenario_bank_sha256": _sha256(bank_path),
        "split": args.split,
        "scenario_count": len(frozen),
        "early_stop_threshold": threshold,
        "inference_seed": args.inference_seed,
        "weak_endpoint": args.weak_endpoint,
        "weak_model": args.weak_model,
        "strong_endpoint": args.strong_endpoint,
        "strong_model": args.strong_model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "oracle_visible_to_controller_or_models": False,
        "validator_scope": "hard safety and feasibility only; not optimality or oracle equality",
    }
    config_path = run_dir / "config.json"
    if config_path.exists():
        prior = json.loads(config_path.read_text(encoding="utf-8"))
        for key in (
            "controller_config_sha256", "policy_bundle_sha256", "scenario_bank_sha256",
            "split", "inference_seed", "weak_model", "strong_model", "max_tokens",
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
    try:
        with request_path.open("a", encoding="utf-8") as request_handle, decision_path.open("a", encoding="utf-8") as decision_handle:
            async with httpx.AsyncClient(limits=limits, timeout=timeout, trust_env=False) as client:
                for scenario_index, (scenario, split, scenario_hash) in enumerate(frozen):
                    if stop_event.is_set() or (args.max_scenarios is not None and newly_completed >= args.max_scenarios):
                        stop_event.set()
                        break
                    if scenario.scenario_id in done:
                        continue
                    cell_started = time.perf_counter()
                    all_calls: list[AgentCall] = []
                    selected: ParsedDecision | None = None
                    final_calls: list[AgentCall] = []
                    organization: dict[str, Any] | None = None
                    attempt_count = 0
                    for attempt_count in range(1, args.max_cell_attempts + 1):
                        selected, final_calls, organization = await execute_a8_cell(
                            client,
                            args=args,
                            scenario=scenario,
                            bundle=bundle,
                            threshold=threshold,
                            episode_index=len(done),
                            seed=args.inference_seed + scenario_index * 100_003,
                        )
                        for call_index, call in enumerate(final_calls):
                            call.row.update({
                                "architecture": "A8",
                                "split": split,
                                "scenario_sha256": scenario_hash,
                                "controller_version": CONTROLLER_VERSION,
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
                        arm="A8",
                        selected=selected,
                        final_calls=final_calls,
                        all_calls=all_calls,
                        organization=organization,
                        decision_latency=time.perf_counter() - cell_started,
                        attempt_count=attempt_count,
                    )
                    row.update({
                        "schema_version": SCHEMA_VERSION,
                        "architecture": "A8",
                        **{
                            key: value for key, value in organization.items()
                            if key not in {"agreement", "action_diversity", "organization_switches", "weak_calls", "strong_calls", "verifier_called", "planner_valid", "worker_aggregate_action", "final_source"}
                        },
                    })
                    row["initial_correct"] = organization["initial_action"] == scenario.oracle_action
                    row["reorganization_gain"] = int(bool(row["correct"])) - int(bool(row["initial_correct"]))
                    row["unnecessary_delegation"] = bool(row["delegated"] and row["initial_correct"] and row["correct"])
                    decision_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    decision_handle.flush()
                    done.add(scenario.scenario_id)
                    newly_completed += 1
                    now = time.perf_counter()
                    status = summarize(read_jsonl(decision_path), expected=len(frozen))
                    status.update({"state": "running", "elapsed_s": now - started, "current_scenario": scenario.scenario_id, "current_route": row["route"]})
                    atomic_json(run_dir / "status.json", status)
                    if now - last_progress >= args.progress_interval:
                        print(
                            f"[A8] {len(done)}/{len(frozen)} ({len(done)/len(frozen):.1%}) "
                            f"route={row['route']} correct={int(row['correct'])} "
                            f"tokens={row['total_tokens']} latency={row['decision_latency_s']:.2f}s errors={row['request_errors']}",
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
    })
    atomic_json(run_dir / "summary.json", summary)
    atomic_json(run_dir / "status.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if complete or args.max_scenarios is not None else 130


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--policy-bundle", required=True)
    parser.add_argument("--controller-config", required=True)
    parser.add_argument("--bank", default="benchmarks/multitown-v0.2-1200/scenario-bank.jsonl")
    parser.add_argument("--split", default="test", choices=("train", "dev", "test"))
    parser.add_argument("--strong-endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--strong-model", default="qwen-game")
    parser.add_argument("--weak-endpoint", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--weak-model", default="qwen-mm-backup")
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--strong-api-key", default="local")
    parser.add_argument("--weak-api-key", default="EMPTY")
    parser.add_argument("--inference-seed", type=int, default=370_801)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--planner-max-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--request-timeout", type=float, default=900)
    parser.add_argument("--monitor-interval", type=float, default=5)
    parser.add_argument("--progress-interval", type=float, default=30)
    parser.add_argument("--max-cell-attempts", type=int, default=3)
    parser.add_argument("--max-scenarios", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    raise SystemExit(asyncio.run(run(build_parser().parse_args())))


if __name__ == "__main__":
    main()
