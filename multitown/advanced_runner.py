"""Time-bounded A3/A4/A5 organization benchmark with dual local models."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import psutil

from .client import ModelResponse, stream_chat_completion
from .parsing import ParsedDecision, aggregate_decisions, parse_decision, strict_json_compliant
from .report import generate_run_report
from .runner import atomic_json, parse_prometheus, read_gpu_metrics, utc_now
from .scenarios import Scenario, build_scenario_bank


DECISION_SYSTEM_PROMPT = (
    "你是赛博小镇应急决策基准中的决策智能体。"
    "所有事实、公式、约束和允许动作都由题目给出。精确计算，不添加外部假设。"
    "不得猜测或请求标准答案。最终严格返回题目指定的单个JSON对象。"
)
PLANNER_SYSTEM_PROMPT = (
    "你是赛博小镇的强模型规划者。只根据题目制定可执行分工，不能访问标准答案。"
    "输出简洁的核验计划，分别给三个弱模型工作者安排互补任务。"
)
INTEGRATOR_SYSTEM_PROMPT = (
    "你是赛博小镇的强模型负责人。根据原题、计划和工作者候选独立复算并作最终决策。"
    "不能访问标准答案，不能因多数意见而放弃约束核验。最终严格返回题目指定的单个JSON对象。"
)
VERIFIER_SYSTEM_PROMPT = (
    "你是隔离上下文中的独立强模型验证者。根据原题重新计算，再审查候选意见。"
    "不能访问标准答案，不能默认规划者或多数意见正确。最终严格返回题目指定的单个JSON对象。"
)

WEAK_SOLO_FAMILIES = ("dependency_recovery", "supply_route", "fault_recovery")
HIGH_RISK_FAMILIES = ("incident_dispatch", "evidence_fusion", "resource_allocation")
CRITICAL_FAMILIES = ("incident_dispatch", "evidence_fusion")
WORKER_ROLES = (
    "约束与可行性核验员",
    "公式计算与候选排序员",
    "反例、边界与输出格式审计员",
)


@dataclass
class AgentCall:
    response: ModelResponse
    parsed: ParsedDecision
    row: dict[str, Any]


def _metrics_url(endpoint: str) -> str:
    value = endpoint.rstrip("/")
    if value.endswith("/v1"):
        value = value[:-3]
    return value + "/metrics"


async def monitor_dual_system(
    *, run_dir: Path, endpoints: dict[str, str], api_key: str, started: float,
    interval: float, stop_event: asyncio.Event,
    api_keys: dict[str, str] | None = None,
) -> None:
    """Record host/GPU telemetry plus independently prefixed llama.cpp metrics."""
    output = run_dir / "system_metrics.jsonl"
    gpu_failure_path = run_dir / "gpu-health-error.json"
    missing_gpu_samples = 0
    psutil.cpu_percent(interval=None)
    with output.open("a", encoding="utf-8") as handle:
        async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
            while not stop_event.is_set():
                now = time.perf_counter()
                memory = psutil.virtual_memory()
                row: dict[str, Any] = {
                    "timestamp_utc": utc_now(),
                    "elapsed_s": now - started,
                    "cpu_percent": psutil.cpu_percent(interval=None),
                    "ram_used_gb": (memory.total - memory.available) / (1024 ** 3),
                    "ram_percent": memory.percent,
                    "load1": os.getloadavg()[0],
                }
                gpu_metrics = await asyncio.to_thread(read_gpu_metrics)
                row.update(gpu_metrics)
                if gpu_metrics.get("gpu_util_percent") is None or gpu_metrics.get("gpu_power_w") is None:
                    missing_gpu_samples += 1
                else:
                    missing_gpu_samples = 0
                row["gpu_missing_consecutive_samples"] = missing_gpu_samples
                for tier, endpoint in endpoints.items():
                    try:
                        tier_api_key = (api_keys or {}).get(tier, api_key)
                        response = await client.get(
                            _metrics_url(endpoint),
                            headers={"Authorization": f"Bearer {tier_api_key}"},
                        )
                        if response.is_success:
                            for name, value in parse_prometheus(response.text).items():
                                safe_name = name.replace(":", "_").replace(".", "_")
                                row[f"{tier}_{safe_name}"] = value
                        else:
                            row[f"{tier}_metrics_error"] = f"HTTP {response.status_code}"
                    except Exception as exc:
                        row[f"{tier}_metrics_error"] = f"{type(exc).__name__}: {exc}"
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                if missing_gpu_samples >= 3:
                    atomic_json(gpu_failure_path, {
                        "timestamp_utc": utc_now(),
                        "reason": "GPU utilization or power telemetry unavailable for three consecutive samples",
                        "last_gpu_metrics": gpu_metrics,
                    })
                    stop_event.set()
                    break
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=interval)
                except asyncio.TimeoutError:
                    pass


def _decision_instruction(scenario: Scenario) -> str:
    return (
        f"允许动作严格限于：{json.dumps(scenario.allowed_actions, ensure_ascii=False)}。\n"
        '最终只返回：{"action":"允许动作之一","confidence":0到1之间的数,'
        '"brief_reason":"不超过40个汉字"}'
    )


def _candidate_text(calls: list[AgentCall]) -> str:
    rows = []
    for index, call in enumerate(calls):
        rows.append({
            "candidate_index": index,
            "role": call.row["role"],
            "action": call.parsed.action,
            "confidence": call.parsed.confidence,
            "valid": call.parsed.valid,
            "brief_reason": call.parsed.brief_reason,
            "raw_output": call.response.content[:1200],
        })
    return json.dumps(rows, ensure_ascii=False)


async def _call_agent(
    client: httpx.AsyncClient,
    *, args: argparse.Namespace, architecture: str, episode_index: int,
    trial_index: int, scenario: Scenario, phase: str, role: str,
    model_tier: str, endpoint: str, model: str, seed: int,
    messages: list[dict[str, str]], expects_decision: bool,
    max_tokens: int | None = None,
) -> AgentCall:
    response = await stream_chat_completion(
        client,
        endpoint=endpoint,
        model=model,
        api_key=(getattr(args, f"{model_tier}_api_key", None) or args.api_key),
        messages=messages,
        seed=seed,
        max_tokens=max_tokens if max_tokens is not None else args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    parsed = (
        parse_decision(response.content, scenario.allowed_actions)
        if expects_decision
        else ParsedDecision(None, 0.0, "not_a_decision_phase", True)
    )
    row: dict[str, Any] = {
        "timestamp_utc": utc_now(),
        "architecture": architecture,
        "episode_index": episode_index,
        "trial_index": trial_index,
        "scenario_id": scenario.scenario_id,
        "family": scenario.family,
        "phase": phase,
        "role": role,
        "model_tier": model_tier,
        "model": model,
        "endpoint": endpoint,
        "inference_seed": seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": max_tokens if max_tokens is not None else args.max_tokens,
        "expects_decision": expects_decision,
        "action": parsed.action,
        "confidence": parsed.confidence,
        "valid": parsed.valid,
        "strict_json_compliant": (
            strict_json_compliant(response.content, scenario.allowed_actions)
            if expects_decision else None
        ),
        "brief_reason": parsed.brief_reason,
        "correct_individual": (
            parsed.action == scenario.oracle_action if expects_decision else None
        ),
        "latency_s": response.latency_s,
        "ttft_s": response.ttft_s,
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
        "total_tokens": response.total_tokens,
        "reasoning_chars": response.reasoning_chars,
        "error": response.error,
        "messages": messages,
        "raw_content": response.content,
    }
    return AgentCall(response=response, parsed=parsed, row=row)


def _final_or_fallback(
    final_call: AgentCall, fallback: ParsedDecision,
) -> tuple[ParsedDecision, str]:
    if final_call.parsed.valid:
        return final_call.parsed, str(final_call.row["role"])
    return fallback, "deterministic_worker_aggregate_fallback"


async def _run_a3(
    client: httpx.AsyncClient, *, args: argparse.Namespace, scenario: Scenario,
    episode_index: int, trial_index: int, seed: int, architecture: str = "A3",
) -> tuple[ParsedDecision, list[AgentCall], dict[str, Any]]:
    plan = await _call_agent(
        client, args=args, architecture=architecture, episode_index=episode_index,
        trial_index=trial_index, scenario=scenario, phase="planning", role="strong_leader_planner",
        model_tier="strong", endpoint=args.strong_endpoint, model=args.strong_model,
        seed=seed + 101,
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": scenario.prompt + "\n请给三个工作者制定互补的核验分工；此阶段不要假装拥有标准答案。"},
        ],
        expects_decision=False, max_tokens=args.planner_max_tokens,
    )
    worker_calls = await asyncio.gather(*[
        _call_agent(
            client, args=args, architecture=architecture, episode_index=episode_index,
            trial_index=trial_index, scenario=scenario, phase="worker_execution",
            role=f"weak_worker_{index + 1}_{role}", model_tier="weak",
            endpoint=args.weak_endpoint, model=args.weak_model, seed=seed + 1001 + index,
            messages=[
                {"role": "system", "content": DECISION_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    scenario.prompt + f"\n\n负责人计划：\n{plan.response.content[:3000]}"
                    f"\n\n你的特定职责：{role}。请独立核验后决策。\n{_decision_instruction(scenario)}"
                )},
            ],
            expects_decision=True,
        )
        for index, role in enumerate(WORKER_ROLES)
    ])
    worker_aggregate, agreement, diversity = aggregate_decisions(
        [call.parsed for call in worker_calls], scenario.allowed_actions
    )
    integration = await _call_agent(
        client, args=args, architecture=architecture, episode_index=episode_index,
        trial_index=trial_index, scenario=scenario, phase="integration", role="strong_leader_integrator",
        model_tier="strong", endpoint=args.strong_endpoint, model=args.strong_model,
        seed=seed + 2001,
        messages=[
            {"role": "system", "content": INTEGRATOR_SYSTEM_PROMPT},
            {"role": "user", "content": (
                scenario.prompt + f"\n\n你先前的计划：\n{plan.response.content[:3000]}"
                f"\n\n工作者候选：\n{_candidate_text(worker_calls)}"
                f"\n\n请复算、解决冲突并最终决策。\n{_decision_instruction(scenario)}"
            )},
        ],
        expects_decision=True,
    )
    selected, final_source = _final_or_fallback(integration, worker_aggregate)
    calls = [plan, *worker_calls, integration]
    return selected, calls, {
        "route": "leader_workers_integration",
        "final_source": final_source,
        "agreement": agreement,
        "action_diversity": diversity,
        "organization_switches": 2,
        "weak_calls": 3,
        "strong_calls": 2,
        "verifier_called": False,
        "planner_valid": plan.response.error is None,
        "worker_aggregate_action": worker_aggregate.action,
    }


async def _run_a4(
    client: httpx.AsyncClient, *, args: argparse.Namespace, scenario: Scenario,
    episode_index: int, trial_index: int, seed: int, architecture: str = "A4",
) -> tuple[ParsedDecision, list[AgentCall], dict[str, Any]]:
    plan = await _call_agent(
        client, args=args, architecture=architecture, episode_index=episode_index,
        trial_index=trial_index, scenario=scenario, phase="planning", role="strong_planner",
        model_tier="strong", endpoint=args.strong_endpoint, model=args.strong_model,
        seed=seed + 101,
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": scenario.prompt + "\n请给三个工作者制定带验收条件的任务契约；不要输出标准答案。"},
        ],
        expects_decision=False, max_tokens=args.planner_max_tokens,
    )
    worker_calls = await asyncio.gather(*[
        _call_agent(
            client, args=args, architecture=architecture, episode_index=episode_index,
            trial_index=trial_index, scenario=scenario, phase="worker_execution",
            role=f"weak_worker_{index + 1}_{role}", model_tier="weak",
            endpoint=args.weak_endpoint, model=args.weak_model, seed=seed + 1001 + index,
            messages=[
                {"role": "system", "content": DECISION_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    scenario.prompt + f"\n\n规划者的任务契约：\n{plan.response.content[:3000]}"
                    f"\n\n你的特定职责：{role}。独立完成并提交候选。\n{_decision_instruction(scenario)}"
                )},
            ],
            expects_decision=True,
        )
        for index, role in enumerate(WORKER_ROLES)
    ])
    worker_aggregate, agreement, diversity = aggregate_decisions(
        [call.parsed for call in worker_calls], scenario.allowed_actions
    )
    verifier = await _call_agent(
        client, args=args, architecture=architecture, episode_index=episode_index,
        trial_index=trial_index, scenario=scenario, phase="independent_verification",
        role="independent_strong_verifier", model_tier="strong",
        endpoint=args.strong_endpoint, model=args.strong_model, seed=seed + 3001,
        messages=[
            {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": (
                scenario.prompt + f"\n\n待审候选（不包含规划上下文）：\n{_candidate_text(worker_calls)}"
                f"\n\n请先独立复算，再核查候选并决策。\n{_decision_instruction(scenario)}"
            )},
        ],
        expects_decision=True,
    )
    selected, final_source = _final_or_fallback(verifier, worker_aggregate)
    calls = [plan, *worker_calls, verifier]
    return selected, calls, {
        "route": "planner_workers_independent_verifier",
        "final_source": final_source,
        "agreement": agreement,
        "action_diversity": diversity,
        "organization_switches": 2,
        "weak_calls": 3,
        "strong_calls": 2,
        "verifier_called": True,
        "planner_valid": plan.response.error is None,
        "worker_aggregate_action": worker_aggregate.action,
    }


async def _run_a5(
    client: httpx.AsyncClient, *, args: argparse.Namespace, scenario: Scenario,
    episode_index: int, trial_index: int, seed: int, architecture: str = "A5",
) -> tuple[ParsedDecision, list[AgentCall], dict[str, Any]]:
    solo = await _call_agent(
        client, args=args, architecture=architecture, episode_index=episode_index,
        trial_index=trial_index, scenario=scenario, phase="initial_weak_attempt",
        role="weak_solo_solver", model_tier="weak", endpoint=args.weak_endpoint,
        model=args.weak_model, seed=seed + 1001,
        messages=[
            {"role": "system", "content": DECISION_SYSTEM_PROMPT},
            {"role": "user", "content": scenario.prompt + "\n" + _decision_instruction(scenario)},
        ],
        expects_decision=True,
    )
    calls = [solo]
    if (
        scenario.family in WEAK_SOLO_FAMILIES
        and solo.parsed.valid
    ):
        return solo.parsed, calls, {
            "route": "solo_weak",
            "final_source": "weak_solo_solver",
            "agreement": 1.0,
            "action_diversity": 1,
            "organization_switches": 0,
            "weak_calls": 1,
            "strong_calls": 0,
            "verifier_called": False,
            "planner_valid": None,
            "worker_aggregate_action": solo.parsed.action,
        }

    extra_workers = await asyncio.gather(*[
        _call_agent(
            client, args=args, architecture=architecture, episode_index=episode_index,
            trial_index=trial_index, scenario=scenario, phase="weak_team_expansion",
            role=f"weak_dynamic_worker_{index + 2}_{role}", model_tier="weak",
            endpoint=args.weak_endpoint, model=args.weak_model, seed=seed + 1002 + index,
            messages=[
                {"role": "system", "content": DECISION_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    scenario.prompt + f"\n\n动态组织给你的职责：{role}。"
                    f"请不受其他候选影响地独立求解。\n{_decision_instruction(scenario)}"
                )},
            ],
            expects_decision=True,
        )
        for index, role in enumerate(WORKER_ROLES)
    ])
    calls.extend(extra_workers)
    weak_calls = [solo, *extra_workers]
    weak_aggregate, agreement, diversity = aggregate_decisions(
        [call.parsed for call in weak_calls], scenario.allowed_actions
    )
    if (
        weak_aggregate.valid
        and agreement >= args.a5_vote_agreement
        and scenario.family not in CRITICAL_FAMILIES
    ):
        return weak_aggregate, calls, {
            "route": "weak_vote",
            "final_source": "deterministic_weak_vote",
            "agreement": agreement,
            "action_diversity": diversity,
            "organization_switches": 1,
            "weak_calls": 4,
            "strong_calls": 0,
            "verifier_called": False,
            "planner_valid": None,
            "worker_aggregate_action": weak_aggregate.action,
        }

    integrator = await _call_agent(
        client, args=args, architecture=architecture, episode_index=episode_index,
        trial_index=trial_index, scenario=scenario, phase="strong_escalation",
        role="dynamic_strong_integrator", model_tier="strong",
        endpoint=args.strong_endpoint, model=args.strong_model, seed=seed + 2001,
        messages=[
            {"role": "system", "content": INTEGRATOR_SYSTEM_PROMPT},
            {"role": "user", "content": (
                scenario.prompt + f"\n\n弱模型候选：\n{_candidate_text(weak_calls)}"
                f"\n\n因低共识或高风险升级给你。请独立复算后决策。\n{_decision_instruction(scenario)}"
            )},
        ],
        expects_decision=True,
    )
    calls.append(integrator)
    strong_selected, strong_source = _final_or_fallback(integrator, weak_aggregate)
    needs_verifier = (
        scenario.family in CRITICAL_FAMILIES
        or not integrator.parsed.valid
        or (
            weak_aggregate.valid
            and integrator.parsed.valid
            and integrator.parsed.action != weak_aggregate.action
        )
    )
    if not needs_verifier:
        return strong_selected, calls, {
            "route": "strong_escalation",
            "final_source": strong_source,
            "agreement": agreement,
            "action_diversity": diversity,
            "organization_switches": 2,
            "weak_calls": 4,
            "strong_calls": 1,
            "verifier_called": False,
            "planner_valid": None,
            "worker_aggregate_action": weak_aggregate.action,
        }

    verifier = await _call_agent(
        client, args=args, architecture=architecture, episode_index=episode_index,
        trial_index=trial_index, scenario=scenario, phase="independent_verification",
        role="dynamic_independent_strong_verifier", model_tier="strong",
        endpoint=args.strong_endpoint, model=args.strong_model, seed=seed + 3001,
        messages=[
            {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": (
                scenario.prompt + "\n\n需要独立核验的候选：\n"
                + json.dumps({
                    "weak_aggregate": weak_aggregate.action,
                    "strong_candidate": integrator.parsed.action,
                    "strong_confidence": integrator.parsed.confidence,
                }, ensure_ascii=False)
                + f"\n\n请独立复算并最终裁决。\n{_decision_instruction(scenario)}"
            )},
        ],
        expects_decision=True,
    )
    calls.append(verifier)
    if verifier.parsed.valid:
        selected, final_source = verifier.parsed, "dynamic_independent_strong_verifier"
    else:
        selected, final_source = strong_selected, strong_source
    return selected, calls, {
        "route": "strong_plus_verifier",
        "final_source": final_source,
        "agreement": agreement,
        "action_diversity": diversity,
        "organization_switches": 3,
        "weak_calls": 4,
        "strong_calls": 2,
        "verifier_called": True,
        "planner_valid": None,
        "worker_aggregate_action": weak_aggregate.action,
    }


async def run(args: argparse.Namespace) -> int:
    run_dir = Path(args.output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    scenarios = build_scenario_bank(args.world_seed, args.case_count)
    method = {
        "A3": "one_strong_leader_three_weak_workers_then_strong_integration",
        "A4": "strong_planner_three_weak_workers_independent_strong_verifier",
        "A5": "dynamic_weak_team_strong_escalation_and_optional_verification",
    }[args.architecture]
    config = vars(args).copy()
    config.update({
        "started_at_utc": utc_now(),
        "method": method,
        "timing_scope": "after_both_models_loaded_and_warmed_until_duration_deadline",
        "deterministic_oracle": True,
        "oracle_visible_to_router_or_models": False,
        "aggregation": "majority_then_mean_confidence_then_action_order",
        "worker_roles": WORKER_ROLES,
        "prompts": {
            "decision_system": DECISION_SYSTEM_PROMPT,
            "planner_system": PLANNER_SYSTEM_PROMPT,
            "integrator_system": INTEGRATOR_SYSTEM_PROMPT,
            "verifier_system": VERIFIER_SYSTEM_PROMPT,
        },
        "a5_policy": {
            "weak_solo_families": WEAK_SOLO_FAMILIES,
            "high_risk_families": HIGH_RISK_FAMILIES,
            "critical_families": CRITICAL_FAMILIES,
            "solo_weak_stop": "historically weak-solo-suitable family and parse-valid output",
            "weak_vote_stop": "valid unanimous weak vote and not a critical family",
            "verify": "critical family, invalid strong result, or strong/weak disagreement",
            "self_reported_confidence_used_for_routing": False,
            "prior_a0_a2_finding": "self-reported confidence is miscalibrated; unanimity is more informative than 3:1",
        },
    })
    atomic_json(run_dir / "config.json", config)
    atomic_json(run_dir / "scenario_bank.json", {"scenarios": [item.to_dict() for item in scenarios]})

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
        with request_path.open("a", encoding="utf-8") as request_handle, decision_path.open("a", encoding="utf-8") as decision_handle:
            async with httpx.AsyncClient(limits=limits, timeout=timeout, trust_env=False) as client:
                while time.perf_counter() < deadline and not stop_event.is_set():
                    scenario = scenarios[decisions % len(scenarios)]
                    trial_index = decisions // len(scenarios)
                    scenario_index = decisions % len(scenarios)
                    base_seed = args.inference_seed + trial_index * 10_000_019 + scenario_index * 100_003
                    decision_started = time.perf_counter()
                    runner = {"A3": _run_a3, "A4": _run_a4, "A5": _run_a5}[args.architecture]
                    selected, calls, organization = await runner(
                        client, args=args, scenario=scenario, episode_index=decisions,
                        trial_index=trial_index, seed=base_seed,
                    )
                    decision_latency = time.perf_counter() - decision_started
                    for call_index, call in enumerate(calls):
                        call.row["request_index_in_episode"] = call_index
                        request_handle.write(json.dumps(call.row, ensure_ascii=False) + "\n")
                    request_handle.flush()

                    is_correct = selected.action == scenario.oracle_action
                    prompt_tokens = sum(call.response.prompt_tokens for call in calls)
                    completion_tokens = sum(call.response.completion_tokens for call in calls)
                    decision_tokens = sum(call.response.total_tokens for call in calls)
                    weak_request_tokens = sum(
                        call.response.total_tokens for call in calls if call.row["model_tier"] == "weak"
                    )
                    strong_request_tokens = decision_tokens - weak_request_tokens
                    decision_errors = sum(call.response.error is not None for call in calls)
                    candidate_calls = [call for call in calls if call.row["expects_decision"]]
                    row: dict[str, Any] = {
                        "timestamp_utc": utc_now(),
                        "elapsed_s": time.perf_counter() - started,
                        "architecture": args.architecture,
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
                        "final_source": organization["final_source"],
                        "organization_switches": organization["organization_switches"],
                        "weak_calls": organization["weak_calls"],
                        "strong_calls": organization["strong_calls"],
                        "verifier_called": organization["verifier_called"],
                        "planner_valid": organization["planner_valid"],
                        "worker_aggregate_action": organization["worker_aggregate_action"],
                        "candidate_actions": [call.parsed.action for call in candidate_calls],
                        "candidate_correct": [call.parsed.action == scenario.oracle_action for call in candidate_calls],
                        "candidate_roles": [call.row["role"] for call in candidate_calls],
                        "strict_json_calls": sum(bool(call.row["strict_json_compliant"]) for call in candidate_calls),
                        "strict_json_rate": (
                            sum(bool(call.row["strict_json_compliant"]) for call in candidate_calls) / len(candidate_calls)
                            if candidate_calls else None
                        ),
                        "request_count": len(calls),
                        "communication_messages": len(calls) * 2,
                        "decision_latency_s": decision_latency,
                        "mean_request_latency_s": sum(call.response.latency_s for call in calls) / len(calls),
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
                        "architecture": args.architecture,
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
                        "current_route": organization["route"],
                    }
                    atomic_json(run_dir / "status.json", status)
                    if now - last_progress >= args.progress_interval:
                        print(
                            f"[{args.architecture}] elapsed={(now-started)/3600:.3f}h "
                            f"remaining={max(0.0, deadline-now)/3600:.3f}h decisions={decisions} "
                            f"accuracy={correct/decisions:.4f} route={organization['route']} "
                            f"tokens={total_tokens} mean_latency={total_latency/decisions:.2f}s "
                            f"errors={request_errors}",
                            flush=True,
                        )
                        last_progress = now
                    if now - last_report >= args.report_interval:
                        try:
                            await asyncio.to_thread(generate_run_report, run_dir)
                        except Exception as exc:
                            print(f"[{args.architecture}] interim report failed: {exc}", flush=True)
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
    final_state = "failed_gpu_health" if gpu_failure_path.exists() else "stopped" if interrupted_before_deadline else "complete"
    final_status = {
        "state": final_state,
        "architecture": args.architecture,
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
    parser.add_argument("--architecture", choices=["A3", "A4", "A5"], required=True)
    parser.add_argument("--strong-endpoint", required=True)
    parser.add_argument("--strong-model", required=True)
    parser.add_argument("--weak-endpoint", required=True)
    parser.add_argument("--weak-model", required=True)
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--duration-seconds", type=float, default=14_400)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--case-count", type=int, default=120)
    parser.add_argument("--world-seed", type=int, default=20_260_807)
    parser.add_argument("--inference-seed", type=int, default=70_801)
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
