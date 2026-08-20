"""Time-bounded A0/A1/A2 benchmark runner with live monitoring."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import signal
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import psutil

from .client import ModelResponse, stream_chat_completion
from .parsing import ParsedDecision, aggregate_decisions, parse_decision
from .report import generate_run_report
from .scenarios import Scenario, build_scenario_bank


SYSTEM_PROMPT = (
    "你是赛博小镇应急决策基准中的独立决策智能体。"
    "所有事实、公式、约束和允许动作都由题目给出。精确计算，不添加外部假设。"
    "最终必须严格按照题目要求返回单个JSON对象。"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        temp_name = handle.name
    os.replace(temp_name, path)


def parse_number(value: str) -> float | None:
    value = value.strip()
    if not value or value.upper() == "N/A":
        return None
    interrupted_before_deadline = False
    try:
        return float(value)
    except ValueError:
        return None


def read_gpu_metrics() -> dict[str, float | None]:
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,power.draw,temperature.gpu,clocks.sm",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=5)
        fields = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
        values = [parse_number(field) for field in fields]
        return {
            "gpu_util_percent": values[0], "gpu_power_w": values[1],
            "gpu_temp_c": values[2], "gpu_clock_mhz": values[3],
        }
    except Exception:
        return {
            "gpu_util_percent": None, "gpu_power_w": None,
            "gpu_temp_c": None, "gpu_clock_mhz": None,
        }


def parse_prometheus(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    wanted = (
        "requests_processing", "requests_deferred", "tokens_predicted_total",
        "tokens_prompt_total", "kv_cache_usage_ratio", "prompt_tokens_seconds",
        "predicted_tokens_seconds",
    )
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        head, _, raw_value = line.rpartition(" ")
        name = head.split("{", 1)[0]
        if not any(token in name for token in wanted):
            continue
        try:
            metrics[name] = float(raw_value)
        except ValueError:
            continue
    return metrics


async def monitor_system(
    *, run_dir: Path, endpoint: str, api_key: str, started: float,
    interval: float, stop_event: asyncio.Event,
) -> None:
    metrics_url = endpoint.rstrip("/")
    if metrics_url.endswith("/v1"):
        metrics_url = metrics_url[:-3]
    metrics_url += "/metrics"
    output = run_dir / "system_metrics.jsonl"
    psutil.cpu_percent(interval=None)
    with output.open("a", encoding="utf-8") as handle:
        async with httpx.AsyncClient(timeout=5) as client:
            while not stop_event.is_set():
                now = time.perf_counter()
                memory = psutil.virtual_memory()
                row: dict[str, Any] = {
                    "timestamp_utc": utc_now(), "elapsed_s": now - started,
                    "cpu_percent": psutil.cpu_percent(interval=None),
                    "ram_used_gb": (memory.total - memory.available) / (1024 ** 3),
                    "ram_percent": memory.percent,
                    "load1": os.getloadavg()[0],
                }
                row.update(await asyncio.to_thread(read_gpu_metrics))
                try:
                    response = await client.get(metrics_url, headers={"Authorization": f"Bearer {api_key}"})
                    if response.is_success:
                        row.update(parse_prometheus(response.text))
                except Exception as exc:
                    row["metrics_error"] = f"{type(exc).__name__}: {exc}"
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=interval)
                except asyncio.TimeoutError:
                    pass


def request_row(
    *, architecture: str, episode_index: int, trial_index: int, scenario: Scenario,
    agent_index: int, inference_seed: int, response: ModelResponse,
    parsed: ParsedDecision,
) -> dict[str, Any]:
    return {
        "timestamp_utc": utc_now(), "architecture": architecture,
        "episode_index": episode_index, "trial_index": trial_index,
        "scenario_id": scenario.scenario_id, "family": scenario.family,
        "agent_index": agent_index, "inference_seed": inference_seed,
        "action": parsed.action, "confidence": parsed.confidence,
        "valid": parsed.valid, "brief_reason": parsed.brief_reason,
        "correct_individual": parsed.action == scenario.oracle_action,
        "latency_s": response.latency_s, "ttft_s": response.ttft_s,
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
        "total_tokens": response.total_tokens,
        "reasoning_chars": response.reasoning_chars,
        "error": response.error,
        "raw_content": response.content[:4096],
    }


async def run(args: argparse.Namespace) -> int:
    run_dir = Path(args.output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    scenarios = build_scenario_bank(args.world_seed, args.case_count)
    config = vars(args).copy()
    config.update({
        "started_at_utc": utc_now(),
        "method": {
            "A0": "single_qwen_4b",
            "A1": "single_qwen_35b_a3b",
            "A2": "four_qwen_4b_independent_deterministic_vote",
        }[args.architecture],
        "reasoning_mode": args.reasoning_mode,
        "deterministic_oracle": True,
        "aggregation": "majority_then_mean_confidence_then_action_order",
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
    monitor_task = asyncio.create_task(monitor_system(
        run_dir=run_dir, endpoint=args.endpoint, api_key=args.api_key,
        started=started, interval=args.monitor_interval, stop_event=stop_event,
    ))

    decisions = 0
    correct = 0
    invalid = 0
    request_errors = 0
    total_tokens = 0
    total_latency = 0.0
    last_progress = started
    last_report = started
    request_path = run_dir / "requests.jsonl"
    decision_path = run_dir / "decisions.jsonl"

    limits = httpx.Limits(max_connections=8, max_keepalive_connections=8)
    timeout = httpx.Timeout(connect=30, read=args.request_timeout, write=30, pool=30)
    try:
        with request_path.open("a", encoding="utf-8") as request_handle, decision_path.open("a", encoding="utf-8") as decision_handle:
            async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
                while time.perf_counter() < deadline and not stop_event.is_set():
                    scenario = scenarios[decisions % len(scenarios)]
                    trial_index = decisions // len(scenarios)
                    agent_count = 4 if args.architecture == "A2" else 1
                    base_inference_seed = args.inference_seed + trial_index * 10007 + (decisions % len(scenarios))
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": scenario.prompt},
                    ]
                    decision_started = time.perf_counter()
                    calls = [
                        stream_chat_completion(
                            client,
                            endpoint=args.endpoint,
                            model=args.model,
                            api_key=args.api_key,
                            messages=messages,
                            seed=base_inference_seed + agent_index * 1_000_003,
                            max_tokens=args.max_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p,
                        )
                        for agent_index in range(agent_count)
                    ]
                    responses = await asyncio.gather(*calls)
                    decision_latency = time.perf_counter() - decision_started
                    parsed = [parse_decision(response.content, scenario.allowed_actions) for response in responses]
                    for agent_index, (response, item) in enumerate(zip(responses, parsed, strict=True)):
                        row = request_row(
                            architecture=args.architecture, episode_index=decisions,
                            trial_index=trial_index, scenario=scenario, agent_index=agent_index,
                            inference_seed=base_inference_seed + agent_index * 1_000_003,
                            response=response, parsed=item,
                        )
                        request_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        request_errors += int(response.error is not None)
                    request_handle.flush()

                    if args.architecture == "A2":
                        aggregate, agreement, diversity = aggregate_decisions(parsed, scenario.allowed_actions)
                    else:
                        aggregate, agreement, diversity = parsed[0], 1.0 if parsed[0].valid else 0.0, int(parsed[0].valid)
                    is_correct = aggregate.action == scenario.oracle_action
                    decision_tokens = sum(response.total_tokens for response in responses)
                    decision_row = {
                        "timestamp_utc": utc_now(), "elapsed_s": time.perf_counter() - started,
                        "architecture": args.architecture, "episode_index": decisions,
                        "trial_index": trial_index, "scenario_id": scenario.scenario_id,
                        "family": scenario.family, "world_seed": scenario.seed,
                        "oracle_action": scenario.oracle_action, "selected_action": aggregate.action,
                        "correct": is_correct, "valid": aggregate.valid,
                        "confidence": aggregate.confidence, "agreement": agreement,
                        "action_diversity": diversity,
                        "agent_actions": [item.action for item in parsed],
                        "agent_correct": [item.action == scenario.oracle_action for item in parsed],
                        "decision_latency_s": decision_latency,
                        "mean_request_latency_s": sum(response.latency_s for response in responses) / len(responses),
                        "prompt_tokens": sum(response.prompt_tokens for response in responses),
                        "completion_tokens": sum(response.completion_tokens for response in responses),
                        "total_tokens": decision_tokens,
                        "request_errors": sum(response.error is not None for response in responses),
                    }
                    decision_handle.write(json.dumps(decision_row, ensure_ascii=False) + "\n")
                    decision_handle.flush()

                    decisions += 1
                    correct += int(is_correct)
                    invalid += int(not aggregate.valid)
                    total_tokens += decision_tokens
                    total_latency += decision_latency
                    now = time.perf_counter()
                    status = {
                        "state": "running", "architecture": args.architecture,
                        "updated_at_utc": utc_now(), "elapsed_s": now - started,
                        "remaining_s": max(0.0, deadline - now), "duration_s": args.duration_seconds,
                        "decisions": decisions, "correct": correct,
                        "accuracy": correct / decisions, "invalid_decisions": invalid,
                        "request_errors": request_errors, "total_tokens": total_tokens,
                        "tokens_per_decision": total_tokens / decisions,
                        "mean_decision_latency_s": total_latency / decisions,
                        "current_scenario": scenario.scenario_id,
                    }
                    atomic_json(run_dir / "status.json", status)
                    if now - last_progress >= args.progress_interval:
                        print(
                            f"[{args.architecture}] elapsed={(now-started)/3600:.3f}h "
                            f"remaining={max(0.0, deadline-now)/3600:.3f}h decisions={decisions} "
                            f"accuracy={correct/decisions:.4f} tokens={total_tokens} "
                            f"mean_latency={total_latency/decisions:.2f}s errors={request_errors}",
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
    final_state = "stopped" if interrupted_before_deadline else "complete"
    final_status = {
        "state": final_state, "architecture": args.architecture,
        "updated_at_utc": utc_now(), "elapsed_s": ended - started,
        "remaining_s": max(0.0, deadline - ended), "duration_s": args.duration_seconds,
        "decisions": decisions, "correct": correct,
        "accuracy": correct / decisions if decisions else 0.0,
        "invalid_decisions": invalid, "request_errors": request_errors,
        "total_tokens": total_tokens,
        "tokens_per_decision": total_tokens / decisions if decisions else 0.0,
        "mean_decision_latency_s": total_latency / decisions if decisions else math.nan,
        "finished_at_utc": utc_now(),
    }
    atomic_json(run_dir / "status.json", final_status)
    generate_run_report(run_dir)
    print(json.dumps(final_status, ensure_ascii=False), flush=True)
    return 0 if final_state == "complete" else 130


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=["A0", "A1", "A2"], required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--duration-seconds", type=float, default=14_400)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--case-count", type=int, default=120)
    parser.add_argument("--world-seed", type=int, default=20_260_807)
    parser.add_argument("--inference-seed", type=int, default=70_801)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--request-timeout", type=float, default=900)
    parser.add_argument("--monitor-interval", type=float, default=5)
    parser.add_argument("--report-interval", type=float, default=300)
    parser.add_argument("--progress-interval", type=float, default=60)
    parser.add_argument("--reasoning-mode", default="off")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
