"""Run local MultiTown organizations on the immutable MASBench subset."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import statistics
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from .client import ModelResponse, stream_chat_completion
from .masbench_eval import aggregate_answer_lists, grade_answers, parse_answers


ARCHITECTURES = ("single", "vote", "heavy")
ROLE_PROMPTS = (
    "你是依赖链核验员。逐项计算，只在最后按指定 JSON answers 数组输出答案。",
    "你是并行子题计算员。独立求解全部问题并检查顺序，只在最后输出指定 JSON。",
    "你是反例与格式审计员。重新计算并排查遗漏、干扰信息和顺序错误，只在最后输出指定 JSON。",
)


def answer_response_format(count: int) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "masbench_answers",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "answers": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "pattern": r"^-?[0-9]{1,15}$",
                            "maxLength": 16,
                        },
                        "minItems": count,
                        "maxItems": count,
                    }
                },
                "required": ["answers"],
                "additionalProperties": False,
            },
        },
    }


def planner_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "masbench_worker_tasks",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "worker_tasks": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 3,
                        "maxItems": 3,
                    }
                },
                "required": ["worker_tasks"],
                "additionalProperties": False,
            },
        },
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_seed(base: int, sample_id: str, offset: int) -> int:
    digest = hashlib.sha256(f"{base}:{sample_id}:{offset}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def _git_revision(project_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project_root, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def load_subset(
    path: Path,
    split: str,
    max_samples: int | None,
    axes: list[str] | None = None,
    sample_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [record for record in records if record["split"] == split]
    if axes:
        records = [record for record in records if record["axis"] in axes]
    if sample_ids:
        wanted = set(sample_ids)
        records = [record for record in records if record["sample_id"] in wanted]
    records.sort(key=lambda record: record["sample_id"])
    return records[:max_samples] if max_samples is not None else records


def solver_messages(record: dict[str, Any], role: str) -> list[dict[str, str]]:
    count = len(record["answers"])
    system = (
        f"{role}\n在内部完成计算和复核，不要展示中间推理。必须给出恰好 {count} 个答案，顺序与题目一致。"
        '最终严格输出一个 JSON 对象：{"answers":["答案1","答案2"]}。'
        "answers 数组长度必须正确；不要在 JSON 后添加文字。"
    )
    return [{"role": "system", "content": system}, *record["messages"]]


async def call_model(
    client: httpx.AsyncClient,
    *,
    endpoint: str,
    model: str,
    api_key: str,
    messages: list[dict[str, str]],
    seed: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    response_format: dict[str, Any],
) -> ModelResponse:
    return await stream_chat_completion(
        client,
        endpoint=endpoint,
        model=model,
        api_key=api_key,
        messages=messages,
        seed=seed,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        response_format=response_format,
    )


def request_record(sample_id: str, role: str, response: ModelResponse, answers: list[str] | None, parser: str) -> dict[str, Any]:
    return {
        "timestamp_utc": utc_now(),
        "sample_id": sample_id,
        "role": role,
        "response": asdict(response),
        "parsed_answers": answers,
        "parser": parser,
    }


async def evaluate_sample(
    client: httpx.AsyncClient, record: dict[str, Any], args: argparse.Namespace
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    started = time.perf_counter()
    expected_count = len(record["answers"])
    requests: list[dict[str, Any]] = []

    async def invoke(
        role: str,
        endpoint: str,
        model: str,
        key: str,
        messages: list[dict[str, str]],
        offset: int,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ):
        response = await call_model(
            client,
            endpoint=endpoint,
            model=model,
            api_key=key,
            messages=messages,
            seed=stable_seed(args.seed, record["sample_id"], offset),
            max_tokens=max_tokens or args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            response_format=response_format or answer_response_format(expected_count),
        )
        answers, parser = parse_answers(response.content, expected_count)
        requests.append(request_record(record["sample_id"], role, response, answers, parser))
        return response, answers, parser

    if args.architecture == "single":
        _, selected, parser = await invoke(
            "single_solver",
            args.primary_endpoint,
            args.primary_model,
            args.primary_api_key,
            solver_messages(record, "你是严谨的数学与信息检索求解器。内部求解并自行复核。"),
            0,
        )
        agreement, diversity, final_source = 1.0, int(selected is not None), f"single_{parser}"
        organization_switches = 0
        communication_messages = 0
    elif args.architecture == "vote":
        tasks = [
            invoke(
                f"independent_solver_{index + 1}",
                args.primary_endpoint,
                args.primary_model,
                args.primary_api_key,
                solver_messages(record, "你是独立求解器。不要假设其他求解器的结论。"),
                100 + index,
            )
            for index in range(args.vote_size)
        ]
        results = await asyncio.gather(*tasks)
        selected, agreement, diversity = aggregate_answer_lists(
            [answers for _, answers, _ in results], expected_count
        )
        final_source = "deterministic_exact_majority"
        organization_switches = 1
        communication_messages = args.vote_size
    else:
        plan_messages = [
            {
                "role": "system",
                "content": (
                    "你是 Planner。分析题目的依赖、并行和干扰结构，为三个独立 Worker 生成职责。"
                    '严格输出 {"worker_tasks":["职责1","职责2","职责3"]}，不要直接给最终答案。'
                ),
            },
            *record["messages"],
        ]
        planner_response, _, _ = await invoke(
            "strong_planner",
            args.strong_endpoint,
            args.strong_model,
            args.strong_api_key,
            plan_messages,
            200,
            args.planner_max_tokens,
            planner_response_format(),
        )
        worker_tasks = []
        try:
            payload = json.loads(planner_response.content.strip())
            if isinstance(payload.get("worker_tasks"), list) and len(payload["worker_tasks"]) == 3:
                worker_tasks = [str(item)[:500] for item in payload["worker_tasks"]]
        except (json.JSONDecodeError, AttributeError):
            pass
        if not worker_tasks:
            worker_tasks = list(ROLE_PROMPTS)
        worker_calls = [
            invoke(
                f"weak_worker_{index + 1}",
                args.primary_endpoint,
                args.primary_model,
                args.primary_api_key,
                solver_messages(record, f"{ROLE_PROMPTS[index]}\nPlanner 分配：{worker_tasks[index]}"),
                300 + index,
            )
            for index in range(3)
        ]
        worker_results = await asyncio.gather(*worker_calls)
        candidate_payload = [
            {
                "worker": index + 1,
                "answers": answers,
                "raw_tail": response.content[-1200:],
            }
            for index, (response, answers, _) in enumerate(worker_results)
        ]
        verifier_messages = [
            {
                "role": "system",
                "content": (
                    "你是与 Planner 隔离的独立 Verifier。重新求解原题，再对比候选；候选可能全错。"
                    f"必须输出恰好 {expected_count} 个答案。"
                    '最终严格输出 {"answers":["答案1","答案2"]}，不要附加文字。'
                ),
            },
            *record["messages"],
            {"role": "user", "content": "候选结果：" + json.dumps(candidate_payload, ensure_ascii=False)},
        ]
        _, selected, parser = await invoke(
            "independent_strong_verifier",
            args.strong_endpoint,
            args.strong_model,
            args.strong_api_key,
            verifier_messages,
            400,
        )
        _, worker_agreement, worker_diversity = aggregate_answer_lists(
            [answers for _, answers, _ in worker_results], expected_count
        )
        agreement, diversity = worker_agreement, worker_diversity
        final_source = f"independent_strong_verifier_{parser}"
        organization_switches = 2
        communication_messages = 8

    valid = selected is not None
    correct = grade_answers(selected, record["answers"])
    total_tokens = sum(item["response"]["total_tokens"] for item in requests)
    prompt_tokens = sum(item["response"]["prompt_tokens"] for item in requests)
    completion_tokens = sum(item["response"]["completion_tokens"] for item in requests)
    errors = sum(item["response"]["error"] is not None for item in requests)
    truncated_calls = sum(item["response"]["finish_reason"] == "length" for item in requests)
    strict_json_calls = sum(item["parser"] == "json" for item in requests)
    decision = {
        "timestamp_utc": utc_now(),
        "architecture": args.label,
        "organization": args.architecture,
        "sample_id": record["sample_id"],
        "axis": record["axis"],
        "axis_value": record["axis_value"],
        "split": record["split"],
        "expected_answers": record["answers"],
        "selected_answers": selected,
        "correct": correct,
        "valid": valid,
        "agreement": agreement,
        "answer_diversity": diversity,
        "final_source": final_source,
        "organization_switches": organization_switches,
        "communication_messages": communication_messages,
        "request_count": len(requests),
        "request_errors": errors,
        "truncated_calls": truncated_calls,
        "strict_json_calls": strict_json_calls,
        "strict_json_rate": strict_json_calls / len(requests),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "decision_latency_s": time.perf_counter() - started,
    }
    return decision, requests


def summarize(decisions: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    count = len(decisions)
    correct = sum(item["correct"] for item in decisions)
    valid = sum(item["valid"] for item in decisions)
    latencies = [item["decision_latency_s"] for item in decisions]
    total_tokens = sum(item["total_tokens"] for item in decisions)
    requests = sum(item["request_count"] for item in decisions)
    return {
        "schema_version": "multitown-masbench-result-v1",
        "architecture": config["label"],
        "organization": config["architecture"],
        "split": config["split"],
        "decisions": count,
        "correct": correct,
        "accuracy": correct / count if count else 0.0,
        "valid": valid,
        "valid_rate": valid / count if count else 0.0,
        "requests": requests,
        "request_errors": sum(item["request_errors"] for item in decisions),
        "truncated_calls": sum(item.get("truncated_calls", 0) for item in decisions),
        "total_tokens": total_tokens,
        "tokens_per_decision": total_tokens / count if count else 0.0,
        "latency_mean_s": statistics.mean(latencies) if latencies else 0.0,
        "latency_p95_s": sorted(latencies)[min(len(latencies) - 1, int(len(latencies) * 0.95))] if latencies else 0.0,
        "config": config,
    }


async def run(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    decisions_path = output / "decisions.jsonl"
    requests_path = output / "requests.jsonl"
    completed: set[str] = set()
    decisions_by_sample: dict[str, dict[str, Any]] = {}
    if decisions_path.exists() and args.resume:
        existing = [json.loads(line) for line in decisions_path.read_text(encoding="utf-8").splitlines() if line]
        for item in existing:
            decisions_by_sample[item["sample_id"]] = item
        completed = {
            sample_id
            for sample_id, item in decisions_by_sample.items()
            if not (
                args.retry_invalid
                and (
                    not item.get("valid", False)
                    or item.get("request_errors", 0) > 0
                    or item.get("truncated_calls", 0) > 0
                )
            )
        }

    project_root = Path(__file__).resolve().parents[1]
    config = {
        "schema_version": "multitown-masbench-config-v1",
        "created_at_utc": utc_now(),
        "source_revision": _git_revision(project_root),
        "platform": platform.platform(),
        "subset": str(args.subset.resolve()),
        "subset_sha256": hashlib.sha256(args.subset.read_bytes()).hexdigest(),
        "split": args.split,
        "architecture": args.architecture,
        "label": args.label,
        "primary_endpoint": args.primary_endpoint,
        "primary_model": args.primary_model,
        "strong_endpoint": args.strong_endpoint if args.architecture == "heavy" else None,
        "strong_model": args.strong_model if args.architecture == "heavy" else None,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "planner_max_tokens": args.planner_max_tokens,
        "vote_size": args.vote_size if args.architecture == "vote" else None,
        "concurrency": args.concurrency,
        "max_samples": args.max_samples,
        "axes": args.axis,
        "sample_ids": args.sample_id,
        "retry_invalid": args.retry_invalid,
    }
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    records = [
        record
        for record in load_subset(
            args.subset,
            args.split,
            args.max_samples,
            axes=args.axis,
            sample_ids=args.sample_id,
        )
        if record["sample_id"] not in completed
    ]
    semaphore = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    timeout = httpx.Timeout(args.timeout_seconds)

    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        async def process(record: dict[str, Any]) -> None:
            async with semaphore:
                decision, request_rows = await evaluate_sample(client, record, args)
            async with write_lock:
                with requests_path.open("a", encoding="utf-8") as handle:
                    for row in request_rows:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                with decisions_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(decision, ensure_ascii=False) + "\n")
                decisions_by_sample[decision["sample_id"]] = decision
                done = len(decisions_by_sample)
                print(
                    f"[{done}] {decision['sample_id']} correct={decision['correct']} "
                    f"tokens={decision['total_tokens']} latency={decision['decision_latency_s']:.2f}s",
                    flush=True,
                )

        await asyncio.gather(*(process(record) for record in records))

    decisions = sorted(decisions_by_sample.values(), key=lambda item: item["sample_id"])
    summary = summarize(decisions, config)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, default=Path("benchmarks/external/masbench-v1/subset.jsonl"))
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--primary-endpoint", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--primary-model", default="qwen-mm-backup")
    parser.add_argument("--primary-api-key", default="EMPTY")
    parser.add_argument("--strong-endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--strong-model", default="qwen-game")
    parser.add_argument("--strong-api-key", default="local")
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--planner-max-tokens", type=int, default=512)
    parser.add_argument("--vote-size", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--axis", action="append", choices=("breadth", "depth", "horizon", "parallel", "robustness"))
    parser.add_argument("--sample-id", action="append")
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--retry-invalid", action="store_true")
    parser.set_defaults(resume=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.concurrency <= 0 or args.vote_size <= 0 or (args.max_samples is not None and args.max_samples <= 0):
        raise SystemExit("concurrency, vote-size and max-samples must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
