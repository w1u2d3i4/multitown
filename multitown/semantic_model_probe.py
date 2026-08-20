"""Collect a cross-factor real-model probe over A13 semantic task views."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from .client import ModelResponse, stream_chat_completion
from .masbench_routing import git_state, utc_now, write_json
from .semantic_tasks import (
    CHOICE_PROMPT_PROTOCOL_VERSION,
    FAMILIES,
    OPTION_LABELS,
    SemanticTask,
    parse_decision,
    read_bank,
    render_worker_messages,
    role_context,
)


SCHEMA_VERSION = "multitown-semantic-model-outcome-v2"
SUPPORTED_SCHEMA_VERSIONS = {"multitown-semantic-model-outcome-v1", SCHEMA_VERSION}
CONFIG_VERSION = "multitown-semantic-model-probe-config-v2"
CELLS = (
    "qwen4b_weak_context",
    "qwen4b_strong_context",
    "qwen4b_no_context",
    "qwen4b_union_context",
    "qwen35b_weak_context",
    "qwen35b_strong_context",
    "qwen35b_no_context",
    "qwen35b_union_context",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def task_sha256(task: SemanticTask) -> str:
    return canonical_sha256(task.to_dict())


def request_payload(
    task: SemanticTask, cell: str, endpoint: "ProbeEndpoint", seed: int,
) -> dict[str, Any]:
    _, context_role = _cell_parts(cell)
    return {
        "endpoint": endpoint.endpoint,
        "model": endpoint.model,
        "messages": render_probe_messages(task, context_role),
        "seed": seed,
        "max_tokens": 48,
        "temperature": 0.0,
        "top_p": 1.0,
        "response_format": None,
    }


@dataclass(frozen=True)
class ProbeEndpoint:
    model_key: str
    endpoint: str
    model: str


@dataclass(frozen=True)
class SemanticMeasuredCall:
    cell: str
    model_key: str
    context_role: str
    model: str
    request_sha256: str
    response_content: str
    parsed_option: int | None
    abstained: bool
    valid: bool
    correct: bool
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_s: float
    ttft_s: float | None
    finish_reason: str | None
    error: str | None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SemanticMeasuredCall":
        return cls(
            cell=str(value["cell"]), model_key=str(value["model_key"]),
            context_role=str(value["context_role"]), model=str(value["model"]),
            request_sha256=str(value["request_sha256"]),
            response_content=str(value["response_content"]),
            parsed_option=None if value.get("parsed_option") is None else int(value["parsed_option"]),
            abstained=bool(value.get("abstained", False)),
            valid=bool(value["valid"]), correct=bool(value["correct"]),
            prompt_tokens=int(value["prompt_tokens"]),
            completion_tokens=int(value["completion_tokens"]),
            total_tokens=int(value["total_tokens"]), latency_s=float(value["latency_s"]),
            ttft_s=None if value.get("ttft_s") is None else float(value["ttft_s"]),
            finish_reason=value.get("finish_reason"), error=value.get("error"),
        )


@dataclass(frozen=True)
class SemanticProbeOutcome:
    task_id: str
    split: str
    family: str
    task_sha256: str
    correct_option: int
    calls: dict[str, SemanticMeasuredCall]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SemanticProbeOutcome":
        if value.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(f"unsupported semantic probe schema: {value.get('schema_version')}")
        calls = {name: SemanticMeasuredCall.from_dict(call) for name, call in value["calls"].items()}
        if set(calls) != set(CELLS):
            raise ValueError(f"probe outcome must contain cells {CELLS}")
        return cls(
            task_id=str(value["task_id"]), split=str(value["split"]),
            family=str(value["family"]), task_sha256=str(value["task_sha256"]),
            correct_option=int(value["evaluator"]["correct_option"]), calls=calls,
        )


def _cell_parts(cell: str) -> tuple[str, str]:
    if cell.startswith("qwen4b_"):
        model_key = "qwen4b"
    elif cell.startswith("qwen35b_"):
        model_key = "qwen35b"
    else:
        raise ValueError(cell)
    suffix = cell.split("_", 1)[1]
    context_role = {
        "weak_context": "weak",
        "strong_context": "strong",
        "no_context": "none",
        "union_context": "union",
    }.get(suffix)
    if context_role is None:
        raise ValueError(cell)
    return model_key, context_role


def render_probe_messages(task: SemanticTask, context_role: str) -> list[dict[str, str]]:
    if context_role in {"weak", "strong"}:
        return render_worker_messages(task, context_role)
    option_text = "\n".join(
        f"{label}. {option}" for label, option in zip(OPTION_LABELS, task.options, strict=True)
    )
    if context_role == "none":
        context = "No role feed is available. Do not invent current operational facts."
    elif context_role == "union":
        context = (
            f"Local feed:\n{role_context(task, 'weak')}\n\n"
            f"Central feed:\n{role_context(task, 'strong')}"
        )
    else:
        raise ValueError(context_role)
    return [
        {
            "role": "system",
            "content": (
                "Select the one compliant action using only the brief, options, and provided "
                "feeds. Treat LIVE AUTHORITATIVE data as current and stale data as non-current. "
                'If a live feed is available, return its selected label as {"option":"A"}. If no '
                'live feed is available, return {"option":"ABSTAIN"}. Return only that JSON object.'
            ),
        },
        {
            "role": "user",
            "content": (
                f"Task brief:\n{task.public_brief}\n\nOptions:\n{option_text}\n\n"
                f"Available feeds:\n{context}"
            ),
        },
    ]


async def call_cell(
    client: httpx.AsyncClient,
    task: SemanticTask,
    cell: str,
    endpoint: ProbeEndpoint,
    *,
    seed: int,
) -> SemanticMeasuredCall:
    model_key, context_role = _cell_parts(cell)
    if endpoint.model_key != model_key:
        raise ValueError(f"endpoint mismatch for {cell}")
    request = request_payload(task, cell, endpoint, seed)
    messages = request["messages"]
    response: ModelResponse = await stream_chat_completion(
        client, endpoint=endpoint.endpoint, model=endpoint.model,
        api_key="local-no-secret", messages=messages, seed=seed,
        max_tokens=48, temperature=0.0, top_p=1.0, response_format=None,
    )
    parsed, abstained, parse_error = parse_decision(response.content)
    errors = [item for item in (response.error, parse_error) if item]
    valid = response.error is None and parse_error is None and response.total_tokens > 0
    if response.error is None and response.total_tokens <= 0:
        errors.append("ValueError: missing positive token usage")
    return SemanticMeasuredCall(
        cell=cell, model_key=model_key, context_role=context_role, model=endpoint.model,
        request_sha256=canonical_sha256(request), response_content=response.content,
        parsed_option=parsed, abstained=abstained, valid=valid,
        correct=valid and not abstained and parsed == task.correct_option,
        prompt_tokens=response.prompt_tokens, completion_tokens=response.completion_tokens,
        total_tokens=response.total_tokens, latency_s=response.latency_s,
        ttft_s=response.ttft_s, finish_reason=response.finish_reason,
        error="; ".join(errors) or None,
    )


async def collect_task(
    client: httpx.AsyncClient,
    task: SemanticTask,
    endpoints: dict[str, ProbeEndpoint],
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    async def limited(cell: str) -> SemanticMeasuredCall:
        model_key, _ = _cell_parts(cell)
        async with semaphore:
            return await call_cell(
                client, task, cell, endpoints[model_key], seed=task.seed * 31 + 17,
            )

    measured = await asyncio.gather(
        *(limited(cell) for cell in CELLS)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": task.task_id,
        "split": task.split,
        "family": task.family,
        "task_sha256": task_sha256(task),
        "evaluator": {"correct_option": task.correct_option},
        "calls": {call.cell: asdict(call) for call in measured},
    }


def read_outcomes(
    path: Path, *, repair_truncated_tail: bool = False,
) -> list[SemanticProbeOutcome]:
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    rows = []
    valid_prefix_bytes = 0
    for index, line in enumerate(lines):
        if not line.strip():
            valid_prefix_bytes += len(line)
            continue
        try:
            rows.append(SemanticProbeOutcome.from_dict(json.loads(line)))
            valid_prefix_bytes += len(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            is_truncated_tail = index == len(lines) - 1 and not raw.endswith(b"\n")
            if not (repair_truncated_tail and is_truncated_tail):
                raise ValueError(f"invalid outcome line {index + 1}: {exc}") from exc
            tail = raw[valid_prefix_bytes:]
            quarantine = path.with_name(path.name + ".truncated-tail")
            if quarantine.exists():
                raise ValueError(f"truncated-tail quarantine already exists: {quarantine}")
            quarantine.write_bytes(tail)
            path.write_bytes(raw[:valid_prefix_bytes])
            break
    if not rows:
        raise ValueError(f"no semantic probe outcomes in {path}")
    if len({row.task_id for row in rows}) != len(rows):
        raise ValueError("duplicate task ids in semantic probe outcomes")
    return rows


def validate_coverage(
    tasks: list[SemanticTask], outcomes: list[SemanticProbeOutcome],
    endpoints: dict[str, ProbeEndpoint] | None = None,
) -> None:
    expected = {task.task_id: task for task in tasks}
    actual = {row.task_id: row for row in outcomes}
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))[:5]
        extra = sorted(set(actual) - set(expected))[:5]
        raise ValueError(f"outcome coverage mismatch; missing={missing}; extra={extra}")
    for task_id, task in expected.items():
        row = actual[task_id]
        if row.task_sha256 != task_sha256(task):
            raise ValueError(f"task hash mismatch: {task_id}")
        if row.correct_option != task.correct_option or row.family != task.family or row.split != task.split:
            raise ValueError(f"task evaluator metadata mismatch: {task_id}")
        for cell, call in row.calls.items():
            model_key, context_role = _cell_parts(cell)
            if call.cell != cell or call.model_key != model_key or call.context_role != context_role:
                raise ValueError(f"cell metadata mismatch: {task_id}/{cell}")
            parsed, abstained, parse_error = parse_decision(call.response_content)
            if call.parsed_option != parsed or call.abstained != abstained:
                raise ValueError(f"parsed option mismatch: {task_id}/{cell}")
            expected_valid = parse_error is None and call.total_tokens > 0 and call.error is None
            if call.valid != expected_valid:
                raise ValueError(f"call validity mismatch: {task_id}/{cell}")
            expected_correct = (
                call.valid and not call.abstained and call.parsed_option == task.correct_option
            )
            if call.correct != expected_correct:
                raise ValueError(f"call correctness mismatch: {task_id}/{cell}")
            if endpoints is not None:
                endpoint = endpoints[model_key]
                if call.model != endpoint.model:
                    raise ValueError(f"model alias mismatch: {task_id}/{cell}")
                seed = task.seed * 31 + 17
                expected_hash = canonical_sha256(request_payload(task, cell, endpoint, seed))
                if call.request_sha256 != expected_hash:
                    raise ValueError(f"request hash mismatch: {task_id}/{cell}")


def _mcnemar_exact(left_only: int, right_only: int) -> dict[str, Any]:
    discordant = left_only + right_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, index) for index in range(min(left_only, right_only) + 1))
        p_value = min(1.0, 2.0 * tail / (2 ** discordant))
    return {
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "discordant": discordant,
        "p_value_two_sided": p_value,
    }


def _expects_abstain(task: SemanticTask, context_role: str) -> bool:
    if context_role == "none":
        return True
    if context_role == "union":
        return False
    live_role = "weak" if task.world_state["authority"] == "local" else "strong"
    return context_role != live_role


def summarize_outcomes(
    outcomes: list[SemanticProbeOutcome], tasks: list[SemanticTask] | None = None,
) -> dict[str, Any]:
    if not outcomes:
        raise ValueError("cannot summarize empty outcomes")
    cells: dict[str, Any] = {}
    for cell in CELLS:
        calls = [row.calls[cell] for row in outcomes]
        by_family: dict[str, list[SemanticMeasuredCall]] = defaultdict(list)
        for row in outcomes:
            by_family[row.family].append(row.calls[cell])
        answered = [call for call in calls if call.valid and not call.abstained]
        cells[cell] = {
            "requests": len(calls),
            "valid_rate": sum(call.valid for call in calls) / len(calls),
            "answer_coverage": len(answered) / len(calls),
            "abstain_rate": sum(call.valid and call.abstained for call in calls) / len(calls),
            "accuracy": sum(call.correct for call in calls) / len(calls),
            "selective_accuracy": (
                sum(call.correct for call in answered) / len(answered) if answered else None
            ),
            "mean_tokens": statistics.mean(call.total_tokens for call in calls),
            "mean_latency_s": statistics.mean(call.latency_s for call in calls),
            "errors": sum(call.error is not None for call in calls),
            "accuracy_by_family": {
                family: sum(call.correct for call in rows) / len(rows)
                for family, rows in sorted(by_family.items())
            },
        }

    task_index = {task.task_id: task for task in tasks or []}
    if tasks is not None and set(task_index) != {row.task_id for row in outcomes}:
        raise ValueError("summary task coverage does not match outcomes")
    if tasks is not None:
        for cell in CELLS:
            _, context_role = _cell_parts(cell)
            cells[cell]["contract_compliance_rate"] = sum(
                row.calls[cell].valid
                and row.calls[cell].abstained == _expects_abstain(task_index[row.task_id], context_role)
                for row in outcomes
            ) / len(outcomes)

    deployed_weak = [row.calls["qwen4b_weak_context"].correct for row in outcomes]
    deployed_strong = [row.calls["qwen35b_strong_context"].correct for row in outcomes]
    both = sum(weak and strong for weak, strong in zip(deployed_weak, deployed_strong, strict=True))
    weak_only = sum(weak and not strong for weak, strong in zip(deployed_weak, deployed_strong, strict=True))
    strong_only = sum(strong and not weak for weak, strong in zip(deployed_weak, deployed_strong, strict=True))
    neither = len(outcomes) - both - weak_only - strong_only
    best_single = max(sum(deployed_weak), sum(deployed_strong)) / len(outcomes)
    union = (both + weak_only + strong_only) / len(outcomes)
    authority_analysis: dict[str, Any] | None = None
    if tasks is not None:
        breakdown: dict[str, dict[str, dict[str, dict[str, float | int]]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        live_by_model: dict[str, list[SemanticMeasuredCall]] = {"qwen4b": [], "qwen35b": []}
        stale_by_model: dict[str, list[SemanticMeasuredCall]] = {"qwen4b": [], "qwen35b": []}
        authority_router: list[bool] = []
        inverse_router: list[bool] = []
        for row in outcomes:
            task = task_index[row.task_id]
            authority = task.world_state["authority"]
            live_role = "weak" if authority == "local" else "strong"
            stale_role = "strong" if live_role == "weak" else "weak"
            for model_key in ("qwen4b", "qwen35b"):
                live_call = row.calls[f"{model_key}_{live_role}_context"]
                stale_call = row.calls[f"{model_key}_{stale_role}_context"]
                live_by_model[model_key].append(live_call)
                stale_by_model[model_key].append(stale_call)
                for label, call in (("live", live_call), ("stale", stale_call)):
                    slot = breakdown[row.family][authority].setdefault(
                        f"{model_key}_{label}", {"requests": 0, "correct": 0},
                    )
                    slot["requests"] += 1
                    slot["correct"] += int(call.correct)
            routed_cell = "qwen4b_weak_context" if authority == "local" else "qwen35b_strong_context"
            inverse_cell = "qwen35b_strong_context" if authority == "local" else "qwen4b_weak_context"
            authority_router.append(row.calls[routed_cell].correct)
            inverse_router.append(row.calls[inverse_cell].correct)
        rendered_breakdown = {}
        for family, authorities in sorted(breakdown.items()):
            rendered_breakdown[family] = {}
            for authority, values in sorted(authorities.items()):
                rendered_breakdown[family][authority] = {
                    key: {**value, "accuracy": value["correct"] / value["requests"]}
                    for key, value in sorted(values.items())
                }
        authority_only = sum(a and not b for a, b in zip(authority_router, inverse_router, strict=True))
        inverse_only = sum(b and not a for a, b in zip(authority_router, inverse_router, strict=True))
        live_stale_stats = {}
        for model in ("qwen4b", "qwen35b"):
            live_calls = live_by_model[model]
            stale_calls = stale_by_model[model]
            live_only = sum(
                live.correct and not stale.correct
                for live, stale in zip(live_calls, stale_calls, strict=True)
            )
            stale_only = sum(
                stale.correct and not live.correct
                for live, stale in zip(live_calls, stale_calls, strict=True)
            )
            live_stale_stats[model] = {
                "live_accuracy": sum(call.correct for call in live_calls) / len(outcomes),
                "stale_accuracy": sum(call.correct for call in stale_calls) / len(outcomes),
                "live_minus_stale": (
                    sum(call.correct for call in live_calls)
                    - sum(call.correct for call in stale_calls)
                ) / len(outcomes),
                "live_answer_coverage": sum(
                    call.valid and not call.abstained for call in live_calls
                ) / len(outcomes),
                "stale_abstain_rate": sum(
                    call.valid and call.abstained for call in stale_calls
                ) / len(outcomes),
                "paired_mcnemar": _mcnemar_exact(live_only, stale_only),
            }
        authority_analysis = {
            "model_live_vs_stale": live_stale_stats,
            "authority_router": {
                "definition": "local -> qwen4b/local; central -> qwen35b/central",
                "accuracy": sum(authority_router) / len(outcomes),
            },
            "inverse_authority_router": {
                "definition": "local -> qwen35b/central-stale; central -> qwen4b/local-stale",
                "accuracy": sum(inverse_router) / len(outcomes),
            },
            "authority_vs_inverse_mcnemar": _mcnemar_exact(authority_only, inverse_only),
            "family_authority_model_context": rendered_breakdown,
        }

    result = {
        "schema_version": "multitown-semantic-model-probe-summary-v2",
        "evaluation_status": (
            "train-only cross-factor task/tool/model probe with explicit abstention"
        ),
        "tasks": len(outcomes),
        "calls": len(outcomes) * len(CELLS),
        "cells": cells,
        "deployed_bundle_complementarity": {
            "weak_bundle": "qwen4b_weak_context",
            "strong_bundle": "qwen35b_strong_context",
            "weak_only_correct_rate": weak_only / len(outcomes),
            "strong_only_correct_rate": strong_only / len(outcomes),
            "both_correct_rate": both / len(outcomes),
            "neither_correct_rate": neither / len(outcomes),
            "deployed_output_oracle_accuracy": union,
            "best_deployed_bundle_accuracy": best_single,
            "deployed_output_oracle_gain": union - best_single,
        },
        "factor_effects": {
            "weak_context_effect_on_qwen4b": (
                cells["qwen4b_weak_context"]["accuracy"]
                - cells["qwen4b_strong_context"]["accuracy"]
            ),
            "strong_context_effect_on_qwen35b": (
                cells["qwen35b_strong_context"]["accuracy"]
                - cells["qwen35b_weak_context"]["accuracy"]
            ),
            "model_effect_with_weak_context": (
                cells["qwen35b_weak_context"]["accuracy"]
                - cells["qwen4b_weak_context"]["accuracy"]
            ),
            "model_effect_with_strong_context": (
                cells["qwen35b_strong_context"]["accuracy"]
                - cells["qwen4b_strong_context"]["accuracy"]
            ),
            "qwen4b_union_gain_over_no_context": (
                cells["qwen4b_union_context"]["accuracy"]
                - cells["qwen4b_no_context"]["accuracy"]
            ),
            "qwen35b_union_gain_over_no_context": (
                cells["qwen35b_union_context"]["accuracy"]
                - cells["qwen35b_no_context"]["accuracy"]
            ),
        },
    }
    if authority_analysis is not None:
        result["authority_analysis"] = authority_analysis
    return result


def balanced_subset(
    tasks: list[SemanticTask], max_tasks: int | None, *, cohort_index: int = 0,
) -> list[SemanticTask]:
    if cohort_index < 0:
        raise ValueError("cohort_index must be non-negative")
    if max_tasks is None or max_tasks >= len(tasks):
        if cohort_index:
            raise ValueError("cohort_index requires max_tasks smaller than the bank")
        return tasks
    strata = len(FAMILIES) * 2
    if max_tasks <= 0 or max_tasks % strata:
        raise ValueError(f"max_tasks must be a positive multiple of {strata}")
    per_family = max_tasks // len(FAMILIES)
    per_authority = per_family // 2
    selected = []
    stratum_index = 0
    for family in FAMILIES:
        for authority in ("local", "central"):
            rows = sorted(
                (
                    task for task in tasks
                    if task.family == family and task.world_state["authority"] == authority
                ),
                key=lambda item: (item.correct_option, item.task_id),
            )
            cohort_end = (cohort_index + 1) * per_authority
            if len(rows) < cohort_end:
                raise ValueError(f"not enough {family}/{authority} tasks")
            by_option = {
                option: [task for task in rows if task.correct_option == option]
                for option in range(len(OPTION_LABELS))
            }
            picked = []
            cursor = stratum_index
            while len(picked) < cohort_end:
                option = cursor % len(OPTION_LABELS)
                if by_option[option]:
                    picked.append(by_option[option].pop(0))
                cursor += 1
                if cursor - stratum_index > (len(rows) + 1) * len(OPTION_LABELS):
                    raise RuntimeError("unable to construct authority-balanced subset")
            selected.extend(picked[cohort_index * per_authority : cohort_end])
            stratum_index += 1
    return sorted(selected, key=lambda item: item.task_id)


async def collect(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    revision, dirty = git_state(project_root)
    if dirty:
        raise RuntimeError("A13 semantic model probe requires a clean source revision")
    bank_path = Path(args.bank).resolve()
    tasks = balanced_subset(
        read_bank(bank_path), args.max_tasks, cohort_index=args.cohort_index,
    )
    if any(task.split != "train" for task in tasks):
        raise ValueError("A13 probe accepts train tasks only")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outcomes_path = output_dir / "outcomes.jsonl"
    config_path = output_dir / "config.json"
    endpoints = {
        "qwen4b": ProbeEndpoint("qwen4b", args.qwen4b_endpoint, args.qwen4b_model),
        "qwen35b": ProbeEndpoint("qwen35b", args.qwen35b_endpoint, args.qwen35b_model),
    }
    model_files = {
        "qwen4b": Path(args.qwen4b_model_file).resolve(),
        "qwen35b": Path(args.qwen35b_model_file).resolve(),
    }
    config = {
        "schema_version": CONFIG_VERSION,
        "source_revision": revision,
        "source_dirty_at_start": dirty,
        "collector_source_sha256": sha256_file(Path(__file__)),
        "semantic_tasks_source_sha256": sha256_file(Path(__file__).with_name("semantic_tasks.py")),
        "choice_prompt_protocol_version": CHOICE_PROMPT_PROTOCOL_VERSION,
        "bank": str(bank_path),
        "bank_sha256": sha256_file(bank_path),
        "task_count": len(tasks),
        "cohort_index": args.cohort_index,
        "task_ids_sha256": canonical_sha256([task.task_id for task in tasks]),
        "cells": list(CELLS),
        "cell_prompt_sha256": {
            cell: canonical_sha256([
                {"task_id": task.task_id, "messages": render_probe_messages(task, _cell_parts(cell)[1])}
                for task in tasks
            ])
            for cell in CELLS
        },
        "endpoints": {key: asdict(value) for key, value in endpoints.items()},
        "model_files": {
            key: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for key, path in model_files.items()
        },
        "decoding": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 48},
        "collection": {
            "concurrency": args.concurrency,
            "batch_tasks": args.batch_tasks,
            "timeout_s": args.timeout,
        },
        "runtime": {
            "version": args.runtime_version,
            "qwen4b_startup_args": args.qwen4b_startup_args,
            "qwen35b_startup_args": args.qwen35b_startup_args,
        },
        "http_proxy_policy": "trust_env=false; localhost bypasses download proxy",
        "selection": (
            "disjoint deterministic family x authority cohort with option round-robin"
        ),
    }
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != config:
            raise ValueError("probe config changed in an existing output directory")
    else:
        write_json(config_path, config)

    completed: dict[str, SemanticProbeOutcome] = {}
    if outcomes_path.exists() and outcomes_path.stat().st_size:
        completed_rows = read_outcomes(outcomes_path, repair_truncated_tail=True)
        completed = {row.task_id: row for row in completed_rows}
    task_index = {task.task_id: task for task in tasks}
    for task_id, row in completed.items():
        if task_id not in task_index or row.task_sha256 != task_sha256(task_index[task_id]):
            raise ValueError(f"cached task mismatch: {task_id}")
    if completed:
        validate_coverage(
            [task_index[task_id] for task_id in completed], list(completed.values()), endpoints,
        )

    pending = [task for task in tasks if task.task_id not in completed]
    started = time.monotonic()
    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = httpx.Timeout(args.timeout)
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        with outcomes_path.open("a", encoding="utf-8") as handle:
            for start in range(0, len(pending), args.batch_tasks):
                batch = pending[start : start + args.batch_tasks]
                values = await asyncio.gather(
                    *(collect_task(client, task, endpoints, semaphore) for task in batch)
                )
                for value in values:
                    handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                done = len(completed) + min(start + len(batch), len(pending))
                print(json.dumps({"completed": done, "total": len(tasks)}, ensure_ascii=False), flush=True)

    outcomes = read_outcomes(outcomes_path)
    validate_coverage(tasks, outcomes, endpoints)
    summary = summarize_outcomes(outcomes, tasks)
    summary.update({
        "source_revision": revision,
        "source_dirty_at_start": dirty,
        "collection_seconds_this_invocation": time.monotonic() - started,
        "input_sha256": {
            "bank": sha256_file(bank_path),
            "config": sha256_file(config_path),
            "outcomes": sha256_file(outcomes_path),
        },
    })
    write_json(output_dir / "summary.json", summary)
    manifest = {
        "schema_version": "multitown-semantic-model-probe-manifest-v2",
        "created_at_utc": utc_now(),
        "source_revision": revision,
        "source_dirty_at_start": dirty,
        "files": {
            name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for name, path in {
                "config.json": config_path,
                "outcomes.jsonl": outcomes_path,
                "summary.json": output_dir / "summary.json",
            }.items()
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--cohort-index", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--batch-tasks", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--qwen4b-endpoint", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--qwen35b-endpoint", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--qwen4b-model", default="qwen3.5-4b")
    parser.add_argument("--qwen35b-model", default="qwen3.5-35b-a3b")
    parser.add_argument("--qwen4b-model-file", required=True)
    parser.add_argument("--qwen35b-model-file", required=True)
    parser.add_argument("--runtime-version", required=True)
    parser.add_argument("--qwen4b-startup-args", required=True)
    parser.add_argument("--qwen35b-startup-args", required=True)
    args = parser.parse_args()
    if args.concurrency <= 0 or args.batch_tasks <= 0:
        parser.error("concurrency and batch-tasks must be positive")
    if args.cohort_index < 0:
        parser.error("cohort-index must be non-negative")
    result = asyncio.run(collect(args))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
