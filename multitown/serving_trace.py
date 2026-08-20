"""Export and validate sanitized A8 agentic-serving traces."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import re
import secrets
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


TRACE_SCHEMA_VERSION = "multitown-agentic-serving-trace-v1"
NANO_SCHEMA_VERSION = "multitown-nanovllm-replay-v1"
MANIFEST_SCHEMA_VERSION = "multitown-serving-trace-export-v1"
CONTROL_ACTIONS = {"stop", "delegate", "escalate", "review", "human"}
REQUIRED_TRACE_FIELDS = {
    "schema_version", "session_id", "turn_id", "event_type", "request_id",
    "scheduled_offset_ms", "shared_prefix_length", "shared_prefix_length_unit",
    "shared_prefix_request_id", "input_tokens", "output_tokens", "action",
    "latency", "validator_state", "weak_disagreement", "budget_state", "request",
}
REQUEST_ACTION = {
    "initial_attempt": "delegate",
    "selective_weak_delegation": "delegate",
    "strong_specialist_escalation": "escalate",
    "independent_review": "review",
}
FORBIDDEN_KEYS = {
    "api_key",
    "endpoint",
    "scenario_id",
    "scenario_sha256",
    "oracle_action",
    "correct",
    "correct_individual",
    "raw_content",
    "brief_reason",
    "confidence",
    "inference_seed",
}
SENSITIVE_PATTERNS = {
    "home_path": re.compile(r"/(?:home|root)/[^\s\"']+"),
    "ipv4": re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)"),
    "secret_assignment": re.compile(
        r"(?i)(?:api[_-]?key|authorization|bearer|password)\s*[:=]\s*\S+"
    ),
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: row is not an object")
                rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pseudonym(salt: bytes, namespace: str, value: str) -> str:
    digest = hmac.new(salt, f"{namespace}:{value}".encode(), hashlib.sha256).hexdigest()
    return f"{namespace[:1]}_{digest[:24]}"


def _timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def render_chatml(messages: list[dict[str, Any]]) -> str:
    """Render messages deterministically for Nano-vLLM performance replay."""

    rendered = []
    for message in messages:
        role = str(message["role"])
        content = str(message["content"])
        rendered.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    rendered.append("<|im_start|>assistant\n")
    return "".join(rendered)


def _longest_common_prefix(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def _apply_prefix_metadata(rows: list[dict[str, Any]]) -> None:
    previous: list[tuple[str, str]] = []
    for row in rows:
        if row["event_type"] != "request":
            row["shared_prefix_length"] = 0
            row["shared_prefix_request_id"] = None
            continue
        prompt = row["request"]["prompt"]
        best_length = 0
        best_request = None
        for request_id, prior_prompt in previous:
            length = _longest_common_prefix(prompt, prior_prompt)
            if length > best_length:
                best_length = length
                best_request = request_id
        row["shared_prefix_length"] = best_length
        row["shared_prefix_request_id"] = best_request
        previous.append((row["request_id"], prompt))


def _validator_state(value: dict[str, Any] | None) -> dict[str, Any]:
    value = value or {}
    return {
        "parse_valid": bool(value.get("parse_valid", False)),
        "hard_constraints_pass": bool(value.get("hard_constraints_pass", False)),
        "issue_codes": sorted(str(item) for item in value.get("issue_codes", [])),
    }


def _phase_validation(decision: dict[str, Any], turn_id: int, request: dict[str, Any]) -> dict[str, Any]:
    trace = decision.get("phase_trace", [])
    if turn_id >= len(trace):
        raise ValueError(f"missing phase_trace row for request turn {turn_id}")
    phase = trace[turn_id]
    if phase.get("phase") != request.get("phase") or phase.get("role") != request.get("role"):
        raise ValueError("request rows and decision phase_trace are not aligned")
    return _validator_state(phase.get("validation"))


def _sample_session_ids(
    decisions: list[dict[str, Any]], session_ids: dict[str, str], count: int
) -> list[str]:
    selected: list[str] = []
    seen_routes: set[str] = set()
    for decision in decisions:
        route = str(decision.get("route"))
        session_id = session_ids[str(decision["scenario_id"])]
        if route not in seen_routes:
            selected.append(session_id)
            seen_routes.add(route)
        if len(selected) >= count:
            return selected
    for decision in decisions:
        session_id = session_ids[str(decision["scenario_id"])]
        if session_id not in selected:
            selected.append(session_id)
        if len(selected) >= count:
            break
    return selected


def trace_to_nano(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    replay = []
    for row in rows:
        if row["event_type"] != "request":
            continue
        request = row["request"]
        replay.append({
            "schema_version": NANO_SCHEMA_VERSION,
            "request_id": row["request_id"],
            "session_id": row["session_id"],
            "turn_id": row["turn_id"],
            "scheduled_offset_ms": row["scheduled_offset_ms"],
            "prompt": request["prompt"],
            "sampling_params": request["sampling_params"],
            "reference": {
                "input_tokens": row["input_tokens"],
                "output_tokens": row["output_tokens"],
                "shared_prefix_length": row["shared_prefix_length"],
                "shared_prefix_length_unit": row["shared_prefix_length_unit"],
                "action": row["action"],
                "model_tier": request["model_tier"],
            },
        })
    return replay


def export_a8_trace(
    *,
    requests_path: Path,
    decisions_path: Path,
    trace_path: Path,
    nano_path: Path,
    manifest_path: Path,
    sample_trace_path: Path | None = None,
    sample_nano_path: Path | None = None,
    sample_sessions: int = 8,
    salt: bytes | None = None,
) -> dict[str, Any]:
    requests = _read_jsonl(requests_path)
    decisions = _read_jsonl(decisions_path)
    salt = salt or secrets.token_bytes(32)
    if not requests or not decisions:
        raise ValueError("requests and decisions must be non-empty")

    decision_by_scenario: dict[str, dict[str, Any]] = {}
    for decision in decisions:
        scenario_id = str(decision["scenario_id"])
        if scenario_id in decision_by_scenario:
            raise ValueError(f"duplicate decision for {scenario_id}")
        decision_by_scenario[scenario_id] = decision

    requests_by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for request in requests:
        requests_by_scenario[str(request["scenario_id"])].append(request)
    if set(requests_by_scenario) != set(decision_by_scenario):
        raise ValueError("request/decision scenario sets differ")

    session_ids = {
        scenario_id: _pseudonym(salt, "session", scenario_id)
        for scenario_id in decision_by_scenario
    }
    first_timestamp = min(_timestamp(str(row["timestamp_utc"])) for row in requests)
    rows: list[dict[str, Any]] = []
    route_counts: Counter[str] = Counter()

    for decision in decisions:
        scenario_id = str(decision["scenario_id"])
        session_id = session_ids[scenario_id]
        session_requests = requests_by_scenario[scenario_id]
        if len(session_requests) != int(decision["request_count"]):
            raise ValueError(f"request count mismatch for {scenario_id}")
        if len(decision.get("phase_trace", [])) != len(session_requests):
            raise ValueError(f"phase trace count mismatch for {scenario_id}")

        cumulative_tokens = 0
        cumulative_latency = 0.0
        weak_actions: list[str] = []
        for turn_id, request in enumerate(session_requests):
            phase = str(request["phase"])
            if phase not in REQUEST_ACTION:
                raise ValueError(f"unsupported A8 phase: {phase}")
            input_tokens = int(request["prompt_tokens"])
            output_tokens = int(request["completion_tokens"])
            request_latency = float(request["latency_s"])
            cumulative_tokens += input_tokens + output_tokens
            cumulative_latency += request_latency
            if request.get("model_tier") == "weak":
                weak_actions.append(str(request.get("action")))
            disagreement = len(weak_actions) >= 2 and len(set(weak_actions[:2])) > 1
            messages = [
                {"role": str(item["role"]), "content": str(item["content"])}
                for item in request["messages"]
            ]
            request_id = _pseudonym(salt, "request", f"{scenario_id}:{turn_id}")
            rows.append({
                "schema_version": TRACE_SCHEMA_VERSION,
                "session_id": session_id,
                "turn_id": turn_id,
                "event_type": "request",
                "request_id": request_id,
                "scheduled_offset_ms": round(
                    (_timestamp(str(request["timestamp_utc"])) - first_timestamp).total_seconds() * 1000,
                    3,
                ),
                "shared_prefix_length": 0,
                "shared_prefix_length_unit": "unicode_codepoints",
                "shared_prefix_request_id": None,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "action": REQUEST_ACTION[phase],
                "latency": {
                    "request_s": request_latency,
                    "ttft_s": request.get("ttft_s"),
                    "e2e_s": None,
                },
                "validator_state": _phase_validation(decision, turn_id, request),
                "weak_disagreement": disagreement if len(weak_actions) >= 2 else None,
                "budget_state": {
                    "mode": "observed_unbounded_formal_a8",
                    "token_limit": None,
                    "latency_limit_s": None,
                    "tokens_used": cumulative_tokens,
                    "latency_used_s": cumulative_latency,
                    "exceeded": False,
                },
                "request": {
                    "phase": phase,
                    "role": str(request["role"]),
                    "model_tier": str(request["model_tier"]),
                    "messages": messages,
                    "prompt": render_chatml(messages),
                    "prompt_format": "qwen_chatml_text_v1",
                    "sampling_params": {
                        "temperature": float(request["temperature"]),
                        "top_p": float(request["top_p"]),
                        "max_tokens": int(request["max_tokens"]),
                    },
                },
            })

        terminal_action = "human" if decision.get("human_escalation_required") else "stop"
        decision_timestamp = _timestamp(str(decision["timestamp_utc"]))
        rows.append({
            "schema_version": TRACE_SCHEMA_VERSION,
            "session_id": session_id,
            "turn_id": len(session_requests),
            "event_type": "terminal",
            "request_id": None,
            "scheduled_offset_ms": round((decision_timestamp - first_timestamp).total_seconds() * 1000, 3),
            "shared_prefix_length": 0,
            "shared_prefix_length_unit": "unicode_codepoints",
            "shared_prefix_request_id": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "action": terminal_action,
            "latency": {
                "request_s": None,
                "ttft_s": None,
                "e2e_s": float(decision["decision_latency_s"]),
            },
            "validator_state": _validator_state(decision.get("final_validation")),
            "weak_disagreement": (
                len(set(weak_actions[:2])) > 1 if len(weak_actions) >= 2 else None
            ),
            "budget_state": {
                "mode": "observed_unbounded_formal_a8",
                "token_limit": None,
                "latency_limit_s": None,
                "tokens_used": cumulative_tokens,
                "latency_used_s": cumulative_latency,
                "exceeded": False,
            },
            "request": None,
        })
        route_counts[str(decision["route"])] += 1

    _apply_prefix_metadata(rows)
    nano_rows = trace_to_nano(rows)
    serialized = json.dumps(rows, ensure_ascii=False)
    leaked_ids = [scenario_id for scenario_id in decision_by_scenario if scenario_id in serialized]
    if leaked_ids:
        raise ValueError(f"original scenario identifiers leaked into export: {leaked_ids[:3]}")
    _write_jsonl(trace_path, rows)
    _write_jsonl(nano_path, nano_rows)

    sample_ids: list[str] = []
    sample_rows: list[dict[str, Any]] = []
    sample_nano_rows: list[dict[str, Any]] = []
    if sample_trace_path or sample_nano_path:
        sample_ids = _sample_session_ids(decisions, session_ids, sample_sessions)
        selected = set(sample_ids)
        sample_rows = [json.loads(json.dumps(row)) for row in rows if row["session_id"] in selected]
        _apply_prefix_metadata(sample_rows)
        sample_nano_rows = trace_to_nano(sample_rows)
        if sample_trace_path:
            _write_jsonl(sample_trace_path, sample_rows)
        if sample_nano_path:
            _write_jsonl(sample_nano_path, sample_nano_rows)

    validations = [validate_trace(trace_path), validate_nano_replay(nano_path)]
    if sample_trace_path:
        validations.append(validate_trace(sample_trace_path))
    if sample_nano_path:
        validations.append(validate_nano_replay(sample_nano_path))
    failed = [result for result in validations if not result["passed"]]
    if failed:
        raise ValueError(f"generated export failed validation: {failed}")

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source": {
            "architecture": "A8",
            "formal_record": "v0.2.0",
            "requests_sha256": _sha256(requests_path),
            "decisions_sha256": _sha256(decisions_path),
        },
        "pseudonymization": {
            "algorithm": "HMAC-SHA256",
            "salt_sha256": hashlib.sha256(salt).hexdigest(),
            "salt_retained": False,
        },
        "privacy": {
            "removed_fields": sorted(FORBIDDEN_KEYS),
            "contains_model_response_text": False,
            "contains_oracle_or_correctness_labels": False,
            "contains_original_scenario_identifiers": False,
        },
        "prefix_semantics": {
            "length_unit": "unicode_codepoints",
            "definition": "maximum exact text prefix shared with any earlier replay request",
            "serving_requirement": "recompute cache hits from engine token IDs and allocated KV blocks",
        },
        "sessions": len(decisions),
        "request_events": len(requests),
        "terminal_events": len(decisions),
        "trace_events": len(rows),
        "route_counts": dict(sorted(route_counts.items())),
        "sample_sessions": len(sample_ids),
        "sample_trace_events": len(sample_rows),
        "sample_replay_requests": len(sample_nano_rows),
        "outputs": {
            "trace_sha256": _sha256(trace_path),
            "nano_replay_sha256": _sha256(nano_path),
            "sample_trace_sha256": _sha256(sample_trace_path) if sample_trace_path else None,
            "sample_nano_replay_sha256": _sha256(sample_nano_path) if sample_nano_path else None,
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def _walk(value: Any, path: str = "$") -> Iterable[tuple[str, str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield path, str(key), child
            yield from _walk(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, f"{path}[{index}]")


def validate_trace(path: Path) -> dict[str, Any]:
    issues: list[str] = []
    try:
        rows = _read_jsonl(path)
    except Exception as error:  # report malformed JSON as validation evidence
        return {"path": str(path), "passed": False, "rows": 0, "issues": [str(error)]}

    sessions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    previous: list[tuple[str, str]] = []
    request_ids: set[str] = set()
    for index, row in enumerate(rows):
        label = f"row {index + 1}"
        missing = REQUIRED_TRACE_FIELDS - set(row)
        if missing:
            issues.append(f"{label}: missing fields {sorted(missing)}")
        if row.get("schema_version") != TRACE_SCHEMA_VERSION:
            issues.append(f"{label}: invalid schema_version")
        if row.get("action") not in CONTROL_ACTIONS:
            issues.append(f"{label}: invalid action")
        session_id = row.get("session_id")
        if not isinstance(session_id, str):
            issues.append(f"{label}: missing session_id")
            continue
        sessions[session_id].append(row)
        if not isinstance(row.get("turn_id"), int) or row["turn_id"] < 0:
            issues.append(f"{label}: invalid turn_id")
        if not isinstance(row.get("scheduled_offset_ms"), (int, float)) or row["scheduled_offset_ms"] < 0:
            issues.append(f"{label}: invalid scheduled_offset_ms")
        if row.get("shared_prefix_length_unit") != "unicode_codepoints":
            issues.append(f"{label}: invalid shared_prefix_length_unit")
        if not isinstance(row.get("shared_prefix_length"), int) or row["shared_prefix_length"] < 0:
            issues.append(f"{label}: invalid shared_prefix_length")
        latency = row.get("latency")
        if not isinstance(latency, dict) or set(latency) != {"request_s", "ttft_s", "e2e_s"}:
            issues.append(f"{label}: invalid latency state")
        validator = row.get("validator_state")
        if (
            not isinstance(validator, dict)
            or set(validator) != {"parse_valid", "hard_constraints_pass", "issue_codes"}
            or not isinstance(validator.get("parse_valid"), bool)
            or not isinstance(validator.get("hard_constraints_pass"), bool)
            or not isinstance(validator.get("issue_codes"), list)
        ):
            issues.append(f"{label}: invalid validator_state")
        budget = row.get("budget_state")
        budget_fields = {
            "mode", "token_limit", "latency_limit_s", "tokens_used", "latency_used_s", "exceeded"
        }
        if not isinstance(budget, dict) or set(budget) != budget_fields:
            issues.append(f"{label}: invalid budget_state")
        if row.get("weak_disagreement") is not None and not isinstance(row["weak_disagreement"], bool):
            issues.append(f"{label}: invalid weak_disagreement")
        for object_path, key, child in _walk(row):
            if key in FORBIDDEN_KEYS:
                issues.append(f"{label}: forbidden key {object_path}.{key}")
            if isinstance(child, str):
                for name, pattern in SENSITIVE_PATTERNS.items():
                    if pattern.search(child):
                        issues.append(f"{label}: sensitive pattern {name} at {object_path}.{key}")

        if row.get("event_type") == "request":
            request_id = row.get("request_id")
            if not isinstance(request_id, str) or request_id in request_ids:
                issues.append(f"{label}: invalid or duplicate request_id")
                continue
            request_ids.add(request_id)
            request = row.get("request")
            if not isinstance(request, dict) or not isinstance(request.get("prompt"), str):
                issues.append(f"{label}: missing replay prompt")
                continue
            prompt = request["prompt"]
            best_length = 0
            best_request = None
            for prior_id, prior_prompt in previous:
                length = _longest_common_prefix(prompt, prior_prompt)
                if length > best_length:
                    best_length = length
                    best_request = prior_id
            if row.get("shared_prefix_length") != best_length:
                issues.append(f"{label}: shared prefix length mismatch")
            if row.get("shared_prefix_request_id") != best_request:
                issues.append(f"{label}: shared prefix source mismatch")
            previous.append((request_id, prompt))
            if int(row.get("input_tokens", 0)) <= 0 or int(row.get("output_tokens", -1)) < 0:
                issues.append(f"{label}: invalid token counts")
        elif row.get("event_type") == "terminal":
            if row.get("request") is not None or row.get("request_id") is not None:
                issues.append(f"{label}: terminal contains request payload")
            if row.get("action") not in {"stop", "human"}:
                issues.append(f"{label}: invalid terminal action")
            if row.get("input_tokens") != 0 or row.get("output_tokens") != 0:
                issues.append(f"{label}: terminal token counts must be zero")
        else:
            issues.append(f"{label}: invalid event_type")

    for session_id, session_rows in sessions.items():
        turns = [row.get("turn_id") for row in session_rows]
        if turns != list(range(len(session_rows))):
            issues.append(f"session {session_id}: non-contiguous turns")
        terminals = [row for row in session_rows if row.get("event_type") == "terminal"]
        if len(terminals) != 1 or session_rows[-1].get("event_type") != "terminal":
            issues.append(f"session {session_id}: terminal must occur exactly once and last")
        cumulative_tokens = 0
        cumulative_latency = 0.0
        for row in session_rows:
            if row.get("event_type") == "request":
                cumulative_tokens += int(row.get("input_tokens", 0)) + int(row.get("output_tokens", 0))
                latency = row.get("latency")
                if isinstance(latency, dict) and isinstance(latency.get("request_s"), (int, float)):
                    cumulative_latency += float(latency["request_s"])
            budget = row.get("budget_state")
            if isinstance(budget, dict):
                if budget.get("tokens_used") != cumulative_tokens:
                    issues.append(f"session {session_id}: cumulative token budget mismatch")
                if abs(float(budget.get("latency_used_s", -1)) - cumulative_latency) > 1e-9:
                    issues.append(f"session {session_id}: cumulative latency budget mismatch")

    return {
        "schema_version": "multitown-serving-trace-validation-v1",
        "path": str(path),
        "passed": not issues,
        "rows": len(rows),
        "sessions": len(sessions),
        "requests": len(request_ids),
        "issues": issues,
    }


def validate_nano_replay(path: Path) -> dict[str, Any]:
    issues: list[str] = []
    try:
        rows = _read_jsonl(path)
    except Exception as error:
        return {"path": str(path), "passed": False, "rows": 0, "issues": [str(error)]}
    request_ids: set[str] = set()
    for index, row in enumerate(rows):
        label = f"row {index + 1}"
        required = {
            "schema_version", "request_id", "session_id", "turn_id",
            "scheduled_offset_ms", "prompt", "sampling_params", "reference",
        }
        if required - set(row):
            issues.append(f"{label}: missing fields {sorted(required - set(row))}")
        if row.get("schema_version") != NANO_SCHEMA_VERSION:
            issues.append(f"{label}: invalid schema_version")
        request_id = row.get("request_id")
        if not isinstance(request_id, str) or request_id in request_ids:
            issues.append(f"{label}: invalid or duplicate request_id")
        else:
            request_ids.add(request_id)
        if not isinstance(row.get("prompt"), str) or not row["prompt"]:
            issues.append(f"{label}: missing prompt")
        params = row.get("sampling_params")
        if not isinstance(params, dict) or not {"temperature", "top_p", "max_tokens"} <= set(params):
            issues.append(f"{label}: incomplete sampling_params")
        for object_path, key, child in _walk(row):
            if key in FORBIDDEN_KEYS:
                issues.append(f"{label}: forbidden key {object_path}.{key}")
            if isinstance(child, str):
                for name, pattern in SENSITIVE_PATTERNS.items():
                    if pattern.search(child):
                        issues.append(f"{label}: sensitive pattern {name} at {object_path}.{key}")
    return {
        "schema_version": "multitown-nanovllm-replay-validation-v1",
        "path": str(path),
        "passed": not issues,
        "rows": len(rows),
        "issues": issues,
    }


def export_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--nano-replay", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sample-trace", type=Path)
    parser.add_argument("--sample-nano-replay", type=Path)
    parser.add_argument("--sample-sessions", type=int, default=8)
    args = parser.parse_args()
    if args.sample_sessions <= 0:
        parser.error("--sample-sessions must be positive")
    manifest = export_a8_trace(
        requests_path=args.requests,
        decisions_path=args.decisions,
        trace_path=args.trace,
        nano_path=args.nano_replay,
        manifest_path=args.manifest,
        sample_trace_path=args.sample_trace,
        sample_nano_path=args.sample_nano_replay,
        sample_sessions=args.sample_sessions,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def validate_main() -> None:
    parser = argparse.ArgumentParser(description="Validate sanitized trace and Nano-vLLM replay JSONL.")
    parser.add_argument("--trace", type=Path, action="append", default=[])
    parser.add_argument("--nano-replay", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.trace and not args.nano_replay:
        parser.error("provide at least one --trace or --nano-replay")
    results = [validate_trace(path) for path in args.trace]
    results.extend(validate_nano_replay(path) for path in args.nano_replay)
    payload = {"passed": all(result["passed"] for result in results), "results": results}
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    raise SystemExit(0 if payload["passed"] else 1)


if __name__ == "__main__":
    export_main()
