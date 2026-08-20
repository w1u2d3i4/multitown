"""Structured response parsing and deterministic ensemble aggregation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedDecision:
    action: str | None
    confidence: float
    brief_reason: str
    valid: bool


def strict_json_compliant(text: str, allowed_actions: tuple[str, ...]) -> bool:
    """Check the prompt contract without the benchmark's recovery parser."""
    try:
        payload = json.loads(text.strip())
    except (json.JSONDecodeError, TypeError):
        return False
    if not isinstance(payload, dict) or payload.get("action") not in allowed_actions:
        return False
    try:
        confidence = float(payload["confidence"])
    except (KeyError, TypeError, ValueError):
        return False
    return 0.0 <= confidence <= 1.0 and isinstance(payload.get("brief_reason"), str)


def parse_decision(text: str, allowed_actions: tuple[str, ...]) -> ParsedDecision:
    candidates = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
    for candidate in reversed(candidates):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        action = str(payload.get("action", "")).strip()
        if action not in allowed_actions:
            continue
        try:
            confidence = min(1.0, max(0.0, float(payload.get("confidence", 0.5))))
        except (TypeError, ValueError):
            confidence = 0.5
        reason = str(payload.get("brief_reason", ""))[:160]
        return ParsedDecision(action, confidence, reason, True)
    mentioned = [action for action in allowed_actions if action in text]
    if len(mentioned) == 1:
        return ParsedDecision(mentioned[0], 0.25, "recovered_from_unstructured_output", True)
    return ParsedDecision(None, 0.0, "invalid_or_ambiguous_output", False)


def aggregate_decisions(
    decisions: list[ParsedDecision], allowed_actions: tuple[str, ...]
) -> tuple[ParsedDecision, float, int]:
    valid = [decision for decision in decisions if decision.valid and decision.action]
    if not valid:
        return ParsedDecision(None, 0.0, "all_agents_invalid", False), 0.0, 0
    stats: dict[str, tuple[int, float]] = {}
    for action in allowed_actions:
        selected = [decision for decision in valid if decision.action == action]
        if selected:
            stats[action] = (len(selected), sum(item.confidence for item in selected) / len(selected))
    winner = max(
        stats,
        key=lambda action: (stats[action][0], stats[action][1], -allowed_actions.index(action)),
    )
    count, mean_confidence = stats[winner]
    agreement = count / len(decisions)
    return (
        ParsedDecision(winner, mean_confidence, f"majority_{count}_of_{len(decisions)}", True),
        agreement,
        len(stats),
    )
