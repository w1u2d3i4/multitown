"""Deterministic answer parsing and scoring for the frozen MASBench subset."""

from __future__ import annotations

import json
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from typing import Any


BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]*)\}")
NUMBER_PATTERN = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])")


def _balanced_json_objects(text: str) -> list[str]:
    objects: list[str] = []
    start: int | None = None
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                objects.append(text[start : index + 1])
                start = None
    return objects


def _coerce_answers(value: Any) -> list[str] | None:
    if not isinstance(value, list) or not value:
        return None
    answers = [str(item).strip() for item in value]
    return answers if all(answers) else None


def parse_answers(text: str, expected_count: int) -> tuple[list[str] | None, str]:
    """Parse answer arrays, then boxed values, without guessing extra numbers."""

    for candidate in reversed(_balanced_json_objects(text)):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        answers = _coerce_answers(payload.get("answers")) if isinstance(payload, dict) else None
        if answers is not None and len(answers) == expected_count:
            return answers, "json"

    boxed = [item.strip() for item in BOXED_PATTERN.findall(text) if item.strip()]
    if len(boxed) == expected_count:
        return boxed, "boxed"
    if len(boxed) == 1 and expected_count > 1:
        separated = [item.strip() for item in re.split(r"<<horizon>>|[,;\n]", boxed[0]) if item.strip()]
        if len(separated) == expected_count:
            return separated, "boxed_joined"

    final_lines = re.findall(r"Problem\s+\d+\s*:\s*([^\n]+)", text, flags=re.IGNORECASE)
    if len(final_lines) >= expected_count:
        extracted = []
        for line in final_lines[-expected_count:]:
            numbers = NUMBER_PATTERN.findall(line)
            if len(numbers) != 1:
                break
            extracted.append(numbers[0])
        if len(extracted) == expected_count:
            return extracted, "problem_lines"
    return None, "invalid"


def normalize_numeric(value: str) -> str | None:
    try:
        number = Decimal(value.strip().replace(",", ""))
    except (InvalidOperation, AttributeError):
        return None
    if not number.is_finite():
        return None
    normalized = format(number.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return "0" if normalized in {"-0", ""} else normalized


def grade_answers(predicted: list[str] | None, expected: list[str]) -> bool:
    if predicted is None or len(predicted) != len(expected):
        return False
    left = [normalize_numeric(item) for item in predicted]
    right = [normalize_numeric(item) for item in expected]
    return None not in left and left == right


def aggregate_answer_lists(
    candidates: list[list[str] | None], expected_count: int
) -> tuple[list[str] | None, float, int]:
    normalized: list[tuple[str, ...]] = []
    original: dict[tuple[str, ...], list[str]] = {}
    for candidate in candidates:
        if candidate is None or len(candidate) != expected_count:
            continue
        key_values = [normalize_numeric(value) for value in candidate]
        if any(value is None for value in key_values):
            continue
        key = tuple(value for value in key_values if value is not None)
        normalized.append(key)
        original.setdefault(key, candidate)
    if not normalized:
        return None, 0.0, 0
    counts = Counter(normalized)
    first_seen = {key: normalized.index(key) for key in counts}
    winner = max(counts, key=lambda key: (counts[key], -first_seen[key]))
    return original[winner], counts[winner] / len(candidates), len(counts)
