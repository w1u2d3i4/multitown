from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from harness.adapters.openai_adapter import OpenAIAdapter
from harness.agent_interface import AdapterResponse, ToolCallAdapter

DEFAULT_HISTORY_CHAR_BUDGET = 20_000


class _BudgetedAdapter(Protocol):
    configured_max_tokens: int

    def set_request_max_tokens(self, value: int) -> None: ...

    def generate_with_tools(
        self, messages: list[dict], system_prompt: str, tools: list[dict]
    ) -> AdapterResponse: ...

    def get_usage(self) -> dict[str, Any]: ...


def deterministic_request_seed(
    base_seed: int,
    *,
    task_id: str,
    role: str,
    request_index: int,
) -> int:
    material = f"{base_seed}\0{task_id}\0{role}\0{request_index}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big") & 0x7FFFFFFF


class SeededOpenAIAdapter(OpenAIAdapter):
    """Attach a per-request seed without patching the upstream TeamBench adapter."""

    def __init__(self, *args: Any, seed_provider: Callable[[], int], **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._seed_provider = seed_provider

    def _call_with_retry(self, max_retries: int = 8, **kwargs: Any) -> Any:
        kwargs["seed"] = self._seed_provider()
        return super()._call_with_retry(max_retries=max_retries, **kwargs)


def _message_chars(message: dict[str, Any]) -> int:
    return len(str(message.get("content", "")))


def compact_messages(
    messages: list[dict[str, Any]],
    *,
    char_budget: int = DEFAULT_HISTORY_CHAR_BUDGET,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Keep the task prompt and recent turns inside a deterministic context window."""
    original_chars = sum(_message_chars(message) for message in messages)
    if len(messages) <= 2 or original_chars <= char_budget:
        copied = [dict(message) for message in messages]
        return copied, {
            "original_message_count": len(messages),
            "effective_message_count": len(copied),
            "dropped_message_count": 0,
            "original_message_chars": original_chars,
            "effective_message_chars": original_chars,
        }

    first = dict(messages[0])
    marker = {
        "role": "user",
        "content": (
            "Earlier intermediate tool transcripts were omitted to stay within the local "
            "model context window. The task prompt is preserved. Re-read workspace files or "
            "rerun checks when an omitted detail is needed."
        ),
    }
    used = _message_chars(first) + _message_chars(marker)
    tail: list[dict[str, Any]] = []
    for message in reversed(messages[1:]):
        size = _message_chars(message)
        if tail and used + size > char_budget:
            break
        if not tail and used + size > char_budget:
            content = str(message.get("content", ""))
            room = max(1_000, char_budget - used)
            trimmed = dict(message)
            trimmed["content"] = content[-room:]
            tail.append(trimmed)
            used += _message_chars(trimmed)
            break
        tail.append(dict(message))
        used += size
    tail.reverse()
    effective = [first, marker, *tail]
    dropped = max(0, len(messages) - len(tail) - 1)
    effective_chars = sum(_message_chars(message) for message in effective)
    return effective, {
        "original_message_count": len(messages),
        "effective_message_count": len(effective),
        "dropped_message_count": dropped,
        "original_message_chars": original_chars,
        "effective_message_chars": effective_chars,
    }


class OutcomeStopAdapter(ToolCallAdapter):
    """End a phase without another model request once its protocol output exists."""

    def __init__(self, inner: ToolCallAdapter, should_stop: Callable[[], bool]):
        self.inner = inner
        self.should_stop = should_stop

    def generate_with_tools(
        self, messages: list[dict], system_prompt: str, tools: list[dict]
    ) -> AdapterResponse:
        if self.should_stop():
            return AdapterResponse(text="DONE", done=True)
        return self.inner.generate_with_tools(messages, system_prompt, tools)

    def get_usage(self) -> dict[str, Any]:
        return self.inner.get_usage()


class ConservativeTokenBudgetAdapter(ToolCallAdapter):
    """Stop before a request that cannot fit a conservative total-token bound.

    UTF-8 payload bytes upper-bound ordinary byte-fallback tokenizer pieces.
    A configurable template margin covers provider chat formatting that is not
    visible in the OpenAI-compatible request. Actual provider usage is audited
    after every request; an observed overrun is recorded rather than hidden.
    """

    def __init__(
        self,
        inner: _BudgetedAdapter,
        *,
        usage_provider: Callable[[], int],
        total_budget: int,
        template_overhead_tokens: int = 4096,
        minimum_completion_tokens: int = 64,
    ):
        if total_budget <= 0:
            raise ValueError("total token budget must be positive")
        if template_overhead_tokens < 0:
            raise ValueError("template overhead must be non-negative")
        if minimum_completion_tokens <= 0:
            raise ValueError("minimum completion tokens must be positive")
        self.inner = inner
        self.usage_provider = usage_provider
        self.total_budget = total_budget
        self.template_overhead_tokens = template_overhead_tokens
        self.minimum_completion_tokens = minimum_completion_tokens
        self.events: list[dict[str, Any]] = []

    def _prompt_upper_bound(
        self, messages: list[dict], system_prompt: str, tools: list[dict]
    ) -> int:
        effective_messages, _ = compact_messages(messages)
        payload = json.dumps(
            {
                "system_prompt": system_prompt,
                "messages": effective_messages,
                "tools": tools,
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        ).encode("utf-8")
        return len(payload) + self.template_overhead_tokens

    def generate_with_tools(
        self, messages: list[dict], system_prompt: str, tools: list[dict]
    ) -> AdapterResponse:
        total_before = int(self.usage_provider())
        remaining_before = max(0, self.total_budget - total_before)
        prompt_upper_bound = self._prompt_upper_bound(
            messages, system_prompt, tools
        )
        completion_limit = min(
            self.inner.configured_max_tokens,
            max(0, remaining_before - prompt_upper_bound),
        )
        if completion_limit < self.minimum_completion_tokens:
            self.events.append({
                "event": "request_blocked",
                "total_tokens_before": total_before,
                "remaining_tokens_before": remaining_before,
                "prompt_token_upper_bound": prompt_upper_bound,
                "minimum_completion_tokens": self.minimum_completion_tokens,
            })
            return AdapterResponse(text="DONE", done=True)
        self.inner.set_request_max_tokens(completion_limit)
        if completion_limit < self.inner.configured_max_tokens:
            self.events.append({
                "event": "completion_capped",
                "total_tokens_before": total_before,
                "remaining_tokens_before": remaining_before,
                "prompt_token_upper_bound": prompt_upper_bound,
                "completion_token_limit": completion_limit,
            })
        response = self.inner.generate_with_tools(messages, system_prompt, tools)
        total_after = int(self.usage_provider())
        if total_after > self.total_budget:
            self.events.append({
                "event": "observed_provider_overrun",
                "total_tokens_after": total_after,
                "overrun_tokens": total_after - self.total_budget,
            })
        return response

    def get_usage(self) -> dict[str, Any]:
        return self.inner.get_usage()


class RecordedAdapter(ToolCallAdapter):
    def __init__(
        self,
        *,
        role: str,
        task_id: str,
        method: str,
        request_log: Path,
        endpoint: str,
        api_key: str,
        model: str,
        max_tokens: int,
        temperature: float,
        sampling_seed: int | None = None,
    ):
        self.role = role
        self.task_id = task_id
        self.method = method
        self.request_log = request_log
        self.request_index = 0
        self.sampling_seed = sampling_seed
        self.configured_max_tokens = max_tokens
        adapter_class = SeededOpenAIAdapter if sampling_seed is not None else OpenAIAdapter
        adapter_kwargs: dict[str, Any] = {}
        if sampling_seed is not None:
            adapter_kwargs["seed_provider"] = lambda: deterministic_request_seed(
                sampling_seed,
                task_id=self.task_id,
                role=self.role,
                request_index=self.request_index,
            )
        self.inner = adapter_class(
            api_key=api_key,
            model=model,
            base_url=endpoint,
            max_tokens=max_tokens,
            temperature=temperature,
            lenient_mode=True,
            **adapter_kwargs,
        )

    def generate_with_tools(
        self,
        messages: list[dict],
        system_prompt: str,
        tools: list[dict],
    ) -> AdapterResponse:
        effective_messages, compaction = compact_messages(messages)
        before = self.inner.get_usage()
        started = time.perf_counter()
        error = ""
        response = AdapterResponse()
        try:
            response = self.inner.generate_with_tools(
                effective_messages, system_prompt, tools
            )
            return response
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            after = self.inner.get_usage()
            payload = {
                "schema_version": "general-mas-request-v2",
                "method": self.method,
                "task_id": self.task_id,
                "role": self.role,
                "request_index": self.request_index,
                "latency_s": time.perf_counter() - started,
                "input_tokens": int(after.get("input_tokens", 0)) - int(before.get("input_tokens", 0)),
                "output_tokens": int(after.get("output_tokens", 0)) - int(before.get("output_tokens", 0)),
                "total_tokens": int(after.get("total_tokens", 0)) - int(before.get("total_tokens", 0)),
                "message_count": len(effective_messages),
                **compaction,
                "prompt_sha256": hashlib.sha256(
                    json.dumps(
                        {"system": system_prompt, "messages": effective_messages},
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str,
                    ).encode("utf-8")
                ).hexdigest(),
                "declared_tools": [item.get("name") for item in tools],
                "returned_tool_calls": len(response.tool_calls),
                "response_chars": len(response.text),
                "error": error,
            }
            if self.sampling_seed is not None:
                payload["sampling_seed"] = self.sampling_seed
                payload["request_seed"] = deterministic_request_seed(
                    self.sampling_seed,
                    task_id=self.task_id,
                    role=self.role,
                    request_index=self.request_index,
                )
            self.request_log.parent.mkdir(parents=True, exist_ok=True)
            with self.request_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self.request_index += 1

    def get_usage(self) -> dict[str, Any]:
        return self.inner.get_usage()

    def set_request_max_tokens(self, value: int) -> None:
        if value <= 0:
            raise ValueError("request max tokens must be positive")
        self.inner.max_tokens = min(value, self.configured_max_tokens)
