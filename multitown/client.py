"""Minimal streaming OpenAI-compatible client with latency and usage telemetry."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class ModelResponse:
    content: str
    reasoning_chars: int
    latency_s: float
    ttft_s: float | None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: str | None = None
    error: str | None = None


async def stream_chat_completion(
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
    response_format: dict[str, Any] | None = None,
) -> ModelResponse:
    url = endpoint.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "seed": seed,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
        "stream_options": {"include_usage": True},
        "cache_prompt": True,
    }
    if response_format is not None:
        body["response_format"] = response_format
    started = time.perf_counter()
    first_token: float | None = None
    pieces: list[str] = []
    reasoning_chars = 0
    usage: dict[str, int] = {}
    finish_reason: str | None = None
    try:
        async with client.stream("POST", url, headers=headers, json=body) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                payload = json.loads(data)
                if payload.get("usage"):
                    usage = payload["usage"]
                choices = payload.get("choices") or []
                if not choices:
                    continue
                if choices[0].get("finish_reason") is not None:
                    finish_reason = str(choices[0]["finish_reason"])
                delta = choices[0].get("delta") or {}
                content = delta.get("content") or ""
                reasoning = delta.get("reasoning_content") or ""
                if (content or reasoning) and first_token is None:
                    first_token = time.perf_counter()
                if isinstance(content, str):
                    pieces.append(content)
                if isinstance(reasoning, str):
                    reasoning_chars += len(reasoning)
        ended = time.perf_counter()
        return ModelResponse(
            content="".join(pieces),
            reasoning_chars=reasoning_chars,
            latency_s=ended - started,
            ttft_s=None if first_token is None else first_token - started,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            total_tokens=int(usage.get("total_tokens", 0)),
            finish_reason=finish_reason,
        )
    except Exception as exc:
        ended = time.perf_counter()
        return ModelResponse(
            content="", reasoning_chars=reasoning_chars, latency_s=ended - started,
            ttft_s=None if first_token is None else first_token - started,
            prompt_tokens=0, completion_tokens=0, total_tokens=0,
            finish_reason=None,
            error=f"{type(exc).__name__}: {exc}",
        )
