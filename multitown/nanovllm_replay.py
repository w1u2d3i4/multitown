"""Consume an MultiTown replay with Nano-vLLM's offline generate API."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from .serving_trace import _read_jsonl, validate_nano_replay


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    prefix_lengths = [int(row["reference"]["shared_prefix_length"]) for row in rows]
    prompt_tokens = [int(row["reference"]["input_tokens"]) for row in rows]
    output_tokens = [int(row["reference"]["output_tokens"]) for row in rows]
    sessions = {str(row["session_id"]) for row in rows}
    sampling = {
        json.dumps(row["sampling_params"], sort_keys=True)
        for row in rows
    }
    return {
        "schema_version": "multitown-nanovllm-replay-summary-v1",
        "requests": len(rows),
        "sessions": len(sessions),
        "reference_input_tokens": sum(prompt_tokens),
        "reference_output_tokens": sum(output_tokens),
        "requests_with_textual_prefix": sum(length > 0 for length in prefix_lengths),
        "shared_prefix_codepoints_median": statistics.median(prefix_lengths) if rows else 0,
        "shared_prefix_codepoints_max": max(prefix_lengths, default=0),
        "distinct_sampling_configs": len(sampling),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    validation = validate_nano_replay(args.input)
    if not validation["passed"]:
        raise SystemExit(json.dumps(validation, ensure_ascii=False, indent=2))
    rows = _read_jsonl(args.input)
    if args.limit is not None:
        if args.limit <= 0:
            parser.error("--limit must be positive")
        rows = rows[: args.limit]
    summary = summarize(rows)
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    if not args.model:
        parser.error("--model is required unless --dry-run is set")

    try:
        from nanovllm import LLM, SamplingParams
    except ImportError as error:
        raise SystemExit(
            "Nano-vLLM is not installed. Install the pinned revision described in "
            "docs/NANOVLLM_REPLAY_BENCHMARK.md."
        ) from error

    prompts = [row["prompt"] for row in rows]
    sampling_params = [
        SamplingParams(
            temperature=float(row["sampling_params"]["temperature"]),
            top_p=float(row["sampling_params"]["top_p"]),
            max_tokens=int(row["sampling_params"]["max_tokens"]),
        )
        for row in rows
    ]
    llm = LLM(
        args.model,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
    )
    llm.generate(["MultiTown neutral warmup."], SamplingParams(max_tokens=1), use_tqdm=False)
    started = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    elapsed = time.perf_counter() - started
    summary.update({
        "model": args.model,
        "elapsed_s": elapsed,
        "reference_output_tokens_per_s": (
            summary["reference_output_tokens"] / elapsed if elapsed else None
        ),
        "outputs": len(outputs),
        "measurement_scope": "compatibility_and_offline_throughput_only",
    })
    rendered = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
