"""Prompt contracts for cost-aware MAGRPO writing experiments."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _tldr_concise(example: dict[str, Any]) -> str:
    prompt = example.get("prompt", "")
    if not prompt:
        return "Error: No prompt provided."
    return f"""Summarize the Reddit post below.

{prompt}

Output contract:
- Write only the summary: no heading, preface, analysis, or bullet list.
- Use 12-20 whitespace-delimited words.
- State only the central situation and request.
- A separate agent will add details, so do not elaborate.
"""


def _tldr_detailed(example: dict[str, Any]) -> str:
    prompt = example.get("prompt", "")
    if not prompt:
        return "Error: No prompt provided."
    return f"""Write the complementary detailed summary of the Reddit post below.

{prompt}

Output contract:
- Write only the summary: no heading, preface, analysis, or bullet list.
- Use 40-60 whitespace-delimited words, about 2-3 times a partner's 12-20 words.
- Add distinct factual details instead of repeating only the central situation.
- Naturally use at least three transition expressions chosen from: furthermore,
  however, therefore, for example, in addition, as a result, in summary.
- Do not mention the partner or these instructions.
"""


def _arxiv_background(example: dict[str, Any]) -> str:
    abstract = example.get("abstract_text", "")
    if not abstract:
        return "Error: No abstract provided."
    return f"""Expand this scientific abstract with background and motivation only.

Abstract:
{abstract}

Output contract:
- Write 135-165 whitespace-delimited words in one paragraph.
- Exclude methods, results, contributions, and implications.
- Use several natural transition expressions and no heading or preface.
"""


def _arxiv_method(example: dict[str, Any]) -> str:
    abstract = example.get("abstract_text", "")
    if not abstract:
        return "Error: No abstract provided."
    return f"""Expand this scientific abstract with methods and implications only.

Abstract:
{abstract}

Output contract:
- Write 135-165 whitespace-delimited words in one paragraph.
- Exclude general background and motivation.
- Cover framework, method, contribution, and implications with several natural
  transition expressions; use no heading or preface.
"""


def budgeted_formatters(dataset: str) -> list[Callable[[dict[str, Any]], str]]:
    if dataset == "tldr":
        return [_tldr_concise, _tldr_detailed]
    if dataset == "arxiv":
        return [_arxiv_background, _arxiv_method]
    raise ValueError(f"unsupported writing dataset: {dataset}")
