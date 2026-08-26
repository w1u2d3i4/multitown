# External Agentic Benchmark Protocol

This document keeps the external Agentic RL work separate from the TeamBench
evidence on the `public-bench` branch and from MultiTown's synthetic town
controller experiments. It freezes the data surface before any optimization is
attempted.

## Current data status

| Method family | Public tasks available locally | Audited rows | Exact-paper status |
| --- | --- | ---: | --- |
| AgentConductor | APPS, HumanEval, MBPP, LiveCodeBench release v4, CodeContests | 10,000 APPS; 164 HumanEval; 500 MBPP test; 713 LCB-v4; 13,610 CodeContests across train/validation/test | Data available; official training code and generated topology-SFT corpus were not released/found |
| MAGRPO | TLDR and arXiv Abstract | 129,722 TLDR; 151,177 arXiv across downloaded splits | Official CoMLRL and collaborative-writing code available; this is the first real-gradient training target |
| MATTRL | RareBench public Task 4 and SuperGPQA | 1,122 RareBench cases; 26,529 SuperGPQA questions | Executable reimplementation only; HLE is gated and the paper's 2,185-case RareBench split is not public |

The audit reports `ready_with_declared_gaps`: every payload used by the local
protocol is present and structurally valid, while unavailable paper-only inputs
remain explicit blockers to an *exact* reproduction claim.

## Frozen comparisons

### MAGRPO

The official collaborative-writing task is the primary training target because
both the method and implementation are public. The frozen protocol uses two
agents, group size four, one interaction turn, and the repository's task reward.
The initial reproducibility pass uses:

- TLDR `train[0:1000]` and `test[0:1000]`;
- arXiv `train[0:1000]` and `validation[0:1000]`;
- single-agent GRPO, untrained two-agent collaboration, and MAGRPO under the
  same model, decoding, sample, and output-token budgets;
- task reward, structure/consistency components, prompt/completion tokens,
  wall latency, peak memory, and energy where the platform exposes it.

A smaller smoke slice may be used to validate the stack, but it is never
reported as the benchmark result.

### AgentConductor

AgentConductor trains an orchestrator to emit a variable-size layered execution
graph for coding agents. Its paper reports APPS, LiveCodeBench v4, CodeContests,
HumanEval, and MBPP, but no official executable repository or topology-SFT
corpus was found. MultiTown will therefore label its implementation
**AgentConductor-inspired**, not an exact reproduction. The comparison must keep
the execution checker, base model, role prompts, maximum graph nodes, sampling,
and total generated-token budget fixed.

### MATTRL

MATTRL is test-time textual experience construction with fixed model weights,
not gradient RL. The public reimplementation will compare single-agent,
multi-agent discussion without experience, and retrieved textual experience on
the frozen 300-question SuperGPQA subset. RareBench is a separate medical
ranking track; its public 1,122-case payload cannot substantiate the paper's
2,185-case headline. Medical outputs are benchmark predictions, not clinical
advice.

## Reproduce the data audit

```bash
export MULTITOWN_AGENTIC_DATA_ROOT=/path/to/agentic-rl
multitown-agentic-data-audit \
  --data-root "$MULTITOWN_AGENTIC_DATA_ROOT" \
  --output /tmp/multitown-agentic-data-manifest-v1.json
sha256sum /tmp/multitown-agentic-data-manifest-v1.json
```

The expected manifest digest and per-payload aggregate digests are frozen in
[`benchmarks/agentic_rl/protocol_v1.json`](../benchmarks/agentic_rl/protocol_v1.json).
The manifest records raw file SHA-256 values and the exact 300 SuperGPQA IDs;
the large source datasets themselves are not committed.

## Run the official MAGRPO training adapter

Install MultiTown's optional dependencies, then clone and install the two
official upstream repositories at the revisions recorded by the resulting run:

```bash
python -m pip install -e '.[agentic,dev]'
git clone https://github.com/OpenMLRL/CoMLRL third_party/CoMLRL
python -m pip install --no-deps -e third_party/CoMLRL
git clone https://github.com/OpenMLRL/LLM_Collab_Writing \
  third_party/LLM_Collab_Writing

export MULTITOWN_AGENTIC_DATA_ROOT=/path/to/agentic-rl
export MULTITOWN_AGENTIC_MODEL=/path/to/Qwen3-1.7B
export MULTITOWN_MAGRPO_WRITING_ROOT=third_party/LLM_Collab_Writing

multitown-magrpo \
  --dataset tldr \
  --data-root "$MULTITOWN_AGENTIC_DATA_ROOT" \
  --model "$MULTITOWN_AGENTIC_MODEL" \
  --writing-root "$MULTITOWN_MAGRPO_WRITING_ROOT" \
  --train-samples 8 --eval-samples 4 \
  --generations 4 --max-new-tokens 64 \
  --output /path/to/new-artifact-directory
```

This small command is a pipeline pilot, not the frozen 1,000-example result.
The output directory contains `result.json`, one-second `monitor.jsonl`, and a
resource curve. Add `--save-model` only when the two policy checkpoints are
needed; the runner refuses to overwrite a non-empty artifact directory.

For an equal-output-budget comparison, evaluate two-agent MAGRPO at 64 output
tokens per agent and single-agent GRPO at 128:

```bash
multitown-magrpo-eval --method magrpo --models /path/to/shared-or-agent-models \
  --eval-samples 32 --seeds 20260830 20260831 20260832 \
  --max-new-tokens 64 --output /path/to/base-eval

multitown-magrpo-eval --method grpo --models /path/to/grpo-model \
  --grpo-split-policy strict-delimiter \
  --eval-samples 32 --seeds 20260830 20260831 20260832 \
  --max-new-tokens 128 --output /path/to/grpo-eval

multitown-magrpo-report \
  --base /path/to/base-eval/result.json \
  --magrpo /path/to/magrpo-eval/result.json \
  --grpo /path/to/grpo-eval/result.json \
  --output /path/to/new-comparison
```

The upstream GRPO evaluator falls back to splitting a response at one third of
its character length when `[PARAGRAPH_SPLIT]` is absent. Because that fallback
can mechanically satisfy the length-ratio reward, MultiTown exposes both the
upstream-compatible `official-fallback` policy and the format-conforming
`strict-delimiter` policy. Reports record delimiter adherence and must state
which policy produced a score.

`--prompt-profile budgeted` is an experimental two-agent profile that makes
role and length contracts explicit while retaining the official task reward.
It is a MultiTown optimization, not part of the original MAGRPO protocol.

## Claim boundary

Passing the audit proves only that the inputs are present, parseable, and pinned.
It does not prove method equivalence or a performance improvement. A result is
eligible for the README only after a held-out equal-budget comparison with
multiple seeds, paired uncertainty, failure accounting, and a saved run
provenance record. Until then the branch may say “benchmark integration” or
“inspired reimplementation,” but not “paper reproduction,” “SOTA,” or “best.”

## Primary sources

- [AgentConductor paper](https://arxiv.org/abs/2602.17100)
- [MAGRPO paper](https://arxiv.org/abs/2508.04652)
- [CoMLRL official code](https://github.com/OpenMLRL/CoMLRL)
- [MAGRPO collaborative-writing code](https://github.com/OpenMLRL/LLM_Collab_Writing)
- [MATTRL paper](https://arxiv.org/abs/2601.09667)
- [MATTRL official repository](https://github.com/zhiyuanhubj/MATTRL)
