# Public benchmark selection

## Primary: TeamBench

[TeamBench](https://github.com/ybkim95/TeamBench) evaluates Planner, Executor
and Verifier coordination on software engineering, data engineering and
incident-response tasks. It provides deterministic graders, parameterized
generators, role ablations and a public stratified 90-task test list. These
properties make it possible to compare A4-TB and A8-TB without using an LLM as
the final judge.

The formal comparison excludes `GH120_redis-py_3863`, which the upstream README
marks as under re-curation and forced to score zero. The remaining 89 public
test tasks are never used for A8 threshold selection. Development tasks are
selected deterministically from non-leaderboard templates.

## Secondary: MultiAgentBench/MARBLE

[MultiAgentBench/MARBLE](https://github.com/ulab-uiuc/MARBLE) is the ACL 2025
benchmark for collaboration and competition across bargaining, coding,
database, Minecraft and research scenarios. The released data contains 100
configurations per domain. It is broader and more interactive than TeamBench,
but several research, code-quality and social metrics are produced by an LLM
evaluator. Objective environment outcomes and LLM-judge scores must therefore
be reported in separate columns.

## Excluded from the primary claim

- SOTOPIA is valuable for social interaction but its main social-goal metrics
  rely on model-based evaluation.
- AgentBench and General AgentBench primarily evaluate single general agents;
  they do not isolate organization effects in the same way.
- MP-Bench evaluates failure attribution over existing traces rather than task
  completion, so it is useful for a later controller-diagnosis study, not this
  A4/A8 effectiveness comparison.

## Claim boundary

External results remain separate from the synthetic town evaluation. A positive
TeamBench result would support transfer to public engineering teamwork tasks;
it would not retroactively change the synthetic score or establish trained
Agentic RL. The historical A4/A8 transfer was mixed/negative. The subsequent
frozen, non-RL MT-CapacityRoute seed-5 confirmation is positive at the local
operating-point level (20/89 passes, 0.64180 partial, 48,296 tokens/task), but
its partial-score confidence intervals cross zero and it is not a literature
SOTA claim.
