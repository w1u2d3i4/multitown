# Related work and evidence design

This note explains which comparison practices MultiTown adopts and which
cross-paper numbers it does not treat as directly comparable.

## Primary references

- [TeamBench](https://arxiv.org/abs/2605.07073) evaluates a canonical Solo agent,
  a restricted Solo agent, a full Planner-Executor-Verifier team, and no-plan /
  no-verifier ablations under OS-enforced role separation and deterministic
  graders. MultiTown therefore adds the canonical full-access Solo condition
  instead of treating an always-on team as the only baseline.
- [Towards a Science of Scaling Agent Systems](https://arxiv.org/abs/2512.08296)
  compares single, independent, centralized, decentralized and hybrid agent
  organizations with standardized tools and token budgets. Its central warning
  is directly relevant here: coordination overhead can erase multi-agent gains,
  especially on sequential tool tasks or when the single-agent anchor is
  already capable.
- [DyLAN](https://arxiv.org/abs/2310.02170) combines task-conditioned agent
  selection with early stopping and reports both performance and computation.
  It motivates testing dynamic organization against static teams, but its
  reported benchmark numbers are not copied into MultiTown because the tasks,
  models and role-access contract differ.
- [AgentPrune](https://openreview.net/forum?id=LkzuPorQ5L) evaluates whether
  communication pruning preserves quality while reducing token cost. This
  supports treating quality and resource use as a joint requirement rather
  than claiming success from token reduction alone.
- [Cost-effective Agent Test-Time Scaling](https://research.google/pubs/cost-effective-agent-test-time-scaling/)
  reports cost-performance curves under token and tool-call budgets. MultiTown
  likewise reports a Pareto view and keeps tokens, latency, energy and role
  activations visible.

## MultiTown comparison matrix

All local headline comparisons use the same 89 evaluable TeamBench test tasks,
task versions, deterministic graders, sandbox image, temperature and per-call
output cap.

| System | Why it is included | Information and organization |
| --- | --- | --- |
| Solo-TB | Canonical mainstream single-agent anchor | One strong agent sees the full specification, edits and tests the workspace, and certifies its result. |
| A4-TB | Mainstream fixed multi-agent anchor | A strong Planner, weak Executor and independent strong Verifier are always activated. |
| A8-TB | MultiTown method | A weak Executor starts; public runtime evidence selectively activates a strong Planner and/or Verifier and can roll back a failed review. |

Solo-TB and the role-separated methods answer different operational questions.
Solo-TB measures what one capable agent can achieve when it can see, edit and
self-certify. A4-TB/A8-TB additionally enforce separation of duties: no role can
hold all three privileges. The relevant A8-TB claim is therefore conditional:
resource efficiency relative to an always-on team under the same governance
boundary, plus an explicit quality comparison to the unrestricted Solo anchor.

Required outputs are pass rate, deterministic partial score, paired bootstrap
intervals, exact McNemar tests, tokens, latency, monitored energy, route mix,
role activations and invocation failures. Solo-TB was protocol-frozen on
2026-08-24 before its first formal test invocation; it is a post-hoc contextual
baseline and cannot be used to retune A8-TB on the same test set.

## Claim boundary

Published TeamBench leaderboard rows and results from DyLAN, AgentPrune or other
agent benchmarks use different models, prompts, tools, budgets or task sets.
They are related-work context, not head-to-head evidence that MultiTown wins.
Only same-harness paired rows support a local superiority or efficiency claim.

If A8-TB improves cost but fails the quality non-inferiority rule, the valid
claim is an efficiency/quality trade-off, not “better than multi-agent methods.”
If a system is dominated by another system in both quality and resource use,
the negative result remains part of the public evidence.
