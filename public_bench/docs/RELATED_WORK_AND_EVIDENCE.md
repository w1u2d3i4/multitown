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
- [AgentDropout](https://arxiv.org/abs/2503.18891) learns to eliminate redundant
  agents and communication links across rounds. It motivates conditional role
  activation, while MultiTown uses OS-isolated roles and preserves a safe
  PlanExecute fallback rather than adopting its communication graph directly.
- [FlowSteer](https://arxiv.org/abs/2602.01664) treats workflow construction as
  multi-turn interaction between a policy and an executable environment. It
  motivates retaining state/action trajectories so the same TeamBench
  controller interface can later support RL instead of a hand-written rule.
- [AgentConductor](https://arxiv.org/abs/2602.17100) learns task-conditioned
  topology density and refines the organization from execution feedback. It
  motivates evaluating topology changes as budgeted actions, while MultiTown's
  current v2.1 keeps a much smaller, auditable Planner/Executor action space.
- [Agent-as-a-Router](https://arxiv.org/abs/2606.22902) frames coding-model
  routing as a Context→Action→Feedback→Context loop and reports that
  execution-grounded history closes an information deficit left by one-shot
  routers. This directly motivates moving beyond the rejected seed-1 static
  category selector.
- [Multi-Agent Routing as Set-Valued Prediction](https://arxiv.org/abs/2606.28925)
  evaluates fixed-catalog routing with both accuracy and cost. Its results
  reinforce a distinction exposed by MultiTown's seed-1 audit: pre-task routing
  and runtime sequential control are different learning problems.
- [MAGRPO](https://arxiv.org/abs/2508.04652) optimizes multi-turn LLM
  collaboration with cooperative multi-agent RL. MultiTown's future RL claim
  is narrower: it will optimize the orchestration controller over frozen role
  models and must beat the hand-written controller in this same harness.
- [Conservative Q-Learning](https://arxiv.org/abs/2006.04779) learns a
  conservative value function to reduce optimistic action selection under
  offline distribution shift. MultiTown v2/v3 adopts the narrower principle of
  penalizing uncertain intervention advantage; it is not a reproduction of the
  full CQL objective.
- [Implicit Q-Learning](https://arxiv.org/abs/2110.06169) avoids querying unseen
  actions while extracting an offline policy. This supports MultiTown's masked
  phase-specific action space and refusal to infer rewards for missing
  counterfactual episodes.
- [Diversified Q-Ensemble](https://arxiv.org/abs/2110.01548) studies ensemble
  disagreement as an offline-RL uncertainty signal. MultiTown uses bootstrap
  disagreement in a lower-confidence-bound gate, then checks it under
  leave-one-generator-seed-out predictions.
- [LEVER](https://arxiv.org/abs/2302.08468) learns to rank generated programs
  from execution results. Its verifier lesson motivates using public test and
  command evidence in the recovery-value model instead of treating an LLM
  review verdict as ground truth.

## What “mainstream multi-agent strategy” means here

The literature contains several recurring organization patterns. A framework
name is not itself a controlled baseline: AutoGen and CAMEL, for example, can
express many different conversations. MultiTown therefore compares concrete
organizations, and cites frameworks only to explain the family they belong to.

| Strategy family | Representative work | Mechanism | Same-harness representative in this branch |
| --- | --- | --- | --- |
| Independent ensemble / vote | Multi-agent debate and the independent architecture in *Scaling Agent Systems* | Several agents propose independently, then vote, judge or debate | Context only. Parallel answer aggregation does not preserve TeamBench's single shared workspace and disjoint OS privileges. |
| Central manager / worker | Centralized architecture in *Scaling Agent Systems*; flexible conversations in [AutoGen](https://arxiv.org/abs/2308.08155) | A coordinator decomposes work and assigns execution | **PlanExecute-TB** isolates the value of planning and handoff. |
| Sequential role pipeline / SOP | [MetaGPT](https://arxiv.org/abs/2308.00352) and [ChatDev](https://arxiv.org/abs/2307.07924) | Specialized roles pass artifacts through a fixed workflow | **FixedTeam-TB** is the full Planner -> Executor -> Verifier pipeline; **PlanExecute-TB** and **ExecuteReview-TB** are its role ablations. |
| Peer conversation / role play | [CAMEL](https://arxiv.org/abs/2303.17760) and AutoGen | Agents converse under assigned roles, sometimes without a central controller | Context only. The current benchmark intentionally tests auditable privilege separation rather than free-form shared-history chat. |
| Dynamic selection / pruning | DyLAN and AgentPrune | Select useful agents or prune communication according to the task | **MultiTown-TB** is the direct local dynamic-activation method. It is analogous at the organization level, but is not a reproduction of either algorithm. |
| Execution-feedback topology control | FlowSteer, AgentConductor and Agent-as-a-Router | Change workflow or routing after environment feedback | **MT-Replan-v2.1** is the local deterministic precursor: interrupt a hard failure, replan once, then delegate bounded recovery. It is not yet a learned or RL policy. |

This boundary prevents a misleading claim such as “MultiTown beats MetaGPT”
when MetaGPT has not been run under the same model, task and access contract.
The direct claim is narrower and testable: MultiTown-TB is compared with Solo,
planning-only, review-only and fixed-full organizations in one TeamBench
harness.

## MultiTown comparison matrix

All local headline comparisons use the same 89 evaluable TeamBench test tasks,
task versions, deterministic graders, sandbox image, temperature and per-call
output cap.

| System | Why it is included | Information and organization |
| --- | --- | --- |
| Solo-TB | Canonical mainstream single-agent anchor | One strong agent sees the full specification, edits and tests the workspace, and certifies its result. |
| PlanExecute-TB | Mainstream manager/worker and no-review anchor | A strong Planner transfers a full-spec plan to a weak, brief-only Executor; no Verifier is called. |
| ExecuteReview-TB | Mainstream execution/review and no-planning anchor | A weak Executor works from the brief, then an independent strong Verifier checks the result; no repair loop is added. |
| FixedTeam-TB (`A4`) | Mainstream fixed role pipeline | A strong Planner, weak Executor and independent strong Verifier are always activated, with at most one remediation/reverification loop. |
| MultiTown-TB (`A8`) | Proposed dynamic organization | A weak Executor starts; public runtime evidence selectively activates a strong Planner and/or Verifier and can roll back a failed review. |

Solo-TB and the role-separated methods answer different operational questions.
Solo-TB measures what one capable agent can achieve when it can see, edit and
self-certify. A4-TB/A8-TB additionally enforce separation of duties: no role can
hold all three privileges. The relevant A8-TB claim is therefore conditional:
resource efficiency relative to an always-on team under the same governance
boundary, plus an explicit quality comparison to the unrestricted Solo anchor.

The two new fixed baselines mirror TeamBench's official `team_no_verify` and
`team_no_plan` conditions rather than being tuned after viewing A8 outcomes.
Their protocol and the five-method matrix were frozen before formal invocation
in [`TEAMBENCH_STRATEGY_MATRIX_PROTOCOL.md`](TEAMBENCH_STRATEGY_MATRIX_PROTOCOL.md).

The completed paired result puts **Solo-TB and PlanExecute-TB on the
quality/token Pareto frontier**. PlanExecute-TB produced 18 passes versus 16 for
Solo-TB while using 40.67% fewer tokens, but its paired partial-score interval
crossed zero, so this is not a quality-superiority claim. ExecuteReview-TB was
significantly worse in mean partial score than PlanExecute-TB and used more
tokens; FixedTeam-TB also used substantially more tokens without a clear
partial-score gain. The local evidence therefore supports selective
organization design, not the slogan that adding agents is inherently better.
The full numbers and provenance boundary are in
[`../records/TEAMBENCH_STRATEGY_QUALITY_V2.md`](../records/TEAMBENCH_STRATEGY_QUALITY_V2.md).

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
