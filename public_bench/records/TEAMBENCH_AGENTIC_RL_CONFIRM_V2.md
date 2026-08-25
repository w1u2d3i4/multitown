# TeamBench Agentic RL v2 independent confirmation

Status: completed. V2 passes its frozen quality-first point-estimate gate, but
does not pass the stronger improvement gate and is not a benchmark winner.

Raw prompts, messages, requests and workspaces remain private.

## What changed from v1

V1 trained on 28 episodes and learned to stop everywhere. V2 uses 85 complete
paired episodes across generator seeds 0, 1 and 2. It adds public executor test
signals, a bounded hash of the public brief, workspace-language features, a
64-member bootstrap advantage ensemble and a lower-confidence-bound action
gate. Beta zero is excluded. The policy also learns when to restore the exact
PlanExecute prefix instead of asking an LLM Verifier to estimate recovery
quality.

This is a trained offline sequential orchestration policy. The language-model
weights are unchanged, and hidden grader information is never a policy input.
The exact design and gate were frozen in the
[v2 protocol](../docs/TEAMBENCH_AGENTIC_RL_V2_PROTOCOL.md) before seed 3.

## Training evidence

Leave-one-generator-seed-out selection records 15 versus 13 full passes and
0.66607 versus 0.65140 mean partial score. Mean tokens rise from 47,368 to
58,401 (+23.3%). This selected alpha 100, uncertainty beta 0.5 and margin 0.
It is model-selection evidence, not an online benchmark result.

## Independent seed-3 result

The frozen policy was evaluated once on generator seed 3 with sampling seed
20260829. The exact pre-policy workspace and reports were graded after routing,
so the comparison uses a same-trajectory PlanExecute baseline.

| Method | Passes | Mean partial | Mean tokens | Median latency | p95 latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Same-trajectory PlanExecute | 7 / 30 | 0.68465 | 45,437 | 45.445 s | 103.582 s |
| MT-Agentic-RL-v2 | 7 / 30 | **0.69343** | 54,062 | 50.873 s | 134.130 s |

V2 improves mean partial score by +0.00878, with a 95% paired bootstrap
interval of [0.00000, +0.02355]. It is better on 2 tasks, ties on 28 and is
worse on none. `LH6_audit_trail` improves from 0.27 to 0.45 and
`DIST3_idempotency` from 0.25 to 0.3333. Neither repair creates a new full pass.

The improvement costs +8,625 tokens/task (+18.98%; 95% paired interval +5,034
to +12,453) and +13.12 seconds/task (95% paired interval +5.98 to +23.66).
The policy stops directly on 13 tasks, stops after replanning on 5, keeps 8
recoveries and rolls 4 recoveries back to the byte-identical prefix. It invokes
no LLM Verifier and records zero invocation errors.

## Gate and failure audit

The frozen basic gate passes: no pass regression, positive point-estimate
partial-score delta, mean token overhead below 30%, and zero invocation or
sandbox errors. The stronger gate fails because there is no new full pass. The
quality interval also touches zero, so this run does not establish a
statistically separated quality advantage or justify replacing PlanExecute.

The run reveals a budget-enforcement defect. `MULTI3_polyglot` consumes 91,740
tokens and `EA2_coverage_gap` consumes 93,714. The learned reserve prevents
some unaffordable actions but the 90,000-token setting is checked before a
generation rather than enforced as a hard end-to-end ceiling. V2 therefore
cannot be described as a strict-budget controller.

## Decision

V2 advances the research state: unlike v1, it deploys non-trivial learned
actions and obtains two no-regression partial repairs on an independent seed.
It is still not “best on TeamBench.” The next iteration must first enforce the
hard token ceiling, then reduce tie-only recoveries with a calibrated
probability-of-improvement/risk-of-harm gate. Seed 3 is now evaluation data and
must not be reused as an independent holdout for that iteration.

Exact aggregate values, frozen decisions and artifact hashes are in the
[summary JSON](teambench-agentic-rl-confirm-v2-summary.json).
