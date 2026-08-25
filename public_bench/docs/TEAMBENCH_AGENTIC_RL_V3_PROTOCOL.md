# TeamBench Agentic RL v3 tail-budget protocol

Status: frozen before the first seed-4 model request.

## Motivation and evidence status

V2 obtained two no-regression partial repairs on independent generator seed 3,
but created no new full pass, added 18.98% mean tokens and exceeded the declared
90,000-token budget twice. Seed 3 is now evaluation evidence and is not an
independent holdout for v3.

V3 asks a narrower question: can the learned quality gain be retained while a
tail-calibrated action reserve and request guard reduce cost and eliminate
observed budget overruns? It does not change Planner, Executor or language-model
weights.

## Frozen changes

The paired finite-horizon advantage ensemble, public feature boundary, action
space, reward and deterministic prefix rollback remain identical to v2. Two
budget changes are made:

1. action reserves use the 90th percentile with the `higher` finite-sample
   quantile rule, instead of the 75th percentile;
2. before every model request, a conservative guard upper-bounds prompt tokens
   by UTF-8 payload bytes plus a 4,096-token chat-template margin. It caps the
   completion or emits `DONE` without sending the request when fewer than 64
   completion tokens can fit under the remaining 90,000-token task budget.

Provider-reported usage is audited after every request. The guard is designed
for the pinned llama.cpp byte-fallback tokenizer endpoints. Until a clean
confirmation observes zero overruns, it is called a conservative request guard,
not a generally proven hard cap for arbitrary external providers.

## Model selection

V3 still trains on the 85 complete seed-0/1/2 paired episodes. Seed 3 is not
added to training. The reserve scan is `[0.80, 0.85, 0.90, 0.95]`. The 0.80,
0.85 and 0.90 candidates retain the best leave-one-seed-out pass count; 0.90 is
selected as the highest reserve quantile in that set. The 0.95 candidate is
rejected because it removes both cross-validated new passes.

The frozen policy SHA-256 is
`8dab24919431c7c6b216661439b94a18668f89e385233c646603f0eff57aa507`.
Its file SHA-256 is
`e67144f1ca7d40acaf4b1c2c32677dbd293db4669cd805055a9fd9690c7fb02d`.
The learned action reserves are 29,787 tokens after execution, 26,855 after
replanning and zero after recovery.

Leave-one-seed-out predictions record 15 versus 13 PlanExecute-prefix passes,
0.66407 versus 0.65140 mean partial score, and 56,724 versus 47,368 mean tokens
(+19.75%). This is model-selection evidence, not an online result.

A replay of seed-3 v2 trajectories suggests that the p90 action reserve would
retain both observed quality repairs while reducing token overhead from 18.98%
to 14.76% and avoiding both observed 90k overruns. This is explicitly post-hoc
diagnostic evidence and is not used as confirmation.

## Seed-4 confirmation

Generator seed 4 is the one independent confirmation set. Sampling seed
20260830, task order, endpoints, prompts, OS isolation, per-call completion
limit and same-trajectory prefix grading are frozen. No threshold, feature,
policy weight, prompt or budget parameter may be changed after the first seed-4
request.

The basic gate passes only if all conditions hold:

1. candidate full passes are not below the same-trajectory prefix;
2. candidate mean partial score is strictly higher;
3. mean token overhead is at most 20%;
4. no task exceeds 90,000 provider-reported total tokens;
5. there are zero invocation, sandbox or provider-overrun errors.

The stronger quality claim additionally requires at least one new full pass,
no lost pass and a paired partial-score interval whose lower bound is above
zero. If the basic gate fails, v3 remains a negative result. If only the strong
gate fails, v3 remains a bounded point-estimate candidate and cannot replace
PlanExecute or be called benchmark-best.

## Leakage and claim boundary

Task ID, full specification, expected output, hidden grader, hidden tests and
labels remain excluded at inference time. Hidden scores are terminal rewards
or post-policy evaluation only. Raw prompts, workspaces and requests remain
private.

The valid claim, if the gate passes, is limited to a trained offline sequential
controller in this frozen local TeamBench protocol. It is not evidence of a
public leaderboard win, universal multi-agent superiority, or language-model
weight training.
