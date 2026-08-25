# TeamBench Agentic RL v4 phase-specific value-of-information protocol

Status: frozen before the first v4 public-test model request.

## Motivation

V3 created one new seed-4 pass with no lost pass and no 90k token overrun, but
missed its mean-cost gate by 0.223 percentage points. Its 20 escalations
contained only two quality gains; five stopped after replanning and five more
ran recovery before restoring the exact prefix.

V4 tests whether separate value-of-information margins at each decision phase
can remove these tie-only paths without sacrificing the learned repairs. It is
not another change to model weights, prompts, reward, public features or hidden
grader access.

## Frozen policy selection

Training remains restricted to the 85 complete counterfactual episodes from
development generator seeds 0, 1 and 2. V3 seed 4 is development diagnostic
evidence for this iteration and cannot be claimed as an independent v4 result.

The lower-confidence-bound margin grid is:

- post execution: `[0, 0.01, 0.02, 0.03, 0.04]`;
- post replan: `[0, 0.01, 0.02]`;
- post recovery: `[0, 0.01]`.

Every combination is evaluated with ridge alpha `[10, 100]`, uncertainty beta
`[0.5, 1, 2]` and the v3 p90 action reserves. Selection first forbids a
leave-one-generator-seed-out pass regression, then maximizes passes and the
frozen cost-aware reward. This selects margins 0.01 after execution, 0.01 after
replanning and 0 after recovery, with alpha 100 and beta 0.5.

The frozen internal policy SHA-256 is
`ef779285f7378a134f16a243f922551185b4d68b4069ef06de7f43273fcbbb89`;
the policy file SHA-256 is
`be52071d07c2384eeecbea6de5c095ca54c56e17997fb755544e3e996b5330b6`.

Leave-one-seed-out predictions record 15 versus 13 prefix passes, 0.66407
versus 0.65140 mean partial score and 54,818 versus 47,368 mean tokens
(+15.73%). This is model-selection evidence only.

A post-hoc replay of the already consumed seed-4 v3 trajectories retains the
6-versus-5 pass result and both quality gains while reducing projected overhead
from 20.223% to 11.23%. Because the v4 margins were developed after the v3 run,
this replay is diagnostic and cannot validate v4.

## One public-test confirmation

V4 advances to the frozen 89-task TeamBench public test split because its
seed-0/1/2 cross-validation preserves both quality and the 20% cost gate. It is
run once with sampling seed 20260836. The exact pre-policy workspace and reports
are graded after routing, giving a same-trajectory PlanExecute baseline for all
89 tasks. Task order, Qwen endpoints, prompts, source revision, OS sandbox,
2,048-token per-call completion limit, p90 reserves and conservative 90k
request guard are frozen.

No v4 feature, margin, weight, prompt or budget parameter may be changed after
the first public-test request. Existing public-test results from other methods
are context only; the causal primary comparison is the same-trajectory prefix.

The basic test gate passes only if all conditions hold:

1. candidate full passes are not below the same-trajectory prefix;
2. candidate mean partial score is strictly higher;
3. mean token overhead is at most 20%;
4. no task exceeds 90,000 provider-reported tokens;
5. there are zero invocation, sandbox or provider-overrun errors.

The stronger performance claim additionally requires at least one new full
pass, no lost pass and a paired partial-score interval whose lower bound is
above zero. If the basic gate fails, v4 remains a negative result. Passing the
strong gate supports superiority only to the same-trajectory PlanExecute arm in
this frozen local run; it does not establish a TeamBench leaderboard SOTA or
universal multi-agent superiority.

## Leakage and publication boundary

Task ID, full specification, expected output, hidden tests, hidden grader output
and labels are excluded at inference time. The controller sees only public
brief hashes, public workspace profile, execution/test-command evidence,
elapsed time and token-budget state. Hidden scores are post-policy evaluation
or offline rewards only.

Only code, configuration, aggregate results and hashes may be published. Raw
prompts, requests, messages and generated workspaces remain private.
