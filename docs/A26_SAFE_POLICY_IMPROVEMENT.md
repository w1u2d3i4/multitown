# A26: Risk-Calibrated Policy Improvement

Status: frozen protocol candidate; no test result is implied by this document.

## Why A26 exists

Earlier controller studies exposed two opposite failures:

- unconstrained PPO improved train-only success and token use but raised unsafe
  episodes from 15.73% to 66.00%;
- a hard review shield recovered safety but could collapse autonomous success or
  add review/token overhead.

A26 tests a smaller and more conservative hypothesis: learn **which specialist
to call**, and allow the controller to skip review only for calibrated
low-severity states where an independent sensor and that specialist agree. All
other states retain review, second-worker escalation, and human fallback.

## Leakage repair

The historical long-horizon fixture derived the correct action from public
family, failure-mode, and severity fields. A sufficiently expressive learner
could therefore infer the label without using the worker evidence. A26 does not
reuse that performance fixture.

`generate_a26_episode()` draws the correct action independently of the public
task fields. Train/dev use 12 family × failure-mode combinations; test uses the
four held-out combinations. Tool reliability, budgets, action costs, and the
eight-action controller interface remain unchanged.

## Frozen protocol

1. Generate 3,000 train episodes with seed offset `41_000_000`.
2. Fit a family-level weak/strong specialist map from train labels only.
3. Generate 500 calibration episodes with seed offset `42_000_000`.
4. Evaluate the fixed threshold grid
   `0,.4,.5,.6,.7,.75,.8,.85,.9,1.01`.
5. A candidate is feasible only when:
   - unsafe-episode rate ≤ calibration A8 + 0.02;
   - wrong executions per incident ≤ calibration A8 + 0.01;
   - autonomous success ≥ calibration A8.
6. Select maximum autonomous success among feasible candidates; tie-break by
   fewer tokens and then higher return.
7. Persist `selection-lock.json` before constructing the OOD test bank.
8. Generate exactly 1,000 test episodes with seed offset `43_000_000`, then run
   A8 and the frozen A26 controller on the same episodes.
9. Report episode-cluster paired bootstrap intervals for success, unsafe
   episodes, tokens, and return.

The source command is:

```bash
multitown-a26-safe-router \
  --output /path/outside/the/repository/a26-run
```

The output directory is intentionally untracked. Public code contains the
generator, protocol, tests, and aggregate claim boundary—not raw trajectories.

## Method boundary

A26 is a **constrained contextual policy-improvement controller**, not full
Agentic RL. Its learned decisions are the specialist map and calibrated review
gate; the remainder of the workflow is deterministic. It trains no language
model weights.

This is intentional. A26 becomes the baseline/fallback for the next sequential
RL experiment. That successor may deviate from A26 only when conservative
advantage and action-support tests pass; otherwise it executes the frozen A26
action. Calling that successor Agentic RL will additionally require multi-step
credit assignment and policy training on train data only.

## Related-work decisions

- [RouteLLM](https://arxiv.org/abs/2406.18665) and
  [FrugalGPT](https://arxiv.org/abs/2305.05176) motivate learned routing and
  cascades across quality/cost trade-offs.
- [SPIBB](https://proceedings.mlr.press/v97/laroche19a.html) motivates reverting
  to a baseline policy when batch support is insufficient.
- [Conservative Q-Learning](https://arxiv.org/abs/2006.04779) and
  [Implicit Q-Learning](https://arxiv.org/abs/2110.06169) motivate pessimistic
  offline value learning without exploiting unseen actions.
- [Conformal Risk Control](https://arxiv.org/abs/2208.02814) motivates a
  calibration-first risk gate. A26's current A8-relative margins are not a
  conformal guarantee; a later version must implement and test the actual risk
  bound before using that name in a claim.
- [Agent Lightning](https://arxiv.org/abs/2508.03680) motivates separating agent
  execution traces from RL training and assigning credit at controller
  decisions.
- [AgentPrune](https://arxiv.org/abs/2410.02506) motivates measuring redundant
  inter-agent communication rather than treating extra messages as free.

No theorem or empirical claim from these works is inherited by MultiTown.

## Claim gates

A26 is a validated synthetic improvement only if all of the following hold on
the once-opened OOD split:

- paired success-difference 95% interval has a positive lower bound;
- unsafe episodes are within A8 + 0.02;
- wrong executions per incident are within A8 + 0.01;
- invalid actions and budget violations are both zero.

Even if all gates pass, the permitted claim is limited to the A26 synthetic
fixture. It is not evidence of general multi-agent superiority and does not
replace the separate TeamBench evidence on `public-bench`.
