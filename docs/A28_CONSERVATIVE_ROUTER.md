# A28: Agreement-Gated Specialist-First Router

Status: confirmation protocol frozen before opening its independent OOD bank.

## Predecessor and one permitted change

A26 was a valid negative result on its once-opened OOD split: autonomous
success was 18.9% versus 18.7% for A8 (paired 95% CI −2.2 to +2.6 percentage
points), while unsafe episodes were 18.3% versus 16.2%. It failed both the
success and unsafe gates.

Trace-level reasoning identified one design error: when sensor and learned
specialist disagreed, A26 reviewed the lower-accuracy sensor first and could
execute it after one false-positive review. A28 makes one structural change:

```text
sensor != preferred specialist
    A26: review(sensor) -> maybe execute(sensor)
    A28: connect(preferred) -> review(preferred) -> maybe execute(preferred)
```

If sensor and specialist agree, the calibration threshold still controls
whether review may be skipped. A reviewer rejection calls the alternate worker,
reconnects, and reviews again; human handoff remains the final fallback.

## Independent protocol

- non-leaking environment: identical code and distribution contract to A26;
- train: 3,000 episodes, new seed offset `51_000_000`;
- calibration: 500 episodes, new seed offset `52_000_000`;
- confirmation: 1,000 OOD episodes, new seed offset `53_000_000`;
- threshold grid, selection order, and A8-relative safety margins: unchanged
  from A26;
- confirmation bank is constructed only after `selection-lock.json` exists;
- the consumed A26 test rows are not used for A28 threshold selection.

A28 adds one stricter gate: mean tokens per episode must not exceed A8. All A26
claim boundaries remain in force.

```bash
multitown-a28-conservative-router \
  --output /path/outside/the/repository/a28-confirmation
```

## Claim boundary

A28 is a learned router plus a deterministic safety workflow. It is not full
Agentic RL and trains no language-model weights. A passing result can establish
only an improvement on this synthetic non-leaking fixture. It cannot establish
general multi-agent superiority; the separate `public-bench` branch carries
TeamBench evidence.

If A28 passes, it becomes the immutable fallback policy for the next offline
sequential learner. The RL policy may deviate only on train-supported actions
whose pessimistic advantage over A28 is positive and whose calibrated risk is
within the frozen budget.
