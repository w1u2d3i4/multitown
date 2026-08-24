# A28: Agreement-Gated Specialist-First Router

Status: completed; all pre-registered confirmation gates passed.

## Independent confirmation result

On 1,000 paired, independently generated OOD episodes:

| Metric | A8 | A28 | Paired A28 − A8 |
| --- | ---: | ---: | ---: |
| Autonomous success | 18.6% | **30.9%** | **+12.3 pp**, 95% CI **+9.1 to +15.5** |
| Unsafe episode | 14.0% | **11.6%** | −2.4 pp, 95% CI −5.3 to +0.4 |
| Tokens / episode | 3,316.3 | **3,211.2** | **−105.1**, 95% CI **−134.0 to −76.5** |
| Mean return | 1.388 | **2.593** | **+1.204**, 95% CI **+0.949 to +1.459** |

A28 also reduced wrong executions per incident from 2.728% to 2.292%, with
zero invalid actions and zero budget violations. The unsafe point estimate was
lower, but its interval crossed zero; the supported safety statement is that
the frozen noninferiority margins passed, not that unsafe episodes were
significantly reduced.

The selected threshold was `1.01`: every sensor–specialist agreement may skip
review, while every disagreement switches to the learned preferred specialist
before review. The train-learned map routed families 0/2 to the weak specialist
and families 1/3 to the strong specialist.

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

A28 now becomes the immutable fallback policy for the next offline
sequential learner. The RL policy may deviate only on train-supported actions
whose pessimistic advantage over A28 is positive and whose calibrated risk is
within the frozen budget.
