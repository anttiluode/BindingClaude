# Part 3 ledger — 2026-10-23

**Status:** The committed Part 3 of [PREREG.md](../PREREG.md) was inspected before implementation. [PART3_PROTOCOL_LOCK.md](../PART3_PROTOCOL_LOCK.md) fixes asynchronous update, masked targets, tie handling, shortcut/context confounds and comparison limitations. Part 1 / 2 results remain untouched.

**Execution:** local CPython + NumPy, 10 deterministic seeds, 20,000 complete experience samples / seed, 2,000 *fresh* test cases / seed, sigma 0.35, M=200 branches of orders 2–4, 200-example consolidation, z > 5, up to 10 asynchronous sweeps. Tests were not independently reproduced by a second execution environment. Accuracy with unresolved zeros is *expected accuracy after random tie-breaking*, not a fabricated +1 response.

Run:
    python part3.py --out results/part3.json
    python part3_controls.py --mode baselines --updates 100 --out results/part3_baselines.json
    python part3_controls.py --mode revision --out results/part3_revision.json
    python -m unittest discover -s tests -p 'test_part3.py'

## Locked gates — frozen binder, ten seeds

| gate | registered criterion | observed | verdict |
| --- | --- | --- | --- |
| S0 | correct chain and both context wiring families present in >=9/10 seeds | **10/10** chain; **10/10** context | **PASS** |
| S1 | one-shot h3 chance ±3%; settled h3 >=95% in >=9/10 seeds | one-shot **50.00%**, settled **99.10%** mean, **10/10** >=95% | **PASS** |
| S2 | context + and - mean accuracy >=95% | **99.5909%**, **99.5892%** | **PASS** |
| S2b | remove branches containing {b,c,t}: context + chance ±3%; context - >=95% | **49.5900%** (+), **99.5892%** (-) | **PASS** |

All h3 one-shot predictions **abstain** (zero) on chain; they receive 50% expected random-tie accuracy by explicit convention. During the first asynchronous sweep, h1, then h2, then h3 resolve; a second sweep detects convergence. This is legitimate *sequential propagation*, not evidence of a slow multi-round attractor requiring many cycles. Given the favourable fixed order, a reverse-order update ablation should be run before making a latency claim.

**Chain per-seed h3:** 99.00, 99.05, 99.40, 99.05, 98.70, 99.20, 99.10, 99.20, 99.25, 99.00%. Each seed found every registered three-site true binding. Accuracy is limited by noise on visible pieces.

**Context audit beyond the gates:** retaining only the learned **explicitly gated** four-site couplings gives 99.5909% (+) and 99.5892% (-). Removing only these four-site couplings leaves **69.30% (+)** and **79.71% (-)**. Ungated three-site correlations are informative but not sufficient for comparable performance; the gated couplings matter causally here. A success without this cut would have been ambiguous.

## S3 — reported baselines, not a superiority gate

A deliberately compute-limited trial, ten seeds, 2,000 held-out samples each: a 64-tanh MLP with mini-batch Adam (256), fixed **100 optimizer updates** for the direct and hybrid arms. **This 100-update choice was made for affordable execution after a small smoke run and is not a preregistered optimizer budget**; it is not the entire SGD learning curve and must not be used to claim a general gradient disadvantage. Lookup uses **all** visible input signs including clutter; hybrid uses the visible input plus covered-piece estimates and 200 learned branch features constructed **after masking the true covered values**.

| world | experiences | lookup | direct MLP | hybrid MLP | frozen settling |
| --- | ---: | ---: | ---: | ---: | ---: |
| chain | 500 | 68.5% | 80.9% | 64.3% | 50.0% |
| chain | 2,000 | 91.7% | 91.3% | 94.3% | 94.3% |
| chain | 20,000 | 99.2% | 91.8% | 99.2% | 99.2% |
| context | 500 | 80.5% | 99.6% | 99.6% | 77.2% |
| context | 2,000 | 98.4% | 99.6% | 99.6% | 89.6% |
| context | 20,000 | 99.6% | 99.6% | 99.6% | 99.6% |

Hybrid MLP has **more parameters and preprocessing compute** than the direct MLP; this is a combined-system feasibility control, **not** a matched-parameter or matched-FLOPs win. Both MLPs receive teacher labels, whereas the binder learns from complete unlabeled scenes. This difference in training supervision and views must remain visible. The trivial finite-world lookup eventually catches up.

## S4 — revision (reported; no pass gate)

In context + only, flip t=b*c to t=-b*c after 20,000 pre-flip experiences. Retain the existing consolidated wiring and update weights with exponential decay **0.99 per 50 examples**, reopening branches when an approximate local z score (200-sample standard-error proxy) falls below 2. Measure every 200 experiences over another 20,000; context - unchanged.

- **1/10** seeds regained >=95% context + (first at **14,600** post-flip samples).
- Remaining **9/10 did not recover**; final context + mean **56.01%**.
- Context - retained final mean **99.61%** (no meaningful observed damage in this tested set).

**Negative result:** the chosen slow EMA plus reopening does not reliably rewrite the old rule. It is not a refutation of other plasticity schedules, nor a success to hide. The S4 z proxy is an implementation choice that requires separate sensitivity tests before any theoretical conclusion.

## Interpretation and limits

Found high-order couplings can function as recurrent completion constraints, and test-time knockout shows specificity. The chain supports fast dependency propagation in a hand-chosen order; it does not establish emergent dendritic biophysics, natural-image completion, long-term iterative purification, or superiority to a lookup or a well-optimized gradient learner. The important separation is **discovering the reusable constraint**, **using it to settle**, and **revising it under change**. This Part 3 passes its frozen completion gates but fails to demonstrate robust ongoing revision.

Raw JSON is emitted by the commands above. This ledger and the scripts are in the pull request; main-branch prior evidence was not overwritten.
