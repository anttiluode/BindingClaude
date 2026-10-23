# Part 3 implementation protocol (locked before code or results)

This document operationalizes, but does not replace or rewrite, the committed **Part 3** section of [PREREG.md](PREREG.md). The existing gates, ten seeds, 2,000 fresh tests, sigma = 0.35, 200 branches, ages of 200 experiences, k in 2..4, 20,000 experiences and fixed test worlds remain intact. All post-hoc changes must be identified as such.

## What the two clocks mean

1. **Learning:** stream complete observations, including h1/h2/h3 or t. The model never receives a target label or supervised error while learning branch couplings. Use c_b = average product of noisy observed branch pieces; only significant branches consolidate. Frozen snapshot precedes tests. Random turnover probes candidate wire sets; consolidated weights do not quietly change during ordinary testing.
2. **Settling:** remove all hidden target evidence *before* updating, clamp visible evidence once to sign(z) (gates are exactly 0/1), initialize hidden variables to 0. Update covered variables in a fixed deterministic order, one site at a time (asynchronous Gauss-Seidel); each update uses only *other* sites and frozen consolidated couplings. Stop on no changes or after 10 sweeps. Zero local field **retains** previous value (including 0). The one-shot control uses one **simultaneous** evaluation from the same all-zero hidden state, never the partially updated values. This matters on W-chain.
3. **Scoring:** unresolved zero is not silently mapped to +1. Report both resolution rate and accuracy giving an unresolved ±1 bit half credit (expected random tie-breaking). Accuracy on resolved samples is separately reportable. Final truth is used for evaluation only.
4. **Causality:** branch-knockout sets their couplings to zero only at test, not the underlying examples. No learned branch gets access to the correct hidden target during inference.

## Confounds and expected difficulties recorded up front

- W-chain also permits four-body `{a,b,c,h2}` and `{h1,c,d,h3}` shortcuts. They cannot connect *all-visible* inputs to h3 in one shot (that would require five sites including h3), but they can shorten settling paths. The critical comparison is one-shot vs settled h3, not that **only** the three listed wires can ever carry information.
- W-context has ungated marginal three-body correlations with `{b,c,t}` and `{d,e,t}`, each at strength 1/2, **as well as** four-body gated correlations. A context success alone does not prove that the four-body gated branches were necessary. Report an additional gated-branch-only and gated-branch-knockout audit without changing registered S2/S2b verdicts.
- A visible-only lookup is allowed the same supervised examples as the visible-only MLP; the unsupervised binder sees full scenes, which is a different information budget. Report comparison honestly, not as a matched-information superiority claim.
- In W-chain, the visible-only target h3 is **four-way parity** a*b*c*d; it needs a five-body interaction only when the **target** h3 is included in an unsupervised energy term. The wording in Part 3's S3 prediction is corrected here, not changed retroactively in the preregistration.
- A learned feature used by the hybrid must never contain a ground-truth hidden value during prediction; use the settled estimate from a masked example, and train/evaluate the hybrid with the same feature construction. Report feature provenance.
- The S4 revision is a distinct adaptation phase after the frozen S0–S3 test, and is not pooled with frozen scores.

## Gate handling

Preserve S0/S1/S2/S2b pass criteria verbatim, explicitly report each seed, and mark a failure as a failure. S3 and S4 remain **reported only**, not reinterpreted as pass gates. No changing thresholds after seeing held-out results.
