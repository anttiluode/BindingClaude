# BindingClaude — the other half of the XOR

**Binding and third-party residue are the same quantity: synergistic information.** It
lives in a combination of parts and in no part alone, so it can only be read by someone
holding the other half. This repo tests that on a toy where everything is exact, and asks
whether a receiver can find the binding itself.

Gates were pre-registered ([PREREG.md](PREREG.md)) before any code. numpy only, CPU,
`python run.py` (about 7 minutes). Raw output in [results/](results/).

## Where it comes from

- **Välikäsi.** B's secret colour Z changes A's note to C, yet every first-order reader
  of the note scores chance. Given the context G (B's script and the topic), the note
  carries 1.14–1.63 of 2 bits about Z (0.5B) and 0.65–1.20 (3B), computed exactly from the
  pushed notes. I(Z; N | G) is large while the readable part of I(Z; N) is near zero. The
  note behaves like Z ⊕ K, with the context as the key.
- **ClaudeRepo1's three-piece world.** Every piece and every pair is 50/50; only eye ×
  eye × mouth differs between worlds. Pairwise and linear learners score 50%.
- **The correction from Sol.** A linear probe *can* read an XOR once a network has built
  it. So the question isn't whether information exists, but whether some reader has
  constructed the conjunction, and that needs the other half.

## The mathematics

For pieces X₁…Xₙ and target Y, the **interaction order** is the smallest k such that some
k pieces carry information about Y. Pure k-th order synergy means I(Y; S) = 0 for every set
of fewer than k pieces, while some k pieces give I(Y; S) = H(Y).

The keyed message is the one-time pad. If m = s·K with K uniform and independent, then
I(s; m) = 0 and I(s; m, K) = H(s). For the receiver, the key raises the order: to read
y = s·e1·e2 through m it must bind four pieces (y = m·K·e1·e2), and without K no reader of
any kind can use m.

| task | receiver sees | target | order (exact) |
|---|---|---|---|
| T1 | e1, e2, m = s | e1 | 1 |
| T2 | e1, e2, m = s | s·e1·e2 | 3 |
| T3 | e1, e2, m = s·K, **K** | s·e1·e2 | 4 |
| T4 | e1, e2, m = s·K | s·e1·e2 | none: 0 bits from every subset |

Each piece is seen as noisy evidence (σ = 0.35); D independent clutter pieces are added.
Reading one piece's sign is supplied. **The binding is not.**

## Results

| gate | claim | result | |
|---|---|---|---|
| B0 | exact interaction orders 1, 3, 4, none (D = 0 and D = 8) | 1, 3, 4, none | **PASS** |
| B1 | readers work: linear on T1 | 99.7% | **PASS** |
| B1b | first order is blind to binding | linear and pairwise 49.7–50.4% on T2 and T3 | **PASS** |
| B2 | no key, no use (T4) | linear 50.4, pairwise 50.4, MLP 50.1, branch 49.8% | **PASS** |
| B3 | branch finds the binding itself | T2 99.4%, T3 99.2%; chosen pieces exactly {e1, e2, m} and {e1, e2, m, K} in **10/10** seeds each | **PASS** |
| B3c | cutting the binding returns chance | key removed at test: 49.7%; re-paired table: T3 49.6%, T2 **56.5%** | **FAIL** |

**B3c failed as registered, and the gate was wrong, not the binding.** A single re-pairing
of an 8-cell table is a coarse random Boolean function: its accuracy moves in steps of
1/8, with a per-seed SD of 0.21 (`posthoc_cut.py`, post-hoc). Ten seeds give a standard
error near 0.066, wider than the ±0.05 band I registered. Averaged over 500 re-pairings the
cut gives 49.9% (T2) and 50.3% (T3). The verdict stays FAIL in the table; the cause was
found after the fact.

**B4: samples needed to reach 95%** (training grid up to 2048, mean of 10 seeds; – = not
reached)

| task (order) | clutter D | branch | MLP |
|---|---|---|---|
| T2 (3) | 0 | 32 | 32 |
| T2 (3) | 8 | 64 | 256 |
| T2 (3) | 16 | 64 | 1024 |
| T3 (4) | 0 | 64 | 128 |
| T3 (4) | 8 | 64 | 2048 |
| T3 (4) | 16 | 128 | – |

As predicted: the branch's sample need barely moves with clutter, while the MLP's grows
fast, and faster for order 4 than order 3. The branch pays in compute instead: it scores
every subset of up to 4 of 20 pieces (6,195 candidates). Samples are cheap for it and the
search is not. That is the trade ClaudeRepo1 left open, now measured. The MLP pattern is
the known sparse-parity difficulty (Barak et al. 2022).

## Part 2 — dendrites: lateral axons + nonlinear branches + Hebb

Pre-registered as Part 2 of [PREREG.md](PREREG.md) before any of this code. `python dendrite.py`
(about 2 minutes). Raw: [results/dendrite.json](results/dendrite.json),
[results/DENDRITE_LEDGER.md](results/DENDRITE_LEDGER.md).

**Why Hebb alone can't bind.** In a pure-synergy world every single and pairwise correlation
with the target is exactly zero: E[y·xᵢ] = E[y·xᵢxⱼ] = 0, while E[y·x₁x₂x₃] = 1. A Hebbian
update over inputs or pairs averages to nothing, so no amount of lateral wiring helps on its
own.

**The learner.** 200 dendritic branches, each fed 2–4 pieces routed at random (the lateral
axons). A branch multiplies its inputs (coincidence detection). Its synapse onto the soma
learns ⟨y·b⟩ (three-factor Hebb: y is the hidden piece, observed during experience). At age
200 a branch is consolidated if its correlation is significant (z > 5); otherwise it is
rewired (turnover).

| gate | claim | result | |
|---|---|---|---|
| D1 | exact: all single and pairwise Hebbian correlations vanish | max \|E[y·xᵢ]\| = max \|E[y·xᵢxⱼ]\| = 0, E[y·Π] = 1 | **PASS** |
| D2 | Hebb works at first order (T1) | 99.8% | **PASS** |
| D2b | Hebb over inputs or pairs is blind (T2, T3) | 49.7–50.0% | **PASS** |
| D3 | multiplicative dendrites + Hebb + turnover bind | T2 99.4%, T3 99.2%; ≥ 95% and wired exactly to the true pieces in **10/10** seeds each | **PASS** |
| D4 | cut: linear dendrites (sum instead of product) | 49.3%, 50.7% | **PASS** |
| D4b | knockout: silence the true-set branches at test | 50.6%, 49.5% | **PASS** |

**D5: experiences to 95%** (10 seeds, median), against the registered prediction
mean ≈ A·(number of wirings)/M:

| task (order) | clutter | median | mean | predicted mean |
|---|---|---|---|---|
| T2 (3) | 8 | 600 | 820 | 495 |
| T2 (3) | 16 | 2,200 | 2,360 | 2,907 |
| T3 (4) | 8 | 1,000 | 1,420 | 1,485 |
| T3 (4) | 16 | 4,600 | 8,760 | 14,535 |

The search behaves as a geometric waiting time (T3, D = 16 spans 200 to 23,600 across seeds),
and cost grows with the number of possible wirings, C(N, k). That is the price of finding a
binding by random lateral growth. Part 1's exhaustive search tested all wirings at once and
needed about 100 examples; dendrites test 200 at a time and need thousands of experiences,
but each one is a single local operation.

**Threshold (AND) dendrites, reported only.** A branch that fires when all its inputs agree
carries 2^(1−k) of a k-way parity. For 3-way binding (T2) that was enough: 98.2%, with true-set
branches consolidated in 10/10 seeds, several AND branches with different input polarities
jointly forming the table. For 4-way (T3) it was not: 49.5%, 0/10. I predicted it would rarely
pass for both; it was wrong for order 3.

**Two code changes, both logged.** The seed mix used Python's `hash()`, which varies between
processes; it was replaced with `crc32` before any result was produced. The D5 sweep then
crashed on T2 with no clutter (3 inputs cannot feed a 4-input branch); branch size is now
capped at the number of inputs, which only affects that one cell.

**What Part 2 means.** Lateral axons + Hebb can't bind. Lateral axons + *multiplicative
dendrites* + Hebb + turnover can, and remove the multiplication (or the branch) and the ability
goes. It is the two-layer neuron (Poirazi & Mel 2003) and structural plasticity used as a
binding machine: known parts, now with exact blindness, a cut, a knockout and a measured
search cost. It also suggests a fix for BindingSol's 75% plateau: a gradient-trained sender
gets no signal from lower-order terms, a sampled-and-tested one doesn't need any.

## What it means

1. **A third party's residue is unusable without the key, for every reader.** That's
   exact here (B0, B2), not a power problem. In Välikäsi, C never held B's script or A's
   context, so the chance-level readouts are what this predicts. It isn't proof that
   nothing leaked.
2. **The key raises the binding order.** Reading a keyed message means binding one more
   piece than reading the plain one. Each layer of indirection (B → A → C) adds a piece
   to the conjunction the receiver must find.
3. **A receiver can find and name the binding with no decoder supplied**, and the cost
   is search, not samples. The interpretable part is the answer itself: "these four
   pieces bind".
4. **For Anthropic's workspace limitation** (concepts visible, binding not), the
   prediction is concrete. A bound concept should be linearly readable only after the
   layer where the network has constructed the conjunction, and only if the key-half was
   in context.

## What it does not show

- Uniform independent bits give pure synergy by construction. Natural data mixes unique,
  redundant and synergistic information.
- Exhaustive subset search does not scale to images or language. B4 shows where it stops.
- Nothing here shows an LLM builds such a binding. The direct test, back in Välikäsi:
  give C the key (B's context) and probe each layer for the conjunction.

## Prior art

Williams & Beer 2010 (partial information decomposition, synergy); Poirazi & Mel 2003 (the neuron as a two-layer network); Chklovskii et al. 2004 (structural plasticity and wiring); Shannon 1949
(one-time pad); Barak et al. 2022 (sparse parity and SGD); ClaudeRepo1 (three-piece gate,
world.html); Välikäsi (keyed residue in LLM notes).


## Part 3 — two clocks, one learned constraint graph (2026-10-23)

See the **[Part 3 ledger](results/PART3_LEDGER.md)**, the **[locked inference protocol](PART3_PROTOCOL_LOCK.md)**, and [PREREG.md](PREREG.md). Python + NumPy, no changes to Parts 1–2.

The *learning clock* watches complete examples, samples high-order wire sets, consolidates statistically persistent products and rewires unsupported branches. It does not use a supervised target at this stage. The *thinking clock* clamps noisy visible signs and iteratively updates hidden sites against **frozen** learned couplings, with targets masked to zero before inference. It uses asynchronous one-bit updates and keeps zero on a tie.

Ten-seed results at 20,000 experiences and 2,000 fresh tests per seed: **S0/S1/S2/S2b all PASS**. The chained final hidden piece goes from **50% expected chance on unresolved one-shot** to **99.10%** after settling; context outcomes are **99.59%** each; selective knockout drops only the intended context to **49.59%**. The chain resolves in one correctly ordered asynchronous sweep followed by a convergence check, not an extended relaxation process.

The important caveats are as valuable as the passes: a regular lookup reaches **99.2% / 99.6%** with enough examples; the 100-update direct and branch-assisted MLPs are **not capacity-matched**; ordinary ungated correlations partly solve context; and **S4 revision fails in 9/10 seeds** after flipping the learned rule, although the untouched context stays accurate. These toy results do **not** establish image completion, biology, or gradient learning's general inferiority.

Reproduce locally:

\`\`\`bash
python part3.py
python part3_controls.py --mode baselines --updates 100
python part3_controls.py --mode revision
python -m unittest discover -s tests -p 'test_part3.py'
\`\`\`

For the baseline experiment, 100 steps is an explicitly **post-smoke compute limit**, not a fully explored or preregistered SGD training budget. The next honest question is whether the learned recurrent structure makes robust **revision** possible and whether its benefits survive neutralizing favourable update order and increasing task complexity.
