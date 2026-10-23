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

Williams & Beer 2010 (partial information decomposition, synergy); Shannon 1949
(one-time pad); Barak et al. 2022 (sparse parity and SGD); ClaudeRepo1 (three-piece gate,
world.html); Välikäsi (keyed residue in LLM notes).
