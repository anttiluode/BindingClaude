# BindingClaude — pre-registration

Committed before any code. 2026-10-23.

## The claim under test

Binding (ClaudeRepo1's eye × eye × mouth) and third-party residue (Välikäsi's note) are
the same quantity: **synergistic information**, which exists in a combination of parts
and in no part alone. Two consequences are tested:

1. A third party's message is usable by a receiver only if the receiver holds what the
   message is synergistic with (the key). Without it, no reader of any kind can use it.
2. A receiver can find the binding itself, with no product feature supplied, and say
   which parts bind. Its sample cost grows with the number of candidate groupings
   (the "search cost" ClaudeRepo1 left open).

## World

Pieces are bits x ∈ {−1, +1}, seen as noisy evidence z = x + N(0, 0.35²). The reader of a
single piece is supplied (sign); **the binding is not**. B's world bit s ∈ {±1} is B's
secret (Välikäsi's Z). Two visible pieces e1, e2; target y = the hidden mouth. D clutter
pieces, independent and uniform. All bits are uniform and independent unless stated.

| task | receiver sees | target | predicted interaction order |
|---|---|---|---|
| T1 control | e1, e2, m = s, clutter | y = e1 | 1 |
| T2 plain message | e1, e2, m = s, clutter | y = s·e1·e2 | 3 |
| T3 keyed message, key held | e1, e2, m = s·K, K, clutter | y = s·e1·e2 = m·K·e1·e2 | 4 |
| T4 keyed message, no key | e1, e2, m = s·K, clutter | y = s·e1·e2 | none (0 bits from every subset) |

T3/T4 are Välikäsi in miniature: A's message carries B's secret only relative to A's
context K.

## Receivers

- **lin**: ridge classifier on the evidence (first order).
- **pair**: ridge on evidence plus all pairwise products.
- **mlp**: one hidden layer of 64 tanh units, Adam, full batch; a generic nonlinear learner.
- **branch**: exhaustive search over subsets of up to 4 pieces. Each subset keeps a table
  from sign patterns to the target, built on half the training data and kept only by its
  self-prediction score on the other half; the best subset is used. No product supplied:
  the table can hold any Boolean function of its pieces.
- **oracle**: ridge with the true product supplied (upper reference).

Training n ∈ {32, 64, 128, 256, 512, 1024, 2048}, fresh test set of 2000, 10 seeds,
clutter D ∈ {0, 8, 16}.

## Gates

| gate | test | pass |
|---|---|---|
| **B0 exact information** | noise-free discrete version, D ∈ {0, 8}: I(y; S) by enumeration for every subset S; interaction order = smallest S with I > 0 | orders exactly 1, 3, 4, none for T1–T4 (tolerance 1e-12) |
| **B1 readers not broken** | n = 256, D = 8 | lin ≥ 95% on T1 |
| **B1b first order is blind** | n = 2048, D = 8 | lin and pair within 50 ± 3% on T2 and T3 |
| **B2 no key, no use** | T4, n = 2048, D = 8 | lin, pair, mlp, branch all within 50 ± 3% |
| **B3 self-found binding** | T2 and T3, n = 2048, D = 8 | branch ≥ 95%, and its chosen subset is exactly the true set in ≥ 9/10 seeds for each task |
| **B3c cut the binding** | the B3 branch with its table re-paired (cells shuffled), and with K removed at test (T3) | both within 50 ± 5% |
| **B4 search cost** | samples to reach 95% test accuracy, branch and mlp, T2 and T3, D ∈ {0, 8, 16} | reported, no threshold |

B4 prediction, written now: the branch's required n grows roughly with
2^k · log(number of candidate subsets), so it rises slowly with D. The mlp's required n
grows faster with D for k = 4 than for k = 3 (sparse parity is hard for gradient
learners: Barak et al. 2022).

## What this does not show

- It is a toy with uniform bits, so synergy is pure by construction. Natural data mixes
  unique, redundant and synergistic information.
- The branch is an exhaustive search. It names the binding, but it doesn't scale past a
  few dozen pieces; B4 measures how fast it stops scaling.
- It does not show that an LLM builds such a binding. The LLM test (does C's hidden state
  ever make m·K linearly readable when given the key?) is the follow-up in Välikäsi.

## Prior art

Partial information decomposition and synergy (Williams & Beer 2010); XOR as the canonical
synergistic case; one-time pad (Shannon 1949); sparse parity learning (Barak et al. 2022);
ClaudeRepo1's three-piece gate and world.html.
