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

---

# Part 2 — dendrites: can lateral axons + nonlinear dendrites + Hebb bind?

Written 2026-10-23, after Part 1's results and before any Part 2 code.

## Claim

In a pure-synergy world, every single-input and pairwise Hebbian correlation is exactly
zero (E[y·xᵢ] = E[y·xᵢxⱼ] = 0 for order ≥ 3), so Hebbian learning over inputs or pairs is
blind to the binding. A dendritic branch that **multiplies** its inputs (coincidence
detection) creates the product locally. A Hebbian synapse from that branch to the soma can
then learn the binding, provided useless branches are rewired (turnover).

## Learner (`dendrite.py`)

- **Lateral axons:** M = 200 branches. Each receives k ∈ {2, 3, 4} inputs (uniform),
  routed at random from all N input pieces.
- **Branch:** b = Π zᵢ over its inputs (multiplicative dendrite).
- **Hebbian soma synapse:** the branch's running mean c = ⟨y·b⟩ since its birth. The teacher
  signal y is the hidden piece, observed during experience (self-prediction). That makes it
  three-factor Hebb, not unsupervised Hebb.
- **Turnover:** at age A = 200 experiences, a branch is consolidated if
  |c| / SE(c) > 5; otherwise it is rewired at random. A branch's weight stays silent (0)
  until consolidated.
- **Soma:** sign(Σ c·b) over consolidated branches (+1 if none).
- Experience is a stream of fresh examples. The test set is 2000 fresh examples, with 10
  seeds.

## Variants

| name | branch computes | role |
|---|---|---|
| H-prod | product of k ∈ {2,3,4} inputs | the claim |
| H-single | one input (k = 1) | Hebb over inputs |
| H-pair | product of exactly 2 inputs | Hebb over pairs |
| H-lin | sum of k ∈ {2,3,4} inputs | linear dendrite: the cut |
| H-and | 1 if all inputs (with random polarity) are positive, else 0 | threshold dendrite, reported only |

## Gates (tasks T1–T3 from Part 1, D = 8 clutter, stream of 20,000 experiences)

| gate | test | pass |
|---|---|---|
| D1 exact blindness | noise-free enumeration, T2 and T3, D = 0 | every E[y·xᵢ] and E[y·xᵢxⱼ] is 0, and E[y·Π true] = 1 (tolerance 1e-12) |
| D2 Hebb works at first order | H-single on T1 | ≥ 95% |
| D2b Hebb over inputs and pairs is blind | H-single and H-pair on T2 and T3 | mean within 50 ± 3% |
| D3 dendrites bind | H-prod on T2 and T3 | ≥ 95% in ≥ 9/10 seeds, and a consolidated branch wired exactly to the true set in ≥ 9/10 seeds |
| D4 cut: linear dendrites | H-lin on T2 and T3 | mean within 50 ± 3% |
| D4b knockout | trained H-prod with its true-set branches silenced at test | mean within 50 ± 3% |
| D5 cost | experiences until test accuracy (500 fresh examples, checked every 200) first reaches 95%, H-prod, T2 and T3, D ∈ {0, 8, 16}, stream of 40,000 | reported |

D5 prediction, written now: a random branch hits the true set with probability
p = (1/3)/C(N, k_true), and all non-consolidated branches are re-drawn every A experiences.
So the expected experiences ≈ A/(M·p) = A·3·C(N, k)/M. For T3: about 1,500 at D = 8 and
about 15,000 at D = 16. That is linear in the number of candidate wirings, unlike Part 1's
search (whose sample need was near-flat), because here each candidate costs A
experiences of on-line testing.

H-and is reported only. A single threshold branch carries just 2^(1−k) of a k-way parity,
so it should rarely pass the consolidation test alone.
