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

---

# Part 3 — two clocks on the same branches: discover, then settle

Written 2026-10-23, after Part 2 and after Astra's review (consolidated branches are never
re-tested; the gradient comparison isn't fair yet). Pre-registered before any Part 3 code.
Not yet built.

## Idea

- **Learning clock (Part 2, made unsupervised).** During experience every piece is
  observed, including the ones later hidden. A branch's coupling is c_b = ⟨Π_{i∈b} zᵢ⟩, which
  is higher-order Hebb on the joint with no teacher. Turnover as in Part 2: M = 200,
  k ∈ {2, 3, 4}, consolidate at age A = 200 if |z| > 5, otherwise rewire.
- **Thinking clock (SighImageSuper's "pass it back until it stops changing", made
  nonlinear).** At test, covered pieces start at 0. Each sweep sets every covered piece
  xⱼ ← sign(Σ_{b∋j} c_b Π_{i∈b, i≠j} xᵢ), with visible pieces as their sign. Iterate until
  nothing changes (at most 10 sweeps). This descends E = −Σ_b c_b Π_{i∈b} xᵢ, a higher-order
  Hopfield energy whose couplings were found by experience.
- **One-shot** = a single simultaneous sweep with every covered piece held at 0.

Noise σ = 0.35 on ±1 pieces as before; 10 seeds; 2000 test cases; stream of 20,000
experiences.

## Worlds

**W-chain.** Visible a, b, c, d; hidden h1, h2, h3; 6 clutter pieces. Rules:
h1 = a·b, h2 = h1·c, h3 = h2·d (every rule is a 3-body binding). h3 = a·b·c·d·(…) needs a
5-body branch to be read in one shot, beyond k ≤ 4. So h3 can only be filled by settling
through the chain. (h2 has a reachable 4-body shortcut {a,b,c,h2}; h1 needs none.)

**W-context.** Visible b, c, d, e (±1, noisy) and two gate pieces g+, g− (one-hot 0/1,
noiseless: an absent context is 0, not −1); target t; 4 clutter pieces. In context + the
rule is t = b·c; in context − it is t = d·e. A multiplicative branch containing a gate piece
is silent when that context is absent. That is the "conducts only when its other inputs are
present" claim.

## Gates

| gate | test | pass |
|---|---|---|
| S0 | W-chain and W-context: the true bindings ({a,b,h1}, {h1,c,h2}, {h2,d,h3}; {b,c,t} or {g+,b,c,t}, and {d,e,t} or {g−,d,e,t}) are among consolidated branches | ≥ 9/10 seeds |
| S1 settling beats one-shot | W-chain, h3 covered (with h1, h2) | one-shot h3 within 50 ± 3%; settled h3 ≥ 95% in ≥ 9/10 seeds |
| S2 context gating | W-context, t covered | ≥ 95% in each context (mean of seeds) |
| S2b selective knockout | silence every consolidated branch containing {b, c, t} | context + within 50 ± 3%; context − still ≥ 95% |
| S3 baselines (reported) | at 500, 2000 and 20,000 experiences: (i) MLP trained directly on the covered target from visible pieces, (ii) lookup table over the visible pieces' signs, (iii) MLP given the consolidated branches' outputs as extra input features (the fair search + gradient combination) | reported, no threshold |
| S4 revision (reported) | W-context, after 20,000 experiences the rule in context + flips sign (t = −b·c). Consolidated branches keep an exponential running mean (decay 0.99 per 50 experiences) and are re-opened if \|z\| < 2 | experiences until context + is back to ≥ 95%, and whether context − is disturbed; reported |

Predictions written now: S1 one-shot h3 ≈ 50% exactly (no reachable branch carries it);
S3 (i) the MLP needs far more experience for h3 (5-way parity of visible pieces) than for t;
S3 (iii) is the best of the baselines; S4 recovers in roughly A·(a few) experiences because the
same wiring stays useful and only the coupling's sign must change.

## What it would and wouldn't show

It would show experience-found bindings reused by a settling process, gated by context, with
causal knockouts. It wouldn't show biological dendrites, natural images, or that this beats
gradient learning in general; S3 is there to keep that honest.

## Prior art

Sigma-pi units (Rumelhart, Hinton & McClelland 1986); higher-order Boltzmann machines
(Sejnowski 1986); dense associative memory (Krotov & Hopfield 2016); Barak et al. 2022 (hidden
progress of SGD on parity, the reason S3 (i) is reported, not gated).
