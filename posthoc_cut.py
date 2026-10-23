"""Post-hoc, NOT pre-registered (written after B3c failed). One re-pairing of a
2^k-cell table is a coarse random Boolean function: per-seed accuracy moves in steps of
1/2^k, so 10 seeds have a standard error near 0.06, above the registered tolerance of
0.05. Here each seed's cut is averaged over 500 re-pairings of only the cells in use."""
import numpy as np
from run import SEEDS, NTEST
from binding import columns, sample, fit_branch

for t in ("T2", "T3"):
    single, many = [], []
    for s in range(SEEDS):
        rng = np.random.default_rng(1000 * s + 7 * 2048 + 8)
        cols, true = columns(t, 8)
        Z, y = sample(t, 8, 2048, rng)
        Zt, yt = sample(t, 8, NTEST, rng)
        B = fit_branch(Z, y).model
        k = len(B.chosen)
        code = B._codes(Zt)[:, B.best]  # once; only the cells change below
        accs = []
        for _ in range(500):
            cells = B.cells.copy()
            cells[: 2 ** k] = rng.permutation(cells[: 2 ** k])
            accs.append((cells[code] == yt).mean())
        many.append(np.mean(accs))
        single.append(accs[0])
    print(f"{t}: one re-pairing per seed {np.mean(single):.3f} (seed SD {np.std(single):.3f}); "
          f"mean of 500 re-pairings {np.mean(many):.3f}")
