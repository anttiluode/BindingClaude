"""BindingClaude core: world, exact information, five receivers. numpy only."""
from itertools import combinations

import numpy as np

SIGMA = 0.35
TASKS = ("T1", "T2", "T3", "T4")


# ---------------------------------------------------------------- world
def columns(task, D):
    cols = ["e1", "e2", "m"] + (["K"] if task == "T3" else []) + [f"c{i}" for i in range(D)]
    true = {"T1": ["e1"], "T2": ["e1", "e2", "m"], "T3": ["e1", "e2", "m", "K"], "T4": []}[task]
    return cols, [cols.index(c) for c in true]


def make_bits(task, D, e1, e2, s, K, clutter):
    """Observed bit columns and target, given the underlying independent bits."""
    m = s * K if task in ("T3", "T4") else s
    obs = [e1, e2, m] + ([K] if task == "T3" else []) + [clutter[:, i] for i in range(D)]
    y = e1 if task == "T1" else s * e1 * e2
    return np.stack(obs, 1), y


def sample(task, D, n, rng):
    b = lambda size: rng.choice([-1, 1], size=size)
    X, y = make_bits(task, D, b(n), b(n), b(n), b(n), b((n, D)))
    return X + rng.normal(0, SIGMA, X.shape), y


# ---------------------------------------------------------------- exact information
def _H(keys):
    _, c = np.unique(keys, return_counts=True, axis=0)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def mutual_info(y, X):
    if X.shape[1] == 0:
        return 0.0
    return _H(y[:, None]) + _H(X) - _H(np.column_stack([y, X]))


def enumerate_world(task, D):
    """Every assignment of the independent bits (e1, e2, s, K, clutter): uniform, exact."""
    nb = 4 + D
    g = ((np.arange(2 ** nb)[:, None] >> np.arange(nb)) & 1) * 2 - 1
    return make_bits(task, D, g[:, 0], g[:, 1], g[:, 2], g[:, 3], g[:, 4:])


def interaction_order(task, D, max_k=4, tol=1e-12):
    """Smallest subset size with I(y; X_S) > 0, and the information of the full set."""
    X, y = enumerate_world(task, D)
    full = mutual_info(y, X)
    for k in range(1, min(max_k, X.shape[1]) + 1):
        for S in combinations(range(X.shape[1]), k):
            if mutual_info(y, X[:, S]) > tol:
                return k, full
    return None, full


# ---------------------------------------------------------------- receivers
def _ridge(F, y, lam=1.0):
    mu, sd = F.mean(0), F.std(0) + 1e-9
    Fs = np.column_stack([(F - mu) / sd, np.ones(len(F))])
    w = np.linalg.solve(Fs.T @ Fs + lam * np.eye(Fs.shape[1]), Fs.T @ y)
    return lambda G: np.where(np.column_stack([(G - mu) / sd, np.ones(len(G))]) @ w >= 0, 1, -1)


def _pairs(Z):
    i, j = np.triu_indices(Z.shape[1], 1)
    return np.column_stack([Z, Z[:, i] * Z[:, j]])


def fit_lin(Z, y, **_):
    return _ridge(Z, y)


def fit_pair(Z, y, **_):
    f = _ridge(_pairs(Z), y)
    return lambda G: f(_pairs(G))


def fit_oracle(Z, y, true_idx, **_):
    prod = lambda G: np.prod(np.sign(G[:, true_idx]), 1, keepdims=True) if true_idx else np.zeros((len(G), 1))
    f = _ridge(np.column_stack([Z, prod(Z)]), y)
    return lambda G: f(np.column_stack([G, prod(G)]))


def fit_mlp(Z, y, seed=0, hidden=64, steps=600, lr=0.02, **_):
    r = np.random.default_rng(seed)
    d = Z.shape[1]
    P = [r.normal(0, 1 / np.sqrt(d), (d, hidden)), np.zeros(hidden),
         r.normal(0, 1 / np.sqrt(hidden), hidden), np.zeros(1)]
    M = [np.zeros_like(p) for p in P]
    V = [np.zeros_like(p) for p in P]
    t01 = (y + 1) / 2
    for t in range(1, steps + 1):
        h = np.tanh(Z @ P[0] + P[1])
        out = 1 / (1 + np.exp(-(h @ P[2] + P[3])))
        g = (out - t01) / len(y)
        gh = np.outer(g, P[2]) * (1 - h ** 2)
        G = [Z.T @ gh, gh.sum(0), h.T @ g, np.array([g.sum()])]
        for i in range(4):
            M[i] = 0.9 * M[i] + 0.1 * G[i]
            V[i] = 0.999 * V[i] + 0.001 * G[i] ** 2
            P[i] -= lr * (M[i] / (1 - 0.9 ** t)) / (np.sqrt(V[i] / (1 - 0.999 ** t)) + 1e-8)
    return lambda G_: np.where(np.tanh(G_ @ P[0] + P[1]) @ P[2] + P[3] >= 0, 1, -1)


class Branch:
    """Search all subsets of up to max_k pieces. Each keeps a sign-pattern table built on
    one half of the experience and is judged by self-prediction on the other half."""

    def __init__(self, n_cols, max_k=4):
        subs = [S for k in range(1, max_k + 1) for S in combinations(range(n_cols), k)]
        self.subs = subs
        self.pad = np.array([list(S) + [-1] * (max_k - len(S)) for S in subs])
        self.ncell = 2 ** max_k

    def _codes(self, Z):
        b = (Z > 0).astype(np.int64)
        c = np.zeros((len(Z), len(self.subs)), np.int64)
        for j in range(self.pad.shape[1]):
            idx = self.pad[:, j]
            ok = idx >= 0
            c[:, ok] += b[:, idx[ok]] << j
        return c

    def _table(self, codes, y, sub_ids):
        flat = (sub_ids[None, :] * self.ncell + codes).ravel()
        s = np.bincount(flat, np.repeat(y, codes.shape[1]), len(self.subs) * self.ncell)
        return np.where(s >= 0, 1, -1)

    def fit(self, Z, y):
        h = len(y) // 2
        ids = np.arange(len(self.subs))
        cA, cB = self._codes(Z[:h]), self._codes(Z[h:])
        tab = self._table(cA, y[:h], ids)
        correct = tab[(ids[None, :] * self.ncell + cB)] == y[h:, None]
        score = correct.mean(0)
        best = max(range(len(self.subs)), key=lambda i: (round(score[i], 12), -len(self.subs[i]), -i))
        self.chosen = self.subs[best]
        self.score = float(score[best])
        cAll = self._codes(Z)[:, best]
        self.cells = np.where(np.bincount(cAll, y, self.ncell) >= 0, 1, -1)
        self.best = best
        return self

    def predict(self, G, cells=None):
        cells = self.cells if cells is None else cells
        return cells[self._codes(G)[:, self.best]]


def fit_branch(Z, y, **_):
    B = Branch(Z.shape[1]).fit(Z, y)
    f = lambda G: B.predict(G)
    f.model = B
    return f


RECEIVERS = {"lin": fit_lin, "pair": fit_pair, "mlp": fit_mlp, "branch": fit_branch, "oracle": fit_oracle}
