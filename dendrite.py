"""Part 2: lateral axons + dendritic branches + three-factor Hebb + turnover.
python dendrite.py  -> applies the Part 2 gates in PREREG.md (CPU, a few minutes)."""
import json
import zlib
import os
from itertools import combinations

import numpy as np

from binding import columns, enumerate_world, sample

M, A, ZCRIT, KMAX, CHUNK = 200, 200, 5.0, 4, 50
SEEDS, NTEST = 10, 2000
KS = {"prod": (2, 3, 4), "lin": (2, 3, 4), "and": (2, 3, 4), "single": (1,), "pair": (2,)}


class Dendrites:
    def __init__(self, n_inputs, kind, rng):
        self.N, self.kind, self.rng = n_inputs, kind, rng
        self.idx = np.full((M, KMAX), -1)
        self.pol = np.ones((M, KMAX))
        self.s1, self.s2, self.age = np.zeros(M), np.zeros(M), np.zeros(M)
        self.cons = np.zeros(M, bool)
        self.rewired = 0
        for b in range(M):
            self._wire(b)

    def _wire(self, b):
        k = self.rng.choice([k for k in KS[self.kind] if k <= self.N])  # T2, D=0 has only 3 inputs
        self.idx[b] = -1
        self.idx[b, :k] = self.rng.choice(self.N, k, replace=False)
        self.pol[b] = self.rng.choice([-1, 1], KMAX)
        self.s1[b] = self.s2[b] = self.age[b] = 0
        self.cons[b] = False

    def branches(self, Z):
        mask = self.idx >= 0
        G = Z[:, np.where(mask, self.idx, 0)]                  # (n, M, KMAX)
        if self.kind in ("prod", "single", "pair"):
            return np.where(mask, G, 1.0).prod(-1)
        if self.kind == "lin":
            return np.where(mask, G, 0.0).sum(-1)
        return np.where(mask, self.pol * G > 0, True).all(-1).astype(float)   # "and"

    def learn(self, Z, y):
        yb = y[:, None] * self.branches(Z)
        self.s1 += yb.sum(0)
        self.s2 += (yb ** 2).sum(0)
        self.age += len(y)
        due = (~self.cons) & (self.age >= A)
        for b in np.where(due)[0]:
            m = self.s1[b] / self.age[b]
            se = np.sqrt(max(self.s2[b] / self.age[b] - m * m, 1e-12) / self.age[b])
            if abs(m) / se > ZCRIT:
                self.cons[b] = True
            else:
                self._wire(b)
                self.rewired += 1

    def predict(self, Z, silence=None):
        w = np.where(self.cons, self.s1 / np.maximum(self.age, 1), 0.0)
        if silence is not None:
            w = np.where(silence, 0.0, w)
        return np.where(self.branches(Z) @ w >= 0, 1, -1)

    def wired_to(self, true_idx):
        t = sorted(true_idx)
        return np.array([sorted(r[r >= 0].tolist()) == t for r in self.idx])


def train(kind, task, D, seed, stream, curve_every=None):
    rng = np.random.default_rng(10_000 * seed + 97 * D + zlib.crc32(f"{kind}{task}".encode()) % 997)
    cols, true = columns(task, D)
    net = Dendrites(len(cols), kind, rng)
    Zc, yc = sample(task, D, 500, rng) if curve_every else (None, None)
    first95, seen = None, 0
    while seen < stream:
        Z, y = sample(task, D, CHUNK, rng)
        net.learn(Z, y)
        seen += CHUNK
        if curve_every and first95 is None and seen % curve_every == 0:
            if (net.predict(Zc) == yc).mean() >= 0.95:
                first95 = seen
    Zt, yt = sample(task, D, NTEST, rng)
    hit = net.wired_to(true) & net.cons
    return dict(acc=float((net.predict(Zt) == yt).mean()), bound=bool(hit.any()),
                knockout=float((net.predict(Zt, silence=hit) == yt).mean()),
                rewired=int(net.rewired), first95=first95)


def d1_exact():
    out, ok = {}, True
    for t in ("T2", "T3"):
        X, y = enumerate_world(t, 0)
        _, true = columns(t, 0)
        s = [float(np.mean(y * X[:, i])) for i in range(X.shape[1])]
        p = [float(np.mean(y * X[:, i] * X[:, j])) for i, j in combinations(range(X.shape[1]), 2)]
        top = float(np.mean(y * np.prod(X[:, true], 1)))
        out[t] = dict(max_single=max(map(abs, s)), max_pair=max(map(abs, p)), top=top)
        ok &= out[t]["max_single"] < 1e-12 and out[t]["max_pair"] < 1e-12 and abs(top - 1) < 1e-12
    return out, ok


def many(kind, task, D=8, stream=20_000):
    return [train(kind, task, D, s, stream) for s in range(SEEDS)]


def main():
    os.makedirs("results", exist_ok=True)
    R, g = {}, {}
    R["D1"], g["D1"] = d1_exact()

    R["D2"] = float(np.mean([r["acc"] for r in many("single", "T1")]))
    g["D2"] = R["D2"] >= 0.95
    R["D2b"] = {f"{k}_{t}": float(np.mean([r["acc"] for r in many(k, t)]))
                for k in ("single", "pair") for t in ("T2", "T3")}
    g["D2b"] = all(abs(v - 0.5) <= 0.03 for v in R["D2b"].values())

    R["D3"], R["D4b"], ok3, okk = {}, {}, True, True
    for t in ("T2", "T3"):
        rs = many("prod", t)
        R["D3"][t] = dict(accs=[r["acc"] for r in rs], n_ge95=sum(r["acc"] >= 0.95 for r in rs),
                          n_bound=sum(r["bound"] for r in rs), rewired=[r["rewired"] for r in rs])
        ok3 &= R["D3"][t]["n_ge95"] >= 9 and R["D3"][t]["n_bound"] >= 9
        R["D4b"][t] = float(np.mean([r["knockout"] for r in rs]))
        okk &= abs(R["D4b"][t] - 0.5) <= 0.03
    g["D3"], g["D4b"] = ok3, okk

    R["D4"] = {t: float(np.mean([r["acc"] for r in many("lin", t)])) for t in ("T2", "T3")}
    g["D4"] = all(abs(v - 0.5) <= 0.03 for v in R["D4"].values())

    R["H_and"] = {t: dict(acc=float(np.mean([r["acc"] for r in rs])), bound=sum(r["bound"] for r in rs))
                  for t in ("T2", "T3") for rs in [many("and", t)]}

    R["D5"] = {}
    for t in ("T2", "T3"):
        for D in (0, 8, 16):
            rs = [train("prod", t, D, s, 40_000, curve_every=200) for s in range(SEEDS)]
            f = [r["first95"] for r in rs]
            reached = [x for x in f if x is not None]
            R["D5"][f"{t}_D{D}"] = dict(first95=f, median=float(np.median(reached)) if reached else None,
                                        reached=len(reached))
            print(t, D, R["D5"][f"{t}_D{D}"]["median"], len(reached), flush=True)

    R["gates"] = {k: ("PASS" if v else "FAIL") for k, v in g.items()}
    json.dump(R, open("results/dendrite.json", "w"), indent=2)
    L = ["| gate | verdict |", "|---|---|"] + [f"| {k} | {v} |" for k, v in R["gates"].items()]
    L += ["", f"D1 exact: {R['D1']}", f"D2 single-input Hebb on T1: {R['D2']:.3f}",
          "D2b blind: " + ", ".join(f"{k} {v:.3f}" for k, v in R["D2b"].items()),
          "D3 dendrites: " + ", ".join(f"{t} mean {np.mean(v['accs']):.3f}, >=95% in {v['n_ge95']}/10, "
                                        f"bound {v['n_bound']}/10" for t, v in R["D3"].items()),
          "D4 linear dendrites: " + ", ".join(f"{t} {v:.3f}" for t, v in R["D4"].items()),
          "D4b knockout: " + ", ".join(f"{t} {v:.3f}" for t, v in R["D4b"].items()),
          "H-and (reported): " + ", ".join(f"{t} {v['acc']:.3f}, bound {v['bound']}/10" for t, v in R["H_and"].items()),
          "", "D5 experiences to 95% (median of seeds that reached it, stream 40,000)", "",
          "| task | D | median | seeds reaching |", "|---|---|---|---|"]
    for k, v in R["D5"].items():
        t, D = k.split("_D")
        L.append(f"| {t} | {D} | {v['median']} | {v['reached']}/10 |")
    open("results/DENDRITE_LEDGER.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
