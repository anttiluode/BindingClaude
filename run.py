"""Applies the gates in PREREG.md. CPU, about two minutes.  python run.py"""
import json
import os

import numpy as np

from binding import RECEIVERS, TASKS, columns, interaction_order, sample

SEEDS, NTEST, NS, DS = 10, 2000, [32, 64, 128, 256, 512, 1024, 2048], [0, 8, 16]


def run(task, D, n, rec, seed):
    rng = np.random.default_rng(1000 * seed + 7 * n + D)
    cols, true = columns(task, D)
    Z, y = sample(task, D, n, rng)
    Zt, yt = sample(task, D, NTEST, rng)
    f = RECEIVERS[rec](Z, y, true_idx=true, seed=seed)
    out = dict(acc=float((f(Zt) == yt).mean()))
    B = getattr(f, "model", None)
    if B is not None:
        out["chosen"] = [cols[i] for i in B.chosen]
        out["exact"] = sorted(B.chosen) == sorted(true)
        out["cut_repair"] = float((B.predict(Zt, rng.permutation(B.cells)) == yt).mean())
        if "K" in cols:
            Zk = Zt.copy()
            Zk[:, cols.index("K")] = rng.choice([-1, 1], NTEST) + rng.normal(0, 0.35, NTEST)
            out["cut_key"] = float((B.predict(Zk) == yt).mean())
    return out


def mean(task, D, n, rec, key="acc"):
    return float(np.mean([run(task, D, n, rec, s)[key] for s in range(SEEDS)]))


def main():
    os.makedirs("results", exist_ok=True)
    R, g = {}, {}

    R["B0"] = {f"{t}_D{D}": interaction_order(t, D) for t in TASKS for D in (0, 8)}
    want = {"T1": 1, "T2": 3, "T3": 4, "T4": None}
    g["B0"] = all(R["B0"][f"{t}_D{D}"][0] == want[t] for t in TASKS for D in (0, 8))

    R["B1"] = mean("T1", 8, 256, "lin")
    g["B1"] = R["B1"] >= 0.95
    R["B1b"] = {f"{t}_{r}": mean(t, 8, 2048, r) for t in ("T2", "T3") for r in ("lin", "pair")}
    g["B1b"] = all(abs(v - 0.5) <= 0.03 for v in R["B1b"].values())
    R["B2"] = {r: mean("T4", 8, 2048, r) for r in ("lin", "pair", "mlp", "branch", "oracle")}
    g["B2"] = all(abs(R["B2"][r] - 0.5) <= 0.03 for r in ("lin", "pair", "mlp", "branch"))

    R["B3"], R["B3c"] = {}, {}
    ok3, okc = True, True
    for t in ("T2", "T3"):
        rs = [run(t, 8, 2048, "branch", s) for s in range(SEEDS)]
        acc = float(np.mean([r["acc"] for r in rs]))
        exact = int(sum(r["exact"] for r in rs))
        R["B3"][t] = dict(acc=acc, exact_of_10=exact, chosen=[r["chosen"] for r in rs])
        ok3 &= acc >= 0.95 and exact >= 9
        R["B3c"][t] = dict(repaired=float(np.mean([r["cut_repair"] for r in rs])))
        if t == "T3":
            R["B3c"][t]["key_removed"] = float(np.mean([r["cut_key"] for r in rs]))
        okc &= all(abs(v - 0.5) <= 0.05 for v in R["B3c"][t].values())
    g["B3"], g["B3c"] = ok3, okc

    R["B4"] = {}
    for t in ("T2", "T3"):
        for D in DS:
            for rec in ("branch", "mlp"):
                curve = {n: mean(t, D, n, rec) for n in NS}
                hit = next((n for n in NS if curve[n] >= 0.95), None)
                R["B4"][f"{t}_D{D}_{rec}"] = dict(curve=curve, n95=hit)
                print(t, D, rec, hit, flush=True)

    R["gates"] = {k: ("PASS" if v else "FAIL") for k, v in g.items()}
    json.dump(R, open("results/results.json", "w"), indent=2, default=str)

    L = ["| gate | verdict |", "|---|---|"] + [f"| {k} | {v} |" for k, v in R["gates"].items()] + ["| B4 | reported |"]
    L += ["", "B0 interaction order (D=0, D=8): " + ", ".join(
        f"{t} {R['B0'][f'{t}_D0'][0]}/{R['B0'][f'{t}_D8'][0]}" for t in TASKS),
          f"B1 lin on T1: {R['B1']:.3f}",
          "B1b first order on T2/T3: " + ", ".join(f"{k} {v:.3f}" for k, v in R["B1b"].items()),
          "B2 no key (T4): " + ", ".join(f"{k} {v:.3f}" for k, v in R["B2"].items()),
          "B3 branch: " + ", ".join(f"{t} {v['acc']:.3f}, exact subset {v['exact_of_10']}/10" for t, v in R["B3"].items()),
          "B3c cut: " + ", ".join(f"{t} {k} {x:.3f}" for t, v in R["B3c"].items() for k, x in v.items()),
          "", "B4 training samples to reach 95% (grid up to 2048; - = not reached)", "",
          "| task | D | branch | mlp |", "|---|---|---|---|"]
    for t in ("T2", "T3"):
        for D in DS:
            b, m = (R["B4"][f"{t}_D{D}_{r}"]["n95"] for r in ("branch", "mlp"))
            L.append(f"| {t} | {D} | {b or '-'} | {m or '-'} |")
    open("results/LEDGER.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
