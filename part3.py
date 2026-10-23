"""Part 3: unsupervised discovery and clamped asynchronous recurrent completion.

Execute: python part3.py [--seeds 10] [--stream 20000] [--out results/part3.json]
Gates are fixed in PREREG.md; operational details in PART3_PROTOCOL_LOCK.md.
"""
import argparse
import json
import os

import numpy as np

M, AGE, ZCRIT, KMAX, SIGMA = 200, 200, 5., 4, .35
NTEST, STREAM, SEEDS, BATCH, SWEEPS = 2000, 20000, 10, 50, 10


def world(name, n, rng, flip=False):
    """Return noisy observations, exact bits, covered indices, visible context."""
    bit = lambda shape: rng.choice([-1., 1.], size=shape)
    p = bit((n, 4))
    if name == 'chain':
        a, b, c, d = p.T
        h1, h2 = a*b, a*b*c
        h3 = h2*d
        exact = np.column_stack((p, h1, h2, h3, bit((n, 6))))
        obs = exact + rng.normal(0, SIGMA, exact.shape)
        return obs, exact, (4, 5, 6), np.zeros(n, int)
    if name == 'context':
        b, c, d, e = p.T
        ctx = rng.integers(0, 2, n)
        gp, gm = (ctx == 0).astype(float), (ctx == 1).astype(float)
        target = np.where(ctx == 0, (-1 if flip else 1)*b*c, d*e)
        exact = np.column_stack((p, gp, gm, target, bit((n, 4))))
        obs = exact + rng.normal(0, SIGMA, exact.shape)
        obs[:, 4:6] = exact[:, 4:6]  # absent gate is genuinely zero
        return obs, exact, (6,), ctx
    raise ValueError(name)


class Binder:
    def __init__(self, n_inputs, rng, m=M):
        self.n, self.rng, self.m = n_inputs, rng, m
        self.idx = np.full((m, KMAX), -1, int)
        self.s1 = np.zeros(m)
        self.s2 = np.zeros(m)
        self.age = np.zeros(m, int)
        self.cons = np.zeros(m, bool)
        self.rewired = 0
        for i in range(m):
            self._rewire(i)

    def _rewire(self, i):
        self.idx[i, :] = -1
        k = int(self.rng.choice((2, 3, 4)))
        self.idx[i, :k] = self.rng.choice(self.n, k, replace=False)
        self.s1[i] = self.s2[i] = 0
        self.age[i] = 0
        self.cons[i] = False

    def product(self, z):
        valid = self.idx >= 0
        return np.where(valid[None, :, :], z[:, self.idx.clip(min=0)], 1.).prod(axis=2)

    def learn(self, z):
        val = self.product(z)
        self.s1 += val.sum(0)
        self.s2 += (val**2).sum(0)
        self.age += len(z)
        due = np.where((~self.cons) & (self.age >= AGE))[0]
        for i in due:
            mean = self.s1[i] / self.age[i]
            variance = max(self.s2[i]/self.age[i] - mean*mean, 1e-12)
            if abs(mean) / np.sqrt(variance / self.age[i]) > ZCRIT:
                self.cons[i] = True
            else:
                self._rewire(i)
                self.rewired += 1

    def weights(self, silence=None):
        w = np.where(self.cons, getattr(self, "override_weights", self.s1 / np.maximum(1, self.age)), 0.)
        if silence is not None:
            w = np.where(silence, 0., w)
        return w

    def containing(self, *pieces):
        return np.array([set(pieces).issubset(set(row[row >= 0])) for row in self.idx])

    def exact_count(self, pieces):
        return int(sum(self.cons[i] and set(row[row>=0]) == set(pieces)
                       for i, row in enumerate(self.idx)))

    def field(self, x, target, weights):
        use = (weights != 0) & (self.idx == target).any(axis=1)
        idx = self.idx[use]
        if not len(idx):
            return np.zeros(len(x))
        valid = (idx >= 0) & (idx != target)
        other = np.where(valid[None, :, :], x[:, idx.clip(min=0)], 1.)
        return other.prod(axis=2) @ weights[use]

    def settle(self, observed, covered, mode='settled', silence=None):
        # Covered noisy samples are discarded *before* any model operation.
        x = np.sign(observed)
        x[:, list(covered)] = 0.
        if self.n == 11:  # Context world: exact one-hot gates
            x[:, 4:6] = observed[:, 4:6]
        w = self.weights(silence)
        changes = []
        initial = x.copy()
        if mode == 'one_shot':
            # Jacobi: every target sees the same all-zero hidden state.
            for j in covered:
                f = self.field(initial, j, w)
                x[:, j] = np.sign(f)
            return x, [int(np.count_nonzero(x != initial))]
        for _ in range(SWEEPS):
            old = x.copy()
            for j in covered:  # fixed, predeclared asynchronous order
                f = self.field(x, j, w)
                x[:, j] = np.where(f > 1e-12, 1., np.where(f < -1e-12, -1., x[:, j]))
            ch = int(np.count_nonzero(old != x))
            changes.append(ch)
            if not ch:
                break
        return x, changes


def accuracy(pred, truth):
    """Expected tie-broken accuracy; a zero is an abstention worth 0.5."""
    p, y = np.asarray(pred), np.asarray(truth)
    return dict(acc=float(np.mean((p == y).astype(float) + .5*(p == 0))),
                resolved=float(np.mean(p != 0)),
                correct_resolved=float(np.mean(p[p!=0] == y[p!=0])) if np.any(p!=0) else None)


def learn_world(name, seed, stream=STREAM):
    rng = np.random.default_rng(33103 + seed*100019 + (0 if name == 'chain' else 191))
    binder = Binder(13 if name == 'chain' else 11, rng)
    for _ in range(0, stream, BATCH):
        z, _, _, _ = world(name, min(BATCH, stream - _), rng)
        binder.learn(z)
    return binder, rng


def experiment(name, seed, stream=STREAM, ntest=NTEST):
    binder, rng = learn_world(name, seed, stream)
    z, exact, covered, contexts = world(name, ntest, rng)
    settled, changes = binder.settle(z, covered)
    shot, _ = binder.settle(z, covered, mode='one_shot')
    info = dict(seed=seed, n_consolidated=int(binder.cons.sum()), rewired=binder.rewired,
                changes=changes)
    if name == 'chain':
        truthsets = ((0,1,4), (4,2,5), (5,3,6))
        info['true_counts'] = [binder.exact_count(p) for p in truthsets]
        info['bound'] = all(c > 0 for c in info['true_counts'])
        info['settled'] = {f'h{i}':accuracy(settled[:,j], exact[:,j]) for i,j in enumerate(covered, 1)}
        info['one_shot_h3'] = accuracy(shot[:, 6], exact[:, 6])
    else:
        plus = binder.exact_count((0,1,6)) + binder.exact_count((4,0,1,6))
        minus = binder.exact_count((2,3,6)) + binder.exact_count((5,2,3,6))
        info['true_counts'] = [plus, minus]
        info['bound'] = plus>0 and minus>0
        info['settled'] = {str(c):accuracy(settled[contexts==c,6], exact[contexts==c,6]) for c in (0,1)}
        cut = binder.containing(0,1,6) & binder.cons
        ko, _ = binder.settle(z, covered, silence=cut)
        info['knockout'] = {str(c):accuracy(ko[contexts==c,6], exact[contexts==c,6]) for c in (0,1)}
        gated = ((binder.containing(4,0,1,6) | binder.containing(5,2,3,6)) & binder.cons)
        gated_only, _ = binder.settle(z, covered, silence=~gated)
        gated_cut, _ = binder.settle(z, covered, silence=gated)
        info['gated_only'] = {str(c):accuracy(gated_only[contexts==c,6], exact[contexts==c,6]) for c in (0,1)}
        info['gated_cut'] = {str(c):accuracy(gated_cut[contexts==c,6], exact[contexts==c,6]) for c in (0,1)}
    return info


def summarize(raw, full=True):
    gate = {'S0':'NOT_RUN','S1':'NOT_RUN','S2':'NOT_RUN','S2b':'NOT_RUN'}
    if full and len(raw['chain']) == 10 and len(raw['context']) == 10:
        gate['S0'] = 'PASS' if all(sum(x['bound'] for x in raw[w]) >= 9 for w in ('chain','context')) else 'FAIL'
        gate['S1'] = 'PASS' if abs(np.mean([x['one_shot_h3']['acc'] for x in raw['chain']]) - .5) <= .03 and sum(x['settled']['h3']['acc'] >= .95 for x in raw['chain']) >= 9 else 'FAIL'
        gate['S2'] = 'PASS' if all(np.mean([x['settled'][str(c)]['acc'] for x in raw['context']]) >= .95 for c in (0,1)) else 'FAIL'
        gate['S2b'] = 'PASS' if abs(np.mean([x['knockout']['0']['acc'] for x in raw['context']]) - .5) <= .03 and np.mean([x['knockout']['1']['acc'] for x in raw['context']]) >= .95 else 'FAIL'
    return gate


def run(seeds=SEEDS, stream=STREAM, ntest=NTEST):
    raw = {w:[experiment(w,s,stream,ntest) for s in range(seeds)] for w in ('chain','context')}
    gates = summarize(raw, seeds==10 and stream==STREAM and ntest==NTEST)
    return dict(settings=dict(seeds=seeds,stream=stream,ntest=ntest, sigma=SIGMA,
                              M=M,age=AGE,z_crit=ZCRIT,settling='asynchronous',
                              tie='keep current / 0 -> 0',score='unresolved half credit'),
                gates=gates, worlds=raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds',type=int,default=SEEDS)
    ap.add_argument('--stream',type=int,default=STREAM)
    ap.add_argument('--ntest',type=int,default=NTEST)
    ap.add_argument('--out',default='results/part3.json')
    opts = ap.parse_args()
    data=run(opts.seeds,opts.stream,opts.ntest)
    os.makedirs(os.path.dirname(opts.out) or '.',exist_ok=True)
    with open(opts.out,'w') as f:json.dump(data,f,indent=2)
    print(json.dumps(data['gates'], indent=2))
    for w,rows in data['worlds'].items():
        print(w,'bound',sum(x['bound'] for x in rows),'/',len(rows))
        for key in ('settled','one_shot_h3','knockout','gated_only','gated_cut'):
            if key in rows[0]:
                first=rows[0][key]
                if isinstance(first,dict) and 'acc' not in first:
                    print(' ',key,{k:round(float(np.mean([x[key][k]['acc'] for x in rows])),4) for k in first})
                else:print(' ',key,round(float(np.mean([x[key]['acc'] for x in rows])),4))

if __name__ == '__main__':
    main()
