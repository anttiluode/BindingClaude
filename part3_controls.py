"""Registered Part 3 S3/S4: distinct supervised baselines and after-training revision.

python part3_controls.py --mode baselines
python part3_controls.py --mode revision
Default outputs under results/. No changes to S0-S2b gate scores.
"""
import argparse
import json
import os
import numpy as np
from part3 import Binder, world, accuracy, BATCH, STREAM, NTEST

CHECKPOINTS = (500, 2000, 20000)
HIDDEN = 64
UPDATES = 600
BATCH_MLP = 256
LR = .012


def visible(z, covered):
    v = [j for j in range(z.shape[1]) if j not in covered]
    s = np.sign(z[:, v]); s[z[:, v] == 0] = 0
    return s


def codes(v):
    return ((v > 0).astype(np.int64) * (1 << np.arange(v.shape[1], dtype=np.int64))).sum(1)


def lookup(train_v, train_y, test_v):
    n = 1 << train_v.shape[1]
    score = np.bincount(codes(train_v), weights=train_y, minlength=n)
    count = np.bincount(codes(train_v), minlength=n)
    return np.where(count[codes(test_v)] == 0, 0., np.sign(score[codes(test_v)]))


def mlp_train(x, y, seed, updates=UPDATES):
    """One fixed 64-tanh-hidden MLP, minibatch Adam, equal updates for both arms."""
    rng = np.random.default_rng(seed)
    n, d = x.shape
    W = rng.normal(0, 1 / np.sqrt(d), (d,HIDDEN))
    b = np.zeros(HIDDEN)
    v = rng.normal(0, 1 / np.sqrt(HIDDEN), HIDDEN)
    a = np.zeros(1)
    P = [W,b,v,a]
    M = [np.zeros_like(p) for p in P]
    V = [np.zeros_like(p) for p in P]
    target=(y+1)/2
    for t in range(1,updates+1):
        ix = rng.integers(0,n,BATCH_MLP)
        xx, yy=x[ix], target[ix]
        h=np.tanh(xx@P[0]+P[1]); lg=np.clip(h@P[2]+P[3],-30,30)
        pp=1/(1+np.exp(-lg)); err=(pp-yy)/len(ix)
        gh=err[:,None]*P[2][None,:]*(1-h*h)
        grads=[xx.T@gh, gh.sum(0), h.T@err, np.array([err.sum()])]
        for i in range(4):
            M[i]=.9*M[i]+.1*grads[i]
            V[i]=.999*V[i]+.001*grads[i]*grads[i]
            P[i]-=LR*(M[i]/(1-.9**t))/(np.sqrt(V[i]/(1-.999**t))+1e-8)
    return lambda xx: np.where(np.tanh(xx@P[0]+P[1])@P[2]+P[3]>=0,1.,-1.)


def training_snapshots(name, seed):
    """One complete unsupervised training stream with frozen checkpoint copies."""
    import copy
    r=np.random.default_rng(33103 + seed*100019 + (0 if name=='chain' else 191))
    net=Binder(13 if name=='chain' else 11,r)
    chunks=[]; snapshots={}
    for seen in range(BATCH,STREAM+BATCH,BATCH):
        z,exact,covered,_=world(name,BATCH,r)
        chunks.append((z,exact[:,covered[-1]]))
        net.learn(z)
        if seen in CHECKPOINTS:
            snapshots[seen]=(copy.deepcopy(net),
                             np.concatenate([a for a,b in chunks]),
                             np.concatenate([b for a,b in chunks]))
    return snapshots


def hybrid_features(net,z,covered):
    """Only masked recurrent estimates; NEVER the supplied exact hidden state."""
    settled,_=net.settle(z,covered)
    branch=net.product(settled)
    return np.column_stack((visible(z,covered),settled[:,list(covered)],branch*net.cons))


def baselines(seeds=10,ntest=NTEST, updates=UPDATES):
    out={}
    for w in ('chain','context'):
        out[w]={str(n):[] for n in CHECKPOINTS}
        for s in range(seeds):
            trace=training_snapshots(w,s)
            test_rng=np.random.default_rng(900008 + s*43 + (0 if w=='chain' else 101))
            zt,xt,covered,_=world(w,ntest,test_rng)
            yt=xt[:,covered[-1]]
            vt=visible(zt,covered)
            for n in CHECKPOINTS:
                net,z,y=trace[n]
                v=visible(z,covered)
                table=lookup(v,y,vt)
                f=mlp_train(v,y,seed=99+s,updates=updates)
                # Hybrid receives the very same observations, plus only frozen-net
                # masked inferences; it has more input parameters (reported).
                h=hybrid_features(net,z,covered)
                ht=hybrid_features(net,zt,covered)
                g=mlp_train(h,y,seed=99+s,updates=updates)
                one, _=net.settle(zt,covered,mode='one_shot')
                settled,_=net.settle(zt,covered)
                out[w][str(n)].append(dict(seed=s,lookup=accuracy(table,yt),mlp=accuracy(f(vt),yt),
                                            hybrid=accuracy(g(ht),yt),
                                            one_shot=accuracy(one[:,covered[-1]],yt),
                                            settled=accuracy(settled[:,covered[-1]],yt),
                                            n_visible=v.shape[1],n_hybrid=h.shape[1],
                                            direct_params=v.shape[1]*HIDDEN+HIDDEN+HIDDEN+1,
                                            hybrid_params=h.shape[1]*HIDDEN+HIDDEN+HIDDEN+1))
            print('S3',w,'seed',s,flush=True)
    return dict(settings=dict(seeds=seeds,ntest=ntest,updates=updates,lr=LR,
                              hidden=HIDDEN,batch=BATCH_MLP,checkpoints=CHECKPOINTS,
                              model='one tanh hidden, minibatch Adam, fixed updates',
                              lookup='all visible signs including clutter; no target',
                              hybrid='visible + covered settled estimates + frozen branch outputs'),
                results=out)


def revision(seed,ntest=NTEST,post_stream=STREAM):
    """S4: phase change with EMA and reopen, separate from frozen evaluation."""
    from part3 import learn_world
    net,r=learn_world('context',seed,STREAM)
    test_rng=np.random.default_rng(929200+seed)
    z,x,covered,ctx=world('context',ntest,test_rng,flip=True)
    old_c=net.weights().copy()
    mean=old_c.copy()
    sq=np.where(net.cons,net.s2/np.maximum(net.age,1),0.)
    trace=[]; reopened=0; first95=None; intact=[]
    for seen in range(BATCH,post_stream+BATCH,BATCH):
        ztrain,_,_,_=world('context',BATCH,r,flip=True)
        vals=net.product(ztrain)
        prior=net.cons.copy()
        # Existing learned couplings are revised slowly, not re-estimated from all past history.
        mean[prior]=.99*mean[prior]+.01*vals[:,prior].mean(0)
        sq[prior]=.99*sq[prior]+.01*(vals[:,prior]**2).mean(0)
        net.learn(ztrain) # updates / tests only unconsolidated branches
        new=net.cons & ~prior
        mean[new]=net.s1[new]/np.maximum(net.age[new],1)
        sq[new]=net.s2[new]/np.maximum(net.age[new],1)
        # Explicit local 200-experience SE proxy, NOT an exact EMA sample-size test.
        variance=np.maximum(sq-mean*mean,1e-12)
        zscore=np.abs(mean)/np.sqrt(variance/200)
        for j in np.where(net.cons & (zscore<2))[0]:
            net._rewire(j); mean[j]=sq[j]=0.; reopened+=1
        net.override_weights=np.where(net.cons,mean,0.)
        if seen % 200 == 0:
            pred,_=net.settle(z,covered)
            acc_plus=accuracy(pred[ctx==0,6],x[ctx==0,6])['acc']
            acc_minus=accuracy(pred[ctx==1,6],x[ctx==1,6])['acc']
            intact.append(acc_minus)
            if first95 is None and acc_plus>=.95:first95=seen
            trace.append([seen,acc_plus,acc_minus])
    return dict(seed=seed,first95=first95,reopened=reopened,
                before_plus=None, final_plus=trace[-1][1],final_minus=trace[-1][2],
                worst_minus=min(intact),trace=trace)


def revisions(seeds=10):
    return dict(settings=dict(seeds=seeds,after_flip=STREAM,decay=.99,interval=50,
                              reopen_z=2,check_every=200,
                              z_assumption='local 200-sample SE proxy; exploratory interpretation'),
                results=[revision(s) for s in range(seeds)])


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=['baselines','revision'],required=True)
    ap.add_argument('--seeds',type=int,default=10)
    ap.add_argument('--updates',type=int,default=UPDATES)
    ap.add_argument('--out')
    opt=ap.parse_args()
    data=baselines(opt.seeds,updates=opt.updates) if opt.mode=='baselines' else revisions(opt.seeds)
    path=opt.out or 'results/part3_'+opt.mode+'.json'
    os.makedirs(os.path.dirname(path) or '.',exist_ok=True)
    with open(path,'w') as f:json.dump(data,f,indent=2)
    if opt.mode=='baselines':
        for w,by_n in data['results'].items():
            for n,rows in by_n.items():
                print('S3',w,n,{key:round(np.mean([r[key]['acc'] for r in rows]),3)
                                     for key in ('lookup','mlp','hybrid','settled','one_shot')})
    else:
        for r in data['results']:
            print('S4',r['seed'],'first95',r['first95'],'final',r['final_plus'],r['final_minus'],
                  'worst -',r['worst_minus'],'reopened',r['reopened'])

if __name__=='__main__':main()
