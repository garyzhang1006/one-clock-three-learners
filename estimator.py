import numpy as np, pandas as pd
def _hat(X):
    U,s,_ = np.linalg.svd(X, full_matrices=False)
    keep = s > s.max()*1e-10
    return U[:,keep], int(keep.sum())
def _design(unit, event, extra=None, include_event=True):
    n=len(unit); u=pd.factorize(unit)[0]
    B=[np.eye(u.max()+1)[u]]
    if include_event:
        e=pd.factorize(event)[0]; E=np.zeros((n,e.max()+1)); E[np.arange(n),e]=1
        B.append(E[:,1:])
    if extra is not None: B.append(np.asarray(extra,dtype=float))
    return np.hstack(B)
def _scan(U,t,ks):
    best=(-2,None)
    for k in ks:
        y=np.log(t-k); p=U@(U.T@y); r=np.corrcoef(y,p)[0,1]
        if r>best[0]: best=(r,k)
    return best
def fit(unit,event,t,extra=None,include_event=True):
    U,rank=_hat(_design(unit,event,extra,include_event))
    lo=t.min()
    r,k=_scan(U,t,np.linspace(-40,lo-0.51,60))
    r,k=_scan(U,t,np.linspace(max(-60,k-3),min(lo-0.51,k+3),60))
    y=np.log(t-k); p=U@(U.T@y)
    r2=1-((y-p)**2).sum()/((y-y.mean())**2).sum()
    return dict(rho=r,r2=r2,k=k,rank=rank,U=U)
def permute_within_unit(unit,t,rng):
    tp=t.copy()
    for u in pd.unique(unit):
        m=(unit==u); v=tp[m].copy(); rng.shuffle(v); tp[m]=v
    return tp
def null_dist(unit,event,t,extra=None,n=300,seed=0):
    rng=np.random.default_rng(seed); U=fit(unit,event,t,extra)['U']
    out=[]
    for _ in range(n):
        tp=permute_within_unit(unit,t,rng); lo=tp.min()
        r,k=_scan(U,tp,np.linspace(-40,lo-0.51,60))
        r,k=_scan(U,tp,np.linspace(max(-60,k-3),min(lo-0.51,k+3),60))
        out.append(r)
    return np.array(out)
def report(name,unit,event,t,extra=None,nperm=300):
    f=fit(unit,event,t,extra); fr=fit(unit,event,t,extra,include_event=False)
    nd=null_dist(unit,event,t,extra,n=nperm)
    z=(f['rho']-nd.mean())/nd.std()
    print(f"{name:26s} n={len(t):5d} units={pd.Series(unit).nunique():3d} events={pd.Series(event).nunique():4d} "
          f"| rho={f['rho']:.4f} R2={f['r2']:.4f} k={f['k']:.2f} | rate-only rho={fr['rho']:.4f} R2={fr['r2']:.4f} "
          f"| null {nd.mean():.4f}+-{nd.std():.4f} max={nd.max():.4f} p={(nd>=f['rho']).mean():.3g} z={z:.1f}")
    return dict(name=name,n=len(t),units=pd.Series(unit).nunique(),events=pd.Series(event).nunique(),
                rho=f['rho'],r2=f['r2'],k=f['k'],rho_rate=fr['rho'],r2_rate=fr['r2'],
                null_mean=nd.mean(),null_sd=nd.std(),null_max=nd.max(),p=(nd>=f['rho']).mean(),z=z)
