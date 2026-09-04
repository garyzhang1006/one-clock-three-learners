# Three controls run on the fitted panels at k=0: (1) the machine coverage pattern and
# a six-unit count imposed on the mammal table, (2) a rival with a second scalar per unit
# (a stretch of the schedule in log time), (3) a ceiling-censoring simulation on children.
import numpy as np, pandas as pd
from itertools import combinations
rng=np.random.default_rng(0)
P={s:pd.read_csv(f'panel_{s}.csv') for s in ['mammals','children','machines']}
def ols(d,event=True):
    y=d.obs.values; U=pd.get_dummies(d.unit).values.astype(float)
    X=np.c_[U,pd.get_dummies(d.event).values.astype(float)[:,1:]] if event else U
    b=np.linalg.lstsq(X,y,rcond=None)[0]; p=X@b
    return 1-((y-p)**2).sum()/((y-y.mean())**2).sum(), np.corrcoef(y,p)[0,1], b
def terms(d):
    _,_,b=ols(d); units=sorted(d.unit.unique()); ev=sorted(d.event.unique())
    return pd.Series(b[:len(units)],index=units), pd.Series(np.r_[0,b[len(units):]],index=ev)
def drop_singletons(d): return d[d.groupby('event').event.transform('size')>=2]

print('=== geometry: machine coverage and unit count imposed on mammals (no primate terms) ===')
m=P['mammals']; a,b=terms(m)
cov=np.array([32,38,36,41,43,44])/48.0          # retained events per size, 14M to 1B, over 48 events
rate=a.sort_values(ascending=False); frac=np.interp(np.linspace(0,1,10),np.linspace(0,1,6),cov)
kl=[];rand=[]
for (u,_),f in zip(rate.items(),frac):
    du=m[m.unit==u]; n=max(2,int(round(f*len(du))))
    kl.append(du.assign(beta=b[du.event].values).sort_values('beta').iloc[:n])
late=drop_singletons(pd.concat(kl))
for rep in range(200):
    kr=[m[m.unit==u].sample(max(2,int(round(f*len(m[m.unit==u])))),random_state=int(rng.integers(1e9))) for (u,_),f in zip(rate.items(),frac)]
    dd=drop_singletons(pd.concat(kr)); rand.append(ols(dd,False)[0])
print(f"baseline rate-only R2 {ols(m,False)[0]:.3f}")
print(f"latest events dropped from the slowest species, machine fractions: rate-only R2 {ols(late,False)[0]:.3f} rho {ols(late)[1]:.4f} cells {len(late)}")
print(f"same counts dropped at random, 200 draws: rate-only R2 mean {np.mean(rand):.3f} [{np.percentile(rand,2.5):.3f}, {np.percentile(rand,97.5):.3f}]")
vals=np.array([ols(drop_singletons(m[m.unit.isin(S)]),False)[0] for S in combinations(sorted(m.unit.unique()),6)])
print(f"all {len(vals)} six-species subsets: rate-only R2 median {np.median(vals):.3f} min {vals.min():.3f} max {vals.max():.3f}")

print('=== rival: log t = a_i + g_i * b_e, alternating least squares, 500 sweeps ===')
def slope(d):
    y=d.obs.values;u=d.unit.values;e=d.event.values
    a,b=terms(d); g=pd.Series(1.0,index=a.index)
    for it in range(500):
        for ui in a.index:
            mk=u==ui; X=np.c_[np.ones(mk.sum()),b[e[mk]].values]; a[ui],g[ui]=np.linalg.lstsq(X,y[mk],rcond=None)[0]
        num=pd.Series(g[u].values*(y-a[u].values)).groupby(e).sum(); den=pd.Series(g[u].values**2).groupby(e).sum()
        b=num/den; sc=b.std(); b=(b-b.mean())/sc; g=g*sc
        a=pd.Series(y-g[u].values*b[e].values).groupby(u).mean()
    p=a[u].values+g[u].values*b[e].values
    return 1-((y-p)**2).sum()/((y-y.mean())**2).sum(), g/g.mean()
def report(name,d):
    r2a=ols(d)[0]; r2s,g=slope(d); n=len(d); k1=d.unit.nunique()+d.event.nunique()-1; q=d.unit.nunique()-1
    F=((r2s-r2a)/q)/((1-r2s)/(n-k1-q))
    print(f"{name:26s} n={n} additive R2 {r2a:.4f} stretch R2 {r2s:.4f} gain {r2s-r2a:.4f} F({q},{n-k1-q})={F:.1f}")
    print('   ', ', '.join(f'{k}={v:.2f}' for k,v in g.round(2).items()))
for s,d in P.items(): report(s,d)
mch=P['machines']; report('machines complete-case',mch[mch.groupby('event').event.transform('size')==6])
for f,name in [('machine_acq_70_dedup.csv','machines deduplicated'),('machine_acq_65.csv','machines criterion 0.65'),('machine_acq_75.csv','machines criterion 0.75')]:
    x=pd.read_csv(f).rename(columns={'size':'unit','paradigm':'event'}); x['obs']=np.log(x.tok); report(name,drop_singletons(x))
st=pd.read_csv('machine_acq_70.csv').rename(columns={'size':'unit','paradigm':'event'}); st['obs']=np.log(st.tok)
report('machines standard, 5 sizes',drop_singletons(st[st.unit!='pythia-1b']))

print('=== child ceiling: slow one language, censor above 36 months, refit ===')
c=P['children']; cap=np.log(36.0); a,b=terms(c)
for slow in [1.25,1.5,2.0]:
    rec=[];span=[];lost=[]
    for L in c.unit.unique():
        d=c.copy(); mk=(d.unit==L).values; d.loc[mk,'obs']+=np.log(slow)
        d=drop_singletons(d[d.obs<=cap+1e-9]); a2,_=terms(d)
        rec.append(np.exp(a2[L]-a[L])); span.append(np.exp(a2.max()-a2.min())); lost.append(mk.sum()-(d.unit==L).sum())
    print(f"slowed x{slow}: recovered fold median {np.median(rec):.3f} min {min(rec):.3f} max {max(rec):.3f}; cells censored median {np.median(lost):.0f}; rate span after median {np.median(span):.3f}")

print('=== rival: random restarts of the alternating fit (initial stretches drawn in [0.5, 2]) ===')
def slope_init(d,g0):
    y=d.obs.values;u=d.unit.values;e=d.event.values
    a,b=terms(d); g=pd.Series(g0,index=a.index)
    for it in range(500):
        for ui in a.index:
            mk=u==ui; X=np.c_[np.ones(mk.sum()),b[e[mk]].values]; a[ui],g[ui]=np.linalg.lstsq(X,y[mk],rcond=None)[0]
        num=pd.Series(g[u].values*(y-a[u].values)).groupby(e).sum(); den=pd.Series(g[u].values**2).groupby(e).sum()
        b=num/den; sc=b.std(); b=(b-b.mean())/sc; g=g*sc
        a=pd.Series(y-g[u].values*b[e].values).groupby(u).mean()
    p=a[u].values+g[u].values*b[e].values
    return 1-((y-p)**2).sum()/((y-y.mean())**2).sum(), (g/g.mean()).min()
for s in ['mammals','machines']:
    d=P[s]; res=[slope_init(d,rng.uniform(0.5,2,d.unit.nunique())) for _ in range(20)]
    r2=[r for r,_ in res]; gm=[g for _,g in res]
    print(f"{s}: 20 restarts, stretch R2 min {min(r2):.4f} max {max(r2):.4f}; smallest fitted stretch across restarts min {min(gm):.2f} max {max(gm):.2f}")

print('=== child ceiling: rate-only R2 under censoring against the uncensored slowed table ===')
for slow in [1.25,1.5,2.0]:
    cen=[];unc=[]
    for L in c.unit.unique():
        d=c.copy(); mk=(d.unit==L).values; d.loc[mk,'obs']+=np.log(slow)
        unc.append(ols(d,False)[0]); cen.append(ols(drop_singletons(d[d.obs<=cap+1e-9]),False)[0])
    print(f"slowed x{slow}: rate-only R2 uncensored median {np.median(unc):.3f}, censored median {np.median(cen):.3f} min {min(cen):.3f} max {max(cen):.3f}")
