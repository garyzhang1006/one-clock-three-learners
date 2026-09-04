import numpy as np, pandas as pd
from estimator import _hat,_design,permute_within_unit

def fit0(u,e,t,ex=None,inc=True):
    U,rank=_hat(_design(u,e,ex,inc)); y=np.log(t); p=U@(U.T@y)
    r2=1-((y-p)**2).sum()/((y-y.mean())**2).sum()
    n=len(y); adj=1-(1-r2)*(n-1)/(n-rank)
    return dict(rho=np.corrcoef(y,p)[0,1],r2=r2,adj=adj,rank=rank,n=n,U=U)

print('=== ATTACK 2: is the rate-only R2 contrast a degrees-of-freedom artefact? ===')
df=pd.read_csv('event_data.csv')
SP=['Hamster','Mouse','Rat','Rabbit','Spiny.Mouse','Guinea.Pig','Ferret','Cat','Macaque','Human']
rows=[(f'ev{e}',s,float(df[s].iloc[e])) for s in SP for e in range(len(df)) if df[s].iloc[e]>0]
R=pd.DataFrame(rows,columns=['ev','sp','t']); c=R.ev.value_counts(); R=R[R.ev.map(c)>=2]
a=pd.read_feather('aoas.feather'); ch=a[(a.measure=='produces')&a.aoa.notna()&a.uni_lemma.notna()]
c=ch.uni_lemma.value_counts(); ch=ch[ch.uni_lemma.map(c)>=2]
A=pd.read_csv('machine_acq_70.csv'); c=A.paradigm.value_counts(); A=A[A.paradigm.map(c)>=2]
SYS=[('mammals',R.sp.values,R.ev.values,R.t.values),
     ('children',ch.language.values,ch.uni_lemma.values,ch.aoa.values.astype(float)),
     ('machines',A['size'].values,A.paradigm.values,A.tok.values.astype(float))]
for nm,u,e,t in SYS:
    f=fit0(u,e,t); fr=fit0(u,e,t,inc=False)
    print(f'{nm:9s} rate-only: R2={fr["r2"]:.4f} adjR2={fr["adj"]:.4f} params={fr["rank"]} n={fr["n"]} '
          f'params/n={fr["rank"]/fr["n"]:.3f} | full: R2={f["r2"]:.4f} adjR2={f["adj"]:.4f} params={f["rank"]}')

print('\n=== ATTACK 3: does non-random missingness drive the machine result? ===')
Afull=pd.read_csv('machine_acq_70.csv')
SZ=['pythia-14m','pythia-31m','pythia-70m','pythia-160m','pythia-410m','pythia-1b']
present=[z for z in SZ if z in set(Afull['size'])]
cc=Afull.paradigm.value_counts()
for thr in [len(present), len(present)-1]:
    keep=cc[cc>=thr].index; C=Afull[Afull.paradigm.isin(keep)]
    if C.paradigm.nunique()<10:
        print(f'  >= {thr}/{len(present)} sizes: only {C.paradigm.nunique()} paradigms, too few to test'); continue
    f=fit0(C['size'].values,C.paradigm.values,C.tok.values.astype(float))
    fr=fit0(C['size'].values,C.paradigm.values,C.tok.values.astype(float),inc=False)
    rng=np.random.default_rng(0); U=f['U']; nd=[]
    for _ in range(1000):
        tp=permute_within_unit(C['size'].values,C.tok.values.astype(float),rng)
        y=np.log(tp); nd.append(np.corrcoef(y,U@(U.T@y))[0,1])
    nd=np.array(nd)
    ui,ul=pd.factorize(C['size'].values); ei,_=pd.factorize(C.paradigm.values); n=len(C)
    E=np.zeros((n,ei.max()+1)); E[np.arange(n),ei]=1
    X=np.hstack([np.eye(ui.max()+1)[ui],E[:,1:]])
    b,*_=np.linalg.lstsq(X,np.log(C.tok.values.astype(float)),rcond=None)
    al=b[:ui.max()+1]; al=al-al.min()
    rates={k:round(float(np.exp(v)),2) for k,v in sorted(zip(ul,al),key=lambda z:present.index(z[0]))}
    print(f'  >= {thr}/{len(present)} sizes: paradigms={C.paradigm.nunique()} cells={len(C)} '
          f'rho={f["rho"]:.4f} R2={f["r2"]:.4f} rate-only R2={fr["r2"]:.4f} '
          f'null={nd.mean():.4f}+-{nd.std():.4f} z={(f["rho"]-nd.mean())/nd.std():.1f}')
    print(f'      rates: {rates}')
