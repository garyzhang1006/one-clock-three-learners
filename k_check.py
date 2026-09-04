import numpy as np, pandas as pd
from estimator import _hat,_design,fit,null_dist
def fit_fixed_k(unit,event,t,k,extra=None,include_event=True):
    U,rank=_hat(_design(unit,event,extra,include_event))
    y=np.log(t-k); p=U@(U.T@y)
    return np.corrcoef(y,p)[0,1], 1-((y-p)**2).sum()/((y-y.mean())**2).sum()

def show(name,unit,event,t,extra=None):
    f=fit(unit,event,t,extra); lo=t.min()
    r0,q0=fit_fixed_k(unit,event,t,0.0,extra)
    rr,qr=fit_fixed_k(unit,event,t,0.0,extra,include_event=False)
    print(f'{name:24s} t range=[{lo:.4g},{t.max():.4g}]  k_hat={f["k"]:.4g}  k_hat/min_t={f["k"]/lo:.3f}  '
          f'| rho(k_hat)={f["rho"]:.4f}  rho(k=0)={r0:.4f} R2(k=0)={q0:.4f}  rate-only R2(k=0)={qr:.4f}')
    return r0,q0

# mammals
df=pd.read_csv('event_data.csv')
SP=['Hamster','Mouse','Rat','Rabbit','Spiny.Mouse','Guinea.Pig','Ferret','Cat','Macaque','Human']
PRIM={'Macaque','Human'}
rows=[(f'ev{e}',s,float(df[s].iloc[e]),int(df['Cortical'].iloc[e]),int(df['Limbic'].iloc[e]),s in PRIM)
      for s in SP for e in range(len(df)) if df[s].iloc[e]>0]
R=pd.DataFrame(rows,columns=['ev','sp','t','cort','limb','prim']); c=R.ev.value_counts(); R=R[R.ev.map(c)>=2]
ex=np.c_[R.prim*R.cort,R.prim*R.limb].astype(float)
show('mammals',R.sp.values,R.ev.values,R.t.values,ex)
# children
a=pd.read_feather('aoas.feather'); ch=a[(a.measure=='produces')&a.aoa.notna()&a.uni_lemma.notna()]
c=ch.uni_lemma.value_counts(); ch=ch[ch.uni_lemma.map(c)>=2]
show('children (produces)',ch.language.values,ch.uni_lemma.values,ch.aoa.values.astype(float))
# machines
A=pd.read_csv('machine_acq_70.csv'); c=A.paradigm.value_counts(); A=A[A.paradigm.map(c)>=2]
show('machines (crit=0.70)',A['size'].values,A.paradigm.values,A.tok.values.astype(float))

# null for machines at k=0
print('\nk=0 permutation nulls:')
for nm,u,e,t,ex2 in [('mammals',R.sp.values,R.ev.values,R.t.values,ex),
                     ('children',ch.language.values,ch.uni_lemma.values,ch.aoa.values.astype(float),None),
                     ('machines',A['size'].values,A.paradigm.values,A.tok.values.astype(float),None)]:
    U,_=_hat(_design(u,e,ex2,True)); rng=np.random.default_rng(0)
    from estimator import permute_within_unit
    nd=[]
    for _ in range(300):
        tp=permute_within_unit(u,t,rng); y=np.log(tp); p=U@(U.T@y)
        nd.append(np.corrcoef(y,p)[0,1])
    nd=np.array(nd); obs=fit_fixed_k(u,e,t,0.0,ex2)[0]
    print(f'  {nm:10s} obs={obs:.4f}  null={nd.mean():.4f}+-{nd.std():.4f} max={nd.max():.4f} p={(nd>=obs).mean():.3g} z={(obs-nd.mean())/nd.std():.1f}')
