import numpy as np, pandas as pd
from estimator import _hat,_design,permute_within_unit
rng=np.random.default_rng(0)
df=pd.read_csv('event_data.csv')
SP=['Hamster','Mouse','Rat','Rabbit','Spiny.Mouse','Guinea.Pig','Ferret','Cat','Macaque','Human']
PRIM={'Macaque','Human'}
rows=[(e,s,float(df[s].iloc[e]),int(df['Cortical'].iloc[e]),int(df['Limbic'].iloc[e]),s in PRIM)
      for s in SP for e in range(len(df)) if df[s].iloc[e]>0]
R=pd.DataFrame(rows,columns=['ev','sp','t','cort','limb','prim']); c=R.ev.value_counts(); R=R[R.ev.map(c)>=2].reset_index(drop=True)
def coefs(D):
    n=len(D); u,_=pd.factorize(D.sp.values); e,_=pd.factorize(D.ev.values)
    E=np.zeros((n,e.max()+1)); E[np.arange(n),e]=1
    X=np.hstack([np.eye(u.max()+1)[u],E[:,1:],
                 (D.prim.values*D.cort.values).reshape(-1,1).astype(float),
                 (D.prim.values*D.limb.values).reshape(-1,1).astype(float)])
    b,*_=np.linalg.lstsq(X,np.log(D.t.values),rcond=None); return b[-2],b[-1]
pc,pl=coefs(R)
B=np.array([coefs(R.sample(len(R),replace=True,random_state=int(s))) for s in rng.integers(0,10**9,2000)])
for nm,v,o in [('primate x cortical',B[:,0],pc),('primate x limbic',B[:,1],pl)]:
    lo,hi=np.percentile(v,[2.5,97.5])
    print(f'{nm:20s} beta={o:+.4f}  95% CI [{lo:+.4f},{hi:+.4f}]  fold={np.exp(o):.3f}  p(<=0)={(v<=0).mean():.4f}')

print('\ncriterion sensitivity at k=0:')
for crit in [65,70,75]:
    A=pd.read_csv(f'machine_acq_{crit}.csv'); cc=A.paradigm.value_counts(); A=A[A.paradigm.map(cc)>=2]
    u,e,t=A['size'].values,A.paradigm.values,A.tok.values.astype(float)
    U,_=_hat(_design(u,e,None,True)); y=np.log(t); p=U@(U.T@y)
    Ur,_=_hat(_design(u,e,None,False)); pr=Ur@(Ur.T@y)
    rho=np.corrcoef(y,p)[0,1]; r2=1-((y-p)**2).sum()/((y-y.mean())**2).sum()
    r2r=1-((y-pr)**2).sum()/((y-y.mean())**2).sum()
    nd=np.array([np.corrcoef(np.log(permute_within_unit(u,t,rng)),U@(U.T@np.log(permute_within_unit(u,t,rng))))[0,1] for _ in range(1)])
    nd=[]
    for _ in range(500):
        tp=permute_within_unit(u,t,rng); yy=np.log(tp); nd.append(np.corrcoef(yy,U@(U.T@yy))[0,1])
    nd=np.array(nd)
    print(f'  acc>={crit/100:.2f}  n={len(t):3d} events={A.paradigm.nunique():2d}  rho={rho:.4f} R2={r2:.4f} rate-only R2={r2r:.4f} '
          f'null={nd.mean():.4f}+-{nd.std():.4f} z={(rho-nd.mean())/nd.std():.1f}')
