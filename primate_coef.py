# Variant fit at k=4.63 (log(t-k), the free-k estimate): NOT the numbers in the paper, which come from
# consistency.py at k=0 (beta +0.240 / -0.101). Kept only to show the interaction survives the change of k.
import numpy as np, pandas as pd
rng=np.random.default_rng(0)
df=pd.read_csv('event_data.csv')
SP=['Hamster','Mouse','Rat','Rabbit','Spiny.Mouse','Guinea.Pig','Ferret','Cat','Macaque','Human']
PRIM={'Macaque','Human'}
rows=[(ei,s,float(df[s].iloc[ei]),int(df['Cortical'].iloc[ei]),int(df['Limbic'].iloc[ei]),s in PRIM)
      for s in SP for ei in range(len(df)) if df[s].iloc[ei]>0]
R=pd.DataFrame(rows,columns=['ev','sp','t','cort','limb','prim'])
c=R.ev.value_counts(); R=R[R.ev.map(c)>=2].reset_index(drop=True)
K=4.63
def coefs(R):
    n=len(R); u,_=pd.factorize(R.sp.values); e,_=pd.factorize(R.ev.values)
    E=np.zeros((n,e.max()+1)); E[np.arange(n),e]=1
    X=np.hstack([np.eye(u.max()+1)[u], E[:,1:],
                 (R.prim.values*R.cort.values).reshape(-1,1).astype(float),
                 (R.prim.values*R.limb.values).reshape(-1,1).astype(float)])
    b,*_=np.linalg.lstsq(X,np.log(R.t.values-K),rcond=None)
    return b[-2],b[-1]
pc,pl=coefs(R)
B=np.array([coefs(R.sample(len(R),replace=True,random_state=int(s))) for s in rng.integers(0,1e9,2000)])
def ci(v): return np.percentile(v,[2.5,97.5])
print(f'primate x cortical  beta={pc:+.4f} log-units  95% CI [{ci(B[:,0])[0]:+.4f},{ci(B[:,0])[1]:+.4f}]  '
      f'fold-late={np.exp(pc):.3f}  p(<=0)={(B[:,0]<=0).mean():.4f}')
print(f'primate x limbic    beta={pl:+.4f} log-units  95% CI [{ci(B[:,1])[0]:+.4f},{ci(B[:,1])[1]:+.4f}]  '
      f'fold-late={np.exp(pl):.3f}  p(<=0)={(B[:,1]<=0).mean():.4f}')
