# Do later events vary more across units? A second shared property, needing no matched item set.
import numpy as np, pandas as pd
from final_results import SYS
from scipy import stats
rows=[]
for nm,u,e,t,ex in SYS:
    u=np.asarray(u); e=np.asarray(e); t=np.asarray(t,float)
    ui,_=pd.factorize(u); ei,ev=pd.factorize(e); n=len(t)
    E=np.zeros((n,ei.max()+1)); E[np.arange(n),ei]=1
    X=np.hstack([np.eye(ui.max()+1)[ui],E[:,1:]])
    b,*_=np.linalg.lstsq(X,np.log(t),rcond=None)
    nu=ui.max()+1
    beta=np.concatenate([[0.0],b[nu:]])          # event terms, first event as reference
    alpha=b[:nu]
    resid=np.log(t)-(alpha[ui]+beta[ei])         # rate-removed deviation
    d=pd.DataFrame({'ev':ei,'r':resid})
    g=d.groupby('ev').r.agg(['std','count'])
    g=g[g['count']>=3]                            # need a few units to estimate a spread
    if len(g)<8: print(nm,"too few events with 3+ units"); continue
    x=beta[g.index.values]; y=g['std'].values
    rho,p=stats.spearmanr(x,y)
    rows.append(dict(system=nm,events=len(g),spearman=rho,p=p))
    print(f"{nm:9s} events={len(g):4d}  spearman(schedule position, cross-unit sd)={rho:+.3f}  p={p:.3g}")
pd.DataFrame(rows).to_csv('lateness_variance.csv',index=False)
