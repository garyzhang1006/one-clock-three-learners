# Bootstrap 95% intervals on the headline correlation of each system, resampling units
# with replacement so the interval reflects the sampling of species, languages, and scales.
import numpy as np, pandas as pd, json
from final_results import fitk0, SYS

B=500; rng=np.random.default_rng(0); recs=[]
for nm,u,e,t,ex in SYS:
    u=np.asarray(u); e=np.asarray(e); t=np.asarray(t,dtype=float)
    ex=None if ex is None else np.asarray(ex,dtype=float)
    units=pd.unique(u); idx_by_unit={x:np.where(u==x)[0] for x in units}
    obs=fitk0(u,e,t,ex)['rho']; draws=[]
    for _ in range(B):
        pick=rng.choice(units,size=len(units),replace=True)
        idx=np.concatenate([idx_by_unit[x] for x in pick])
        # relabel duplicated units so each draw keeps its own rate parameter
        lab=np.concatenate([np.full(len(idx_by_unit[x]),f"{x}#{j}") for j,x in enumerate(pick)])
        ee=e[idx]; c=pd.Series(ee).value_counts()
        keep=pd.Series(ee).map(c).values>=2
        if keep.sum()<20: continue
        try:
            r=fitk0(lab[keep],ee[keep],t[idx][keep],None if ex is None else ex[idx][keep])['rho']
        except Exception: continue
        if np.isfinite(r): draws.append(r)
    d=np.array(draws)
    recs.append(dict(system=nm,rho=obs,lo=np.percentile(d,2.5),hi=np.percentile(d,97.5),
                     n_draws=len(d)))
    print(f"{nm:9s} rho={obs:.4f}  95% CI [{np.percentile(d,2.5):.4f},{np.percentile(d,97.5):.4f}]  draws={len(d)}")
pd.DataFrame(recs).to_csv('bootstrap_ci.csv',index=False)
