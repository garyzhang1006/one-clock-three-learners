# Does the machine schedule survive a change of training corpus? Same six sizes, same 17 steps,
# same 67 paradigms, but the deduped Pythia ladder instead of the standard one.
import numpy as np, pandas as pd, sys
from scipy.optimize import curve_fit
from scipy import stats
from final_results import fitk0, nullk0
SRC=sys.argv[1] if len(sys.argv)>1 else 'blimp_by_step_dedup.csv'
TOK=1024*2049.0; CRIT=0.70
d=pd.read_csv(SRC)
from machines import acq, logistic
d['tok']=d.step*TOK
rows=[]
for (sz,par),g in d.groupby(['size','paradigm']):
    a=acq(g,CRIT)
    if a==a and a>0: rows.append({'size':sz,'paradigm':par,'tok':a})
A=pd.DataFrame(rows); c=A.paradigm.value_counts(); A=A[A.paradigm.map(c)>=2]
u,e,t=A['size'].values,A.paradigm.values,A.tok.values.astype(float)
f=fitk0(u,e,t); fr=fitk0(u,e,t,inc=False); nd=nullk0(u,e,t,n=1000)
z=(f['rho']-nd.mean())/nd.std()
print(f"DEDUP  sizes={A['size'].nunique()} events={A.paradigm.nunique()} cells={len(A)} "
      f"rho={f['rho']:.4f} R2={f['r2']:.4f} rate-only R2={fr['r2']:.4f} null={nd.mean():.4f}+-{nd.std():.4f} z={z:.1f}")
ui,ul=pd.factorize(u); ei,_=pd.factorize(e); n=len(t)
E=np.zeros((n,ei.max()+1)); E[np.arange(n),ei]=1
X=np.hstack([np.eye(ui.max()+1)[ui],E[:,1:]])
b,*_=np.linalg.lstsq(X,np.log(t),rcond=None); al=b[:ui.max()+1]; al=al-al.min()
print("dedup rates:",{k:round(float(v),3) for k,v in zip(ul,np.exp(al))})
# does the deduped schedule agree with the standard one, event by event?
S=pd.read_csv('machine_event_terms.csv')[['paradigm','beta']].set_index('paradigm').beta
beta=np.concatenate([[0.0],b[ui.max()+1:]]); ev=pd.factorize(e)[1]
D=pd.Series(beta,index=ev)
both=D.index.intersection(S.index)
r,p=stats.spearmanr(D[both],S[both])
print(f"schedule agreement with standard Pythia: spearman={r:+.3f} p={p:.3g} over {len(both)} shared paradigms")
A.to_csv('machine_acq_70_dedup.csv',index=False)
