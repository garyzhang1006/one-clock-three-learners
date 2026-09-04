# Does the leave-one-out result depend on how many anchor events the held-out unit is granted?
import numpy as np, pandas as pd, loo as L
from loo import loo
df=pd.read_csv('event_data.csv')
SP=['Hamster','Mouse','Rat','Rabbit','Spiny.Mouse','Guinea.Pig','Ferret','Cat','Macaque','Human']
rows=[(f'ev{e}',s,float(df[s].iloc[e])) for s in SP for e in range(len(df)) if df[s].iloc[e]>0]
R=pd.DataFrame(rows,columns=['ev','sp','t']); c=R.ev.value_counts(); R=R[R.ev.map(c)>=2]
a=pd.read_feather('aoas.feather')
P=a[(a.measure=='produces')&a.aoa.notna()&a.uni_lemma.notna()]
c=P.uni_lemma.value_counts(); P=P[P.uni_lemma.map(c)>=2]
M=pd.read_csv('machine_acq_70.csv'); c=M.paradigm.value_counts(); M=M[M.paradigm.map(c)>=2]
SETS=[('mammals',R.sp.values,R.ev.values,R.t.values),
      ('children',P.language.values,P.uni_lemma.values,P.aoa.values.astype(float)),
      ('machines',M['size'].values,M.paradigm.values,M.tok.values.astype(float))]
def cell(u,e,t,na):
    L.rng=np.random.default_rng(0)   # reset per cell so every entry of the table reproduces on its own
    return loo(u,e,t,n_anchor=na,reps=200)[0]
for na in [1,2,3,5]:
    out=[f"{nm} R2={cell(u,e,t,na):.3f}" for nm,u,e,t in SETS]
    print(f"anchors={na}  "+"  |  ".join(out))
