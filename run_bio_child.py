import numpy as np, pandas as pd
from estimator import report
res=[]
# ---------- MAMMALS ----------
df=pd.read_csv('event_data.csv')
SP=['Hamster','Mouse','Rat','Rabbit','Spiny.Mouse','Guinea.Pig','Ferret','Cat','Macaque','Human']
PRIM={'Macaque','Human'}
rows=[]
for s in SP:
    for ei in range(len(df)):
        t=df[s].iloc[ei]
        if t and t>0: rows.append((f'ev{ei}',s,float(t),int(df['Cortical'].iloc[ei]),int(df['Limbic'].iloc[ei]),s in PRIM))
R=pd.DataFrame(rows,columns=['ev','sp','t','cort','limb','prim'])
cnt=R.ev.value_counts(); R=R[R.ev.map(cnt)>=2].reset_index(drop=True)
extra=np.c_[R.prim*R.cort, R.prim*R.limb].astype(float)
res.append(report('mammals (+primate terms)', R.sp.values, R.ev.values, R.t.values, extra))
res.append(report('mammals (no extra terms)', R.sp.values, R.ev.values, R.t.values, None))
# ---------- CHILDREN ----------
a=pd.read_feather('aoas.feather')
for meas in ['produces','understands']:
    d=a[(a.measure==meas)&a.aoa.notna()&a.uni_lemma.notna()].copy()
    c=d.uni_lemma.value_counts(); d=d[d.uni_lemma.map(c)>=2]
    res.append(report(f'children ({meas})', d.language.values, d.uni_lemma.values, d.aoa.values.astype(float)))
pd.DataFrame(res).drop(columns=[]).to_csv('fit_results_bio_child.csv',index=False)
print('\nsaved fit_results_bio_child.csv')
