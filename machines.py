import numpy as np, pandas as pd
from scipy.optimize import curve_fit
from estimator import report
TOK = 1024*2049.0                      # tokens per optimisation step (Pythia)
d = pd.read_csv('blimp_by_step_all.csv')
d['tok'] = d.step*TOK
SIZES=['pythia-14m','pythia-31m','pythia-70m','pythia-160m','pythia-410m','pythia-1b']

def logistic(x,L,x0,k): return 0.5 + (L-0.5)/(1+np.exp(-k*(x-x0)))
def acq(g, crit):
    g=g.sort_values('tok'); x=np.log(g.tok.values); y=g.acc.values
    if y.max() < crit: return np.nan
    try:
        p,_=curve_fit(logistic,x,y,p0=[max(y.max(),0.6),x.mean(),1.0],
                      bounds=([0.5,x.min()-5,0.01],[1.0,x.max()+5,20]),maxfev=20000)
        L,x0,k=p
        if L<=crit: return np.nan
        xc = x0 - np.log((L-0.5)/(crit-0.5)-1)/k
        return float(np.exp(xc)) if x.min()-2 <= xc <= x.max()+2 else np.nan
    except Exception:
        return np.nan

for crit in [0.65,0.70,0.75]:
    rows=[]
    for (s,p),g in d.groupby(['size','paradigm']):
        a=acq(g,crit)
        if a==a and a>0: rows.append((s,p,a))
    A=pd.DataFrame(rows,columns=['size','paradigm','tok'])
    A['size']=pd.Categorical(A['size'],SIZES,ordered=True)
    c=A.paradigm.value_counts(); A2=A[A.paradigm.map(c)>=2]
    print(f'\n### criterion acc={crit}   cells={len(A)}  paradigms={A.paradigm.nunique()}  per-size={dict(A["size"].value_counts().sort_index())}')
    if A2.paradigm.nunique()>=10 and A2['size'].nunique()>=3:
        r=report(f'machines (crit={crit})', A2['size'].astype(str).values, A2.paradigm.values, A2.tok.values.astype(float))
        # rate offsets
        u,ul=pd.factorize(A2['size'].astype(str).values); e,el=pd.factorize(A2.paradigm.values)
        n=len(A2); E=np.zeros((n,e.max()+1)); E[np.arange(n),e]=1
        X=np.hstack([np.eye(u.max()+1)[u],E[:,1:]])
        b,*_=np.linalg.lstsq(X,np.log(A2.tok.values-r['k']),rcond=None)
        a0=b[:u.max()+1]; a0=a0-a0.min()
        print('   size rate offsets (fold-slower than fastest):',
              {k:round(float(np.exp(v)),2) for k,v in sorted(zip(ul,a0), key=lambda z: SIZES.index(z[0]))})
    A.to_csv(f'machine_acq_{int(crit*100)}.csv',index=False)
