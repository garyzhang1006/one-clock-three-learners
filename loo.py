# Leave-one-unit-out: fit the shared schedule WITHOUT a unit, anchor that unit's rate on a few
# of its own events, then predict the rest. This is what Translating Time is actually used for.
import numpy as np, pandas as pd
rng=np.random.default_rng(0)

def loo(unit,event,t,n_anchor=3,reps=200):
    unit=np.asarray(unit); event=np.asarray(event); y=np.log(np.asarray(t,dtype=float))
    out=[]
    for u in pd.unique(unit):
        tr=unit!=u; te=unit==u
        if te.sum()<n_anchor+3: continue
        ev_tr=pd.unique(event[tr])
        # fit shared schedule on the other units
        ui,_=pd.factorize(unit[tr]); ei=pd.Categorical(event[tr],categories=ev_tr).codes
        n=tr.sum(); E=np.zeros((n,len(ev_tr))); E[np.arange(n),ei]=1
        X=np.hstack([np.eye(ui.max()+1)[ui],E[:,1:]])
        b,*_=np.linalg.lstsq(X,y[tr],rcond=None)
        beta=np.concatenate([[0.0],b[ui.max()+1:]])          # event terms, first event as reference
        bmap=dict(zip(ev_tr,beta))
        idx=np.where(te)[0]
        usable=[i for i in idx if event[i] in bmap]
        if len(usable)<n_anchor+3: continue
        for _ in range(reps):
            perm=rng.permutation(usable)
            anc,tst=perm[:n_anchor],perm[n_anchor:]
            alpha=np.mean([y[i]-bmap[event[i]] for i in anc])  # the unit's single rate
            pred=np.array([alpha+bmap[event[i]] for i in tst]); obs=y[tst]
            out.append((u,np.mean((obs-pred)**2),np.var(obs),np.median(np.abs(np.exp(obs)-np.exp(pred))/np.exp(obs))))
    d=pd.DataFrame(out,columns=['unit','mse','var','mape'])
    r2=1-d.mse.sum()/d['var'].sum()
    return r2, d.mape.median(), d.unit.nunique()

res=[]
df=pd.read_csv('event_data.csv')
SP=['Hamster','Mouse','Rat','Rabbit','Spiny.Mouse','Guinea.Pig','Ferret','Cat','Macaque','Human']
rows=[(f'ev{e}',s,float(df[s].iloc[e])) for s in SP for e in range(len(df)) if df[s].iloc[e]>0]
R=pd.DataFrame(rows,columns=['ev','sp','t']); c=R.ev.value_counts(); R=R[R.ev.map(c)>=2]
res.append(('mammals',)+loo(R.sp.values,R.ev.values,R.t.values))

a=pd.read_feather('aoas.feather')
for meas in ['produces','understands']:
    ch=a[(a.measure==meas)&a.aoa.notna()&a.uni_lemma.notna()]
    c=ch.uni_lemma.value_counts(); ch=ch[ch.uni_lemma.map(c)>=2]
    res.append((f'children ({meas})',)+loo(ch.language.values,ch.uni_lemma.values,ch.aoa.values.astype(float)))

A=pd.read_csv('machine_acq_70.csv'); c=A.paradigm.value_counts(); A=A[A.paradigm.map(c)>=2]
res.append(('machines',)+loo(A['size'].values,A.paradigm.values,A.tok.values.astype(float)))

T=pd.DataFrame(res,columns=['system','oos_R2','median_rel_err','units_tested'])
print(T.round(4).to_string(index=False)); T.to_csv('loo_results.csv',index=False)
