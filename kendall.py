# Assumption-free check: Kendall's W on event RANKS within each unit. No log, no additivity,
# no rate parameter. This is the statistic the acquisition literature uses for order consistency.
import numpy as np, pandas as pd, json
from itertools import combinations
rng=np.random.default_rng(0)

def kendall_W(unit,event,t,reps=2000):
    d=pd.DataFrame({'u':unit,'e':event,'t':np.asarray(t,float)})
    d['r']=d.groupby('u').t.rank()
    # pairwise concordance over events shared by each pair of units (handles missing cells)
    def conc(dd):
        agree=tot=0
        for (u1,u2) in combinations(dd.u.unique(),2):
            a=dd[dd.u==u1].set_index('e').r; b=dd[dd.u==u2].set_index('e').r
            sh=a.index.intersection(b.index)
            if len(sh)<3: continue
            av,bv=a[sh].values,b[sh].values
            for i,j in combinations(range(len(sh)),2):
                if av[i]==av[j] or bv[i]==bv[j]: continue
                tot+=1; agree+= ((av[i]<av[j])==(bv[i]<bv[j]))
        return agree/tot if tot else np.nan, tot
    obs,npairs=conc(d)
    null=[]
    for _ in range(reps):
        dd=d.copy()
        dd['r']=dd.groupby('u').r.transform(lambda x: rng.permutation(x.values))
        null.append(conc(dd)[0])
    null=np.array([x for x in null if x==x])
    return obs,npairs,null.mean(),null.std(),(null>=obs).mean()

res=[]
df=pd.read_csv('event_data.csv')
SP=['Hamster','Mouse','Rat','Rabbit','Spiny.Mouse','Guinea.Pig','Ferret','Cat','Macaque','Human']
rows=[(f'ev{e}',s,float(df[s].iloc[e])) for s in SP for e in range(len(df)) if df[s].iloc[e]>0]
R=pd.DataFrame(rows,columns=['ev','sp','t']); c=R.ev.value_counts(); R=R[R.ev.map(c)>=2]
res.append(('mammals',)+kendall_W(R.sp.values,R.ev.values,R.t.values,reps=400))

a=pd.read_feather('aoas.feather'); ch=a[(a.measure=='produces')&a.aoa.notna()&a.uni_lemma.notna()]
c=ch.uni_lemma.value_counts(); ch=ch[ch.uni_lemma.map(c)>=2]
res.append(('children',)+kendall_W(ch.language.values,ch.uni_lemma.values,ch.aoa.values.astype(float),reps=100))

A=pd.read_csv('machine_acq_70.csv'); c=A.paradigm.value_counts(); A=A[A.paradigm.map(c)>=2]
res.append(('machines',)+kendall_W(A['size'].values,A.paradigm.values,A.tok.values.astype(float),reps=400))

T=pd.DataFrame(res,columns=['system','concordance','pairs','null_mean','null_sd','p'])
T['z']=(T.concordance-T.null_mean)/T.null_sd
print(T.round(4).to_string(index=False)); T.to_csv('concordance_results.csv',index=False)
