import numpy as np, pandas as pd, json
rng=np.random.default_rng(0)
FIELD={k:v['field'] for k,v in json.load(open('blimp_fields.json')).items()}  # paradigm -> BLiMP phenomenon, exported from the original scratch subset
A=pd.read_csv('machine_acq_70.csv'); c=A.paradigm.value_counts(); A=A[A.paradigm.map(c)>=2].copy()
ui,ul=pd.factorize(A['size'].values); ei,el=pd.factorize(A.paradigm.values); n=len(A)
E=np.zeros((n,ei.max()+1)); E[np.arange(n),ei]=1
X=np.hstack([np.eye(ui.max()+1)[ui],E[:,1:]])
y=np.log(A.tok.values.astype(float))
b,*_=np.linalg.lstsq(X,y,rcond=None)
beta=np.concatenate([[0.0],b[ui.max()+1:]])          # event terms, first paradigm as reference
T=pd.DataFrame({'paradigm':el,'beta':beta}); T['field']=T.paradigm.map(FIELD)
T['tokens_at_ref_rate']=np.exp(T.beta+b[:ui.max()+1].mean())
g=T.groupby('field').agg(n=('beta','size'),mean_beta=('beta','mean')).sort_values('mean_beta')
g['fold_vs_earliest']=np.exp(g.mean_beta-g.mean_beta.min())
print('Machine acquisition order by BLiMP phenomenon (earliest first):')
print(g.round(3).to_string())

# does phenomenon explain the event terms? one-way permutation F-test
def ss_between(vals,labels):
    m=vals.mean(); return sum(len(v)*(v.mean()-m)**2 for _,v in pd.Series(vals).groupby(labels))
obs=ss_between(T.beta.values,T.field.values)
null=np.array([ss_between(T.beta.values,rng.permutation(T.field.values)) for _ in range(20000)])
tot=((T.beta-T.beta.mean())**2).sum()
print(f'\nphenomenon explains {obs/tot:.3f} of the variance in machine event terms, '
      f'permutation p={(null>=obs).mean():.4g} over 20000 label shuffles ({T.field.nunique()} phenomena, {len(T)} paradigms)')
T.sort_values('beta').to_csv('machine_event_terms.csv',index=False)
print('\nearliest 5:',list(T.sort_values('beta').paradigm[:5]))
print('latest 5:  ',list(T.sort_values('beta').paradigm[-5:]))
