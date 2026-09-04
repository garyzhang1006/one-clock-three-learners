# Does a domain label organise the conserved schedule? Same statistic, same null, all three systems.
import numpy as np, pandas as pd, json
rng=np.random.default_rng(0)

def event_terms(unit,event,t):
    unit=np.asarray(unit); event=np.asarray(event); y=np.log(np.asarray(t,float))
    ui,_=pd.factorize(unit); ei,el=pd.factorize(event); n=len(y)
    E=np.zeros((n,ei.max()+1)); E[np.arange(n),ei]=1
    X=np.hstack([np.eye(ui.max()+1)[ui],E[:,1:]])
    b,*_=np.linalg.lstsq(X,y,rcond=None)
    return pd.DataFrame({'event':el,'beta':np.concatenate([[0.0],b[ui.max()+1:]])})

def domain_test(T,label,reps=20000):
    T=T.dropna(subset=[label]); v=T.beta.values; lab=T[label].values
    def ssb(vals,l):
        m=vals.mean(); return sum(len(g)*(g.mean()-m)**2 for _,g in pd.Series(vals).groupby(l))
    tot=((v-v.mean())**2).sum(); obs=ssb(v,lab)
    null=np.array([ssb(v,rng.permutation(lab)) for _ in range(reps)])
    return obs/tot,(null>=obs).mean(),T[label].nunique(),len(T)

res=[]
# mammals: cortical / limbic / neither
df=pd.read_csv('event_data.csv')
SP=['Hamster','Mouse','Rat','Rabbit','Spiny.Mouse','Guinea.Pig','Ferret','Cat','Macaque','Human']
rows=[(f'ev{e}',s,float(df[s].iloc[e])) for s in SP for e in range(len(df)) if df[s].iloc[e]>0]
R=pd.DataFrame(rows,columns=['ev','sp','t']); c=R.ev.value_counts(); R=R[R.ev.map(c)>=2]
T=event_terms(R.sp.values,R.ev.values,R.t.values)
cls=df.apply(lambda r:'cortical' if r.Cortical else ('limbic' if r.Limbic else 'other'),axis=1)
T['structure']=T.event.str.replace('ev','').astype(int).map(cls)
res.append(('mammals','brain structure class')+domain_test(T,'structure'))
gm=T.groupby('structure').beta.agg(['size','mean']).sort_values('mean')
print('MAMMALS by structure class (earliest first)\n',gm.round(3).to_string(),'\n')

# children: CDI lexical class
it=pd.read_feather('items.feather')
lex=(it[it.type=='word'].dropna(subset=['uni_lemma'])
       .groupby('uni_lemma').lexical_class.agg(lambda x:x.mode().iat[0]))
a=pd.read_feather('aoas.feather'); ch=a[(a.measure=='produces')&a.aoa.notna()&a.uni_lemma.notna()]
c=ch.uni_lemma.value_counts(); ch=ch[ch.uni_lemma.map(c)>=2]
Tc=event_terms(ch.language.values,ch.uni_lemma.values,ch.aoa.values.astype(float))
Tc['lexclass']=Tc.event.map(lex)
res.append(('children','CDI lexical class')+domain_test(Tc,'lexclass'))
gc=Tc.groupby('lexclass').beta.agg(['size','mean']).sort_values('mean')
print('CHILDREN by lexical class (earliest first)\n',gc.round(3).to_string(),'\n')

# machines: BLiMP phenomenon
FIELD={k:v['field'] for k,v in json.load(open('blimp_fields.json')).items()}  # paradigm -> BLiMP phenomenon, exported from the original scratch subset
A=pd.read_csv('machine_acq_70.csv'); c=A.paradigm.value_counts(); A=A[A.paradigm.map(c)>=2]
Tm=event_terms(A['size'].values,A.paradigm.values,A.tok.values.astype(float))
Tm['field']=Tm.event.map(FIELD)
res.append(('machines','BLiMP phenomenon')+domain_test(Tm,'field'))

O=pd.DataFrame(res,columns=['system','domain label','var explained','perm p','classes','events'])
print(O.round(4).to_string(index=False)); O.to_csv('domain_results.csv',index=False)
