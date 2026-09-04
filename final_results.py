import numpy as np, pandas as pd, json
from estimator import _hat,_design,permute_within_unit
def fitk0(u,e,t,ex=None,inc=True):
    U,rank=_hat(_design(u,e,ex,inc)); y=np.log(t); p=U@(U.T@y)
    return dict(rho=float(np.corrcoef(y,p)[0,1]), r2=float(1-((y-p)**2).sum()/((y-y.mean())**2).sum()), rank=rank, U=U)
def nullk0(u,e,t,ex=None,n=1000,seed=0):
    U=fitk0(u,e,t,ex)['U']; rng=np.random.default_rng(seed); o=[]
    for _ in range(n):
        y=np.log(permute_within_unit(u,t,rng)); p=U@(U.T@y); o.append(np.corrcoef(y,p)[0,1])
    return np.array(o)
def rates(u,e,t):
    ui,ul=pd.factorize(u); ei,_=pd.factorize(e); n=len(t)
    E=np.zeros((n,ei.max()+1)); E[np.arange(n),ei]=1
    X=np.hstack([np.eye(ui.max()+1)[ui],E[:,1:]])
    b,*_=np.linalg.lstsq(X,np.log(t),rcond=None); a=b[:ui.max()+1]; a=a-a.min()
    return dict(zip(ul,np.exp(a)))

out={}
# mammals
df=pd.read_csv('event_data.csv')
SP=['Hamster','Mouse','Rat','Rabbit','Spiny.Mouse','Guinea.Pig','Ferret','Cat','Macaque','Human']
PRIM={'Macaque','Human'}
rows=[(f'ev{e}',s,float(df[s].iloc[e]),int(df['Cortical'].iloc[e]),int(df['Limbic'].iloc[e]),s in PRIM)
      for s in SP for e in range(len(df)) if df[s].iloc[e]>0]
R=pd.DataFrame(rows,columns=['ev','sp','t','cort','limb','prim']); c=R.ev.value_counts(); R=R[R.ev.map(c)>=2]
ex=np.c_[R.prim*R.cort,R.prim*R.limb].astype(float)
# children
a=pd.read_feather('aoas.feather'); ch=a[(a.measure=='produces')&a.aoa.notna()&a.uni_lemma.notna()]
c=ch.uni_lemma.value_counts(); ch=ch[ch.uni_lemma.map(c)>=2]
# machines
A=pd.read_csv('machine_acq_70.csv'); c=A.paradigm.value_counts(); A=A[A.paradigm.map(c)>=2]

SYS=[('mammals',R.sp.values,R.ev.values,R.t.values,ex),
     ('children',ch.language.values,ch.uni_lemma.values,ch.aoa.values.astype(float),None),
     ('machines',A['size'].values,A.paradigm.values,A.tok.values.astype(float),None)]
recs=[]
for nm,u,e,t,exx in SYS:
    f=fitk0(u,e,t,exx); fr=fitk0(u,e,t,exx,inc=False); nd=nullk0(u,e,t,exx)
    rt=rates(u,e,t)
    recs.append(dict(system=nm,n=len(t),units=len(set(u)),events=len(set(e)),
                     rho=f['rho'],r2=f['r2'],rho_rate=fr['rho'],r2_rate=fr['r2'],
                     null_mean=nd.mean(),null_sd=nd.std(),null_max=nd.max(),
                     p=float((nd>=f['rho']).mean()), z=(f['rho']-nd.mean())/nd.std(),
                     rate_span=max(rt.values())/min(rt.values())))
    out[nm+'_rates']={k:round(float(v),3) for k,v in rt.items()}
    # residuals for the figure
    y=np.log(t); p=f['U']@(f['U'].T@y)
    pd.DataFrame({'obs':y,'pred':p,'unit':u,'event':e}).to_csv(f'panel_{nm}.csv',index=False)
T=pd.DataFrame(recs); T.to_csv('final_table.csv',index=False)
print(T.round(4).to_string(index=False))
print(); [print(k,v) for k,v in out.items()]
json.dump(out,open('final_rates.json','w'),indent=1)
