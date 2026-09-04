import numpy as np, pandas as pd
def rates(unit,event,t,k):
    y=np.log(t-k); u,ul=pd.factorize(unit); e,el=pd.factorize(event)
    X=np.hstack([np.eye(u.max()+1)[u], np.zeros((len(t),e.max()))])
    E=np.zeros((len(t),e.max()+1)); E[np.arange(len(t)),e]=1; X[:,u.max()+1:]=E[:,1:]
    b,*_=np.linalg.lstsq(X,y,rcond=None)
    a=b[:u.max()+1]; a=a-a.min()
    return pd.Series(a,index=ul).sort_values()

df=pd.read_csv('event_data.csv')
SP=['Hamster','Mouse','Rat','Rabbit','Spiny.Mouse','Guinea.Pig','Ferret','Cat','Macaque','Human']
rows=[(f'ev{ei}',s,float(df[s].iloc[ei])) for s in SP for ei in range(len(df)) if df[s].iloc[ei]>0]
R=pd.DataFrame(rows,columns=['ev','sp','t']); c=R.ev.value_counts(); R=R[R.ev.map(c)>=2]
rm=rates(R.sp.values,R.ev.values,R.t.values,4.63)
print('MAMMAL rate offsets (log units above fastest); exp() = fold-slower than hamster')
print(pd.DataFrame({'log_rate':rm.round(3),'fold':np.exp(rm).round(2)}).to_string())

a=pd.read_feather('aoas.feather'); d=a[(a.measure=='produces')&a.aoa.notna()&a.uni_lemma.notna()]
c=d.uni_lemma.value_counts(); d=d[d.uni_lemma.map(c)>=2]
rc=rates(d.language.values,d.uni_lemma.values,d.aoa.values.astype(float),3.41)
print('\nCHILD (produces) language rate offsets; exp() = fold-slower than fastest language')
print(pd.DataFrame({'log_rate':rc.round(3),'fold':np.exp(rc).round(2)}).to_string())
