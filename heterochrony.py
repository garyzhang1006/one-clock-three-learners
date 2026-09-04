# Event-pairing sequence heterochrony (Jeffery et al. 2005), simplified to a two-group contrast.
import numpy as np, pandas as pd, itertools
from collections import defaultdict
rng=np.random.default_rng(0)

df=pd.read_csv('event_data.csv')
SP=['Hamster','Mouse','Rat','Rabbit','Spiny.Mouse','Guinea.Pig','Ferret','Cat','Macaque','Human']
PRIM={'Macaque','Human'}
T={s:{ei:float(df[s].iloc[ei]) for ei in range(len(df)) if df[s].iloc[ei]>0} for s in SP}
CORT={ei:int(df['Cortical'].iloc[ei]) for ei in range(len(df))}

def rel(a,b): return 0 if a<b else (2 if a>b else 1)
pairs=defaultdict(lambda: {'np':[], 'p':[]})
for s in SP:
    ev=sorted(T[s])
    for i,j in itertools.combinations(ev,2):
        pairs[(i,j)]['p' if s in PRIM else 'np'].append(rel(T[s][i],T[s][j]))

flips=[]; tested=0
for (i,j),d in pairs.items():
    if len(d['np'])>=3 and len(d['p'])>=1:
        tested+=1
        npmode=max(set(d['np']),key=d['np'].count)
        if d['np'].count(npmode)/len(d['np'])>=0.8 and all(x!=npmode for x in d['p']):
            flips.append((i,j))
print(f'event pairs tested={tested}  primate-shifted pairs={len(flips)} ({100*len(flips)/tested:.1f}%)')

ev_in_flip=set()
for i,j in flips: ev_in_flip |= {i,j}
ev_tested=set()
for (i,j),d in pairs.items():
    if len(d['np'])>=3 and len(d['p'])>=1: ev_tested |= {i,j}
obs=np.mean([CORT[e] for e in ev_in_flip]); base=np.mean([CORT[e] for e in ev_tested])
null=[np.mean([CORT[e] for e in rng.choice(list(ev_tested),size=len(ev_in_flip),replace=False)]) for _ in range(20000)]
null=np.array(null)
print(f'cortical fraction: shifted events={obs:.3f}  all tested events={base:.3f}  '
      f'permutation p={(null>=obs).mean():.4g}  (n_shifted={len(ev_in_flip)}, n_tested={len(ev_tested)})')
