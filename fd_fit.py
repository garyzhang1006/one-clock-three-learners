import numpy as np, pandas as pd, time
rng = np.random.default_rng(0)
df = pd.read_csv('event_data.csv')
SP = ['Hamster','Mouse','Rat','Rabbit','Spiny.Mouse','Guinea.Pig','Ferret','Cat','Macaque','Human']
PRIM = {'Macaque','Human'}
rows=[]
for si,s in enumerate(SP):
    for ei in range(len(df)):
        t = df[s].iloc[ei]
        if t and t>0:
            rows.append((ei,si,float(t),int(df['Cortical'].iloc[ei]),int(df['Limbic'].iloc[ei]), s in PRIM))
R = pd.DataFrame(rows, columns=['ev','sp','t','cort','limb','prim'])
print(f'observations={len(R)}  events={R.ev.nunique()}  species={R.sp.nunique()}')

def design(R, include_event=True):
    n=len(R); blocks=[np.eye(10)[R.sp.values]]
    if include_event:
        ids=pd.factorize(R.ev.values)[0]
        E=np.zeros((n,ids.max()+1)); E[np.arange(n),ids]=1
        blocks.append(E[:,1:])
    blocks.append((R.prim.values*R.cort.values).reshape(-1,1).astype(float))
    blocks.append((R.prim.values*R.limb.values).reshape(-1,1).astype(float))
    return np.hstack(blocks)

def hat(X):
    U,s,Vt = np.linalg.svd(X, full_matrices=False)
    keep = s > s.max()*1e-10
    return U[:,keep], keep.sum()

def scan(U, t, ks):
    best=(-2,None)
    for k in ks:
        y=np.log(t-k); p=U@(U.T@y)
        r=np.corrcoef(y,p)[0,1]
        if r>best[0]: best=(r,k)
    return best

def coarse_fine(U, t):
    lo=t.min()
    r,k = scan(U,t,np.linspace(-40, lo-0.51, 60))
    r,k = scan(U,t,np.linspace(max(-60,k-3), min(lo-0.51,k+3), 60))
    return r,k

def r2(U,t,k):
    y=np.log(t-k); p=U@(U.T@y)
    return 1-((y-p)**2).sum()/((y-y.mean())**2).sum()

t=R.t.values
Uf,df_f = hat(design(R,True));  rho_full,k_full = coarse_fine(Uf,t)
Ur,df_r = hat(design(R,False)); rho_rate,k_rate = coarse_fine(Ur,t)
print(f'FULL  species+event+primate : rho={rho_full:.4f} R2={r2(Uf,t,k_full):.4f} k={k_full:.2f} rank={df_f}')
print(f'RATE-ONLY species+primate   : rho={rho_rate:.4f} R2={r2(Ur,t,k_rate):.4f} k={k_rate:.2f} rank={df_r}')

cnt=R.ev.value_counts(); R2p=R[R.ev.map(cnt)>=2].reset_index(drop=True)
t2=R2p.t.values; U2,df2 = hat(design(R2p,True)); rho_2p,k_2p = coarse_fine(U2,t2)
print(f'>=2 species  n={len(R2p)} events={R2p.ev.nunique()} rank={df2}: rho={rho_2p:.4f} R2={r2(U2,t2,k_2p):.4f} k={k_2p:.2f}')

def perm_t(R, t, rng):
    tp=t.copy()
    for s in np.unique(R.sp.values):
        m=R.sp.values==s; v=tp[m].copy(); rng.shuffle(v); tp[m]=v
    return tp

t0=time.time()
for label,Rb,U,tb,obs in [('ALL',R,Uf,t,rho_full), ('>=2SP',R2p,U2,t2,rho_2p)]:
    null=np.array([coarse_fine(U, perm_t(Rb,tb,rng))[0] for _ in range(300)])
    print(f'{label}: obs rho={obs:.4f} | null mean={null.mean():.4f} sd={null.std():.4f} max={null.max():.4f} | p={(null>=obs).mean():.4g}')
print(f'{time.time()-t0:.1f}s')
