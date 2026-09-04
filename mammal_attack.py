# Does the mammal headline rest on the two fitted primate interaction terms, or on the primates?
import numpy as np, pandas as pd
from final_results import fitk0, nullk0
df=pd.read_csv('event_data.csv')
SP=['Hamster','Mouse','Rat','Rabbit','Spiny.Mouse','Guinea.Pig','Ferret','Cat','Macaque','Human']
PRIM={'Macaque','Human'}
rows=[(f'ev{e}',s,float(df[s].iloc[e]),int(df['Cortical'].iloc[e]),int(df['Limbic'].iloc[e]),s in PRIM)
      for s in SP for e in range(len(df)) if df[s].iloc[e]>0]
R=pd.DataFrame(rows,columns=['ev','sp','t','cort','limb','prim'])
def run(tag,D,use_ex):
    c=D.ev.value_counts(); D=D[D.ev.map(c)>=2]
    ex=np.c_[D.prim*D.cort,D.prim*D.limb].astype(float) if use_ex else None
    f=fitk0(D.sp.values,D.ev.values,D.t.values,ex)
    fr=fitk0(D.sp.values,D.ev.values,D.t.values,ex,inc=False)
    nd=nullk0(D.sp.values,D.ev.values,D.t.values,ex,n=1000)
    z=(f['rho']-nd.mean())/nd.std()
    print(f"{tag:34s} species={D.sp.nunique():2d} events={D.ev.nunique():3d} cells={len(D):4d} "
          f"rho={f['rho']:.4f} R2={f['r2']:.4f} rate-only R2={fr['r2']:.4f} null={nd.mean():.4f} z={z:.1f}")
run("full, with primate terms",       R, True)
run("full, primate terms dropped",    R, False)
run("primates removed entirely",      R[~R.prim], False)
