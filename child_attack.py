# Does the child panel ride on non-random missingness the way the machine panel might?
# Restrict to lemmas present in progressively more of the fifteen languages and refit.
import numpy as np, pandas as pd
from final_results import fitk0, nullk0
a=pd.read_feather('aoas.feather')
ch=a[(a.measure=='produces')&a.aoa.notna()&a.uni_lemma.notna()]
nl=ch.language.nunique()
print("languages",nl)
for thr in [2,8,12,15]:
    c=ch.uni_lemma.value_counts(); sub=ch[ch.uni_lemma.map(c)>=thr]
    if sub.uni_lemma.nunique()<5: print(f">= {thr:2d}/15  too few lemmas"); continue
    u=sub.language.values; e=sub.uni_lemma.values; t=sub.aoa.values.astype(float)
    f=fitk0(u,e,t); fr=fitk0(u,e,t,inc=False); nd=nullk0(u,e,t,n=300)
    z=(f['rho']-nd.mean())/nd.std()
    print(f">= {thr:2d}/15  lemmas={sub.uni_lemma.nunique():4d} cells={len(t):5d} "
          f"rho={f['rho']:.4f} R2={f['r2']:.4f} rate-only R2={fr['r2']:.4f} "
          f"null={nd.mean():.4f}+-{nd.std():.4f} z={z:.1f}")
