import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, pandas as pd, numpy as np
plt.rcParams.update({'font.size':8,'font.family':'serif','axes.linewidth':0.6})
T=pd.read_csv('final_table.csv').set_index('system')
import pandas as _pd
_T=_pd.read_csv('final_table.csv').set_index('system')
LAB={'mammals':f"Mammals ({int(_T.loc['mammals','units'])} species)",
     'children':f"Children ({int(_T.loc['children','units'])} languages)",
     'machines':f"Machines ({int(_T.loc['machines','units'])} model scales)"}
UNITS={'mammals':'log post-conceptional day','children':'log age (months)','machines':'log cumulative tokens'}
fig,ax=plt.subplots(1,3,figsize=(7.0,2.45))
for i,s in enumerate(['mammals','children','machines']):
    d=pd.read_csv(f'panel_{s}.csv'); a=ax[i]
    lo,hi=min(d.pred.min(),d.obs.min()),max(d.pred.max(),d.obs.max()); pad=0.06*(hi-lo)
    a.plot([lo-pad,hi+pad],[lo-pad,hi+pad],color='0.6',lw=0.7,ls='--',zorder=1)
    a.scatter(d.pred,d.obs,s=5,alpha=0.45,edgecolors='none',color='#1f4e79',zorder=2)
    a.set_xlim(lo-pad,hi+pad); a.set_ylim(lo-pad,hi+pad)
    a.set_title(LAB[s],fontsize=8,pad=4)
    a.set_xlabel('predicted '+UNITS[s],fontsize=7)
    if i==0: a.set_ylabel('observed',fontsize=7)
    r=T.loc[s]
    a.text(0.04,0.95,f"$\\rho$ = {r.rho:.3f}\n$R^2$ = {r.r2:.3f}\nnull $\\rho$ = {r.null_mean:.3f}\nz = {r.z:.1f}",
           transform=a.transAxes,va='top',ha='left',fontsize=6.6)
    a.tick_params(labelsize=6.5,length=2)
plt.tight_layout(); plt.savefig('fig1.pdf',bbox_inches='tight'); plt.close()

fig,ax=plt.subplots(1,2,figsize=(7.0,2.35))
sy=['mammals','children','machines']; x=np.arange(3); w=0.36
ax[0].bar(x-w/2,[T.loc[s,'r2_rate'] for s in sy],w,label='unit rate only',color='#9ecae1',edgecolor='k',lw=0.4)
ax[0].bar(x+w/2,[T.loc[s,'r2'] for s in sy],w,label='rate + conserved order',color='#1f4e79',edgecolor='k',lw=0.4)
ax[0].set_xticks(x); ax[0].set_xticklabels(['mammals','children','machines'],fontsize=7)
ax[0].set_ylabel('$R^2$ of log developmental time',fontsize=7.5); ax[0].set_ylim(0,1.05)
ax[0].legend(fontsize=6.5,frameon=False,loc='upper right'); ax[0].tick_params(labelsize=6.5,length=2)
import json; RT=json.load(open('final_rates.json'))
STY={'mammals':('#1f4e79','o'),'children':('#4f8a3d','^'),'machines':('#c0504d','s')}
for sysname in ['mammals','children','machines']:
    v=sorted(RT[sysname+'_rates'].values()); v=[y/v[0] for y in v]
    xr=np.linspace(0,1,len(v)) if len(v)>1 else [0.5]
    c,m=STY[sysname]
    ax[1].plot(xr,v,marker=m,ms=3.5,lw=1.0,color=c,
               label=f'{sysname} ({len(v)} units, {T.loc[sysname,"rate_span"]:.2f}$\\times$)')
ax[1].set_ylabel('unit rate (fold slower than fastest unit)',fontsize=7.5)
ax[1].set_xlabel('unit rank within system, fastest to slowest (normalised)',fontsize=7)
ax[1].set_ylim(0.8,6.1)   # must clear the 5.73x machine span, which 5.4 was clipping
ax[1].legend(fontsize=6.3,frameon=False,loc='upper left'); ax[1].tick_params(labelsize=6.5,length=2)
plt.tight_layout(); plt.savefig('fig2.pdf',bbox_inches='tight'); plt.close()
print('figs written')
