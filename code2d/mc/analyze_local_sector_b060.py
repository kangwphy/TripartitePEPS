#!/usr/bin/env python3
"""Pool the sector-mixing campaign and compare it with old local and SW data."""
import csv, glob, importlib, math, os, sys
from collections import defaultdict
PLOTS=getattr(os,'environ',{}).get('MC_NO_PLOTS','0') != '1'
if PLOTS:
    # Some SMP compute images expose user-site matplotlib without packaging.
    # Setuptools vendors the same API, so use it as a compatibility fallback.
    try:
        import packaging  # noqa: F401
    except ModuleNotFoundError:
        import setuptools._vendor.packaging as packaging
        sys.modules['packaging']=packaging
        for name in ('version','specifiers','requirements','markers','utils','tags'):
            sys.modules[f'packaging.{name}']=importlib.import_module(f'setuptools._vendor.packaging.{name}')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..'))
OUT=os.path.join(ROOT,'data/rk_ising/mc','campaigns/local_sector_beta060_100k_16chains')
OLD=os.path.join(ROOT,'data/rk_ising/mc','campaigns/local_full_100k_16chains','pooled_components.csv')
SW=os.path.join(ROOT,'data/rk_ising/mc','campaigns/sw_components_100k_16chains','pooled_components.csv')
OBS=('S2A','S2B','S2C','S3','tS'); SIZES=(4,8,12,16,20,24,32,48,64,80,96)

def pool(vals):
    n=len(vals); mean=sum(v for v,_ in vals)/n
    within=math.sqrt(sum(e*e for _,e in vals))/n
    sem=math.sqrt(sum((v-mean)**2 for v,_ in vals)/(n*(n-1))) if n>1 else within
    return mean,max(within,sem),within,sem,n

raw=defaultdict(list); chain_rows=[]
for obs in OBS:
    for path in sorted(glob.glob(os.path.join(OUT,obs,'raw','*.csv'))):
        with open(path) as f: r=next(csv.DictReader(f))
        row={'observable':obs,**r}; chain_rows.append(row)
        raw[(int(r['L']),obs)].append((float(r['tildeS']),float(r['error'])))
with open(os.path.join(OUT,'all_chains.csv'),'w',newline='') as f:
    fields=('observable','L','beta','n_sweeps','n_eq','seed','tildeS','error','runtime_s','n_gl','update','quantity','gauge','tag')
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows({k:r[k] for k in fields} for r in chain_rows)

new={}
with open(os.path.join(OUT,'pooled_components.csv'),'w',newline='') as f:
    w=csv.writer(f); w.writerow(['beta','L','observable','value','error','within_chain_error','chain_sem','n_chains'])
    for L in SIZES:
        for obs in OBS:
            p=pool(raw[(L,obs)]); new[(L,obs)]=p; w.writerow(['0.60',L,obs,*p])

def read_pooled(path):
    out={}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r['beta']=='0.60': out[(int(r['L']),r['observable'])]=(float(r['value']),float(r['error']),int(r['n_chains']))
    return out
old=read_pooled(OLD); sw=read_pooled(SW)

comparison=[]
with open(os.path.join(OUT,'sector_local_vs_old_local_vs_sw.csv'),'w',newline='') as f:
    fields=['L','observable','sector_value','sector_error','sector_n','old_local_value','old_local_error','old_local_n','sw_value','sw_error','sw_n','sector_minus_sw','sector_minus_sw_error','sector_minus_sw_z']
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
    for L in SIZES:
        for obs in OBS:
            nv,ne,_,_,nn=new[(L,obs)]; ov,oe,on=old[(L,obs)]; sv,se,sn=sw[(L,obs)]
            de=math.hypot(ne,se); row=dict(L=L,observable=obs,sector_value=nv,sector_error=ne,sector_n=nn,
                old_local_value=ov,old_local_error=oe,old_local_n=on,sw_value=sv,sw_error=se,sw_n=sn,
                sector_minus_sw=nv-sv,sector_minus_sw_error=de,sector_minus_sw_z=(nv-sv)/de)
            comparison.append(row); w.writerow(row)

titles={'S2A':r'$S_2(A)$','S2B':r'$S_2(B)$','S2C':r'$S_2(C)$','S3':r'$S_2^{(3)}$','tS':r'$\widetilde S$'}
if PLOTS:
    fig,axes=plt.subplots(1,5,figsize=(16,3.3),dpi=180)
    for ax,obs in zip(axes,OBS):
        q=[r for r in comparison if r['observable']==obs]
        ax.errorbar([r['L']*.98 for r in q],[r['old_local_value'] for r in q],yerr=[r['old_local_error'] for r in q],fmt='^-',ms=3,lw=1,capsize=2,color='0.45',label='old local')
        ax.errorbar([r['L'] for r in q],[r['sector_value'] for r in q],yerr=[r['sector_error'] for r in q],fmt='s-',ms=3,lw=1,capsize=2,color='#b55223',label='local + sector')
        ax.errorbar([r['L']*1.02 for r in q],[r['sw_value'] for r in q],yerr=[r['sw_error'] for r in q],fmt='o-',ms=3,lw=1,capsize=2,color='#1f5a85',label='SW')
        ax.set_title(titles[obs]); ax.set_xlabel(r'$L$'); ax.set_xticks(SIZES); ax.tick_params(axis='x',rotation=60,labelsize=7); ax.grid(alpha=.22)
    axes[0].set_ylabel(r'$\beta=0.60$'); handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False); fig.tight_layout(rect=(0,0,1,.9)); fig.savefig(os.path.join(OUT,'sector_local_vs_old_local_vs_sw.png'),bbox_inches='tight')

# Direct-vs-reconstructed endpoint-identity check.  The two estimators use
# different TI paths, so agreement is also a stringent quadrature test.
reconstruction=[]
with open(os.path.join(OUT,'reconstruction_vs_direct.csv'),'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['beta','L','reconstructed_tS','reconstructed_error',
        'direct_tS','direct_error','difference','difference_error','z_score',
        'sw_reconstructed_tS','sw_reconstructed_error','sw_direct_tS','sw_direct_error',
        'sw_difference','sw_difference_error','sw_z_score'])
    w.writeheader()
    for L in SIZES:
        p={o:new[(L,o)] for o in OBS}
        rv=2*p['S3'][0]-p['S2A'][0]-p['S2B'][0]-p['S2C'][0]
        re=math.sqrt((2*p['S3'][1])**2+sum(p[o][1]**2 for o in ('S2A','S2B','S2C')))
        diff=rv-p['tS'][0]; de=math.hypot(re,p['tS'][1])
        sp={o:sw[(L,o)] for o in OBS}
        srv=2*sp['S3'][0]-sp['S2A'][0]-sp['S2B'][0]-sp['S2C'][0]
        sre=math.sqrt((2*sp['S3'][1])**2+sum(sp[o][1]**2 for o in ('S2A','S2B','S2C')))
        sdiff=srv-sp['tS'][0]; sde=math.hypot(sre,sp['tS'][1])
        row={'beta':'0.60','L':L,'reconstructed_tS':rv,'reconstructed_error':re,
             'direct_tS':p['tS'][0],'direct_error':p['tS'][1],
             'difference':diff,'difference_error':de,'z_score':diff/de,
             'sw_reconstructed_tS':srv,'sw_reconstructed_error':sre,
             'sw_direct_tS':sp['tS'][0],'sw_direct_error':sp['tS'][1],
             'sw_difference':sdiff,'sw_difference_error':sde,'sw_z_score':sdiff/sde}
        reconstruction.append(row); w.writerow(row)

if PLOTS:
    fig,ax=plt.subplots(figsize=(7.2,4.2),dpi=180)
    Ls=[r['L'] for r in reconstruction]
    ax.errorbar([x*.985 for x in Ls],[r['reconstructed_tS'] for r in reconstruction],
                yerr=[r['reconstructed_error'] for r in reconstruction],fmt='s-',ms=4,lw=1,
                capsize=2,color='#b55223',label='local + sector: components')
    ax.errorbar([x*1.015 for x in Ls],[r['direct_tS'] for r in reconstruction],
                yerr=[r['direct_error'] for r in reconstruction],fmt='o-',ms=4,lw=1,
                capsize=2,color='#7a3e9d',label='local + sector: direct')
    ax.errorbar(Ls,[r['sw_reconstructed_tS'] for r in reconstruction],
                yerr=[r['sw_reconstructed_error'] for r in reconstruction],fmt='s--',mfc='none',
                ms=4,lw=1,capsize=2,color='#1f5a85',label='SW: components')
    ax.errorbar(Ls,[r['sw_direct_tS'] for r in reconstruction],
                yerr=[r['sw_direct_error'] for r in reconstruction],fmt='o--',mfc='none',
                ms=4,lw=1,capsize=2,color='#287271',label='SW: direct')
    ax.axhline(0,color='0.55',lw=.8,ls=':'); ax.set_xlabel(r'$L$'); ax.set_ylabel(r'$\widetilde S$')
    ax.set_xticks(SIZES); ax.tick_params(axis='x',rotation=45); ax.grid(alpha=.22)
    ax.legend(frameon=False,ncol=2,fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(OUT,'direct_vs_reconstructed.png'),bbox_inches='tight')

# Pool per-node sector diagnostics over independent chains.
diag=defaultdict(list)
for obs in OBS:
    for path in glob.glob(os.path.join(OUT,obs,'diagnostics','*.csv')):
        with open(path) as f:
            for r in csv.DictReader(f): diag[(obs,int(r['L']),int(r['node']))].append(r)
diag_pool={}
with open(os.path.join(OUT,'sector_diagnostics_pooled.csv'),'w',newline='') as f:
    fields=['observable','L','node','lambda','mean_deltaK','error','acceptance','mean_sectors_visited','mean_dominant_fraction','mean_transitions','n_chains']
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
    for obs in OBS:
        for L in SIZES:
            for node in range(1,13):
                rows=diag[(obs,L,node)]; vals=[(float(r['mean_deltaK']),float(r['error'])) for r in rows]; p=pool(vals)
                attempts=sum(int(r['attempts']) for r in rows); accepts=sum(int(r['accepts']) for r in rows)
                out={'observable':obs,'L':L,'node':node,
                    'lambda':sum(float(r['lambda']) for r in rows)/len(rows),
                    'mean_deltaK':p[0],'error':p[1],'acceptance':accepts/attempts,
                    'mean_sectors_visited':sum(int(r['sectors_visited']) for r in rows)/len(rows),
                    'mean_dominant_fraction':sum(float(r['dominant_fraction']) for r in rows)/len(rows),
                    'mean_transitions':sum(int(r['transitions']) for r in rows)/len(rows),
                    'n_chains':len(rows)}
                diag_pool[(obs,L,node)]=out; w.writerow(out)

if PLOTS:
    fig,axes=plt.subplots(2,5,figsize=(16,6),dpi=180,sharex='col')
    colors={24:'#4c78a8',48:'#d07c32',96:'#287271'}
    for j,obs in enumerate(OBS):
        for L in (24,48,96):
            q=[diag_pool[(obs,L,n)] for n in range(1,13)]
            axes[0,j].plot([r['lambda'] for r in q],[r['acceptance'] for r in q],'o-',ms=3,lw=1,color=colors[L],label=f'L={L}')
            axes[1,j].plot([r['lambda'] for r in q],[r['mean_dominant_fraction'] for r in q],'o-',ms=3,lw=1,color=colors[L])
        axes[0,j].set_title(titles[obs]); axes[1,j].set_xlabel(r'$\lambda$')
        for i in (0,1): axes[i,j].grid(alpha=.22); axes[i,j].set_xlim(0,1)
    axes[0,0].set_ylabel('sector-flip acceptance'); axes[1,0].set_ylabel('dominant-sector fraction')
    axes[0,0].legend(frameon=False,fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(OUT,'sector_mixing_diagnostics.png'),bbox_inches='tight')

# The direct six-layer path develops narrow endpoint structure as L grows.
# Showing the sampled integrand makes fixed-order quadrature limitations
# visible instead of conflating them with Markov-chain convergence.
if PLOTS:
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),dpi=180)
    profile_sizes=(24,48,64,80,96)
    profile_colors=plt.cm.viridis([0,.25,.5,.75,1])
    for L,c in zip(profile_sizes,profile_colors):
        q=[diag_pool[('tS',L,n)] for n in range(1,13)]
        x=[r['lambda'] for r in q]; y=[r['mean_deltaK'] for r in q]
        for ax in axes: ax.plot(x,y,'o-',ms=3,lw=1,color=c,label=f'L={L}')
    axes[0].set_xlim(0,1); axes[0].set_title('full path')
    axes[1].set_xlim(0,.13); axes[1].set_title(r'$\lambda\to0$')
    axes[2].set_xlim(.87,1); axes[2].set_title(r'$\lambda\to1$')
    for ax in axes:
        ax.axhline(0,color='0.55',lw=.8,ls=':'); ax.set_xlabel(r'$\lambda$'); ax.grid(alpha=.22)
    axes[0].set_ylabel(r'$\langle\Delta K\rangle_\lambda$')
    axes[0].legend(frameon=False,fontsize=8,ncol=1); fig.tight_layout()
    fig.savefig(os.path.join(OUT,'direct_integrand_endpoint_profiles.png'),bbox_inches='tight')

with open(os.path.join(OUT,'comparison_summary.txt'),'w') as f:
    for obs in OBS:
        q=[r for r in comparison if r['observable']==obs]
        worst=max(q,key=lambda r:abs(r['sector_minus_sw_z']))
        f.write(f"{obs}: max_abs_z={abs(worst['sector_minus_sw_z']):.6g} at L={worst['L']}\n")
    worst=max(reconstruction,key=lambda r:abs(r['z_score']))
    sw_worst=max(reconstruction,key=lambda r:abs(r['sw_z_score']))
    f.write(f"local_sector identity: max_abs_z={abs(worst['z_score']):.6g} at L={worst['L']}\n")
    f.write(f"SW identity: max_abs_z={abs(sw_worst['sw_z_score']):.6g} at L={sw_worst['L']}\n")
print('wrote sector-aware local analysis to',OUT)
