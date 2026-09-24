#!/usr/bin/env python3
"""Pool local full entropy and single-layer Ising benchmark outputs."""
import csv, glob, math, os
from collections import defaultdict
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..')); OUT=os.path.join(ROOT,'data/rk_ising/mc','campaigns/local_full_100k_16chains')
OBS=('S2A','S2B','S2C','S3','tS'); BETAS=('0.30','0.60','crit'); SIZES=(4,8,12,16,20,24,32,48,64,80,96)
raw=defaultdict(list)
with open(os.path.join(OUT,'all_chains.csv')) as f:
    for r in csv.DictReader(f): raw[(r['beta'],int(r['L']),r['observable'])].append((float(r['value']),float(r['error'])))
def pool(vals):
    if not vals:return None
    n=len(vals); m=sum(v for v,_ in vals)/n
    within=math.sqrt(sum(e*e for _,e in vals))/n
    sem=math.sqrt(sum((v-m)**2 for v,_ in vals)/(n*(n-1))) if n>1 else within
    return m,max(within,sem),within,sem,n

def ising_infinite_volume(beta):
    """Exact square-lattice Ising magnetization and energy per site."""
    beta_c=0.5*math.log(1+math.sqrt(2))
    if abs(beta-beta_c)<1e-12:
        return 0.0,-math.sqrt(2)
    magnetization=(1-math.sinh(2*beta)**-4)**0.125 if beta>beta_c else 0.0
    k=2*math.sinh(2*beta)/math.cosh(2*beta)**2
    a,b=1.0,math.sqrt(max(0.0,1-k*k))
    for _ in range(100):
        na=(a+b)/2; nb=math.sqrt(a*b)
        if abs(na-nb)<1e-15: a=na; break
        a,b=na,nb
    elliptic_k=math.pi/(2*a)
    energy=-1/math.tanh(2*beta)*(1+(2/math.pi)*(2*math.tanh(2*beta)**2-1)*elliptic_k)
    return magnetization,energy
pooled={}
with open(os.path.join(OUT,'pooled_components.csv'),'w',newline='') as f:
    w=csv.writer(f); w.writerow(['beta','L','observable','value','error','within_chain_error','chain_sem','n_chains'])
    for b in BETAS:
        for L in SIZES:
            for o in OBS:
                p=pool(raw[(b,L,o)])
                if p: pooled[(b,L,o)]=p; w.writerow([b,L,o,*p])
with open(os.path.join(OUT,'reconstruction_vs_direct.csv'),'w',newline='') as f:
    w=csv.writer(f); w.writerow(['beta','L','reconstructed_tS','reconstructed_error','direct_tS','direct_error','difference','difference_error','z_score'])
    for b in BETAS:
        for L in SIZES:
            if not all((b,L,o) in pooled for o in OBS): continue
            p={o:pooled[(b,L,o)] for o in OBS}; rv=2*p['S3'][0]-p['S2A'][0]-p['S2B'][0]-p['S2C'][0]; re=math.sqrt((2*p['S3'][1])**2+sum(p[o][1]**2 for o in ('S2A','S2B','S2C'))); d=rv-p['tS'][0]; de=math.hypot(re,p['tS'][1]); w.writerow([b,L,rv,re,p['tS'][0],p['tS'][1],d,de,d/de])
colors={'S2A':'#287271','S2B':'#d07c32','S2C':'#4c78a8','S3':'#222222'}; labels={'S2A':'S2(A)','S2B':'S2(B)','S2C':'S2(C)','S3':'S2^(3)'}
fig,ax=plt.subplots(2,3,figsize=(12,7),dpi=180,sharex='col')
for j,b in enumerate(BETAS):
    for o in ('S2A','S2B','S2C','S3'):
        q=[(L,*pooled[(b,L,o)][:2]) for L in SIZES if (b,L,o) in pooled]
        if q: ax[0,j].errorbar([x[0] for x in q],[x[1] for x in q],yerr=[x[2] for x in q],fmt='o-',ms=3,capsize=2,label=labels[o],color=colors[o])
    q=[]
    for L in SIZES:
        if all((b,L,o) in pooled for o in OBS):
            p={o:pooled[(b,L,o)] for o in OBS}; rv=2*p['S3'][0]-p['S2A'][0]-p['S2B'][0]-p['S2C'][0]; re=math.sqrt((2*p['S3'][1])**2+sum(p[o][1]**2 for o in ('S2A','S2B','S2C'))); q.append((L,rv,re,p['tS'][0],p['tS'][1]))
    if q:
        ax[1,j].errorbar([x[0] for x in q],[x[1] for x in q],yerr=[x[2] for x in q],fmt='o-',label='reconstructed',color='#b55223'); ax[1,j].errorbar([x[0] for x in q],[x[3] for x in q],yerr=[x[4] for x in q],fmt='s-',label='direct',color='#2e5fa3')
    ax[0,j].set_title('beta = beta_c' if b=='crit' else 'beta = '+b); ax[1,j].set_xlabel('L')
    for k in (0,1): ax[k,j].grid(alpha=.22)
ax[0,0].set_ylabel('component entropy'); ax[1,0].set_ylabel('tilde S'); ax[0,0].legend(frameon=False,fontsize=8); ax[1,0].legend(frameon=False,fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(OUT,'local_full_summary.png'),bbox_inches='tight')

# Pool physical single-layer observables measured by the benchmark chains.
bench=defaultdict(list); benchmark_file=os.path.join(OUT,'benchmark','all_chains.csv')
if os.path.exists(benchmark_file):
    with open(benchmark_file) as f:
        for r in csv.DictReader(f):
            error_field={'m_abs':'m_abs_error','m2':'m2_error',
                         'energy_per_site':'energy_error','chi':'chi_error'}
            for name in ('m_abs','m2','energy_per_site','chi'):
                bench[(r['beta'],int(r['L']),name)].append(
                    (float(r[name]),float(r[error_field[name]])))
            beta=0.5*math.log(1+math.sqrt(2)) if r['beta']=='crit' else float(r['beta'])
            nsite=int(r['L'])**2; ma=float(r['m_abs']); ma_err=float(r['m_abs_error'])
            m2=float(r['m2']); m2_err=float(r['m2_error'])
            chi_abs=beta*nsite*(m2-ma*ma)
            chi_abs_err=beta*nsite*math.sqrt(m2_err*m2_err+(2*ma*ma_err)**2)
            bench[(r['beta'],int(r['L']),'chi_abs')].append((chi_abs,chi_abs_err))
    with open(os.path.join(OUT,'benchmark_pooled.csv'),'w',newline='') as f:
        w=csv.writer(f); w.writerow(['beta','L','observable','value','error','within_chain_error','chain_sem','n_chains'])
        for b in BETAS:
            for L in SIZES:
                for name in ('m_abs','m2','energy_per_site','chi','chi_abs'):
                    p=pool(bench[(b,L,name)])
                    if p: w.writerow([b,L,name,*p])
    fig,axes=plt.subplots(3,3,figsize=(12,9),dpi=180,sharex='col')
    names=('m_abs','energy_per_site','chi_abs'); ylabels=(r'$\langle |m|\rangle$',r'$E/N$',r'$\chi_{|m|}$')
    for j,b in enumerate(BETAS):
        beta=0.5*math.log(1+math.sqrt(2)) if b=='crit' else float(b)
        exact_m,exact_e=ising_infinite_volume(beta)
        for i,(name,ylabel) in enumerate(zip(names,ylabels)):
            q=[(L,*pool(bench[(b,L,name)])[:2]) for L in SIZES if pool(bench[(b,L,name)])]
            if q: axes[i,j].errorbar([x[0] for x in q],[x[1] for x in q],yerr=[x[2] for x in q],fmt='o-',ms=3,capsize=2,color=('#287271','#4c78a8','#b55223')[i])
            axes[i,j].grid(alpha=.22); axes[i,0].set_ylabel(ylabel)
        axes[0,j].axhline(exact_m,color='0.25',ls='--',lw=1,label='exact infinite-volume')
        axes[1,j].axhline(exact_e,color='0.25',ls='--',lw=1)
        axes[0,j].set_title('beta = beta_c' if b=='crit' else 'beta = '+b); axes[2,j].set_xlabel('L')
    axes[0,0].legend(frameon=False,fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(OUT,'ising_benchmark.png'),bbox_inches='tight')
print('wrote analysis to',OUT)
