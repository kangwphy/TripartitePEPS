#!/usr/bin/env python3
"""Compare the complete fresh local campaign with available SW components."""
import csv, math, os
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..'))
LOCAL=os.path.join(ROOT,'data/rk_ising/mc','campaigns/local_full_100k_16chains','pooled_components.csv')
SW=os.path.join(ROOT,'data/rk_ising/mc','campaigns/sw_components_100k_16chains','pooled_components.csv')
OUT=os.path.join(ROOT,'data/rk_ising/mc','campaigns/local_full_100k_16chains')
BETAS=('0.30','0.60','crit'); SIZES=(4,8,12,16,20,24,32,48,64,80,96); OBS=('S2A','S2B','S2C','S3','tS')

def read(path):
    d={}
    with open(path) as f:
        for r in csv.DictReader(f):
            d[(r['beta'],int(r['L']),r['observable'])]=(float(r['value']),float(r['error']),int(r['n_chains']))
    return d

local,sw=read(LOCAL),read(SW); rows=[]
for b in BETAS:
    for L in SIZES:
        for o in OBS:
            lp=local.get((b,L,o)); sp=sw.get((b,L,o))
            if lp is None: continue
            if sp is None:
                rows.append((b,L,o,*lp,'','','','','missing'))
            else:
                diff=lp[0]-sp[0]; err=math.hypot(lp[1],sp[1]); rows.append((b,L,o,*lp,*sp,diff,err,diff/err,'complete' if sp[2]>=16 else 'incomplete'))
with open(os.path.join(OUT,'local_vs_sw_all_observables.csv'),'w',newline='') as f:
    w=csv.writer(f);w.writerow(['beta','L','observable','local_value','local_error','local_n','sw_value','sw_error','sw_n','difference','difference_error','z_score','sw_status']);w.writerows(rows)

fig,axes=plt.subplots(3,5,figsize=(16,8.5),dpi=180,sharex='col')
for i,b in enumerate(BETAS):
    for j,o in enumerate(OBS):
        ax=axes[i,j]; q=[r for r in rows if r[0]==b and r[2]==o]
        ax.errorbar([r[1]*.985 for r in q],[r[3] for r in q],yerr=[r[4] for r in q],fmt='s-',ms=3,capsize=2,lw=1,color='#b55223',label='local')
        qs=[r for r in q if r[-1] != 'missing']
        if qs:
            ax.errorbar([r[1]*1.015 for r in qs],[r[6] for r in qs],yerr=[r[7] for r in qs],fmt='o-',ms=3,capsize=2,lw=1,color='#1f5a85',label='SW')
            for r in qs:
                if r[-1]=='incomplete': ax.plot(r[1]*1.015,r[6],marker='x',ms=7,color='black')
        ax.set_xticks(SIZES);ax.set_xticklabels([str(x) for x in SIZES],rotation=60,fontsize=7);ax.grid(alpha=.2)
        if i==0: ax.set_title({'S2A':r'$S_2(A)$','S2B':r'$S_2(B)$','S2C':r'$S_2(C)$','S3':r'$S_2^{(3)}$','tS':r'$\widetilde S$'}[o])
        if j==0: ax.set_ylabel(r'$\beta=\beta_c$' if b=='crit' else rf'$\beta={b}$')
        if i==2: ax.set_xlabel(r'$L$')
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=2,frameon=False);fig.tight_layout(rect=(0,0,1,.96));fig.savefig(os.path.join(OUT,'local_vs_sw_all_observables.png'),bbox_inches='tight')

with open(os.path.join(OUT,'local_vs_sw_summary.txt'),'w') as f:
    for b in BETAS:
        q=[r for r in rows if r[0]==b and r[-1] != 'missing']
        worst=max(q,key=lambda r:abs(float(r[-2])))
        f.write(f"{b}: compared={len(q)}, max_abs_z={abs(float(worst[-2])):.6g} at L={worst[1]} {worst[2]}\n")
    f.write(f"missing_SW={sum(r[-1]=='missing' for r in rows)}, incomplete_SW={sum(r[-1]=='incomplete' for r in rows)}\n")
print('wrote local-vs-SW comparison to',OUT)
