#!/usr/bin/env python3
"""Report result/diagnostic coverage for the beta=0.6 sector-mixing campaign."""
import os

ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..'))
OUT=os.path.join(ROOT,'data/rk_ising/mc','campaigns/local_sector_beta060_100k_16chains')
OBS=('S2A','S2B','S2C','S3','tS')
SIZES=(4,8,12,16,20,24,32,48,64,80,96)
complete=0; missing=[]
for obs in OBS:
    for L in SIZES:
        for chain in range(1,17):
            seed=860000+100*L+chain
            stem=f'L{L:03d}_b0p60_n100000_eq100000_seed{seed}'
            result=os.path.join(OUT,obs,'raw',stem+'.csv')
            diag=os.path.join(OUT,obs,'diagnostics',stem+'_sectors.csv')
            if os.path.getsize(result) > 0 if os.path.exists(result) else False:
                if os.path.getsize(diag) > 0 if os.path.exists(diag) else False:
                    complete += 1
                    continue
            missing.append((obs,L,chain,not os.path.exists(result),not os.path.exists(diag)))
print(f'complete={complete}/880 missing={len(missing)}')
for item in missing[:30]: print(*item)
