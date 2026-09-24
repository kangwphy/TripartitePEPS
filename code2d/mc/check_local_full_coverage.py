#!/usr/bin/env python3
"""Write current per-point coverage for the isolated local campaign."""
import csv, glob, os
from collections import Counter
root=os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..'))
out=os.path.join(root,'data/rk_ising/mc','campaigns/local_full_100k_16chains')
sizes=(4,8,12,16,20,24,32,48,64,80,96); betas=('0.30','0.60','crit'); obs=('S2A','S2B','S2C','S3','tS','benchmark')
with open(os.path.join(out,'coverage_current.csv'),'w',newline='') as f:
    w=csv.writer(f); w.writerow(['observable','beta','L','n_chains','complete_16'])
    for o in obs:
        c=Counter()
        for p in glob.glob(os.path.join(out,o,'raw','*.csv')):
            with open(p) as stream:r=next(csv.DictReader(stream))
            c[(r['beta'],int(r['L']))]+=1
        for b in betas:
            for L in sizes:w.writerow([o,b,L,c[(b,L)],c[(b,L)]>=16])
print('wrote',os.path.join(out,'coverage_current.csv'))
