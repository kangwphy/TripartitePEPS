#!/usr/bin/env python3
"""Strictly validate every chain and node diagnostic in the sector campaign."""
import csv, math, os, sys

ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..'))
OUT=os.path.join(ROOT,'data/rk_ising/mc','campaigns/local_sector_beta060_100k_16chains')
OBS=('S2A','S2B','S2C','S3','tS'); SIZES=(4,8,12,16,20,24,32,48,64,80,96)
errors=[]; n=0
for obs in OBS:
    for L in SIZES:
        for chain in range(1,17):
            seed=860000+100*L+chain
            stem=f'L{L:03d}_b0p60_n100000_eq100000_seed{seed}'
            result=os.path.join(OUT,obs,'raw',stem+'.csv')
            diag=os.path.join(OUT,obs,'diagnostics',stem+'_sectors.csv')
            try:
                with open(result) as f: rows=list(csv.DictReader(f))
                if len(rows)!=1: raise ValueError(f'{len(rows)} result rows')
                r=rows[0]
                expected={'L':str(L),'beta':'0.60','n_sweeps':'100000','n_eq':'100000',
                          'seed':str(seed),'n_gl':'12','update':'local','quantity':obs,'gauge':'user'}
                for key,value in expected.items():
                    if r.get(key)!=value: raise ValueError(f'{key}={r.get(key)!r}, expected {value!r}')
                required_tag='SECTORINITSECTORSUBSET' if obs=='tS' else 'SECTORINITSECTORMOVE'
                if required_tag not in r.get('tag',''): raise ValueError(f'missing sector tag {required_tag}')
                if not all(math.isfinite(float(r[key])) for key in ('tildeS','error','runtime_s')):
                    raise ValueError('non-finite result')
                with open(diag) as f: drows=list(csv.DictReader(f))
                if len(drows)!=12: raise ValueError(f'{len(drows)} diagnostic rows')
                nodes=set()
                for d in drows:
                    if (d['L'],d['beta'],d['quantity'],d['seed']) != (str(L),'0.60',obs,str(seed)):
                        raise ValueError('diagnostic metadata mismatch')
                    node=int(d['node']); nodes.add(node)
                    attempts=int(d['attempts']); accepts=int(d['accepts'])
                    counts=[int(x) for x in d['relative_sector_counts'].split(';')]
                    if attempts!=100000 or not (0<=accepts<=attempts): raise ValueError('invalid proposal counts')
                    if len(counts)!=1<<(2-1 if obs.startswith('S2') else 4-1 if obs=='S3' else 6-1):
                        raise ValueError('wrong sector histogram length')
                    if sum(counts)!=100000: raise ValueError('sector histogram does not sum to measurements')
                    if int(d['sectors_visited']) != sum(x>0 for x in counts):
                        raise ValueError('sectors_visited disagrees with histogram')
                    if abs(float(d['dominant_fraction'])-max(counts)/100000) > 1e-9:
                        raise ValueError('dominant_fraction disagrees with histogram')
                    if abs(float(d['acceptance'])-accepts/attempts) > 1e-9:
                        raise ValueError('acceptance disagrees with proposal counts')
                    if not (0 <= int(d['transitions']) < 100000):
                        raise ValueError('invalid transition count')
                    for key in ('lambda','mean_deltaK','error','acceptance','dominant_fraction'):
                        if not math.isfinite(float(d[key])): raise ValueError(f'non-finite {key}')
                if nodes != set(range(1,13)): raise ValueError('wrong GL node indices')
                n += 1
            except Exception as exc:
                errors.append(f'{obs} L={L} chain={chain}: {exc}')
if errors:
    print('\n'.join(errors[:100]),file=sys.stderr); print(f'validation failed: {len(errors)} files',file=sys.stderr); sys.exit(1)
print(f'validated {n} sector-mixing chains and {12*n} node diagnostics')
