#!/usr/bin/env python3
"""Fail unless every requested local entropy and benchmark chain is valid."""
import csv, glob, os, sys
root=os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..'))
out=os.path.join(root,'data/rk_ising/mc','campaigns/local_full_100k_16chains')
sizes=(4,8,12,16,20,24,32,48,64,80,96); betas=('0.30','0.60','crit'); obs=('S2A','S2B','S2C','S3','tS')
errors=[]
def beta_key(value): return value if value=='crit' else f'{float(value):.2f}'
for name in (*obs,'benchmark'):
    rows={}
    for path in glob.glob(os.path.join(out,name,'raw','*.csv')):
        try:
            with open(path) as f:r=next(csv.DictReader(f))
            key=(beta_key(r['beta']),int(r['L']),int(r['seed']))
            if key in rows: errors.append(f'duplicate key {name} {key}')
            rows[key]=r
            if int(r['n_sweeps'])!=100000 or int(r['n_eq'])!=100000: errors.append(f'wrong sweeps {path}')
            if r['update']!='local': errors.append(f'wrong update {path}')
            if name in obs and r['quantity']!=name: errors.append(f'wrong quantity {path}')
            if name in obs and int(r['n_gl'])!=12: errors.append(f'wrong quadrature order {path}')
            if name=='benchmark' and r['quantity']!='benchmark': errors.append(f'wrong benchmark quantity {path}')
            if name=='benchmark' and r['tag']!='LOCAL_ISING_OPEN_V2': errors.append(f'wrong benchmark version {path}')
        except Exception as exc: errors.append(f'unreadable {path}: {exc}')
    for beta in betas:
        boff=0 if beta=='0.30' else 100000 if beta=='0.60' else 200000
        for L in sizes:
            offset={'S2A':0,'S2B':1000,'S2C':2000,'S3':3000,'tS':4000}.get(name)
            expected=[900000+boff+L*100+c if name=='benchmark' else 700000+boff+L*100+offset+c for c in range(1,17)]
            missing=[seed for seed in expected if (beta,L,seed) not in rows]
            if missing: errors.append(f'missing {name} beta={beta} L={L}: {missing}')
if errors:
    print('\n'.join(errors)); print(f'validation failed with {len(errors)} issue(s)'); sys.exit(1)
print('validated 3168 independent-chain files: 2640 entropy + 528 benchmark')
