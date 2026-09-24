#!/usr/bin/env python3
"""Plot the original L=24 audit together with the 160-chain long audit."""
import csv, glob, math, os, re
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT = os.path.join(ROOT, 'data/rk_ising/mc', 'audits/critical_l24/long_thermalization')
os.makedirs(OUT, exist_ok=True)

def pool(values):
    # values are (mean, error), with chain scatter retained separately.
    w = [1.0/e**2 for v,e in values]
    m = sum(v*wi for (v,e),wi in zip(values,w))/sum(w)
    formal = 1.0/math.sqrt(sum(w))
    scatter = (math.sqrt(sum((v-m)**2 for v,e in values)/
                         (len(values)*(len(values)-1)))
               if len(values) > 1 else formal)
    return m, max(formal, scatter), formal, scatter

records = []
with open(os.path.join(ROOT, 'data/rk_ising/mc', 'audits/critical_l24/short', 'summary.csv')) as f:
    records.extend(list(csv.DictReader(f)))
records = [{**r, 'value': float(r['value']), 'error': float(r['error'])}
           for r in records]

profiles = defaultdict(list)
for r in records:
    path = os.path.join(ROOT, 'logs2d', r['log'])
    text = open(path).read()
    for lam, mean, err in re.findall(
            r'lam=([0-9.]+)\s+<dE/dl>=([+-]?[0-9.]+)\s+\+\-\s+([0-9.]+)', text):
        profiles[(r['mode'], float(lam))].append((float(mean), float(err)))

long_records = []
long_profiles = defaultdict(list)
for path in glob.glob(os.path.join(ROOT, 'logs2d', 'mc_l24_100w_*.out')):
    text = open(path).read()
    h = re.search(r'eq=1000000 seed=(\d+)', text)
    z = re.search(r'RESULT_MC.*tildeS = ([+-]?[0-9.]+) \+\- ([0-9.]+)', text)
    nodes = re.findall(r'lam=([0-9.]+)\s+<dE/dl>=([+-]?[0-9.]+)\s+\+\-\s+([0-9.]+)', text)
    if not h or not z or len(nodes) != 12:
        continue
    long_records.append((float(z.group(1)), float(z.group(2))))
    for lam, mean, err in nodes:
        long_profiles[float(lam)].append((float(mean), float(err)))

if len(long_records) != 160:
    raise RuntimeError(f'expected 160 long-audit records, found {len(long_records)}')

# Equally spaced lambda audits use composite trapezoidal integration.  These
# have no node-profile logs in the audit directory, but their per-chain CSVs
# provide the integrated values and uncertainties for the left panel.
trap_records = defaultdict(list)
for path in glob.glob(os.path.join(ROOT, 'data/rk_ising/mc', 'audits/critical_l24/linear_lambda',
                                   'raw', '*.csv')):
    with open(path) as f:
        row = next(csv.DictReader(f))
    tag = row.get('tag', '')
    if not tag.startswith('TRAP'):
        continue
    nlambda = int(row['n_gl'])
    trap_records[nlambda].append((float(row['tildeS']), float(row['error'])))

trap_profiles = defaultdict(lambda: defaultdict(list))
seed_to_nlambda = {}
for path in glob.glob(os.path.join(ROOT, 'data/rk_ising/mc', 'audits/critical_l24/linear_lambda',
                                   'raw', '*.csv')):
    with open(path) as f:
        row = next(csv.DictReader(f))
    seed_to_nlambda[row['seed']] = int(row['n_gl'])
for path in glob.glob(os.path.join(ROOT, 'logs2d', 'l24_linear_lambda_*.out')):
    text = open(path).read()
    header = re.search(r'seed=(\d+)', text)
    if not header or header.group(1) not in seed_to_nlambda:
        continue
    nlambda = seed_to_nlambda[header.group(1)]
    for lam, mean, err in re.findall(
            r'lam=([0-9.]+)\s+<dE/dl>=([+-]?[0-9.]+)\s+\+\-\s+([0-9.]+)', text):
        trap_profiles[nlambda][float(lam)].append((float(mean), float(err)))

summary = []
for mode in ('short', 'long', 'fresh', 'reverse'):
    vals = [(r['value'], r['error']) for r in records if r['mode'] == mode]
    summary.append((mode, *pool(vals)))
summary.append(('long-1e6', *pool(long_records)))
for nlambda in (10, 20, 30):
    vals = trap_records.get(nlambda, [])
    if vals:
        summary.append((f'trap-{nlambda}', *pool(vals)))

with open(os.path.join(OUT, 'combined_summary.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['mode', 'value', 'quoted_error', 'formal_error', 'chain_sem'])
    w.writerows(summary)

colors = {'short': '#777777', 'long': '#1f4e79', 'fresh': '#b55223',
          'reverse': '#4b8b3b', 'long-1e6': '#000000',
          'trap-10': '#c06c84', 'trap-20': '#8f5f9f', 'trap-30': '#5f4b8b'}
display = {'long-1e6': 'long-1e6', 'trap-10': 'trap-10',
           'trap-20': 'trap-20', 'trap-30': 'trap-30'}
category_order = ['short', 'long', 'fresh', 'reverse', 'long-1e6',
                  'trap-10', 'trap-20', 'trap-30']
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.6, 4.5), dpi=180)
summary_by_mode = {row[0]: row for row in summary}
for mode in category_order:
    if mode not in summary_by_mode:
        continue
    _, mean, err, formal, scatter = summary_by_mode[mode]
    ax0.errorbar(mode, mean, yerr=err, fmt='o', ms=6, capsize=3,
                 color=colors[mode], label=display.get(mode, mode))
    if mode == 'long-1e6':
        vals = [v for v,e in long_records]
        ax0.scatter([mode]*len(vals), vals, marker='.', s=10,
                    color=colors[mode], alpha=.45, zorder=1)
    elif mode.startswith('trap-'):
        nlambda = int(mode.split('-')[1])
        vals = [v for v,e in trap_records[nlambda]]
        ax0.scatter([mode]*len(vals), vals, marker='^', s=36,
                    color=colors[mode], alpha=.65, zorder=1)
    else:
        vals = [r['value'] for r in records if r['mode'] == mode]
        ax0.scatter([mode]*len(vals), vals, marker='x', s=48,
                    color=colors[mode], alpha=.65, zorder=1)
ax0.axhline(0, color='0.7', ls=':')
ax0.set_xticks(category_order, category_order, rotation=25, ha='right')
ax0.set_ylabel(r'$\widetilde S$')
ax0.set_title(r'$L=24$, $\beta_c$')
ax0.grid(alpha=.2)
ax0.legend(frameon=False, fontsize=8)
ax0.text(0.02, 0.02,
         'circle = pooled mean +/- error\n'
         'x / dot = one independent chain',
         transform=ax0.transAxes, fontsize=8, color='0.25',
         bbox=dict(facecolor='white', edgecolor='0.8', alpha=0.85,
                   boxstyle='round,pad=0.3'))

for mode in ('short', 'long', 'fresh', 'reverse'):
    pts = []
    for (m, lam), vals in profiles.items():
        if m == mode:
            pts.append((lam, *pool(vals)[:2]))
    pts.sort()
    ax1.errorbar([p[0] for p in pts], [p[1] for p in pts],
                 yerr=[p[2] for p in pts], marker='o', ms=3, lw=1,
                 color=colors[mode], label=mode)
pts = [(lam, *pool(vals)[:2]) for lam, vals in long_profiles.items()]
pts.sort()
ax1.errorbar([p[0] for p in pts], [p[1] for p in pts],
             yerr=[p[2] for p in pts], marker='.', ms=4, lw=1.6,
             color=colors['long-1e6'], label='long-1e6 (160 chains)')
trap_profile_colors = {10: '#c06c84', 20: '#8f5f9f', 30: '#5f4b8b'}
for nlambda in (10, 20, 30):
    pts = [(lam, *pool(vals)[:2])
           for lam, vals in trap_profiles[nlambda].items()]
    pts.sort()
    if not pts:
        continue
    ax1.errorbar([p[0] for p in pts], [p[1] for p in pts],
                 yerr=[p[2] for p in pts], marker='o', ms=3.5,
                 lw=1.0, ls='--', capsize=1.5,
                 color=trap_profile_colors[nlambda],
                 label=f'trap-{nlambda}')
ax1.axhline(0, color='0.7', ls=':')
ax1.set_xlabel(r'$\lambda$')
ax1.set_ylabel(r'$\langle\Delta K\rangle_\lambda$')
ax1.set_title('GL12 node profiles')
ax1.grid(alpha=.2)
ax1.legend(frameon=False, fontsize=8)
fig.subplots_adjust(bottom=0.28, wspace=0.22)
fig.text(0.5, 0.015,
         'Audit parameters (all: L=24, beta=beta_c, SW, MCQ=tS, GL12, forward unless noted):\n'
         'short: eq=2,000, meas=16,000, n=4, warm start   |   '
         'long: eq=16,000, meas=64,000, n=4, warm start\n'
         'fresh: eq=16,000, meas=64,000, n=4, random restart at every lambda   |   '
         'reverse: eq=16,000, meas=64,000, n=4, lambda 1 -> 0\n'
         'long-1e6: eq=1,000,000, meas=640,000, n=160, warm start   |   '
         'trap-N: eq=16,000, meas=64,000, n=4, N equally spaced lambda points',
         ha='center', va='bottom', fontsize=8, color='0.2')
output = os.path.join(OUT, 'audit_combined_trapezoid.png')
fig.savefig(output, bbox_inches='tight')
print('wrote', output)
for row in summary:
    print('%-10s %.8f %.8f formal %.8f chain %.8f' % row)
