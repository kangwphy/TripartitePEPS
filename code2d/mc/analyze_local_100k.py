#!/usr/bin/env python3
"""Pool the independent 100k local-update chains and plot S-tilde versus L."""
import csv, glob, math, os
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
out = os.path.join(root, 'data/rk_ising/mc', 'campaigns/local_tS_100k_16chains')
raw = defaultdict(list)
for path in glob.glob(os.path.join(out, 'raw', '*.csv')):
    with open(path) as stream:
        row = next(csv.DictReader(stream))
    raw[(row['beta'], int(row['L']))].append((float(row['tildeS']), float(row['error'])))

def pool(values):
    weights = [1.0/e**2 for _, e in values]
    mean = sum(v*w for (v, _), w in zip(values, weights)) / sum(weights)
    formal = 1.0 / math.sqrt(sum(weights))
    scatter = (math.sqrt(sum((v-mean)**2 for v, _ in values) /
                          (len(values)*(len(values)-1)))
               if len(values) > 1 else formal)
    return mean, max(formal, scatter), formal, scatter

betas = ('0.30', '0.60', 'crit')
sizes = (4, 8, 12, 16, 20, 24, 32, 48, 64, 80, 96)
with open(os.path.join(out, 'pooled.csv'), 'w', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(['beta', 'L', 'value', 'error', 'formal_error', 'chain_sem', 'n_chains', 'complete_16'])
    for beta in betas:
        for L in sizes:
            vals = raw[(beta, L)]
            if vals:
                writer.writerow([beta, L, *pool(vals), len(vals), len(vals) >= 16])

with open(os.path.join(out, 'coverage.csv'), 'w', newline='') as stream:
    writer = csv.writer(stream); writer.writerow(['beta', 'L', 'n_chains', 'complete_16'])
    for beta in betas:
        for L in sizes:
            n = len(raw[(beta, L)]); writer.writerow([beta, L, n, n >= 16])

colors = {'0.30': '#287271', '0.60': '#d07c32', 'crit': '#222222'}
fig, ax = plt.subplots(figsize=(6.6, 4.3), dpi=180)
for beta in betas:
    pts = [(L, *pool(raw[(beta, L)][:])) for L in sizes if raw[(beta, L)]]
    ax.errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[p[2] for p in pts],
                marker='o', ms=4, capsize=2, lw=1.2,
                color=colors[beta], label='beta = beta_c' if beta == 'crit' else f'beta = {beta}')
ax.axhline(0, color='0.7', ls=':'); ax.grid(alpha=.22)
ax.set_xlabel(r'$L$'); ax.set_ylabel(r'local-update $\widetilde S$')
ax.legend(frameon=False); fig.tight_layout()
fig.savefig(os.path.join(out, 'local_update_100k.png'), bbox_inches='tight')
print('wrote local-update analysis to', out)
