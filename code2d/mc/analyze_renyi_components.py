#!/usr/bin/env python3
"""Pool component chains and compare reconstructed and direct tilde S."""
import csv, math, os
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT = os.path.join(ROOT, 'data/rk_ising/mc', 'campaigns/sw_components_100k_16chains')
OBS = ('S2A', 'S2B', 'S2C', 'S3', 'tS')
BETAS = ('0.30', '0.60', 'crit')
SIZES = (4, 8, 12, 16, 20, 24, 32, 48, 64, 80, 96)

raw = defaultdict(list)
with open(os.path.join(OUT, 'all_chains.csv')) as stream:
    for row in csv.DictReader(stream):
        raw[(row['beta'], int(row['L']), row['observable'])].append(
            (float(row['value']), float(row['error'])))

def pool(values):
    weights = [1.0/e**2 for _, e in values]
    mean = sum(v*w for (v, _), w in zip(values, weights))/sum(weights)
    formal = 1.0/math.sqrt(sum(weights))
    chain_sem = (math.sqrt(sum((v-mean)**2 for v, _ in values)/
                           (len(values)*(len(values)-1)))
                 if len(values) > 1 else formal)
    return mean, max(formal, chain_sem), formal, chain_sem

with open(os.path.join(OUT, 'coverage.csv'), 'w', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(['beta', 'L', *OBS, 'complete_16_each'])
    for beta in BETAS:
        for L in SIZES:
            counts = [len(raw[(beta, L, obs)]) for obs in OBS]
            writer.writerow([beta, L, *counts, all(n >= 16 for n in counts)])

pooled = {}
with open(os.path.join(OUT, 'pooled_components.csv'), 'w', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(['beta', 'L', 'observable', 'value', 'error',
                     'formal_error', 'chain_sem', 'n_chains', 'complete_16'])
    for beta in BETAS:
        for L in SIZES:
            for obs in OBS:
                values = raw[(beta, L, obs)]
                if not values:
                    continue
                result = pool(values); pooled[(beta, L, obs)] = result
                writer.writerow([beta, L, obs, *result, len(values), len(values) >= 16])

comparisons = []
with open(os.path.join(OUT, 'reconstruction_vs_direct.csv'), 'w', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(['beta', 'L', 'reconstructed_tildeS', 'reconstructed_error',
                     'direct_tildeS', 'direct_error', 'difference',
                     'difference_error', 'z_score', 'min_chain_count', 'complete_16_each'])
    for beta in BETAS:
        for L in SIZES:
            if not all((beta, L, obs) in pooled for obs in OBS):
                continue
            p = {obs: pooled[(beta, L, obs)] for obs in OBS}
            reconstructed = 2*p['S3'][0] - p['S2A'][0] - p['S2B'][0] - p['S2C'][0]
            reconstructed_error = math.sqrt(
                (2*p['S3'][1])**2 + p['S2A'][1]**2 +
                p['S2B'][1]**2 + p['S2C'][1]**2)
            difference = reconstructed - p['tS'][0]
            difference_error = math.hypot(reconstructed_error, p['tS'][1])
            counts = [len(raw[(beta, L, obs)]) for obs in OBS]
            row = (beta, L, reconstructed, reconstructed_error, p['tS'][0],
                   p['tS'][1], difference, difference_error,
                   difference/difference_error, min(counts), all(n >= 16 for n in counts))
            comparisons.append(row); writer.writerow(row)

colors = {'S2A':'#287271', 'S2B':'#d07c32', 'S2C':'#4c78a8', 'S3':'#222222'}
labels = {'S2A':r'$S_2(A)$', 'S2B':r'$S_2(B)$',
          'S2C':r'$S_2(C)$', 'S3':r'$S_2^{(3)}$'}
fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.8), dpi=180)
for ax, beta in zip(axes, BETAS):
    for obs in ('S2A','S2B','S2C','S3'):
        points = [(L, *pooled[(beta,L,obs)][:2]) for L in SIZES
                  if (beta,L,obs) in pooled]
        if points:
            ax.errorbar([x[0] for x in points], [x[1] for x in points],
                        yerr=[x[2] for x in points], marker='o', ms=3,
                        capsize=2, lw=1.1, color=colors[obs], label=labels[obs])
    title = 'beta = beta_c' if beta == 'crit' else f'beta = {beta}'
    ax.set_title(title)
    ax.set_xlabel(r'$L$'); ax.grid(alpha=.22)
axes[0].set_ylabel('component entropy')
axes[0].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'components_vs_L.png'), bbox_inches='tight')

fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.8), dpi=180)
for ax, beta in zip(axes, BETAS):
    rows = [r for r in comparisons if r[0] == beta]
    if rows:
        L = [r[1] for r in rows]
        ax.errorbar(L, [r[2] for r in rows], yerr=[r[3] for r in rows],
                    fmt='o-', ms=4, capsize=2, color='#b55223', label='reconstructed')
        ax.errorbar(L, [r[4] for r in rows], yerr=[r[5] for r in rows],
                    fmt='s-', ms=4, capsize=2, color='#2e5fa3', label='direct')
        for r in rows:
            if not r[-1]: ax.plot(r[1], r[2], marker='x', color='black', ms=7)
    ax.axhline(0, color='0.7', ls=':'); ax.grid(alpha=.22)
    title = 'beta = beta_c' if beta == 'crit' else f'beta = {beta}'
    ax.set_title(title)
    ax.set_xlabel(r'$L$')
axes[0].set_ylabel(r'$\widetilde S$')
axes[0].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'reconstruction_vs_direct.png'), bbox_inches='tight')

# Combined vertical summary for the note: components on top, reconstruction below.
fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.4), dpi=180,
                         sharex='col', constrained_layout=True)
for ax, beta in zip(axes[0], BETAS):
    for obs in ('S2A', 'S2B', 'S2C', 'S3'):
        points = [(L, *pooled[(beta, L, obs)][:2]) for L in SIZES
                  if (beta, L, obs) in pooled]
        if points:
            ax.errorbar([x[0] for x in points], [x[1] for x in points],
                        yerr=[x[2] for x in points], marker='o', ms=3,
                        capsize=2, lw=1.1, color=colors[obs], label=labels[obs])
    ax.set_title('beta = beta_c' if beta == 'crit' else f'beta = {beta}')
    ax.grid(alpha=.22)
for ax, beta in zip(axes[1], BETAS):
    rows = [r for r in comparisons if r[0] == beta]
    if rows:
        x = [r[1] for r in rows]
        ax.errorbar(x, [r[2] for r in rows], yerr=[r[3] for r in rows],
                    fmt='o-', ms=4, capsize=2, color='#b55223', label='reconstructed')
        ax.errorbar(x, [r[4] for r in rows], yerr=[r[5] for r in rows],
                    fmt='s-', ms=4, capsize=2, color='#2e5fa3', label='direct')
        for r in rows:
            if not r[-1]:
                ax.plot(r[1], r[2], marker='x', color='black', ms=7)
    ax.axhline(0, color='0.7', ls=':')
    ax.grid(alpha=.22)
    ax.set_xlabel(r'$L$')
    ax.set_title('beta = beta_c' if beta == 'crit' else f'beta = {beta}')
for ax in axes[0]:
    ax.set_xlabel(r'$L$')
axes[0, 0].set_ylabel('component entropy')
axes[1, 0].set_ylabel(r'$\widetilde S$')
axes[0, 0].legend(frameon=False, fontsize=8)
axes[1, 0].legend(frameon=False, fontsize=8)
fig.savefig(os.path.join(OUT, 'renyi_components_summary.png'), bbox_inches='tight')

# Same combined view with a logarithmic size axis (semilog-x).
fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.4), dpi=180,
                         sharex='col', constrained_layout=True)
for ax, beta in zip(axes[0], BETAS):
    for obs in ('S2A', 'S2B', 'S2C', 'S3'):
        points = [(L, *pooled[(beta, L, obs)][:2]) for L in SIZES
                  if (beta, L, obs) in pooled]
        if points:
            ax.errorbar([x[0] for x in points], [x[1] for x in points],
                        yerr=[x[2] for x in points], marker='o', ms=3,
                        capsize=2, lw=1.1, color=colors[obs], label=labels[obs])
    ax.set_xscale('log')
    ax.set_title('beta = beta_c' if beta == 'crit' else f'beta = {beta}')
    ax.grid(alpha=.22, which='both')
for ax, beta in zip(axes[1], BETAS):
    rows = [r for r in comparisons if r[0] == beta]
    if rows:
        x = [r[1] for r in rows]
        ax.errorbar(x, [r[2] for r in rows], yerr=[r[3] for r in rows],
                    fmt='o-', ms=4, capsize=2, color='#b55223', label='reconstructed')
        ax.errorbar(x, [r[4] for r in rows], yerr=[r[5] for r in rows],
                    fmt='s-', ms=4, capsize=2, color='#2e5fa3', label='direct')
        for r in rows:
            if not r[-1]:
                ax.plot(r[1], r[2], marker='x', color='black', ms=7)
    ax.set_xscale('log')
    ax.axhline(0, color='0.7', ls=':')
    ax.grid(alpha=.22, which='both')
    ax.set_xlabel(r'$L$')
    ax.set_title('beta = beta_c' if beta == 'crit' else f'beta = {beta}')
for ax in axes[0]:
    ax.set_xlabel(r'$L$')
axes[0, 0].set_ylabel('component entropy')
axes[1, 0].set_ylabel(r'$\widetilde S$')
axes[0, 0].legend(frameon=False, fontsize=8)
axes[1, 0].legend(frameon=False, fontsize=8)
fig.savefig(os.path.join(OUT, 'renyi_components_summary_semilogx.png'),
            bbox_inches='tight')
print('wrote analysis to', OUT)
