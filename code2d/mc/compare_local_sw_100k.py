#!/usr/bin/env python3
"""Compare completed 100k local-update and SW tS campaigns."""
import csv, glob, math, os
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
LOCAL = os.path.join(ROOT, 'data/rk_ising/mc', 'campaigns/local_tS_100k_16chains', 'raw')
SW = os.path.join(ROOT, 'data/rk_ising/mc', 'campaigns/sw_components_100k_16chains', 'tS', 'raw')
OUT = os.path.join(ROOT, 'data/rk_ising/mc', 'campaigns/local_tS_100k_16chains')
BETAS = ('0.30', '0.60', 'crit')
SIZES = (4, 8, 12, 16, 20, 24, 32, 48, 64, 80, 96)

def canonical_beta(value):
    if value == 'crit':
        return value
    return f'{float(value):.2f}'

def read(pattern):
    data = defaultdict(list)
    for path in glob.glob(pattern):
        with open(path, newline='') as stream:
            row = next(csv.DictReader(stream))
        data[(canonical_beta(row['beta']), int(row['L']))].append(
            (float(row['tildeS']), float(row['error']), int(row['seed'])))
    return data

def pool(values):
    if not values:
        return None
    weights = [1.0 / max(err, 1e-15)**2 for _, err, _ in values]
    mean = sum(value * weight for (value, _, _), weight in zip(values, weights)) / sum(weights)
    formal = 1.0 / math.sqrt(sum(weights))
    if len(values) > 1:
        chain_sem = math.sqrt(sum((value - mean)**2 for value, _, _ in values)
                              / (len(values) * (len(values) - 1)))
    else:
        chain_sem = formal
    return mean, max(formal, chain_sem), formal, chain_sem, len(values)

local = read(os.path.join(LOCAL, '*.csv'))
sw = read(os.path.join(SW, '*.csv'))
rows = []
for beta in BETAS:
    for L in SIZES:
        lp, sp = pool(local[(beta, L)]), pool(sw[(beta, L)])
        if lp is None or sp is None:
            continue
        diff = lp[0] - sp[0]
        diff_err = math.hypot(lp[1], sp[1])
        rows.append((beta, L, lp, sp, diff, diff_err, diff / diff_err))

summary = os.path.join(OUT, 'local_vs_sw_100k_summary.csv')
with open(summary, 'w', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(['beta', 'L', 'local_value', 'local_error', 'local_n',
                     'sw_value', 'sw_error', 'sw_n', 'difference',
                     'difference_error', 'z_score'])
    for beta, L, lp, sp, diff, diff_err, z in rows:
        writer.writerow([beta, L, lp[0], lp[1], lp[4], sp[0], sp[1], sp[4],
                         diff, diff_err, z])

fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.9), dpi=180, sharey=True)
titles = {'0.30': r'$\beta=0.30$', '0.60': r'$\beta=0.60$',
          'crit': r'$\beta=\beta_c$'}
for ax, beta in zip(axes, BETAS):
    points = [row for row in rows if row[0] == beta]
    if points:
        x = [row[1] for row in points]
        ax.errorbar([v * 0.975 for v in x], [row[2][0] for row in points],
                    yerr=[row[2][1] for row in points], fmt='s-', ms=4,
                    capsize=2, lw=1.1, color='#b55223', label='local update')
        ax.errorbar([v * 1.025 for v in x], [row[3][0] for row in points],
                    yerr=[row[3][1] for row in points], fmt='o-', ms=4,
                    capsize=2, lw=1.1, color='#1f5a85', label='SW')
    ax.set_xscale('log', base=2)
    ax.set_xticks(SIZES)
    ax.set_xticklabels([str(v) for v in SIZES], rotation=45)
    ax.set_title(titles[beta])
    ax.set_xlabel(r'$L$')
    ax.grid(alpha=.22, which='both')
axes[0].set_ylabel(r'$\widetilde S$')
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, loc='upper center', ncol=2,
           bbox_to_anchor=(0.5, 1.04))
fig.tight_layout()
figure = os.path.join(OUT, 'local_vs_sw_100k.png')
fig.savefig(figure, bbox_inches='tight')

fig, ax = plt.subplots(figsize=(7.0, 3.8), dpi=180)
for beta, color in zip(BETAS, ('#287271', '#d07c32', '#222222')):
    points = [row for row in rows if row[0] == beta]
    if points:
        ax.errorbar([row[1] for row in points], [row[4] for row in points],
                    yerr=[row[5] for row in points], fmt='o-', ms=4,
                    capsize=2, lw=1.1, color=color, label=titles[beta])
ax.axhline(0, color='0.55', ls=':', lw=1)
ax.set_xscale('log', base=2); ax.set_xticks(SIZES)
ax.set_xticklabels([str(v) for v in SIZES], rotation=45)
ax.set_xlabel(r'$L$'); ax.set_ylabel(r'$\widetilde S_{\rm local}-\widetilde S_{\rm SW}$')
ax.grid(alpha=.22, which='both'); ax.legend(frameon=False)
fig.tight_layout()
diff_figure = os.path.join(OUT, 'local_vs_sw_100k_difference.png')
fig.savefig(diff_figure, bbox_inches='tight')
print(f'wrote {summary}')
print(f'wrote {figure}')
print(f'wrote {diff_figure}')
for beta in BETAS:
    selected = [r for r in rows if r[0] == beta]
    print(beta, 'points=', len(selected),
          'max_abs_z=', max((abs(r[6]) for r in selected), default=float('nan')))
