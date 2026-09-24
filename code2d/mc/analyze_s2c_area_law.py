#!/usr/bin/env python3
"""Pool existing S2C runs and test the leading area-law dependence."""

import csv
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SOURCE = os.path.join(ROOT, "data/rk_ising/mc", "tables", "mc_results_controls.csv")
OUT = os.path.join(ROOT, "data/rk_ising/mc", "derived", "s2c_area_law")
os.makedirs(OUT, exist_ok=True)


def pool(rows):
    weights = np.array([1.0 / error**2 for _, error in rows])
    values = np.array([value for value, _ in rows])
    mean = float(np.dot(weights, values) / weights.sum())
    formal = float(1.0 / math.sqrt(weights.sum()))
    if len(rows) > 1:
        chain_sem = math.sqrt(float(np.sum((values - mean)**2)) /
                              (len(rows) * (len(rows) - 1)))
    else:
        chain_sem = formal
    return mean, max(formal, chain_sem), formal, chain_sem


def weighted_fit(L, y, error, include_log=False):
    columns = [L]
    if include_log:
        columns.append(np.log(L))
    columns.append(np.ones_like(L))
    design = np.column_stack(columns)
    inv_var = np.diag(1.0 / error**2)
    covariance = np.linalg.inv(design.T @ inv_var @ design)
    coeff = covariance @ design.T @ inv_var @ y
    residual = y - design @ coeff
    chi2 = float(residual.T @ inv_var @ residual)
    dof = len(y) - len(coeff)
    return coeff, np.sqrt(np.diag(covariance)), chi2, dof


raw = defaultdict(list)
with open(SOURCE) as stream:
    for row in csv.reader(stream):
        if len(row) < 8 or "S2C" not in row[7]:
            continue
        raw[(row[1], int(row[0]))].append((float(row[4]), float(row[5])))

pooled = []
for (beta, L), rows in sorted(raw.items(), key=lambda item: (item[0][0], item[0][1])):
    mean, error, formal, chain_sem = pool(rows)
    pooled.append((beta, L, mean, error, formal, chain_sem, len(rows)))

with open(os.path.join(OUT, "pooled.csv"), "w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["beta", "L", "S2C", "error", "formal_error",
                     "chain_sem", "n_chains"])
    writer.writerows(pooled)

fit_rows = []
for beta in ("0.30", "crit", "0.47"):
    rows = [row for row in pooled if row[0] == beta]
    L = np.array([row[1] for row in rows], dtype=float)
    y = np.array([row[2] for row in rows])
    error = np.array([row[3] for row in rows])
    coeff, coeff_error, chi2, dof = weighted_fit(L, y, error)
    fit_rows.append((beta, "aL+b", *coeff, *coeff_error, chi2, dof))
    print(f"beta={beta:>4s}: S2C = ({coeff[0]:.8f} +- {coeff_error[0]:.8f}) L "
          f"+ ({coeff[1]:.6f} +- {coeff_error[1]:.6f}); "
          f"chi2/dof={chi2:.2f}/{dof}")
    if beta == "crit":
        c, ce, chi2_log, dof_log = weighted_fit(L, y, error, include_log=True)
        fit_rows.append((beta, "aL+c_logL+b", *c, *ce, chi2_log, dof_log))
        print(f"beta=crit log fit: a={c[0]:.8f} +- {ce[0]:.8f}, "
              f"c={c[1]:.6f} +- {ce[1]:.6f}, b={c[2]:.6f} +- {ce[2]:.6f}; "
              f"chi2/dof={chi2_log:.2f}/{dof_log}")

with open(os.path.join(OUT, "fits.csv"), "w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["beta", "model", "parameters_then_errors", "chi2", "dof"])
    for row in fit_rows:
        writer.writerow(row)

fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.7), dpi=170)
colors = {"0.30": "#287271", "crit": "#222222", "0.47": "#b55223"}
for ax, beta in zip(axes, ("0.30", "crit", "0.47")):
    rows = [row for row in pooled if row[0] == beta]
    L = np.array([row[1] for row in rows], dtype=float)
    y = np.array([row[2] for row in rows])
    error = np.array([row[3] for row in rows])
    coeff, _, chi2, dof = weighted_fit(L, y, error)
    grid = np.linspace(L.min(), L.max(), 300)
    ax.errorbar(L, y, yerr=error, fmt="o", capsize=3, color=colors[beta],
                label="pooled MC")
    ax.plot(grid, coeff[0] * grid + coeff[1], color=colors[beta], lw=1.5,
            label=rf"$aL+b$, $\chi^2/\nu={chi2:.1f}/{dof}$")
    if beta == "crit":
        c, _, chi2_log, dof_log = weighted_fit(L, y, error, include_log=True)
        ax.plot(grid, c[0]*grid + c[1]*np.log(grid) + c[2], "--",
                color="#4c78a8", lw=1.4,
                label=rf"$aL+c\ln L+b$, $\chi^2/\nu={chi2_log:.1f}/{dof_log}$")
    ax.set_xlabel(r"$L$ ")
    beta_label = r"\beta_c" if beta == "crit" else beta
    ax.set_title(rf"$\beta={beta_label}$")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=7)
axes[0].set_ylabel(r"$S_2(C)$")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "s2c_area_law.png"), bbox_inches="tight")
print("wrote", OUT)
