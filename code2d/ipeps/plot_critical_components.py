#!/usr/bin/env python3
"""Components of Stilde_J across the critical window, in the convention of
figures/split_replica/critical_components.png:

    Stilde_J = 2 kappa_A + (-kappa_Y),

so the two plotted curves ADD to the black one.  The point of separating them
here is that the combination is a near-cancellation over most of the window --
where the two components are individually large and nearly equal, the drift of
their difference understates the error, which is precisely the trap that made
beta=0.43 at chi1=8 read as a quiet +0.0007 while both components were wrong by
about 0.05.
"""
import csv, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/ix/zdai/kangw/PEPS3EE/PEPS/clean"
SCAN = f"{R}/data/rk_ising/ctmrg/tables/beta_chi_scan_split/summary.csv"
CRIT = f"{R}/data/rk_ising/ctmrg/tables/split_replica/summary.csv"
OUT = f"{R}/data/rk_ising/ctmrg/figures/beta_chi_scan_split/critical_window_components.png"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
BETA_C = 0.5*math.log(1+math.sqrt(2))
LO, HI = 0.4195, 0.4605

scan = {}                       # beta -> chi1 -> (2kA, -kY, tSJ)
for r in csv.DictReader(open(SCAN)):
    b, c = float(r["beta"]), int(r["chi1"])
    # beta_c is collected from the split_replica table and drawn separately;
    # since the dense recomputation it ALSO appears in the scan table, so it
    # must be excluded here or it is plotted twice.
    if LO <= b <= HI and abs(b - BETA_C) > 1e-6:
        scan.setdefault(b, {})[c] = (2*float(r["kA"]), -float(r["kY"]), float(r["tSJ"]))
crit = {}
for r in csv.DictReader(open(CRIT)):
    if abs(float(r["beta"]) - BETA_C) < 1e-6:
        crit[int(r["chi1"])] = (float(r["two_kA"]), float(r["minus_kY"]), float(r["tSJ"]))

betas = sorted(scan)
cmap = plt.cm.plasma(np.linspace(0.05, 0.85, len(betas)))
fig, ax = plt.subplots(1, 3, figsize=(17.0, 5.4))

for k, (idx, name, sym) in enumerate(((0, r"$2\kappa_A$", "o"), (1, r"$-\kappa_Y$", "s"))):
    a = ax[k]
    for b, col in zip(betas, cmap):
        cs = sorted(scan[b]); ys = [scan[b][c][idx] for c in cs]
        a.plot(cs, ys, "-", marker=sym, ms=5, color=col, lw=1.3)
        a.annotate(fr"${b:.4g}$", (cs[-1], ys[-1]), xytext=(5, 0),
                   textcoords="offset points", fontsize=7.5, color=col, va="center")
    cc = sorted(crit)
    a.plot(cc, [crit[c][idx] for c in cc], "-", marker="D", ms=5, color="black", lw=2.2, zorder=5)
    a.annotate(r"$\beta_c$", (cc[-1], crit[cc[-1]][idx]), xytext=(6, -2),
               textcoords="offset points", fontsize=10, fontweight="bold")
    a.set_xscale("log"); a.set_xticks([4, 8, 16, 24, 32])
    a.set_xticklabels(["4", "8", "16", "24", "32"]); a.minorticks_off()
    a.set_xlim(3.4, 46); a.grid(alpha=0.25, which="both")
    a.axhline(0, color="0.5", lw=0.8, ls=":")
    a.set_xlabel(r"one-replica $\chi_1$"); a.set_ylabel(name)
    a.set_title(f"({'ab'[k]}) {name}" + (r" grows slowly" if k == 0 else
                                          r" carries the divergence"))

# (c) both components and their sum against beta, at the deepest common chi1
c_ = ax[2]
CHI = 24
pts = [(b, scan[b][CHI]) for b in betas if CHI in scan[b]]
if CHI in crit:
    pts.append((BETA_C, crit[CHI]))
pts.sort()
bs = [p[0] for p in pts]
for idx, lab, col, sym in ((0, r"$2\kappa_A$", "tab:blue", "o"),
                           (1, r"$-\kappa_Y$", "tab:orange", "s"),
                           (2, r"$\widetilde S_J=2\kappa_A-\kappa_Y$", "black", "^")):
    c_.plot(bs, [p[1][idx] for p in pts], "-", marker=sym, ms=6, color=col,
            lw=2.0 if idx == 2 else 1.5, label=lab)
c_.axvline(BETA_C, color="crimson", lw=1.1, ls="--")
c_.annotate(r"$\beta_c$", (BETA_C, c_.get_ylim()[1]), xytext=(4, -16),
            textcoords="offset points", color="crimson", fontsize=10)
c_.axhline(0, color="0.5", lw=0.8, ls=":")
c_.set_xlabel(r"$\beta$"); c_.set_ylabel("contribution")
c_.set_title(rf"(c) at $\chi_1={CHI}$: both components dip at $\beta_c$")
c_.legend(fontsize=9, loc="lower left"); c_.grid(alpha=0.25)

fig.suptitle(r"components of $\widetilde S_J$ in the critical window "
             r"$0.42\leq\beta\leq0.46$, with $\beta_c$", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT, dpi=170)
print("wrote", OUT)

print(f"\ncancellation at chi1={CHI}: how much larger the components are than their sum")
print(f"{'beta':>8} {'2kA':>11} {'-kY':>11} {'tSJ':>11} {'amplif':>8}")
for b, (A, Y, T) in pts:
    tag = "  <- beta_c" if abs(b-BETA_C) < 1e-6 else ""
    print(f"{b:8.4f} {A:+11.6f} {Y:+11.6f} {T:+11.6f} "
          f"{max(abs(A),abs(Y))/abs(T) if T else float('inf'):8.2f}{tag}")
