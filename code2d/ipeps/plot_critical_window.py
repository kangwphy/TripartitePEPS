#!/usr/bin/env python3
"""The critical window 0.42 <= beta <= 0.46, with beta_c itself on the same axes.

Two things this figure has to make visible:
  (a) chi convergence: the edges settle, the near-critical betas drift, and
      beta_c does not converge at all -- it is still falling at chi1=32.
  (b) the beta dependence at FIXED chi1: every curve dips to a minimum exactly
      at beta_c and the minimum deepens with chi1.  That dip is the finite-chi
      signature, since xi is largest there.

The sign change of Stilde_J between the phases is visible in (b) but is NOT
analysed: the location of the zero is not a meaningful quantity, and an earlier
version of this figure that extracted and extrapolated it has been removed.
"""
import csv, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/ix/zdai/kangw/PEPS3EE/PEPS/clean"
SCAN = f"{R}/data/rk_ising/ctmrg/tables/beta_chi_scan_split/summary.csv"
CRIT = f"{R}/data/rk_ising/ctmrg/tables/split_replica/summary.csv"
OUT = f"{R}/data/rk_ising/ctmrg/figures/beta_chi_scan_split/critical_window.png"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
BETA_C = 0.5*math.log(1+math.sqrt(2))
LO, HI = 0.4195, 0.4605   # crossover betas 0.4380-0.4404 now fall inside

# The pstar grid (16 betas at 1e-4 spacing in [0.4390,0.4405]) and the logbc
# ladder (16 betas hugging beta_c) both live inside this window.  They exist for
# the beta* location and the tensor-bound test, and labelling all of them stacks
# forty annotations outside the axes, which is what squashed this figure.  They
# are still DRAWN -- they carry the shape of the dip -- but only the coarse scan
# betas get a label.
def labelled(b):
    if abs(b - BETA_C) < 1e-9:
        return True
    if 0.4389 < b < 0.4406:            # pstar fine grid
        return False
    if abs(b - BETA_C) < 1.5e-3:       # logbc ladder
        return False
    return True

scan = {}                                   # beta -> chi1 -> tSJ
for r in csv.DictReader(open(SCAN)):
    b, c = float(r["beta"]), int(r["chi1"])
    # beta_c is collected from the split_replica table and drawn separately;
    # since the dense recomputation it ALSO appears in the scan table, so it
    # must be excluded here or it is plotted twice.
    if LO <= b <= HI and abs(b - BETA_C) > 1e-6:
        scan.setdefault(b, {})[c] = float(r["tSJ"])

crit = {}                                   # chi1 -> tSJ at beta_c
for r in csv.DictReader(open(CRIT)):
    if abs(float(r["beta"]) - BETA_C) < 1e-6:
        crit[int(r["chi1"])] = float(r["tSJ"])

def cls(series):
    cs = sorted(series)
    if len(cs) < 2:
        return "UNKNOWN"
    d = abs(series[cs[-1]] - series[cs[-2]])
    rel = d/abs(series[cs[-1]]) if series[cs[-1]] else float("inf")
    return "SAFE" if rel <= 1e-3 else ("MARGINAL" if rel <= 1e-2 else "EXCLUDE")


def label_ends(axis, entries, xpad=0.045, gap=0.028):
    """Labels beside their own curve, pushed apart in axes-fraction space.

    Sixteen betas in this window put seven ordered-side curves inside a band of
    0.03 in Stilde; plain annotate stacks their labels on top of each other."""
    if not entries:
        return
    tr, inv = axis.transData, axis.transAxes.inverted()
    x0 = axis.get_xlim()[0]
    to_f = lambda y: inv.transform(tr.transform((x0, y)))[1]
    from_f = lambda f: tr.inverted().transform(axis.transAxes.transform((0.0, f)))[1]
    ent = sorted(entries, key=lambda e: to_f(e[1]))
    placed = []
    for x, y, txt, col in ent:
        f = to_f(y)
        fl = f if not placed else max(f, placed[-1][1] + gap)
        placed.append((x, fl, y, txt, col))
    over = placed[-1][1] - (1.0 - 0.5*gap)
    if over > 0:
        sh = min(over, placed[0][1] - 0.5*gap)
        if sh > 0:
            placed = [(x, fl-sh, y, t, c) for x, fl, y, t, c in placed]
    lo, hi = axis.get_xlim()
    xt = math.exp(math.log(hi) + xpad*(math.log(hi)-math.log(lo))) \
         if axis.get_xscale() == "log" else hi + xpad*(hi-lo)
    for x, fl, y, txt, col in placed:
        yl = from_f(fl)
        moved = abs(fl - to_f(y)) > 0.4*gap
        axis.annotate(txt, xy=(x, y), xytext=(xt, yl), textcoords="data",
                      color=col, fontsize=7, va="center", ha="left",
                      annotation_clip=False, clip_on=False,
                      arrowprops=dict(arrowstyle="-", color=col, lw=0.45,
                                      alpha=0.4) if moved else None)

fig, ax = plt.subplots(1, 2, figsize=(15.0, 7.2))

# ---- (a) chi convergence, beta_c included ----------------------------------
a = ax[0]
labels_a = []
betas = sorted(scan)
cmap = plt.cm.plasma(np.linspace(0.05, 0.85, len(betas)))
for b, col in zip(betas, cmap):
    cs = sorted(scan[b]); ys = [scan[b][c] for c in cs]
    k = cls(scan[b])
    a.plot(cs, ys, "-", color=col, lw=1.3)
    a.scatter(cs, ys, marker="s" if k == "EXCLUDE" else "o",
              s=44, edgecolor=col, facecolor=col if k == "SAFE" else "white", zorder=3)
    if labelled(b):
        labels_a.append((cs[-1], ys[-1], fr"${b:.4g}$", col))
cc = sorted(crit)
a.plot(cc, [crit[c] for c in cc], "-", color="black", lw=2.2, zorder=4)
a.scatter(cc, [crit[c] for c in cc], marker="D", s=40, color="black", zorder=5)
labels_a.append((cc[-1], crit[cc[-1]], r"$\beta_c$", "black"))
a.axhline(0, color="0.5", lw=0.8, ls=":")
a.set_xlabel(r"one-replica $\chi_1$"); a.set_ylabel(r"$\widetilde S_J$")
a.set_title(r"(a) $\beta_c$ never converges; the window drifts")
# Stilde spans 1e-3 to 1 in this window with both signs; on a linear axis the
# seven ordered-side curves collapse into one band.
a.set_yscale("symlog", linthresh=1e-2, linscale=0.4)
a.set_xscale("log")
a.set_xlim(3.4, 46); a.grid(alpha=0.25, which="both")
a.set_xticks([4, 8, 16, 24, 32]); a.set_xticklabels(["4", "8", "16", "24", "32"])
a.minorticks_off()
label_ends(a, labels_a)

# ---- (b) beta dependence at fixed chi1 -------------------------------------
b_ = ax[1]
for chi1, col in zip((4, 8, 16, 24), plt.cm.viridis([0.05, 0.35, 0.62, 0.88])):
    pts = [(b, scan[b][chi1]) for b in betas if chi1 in scan[b]]
    if chi1 in crit:
        pts.append((BETA_C, crit[chi1]))
    pts.sort()
    b_.plot([p[0] for p in pts], [p[1] for p in pts], "o-", ms=5, color=col,
            lw=1.5, label=fr"$\chi_1={chi1}$")
    if chi1 in crit:
        b_.plot([BETA_C], [crit[chi1]], "D", ms=8, mfc="none", mec=col, mew=1.8)
    # The minimum of each fixed-chi1 curve is the PSEUDO-CRITICAL point
    # beta*(chi1) and it sits BELOW beta_c.  Earlier versions of this figure had
    # no beta between 0.438 and beta_c, so the dip looked pinned at beta_c.
    mn = min(pts, key=lambda q: q[1])
    b_.plot([mn[0]], [mn[1]], "v", ms=12, mfc="none", mec=col, mew=2.2, zorder=6)
b_.axhline(0, color="0.5", lw=0.8, ls=":")
b_.axvline(BETA_C, color="crimson", lw=1.1, ls="--")
b_.annotate(r"$\beta_c$", (BETA_C, b_.get_ylim()[0]), xytext=(4, 12),
            textcoords="offset points", color="crimson", fontsize=10)
b_.set_xlabel(r"$\beta$"); b_.set_ylabel(r"$\widetilde S_J$")
b_.set_title(r"(b) the dip sits BELOW $\beta_c$ at $\beta^*(\chi_1)$ "
             r"($\bigtriangledown$), and deepens with $\chi_1$")
b_.legend(fontsize=9, loc="lower left"); b_.grid(alpha=0.25)

fig.suptitle(r"critical window $0.42\leq\beta\leq0.46$ of the split-replica scan, "
             r"with $\beta_c$ on the same axes", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT, dpi=170)
print("wrote", OUT)
print(f"\nbeta_c series (chi1: tSJ):")
for c in sorted(crit):
    print(f"  {c:3d}  {crit[c]:+.6f}")
