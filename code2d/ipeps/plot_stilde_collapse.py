#!/usr/bin/env python3
"""S-tilde against ln xi: the requested replot, and the collapse.

Two statements are being tested.
  (1) physical:   Stilde = c ln xi_beta   as beta -> beta_c at converged chi
  (2) finite-chi: Stilde = c ln xi_chi = c*kappa ln chi   at beta = beta_c
The collapse variable is the MEASURED xi_1, one value per (beta, chi1).  It is
tempting to write it as min(xi_beta, xi_chi), and that is wrong here: measured
against the beta_c value of xi at the same chi1, the min never binds -- xi_chi is
119 already at chi1=8, larger than every off-critical xi_beta in the scan (the
largest is 92.9 at beta=0.438).  What xi_1 actually is off-criticality is xi_beta
UNDER-RESOLVED, by a factor that degrades from 0.99 deep in either phase to 0.53
at beta=0.438, chi1=8 (137 points, mean 0.843).  And xi_chi is not a property of
chi1 alone: it is the finite-entanglement length of the CRITICAL transfer matrix,
so transplanting it to another beta has no justification.  Hence no formula --
the effective length is measured, not constructed.

Ordered side: the degeneracy-aware estimator returns the single-domain-wall
length, 2*xi_spin, so it is halved before use.  This is a measured factor
(ratios 1.89-1.97 across the ordered betas), not an assumption.
"""
import csv, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/ix/zdai/kangw/PEPS3EE/PEPS/clean"
SCAN = f"{R}/data/rk_ising/ctmrg/tables/beta_chi_scan_split/summary.csv"
OUT = f"{R}/data/rk_ising/ctmrg/figures/beta_chi_scan_split/stilde_collapse.png"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
BC = 0.5*math.log(1+math.sqrt(2))

def xi_beta(b):
    bd = -0.5*math.log(math.tanh(b))
    if abs(b-BC) < 1e-9: return math.inf
    return 1/(2*(bd-b)) if b < BC else 1/(4*(b-bd))

rows = []
for r in csv.DictReader(open(SCAN)):
    b, c, t = float(r["beta"]), int(r["chi1"]), float(r["tSJ"])
    try: xm = float(r["xi1"])
    except ValueError: xm = float("nan")
    if not math.isfinite(xm) or xm > 1e6 or xm <= 0: continue
    # NO transformation.  The measured xi is used exactly as the transfer
    # spectrum returns it.  An earlier version divided the ordered side by two,
    # on the reading that the degeneracy-aware estimator returns the single
    # domain-wall length 2*xi_spin.  That is an interpretation, and building it
    # into the axis hides whether the collapse needs it; both are reported below
    # instead.
    rows.append(dict(beta=b, chi=c, tS=t, xim=xm, xib=xi_beta(b)))

CH = sorted({r["chi"] for r in rows})
cmap = {c: plt.cm.viridis(i/max(len(CH)-1, 1)) for i, c in enumerate(CH)}
fig, ax = plt.subplots(1, 3, figsize=(18.5, 5.8))

# ---- (a) the requested replot: x = ln xi_beta ---------------------------------
A = ax[0]
# The two phases share the same xi_beta values but not the same Stilde, so they
# must be drawn as separate branches; joining them produces a zig-zag that is an
# artefact of sorting, not data.
for c in (4, 8, 16, 24):
    for side, ls, mk in (("dis", "-", "o"), ("ord", "--", "s")):
        sel = [r for r in rows if r["chi"] == c and math.isfinite(r["xib"])
               and ((r["beta"] < BC) if side == "dis" else (r["beta"] > BC))]
        pts = sorted(sel, key=lambda r: r["xib"])
        if not pts: continue
        A.plot([math.log(p["xib"]) for p in pts], [p["tS"] for p in pts],
               ls, marker=mk, ms=4.5, color=cmap[c], lw=1.4,
               label=(fr"$\chi_1={c}$" if side == "dis" else None))
A.plot([], [], "k-o", ms=4.5, label=r"$\beta<\beta_c$")
A.plot([], [], "k--s", ms=4.5, label=r"$\beta>\beta_c$")
A.axhline(0, color="0.5", lw=0.8, ls=":")
A.set_xlabel(r"$\ln\xi_\beta$  (exact Ising correlation length)")
A.set_ylabel(r"$\widetilde S_J$")
A.set_title(r"(a) panel (b) replotted against $\ln\xi_\beta$", fontsize=12)
A.legend(fontsize=9); A.grid(alpha=0.3)
A.text(0.03, 0.05, r"$\beta_c$ is at $\ln\xi_\beta=\infty$ — off the right edge",
       transform=A.transAxes, fontsize=9, color="crimson")

# ---- (b) the collapse: x = ln xi_measured -------------------------------------
B = ax[1]
for c in CH:
    pts = sorted([r for r in rows if r["chi"] == c], key=lambda r: r["xim"])
    B.plot([math.log(p["xim"]) for p in pts], [p["tS"] for p in pts], "o",
           ms=5, color=cmap[c], label=fr"$\chi_1={c}$" if c in (4,8,16,24,32) else None)
crit = sorted([r for r in rows if abs(r["beta"]-BC) < 1e-9], key=lambda r: r["xim"])
if crit:
    B.plot([math.log(p["xim"]) for p in crit], [p["tS"] for p in crit], "k-",
           lw=2.2, zorder=5, label=r"$\beta=\beta_c$ (message 2)")
B.axhline(0, color="0.5", lw=0.8, ls=":")
B.set_xlabel(r"$\ln\xi_1$  (measured, per $(\beta,\chi_1)$)")
B.set_ylabel(r"$\widetilde S_J$")
B.set_title(r"(b) collapse: every $\beta$, every $\chi_1$", fontsize=12)
B.legend(fontsize=8, ncol=2); B.grid(alpha=0.3)

# ---- (c) message 2: the slope at beta_c ---------------------------------------
C = ax[2]
x = np.array([math.log(p["xim"]) for p in crit]); y = np.array([p["tS"] for p in crit])
C.plot(x, y, "ko-", ms=7)
fits = {}
for lo in (0, 2, 4):
    if len(x)-lo >= 3:
        s, b0 = np.polyfit(x[lo:], y[lo:], 1); fits[lo] = (s, b0)
        C.plot(x, s*x+b0, "--", lw=1.2,
               label=fr"fit from $\chi_1\!\geq\!{sorted({r['chi'] for r in crit})[lo]}$:  $c={s:+.4f}$")
for p in crit:
    C.annotate(f"{p['chi']}", (math.log(p["xim"]), p["tS"]), xytext=(5, -10),
               textcoords="offset points", fontsize=8)
C.set_xlabel(r"$\ln\xi_1$ at $\beta_c$"); C.set_ylabel(r"$\widetilde S_J$")
C.set_title(r"(c) message 2:  $\widetilde S=c\ln\xi_\chi$", fontsize=12)
C.legend(fontsize=9); C.grid(alpha=0.3)

fig.suptitle(r"$\widetilde S_J$ against $\ln\xi$: the two messages and the collapse",
             fontsize=14)
fig.tight_layout(rect=[0,0,1,0.94])
fig.savefig(OUT, dpi=160)
print("wrote", OUT)

print(f"\nMESSAGE 2 -- slope at beta_c ({len(crit)} points, chi1="
      f"{[p['chi'] for p in crit]}):")
for lo, (s, b0) in fits.items():
    chis = sorted({r['chi'] for r in crit})
    print(f"   fit over chi1>={chis[lo]:2d}: Stilde = {s:+.5f} ln xi {b0:+.5f}")
print("\n   local slope between consecutive chi1:")
for i in range(len(x)-1):
    print(f"     chi1 {crit[i]['chi']:2d}->{crit[i+1]['chi']:2d}: "
          f"c = {(y[i+1]-y[i])/(x[i+1]-x[i]):+.4f}")

print("\nMESSAGE 1 -- converged off-critical points (SAFE only), disordered side:")
best = {}
for r in rows:
    if r["beta"] < BC: best.setdefault(r["beta"], []).append(r)
print(f"   {'beta':>6} {'xi_beta':>9} {'ln xi_b':>8} {'tS(max chi)':>12} {'chi':>5}")
for b in sorted(best):
    p = max(best[b], key=lambda r: r["chi"])
    print(f"   {b:6.3f} {p['xib']:9.3f} {math.log(p['xib']):8.3f} {p['tS']:+12.6f} {p['chi']:5d}")
