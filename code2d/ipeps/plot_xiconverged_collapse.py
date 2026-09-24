#!/usr/bin/env python3
"""Collapse restricted to points whose xi has itself converged in chi1.

The convergence class used elsewhere is about Stilde.  Here the gate is on the
LENGTH: a (beta, chi1) point is kept only if the measured xi_1 has stopped
moving with chi1, i.e. the environment is resolving the physical correlation
length rather than truncating it.  Those points are the only ones for which the
x axis really is xi_beta, so they are the only ones that can test the physical
statement Stilde = c ln xi_beta without finite-chi contamination.

The two convergences are NOT the same and the figure reports both: xi converges
much more slowly than Stilde, because Stilde is a ratio of window sums that is
insensitive to the tail of the transfer spectrum while xi is defined by it.
"""
import csv, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/ix/zdai/kangw/PEPS3EE/PEPS/clean"
SCAN = f"{R}/data/rk_ising/ctmrg/tables/beta_chi_scan_split/summary.csv"
OUT = f"{R}/data/rk_ising/ctmrg/figures/beta_chi_scan_split/xiconverged_collapse.png"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
BC = 0.5*math.log(1+math.sqrt(2))

def xi_beta(b):
    bd = -0.5*math.log(math.tanh(b))
    return math.inf if abs(b-BC) < 1e-9 else (1/(2*(bd-b)) if b < BC else 1/(4*(b-bd)))

series = {}
for r in csv.DictReader(open(SCAN)):
    b, c = float(r["beta"]), int(r["chi1"])
    try: x = float(r["xi1"])
    except ValueError: continue
    if not math.isfinite(x) or x <= 0 or x > 1e6: continue
    series.setdefault(b, {})[c] = (x, float(r["tSJ"]))

THRESH = [1e-2, 3e-3, 1e-3]
kept = {t: [] for t in THRESH}
print(f"{'beta':>8} {'chi_max':>7} {'xi_1':>10} {'xi drift':>10} {'S drift':>10}  kept at")
for b in sorted(series):
    cs = sorted(series[b])
    if len(cs) < 2: continue
    (x1, s1), (x0, s0) = series[b][cs[-1]], series[b][cs[-2]]
    dxi = abs(x1-x0)/abs(x1)
    dS = abs(s1-s0)/abs(s1) if s1 else float("inf")
    tags = [f"{t:.0e}" for t in THRESH if dxi < t]
    for t in THRESH:
        if dxi < t:
            kept[t].append(dict(beta=b, chi=cs[-1], xi=x1, tS=s1,
                                dxi=dxi, dS=dS, ph="dis" if b < BC else "ord"))
    print(f"{b:8.4f} {cs[-1]:7d} {x1:10.3f} {dxi:10.2e} {dS:10.2e}  "
          f"{','.join(tags) if tags else '-'}")

fig, ax = plt.subplots(1, 2, figsize=(14.5, 5.8))
COL = {"dis": "royalblue", "ord": "crimson"}
A = ax[0]
for t, mk, a_ in zip(THRESH, ("o", "s", "^"), (0.30, 0.62, 1.0)):
    for ph in ("dis", "ord"):
        pts = sorted([p for p in kept[t] if p["ph"] == ph], key=lambda p: p["xi"])
        if not pts: continue
        A.plot([math.log(p["xi"]) for p in pts], [p["tS"] for p in pts], mk,
               ms=7, color=COL[ph], alpha=a_, mfc=COL[ph] if t == THRESH[-1] else "none",
               label=f"{'dis' if ph=='dis' else 'ord'}, $\\delta\\xi<${t:.0e}")
A.axhline(0, color="0.5", lw=0.8, ls=":")
A.set_xlabel(r"$\ln\xi_1$"); A.set_ylabel(r"$\widetilde S_J$")
A.set_title(r"(a) only points whose $\xi$ has converged in $\chi_1$", fontsize=12)
A.legend(fontsize=8, ncol=2); A.grid(alpha=0.3)

B = ax[1]
sel = kept[3e-3]
for ph in ("dis", "ord"):
    pts = sorted([p for p in sel if p["ph"] == ph], key=lambda p: p["xi"])
    if not pts: continue
    x = np.array([math.log(p["xi"]) for p in pts]); y = np.array([p["tS"] for p in pts])
    B.plot(x, y, "o-", ms=7, color=COL[ph],
           label=("disordered" if ph == "dis" else "ordered"))
    for p in pts:
        B.annotate(f"{p['beta']:.3g}", (math.log(p["xi"]), p["tS"]),
                   xytext=(4, -10), textcoords="offset points", fontsize=7,
                   color=COL[ph])
    m = x > 1.0
    if m.sum() >= 3:
        s, b0 = np.polyfit(x[m], y[m], 1)
        B.plot(x[m], s*x[m]+b0, "--", color=COL[ph], lw=1.2,
               label=fr"fit $\ln\xi>1$: $c={s:+.4f}$")
B.axhline(0, color="0.5", lw=0.8, ls=":")
B.set_xlabel(r"$\ln\xi_1$"); B.set_ylabel(r"$\widetilde S_J$")
B.set_title(r"(b) $\delta\xi<3\times10^{-3}$, both phases, with slopes", fontsize=12)
B.legend(fontsize=9); B.grid(alpha=0.3)

fig.suptitle(r"collapse using ONLY $\xi$-converged points", fontsize=14)
fig.tight_layout(rect=[0,0,1,0.94])
fig.savefig(OUT, dpi=160)
print("\nwrote", OUT)
for t in THRESH:
    d = [p for p in kept[t] if p["ph"] == "dis"]; o = [p for p in kept[t] if p["ph"] == "ord"]
    rng = lambda z: (f"{min(math.log(p['xi']) for p in z):.2f}..{max(math.log(p['xi']) for p in z):.2f}"
                     if z else "-")
    print(f"  dxi<{t:.0e}: {len(d)} disordered (ln xi {rng(d)}), "
          f"{len(o)} ordered (ln xi {rng(o)})")
