#!/usr/bin/env python3
# fit_tail.py -- FINAL tail fit of the critical tildeS(L) descent.
# Pools mc_results.csv crit rows per L (weighted mean; error = max(propagated,
# scatter sem)), then fits the tail L >= Lmin with four models:
#   [LIN ]  S = a - b*L                       (linear descent)
#   [POW ]  S = a - c*L^p                     (pure power tail, p free)
#   [LOGP]  S = a + b*ln(L) - c*L^p           (CFT log + power, b free)
#   [LOGF]  S = a + 0.103*ln(L) - c*L^p       (CFT log coefficient FIXED)
# Reports chi2/dof, parameters, and S(128)/S(192) forecasts (the numbers a
# future run would discriminate on).
import os, csv, math
import numpy as np
from collections import defaultdict
from scipy.optimize import curve_fit

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
DATA = os.environ.get("DATA", os.path.join(ROOT, "data2d"))
MC_FILE = os.environ.get("MC_FILE", os.path.join(ROOT, "data/rk_ising/mc", "mc_results.csv"))
pools = defaultdict(list)
with open(MC_FILE) as f:
    for row in csv.reader(f):
        if len(row) < 7 or row[1] != "crit":
            continue
        tag = row[7] if len(row) > 7 else ""
        if "S2C" in tag or "K4C" in tag:   # diagnostic observables, not tildeS
            continue
        pools[int(row[0])].append((float(row[4]), float(row[5])))

Ls, Ss, Es = [], [], []
print("pooled critical points:")
for L in sorted(pools):
    v = np.array([x for x, _ in pools[L]])
    e = np.array([x for _, x in pools[L]])
    w = 1.0 / e**2
    mean = float(np.sum(w * v) / np.sum(w))
    err_prop = 1.0 / math.sqrt(float(np.sum(w)))
    err_scat = float(np.std(v, ddof=1) / math.sqrt(len(v))) if len(v) > 1 else err_prop
    err = max(err_prop, err_scat)
    print(f"  L={L:3d}  n={len(v):2d}  S={mean:+.5f} +- {err:.5f}")
    Ls.append(L); Ss.append(mean); Es.append(err)
Ls = np.array(Ls, float); Ss = np.array(Ss); Es = np.array(Es)

def report(name, f, p0, Lmin, bounds=(-np.inf, np.inf)):
    m = Ls >= Lmin
    n = int(m.sum()); k = len(p0)
    if n <= k:
        print(f"  {name:5s} Lmin={Lmin:2d}: skipped (n={n} <= k={k})")
        return
    try:
        popt, pcov = curve_fit(f, Ls[m], Ss[m], p0=p0, sigma=Es[m],
                               absolute_sigma=True, bounds=bounds, maxfev=200000)
    except Exception as ex:
        print(f"  {name:5s} Lmin={Lmin:2d}: FIT FAILED ({ex})")
        return
    r = (f(Ls[m], *popt) - Ss[m]) / Es[m]
    chi2 = float(np.sum(r**2)); dof = n - k
    perr = np.sqrt(np.diag(pcov))
    ps = "  ".join(f"{v:+.4f}({e:.4f})" for v, e in zip(popt, perr))
    print(f"  {name:5s} Lmin={Lmin:2d}: chi2/dof={chi2/max(dof,1):6.2f}  [{ps}]"
          f"   S(128)={f(128.0,*popt):+.3f}  S(192)={f(192.0,*popt):+.3f}")

lin  = lambda L, a, b: a - b * L
powm = lambda L, a, c, p: a - c * L**p
logp = lambda L, a, b, c, p: a + b * np.log(L) - c * L**p
logf = lambda L, a, c, p: a + 0.103 * np.log(L) - c * L**p

for Lmin in (16, 20, 24, 32):
    print(f"\n--- tail fits, Lmin={Lmin} ---")
    report("LIN ", lin,  (0.1, 0.005), Lmin)
    report("POW ", powm, (0.5, 0.05, 0.6), Lmin, bounds=([-20, 0, 0.05], [20, 20, 2.0]))
    report("LOGP", logp, (0.0, 0.103, 0.05, 0.6), Lmin,
           bounds=([-20, -2, 0, 0.05], [20, 2, 20, 2.0]))
    report("LOGF", logf, (0.0, 0.05, 0.6), Lmin, bounds=([-20, 0, 0.05], [20, 20, 2.0]))
print("\nDONE FIT-TAIL")
