# plot_collapse_clogL.py -- test whether subtracting c*ln(L) collapses the
# near-critical S~(L,beta) data onto a function of L/xi_spin(beta).
#   c := large-L slope of the CRITICAL curve, dS~/d ln L  (fitted below).
# Rationale: if  S~(L,xi) = c ln(min(L,xi)) + f(L/xi), then
#   S~ - c ln L = c ln(min(L,xi)/L) + f(L/xi)  is a pure function of L/xi.
import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NC = ("/ix/zdai/kangw/PEPS3EE/PEPS/clean/data/rk_ising/mc/campaigns/"
      "sw_components_near_critical_100k_16chains/direct_tS.csv")
CR = ("/ix/zdai/kangw/PEPS3EE/PEPS/clean/data/rk_ising/mc/campaigns/"
      "sw_components_100k_16chains/pooled_components.csv")
OUT = ("/ix/zdai/kangw/PEPS3EE/PEPS/clean/data/rk_ising/mc/campaigns/sw_components_near_critical_100k_16chains/"
       "collapse_clogL.png")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
BETA_C = 0.5*np.log(1+np.sqrt(2))
LMIN_FIT = 48          # critical fit window

# ---- critical curve, fit c = dS/d ln L ---------------------------------------
cl, cs, ce = [], [], []
for r in csv.DictReader(open(CR)):
    if r["beta"] == "crit" and r["observable"] == "tS":
        cl.append(int(r["L"])); cs.append(float(r["value"])); ce.append(float(r["error"]))
o = np.argsort(cl)
cl, cs, ce = np.array(cl)[o], np.array(cs)[o], np.array(ce)[o]
m = cl >= LMIN_FIT
w = 1/ce[m]**2
X = np.log(cl[m]); Y = cs[m]
Sw, Sx, Sy = w.sum(), (w*X).sum(), (w*Y).sum()
Sxx, Sxy = (w*X*X).sum(), (w*X*Y).sum()
det = Sw*Sxx - Sx*Sx
c = (Sw*Sxy - Sx*Sy)/det
b0 = (Sxx*Sy - Sx*Sxy)/det
c_err = np.sqrt(Sw/det)
# window stability
alt = {}
for lmin in (32, 48, 64):
    mm = cl >= lmin
    ww = 1/ce[mm]**2; XX = np.log(cl[mm]); YY = cs[mm]
    S0, S1, S2 = ww.sum(), (ww*XX).sum(), (ww*XX*XX).sum()
    T0, T1 = (ww*YY).sum(), (ww*XX*YY).sum()
    alt[lmin] = (S0*T1 - S1*T0)/(S0*S2 - S1*S1)

# ---- near-critical data ------------------------------------------------------
d = {}
for r in csv.DictReader(open(NC)):
    b = float(r["beta"])
    d.setdefault(b, []).append((int(r["L"]), float(r["L_over_xi"]),
                                float(r["direct_tS"]), float(r["error"])))
betas = sorted(d)
cmap = plt.cm.coolwarm(np.linspace(0, 1, len(betas)))
COL = {b: cmap[i] for i, b in enumerate(betas)}

fig, ax = plt.subplots(2, 3, figsize=(16.5, 9.4))
ax = ax.ravel()
fig.suptitle(r"does $\tilde S-c\ln L$ collapse onto a function of $L/\xi$?"
             rf"   ($c$ = critical large-$L$ slope $={c:.3f}$)", fontsize=12)

# (a) the critical fit
a = ax[0]
a.errorbar(cl, cs, yerr=ce, fmt="o", ms=5, color="black", label=r"MC, $\beta=\beta_c$")
xx = np.linspace(np.log(cl[0]), np.log(cl[-1]), 50)
a.plot(np.exp(xx), b0 + c*xx, "-", color="crimson", lw=1.5,
       label=rf"fit $L\geq{LMIN_FIT}$: $c={c:.3f}({c_err*1e3:.0f})$")
a.set_xscale("log"); a.axhline(0, color="gray", lw=0.6)
a.set_xlabel(r"$L$"); a.set_ylabel(r"$\tilde S(\beta_c,L)$")
a.set_title(rf"(a) critical slope; windows: " +
            ", ".join(rf"$L\geq{k}$: {v:.3f}" for k, v in alt.items()), fontsize=9)
a.legend(fontsize=8)

# (b) raw data vs L/xi
b_ = ax[1]
for bb in betas:
    rows = sorted(d[bb])
    x = [r[1] for r in rows]; y = [r[2] for r in rows]; e = [r[3] for r in rows]
    b_.errorbar(x, y, yerr=e, fmt="o-", ms=4, lw=1, color=COL[bb],
                label=rf"$\beta={bb:.2f}$")
b_.axhline(0, color="gray", lw=0.6); b_.set_xscale("log")
b_.set_xlabel(r"$L/\xi_{\rm spin}$"); b_.set_ylabel(r"$\tilde S$")
b_.set_title("(b) raw (right panel of the campaign figure)")
b_.legend(fontsize=7, ncol=2)

# (c) collapse attempt
c_ = ax[2]
for bb in betas:
    rows = sorted(d[bb])
    x = np.array([r[1] for r in rows]); L = np.array([r[0] for r in rows])
    y = np.array([r[2] for r in rows]) - c*np.log(L)
    e = np.array([r[3] for r in rows])
    ls = "-" if bb < BETA_C else "--"
    c_.errorbar(x, y, yerr=e, fmt="o", ls=ls, ms=4, lw=1, color=COL[bb],
                label=rf"$\beta={bb:.2f}$" + (" (dis)" if bb < BETA_C else " (ord)"))
c_.set_xscale("log")
c_.set_xlabel(r"$L/\xi_{\rm spin}$"); c_.set_ylabel(r"$\tilde S-c\ln L$")
c_.set_title(r"(c) collapse attempt: $\tilde S-c\ln L$ vs $L/\xi$")
c_.legend(fontsize=7, ncol=2)


# ---- (d) spread vs subtraction coefficient -----------------------------------
def spread(cc, subset=None):
    """mean std across beta of (tS - cc lnL) in ln(L/xi) bins with >=3 betas."""
    bs = [b for b in betas if subset is None or subset(b)]
    edges = np.exp(np.linspace(np.log(1.0), np.log(40.0), 11))
    tot, n = 0.0, 0
    for lo, hi in zip(edges[:-1], edges[1:]):
        vals = []
        for bb in bs:
            sel = [(L, lx, v) for L, lx, v, e in d[bb] if lo <= lx < hi]
            if sel:
                vals.append(np.mean([v - cc*np.log(L) for L, lx, v in sel]))
        if len(vals) >= 3:
            tot += np.std(vals); n += 1
    return tot/max(n, 1)

cs_scan = np.linspace(-0.6, 0.4, 201)
sp_all = np.array([spread(x) for x in cs_scan])
sp_dis = np.array([spread(x, lambda b: b < BETA_C) for x in cs_scan])
sp_ord = np.array([spread(x, lambda b: b > BETA_C) for x in cs_scan])
c_all, c_dis, c_ord = (cs_scan[np.argmin(v)] for v in (sp_all, sp_dis, sp_ord))
dd = ax[3]
dd.plot(cs_scan, sp_all, "-", color="black", label=rf"all $\beta$ (min at {c_all:+.3f})")
dd.plot(cs_scan, sp_dis, "-", color="royalblue", label=rf"disordered (min {c_dis:+.3f})")
dd.plot(cs_scan, sp_ord, "-", color="firebrick", label=rf"ordered (min {c_ord:+.3f})")
dd.axvline(c, color="crimson", ls="--", lw=1.2, label=rf"critical slope $c={c:.3f}$")
dd.set_xlabel("subtraction coefficient"); dd.set_ylabel("collapse spread")
dd.set_title("(d) no coefficient collapses both regimes")
dd.legend(fontsize=8)

# ---- (e) collapse at the best coefficient ------------------------------------
ee = ax[4]
for bb in betas:
    rows = sorted(d[bb])
    x = np.array([r[1] for r in rows]); L = np.array([r[0] for r in rows])
    y = np.array([r[2] for r in rows]) - c_all*np.log(L)
    ls = "-" if bb < BETA_C else "--"
    ee.plot(x, y, "o", ls=ls, ms=4, lw=1, color=COL[bb], label=rf"$\beta={bb:.2f}$")
ee.set_xscale("log"); ee.set_xlabel(r"$L/\xi_{\rm spin}$")
ee.set_ylabel(rf"$\tilde S-({c_all:+.3f})\ln L$")
ee.set_title(rf"(e) best-fit coefficient {c_all:+.3f}: still no collapse")
ee.legend(fontsize=6, ncol=2)

# ---- (f) plateaus vs ln xi ----------------------------------------------------
PL = ("/ix/zdai/kangw/PEPS3EE/PEPS/clean/data/rk_ising/mc/campaigns/"
      "sw_components_near_critical_100k_16chains/plateau_summary.csv")
ff = ax[5]
for phase, col in (("disordered", "royalblue"), ("ordered", "firebrick")):
    xs, ys, es = [], [], []
    for r in csv.DictReader(open(PL)):
        if r["phase"] != phase:
            continue
        xs.append(np.log(float(r["xi_spin"]))); ys.append(float(r["direct_plateau"]))
        es.append(float(r["direct_error"]))
    o2 = np.argsort(xs); xs = np.array(xs)[o2]; ys = np.array(ys)[o2]; es = np.array(es)[o2]
    sl = np.polyfit(xs, ys, 1)[0]
    ff.errorbar(np.exp(xs), ys, yerr=es, fmt="o-", ms=5, color=col,
                label=rf"{phase}: slope {sl:+.3f}")
ff.axhline(0, color="gray", lw=0.6); ff.set_xscale("log")
ff.set_xlabel(r"$\xi_{\rm spin}$"); ff.set_ylabel("plateau value")
ff.set_title(rf"(f) plateau log-slope $\neq c={c:.3f}$ (sign and size)")
ff.legend(fontsize=8)

fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT, dpi=160)
print("wrote", OUT)
print(f"critical fit (L>={LMIN_FIT}): c = {c:.4f} +- {c_err:.4f}, intercept {b0:.4f}")
print("window stability:", {k: round(v, 4) for k, v in alt.items()})
print("\ncritical curve:")
for L, s, e in zip(cl, cs, ce):
    print(f"   L={L:4d} tS={s:+.5f}({e*1e5:.0f})")
print("\ncollapse spread: at fixed L/xi bins, std of (tS - c lnL) across beta")
allpts = []
for bb in betas:
    for L, lx, v, e in d[bb]:
        allpts.append((lx, v - c*np.log(L), bb))
allpts.sort()
for lo, hi in ((0.5, 1), (1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 100)):
    sel = [p for p in allpts if lo <= p[0] < hi]
    if len(sel) > 1:
        vals = np.array([p[1] for p in sel])
        print(f"   L/xi in [{lo},{hi}): n={len(sel):3d}  mean={vals.mean():+.3f}  "
              f"std={vals.std():.3f}  range={vals.max()-vals.min():.3f}")
