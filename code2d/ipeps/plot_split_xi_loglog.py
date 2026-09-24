# plot_split_xi_loglog.py -- log-log view of the converged split-replica values.
# tildeS_J changes sign across beta_c, so |tildeS_J| is plotted and the two
# phases are kept as separate branches.  The second panel is the local
# logarithmic slope p = d ln|tildeS| / d ln xi between consecutive points: a
# power law is a plateau there, a logarithm drifts to zero.
#
# Two traps this script used to fall into, both fixed here.
#  (1) The branch split used to key on beta < beta_c.  beta=0.43 is on the
#      disordered side but its converged-so-far value is NEGATIVE (-0.0328), so
#      it was drawn on the line whose legend asserts "disordered => S>0".  The
#      split now keys on the SIGN of tildeS, which is what the two branches
#      actually mean.
#  (2) The local slope ln|tS2/tS1| / ln(xi2/xi1) was computed straight through
#      that sign change, returning p = -2.38 at xi ~ 16.6 -- a logarithmic slope
#      across a zero, i.e. meaningless.  Such pairs are now skipped.
# Convergence class is shown, not hidden: sigma = |last chi1 drift| and points
# with sigma/|tildeS| > 1e-2 (only beta=0.43) are drawn hollow and OFF the
# joining line, because a chi-unconverged point on a joined curve reads as
# physics.
import os, csv, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/ix/zdai/kangw/PEPS3EE/PEPS/clean"
SPLIT = f"{R}/data/rk_ising/ctmrg/tables/beta_chi_scan_split/summary.csv"
MC_PLAT = f"{R}/data/rk_ising/mc/campaigns/sw_components_near_critical_100k_16chains/plateau_summary.csv"
OUT = f"{R}/data/rk_ising/ctmrg/figures/beta_chi_scan_split/stilde_vs_xi_loglog.png"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
BETA_C = 0.5*math.log(1+math.sqrt(2))

def xi_spin(beta):
    bd = -0.5*math.log(math.tanh(beta))
    return 1/(2*(bd-beta)) if beta < BETA_C else 1/(4*(beta-bd))

rows = {}
for r in csv.DictReader(open(SPLIT)):
    b, chi1 = float(r["beta"]), int(r["chi1"])
    rows.setdefault(b, {})[chi1] = float(r["tSJ"])
def classify(rel):
    if not math.isfinite(rel):
        return "UNKNOWN"
    return "SAFE" if rel <= 1e-3 else ("MARGINAL" if rel <= 1e-2 else "EXCLUDE")

pts = []
for b, d in sorted(rows.items()):
    chis = sorted(d)
    drift = abs(d[chis[-1]] - d[chis[-2]]) if len(chis) > 1 else float("nan")
    tS = d[chis[-1]]
    rel = drift / abs(tS) if tS else float("inf")
    pts.append(dict(beta=b, xi=xi_spin(b), tS=tS, chi=chis[-1], drift=drift,
                    rel=rel, cls=classify(rel)))

mc = {}
for r in csv.DictReader(open(MC_PLAT)):
    mc[float(r["beta"])] = (float(r["direct_plateau"]), float(r["direct_error"]),
                            float(r["direct_chi2_dof"]))

# Branch by the SIGN of the observable, not by the phase: beta=0.43 is
# disordered but negative, and putting it on the positive branch mislabels it.
dis = [p for p in pts if p["tS"] > 0]
ordd = [p for p in pts if p["tS"] < 0]
excl = [p for p in pts if p["cls"] == "EXCLUDE"]

fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2))
fig.suptitle(r"split-replica $|\tilde S_J|$ vs $\xi_{\rm spin}$, log--log "
             r"(sign: $+$ disordered, $-$ ordered)", fontsize=13)

a = ax[0]
for branch, col, lab in ((dis, "royalblue", r"$\tilde S>0$ ($\beta<\beta_c$)"),
                         (ordd, "crimson", r"$\tilde S<0$")):
    keep = sorted([p for p in branch if p["cls"] != "EXCLUDE"], key=lambda p: p["xi"])
    x = [p["xi"] for p in keep]; y = [abs(p["tS"]) for p in keep]
    yerr = [max(p["drift"], abs(p["tS"])*1e-4) for p in keep]
    a.errorbar(x, y, yerr=yerr, fmt="o-", ms=7, color=col, capsize=2, label=lab)
    for p in keep:
        a.annotate(rf"{p['beta']:.2f}", (p["xi"], abs(p["tS"])),
                   xytext=(5, -9), textcoords="offset points", fontsize=7, color=col)
# chi-unconverged points: hollow, off the line, with their drift as the bar.
for p in sorted(excl, key=lambda q: q["xi"]):
    a.errorbar(p["xi"], abs(p["tS"]), yerr=p["drift"], fmt="s", ms=10,
               mfc="none", mec="black", ecolor="black", capsize=3, zorder=5,
               label=r"$\chi$-unconverged (excluded)" if p is excl[0] else None)
    a.annotate(rf"{p['beta']:.2f}", (p["xi"], abs(p["tS"])),
               xytext=(7, 4), textcoords="offset points", fontsize=8, color="black")
for b, (v, e, c2) in mc.items():
    if b in rows:
        a.errorbar(xi_spin(b)*1.06, abs(v), yerr=e, fmt="o", ms=6,
                   color="black" if c2 < 2 else "gray",
                   mfc="black" if c2 < 2 else "none", capsize=2, zorder=1)
xr = np.array([0.5, 13.0])
for p0, x0, y0, sty in ((1.0, 6.0, 0.12, ":"), (2.0, 2.0, 0.02, "--"),
                        (4.0, 1.0, 0.0035, "-.")):
    a.loglog(xr, y0*(xr/x0)**p0, sty, color="gray", lw=0.9)
    a.annotate(rf"$\xi^{{{p0:.0f}}}$", (xr[0]*1.15, y0*(xr[0]*1.15/x0)**p0),
               fontsize=8, color="gray")
a.set_xlabel(r"$\xi_{\rm spin}(\beta)$"); a.set_ylabel(r"$|\tilde S_J|$")
a.set_title("(a) both branches bend: no single power law")
a.legend(fontsize=8, loc="lower right")

b_ = ax[1]
for branch, col, lab in ((dis, "royalblue", r"$\tilde S>0$"),
                         (ordd, "crimson", r"$\tilde S<0$")):
    branch = sorted(branch, key=lambda p: p["xi"])
    xs, ps = [], []
    for i in range(len(branch)-1):
        p1, p2 = branch[i], branch[i+1]
        # A logarithmic slope through a sign change, or across a chi-unconverged
        # point, is not a slope.  Skip both.
        if p1["tS"] * p2["tS"] <= 0:
            continue
        if "EXCLUDE" in (p1["cls"], p2["cls"]):
            continue
        xs.append(math.sqrt(p1["xi"]*p2["xi"]))
        ps.append(math.log(abs(p2["tS"]/p1["tS"]))/math.log(p2["xi"]/p1["xi"]))
    b_.semilogx(xs, ps, "s-", ms=7, color=col, label=lab)
b_.axhline(0, color="gray", lw=0.6)
b_.set_xlabel(r"$\xi_{\rm spin}$ (geometric mean of the pair)")
b_.set_ylabel(r"local slope $p=\mathrm{d}\ln|\tilde S_J|/\mathrm{d}\ln\xi$")
b_.set_title(r"(b) effective power falls from $\sim4$ toward $\sim0$")
b_.legend(fontsize=9)

fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(OUT, dpi=160)
print("wrote", OUT)
for branch, name in ((dis, "tildeS > 0"), (ordd, "tildeS < 0")):
    print(f"\n{name}:")
    branch = sorted(branch, key=lambda p: p["xi"])
    for i, p in enumerate(branch):
        s = ""
        if i:
            q = branch[i-1]
            s = (f"  local p={math.log(abs(p['tS']/q['tS']))/math.log(p['xi']/q['xi']):5.2f}")
        print(f"  beta={p['beta']:.2f} xi={p['xi']:7.3f} chi1={p['chi']:3d} "
              f"tS={p['tS']:+.7f} drift={p['drift']:.1e} rel={p['rel']:.1e} "
              f"{p['cls']:9s}{s}")
