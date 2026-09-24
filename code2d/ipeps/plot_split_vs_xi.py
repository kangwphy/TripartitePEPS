# plot_split_vs_xi.py -- converged split-replica tildeS_J against the exact
# Ising correlation length.  The CTMRG-measured xi1 is unusable on the ordered
# side (degenerate leading transfer eigenvalues of the Z2-symmetric environment
# give xi -> inf), so the exact analytic xi_spin(beta) is used for both phases,
# matching the Monte Carlo campaign convention.
import os, csv, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/ix/zdai/kangw/PEPS3EE/PEPS/clean"
SPLIT = f"{R}/data/rk_ising/ctmrg/tables/beta_chi_scan_split/summary.csv"
MC_PLAT = f"{R}/data/rk_ising/mc/campaigns/sw_components_near_critical_100k_16chains/plateau_summary.csv"
MC_MAIN = f"{R}/data/rk_ising/mc/campaigns/sw_components_100k_16chains/pooled_components.csv"
OUT = f"{R}/data/rk_ising/ctmrg/figures/beta_chi_scan_split/stilde_vs_xi.png"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
BETA_C = 0.5*math.log(1+math.sqrt(2))

def xi_spin(beta):
    """Exact connected-spin exponential correlation length along an axis."""
    beta_dual = -0.5*math.log(math.tanh(beta))
    if beta < BETA_C:
        return 1.0/(2.0*(beta_dual - beta))
    return 1.0/(4.0*(beta - beta_dual))

# ---- split-replica: largest converged chi1 per beta ---------------------------
rows = {}
for r in csv.DictReader(open(SPLIT)):
    b, chi1 = float(r["beta"]), int(r["chi1"])
    rows.setdefault(b, {})[chi1] = (float(r["tSJ"]), float(r["s2"]),
                                    float(r["kA"]), int(r["env_converged"]))
split = {}
for b, d in rows.items():
    chis = sorted(d)
    tS = d[chis[-1]][0]
    prev = d[chis[-2]][0] if len(chis) > 1 else float("nan")
    split[b] = dict(chi=chis[-1], tS=tS, drift=abs(tS-prev),
                    s2=d[chis[-1]][1], kA=d[chis[-1]][2], xi=xi_spin(b))

# ---- Monte Carlo reference ---------------------------------------------------
mc = {}
for r in csv.DictReader(open(MC_PLAT)):
    b = float(r["beta"])
    mc[b] = dict(v=float(r["direct_plateau"]), e=float(r["direct_error"]),
                 chi2=float(r["direct_chi2_dof"]), phase=r["phase"])
mcL = {}
for r in csv.DictReader(open(MC_MAIN)):
    if r["observable"] == "tS" and r["beta"] != "crit":
        mcL.setdefault(float(r["beta"]), []).append(
            (int(r["L"]), float(r["value"]), float(r["error"])))

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(r"split-replica $\tilde S_J$ vs the exact correlation length "
             r"$\xi_{\rm spin}(\beta)$", fontsize=13)

for a, xlog in zip(ax, (False, True)):
    for b in sorted(split):
        s = split[b]
        ordered = b > BETA_C
        col = "crimson" if ordered else "royalblue"
        a.errorbar(s["xi"], s["tS"], yerr=max(s["drift"], 1e-6), fmt="s", ms=10,
                   color=col, capsize=3, zorder=3)
        a.annotate(rf"$\beta$={b:.2f}" + ("\n" + rf"$\chi_1$={s['chi']}" if not xlog else ""),
                   (s["xi"], s["tS"]), xytext=(6, 7), textcoords="offset points",
                   fontsize=8, color=col)
        if b in mc:
            m = mc[b]
            good = m["chi2"] < 2
            a.errorbar(s["xi"]*1.06, m["v"], yerr=m["e"], fmt="o", ms=7,
                       color="black" if good else "gray",
                       mfc="black" if good else "none", capsize=3, zorder=2)
        elif b in mcL:
            L, v, e = sorted(mcL[b])[-1]
            a.errorbar(s["xi"]*1.06, v, yerr=e, fmt="o", ms=7, color="gray",
                       mfc="none", capsize=3, zorder=2)
    a.axhline(0, color="gray", lw=0.6)
    if xlog:
        a.set_xscale("log")
    a.set_xlabel(r"$\xi_{\rm spin}(\beta)$  (exact)")
    a.set_ylabel(r"$\tilde S_J$")
a.plot([], [], "s", ms=9, color="royalblue", label=r"split, disordered $\beta<\beta_c$")
a.plot([], [], "s", ms=9, color="crimson", label=r"split, ordered $\beta>\beta_c$")
a.plot([], [], "o", ms=7, color="black", label=r"MC plateau ($\chi^2/\nu<2$), offset $\times1.06$")
a.plot([], [], "o", ms=7, color="gray", mfc="none",
       label=r"MC, unconverged/uncertified")
ax[0].set_title("(a) linear in $\\xi$"); ax[1].set_title("(b) log in $\\xi$")
ax[1].legend(fontsize=8, loc="lower left")

fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(OUT, dpi=160)
print("wrote", OUT)
print(f"{'beta':>6} {'phase':>10} {'xi_exact':>9} {'chi1':>5} {'tSJ':>12} "
      f"{'chi-drift':>10} {'MC':>12} {'MC chi2/dof':>11}")
for b in sorted(split):
    s = split[b]
    ph = "ordered" if b > BETA_C else "disordered"
    if b in mc:
        mtxt = f"{mc[b]['v']:+.5f}"; c2 = f"{mc[b]['chi2']:.1f}"
    elif b in mcL:
        L, v, e = sorted(mcL[b])[-1]; mtxt = f"{v:+.5f}(L{L})"; c2 = "-"
    else:
        mtxt = "-"; c2 = "-"
    print(f"{b:6.2f} {ph:>10} {s['xi']:9.3f} {s['chi']:5d} {s['tS']:+12.6f} "
          f"{s['drift']:10.2e} {mtxt:>12} {c2:>11}")
