# plot_split_vs_measured_xi.py -- same figure as stilde_vs_xi.png, but the axis
# is the correlation length the CODE measures, not the analytic one.
#
# Source of the measured xi (both use the degeneracy-aware estimator):
#   * xi1 in beta_chi_scan_split/summary.csv for runs made after the patch
#     (and for every disordered run, where the old and new estimators agree);
#   * logs2d/measurexi_*.out for the ordered betas whose scan predates the
#     patch and therefore stored 1e15 / -inf.
# Ordered-phase caveat kept visible: the first non-degenerate channel level is
# the single-domain-wall state, so the measured length is ~2 xi_spin.
import os, csv, math, re, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/ix/zdai/kangw/PEPS3EE/PEPS/clean"
SPLIT = f"{R}/data/rk_ising/ctmrg/tables/beta_chi_scan_split/summary.csv"
XILOGS = f"{R}/logs2d/measurexi_*.out"
MC_PLAT = f"{R}/data/rk_ising/mc/campaigns/sw_components_near_critical_100k_16chains/plateau_summary.csv"
MC_MAIN = f"{R}/data/rk_ising/mc/campaigns/sw_components_100k_16chains/pooled_components.csv"
OUT = f"{R}/data/rk_ising/ctmrg/figures/beta_chi_scan_split/stilde_vs_measured_xi.png"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
BETA_C = 0.5*math.log(1+math.sqrt(2))

def xi_exact(beta):
    bd = -0.5*math.log(math.tanh(beta))
    return 1/(2*(bd-beta)) if beta < BETA_C else 1/(4*(beta-bd))

# ---- measured xi from the standalone diagnostic --------------------------------
xi_diag = {}
for path in glob.glob(XILOGS):
    for line in open(path):
        m = re.search(r"SUMMARY-XI beta=([\d.]+) chi1=\s*(\d+).*?xi_conn=\s*([\d.]+)", line)
        if m:
            xi_diag[(round(float(m.group(1)), 4), int(m.group(2)))] = float(m.group(3))

# ---- split values, largest chi1 per beta ---------------------------------------
rows = {}
for r in csv.DictReader(open(SPLIT)):
    b, chi1 = float(r["beta"]), int(r["chi1"])
    try:
        xi = float(r["xi1"])
    except ValueError:
        xi = math.inf
    rows.setdefault(b, {})[chi1] = (float(r["tSJ"]), xi)

pts = []
for b, d in sorted(rows.items()):
    chis = sorted(d)
    tS, xi = d[chis[-1]]
    chi_used = chis[-1]
    if not math.isfinite(xi) or xi > 1e6:          # pre-patch ordered row
        for c in reversed(chis):                    # fall back to the diagnostic
            if (round(b, 4), c) in xi_diag:
                xi, chi_used = xi_diag[(round(b, 4), c)], c
                break
        else:
            continue
    drift = abs(d[chis[-1]][0] - d[chis[-2]][0]) if len(chis) > 1 else float("nan")
    pts.append(dict(beta=b, xi=xi, chi=chis[-1], chi_xi=chi_used, tS=tS,
                    drift=drift, xe=xi_exact(b)))

mc = {}
for r in csv.DictReader(open(MC_PLAT)):
    mc[float(r["beta"])] = (float(r["direct_plateau"]), float(r["direct_error"]),
                            float(r["direct_chi2_dof"]))
mcL = {}
for r in csv.DictReader(open(MC_MAIN)):
    if r["observable"] == "tS" and r["beta"] != "crit":
        mcL.setdefault(float(r["beta"]), []).append(
            (int(r["L"]), float(r["value"]), float(r["error"])))

fig, ax = plt.subplots(1, 3, figsize=(17.5, 5.2))
fig.suptitle(r"split-replica $\tilde S_J$ against the CODE-MEASURED correlation "
             r"length $\xi_1$ (degeneracy-aware estimator)", fontsize=13)

for k, (a, xlog) in enumerate(zip(ax[:2], (False, True))):
    for p in pts:
        col = "crimson" if p["beta"] > BETA_C else "royalblue"
        a.errorbar(p["xi"], p["tS"], yerr=max(p["drift"], 1e-6), fmt="s", ms=9,
                   color=col, capsize=3, zorder=3)
        a.annotate(rf"{p['beta']:.2f}", (p["xi"], p["tS"]), xytext=(6, 6),
                   textcoords="offset points", fontsize=8, color=col)
        v = e = None
        if p["beta"] in mc:
            v, e = mc[p["beta"]][0], mc[p["beta"]][1]
            good = mc[p["beta"]][2] < 2
        elif p["beta"] in mcL:
            _, v, e = sorted(mcL[p["beta"]])[-1]; good = False
        if v is not None:
            a.errorbar(p["xi"]*1.06, v, yerr=e, fmt="o", ms=6,
                       color="black" if good else "gray",
                       mfc="black" if good else "none", capsize=2, zorder=2)
    a.axhline(0, color="gray", lw=0.6)
    if xlog:
        a.set_xscale("log")
    a.set_xlabel(r"measured $\xi_1$")
    a.set_ylabel(r"$\tilde S_J$")
    a.set_title(("(b) log in measured $\\xi_1$" if xlog else
                 "(a) linear in measured $\\xi_1$"))
ax[0].plot([], [], "s", ms=9, color="royalblue", label=r"split, disordered")
ax[0].plot([], [], "s", ms=9, color="crimson", label=r"split, ordered")
ax[0].plot([], [], "o", ms=6, color="black", label=r"MC plateau ($\chi^2/\nu<2$)")
ax[0].plot([], [], "o", ms=6, color="gray", mfc="none", label="MC, uncertified")
ax[0].legend(fontsize=8, loc="lower right")

c = ax[2]
for p in pts:
    col = "crimson" if p["beta"] > BETA_C else "royalblue"
    c.plot(p["xe"], p["xi"], "s", ms=9, color=col)
    c.annotate(rf"{p['beta']:.2f}", (p["xe"], p["xi"]), xytext=(6, -3),
               textcoords="offset points", fontsize=8, color=col)
xr = np.array([0.4, 14])
c.plot(xr, xr, "-", color="royalblue", lw=1, alpha=0.7, label=r"$\xi_1=\xi_{\rm spin}$")
c.plot(xr, 2*xr, "--", color="crimson", lw=1, alpha=0.7, label=r"$\xi_1=2\xi_{\rm spin}$")
c.set_xscale("log"); c.set_yscale("log")
c.set_xlabel(r"exact $\xi_{\rm spin}$"); c.set_ylabel(r"measured $\xi_1$")
c.set_title("(c) calibration: ordered branch measures the domain-wall length")
c.legend(fontsize=9)

fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(OUT, dpi=160)
print("wrote", OUT)
print(f"{'beta':>6} {'phase':>10} {'chi1':>5} {'xi_meas':>9} {'xi_exact':>9} "
      f"{'ratio':>6} {'tSJ':>12} {'source':>12}")
for p in pts:
    ph = "ordered" if p["beta"] > BETA_C else "disordered"
    src = "scan" if p["chi_xi"] == p["chi"] else f"diag(chi{p['chi_xi']})"
    print(f"{p['beta']:6.2f} {ph:>10} {p['chi']:5d} {p['xi']:9.4f} {p['xe']:9.4f} "
          f"{p['xi']/p['xe']:6.3f} {p['tS']:+12.7f} {src:>12}")
