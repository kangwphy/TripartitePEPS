# check_split_vs_mc.py -- is the split-replica (product-environment) prototype
# consistent with Monte Carlo?
#   (a) off-critical comparison, disordered vs ordered side;
#   (b) ordered-side MC convergence audit (the reference itself);
#   (c) critical: split has no plateau -- map it onto the MC critical curve.
import os, csv
import math
import re
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/ix/zdai/kangw/PEPS3EE/PEPS/clean"
SPLIT_SCAN = f"{R}/data/rk_ising/ctmrg/tables/beta_chi_scan_split/summary.csv"
SPLIT_CRIT = f"{R}/data/rk_ising/ctmrg/tables/split_replica/summary.csv"
MC_MAIN = f"{R}/data/rk_ising/mc/campaigns/sw_components_100k_16chains/pooled_components.csv"
MC_NEAR = f"{R}/data/rk_ising/mc/campaigns/sw_components_near_critical_100k_16chains/direct_tS.csv"
MC_PLAT = f"{R}/data/rk_ising/mc/campaigns/sw_components_near_critical_100k_16chains/plateau_summary.csv"
OUT = f"{R}/data/rk_ising/ctmrg/figures/split_replica/split_vs_mc.png"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
BETA_C = 0.5*np.log(1+np.sqrt(2))

# ---- split-replica: converged off-critical values, and the critical sequence --
split_off = {}
for r in csv.DictReader(open(SPLIT_SCAN)):
    b, chi1 = float(r["beta"]), int(r["chi1"])
    # xi1 carries a sentinel on the ordered side: the leading transfer eigenvalue
    # is a degenerate Z2 pair there, so the naive estimator returns 1e15 or -inf.
    # Those rows must not be printed as a correlation length.  The degeneracy-
    # aware value lives in logs2d/measurexi_*.out and is loaded below, exactly as
    # plot_split_vs_measured_xi.py already does.
    try:
        xi_raw = float(r["xi1"])
    except (ValueError, TypeError):
        xi_raw = float("inf")
    if not math.isfinite(xi_raw) or xi_raw > 1e6:
        xi_raw = float("nan")
    split_off.setdefault(b, {})[chi1] = (float(r["tSJ"]), xi_raw)
split_crit = []
for r in csv.DictReader(open(SPLIT_CRIT)):
    if abs(float(r["beta"]) - BETA_C) < 1e-6:
        split_crit.append((int(r["chi1"]), float(r["xi1"]), float(r["tSJ"])))
split_crit.sort()

# ---- MC: tS vs L (main campaign for 0.30/0.60/crit, near-crit for the rest) ---
mc_L = {}
for r in csv.DictReader(open(MC_MAIN)):
    if r["observable"] != "tS":
        continue
    b = r["beta"]
    key = "crit" if b == "crit" else float(b)
    mc_L.setdefault(key, []).append((int(r["L"]), float(r["value"]), float(r["error"])))
for r in csv.DictReader(open(MC_NEAR)):
    mc_L.setdefault(float(r["beta"]), []).append(
        (int(r["L"]), float(r["direct_tS"]), float(r["error"])))
for k in mc_L:
    mc_L[k].sort()
plat = {}
for r in csv.DictReader(open(MC_PLAT)):
    plat[float(r["beta"])] = (r["phase"], float(r["direct_plateau"]),
                              float(r["direct_error"]), float(r["direct_chi2_dof"]),
                              float(r["reconstructed_plateau"]),
                              float(r["reconstructed_error"]),
                              float(r["reconstructed_chi2_dof"]))

# ---- critical MC curve + its large-L log fit ---------------------------------
cl = np.array([p[0] for p in mc_L["crit"]]); cv = np.array([p[1] for p in mc_L["crit"]])
ce = np.array([p[2] for p in mc_L["crit"]])
mm = cl >= 48
w = 1/ce[mm]**2; X = np.log(cl[mm]); Y = cv[mm]
A = np.array([[w.sum(), (w*X).sum()], [(w*X).sum(), (w*X*X).sum()]])
bvec = np.array([(w*Y).sum(), (w*X*Y).sum()])
b0, cslope = np.linalg.solve(A, bvec)

fig, ax = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("split-replica (product-environment) prototype vs Monte Carlo", fontsize=13)

# ---- (a) off-critical -------------------------------------------------------
a = ax[0]
bs = sorted(split_off)
for i, b in enumerate(bs):
    chis = sorted(split_off[b]); v = split_off[b][chis[-1]][0]
    ordered = b > BETA_C
    a.plot(b, v, "s", ms=10, color="crimson" if ordered else "royalblue",
           label="split (iPEPS)" if i == 0 else None)
    if b in plat:
        ph, dp, de, dchi, rp, re_, rchi = plat[b]
        a.errorbar(b-0.004, dp, yerr=de, fmt="o", ms=7, color="black",
                   label="MC plateau (direct)" if i == 0 else None)
        a.errorbar(b+0.004, rp, yerr=re_, fmt="^", ms=7, color="gray",
                   label="MC plateau (reconstructed)" if i == 0 else None)
        a.annotate(rf"$\chi^2/\nu$={dchi:.0f}/{rchi:.0f}", (b, min(dp, rp)),
                   xytext=(0, -16), textcoords="offset points", fontsize=7,
                   ha="center", color="dimgray")
    elif b in mc_L:                       # no plateau fit: show the largest L
        L, v2, e2 = mc_L[b][-1]
        a.errorbar(b-0.004, v2, yerr=e2, fmt="o", ms=7, color="black")
        a.annotate(rf"MC $L$={L}", (b-0.004, v2), xytext=(0, 9),
                   textcoords="offset points", fontsize=7, ha="center")
a.axvline(BETA_C, color="green", ls=":", lw=1.2)
a.text(BETA_C+0.004, a.get_ylim()[1]*0.85, r"$\beta_c$", color="green", fontsize=9)
a.axhline(0, color="gray", lw=0.6)
a.set_xlabel(r"$\beta$"); a.set_ylabel(r"$\tilde S$")
a.set_title("(a) off-critical: agree for $\\beta<\\beta_c$, differ for $\\beta>\\beta_c$")
a.legend(fontsize=8)

# ---- (b) ordered-side MC audit ----------------------------------------------
b_ = ax[1]
for bb in sorted(x for x in mc_L if x != "crit" and x > BETA_C):
    arr = np.array(mc_L[bb])
    b_.errorbar(arr[:, 0], arr[:, 1], yerr=arr[:, 2], fmt="o-", ms=4, lw=1,
                label=rf"MC $\beta={bb:.2f}$")
    if bb in split_off:
        v = split_off[bb][sorted(split_off[bb])[-1]][0]
        b_.axhline(v, ls="--", lw=1, alpha=0.6,
                   color=b_.lines[-1].get_color())
b_.axhline(0, color="gray", lw=0.6)
b_.axhline(0.0, color="purple", ls=":", lw=1.4)
b_.text(5, 0.03, r"deep-ordered (GHZ$_2$) limit: $\tilde S=0$"
        "\n(sector counting cancels in the combination)",
        color="purple", fontsize=8)
b_.set_xscale("log"); b_.set_xlabel(r"$L$"); b_.set_ylabel(r"$\tilde S$")
b_.set_title("(b) ordered side: MC drifts away from its own small-$L$ values")
b_.legend(fontsize=7, ncol=2)

# ---- (c) critical: map split onto the MC critical curve ----------------------
c_ = ax[2]
arr = np.array(mc_L["crit"])
c_.errorbar(arr[:, 0], arr[:, 1], yerr=arr[:, 2], fmt="o", ms=5, color="black",
            label="MC critical curve")
xx = np.linspace(np.log(8), np.log(4000), 50)
c_.plot(np.exp(xx), b0 + cslope*xx, "-", color="gray", lw=1,
        label=rf"MC log fit ($c={cslope:.3f}$), extrapolated")
sc = np.array(split_crit)
Leq = np.exp((b0 - sc[:, 2])/(-cslope))
c_.plot(Leq, sc[:, 2], "s", ms=7, color="crimson",
        label=r"split, placed at $L_{\rm eff}$")
# degeneracy-aware xi from the standalone diagnostic, used wherever the scan
# stored a sentinel (ordered phase).
XI_DIAG = {}
for _path in glob.glob(f"{R}/logs2d/measurexi_*.out"):
    for _line in open(_path):
        _m = re.search(r"SUMMARY-XI beta=([\d.]+) chi1=\s*(\d+).*?xi_conn=\s*([\d.]+)", _line)
        if _m:
            XI_DIAG[(round(float(_m.group(1)), 4), int(_m.group(2)))] = float(_m.group(3))

for chi1, xi1, ts in split_crit:
    if chi1 in (6, 12, 24, 32):
        Le = np.exp((b0 - ts)/(-cslope))
        c_.annotate(rf"$\chi_1$={chi1}", (Le, ts), xytext=(4, -10),
                    textcoords="offset points", fontsize=8, color="crimson")
c_.set_xscale("log"); c_.set_xlabel(r"$L$  (MC)   /   $L_{\rm eff}$  (split)")
c_.set_ylabel(r"$\tilde S$")
ratio = Leq/sc[:, 0]
sel = sc[:, 0] >= 6
c_.set_title(rf"(c) critical: split has no plateau; $L_{{\rm eff}}\simeq"
             rf"{ratio[sel].mean():.1f}\,\chi_1$ (±{ratio[sel].std():.1f})")
c_.legend(fontsize=8)

fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUT, dpi=160)
print("wrote", OUT)

print(f"\nMC critical log fit (L>=48): tS = {b0:.4f} {cslope:+.4f} ln L")
print("\n--- off-critical comparison ---")
for b in bs:
    chis = sorted(split_off[b]); v, xi = split_off[b][chis[-1]]
    if not math.isfinite(xi):
        xi = XI_DIAG.get((round(b, 4), chis[-1]), float("nan"))
        xitag = "xi_conn(diag)" if math.isfinite(xi) else "xi1"
    else:
        xitag = "xi1"
    xistr = f"{xi:.3g}" if math.isfinite(xi) else "degenerate (Z2 pair)"
    line = f"beta={b:.2f} split(chi1={chis[-1]})={v:+.6f}  {xitag}={xistr}"
    if b in plat:
        ph, dp, de, dchi, rp, re_, rchi = plat[b]
        line += (f"\n           MC[{ph}] direct={dp:+.5f}({de*1e5:.0f}) chi2/dof={dchi:.1f}"
                 f" | recon={rp:+.5f}({re_*1e5:.0f}) chi2/dof={rchi:.1f}"
                 f" | split-direct={v-dp:+.5f} ({abs(v-dp)/de:.1f} sigma)")
    elif b in mc_L:
        L, v2, e2 = mc_L[b][-1]
        line += (f"\n           MC largest L={L}: {v2:+.5f}({e2*1e5:.0f})"
                 f" | split-MC={v-v2:+.5f} ({abs(v-v2)/e2:.1f} sigma)")
    print(line)

print("\n--- critical: split placed on the MC curve ---")
print(" chi1   xi1        tS_split     L_eff    L_eff/chi1  L_eff/sqrt(xi1)")
for (chi1, xi1, ts), Le in zip(split_crit, Leq):
    print(f" {chi1:4d}  {xi1:9.1f}  {ts:+.6f}  {Le:8.1f}   {Le/chi1:6.2f}"
          f"      {Le/np.sqrt(xi1) if xi1 > 0 else float('nan'):6.2f}")
print(f"\nL_eff/chi1 over chi1>=6: mean={ratio[sel].mean():.2f} std={ratio[sel].std():.2f}"
      f"  (spread {100*ratio[sel].std()/ratio[sel].mean():.0f}%)")
