# plot_crit_stilde.py -- critical-point S~(chi) campaign figure (4 panels)
# reads data/rk_ising/ctmrg/tables/defect_ctmrg/critical_plain.csv
import csv, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "/ix/zdai/kangw/PEPS3EE/PEPS/clean/data/rk_ising/ctmrg/tables/defect_ctmrg/critical_plain.csv"
OUT = "/ix/zdai/kangw/PEPS3EE/PEPS/clean/data/rk_ising/ctmrg/figures/defect_ctmrg/crit_stilde_scan.png"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

rows = []
with open(CSV) as f:
    for r in csv.DictReader(f):
        try:
            rows.append(dict(
                var=r["variant"], chi=int(r["chi_req"]), chi4=int(r["chi4_fin"]),
                s2=float(r["s2"]), xi2=float(r["xi2"]), xi4=float(r["xi4"]),
                kA=float(r["kA"]), kY=float(r["kY"]), tSJ=float(r["tSJ"]),
                it2=int(r["it2"]), it4=int(r["it4"]),
                asym=float(r["seam_asym"])))
        except ValueError:
            continue

# dedupe bit-identical reruns (same node => same branch => same numbers)
seen, uniq = set(), []
for r in rows:
    key = (r["var"], r["chi"], round(r["tSJ"], 8))
    if key in seen: continue
    seen.add(key); uniq.append(r)
rows = uniq
# three-part quality gate:
#   (i) three seams exactly equal (defect environments consistent)
#  (ii) k=4 fast lock-in it4<300 (Y-window on-branch; wanderers: 900-2100)
# (iii) k=2 on-branch it2 < 20*xi2 (kappa_A outliers all show 97-220x)
def gold(r):
    return r["asym"] < 1e-3 and r["it4"] < 300 and r["it2"] < 20*r["xi2"]
SYM = [r for r in rows if r["asym"] < 1e-3]
BRK = [r for r in rows if r["asym"] >= 1e-3]
QUAR = [r for r in SYM if not gold(r)]      # symmetric but wandering: quarantine
# dedupe identical twins (same branch => identical numbers)
s_seen, SYMU = set(), []
for r in sorted((r for r in SYM if gold(r)), key=lambda r: r["chi"]):
    key = round(r["tSJ"], 6)
    if key in s_seen: continue
    s_seen.add(key); SYMU.append(r)
q_seen, QUARU = set(), []
for r in sorted(QUAR, key=lambda r: r["chi"]):
    key = round(r["tSJ"], 6)
    if key in q_seen: continue
    q_seen.add(key); QUARU.append(r)

fig, ax = plt.subplots(2, 2, figsize=(11.5, 9))
fig.suptitle(r"critical point $\beta_c$: $\tilde S_J(\chi)$ via CTMRG windows"
             "  (branch = three-seam symmetry gate)", fontsize=13)

# (a) tSJ vs ln chi, branch-colored ------------------------------------------
a = ax[0][0]
for r in BRK:
    a.semilogx(r["chi"], r["tSJ"], "x", ms=9, mew=2.2, color="crimson")
for r in QUARU:
    a.semilogx(r["chi"], r["tSJ"], "o", ms=9, mfc="none", mec="darkorange", mew=1.8)
for r in SYMU:
    a.semilogx(r["chi"], r["tSJ"], "o", ms=9, color="seagreen")
a.plot([], [], "x", ms=9, mew=2.2, color="crimson", label="broken (seams unequal)")
a.plot([], [], "o", ms=9, mfc="none", mec="darkorange", mew=1.8,
       label="quarantine (sym. seams, wandering lock-in)")
a.plot([], [], "o", ms=9, color="seagreen", label="gold (sym. + fast lock-in both sectors)")
a.axhline(0, color="gray", lw=0.7)
a.set_xlabel(r"$\chi_{\rm env}$"); a.set_ylabel(r"$\tilde S_J$")
a.set_title(r"(a) all samples vs $\ln\chi$, three-part quality gate")
a.legend(fontsize=8, loc="upper left")

# (b) symmetric branch: tSJ vs ln chi (semilogx), CFT ref via xi4(chi) law ---
b = ax[0][1]
xs = np.array([r["chi"] for r in SYMU]); ys = np.array([r["tSJ"] for r in SYMU])
o = np.argsort(xs)
b.semilogx(xs[o], ys[o], "o-", color="seagreen", ms=9, lw=1.2)
for r in SYMU:
    b.annotate(rf"$\chi{{=}}{r['chi']}$", (r["chi"], r["tSJ"]),
               textcoords="offset points", xytext=(5, 6), fontsize=7)
m, s = ys.mean(), ys.std()
b.axhspan(m - s, m + s, color="seagreen", alpha=0.12)
b.axhline(m, color="seagreen", lw=0.8, ls="--")
xr = np.linspace(8, 100, 60)
# CFT reference translated to chi: 0.103*ln xi4(chi), xi4 = 0.43 chi^0.87
b.semilogx(xr, 0.249 + 0.103*0.8697*np.log(xr/24.0), ":", color="k", lw=1.5,
           label=r"CFT ref $0.103\ln\xi_4 = 0.090\ln\chi$ (anchored $\chi{=}24$)")
b.set_xlabel(r"$\chi_{\rm env}$  (log axis)"); b.set_ylabel(r"$\tilde S_J$")
b.set_title(rf"(b) EXPLORATORY: apparent plateau ${m:.3f}\pm{s:.3f}$ — ALL points fail the $2s_2$ gate by 38–59%", fontsize=10, color="darkred")
b.legend(fontsize=9, loc="upper left")

# (c) kappa_A vs ln xi2: branch scatter, no clean CP slope -------------------
c = ax[1][0]
for r in BRK:
    c.plot(math.log(r["xi2"]), r["kA"], "x", ms=8, mew=2, color="crimson")
for r in QUARU:
    c.plot(math.log(r["xi2"]), r["kA"], "o", ms=8, mfc="none", mec="darkorange", mew=1.8)
for r in SYMU:
    c.plot(math.log(r["xi2"]), r["kA"], "o", ms=8, color="seagreen")
c.set_xlabel(r"$\ln \xi_2(\chi)$"); c.set_ylabel(r"$\kappa_A$")
c.set_title(r"(c) $\kappa_A$ vs $\ln\xi_2$: gold on one branch, wanderers hi/lo")

# (d) FES laws ---------------------------------------------------------------
d = ax[1][1]
ch = np.array([r["chi"] for r in rows]); x2 = np.array([r["xi2"] for r in rows])
x4 = np.array([r["xi4"] for r in rows])
ok4 = x4 < 1e3
d.loglog(ch, x2, "s", color="royalblue", ms=7, label=r"$\xi_2(\chi)$ [k=2, $c_{\rm eff}{=}1$]")
d.loglog(ch[ok4], x4[ok4], "d", color="darkorange", ms=7, label=r"$\xi_4(\chi)$ [k=4, $c_{\rm eff}{=}2$]")
cr = np.linspace(8, 100, 50)
d.loglog(cr, cr**1.3441, "-", color="royalblue", lw=1, alpha=0.6, label=r"$\chi^{1.344}$")
d.loglog(cr, 0.43*cr**0.8697, "-", color="darkorange", lw=1, alpha=0.6, label=r"$0.43\,\chi^{0.870}$")
d.set_xlabel(r"$\chi_{\rm env}$"); d.set_ylabel(r"$\xi(\chi)$")
d.set_title(r"(d) FES laws: upper envelopes follow $\chi^{6/(c(\sqrt{12/c}+1))}$")
d.legend(fontsize=8, loc="upper left")

fig.tight_layout(rect=[0, 0, 1, 0.965])
fig.savefig(OUT, dpi=160)
print("wrote", OUT)
print("GOLD points (chi, xi4, tSJ):")
for r in SYMU: print(f"  chi={r['chi']:3d} xi4={r['xi4']:7.2f} tSJ={r['tSJ']:+.4f}")
print(f"gold plateau: {m:.4f} +- {s:.4f}")
print("quarantined (sym seams, wandering):",
      [(r["chi"], round(r["tSJ"], 3)) for r in QUARU])
