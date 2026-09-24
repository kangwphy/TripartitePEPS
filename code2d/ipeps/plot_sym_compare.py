# plot_sym_compare.py -- symmetrized-CTMRG campaign vs the original chi-scan.
# 4 panels: (a) tSJ overlay, (b) lock-in speed, (c) 2s2-gate residual wall,
# (d) three-seam asymmetry. Post-review framing: methodology cured, accuracy
# wall (factorization gate ~50%) remains -> all tSJ values exploratory.
import csv, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OLD = "/ix/zdai/kangw/PEPS3EE/PEPS/clean/data/rk_ising/ctmrg/tables/defect_ctmrg/critical_plain.csv"
SYM = "/ix/zdai/kangw/PEPS3EE/PEPS/clean/data/rk_ising/ctmrg/tables/defect_ctmrg/critical_symmetrized.csv"
OUT = "/ix/zdai/kangw/PEPS3EE/PEPS/clean/data/rk_ising/ctmrg/figures/defect_ctmrg/crit_stilde_sym.png"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

def readcsv(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                rows.append({k: (v if k in ("variant", "jobid") else float(v))
                             for k, v in r.items()})
            except ValueError:
                continue
    return rows

old = readcsv(OLD)
sym = readcsv(SYM)
# old-campaign classes (same gates as plot_crit_stilde.py)
for r in old:
    r["cls"] = ("brk" if r["seam_asym"] >= 1e-3 else
                ("gold" if r["it4"] < 300 and r["it2"] < 20*r["xi2"] else "quar"))
# dedupe old identical reruns
seen, oldu = set(), []
for r in old:
    key = (round(r["tSJ"], 8),)
    if key in seen: continue
    seen.add(key); oldu.append(r)
old = oldu
S4  = [r for r in sym if r["variant"] == "sym"]
FL  = [r for r in sym if r["variant"] == "flip"]

fig, ax = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle(r"symmetrized defect CTMRG ($S_4$ / $S_4{\times}Z_2$ twirl) vs original scan at $\beta_c$"
             "\nEXPLORATORY throughout: all runs fail the $2s_2$ factorization gate by ~40-60%",
             fontsize=12, color="darkred")

# (a) tSJ overlay --------------------------------------------------------------
a = ax[0][0]
for r in old:
    st = {"brk": ("x", "lightcoral"), "quar": ("o", "navajowhite"),
          "gold": ("o", "lightgreen")}[r["cls"]]
    a.semilogx(r["chi_req"], r["tSJ"], st[0], ms=6,
               color=st[1], mew=1.5, mfc=(st[1] if st[0] == "o" else None))
for r in S4:
    a.semilogx(r["chi"], r["tSJ"], "s", ms=9, color="royalblue", mfc="none", mew=2)
for r in FL:
    a.semilogx(r["chi"], r["tSJ"], "D", ms=8, color="purple")
a.axhspan(0.23, 0.27, color="seagreen", alpha=0.10)
a.plot([], [], "x", color="lightcoral", label="old: broken")
a.plot([], [], "o", color="navajowhite", label="old: quarantine")
a.plot([], [], "o", color="lightgreen", label="old: gold")
a.plot([], [], "s", color="royalblue", mfc="none", mew=2, label=r"$S_4$ twirl")
a.plot([], [], "D", color="purple", label=r"$S_4{\times}Z_2$ twirl")
a.set_xlabel(r"$\chi$"); a.set_ylabel(r"$\tilde S_J$")
a.set_title("(a) branch lottery (old) vs deterministic twirl runs")
a.legend(fontsize=8, loc="lower left", ncol=2)

# (b) lock-in speed ------------------------------------------------------------
b = ax[0][1]
for r in old:
    c = {"brk": "lightcoral", "quar": "navajowhite", "gold": "lightgreen"}[r["cls"]]
    b.loglog(r["chi_req"], max(r["it4"], 1), "o", ms=5, color=c)
for r in S4:
    b.loglog(r["chi"], r["it4"], "s", ms=9, color="royalblue", mfc="none", mew=2)
for r in FL:
    b.loglog(r["chi"], r["it4"], "D", ms=8, color="purple")
b.axhline(300, color="gray", ls="--", lw=0.8)
b.text(9, 330, "it$_4$=300 fingerprint", fontsize=8, color="gray")
b.set_xlabel(r"$\chi$"); b.set_ylabel(r"$k{=}4$ lock-in iterations")
b.set_title("(b) wandering (900-50000) vs fast lock-in (57-222)")

# (c) the accuracy wall: gate residual vs xi4 ----------------------------------
c = ax[1][0]
for r in old:
    if r["xi4"] < 1e3 and r["gate_rel"] == r["gate_rel"]:
        col = {"brk": "lightcoral", "quar": "navajowhite", "gold": "lightgreen"}[r["cls"]]
        c.loglog(r["xi4"], abs(r["gate_rel"]), "o", ms=5, color=col)
for r, mk, col in [(r, "s", "royalblue") for r in S4] + [(r, "D", "purple") for r in FL]:
    if r["xi4"] < 1e3:
        c.loglog(r["xi4"], abs(r["gate_rel"]), mk, ms=8, color=col,
                 mfc=("none" if mk == "s" else col), mew=2)
xr = np.linspace(4, 200, 60)
c.loglog(xr, 0.83*xr**-0.6, ":", color="gray", lw=1.0,
         label=r"guide $0.83\,\xi_4^{-0.6}$ (NOT a fit; data tail $\sim\xi_4^{-0.3}$)")
c.axhline(0.10, color="darkred", ls="--", lw=1)
c.text(4.5, 0.083, "10% target: data-tail extrapolation needs $\\xi_4\\sim4{\\times}10^3$ ($\\chi\\sim10^4$)",
       fontsize=8, color="darkred")
c.set_xlabel(r"$\xi_4(\chi)$"); c.set_ylabel(r"$|t_{\rm seam}-2s_2|/2s_2$")
c.set_title("(c) THE WALL: factorization-gate residual (review's point)")
c.legend(fontsize=8)

# (d) three-seam asymmetry -----------------------------------------------------
d = ax[1][1]
for r in old:
    col = {"brk": "lightcoral", "quar": "navajowhite", "gold": "lightgreen"}[r["cls"]]
    d.semilogy(r["chi_req"], max(r["seam_asym"], 1e-16), "o", ms=5, color=col)
for r in S4:
    d.semilogy(r["chi"], max(r["asym"], 1e-16), "s", ms=9, color="royalblue",
               mfc="none", mew=2)
for r in FL:
    d.semilogy(r["chi"], max(r["asym"], 1e-16), "D", ms=8, color="purple")
d.set_xlabel(r"$\chi$"); d.set_ylabel("three-seam asymmetry")
d.set_title("(d) replica symmetry: lottery (up to 0.8) -> machine precision")

fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT, dpi=160)
print("wrote", OUT)
print("sym-campaign points:")
for r in sorted(sym, key=lambda r: (r["chi"], r["variant"])):
    print(f"  chi={int(r['chi']):3d} {r['variant']:4s} tSJ={r['tSJ']:+.4f} "
          f"it4={int(r['it4']):5d} asym={r['asym']:.1e} gate_rel={r['gate_rel']:+.3f} xi4={r['xi4']:.2f}")
