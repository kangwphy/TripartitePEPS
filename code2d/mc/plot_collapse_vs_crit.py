# plot_collapse_vs_crit.py -- proper scaling test: subtract the MEASURED critical
# curve (not a log fit) and check whether the remainder is a function of L/xi:
#     Delta(L,beta) = S~(L,beta) - S~_crit(L)   vs   L/xi_spin(beta)
# S~_crit(L) is interpolated in ln L over the measured critical sizes (4..96);
# points needing extrapolation beyond L=96 are drawn hollow.
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
       "collapse_vs_crit.png")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
BETA_C = 0.5*np.log(1+np.sqrt(2))

cl, cs = [], []
for r in csv.DictReader(open(CR)):
    if r["beta"] == "crit" and r["observable"] == "tS":
        cl.append(int(r["L"])); cs.append(float(r["value"]))
o = np.argsort(cl); cl = np.array(cl)[o]; cs = np.array(cs)[o]
LMAX = cl[-1]
# log-slope of the last decade, used only to extend beyond L=96
sl = (cs[-1]-cs[-3])/(np.log(cl[-1])-np.log(cl[-3]))
def scrit(L):
    if L <= LMAX:
        return float(np.interp(np.log(L), np.log(cl), cs))
    return float(cs[-1] + sl*(np.log(L)-np.log(LMAX)))

d = {}
for r in csv.DictReader(open(NC)):
    d.setdefault(float(r["beta"]), []).append(
        (int(r["L"]), float(r["L_over_xi"]), float(r["direct_tS"]), float(r["error"])))
betas = sorted(d)
cmap = plt.cm.coolwarm(np.linspace(0, 1, len(betas)))
COL = {b: cmap[i] for i, b in enumerate(betas)}

fig, ax = plt.subplots(1, 3, figsize=(16.5, 5))
fig.suptitle(r"scaling test without assuming a log: "
             r"$\Delta=\tilde S(L,\beta)-\tilde S_{\rm crit}(L)$ vs $L/\xi$", fontsize=12)

for k, (sub, ttl) in enumerate(((None, "(a) all couplings"),
                                (lambda b: b < BETA_C, r"(b) disordered ($\beta<\beta_c$)"),
                                (lambda b: b > BETA_C, r"(c) ordered ($\beta>\beta_c$)"))):
    a = ax[k]
    for bb in betas:
        if sub is not None and not sub(bb):
            continue
        rows = sorted(d[bb])
        x = np.array([r[1] for r in rows]); L = np.array([r[0] for r in rows])
        y = np.array([r[2] - scrit(r[0]) for r in rows])
        e = np.array([r[3] for r in rows])
        ext = L > LMAX
        a.errorbar(x[~ext], y[~ext], yerr=e[~ext], fmt="o-", ms=4, lw=1,
                   color=COL[bb], label=rf"$\beta={bb:.2f}$")
        if ext.any():
            a.plot(x[ext], y[ext], "o", ms=4, mfc="none", color=COL[bb])
    a.axhline(0, color="gray", lw=0.6); a.axvline(1, color="gray", ls=":", lw=0.8)
    a.set_xscale("log"); a.set_xlabel(r"$L/\xi_{\rm spin}$")
    a.set_ylabel(r"$\tilde S-\tilde S_{\rm crit}(L)$")
    a.set_title(ttl); a.legend(fontsize=7, ncol=2)

fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(OUT, dpi=160)
print("wrote", OUT)
print(f"critical curve interpolated over L=4..{LMAX}; extension slope {sl:+.3f}")
print("\nspread of Delta across beta in L/xi bins (hollow/extrapolated excluded):")
pts = [(lx, v - scrit(L), bb) for bb in betas for L, lx, v, e in d[bb] if L <= LMAX]
for lo, hi in ((0.5, 1), (1, 2), (2, 4), (4, 8), (8, 16), (16, 40)):
    sel = [p for p in pts if lo <= p[0] < hi]
    if len(sel) > 2:
        v = np.array([p[1] for p in sel])
        print(f"   L/xi in [{lo},{hi}): n={len(v):3d} mean={v.mean():+.4f} "
              f"std={v.std():.4f} range={v.max()-v.min():.4f}")
