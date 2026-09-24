#!/usr/bin/env python3
# plot_semilogx.py -- semilogx view: tildeS (LINEAR y) vs L (log x).
# A c*ln(L) regime would be a straight line here; the old 0.103*lnL
# conjecture is drawn as a guide through the critical rise for reference.
import os, csv, math
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
DATA = os.environ.get("DATA", os.path.join(ROOT, "data/rk_ising/mc"))
MC_FILE = os.environ.get("MC_FILE", os.path.join(DATA, "mc_results.csv"))
FIG = os.environ.get("FIG", os.path.join(ROOT, "data/rk_ising/mc", "figures")); os.makedirs(FIG, exist_ok=True)

pools = defaultdict(list)
with open(MC_FILE) as f:
    for row in csv.reader(f):
        if len(row) < 7: continue
        tag = row[7] if len(row) > 7 else ""
        if "S2C" in tag or "K4C" in tag: continue
        pools[(row[1], int(row[0]))].append((float(row[4]), float(row[5])))

def pool(v):
    w = [1/e**2 for _, e in v]
    m = sum(t*wi for (t, _), wi in zip(v, w))/sum(w)
    ew = 1/math.sqrt(sum(w))
    if len(v) > 1:
        sc = math.sqrt(sum((t-m)**2 for t, _ in v)/(len(v)-1)/len(v))
        ew = max(ew, sc)
    return m, ew

series = defaultdict(list)
for (b, L), v in sorted(pools.items()):
    m, e = pool(v)
    series[b].append((L, m, e))

STYLE = {
    "0.30": ("#8FB3DC", "o", "dis 0.30"),
    "0.42": ("#4C7BB8", "s", "dis 0.42"),
    "0.43": ("#1F4E8C", "^", "dis 0.43"),
    "0.45": ("#8C2F12", "^", "ord 0.45"),
    "0.46": ("#C4552D", "s", "ord 0.46"),
    "0.47": ("#DC8A5F", "v", "ord 0.47"),
    "0.60": ("#EFB08C", "D", "ord 0.60"),
    "crit": ("#111111", "*", r"$\beta_c$"),
}
ORDER = ["0.30", "0.42", "0.43", "crit", "0.45", "0.46", "0.47", "0.60"]

fig, ax = plt.subplots(figsize=(9.2, 5.8), dpi=160)
ax.axhline(0, color="#999999", lw=0.8, ls=":")
for b in ORDER:
    if b not in series: continue
    col, mk, lab = STYLE[b]
    pts = sorted(series[b])
    Ls = [p[0] for p in pts]; vs = [p[1] for p in pts]; es = [p[2] for p in pts]
    ax.errorbar(Ls, vs, yerr=es, fmt=mk + "-", color=col,
                ms=8 if b == "crit" else 5, lw=1.8 if b == "crit" else 1.0,
                capsize=2.5, alpha=0.95, label=lab, zorder=5 if b == "crit" else 3)
# the old conjecture 0.103*lnL as a guide through the small-L rise
Lg = np.geomspace(6, 20, 40)
ax.plot(Lg, 0.103*np.log(Lg) - 0.103*np.log(8) + 0.145, "--", color="#777777", lw=1.2)
ax.annotate(r"$0.103\,\ln L$ (old conjecture)", (9.5, 0.20), fontsize=9, color="#777777")
ax.set_xscale("log")
ax.set_xticks([4, 8, 12, 16, 24, 32, 48, 64, 96, 128])
ax.set_xticklabels(["4", "8", "12", "16", "24", "32", "48", "64", "96", "128"])
ax.minorticks_off()
ax.set_xlim(3.5, 145)
ax.set_xlabel(r"$L$  (log scale)")
ax.set_ylabel(r"$\tilde S(\beta, L)$   (linear)")
ax.set_title(r"semilogx: a $c\ln L$ regime would be a straight line here", fontsize=11)
ax.grid(alpha=0.22, lw=0.5)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=8, loc="lower left", ncol=2, framealpha=0.9)
fig.tight_layout()
out = os.path.join(FIG, "tildeS_semilogx.png")
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
print("DONE_SEMILOGX")
