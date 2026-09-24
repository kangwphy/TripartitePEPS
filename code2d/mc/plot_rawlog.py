#!/usr/bin/env python3
# plot_rawlog.py -- log-axis views of the raw MC tildeS(beta, L):
#  (a) symlog y (linear inside |S|<0.02, log outside): both signs visible;
#  (b) log-log |S~| vs L, sign encoded (filled = positive, open = negative),
#      with an L^{0.39} guide on the critical descent (junction power law).
import os, csv, math
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.6), dpi=160)

# ---- (a) symlog y ----
ax1.axhline(0, color="#999999", lw=0.8, ls=":")
for b in ORDER:
    if b not in series: continue
    col, mk, lab = STYLE[b]
    pts = sorted(series[b])
    Ls = [p[0] for p in pts]; vs = [p[1] for p in pts]; es = [p[2] for p in pts]
    ax1.errorbar(Ls, vs, yerr=es, fmt=mk + "-", color=col,
                 ms=7 if b == "crit" else 5, lw=1.6 if b == "crit" else 1.0,
                 capsize=2.5, alpha=0.95, label=lab,
                 zorder=5 if b == "crit" else 3)
ax1.set_xscale("log", base=2)
ax1.set_yscale("symlog", linthresh=0.02, linscale=0.6)
ax1.set_xticks([4, 8, 12, 16, 24, 32, 48, 64, 96, 128]); ax1.set_xticklabels(["4","8","12","16","24","32","48","64","96","128"])
ax1.set_xlim(3.5, 145)
ax1.set_yticks([-0.2, -0.1, -0.05, -0.02, 0, 0.02, 0.05, 0.1, 0.2])
ax1.set_yticklabels(["-0.2","-0.1","-0.05","-0.02","0","0.02","0.05","0.1","0.2"])
ax1.set_xlabel(r"$L$"); ax1.set_ylabel(r"$\tilde S(\beta, L)$")
ax1.set_title(r"symlog $y$ (linear inside $|\tilde S|<0.02$)", fontsize=10)
ax1.grid(alpha=0.22, lw=0.5); ax1.spines[["top", "right"]].set_visible(False)
ax1.legend(fontsize=8, loc="lower left", ncol=2, framealpha=0.9)

# ---- (b) log-log |S| with sign markers ----
for b in ORDER:
    if b not in series: continue
    col, mk, lab = STYLE[b]
    pts = sorted(series[b])
    for L, v, e in pts:
        if abs(v) < 1e-4: continue
        filled = v > 0
        ax2.errorbar([L], [abs(v)], yerr=[min(e, abs(v)*0.9)], fmt=mk, color=col,
                     ms=8 if b == "crit" else 6,
                     mfc=col if filled else "white", mew=1.3, capsize=2.2,
                     alpha=0.95, zorder=5 if b == "crit" else 3)
    Ls = [p[0] for p in pts if abs(p[1]) > 1e-4]
    Vs = [abs(p[1]) for p in pts if abs(p[1]) > 1e-4]
    ax2.plot(Ls, Vs, "-", color=col, lw=1.5 if b == "crit" else 0.8, alpha=0.5,
             label=lab)
# L^0.39 guide through the critical negative tail (L=48..64)
import numpy as np
Lg = np.array([30.0, 140.0])
ax2.plot(Lg, 0.256*(Lg/64.0)**0.39, "--", color="#555555", lw=1.2)
ax2.annotate(r"$\propto L^{0.39}$", (70, 0.27), fontsize=10, color="#555555")
ax2.set_xscale("log", base=2); ax2.set_yscale("log")
ax2.set_xticks([4, 8, 12, 16, 24, 32, 48, 64, 96, 128]); ax2.set_xticklabels(["4","8","12","16","24","32","48","64","96","128"])
ax2.set_xlim(3.5, 145)
ax2.set_xlabel(r"$L$"); ax2.set_ylabel(r"$|\tilde S(\beta, L)|$")
ax2.set_title(r"log-log $|\tilde S|$: filled = positive, open = negative", fontsize=10)
ax2.grid(alpha=0.22, lw=0.5, which="both"); ax2.spines[["top", "right"]].set_visible(False)
ax2.legend(fontsize=8, loc="lower left", ncol=2, framealpha=0.9)

fig.tight_layout()
out = os.path.join(FIG, "tildeS_raw_log.png")
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
print("DONE_RAWLOG")
