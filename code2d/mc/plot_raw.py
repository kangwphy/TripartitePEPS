#!/usr/bin/env python3
# plot_raw.py -- the DIRECT plot the collapse figure obscured:
#   S~(beta, L) vs L, no rescaling, every measured point, with a coverage
#   table printed to stdout (which (beta, L) exist, how many pooled runs).
# Color: two fixed families (disordered blues, ordered reds; darker = closer
# to beta_c), critical curve in black; markers double-encode identity.
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

series = defaultdict(list)          # beta-string -> [(L, val, err, nruns)]
for (b, L), v in sorted(pools.items()):
    m, e = pool(v)
    series[b].append((L, m, e, len(v)))

print("=== MC data coverage (beta : L[nruns]) ===")
for b in sorted(series, key=lambda x: (x != "crit", x)):
    row = "  ".join(f"{L}[{n}]" for L, _, _, n in sorted(series[b]))
    print(f"beta={b:>5}: {row}")

STYLE = {  # fixed assignment: family + within-family darkness ~ closeness to beta_c
    "0.30": ("#8FB3DC", "o", "dis 0.30"),
    "0.42": ("#4C7BB8", "s", "dis 0.42"),
    "0.43": ("#1F4E8C", "^", "dis 0.43"),
    "0.45": ("#8C2F12", "^", "ord 0.45"),
    "0.46": ("#C4552D", "s", "ord 0.46"),
    "0.47": ("#DC8A5F", "v", "ord 0.47"),
    "0.60": ("#EFB08C", "D", "ord 0.60"),
    "crit": ("#111111", "*", r"$\beta_c$"),
}

fig, ax = plt.subplots(figsize=(8.6, 5.6), dpi=160)
ax.axhline(0, color="#999999", lw=0.8, ls=":")
order = ["0.30", "0.42", "0.43", "crit", "0.45", "0.46", "0.47", "0.60"]
for b in order:
    if b not in series: continue
    col, mk, lab = STYLE[b]
    pts = sorted(series[b])
    Ls = [p[0] for p in pts]; vs = [p[1] for p in pts]; es = [p[2] for p in pts]
    ax.errorbar(Ls, vs, yerr=es, fmt=mk + "-", color=col, ms=7 if b == "crit" else 5,
                lw=1.6 if b == "crit" else 1.0, capsize=2.5, alpha=0.95, label=lab,
                zorder=5 if b == "crit" else 3)
    ax.annotate(lab, (Ls[-1], vs[-1]), textcoords="offset points",
                xytext=(8, -2), fontsize=8, color=col)
ax.set_xscale("log", base=2)
ax.set_xticks([4, 8, 12, 16, 24, 32, 48, 64, 96, 128])
ax.set_xticklabels(["4", "8", "12", "16", "24", "32", "48", "64", "96", "128"])
ax.set_xlim(3.5, 145)
ax.set_xlabel(r"$L$")
ax.set_ylabel(r"$\tilde S(\beta, L)$")
ax.set_title(r"raw MC data, no rescaling: $\tilde S$ vs $L$ for every measured $(\beta, L)$",
             fontsize=11)
ax.grid(alpha=0.22, lw=0.5)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=8, loc="lower left", framealpha=0.9, ncol=2)
fig.tight_layout()
out = os.path.join(FIG, "tildeS_raw_all.png")
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
print("DONE_RAW")
