#!/usr/bin/env python3
# plot_mc_scaling.py -- the MC-campaign money figure.
# Panel (a): critical tildeS(L): pooled MC (error bars) vs bmps chi-series.
# Panel (b): gapped tildeS(L): MC flat lines vs the chi-artifact of bmps chi=128.
# Reads data2d/{mc_results,scanL_results,scale_plain_crit_cpu,scale_plain_0.30_cpu,scale_plain_0.60_cpu}.csv
# Writes figs2d/tildeS_mc_campaign.png
import os, csv, math
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = os.environ.get("DATA", os.path.join(os.path.dirname(__file__), "..", "..", "data2d"))
MC_TABLE = os.environ.get("MC_TABLE", os.path.join(os.path.dirname(__file__), "..", "..", "data/rk_ising/mc", "tables", "mc_results.csv"))
FIG = os.environ.get("FIG", os.path.join(os.path.dirname(__file__), "..", "..", "data/rk_ising/mc", "figures")); os.makedirs(FIG, exist_ok=True)

# palette: identity by method (fixed order), validated for CVD separation
C_MC   = "#2E5FA3"   # MC (blue)
C_BMPS = "#C4552D"   # boundary MPS (orange-red)
C_B030 = "#3A7D44"   # gapped beta=0.30 (green)
C_B060 = "#7B6FA0"   # gapped beta=0.60 (violet-gray)
GRAY   = "#8A8A8A"

# ---- pooled MC tS rows (plain junction mode only: tag empty or GL\d+ with optional R/F) ----
def mc_pool():
    crit, gap = defaultdict(list), defaultdict(list)
    with open(MC_TABLE) as f:
        for row in csv.reader(f):
            if len(row) < 7: continue
            L, bstr, tS, err = int(row[0]), row[1], float(row[4]), float(row[5])
            tag = row[7] if len(row) > 7 else ""
            if "S2C" in tag or "K4C" in tag: continue   # isolation gates, not tildeS
            (crit if bstr == "crit" else gap)[(L, bstr)].append((tS, err))
    def pool(d):
        out = {}
        for k, v in d.items():
            w = [1.0/e**2 for _, e in v]
            m = sum(t*wi for (t, _), wi in zip(v, w))/sum(w)
            ew = 1.0/math.sqrt(sum(w))
            if len(v) > 1:
                sc = math.sqrt(sum((t-m)**2 for t, _ in v)/(len(v)-1)/len(v))
                ew = max(ew, sc)
            out[k] = (m, ew, len(v))
        return out
    return pool(crit), pool(gap)

mc_crit, mc_gap = mc_pool()

# ---- bmps rows ----
def read_scan():
    rows = []
    with open(os.path.join(DATA, "scanL_results.csv")) as f:
        for row in csv.reader(f):
            if len(row) < 9: continue
            rows.append((int(row[0]), row[1], int(row[2]), float(row[8])))
    return rows
scan = read_scan()

def read_cert(fn):
    out = defaultdict(dict)   # L -> chi -> tS
    p = os.path.join(DATA, fn)
    if not os.path.exists(p): return out
    with open(p) as f:
        for r in csv.DictReader(f):
            out[int(float(r["Lx"]))][int(float(r["chi"]))] = float(r["tildeS"])
    return out
cert_crit = read_cert("scale_plain_crit_cpu.csv")
cert_030 = read_cert("scale_plain_0.30_cpu.csv")
cert_060 = read_cert("scale_plain_0.60_cpu.csv")

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=160)

# ================= panel (a): critical =================
Ls = sorted(L for (L, b) in mc_crit)
y  = [mc_crit[(L, "crit")][0] for L in Ls]
ye = [mc_crit[(L, "crit")][1] for L in Ls]
ax.errorbar(Ls, y, yerr=ye, fmt="o", ms=5, lw=1.6, capsize=3, color=C_MC,
            label="MC (SW + TI, pooled seeds)", zorder=5)

# bmps: certified small-L (largest chi per L) + new chi-series at L=12..20
bx, by = [], []
for L in sorted(cert_crit):
    chi = max(cert_crit[L]); bx.append(L); by.append(cert_crit[L][chi])
ax.plot(bx, by, "s", ms=6, mfc="none", mew=1.6, color=C_BMPS,
        label=r"bMPS, largest $\chi$ (biased $\uparrow$ at $L\geq 12$)", zorder=4)
# chi-series arrows at L=12(cert) and L=14,16,18,20(scan)
series = defaultdict(list)
for (L, b, chi, tS) in scan:
    if b == "crit": series[L].append((chi, tS))
for L, chi_d in sorted(cert_crit.items()):
    if L >= 12:
        for chi, tS in sorted(chi_d.items()): series[L].append((chi, tS))
for L, pts in sorted(series.items()):
    pts = sorted(set(pts))
    if len(pts) < 2: continue
    xs = [L]*len(pts); ys = [t for _, t in pts]
    ax.plot(xs, ys, "-", lw=0.9, color=C_BMPS, alpha=0.55, zorder=3)
    ax.plot(xs, ys, "v", ms=3.5, color=C_BMPS, alpha=0.55, zorder=3)
    ax.annotate("", xy=(L, min(ys)-0.012), xytext=(L, min(ys)),
                arrowprops=dict(arrowstyle="->", color=C_BMPS, alpha=0.8))
ax.axhline(0, color=GRAY, lw=0.8, ls=":")
ax.set_xlabel(r"$L$"); ax.set_ylabel(r"$\tilde S$")
ax.set_title(r"critical $\beta=\beta_c$: junction term peaks at $L\!\approx\!12$, then quasi-linear descent"
             "\n" r"(bMPS $\chi$-series still descending toward MC; arrows = direction of $\chi\to\infty$)",
             fontsize=9)
ax.legend(fontsize=8, loc="lower left", framealpha=0.9)
# direct labels for the chi-series: chi_min sits at the top value, chi_max at the bottom
for L in (16,):
    pts = sorted(set(series.get(L, [])))          # ascending chi; tS descends with chi
    if len(pts) >= 2:
        ax.annotate(rf"$\chi={pts[0][0]}$", (L, pts[0][1]), textcoords="offset points",
                    xytext=(7, 3), fontsize=7, color=C_BMPS)
        ax.annotate(rf"$\chi={pts[-1][0]}$", (L, pts[-1][1]), textcoords="offset points",
                    xytext=(7, -9), fontsize=7, color=C_BMPS)

# ================= panel (b): gapped =================
for cert, mcb, color, lab in ((cert_030, "0.30", C_B030, r"$\beta=0.30$"),
                              (cert_060, "0.60", C_B060, r"$\beta=0.60$")):
    bx = sorted(cert); by = [cert[L][max(cert[L])] for L in bx]
    ax2.plot(bx, by, "s--", ms=5, mfc="none", lw=1.0, color=color,
             label=lab + r" bMPS certified ($L\leq 12$)")
    Ls = sorted(L for (L, b) in mc_gap if b == mcb)
    if Ls:
        y  = [mc_gap[(L, mcb)][0] for L in Ls]
        ye = [mc_gap[(L, mcb)][1] for L in Ls]
        ax2.errorbar(Ls, y, yerr=ye, fmt="o", ms=5, lw=1.4, capsize=3, color=color,
                     label=lab + " MC")
# the chi=128 artifact curve
art = sorted((L, tS) for (L, b, chi, tS) in scan if b == "0.30" and chi == 128)
if art:
    ax2.plot([L for L, _ in art], [t for _, t in art], "x-", ms=5, lw=0.9,
             color=GRAY, alpha=0.8, label=r"bMPS $\chi=128$ (artifact, withdrawn)")
ax2.axhline(0, color=GRAY, lw=0.8, ls=":")
ax2.set_xlabel(r"$L$"); ax2.set_ylabel(r"$\tilde S$")
ax2.set_title("gapped: junction term flat/vanishing (MC);\n"
              r"bMPS $\chi=128$ growth at $L\geq16$ exposed as truncation artifact", fontsize=9)
ax2.legend(fontsize=8, loc="upper left", framealpha=0.9)

for a in (ax, ax2):
    a.grid(alpha=0.25, lw=0.5)
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
out = os.path.join(FIG, "tildeS_mc_campaign.png")
fig.savefig(out, bbox_inches="tight")
print("wrote", out)

# ---- console table of pooled numbers for the report ----
print("\npooled MC crit:")
for L in sorted(L for (L, b) in mc_crit):
    m, e, n = mc_crit[(L, "crit")]
    print(f"  L={L:<3d} tS={m:+.4f} +- {e:.4f}  ({n} runs)")
print("DONE_PLOT")
