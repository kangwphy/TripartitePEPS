#!/usr/bin/env python3
# plot_collapse.py -- test the user's FSS ansatz:
#     c*ln(L) - S̃(ξ,L)  =  Φ±( ln(ξ/L) )
# one universal curve per side of the transition (+ = disordered, − = ordered).
# ξ from the exact McCoy–Wu formulas:  1/ξ+ = 2(β*−β),  1/ξ− = 4(β−β*),
# e^{-2β*} = tanh β.  c is scanned to minimize the collapse residual (both
# branches share one c; the critical pooled curve is shown for reference).
# Writes figs2d/collapse_test.png and prints the best c per branch.
import os, csv, math
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
DATA = os.environ.get("DATA", os.path.join(ROOT, "data2d"))
MC_FILE = os.environ.get("MC_FILE", os.path.join(ROOT, "data/rk_ising/mc", "mc_results.csv"))
FIG = os.environ.get("FIG", os.path.join(ROOT, "data/rk_ising/mc", "figures")); os.makedirs(FIG, exist_ok=True)
BETA_C = 0.5*math.log(1+math.sqrt(2))

def beta_star(b): return 0.5*math.log(1.0/math.tanh(b))
def xi_of(b):
    bs = beta_star(b)
    if b < BETA_C:  return 1.0/(2.0*(bs-b))     # disordered
    else:           return 1.0/(4.0*(b-bs))     # ordered

# ---- pool MC tS rows per (beta_string, L) ----
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

# branches: list of (beta_value, xi, L, tS, err)
dis, orde, crit = [], [], []
for (bstr, L), v in sorted(pools.items()):
    m, e = pool(v)
    if bstr == "crit":
        crit.append((L, m, e)); continue
    b = float(bstr)
    if b in (0.30, 0.60):  # deep points: xi < 2 lattice units, formula still fine
        pass
    (dis if b < BETA_C else orde).append((b, xi_of(b), L, m, e))
crit.sort()

def branch_residual(branch, c):
    # quality of collapse: fit y(x) with a quadratic, weighted residual
    if len(branch) < 4: return 0.0
    x = np.array([math.log(xi/L) for (_, xi, L, _, _) in branch])
    y = np.array([c*math.log(L) - t for (_, _, L, t, _) in branch])
    e = np.array([err for (*_, err) in branch])
    A = np.vstack([np.ones_like(x), x, x*x]).T
    W = 1/e**2
    coef, *_ = np.linalg.lstsq(A*W[:, None], y*W, rcond=None)
    r = (y - A@coef)/e
    return float(np.sum(r**2))/max(1, len(x)-3)

cs = np.arange(-0.40, 0.41, 0.005)
best = min(cs, key=lambda c: branch_residual(dis, c) + branch_residual(orde, c))
bd = min(cs, key=lambda c: branch_residual(dis, c))
bo = min(cs, key=lambda c: branch_residual(orde, c))
print(f"best shared c = {best:+.3f}   (dis-only {bd:+.3f}, ord-only {bo:+.3f})")
print(f"chi2/dof at best: dis {branch_residual(dis, best):.2f}  ord {branch_residual(orde, best):.2f}")

C_DIS, C_ORD, C_CRIT = "#2E5FA3", "#C4552D", "#666666"
cl = np.array([math.log(L) for L, _, _ in crit]); cv = np.array([t for _, t, _ in crit])
def S_crit(L): return float(np.interp(math.log(L), cl, cv))
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4), dpi=160)
for ax, c, ttl in ((axes[0], best, f"user ansatz, shared best c = {best:+.3f}"),):
    for branch, col, lab in ((dis, C_DIS, "disordered"), (orde, C_ORD, "ordered")):
        bs = sorted(set(b for (b, *_ ) in branch))
        marks = dict(zip(bs, ["o", "s", "^", "D", "v"]))
        for b0 in bs:
            pts = [(math.log(xi/L), c*math.log(L)-t, e) for (b, xi, L, t, e) in branch if b == b0]
            pts.sort()
            ax.errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[p[2] for p in pts],
                        fmt=marks[b0]+"-", ms=5, lw=0.8, capsize=2.5, color=col, alpha=0.85,
                        label=f"{lab} β={b0}")
    ax.set_xlabel(r"$\ln(\xi/L)$"); ax.set_ylabel(r"$c\ln L-\tilde S$")
    ax.set_title(ttl, fontsize=10)
    ax.grid(alpha=0.25, lw=0.5); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, loc="best", framealpha=0.9)
ax2 = axes[1]
for branch, col, lab in ((dis, C_DIS, "disordered"), (orde, C_ORD, "ordered")):
    bs = sorted(set(b for (b, *_ ) in branch))
    marks = dict(zip(bs, ["o", "s", "^", "D", "v"]))
    for b0 in bs:
        pts = sorted((math.log(xi/L), t - S_crit(L), e)
                     for (b, xi, L, t, e) in branch if b == b0 and 8 <= L <= 64)
        ax2.errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[p[2] for p in pts],
                     fmt=marks[b0]+"-", ms=5, lw=0.8, capsize=2.5, color=col, alpha=0.85,
                     label=f"{lab} {b0}")
ax2.axhline(0, color=C_CRIT, lw=0.8, ls=":")
ax2.set_xlabel(r"$\ln(\xi/L)$"); ax2.set_ylabel(r"$\tilde S-\tilde S_c(L)$")
ax2.set_title(r"critical-curve subtraction: no vertical $L$ rescaling", fontsize=10)
ax2.grid(alpha=0.25, lw=0.5); ax2.spines[["top", "right"]].set_visible(False)
ax2.legend(fontsize=7, loc="best", framealpha=0.9)
ax3 = axes[2]
for branch, col, lab in ((dis, C_DIS, "disordered"), (orde, C_ORD, "ordered")):
    bs = sorted(set(b for (b, *_ ) in branch))
    marks = dict(zip(bs, ["o", "s", "^", "D", "v"]))
    for b0 in bs:
        pts = sorted((math.log(xi/L), t - S_crit(L), e) for (b, xi, L, t, e) in branch
                     if b == b0 and 8 <= L <= 64)
        ax3.errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[p[2] for p in pts],
                     fmt=marks[b0]+"-", ms=5, lw=0.8, capsize=2.5, color=col, alpha=0.85,
                     label=f"{lab} {b0}")
ax3.axhline(0, color=C_CRIT, lw=0.8, ls=":")
ax3.set_xlabel(r"$\ln(\xi/L)$"); ax3.set_ylabel(r"$\tilde S(\xi,L)-\tilde S_c(L)$")
ax3.set_title("generalized: subtract MEASURED critical curve", fontsize=10)
ax3.grid(alpha=0.25, lw=0.5); ax3.spines[["top", "right"]].set_visible(False)
ax3.legend(fontsize=7, loc="best", framealpha=0.9)
fig.suptitle(r"FSS-collapse test:  $c\ln L-\tilde S(\xi,L)$ vs $\ln(\xi/L)$"
             "  (critical branch: " + ", ".join(f"S̃({L})={t:.3f}" for L, t, _ in crit[-3:]) + " ...)",
             fontsize=9)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = os.path.join(FIG, "collapse_test.png")
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
print("\nbranch data (beta, xi, L, tS, err):")
for br, nm in ((dis, "DIS"), (orde, "ORD")):
    for row in br: print(nm, *["%.4g" % v for v in row])
print("DONE_COLLAPSE")
