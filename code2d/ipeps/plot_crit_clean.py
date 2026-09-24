# plot_crit_clean.py -- the SIMPLE view: healthy runs vs diagnosed-bad runs.
# One point = one full tSJ computation. Two classes only.
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OLD = "/ix/zdai/kangw/PEPS3EE/PEPS/clean/data/rk_ising/ctmrg/tables/defect_ctmrg/critical_plain.csv"
SYM = "/ix/zdai/kangw/PEPS3EE/PEPS/clean/data/rk_ising/ctmrg/tables/defect_ctmrg/critical_symmetrized.csv"
OUT = "/ix/zdai/kangw/PEPS3EE/PEPS/clean/data/rk_ising/ctmrg/figures/defect_ctmrg/crit_stilde_clean.png"
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

pts = []   # (chi, tSJ, healthy)
for r in readcsv(OLD):
    ok = r["seam_asym"] < 1e-3 and r["it4"] < 300 and r["it2"] < 20*r["xi2"]
    pts.append((r["chi_req"], r["tSJ"], ok))
for r in readcsv(SYM):
    ok = r["asym"] < 1e-3 and r["it4"] < 300 and abs(r["kY"]) < 1
    pts.append((r["chi"], r["tSJ"], ok))
# dedupe identical values
seen, uniq = set(), []
for p in pts:
    key = (round(p[0]), round(p[1], 6))
    if key in seen: continue
    seen.add(key); uniq.append(p)
pts = uniq

fig, a = plt.subplots(figsize=(8.5, 5.5))
bad = [(c, t) for c, t, ok in pts if not ok]
good = sorted([(c, t) for c, t, ok in pts if ok])
for c, t in bad:
    a.semilogx(c, t, "x", ms=7, mew=1.8, color="silver")
gx = [c for c, _ in good]; gy = [t for _, t in good]
a.semilogx(gx, gy, "o", ms=9, color="seagreen")
m, s = np.mean([t for c, t in good if c >= 20]), np.std([t for c, t in good if c >= 20])
a.axhspan(m - s, m + s, color="seagreen", alpha=0.12)
a.axhline(m, color="seagreen", ls="--", lw=0.9)
a.plot([], [], "o", ms=9, color="seagreen",
       label=f"healthy runs (all checks pass): {m:.2f}$\\pm${s:.2f}")
a.plot([], [], "x", ms=7, mew=1.8, color="silver",
       label="diagnosed-bad runs (auto-flagged, discarded)")
xr = np.linspace(10, 100, 50)
a.semilogx(xr, m + 0.0896*np.log(xr/30.), ":", color="k", lw=1.5,
           label=r"CFT log-growth reference $0.090\ln\chi$")
a.axhline(0, color="gray", lw=0.6)
a.set_xlabel(r"$\chi$  (log axis)"); a.set_ylabel(r"$\tilde S_J$")
a.set_title(r"$\tilde S_J(\beta_c)$, all runs, two classes."
            "  EXPLORATORY: healthy runs still fail the $2s_2$ gate (~50%)",
            fontsize=10)
a.legend(fontsize=9, loc="upper left")
fig.tight_layout()
fig.savefig(OUT, dpi=160)
print("wrote", OUT, f" healthy n={len(good)} band {m:.3f}+-{s:.3f}, bad n={len(bad)}")
