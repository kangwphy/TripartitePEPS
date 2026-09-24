#!/usr/bin/env python3
"""Plot the modular square-RVB D=3 boundary-MPS scan."""
import csv
from pathlib import Path
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[2] / "data"
repo_root = Path(__file__).resolve().parents[5]
files = [
    root / "square_rvb_d3" / "scan.csv",
    root / "square_rvb_d3_chi3" / "scan.csv",
    root / "square_rvb_d3_chi4" / "scan.csv",
    root / "square_rvb_d3_chi8" / "scan.csv",
    repo_root / "data" / "square_rvb_d3_chi10_relaxed" / "scan.csv",
    repo_root / "data" / "square_rvb_d3_chi12_relaxed2" / "scan.csv",
    repo_root / "data" / "square_rvb_d3_chi16_relaxed2" / "scan.csv",
    repo_root / "data" / "square_rvb_d3_chi24_relaxed2" / "scan.csv",
    repo_root / "data" / "square_rvb_d3_chi24_relaxed3" / "scan.csv",
    repo_root / "data" / "square_rvb_d3_chi32_relaxed2" / "scan.csv",
]
rows = []
for path in files:
    if not path.exists():
        continue
    with path.open() as f:
        for row in csv.DictReader(f):
            try:
                if row["error"]:
                    continue
                rows.append({k: float(row[k]) for k in ("chi", "xi_mean", "stilde")})
            except (KeyError, ValueError):
                continue
rows.sort(key=lambda r: r["chi"])
if not rows:
    raise SystemExit("no successful RVB scan points found")
chi = [r["chi"] for r in rows]
xi = [r["xi_mean"] for r in rows]
st = [r["stilde"] for r in rows]

fig, ax = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
ax[0].plot(chi, xi, "o-", color="tab:blue")
ax[0].set(xlabel=r"boundary bond dimension $\chi$", ylabel=r"$\xi_{\rm bMPS}$",
          title="RVB $D=3$: boundary $\\xi$ vs $\\chi$\n(relaxed-tail points included)")
ax[1].plot(chi, st, "s-", color="tab:red")
ax[1].set(xlabel=r"boundary bond dimension $\chi$", ylabel=r"$\widetilde S$",
          title=r"RVB $D=3$: $\widetilde S$ vs $\chi$")
ax[2].plot(xi, st, "D-", color="tab:green")
for x, y, c in zip(xi, st, chi):
    ax[2].annotate(f"$\chi={int(c)}$", (x, y), xytext=(4, 4),
                   textcoords="offset points", fontsize=8)
ax[2].set(xlabel=r"$\xi_{\rm bMPS}$", ylabel=r"$\widetilde S$",
          title=r"RVB $D=3$: $\widetilde S$ vs $\xi$")
for a in ax:
    a.grid(True, alpha=0.3)
out = root / "square_rvb_d3" / "stilde_xi_chi.png"
fig.savefig(out, dpi=180)
print(out)
