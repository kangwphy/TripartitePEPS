#!/usr/bin/env python3
"""Pool the local-update audit and compare it with production SW data."""

import csv
import glob
import math
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOCAL_GLOB = os.path.join(ROOT, "data/rk_ising/mc", "audits", "local_update_pilot", "raw",
                          "L*_n10000_eq5000_seed*.csv")
SW_FILE = os.path.join(ROOT, "data/rk_ising/mc", "tables", "mc_target_tS.csv")
OUTDIR = os.path.join(ROOT, "data/rk_ising/mc", "audits", "local_update_pilot")
FIGDIR = os.path.join(OUTDIR, "figures")
os.makedirs(FIGDIR, exist_ok=True)

SIZES = (4, 8, 16, 32, 48)
BETAS = ("0.3", "0.6", "crit")
BETA_ALIASES = {"0.3": "0.3", "0.30": "0.3",
                "0.6": "0.6", "0.60": "0.6", "crit": "crit"}
EXACT_L4 = {"0.3": 0.008925087523114217,
            "0.6": 0.16918765857412854,
            "crit": 0.072702111439829}


def pooled(runs):
    """Return weighted mean and conservative error diagnostics."""
    weights = [1.0 / r[1]**2 for r in runs]
    mean = sum(r[0] * w for r, w in zip(runs, weights)) / sum(weights)
    formal = 1.0 / math.sqrt(sum(weights))
    if len(runs) > 1:
        chain_sem = math.sqrt(sum((r[0] - mean)**2 for r in runs)
                              / ((len(runs) - 1) * len(runs)))
    else:
        chain_sem = formal
    return mean, formal, chain_sem, max(formal, chain_sem)


local = defaultdict(list)
for path in glob.glob(LOCAL_GLOB):
    with open(path, newline="") as stream:
        row = next(csv.DictReader(stream))
    beta = BETA_ALIASES[row["beta"]]
    key = (beta, int(row["L"]))
    local[key].append((float(row["tildeS"]), float(row["error"]),
                       float(row["runtime_s"]), int(row["seed"])))

# The critical reference is the curated production snapshot.  The beta=0.3
# and 0.6 comparison runs were launched later and are read from their Slurm
# logs, whose job IDs are fixed here to keep the selection reproducible.
sw = defaultdict(list)
with open(SW_FILE, newline="") as stream:
    for row in csv.DictReader(stream):
        beta = BETA_ALIASES.get(row["beta"])
        if beta != "crit":
            continue
        lattice_size = int(row["L"])
        if lattice_size not in SIZES:
            continue
        tag = row.get("tag", "")
        if "S2C" in tag or "K4C" in tag:
            continue
        sw[(beta, lattice_size)].append(
            (float(row["value"]), float(row["error"]),
             float(row["seconds"]), int(row["seed"])))

sw_job_ids = (list(range(23758824, 23758872))
              + list(range(23758792, 23758796))
              + list(range(23758808, 23758812)))
start_pattern = re.compile(r"L=(\d+) beta=([0-9.]+).*seed=(\d+)")
result_pattern = re.compile(
    r"tildeS = ([+-]?[0-9.]+) \+- ([0-9.]+).*\(([0-9]+)s total\)")
for job_id in sw_job_ids:
    path = os.path.join(ROOT, "logs2d", f"mc_{job_id}.out")
    with open(path) as stream:
        contents = stream.read()
    start = start_pattern.search(contents)
    result = result_pattern.search(contents)
    if start is None or result is None:
        raise RuntimeError(f"could not parse SW log {path}")
    lattice_size, beta_value, seed = start.groups()
    beta = BETA_ALIASES.get(str(float(beta_value)))
    if beta not in ("0.3", "0.6") or int(lattice_size) not in SIZES:
        continue
    value, error, runtime = result.groups()
    sw[(beta, int(lattice_size))].append(
        (float(value), float(error), float(runtime), int(seed)))

summary = []
for beta in BETAS:
    for lattice_size in SIZES:
        key = (beta, lattice_size)
        if len(local[key]) != 4:
            raise RuntimeError(f"expected four local chains for {key}, got {len(local[key])}")
        lm, lf, ls, le = pooled(local[key])
        sm, sf, ss, se = pooled(sw[key])
        combined = math.hypot(le, se)
        summary.append({
            "beta": beta, "L": lattice_size,
            "local_n": len(local[key]), "local_mean": lm,
            "local_formal_error": lf, "local_chain_sem": ls,
            "local_error": le,
            "local_runtime_mean_s": sum(r[2] for r in local[key]) / len(local[key]),
            "local_min": min(r[0] for r in local[key]),
            "local_max": max(r[0] for r in local[key]),
            "sw_n": len(sw[key]), "sw_mean": sm, "sw_error": se,
            "difference": lm-sm, "pull": (lm-sm)/combined,
        })

columns = list(summary[0])
summary_file = os.path.join(OUTDIR, "local_vs_sw_summary.csv")
with open(summary_file, "w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(summary)

print("beta   L       local update          SW reference       pull   local range")
for row in summary:
    print(f"{row['beta']:>4} {row['L']:3d}  "
          f"{row['local_mean']:+.5f} +- {row['local_error']:.5f}   "
          f"{row['sw_mean']:+.5f} +- {row['sw_error']:.5f}   "
          f"{row['pull']:+5.2f}   "
          f"[{row['local_min']:+.3f},{row['local_max']:+.3f}]")

fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.7), dpi=180)
titles = {"0.3": r"$\beta=0.3$", "0.6": r"$\beta=0.6$",
          "crit": r"$\beta=\beta_c$"}
for ax, beta in zip(axes, BETAS):
    rows = [r for r in summary if r["beta"] == beta]
    xs = [r["L"] for r in rows]
    ax.axhline(0.0, color="0.75", lw=0.8, ls=":")
    ax.errorbar([x*0.97 for x in xs], [r["sw_mean"] for r in rows],
                yerr=[r["sw_error"] for r in rows], fmt="o", ms=4.8,
                color="#1f4e79", capsize=2.5, label="SW")
    ax.errorbar([x*1.03 for x in xs], [r["local_mean"] for r in rows],
                yerr=[r["local_error"] for r in rows], fmt="s", ms=4.6,
                color="#b55223", capsize=2.5, label="local")
    ax.plot(4, EXACT_L4[beta], marker="*", color="black", ms=8,
            linestyle="none", label=r"exact $4\times4$")
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_title(titles[beta])
    ax.set_xlabel(r"$L$")
    ax.grid(alpha=0.18, lw=0.5)
axes[0].set_ylabel(r"$\widetilde S$")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, 1.04))
fig.tight_layout()
figure_file = os.path.join(FIGDIR, "local_vs_sw.png")
fig.savefig(figure_file, bbox_inches="tight")
print("wrote", summary_file)
print("wrote", figure_file)
