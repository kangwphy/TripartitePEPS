#!/usr/bin/env python3
"""Summarize the L=32 critical SW thermalization audit from Slurm logs."""

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
LOG_GLOB = os.path.join(ROOT, "logs2d", "mc_audit_*.out")
OUT = os.path.join(ROOT, "data/rk_ising/mc", "audits/thermalization")
os.makedirs(OUT, exist_ok=True)

header_re = re.compile(
    r"L=(?P<L>\d+) beta=(?P<beta>[0-9.]+) sweeps=(?P<sweeps>\d+) "
    r"eq=(?P<eq>\d+) seed=(?P<seed>\d+).*update=(?P<update>\w+) "
    r"quantity=(?P<quantity>\w+) reverse=(?P<reverse>\w+) fresh=(?P<fresh>\w+)")
node_re = re.compile(
    r"lam=(?P<lam>[0-9.]+)\s+<dE/dl>=(?P<mean>[+-]?[0-9.]+) "
    r"\+\-\s+(?P<err>[0-9.]+)")
result_re = re.compile(
    r"RESULT_MC L=(?P<L>\d+) beta=(?P<beta>\S+) update=(?P<update>\w+) "
    r"tildeS = (?P<value>[+-]?[0-9.]+) \+- (?P<error>[0-9.]+)")

records = []
profiles = defaultdict(list)
for path in sorted(glob.glob(LOG_GLOB)):
    text = open(path).read()
    header = header_re.search(text)
    result = result_re.search(text)
    nodes = list(node_re.finditer(text))
    if not header or not result or len(nodes) != 12:
        continue
    h = header.groupdict()
    r = result.groupdict()
    if h["fresh"] == "true":
        mode = "fresh"
    elif h["reverse"] == "true":
        mode = "reverse"
    elif int(h["eq"]) <= 1000:
        mode = "short"
    else:
        mode = "long"
    runtime_match = re.search(r"\(([0-9]+)s total\)", text)
    records.append({"mode": mode, "seed": int(h["seed"]), "L": int(h["L"]),
                    "beta": h["beta"], "eq": int(h["eq"]),
                    "value": float(r["value"]), "error": float(r["error"]),
                    "runtime_s": int(runtime_match.group(1)),
                    "log": os.path.basename(path)})
    for n in nodes:
        profiles[(mode, float(n["lam"]))].append(
            (float(n["mean"]), float(n["err"]), int(h["seed"])))


def pool(values):
    weights = [1.0 / e**2 for _, e, _ in values]
    mean = sum(v*w for (v, _, _), w in zip(values, weights)) / sum(weights)
    formal = 1.0 / math.sqrt(sum(weights))
    scatter = (math.sqrt(sum((v-mean)**2 for v, _, _ in values)
                         / ((len(values)-1)*len(values)))
               if len(values) > 1 else formal)
    return mean, max(formal, scatter), scatter


if len(records) != 12:
    raise RuntimeError(f"expected 12 completed audit logs, found {len(records)}")

records.sort(key=lambda r: (r["mode"], r["seed"]))
summary_path = os.path.join(OUT, "summary.csv")
with open(summary_path, "w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)

print("mode     chain values (tildeS)                         pooled")
for mode in ("short", "long", "fresh", "reverse"):
    vals = [r for r in records if r["mode"] == mode]
    mean, err, scatter = pool([(r["value"], r["error"], r["seed"]) for r in vals])
    pieces = " ".join(f"{r['value']:+.5f}" for r in vals)
    print(f"{mode:7s} {pieces}   {mean:+.5f} +- {err:.5f} "
          f"(chain SEM {scatter:.5f})")

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4), dpi=170)
colors = {"short": "#777777", "long": "#1f4e79",
          "fresh": "#b55223", "reverse": "#4b8b3b"}
for mode in ("short", "long", "fresh", "reverse"):
    vals = [r for r in records if r["mode"] == mode]
    mean, err, _ = pool([(r["value"], r["error"], r["seed"]) for r in vals])
    ax0.errorbar(mode, mean, yerr=err, fmt="o", color=colors[mode], capsize=3)
    for r in vals:
        ax0.plot(mode, r["value"], "x", color=colors[mode], alpha=0.75)
    points = sorted((lam, pool(v)[0], pool(v)[1])
                    for (m, lam), v in profiles.items() if m == mode)
    ax1.errorbar([p[0] for p in points], [p[1] for p in points],
                 yerr=[p[2] for p in points], marker="o", ms=3,
                 lw=1, color=colors[mode], label=mode)
ax0.axhline(0, color="0.7", ls=":")
ax0.set_ylabel(r"$\widetilde S$")
ax0.set_title(r"$L=32$, $\beta_c$: integrated result")
ax0.grid(alpha=0.2)
ax1.axhline(0, color="0.7", ls=":")
ax1.set_xlabel(r"$\lambda$")
ax1.set_ylabel(r"$\langle\Delta K\rangle_\lambda$")
ax1.set_title("GL12 node profile")
ax1.legend(frameon=False)
ax1.grid(alpha=0.2)
fig.tight_layout()
figure_path = os.path.join(OUT, "thermalization_audit.png")
fig.savefig(figure_path, bbox_inches="tight")
print("wrote", summary_path)
print("wrote", figure_path)
