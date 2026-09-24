#!/usr/bin/env python3
"""Pool the L=24 critical SW audit and test node-level consistency."""

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
LOG_GLOB = os.path.join(ROOT, "logs2d", "mc_l24_audit_*.out")
OUT = os.path.join(ROOT, "data/rk_ising/mc", "audits/critical_l24/short")
os.makedirs(OUT, exist_ok=True)

header_re = re.compile(
    r"L=(?P<L>\d+) beta=(?P<beta>[0-9.]+) sweeps=(?P<sweeps>\d+) "
    r"eq=(?P<eq>\d+) seed=(?P<seed>\d+).*update=(?P<update>\w+) "
    r"quantity=(?P<quantity>\w+) reverse=(?P<reverse>\w+) fresh=(?P<fresh>\w+)")
node_re = re.compile(
    r"lam=(?P<lam>[0-9.]+)\s+<dE/dl>=(?P<mean>[+-]?[0-9.]+)\s+"
    r"\+\-\s+(?P<err>[0-9.]+)")
result_re = re.compile(
    r"RESULT_MC L=(?P<L>\d+) beta=(?P<beta>\S+) update=(?P<update>\w+) "
    r"tildeS = (?P<value>[+-]?[0-9.]+) \+- (?P<error>[0-9.]+)")

records = []
profiles = defaultdict(list)
for path in sorted(glob.glob(LOG_GLOB)):
    text = open(path).read()
    h = header_re.search(text)
    r = result_re.search(text)
    nodes = list(node_re.finditer(text))
    if not h or not r or len(nodes) != 12:
        continue
    hd, rd = h.groupdict(), r.groupdict()
    if hd["fresh"] == "true": mode = "fresh"
    elif hd["reverse"] == "true": mode = "reverse"
    elif int(hd["eq"]) <= 2000: mode = "short"
    else: mode = "long"
    tm = re.search(r"\(([0-9]+)s total\)", text)
    records.append({"mode": mode, "seed": int(hd["seed"]), "L": int(hd["L"]),
                    "beta": hd["beta"], "sweeps": int(hd["sweeps"]),
                    "eq": int(hd["eq"]), "value": float(rd["value"]),
                    "error": float(rd["error"]), "runtime_s": int(tm.group(1)),
                    "log": os.path.basename(path)})
    for n in nodes:
        profiles[(mode, float(n["lam"]))].append(
            (float(n["mean"]), float(n["err"]), int(hd["seed"])))


def pool(values):
    weights = [1.0 / e**2 for _, e, _ in values]
    mean = sum(v*w for (v, _, _), w in zip(values, weights)) / sum(weights)
    formal = 1.0 / math.sqrt(sum(weights))
    scatter = (math.sqrt(sum((v-mean)**2 for v, _, _ in values)
                         / ((len(values)-1)*len(values)))
               if len(values) > 1 else formal)
    return mean, formal, scatter, max(formal, scatter)


if len(records) != 16:
    raise RuntimeError(f"expected 16 completed audit logs, found {len(records)}")
records.sort(key=lambda x: (x["mode"], x["seed"]))

with open(os.path.join(OUT, "summary.csv"), "w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=records[0].keys())
    writer.writeheader(); writer.writerows(records)

print("mode     values                                      pooled")
for mode in ("short", "long", "fresh", "reverse"):
    rs = [r for r in records if r["mode"] == mode]
    m, f, s, e = pool([(r["value"], r["error"], r["seed"]) for r in rs])
    print(f"{mode:7s} " + " ".join(f"{r['value']:+.5f}" for r in rs)
          + f"   {m:+.5f} +- {e:.5f} (formal {f:.5f}, chain SEM {s:.5f})")

with open(os.path.join(OUT, "node_profiles.csv"), "w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["mode", "lambda", "mean", "error", "chain_sem"])
    for mode in ("short", "long", "fresh", "reverse"):
        for (m0, lam), values in sorted(profiles.items()):
            if m0 != mode: continue
            mean, formal, scatter, error = pool(values)
            writer.writerow([mode, lam, mean, error, scatter])

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.5, 4.0), dpi=170)
colors = {"short": "#777777", "long": "#1f4e79",
          "fresh": "#b55223", "reverse": "#4b8b3b"}
for mode in ("short", "long", "fresh", "reverse"):
    rs = [r for r in records if r["mode"] == mode]
    m, _, _, e = pool([(r["value"], r["error"], r["seed"]) for r in rs])
    ax0.errorbar(mode, m, yerr=e, fmt="o", capsize=3, color=colors[mode])
    for r in rs: ax0.plot(mode, r["value"], "x", color=colors[mode], alpha=.7)
    pts = sorted((lam, pool(v)[0], pool(v)[3])
                 for (m0, lam), v in profiles.items() if m0 == mode)
    ax1.errorbar([x[0] for x in pts], [x[1] for x in pts],
                 yerr=[x[2] for x in pts], marker="o", ms=3, lw=1,
                 color=colors[mode], label=mode)
ax0.axhline(0, color="0.7", ls=":")
ax0.set_ylabel(r"$\widetilde S$"); ax0.set_title(r"$L=24$, $\beta_c$")
ax0.grid(alpha=.2)
ax1.axhline(0, color="0.7", ls=":")
ax1.set_xlabel(r"$\lambda$"); ax1.set_ylabel(r"$\langle\Delta K\rangle_\lambda$")
ax1.set_title("GL12 node profiles"); ax1.legend(frameon=False); ax1.grid(alpha=.2)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "audit.png"), bbox_inches="tight")
print("wrote", OUT)
