#!/usr/bin/env python3
"""How large a chi1 would each beta need, and does it fit in memory?

WARNING, established after this script was written: the deficit screen below is
a JOB-SIZING heuristic, not an error bar, and it is falsified in sample as an
error predictor.  At the same deficit 0.043, beta=0.40 at chi1=16 has relative
error 7e-5 while beta=0.42 at chi1=24 has 7.8e-4 -- the deficit-to-error map
spreads one to two decades, so any chi1 read off it carries a factor >=2
uncertainty.  Convergence is decided by the chi1 drift of the components
2*kappa_A and kappa_Y (printed at the bottom), never by the deficit.  Using the
deficit for that once produced a wrong verdict: it put beta=0.43 at chi1~35 when
a direct fit to S-tilde itself gives chi1~65, and the honest conclusion is that
beta=0.43 is unreachable.

The measured environment deficit  delta(chi1) = 1 - xi1_measured/xi_exact  is fitted
at fixed beta.  Empirically delta ~ C(beta)/chi1 at the larger chi1, so C is read off
from the largest available chi1 and cross-checked against the whole series.  The
target deficit is the one achieved by betas whose S-tilde is converged to <1e-6.

Memory law, measured on this cluster at chi1 = 16, 24, 32 (24.8 / ~273 / 1543 GiB):
    MaxRSS  ~  3.0 * 64 * chi1^6 * 8 bytes
The largest shared-memory node is 2048 GB.
"""
import csv, math, os

ROOT = "/ix/zdai/kangw/PEPS3EE/PEPS/clean"
BETA_C = 0.5 * math.log(1 + math.sqrt(2))
NODE_GIB = 2048 * 1000**3 / 2**30      # 2048 GB node in GiB


def xi_exact(beta):
    bd = -0.5 * math.log(math.tanh(beta))
    return 1 / (2 * (bd - beta)) if beta < BETA_C else 1 / (4 * (beta - bd))


def rss_gib(chi1):
    return 3.0 * 64 * chi1**6 * 8 / 2**30


rows = {}
with open(f"{ROOT}/data/rk_ising/ctmrg/tables/beta_chi_scan_split/summary.csv") as fh:
    for r in csv.DictReader(fh):
        b, c = float(r["beta"]), int(r["chi1"])
        def f(k):
            try: return float(r[k])
            except (ValueError, KeyError, TypeError): return float("nan")
        rows.setdefault(b, {})[c] = (f("tSJ"), f("xi1"), f("kA"), f("kY"))

print("measured memory law check:")
for c, meas in ((16, 24.8), (24, 273.0), (32, 1543.0)):
    print(f"  chi1={c:3d}  model {rss_gib(c):8.1f} GiB   measured {meas:8.1f} GiB"
          f"   ratio {meas/rss_gib(c):.2f}")
print(f"  2048 GB node = {NODE_GIB:.0f} GiB  ->  largest feasible chi1 = "
      f"{max(c for c in range(2, 64) if rss_gib(c) < NODE_GIB)}")

print("\ndeficit delta = 1 - xi_measured/xi_exact, and C = delta*chi1 (flat if delta ~ 1/chi1)")
print(f"{'beta':>6} {'xi_ex':>7}   " + "  ".join(f"c{c}" for c in (4, 8, 16, 24, 32)))
C = {}
for b in sorted(rows):
    if b > BETA_C:
        continue                              # ordered side: xi is the domain wall, skip
    xe = xi_exact(b)
    cells, Cs = [], []
    for c in (4, 8, 16, 24, 32):
        if c in rows[b] and rows[b][c][1] < 1e6:
            d = 1 - rows[b][c][1] / xe
            cells.append(f"{d:.3f}/{d*c:.2f}")
            Cs.append((c, d * c))
        else:
            cells.append("   --   ")
    if Cs:
        C[b] = Cs[-1][1]
    print(f"{b:6.2f} {xe:7.2f}   " + "  ".join(f"{x:>12s}" for x in cells))
print("   (cells are delta/C ; C flat across chi1 confirms delta ~ C/chi1)")

# target: the deficit at which tSJ is converged.  beta=0.40 is converged to 1e-7 by
# chi1=16, where its deficit is 0.043; use that as the acceptance level.
TARGET = 0.043
print(f"\nchi1 needed to reach deficit {TARGET} (the level at which beta=0.40 is "
      f"converged to 1e-7), and its cost:")
print(f"{'beta':>6} {'xi_ex':>7} {'C':>6} {'chi1_need':>10} {'RSS(GiB)':>12} {'fits 2048GB?':>13}")
for b in sorted(C):
    need = C[b] / TARGET
    need_i = math.ceil(need)
    r = rss_gib(need_i)
    print(f"{b:6.2f} {xi_exact(b):7.2f} {C[b]:6.2f} {need_i:10d} {r:12.1f} "
          f"{'YES' if r < NODE_GIB else 'NO':>13}")

print("\ncomponent drift (the criterion that actually matters): |change in 2kA| and "
      "|change in kY| between the two largest chi1")
print(f"{'beta':>6} {'d(2kA)':>10} {'d(kY)':>10} {'d(tS)':>10}  verdict")
for b in sorted(rows):
    cs = sorted(rows[b])
    if len(cs) < 2:
        continue
    a, z = rows[b][cs[-2]], rows[b][cs[-1]]
    dA, dY, dT = abs(2*z[2] - 2*a[2]), abs(z[3] - a[3]), abs(z[0] - a[0])
    v = "EXCLUDE" if max(dA, dY) > 1e-2 else ("MARGINAL" if max(dA, dY) > 1e-3 else "SAFE")
    print(f"{b:6.2f} {dA:10.2e} {dY:10.2e} {dT:10.2e}  {v}")
