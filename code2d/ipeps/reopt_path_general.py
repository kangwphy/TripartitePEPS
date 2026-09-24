#!/usr/bin/env python3
"""Re-optimise the junction path with the TRUE dimensions.

The compiled path was found by scoring intermediates lexicographically on
(number of chi legs, number of D legs).  That is right for the RK-Ising case,
where the one-replica virtual bond is D_eff = 2 and chi1 >> D_eff, so a chi leg
always dominates.  It is WRONG for a general quantum state: there one replica is
a bra-ket double layer, D_eff = D^2, and at D = 3 that is 9 against a chi1 that
memory caps near 7 -- the D legs are then the BIGGER ones and the ordering that
minimises chi width is no longer the ordering that minimises memory.

This searches again with the actual sizes, peak = max over steps of
prod(dim) of the result, for a grid of (D_eff, chi1) regimes.
"""
import argparse, math, random
from optimize_split_junction_path import junction_network
from optimize_split_junction_path2 import CURRENT_PATH, label_kind, label_graph, min_fill_order

factors, dims = junction_network(10)
KIND = {l: label_kind(l, dims) for l in dims}

def sizes(labels, chi, deff):
    n = 1.0
    for l in labels:
        n *= chi if KIND[l] == "chi" else deff
    return n

def exps(labels):
    c = sum(1 for l in labels if KIND[l] == "chi")
    return c, len(labels) - c

def replay(path, chi, deff):
    ops = [frozenset(f) for f in factors]
    peak = 0.0; peak_ex = None; fl = 0.0
    for i, j in path:
        a, b = ops[i], ops[j]; com = a & b; out = (a | b) - com
        s = sizes(out, chi, deff)
        fl += s * sizes(com, chi, deff)
        if s > peak: peak, peak_ex = s, exps(out)
        del ops[j]; del ops[i]; ops.append(out)
    ok = len(ops) == 1 and not ops[0]
    return peak, peak_ex, fl, ok

def greedy(rng, chi, deff, temp):
    ops = [frozenset(f) for f in factors]; path = []
    while len(ops) > 1:
        cand = []
        for i in range(len(ops)):
            for j in range(i+1, len(ops)):
                if ops[i] & ops[j] or len(ops) <= 2:
                    out = (ops[i] | ops[j]) - (ops[i] & ops[j])
                    cand.append((sizes(out, chi, deff), i, j))
        if not cand:
            cand = [(sizes((ops[0]|ops[1])-(ops[0]&ops[1]), chi, deff), 0, 1)]
        best = min(c[0] for c in cand)
        if temp <= 0:
            pool = [c for c in cand if c[0] == best]
            _, i, j = rng.choice(pool)
        else:
            w = [math.exp(-(math.log(c[0]/best))/temp) for c in cand]
            _, i, j = rng.choices(cand, weights=w, k=1)[0]
        out = (ops[i] | ops[j]) - (ops[i] & ops[j])
        del ops[j]; del ops[i]; ops.append(out); path.append((i, j))
    return path

def order_to_path(order):
    ops = [frozenset(f) for f in factors]; path = []
    for lab in order:
        h = [i for i, o in enumerate(ops) if lab in o]
        while len(h) > 1:
            i, j = sorted(h[:2])
            out = (ops[i] | ops[j]) - (ops[i] & ops[j])
            del ops[j]; del ops[i]; ops.append(out); path.append((i, j))
            h = [k for k, o in enumerate(ops) if lab in o]
    while len(ops) > 1:
        out = (ops[0] | ops[1]) - (ops[0] & ops[1])
        del ops[1]; del ops[0]; ops.append(out); path.append((0, 1))
    return path

ap = argparse.ArgumentParser(); ap.add_argument("--restarts", type=int, default=3000)
ap.add_argument("--minfill", type=int, default=20000)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--only", type=str, default="")
args = ap.parse_args()
adj = label_graph(factors)
REG = [("RK-Ising      ", 2, 24), ("PEPS D=2      ", 4, 16), ("PEPS D=3      ", 9, 7),
       ("PEPS D=3 wish ", 9, 12), ("PEPS D=4      ", 16, 4), ("PEPS D=5      ", 25, 3)]
print(f"{'regime':16} {'Deff':>5} {'chi1':>5} {'compiled path':>22} {'re-optimised':>22} {'gain':>8}")
for name, deff, chi in REG:
    if args.only and args.only not in name: continue
    p0, e0, f0, ok0 = replay(CURRENT_PATH, chi, deff)
    rng = random.Random(args.seed); best = None
    for t in range(args.restarts):
        path = greedy(rng, chi, deff, 0.0 if t % 20 == 0 else rng.choice([0.05, 0.3, 1.0]))
        pk, ex, fl, ok = replay(path, chi, deff)
        if ok and (best is None or pk < best[0]): best = (pk, ex, fl, path)
    for t in range(args.minfill):
        order, _ = min_fill_order(adj, KIND, rng, 0.0 if t % 10 == 0 else rng.choice([0.5, 2.0, 6.0]))
        pk, ex, fl, ok = replay(order_to_path(order), chi, deff)
        if ok and (best is None or pk < best[0]): best = (pk, ex, fl, order_to_path(order))
    g0 = p0*8/2**30; g1 = best[0]*8/2**30
    print(f"{name:16} {deff:5d} {chi:5d}  chi^{e0[0]}D^{e0[1]:<2} {g0:10.1f} GiB   "
          f"chi^{best[1][0]}D^{best[1][1]:<2} {g1:10.1f} GiB {p0/best[0]:7.1f}x")
