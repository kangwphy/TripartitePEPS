#!/usr/bin/env python3
"""Search a lower-WIDTH contraction path for the replica-factorised Y window.

Why not cotengra: it is not installed on this cluster (checked), and the search
here is small enough to do exactly.  More importantly this file also SIMULATES a
given path, which lets us

  * reproduce the peak of the path currently hard-coded in
    split_replica_ctmrg.jl (a self-test of the simulator), and
  * report the new path in the very same convention the Julia loop consumes:
    a LINEAR path of 0-based pairs (i<j) indexing the CURRENT operand list, with
    the contraction result appended at the END.

Cost model: every intermediate is scored by the exponent pair
(number of chi-labels, number of d-labels); chi >> d, so peaks are compared
lexicographically on the chi exponent first.  Minimising the chi width is what
matters here because memory, not flops, is the binding constraint (64*chi^6
doubles is 288 GB at chi1=24, measured).

VERDICT (2026-08, 3.2e5 orderings over four seeds each):
  compiled path      chi^6 d^6   91.1 GiB @chi24   flops 10^13.76
  best min-fill      chi^6 d^6   91.1 GiB @chi24   flops 10^13.77   (ties)
  best greedy        chi^6 d^7  182.2 GiB @chi24   flops 10^14.35   (loses)
No ordering ever reached chi-width < 6.  Width 6 has a structural reading --
one replica must expose 2 boundary legs at each of the 3 seam slots -- and the
minor-min-width bound (minor-monotone, hence valid) gives tw >= 13 for the full
96-label graph and tw >= 6 for its chi-minor.  The compiled path is kept.

Usage:
  python3 optimize_split_junction_path2.py --restarts 20000 --seed 0
  python3 optimize_split_junction_path2.py --minfill 20000 --bounds --seed 1
  python3 optimize_split_junction_path2.py --check-current
"""

from __future__ import annotations

import argparse
import math
import random
from collections import Counter

from optimize_split_junction_path import junction_network

# the path presently compiled into split_replica_ctmrg.jl
CURRENT_PATH = [
    (0, 20), (3, 50), (40, 49), (39, 42), (46, 47), (3, 24), (19, 45), (14, 44),
    (42, 43), (2, 33), (7, 41), (25, 40), (23, 34), (37, 38), (36, 37), (1, 21),
    (8, 35), (22, 34), (24, 30), (31, 32), (0, 13), (14, 30), (8, 29), (27, 28),
    (26, 27), (10, 12), (10, 25), (2, 24), (4, 23), (16, 22), (16, 21), (16, 20),
    (7, 19), (4, 8), (16, 17), (2, 6), (4, 15), (13, 14), (12, 13), (2, 4), (0, 11),
    (5, 10), (7, 9), (7, 8), (1, 5), (0, 6), (1, 5), (0, 1), (0, 3), (1, 2), (0, 1),
]


def label_kind(label: str, dimensions: dict[str, int]) -> str:
    """chi-labels carry the environment bond, everything else is an Ising leg."""
    return "chi" if dimensions[label] > 2 else "d"


def exponents(labels, kinds) -> tuple[int, int]:
    nchi = sum(1 for l in labels if kinds[l] == "chi")
    return nchi, len(labels) - nchi


def simulate(path, factors, kinds):
    """Replay a linear path; return (peak, peak_at, flops_log10, ok)."""
    operands = [frozenset(f) for f in factors]
    if len(set(map(len, [[l for f in factors for l in f]]))) == 0:
        return None
    peak = (-1, -1)
    peak_step = -1
    flops = 0.0
    CHI, D = 24.0, 2.0          # only for a human-readable flop estimate
    for step, (i0, j0) in enumerate(path):
        if not (0 <= i0 < j0 < len(operands)):
            raise ValueError(f"step {step}: indices {(i0, j0)} invalid for "
                             f"{len(operands)} operands (linear-path convention)")
        a, b = operands[i0], operands[j0]
        common = a & b
        out = (a | b) - common
        na, _ = exponents(a, kinds)
        nb, _ = exponents(b, kinds)
        nc, _ = exponents(common, kinds)
        # flops ~ |out| * |common|
        e_chi = exponents(out, kinds)[0] + nc
        e_d = exponents(out, kinds)[1] + exponents(common, kinds)[1]
        flops += CHI ** e_chi * D ** e_d
        del operands[j0]
        del operands[i0]
        operands.append(out)
        ex = exponents(out, kinds)
        if ex > peak:
            peak, peak_step = ex, step
    ok = len(operands) == 1 and len(operands[0]) == 0
    return peak, peak_step, math.log10(flops + 1e-300), ok


def greedy_path(factors, kinds, rng, temperature):
    """One randomised-greedy sweep: repeatedly contract a cheap pair."""
    operands = [frozenset(f) for f in factors]
    path = []
    while len(operands) > 1:
        candidates = []
        for i in range(len(operands)):
            for j in range(i + 1, len(operands)):
                if operands[i] & operands[j] or len(operands) <= 2:
                    out = (operands[i] | operands[j]) - (operands[i] & operands[j])
                    candidates.append((exponents(out, kinds), i, j))
        if not candidates:                      # disconnected remainder
            candidates = [(exponents((operands[0] | operands[1]) -
                                     (operands[0] & operands[1]), kinds), 0, 1)]
        best = min(c[0] for c in candidates)
        if temperature <= 0:
            pool = [c for c in candidates if c[0] == best]
        else:
            weights = []
            for c in candidates:
                excess = (c[0][0] - best[0]) + 0.1 * (c[0][1] - best[1])
                weights.append(math.exp(-excess / temperature))
            pool = [rng.choices(candidates, weights=weights, k=1)[0]]
        _, i, j = rng.choice(pool)
        a, b = operands[i], operands[j]
        out = (a | b) - (a & b)
        del operands[j]
        del operands[i]
        operands.append(out)
        path.append((i, j))
    return path



# ---------------------------------------------------------------------------
# Min-fill elimination search.  Greedy-on-size is myopic and (measured) cannot
# even match the compiled path; contraction WIDTH is what memory scales with,
# and width is the induced width of an elimination ordering on the label graph
# (labels = vertices, each factor's labels = a clique).  min-fill with
# randomised tie-breaking is the standard heuristic for that.
# ---------------------------------------------------------------------------
def label_graph(factors):
    adj = {}
    for f in factors:
        for a in f:
            adj.setdefault(a, set())
            for b in f:
                if a != b:
                    adj[a].add(b)
    return adj


def min_fill_order(adj, kinds, rng, noise):
    """Randomised min-fill ordering; chi-labels are penalised in the bag score."""
    adj = {v: set(n) for v, n in adj.items()}
    order = []
    width = (0, 0)
    while adj:
        best, pool = None, []
        for v, nbrs in adj.items():
            missing = 0
            nl = list(nbrs)
            for i in range(len(nl)):
                for j in range(i + 1, len(nl)):
                    if nl[j] not in adj[nl[i]]:
                        missing += 1
            bag = nbrs | {v}
            nchi = sum(1 for x in bag if kinds[x] == "chi")
            score = (missing, nchi, len(bag) - nchi)
            score = (score[0] + noise * rng.random(), score[1], score[2])
            if best is None or score < best:
                best, pool = score, [v]
            elif score == best:
                pool.append(v)
        v = rng.choice(pool)
        nbrs = adj[v]
        bag = nbrs | {v}
        nchi = sum(1 for x in bag if kinds[x] == "chi")
        width = max(width, (nchi, len(bag) - nchi))
        nl = list(nbrs)
        for i in range(len(nl)):
            for j in range(i + 1, len(nl)):
                adj[nl[i]].add(nl[j])
                adj[nl[j]].add(nl[i])
        for n in nbrs:
            adj[n].discard(v)
        del adj[v]
        order.append(v)
    return order, width


def order_to_path(order, factors):
    """Eliminate labels in the given order, contracting all tensors sharing one."""
    operands = [frozenset(f) for f in factors]
    path = []
    for label in order:
        holders = [i for i, o in enumerate(operands) if label in o]
        while len(holders) > 1:
            i, j = holders[0], holders[1]
            if i > j:
                i, j = j, i
            a, b = operands[i], operands[j]
            out = (a | b) - (a & b)
            del operands[j]
            del operands[i]
            operands.append(out)
            path.append((i, j))
            holders = [k for k, o in enumerate(operands) if label in o]
    while len(operands) > 1:                      # close any disjoint remainder
        a, b = operands[0], operands[1]
        out = (a | b) - (a & b)
        del operands[1]
        del operands[0]
        operands.append(out)
        path.append((0, 1))
    return path


def search_min_fill(factors, kinds, rng, restarts):
    adj = label_graph(factors)
    best = None
    for trial in range(restarts):
        noise = 0.0 if trial % 10 == 0 else rng.choice([0.5, 2.0, 6.0])
        order, _ = min_fill_order(adj, kinds, rng, noise)
        path = order_to_path(order, factors)
        peak, step, lf, ok = simulate(path, factors, kinds)
        if not ok:
            continue
        key = (peak, lf)
        if best is None or key < best[0]:
            best = (key, path, peak, lf)
            print(f"  minfill trial {trial:5d}: peak = chi^{peak[0]} d^{peak[1]}, "
                  f"log10(flops@chi24)={lf:.2f}", flush=True)
    return best



# ---------------------------------------------------------------------------
# Lower-bound diagnostics.  If no search finds width < 6 it matters whether that
# is a search failure or a property of the network, so we also compute the
# minor-min-width (MMW) bound: repeatedly contract a minimum-degree vertex into
# its least-connected neighbour, tracking max(min degree).  Vertex contraction
# produces a minor, and treewidth is minor-monotone, so MMW is a genuine lower
# bound on the treewidth of the label graph -- hence on the contraction width.
# Run on two graphs: all 96 labels, and the chi-minor (every d-label contracted
# into a chi neighbour), which speaks to the chi-width specifically.
# ---------------------------------------------------------------------------
def minor_min_width(adj):
    adj = {v: set(n) for v, n in adj.items()}
    lb = 0
    while len(adj) > 1:
        v = min(adj, key=lambda x: len(adj[x]))
        if not adj[v]:
            del adj[v]
            continue
        lb = max(lb, len(adj[v]))
        u = min(adj[v], key=lambda x: len(adj[v] & adj[x]))
        adj[u] |= adj[v] - {u}
        for w in adj[v] - {u}:
            adj[w].add(u)
        for w in adj[v]:
            adj[w].discard(v)
        del adj[v]
    return lb


def chi_minor(adj, kinds):
    """Contract every d-label into an adjacent label: a minor on chi-labels."""
    adj = {v: set(n) for v, n in adj.items()}
    for v in [x for x in list(adj) if kinds[x] == "d"]:
        if v not in adj:
            continue
        chi_nbrs = [x for x in adj[v] if kinds[x] == "chi"]
        target = chi_nbrs[0] if chi_nbrs else (sorted(adj[v])[0] if adj[v] else None)
        if target is None:
            del adj[v]
            continue
        adj[target] |= adj[v] - {target}
        for w in adj[v] - {target}:
            adj[w].add(target)
        for w in adj[v]:
            adj[w].discard(v)
        del adj[v]
    return {v: n for v, n in adj.items() if kinds[v] == "chi"}


def report_bounds(factors, kinds):
    adj = label_graph(factors)
    full = minor_min_width(adj)
    cm = chi_minor(adj, kinds)
    chi_lb = minor_min_width(cm)
    print(f"lower bounds (minor-min-width, minor-monotone => valid):")
    print(f"  full label graph (96 labels): treewidth >= {full}")
    print(f"  chi-minor ({len(cm)} chi labels) : treewidth >= {chi_lb}")
    return full, chi_lb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chi", type=int, default=24)
    parser.add_argument("--restarts", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check-current", action="store_true")
    parser.add_argument("--bounds", action="store_true",
                        help="report minor-min-width lower bounds")
    parser.add_argument("--minfill", type=int, default=0,
                        help="restarts of the min-fill elimination search")
    args = parser.parse_args()

    factors, dimensions = junction_network(args.chi)
    kinds = {label: label_kind(label, dimensions) for label in dimensions}
    counts = Counter(l for f in factors for l in f)
    assert set(counts.values()) == {2}, "network is not pairwise closed"
    nchi = sum(1 for l in kinds.values() if l == "chi")
    print(f"network: {len(factors)} factors, {len(dimensions)} labels "
          f"({nchi} chi, {len(dimensions)-nchi} d)")

    peak, step, lf, ok = simulate(CURRENT_PATH, factors, kinds)
    print(f"CURRENT path: peak = chi^{peak[0]} d^{peak[1]} at step {step}, "
          f"closes={ok}, log10(flops@chi24)={lf:.2f}")
    print(f"              peak doubles = {2**peak[1]} * chi^{peak[0]} "
          f"= {2**peak[1] * args.chi**peak[0] * 8 / 2**30:.1f} GiB at chi={args.chi}")
    if args.check_current:
        return

    if args.bounds:
        report_bounds(factors, kinds)

    rng = random.Random(args.seed)
    if args.minfill:
        best = search_min_fill(factors, kinds, rng, args.minfill)
        if best is None:
            raise SystemExit("min-fill found no closing path")
        key, path, peak, lf = best
        print(f"\nBEST(min-fill): peak = chi^{peak[0]} d^{peak[1]} = "
              f"{2**peak[1] * args.chi**peak[0] * 8 / 2**30:.2f} GiB at chi={args.chi}, "
              f"log10(flops@chi24)={lf:.2f}")
        print("path = (")
        for k in range(0, len(path), 8):
            print("    " + "".join(f"({i},{j})," for i, j in path[k:k+8]))
        print(")")
        return
    best = None
    for trial in range(args.restarts):
        temperature = 0.0 if trial % 20 == 0 else rng.choice([0.05, 0.2, 0.6])
        path = greedy_path(factors, kinds, rng, temperature)
        peak, step, lf, ok = simulate(path, factors, kinds)
        if not ok:
            continue
        key = (peak, lf)
        if best is None or key < best[0]:
            best = (key, path, peak, lf)
            print(f"  trial {trial:6d}: peak = chi^{peak[0]} d^{peak[1]}, "
                  f"log10(flops)={lf:.2f}")
    key, path, peak, lf = best
    print(f"\nBEST: peak = chi^{peak[0]} d^{peak[1]} "
          f"= {2**peak[1] * args.chi**peak[0] * 8 / 2**30:.2f} GiB at chi={args.chi}, "
          f"log10(flops@chi24)={lf:.2f}")
    print("path = (")
    for k in range(0, len(path), 8):
        print("    " + "".join(f"({i},{j})," for i, j in path[k:k+8]))
    print(")")


if __name__ == "__main__":
    main()
