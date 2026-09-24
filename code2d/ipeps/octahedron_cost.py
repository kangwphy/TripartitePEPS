#!/usr/bin/env python3
"""Contraction cost of the six-R octahedron of refs/multientropy_PEPS.pdf.

gamma_2^(3) is the tensor trace of SIX R tensors on the octahedron K_{2,2,2}:
six vertices, twelve edges, every vertex of degree 4, each edge carrying the
PEPS BOUNDARY dimension chi.  Antipodal vertices are the two cycles of the same
seam and share no edge.  This searches the contraction exhaustively -- 6 tensors
is small enough to enumerate every order -- and reports the true peak and flops.
"""
import itertools, math

# antipodal pairs (0,1), (2,3), (4,5): every vertex joins all but its antipode
ANTI = {0:1, 1:0, 2:3, 3:2, 4:5, 5:4}
edges = [(i,j) for i in range(6) for j in range(i+1,6) if ANTI[i] != j]
assert len(edges) == 12
labels = {e: k for k, e in enumerate(edges)}
factors = [frozenset(labels[e] for e in edges if v in e) for v in range(6)]
for f in factors:
    assert len(f) == 4                       # degree 4

best = None
for order in itertools.permutations(range(1, 6)):      # fix vertex 0 first
    ops = [factors[0]]; used = {0}; peak = 0; fl = 0
    for v in order:
        a = ops[-1]; b = factors[v]
        com = a & b; out = (a | b) - com
        fl += len(out) + len(com)                       # exponents of chi
        peak = max(peak, len(out))
        ops.append(out)
    if ops[-1]:                                        # must close to a scalar
        continue
    key = (peak,)
    if best is None or key < best[0]:
        best = (key, peak, order)
print("sequential orders (absorb one R at a time):")
print(f"  best peak = chi^{best[1]}   order {(0,)+best[2]}")

# allow arbitrary binary trees, not just sequential absorption
import functools
@functools.lru_cache(maxsize=None)
def solve(mask):
    if bin(mask).count("1") == 1:
        v = mask.bit_length()-1
        return (len(factors[v]), factors[v])
    bestp, bestl = 10**9, None
    sub = (mask-1) & mask
    while sub:
        comp = mask ^ sub
        if sub < comp:
            p1, l1 = solve(sub); p2, l2 = solve(comp)
            out = (l1 | l2) - (l1 & l2)
            p = max(p1, p2, len(out))
            if p < bestp: bestp, bestl = p, out
        sub = (sub-1) & mask
    return bestp, bestl
p, l = solve(0b111111)
print(f"all binary trees        : optimal peak = chi^{p}, closes={not l}")
print()
print("cost of the optimal order, edge dimension chi:")
print(f"   peak storage : chi^{p}")
print(f"   arithmetic   : chi^{p+2}   (result chi^{p} x two summed edges)")
print()
for chi in (16, 24, 32, 48, 64, 96, 128):
    g = chi**p * 8 / 2**30
    print(f"   chi={chi:4d}:  storage {g:12.2f} GiB   flops {float(chi)**(p+2):.2e}")
