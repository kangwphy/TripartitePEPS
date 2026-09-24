#!/usr/bin/env python3
"""Every one of the 51 contraction steps, as actually performed.

No four-replica object is ever built.  The inputs are one-replica C1, T1, a1 and
two-replica seam objects; the cost of a step is

    flops ~ |output| x |summed indices|,

both measured in chi1 and D exponents.  This prints the step-by-step ledger and
identifies which handful of steps actually cost anything.
"""
from collections import defaultdict
from optimize_split_junction_path import junction_network
from optimize_split_junction_path2 import CURRENT_PATH, label_kind

factors, dims = junction_network(10)
kinds = {l: label_kind(l, dims) for l in dims}

names = []
for r in range(1, 5):
    names += [f"C[NW]r{r}", f"C[NE]r{r}", f"C[SE]r{r}", f"C[SW]r{r}"]
for r in range(1, 5):
    names.append(f"a[regionB]r{r}")
PAIRS = {"AB": ((1, 2), (3, 4)), "CA": ((1, 3), (2, 4)), "CB": ((1, 4), (2, 3))}
for lo, hi, sec, tag in (("n0","n1","AB","E*AB"), ("n1","n2",None,"T[N]"),
                         ("e0","e1",None,"T[E]"), ("e1","e2","CB","E*CB"),
                         ("s0","s1",None,"T[SW]"), ("s1","s2",None,"T[SE]"),
                         ("w0","w1","CA","E*CA"), ("w1","w2",None,"T[W]")):
    if sec is None:
        names += [f"{tag}r{r}" for r in range(1, 5)]
    else:
        names += [f"{tag}{p}" for p in PAIRS[sec]]
for site, sec, tag in (("tl","AB","A[regA]"), ("bl","CA","A[regC-CA]"), ("br","CB","A[regC-CB]")):
    names += [f"{tag}{p}" for p in PAIRS[sec]]

def ex(labels):
    nc = sum(1 for l in labels if kinds[l] == "chi")
    return nc, len(labels) - nc

def fmt(e):
    c, d = e
    a = f"chi^{c}" if c else ""
    b = f"D^{d}" if d else ""
    return (a + ("*" if a and b else "") + b) or "1"

ops = [frozenset(f) for f in factors]
prov = [{i} for i in range(52)]
print(f"{'st':>3} {'merges':<44} {'summed':>10} {'result':>12} {'flops':>13}")
print("-"*88)
costs = []
for step, (i, j) in enumerate(CURRENT_PATH):
    a, b = ops[i], ops[j]
    common = a & b
    out = (a | b) - common
    fl = (ex(out)[0] + ex(common)[0], ex(out)[1] + ex(common)[1])
    costs.append((step, fl, ex(out)))
    pa, pb = prov[i], prov[j]
    def tag(p):
        if len(p) == 1: return names[next(iter(p))]
        return f"<{len(p)} factors>"
    print(f"{step:3d} {tag(pa)+' x '+tag(pb):<44} {fmt(ex(common)):>10} "
          f"{fmt(ex(out)):>12} {fmt(fl):>13}")
    del ops[j]; del ops[i]; ops.append(out)
    del prov[j]; del prov[i]; prov.append(pa | pb)

print("\n" + "="*88)
CHI = 24.0
tot = sum(CHI**c * 2.0**d for _, (c, d), _ in costs)
print(f"total flops at chi1=24: {tot:.3e}\n")
print("steps ranked by cost (top 12) -- this is where the time goes:")
for step, fl, o in sorted(costs, key=lambda x: -(CHI**x[1][0] * 2.0**x[1][1]))[:12]:
    f = CHI**fl[0] * 2.0**fl[1]
    print(f"  step {step:3d}  flops {fmt(fl):>12} = {f:10.3e}  ({100*f/tot:5.1f}% of total)"
          f"   result {fmt(o)}")
by = defaultdict(float)
for _, fl, _ in costs:
    by[fl[0]] += CHI**fl[0] * 2.0**fl[1]
print("\nflops grouped by the chi exponent of the step:")
for c in sorted(by, reverse=True):
    print(f"  chi^{c}: {by[c]:10.3e}  ({100*by[c]/tot:5.1f}%)")
