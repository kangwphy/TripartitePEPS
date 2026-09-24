#!/usr/bin/env python3
"""Find contraction paths for the replica-factorized 2x2 Y window.

This is a development helper only.  It represents every chi_1 and Ising
index separately, so no chi_1^4 edge is ever introduced.  The resulting
path is consumed by the dense labeled-tensor implementation in
``split_replica_ctmrg.jl`` after small-chi regression against the original
fused contraction.
"""

from __future__ import annotations

import argparse
from collections import Counter


PAIRS = {
    "AB": ((1, 2), (3, 4)),
    "CA": ((1, 3), (2, 4)),
    "CB": ((1, 4), (2, 3)),
}


def junction_network(chi: int):
    factors: list[tuple[str, ...]] = []
    dimensions: dict[str, int] = {}

    def variable(name: str, dimension: int) -> str:
        dimensions[name] = dimension
        return name

    def factor(*labels: str) -> None:
        factors.append(tuple(labels))

    for replica in range(1, 5):
        for boundary in (
            "n0", "n1", "n2", "e0", "e1", "e2",
            "s0", "s1", "s2", "w0", "w1", "w2",
        ):
            variable(f"{boundary}_{replica}", chi)
        factor(f"w2_{replica}", f"n0_{replica}")
        factor(f"n2_{replica}", f"e0_{replica}")
        factor(f"e2_{replica}", f"s2_{replica}")
        factor(f"s0_{replica}", f"w0_{replica}")

    site_legs: dict[str, dict[int, tuple[str, ...]]] = {
        site: {} for site in ("tl", "tr", "bl", "br")
    }
    for replica in range(1, 5):
        site_legs["tl"][replica] = (
            f"utl_{replica}", f"ltl_{replica}",
            f"vleft_{replica}", f"htop_{replica}",
        )
        site_legs["tr"][replica] = (
            f"utr_{replica}", f"htop_{replica}",
            f"vright_{replica}", f"rtr_{replica}",
        )
        site_legs["bl"][replica] = (
            f"vleft_{replica}", f"lbl_{replica}",
            f"dbl_{replica}", f"hbot_{replica}",
        )
        site_legs["br"][replica] = (
            f"vright_{replica}", f"hbot_{replica}",
            f"dbr_{replica}", f"rbr_{replica}",
        )
        for legs in site_legs.values():
            for label in legs[replica]:
                variable(label, 2)
        factor(*site_legs["tr"][replica])

    def edge(left: str, right: str, physical, sector: str | None = None):
        if sector is None:
            for replica in range(1, 5):
                factor(
                    f"{left}_{replica}", f"{right}_{replica}", physical(replica)
                )
            return
        for first, second in PAIRS[sector]:
            factor(
                f"{left}_{first}", f"{right}_{first}", physical(first),
                f"{left}_{second}", f"{right}_{second}", physical(second),
            )

    edge("n0", "n1", lambda r: f"utl_{r}", "AB")
    edge("n1", "n2", lambda r: f"utr_{r}")
    edge("e0", "e1", lambda r: f"rtr_{r}")
    edge("e1", "e2", lambda r: f"rbr_{r}", "CB")
    edge("s0", "s1", lambda r: f"dbl_{r}")
    edge("s1", "s2", lambda r: f"dbr_{r}")
    edge("w0", "w1", lambda r: f"lbl_{r}", "CA")
    edge("w1", "w2", lambda r: f"ltl_{r}")

    for site, sector in (("tl", "AB"), ("bl", "CA"), ("br", "CB")):
        for first, second in PAIRS[sector]:
            factor(*site_legs[site][first], *site_legs[site][second])

    counts = Counter(label for labels in factors for label in labels)
    if set(counts.values()) != {2}:
        raise RuntimeError("the closed junction network is not pairwise")
    return factors, dimensions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chi", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=256)
    args = parser.parse_args()

    try:
        import cotengra as ctg
    except ImportError as error:
        raise SystemExit("install cotengra in a temporary development environment") from error

    factors, dimensions = junction_network(args.chi)
    optimizer = ctg.HyperOptimizer(
        methods=["greedy", "random-greedy"],
        max_repeats=args.repeats,
        parallel=False,
        minimize="combo",
        progbar=False,
    )
    tree = optimizer.search(factors, (), dimensions)
    print(f"factors={len(factors)} variables={len(dimensions)}")
    print(f"max_size={tree.max_size()} total_flops={tree.total_flops()}")
    print(f"contraction_width={tree.contraction_width()}")
    print(f"path={tree.get_path()!r}")


if __name__ == "__main__":
    main()
