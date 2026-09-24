#!/usr/bin/env python3
"""Fixed-network certificate for the 52-factor versus six-R regrouping.

This does not solve CTMRG.  It uses the exact label graph of
``junction_window_pair_log`` and generic dense tensors with the same dimensions.
The twelve pair edge/site factors are first combined into the six 2-cycle
cores.  The remaining one-replica factors form twelve region--replica bridges;
parallel direct core bonds are included in the same bridge.  Every bridge is
SVD-factorized without truncation and its two halves are absorbed into the
adjacent cores.  The resulting six tensors must form K_{2,2,2} and contract to
the same scalar as the original 52-factor graph.

The certificate also reports the exact generic bridge ranks.  These ranks,
not the one-copy CTMRG chi by itself, are the edge dimensions of a dense six-R
octahedron made from the current 2x2 closure.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from optimize_split_junction_path import junction_network


@dataclass
class Tensor:
    data: np.ndarray
    labels: list[str]
    name: str


def contract(left: Tensor, right: Tensor, name: str) -> Tensor:
    right_positions = {label: i for i, label in enumerate(right.labels)}
    common = [label for label in left.labels if label in right_positions]
    left_axes = [left.labels.index(label) for label in common]
    right_axes = [right_positions[label] for label in common]
    data = np.tensordot(left.data, right.data, axes=(left_axes, right_axes))
    labels = [label for label in left.labels if label not in common]
    labels += [label for label in right.labels if label not in common]
    return Tensor(np.asarray(data), labels, name)


def outer(left: Tensor, right: Tensor, name: str) -> Tensor:
    if set(left.labels) & set(right.labels):
        raise RuntimeError("outer product received a shared label")
    data = np.tensordot(left.data, right.data, axes=0)
    return Tensor(np.asarray(data), left.labels + right.labels, name)


def contract_path(factors: list[Tensor], path: tuple[tuple[int, int], ...]) -> float:
    work = [Tensor(np.array(t.data, copy=True), list(t.labels), t.name) for t in factors]
    for step, (i, j) in enumerate(path, 1):
        if not i < j < len(work):
            raise RuntimeError(f"invalid path entry {(i, j)} at step {step}")
        result = contract(work[i], work[j], f"raw_{step}")
        del work[j]
        del work[i]
        work.append(result)
    if len(work) != 1 or work[0].labels:
        raise RuntimeError("raw path did not close the graph")
    return float(work[0].data)


# Exact production path in split_replica_ctmrg.jl.
RAW_PATH = (
    (0, 20), (3, 50), (40, 49), (39, 42), (46, 47), (3, 24),
    (19, 45), (14, 44), (42, 43), (2, 33), (7, 41), (25, 40),
    (23, 34), (37, 38), (36, 37), (1, 21), (8, 35), (22, 34),
    (24, 30), (31, 32), (0, 13), (14, 30), (8, 29), (27, 28),
    (26, 27), (10, 12), (10, 25), (2, 24), (4, 23), (16, 22),
    (16, 21), (16, 20), (7, 19), (4, 8), (16, 17), (2, 6),
    (4, 15), (13, 14), (12, 13), (2, 4), (0, 11), (5, 10),
    (7, 9), (7, 8), (1, 5), (0, 6), (1, 5), (0, 1), (0, 3),
    (1, 2), (0, 1),
)


# Zero-based factor positions.  Each pair is one defect edge plus the dressed
# core site carrying the same 2-cycle.
CORE_GROUPS = {
    "AB12": (20, 46),
    "AB34": (21, 47),
    "CA13": (40, 48),
    "CA24": (41, 49),
    "CB14": (30, 50),
    "CB23": (31, 51),
}


def random_network(chi: int, physical: int, seed: int) -> tuple[list[Tensor], dict[str, int]]:
    label_lists, dimensions = junction_network(chi)
    # junction_network fixes every microscopic Ising leg to dimension two.
    if physical != 2:
        for label in list(dimensions):
            if dimensions[label] == 2 and not label[0] in "nesw":
                dimensions[label] = physical
    rng = np.random.default_rng(seed)
    factors: list[Tensor] = []
    for i, labels in enumerate(label_lists):
        shape = tuple(dimensions[label] for label in labels)
        scale = np.sqrt(max(np.prod(shape), 1))
        factors.append(Tensor(rng.standard_normal(shape) / scale, list(labels), f"f{i+1}"))
    return factors, dimensions


def connected_components(factors: list[Tensor]) -> list[list[int]]:
    label_to_factors: dict[str, list[int]] = defaultdict(list)
    for i, factor in enumerate(factors):
        for label in factor.labels:
            label_to_factors[label].append(i)
    adjacency = [set() for _ in factors]
    for indices in label_to_factors.values():
        for i in indices:
            adjacency[i].update(j for j in indices if j != i)
    seen: set[int] = set()
    components: list[list[int]] = []
    for start in range(len(factors)):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component: list[int] = []
        while stack:
            i = stack.pop()
            component.append(i)
            for j in adjacency[i]:
                if j not in seen:
                    seen.add(j)
                    stack.append(j)
        components.append(sorted(component))
    return components


def contract_component(factors: list[Tensor], indices: list[int], name: str) -> Tensor:
    work = [factors[i] for i in indices]
    while len(work) > 1:
        best: tuple[int, int] | None = None
        for i, j in combinations(range(len(work)), 2):
            if set(work[i].labels) & set(work[j].labels):
                best = (i, j)
                break
        if best is None:
            result = outer(work[0], work[1], name)
            del work[1]
            work[0] = result
            continue
        i, j = best
        result = contract(work[i], work[j], name)
        del work[j]
        del work[i]
        work.append(result)
    return work[0]


def six_r_regroup(factors: list[Tensor], dimensions: dict[str, int]):
    core_factor_indices = {i for group in CORE_GROUPS.values() for i in group}
    cores = {
        name: contract(factors[i], factors[j], name)
        for name, (i, j) in CORE_GROUPS.items()
    }

    # Split every direct core--core label into two endpoint labels and insert an
    # explicit identity.  This lets it be combined with the outer bridge that
    # connects the same pair of cores.
    label_to_cores: dict[str, list[str]] = defaultdict(list)
    for core_name, core in cores.items():
        for label in core.labels:
            label_to_cores[label].append(core_name)

    bridge_factors = [
        Tensor(np.array(factors[i].data, copy=True), list(factors[i].labels), factors[i].name)
        for i in range(len(factors)) if i not in core_factor_indices
    ]
    for label, owners in list(label_to_cores.items()):
        if len(owners) != 2:
            continue
        left_name, right_name = sorted(owners)
        left_label = f"{label}@{left_name}"
        right_label = f"{label}@{right_name}"
        left_core = cores[left_name]
        right_core = cores[right_name]
        left_core.labels[left_core.labels.index(label)] = left_label
        right_core.labels[right_core.labels.index(label)] = right_label
        dimension = dimensions[label]
        bridge_factors.append(
            Tensor(np.eye(dimension), [left_label, right_label], f"delta_{label}")
        )

    # Recompute the endpoint ownership after the direct labels were renamed.
    endpoint_owner: dict[str, str] = {}
    for core_name, core in cores.items():
        for label in core.labels:
            if label in endpoint_owner:
                raise RuntimeError(f"label {label} still joins two cores directly")
            endpoint_owner[label] = core_name

    # Contract each connected bridge component, then combine parallel
    # components belonging to the same unordered core pair.
    grouped: dict[tuple[str, str], list[Tensor]] = defaultdict(list)
    for component_id, indices in enumerate(connected_components(bridge_factors), 1):
        bridge = contract_component(bridge_factors, indices, f"bridge_component_{component_id}")
        owners = sorted({endpoint_owner[label] for label in bridge.labels if label in endpoint_owner})
        if len(owners) != 2:
            raise RuntimeError(f"bridge component touches {owners}, expected exactly two cores")
        grouped[(owners[0], owners[1])].append(bridge)
    if len(grouped) != 12:
        raise RuntimeError(f"expected 12 octahedron edges, found {len(grouped)}")

    ranks: dict[tuple[str, str], int] = {}
    singular_values: dict[tuple[str, str], np.ndarray] = {}
    for edge_id, (owners, pieces) in enumerate(sorted(grouped.items()), 1):
        bridge = pieces[0]
        for piece in pieces[1:]:
            bridge = outer(bridge, piece, f"bridge_{edge_id}")
        left_name, right_name = owners
        left_labels = [label for label in bridge.labels if endpoint_owner.get(label) == left_name]
        right_labels = [label for label in bridge.labels if endpoint_owner.get(label) == right_name]
        if set(left_labels + right_labels) != set(bridge.labels):
            raise RuntimeError("bridge contains a label not owned by either endpoint")
        permutation = [bridge.labels.index(label) for label in left_labels + right_labels]
        data = np.transpose(bridge.data, permutation)
        left_shape = tuple(dimensions[label.split("@")[0]] if "@" in label else dimensions[label]
                           for label in left_labels)
        right_shape = tuple(dimensions[label.split("@")[0]] if "@" in label else dimensions[label]
                            for label in right_labels)
        matrix = np.reshape(data, (int(np.prod(left_shape)), int(np.prod(right_shape))))
        u, s, vh = np.linalg.svd(matrix, full_matrices=False)
        tolerance = max(matrix.shape) * np.finfo(float).eps * (s[0] if len(s) else 0.0)
        rank = int(np.count_nonzero(s > tolerance))
        u = u[:, :rank]
        s = s[:rank]
        vh = vh[:rank, :]
        rho_label = f"rho_{edge_id}"
        dimensions[rho_label] = rank
        root = np.sqrt(s)
        left = Tensor(np.reshape(u * root, left_shape + (rank,)), left_labels + [rho_label],
                      f"left_{edge_id}")
        right_matrix = vh.T * root
        right = Tensor(np.reshape(right_matrix, right_shape + (rank,)),
                       right_labels + [rho_label], f"right_{edge_id}")
        cores[left_name] = contract(cores[left_name], left, left_name)
        cores[right_name] = contract(cores[right_name], right, right_name)
        ranks[owners] = rank
        singular_values[owners] = s

    if any(len(core.labels) != 4 for core in cores.values()):
        raise RuntimeError(f"regrouped cores are not rank four: "
                           f"{ {name: len(core.labels) for name, core in cores.items()} }")
    degree = {name: set() for name in cores}
    for left_name, right_name in ranks:
        degree[left_name].add(right_name)
        degree[right_name].add(left_name)
    if any(len(neighbours) != 4 for neighbours in degree.values()):
        raise RuntimeError(f"six cores do not form K_{{2,2,2}}: {degree}")
    return cores, ranks, singular_values


def contract_six_r(cores: dict[str, Tensor]) -> float:
    work = list(cores.values())
    expected_shared = (1, 2, 2, 3, 4)
    partial = work.pop(0)
    for step, expected in enumerate(expected_shared, 1):
        candidates = []
        for i, tensor in enumerate(work):
            shared = len(set(partial.labels) & set(tensor.labels))
            if shared:
                output_size = np.prod(
                    [partial.data.shape[partial.labels.index(label)]
                     for label in partial.labels if label not in tensor.labels]
                    + [tensor.data.shape[tensor.labels.index(label)]
                       for label in tensor.labels if label not in partial.labels]
                )
                candidates.append((-shared, output_size, i))
        _, _, index = min(candidates)
        tensor = work.pop(index)
        shared = len(set(partial.labels) & set(tensor.labels))
        if shared != expected:
            raise RuntimeError(f"unexpected octahedron step {step}: shared {shared}, expected {expected}")
        partial = contract(partial, tensor, f"octa_{step}")
    if partial.labels:
        raise RuntimeError("six-R contraction left open labels")
    return float(partial.data)


def rank_summary(ranks: dict[tuple[str, str], int]) -> dict[str, list[int]]:
    summary: dict[str, list[int]] = {"A": [], "B": [], "C": []}
    for (left, right), rank in ranks.items():
        seams = {left[:2], right[:2]}
        # The common region of the two seam names labels this octahedron edge.
        if seams == {"AB", "CA"}:
            summary["A"].append(rank)
        elif seams == {"AB", "CB"}:
            summary["B"].append(rank)
        elif seams == {"CA", "CB"}:
            summary["C"].append(rank)
        else:
            raise RuntimeError(f"cannot classify edge {left}--{right}")
    return {region: sorted(values) for region, values in summary.items()}


def main() -> None:
    for chi in (2, 3):
        factors, dimensions = random_network(chi, 2, seed=20260816 + chi)
        raw = contract_path(factors, RAW_PATH)
        cores, ranks, _ = six_r_regroup(factors, dimensions)
        octa = contract_six_r(cores)
        relative = abs(raw - octa) / max(abs(raw), np.finfo(float).tiny)
        summary = rank_summary(ranks)
        print(
            f"chi={chi} raw={raw:+.16e} sixR={octa:+.16e} "
            f"rel={relative:.3e} ranks_A={summary['A']} "
            f"ranks_B={summary['B']} ranks_C={summary['C']}"
        )
        expected = {"A": [2 * chi] * 4, "B": [2 * chi] * 4, "C": [2 * chi] * 4}
        if summary != expected:
            raise RuntimeError(f"generic bridge ranks {summary} != expected {expected}")
        if relative > 5e-11:
            raise RuntimeError(f"52-factor/six-R identity failed at chi={chi}: {relative}")
    print("PASS: exact 52-factor -> six-R regrouping; every dense octahedron "
          "edge has generic rank rho=2chi for the current Ising 2x2 geometry")


if __name__ == "__main__":
    main()
