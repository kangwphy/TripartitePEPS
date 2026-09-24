"""Small algebra checks for the research note; run via check_slurm.sh.

This does not implement or certify a fermionic LMPS/six-R contraction.
"""
from pathlib import Path
import itertools
import json
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
GAUSSIAN = HERE.parents[2] / "Gaussian" / "code"
sys.path.insert(0, str(GAUSSIAN))
from maj_tools import ops, random_bdg_state, cube_occupation, covariance
from cube_bdg import A_cube, gmat, PAIRS


def multiply(a, b):
    out = {}
    for u, x in a.items():
        for v, y in b.items():
            if set(u) & set(v):
                continue
            sign = (-1) ** sum(i > j for i in u for j in v)
            w = tuple(sorted(u + v))
            out[w] = out.get(w, 0) + sign * x * y
    return {w: x for w, x in out.items() if x != 0}


def add(*args):
    out = {}
    for a in args:
        for w, x in a.items():
            out[w] = out.get(w, 0) + x
    return {w: x for w, x in out.items() if x != 0}


def scale(a, x):
    return {w: x * z for w, z in a.items()}


def local_tensor_check():
    # Canonical MONOMIAL order [c^dagger, delta, beta, gamma, alpha].
    c, d, b, g, a = ({(i,): 1} for i in range(5))
    F = add(multiply(add(scale(a, 1j), b), add(scale(g, -1), scale(d, 1j))),
            multiply(a, b), multiply(g, d),
            multiply(c, add(scale(a, -1j), scale(b, -1), scale(g, -1), scale(d, 1j))))
    Q = add({(): 1}, F, scale(multiply(F, F), 0.5))
    assert not multiply(multiply(F, F), F)
    # Independent evaluation as 32 x 32 CAR operator matrices.
    annih = ops(5)
    C, D, B, G, A = [annih[0].conj().T, *annih[1:]]
    fm = (1j * A + B) @ (-G + 1j * D) + A @ B + G @ D
    fm = fm + C @ (-1j * A - B - G + 1j * D)
    qm = np.eye(32) + fm + fm @ fm / 2
    basis = [C, D, B, G, A]
    reconstructed = np.zeros((32, 32), dtype=complex)
    table = []
    for word, value in sorted(Q.items(), key=lambda z: (len(z[0]), z[0])):
        term = np.eye(32, dtype=complex)
        for i in word:
            term = term @ basis[i]
        reconstructed += value * term
        bits = [int(i in word) for i in range(5)]
        assert sum(bits) % 2 == 0
        table.append({"kdrul": "".join(map(str, bits)), "coefficient": str(complex(value))})
    error = float(np.max(np.abs(reconstructed - qm)))
    assert error == 0
    return Q, {"matrix_error": error, "nonzero_entries": len(Q), "table": table}


def permutation_sign(destination, parity):
    return (-1) ** sum(parity[i] * parity[j]
                      for i in range(len(parity)) for j in range(i + 1, len(parity))
                      if destination[i] > destination[j])


def check_permutations():
    checks = 0
    for dest in itertools.permutations(range(4)):
        for parity in itertools.product((0, 1), repeat=4):
            # Independently sort all wires by adjacent crossings.
            wires = list(range(4))
            sign = 1
            for end in range(3, 0, -1):
                for j in range(end):
                    if dest[wires[j]] > dest[wires[j + 1]]:
                        sign *= (-1) ** (parity[wires[j]] * parity[wires[j + 1]])
                        wires[j], wires[j + 1] = wires[j + 1], wires[j]
            assert sign == permutation_sign(dest, parity)
            checks += 1
    # Source order [1,2,3,4], output [3,4,1,2]: total parity of this
    # four-wire object may be even while the crossing sign is negative.
    assert permutation_sign((2, 3, 0, 1), (1, 0, 1, 0)) == -1
    swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0],
                     [0, 1, 0, 0], [0, 0, 0, 1]])
    fswap = swap @ np.diag([1, 1, 1, -1])
    occupied = np.diag([0, 1])
    two_copies = np.kron(occupied, occupied)
    ordinary = int(np.trace(two_copies @ swap))
    fermionic = int(np.trace(two_copies @ fswap))
    assert ordinary == 1 and fermionic == -1
    return {"permutation_parity_cases": checks, "even_object_negative_crossing": True,
            "occupied_region_ordinary_swap": ordinary, "occupied_region_fswap": fermionic}


def reorder_fock(psi, new_order):
    destination = [new_order.index(i) for i in range(len(new_order))]
    out = np.zeros_like(psi)
    for mask, amplitude in enumerate(psi):
        bits = [(mask >> i) & 1 for i in range(len(new_order))]
        target = sum(bits[old] << new for new, old in enumerate(new_order))
        out[target] = permutation_sign(destination, bits) * amplitude
    return out


def check_gaussian_oracle():
    glue = {"A": gmat(PAIRS["A"], (1, 1)),
            "B": gmat(PAIRS["B"], (1, -1)),
            "C": gmat(PAIRS["C"], (1, -1))}
    for x in glue:
        assert np.array_equal(glue[x] @ glue[x], -np.eye(4))
    assert np.array_equal(glue["A"] @ glue["B"], -glue["C"])
    errors = []
    order = [0, 2, 1, 3]
    majorder = [j for i in order for j in (2 * i, 2 * i + 1)]
    psi, gamma, _ = random_bdg_state(4, seed=20260909)
    grouped = reorder_fock(psi, order)
    grouped_gamma = gamma[np.ix_(majorder, majorder)]
    covariance_error = float(np.max(np.abs(covariance(grouped, ops(4)) - grouped_gamma)))
    assert covariance_error < 1e-12
    raw_cube = cube_occupation(psi, ["A", "B", "A", "C"])
    grouped_cube = cube_occupation(grouped, ["A", "A", "B", "C"])
    for n, state, cov, reg in [(4, grouped, grouped_gamma, ["A", "A", "B", "C"])]:
        z = cube_occupation(state, reg)
        phase, ld = np.linalg.slogdet(A_cube(cov, reg, glue))
        magnitude = np.exp(-2 * n * np.log(2) + ld / 2)
        errors.append(float(abs(magnitude - z)))
        assert abs(phase - 1) < 1e-10 and errors[-1] < 1e-12
    # A concrete map from Gaussian oracle permutations to the PEPS convention:
    # common left composition by A, followed by simultaneous conjugation.
    a, b, c = (2, 3, 0, 1), (1, 0, 3, 2), (3, 2, 1, 0)
    compose = lambda p, q: tuple(p[q[i]] for i in range(4))
    target = (tuple(range(4)), b, a)
    conjugators = []
    for r in itertools.permutations(range(4)):
        ri = tuple(r.index(i) for i in range(4))
        got = tuple(compose(compose(r, compose(a, s)), ri) for s in (a, b, c))
        if got == target:
            conjugators.append([i + 1 for i in r])
    assert conjugators
    return {"covariance_reordering_error": covariance_error,
            "grouped_cube_error": max(errors),
            "raw_interleaved_cube": [raw_cube.real, raw_cube.imag],
            "grouped_cube": [grouped_cube.real, grouped_cube.imag],
            "ordering_difference": float(abs(raw_cube - grouped_cube)),
            "replica_conjugator_destinations_1based": conjugators[0]}


def main():
    _, tensor = local_tensor_check()
    result = {"scope": "local CAR algebra and occupation/Gaussian conventions only; no six-R validation",
              "local_tensor": tensor, "permutations": check_permutations(),
              "gaussian_oracle": check_gaussian_oracle()}
    (HERE / "checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
