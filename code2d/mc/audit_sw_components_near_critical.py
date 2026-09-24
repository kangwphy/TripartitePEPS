#!/usr/bin/env python3
"""Audit strict chain coverage for the near-critical SW component campaign."""

import csv
import math
import os


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CAMPAIGN = os.path.join(
    ROOT, "data/rk_ising/mc", "campaigns",
    "sw_components_near_critical_100k_16chains",
)
BETAS = ("0.36", "0.38", "0.40", "0.42", "0.46", "0.48", "0.50", "0.52", "0.54")
OBSERVABLES = ("S2A", "S2B", "S2C", "S3", "tS")
SIZES = {
    "0.36": (4, 8, 12, 16, 20, 24, 28, 32, 36, 40),
    "0.38": (4, 8, 12, 16, 20, 24, 28, 32, 36, 40),
    "0.40": (4, 8, 12, 16, 24, 32, 40, 48, 56, 64),
    "0.42": (4, 8, 12, 16, 24, 32, 48, 64, 80, 96, 112, 128),
    "0.46": (4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 96, 112, 128),
    "0.48": (4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 72, 80, 96, 112, 128),
    "0.50": (4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 72, 80, 96, 112, 128),
    "0.52": (4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 72, 80, 96, 112, 128),
    "0.54": (4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 72, 80, 96, 112, 128),
}
BETA_OFFSET = {beta: 200000 * index for index, beta in enumerate(BETAS)}
OBSERVABLE_OFFSET = {"S2A": 0, "S2B": 100, "S2C": 200, "S3": 300, "tS": 400}


def expected_seed(beta, size, observable, chain):
    """Return the deterministic seed assigned to one independent chain."""
    return 4000000 + BETA_OFFSET[beta] + 1000 * size + OBSERVABLE_OFFSET[observable] + chain


def result_path(beta, size, observable, seed):
    """Return the unique CSV path associated with an expected chain."""
    beta_key = beta.replace(".", "p")
    name = f"L{size:03d}_b{beta_key}_n100000_eq100000_seed{seed}.csv"
    return os.path.join(CAMPAIGN, observable, "raw", name)


def valid_result(path, beta, size, observable, seed):
    """Apply the same strict metadata checks used by the recovery submitter."""
    try:
        with open(path, newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 1:
            return False
        row = rows[0]
        float(row["tildeS"])
        float(row["error"])
        return (
            int(row["L"]) == size
            and math.isclose(float(row["beta"]), float(beta), rel_tol=0.0, abs_tol=1e-14)
            and int(row["n_sweeps"]) == 100000
            and int(row["n_eq"]) == 100000
            and int(row["seed"]) == seed
            and int(row["n_gl"]) == 12
            and row["update"] == "sw"
            and row["quantity"] == observable
            and row["gauge"] == "user"
        )
    except (FileNotFoundError, KeyError, ValueError):
        return False


def audit():
    """Write per-size and per-beta coverage tables and print a compact audit."""
    size_rows = []
    beta_rows = []
    total_expected = 0
    total_valid = 0
    for beta in BETAS:
        beta_expected = 0
        beta_valid = 0
        complete_all = []
        for size in SIZES[beta]:
            counts = {}
            for observable in OBSERVABLES:
                valid = sum(
                    valid_result(
                        result_path(beta, size, observable,
                                    expected_seed(beta, size, observable, chain)),
                        beta, size, observable,
                        expected_seed(beta, size, observable, chain),
                    )
                    for chain in range(1, 17)
                )
                counts[observable] = valid
                beta_expected += 16
                beta_valid += valid
            is_complete = all(counts[q] == 16 for q in OBSERVABLES)
            if is_complete:
                complete_all.append(size)
            size_rows.append([
                beta, size, *(counts[q] for q in OBSERVABLES),
                sum(counts.values()), 80, "complete" if is_complete else "incomplete",
            ])
        total_expected += beta_expected
        total_valid += beta_valid
        beta_rows.append([
            beta, beta_valid, beta_expected, beta_expected - beta_valid,
            len(complete_all), len(SIZES[beta]),
            ";".join(map(str, complete_all)),
        ])

    with open(os.path.join(CAMPAIGN, "coverage_by_size_current.csv"), "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "beta", "L", *[f"valid_{q}" for q in OBSERVABLES],
            "valid_total", "expected_total", "status",
        ])
        writer.writerows(size_rows)
    with open(os.path.join(CAMPAIGN, "coverage_by_beta_current.csv"), "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "beta", "valid_chains", "expected_chains", "missing_chains",
            "complete_sizes", "planned_sizes", "complete_size_list",
        ])
        writer.writerows(beta_rows)

    print("beta  valid/expected  complete sizes")
    for beta, valid, expected, _, n_complete, n_sizes, complete in beta_rows:
        print(f"{beta:>4}  {valid:4d}/{expected:<4d}    {n_complete:2d}/{n_sizes:<2d}  {complete or '-'}")
    print(f"TOTAL {total_valid}/{total_expected}; missing {total_expected - total_valid}")
    return total_valid == total_expected


if __name__ == "__main__":
    raise SystemExit(0 if audit() else 1)
