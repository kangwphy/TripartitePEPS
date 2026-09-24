#!/usr/bin/env python3
"""Strictly pool the near-critical SW campaign and plot its plateaus."""

import csv
import math
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(
    ROOT, "data/rk_ising/mc", "campaigns",
    "sw_components_near_critical_100k_16chains",
)
BETA_C = 0.5 * math.log(1.0 + math.sqrt(2.0))
BETAS = ("0.36", "0.38", "0.40", "0.42", "0.46", "0.48", "0.50", "0.52", "0.54")
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
OBSERVABLES = ("S2A", "S2B", "S2C", "S3", "tS")
BETA_OFFSET = {beta: 200000 * index for index, beta in enumerate(BETAS)}
OBSERVABLE_OFFSET = {"S2A": 0, "S2B": 100, "S2C": 200, "S3": 300, "tS": 400}


def expected_seeds(beta, size, observable):
    """Return the 16 seeds assigned to one beta/size/observable point."""
    base = 4000000 + BETA_OFFSET[beta] + 1000 * size + OBSERVABLE_OFFSET[observable]
    return {base + chain for chain in range(1, 17)}


def xi_spin(beta):
    """Exact connected-spin exponential correlation length along an axis."""
    beta_dual = -0.5 * math.log(math.tanh(beta))
    if beta < BETA_C:
        return 1.0 / (2.0 * (beta_dual - beta))
    return 1.0 / (4.0 * (beta - beta_dual))


def pool(values):
    """Pool independent chains, guarding against chain-to-chain scatter."""
    weights = [1.0 / error**2 for _, error in values]
    mean = sum(value * weight for (value, _), weight in zip(values, weights))
    mean /= sum(weights)
    formal = 1.0 / math.sqrt(sum(weights))
    chain_sem = math.sqrt(
        sum((value - mean) ** 2 for value, _ in values)
        / (len(values) * (len(values) - 1))
    )
    return mean, max(formal, chain_sem), formal, chain_sem


def constant_fit(points):
    """Fit a constant to value/error pairs and return chi-square per dof."""
    weights = [1.0 / error**2 for _, error in points]
    mean = sum(value * weight for (value, _), weight in zip(points, weights))
    mean /= sum(weights)
    error = 1.0 / math.sqrt(sum(weights))
    chi2 = sum(((value - mean) / sigma) ** 2 for value, sigma in points)
    return mean, error, chi2 / (len(points) - 1)


def read_and_validate():
    """Read aggregate rows and require exact campaign parameters and coverage."""
    raw = defaultdict(list)
    path = os.path.join(OUT, "all_chains.csv")
    with open(path, newline="") as stream:
        for row in csv.DictReader(stream):
            beta = f"{float(row['beta']):.2f}"
            size = int(row["L"])
            observable = row["observable"]
            if beta not in BETAS or size not in SIZES[beta]:
                raise ValueError(f"unexpected beta/L row: {row}")
            if observable not in OBSERVABLES:
                raise ValueError(f"unexpected observable row: {row}")
            if (int(row["n_sweeps"]) != 100000
                    or int(row["n_eq"]) != 100000
                    or int(row["n_lambda"]) != 12
                    or row["update"] != "sw"
                    or row["quantity"] != observable
                    or row["gauge"] != "user"):
                raise ValueError(f"nonstandard campaign row: {row}")
            raw[(beta, size, observable)].append({
                "seed": int(row["seed"]),
                "value": float(row["value"]),
                "error": float(row["error"]),
            })

    for beta in BETAS:
        for size in SIZES[beta]:
            for observable in OBSERVABLES:
                rows = raw[(beta, size, observable)]
                seeds = {row["seed"] for row in rows}
                expected = expected_seeds(beta, size, observable)
                if len(rows) != 16 or seeds != expected:
                    raise RuntimeError(
                        f"incomplete/wrong seeds at {beta=} {size=} {observable=}: "
                        f"rows={len(rows)}, missing={sorted(expected - seeds)}, "
                        f"unexpected={sorted(seeds - expected)}"
                    )
                if any(row["error"] <= 0.0 for row in rows):
                    raise RuntimeError(
                        f"nonpositive error at {beta=} {size=} {observable=}"
                    )
    return raw


def write_tables(raw):
    """Write strict coverage, pooled observables, identities, and plateaus."""
    with open(os.path.join(OUT, "coverage.csv"), "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["beta", "L", *OBSERVABLES, "complete_16_each"])
        for beta in BETAS:
            for size in SIZES[beta]:
                counts = [len(raw[(beta, size, obs)]) for obs in OBSERVABLES]
                writer.writerow([beta, size, *counts, all(n == 16 for n in counts)])

    pooled = {}
    with open(os.path.join(OUT, "pooled_components.csv"), "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "beta", "L", "observable", "value", "error", "formal_error",
            "chain_sem", "n_chains", "complete_16",
        ])
        for beta in BETAS:
            for size in SIZES[beta]:
                for observable in OBSERVABLES:
                    rows = raw[(beta, size, observable)]
                    result = pool([(row["value"], row["error"]) for row in rows])
                    pooled[(beta, size, observable)] = result
                    writer.writerow([beta, size, observable, *result, 16, True])

    comparisons = []
    with open(os.path.join(OUT, "reconstruction_vs_direct.csv"), "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "beta", "L", "reconstructed_tildeS", "reconstructed_error",
            "direct_tildeS", "direct_error", "difference", "difference_error",
            "z_score",
        ])
        for beta in BETAS:
            for size in SIZES[beta]:
                p = {obs: pooled[(beta, size, obs)] for obs in OBSERVABLES}
                reconstructed = (
                    2.0 * p["S3"][0] - p["S2A"][0]
                    - p["S2B"][0] - p["S2C"][0]
                )
                reconstructed_error = math.sqrt(
                    (2.0 * p["S3"][1]) ** 2 + p["S2A"][1] ** 2
                    + p["S2B"][1] ** 2 + p["S2C"][1] ** 2
                )
                difference = reconstructed - p["tS"][0]
                difference_error = math.hypot(reconstructed_error, p["tS"][1])
                row = (
                    beta, size, reconstructed, reconstructed_error,
                    p["tS"][0], p["tS"][1], difference, difference_error,
                    difference / difference_error,
                )
                comparisons.append(row)
                writer.writerow(row)

    plateau_rows = []
    for beta in BETAS:
        beta_rows = [row for row in comparisons if row[0] == beta]
        tail = beta_rows[-3:]
        direct = constant_fit([(row[4], row[5]) for row in tail])
        reconstructed = constant_fit([(row[2], row[3]) for row in tail])
        correlation_length = xi_spin(float(beta))
        max_identity_z = max(abs(row[8]) for row in tail)
        plateau_rows.append({
            "beta": beta,
            "phase": "disordered" if float(beta) < BETA_C else "ordered",
            "xi_spin": correlation_length,
            "tail_sizes": ";".join(str(row[1]) for row in tail),
            "tail_min_L_over_xi": tail[0][1] / correlation_length,
            "direct_plateau": direct[0],
            "direct_error": direct[1],
            "direct_chi2_dof": direct[2],
            "reconstructed_plateau": reconstructed[0],
            "reconstructed_error": reconstructed[1],
            "reconstructed_chi2_dof": reconstructed[2],
            "max_identity_abs_z": max_identity_z,
            "direct_certified": (
                tail[0][1] / correlation_length >= 8.0
                and direct[2] <= 3.0 and max_identity_z <= 3.0
            ),
        })
    with open(os.path.join(OUT, "plateau_summary.csv"), "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(plateau_rows[0]))
        writer.writeheader()
        writer.writerows(plateau_rows)
    return pooled, plateau_rows


def plot_plateaus(plateau_rows):
    """Plot linear, logarithmic-coordinate, and log-log plateau views."""
    by_phase = {
        phase: [row for row in plateau_rows if row["phase"] == phase]
        for phase in ("disordered", "ordered")
    }
    phases = (
        ("disordered", "#2e5fa3", "o"),
        ("ordered", "#c4552d", "s"),
    )
    offsets = {beta: (5, 5) for beta in BETAS}
    offsets.update({"0.36": (5, -13), "0.46": (5, -13), "0.54": (5, -13)})

    fig, axes = plt.subplots(1, 3, figsize=(16.3, 4.7), dpi=180)
    transforms = (
        (lambda row: row["xi_spin"], r"$\xi_{\rm spin}(\beta)$",
         "(a) linear--linear"),
        (lambda row: math.log(row["xi_spin"]),
         r"$\ln \xi_{\rm spin}(\beta)$", "(b) log coordinate--linear"),
    )
    for axis, (x_value, x_label, title) in zip(axes[:2], transforms):
        for phase, color, marker in phases:
            rows = by_phase[phase]
            for row in rows:
                axis.errorbar(
                    x_value(row), row["direct_plateau"],
                    yerr=row["direct_error"], fmt=marker, ms=5.5,
                    capsize=2.5, color=color,
                    markerfacecolor=(color if row["direct_certified"] else "none"),
                    markeredgewidth=1.2,
                )
                axis.annotate(
                    rf"$\beta={row['beta']}$",
                    (x_value(row), row["direct_plateau"]),
                    xytext=offsets[row["beta"]], textcoords="offset points",
                    fontsize=7.2,
                )
        axis.axhline(0.0, color="0.65", ls=":", lw=0.9)
        axis.set_xlabel(x_label)
        axis.set_title(title)
        axis.grid(alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(r"tail constant-fit estimate of $\widetilde S(\beta)$")
    axes[0].legend(
        handles=[
            Line2D([], [], marker=marker, ls="none", color=color, label=phase)
            for phase, color, marker in phases
        ] + [
            Line2D([], [], marker="o", ls="none", color="0.25",
                   markerfacecolor="none", label="not certified")
        ],
        frameon=False, fontsize=8,
    )
    axes[1].set_ylim(axes[0].get_ylim())

    axis = axes[2]
    for phase, color, marker in phases:
        axis.plot([], [], marker=marker, ls="none", color=color, label=phase)
        for row in by_phase[phase]:
            x_point = row["xi_spin"]
            value = row["direct_plateau"]
            error = row["direct_error"]
            if value - error > 0.0:
                y_point = value
                axis.errorbar(
                    x_point, y_point, yerr=error, fmt=marker,
                    ms=5.5, capsize=2.5, color=color,
                    markerfacecolor=(color if row["direct_certified"] else "none"),
                    markeredgewidth=1.2,
                )
            else:
                y_point = value + 1.645 * error
                if y_point <= 0.0:
                    continue
                axis.scatter(
                    [x_point], [y_point], marker="v", s=38,
                    facecolors="none", edgecolors=color, linewidths=1.2,
                )
            axis.annotate(
                rf"$\beta={row['beta']}$", (x_point, y_point),
                xytext=offsets[row["beta"]], textcoords="offset points",
                fontsize=7.2,
            )
    axis.plot(
        [], [], marker="v", ls="none", markerfacecolor="none",
        markeredgecolor="0.25", color="0.25", label="95% upper limit",
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(r"$\xi_{\rm spin}(\beta)$")
    axis.set_ylabel(r"tail constant-fit estimate of $\widetilde S(\beta)$")
    axis.set_title("(c) log--log")
    axis.grid(alpha=0.22, which="both")
    axis.spines[["top", "right"]].set_visible(False)
    axis.text(
        0.02, 0.02,
        "Negative estimates are omitted on the logarithmic y axis; see (a,b).",
        transform=axis.transAxes, fontsize=7.2, va="bottom",
    )
    axis.legend(frameon=False, fontsize=8)
    fig.suptitle("Near-critical Swendsen--Wang tail fits")
    fig.tight_layout()
    fig.savefig(
        os.path.join(OUT, "near_only_plateau_three_axis_views.png"),
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    raw = read_and_validate()
    _, plateau_rows = write_tables(raw)
    plot_plateaus(plateau_rows)
    print(f"strictly validated and plotted {len(plateau_rows)} near-critical betas")
    for row in plateau_rows:
        print(
            f"beta={row['beta']} xi={row['xi_spin']:.6f} "
            f"tail={row['tail_sizes']} "
            f"plateau={row['direct_plateau']:+.7f}+-{row['direct_error']:.7f} "
            f"chi2/dof={row['direct_chi2_dof']:.2f} "
            f"identity|max z|={row['max_identity_abs_z']:.2f} "
            f"certified={row['direct_certified']}"
        )


if __name__ == "__main__":
    main()
