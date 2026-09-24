#!/usr/bin/env python3
"""Pool the additional-beta SW campaign and test large-L plateaus."""

import csv
import math
import os
from collections import defaultdict

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MATPLOTLIB = True
except ModuleNotFoundError:
    # Some compute-node Python installations lack matplotlib dependencies.
    # The recovery job must still be able to validate and pool completed MC
    # chains; figures can be regenerated on the login node afterward.
    HAVE_MATPLOTLIB = False


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(
    ROOT, "data/rk_ising/mc", "campaigns",
    "sw_components_extra_betas_100k_16chains",
)
BETAS = ("0.10", "0.20", "0.25", "0.35", "0.70", "0.80")
SIZES = {
    "0.10": (4, 8, 12, 16, 20, 24),
    "0.20": (4, 8, 12, 16, 20, 24),
    "0.25": (4, 8, 12, 16, 20, 24),
    "0.35": (4, 8, 12, 16, 20, 24, 32, 40, 48),
    "0.70": (4, 8, 12, 16, 20, 24),
    "0.80": (4, 8, 12, 16, 20, 24),
}
OBSERVABLES = ("S2A", "S2B", "S2C", "S3", "tS")
BETA_C = 0.5 * math.log(1.0 + math.sqrt(2.0))


def xi_spin(beta):
    """Exact connected-spin exponential correlation length along an axis."""
    beta_dual = -0.5 * math.log(math.tanh(beta))
    if beta < BETA_C:
        return 1.0 / (2.0 * (beta_dual - beta))
    return 1.0 / (4.0 * (beta - beta_dual))


def pool(values):
    """Pool independent chains and retain the larger of two error estimates."""
    weights = [1.0 / error**2 for _, error in values]
    mean = sum(value * weight for (value, _), weight in zip(values, weights))
    mean /= sum(weights)
    formal = 1.0 / math.sqrt(sum(weights))
    if len(values) > 1:
        chain_sem = math.sqrt(
            sum((value - mean) ** 2 for value, _ in values)
            / (len(values) * (len(values) - 1))
        )
    else:
        chain_sem = formal
    return mean, max(formal, chain_sem), formal, chain_sem


def constant_fit(points):
    """Fit a constant to (value,error) pairs and return mean,error,chi2/dof."""
    weights = [1.0 / error**2 for value, error in points]
    mean = sum(value * weight for (value, _), weight in zip(points, weights))
    mean /= sum(weights)
    error = 1.0 / math.sqrt(sum(weights))
    chi2 = sum(((value - mean) / sigma) ** 2 for value, sigma in points)
    dof = max(1, len(points) - 1)
    return mean, error, chi2 / dof


def read_raw():
    """Read the aggregate cache and verify run parameters."""
    raw = defaultdict(list)
    with open(os.path.join(OUT, "all_chains.csv"), newline="") as stream:
        for row in csv.DictReader(stream):
            if (int(row["n_sweeps"]) != 100000
                    or int(row["n_eq"]) != 100000
                    or int(row["n_lambda"]) != 12
                    or row["update"] != "sw"):
                raise ValueError(f"nonstandard campaign row: {row}")
            key = (row["beta"], int(row["L"]), row["observable"])
            raw[key].append((float(row["value"]), float(row["error"])))
    return raw


def write_tables(raw):
    """Write coverage, pooled components, endpoint checks, and plateau tests."""
    with open(os.path.join(OUT, "coverage.csv"), "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["beta", "L", *OBSERVABLES, "complete_16_each"])
        for beta in BETAS:
            for size in SIZES[beta]:
                counts = [
                    len(raw[(beta, size, observable)])
                    for observable in OBSERVABLES
                ]
                writer.writerow(
                    [beta, size, *counts, all(count == 16 for count in counts)]
                )

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
                    values = raw[(beta, size, observable)]
                    if not values:
                        continue
                    result = pool(values)
                    pooled[(beta, size, observable)] = result
                    writer.writerow([
                        beta, size, observable, *result, len(values),
                        len(values) == 16,
                    ])

    comparisons = []
    comparison_path = os.path.join(OUT, "reconstruction_vs_direct.csv")
    with open(comparison_path, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "beta", "L", "reconstructed_tildeS", "reconstructed_error",
            "direct_tildeS", "direct_error", "difference",
            "difference_error", "z_score", "min_chain_count",
            "complete_16_each",
        ])
        for beta in BETAS:
            for size in SIZES[beta]:
                if not all(
                    (beta, size, observable) in pooled
                    for observable in OBSERVABLES
                ):
                    continue
                p = {
                    observable: pooled[(beta, size, observable)]
                    for observable in OBSERVABLES
                }
                reconstructed = (
                    2.0 * p["S3"][0] - p["S2A"][0]
                    - p["S2B"][0] - p["S2C"][0]
                )
                reconstructed_error = math.sqrt(
                    (2.0 * p["S3"][1]) ** 2 + p["S2A"][1] ** 2
                    + p["S2B"][1] ** 2 + p["S2C"][1] ** 2
                )
                difference = reconstructed - p["tS"][0]
                difference_error = math.hypot(
                    reconstructed_error, p["tS"][1]
                )
                counts = [
                    len(raw[(beta, size, observable)])
                    for observable in OBSERVABLES
                ]
                row = (
                    beta, size, reconstructed, reconstructed_error,
                    p["tS"][0], p["tS"][1], difference, difference_error,
                    difference / difference_error, min(counts),
                    all(count == 16 for count in counts),
                )
                comparisons.append(row)
                writer.writerow(row)

    plateau_rows = []
    for beta in BETAS:
        rows = [row for row in comparisons if row[0] == beta]
        rows.sort(key=lambda row: row[1])
        if len(rows) < 3:
            continue
        tail = rows[-3:]
        direct = constant_fit([(row[4], row[5]) for row in tail])
        reconstructed = constant_fit([(row[2], row[3]) for row in tail])
        max_identity_z = max(abs(row[8]) for row in tail)
        correlation_length = xi_spin(float(beta))
        min_l_over_xi = tail[0][1] / correlation_length
        direct_flat = direct[2] <= 3.0
        reconstructed_flat = reconstructed[2] <= 3.0
        identity_pass = max_identity_z <= 3.0
        direct_certified = (
            min_l_over_xi >= 4.0 and direct_flat and identity_pass
        )
        plateau_rows.append({
            "beta": beta,
            "xi_spin": correlation_length,
            "tail_sizes": ";".join(str(row[1]) for row in tail),
            "tail_min_L_over_xi": min_l_over_xi,
            "direct_plateau": direct[0],
            "direct_error": direct[1],
            "direct_chi2_dof": direct[2],
            "reconstructed_plateau": reconstructed[0],
            "reconstructed_error": reconstructed[1],
            "reconstructed_chi2_dof": reconstructed[2],
            "max_identity_abs_z": max_identity_z,
            "direct_flat": direct_flat,
            "reconstructed_flat": reconstructed_flat,
            "identity_pass": identity_pass,
            "direct_plateau_certified": direct_certified,
            "both_paths_certified": direct_certified and reconstructed_flat,
        })

    plateau_path = os.path.join(OUT, "plateau_summary.csv")
    if plateau_rows:
        with open(plateau_path, "w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(plateau_rows[0])
            )
            writer.writeheader()
            writer.writerows(plateau_rows)
    return pooled, comparisons, plateau_rows


def make_plots(pooled, comparisons, plateau_rows):
    """Plot all components and the direct-versus-reconstructed plateau test."""
    colors = {
        "S2A": "#287271", "S2B": "#d07c32",
        "S2C": "#4c78a8", "S3": "#222222",
    }
    labels = {
        "S2A": r"$S_2(A)$", "S2B": r"$S_2(B)$",
        "S2C": r"$S_2(C)$", "S3": r"$S_2^{(3)}$",
    }

    fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.2), dpi=180)
    for axis, beta in zip(axes.flat, BETAS):
        for observable in ("S2A", "S2B", "S2C", "S3"):
            points = [
                (size, *pooled[(beta, size, observable)][:2])
                for size in SIZES[beta]
                if (beta, size, observable) in pooled
            ]
            if points:
                axis.errorbar(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    yerr=[point[2] for point in points],
                    fmt="o-", ms=3.5, lw=1.0, capsize=2,
                    color=colors[observable], label=labels[observable],
                )
        axis.set_title(
            rf"$\beta={beta}$, $\xi={xi_spin(float(beta)):.3f}$"
        )
        axis.set_xlabel(r"$L$")
        axis.grid(alpha=0.22)
    axes[0, 0].set_ylabel("component entropy")
    axes[1, 0].set_ylabel("component entropy")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "components_vs_L.png"), bbox_inches="tight")
    plt.close(fig)

    summary = {row["beta"]: row for row in plateau_rows}
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.2), dpi=180)
    for axis, beta in zip(axes.flat, BETAS):
        rows = sorted(
            (row for row in comparisons if row[0] == beta),
            key=lambda row: row[1],
        )
        if rows:
            sizes = [row[1] for row in rows]
            axis.errorbar(
                sizes, [row[2] for row in rows],
                yerr=[row[3] for row in rows],
                fmt="o-", ms=4, capsize=2, color="#b55223",
                label="components",
            )
            axis.errorbar(
                sizes, [row[4] for row in rows],
                yerr=[row[5] for row in rows],
                fmt="s-", ms=4, capsize=2, color="#2e5fa3",
                label="direct",
            )
            if beta in summary:
                result = summary[beta]
                axis.axhspan(
                    result["direct_plateau"] - result["direct_error"],
                    result["direct_plateau"] + result["direct_error"],
                    color="#2e5fa3", alpha=0.12,
                )
                if result["both_paths_certified"]:
                    status = "pass"
                elif result["direct_plateau_certified"]:
                    status = "direct pass; component tail check"
                else:
                    status = "check"
                axis.text(
                    0.03, 0.04,
                    f"tail: {result['tail_sizes']}\n"
                    f"min L/xi={result['tail_min_L_over_xi']:.1f}; {status}",
                    transform=axis.transAxes, fontsize=8,
                    bbox={"facecolor": "white", "edgecolor": "0.8"},
                )
        axis.axhline(0.0, color="0.7", ls=":", lw=0.8)
        axis.set_title(
            rf"$\beta={beta}$, $\xi={xi_spin(float(beta)):.3f}$"
        )
        axis.set_xlabel(r"$L$")
        axis.grid(alpha=0.22)
    axes[0, 0].set_ylabel(r"$\widetilde S$")
    axes[1, 0].set_ylabel(r"$\widetilde S$")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(
        os.path.join(OUT, "tildeS_plateau_diagnostic.png"),
        bbox_inches="tight",
    )
    plt.close(fig)

    # Put the directly sampled tilde-S curves on common axes so that the
    # finite-size crossover and the large-L plateaus can be compared without
    # switching between beta-specific panels.
    beta_styles = {
        "0.10": ("#8bb8e8", "o"),
        "0.20": ("#5795d1", "s"),
        "0.25": ("#2f6fad", "^"),
        "0.35": ("#173f75", "D"),
        "0.70": ("#d46a3a", "v"),
        "0.80": ("#8f2d1b", "P"),
    }
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 8.0), dpi=180)
    for axis in axes:
        for beta in BETAS:
            points = [
                (size, *pooled[(beta, size, "tS")][:2])
                for size in SIZES[beta]
                if (beta, size, "tS") in pooled
            ]
            if axis is axes[1]:
                points = [point for point in points if point[0] >= 8]
            color, marker = beta_styles[beta]
            axis.errorbar(
                [point[0] for point in points],
                [point[1] for point in points],
                yerr=[point[2] for point in points],
                fmt=marker + "-", ms=5, lw=1.2, capsize=2.5,
                color=color, label=rf"$\beta={beta}$",
            )
        axis.axhline(0.0, color="0.6", ls=":", lw=0.9)
        axis.set_ylabel(r"direct $\widetilde S$")
        axis.grid(alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("Direct Swendsen--Wang estimates: all sizes")
    axes[1].set_title(r"Large-size view ($L\geq 8$)")
    axes[1].set_xlabel(r"$L$")
    axes[0].legend(frameon=False, ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(
        os.path.join(OUT, "tildeS_all_betas_vs_L.png"),
        bbox_inches="tight",
    )
    plt.close(fig)

    # Compact thermodynamic-limit summary for the correlation-length study.
    fig, axis = plt.subplots(figsize=(6.8, 4.7), dpi=180)
    label_offsets = {
        "0.80": (7, 10), "0.10": (7, -16), "0.70": (7, 12),
        "0.20": (7, 5), "0.25": (7, 5), "0.35": (7, 5),
    }
    for phase, phase_betas, color, marker in (
        ("disordered", ("0.10", "0.20", "0.25", "0.35"), "#2e5fa3", "o"),
        ("ordered", ("0.70", "0.80"), "#c4552d", "s"),
    ):
        rows = [summary[beta] for beta in phase_betas]
        axis.errorbar(
            [math.log(row["xi_spin"]) for row in rows],
            [row["direct_plateau"] for row in rows],
            yerr=[row["direct_error"] for row in rows],
            fmt=marker, ms=6, capsize=3, color=color, label=phase,
        )
        for beta, row in zip(phase_betas, rows):
            axis.annotate(
                rf"$\beta={beta}$",
                (math.log(row["xi_spin"]), row["direct_plateau"]),
                xytext=label_offsets[beta], textcoords="offset points",
                fontsize=8,
            )
    axis.axhline(0.0, color="0.65", ls=":", lw=0.9)
    axis.set_xlabel(r"$\ln \xi_{\rm spin}(\beta)$")
    axis.set_ylabel(r"direct plateau $\widetilde S_\infty(\beta)$")
    axis.set_title("Additional-beta SW plateaus")
    axis.grid(alpha=0.22)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plateau_vs_ln_xi.png"), bbox_inches="tight")
    plt.close(fig)

    # Show the same plateau estimates against xi itself and against ln(xi).
    # The paired panels make the horizontal rescaling explicit while keeping
    # all Monte Carlo values and uncertainties identical.
    fig, axes = plt.subplots(
        1, 2, figsize=(11.4, 4.7), dpi=180, sharey=True,
    )
    x_transforms = (
        (lambda row: row["xi_spin"], r"$\xi_{\rm spin}(\beta)$", "linear"),
        (
            lambda row: math.log(row["xi_spin"]),
            r"$\ln \xi_{\rm spin}(\beta)$", "logarithm",
        ),
    )
    combined_offsets = {
        "0.80": (6, -15), "0.10": (6, 7), "0.70": (6, 10),
        "0.20": (6, 5), "0.25": (6, 5), "0.35": (6, 5),
    }
    phases = (
        ("disordered", ("0.10", "0.20", "0.25", "0.35"),
         "#2e5fa3", "o"),
        ("ordered", ("0.70", "0.80"), "#c4552d", "s"),
    )
    for axis, (x_value, x_label, title) in zip(axes, x_transforms):
        for phase, phase_betas, color, marker in phases:
            rows = [summary[beta] for beta in phase_betas]
            axis.errorbar(
                [x_value(row) for row in rows],
                [row["direct_plateau"] for row in rows],
                yerr=[row["direct_error"] for row in rows],
                fmt=marker, ms=6, capsize=3, color=color, label=phase,
            )
            for beta, row in zip(phase_betas, rows):
                axis.annotate(
                    rf"$\beta={beta}$",
                    (x_value(row), row["direct_plateau"]),
                    xytext=combined_offsets[beta],
                    textcoords="offset points", fontsize=8,
                )
        axis.axhline(0.0, color="0.65", ls=":", lw=0.9)
        axis.set_xlabel(x_label)
        axis.set_title(title + " horizontal coordinate")
        axis.grid(alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(r"direct plateau $\widetilde S_\infty(\beta)$")
    axes[0].legend(frameon=False)
    fig.suptitle("Additional-beta Swendsen--Wang plateaus")
    fig.tight_layout()
    fig.savefig(
        os.path.join(OUT, "plateau_vs_xi_linear_and_log.png"),
        bbox_inches="tight",
    )
    plt.close(fig)

    # A true log-log view cannot display a signed estimate at or below zero.
    # Plot statistically resolved positive plateaus normally and represent all
    # zero-compatible estimates by Gaussian one-sided 95% upper limits.  This
    # retains the sign information instead of taking an unjustified absolute
    # value merely to satisfy the logarithmic axis.
    fig, axis = plt.subplots(figsize=(6.8, 4.8), dpi=180)
    loglog_offsets = {
        "0.80": (6, -6), "0.10": (6, 5), "0.70": (6, 5),
        "0.20": (6, 5), "0.25": (6, 5), "0.35": (6, 5),
    }
    for phase, phase_betas, color, marker in phases:
        # Legend proxy, since some phases can contain only upper limits.
        axis.plot(
            [], [], marker=marker, ls="none", color=color,
            label=phase,
        )
        for beta in phase_betas:
            row = summary[beta]
            x_point = row["xi_spin"]
            value = row["direct_plateau"]
            error = row["direct_error"]
            if value - error > 0.0:
                y_point = value
                axis.errorbar(
                    x_point, y_point, yerr=error, fmt=marker,
                    ms=6, capsize=3, color=color,
                )
            else:
                # One-sided 95% Gaussian upper confidence limit.
                y_point = max(value + 1.645 * error, 1.0e-12)
                axis.scatter(
                    [x_point], [y_point], marker="v", s=42,
                    facecolors="none", edgecolors=color, linewidths=1.3,
                )
            axis.annotate(
                rf"$\beta={beta}$", (x_point, y_point),
                xytext=loglog_offsets[beta], textcoords="offset points",
                fontsize=8,
            )
    axis.plot(
        [], [], marker="v", ls="none", markerfacecolor="none",
        markeredgecolor="0.25", color="0.25", label="95% upper limit",
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(r"$\xi_{\rm spin}(\beta)$")
    axis.set_ylabel(r"direct plateau $\widetilde S_\infty(\beta)$")
    axis.set_title("Log--log view of additional-beta SW plateaus")
    axis.grid(alpha=0.22, which="both")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(
        os.path.join(OUT, "plateau_vs_xi_loglog.png"),
        bbox_inches="tight",
    )
    plt.close(fig)

    # Publication-style comparison of all three horizontal/vertical axis
    # choices in one row.  The first two panels retain signed plateau values;
    # the log-log panel uses the same upper-limit convention as above.
    fig, axes = plt.subplots(1, 3, figsize=(16.3, 4.7), dpi=180)
    for axis, (x_value, x_label, panel_title) in zip(
        axes[:2],
        (
            (lambda row: row["xi_spin"],
             r"$\xi_{\rm spin}(\beta)$", "(a) linear--linear"),
            (lambda row: math.log(row["xi_spin"]),
             r"$\ln \xi_{\rm spin}(\beta)$", "(b) log coordinate--linear"),
        ),
    ):
        for phase, phase_betas, color, marker in phases:
            rows = [summary[beta] for beta in phase_betas]
            axis.errorbar(
                [x_value(row) for row in rows],
                [row["direct_plateau"] for row in rows],
                yerr=[row["direct_error"] for row in rows],
                fmt=marker, ms=5.5, capsize=2.5, color=color,
                label=phase,
            )
            for beta, row in zip(phase_betas, rows):
                axis.annotate(
                    rf"$\beta={beta}$",
                    (x_value(row), row["direct_plateau"]),
                    xytext=combined_offsets[beta],
                    textcoords="offset points", fontsize=7.5,
                )
        axis.axhline(0.0, color="0.65", ls=":", lw=0.9)
        axis.set_xlabel(x_label)
        axis.set_title(panel_title)
        axis.grid(alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(r"direct plateau $\widetilde S_\infty(\beta)$")
    axes[0].legend(frameon=False, fontsize=8)
    # The same data give identical vertical limits in the first two panels.
    axes[1].set_ylim(axes[0].get_ylim())

    axis = axes[2]
    for phase, phase_betas, color, marker in phases:
        axis.plot(
            [], [], marker=marker, ls="none", color=color, label=phase,
        )
        for beta in phase_betas:
            row = summary[beta]
            x_point = row["xi_spin"]
            value = row["direct_plateau"]
            error = row["direct_error"]
            if value - error > 0.0:
                y_point = value
                axis.errorbar(
                    x_point, y_point, yerr=error, fmt=marker,
                    ms=5.5, capsize=2.5, color=color,
                )
            else:
                y_point = max(value + 1.645 * error, 1.0e-12)
                axis.scatter(
                    [x_point], [y_point], marker="v", s=38,
                    facecolors="none", edgecolors=color, linewidths=1.2,
                )
            axis.annotate(
                rf"$\beta={beta}$", (x_point, y_point),
                xytext=loglog_offsets[beta], textcoords="offset points",
                fontsize=7.5,
            )
    axis.plot(
        [], [], marker="v", ls="none", markerfacecolor="none",
        markeredgecolor="0.25", color="0.25", label="95% upper limit",
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(r"$\xi_{\rm spin}(\beta)$")
    axis.set_ylabel(r"direct plateau $\widetilde S_\infty(\beta)$")
    axis.set_title("(c) log--log")
    axis.grid(alpha=0.22, which="both")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=8)
    fig.suptitle("Additional-beta Swendsen--Wang plateaus")
    fig.tight_layout()
    fig.savefig(
        os.path.join(OUT, "plateau_three_axis_views.png"),
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    raw = read_raw()
    pooled, comparisons, plateau_rows = write_tables(raw)
    if HAVE_MATPLOTLIB:
        make_plots(pooled, comparisons, plateau_rows)
    else:
        print("matplotlib unavailable; wrote tables without figures")
    print(f"wrote additional-beta analysis to {OUT}")
    for row in plateau_rows:
        print(
            f"beta={row['beta']} xi={row['xi_spin']:.6f} "
            f"tail={row['tail_sizes']} "
            f"direct={row['direct_plateau']:+.6f}"
            f"+-{row['direct_error']:.6f} "
            f"chi2/dof={row['direct_chi2_dof']:.2f} "
            f"identity|max z|={row['max_identity_abs_z']:.2f} "
            f"direct_certified={row['direct_plateau_certified']} "
            f"both_paths={row['both_paths_certified']}"
        )


if __name__ == "__main__":
    main()
