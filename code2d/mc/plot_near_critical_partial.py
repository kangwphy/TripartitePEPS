#!/usr/bin/env python3
"""Plot the complete direct-tS near-critical campaign and plateau diagnostics."""

import csv
import glob
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(
    ROOT, "data/rk_ising/mc", "campaigns",
    "sw_components_near_critical_100k_16chains",
)
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
BETA_C = 0.5 * math.log(1.0 + math.sqrt(2.0))


def xi_spin(beta):
    """Exact connected-spin exponential correlation length along an axis."""
    beta_dual = -0.5 * math.log(math.tanh(beta))
    if beta < BETA_C:
        return 1.0 / (2.0 * (beta_dual - beta))
    return 1.0 / (4.0 * (beta - beta_dual))


def pool(values):
    """Pool 16 chains using the larger formal or chain-scatter error."""
    weights = [1.0 / error**2 for _, error in values]
    mean = sum(value * weight for (value, _), weight in zip(values, weights))
    mean /= sum(weights)
    formal = 1.0 / math.sqrt(sum(weights))
    chain_sem = math.sqrt(
        sum((value - mean) ** 2 for value, _ in values)
        / (len(values) * (len(values) - 1))
    )
    return mean, max(formal, chain_sem)


def constant_fit(points):
    """Fit a constant to three pooled value/error pairs."""
    weights = [1.0 / error**2 for _, error in points]
    mean = sum(value * weight for (value, _), weight in zip(points, weights))
    mean /= sum(weights)
    error = 1.0 / math.sqrt(sum(weights))
    chi2 = sum(((value - mean) / sigma) ** 2 for value, sigma in points)
    return mean, error, chi2 / (len(points) - 1)


def read_complete_points():
    """Accept a point only when all 16 expected direct-tS seeds are valid."""
    points = {beta: [] for beta in BETAS}
    pooled_path = os.path.join(OUT, "pooled_components.csv")
    if os.path.isfile(pooled_path):
        with open(pooled_path, newline="") as stream:
            for row in csv.DictReader(stream):
                if row["observable"] != "tS" or row["complete_16"] != "True":
                    continue
                beta = f"{float(row['beta']):.2f}"
                if beta not in points:
                    continue
                points[beta].append(
                    (int(row["L"]), float(row["value"]), float(row["error"]))
                )
        for beta in BETAS:
            points[beta].sort()
            found = tuple(row[0] for row in points[beta])
            if found != SIZES[beta]:
                raise RuntimeError(
                    f"strict pooled coverage mismatch at beta={beta}: {found}"
                )
        return points

    raw_dir = os.path.join(OUT, "tS", "raw")
    for beta in BETAS:
        betakey = beta.replace(".", "p")
        for size in SIZES[beta]:
            pattern = os.path.join(
                raw_dir,
                f"L{size:03d}_b{betakey}_n100000_eq100000_seed*.csv",
            )
            values = []
            seeds = set()
            for path in glob.glob(pattern):
                with open(path, newline="") as stream:
                    rows = list(csv.DictReader(stream))
                if len(rows) != 1:
                    continue
                row = rows[0]
                if (int(row["n_sweeps"]) != 100000
                        or int(row["n_eq"]) != 100000
                        or int(row["n_gl"]) != 12
                        or row["update"] != "sw"
                        or row["quantity"] != "tS"
                        or row["gauge"] != "user"):
                    continue
                seed = int(row["seed"])
                if seed in seeds:
                    continue
                seeds.add(seed)
                values.append((float(row["tildeS"]), float(row["error"])))
            if len(values) == 16:
                mean, error = pool(values)
                points[beta].append((size, mean, error))
    return points


def write_table(points):
    """Record exactly which complete points entered the final figure."""
    rows = []
    for beta in BETAS:
        xi = xi_spin(float(beta))
        for size, value, error in points[beta]:
            rows.append([beta, size, size / xi, value, error, 16])
    for name in ("direct_tS.csv", "partial_direct_tS.csv"):
        with open(os.path.join(OUT, name), "w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow([
                "beta", "L", "L_over_xi", "direct_tS", "error", "n_chains"
            ])
            writer.writerows(rows)


def read_plateau_summary():
    """Read the strict final tail fits produced by the campaign analysis."""
    path = os.path.join(OUT, "plateau_summary.csv")
    with open(path, newline="") as stream:
        rows = {row["beta"]: row for row in csv.DictReader(stream)}
    if set(rows) != set(BETAS):
        raise RuntimeError(f"expected nine final plateau rows, found {sorted(rows)}")
    return rows


def plot(points, plateaus):
    """Show all complete points versus L and L/xi, including tail fits."""
    colors = plt.get_cmap("turbo")
    beta_colors = {
        beta: colors(index / (len(BETAS) - 1))
        for index, beta in enumerate(BETAS)
    }
    markers = ("o", "s", "^", "D", "v", "P", "X", "<", ">")
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), dpi=180)
    for beta, marker in zip(BETAS, markers):
        rows = points[beta]
        if not rows:
            continue
        xi = xi_spin(float(beta))
        for axis, x_values in (
            (axes[0], [row[0] for row in rows]),
            (axes[1], [row[0] / xi for row in rows]),
        ):
            axis.errorbar(
                x_values,
                [row[1] for row in rows],
                yerr=[row[2] for row in rows],
                fmt=marker + "-", ms=4.5, lw=1.1, capsize=2,
                color=beta_colors[beta], label=rf"$\beta={beta}$",
            )
            tail_x = x_values[-3:]
            plateau = float(plateaus[beta]["direct_plateau"])
            plateau_error = float(plateaus[beta]["direct_error"])
            axis.hlines(
                plateau, min(tail_x), max(tail_x),
                color=beta_colors[beta], ls="--", lw=1.5, alpha=0.9,
            )
            axis.fill_between(
                [min(tail_x), max(tail_x)],
                plateau - plateau_error, plateau + plateau_error,
                color=beta_colors[beta], alpha=0.10,
            )
    for axis in axes:
        axis.axhline(0.0, color="0.65", ls=":", lw=0.9)
        axis.set_ylabel(r"direct $\widetilde S(L)$")
        axis.grid(alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_xlabel(r"$L$")
    axes[0].set_title("all complete points; dashed segments are tail fits")
    axes[1].axvline(8.0, color="0.35", ls="--", lw=1.0, label=r"$L/\xi=8$")
    axes[1].set_xlabel(r"$L/\xi_{\rm spin}(\beta)$")
    axes[1].set_title("correlation-length-scaled size")
    axes[0].legend(frameon=False, fontsize=7.5, ncol=3)
    axes[1].legend(frameon=False, fontsize=7.5, ncol=2)
    fig.suptitle("Near-critical SW campaign: complete direct measurements")
    fig.tight_layout()
    for name in ("direct_tS_vs_L.png", "partial_direct_tS_vs_L.png"):
        fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)


def plot_plateau_diagnostics(points, plateaus):
    """Give each beta its own panel so tail drift cannot hide in a crowded plot."""
    colors = plt.get_cmap("turbo")
    beta_colors = {
        beta: colors(index / (len(BETAS) - 1))
        for index, beta in enumerate(BETAS)
    }
    fig, axes = plt.subplots(3, 3, figsize=(12.8, 10.2), dpi=180)
    for axis, beta in zip(axes.flat, BETAS):
        rows = points[beta]
        xi = xi_spin(float(beta))
        x = [row[0] / xi for row in rows]
        y = [row[1] for row in rows]
        errors = [row[2] for row in rows]
        color = beta_colors[beta]
        axis.errorbar(
            x, y, yerr=errors, fmt="o-", ms=4.2, lw=1.0,
            capsize=2, color=color,
        )
        axis.errorbar(
            x[-3:], y[-3:], yerr=errors[-3:], fmt="o", ms=6.2,
            capsize=2.5, color=color, markeredgecolor="black",
            markeredgewidth=0.55, label="tail fit points",
        )
        fit = plateaus[beta]
        plateau = float(fit["direct_plateau"])
        plateau_error = float(fit["direct_error"])
        chi2 = float(fit["direct_chi2_dof"])
        certified = fit["direct_certified"] == "True"
        axis.axhline(plateau, color="black", ls="--", lw=1.15)
        axis.fill_between(
            [x[-3], x[-1]], plateau - plateau_error, plateau + plateau_error,
            color="black", alpha=0.10,
        )
        axis.axvline(8.0, color="0.45", ls=":", lw=0.9)
        axis.axhline(0.0, color="0.7", ls=":", lw=0.8)
        status = "PASS" if certified else "FAIL"
        axis.set_title(
            rf"$\beta={beta}$, $\xi={xi:.2f}$: {status}"
            + "\n"
            + rf"$\widetilde S_\infty={plateau:+.5f}\pm{plateau_error:.5f}$, "
              rf"$\chi^2_\nu={chi2:.2f}$",
            fontsize=9,
            color=("#176b2c" if certified else "#a32620"),
        )
        axis.set_xlabel(r"$L/\xi_{\rm spin}$")
        axis.set_ylabel(r"direct $\widetilde S(L)$")
        axis.grid(alpha=0.20)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Near-critical SW plateau diagnostics: last three sizes per beta",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(
        os.path.join(OUT, "direct_tS_plateau_diagnostics.png"),
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_provisional_plateaus(points):
    """Plot only betas whose planned final three sizes are all complete."""
    plateaus = []
    for beta in BETAS:
        rows = {size: (value, error) for size, value, error in points[beta]}
        tail_sizes = SIZES[beta][-3:]
        if not all(size in rows for size in tail_sizes):
            continue
        fit = constant_fit([rows[size] for size in tail_sizes])
        xi = xi_spin(float(beta))
        plateaus.append((beta, xi, *fit, tail_sizes))

    with open(os.path.join(OUT, "partial_plateau_summary.csv"), "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["beta", "xi_spin", "plateau", "error", "chi2_dof", "tail_sizes"])
        for beta, xi, value, error, chi2, tail_sizes in plateaus:
            writer.writerow([beta, xi, value, error, chi2, ";".join(map(str, tail_sizes))])

    if not plateaus:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.6), dpi=180)
    transforms = (
        (lambda xi: xi, r"$\xi_{\rm spin}(\beta)$", "(a) linear--linear"),
        (math.log, r"$\ln\xi_{\rm spin}(\beta)$", "(b) log coordinate--linear"),
    )
    for axis, (x_value, x_label, title) in zip(axes[:2], transforms):
        axis.errorbar(
            [x_value(row[1]) for row in plateaus],
            [row[2] for row in plateaus],
            yerr=[row[3] for row in plateaus],
            fmt="o", ms=6, capsize=3, color="#2e5fa3",
        )
        for beta, xi, value, _, chi2, _ in plateaus:
            axis.annotate(
                rf"$\beta={beta}$, $\chi^2_\nu={chi2:.2f}$",
                (x_value(xi), value), xytext=(5, 5),
                textcoords="offset points", fontsize=7.5,
            )
        axis.set_xlabel(x_label)
        axis.set_title(title)
        axis.grid(alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(r"provisional direct plateau $\widetilde S_\infty$")
    axes[1].set_ylim(axes[0].get_ylim())

    axis = axes[2]
    axis.errorbar(
        [row[1] for row in plateaus], [row[2] for row in plateaus],
        yerr=[row[3] for row in plateaus],
        fmt="o", ms=6, capsize=3, color="#2e5fa3",
    )
    for beta, xi, value, _, chi2, _ in plateaus:
        axis.annotate(
            rf"$\beta={beta}$, $\chi^2_\nu={chi2:.2f}$",
            (xi, value), xytext=(5, 5),
            textcoords="offset points", fontsize=7.5,
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(r"$\xi_{\rm spin}(\beta)$")
    axis.set_ylabel(r"provisional direct plateau $\widetilde S_\infty$")
    axis.set_title("(c) log--log")
    axis.grid(alpha=0.22, which="both")
    axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Near-critical SW campaign: currently certified plateaus")
    fig.tight_layout()
    fig.savefig(
        os.path.join(OUT, "partial_plateau_three_axis_views.png"),
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    points = read_complete_points()
    plateaus = read_plateau_summary()
    write_table(points)
    plot(points, plateaus)
    plot_plateau_diagnostics(points, plateaus)
    for beta in BETAS:
        if points[beta]:
            sizes = ",".join(str(row[0]) for row in points[beta])
            print(f"beta={beta}: {len(points[beta])} complete sizes ({sizes})")


if __name__ == "__main__":
    main()
