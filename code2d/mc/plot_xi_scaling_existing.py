#!/usr/bin/env python3
"""Plot the existing direct-MC data in finite-correlation-length variables.

The horizontal coordinate is L/xi_spin, where xi_spin is the exact
exponential correlation length of the connected spin-spin correlation
function along a lattice axis of the infinite square-lattice Ising model.
The script deliberately reports largest-L values as *proxies*, not fitted
thermodynamic limits: fixed-order GL12 thermodynamic integration is known to
fail its path-independence check for the large-L beta=0.60 data.
"""

import csv
import math
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INPUT = os.path.join(ROOT, "data/rk_ising/mc", "tables", "mc_results_tS.csv")
COMPONENT_CHECK = os.path.join(
    ROOT, "data/rk_ising/mc", "campaigns/sw_components_100k_16chains",
    "reconstruction_vs_direct.csv"
)
OUT = os.path.join(ROOT, "data/rk_ising/mc", "derived", "correlation_length_scaling")
os.makedirs(OUT, exist_ok=True)

BETA_C = 0.5 * math.log(1.0 + math.sqrt(2.0))
BETAS = (0.30, 0.42, 0.43, 0.45, 0.46, 0.47, 0.60)


def dual_beta(beta):
    """Return beta* defined by exp(-2 beta*) = tanh(beta)."""
    return -0.5 * math.log(math.tanh(beta))


def xi_spin(beta):
    """Exact connected-spin exponential xi along a lattice axis.

    Above Tc (beta < beta_c), xi^-1 = 2(beta* - beta).  Below Tc,
    the connected spin correlator starts in the two-particle channel and
    xi^-1 = 4(beta - beta*).
    """
    beta_dual = dual_beta(beta)
    if beta < BETA_C:
        return 1.0 / (2.0 * (beta_dual - beta))
    return 1.0 / (4.0 * (beta - beta_dual))


def xi_gap(beta):
    """Fundamental transfer-matrix gap length, included for convention checks."""
    return 1.0 / (2.0 * abs(beta - dual_beta(beta)))


def conservative_pool(values):
    """Pool independent chains with equal weight and a conservative error."""
    means = [item[0] for item in values]
    errors = [item[1] for item in values]
    n = len(values)
    mean = sum(means) / n
    propagated = math.sqrt(sum(error * error for error in errors)) / n
    if n > 1:
        chain_sem = math.sqrt(
            sum((value - mean) ** 2 for value in means) / (n * (n - 1))
        )
    else:
        chain_sem = 0.0
    return mean, max(propagated, chain_sem), n


def read_direct_results(checks):
    """Read and pool the best available direct-tS results.

    The dedicated 100k-sweep, 16-chain component campaign supersedes the
    lower-statistics canonical-table values at beta=0.30 and beta=0.60.  It
    also provides an independent endpoint-identity check at every size.
    """
    grouped = defaultdict(list)
    with open(INPUT, newline="") as stream:
        for row in csv.reader(stream):
            if len(row) < 7 or row[1] == "crit":
                continue
            tag = row[7] if len(row) > 7 else ""
            if "S2C" in tag or "K4C" in tag:
                continue
            beta = float(row[1])
            if not any(abs(beta - target) < 1e-12 for target in BETAS):
                continue
            grouped[(beta, int(row[0]))].append(
                (float(row[4]), float(row[5]), tag or "legacy")
            )

    pooled = []
    for (beta, size), values in sorted(grouped.items()):
        mean, error, n_chains = conservative_pool(values)
        pooled.append({
            "beta": beta,
            "phase": "disordered" if beta < BETA_C else "ordered",
            "xi_spin": xi_spin(beta),
            "xi_gap": xi_gap(beta),
            "L": size,
            "L_over_xi_spin": size / xi_spin(beta),
            "value": mean,
            "error": error,
            "n_chains": n_chains,
            "tags": ";".join(sorted({item[2] for item in values})),
            "source": "canonical_direct_table",
        })

    pooled = [row for row in pooled if row["beta"] not in (0.30, 0.60)]
    for (beta, size), check in sorted(checks.items()):
        if beta not in (0.30, 0.60):
            continue
        pooled.append({
            "beta": beta,
            "phase": "disordered" if beta < BETA_C else "ordered",
            "xi_spin": xi_spin(beta),
            "xi_gap": xi_gap(beta),
            "L": size,
            "L_over_xi_spin": size / xi_spin(beta),
            "value": check["direct"],
            "error": check["direct_error"],
            "n_chains": 16,
            "tags": "GL12;100k",
            "source": "campaigns/sw_components_100k_16chains",
        })
    return sorted(pooled, key=lambda row: (row["beta"], row["L"]))


def read_identity_check():
    """Read direct-versus-component checks available at beta=0.30 and 0.60."""
    checks = {}
    with open(COMPONENT_CHECK, newline="") as stream:
        for row in csv.DictReader(stream):
            if row["beta"] == "crit":
                continue
            beta = float(row["beta"])
            checks[(beta, int(row["L"]))] = {
                "reconstructed": float(row["reconstructed_tildeS"]),
                "reconstructed_error": float(row["reconstructed_error"]),
                "direct": float(row["direct_tildeS"]),
                "direct_error": float(row["direct_error"]),
                "z_score": float(row["z_score"]),
            }
    return checks


def write_tables(rows, checks):
    """Write the exact xi values, pooled curves, and largest-L proxies."""
    with open(os.path.join(OUT, "exact_ising_xi.csv"), "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["beta", "phase", "xi_spin_axis", "xi_gap"])
        for beta in BETAS:
            writer.writerow([
                f"{beta:.2f}",
                "disordered" if beta < BETA_C else "ordered",
                f"{xi_spin(beta):.12g}",
                f"{xi_gap(beta):.12g}",
            ])

    fields = list(rows[0])
    with open(os.path.join(OUT, "pooled_existing_tS.csv"), "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    proxies = []
    for beta in BETAS:
        curve = [row for row in rows if row["beta"] == beta]
        last = max(curve, key=lambda row: row["L"])
        check = checks.get((beta, last["L"]))
        identity_status = "not_measured"
        identity_z = ""
        if check is not None:
            identity_z = check["z_score"]
            identity_status = "pass" if abs(identity_z) <= 3.0 else "fail"
        coverage = "adequate" if last["L_over_xi_spin"] >= 4.0 else "short"
        proxies.append({
            "beta": beta,
            "phase": last["phase"],
            "ln_xi_spin": math.log(last["xi_spin"]),
            "xi_spin": last["xi_spin"],
            "L_max": last["L"],
            "L_max_over_xi_spin": last["L_over_xi_spin"],
            "largest_L_value": last["value"],
            "largest_L_error": last["error"],
            "coverage": coverage,
            "endpoint_identity": identity_status,
            "identity_z_score": identity_z,
        })

    with open(os.path.join(OUT, "largest_L_proxies.csv"), "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(proxies[0]))
        writer.writeheader()
        writer.writerows(proxies)
    return proxies


def plot(rows, proxies, checks):
    """Make the L/xi curves and the deliberately qualified largest-L summary."""
    colors = {
        0.30: "#7aa6d8", 0.42: "#377eb8", 0.43: "#164a8a",
        0.45: "#8c2d04", 0.46: "#c94c1a", 0.47: "#e98243",
        0.60: "#f4ad7d",
    }
    markers = {
        0.30: "o", 0.42: "s", 0.43: "^",
        0.45: "^", 0.46: "s", 0.47: "v", 0.60: "D",
    }

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8.4, 9.0), dpi=180)
    for beta in BETAS:
        curve = sorted(
            (row for row in rows if row["beta"] == beta),
            key=lambda row: row["L"],
        )
        ax.errorbar(
            [row["L_over_xi_spin"] for row in curve],
            [row["value"] for row in curve],
            yerr=[row["error"] for row in curve],
            fmt=markers[beta] + "-", ms=4.5, lw=1.0, capsize=2.5,
            color=colors[beta], label=rf"$\beta={beta:.2f}$",
        )

    # For beta=0.60, show where the independent endpoint construction first
    # disagrees by more than three combined standard errors.
    failing = [
        (size / xi_spin(beta), check["direct"])
        for (beta, size), check in checks.items()
        if abs(beta - 0.60) < 1e-12 and abs(check["z_score"]) > 3.0
    ]
    if failing:
        ax.scatter(
            [point[0] for point in failing], [point[1] for point in failing],
            marker="x", s=55, linewidths=1.8, color="black", zorder=8,
            label=r"$\beta=0.60$: endpoint identity fails ($>3\sigma$)",
        )

    ax.axvline(1.0, color="0.45", ls="--", lw=0.9)
    ax.axvline(4.0, color="0.65", ls=":", lw=0.9)
    ax.axhline(0.0, color="0.7", ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel(r"$L/\xi_{\rm spin}(\beta)$")
    ax.set_ylabel(r"$\widetilde S(L,\beta)$")
    ax.set_title("Existing direct-MC curves in finite-correlation-length units")
    ax.text(1.03, 0.98, r"$L=\xi$", transform=ax.get_xaxis_transform(),
            ha="left", va="top", fontsize=8, color="0.35")
    ax.text(4.12, 0.98, r"$L=4\xi$", transform=ax.get_xaxis_transform(),
            ha="left", va="top", fontsize=8, color="0.45")
    ax.legend(ncol=2, fontsize=8, frameon=False)

    for proxy in proxies:
        beta = proxy["beta"]
        short = proxy["coverage"] == "short"
        identity_fail = proxy["endpoint_identity"] == "fail"
        face = "none" if short or identity_fail else colors[beta]
        ax2.errorbar(
            proxy["ln_xi_spin"], proxy["largest_L_value"],
            yerr=proxy["largest_L_error"], fmt=markers[beta], ms=7,
            capsize=3, color=colors[beta], markerfacecolor=face,
            markeredgewidth=1.4,
        )
        ax2.annotate(
            rf"${beta:.2f}$; $L_{{\max}}/\xi={proxy['L_max_over_xi_spin']:.1f}$",
            (proxy["ln_xi_spin"], proxy["largest_L_value"]),
            xytext=(5, 5), textcoords="offset points", fontsize=7.5,
        )
        if identity_fail:
            ax2.scatter(proxy["ln_xi_spin"], proxy["largest_L_value"],
                        marker="x", s=75, linewidths=1.7, color="black", zorder=8)

    ax2.axhline(0.0, color="0.7", ls=":", lw=0.8)
    ax2.set_xlabel(r"$\ln \xi_{\rm spin}(\beta)$")
    ax2.set_ylabel(r"$\widetilde S(L_{\max},\beta)$")
    ax2.set_title("Largest available size (a proxy, not an extrapolation)")
    ax2.text(
        0.01, 0.02,
        "open marker: $L_{max}/\\xi<4$ or endpoint identity fails\n"
        "black x: direct and component paths disagree by more than $3\\sigma$",
        transform=ax2.transAxes, ha="left", va="bottom", fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
    )

    for axis in (ax, ax2):
        axis.grid(alpha=0.24, lw=0.6)
        axis.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        r"Square-lattice Ising: $\xi_{\rm spin}^{-1}=2(\beta^*-\beta)$ "
        r"or $4(\beta-\beta^*)$",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    path = os.path.join(OUT, "existing_mc_xi_scaling.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    checks = read_identity_check()
    rows = read_direct_results(checks)
    proxies = write_tables(rows, checks)
    path = plot(rows, proxies, checks)
    print(f"wrote {path}")
    for proxy in proxies:
        print(
            f"beta={proxy['beta']:.2f} xi={proxy['xi_spin']:.6g} "
            f"Lmax/xi={proxy['L_max_over_xi_spin']:.3f} "
            f"proxy={proxy['largest_L_value']:+.6f} "
            f"coverage={proxy['coverage']} "
            f"identity={proxy['endpoint_identity']}"
        )


if __name__ == "__main__":
    main()
