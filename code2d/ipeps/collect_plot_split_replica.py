#!/usr/bin/env python3
"""Collect synchronized split-replica Slurm logs and plot diagnostics."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
LOGDIR = ROOT / "data/rk_ising/ctmrg" / "logs" / "split_replica"
TABLE = ROOT / "data/rk_ising/ctmrg" / "tables" / "split_replica" / "summary.csv"
COMPONENT_TABLE = (
    ROOT / "data/rk_ising/ctmrg" / "tables" / "split_replica" / "critical_components.csv"
)
FIT_TABLE = (
    ROOT / "data/rk_ising/ctmrg" / "tables" / "split_replica" / "critical_logxi_fits.csv"
)
FIGURE = ROOT / "data/rk_ising/ctmrg" / "figures" / "split_replica" / "diagnostics.png"
COMPONENT_FIGURE = (
    ROOT / "data/rk_ising/ctmrg" / "figures" / "split_replica" / "critical_components.png"
)
FUSED = ROOT / "data/rk_ising/ctmrg" / "tables" / "defect_ctmrg" / "critical_local_components.csv"

SUMMARY = re.compile(
    r"SUMMARY-SPLIT beta=(?P<beta>[\d.]+) chi1=(?P<chi1>\d+) "
    r"chi2=(?P<chi2>\d+) chi4=(?P<chi4>\d+) xi1=(?P<xi1>[\d.eE+\-]+) "
    r"plain2=(?P<plain2>[\d.eE+\-]+) plain4=(?P<plain4>[\d.eE+\-]+) "
    r"s2=(?P<s2>[\d.eE+\-]+) kA=(?P<kA>[\d.eE+\-]+) "
    r"tCA=(?P<tCA>[\d.eE+\-]+) tCB=(?P<tCB>[\d.eE+\-]+) "
    r"tAB=(?P<tAB>[\d.eE+\-]+) asym=(?P<asym>[\d.eE+\-]+) "
    r"factgate=(?P<factgate>[\d.eE+\-]+) kY=(?P<kY>[\d.eE+\-]+) "
    r"tSJ=(?P<tSJ>[\d.eE+\-]+) it1=(?P<it1>\d+) "
    r"it2d=(?P<it2d>\d+) it4max=(?P<it4max>\d+)"
    r"(?: mode=(?P<mode>\w+) pairres=(?P<pairres>[\d.eE+\-]+) "
    r"(?:juncgate=(?P<juncgate>[\d.eE+\-]+) )?"
    r"envsec=(?P<envsec>[\d.eE+\-]+) "
    r"(?:juncsec=(?P<juncsec>[\d.eE+\-]+) )?"
    r"totalsec=(?P<totalsec>[\d.eE+\-]+))?"
)


def collect() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for logfile in sorted(LOGDIR.glob("splitrep_*.out")):
        text = logfile.read_text(errors="replace")
        # Frozen-projector jobs are deliberately excluded even when they have
        # a SUMMARY-SPLIT line: they fail the certified xi1 regression.
        if ("synchronized env:" not in text
                or "DONE SPLIT-REPLICA-CTMRG" not in text
                or "WARN pair coevolve hit nmax" in text):
            continue
        jobid = logfile.stem.rsplit("_", 1)[-1]
        for match in SUMMARY.finditer(text):
            row: dict[str, object] = {"jobid": jobid, "logfile": logfile.name}
            for key, value in match.groupdict().items():
                if value is None:
                    row[key] = ""
                elif key == "mode":
                    row[key] = value
                elif key in {"chi1", "chi2", "chi4", "it1", "it2d", "it4max"}:
                    row[key] = int(value)
                else:
                    row[key] = float(value)
            # The pair-vs-fused junction residual exists only when the
            # SRFULLGATE oracle was evaluated.  Older fast logs printed zero
            # as a placeholder; expose that as missing, not as a passed test.
            if "fullgate=1" not in text:
                row["juncgate"] = ""
            # Explicit local-gauge components.  Keeping these next to tSJ
            # prevents the final cancellation from hiding either contribution.
            kappa_a = float(row["kA"])
            kappa_y = float(row["kY"])
            row["two_kA"] = 2.0 * kappa_a
            row["minus_kY"] = -kappa_y
            row["S2A_loc"] = -kappa_a
            row["S2B_loc"] = -kappa_a
            row["S2C_loc"] = 0.0
            row["S3_loc"] = -0.5 * kappa_y
            rows.append(row)
    # Keep the largest synchronized job id if a point was rerun.
    latest: dict[tuple[float, int], dict[str, object]] = {}
    for row in sorted(rows, key=lambda item: int(str(item["jobid"]))):
        latest[(round(float(row["beta"]), 10), int(row["chi1"]))] = row
    rows = sorted(latest.values(), key=lambda item: (float(item["beta"]), int(item["chi1"])))
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    fields = ["jobid", "logfile"] + list(SUMMARY.groupindex) + [
        "two_kA", "minus_kY", "S2A_loc", "S2B_loc", "S2C_loc", "S3_loc"
    ]
    with TABLE.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    component_fields = [
        "jobid", "chi1", "chi2", "chi4", "xi1", "s2",
        "S2A_loc", "S2B_loc", "S2C_loc", "S3_loc",
        "kA", "kY", "two_kA", "minus_kY", "tSJ",
        "tCA", "tCB", "tAB", "factgate", "juncgate",
        "envsec", "juncsec", "totalsec",
    ]
    critical_rows = [
        {field: row[field] for field in component_fields}
        for row in rows
        if abs(float(row["beta"]) - 0.4406867935) < 1e-7
    ]
    with COMPONENT_TABLE.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=component_fields)
        writer.writeheader()
        writer.writerows(critical_rows)
    return rows


def read_fused_stable() -> list[dict[str, float]]:
    if not FUSED.exists():
        return []
    rows: list[dict[str, float]] = []
    with FUSED.open() as handle:
        for row in csv.DictReader(handle):
            if row["stable"] != "True":
                continue
            rows.append({key: float(row[key]) for key in ("chi", "xi4", "tSJ")})
    return rows


def fit_logxi(
    rows: list[dict[str, object]], chi1_min: int = 4
) -> list[dict[str, object]]:
    """Fit the visually linear high-chi regime to y = slope*ln(xi) + intercept."""
    selected = [
        row for row in rows
        if int(row["chi1"]) >= chi1_min and float(row["xi1"]) > 0.0
    ]
    if len(selected) < 2:
        return []
    log_xi = np.log([float(row["xi1"]) for row in selected])
    fits: list[dict[str, object]] = []
    for quantity in ("two_kA", "minus_kY", "tSJ"):
        values = np.asarray([float(row[quantity]) for row in selected])
        slope, intercept = np.polyfit(log_xi, values, 1)
        predicted = slope * log_xi + intercept
        residual_ss = float(np.sum((values - predicted) ** 2))
        total_ss = float(np.sum((values - np.mean(values)) ** 2))
        r_squared = 1.0 - residual_ss / total_ss if total_ss > 0.0 else 1.0
        fits.append({
            "quantity": quantity,
            "chi1_min": chi1_min,
            "chi1_max": max(int(row["chi1"]) for row in selected),
            "n_points": len(selected),
            "lnxi_min": float(np.min(log_xi)),
            "lnxi_max": float(np.max(log_xi)),
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": r_squared,
        })
    FIT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    with FIT_TABLE.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fits[0]))
        writer.writeheader()
        writer.writerows(fits)
    return fits


def plot(rows: list[dict[str, object]]) -> None:
    if not rows:
        print("no completed synchronized rows; table header written, plot skipped")
        return
    critical = [row for row in rows if abs(float(row["beta"]) - 0.4406867935) < 1e-7]
    gapped = [row for row in rows if row not in critical]
    fused = read_fused_stable()
    colors = {0.30: "#2166ac", 0.35: "#1b9e77", 0.4406867935: "#b2182b"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.8))
    fig.suptitle("split-replica product environment: synchronized CTMRG diagnostics", fontsize=13)

    ax = axes[0, 0]
    if fused:
        ax.semilogx([row["xi4"] for row in fused], [row["tSJ"] for row in fused],
                    "x", color="silver", ms=6, label="fused k=4 CTMRG (internally stable)")
    if critical:
        ax.semilogx([row["xi1"] for row in critical], [row["tSJ"] for row in critical],
                    "o-", color=colors[0.4406867935], ms=7,
                    label="split replicas (product xi4=xi1)")
        for row in critical:
            ax.annotate(f"chi1={row['chi1']}", (float(row["xi1"]), float(row["tSJ"])),
                        xytext=(4, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("four-replica controlling correlation length")
    ax.set_ylabel(r"$\widetilde S_J$")
    ax.set_title("(a) critical: fused versus product environment")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for beta in sorted({round(float(row["beta"]), 2) for row in gapped}):
        group = [row for row in gapped if round(float(row["beta"]), 2) == beta]
        ax.plot([row["chi1"] for row in group], [row["tSJ"] for row in group], "o-",
                color=colors.get(beta), label=fr"$\beta={beta:.2f}$")
    ax.axhline(0.01104, color=colors[0.30], ls="--", lw=1, alpha=0.65,
               label=r"finite PEPS $\beta=.30$, $L=12$")
    ax.axhline(0.03339, color=colors[0.35], ls="--", lw=1, alpha=0.65,
               label=r"finite PEPS $\beta=.35$, $L=12$")
    ax.set_xlabel(r"one-replica $\chi_1$")
    ax.set_ylabel(r"$\widetilde S_J$")
    ax.set_title("(b) gapped controls")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for beta in sorted({float(row["beta"]) for row in rows}):
        group = [row for row in rows if float(row["beta"]) == beta]
        label = "critical" if abs(beta - 0.4406867935) < 1e-7 else f"beta={beta:.2f}"
        ax.semilogy([row["chi1"] for row in group], [row["factgate"] for row in group],
                    "o-", label=label)
    ax.axhline(1e-10, color="gray", ls=":", lw=1)
    ax.set_xlabel(r"one-replica $\chi_1$")
    ax.set_ylabel(r"$|t_4-2s_2|/(2s_2)$")
    ax.set_title("(c) exact factorization gate")
    ax.grid(which="both", alpha=0.22)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    if critical:
        ax.plot([row["chi1"] for row in critical], [2*float(row["kA"]) for row in critical],
                "o-", label=r"$2\kappa_A$")
        ax.plot([row["chi1"] for row in critical], [-float(row["kY"]) for row in critical],
                "s-", label=r"$-\kappa_Y$")
        ax.plot([row["chi1"] for row in critical], [row["tSJ"] for row in critical],
                "^-", color="black", label=r"$\widetilde S_J=2\kappa_A-\kappa_Y$")
    ax.set_xlabel(r"one-replica $\chi_1$")
    ax.set_ylabel("local contribution")
    ax.set_title("(d) critical component balance")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8)

    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)
    print(f"wrote {FIGURE}")

    if critical:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
        chis = [int(row["chi1"]) for row in critical]
        # chi1=1 has xi=0 and cannot appear on the logarithmic right axis.
        # Keep it in the left panel, and explicitly label the surviving
        # points on the right so the shared S_t data cannot be mistaken for
        # a different series.
        critical_xi = [row for row in critical if float(row["xi1"]) > 0]
        xis = [float(row["xi1"]) for row in critical_xi]
        logxi_fits = fit_logxi(critical_xi, chi1_min=4)
        axes[0].plot(chis, [float(row["S2A_loc"]) for row in critical], "o-",
                     label=r"$S_2(A)=S_2(B)$ (local normalization)")
        axes[0].plot(chis, [float(row["S2C_loc"]) for row in critical], "d-",
                     label=r"$S_2(C)$ (reference $=0$)")
        axes[0].plot(chis, [float(row["S3_loc"]) for row in critical], "s-",
                     label=r"$S_2^{(3)}$ (local normalization)")
        axes[0].plot(chis, [float(row["tSJ"]) for row in critical], "^-", color="black",
                     label=r"$\widetilde S_J$")
        axes[0].set_xscale("log")
        # Keep the logarithmic axis readable when the extended scan reaches
        # chi1=32; every computed point is drawn, but only a sparse subset is
        # labeled.
        preferred_ticks = {1, 2, 4, 6, 8, 12, 16, 24, 32}
        tick_chis = [chi for chi in chis if chi in preferred_ticks]
        axes[0].set_xticks(tick_chis)
        axes[0].set_xticklabels([str(chi) for chi in tick_chis])
        axes[0].set_xlabel(r"one-replica $\chi_1$")
        axes[0].set_ylabel("local value")
        axes[0].set_title("stored entropy components")
        axes[0].grid(alpha=0.22)
        axes[0].legend(fontsize=8)

        axes[1].semilogx(xis, [float(row["two_kA"]) for row in critical_xi], "o-",
                         label=r"$-S_2(A)-S_2(B)=2\kappa_A$")
        axes[1].semilogx(xis, [float(row["minus_kY"]) for row in critical_xi], "s-",
                         label=r"$2S_2^{(3)}=-\kappa_Y$")
        axes[1].semilogx(xis, [float(row["tSJ"]) for row in critical_xi], "^-", color="black",
                         label=r"same $\widetilde S_J=2\kappa_A-\kappa_Y$")
        fit_styles = {
            "two_kA": "tab:blue",
            "minus_kY": "tab:orange",
            "tSJ": "black",
        }
        if logxi_fits:
            fit_x = np.geomspace(
                np.exp(float(logxi_fits[0]["lnxi_min"])),
                np.exp(float(logxi_fits[0]["lnxi_max"])),
                200,
            )
            for fit in logxi_fits:
                fit_y = (float(fit["slope"]) * np.log(fit_x)
                         + float(fit["intercept"]))
                axes[1].plot(
                    fit_x, fit_y, "--", lw=1.35,
                    color=fit_styles[str(fit["quantity"])], alpha=0.8,
                )
            display_names = {
                "two_kA": r"$2\kappa_A$",
                "minus_kY": r"$-\kappa_Y$",
                "tSJ": r"$\widetilde S_J$",
            }
            fit_text = [r"fits for $\chi_1\geq4$: $y=a\ln\xi+b$"]
            for fit in logxi_fits:
                fit_text.append(
                    f"{display_names[str(fit['quantity'])]}: "
                    f"$a={float(fit['slope']):+.4f}$, "
                    f"$R^2={float(fit['r_squared']):.4f}$"
                )
            axes[1].text(
                0.98, 0.98, "\n".join(fit_text),
                transform=axes[1].transAxes, ha="right", va="top", fontsize=7.4,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.88,
                      "edgecolor": "0.75"},
            )
        for row in critical_xi:
            if int(row["chi1"]) not in preferred_ticks:
                continue
            axes[1].annotate(
                fr"$\chi_1={int(row['chi1'])}$",
                (float(row["xi1"]), float(row["tSJ"])),
                xytext=(3, 4), textcoords="offset points", fontsize=7,
            )
        axes[1].set_xlabel(r"$\xi(\chi_1)$")
        axes[1].set_ylabel("contribution")
        axes[1].set_title(r"same $\widetilde S_J$, replotted versus $\xi$")
        axes[1].grid(which="both", alpha=0.22)
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(COMPONENT_FIGURE, dpi=180)
        plt.close(fig)
        print(f"wrote {COMPONENT_FIGURE}")
        if logxi_fits:
            print(f"wrote {FIT_TABLE}")


def main() -> None:
    rows = collect()
    print(f"wrote {TABLE} ({len(rows)} synchronized rows)")
    print(f"wrote {COMPONENT_TABLE}")
    plot(rows)


if __name__ == "__main__":
    main()
