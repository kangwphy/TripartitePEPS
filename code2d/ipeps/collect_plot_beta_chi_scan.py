#!/usr/bin/env python3
"""Collect and plot the requested sparse iPEPS beta/chi scan."""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
LOGDIR = ROOT / "data/rk_ising/ctmrg" / "logs" / "beta_chi_scan"
OUTDIR = ROOT / "data/rk_ising/ctmrg" / "tables" / "beta_chi_scan"
FIGDIR = ROOT / "data/rk_ising/ctmrg" / "figures" / "beta_chi_scan"
BETAS = (0.30, 0.40, 0.46, 0.48, 0.60)
CHIS = (4, 8, 12, 16, 24, 32)

SUMMARY = re.compile(
    r"SUMMARY-SYM beta=(?P<beta>[\d.]+) chi=(?P<chi>\d+) "
    r"s2=(?P<s2>[\d.eE+\-]+) xi2=(?P<xi2>[\d.eE+\-]+) "
    r"xi4=(?P<xi4>[\d.eE+\-]+) kA=(?P<kA>[\d.eE+\-]+) "
    r"kY=(?P<kY>[\d.eE+\-]+) tSJ=(?P<tSJ>[\d.eE+\-]+) "
    r"it2=(?P<it2>\d+) it4=(?P<it4>\d+) "
    r"dr2=(?P<dr2>[\d.eE+\-]+) dr4=(?P<dr4>[\d.eE+\-]+) "
    r"lk2=(?P<lk2>[\d.eE+\-]+) lk4=(?P<lk4>[\d.eE+\-]+) "
    r"asym=(?P<asym>[\d.eE+\-]+)"
)
SEAM = re.compile(
    r"seam (?:CA|CB|AB): tension=(?P<t>[\d.eE+\-]+) "
    r"vs 2\*s2win=(?P<target>[\d.eE+\-]+)"
)


def collect() -> list[dict[str, object]]:
    """Keep the latest complete summary for every requested beta/chi pair."""
    latest: dict[tuple[float, int], dict[str, object]] = {}
    for logfile in sorted(LOGDIR.glob("scan_*.out")):
        text = logfile.read_text(errors="replace")
        job_match = re.search(r"scan_(\d+)_(\d+)\.out$", logfile.name)
        jobid = job_match.group(1) if job_match else ""
        blocks = re.split(r"(?=chi_env=\d+)", text)
        for block in blocks:
            match = SUMMARY.search(block)
            if not match:
                continue
            beta = float(match.group("beta"))
            chi = int(match.group("chi"))
            if not any(math.isclose(beta, value, abs_tol=5e-9) for value in BETAS):
                continue
            if chi not in CHIS:
                continue
            row: dict[str, object] = {
                "jobid": jobid,
                "logfile": logfile.name,
            }
            for key, value in match.groupdict().items():
                row[key] = int(value) if key in {"chi", "it2", "it4"} else float(value)
            seam_rows = [(float(item.group("t")), float(item.group("target")))
                         for item in SEAM.finditer(block)]
            row["factor_gate_max_rel"] = (
                max(abs(tension - target) / abs(target)
                    for tension, target in seam_rows if target != 0.0)
                if seam_rows else float("nan")
            )
            key = (round(beta, 8), chi)
            if key not in latest or int(jobid or 0) >= int(latest[key]["jobid"] or 0):
                latest[key] = row
    rows = sorted(latest.values(), key=lambda row: (float(row["beta"]), int(row["chi"])))
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fields = [
        "beta", "chi", "tSJ", "kA", "kY", "s2", "xi2", "xi4",
        "factor_gate_max_rel", "asym", "dr2", "dr4", "lk2", "lk4",
        "it2", "it4", "jobid", "logfile",
    ]
    with (OUTDIR / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def plot(rows: list[dict[str, object]]) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig, (axis, gate_axis) = plt.subplots(1, 2, figsize=(11.4, 4.5), dpi=180)
    colors = plt.get_cmap("viridis")
    for index, beta in enumerate(BETAS):
        group = [row for row in rows if math.isclose(float(row["beta"]), beta, abs_tol=5e-9)]
        if not group:
            continue
        color = colors(index / (len(BETAS) - 1))
        label = rf"$\beta={beta:.2f}$"
        axis.plot([row["chi"] for row in group], [row["tSJ"] for row in group],
                  "o-", color=color, label=label)
        gate_axis.semilogy(
            [row["chi"] for row in group],
            [max(float(row["factor_gate_max_rel"]), 1e-16) for row in group],
            "o-", color=color, label=label,
        )
    axis.axhline(0.0, color="0.65", ls=":", lw=0.9)
    axis.set_xlabel(r"environment bond dimension $\chi$")
    axis.set_ylabel(r"fused defect-CTMRG $\widetilde S_J(\chi)$")
    axis.set_title("requested off-critical iPEPS scan")
    gate_axis.axhline(1e-2, color="0.55", ls=":", lw=0.9)
    gate_axis.set_xlabel(r"environment bond dimension $\chi$")
    gate_axis.set_ylabel(r"$\max|t_{XY}-2s_2|/(2s_2)$")
    gate_axis.set_title("factorization gate (smaller is better)")
    for ax in (axis, gate_axis):
        ax.grid(alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "stilde_vs_chi.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = collect()
    plot(rows)
    expected = len(BETAS) * len(CHIS)
    print(f"collected {len(rows)}/{expected} requested beta/chi points")
    for beta in BETAS:
        chis = [int(row["chi"]) for row in rows
                if math.isclose(float(row["beta"]), beta, abs_tol=5e-9)]
        print(f"beta={beta:.2f}: {chis or '-'}")


if __name__ == "__main__":
    main()
