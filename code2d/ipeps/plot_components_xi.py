#!/usr/bin/env python3
"""Plot critical defect-CTMRG local entropy pieces and correlation lengths.

The CTMRG window construction only forms ratios within a replica sector, so
absolute infinite-cut entropies are not available.  We choose the harmless
local gauge S2_loc(C)=0.  Then

    S2_loc(A) = S2_loc(B) = -kappa_A,
    S2_loc^(3) = -kappa_Y/2,

and 2*S2_loc^(3)-sum_X S2_loc(X) equals the reported tilde S_J exactly.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "data/rk_ising/ctmrg" / "tables" / "defect_ctmrg"
FIGURES = ROOT / "data/rk_ising/ctmrg" / "figures" / "defect_ctmrg"
PLAIN = TABLES / "critical_plain.csv"
SYM = TABLES / "critical_symmetrized.csv"
DERIVED = TABLES / "critical_local_components.csv"
OUT = FIGURES / "critical_local_components_and_xi.png"


def numeric_row(row: dict[str, str], text_fields: set[str]) -> dict[str, object]:
    return {key: (value if key in text_fields else float(value)) for key, value in row.items()}


def load_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with PLAIN.open() as handle:
        for raw in csv.DictReader(handle):
            try:
                row = numeric_row(raw, {"jobid", "variant"})
            except ValueError:
                continue
            stable = (
                float(row["seam_asym"]) < 1e-3
                and float(row["it4"]) < 300
                and float(row["it2"]) < 20 * float(row["xi2"])
            )
            rows.append(
                {
                    "jobid": row["jobid"],
                    "campaign": "original",
                    "variant": row["variant"],
                    "chi": row["chi_req"],
                    "xi2": row["xi2"],
                    "xi4": row["xi4"],
                    "kA": row["kA"],
                    "kY": row["kY"],
                    "tSJ": row["tSJ"],
                    "gate_rel": row["gate_rel"],
                    "stable": stable,
                }
            )
    with SYM.open() as handle:
        for raw in csv.DictReader(handle):
            try:
                row = numeric_row(raw, {"jobid", "variant"})
            except ValueError:
                continue
            stable = (
                float(row["asym"]) < 1e-3
                and float(row["it4"]) < 300
                and abs(float(row["kY"])) < 1
            )
            campaign = "S4xZ2" if row["variant"] == "flip" else "S4"
            rows.append(
                {
                    "jobid": row["jobid"],
                    "campaign": campaign,
                    "variant": row["variant"],
                    "chi": row["chi"],
                    "xi2": row["xi2"],
                    "xi4": row["xi4"],
                    "kA": row["kA"],
                    "kY": row["kY"],
                    "tSJ": row["tSJ"],
                    "gate_rel": row["gate_rel"],
                    "stable": stable,
                }
            )

    # Remove bitwise-identical reruns without hiding distinct branches at the
    # same requested chi.
    unique: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    for row in rows:
        key = (
            row["campaign"],
            round(float(row["chi"])),
            round(float(row["kA"]), 8),
            round(float(row["kY"]), 8),
            round(float(row["xi2"]), 4),
            round(float(row["xi4"]), 4),
        )
        if key not in seen:
            seen.add(key)
            unique.append(row)

    for row in unique:
        row["S2A_loc"] = -float(row["kA"])
        row["S2B_loc"] = -float(row["kA"])
        row["S2C_loc"] = 0.0
        row["S3_loc"] = -0.5 * float(row["kY"])
        reconstructed = (
            2 * float(row["S3_loc"])
            - float(row["S2A_loc"])
            - float(row["S2B_loc"])
            - float(row["S2C_loc"])
        )
        if abs(reconstructed - float(row["tSJ"])) > 2e-7:
            raise RuntimeError(f"local decomposition failed for job {row['jobid']}")
    return unique


def write_derived(rows: list[dict[str, object]]) -> None:
    fields = [
        "jobid",
        "campaign",
        "variant",
        "chi",
        "stable",
        "xi2",
        "xi4",
        "kA",
        "kY",
        "S2A_loc",
        "S2B_loc",
        "S2C_loc",
        "S3_loc",
        "tSJ",
        "gate_rel",
    ]
    TABLES.mkdir(parents=True, exist_ok=True)
    with DERIVED.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, object]]) -> None:
    stable = [row for row in rows if bool(row["stable"])]
    families = {
        "original": ("o", "original stable branch"),
        "S4": ("s", r"$S_4$ twirl"),
        "S4xZ2": ("D", r"$S_4\!\times\!Z_2$ twirl"),
    }
    components = {
        "S3_loc": ("#6a3d9a", r"$S_{2,\mathrm{loc}}^{(3)}=-\kappa_Y/2$"),
        "S2A_loc": ("#e66101", r"$S_{2,\mathrm{loc}}(A)=S_{2,\mathrm{loc}}(B)=-\kappa_A$"),
    }

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9.2, 9.2), sharex=True)
    fig.suptitle(
        r"critical defect CTMRG: local entropy pieces and correlation lengths"
        "\nexploratory: internally stable branches still fail the four-replica factorization gate",
        fontsize=12,
    )

    for component, (color, label) in components.items():
        first = True
        for family, (marker, _) in families.items():
            group = sorted(
                [row for row in stable if row["campaign"] == family],
                key=lambda row: float(row["chi"]),
            )
            if not group:
                continue
            top.plot(
                [row["chi"] for row in group],
                [row[component] for row in group],
                marker=marker,
                ms=6.5,
                lw=1.0,
                color=color,
                alpha=0.9,
                label=label if first else None,
            )
            first = False
    top.axhline(0.0, color="#555555", lw=1.0, label=r"$S_{2,\mathrm{loc}}(C)=0$ (reference)")
    top.set_xscale("log")
    top.set_ylabel("local entropy contribution")
    top.set_title("(a) Gauge-fixed local decomposition; A and B coincide by mirror symmetry")
    top.grid(alpha=0.22)
    component_legend = top.legend(loc="best", fontsize=9)
    top.add_artist(component_legend)
    family_handles = [
        Line2D([], [], marker=marker, ls="none", color="black", label=label)
        for marker, label in families.values()
    ]
    top.legend(handles=family_handles, loc="lower left", fontsize=8, ncol=3)

    for quantity, color, label in (
        ("xi2", "#2166ac", r"$\xi_2(\chi)$"),
        ("xi4", "#b2182b", r"$\xi_4(\chi)$"),
    ):
        first = True
        for family, (marker, _) in families.items():
            group = sorted(
                [row for row in stable if row["campaign"] == family],
                key=lambda row: float(row["chi"]),
            )
            if not group:
                continue
            bottom.loglog(
                [row["chi"] for row in group],
                [row[quantity] for row in group],
                marker=marker,
                ms=6.5,
                lw=1.1,
                color=color,
                alpha=0.9,
                label=label if first else None,
            )
            first = False
    bottom.set_xlabel(r"requested CTMRG environment dimension $\chi$")
    bottom.set_ylabel(r"correlation length $\xi$")
    bottom.set_title(r"(b) The four-replica environment is scale-starved: $\xi_4\ll\xi_2$")
    bottom.grid(which="both", alpha=0.22)
    bottom.legend(fontsize=9)

    fig.text(
        0.5,
        0.012,
        r"Gauge identity: $2S_{2,\mathrm{loc}}^{(3)}-S_{2,\mathrm{loc}}(A)"
        r"-S_{2,\mathrm{loc}}(B)-S_{2,\mathrm{loc}}(C)=2\kappa_A-\kappa_Y=\widetilde S_J$.",
        ha="center",
        fontsize=9,
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.045, 1, 0.94])
    fig.savefig(OUT, dpi=180)
    plt.close(fig)


def main() -> None:
    rows = load_rows()
    write_derived(rows)
    plot(rows)
    nstable = sum(bool(row["stable"]) for row in rows)
    print(f"wrote {DERIVED} ({len(rows)} unique rows; {nstable} internally stable)")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
