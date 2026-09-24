#!/usr/bin/env python3
"""Combine established additional-beta and current near-critical plateaus."""

import csv
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXTRA = os.path.join(
    ROOT, "data/rk_ising/mc", "campaigns",
    "sw_components_extra_betas_100k_16chains",
)
NEAR = os.path.join(
    ROOT, "data/rk_ising/mc", "campaigns",
    "sw_components_near_critical_100k_16chains",
)
BETA_C = 0.5 * math.log(1.0 + math.sqrt(2.0))


def as_bool(value):
    """Interpret a CSV Boolean written by Python."""
    return str(value).strip().lower() in ("true", "1", "yes")


def read_rows():
    """Read final extra plateaus and the best available near-critical fits."""
    rows = []
    with open(os.path.join(EXTRA, "plateau_summary.csv"), newline="") as stream:
        for row in csv.DictReader(stream):
            rows.append({
                "beta": row["beta"],
                "xi": float(row["xi_spin"]),
                "value": float(row["direct_plateau"]),
                "error": float(row["direct_error"]),
                "chi2": float(row["direct_chi2_dof"]),
                "tail": row["tail_sizes"],
                "source": "additional beta",
                "certified": as_bool(row["direct_plateau_certified"]),
                "provisional": False,
            })

    final_path = os.path.join(NEAR, "plateau_summary.csv")
    partial_path = os.path.join(NEAR, "partial_plateau_summary.csv")
    if os.path.isfile(final_path):
        with open(final_path, newline="") as stream:
            final_rows = list(csv.DictReader(stream))
    else:
        final_rows = []
    if len(final_rows) == 9:
        for row in final_rows:
            rows.append({
                "beta": row["beta"],
                "xi": float(row["xi_spin"]),
                "value": float(row["direct_plateau"]),
                "error": float(row["direct_error"]),
                "chi2": float(row["direct_chi2_dof"]),
                "tail": row["tail_sizes"],
                "source": "near critical",
                "certified": as_bool(row["direct_certified"]),
                "provisional": False,
            })
    else:
        with open(partial_path, newline="") as stream:
            for row in csv.DictReader(stream):
                rows.append({
                    "beta": row["beta"],
                    "xi": float(row["xi_spin"]),
                    "value": float(row["plateau"]),
                    "error": float(row["error"]),
                    "chi2": float(row["chi2_dof"]),
                    "tail": row["tail_sizes"],
                    "source": "near critical",
                    "certified": float(row["chi2_dof"]) <= 3.0,
                    "provisional": True,
                })
    rows.sort(key=lambda row: row["xi"])
    return rows


def style(row):
    """Encode phase by color and campaign by marker."""
    color = "#2e5fa3" if float(row["beta"]) < BETA_C else "#c4552d"
    if row["source"] == "near critical":
        marker = "D"
    elif float(row["beta"]) < BETA_C:
        marker = "o"
    else:
        marker = "s"
    return color, marker


def plot_point(axis, x, row, log_y=False):
    """Plot a measurement, an uncertified open point, or an upper limit."""
    color, marker = style(row)
    value, error = row["value"], row["error"]
    if log_y and value - error <= 0.0:
        upper = value + 1.645 * error
        if upper <= 0.0:
            return None
        axis.scatter(
            [x], [upper], marker="v", s=43, facecolors="none",
            edgecolors=color, linewidths=1.3,
        )
        return upper
    marker_face = color if row["certified"] else "none"
    axis.errorbar(
        x, value, yerr=error, fmt=marker, ms=6, capsize=3,
        color=color, markerfacecolor=marker_face,
        markeredgecolor=color, markeredgewidth=1.2,
    )
    return value


def write_table(rows):
    """Record the exact combined inputs displayed in the figure."""
    fields = (
        "beta", "xi", "value", "error", "chi2", "tail", "source",
        "certified", "provisional",
    )
    provisional = any(row["provisional"] for row in rows)
    name = (
        "partial_combined_plateau_summary.csv"
        if provisional else "combined_plateau_summary.csv"
    )
    with open(os.path.join(NEAR, name), "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_plot(rows):
    """Draw the combined linear, log-coordinate, and log-log views."""
    provisional = any(row["provisional"] for row in rows)
    offsets = {
        "0.80": (5, -14), "0.10": (5, 7), "0.70": (5, 9),
        "0.20": (5, 5), "0.25": (5, 5), "0.35": (5, -13),
        "0.36": (5, 7), "0.38": (5, 5), "0.40": (5, 5),
    }
    fig, axes = plt.subplots(1, 3, figsize=(16.4, 4.8), dpi=180)
    transforms = (
        (lambda row: row["xi"], r"$\xi_{\rm spin}(\beta)$", "(a) linear--linear"),
        (lambda row: math.log(row["xi"]), r"$\ln\xi_{\rm spin}(\beta)$",
         "(b) log coordinate--linear"),
    )
    for axis, (x_value, x_label, title) in zip(axes[:2], transforms):
        for row in rows:
            x = x_value(row)
            y = plot_point(axis, x, row)
            axis.annotate(
                rf"$\beta={row['beta']}$", (x, y),
                xytext=offsets.get(row["beta"], (5, 5)), textcoords="offset points",
                fontsize=7.3,
            )
        axis.axhline(0.0, color="0.65", ls=":", lw=0.9)
        axis.set_xlabel(x_label)
        axis.set_title(title)
        axis.grid(alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(r"direct plateau $\widetilde S_\infty(\beta)$")
    axes[1].set_ylim(axes[0].get_ylim())

    axis = axes[2]
    for row in rows:
        x = row["xi"]
        y = plot_point(axis, x, row, log_y=True)
        if y is None:
            continue
        axis.annotate(
            rf"$\beta={row['beta']}$", (x, y),
            xytext=offsets.get(row["beta"], (5, 5)), textcoords="offset points",
            fontsize=7.3,
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(r"$\xi_{\rm spin}(\beta)$")
    axis.set_ylabel(r"direct plateau $\widetilde S_\infty(\beta)$")
    axis.set_title("(c) log--log")
    axis.grid(alpha=0.22, which="both")
    axis.spines[["top", "right"]].set_visible(False)
    axis.text(
        0.02, 0.02,
        "Negative estimates are omitted on the logarithmic y axis; see (a,b).",
        transform=axis.transAxes, fontsize=7.2, va="bottom",
    )

    handles = [
        Line2D([], [], marker="o", ls="none", color="#2e5fa3",
               label="additional: disordered"),
        Line2D([], [], marker="s", ls="none", color="#c4552d",
               label="additional: ordered"),
        Line2D([], [], marker="D", ls="none", color="#2e5fa3",
               label=("near-critical: incomplete" if provisional
                      else "near-critical: complete")),
        Line2D([], [], marker="o", ls="none", color="0.25",
               markerfacecolor="none", label="tail check failed"),
        Line2D([], [], marker="v", ls="none", color="0.25",
               markerfacecolor="none", label="95% upper limit"),
    ]
    axes[0].legend(handles=handles, frameon=False, fontsize=7.5, loc="upper left")
    title = "Combined Swendsen--Wang plateau estimates"
    if provisional:
        title += " (incomplete near-critical coverage)"
    fig.suptitle(title)
    fig.tight_layout()
    name = (
        "partial_combined_plateau_three_axis_views.png"
        if provisional else "plateau_three_axis_views.png"
    )
    fig.savefig(os.path.join(NEAR, name), bbox_inches="tight")
    plt.close(fig)


def main():
    rows = read_rows()
    write_table(rows)
    make_plot(rows)
    print(f"combined {len(rows)} plateau estimates")
    for row in rows:
        print(
            f"beta={row['beta']} source={row['source']} "
            f"plateau={row['value']:+.7f}+-{row['error']:.7f} "
            f"chi2/dof={row['chi2']:.2f} certified={row['certified']}"
        )


if __name__ == "__main__":
    main()
