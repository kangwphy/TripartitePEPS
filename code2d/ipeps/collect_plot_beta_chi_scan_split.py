#!/usr/bin/env python3
"""Collect the dedicated split-replica beta/chi1 campaign and plot it."""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
LOGDIR = ROOT / "data/rk_ising/ctmrg" / "logs" / "beta_chi_scan_split"
TABLE = ROOT / "data/rk_ising/ctmrg" / "tables" / "beta_chi_scan_split" / "summary.csv"
FIGURE = ROOT / "data/rk_ising/ctmrg" / "figures" / "beta_chi_scan_split" / "stilde_vs_chi1_xi1.png"
LINE = re.compile(r"SUMMARY-SPLIT\s+(.*)")
SEAM = re.compile(r"k2 product:.*?seam_res=([+\-0-9.eE]+)")
BETA_C = 0.5 * math.log(1.0 + math.sqrt(2.0))


def parse_rows() -> list[dict[str, object]]:
    latest: dict[tuple[float, int], dict[str, object]] = {}
    for logfile in sorted(LOGDIR.glob("scan_*.out")):
        text = logfile.read_text(errors="replace")
        seam_matches = SEAM.findall(text)
        seam_res = float(seam_matches[-1]) if seam_matches else math.nan
        hit_nmax = "WARN pair coevolve hit nmax" in text
        for match in LINE.finditer(text):
            values: dict[str, str] = {}
            for token in match.group(1).split():
                if "=" in token:
                    key, value = token.split("=", 1)
                    values[key] = value
            if "beta" not in values or "chi1" not in values:
                continue
            row: dict[str, object] = {"logfile": logfile.name}
            for key, value in values.items():
                if key in {"chi1", "chi2", "chi4", "it1", "it2d", "it4max"}:
                    row[key] = int(value)
                elif key == "mode":
                    row[key] = value
                else:
                    try:
                        row[key] = float(value)
                    except ValueError:
                        row[key] = value
            if row.get("mode") != "pair":
                raise RuntimeError(f"non-pair result found in {logfile}: {row.get('mode')}")
            # Runs that predate the SRFLIP switch always used the flip
            # projection unconditionally, so a missing field means flip=1.
            row.setdefault("flip", 1)
            row.setdefault("stop", "edge")   # pre-switch rows used the edge gate
            row["seam_res"] = seam_res
            # NOT a convergence flag.  It records only whether the retired
            # elementwise gate |dT_d2| let the loop exit before SRNMAX.  That
            # gate is gauge and normalization dependent and in practice never
            # fires, so it is simultaneously too strict (it marks the two
            # deepest points in the campaign unconverged although their
            # gauge-invariant corner drift is ~1e-16) and too lax (it passes
            # rows whose stored residual is 0.25).  Kept for provenance only;
            # marker encoding uses sigma_chi, computed below.
            row["pair_edge_iter_capped"] = 1 if hit_nmax else 0
            row["env_converged"] = 0 if hit_nmax else 1
            latest[(round(float(row["beta"]), 10), int(row["chi1"]))] = row
    ordered = sorted(latest.values(),
                     key=lambda row: (float(row["beta"]), int(row["chi1"])))
    # sigma_chi classification: the drift of tSJ between the two largest chi1 at
    # each beta, relative to the value.  This is what decides whether a point may
    # be quoted, and it is what the plot markers encode.  Where the components
    # 2kA and kY nearly cancel, the drift of the DIFFERENCE understates the
    # error, so the component drift is used instead (guard factor 10).
    by_beta = {}
    for row in ordered:
        by_beta.setdefault(round(float(row["beta"]), 10), []).append(row)
    for beta, group in by_beta.items():
        group.sort(key=lambda r: int(r["chi1"]))
        if len(group) < 2:
            for r in group:
                r["sigma_chi"] = float("nan"); r["conv_class"] = "UNKNOWN"
            continue
        a, z = group[-2], group[-1]
        dT = abs(float(z["tSJ"]) - float(a["tSJ"]))
        dA = abs(2*float(z["kA"]) - 2*float(a["kA"]))
        dY = abs(float(z["kY"]) - float(a["kY"]))
        sigma = max(dA, dY) if (dT > 0 and max(dA, dY)/dT > 10) else dT
        rel = sigma/abs(float(z["tSJ"])) if float(z["tSJ"]) else float("inf")
        cls = "SAFE" if rel <= 1e-3 else ("MARGINAL" if rel <= 1e-2 else "EXCLUDE")
        for r in group:
            r["sigma_chi"] = sigma
            r["sigma_rel"] = rel
            r["conv_class"] = cls
    return ordered



def label_at_line_ends(axis, entries, xpad=0.035, min_gap_frac=0.026):
    """Write each series' label beside its own right-hand end, not in a legend.

    Eighteen betas on a continuous colormap are not separable by eye -- 0.41,
    0.42, 0.43, 0.46 and 0.48 come out the same green -- so the beta is placed
    next to the curve it belongs to.  Labels are pushed apart vertically by a
    single upward sweep so that near-degenerate series (the whole ordered side
    lives within 0.007 of zero) stay readable, and a thin leader line keeps each
    label attached to its curve once it has been displaced.
    """
    if not entries:
        return
    # Separate in AXES-FRACTION space, not data space: the y axis is symlog, so a
    # fixed data gap would be huge near zero and invisible at 0.17.
    tr = axis.transData
    inv = axis.transAxes.inverted()
    def to_frac(y):
        return inv.transform(tr.transform((axis.get_xlim()[0], y)))[1]
    def from_frac(f):
        return tr.inverted().transform(axis.transAxes.transform((0.0, f)))[1]
    gap = min_gap_frac
    entries = sorted(entries, key=lambda e: to_frac(e[1]))
    placed = []
    for x, y, text, color in entries:
        f = to_frac(y)
        fl = f if not placed else max(f, placed[-1][1] + gap)
        placed.append((x, fl, y, text, color))
    overflow = placed[-1][1] - (1.0 - 0.5 * gap)
    if overflow > 0:
        shift = min(overflow, placed[0][1] - 0.5 * gap)
        if shift > 0:
            placed = [(x, fl - shift, y, t, c) for x, fl, y, t, c in placed]
    placed = [(x, from_frac(fl), y, t, c, abs(fl - to_frac(y)))
              for x, fl, y, t, c in placed]
    xlo, xhi = axis.get_xlim()
    if axis.get_xscale() == "log":
        import math as _m
        xtext = _m.exp(_m.log(xhi) + xpad * (_m.log(xhi) - _m.log(xlo)))
    else:
        xtext = xhi + xpad * (xhi - xlo)
    for x, yl, y, text, color, moved in placed:
        # A leader is drawn only when the label had to be displaced; otherwise it
        # sits level with its own curve and a line across the panel would read as
        # spurious data.
        arrow = (dict(arrowstyle="-", color=color, lw=0.45, alpha=0.4,
                      shrinkA=0, shrinkB=1) if moved > 0.4 * min_gap_frac
                 else None)
        axis.annotate(text, xy=(x, y), xytext=(xtext, yl),
                      textcoords="data", color=color, fontsize=7.5,
                      va="center", ha="left", clip_on=False,
                      annotation_clip=False, arrowprops=arrow)


# ---------------------------------------------------------------------------
# xi sanity gate and diagnostic override.
#
# Two ways the stored xi1 goes wrong, both seen in the data:
#   * pre-patch rows used the naive estimator, which divides by log(1) in the
#     ordered phase and returns 1e15 or -inf (13 rows at beta = 0.46/0.48/0.60);
#   * near beta_c on the ordered side the Z2 splitting is resolved but not yet at
#     machine precision, and at one intermediate chi1 the truncation squeezes it
#     anomalously small, so -1/log(lambda2/lambda1) blows up.  At beta=0.441425,
#     chi1=10 the splitting reads 3.7e-4 and xi comes out 2703 against a domain
#     wall length of 339 -- eight times too long -- while chi1=16 and above have
#     the splitting at 1e-16 and behave.
# The robust test is physical rather than spectral: xi cannot exceed the length
# it is trying to resolve, xi_spin in the disordered phase and 2*xi_spin (the
# single domain wall) in the ordered one.  Rows failing it are replaced by the
# dedicated xi-only diagnostic when one exists, and otherwise blanked.
# ---------------------------------------------------------------------------
def _xi_target(beta):
    bd = -0.5*math.log(math.tanh(beta))
    if abs(beta - BETA_C) < 1e-9:
        return math.inf
    return 1/(2*(bd-beta)) if beta < BETA_C else 2/(4*(beta-bd))


def _load_xi_diagnostic():
    import glob, re
    out = {}
    for path in glob.glob("/ix/zdai/kangw/PEPS3EE/PEPS/clean/logs2d/xi*_*.out"):
        for line in open(path, errors="replace"):
            m = re.search(r"SUMMARY-XI beta=([\d.]+)\s+chi1=\s*(\d+).*?"
                          r"xi_conn=\s*([\d.eE+-]+)", line)
            if m:
                out[(round(float(m.group(1)), 6), int(m.group(2)))] = float(m.group(3))
    return out


def apply_xi_gate(rows):
    diag = _load_xi_diagnostic()
    nfix = nblank = 0
    for r in rows:
        b, c = float(r["beta"]), int(r["chi1"])
        try:
            x = float(r["xi1"])
        except (ValueError, TypeError):
            x = float("nan")
        tgt = _xi_target(b)
        ok = math.isfinite(x) and 0 < x <= 1.15*tgt
        if ok:
            continue
        alt = diag.get((round(b, 6), c))
        if alt is not None and 0 < alt <= 1.15*tgt:
            r["xi1"] = f"{alt:.8f}"; nfix += 1
        else:
            # "nan" rather than "": downstream readers call float() on this
            # field, and every consumer already filters on isfinite.
            r["xi1"] = "nan"; nblank += 1
    if nfix or nblank:
        print(f"xi gate: {nfix} rows replaced from the xi-only diagnostic, "
              f"{nblank} blanked (no valid value available)")
    return rows

def main() -> None:
    rows = apply_xi_gate(parse_rows())
    if not rows:
        raise SystemExit(f"no completed SUMMARY-SPLIT records under {LOGDIR}")

    TABLE.parent.mkdir(parents=True, exist_ok=True)
    # sigma_chi / sigma_rel / conv_class are the columns that decide whether a
    # row may be quoted; env_converged is retained only for provenance and is
    # renamed in spirit by pair_edge_iter_capped sitting next to it.
    fields = ["logfile", "beta", "chi1", "chi2", "chi4", "xi1", "z2split",
              "tSJ", "s2", "kA", "kY", "sigma_chi", "sigma_rel", "conv_class",
              "factgate", "plain2", "plain4", "seam_res",
              "pair_edge_iter_capped", "env_converged",
              "mode", "stop", "flip", "totalsec"]
    with TABLE.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    # Two phases, two rows.  With forty betas a single panel stacks forty labels
    # and the ordered side (all inside |S| < 0.2) is squeezed against the zero
    # line by the disordered one.  Splitting them lets each row keep its own
    # vertical range and its own label column.  beta_c is drawn in both chi1
    # panels, in black, as the common reference.
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 11.5))
    betas = sorted({float(row["beta"]) for row in rows})
    dis = [b for b in betas if b < BETA_C - 1e-9]
    ordd = [b for b in betas if b > BETA_C + 1e-9]
    crit = [b for b in betas if abs(b - BETA_C) < 1e-9]

    def draw(row_axes, blist, labels_chi, labels_xi):
        cmap = plt.cm.viridis(
            [i / max(len(blist) - 1, 1) for i in range(len(blist))])
        for beta, color in zip(blist, cmap):
            group = sorted((r for r in rows
                            if abs(float(r["beta"]) - beta) < 1e-10),
                           key=lambda r: int(r["chi1"]))
            if not group:
                continue
            xs = [int(r["chi1"]) for r in group]
            ys = [float(r["tSJ"]) for r in group]
            cls = group[0].get("conv_class", "UNKNOWN")
            mk = "s" if cls == "EXCLUDE" else "o"
            row_axes[0].plot(xs, ys, "-", color=color, lw=1.4)
            row_axes[0].scatter(xs, ys, marker=mk, s=52 if mk == "s" else 42,
                                edgecolor=color,
                                facecolor=color if cls == "SAFE" else "white",
                                zorder=3)
            labels_chi.append((xs[-1], ys[-1], fr"${beta:.4g}$", color))
            xg = [r for r in group
                  if math.isfinite(float(r["xi1"])) and 0 < float(r["xi1"]) < 1e6]
            if xg:
                xv = [float(r["xi1"]) for r in xg]
                yv = [float(r["tSJ"]) for r in xg]
                row_axes[1].plot(xv, yv, "-", color=color, lw=1.4)
                row_axes[1].scatter(xv, yv, marker=mk, s=52 if mk == "s" else 42,
                                    edgecolor=color,
                                    facecolor=color if cls == "SAFE" else "white",
                                    zorder=3)
                labels_xi.append((xv[-1], yv[-1], fr"${beta:.4g}$", color))

    lab = [[], [], [], []]
    draw(axes[0], dis, lab[0], lab[1])
    draw(axes[1], ordd, lab[2], lab[3])
    for r in (0, 1):                       # beta_c reference in both chi1 panels
        for beta in crit:
            g = sorted((x for x in rows if abs(float(x["beta"]) - beta) < 1e-10),
                       key=lambda x: int(x["chi1"]))
            axes[r][0].plot([int(x["chi1"]) for x in g],
                            [float(x["tSJ"]) for x in g], "k-D", ms=5, lw=2.0,
                            zorder=5)
            lab[0 if r == 0 else 2].append(
                (int(g[-1]["chi1"]), float(g[-1]["tSJ"]), r"$\beta_c$", "black"))

    axes[0][0].set_title(r"disordered  $\beta<\beta_c$   vs $\chi_1$", fontsize=12)
    axes[0][1].set_title(r"disordered  $\beta<\beta_c$   vs $\xi_1$", fontsize=12)
    axes[1][0].set_title(r"ordered  $\beta>\beta_c$   vs $\chi_1$", fontsize=12)
    axes[1][1].set_title(r"ordered  $\beta>\beta_c$   vs $\xi_1$", fontsize=12)
    for r in (0, 1):
        axes[r][0].set_xlabel(r"one-replica $\chi_1$")
        axes[r][1].set_xlabel(r"one-replica $\xi_1$")
        axes[r][1].set_xscale("log")
        for a in axes[r]:
            a.set_ylabel(r"split-replica $\widetilde S_J$")
            a.axhline(0.0, color="0.45", lw=0.8, ls=":")
            a.grid(alpha=0.25, which="both")
    # The disordered chi1 panel spans 1e-3 to 1 in both signs once beta_c is on
    # it, so it stays symlog; every other panel keeps a linear ordinate, which is
    # what makes the growth of |S| with xi readable.
    axes[0][0].set_yscale("symlog", linthresh=1e-3, linscale=0.5)
    axes[1][0].set_yscale("symlog", linthresh=1e-4, linscale=0.5)
    for k, (a, entries) in enumerate(((axes[0][0], lab[0]), (axes[0][1], lab[1]),
                                      (axes[1][0], lab[2]), (axes[1][1], lab[3]))):
        lo, hi = a.get_xlim()
        if a.get_xscale() == "log":
            a.set_xlim(lo, hi * 1.6)
        else:
            a.set_xlim(lo, hi + 0.20 * (hi - lo))
        label_at_line_ends(a, entries)

    status_handles = [
        Line2D([0], [0], marker="o", color="0.25", markerfacecolor="0.25",
               lw=0, label=r"SAFE ($\sigma_\chi/|\widetilde S|\leq10^{-3}$)"),
        Line2D([0], [0], marker="o", color="0.25", markerfacecolor="white",
               lw=0, label=r"MARGINAL ($10^{-3}$--$10^{-2}$)"),
        Line2D([0], [0], marker="s", color="0.25", markerfacecolor="white",
               lw=0, label=r"EXCLUDE ($>10^{-2}$)"),
        Line2D([0], [0], color="black", lw=2.0, marker="D", ms=5,
               label=r"$\beta=\beta_c$"),
    ]
    fig.legend(handles=status_handles, fontsize=9.5, ncol=4,
               loc="lower center", bbox_to_anchor=(0.5, -0.01), frameon=False)
    present = {(round(float(r["beta"]), 10), int(r["chi1"])) for r in rows}
    nbeta = len({b for b, _ in present})
    classes = {round(float(r["beta"]), 10): r.get("conv_class", "UNKNOWN")
               for r in rows}
    tally = {c: sum(1 for v in classes.values() if v == c)
             for c in ("SAFE", "MARGINAL", "EXCLUDE", "UNKNOWN")}
    missing = []
    fig.suptitle(f"scale-matched split-replica scan: {nbeta} "
                 fr"$\beta$, {len(present)} points  |  "
                 f"SAFE {tally['SAFE']}, MARGINAL {tally['MARGINAL']}, "
                 f"EXCLUDE {tally['EXCLUDE']}", fontsize=14)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(FIGURE, dpi=190)
    print(f"wrote {TABLE}")
    print(f"wrote {FIGURE}")
    if missing:
        print("missing: " + ", ".join(f"beta={beta:.2f},chi1={chi}" for beta, chi in missing))


if __name__ == "__main__":
    main()
