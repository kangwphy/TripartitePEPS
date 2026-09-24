#!/usr/bin/env python3
"""Collect the CONTROLLED twirl on/off comparison.

This is a separate experiment from the production beta scan and must never be
merged into data/rk_ising/ctmrg/tables/beta_chi_scan_split/summary.csv, whose rows are
all flip=1.  Here SRFLIP toggles ONLY the global spin-flip projection of the
one-replica CTMRG (ctm_step_Z_flip -> ctm_step_Z, and sym_flip_edge skipped);
every other setting -- product environment, seam-edge evolution, windows,
junction path, tolerances -- is identical, so the pair (flip=0, flip=1) at the
same (beta, chi1) isolates the effect of the projection.

Also archived here: srsmoke_*.out, the regression run that checked the
degeneracy-aware xi_from_T patch against the stored value
tSJ(beta=0.30, chi1=4) = +0.0109138687.
"""
import csv
import pathlib
import re

ROOT = pathlib.Path("/ix/zdai/kangw/PEPS3EE/PEPS/clean")
LOGDIR = ROOT / "data/rk_ising/ctmrg/logs/twirl_ab"
TABLE = ROOT / "data/rk_ising/ctmrg/tables/twirl_ab/summary.csv"
LINE = re.compile(r"SUMMARY-SPLIT\s+(.*)")
KEEP = ["logfile", "role", "beta", "chi1", "flip", "xi1", "z2split", "tSJ",
        "kA", "kY", "s2", "it1", "hit_nmax", "totalsec"]


def rows():
    out = []
    for logfile in sorted(LOGDIR.glob("*.out")):
        text = logfile.read_text(errors="replace")
        hit = int("WARN pair coevolve hit nmax" in text)
        for match in LINE.finditer(text):
            values = dict(
                token.split("=", 1) for token in match.group(1).split() if "=" in token
            )
            row = {"logfile": logfile.name, "hit_nmax": hit}
            for key, value in values.items():
                if key in {"chi1", "it1", "flip"}:
                    row[key] = int(float(value))
                elif key == "mode":
                    row[key] = value
                else:
                    try:
                        row[key] = float(value)
                    except ValueError:
                        row[key] = value
            row.setdefault("flip", 1)
            row["role"] = ("twirl-off" if row["flip"] == 0 else
                           "xi-patch regression" if logfile.name.startswith("srsmoke")
                           else "twirl-on control")
            out.append(row)
    return sorted(out, key=lambda r: (r["beta"], r["chi1"], r["flip"]))


def main():
    data = rows()
    if not data:
        raise SystemExit(f"no SUMMARY-SPLIT records under {LOGDIR}")
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    with TABLE.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=KEEP, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(data)
    print(f"wrote {TABLE}  ({len(data)} rows)")

    pairs = {}
    for row in data:
        if row["role"].startswith("twirl"):
            pairs.setdefault((row["beta"], row["chi1"]), {})[row["flip"]] = row
    print(f"\n{'beta':>6} {'chi1':>5} {'tSJ(flip=1)':>14} {'tSJ(flip=0)':>14} "
          f"{'|difference|':>13} {'z2split on/off':>22} {'hit nmax on/off':>16}")
    for (beta, chi1), pair in sorted(pairs.items()):
        if 0 in pair and 1 in pair:
            on, off = pair[1], pair[0]
            print(f"{beta:6.2f} {chi1:5d} {on['tSJ']:+14.10f} {off['tSJ']:+14.10f} "
                  f"{abs(on['tSJ']-off['tSJ']):13.2e} "
                  f"{on['z2split']:10.2e} /{off['z2split']:10.2e} "
                  f"{on['hit_nmax']:8d} /{off['hit_nmax']:6d}")
        else:
            only = next(iter(pair.values()))
            print(f"{beta:6.2f} {chi1:5d}  flip={only['flip']} only: "
                  f"tSJ={only['tSJ']:+.10f} (partner from the production scan)")


if __name__ == "__main__":
    main()
