#!/usr/bin/env python3
"""Collect per-point atomic outputs without ever mutating the source files."""

from __future__ import annotations

import csv
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("VUMPS_BATCH_ROOT", HERE.parent.parent / "data" / "production_v1"))
GRID = Path(os.environ.get("VUMPS_BATCH_GRID", HERE / "beta_grid.csv"))


def beta_tag(beta: float) -> str:
    return f"{beta:.14f}".replace(".", "p").replace("-", "m")


def one_row(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"expected one row in {path}, found {len(rows)}")
    return rows[0]


rows: list[dict[str, str]] = []
with GRID.open(newline="") as handle:
    for spec in csv.DictReader(handle):
        beta = float(spec["beta"])
        for chi_text in spec["chis"].split(":"):
            chi = int(chi_text)
            point = ROOT / "points" / f"beta_{beta_tag(beta)}" / f"chi_{chi:02d}"
            obs = one_row(point / "observables.csv")
            sixr = one_row(point / "sixr.csv")
            row = {
                "status": "complete" if obs and sixr else
                          "boundary_only" if obs else
                          "failed" if point.exists() else "missing",
                "grid_group": spec["group"],
                "grid_beta": spec["beta"],
                "grid_chi": str(chi),
                "point_dir": str(point.relative_to(ROOT)),
                "boundary_error": str((point / "boundary_error.txt").relative_to(ROOT))
                    if (point / "boundary_error.txt").is_file() else "",
                "observable_error": str((point / "observable_error.txt").relative_to(ROOT))
                    if (point / "observable_error.txt").is_file() else "",
                "sixr_error": str((point / "sixr_error.txt").relative_to(ROOT))
                    if (point / "sixr_error.txt").is_file() else "",
            }
            row.update({f"obs_{key}": value for key, value in obs.items()})
            row.update({f"sixr_{key}": value for key, value in sixr.items()})
            rows.append(row)

fixed = [
    "status", "grid_group", "grid_beta", "grid_chi", "point_dir",
    "boundary_error", "observable_error", "sixr_error",
]
extras = sorted({key for row in rows for key in row if key not in fixed})
fields = fixed + extras
ROOT.mkdir(parents=True, exist_ok=True)
target = ROOT / "summary.csv"
temporary = target.with_suffix(".csv.tmp")
with temporary.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
os.replace(temporary, target)

counts: dict[str, int] = {}
for row in rows:
    counts[row["status"]] = counts.get(row["status"], 0) + 1
print(f"wrote {target} with {len(rows)} requested points: {counts}")
