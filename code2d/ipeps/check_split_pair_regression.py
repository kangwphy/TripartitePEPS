#!/usr/bin/env python3
"""Compare optimized pair-factorized runs with the frozen legacy baselines."""

from __future__ import annotations

import csv
from pathlib import Path

from collect_plot_split_replica import SUMMARY


ROOT = Path(__file__).resolve().parents[2]
LOGDIR = ROOT / "data/rk_ising/ctmrg" / "logs" / "split_replica"
OUTPUT = ROOT / "data/rk_ising/ctmrg" / "tables" / "split_replica" / "pair_regression.csv"
JUNCTION_OUTPUT = (
    ROOT / "data/rk_ising/ctmrg" / "tables" / "split_replica" / "junction_regression.csv"
)

# Baselines are the synchronized (non-frozen-projector) production jobs.
BASELINES = {
    (0.30, 2): 23792472,
    (0.30, 3): 23792472,
    (0.30, 4): 23792501,
    (0.35, 2): 23792473,
    (0.35, 3): 23792473,
    (0.35, 4): 23792502,
    (0.44068679351, 2): 23792470,
    (0.44068679351, 3): 23792471,
    (0.44068679351, 4): 23792500,
}
OPTIMIZED = {0.30: 23792600, 0.35: 23792601, 0.44068679351: 23792599}

# Slurm elapsed times for the old single-chi1=4 jobs, recorded by sacct.
OLD_WALL_SECONDS = {23792501: 177.0, 23792502: 235.0, 23792500: 2573.0}
FIELDS = ("xi1", "s2", "kA", "tCA", "tCB", "tAB", "kY", "tSJ")
JUNCTION_RUNS = {
    2: (23792470, 23792638),
    3: (23792471, 23792638),
    4: (23792500, 23792650),
    5: (23792613, 23792639),
    6: (23792614, 23792640),
}


def rows(jobid: int) -> dict[int, dict[str, str]]:
    text = (LOGDIR / f"splitrep_{jobid}.out").read_text()
    return {int(m["chi1"]): m.groupdict() for m in SUMMARY.finditer(text)}


def main() -> None:
    cache = {job: rows(job) for job in set(BASELINES.values()) | set(OPTIMIZED.values())}
    output_rows = []
    for (beta, chi1), old_job in BASELINES.items():
        new_job = OPTIMIZED[beta]
        old, new = cache[old_job][chi1], cache[new_job][chi1]
        diffs = {field: abs(float(new[field]) - float(old[field])) for field in FIELDS}
        maxdiff = max(diffs.values())
        if maxdiff > 5e-10:
            raise RuntimeError(f"regression failed beta={beta} chi1={chi1}: {maxdiff}")
        if float(new["factgate"]) > 1e-12:
            raise RuntimeError(f"factorization gate failed beta={beta} chi1={chi1}")
        old_wall = OLD_WALL_SECONDS.get(old_job, "") if chi1 == 4 else ""
        new_wall = float(new["totalsec"])
        output_rows.append(
            {
                "beta": beta,
                "chi1": chi1,
                "legacy_job": old_job,
                "pair_job": new_job,
                "max_abs_observable_difference": maxdiff,
                "factorization_gate": float(new["factgate"]),
                "pair_environment_seconds": float(new["envsec"]),
                "pair_total_seconds": new_wall,
                "legacy_wall_seconds": old_wall,
                "chi1_4_speedup": float(old_wall) / new_wall if old_wall != "" else "",
            }
        )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_rows[0])
        writer.writeheader()
        writer.writerows(output_rows)

    junction_cache = {
        job: rows(job)
        for pair in JUNCTION_RUNS.values()
        for job in pair
    }
    junction_rows = []
    for chi1, (fused_job, factor_job) in JUNCTION_RUNS.items():
        fused = junction_cache[fused_job][chi1]
        factor = junction_cache[factor_job][chi1]
        diffs = {field: abs(float(factor[field])-float(fused[field])) for field in FIELDS}
        maxdiff = max(diffs.values())
        if maxdiff > 2e-10:
            raise RuntimeError(f"junction regression failed chi1={chi1}: {maxdiff}")
        direct_gate = (factor.get("juncgate") or "") if factor_job == 23792638 else ""
        if direct_gate != "" and float(direct_gate) > 1e-12:
            raise RuntimeError(f"direct fused junction gate failed chi1={chi1}")
        junction_rows.append({
            "chi1": chi1,
            "fused_job": fused_job,
            "factorized_job": factor_job,
            "max_abs_observable_difference": maxdiff,
            "direct_pair_vs_fused_gate": direct_gate,
            "factorized_junction_seconds": factor.get("juncsec") or "",
            "factorized_total_seconds": factor.get("totalsec") or "",
        })
    with JUNCTION_OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=junction_rows[0])
        writer.writeheader()
        writer.writerows(junction_rows)
    print(f"PASS: 9 pair-vs-legacy rows; wrote {OUTPUT}")
    print(f"PASS: 5 factorized-vs-fused junction rows; wrote {JUNCTION_OUTPUT}")


if __name__ == "__main__":
    main()
