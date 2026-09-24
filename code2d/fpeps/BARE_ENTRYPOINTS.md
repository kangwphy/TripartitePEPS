# Fermionic bare/direct entry points

All paths below are relative to `clean/code2d/fpeps/`.

| Purpose | File |
| --- | --- |
| Original two-/three-column fixed points, no grown tensor/endmap | `benchmark/bare_direct_core.jl` |
| Larger-boundary fixed-point backend | `benchmark/wide_bare_direct_core.jl` |
| Bare LMPS, signed replica sewing and depth convergence | `benchmark/wide_bare_direct_scan_point.jl` |
| Independently solved boundary-pair adapter and input gates | `benchmark/independent_bivumps/bare_wide.py` |
| Saved boundary audit | `benchmark/independent_bivumps/audit_saved.jl` |
| Backend load check only | `test/bare_load_smoke.jl` |

The other `bare_*.py` files preserve experiment-specific gates and provenance;
they are not interchangeable acceptance policies. Do not relabel the older
`direct_graded_lmps_probe.jl` grown-tail diagnostic as this bare route.
Boundary convergence, successful contraction, and physical accuracy remain
separate checks. A migration does not certify historical entropy points.

## Submission

The Python adapter takes four **absolute** arguments: native-boundary directory,
rotated-boundary directory, derivative-control directory, and a new output
directory. It verifies the saved reports/checkpoints before calling the backend.
From a shell with `CLEAN_ROOT=/ix/zdai/kangw/PEPS3EE/PEPS/clean` exported:

```bash
sbatch -M htc --partition=preempt \
  --output="$CLEAN_ROOT/verification/fermion-bare-%j.out" \
  "$CLEAN_ROOT/code2d/fpeps/jobs/run_python.sh" \
  benchmark/independent_bivumps/bare_wide.py \
  /absolute/native /absolute/rotated /absolute/control /absolute/new_output
```

`jobs/run_cpu.sh` is a compatibility wrapper for Julia subprocess calls from
the unchanged Python adapters; it delegates to the clean, independent fermion
environment and refuses execution outside a `preempt` job. It does not submit
an extra job when called inside an existing allocation. Historical README
commands with other partitions are not current instructions.
The Python launcher reuses the existing `.plot_deps` package installation
(override with `FPEPS_PYTHON_DEPS`); Julia similarly reuses dependency depots.
These are explicit software-environment dependencies, not old source entry points.

The full raw data is `../../data/fermion/store/` (also available through local
`data/`). Original absolute provenance paths remain supported by compatibility
aliases. Original code remains available for existing queued jobs and rollback.
Copied scientific code is byte-identical; the manifest is
`../../verification/fermion_bare_migration.json`.
