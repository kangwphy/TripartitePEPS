# RK-Ising: independent symmetry and contraction options

This is the unified, opt-in entry point for the RK-Ising VUMPS + bare/direct
calculation. It does not replace the historical campaign scripts or their
datasets. The algorithm is selected by two independent Boolean configuration
keys, not by editing source files:

| Series | `use_z2` | `use_cap_fix` |
| --- | --- | --- |
| No constraint | `false` | `false` |
| Z2 | `true` | `false` |
| Z2 + fix | `true` | `true` |
| No constraint + fix (optional diagnostic) | `false` | `true` |

Both options default to `false` in the point-driver API. Campaign initialization
requires both choices explicitly. A campaign's switches, grid, input copies,
and source snapshot are immutable after initialization. Use a different output
directory to change them. Do not merge results solely on `(beta, chi)` without
retaining the series and provenance.

## What is changed, and what is not

1. **Boundary MPS.** `use_z2=false` uses the existing dense VUMPS solver.
   `use_z2=true` uses the bosonic Z2 adapter, with an even/odd virtual-space
   split of `chi/2` each in this campaign. It exports the graded solution
   losslessly to dense arrays for subsequent contractions. It is not a dense
   reoptimization. Saved tensors include `AL`, `AR`, `AC`, and `C`.
2. **LMPS construction.** All modes use the same literal two-column `LR` and
   three-column `LTR` fixed-point construction and the same cardinal regional
   rails. They do not switch to the grown-tail/endpoint-map LMPS approximation.
   Reconstruction is checked against the original strip and boundary MPS.
3. **Final contraction.** `use_cap_fix=false` uses the historical procedure:
   an Arnoldi direction, followed by propagation of the specified far boundary
   to obtain its amplitude. The new static adapter provides an explicit tail
   iteration limit and preserves finite diagnostic output; it does not mutate
   the legacy solver or rewrite its source at runtime.
4. **Cap-selected fix.** `use_cap_fix=true` propagates the specified far boundary
   itself with normalized power iteration. For an exactly degenerate dominant
   eigenvalue this preserves the combination selected by that boundary,
   instead of comparing it with an independently chosen eigenvector. The fix
   is applied consistently to all three one-copy metric seams and all six
   replica seams. The API name `endpoint_solver` refers to these final seam
   environments, not to an endpoint-map replacement of `LR` or `LTR`.

The one-copy seed is the identity in the specified seam coordinates; replica
seeds are synchronized products of the one-copy solutions, including their
amplitudes. The stored seam checkpoints contain the actual seeds, propagated
environments, and convergence histories. No seed is fitted to CTMRG or to a
broken-state entropy. This fixes the numerical selection of a **specified**
far boundary; it does not prove that the boundary is appropriate for every
physical question or arbitrary independent gauge changes. Unequal-phase
peripheral eigenvalues can also prevent an ordinary power limit from existing.

At fixed total `chi`, a symmetric state and a broken state need not have the
same effective capacity per symmetry sector. Agreement should be checked, not
assumed or enforced. Likewise, the full transfer-spectrum correlation length
can be infinite in a cat state. The Z2 output separately records the full,
nonperipheral, and symmetry-sector lengths. These are not interchangeable with
the exact connected spin correlation length `xi_spin_exact`.

## Convergence and diagnostic values

Every solver has a maximum iteration count and may stop early on convergence.
For the fixed contraction, the state/eigen-residual and amplitude/phase
stability are checked separately. With retention enabled, reaching the limit
can still produce a finite entropy and its components. Such a value is saved
as diagnostic, not silently discarded and not relabelled as converged.
Nonfinite or structurally invalid contractions can still fail without an
entropy. Every requested point remains represented in `summary.csv`.

`numerical_pass` additionally checks the boundary/direct solves, reconstruction,
symmetry audits where applicable, seam residuals, closure consistency, and
retained rank. It is a numerical status, not a proof of physical correctness.
`production_release=false` is intentionally retained pending scientific review.
Plots should distinguish diagnostics but may connect them when explicitly
requested; no automatic dip removal or curve-shape selection is performed.

## Data layout

The present three series are separated under `clean/data/rk_ising/`:

- `vumps_direct_logscan_v1/`: historical no-constraint data, unchanged.
- `vumps_direct_z2_logscan_v1/`: historical Z2 data, unchanged.
- `vumps_direct_z2_fix_logscan_v1/`: complete-grid Z2 + fix rerun.

Each new campaign has only a small set of functional subdirectories:

```text
campaign/
  config.toml, requested_grid.csv, inputs.csv, source_hashes.json
  summary.csv, collection.json
  points/        # readable measurements, spectra, and seam audits
  checkpoints/   # MPS, direct G/B, seam states, result; readable tensor CSV too
  figures/       # figures only
  logs/          # scheduler logs
  provenance/    # source configurations, checksums, grid statistics
  code_snapshot/ # frozen runnable package
```

An import is a byte-preserving snapshot, never a live symlink or an output into
the source series. Reuse does not require the previous six-R job to finish:
the MPS and direct checkpoints are validated separately. Missing or unconverged
boundary inputs trigger a fresh solve; mismatched direct parents are recomputed.
Corrupt identities/checksums stop the point rather than silently substituting
an unverified state. Checkpoints and raw data are excluded from GitHub. Source,
configuration examples, and tests are published.

## Running on Pitt CRC

All computations, including initialization, tests, collection, and plotting,
must use Slurm **preempt**. The launcher requests the SMP cluster, four CPUs,
requeue, and a 21-day wall-time limit; a finite limit is not an iteration target.
The examples below contain no artificial `%N` array-concurrency cap.

Set paths to your checkout and a new, independent output root:

```bash
export CLEAN_ROOT=/path/to/clean
export RK_CAMPAIGN_ROOT=/path/to/new-rk-campaign
launcher="$CLEAN_ROOT/code2d/lmps/vumps/jobs/rk_ising_campaign.sh"
mkdir -p "$RK_CAMPAIGN_ROOT/logs"
sbatch -M smp --partition=preempt --export=ALL \
  --output="$RK_CAMPAIGN_ROOT/logs/test_%j.log" "$launcher" test
sbatch -M smp --partition=preempt --export=ALL \
  --output="$RK_CAMPAIGN_ROOT/logs/init_%j.log" "$launcher" init \
  --source-grid "$CLEAN_ROOT/code2d/lmps/vumps/production/rk_ising/example_grid.csv" \
  --source-config "$CLEAN_ROOT/code2d/lmps/vumps/production/rk_ising/example_config.toml" \
  --z2 --fix --expected-count 2
```

After successful initialization, submit the frozen point driver. This example
grid has two points; use the exact task lists and memory requests in
`provenance/grid_statistics.json` for a full campaign:

```bash
sbatch -M smp --partition=preempt --export=ALL --array=0-1 --mem=32G \
  --output="$RK_CAMPAIGN_ROOT/logs/point_%A_%a.log" "$launcher" point
sbatch -M smp --partition=preempt --export=ALL \
  --output="$RK_CAMPAIGN_ROOT/logs/collect_%j.log" "$launcher" collect
squeue -M smp -u "$USER" -o '%.18i %.12P %.10T %.30j'
```

Collection is a snapshot; run it again after jobs finish, or use an `afterany`
dependency on the array job IDs. Verification/collection checks the frozen
source and imported-input hashes. To reuse a previous campaign, supply its
exact `requested_grid.csv` and add `--reuse-from /path/to/source`. Cross-Z2
imports are rejected. To run the other two principal modes use respectively
`--no-z2 --no-fix` or `--z2 --no-fix`, each in a separate output root.

The launcher accepts `JULIA_EXE`, `JULIA_DEPOT_PATH`, and `RK_PYTHON_EXE`
overrides. Python 3.11+ is required. At CRC the default Python module is loaded
when no interpreter override is supplied. Julia dependencies are pinned by
the package's `Project.toml` and `Manifest.toml`.

## Verification

The batch `test` action runs controller/provenance tests, cap-selected synthetic
degeneracy tests, the legacy package regressions, the four-mode dispatch and
analytic-product checks, and the graded-boundary/export smoke test. It checks
that enabling the fix is independent of enabling Z2. Real saved-state replay
is also required before promoting a new campaign: compare individual entropy
components and residuals, not only the final plotted line.
