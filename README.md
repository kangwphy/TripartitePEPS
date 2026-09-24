# TripartitePEPS

This directory is the new entry point. The cleanup changes file layout, not
the numerical definitions or existing acceptance criteria. Original code is
retained outside this directory for rollback and jobs already in the queue.

The GitHub repository contains code, source notes and regression tests only.
Raw data, checkpoints, local archives and migration manifests remain on the
cluster and are intentionally excluded. Paths in `DATA_GUIDE.md` describe that
local data installation; a fresh clone does not contain those datasets.

Set `CLEAN_ROOT` to the root of this checkout. The launchers are configured for
Pitt CRC and require **preempt**; `JULIA_EXE`, `JULIA_DEPOT_PATH` and
`FPEPS_PYTHON_DEPS` can override the local software installations. Instantiate
`code2d/lmps/vumps` and `code2d/fpeps/environments/default` as separate Julia
projects in an appropriate allocation before first use on another machine.

```
code2d/
  lmps/vumps/
    src/algorithms/          # boundary solvers, direct LR/LTR, bridge, six-R
    src/models/              # RK-Ising, TFIM, RVB-D3
    src/LMPSVUMPS.jl         # same public module/API
    production/{rk_ising,tfim,rvb_d3}/
    jobs/                    # preempt-only launcher
    test/
  fpeps/                     # graded fermion code, independent environment
  ipeps/                     # RK-Ising CTMRG reference
  mc/                        # RK-Ising Monte Carlo
data/
  rk_ising/{mc,ctmrg,results,vumps}/
  tfim/{current,store}/       # live D2–D8 entry; complete raw store + checkpoints
  fermion/store/             # complete independent fermion data store
  rvb_d3/store/              # complete raw campaigns; scientific status unchanged
  runtime_logs/             # relocated solver logs
```

## What to use

| Task | Entry point, relative to `code2d/lmps/vumps/` |
| --- | --- |
| Public Julia API | `src/LMPSVUMPS.jl` |
| TFIM C4v variational GS | `production/tfim/quantum_tfim_c4v_scan_point.jl` |
| TFIM **bare/direct** LR/LTR | `production/tfim/run_dense_bare_cardinal.jl` |
| Six-R from saved bare cardinal network | `production/tfim/quantum_tfim_bare_ladder_sixr_sliced.jl` |
| RK-Ising bare saved-state benchmark | `production/rk_ising/rk_ising_bare_saved_points.jl` |
| Historical RK-Ising scan | `production/rk_ising/production_scan_beta.jl` |
| RVB-D3 adapter | `src/models/rvb_d3.jl` |

TFIM GS optimization uses the existing CTMRG-based variational routine; VUMPS
contracts a saved PEPS boundary. They are not the same optimization stage.
The main TFIM entropy route above is **bare/direct**, not grown-tail or endpoint.
The old endpoint routines remain in the API for reproducibility; organizing
them here does not establish their equivalence to the original network.
The existing RVB driver is retained as an experimental common-chart driver,
not relabeled as direct or validated.

Fermionic bare/direct entry points and their independent Slurm launchers are
listed in [fpeps/BARE_ENTRYPOINTS.md](code2d/fpeps/BARE_ENTRYPOINTS.md).

## Data and provenance

* `DATA_MOVES.json` and `RK_RAW_MOVES.json`: physically relocated inactive RK/MC/CTMRG stores; the old
  paths are compatibility symlinks. Nothing was deleted.
* `DATA_FILES.csv`: stable live-data snapshots, source paths, sizes, hashes and
  status labels. A snapshot is NOT a new convergence/physics validation.
* `data/tfim/GS_INDEX.csv`: converged-in-source-record GS checkpoints, locally
  stored by SHA-256; records retain D, h, branch, CTM chi and original provenance.
* `DATA_SNAPSHOT_ISSUES.json`: missing references, changed files or hash failures.
  These are not silently promoted into the valid dataset.
* **Start with `data/tfim/current/` for live TFIM results.** The complete raw
  store, including intermediate optimization checkpoints and historical runs,
  is physically in `data/tfim/store/`. `live` is an alias to its `current` view.
* Complete fermion and RVB raw stores are physically under `data/fermion/store/`
  and `data/rvb_d3/store/`. Old data paths are compatibility symlinks into this
  tree, so previously submitted jobs continue writing to the relocated data.
* `verification/full_migration/` records whole-directory moves, preserved
  directory inodes, and an unfiltered checkpoint inventory. Earlier snapshot
  indexes remain useful frozen selections, but are not the full live store.
* Fermion finite-window acceptance does not certify infinite-plane entropy;
  RVB and unaccepted infinite-fermion results stay in `diagnostics`.

`SOURCE_MAP.csv` records the source and cleaned-file hashes. A refactored file
can have a different hash because includes and model placement changed.
Original serialized checkpoints were relocated without deserialization or rewriting.

## Running (Pitt CRC: preempt only)

Set the root explicitly; Slurm copies batch scripts, so they must not infer
their installation path from the submitted script's location.

```bash
export CLEAN_ROOT=/ix/zdai/kangw/PEPS3EE/PEPS/clean
sbatch -M smp --partition=preempt --export=ALL,CLEAN_SMOKE=1 \
  --output="$CLEAN_ROOT/verification/vumps-%j.out" \
  "$CLEAN_ROOT/code2d/lmps/vumps/jobs/run_preempt.sh" test/runtests.jl
```

Drivers retain their documented environment-variable inputs. Set explicit
input/output paths for each new run; do not overwrite the historical snapshots.
Use the separate fpeps launcher for fermions. MC/CTMRG and Python plotting can
use `code2d/run_preempt.sh` (Python: first argument `--python`). Every numerical
test, analysis and plot must also be submitted to `preempt`.

Migration checks: VUMPS unit tests **50/50** and the standalone fermionic
Bell-pair check **6/6** passed in `preempt` jobs 24163688 and 24163689. Static
checks found no missing literal Julia includes and verified the preserved
boundary/quantum implementation text and direct/bridge/six-R source hashes.
These are refactoring checks, not a rerun or validation of all historical
scientific results. Logs and job provenance are under `verification/`.

Fermion tests now use `code2d/fpeps/test/runtests.jl` with self-contained cases
under `test/regression/`; one-off checkpoint probes were archived locally.
See each module's test README. `python tools/check_repository.py --staged`
performs a static publication audit without running numerical calculations.

## Deliberately not imported

`.git`, Julia/Python caches, old job arrays, superseded controllers,
and unrelated finite-PEPS code are not copied into the new code tree. Originals
remain available outside `clean/`; none have been classified as disposable
merely because they were not selected. Only the small CTMRG and MC support
dependencies needed by the retained entries are included.
Solver logs are preserved separately in `data/runtime_logs/`, not discarded.
