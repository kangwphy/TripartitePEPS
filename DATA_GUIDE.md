# Where to find data

## RK-Ising

`data/rk_ising/results/` is the compact figure/table entry point:

* `VUMPS/`: original scans and convergence comparisons.
* `CTMRG/`: reference tables and figures.
* `MC/`: finite-size Monte Carlo comparison.
* `bare_chi16_saved_v1/`: bare LR/LTR benchmark; consult per-point diagnostics.
* `scaling_interpretation/`: saved scaling comparison, not a new fit.

Complete MC raw campaigns, pooled tables, error analyses and figures are in
`data/rk_ising/mc/`. CTMRG tables/logs/figures are in `data/rk_ising/ctmrg/`.
VUMPS saved-boundary and measurement snapshots are in `data/rk_ising/vumps/`.
Historical tables may contain unaccepted points: keep their convergence labels.

## TFIM

**Start with `data/tfim/current/` for current results**, with D2–D8 GS and
measurement entries and `results/` for shared outputs. This is an alias to
`data/tfim/store/current/`, not a copy. The **entire** old `quantum_tfim` store
is now physically in `data/tfim/store/`, including optimization intermediate
checkpoints, archived campaigns, failed/provisional runs and existing trash.
Supplementary QR/eigensolver campaigns are in `data/tfim/supplementary/`.
No raw checkpoint was filtered out during this whole-store move.

`data/tfim/GS_INDEX.csv` is the earlier frozen selection: it maps
(D, h, branch, CTM chi) to the copied
checkpoint and its SHA-256. `source_record_converged` records the original flag;
it does not mean the snapshot was reoptimized or extrapolated in chi.
Some old local summaries did not store their output hash; recovered directory
aliases are marked `original_output_hash_unavailable`. Two cross-machine D4,
CTM-96 imports lack an original output hash: local candidates are retained in
`diagnostics/foreign_imports/`, not promoted into the main GS index.

* `D2/` ... `D8/`: earlier frozen per-D snapshots, NOT the live results entry.
* `dense_h_scan/`: dense-scan measurement metadata and selected-source records.
* `checkpoints/`: actual serialized GS+environment files, content-addressed to
  avoid copying the same checkpoint under several misleading names.
* `provenance/`: original release/direct-only policies and manifests.
* `current/` and `live/`: aliases to `store/current/`, updated by ongoing jobs.
* `store/`: complete raw store, including all intermediate checkpoints.

Raw CSV/JSON contents are preserved. Absolute filenames inside them describe
original provenance; use `GS_INDEX.csv` for relocated GS inputs. Do not replace
failed/missing direct points with endpoint points. A numerical diagnostic PASS
and `production_release=false` must remain distinguished.

## Fermionic PEPS

`data/fermion/finite_window/` contains the saved finite-window benchmark and
its `gapped_convergence_audit/acceptance.json`. Its acceptance scope is finite
window, specified model/gauge/rank; it does not validate every infinite-plane
calculation. Infinite-boundary trial curves remain under `diagnostics/`.
`store/` contains the physically relocated complete independent fermionic
solver store, and `live/` aliases it. Historical experiments are preserved;
their presence does not mean acceptance.

## RVB D=3

Existing RVB curves are preserved under `data/rvb_d3/diagnostics/`. They are
provisional: some used relaxed thresholds, and the retained driver is not a
bare/direct benchmark. Folder cleanup does not change this status.
Complete original campaigns are in `data/rvb_d3/store/{vumps,root_runs}/`.

## Integrity and compatibility

See `DATA_FILES.csv`, `DATA_MOVES.json`, `DATA_SNAPSHOT_SUMMARY.json` and
`DATA_SNAPSHOT_ISSUES.json` at the root. No failed records are silently discarded.
The complete active stores were moved on the same filesystem; directory inode
identity was verified. Old paths are compatibility symlinks, not second data
copies. Running jobs were not cancelled or resubmitted. Existing absolute
paths in serialized metadata remain usable through those aliases. The old
code tree is retained for queued scripts and rollback; use the cleaned code
and explicit output paths for new runs.

`verification/full_migration/moves.json` is the relocation journal;
`checkpoint_inventory.csv` lists unfiltered raw checkpoint/container files,
their new paths, old compatible paths, sizes and inode metadata. It is an
inventory, not an acceptance manifest. Live files can continue changing after
the inventory timestamp. Runtime logs are in `data/runtime_logs/`.

Earlier root-level RK tables/figures and classical LMPS data are preserved in
`data/rk_ising/legacy/`; earlier fermion tables are in `data/fermion/legacy/`.
Mixed historical VUMPS archives (including older TFIM inputs) and organization
manifests are in `data/_legacy/`. Their relocation journal and checkpoint
inventory are in `verification/full_migration/legacy/`. These are archived
records, not newly accepted datasets. No historical data was deleted.
