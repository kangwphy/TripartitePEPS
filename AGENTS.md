# Working rules

1. **Pitt CRC: every CPU/GPU computation uses Slurm `--partition=preempt`.**
   This includes numerical tests, analysis and plotting. Login-node work is
   limited to file inspection/editing, administrative migration and submission.
   Never change to a paid/default partition or fabricate Slurm variables.
2. Read `README.md` and `DATA_GUIDE.md` before choosing an input dataset.
   Preserve source hashes, parameter values, convergence flags, route names,
   and release status. Completed files are not automatically accepted results.
3. The primary TFIM entropy route is bare/direct LR/LTR. Do not silently replace
   it with grown-tail or endpoint results. Retaining the latter in the API is
   for reproducibility, not a proof of equivalence.
4. Put shared boundary/bridge/six-R algorithms in `lmps/vumps/src/algorithms`,
   model definitions in `src/models`, and model-specific drivers in the matching
   `production` subdirectory. Keep the exported `LMPSVUMPS` API compatible.
5. Fermion PEPS uses its own graded code and `fpeps/environments/default`.
   Do not route fermion contractions through ordinary bosonic adapters.
6. Canonical raw stores are `data/{tfim,fermion,rvb_d3}/store`; old external
   paths are compatibility aliases still serving queued/running jobs. Do not
   remove aliases, rename stores or redirect writes without checking job status
   and provenance dependencies. Use `data/tfim/current` for live TFIM results,
   not the earlier frozen D2–D8 snapshots. Use explicit paths for new outputs.
