# TFIM bare/direct measurement: CPU and CUDA

Run `production/tfim/run_bare_sixr.jl` with the LMPSVUMPS Julia project.
Set these environment variables in your Slurm job:

```bash
export TFIM_SELECTED_STATE=/absolute/path/to/checkpoint.jls
export TFIM_ROUTE_CHI=24
export TFIM_BACKEND=cpu  # or cuda
export TFIM_BARE_OUTPUT=/absolute/path/to/new/output_directory
julia --startup-file=no --project=. production/tfim/run_bare_sixr.jl
```

CPU is the default. CUDA requires a GPU allocation and errors if CUDA is
unavailable; it never silently falls back to CPU. On PittCrc use partition
`preempt` for every CPU or GPU computation. For A100 request the GPU cluster,
preempt QOS and A100 constraint. No existing CPU package source or production
entry is modified. CUDA is loaded only by the explicit CUDA driver branch.

The driver follows the validated bare/direct LR/LTR benchmark: four cardinal
sides, three seeds per side, VUMPS tolerance 1e-8 (max4000), direct residual
1e-11 (max24000), and six-R tolerance 1e-10 (max8000). Only converged boundary
candidates are selected. The input is a one-site ungraded quantum PEPS saved
under `selected_groundstate.peps`. Measurement chi is separate from GS CTM chi.

Outputs in the fresh output directory:

- `measurement.csv`: backend, source/code hashes, convergence flags, S_tilde,
  residuals and stage timings.
- `measurement.jls`: scalar result, six-R audits, per-side/trial diagnostics,
  source configuration and metadata.
- `measurement_direct.jls`: portable CPU arrays for selected cardinal M, bulk
  a, and direct G/B/H, usable without a GPU.

Existing files are not overwritten. Optional `TFIM_BARE_SOURCE_SHA` pins the
expected input SHA256. The input GS convergence flag is preserved, not promoted.
A numerical diagnostic does not replace independent GS/strip/fidelity audits;
`production_release` remains false until that scientific review is performed.

Set `TFIM_BARE_WARMUP=true` for a complete warmup followed by a timed run in
the same process, saved as separate `warmup*` and `timed*` files. Timing sums
cover the measured boundary/direct/six-R stages, not all loading, setup and
file I/O. Compare only matching routes and settings. The prior D4, chi24 test
measured 703.05s on 16 CPU threads and 132.19s on A100, |delta S_tilde|=4e-9.
That does not validate GPU memory capacity or performance at chi96.

The GPU kernels are an explicitly namespaced derivative of the authoritative
CPU source (hashes in `src/algorithms/bare_cuda/SOURCE_SHA256.txt`). Update both
implementations and rerun parity tests after changing replica wiring or
normalization. Exact one-bond slicing reduces memory use but still grows
rapidly at high chi. No automatic reduction in chi is performed.

Tests: `test/bare_driver_physical.jl` with TFIM_BACKEND=cpu/cuda,
TFIM_TEST_SOURCE, TFIM_TEST_OUTPUT and optional TFIM_TEST_CHI (default 4);
`test/bare_cuda_regression.jl` for complex contraction parity; and
`test/bare_cuda_bridge.jl` with TFIM_CUDA_BRIDGE_INPUT pointing to saved
physical boundary trial files. Run all tests in Slurm compute allocations.
