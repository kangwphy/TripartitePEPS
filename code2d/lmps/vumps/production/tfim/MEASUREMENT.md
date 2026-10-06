# Final-state measurement contract

`measure_bidirectional_point.jl` uses the canonical LMPSVUMPS APIs in Pitt clean revision `1e7cfd3`. It measures D=3, CTM optimization chi48, and measurement chi48. It must run under a Slurm allocation; it refuses an absent `SLURM_JOB_ID`. It does not optimize the PEPS or calculate S_tilde.

From the allocated compute node, run:

```sh
julia --project=/path/to/worktree/code2d/lmps/vumps --threads=16 \
  /path/to/worktree/code2d/lmps/vumps/production/tfim/measure_bidirectional_point.jl \
  --point-dir /path/to/campaign/D3/increasing_h/h3.04000000
```

The controller may also pass `--source-sha FINAL_FILE_SHA`, `--expected-h H`, and `--expected-branch BRANCH`. `--vumps-maxiter` defaults to 1000, `--ctm-maxiter` to 2400; `--seed` can be specified for a retry. The driver fixes chi48 and measurement tolerance 1e-9. VUMPS diagnostics additionally require left/right environment and center residuals <=1e-8. CPU BLAS threads follow `SLURM_CPUS_PER_TASK`.

Input is `POINT/warmup_state.jls` plus a terminal `POINT/point_result.json` with one `checkpoint_sha256`. The state payload must contain `config` (D, h, branch, ctm_chi) and the CPU `selected_groundstate`. `source_sha256` in every measurement means the SHA256 of this **final checkpoint file**; the optimizer's incoming-checkpoint SHA is separately named `optimization_input_sha256`. A step-limit endpoint is measured but keeps `gs_converged=false` and its stopping reason.

The optimizer's final CPU QR environment is reused if explicit payload fields `ctm_converged`, `ctm_convergence_error`, `ctm_tolerance`, `ctm_projector`, and `ctm_measurement_backend` show sufficient convergence. In this campaign the optimizer saves tol1e-10, tighter than the measurement request. Otherwise the driver performs fixed-PEPS CPU C4vQR CTM refinement at tol1e-9. It compares measured CTM energy with `gs.energy` (tolerance 1e-10 for reuse; 1e-8 for a new refinement) and records the difference and actual tolerance.

VUMPS is the same independent north boundary solve used by the bare/six-R route, directly on the identical final PEPS, without the cardinal-boundary or entropy closures. `xi_M` is `MPSKit.correlation_length(boundary.state)`, while `xi_CTM_horizontal` and `xi_CTM_vertical` come from `MPSKit.correlation_length(peps, CTM_environment)`. CTM and VUMPS x/z are independent measurements; signed z and abs(z) are retained.

Outputs are in `POINT/measurement/`:

- `ctm.csv`, `vumps.csv`, `summary.csv`: explicitly named method-specific values and diagnostics, joined only by the same final-checkpoint SHA.
- `ctm_state.jls`: the actual measured CPU CTM environment, source PEPS and provenance.
- `boundary_state.jls`: the full CPU `selected_boundary`, including InfiniteMPS AL/AR/AC/C, environments GLs/GRs, transfer and source PEPS, plus provenance. Its actual dense dimensions must be `(48, D^2, 48)`.
- `vumps_diagnostics.csv`: convergence residuals, iterations, saved-boundary hash and reload evidence.
- `result.json`: controller marker. `status="complete"` requires accepted CTM and VUMPS, finite observables, unchanged final-GS hash, and a saved-boundary reload check. It contains `source_sha256`, `summary_csv/summary_sha256`, `boundary_checkpoint/boundary_sha256`, `ctm_checkpoint/ctm_checkpoint_sha256`, `measurement_converged`, `ctm_converged`, `vumps_converged`, and `gs_converged`. A measurement failure writes `status="failed"` and exits nonzero; the optimization controller should record pending measurement and continue its GS chain.

The boundary reload audit checks all AL/AR/AC/C and GLs/GRs tensors, transfer tensors, source PEPS, CPU storage and actual bond dimensions. The state format is Julia Serialization, so it is portable between the two CPU installations using the same Julia/package environment; it is not a cross-version interchange format. An unaccepted VUMPS state is retained as `boundary_unconverged.jls` with diagnostics and is never published as complete. A successful local syntax parse is not a numerical benchmark; the Slurm pilot must confirm the runtime APIs and timing.
