# Generic CUDA QR scan-point driver

`run_gs_qr_bidirectional_point.jl` is an independent opt-in driver based on the
qualified clean `run_gs_qr_cuda.jl`. It does not alter CPU drivers or support
imports. Deploy it inside the clean package's `production/tfim` directory so
the relative opt-in CUDA support include resolves correctly.

Required environment: `TFIM_SOURCE_STATE`, `TFIM_SOURCE_SHA256`,
`TFIM_POINT_ROOT`, `TFIM_H`, `TFIM_BRANCH`, `TFIM_CODE_SHA256`.
The branch is the current scan label; historical source labels are not used
to decide the physical state. The source can have a different h. Its actual
source h must exist in payload/config/common metadata or `TFIM_SOURCE_H`.
Raw root-seed payloads must contain a one-site CPU ComplexF64 `peps`; an
environment is optional. Prior unconverged GS checkpoints are accepted only
when their payload has `validated_endpoint=true`.

Defaults: `TFIM_D=3`, `TFIM_CTM_CHI=48`, `TFIM_AD_TOLERANCE=1e-5`,
`TFIM_MAX_STEPS=1000` (maximum 1000), `TFIM_SEGMENT_STEPS=1000`,
`TFIM_RUN_SECONDS=6000`, `TFIM_CTM_TOLERANCE=1e-9`,
`TFIM_GRADIENT_TOLERANCE=2e-7`, `TFIM_BASE_SEED=20261006`.
Set `TFIM_BASE_GIT_COMMIT` and a full included-source/Manifest digest in
`TFIM_CODE_SHA256`. A same-chi input environment is reused and QR-refreshed.
A failed bootstrap QR may use the documented CPU Eigh bootstrap, then hand
back to QR. Every optimization and final saved CTM refresh uses QR.

`TFIM_STOP_FILE` defaults to `POINT_ROOT/stop_after_step`. Runtime, segment and
stop-file controls do not enter the configuration hash, permitting a two-step
pilot and continuation with the same immutable numerical/source identity.
Delete a previous stop request before resume. PEPS/environment and cumulative
accepted steps are restored; LBFGS memory restarts at segment boundaries.
If preemption occurs after saving an accepted update but before its history
append, the exact final row is recovered from that checkpoint. Older missing
rows are errors rather than fabricated process data.

The optimizer stops on the projected gradient criterion or cumulative step
cap. A loose-gradient stop is provisional: tight independent CPU/GPU energy
and gradient checks are mandatory. If strict gradient exceeds tolerance and
budget remains, the driver continues with tighter solvers and optimizer stop
tolerance 0.9 times the requested tolerance. This mode is checkpointed, so a
resume cannot repeatedly stop at the old loose threshold without updates.

Every accepted update writes portable CPU `iteration_checkpoint.jls`,
`progress.csv` and `optimization_history.csv`. Initial paired audit writes
`initial_cpu_gpu_audit.csv`; `bootstrap.csv` records initial CTM convergence.
Per-update CTM convergence is not invented when the optimizer does not expose
it. Final CTM error belongs to the actual saved CPU QR environment.

Exit 42 means nonterminal segment/runtime/stop-file interruption: resume this
same point, and do not advance the chain. Exit 0 means a validated terminal
checkpoint. Other exits are failures. Successful terminal output consists of
`warmup_state.jls` (selected_groundstate, CPU PEPS/environment, metadata),
`ctm_final_xi.csv`, `final_summary.csv`, `accepted.csv` and `point_result.json`.
The JSON includes terminal, stopping_reason, completed_iterations, converged,
checkpoint path/SHA and exact source/code identity. A `step_limit` endpoint
has `converged=false` but is valid as the next point's explicit source. The
controller must verify the final SHA before advancing. `accepted.csv` records
backend agreement, not a claim of gradient convergence.

Final measurements include energy, x/z magnetization, horizontal/vertical CTM
xi and CTM tolerance/error. QR physical discarded weight is unavailable and
is explicitly marked so; its library placeholder zero is not physical data.
VUMPS measurement is a separate same-checkpoint step. No S_tilde is computed.

All Pitt scientific computation and validation must run in Slurm preempt.
