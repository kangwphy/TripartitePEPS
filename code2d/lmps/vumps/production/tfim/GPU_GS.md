# Opt-in CUDA GS continuation

`run_gs_qr_cuda.jl` brings the existing D4 CTM96 GPU QR continuation into the
clean code tree. CPU drivers and the public LMPSVUMPS import are unchanged.
It loads `src/algorithms/c4v_gs_cuda_support.jl` only in the CUDA process.

The input must be a converged D4 CTM48 checkpoint at the requested h/branch.
The PEPS is normalized, the chi96 environment is initialized, then QR CTMRG
and the implicit gradient/optimization run on CUDA. Small C4v gauge and
symmetry operations and the independent endpoint measurements use CPU arrays,
as in the previously qualified driver. A failed initial QR bootstrap may use
the existing CPU Eigh bootstrap before handing the environment back to QR.
This is recorded in bootstrap.csv; optimization remains QR.

Required environment: TFIM_SOURCE_STATE, TFIM_SOURCE_SHA256, TFIM_H,
TFIM_BRANCH, TFIM_POINT_ROOT, TFIM_CODE_SHA256. TFIM_MAX_STEPS defaults to 500
and is capped at 500; TFIM_SEGMENT_STEPS defaults to 100. Campaign launchers
set 500 for a single optimizer segment. Resuming restores PEPS/environment,
not LBFGS memory. TFIM_RUN_SECONDS bounds a segment between accepted updates.

Every accepted update saves portable CPU arrays and appends energy/gradient
history. Initial and final CPU/GPU energy/gradient checks are mandatory.
The final convergence flag is computed from the measured gradient; reaching
the step limit is not labeled gradient convergence. accepted.csv means the
endpoint passed backend agreement checks, not that its gradient converged.
The final checkpoint, magnetization, energy and CTM correlation length are
saved for every validated endpoint, including unconverged ones.

The underlying C4v QR projector reports a placeholder zero truncation error
(PEPSKit 0.8.1); it must not be interpreted as zero discarded weight.
CTM convergence error is saved at the final environment refresh.

Pitt runs must use --cluster=gpu --partition=preempt --constraint=a100.
