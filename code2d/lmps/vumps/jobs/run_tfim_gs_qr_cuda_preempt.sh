#!/bin/bash
#SBATCH --job-name=tfim-gs-qr-cuda
#SBATCH --cluster=gpu
#SBATCH --partition=preempt
#SBATCH --qos=gpu-preempt-s
#SBATCH --constraint=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
set -euo pipefail
: "${CLEAN_ROOT:?Set CLEAN_ROOT to the clean checkout}"
: "${JULIA_EXE:?Set JULIA_EXE to the Julia 1.12 executable}"
: "${TFIM_SOURCE_STATE:?Set source checkpoint path}"
: "${TFIM_SOURCE_SHA256:?Set expected source hash}"
: "${TFIM_POINT_ROOT:?Set a separate output directory}"
: "${TFIM_H:?Set field}"
: "${TFIM_BRANCH:?Set ordered or disordered}"
PACKAGE="$CLEAN_ROOT/code2d/lmps/vumps"
export TFIM_CODE_SHA256=$(cat "$PACKAGE/production/tfim/run_gs_qr_cuda.jl" "$PACKAGE/src/algorithms/c4v_gs_cuda_support.jl" "$PACKAGE/Manifest.toml" | sha256sum | cut -d' ' -f1)
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK" OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" JULIA_PKG_PRECOMPILE_AUTO=0
export TFIM_MAX_STEPS=${TFIM_MAX_STEPS:-500} TFIM_SEGMENT_STEPS=${TFIM_SEGMENT_STEPS:-500}
cd "$PACKAGE"
exec "$JULIA_EXE" -O3 --startup-file=no --compiled-modules=existing --project=. production/tfim/run_gs_qr_cuda.jl
