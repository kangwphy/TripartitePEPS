#!/bin/bash
#SBATCH --job-name=D3-QR48-extra
#SBATCH --partition=fat6326
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-5%6
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
set -euo pipefail
: "${SLURM_JOB_ID:?Slurm allocation required}"
: "${SLURM_ARRAY_TASK_ID:?Slurm array required}"
: "${FIXED_CAMPAIGN:?Specify campaign root}"
: "${FIXED_PACKAGE:?Specify frozen physics package}"
: "${FIXED_ADDITION:?Specify immutable addition manifest}"
: "${TFIM_ADDITIONAL_REVISION:?Pinned worker Git revision}"
: "${TFIM_ADDITIONAL_WORKER_SHA256:?Pinned worker SHA}"
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS"
export JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_DEPOT_PATH=/home/xiang_15/chenjl/TFIM/.julia_depot_tfim
cd "$FIXED_PACKAGE"
python3 "$FIXED_CAMPAIGN/controller/run_fixed_additional_point.py" \
  --package "$FIXED_PACKAGE" --root "$FIXED_CAMPAIGN/D3" \
  --controller "$FIXED_CAMPAIGN/controller/run_fixed_endpoints.py" \
  --addition "$FIXED_ADDITION" --task-index "$SLURM_ARRAY_TASK_ID" \
  --julia /home/xiang_15/chenjl/TFIM/runtime/julia-1.12.6/bin/julia &
controller_pid=$!
trap 'kill -USR1 "$controller_pid" 2>/dev/null || true' USR1
trap 'kill -TERM "$controller_pid" 2>/dev/null || true' TERM
trap 'kill -INT "$controller_pid" 2>/dev/null || true' INT
set +e
while true; do
  wait "$controller_pid"
  result=$?
  kill -0 "$controller_pid" 2>/dev/null || break
done
set -e
if [[ "$result" -eq 42 ]]; then
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
exit "$result"
