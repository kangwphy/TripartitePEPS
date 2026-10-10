#!/bin/bash
#SBATCH --job-name=D3-QR48-env-repair
#SBATCH --partition=fat6326
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/home/xiang_15/chenjl/TFIM/PEPS/campaigns/qr_fixed_endpoints_20261006/recovery/qr_repair_%A_%a.out
#SBATCH --error=/home/xiang_15/chenjl/TFIM/PEPS/campaigns/qr_fixed_endpoints_20261006/recovery/qr_repair_%A_%a.err
set -euo pipefail
: "${SLURM_JOB_ID:?Slurm allocation required}"
: "${SLURM_ARRAY_TASK_ID:?Slurm array required}"
: "${TFIM_REPAIR_SCRIPT_SHA256:?Pinned repair-script SHA required}"
campaign=/home/xiang_15/chenjl/TFIM/PEPS/campaigns/qr_fixed_endpoints_20261006
package=/home/xiang_15/chenjl/TFIM/PEPS/campaigns/qr_bidirectional_fresh_20261006/code/code2d/lmps/vumps
script="$campaign/recovery/repair_same_peps_qr_environment_20261010.jl"
actual_sha=$(sha256sum "$script")
[[ "${actual_sha%% *}" == "$TFIM_REPAIR_SCRIPT_SHA256" ]] || exit 1
case "$SLURM_ARRAY_TASK_ID" in
  0) export TFIM_REPAIR_EXPECTED_H=3.0505
     point_name=h3.05050000
     export TFIM_REPAIR_EXPECTED_CHECKPOINT_SHA256=6bb3e17f11e6cd20c939db7d18f767aa856edc9ce4c198db09d6e579dd679509 ;;
  1) export TFIM_REPAIR_EXPECTED_H=3.051
     point_name=h3.05100000
     export TFIM_REPAIR_EXPECTED_CHECKPOINT_SHA256=307065ef64f7c18be43db9f046049b05c794827cdca06994ef89e76a260ba1cc ;;
  *) exit 2 ;;
esac
export TFIM_REPAIR_POINT="$campaign/D3/from_h3.03800000/$point_name"
export TFIM_REPAIR_MAXITER=10000
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=16
export OMP_NUM_THREADS=16
export JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_DEPOT_PATH=/home/xiang_15/chenjl/TFIM/.julia_depot_tfim
exec /home/xiang_15/chenjl/TFIM/runtime/julia-1.12.6/bin/julia -O3 --startup-file=no \
  --compiled-modules=existing --project="$package" "$script"
