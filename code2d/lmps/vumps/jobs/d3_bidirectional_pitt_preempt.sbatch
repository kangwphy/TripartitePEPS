#!/bin/bash
#SBATCH --job-name=D3-QR48-bidir
#SBATCH --cluster=gpu
#SBATCH --partition=preempt
#SBATCH --qos=gpu-preempt-s
#SBATCH --constraint=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == preempt ]] || { echo 'Pitt requires preempt' >&2; exit 2; }
export CHAIN_PACKAGE=${CHAIN_PACKAGE:-/ix/zdai/kangw/PEPS3EE/PEPS/worktrees/tfim-bidirectional-20261006/code2d/lmps/vumps}
export CHAIN_CAMPAIGN_ROOT=${CHAIN_CAMPAIGN_ROOT:-/ix/zdai/kangw/PEPS3EE/PEPS/clean/data/tfim/store/campaigns/qr_bidirectional_20261006/D3}
export JULIA_EXE=${JULIA_EXE:-/ihome/zdai/kaw593/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia}
export JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH:-/ix/zdai/kangw/PEPS3EE/.julia_depot_vumps_gpu:/ix/zdai/kangw/PEPS3EE/.julia_depot:/ix/zdai/kangw/PEPS3EE/.julia_depot_vumps_cpu:/ihome/zdai/kaw593/.julia}
export CHAIN_SLURM_CLUSTER=gpu
exec bash "$CHAIN_PACKAGE/jobs/run_bidirectional_allocated.sh" "${1:?Pass direction}"
