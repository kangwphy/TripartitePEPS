#!/bin/bash
#SBATCH --job-name=D3-QR48-CPU-bidir
#SBATCH --partition=fat6326
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
set -euo pipefail
export TFIM_BACKEND=cpu
export CHAIN_PACKAGE=${CHAIN_PACKAGE:-/home/xiang_15/chenjl/TFIM/PEPS/campaigns/qr_bidirectional_fresh_20261006/code/code2d/lmps/vumps}
export CHAIN_CAMPAIGN_ROOT=${CHAIN_CAMPAIGN_ROOT:-/home/xiang_15/chenjl/TFIM/PEPS/campaigns/qr_bidirectional_fresh_20261006/D3}
export CHAIN_PLAN=${CHAIN_PLAN:-/home/xiang_15/chenjl/TFIM/PEPS/campaigns/qr_bidirectional_fresh_20261006/campaign_plan.json}
export CHAIN_SEED=${CHAIN_SEED:-/home/xiang_15/chenjl/TFIM/PEPS/campaigns/qr_bidirectional_fresh_20261006/seeds/${1:?Pass direction}.jls}
export JULIA_EXE=${JULIA_EXE:-/home/xiang_15/chenjl/TFIM/runtime/julia-1.12.6/bin/julia}
export JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH:-/home/xiang_15/chenjl/TFIM/.julia_depot_tfim}
unset CHAIN_SLURM_CLUSTER
exec bash "$CHAIN_PACKAGE/jobs/run_bidirectional_allocated.sh" "${1:?Pass direction}"
