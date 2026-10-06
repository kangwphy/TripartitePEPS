#!/bin/bash
#SBATCH --job-name=D3-QR48-bidir
#SBATCH --partition=gpu
#SBATCH --gres=gpu:H100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
set -euo pipefail
export CHAIN_PACKAGE=${CHAIN_PACKAGE:-/data/kangw/TFIM/campaigns/qr_bidirectional_20261006/code/code2d/lmps/vumps}
export CHAIN_CAMPAIGN_ROOT=${CHAIN_CAMPAIGN_ROOT:-/data/kangw/TFIM/campaigns/qr_bidirectional_20261006/D3}
export CHAIN_PLAN=${CHAIN_PLAN:-/data/kangw/TFIM/campaigns/qr_bidirectional_20261006/campaign_plan.json}
export CHAIN_SEED=${CHAIN_SEED:-/data/kangw/TFIM/campaigns/qr_bidirectional_20261006/seeds/${1:?Pass direction}.jls}
export JULIA_EXE=${JULIA_EXE:-/data/kangw/TFIM/runtime/julia-1.12.6/bin/julia}
export JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH:-/data/kangw/TFIM/depot}
exec bash "$CHAIN_PACKAGE/jobs/run_bidirectional_allocated.sh" "${1:?Pass direction}"
